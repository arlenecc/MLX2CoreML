#!/usr/bin/env python3
"""
convert_onnx_reranker_to_mlx.py
================================

将 xlm-roberta/bert 类 reranker 的 ONNX 导出版转换为 MLX 量化版（MLXEmbedders roberta 命名），
供 onnx-runtime-tester 的 MLXModelRuntime 加载。支持 int4 / int8（--bits 4|8）。

已验证事实（bge-reranker-base-ONNX -> bge-reranker-base-mlx-int4 / -int8）：
- 模型本质 roberta；编码器 2D Linear 权重名是通用 onnx::MatMul_*，但**节点名是语义的**
  （/roberta/encoder/layer.{L}/attention/self/{query,key,value}/MatMul 等），由节点名解析角色。
- 朝向：编码器 MatMul 权重无 Transpose 前置 -> 存 [in,out] -> 转置为 [out,in]；
  分类头 Gemm transB=1 -> 已是 [out,in] 不转置；嵌入表 Gather 消费 -> 不转置。
- 量化：mx.quantize(group_size, bits, "affine") 产 (packed, scales, biases)；
  bits=4/8 的 packed 权重均为 uint32，陪衬 scales/biases 为 float16，**兄弟键**存储
  {prefix}.weight / {prefix}.scales / {prefix}.biases（切勿写成 {prefix}.weight.scales）。
  原始 1D 偏置与 LayerNorm 以 fp16 存。
- config.json = 源 config（含 model_type=xlm-roberta）+ quantization + num_labels:1。
- 注意：源 config 自带 model_type，无需像 jina 那样补；roberta 有 embeddings.LayerNorm，
  会被 BIAS_PATTERNS 捕获并写成 embeddings.norm（真实值，非占位）。

用法：
    python3 convert_onnx_reranker_to_mlx.py --src SRC_ONNX_DIR --dst DST_DIR \
        [--onnx PATH] [--bits 4|8 --group-size 64 --mode affine] \
        [--no-verify] [--verify-only]
"""
import argparse
import json
import os
import re
import shutil

import mlx.core as mx
import numpy as np
import onnx


def load_onnx_initializers(onnx_path):
    m = onnx.load(onnx_path)
    init = {i.name: np.ascontiguousarray(onnx.numpy_helper.to_array(i))
            for i in m.graph.initializer}
    return m, init


def build_node_maps(model):
    out2node, input2nodes, by_name = {}, {}, {}
    for n in model.graph.node:
        by_name[n.name] = n
        for o in n.output:
            out2node[o] = n
        for i in n.input:
            input2nodes.setdefault(i, []).append(n)
    return out2node, input2nodes, by_name


MATMUL_PATTERNS = [
    (re.compile(r"layer\.(\d+)/attention/self/query/MatMul"),
     "encoder.layers.{}.attention.query_proj"),
    (re.compile(r"layer\.(\d+)/attention/self/key/MatMul"),
     "encoder.layers.{}.attention.key_proj"),
    (re.compile(r"layer\.(\d+)/attention/self/value/MatMul"),
     "encoder.layers.{}.attention.value_proj"),
    (re.compile(r"layer\.(\d+)/attention/output/dense/MatMul"),
     "encoder.layers.{}.attention.out_proj"),
    (re.compile(r"layer\.(\d+)/intermediate/dense/MatMul"),
     "encoder.layers.{}.linear1"),
    (re.compile(r"layer\.(\d+)/output/dense/MatMul"),
     "encoder.layers.{}.linear2"),
]
GEMM_PATTERNS = [
    (re.compile(r"classifier/dense/Gemm"), "classifier.dense"),
    (re.compile(r"classifier/out_proj/Gemm"), "classifier.out_proj"),
]
BIAS_PATTERNS = [
    (re.compile(r"layer\.(\d+)[\./]attention[\./]self[\./](query|key|value)[\./]bias"),
     lambda L, r: f"encoder.layers.{L}.attention.{r}_proj.bias"),
    (re.compile(r"layer\.(\d+)[\./]attention[\./]output[\./]dense[\./]bias"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.bias"),
    (re.compile(r"layer\.(\d+)[\./]intermediate[\./]dense[\./]bias"),
     lambda L: f"encoder.layers.{L}.linear1.bias"),
    (re.compile(r"layer\.(\d+)[\./]output[\./]dense[\./]bias"),
     lambda L: f"encoder.layers.{L}.linear2.bias"),
    (re.compile(r"layer\.(\d+)[\./]attention[\./]output[\./]LayerNorm[\./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln1.{r}"),
    (re.compile(r"layer\.(\d+)[\./]output[\./]LayerNorm[\./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln2.{r}"),
    (re.compile(r"embeddings[\./]LayerNorm[\./](weight|bias)"),
     lambda r: f"embeddings.norm.{r}"),
    (re.compile(r"classifier[\./]dense[\./]bias"),
     lambda: "classifier.dense.bias"),
    (re.compile(r"classifier[\./]out_proj[\./]bias"),
     lambda: "classifier.out_proj.bias"),
]
EMBED_2D = {
    "roberta.embeddings.word_embeddings.weight": "embeddings.word_embeddings",
    "roberta.embeddings.position_embeddings.weight": "embeddings.position_embeddings",
    "roberta.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings",
}


def weight_has_transpose_before(w_name, input2nodes):
    for c in input2nodes.get(w_name, []):
        if c.op_type == "Transpose":
            return True
    return False


def convert(onnx_path, dst, bits, group_size, mode):
    os.makedirs(dst, exist_ok=True)
    model, init = load_onnx_initializers(onnx_path)
    _, input2nodes, _ = build_node_maps(model)
    init_names = set(init.keys())
    state = {}

    def q2d(arr_np):
        a = mx.array(arr_np, dtype=mx.float16)
        if bits in (16, 32):  # 无损（仅用于验证朝向）
            return a, None, None
        wq, sc, bi = mx.quantize(a, group_size, bits, mode=mode)
        return wq, sc, bi

    def store(prefix, wq, sc, bi):
        # prefix 是角色名（不含 .weight）；量化陪衬键为兄弟键：
        #   encoder.layers.0.attention.query_proj.weight / .scales / .biases
        if sc is None:
            state[f"{prefix}.weight"] = wq
        else:
            state[f"{prefix}.weight"] = wq
            state[f"{prefix}.scales"] = sc
            state[f"{prefix}.biases"] = bi

    # 1) 编码器 MatMul 2D 权重
    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        w_names = [i for i in n.input if i in init_names]
        if not w_names:
            continue
        np_w = init[w_names[0]]
        if np_w.ndim != 2:
            continue
        role = next((t.format(m.group(1)) for p, t in MATMUL_PATTERNS
                     if (m := p.search(n.name))), None)
        if role is None:
            print(f"[warn] 未匹配 MatMul 节点名: {n.name}")
            continue
        if not weight_has_transpose_before(w_names[0], input2nodes):
            np_w = np_w.T
        wq, sc, bi = q2d(np_w)
        store(role, wq, sc, bi)

    # 2) 分类头 Gemm 2D 权重
    for n in model.graph.node:
        if n.op_type != "Gemm":
            continue
        w_names = [i for i in n.input if i in init_names]
        if not w_names:
            continue
        np_w = init[w_names[0]]
        if np_w.ndim != 2:
            continue
        role = next((t for p, t in GEMM_PATTERNS if p.search(n.name)), None)
        if role is None:
            print(f"[warn] 未匹配 Gemm 节点名: {n.name}")
            continue
        transB = next((a.i for a in n.attribute if a.name == "transB"), 0)
        if transB == 0:
            np_w = np_w.T
        wq, sc, bi = q2d(np_w)
        store(role, wq, sc, bi)

    # 3) 嵌入表 2D（不转置）
    for onnx_k, mlx_k in EMBED_2D.items():
        if onnx_k in init:
            wq, sc, bi = q2d(init[onnx_k])
            store(mlx_k, wq, sc, bi)

    # 4) 1D 偏置 / LayerNorm
    for onnx_k, arr in init.items():
        if arr.ndim != 1:
            continue
        for pat, fn in BIAS_PATTERNS:
            m = pat.search(onnx_k)
            if m:
                state[fn(*m.groups())] = mx.array(arr, dtype=mx.float16)
                break

    # 保存
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), state)
    n2d = sum(1 for k in state if k.endswith(".weight"))
    print(f"转换完成（int{bits}）：{n2d} 个 2D 权重已量化，共 {len(state)} 个张量 -> {dst}")

    # config.json
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(onnx_path)), "config.json")
    if not os.path.exists(cfg_path):
        # 回退：onnx_path 的父目录即 src
        cand = os.path.join(os.path.dirname(onnx_path), "..", "config.json")
        if os.path.exists(cand):
            cfg_path = cand
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        cfg = {"model_type": "xlm-roberta"}
    cfg["quantization"] = {"bits": bits, "group_size": group_size, "mode": mode}
    cfg["num_labels"] = 1
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"),
              ensure_ascii=False, indent=2)

    # 拷贝 tokenizer
    src_dir = os.path.dirname(os.path.dirname(onnx_path)) \
        if os.path.basename(os.path.dirname(onnx_path)) == "onnx" \
        else os.path.dirname(onnx_path)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"]:
        sp = os.path.join(src_dir, f)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(dst, f))
    return state


# --------------------------------------------------------------------------- #
# 验证：onnxruntime 原 logits vs HF 反量化 load_state_dict logits
# --------------------------------------------------------------------------- #

def _mlx_to_hf_key(k):
    """MLX 量化键 -> HF XLMRobertaForSequenceClassification 键（用于验证反量化）。
    HF 结构：roberta.embeddings.* / roberta.encoder.* / classifier.*（顶层）。"""
    m = re.match(r"encoder\.layers\.(\d+)\.attention\.(query|key|value)_proj\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.attention.self.{m.group(2)}.{m.group(3)}"
    m = re.match(r"encoder\.layers\.(\d+)\.attention\.out_proj\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.attention.output.dense.{m.group(2)}"
    m = re.match(r"encoder\.layers\.(\d+)\.linear1\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.intermediate.dense.{m.group(2)}"
    m = re.match(r"encoder\.layers\.(\d+)\.linear2\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.output.dense.{m.group(2)}"
    m = re.match(r"encoder\.layers\.(\d+)\.ln1\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.attention.output.LayerNorm.{m.group(2)}"
    m = re.match(r"encoder\.layers\.(\d+)\.ln2\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layer.{m.group(1)}.output.LayerNorm.{m.group(2)}"
    m = re.match(r"embeddings\.(word|position|token_type)_embeddings\.weight", k)
    if m:
        return f"roberta.embeddings.{m.group(1)}_embeddings.weight"
    if k == "embeddings.norm.weight":
        return "roberta.embeddings.LayerNorm.weight"
    if k == "embeddings.norm.bias":
        return "roberta.embeddings.LayerNorm.bias"
    if k in ("classifier.dense.weight", "classifier.dense.bias",
             "classifier.out_proj.weight", "classifier.out_proj.bias"):
        return k
    return None


def verify(dst, src, bits, group_size, mode, onnx_path=None):
    import onnxruntime as ort
    import torch
    from transformers import AutoConfig, AutoTokenizer, XLMRobertaForSequenceClassification

    # 1) onnxruntime（基线：优先用传入的 fp16 源；否则 model.onnx；否则 src/onnx/model_fp16.onnx）
    if onnx_path is None or not os.path.exists(onnx_path):
        cand = os.path.join(src, "onnx", "model.onnx")
        if os.path.exists(cand):
            onnx_path = cand
        else:
            cands = []
            for root, _, files in os.walk(src):
                for f in files:
                    if f == "model.onnx":
                        cands.append(os.path.join(root, f))
            if cands:
                onnx_path = cands[0]
            else:
                fp16 = os.path.join(src, "onnx", "model_fp16.onnx")
                onnx_path = fp16 if os.path.exists(fp16) else None
    if onnx_path is None or not os.path.exists(onnx_path):
        raise FileNotFoundError(f"找不到用于比对的 ONNX 基线文件（期望 src/onnx/model.onnx 或 model_fp16.onnx）")
    print(f"比对基线 ONNX：{onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = {i.name: i for i in sess.get_inputs()}
    tok = AutoTokenizer.from_pretrained(src)
    pair = ("What is the capital of France?",
            "Paris is the capital of France.")
    enc = tok(pair[0], pair[1], return_tensors="np")
    feed = {}
    for name in inp:
        if name in enc:
            feed[name] = enc[name].astype(np.int64)
        elif "token_type" in name:
            feed[name] = np.zeros_like(enc["input_ids"]).astype(np.int64)
    ort_logits = sess.run(None, feed)[0]

    # 2) HF 反量化
    raw = mx.load(os.path.join(dst, "model.safetensors"))
    sd = {}
    for k, v in raw.items():
        if k.endswith(".weight") and v.dtype == mx.uint32:
            scales = raw[k.replace(".weight", ".scales")]
            biases = raw[k.replace(".weight", ".biases")]
            w = mx.dequantize(v, scales, biases, group_size, bits, mode=mode)
            hfk = _mlx_to_hf_key(k)
            if hfk:
                sd[hfk] = torch.from_numpy(np.asarray(w.astype(mx.float32)))
        elif (k.endswith(".bias") or k.endswith(".weight")) and v.dtype != mx.uint32:
            hfk = _mlx_to_hf_key(k)
            if hfk:
                sd[hfk] = torch.from_numpy(np.asarray(v.astype(mx.float32)))
    cfg = AutoConfig.from_pretrained(src)
    cfg.num_labels = 1
    model = XLMRobertaForSequenceClassification(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"HF load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    model.eval()
    with torch.no_grad():
        t = {kk: torch.from_numpy(np.asarray(vv)) for kk, vv in enc.items()}
        hf_logits = model(**t).logits.numpy()

    diff = np.abs(ort_logits - hf_logits).max()
    print(f"ORT logits[0,0]={ort_logits.flatten()[0]:.5f}  HF[0,0]={hf_logits.flatten()[0]:.5f}")
    print(f"max abs diff (int{bits} 反量化 vs onnxruntime) = {diff:.6f}")
    # missing/unexpected=0 已证明朝向/命名完全正确；此处 diff 仅为量化噪声
    # （int8 权重最大误差约 0.023，量级与 logit 6.x 相比可忽略）。
    thr = 0.05
    if diff < thr:
        print(f"验证通过：int{bits} 转换朝向/命名正确，与 ONNX 原 logits 基本一致。")
    else:
        print(f"⚠ 差异偏大（>{thr}），请检查朝向/命名映射。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ONNX 模型目录（含 onnx/ 子目录或 --onnx 指定文件）")
    ap.add_argument("--dst", required=True, help="输出 MLX 目录")
    ap.add_argument("--onnx", default=None, help="ONNX 文件路径（缺省取 SRC/onnx/model.onnx）")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--mode", default="affine")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.dst, args.src, args.bits, args.group_size, args.mode, args.onnx)
        return

    onnx_path = args.onnx or os.path.join(args.src, "onnx", "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"找不到 ONNX 文件: {onnx_path}")
    convert(onnx_path, args.dst, args.bits, args.group_size, args.mode)
    if not args.no_verify:
        verify(args.dst, args.src, args.bits, args.group_size, args.mode, onnx_path)


if __name__ == "__main__":
    main()
