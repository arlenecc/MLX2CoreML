#!/usr/bin/env python3
"""
convert_onnx_jina_reranker_v2_to_mlx_int8.py
============================================

将 jina-reranker-v2-base-multilingual 的 **fp16 ONNX**（`onnx/model_fp16.onnx`）转换为
**MLX int8**（affine 量化，group_size=64），供 mlx-swift-lm / MLXEmbedders 的
XLMRobertaForSequenceClassification 加载（与 bge-reranker-base-mlx-int4 同款命名）。

与 `convert_onnx_jina_reranker_v2_to_mlx_int4.py` 的唯一区别：默认 `bits=8`（int8 比 int4
更精细，权重误差更小）。架构映射、朝向、融合 QKV 拆分、验证逻辑完全一致。

与 bge 版（convert_onnx_reranker_to_mlx_int4.py）的区别：
- jina 的注意力/FFN 节点命名为 `mixer/Wqkv`、`mixer/out_proj`、`mlp/fc1`、`mlp/fc2`，
  且 QKV 是**融合**的（`Wqkv`，shape [in, 3H]），需拆成 query/key/value_proj。
- 归一化命名为 `norm1`/`norm2`（无 embeddings.LayerNorm）。
- ONNX 仅有 `input_ids`/`attention_mask` 两个输入（无 token_type_ids）。

权重朝向（已对照 HF model.safetensors 确认）：
- 编码器 MatMul 权重无 Transpose 前置 → 存为 [in,out]，转置为 [out,in]；
  Wqkv 转置后按行拆成 q[:H] / k[H:2H] / v[2H:3H]。
- 分类头 Gemm transB=1 → 权重已是 [out,in]，不转置。
- 所有 ONNX 初始化器名与 HF 完全一致，验证直接以 HF 为基准。

验证：
1) 逐权重对比：MLX int8 反量化 vs HF 权重（捕捉朝向 / 命名 / 拆分 / 量化误差）。
2) onnxruntime 跑 fp16 ONNX 取基线 logits（确认源模型可跑出有限值）。

用法：
    python3 convert_onnx_jina_reranker_v2_to_mlx_int8.py --src SRC_DIR --dst DST_DIR \
        [--onnx PATH] [--bits 8] [--group-size 64] [--no-quantize] [--verify-only] [--max-length 512]
"""
import argparse
import json
import os
import re
import shutil

import numpy as np
import onnx
import mlx.core as mx
from onnx import numpy_helper

GROUP_SIZE = 64
BITS = 8
MODE = "affine"

# ONNX/HF 嵌入表 -> MLX 键（不转置）
EMBED = {
    "roberta.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "roberta.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "roberta.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
}

LAYER_RE = re.compile(r"roberta\.encoder\.layers\.(\d+)\.(mixer|mlp)\.(\w+)\.(weight|bias)")
NORM_RE = re.compile(r"roberta\.encoder\.layers\.(\d+)\.(norm1|norm2)\.(weight|bias)")
CLS_RE = re.compile(r"classifier\.(\w+)\.(weight|bias)")

HIDDEN = 768


def resolve_onnx(src, onnx_arg):
    if onnx_arg:
        return onnx_arg
    cand = os.path.join(src, "onnx", "model_fp16.onnx")
    if os.path.exists(cand):
        return cand
    # 退回其它候选
    for name in ("model_fp16.onnx", "model.onnx", "model_quantized.onnx"):
        p = os.path.join(src, "onnx", name)
        if os.path.exists(p):
            return p
    return os.path.join(src, "onnx", "model_fp16.onnx")


def q2d(arr, bits, group_size, mode):
    a = mx.array(arr, dtype=mx.float16) if arr.dtype == np.float16 \
        else mx.array(arr.astype(np.float32), dtype=mx.float32)
    if bits in (16, 32):
        return a, None, None
    return mx.quantize(a, group_size, bits, mode=mode)


def store(state, prefix, wq, sc, bi):
    if sc is None:
        state[prefix] = wq
    else:
        # Swift/标准 MLX 量化约定：陪衬键是兄弟键（去掉 .weight 后加 .scales/.biases），
        # 例如 query_proj.weight -> query_proj.scales / query_proj.biases（切勿写成 query_proj.weight.scales）。
        state[prefix] = wq
        base = prefix.rsplit(".weight", 1)[0]
        state[base + ".scales"] = sc
        state[base + ".biases"] = bi


def convert(m, init, dst, src, bits, group_size, mode, no_quantize):
    state = {}
    # 1) 嵌入表（不转置，roberta.* 命名）
    for onnx_k, mlx_k in EMBED.items():
        if onnx_k in init:
            wq, sc, bi = q2d(init[onnx_k], bits, group_size, mode)
            store(state, mlx_k, wq, sc, bi)

    def matmul_wname(n):
        return next((i for i in n.input if i in init), None)

    # 2) 编码器 MatMul 权重（名为 onnx::MatMul_NNNN，需按节点名定位角色）
    for n in m.graph.node:
        if n.op_type == "MatMul":
            mm = re.match(r"/roberta/encoder/layers\.(\d+)/mixer/Wqkv/MatMul", n.name)
            if mm:
                L = int(mm.group(1))
                wn = matmul_wname(n)
                if wn is None:
                    continue
                w = init[wn].T  # [in,out] -> [out=3H, in=H]
                for proj, piece in (("query_proj", w[0:HIDDEN]),
                                    ("key_proj", w[HIDDEN:2 * HIDDEN]),
                                    ("value_proj", w[2 * HIDDEN:3 * HIDDEN])):
                    wq, sc, bi = q2d(piece, bits, group_size, mode)
                    store(state, f"encoder.layers.{L}.attention.{proj}.weight", wq, sc, bi)
                continue
            mm = re.match(r"/roberta/encoder/layers\.(\d+)/mixer/out_proj/MatMul", n.name)
            if mm:
                L = int(mm.group(1)); wn = matmul_wname(n)
                if wn is None:
                    continue
                wq, sc, bi = q2d(init[wn].T, bits, group_size, mode)  # 权重 [in,out]->[out,in]
                store(state, f"encoder.layers.{L}.attention.out_proj.weight", wq, sc, bi)
                continue
            mm = re.match(r"/roberta/encoder/layers\.(\d+)/mlp/fc1/MatMul", n.name)
            if mm:
                L = int(mm.group(1)); wn = matmul_wname(n)
                if wn is None:
                    continue
                wq, sc, bi = q2d(init[wn].T, bits, group_size, mode)
                store(state, f"encoder.layers.{L}.linear1.weight", wq, sc, bi)
                continue
            mm = re.match(r"/roberta/encoder/layers\.(\d+)/mlp/fc2/MatMul", n.name)
            if mm:
                L = int(mm.group(1)); wn = matmul_wname(n)
                if wn is None:
                    continue
                wq, sc, bi = q2d(init[wn].T, bits, group_size, mode)
                store(state, f"encoder.layers.{L}.linear2.weight", wq, sc, bi)
                continue
        elif n.op_type == "Gemm":  # 分类头 transB=1 -> 权重已是 [out,in]
            gm = re.match(r"/classifier/(dense|out_proj)/Gemm", n.name)
            if gm:
                kind = gm.group(1); wn = matmul_wname(n)
                if wn is None:
                    continue
                wq, sc, bi = q2d(init[wn], bits, group_size, mode)
                store(state, f"classifier.{kind}.weight", wq, sc, bi)
                continue

    # 3) 偏差 / 归一化 / 分类头偏差（roberta.* 命名，含 Wqkv 融合 bias 拆分）
    for name, arr in init.items():
        m = LAYER_RE.match(name)
        if m:
            L, block, kind, wt = m.groups()
            L = int(L)
            if block == "mixer" and kind == "Wqkv":
                if wt == "weight":
                    continue  # 权重由节点处理
                bq, bk, bv = arr[0:HIDDEN], arr[HIDDEN:2 * HIDDEN], arr[2 * HIDDEN:3 * HIDDEN]
                for proj, piece in (("query_proj", bq), ("key_proj", bk), ("value_proj", bv)):
                    state[f"encoder.layers.{L}.attention.{proj}.bias"] = \
                        mx.array(piece, dtype=mx.float16)
                continue
            if block == "mixer" and kind == "out_proj":
                state[f"encoder.layers.{L}.attention.out_proj.{wt}"] = mx.array(arr, dtype=mx.float16)
            elif block == "mlp" and kind == "fc1":
                state[f"encoder.layers.{L}.linear1.{wt}"] = mx.array(arr, dtype=mx.float16)
            elif block == "mlp" and kind == "fc2":
                state[f"encoder.layers.{L}.linear2.{wt}"] = mx.array(arr, dtype=mx.float16)
            continue

        m = NORM_RE.match(name)
        if m:
            L, which, wt = m.groups()
            mlxk = f"encoder.layers.{L}.{'ln1' if which == 'norm1' else 'ln2'}.{wt}"
            state[mlxk] = mx.array(arr, dtype=mx.float16)
            continue

        m = CLS_RE.match(name)
        if m:
            kind, wt = m.groups()
            if wt == "weight":
                continue  # 权重由 Gemm 节点处理
            state[f"classifier.{kind}.{wt}"] = mx.array(arr, dtype=mx.float16)
            continue

    # jina 的参考实现 embeddings 无 LayerNorm，但 MLXEmbedders 的 BertEmbedding 总会创建并
    # 应用 embeddings.norm。加载器以 verify:[.all] 加载，缺失该权重会触发 updateMissing 崩溃。
    # 故必须提供（ones/zeros 为 LayerNorm 的“默认”初始化，经第一层 ln1 抵消，不会破坏 reranker）。
    state["embeddings.norm.weight"] = mx.ones((HIDDEN,), dtype=mx.float32)
    state["embeddings.norm.bias"] = mx.zeros((HIDDEN,), dtype=mx.float32)

    # 4) 写出
    os.makedirs(dst, exist_ok=True)
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), state)
    cfg = {}
    if os.path.exists(os.path.join(src, "config.json")):
        cfg = json.load(open(os.path.join(src, "config.json")))
    # 源 config 无 model_type（依赖 auto_map，Swift/MLXEmbedders 无法据此路由）；
    # 缺它会令 BertConfiguration 解码抛错导致加载失败。必须显式补上。
    cfg["model_type"] = "xlm-roberta"
    cfg["quantization"] = {
        "group_size": group_size if bits not in (16, 32) else None,
        "bits": bits,
        "mode": MODE if bits not in (16, 32) else "none",
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), ensure_ascii=False, indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.txt", "vocab.json", "merges.txt"):
        sp = os.path.join(src, f)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(dst, f))
    n2d = sum(1 for k in state if k.endswith(".weight") and state[k].dtype == mx.uint32)
    nw = sum(1 for k in state if k.endswith(".weight"))
    print(f"已写出 MLX {'fp16' if bits in (16,32) else 'int%d'%bits} 模型：{os.path.join(dst,'model.safetensors')}")
    print(f"张量总数：{len(state)}（权重 {nw}，其中量化 {n2d}），期望 ~153")
    return state


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #

def _mlx_to_hf(k, raw):
    """MLX 键 -> (HF 键, 是否需从 Wqkv 切片)。HF 为基准。"""
    m = re.match(r"embeddings\.(word|position|token_type)_embeddings\.weight", k)
    if m:
        return f"roberta.embeddings.{m.group(1)}_embeddings.weight", None
    m = re.match(r"encoder\.layers\.(\d+)\.attention\.(query|key|value)_proj\.weight", k)
    if m:
        L, r = m.groups()
        return f"roberta.encoder.layers.{L}.mixer.Wqkv.weight", {"query": 0, "key": 1, "value": 2}[r]
    m = re.match(r"encoder\.layers\.(\d+)\.attention\.(query|key|value)_proj\.bias", k)
    if m:
        L, r = m.groups()
        return f"roberta.encoder.layers.{L}.mixer.Wqkv.bias", {"query": 0, "key": 1, "value": 2}[r]
    m = re.match(r"encoder\.layers\.(\d+)\.attention\.out_proj\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layers.{m.group(1)}.mixer.out_proj.{m.group(2)}", None
    m = re.match(r"encoder\.layers\.(\d+)\.linear1\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layers.{m.group(1)}.mlp.fc1.{m.group(2)}", None
    m = re.match(r"encoder\.layers\.(\d+)\.linear2\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layers.{m.group(1)}.mlp.fc2.{m.group(2)}", None
    m = re.match(r"encoder\.layers\.(\d+)\.ln1\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layers.{m.group(1)}.norm1.{m.group(2)}", None
    m = re.match(r"encoder\.layers\.(\d+)\.ln2\.(weight|bias)", k)
    if m:
        return f"roberta.encoder.layers.{m.group(1)}.norm2.{m.group(2)}", None
    m = re.match(r"classifier\.(\w+)\.(weight|bias)", k)
    if m:
        return f"classifier.{m.group(1)}.{m.group(2)}", None
    return None, None


def verify(dst, src, onnx_path, max_length):
    import torch
    import safetensors.torch as st
    from transformers import AutoTokenizer
    import onnxruntime as ort

    qcfg = json.load(open(os.path.join(dst, "config.json"))).get("quantization", {})
    bits = qcfg.get("bits", 8)
    group_size = qcfg.get("group_size", 64) or 64
    mode = qcfg.get("mode", "affine")

    # 1) 逐权重对比 MLX 反量化 vs HF
    hf = st.load_file(os.path.join(src, "model.safetensors"))
    raw = mx.load(os.path.join(dst, "model.safetensors"))
    mlx = {}
    for k, v in raw.items():
        if k.endswith(".weight") and v.dtype == mx.uint32:
            base = k.rsplit(".weight", 1)[0]
            scales = raw[base + ".scales"]
            biases = raw[base + ".biases"]
            mlx[k] = mx.dequantize(v, scales, biases, group_size, bits, mode=mode).astype(mx.float32)
        elif k.endswith(".weight") or k.endswith(".bias"):
            mlx[k] = v.astype(mx.float32)

    max_diff = 0.0
    min_cos = 1.0
    sum_cos = 0.0
    n_compared = 0
    worst = ("", 0.0)
    for k, v in mlx.items():
        hfk, sl = _mlx_to_hf(k, raw)
        if hfk is None or hfk not in hf:
            continue
        hv = hf[hfk]
        if sl is not None:
            hv = hv[sl * HIDDEN:(sl + 1) * HIDDEN]
        if hasattr(hv, "to"):  # torch.Tensor（可能含 bfloat16）
            hv = hv.to(torch.float32).numpy()
        hv = np.asarray(hv, dtype=np.float32).reshape(v.shape)
        a = np.asarray(v, dtype=np.float32).ravel()
        b = hv.ravel()
        d = float(np.max(np.abs(a - b)))
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        max_diff = max(max_diff, d)
        min_cos = min(min_cos, cos)
        sum_cos += cos
        n_compared += 1
        if d > worst[1]:
            worst = (k, d)
    mean_cos = sum_cos / n_compared if n_compared else 1.0
    print(f"[权重对比] 对比 {n_compared} 个张量，最大绝对误差 = {max_diff:.5f}（最差：{worst[0]} {worst[1]:.5f}）")
    print(f"          最小余弦 = {min_cos:.6f}，平均余弦 = {mean_cos:.6f}（int8 噪声下应 >0.99）")
    # int8 量化更精细，余弦应 >0.99；朝向/命名错误会使余弦暴跌至 ~0.5 以下
    weights_ok = (min_cos > 0.99) if bits not in (16, 32) else (max_diff < 1e-2)

    # 2) onnxruntime fp16 基线 logits（仅确认源可跑）
    tok = AutoTokenizer.from_pretrained(src)
    texts = ["The cat sat on the mat.", "A dog played in the park.",
             "Machine learning is a subset of artificial intelligence."]
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                          "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    logits = out.reshape(-1)
    finite = bool(np.all(np.isfinite(logits)))
    print(f"[ONNX 基线] logits = {np.round(logits, 3).tolist()}  finite={finite}")

    if weights_ok and finite:
        print("✅ 验证通过：MLX 权重与 HF 基准一致（朝向/命名/拆分正确），ONNX 基线有限。")
    else:
        print("⚠ 验证警告：权重误差过大或 ONNX 基线异常，请检查转换。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="jina reranker v2 ONNX 目录（含 onnx/）")
    ap.add_argument("--dst", required=True, help="MLX 输出目录")
    ap.add_argument("--onnx", default=None, help="显式指定 ONNX 文件路径")
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--no-quantize", action="store_true", help="仅转 fp16（无损，用于验证朝向）")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    onnx_path = resolve_onnx(args.src, args.onnx)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)

    if args.verify_only:
        verify(args.dst, args.src, onnx_path, args.max_length)
        return

    m = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    bits = 16 if args.no_quantize else args.bits
    convert(m, init, args.dst, args.src, bits, args.group_size, MODE, args.no_quantize)
    verify(args.dst, args.src, onnx_path, args.max_length)


if __name__ == "__main__":
    main()
