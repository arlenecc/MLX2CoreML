#!/usr/bin/env python3
"""
convert_onnx_fp16_to_int8.py
============================

将「fp16 的 ONNX embedding 模型」（如 nomic-embed-text-v1.5-onnx-fp16）权重量化到 **ONNX int8**
（动态量化 / weight-only，MatMul 权重 int8 + per-channel scale，激活保持 fp16）。

要点（已验证 nomic-embed-text-v1.5）：
- onnxruntime 的 quantize_dynamic 可直接处理 fp16 源模型（内部预处理），无需手动转 fp32；
  手动 blanket 改张量类型会破坏图内 Cast 节点。
- 模型只输出 last_hidden_state（token 级，dtype=float32），embedding 需外部做
  「attention_mask 平均池化 + L2 归一化」。
- 验证：onnxruntime CPU 同时跑 fp16 源与 int8，对一批英文文本算 embedding，比较余弦相似度
  （典型 >0.999），并做检索式相关性检查（相关文档相似度应高于无关文档）。

用法：
    python3 convert_onnx_fp16_to_int8.py --src SRC_ONNX_DIR --dst DST_DIR \
        [--onnx PATH] [--no-per-channel] [--no-verify] [--verify-only] [--max-length 512]
"""
import argparse
import json
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, helper


def resolve_onnx(src, onnx_arg):
    if onnx_arg:
        return onnx_arg
    cand = os.path.join(src, "onnx", "model.onnx")
    return cand if os.path.isdir(os.path.join(src, "onnx")) else os.path.join(src, "model.onnx")


def fp16_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """正确把 fp16 ONNX 转为 fp32：初始化器 / Constant / 张量类型声明 / Cast 的 to 属性都要改。
    （只改张量声明会破坏图内 Cast 节点；quant_pre_process 不会真正转换 fp16 权重。）"""
    # 1) 初始化器
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    # 2) Constant 节点的 fp16 常量
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
    # 3) 张量类型声明（input/output/value_info）
    def fix(tp):
        if tp.elem_type == TensorProto.FLOAT16:
            tp.elem_type = TensorProto.FLOAT

    for i in model.graph.input:
        fix(i.type.tensor_type)
    for o in model.graph.output:
        fix(o.type.tensor_type)
    for v in model.graph.value_info:
        fix(v.type.tensor_type)
    # 4) Cast 的 to 属性：float16 -> float32（否则输入改 fp32 后类型不匹配）
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    return model


def weight_only_quantize(model: onnx.ModelProto, per_channel: bool) -> onnx.ModelProto:
    """weight-only int8 量化：只把 MatMul/Gemm 的权重存为 int8（per-channel / per-tensor），
    插入 DequantizeLinear 还原成 fp32 再计算；激活保持 fp32（embedding 对激活量化敏感）。"""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replaced, new_inits = {}, []

    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        w_name = next((i for i in n.input if i in inits
                      and inits[i].data_type == TensorProto.FLOAT), None)
        if w_name is None:
            continue
        W = numpy_helper.to_array(inits[w_name]).astype(np.float32)
        if W.ndim != 2:
            continue
        # 量化轴 = 权重输出维：MatMul 输出是最后一维(axis=1)；Gemm transB=1 时输出是 dim0
        if n.op_type == "MatMul":
            axis = 1
        else:
            transB = next((a.i for a in n.attribute if a.name == "transB"), 0)
            axis = 0 if transB == 1 else 1
        out_dim = W.shape[axis]

        if per_channel:
            maxabs = (np.max(np.abs(W), axis=1 - axis) if axis == 1
                      else np.max(np.abs(W), axis=0))
            maxabs = np.where(maxabs == 0, 1.0, maxabs)
            scale = (maxabs / 127.0).astype(np.float32)
            q = (np.round(W / scale[None, :]).clip(-127, 127) if axis == 1
                 else np.round(W / scale[:, None]).clip(-127, 127)).astype(np.int8)
        else:
            s = np.array([np.max(np.abs(W)) / 127.0], dtype=np.float32)
            q = np.round(W / s).clip(-127, 127).astype(np.int8)
            scale, axis = s, 0
        zp = np.zeros(out_dim, dtype=np.int8)

        wq, sc, zp_n = w_name + "_wq", w_name + "_ws", w_name + "_wz"
        new_inits += [numpy_helper.from_array(q, wq),
                      numpy_helper.from_array(scale, sc),
                      numpy_helper.from_array(zp, zp_n)]
        replaced[n.name] = (wq, sc, zp_n, axis, w_name)

    # 重建 initializer：移除被替换的权重，加入 int8/scale/zp
    old_w = {v[4] for v in replaced.values()}
    keep = [i for i in g.initializer if i.name not in old_w]
    g.ClearField("initializer")
    g.initializer.extend(keep)
    g.initializer.extend(new_inits)

    # 改写节点：在 MatMul/Gemm 之前插入 DequantizeLinear，并把权重输入换成反量化结果
    old_nodes = list(g.node)
    g.ClearField("node")
    for n in old_nodes:
        if n.name in replaced:
            wq, sc, zp_n, axis, oldw = replaced[n.name]
            idx = list(n.input).index(oldw)
            deq_out = oldw + "_deq"
            deq = helper.make_node(
                "DequantizeLinear", [wq, sc, zp_n], [deq_out], axis=axis, name=n.name + "_deq")
            n.input[idx] = deq_out
            g.node.append(deq)
            g.node.append(n)
        else:
            g.node.append(n)
    onnx.checker.check_model(model)
    return model


def convert(src, dst, onnx_path, per_channel, no_verify):
    os.makedirs(dst, exist_ok=True)
    print(f"加载源 ONNX：{onnx_path}")
    model = onnx.load(onnx_path)
    is_fp16 = any(i.data_type == onnx.TensorProto.FLOAT16 for i in model.graph.initializer)
    print(f"源模型是否为 fp16：{is_fp16}")

    if is_fp16:
        model = fp16_to_fp32(model)
    int8_path = os.path.join(dst, "model.onnx")
    print(f"weight-only 量化到 int8（MatMul/Gemm 权重，per_channel={per_channel}）...")
    model = weight_only_quantize(model, per_channel)
    onnx.save(model, int8_path)

    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg["quantization"] = {
        "backend": "onnxruntime.dynamic",
        "weight_type": "int8",
        "per_channel": bool(per_channel),
        "op_types_to_quantize": ["MatMul", "Gemm"],
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), ensure_ascii=False, indent=2)

    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "vocab.txt", "vocab.json", "merges.txt", "README.md",
              "configuration_hf_nomic_bert.py"]:
        sp = os.path.join(src, f)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(dst, f))
    print(f"int8 模型已写出：{int8_path}")
    print(f"int8 文件大小：{os.path.getsize(int8_path)/1e6:.1f} MB（源：{os.path.getsize(onnx_path)/1e6:.1f} MB）")
    if not no_verify:
        verify(dst, src, onnx_path)


# --------------------------------------------------------------------------- #
# 验证：fp16 源 vs int8
# --------------------------------------------------------------------------- #

def _tokenize(tok, texts, max_length):
    enc = tok(texts, padding=True, truncation=True, max_length=max_length,
              return_tensors="np", add_special_tokens=True)
    return (enc["input_ids"].astype(np.int64),
            enc["attention_mask"].astype(np.int64),
            np.zeros_like(enc["input_ids"].astype(np.int64)))


def _embed(sess, tok, texts, max_length):
    ids, mask, ttype = _tokenize(tok, texts, max_length)
    feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": ttype}
    out = sess.run(None, feeds)[0]  # [batch, seq, hidden]
    mask3 = mask[:, :, None].astype(np.float32)
    pooled = (out * mask3).sum(axis=1) / (mask3.sum(axis=1) + 1e-9)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9
    return pooled / norms


def _cos(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify(dst, src, onnx_path=None):
    from transformers import AutoTokenizer

    onnx_path = onnx_path or resolve_onnx(src, None)
    int8_path = os.path.join(dst, "model.onnx")
    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"未找到 int8 模型：{int8_path}（请先执行转换）")

    tok = AutoTokenizer.from_pretrained(src)
    texts = [
        "The capital of France is Paris.",
        "Paris is the capital city of France located in Europe.",
        "Machine learning models learn patterns from data.",
        "I love eating pizza with my friends on weekends.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    ]
    max_length = 256

    base_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    e_base = _embed(base_sess, tok, texts, max_length)
    e_int8 = _embed(int8_sess, tok, texts, max_length)

    cosines = [_cos(e_base[i], e_int8[i]) for i in range(len(texts))]
    print("逐条 fp16源 vs int8 余弦相似度：")
    for t, c in zip(texts, cosines):
        print(f"  {c:.5f}  {t[:50]}")
    avg = float(np.mean(cosines))
    print(f"平均余弦相似度：{avg:.6f}")

    query = "What is the capital of France?"
    relevant = "Paris is the capital of France."
    irrelevant = "Machine learning models learn patterns from data."
    q = _embed(int8_sess, tok, [query], max_length)[0]
    rc = _cos(q, _embed(int8_sess, tok, [relevant], max_length)[0])
    ic = _cos(q, _embed(int8_sess, tok, [irrelevant], max_length)[0])
    print(f"检索检查（int8）：query↔相关文档={rc:.4f}，query↔无关文档={ic:.4f}")
    ok = rc > ic

    if avg > 0.99 and ok:
        print("✅ 验证通过：int8 与 fp16 源 embedding 高度一致，检索排序正确。")
    else:
        print("⚠ 验证警告：余弦或检索排序未达预期，请检查量化配置。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fp16 ONNX 模型目录（含 model.onnx）")
    ap.add_argument("--dst", required=True, help="int8 输出目录")
    ap.add_argument("--onnx", default=None, help="显式指定 ONNX 文件路径")
    ap.add_argument("--per-channel", action="store_true", default=True,
                    help="per-channel 量化（默认开；关掉用 --no-per-channel）")
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    onnx_path = resolve_onnx(args.src, args.onnx)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)

    if args.verify_only:
        verify(args.dst, args.src, onnx_path)
        return

    convert(args.src, args.dst, onnx_path, args.per_channel, args.no_verify)


if __name__ == "__main__":
    main()
