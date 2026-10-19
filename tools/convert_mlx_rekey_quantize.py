#!/usr/bin/env python3
"""
convert_mlx_rekey_quantize.py
=============================

通用 MLX 重命名 + 量化工具。把一个「HF roberta/bert 命名」的 MLX 权重目录
（如 bge-reranker-v2-m3-mlx-fp16，键名 roberta.encoder.layer.N...）重命名为
MLXEmbedders roberta 命名（encoder.layers.N.attention.query_proj...），并可选量化。

用途（已产出 bge-reranker-v2-m3-mlx-fp16-fixed / -int8-fixed / -int4）：
- 重命名：剥离 roberta. 前缀；encoder.layer.->encoder.layers.；
  attention.self.{query,key,value}.->attention.{q,k,v}_proj.；
  attention.output.dense.->attention.out_proj.；attention.output.LayerNorm.->ln1.；
  output.LayerNorm.->ln2.；intermediate.dense.->linear1.；output.dense.->linear2.；
  embeddings.LayerNorm.->embeddings.norm.；丢弃 pooler.*。
- 量化：所有 2D 线性/嵌入权重量化（mx.quantize），1D 偏置/LayerNorm 保留 fp16；
  若源已是量化(uint32)则仅重命名不重复量化。
- config.json：复制并写入 quantization 字段（fp16 时不写 / 写 bits=16）。

用法：
    python3 convert_mlx_rekey_quantize.py --src SRC_MLX_DIR --dst DST_DIR
                                        [--bits 4 --group-size 64 --mode affine]
                                        [--no-quantize] [--no-verify]
"""
import argparse
import json
import os
import re
import shutil

import mlx.core as mx


def rekey(k):
    """HF roberta/bert 键 -> MLXEmbedders roberta 键。返回 None 表示丢弃。"""
    if "pooler" in k:
        return None
    if k.startswith("roberta."):
        k = k[len("roberta."):]
    k = k.replace("encoder.layer.", "encoder.layers.")
    # 注意顺序：更具体的子串先替换
    k = k.replace("attention.output.dense.", "attention.out_proj.")
    k = re.sub(r"attention\.self\.(query|key|value)\.", r"attention.\1_proj.", k)
    k = k.replace("attention.output.LayerNorm.", "ln1.")
    k = k.replace("output.LayerNorm.", "ln2.")
    k = k.replace("intermediate.dense.", "linear1.")
    k = k.replace("output.dense.", "linear2.")
    k = k.replace("embeddings.LayerNorm.", "embeddings.norm.")
    return k


def convert(src, dst, bits, group_size, mode, no_quantize):
    os.makedirs(dst, exist_ok=True)
    raw = mx.load(os.path.join(src, "model.safetensors"))
    state = {}
    quantized = (not no_quantize) and bits in (4, 8, 5, 6, 3, 2)
    n_2d = 0
    for k, v in raw.items():
        nk = rekey(k)
        if nk is None:
            continue
        if v.dtype == mx.uint32:
            # 已是量化权重：仅重命名，不重复量化
            state[nk] = v
            continue
        if quantized and nk.endswith(".weight") and v.ndim == 2:
            a = v.astype(mx.float16) if v.dtype != mx.float16 else v
            wq, sc, bi = mx.quantize(a, group_size, bits, mode=mode)
            base = nk[:-len(".weight")]
            state[f"{base}.weight"] = wq
            state[f"{base}.scales"] = sc
            state[f"{base}.biases"] = bi
            n_2d += 1
        else:
            state[nk] = v.astype(mx.float16) if v.dtype != mx.float16 else v
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), state)

    # config.json
    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"model_type": "xlm-roberta"}
    if quantized:
        cfg["quantization"] = {"bits": bits, "group_size": group_size, "mode": mode}
    else:
        cfg.pop("quantization", None)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), ensure_ascii=False, indent=2)

    # tokenizer 等
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"]:
        sp = os.path.join(src, f)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(dst, f))
    # 若源带 adapters 也拷
    if os.path.isdir(os.path.join(src, "adapters")):
        shutil.copytree(os.path.join(src, "adapters"), os.path.join(dst, "adapters"),
                        dirs_exist_ok=True)
    print(f"重命名+量化完成：{n_2d} 个 2D 权重量化，共 {len(state)} 张量 -> {dst}")
    return state


def verify(dst, ref=None):
    raw = mx.load(os.path.join(dst, "model.safetensors"))
    keys = sorted(raw.keys())
    bad = [k for k in keys if k.startswith("roberta.") or "pooler" in k
           or "attention.self." in k or "intermediate.dense" in k or "output.dense" in k]
    qw = [k for k in keys if k.endswith(".weight") and raw[k].dtype == mx.uint32]
    print(f"输出张量数：{len(keys)}；残留旧命名/HF 命名：{len(bad)}；量化权重数：{len(qw)}")
    if bad:
        print("  ⚠ 残留旧键：", bad[:8])
    if ref:
        ref_keys = set(mx.load(os.path.join(ref, "model.safetensors")).keys())
        missing = ref_keys - set(keys)
        extra = set(keys) - ref_keys
        print(f"与参考 {ref} 对比：missing={len(missing)} extra={len(extra)}")
        if missing:
            print("  missing:", sorted(missing)[:8])
    if not bad:
        print("验证通过：键名均已转换为 MLXEmbedders roberta 命名。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--mode", default="affine")
    ap.add_argument("--no-quantize", action="store_true", help="仅重命名，不量化（fp16-fixed）")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--ref", default=None, help="参考交付物目录，用于键名对比验证")
    args = ap.parse_args()
    convert(args.src, args.dst, args.bits, args.group_size, args.mode, args.no_quantize)
    if not args.no_verify:
        verify(args.dst, args.ref)


if __name__ == "__main__":
    main()
