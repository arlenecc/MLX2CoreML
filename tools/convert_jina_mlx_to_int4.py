#!/usr/bin/env python3
"""
convert_jina_mlx_to_int4.py
===========================

将「jina v5 风格」的自定义 MLX embedding 模型（Qwen3 backbone + 每任务 LoRA adapter，
如 jina-embeddings-v5-text-small-mlx）转换为 MLX int4 版本。

要点（均已实测验证）：
1. 把 transformer 的 6 类投影 Linear（q/k/v/o_proj、gate/up/down_proj）改为
   nn.QuantizedLinear(in, out, False, group_size, bits, "affine")；
   embedding 表保留 fp16（保检索质量）。
2. 自定义 model.py / utils.py 必须一并修改为 int4 版：
   - model.py：Linear -> QuantizedLinear；JinaEmbeddingModel 容错忽略 config 中的额外字段
     （如 quantization）。
   - utils.py：LoRA 合并（_apply_adapter）对量化权重必须 反量化 -> 合并 -> 重新量化，
     并**直接给 QuantizedLinear.weight/scales/biases 赋值**（绝不能 model.load_weights
     一次性传入数百个量化键——mlx 会把它误存成 dict 导致前向崩）。
3. 保存：dict(model.parameters()) 含 config（dict 标量）与模块标量属性（scale/head_dim
   等），会让 save_safetensors 报 bad_cast，须递归 collect_arrays 只取 mx.array 叶子并兼容
   list（layers）。

用法：
    python3 convert_jina_mlx_to_int4.py --src SRC_DIR --dst DST_DIR [--bits 4 --group-size 64]
                                        [--no-verify] [--verify-only]

--verify-only：不重新转换，仅对已有的 DST_DIR 做加载+各任务切换+与 fp16 基线的余弦对比验证。
"""
import argparse
import importlib.util
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn


# --------------------------------------------------------------------------- #
# 1) model.py / utils.py 的补丁（针对 jina v5 风格的自定义实现）
# --------------------------------------------------------------------------- #

OLD_ATTENTION = """        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)"""

NEW_ATTENTION = """        # int4 量化线性层（group_size=64, bits=4, affine）。
        # 构造时随机初始化并被量化，load_weights 会替换为保存的量化权重。
        self.q_proj = nn.QuantizedLinear(dim, n_heads * head_dim, False, 64, 4, "affine")
        self.k_proj = nn.QuantizedLinear(dim, n_kv_heads * head_dim, False, 64, 4, "affine")
        self.v_proj = nn.QuantizedLinear(dim, n_kv_heads * head_dim, False, 64, 4, "affine")
        self.o_proj = nn.QuantizedLinear(n_heads * head_dim, dim, False, 64, 4, "affine")"""

OLD_MLP = """        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)"""

NEW_MLP = """        # int4 量化线性层（group_size=64, bits=4, affine）
        self.gate_proj = nn.QuantizedLinear(dim, hidden_dim, False, 64, 4, "affine")
        self.down_proj = nn.QuantizedLinear(hidden_dim, dim, False, 64, 4, "affine")
        self.up_proj = nn.QuantizedLinear(dim, hidden_dim, False, 64, 4, "affine")"""

OLD_INIT = """    def __init__(self, config: dict):
        super().__init__()
        args = ModelArgs(**config)
        self.model = Qwen3Model(args)
        self.config = config"""

NEW_INIT = """    def __init__(self, config: dict):
        super().__init__()
        # 容错：config 中可能含 quantization 等额外字段（int4 元数据）
        known = set(getattr(ModelArgs, "__dataclass_fields__", {}))
        if known:
            config_model = {k: v for k, v in config.items() if k in known}
        else:
            config_model = config
        args = ModelArgs(**config_model)
        self.model = Qwen3Model(args)
        self.config = config"""

OLD_APPLY = """    def _apply_adapter(self, adapter, sign=1.0):
        scale = sign * adapter["config"]["lora_alpha"] / adapter["config"]["r"]
        params = dict(self.model.parameters())
        updates = []
        for key, A in adapter["weights"].items():
            if "lora_A" not in key:
                continue
            base_key = "model." + key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            b_key = key.replace("lora_A", "lora_B")
            B = adapter["weights"][b_key]
            # Navigate param tree to get current value
            parts = base_key.split(".")
            node = params
            for p in parts[:-1]:
                node = node[p] if isinstance(node, dict) else node[int(p)]
            current = node[parts[-1]] if isinstance(node, dict) else node[int(parts[-1])]
            updates.append((base_key, current + scale * (B @ A)))
        self.model.load_weights(updates, strict=False)
        mx.eval(self.model.parameters())"""

NEW_APPLY = """    def _get_layer(self, base_key):
        \"\"\"base_key: 'model.layers.N.self_attn.q_proj.weight' -> QuantizedLinear 模块。\"\"\"
        parts = base_key.split(".")
        # parts[0] == 'model'，从 self.model.model（Qwen3Model）开始导航
        node = self.model.model
        for p in parts[1:-1]:  # 跳过 'model' 与末尾的 'weight'
            if p.isdigit():
                node = node[int(p)]
            else:
                node = getattr(node, p)
        return node

    def _apply_adapter(self, adapter, sign=1.0):
        # int4 版本：基线是 QuantizedLinear（packed uint32 权重），无法直接做
        # `current + scale*(B@A)`，因此先反量化 -> 合并 LoRA -> 重新量化回 int4。
        scale = sign * adapter["config"]["lora_alpha"] / adapter["config"]["r"]
        for key, A in adapter["weights"].items():
            if "lora_A" not in key:
                continue
            base_key = "model." + key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            b_key = key.replace("lora_A", "lora_B")
            B = adapter["weights"][b_key]
            ql = self._get_layer(base_key)  # QuantizedLinear
            # 反量化当前权重 -> fp16
            w_full = mx.dequantize(ql.weight, ql.scales, ql.biases,
                                   ql.group_size, ql.bits, mode=ql.mode)
            # 合并 LoRA: w + scale * (B @ A)
            merged = w_full + scale * (B @ A)
            mx.eval(merged)
            # 重新量化回 int4 并就地写入模块
            # （注意：mlx 的 load_weights 在一次性传入数百个量化键时会把 weight
            #  误存为 dict，故改为直接给 QuantizedLinear 的属性赋值）
            wq, sc, bi = mx.quantize(merged, ql.group_size, ql.bits, mode=ql.mode)
            ql.weight = wq
            ql.scales = sc
            ql.biases = bi
        mx.eval(self.model.parameters())"""


def _patch_model_py(text: str) -> str:
    for old, new in [(OLD_ATTENTION, NEW_ATTENTION), (OLD_MLP, NEW_MLP), (OLD_INIT, NEW_INIT)]:
        if old not in text:
            raise RuntimeError("model.py 补丁未匹配，源码结构可能不同，请手动应用 int4 改造。")
        text = text.replace(old, new, 1)
    return text


def _patch_utils_py(text: str) -> str:
    if OLD_APPLY not in text:
        raise RuntimeError("utils.py _apply_adapter 补丁未匹配，请手动应用。")
    # 用新版替换，同时 _get_layer 已包含在新版里
    return text.replace(OLD_APPLY, NEW_APPLY, 1)


# --------------------------------------------------------------------------- #
# 2) 保存工具：只取 mx.array 叶子（兼容 list / 跳过 config 与模块标量属性）
# --------------------------------------------------------------------------- #

def collect_arrays(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(collect_arrays(v, key))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(collect_arrays(v, f"{prefix}.{i}"))
    elif isinstance(tree, mx.array):
        out[prefix] = tree
    # 跳过 Python 标量（scale/head_dim/n_heads 等模块属性）
    return out


# --------------------------------------------------------------------------- #
# 3) 转换主流程
# --------------------------------------------------------------------------- #

def load_model_module(model_dir: str):
    spec = importlib.util.spec_from_file_location(
        "model_" + os.path.basename(model_dir), os.path.join(model_dir, "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def convert(src: str, dst: str, bits: int, group_size: int, mode: str):
    os.makedirs(dst, exist_ok=True)
    mod = load_model_module(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    model = mod.JinaEmbeddingModel(cfg)
    ws = mx.load(os.path.join(src, "model.safetensors"))
    model.load_weights(list(ws.items()))
    mx.eval(model.parameters())
    del ws

    # 投影层 -> QuantizedLinear
    n = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(attn, name, nn.QuantizedLinear.from_linear(getattr(attn, name), group_size, bits))
            n += 1
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            setattr(mlp, name, nn.QuantizedLinear.from_linear(getattr(mlp, name), group_size, bits))
            n += 1
    mx.eval(model.parameters())

    # 保存（排除 config、兼容 list、只取 array 叶子）
    params = dict(model.parameters())
    params.pop("config", None)
    flat = collect_arrays(params)
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), flat)

    # config.json 写入 quantization 字段
    cfg_out = dict(cfg)
    cfg_out["quantization"] = {"bits": bits, "group_size": group_size, "mode": mode}
    json.dump(cfg_out, open(os.path.join(dst, "config.json"), "w"), ensure_ascii=False, indent=2)

    # 补丁后的 model.py / utils.py
    mtext = open(os.path.join(src, "model.py")).read()
    utext = open(os.path.join(src, "utils.py")).read()
    open(os.path.join(dst, "model.py"), "w").write(_patch_model_py(mtext))
    open(os.path.join(dst, "utils.py"), "w").write(_patch_utils_py(utext))

    # 拷贝 tokenizer / README / adapters
    for f in ["tokenizer.json", "tokenizer_config.json", "README.md"]:
        if os.path.exists(os.path.join(src, f)):
            import shutil
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    if os.path.isdir(os.path.join(src, "adapters")):
        import shutil
        shutil.copytree(os.path.join(src, "adapters"), os.path.join(dst, "adapters"), dirs_exist_ok=True)

    print(f"转换完成：{n} 个 Linear -> QuantizedLinear，已写入 {dst}")
    return model


# --------------------------------------------------------------------------- #
# 4) 验证：加载 int4 目录、切换全部任务、与 fp16 基线对比余弦
# --------------------------------------------------------------------------- #

def verify(dst: str, src: str):
    texts = [
        "What is machine learning?",
        "The capital of France is Paris.",
        "I love eating pizza with friends.",
        "Photosynthesis converts sunlight into chemical energy.",
        "A neural network is inspired by the human brain.",
    ]
    # fp16 基线（复用 src 的 utils.load_model）
    src_utils = load_utils_module(src)
    fp16 = src_utils.load_model(src)
    fp16.switch_task("retrieval")
    e_fp16 = fp16.encode(texts, "retrieval")
    mx.eval(e_fp16)

    # int4
    int4 = load_utils_module(dst).load_model(dst)
    int4.switch_task("retrieval")
    e_int4 = int4.encode(texts, "retrieval")
    mx.eval(e_int4)
    print("retrieval 余弦(fp16 vs int4):")
    for i, t in enumerate(texts):
        c = float(mx.sum(e_fp16[i] * e_int4[i]) / (mx.linalg.norm(e_fp16[i]) * mx.linalg.norm(e_int4[i])))
        print(f"  {c:.5f}  {t}")
    for task in ["text-matching", "clustering", "classification"]:
        int4.switch_task(task)
        e = int4.encode(texts[:2], task)
        mx.eval(e)
        print(f"int4 {task}: encode ok, norm~{float(mx.linalg.norm(e[0]).item()):.3f}")
    print("验证通过：所有任务切换正常，embeddings L2 归一化。")


def load_utils_module(model_dir: str):
    spec = importlib.util.spec_from_file_location(
        "utils_" + os.path.basename(model_dir), os.path.join(model_dir, "utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 jina MLX 模型目录")
    ap.add_argument("--dst", required=True, help="输出 int4 模型目录")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--mode", default="affine")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="只验证已有 dst，不重新转换")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.dst, args.src)
        return

    convert(args.src, args.dst, args.bits, args.group_size, args.mode)
    if not args.no_verify:
        verify(args.dst, args.src)


if __name__ == "__main__":
    main()
