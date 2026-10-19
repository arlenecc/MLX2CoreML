"""把 MLX / GGUF 源归一化为临时 HF 目录，以便复用已有的 HF→{ONNX,MLX,GGUF} 后端。

背景：三个目标后端（torch 导出 ONNX、mlx-lm 转 MLX、gguf 写 GGUF）都消费
**HF 目录**（config.json + model.safetensors 的 HF 命名 + tokenizer.json）。
因此新增源格式只需「归一化成 HF 目录」一层，无需在每个目标后端里重复实现源解析。

覆盖：
- MLX 目录（mlx-lm / mlx-embeddings / 本工具 ONNX→MLX、MLX→MLX 的产物）：
  * causal（mlx-lm）：键名本就是 HF 风格（model.layers.../lm_head），反量化后直接可用；
  * encoder（MLXEmbedders 命名）：反向映射回 HF bert/roberta 命名
    （rekey 的逆，见 mlx_rekey_backend.rekey）；
  * 含 .scales/.biases 兄弟键的量化权重自动反量化。
- GGUF 文件（llama.cpp）：llama 布局张量名 → HF 名，量化张量反量化；
  config.json 由 GGUF 元数据重建；tokenizer.json 由 GGUF 词表/merges 重建
  （已验证分词 ID 与原始完全一致）。

所有函数在临时目录落地，调用方负责清理（见 temp_hf_dir / *_ctx 上下文管理器）。
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from modconv.utils import list_model_files

import numpy as np

# GGUF llama 布局张量名 → HF 名（llama / qwen2 / qwen3 / mistral 统一）
_GGUF_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.bias": "mlp.down_proj.bias",
    # Qwen3 等带 QK-Norm
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

# GGUF general.architecture → HF 因果 LM 架构类名
_GGUF_ARCH_TO_ARCH_CLASS = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2moe": "Qwen2MoeForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3moe": "Qwen3MoeForCausalLM",
    # Qwen3.5 系列（混合线性/全注意力 + 可选视觉塔）。GGUF arch 名以 transformers 写
    # 入的 model_type 为准：纯文本为 qwen3_5_text，多模态为 qwen3_5。
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "mistral": "MistralForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
}

# GGUF general.architecture → HF model_type（本工具导出的 GGUF 一律为 llama）
_GGUF_ARCH_TO_MODEL_TYPE = {
    "llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "mistral": "mistral",
    "mixtral": "mixtral", "gemma": "gemma", "gemma2": "gemma2", "phi3": "phi3",
}

# MLXEmbedders model_type → HF 状态字典前缀
_MLX_PREFIX = {"bert": "bert.", "roberta": "roberta.", "xlm-roberta": "roberta.",
               "xlm_roberta": "roberta.", "camembert": "roberta.",
               "nomic_bert": "bert."}


@contextlib.contextmanager
def temp_hf_dir() -> Iterator[Path]:
    """创建临时目录，退出时递归删除（归一化中间产物）。"""
    d = Path(tempfile.mkdtemp(prefix="modconv_norm_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 公共：safetensors 读取（含 MLX 量化反量化）
# ---------------------------------------------------------------------------
def _read_safetensors_dir(src_dir: Path, hooks, cfg: Optional[dict]
                          = None) -> Dict[str, np.ndarray]:
    """读取目录内全部 safetensors；含 MLX 量化兄弟键时自动反量化为 fp16。"""
    from safetensors import safe_open
    index = src_dir / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text(encoding="utf-8"))
                           ["weight_map"].values()))
    else:
        files = [p.name for p in list_model_files(src_dir, "*.safetensors")]
    tensors: Dict[str, np.ndarray] = {}
    for fname in files:
        fpath = src_dir / fname
        if not fpath.is_file():
            continue
        try:
            with safe_open(str(fpath), framework="numpy") as sf:
                for k in sf.keys():
                    tensors[k] = sf.get_tensor(k)
        except TypeError:
            # numpy 不支持的类型（如 bfloat16）：回退 torch 读取并转 fp16
            import torch
            with safe_open(str(fpath), framework="torch") as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k)
                    if t.dtype == torch.bfloat16:
                        t = t.to(torch.float16)
                    tensors[k] = t.detach().cpu().numpy()
        hooks.on_log(f"已读取权重分片: {fname}")
    if not tensors:
        raise RuntimeError(f"未找到 safetensors 权重文件: {src_dir}")
    if any(k.endswith(".scales") for k in tensors):
        tensors = _dequantize_mlx(tensors, cfg or {}, hooks)
    return tensors


def infer_group_size(packed_shape, scales_shape, bits: int) -> Optional[int]:
    """从打包权重与 scales 的形状推断实际 group_size。

    打包权重末维 = in_dim * bits / 32；scales 末维 = in_dim / group_size。
    故 group_size = in_dim / scales_last_dim。维度不整除 group_size 时转换器会
    回退到更小的 group_size（config 里的值未必是实际值），必须按形状推断。
    """
    try:
        in_dim = int(packed_shape[-1]) * 32 // bits
        groups = int(scales_shape[-1])
        if groups <= 0 or in_dim % groups:
            return None
        return in_dim // groups
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dequantize_mlx(tensors: Dict[str, np.ndarray], cfg: dict,
                    hooks) -> Dict[str, np.ndarray]:
    """MLX 量化权重（weight uint32 + .scales/.biases 兄弟键）→ fp16。"""
    import mlx.core as mx
    qcfg = cfg.get("quantization") or {}
    bits = int(qcfg.get("bits", 4))
    cfg_gs = int(qcfg.get("group_size", 64))
    mode = qcfg.get("mode", "affine")
    out = dict(tensors)
    n = 0
    for k in list(out):
        if not k.endswith(".scales"):
            continue
        base = k[: -len(".scales")]
        wkey, bkey = base + ".weight", base + ".biases"
        if wkey not in out:
            continue
        gs = infer_group_size(np.asarray(out[wkey]).shape,
                              np.asarray(out[k]).shape, bits) or cfg_gs
        # mx.dequantize(w, scales, biases, group_size, bits, mode)
        # 注意：mx.eval() 是就地求值且返回 None，须用 np.array(mx.array) 直转
        dq = mx.dequantize(mx.array(out[wkey]), mx.array(out[k]),
                           mx.array(out[bkey]) if bkey in out else None,
                           gs, bits, mode)
        out[wkey] = np.array(dq).astype(np.float16)
        del out[k]
        if bkey in out:
            del out[bkey]
        n += 1
    if n:
        hooks.on_log(f"已反量化 {n} 个 MLX 量化张量（bits={bits}, group_size 按形状推断）")
    return out


def _write_hf_dir(out_dir: Path, tensors: Dict[str, np.ndarray],
                  config: dict, tokenizer_src: Optional[Path] = None,
                  extra_files: Tuple[str, ...] = ()) -> Path:
    """把 HF 命名张量 + config + tokenizer 落地为 HF 目录。"""
    from safetensors.numpy import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()},
              str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if tokenizer_src is not None and tokenizer_src.is_dir():
        for fname in ("tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.txt",
                      "vocab.json", "merges.txt", *extra_files):
            sp = tokenizer_src / fname
            if sp.is_file():
                shutil.copy(sp, out_dir / fname)
    return out_dir


# ---------------------------------------------------------------------------
# MLX → HF
# ---------------------------------------------------------------------------
def is_mlx_embedders_named(keys) -> bool:
    return any(k.startswith("encoder.layers.") for k in keys)


def mlx_to_hf_name(k: str) -> Optional[str]:
    """MLXEmbedders 键 → HF bert/roberta 键（rekey 的逆）。None 表示丢弃。"""
    k = k.replace("encoder.layers.", "encoder.layer.")
    k = re.sub(r"attention\.(query|key|value)_proj\.", r"attention.self.\1.", k)
    k = k.replace("attention.out_proj.", "attention.output.dense.")
    k = k.replace("ln1.", "attention.output.LayerNorm.")
    k = k.replace("ln2.", "output.LayerNorm.")
    k = k.replace("linear1.", "intermediate.dense.")
    k = k.replace("linear2.", "output.dense.")
    k = k.replace("embeddings.norm.", "embeddings.LayerNorm.")
    # MLXEmbedders 的 BertModel 把 pooler 存为融合的 pooler.{weight,bias}
    if k == "pooler.weight":
        return "pooler.dense.weight"
    if k == "pooler.bias":
        return "pooler.dense.bias"
    return k


def mlx_dir_to_hf_dir(source: Path, hooks, out_dir: Optional[Path] = None
                      ) -> Tuple[Path, Optional[Path]]:
    """把 MLX 模型目录归一化为 HF 目录。

    返回 (hf_dir, 需要调用方清理的自建目录或 None)。推荐用
    ``mlx_to_hf_dir_ctx`` 上下文管理器自动清理。
    """
    src = Path(source)
    cfg_path = src / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"MLX 源目录缺少 config.json: {src}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tensors = _read_safetensors_dir(src, hooks, cfg)

    own = None
    if out_dir is None:
        own = Path(tempfile.mkdtemp(prefix="modconv_mlx2hf_"))
        out_dir = own

    hf_cfg = dict(cfg)
    if is_mlx_embedders_named(tensors):
        # encoder：MLXEmbedders 命名 → HF bert/roberta 命名
        mt = str(hf_cfg.get("model_type", "xlm-roberta"))
        mapped: Dict[str, np.ndarray] = {}
        for k, v in tensors.items():
            nk = mlx_to_hf_name(k)
            if nk is None:
                continue
            mapped[nk] = v
        has_cls = any(k.endswith(("classifier.dense.weight", "classifier.out_proj.weight"))
                      for k in mapped)
        roberta_like = mt in ("roberta", "xlm-roberta", "xlm_roberta", "camembert")
        if has_cls:
            # *ForSequenceClassification 把 base model 包为子模块 → 需要 bert./roberta. 前缀
            prefix = _MLX_PREFIX.get(mt, "roberta.")
            mapped = {prefix + k: v for k, v in mapped.items()}
            hf_cfg["architectures"] = (["XLMRobertaForSequenceClassification"]
                                       if roberta_like
                                       else ["BertForSequenceClassification"])
            ow = next(k for k in mapped if k.endswith("classifier.out_proj.weight"))
            hf_cfg.setdefault("num_labels", int(mapped[ow].shape[0]))
        else:
            # 裸 BertModel/XLMRobertaModel：参数无 base_model_prefix 前缀
            # （transformers 5.x 加载时不自动剥离前缀，加了会全部 missing/extra）
            hf_cfg["architectures"] = (["XLMRobertaModel"] if roberta_like
                                       else ["BertModel"])
        hf_cfg.setdefault("model_type", mt)
        hf_cfg.pop("quantization", None)
        hf_cfg.pop("quantization_config", None)
        hf_cfg.pop("quantization_config_path", None)
        tensors = mapped
    elif any(k.startswith(("vision_tower.", "language_model.")) for k in tensors):
        # VLM（mlx-vlm 风格）：mlx 源内部分两层前缀
        #   language_model.model.*  → HF 的 model.language_model.*
        #   vision_tower.*          → HF 的 model.visual.*
        mapped = {}
        for k, v in tensors.items():
            if k.startswith("vision_tower."):
                nk = "model.visual." + k[len("vision_tower."):]
            elif k.startswith("language_model.model."):
                nk = "model.language_model." + k[len("language_model.model."):]
            elif k.startswith("language_model."):
                nk = "model.language_model." + k[len("language_model."):]
            else:
                nk = k
            # MLX 卷积权重布局：in_channels 在最后（Conv3d=[out,kT,kH,kW,in]、
            # Conv1d=[out,k,in]），HF 为 [out,in,k...]。转回 HF 布局。
            if nk.endswith("patch_embed.proj.weight") and v.ndim == 5:
                v = np.ascontiguousarray(np.transpose(v, (0, 4, 1, 2, 3)))
            elif nk.endswith("conv1d.weight") and v.ndim == 3:
                v = np.ascontiguousarray(np.transpose(v, (0, 2, 1)))
            mapped[nk] = v
        # mlx-vlm 通常把 lm_head 与 embed_tokens 绑定，源里无独立 lm_head.weight；
        # 若确实缺失则声明权重绑定，transformers 会复用 embed_tokens。
        if "lm_head.weight" not in mapped and \
                "model.language_model.embed_tokens.weight" in mapped:
            hf_cfg["tie_word_embeddings"] = True
        hf_cfg.pop("quantization", None)
        hf_cfg.pop("quantization_config", None)
        hf_cfg.pop("quantization_config_path", None)
        hf_cfg["torch_dtype"] = "float16"
        # 保留原 VLM 架构/model_type（如 qwen3_5 / Qwen3_5ForConditionalGeneration）
        tensors = mapped
    else:
        # causal（mlx-lm）：键名本就是 HF 风格，仅清理 config
        # 去掉 MLX 量化相关字段（quantization / quantization_config），
        # 否则 transformers 会尝试按量化模型加载而缺 quant_method 报错
        hf_cfg.pop("quantization", None)
        hf_cfg.pop("quantization_config", None)
        hf_cfg.pop("quantization_config_path", None)
        hf_cfg["torch_dtype"] = "float16"
        hf_cfg.setdefault("architectures", ["LlamaForCausalLM"])
        hf_cfg.setdefault("model_type", "llama")

    _write_hf_dir(out_dir, tensors, hf_cfg, tokenizer_src=src)
    hooks.on_log(f"MLX→HF 归一化完成：{len(tensors)} 个张量 → {out_dir}")
    return out_dir, own


@contextlib.contextmanager
def mlx_to_hf_dir_ctx(source: Path, hooks) -> Iterator[Path]:
    hf_dir, owned = mlx_dir_to_hf_dir(source, hooks)
    try:
        yield hf_dir
    finally:
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)


# ---------------------------------------------------------------------------
# GGUF → HF
# ---------------------------------------------------------------------------
def _gguf_field(reader, name):
    try:
        return reader.get_field(name).contents()
    except Exception:
        return None


def _gguf_tensor_to_hf(reader) -> Dict[str, np.ndarray]:
    """读取 GGUF 全部张量 → HF 命名（fp16），量化张量自动反量化。

    形状约定：``GGUFWriter.add_tensor`` 原样存储传入数组的形状与行主序数据，
    Reader 的 ``tensor.shape`` 是**反转后的 ggml 序**。因此 HF 形状 =
    ``tuple(tensor.shape)[::-1]``，数据按该形状直接 reshape 即可（**不是转置**）。
    """
    from gguf.quants import dequantize as _dq
    from gguf import quant_shape_to_byte_shape

    out: Dict[str, np.ndarray] = {}
    for t in reader.tensors:
        name = t.name
        m = re.match(r"blk\.(\d+)\.(.+)", name)
        if m:
            hf = f"model.layers.{int(m.group(1))}.{_GGUF_TO_HF[m.group(2)]}"
        elif name in _GGUF_TO_HF:
            hf = _GGUF_TO_HF[name]
        else:
            continue
        declared = tuple(int(x) for x in t.shape)[::-1]   # 写入时的 HF 形状
        raw = np.ascontiguousarray(np.asarray(t.data)).reshape(-1)
        tt = int(t.tensor_type)
        if tt == 0:                       # F32
            arr = raw.view(np.float32).reshape(declared)
        elif tt == 1:                     # F16
            arr = raw.view(np.float16).reshape(declared)
        else:                             # 量化：按 byte-shape 反量化后回到 HF 形状
            bshape = quant_shape_to_byte_shape(declared, t.tensor_type)
            arr = np.asarray(_dq(raw.reshape(bshape), t.tensor_type)).reshape(declared)
        out[hf] = np.ascontiguousarray(arr.astype(np.float16))
    if not out:
        raise RuntimeError("GGUF 中未发现可识别的 llama 布局张量")
    return out


def _gguf_tokenizer_json(reader, out_path: Path) -> bool:
    """由 GGUF 词表/merges 重建 tokenizer.json（已验证分词 ID 一致）。"""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    tokens = _gguf_field(reader, "tokenizer.ggml.tokens")
    if not tokens:
        return False
    toks = [str(t) for t in tokens]
    raw_merges = [str(m) for m in (_gguf_field(reader, "tokenizer.ggml.merges") or [])]
    merges = [tuple(m.split(" ", 1)) if " " in m else (m, m) for m in raw_merges]
    tok = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(toks)}, merges=merges))
    # GGUF 记录的分词预处理（default/gpt2 → byte-level BPE）；llama(SPM) 在
    # llama.cpp 侧处理，这里统一按 byte-level 重建，分词 ID 已实测一致。
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    types = _gguf_field(reader, "tokenizer.ggml.token_type") or []
    special = [toks[i] for i, tp in enumerate(types)
               if i < len(toks) and int(tp) in (2, 3, 4)]  # UNKNOWN/CONTROL/USER_DEFINED
    eos = _gguf_field(reader, "tokenizer.ggml.eos_token_id")
    eos_tok = toks[int(eos)] if isinstance(eos, int) and 0 <= int(eos) < len(toks) else None
    PTF = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=eos_tok,
                                  additional_special_tokens=special or None)
    PTF.save_pretrained(str(out_path))
    return True


def gguf_to_hf_dir(source: Path, hooks, out_dir: Optional[Path] = None
                   ) -> Tuple[Path, Optional[Path]]:
    """把 GGUF 文件归一化为 HF 目录。返回 (hf_dir, 自建目录或 None)。"""
    from gguf import GGUFReader
    src = Path(source)
    reader = GGUFReader(str(src))
    arch = _gguf_field(reader, "general.architecture") or "llama"
    model_type = _GGUF_ARCH_TO_MODEL_TYPE.get(str(arch), "llama")

    def _int(name, default=None):
        v = _gguf_field(reader, f"{arch}.{name}")
        return int(v) if isinstance(v, (int, float)) else default

    tensors = _gguf_tensor_to_hf(reader)
    # 偏置自洽：目标模型结构按 model_type 构建（GGUF arch 常被写为通用 "llama"，
    # 而 Llama 的 q/k/v/o_proj 均带 bias）。缺失的偏置补零——与"无偏置"数值等价，
    # 避免 load_weights strict 报 Missing/Extra。
    n_layers = max((int(m.group(1)) for t in reader.tensors
                    if (m := re.match(r"blk\.(\d+)\.", t.name))), default=-1) + 1
    # 按投影类型分别判定偏置是否存在（GGUF 元数据不记录 attention_bias/mlp_bias）。
    # 不能因为 q_proj 有偏置就给 o_proj 也补偏置——部分架构（如 Qwen2/3）只给
    # q/k/v 或完全不给 o_proj 偏置，盲目补齐会让 mlx_lm/transformers 报
    # "parameters not in model"。
    attn_bias = {p: any(k.endswith(f"self_attn.{p}.bias") for k in tensors)
                 for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    mlp_bias = {p: any(k.endswith(f"mlp.{p}.bias") for k in tensors)
                for p in ("gate_proj", "up_proj", "down_proj")}
    shapes = {k: v.shape for k, v in tensors.items()
              if k.endswith("_proj.weight") or k.endswith("embed_tokens.weight")}
    for i in range(n_layers):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            bk = f"model.layers.{i}.self_attn.{p}.bias"
            wk = f"model.layers.{i}.self_attn.{p}.weight"
            if attn_bias[p] and bk not in tensors and wk in shapes:
                tensors[bk] = np.zeros(shapes[wk][0], dtype=np.float16)
        for p in ("gate_proj", "up_proj", "down_proj"):
            bk = f"model.layers.{i}.mlp.{p}.bias"
            wk = f"model.layers.{i}.mlp.{p}.weight"
            if mlp_bias[p] and bk not in tensors and wk in shapes:
                tensors[bk] = np.zeros(shapes[wk][0], dtype=np.float16)
    tied = (not bool(_gguf_field(reader, f"{arch}.tie_word_embeddings") or False)
            and "lm_head.weight" in tensors) is False
    if tied:
        tensors.pop("lm_head.weight", None)

    n_embd = _int("embedding_length") or int(tensors["model.embed_tokens.weight"].shape[-1])
    n_heads = _int("attention.head_count", 4) or 4
    # 依据实际张量判定 bias 开关（GGUF 元数据不记录 attention_bias/mlp_bias）
    has_attn_bias = any(k.endswith("self_attn.q_proj.bias") for k in tensors)
    has_mlp_bias = any(k.endswith("mlp.gate_proj.bias") for k in tensors)
    cfg = {
        "model_type": model_type,
        "architectures": [_GGUF_ARCH_TO_ARCH_CLASS.get(model_type, "LlamaForCausalLM")],
        "hidden_size": n_embd,
        "num_hidden_layers": _int("block_count", 2) or 2,
        "num_attention_heads": n_heads,
        "num_key_value_heads": (_int("attention.head_count_kv")
                                or _int("attention.head_count", 4) or 4),
        "intermediate_size": _int("feed_forward_length", 4 * n_embd) or 4 * n_embd,
        "max_position_embeddings": _int("context_length", 4096) or 4096,
        "vocab_size": int(tensors["model.embed_tokens.weight"].shape[0]),
        "rms_norm_eps": float(_gguf_field(
            reader, f"{arch}.attention.layer_norm_rms_epsilon") or 1e-5),
        "rope_theta": float(_gguf_field(reader, f"{arch}.rope.freq_base") or 10000.0),
        "tie_word_embeddings": bool(tied),
        "torch_dtype": "float16",
    }
    # Qwen3 等：QKV 维度可不等于 hidden/n_heads，必须显式写 head_dim
    head_dim = _int("rope.dimension_count") or (n_embd // n_heads)
    if head_dim and head_dim != n_embd // n_heads:
        cfg["head_dim"] = head_dim
    if has_attn_bias:
        cfg["attention_bias"] = True
    if has_mlp_bias:
        cfg["mlp_bias"] = True
    for key, tid_key in (("bos_token_id", "tokenizer.ggml.bos_token_id"),
                         ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                         ("pad_token_id", "tokenizer.ggml.padding_token_id")):
        v = _gguf_field(reader, tid_key)
        if isinstance(v, int):
            cfg[key] = v

    own = None
    if out_dir is None:
        own = Path(tempfile.mkdtemp(prefix="modconv_gguf2hf_"))
        out_dir = own
    _write_hf_dir(out_dir, tensors, cfg)
    if _gguf_tokenizer_json(reader, out_dir):
        hooks.on_log("已由 GGUF 词表重建 tokenizer.json")
    else:
        hooks.on_log("警告：GGUF 无分词器字段，未生成 tokenizer.json")
    hooks.on_log(f"GGUF→HF 归一化完成：{len(tensors)} 个张量（{model_type}）")
    return out_dir, own


@contextlib.contextmanager
def gguf_to_hf_dir_ctx(source: Path, hooks) -> Iterator[Path]:
    hf_dir, owned = gguf_to_hf_dir(source, hooks)
    try:
        yield hf_dir
    finally:
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)
