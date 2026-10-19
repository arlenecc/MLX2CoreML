"""转换后端探测与选择（混合模式）。

优先使用 coremltools 真实转换；MLX 量化模型在 mlx/mlx-lm 可用时
先反量化再真实转换。依赖缺失、架构不支持时自动回退到打包后端，
保证工具始终可用。
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from modconv import real_backend
from modconv.utils import list_model_files

# 注：Qwen3.5 的 text-only 检查点（config.json 由 TextConfig 落盘）model_type 为
# qwen3_5_text / qwen3_5_moe_text，与复合配置（qwen3_5 / qwen3_5_moe）一并支持。
SUPPORTED_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text",
                         "qwen3_5_moe", "qwen3_5_moe_text", "gemma", "gemma2", "phi3"}


def _mlx_available() -> bool:
    try:
        return (importlib.util.find_spec("mlx") is not None
                and importlib.util.find_spec("mlx_lm") is not None)
    except Exception:
        return False


# 真实转换依赖（仅做 find_spec 轻量探测，不真正导入 torch 等重型库）。
# 缓存结果，供 UI 体积预估 / 架构提示复用：避免每次扫描结束都触发一次重型导入
# （import torch 会卡住界面），同时让"预估口径"与 probe_backend 实际选定的后端一致，
# 否则依赖缺失时界面显示"4-bit 压缩估算"、实际却走 1:1 打包后端，造成误导。
_REAL_BACKEND_DEPS = ("torch", "transformers", "coremltools", "numpy", "safetensors")
_real_backend_ready: Optional[bool] = None


def real_backend_available() -> bool:
    """真实转换依赖是否齐备（find_spec 轻量探测，结果缓存）。"""
    global _real_backend_ready
    if _real_backend_ready is None:
        try:
            _real_backend_ready = all(
                importlib.util.find_spec(m) is not None for m in _REAL_BACKEND_DEPS)
        except Exception:
            _real_backend_ready = False
    return _real_backend_ready


# ---------------------------------------------------------------------------
# 新增：源/目标格式维度（支持 HF→ONNX / ONNX→MLX 等转换任务）
# ---------------------------------------------------------------------------
SOURCE_FORMAT_HF = "hf"
SOURCE_FORMAT_ONNX = "onnx"
SOURCE_FORMAT_MLX = "mlx"
SOURCE_FORMAT_GGUF = "gguf"
TARGET_FORMATS = ("coreml", "onnx", "mlx", "gguf")


def detect_source_format(source: Path) -> str:
    """判定源目录/文件的格式。

    - 单文件：.onnx → onnx；.gguf → gguf；其余按 HF 处理；
    - 目录：含 config.json 时，已量化（quantization 字段）或 MLXEmbedders 命名
      的 safetensors → mlx；否则 hf；无 config.json 则看是否含 .onnx/.gguf。
    """
    source = Path(source)
    if source.is_file():
        suf = source.suffix.lower()
        if suf == ".onnx":
            return SOURCE_FORMAT_ONNX
        if suf == ".gguf":
            return SOURCE_FORMAT_GGUF
        return SOURCE_FORMAT_HF
    if source.is_dir():
        if (source / "config.json").is_file():
            try:
                cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cfg = {}
            if isinstance(cfg, dict):
                if "quantization" in cfg:
                    return SOURCE_FORMAT_MLX
                if safetensors_keys_hf_named(source) is False:
                    return SOURCE_FORMAT_MLX
            return SOURCE_FORMAT_HF
        if any(p.suffix.lower() == ".onnx" for p in list_model_files(source)):
            return SOURCE_FORMAT_ONNX
        if any(p.suffix.lower() == ".gguf" for p in list_model_files(source)):
            return SOURCE_FORMAT_GGUF
    return SOURCE_FORMAT_HF


def hf_to_onnx_available() -> bool:
    """HF→ONNX 依赖：torch / transformers / onnx / onnxruntime。"""
    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("torch", "transformers", "onnx", "onnxruntime"))
    except Exception:
        return False


def onnx_to_mlx_available() -> bool:
    """ONNX→MLX（自定义映射）依赖：onnx / mlx。

    直接读 ONNX initializers 映射到 MLX（不依赖 onnx2torch）。
    """
    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("onnx", "mlx"))
    except Exception:
        return False


def hf_to_mlx_available() -> bool:
    """HF→MLX（经 mlx-lm / mlx-embeddings / mlx-vlm）依赖。

    任一 MLX 转换库 + transformers + torch（HF 权重读取）。
    """
    try:
        libs = ("mlx_lm", "mlx_embeddings", "mlx_vlm")
        has_lib = any(importlib.util.find_spec(m) is not None for m in libs)
        return has_lib and importlib.util.find_spec("transformers") is not None \
            and importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def hf_to_gguf_available() -> bool:
    """HF→GGUF 依赖：gguf（写 GGUF）+ transformers（分词器/结构）。"""
    try:
        return (importlib.util.find_spec("gguf") is not None
                and importlib.util.find_spec("transformers") is not None)
    except Exception:
        return False


def mlx_rekey_available() -> bool:
    """MLX→MLX（重命名+量化，纯 MLX 实现）依赖：mlx。"""
    try:
        return importlib.util.find_spec("mlx") is not None
    except Exception:
        return False


# encoder 家族（embedding / reranker 底座）：bge / jina / nomic 等均属此类
ENCODER_MODEL_TYPES = {"bert", "roberta", "xlm-roberta", "xlm_roberta",
                       "camembert", "nomic_bert"}
_ENCODER_ARCH_RE = re.compile(r"(Bert|Roberta|XLMRoberta|NomicBert|Camembert)")


def resolve_sidecar_dir(source: Path, max_up: int = 3) -> Path:
    """为单文件源（.onnx / .gguf）定位伴随目录（config.json / tokenizer 所在）。

    优先文件所在目录；若其中没有 config.json，则逐级向上查找——Optimum 导出常把
    config.json / tokenizer.json 放在 ``onnx/`` 子目录的**父目录**。
    目录源或找不到时原样返回其所在目录。
    """
    src_dir = source.parent if source.is_file() else Path(source)
    d = src_dir
    for _ in range(max_up):
        if (d / "config.json").is_file():
            return d
        if d.parent == d:
            break
        d = d.parent
    return src_dir


def is_encoder_family(cfg: dict) -> bool:
    """判定 config 是否为 encoder 家族（embedding/reranker 底座）。"""
    if not isinstance(cfg, dict):
        return False
    mt = str(cfg.get("model_type", "") or "").lower()
    if mt in ENCODER_MODEL_TYPES:
        return True
    archs = " ".join(list(cfg.get("architectures") or []))
    if archs and _ENCODER_ARCH_RE.search(archs):
        return True
    # jina-reranker-v2 类源：无 model_type/architectures，仅 auto_map（已验证按 encoder 处理）
    return bool(not mt and not archs and cfg.get("auto_map"))


def safetensors_keys_hf_named(source: Path) -> Optional[bool]:
    """读首个 safetensors 的键名判定权重命名风格（仅读键，不载入数据）。

    True=HF 命名（roberta./bert. 前缀或 attention.self.，含无前缀 BERT）；
    False=MLXEmbedders 命名（encoder.layers.N.attention.query_proj 等）；
    None=无法判定（无 safetensors 或读取失败）。
    """
    files = list_model_files(Path(source), "*.safetensors")
    if not files:
        return None
    try:
        from safetensors import safe_open
        with safe_open(str(files[0]), framework="numpy") as f:
            keys = list(f.keys())
    except Exception:
        return None
    if any(k.startswith(("roberta.", "bert.", "module."))
           or "attention.self." in k for k in keys):
        return True
    if any("attention.query_proj" in k or "encoder.layers." in k for k in keys):
        return False
    return None


@dataclass(frozen=True)
class BackendPlan:
    name: str  # "coremltools" | "package" | "onnx" | "mlx" | "mlx_libs" | "mlx_rekey" | "gguf"
    # onnx 后端按 task 分派 hf->onnx / onnx->onnx / mlx->onnx / gguf->onnx
    reason: str = ""
    model_type: Optional[str] = None
    task: str = "hf->coreml"   # "hf->coreml" | "hf->onnx" | "onnx->mlx" | "hf->mlx" | "hf->gguf"
    target: str = "coreml"      # "coreml" | "onnx" | "mlx" | "gguf"
    available: bool = True      # 依赖齐备/源-目标匹配；False 时调用方应明确报错而非静默

    def describe(self) -> str:
        if not self.available:
            return f"无法执行该转换：{self.reason}"
        if self.name == "coremltools":
            base = f"检测到支持的架构 '{self.model_type}'，使用 coremltools 进行真实转换"
            return f"{base}（{self.reason}）" if self.reason else base
        if self.task == "hf->onnx":
            return f"HF→ONNX（{self.reason}）" if self.reason else "HF→ONNX 转换"
        if self.task == "onnx->onnx":
            return f"ONNX→ONNX（{self.reason}，embedding/reranker 友好）" \
                if self.reason else "ONNX→ONNX 转换（weight-only int8）"
        if self.task == "onnx->mlx":
            return "ONNX→MLX 转换（encoder 类 embedding/reranker，MLXEmbedders 命名）"
        if self.task == "mlx->mlx":
            return f"MLX→MLX（重命名为 MLXEmbedders 命名 + 量化；{self.reason}）" \
                if self.reason else "MLX→MLX（重命名为 MLXEmbedders 命名 + 量化）"
        if self.task == "hf->mlx":
            return f"HF→MLX（{self.reason}）" if self.reason else "HF→MLX 转换"
        if self.task == "hf->gguf":
            return f"HF→GGUF（{self.reason}）" if self.reason else "HF→GGUF 转换"
        if self.task == "onnx->gguf":
            return "ONNX→GGUF 转换（权重取自 .onnx，结构/分词器取自同目录）"
        if self.task == "mlx->gguf":
            return f"MLX→GGUF（{self.reason}）" if self.reason else "MLX→GGUF 转换"
        if self.task == "gguf->mlx":
            return f"GGUF→MLX（{self.reason}）" if self.reason else "GGUF→MLX 转换（llama 布局）"
        if self.task == "mlx->onnx":
            return f"MLX→ONNX（{self.reason}）" if self.reason else "MLX→ONNX 转换"
        if self.task == "gguf->onnx":
            return f"GGUF→ONNX（{self.reason}）" if self.reason else "GGUF→ONNX 转换（llama 布局）"
        return f"使用打包后端（{self.reason}）"


def detect_model_type(source: Path) -> Optional[str]:
    """轻量读取 config.json 的 model_type（不导入 torch/coremltools）。

    供 UI 线程做"会不会走真实转换"的预判（如体积预估），避免阻塞界面；
    真实后端选择仍以 probe_backend 为准（它还会校验重量级依赖是否可用）。
    """
    try:
        config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    model_type = config.get("model_type")
    return model_type if isinstance(model_type, str) else None


def probe_backend(source: Path, requested: str = "auto", target: str = "coreml") -> BackendPlan:
    """探测源模型目录，决定使用哪个转换后端/任务。

    target: "coreml" | "onnx" | "mlx" | "gguf"（GUI 的目标格式下拉）。
    requested: "auto" 自动探测；"package" 强制打包；其余透传给真实后端。
    """
    source = Path(source)
    src_fmt = detect_source_format(source)

    # ---- HF → ONNX / ONNX → ONNX / MLX → ONNX / GGUF → ONNX ----
    if target == "onnx":
        if src_fmt == SOURCE_FORMAT_ONNX:
            # ONNX 源 → ONNX int8（weight-only，nomic-embed 等已验证）
            if not hf_to_onnx_available():
                return BackendPlan("onnx",
                                   reason="缺少 ONNX→ONNX 依赖（onnx/onnxruntime）",
                                   task="onnx->onnx", target="onnx", available=False)
            return BackendPlan("onnx", task="onnx->onnx", target="onnx",
                               available=True, reason="weight-only int8")
        if src_fmt == SOURCE_FORMAT_MLX:
            # MLX → ONNX：归一化为 HF 目录（反量化 + 命名映射）后 torch 导出
            if not hf_to_onnx_available():
                return BackendPlan("onnx",
                                   reason="缺少 MLX→ONNX 依赖（torch/transformers/onnx/onnxruntime）",
                                   task="mlx->onnx", target="onnx", available=False)
            return BackendPlan("onnx", task="mlx->onnx", target="onnx",
                               available=True, reason="归一化为 HF 后导出")
        if src_fmt == SOURCE_FORMAT_GGUF:
            # GGUF → ONNX：llama 布局反量化 + 重建 config/tokenizer 后 torch 导出
            if not (hf_to_onnx_available() and importlib.util.find_spec("gguf")):
                return BackendPlan("onnx",
                                   reason="缺少 GGUF→ONNX 依赖（gguf + torch/transformers/onnx）",
                                   task="gguf->onnx", target="onnx", available=False)
            return BackendPlan("onnx", task="gguf->onnx", target="onnx",
                               available=True, reason="llama 布局")
        if not hf_to_onnx_available():
            return BackendPlan("onnx",
                               reason="缺少 HF→ONNX 依赖（torch/transformers/onnx/onnxruntime）",
                               task="hf->onnx", target="onnx", available=False)
        mt = detect_model_type(source)
        return BackendPlan("onnx", model_type=mt, task="hf->onnx", target="onnx",
                           available=True, reason=f"架构 '{mt}'" if mt else "")

    # ---- ONNX / GGUF / MLX / HF → MLX ----
    if target == "mlx":
        if src_fmt == SOURCE_FORMAT_ONNX:
            if not onnx_to_mlx_available():
                return BackendPlan("mlx", reason="缺少 ONNX→MLX 依赖（onnx/mlx）",
                                   task="onnx->mlx", target="mlx", available=False)
            return BackendPlan("mlx", task="onnx->mlx", target="mlx", available=True)
        if src_fmt == SOURCE_FORMAT_GGUF:
            # GGUF → MLX：归一化为 HF 目录（反量化 + 重建 config/tokenizer）后走 mlx-lm
            if not (hf_to_mlx_available() and importlib.util.find_spec("gguf")):
                return BackendPlan("mlx_libs",
                                   reason="缺少 GGUF→MLX 依赖（gguf + mlx-lm + transformers/torch）",
                                   task="gguf->mlx", target="mlx", available=False)
            return BackendPlan("mlx_libs", task="gguf->mlx", target="mlx",
                               available=True, reason="llama 布局")
        # 目录源（hf / mlx）：encoder 家族（bge/jina/nomic 等 embedding/reranker 底座）
        try:
            cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = None
        if cfg is not None and is_encoder_family(cfg):
            hf_named = safetensors_keys_hf_named(source)
            already_quant = "quantization" in cfg
            # 已量化或已是 MLXEmbedders 命名 → 重命名/重量化后端（纯 MLX）
            if (already_quant or hf_named is False) and mlx_rekey_available():
                return BackendPlan("mlx_rekey", model_type=cfg.get("model_type"),
                                   task="mlx->mlx", target="mlx", available=True)
            # HF 命名优先官方库路径；库缺失时回退纯 MLX 重命名后端
            if hf_named and hf_to_mlx_available():
                return BackendPlan("mlx_libs", model_type=cfg.get("model_type"),
                                   task="hf->mlx", target="mlx", available=True,
                                   reason=f"架构 '{cfg.get('model_type')}'")
            if mlx_rekey_available():
                return BackendPlan("mlx_rekey", model_type=cfg.get("model_type"),
                                   task="mlx->mlx", target="mlx", available=True,
                                   reason="mlx-embeddings 缺失，走纯 MLX 重命名后端")
            return BackendPlan(
                "mlx_rekey",
                reason="缺少 encoder 模型转换依赖（mlx 或 mlx-embeddings + transformers + torch）",
                model_type=cfg.get("model_type"), task="mlx->mlx", target="mlx",
                available=False)
        # 非 encoder（causal / vlm）：走 mlx-lm / mlx-embeddings / mlx-vlm
        if not hf_to_mlx_available():
            return BackendPlan("mlx_libs",
                               reason="缺少 HF→MLX 依赖（mlx-lm/mlx-embeddings/mlx-vlm + transformers + torch）",
                               task="hf->mlx", target="mlx", available=False)
        mt = detect_model_type(source)
        return BackendPlan("mlx_libs", model_type=mt, task="hf->mlx", target="mlx",
                           available=True, reason=f"架构 '{mt}'" if mt else "")

    # ---- HF/MLX/ONNX → GGUF ----
    if target == "gguf":
        if src_fmt == SOURCE_FORMAT_ONNX:
            # ONNX 文件：权重来自 .onnx，结构与分词器取自同目录的 config.json/tokenizer.json
            if not hf_to_gguf_available():
                return BackendPlan("gguf", reason="缺少 GGUF 依赖（gguf/transformers）",
                                   task="onnx->gguf", target="gguf", available=False)
            return BackendPlan("gguf", task="onnx->gguf", target="gguf",
                               available=True, reason="ONNX→GGUF")
        if src_fmt == SOURCE_FORMAT_GGUF:
            return BackendPlan("gguf",
                               reason="GGUF→GGUF（重量化）暂不支持，请先转为 HF/MLX/ONNX",
                               task="gguf->gguf", target="gguf", available=False)
        if src_fmt != SOURCE_FORMAT_HF and src_fmt != SOURCE_FORMAT_MLX:
            return BackendPlan("gguf", reason="GGUF 目标需 HF/MLX 目录或 .onnx 文件",
                               task="hf->gguf", target="gguf", available=False)
        if not hf_to_gguf_available():
            return BackendPlan("gguf", reason="缺少 HF→GGUF 依赖（gguf/transformers）",
                               task="hf->gguf", target="gguf", available=False)
        mt = detect_model_type(source)
        return BackendPlan("gguf", model_type=mt,
                           task=("mlx->gguf" if src_fmt == SOURCE_FORMAT_MLX else "hf->gguf"),
                           target="gguf", available=True,
                           reason=f"架构 '{mt}'" if mt else "")

    # ---- HF → Core ML（既有逻辑）----
    if requested == "package":
        return BackendPlan("package", "用户指定打包后端")

    config_path = source / "config.json"
    if not config_path.is_file():
        return BackendPlan("package", "缺少 config.json，不是 HF 架构的模型目录")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return BackendPlan("package", f"config.json 解析失败: {exc}")
    if not isinstance(config, dict):
        return BackendPlan("package", "config.json 格式异常")
    model_type = config.get("model_type")
    if not model_type or not isinstance(model_type, str):
        return BackendPlan("package", "config.json 缺少 model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        return BackendPlan("package", f"暂不支持的架构 '{model_type}'")

    try:
        real_backend._import_heavy()
    except Exception as exc:
        return BackendPlan("package", f"缺少重量级依赖（torch/transformers/coremltools）: {exc}")

    if "quantization" in config:
        # MLX 量化模型：需要 mlx-lm 先反量化为浮点权重再转换
        if not _mlx_available():
            return BackendPlan("package",
                               "MLX 量化模型需要 mlx/mlx-lm 反量化，但依赖不可用")
        return BackendPlan("coremltools", reason="MLX 量化模型：先反量化再用 coremltools 转换",
                           model_type=model_type, task="hf->coreml", target="coreml")
    return BackendPlan("coremltools", model_type=model_type, task="hf->coreml", target="coreml")
