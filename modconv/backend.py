"""转换后端探测与选择（混合模式）。

优先使用 coremltools 真实转换；MLX 量化模型在 mlx/mlx-lm 可用时
先反量化再真实转换。依赖缺失、架构不支持时自动回退到打包后端，
保证工具始终可用。
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from modconv import real_backend

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


@dataclass(frozen=True)
class BackendPlan:
    name: str  # "coremltools" | "package"
    reason: str = ""
    model_type: Optional[str] = None

    def describe(self) -> str:
        if self.name == "coremltools":
            base = f"检测到支持的架构 '{self.model_type}'，使用 coremltools 进行真实转换"
            return f"{base}（{self.reason}）" if self.reason else base
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


def probe_backend(source: Path, requested: str = "auto") -> BackendPlan:
    """探测源模型目录，决定使用真实转换还是打包后端。

    requested: "auto" 自动探测；"package" 强制打包；"coremltools" 仍会校验
    依赖与架构，不满足时回退打包。
    """
    source = Path(source)
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
                           model_type=model_type)

    return BackendPlan("coremltools", model_type=model_type)
