"""基于 coremltools 的真实转换后端。

把 HF/MLX 架构的模型（config.json + *.safetensors）通过
PyTorch 追踪 + coremltools 转换为带 KV cache 状态的 ML Program（.mlpackage）。

说明：
- stateful mlprogram：input_ids (1, 1..max_seq_len) + cache_position (1..max_seq_len)
  为动态长度输入，KV cache 以 Core ML State 存储（iOS18 目标），
  prefill 与 decode 共用同一图，Swift 侧无需重算前缀。
- 支持 config.json 中 model_type ∈ SUPPORTED_MODEL_TYPES 的模型；
  MLX 量化模型（config 含 "quantization"）先用 mlx-lm 反量化再转换。
- 转换成功后会写入模型元数据，并在输出目录生成 Swift 调用示例
  （<名称>-swift-example.swift），方便 Xcode/Swift 直接使用。
- 任何阶段失败都会抛异常并向用户报错（不降级、不回退：打包后端产物不是
  可用的 Core ML 模型）。

依赖：torch / transformers / coremltools / numpy / safetensors；
量化模型另需 mlx / mlx-lm（仅 macOS Apple Silicon）。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings
import atexit
import signal
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from modconv.converter import ConversionResult
# ATTENTION_LAYER_TYPES：混合注意力架构的层类型名单，与 kv_wrapper 共用同一份
from modconv.utils import ATTENTION_LAYER_TYPES, format_size

# ---------------------------------------------------------------------------
# 临时产物兜底清理
#
# 转换过程会产生 GB 级的临时包：反量化/归一化的权重目录（可达数十 GB）、
# 以及 coremltools 在系统临时目录下生成的临时 .mlpackage。正常流程里它们由
# 调用点的 finally（`tmp_dirs`）与显式 `_drop_package` 删除；但若进程**异常
# 退出**（崩溃抛异常、被 SIGTERM / kill、未走到 finally），coremltools 自带的
# atexit 清理在长驻 GUI 进程里不可靠，临时包会残留并很快吃满启动盘。
#
# 这里用模块级集合记录所有待清理路径，并在以下两种情况强制兜底删除：
#   1) 进程正常退出 / 抛未捕获异常 / Ctrl-C —— atexit 钩子；
#   2) 收到 SIGTERM / SIGINT —— 终止信号处理器（清完再恢复原处置，
#      让进程仍按预期退出，不改变原有信号语义）。
# ---------------------------------------------------------------------------
_PENDING_CLEANUP: set = set()


def _register_pending(path) -> None:
    """登记一个待清理的临时路径（目录或文件），进程异常退出时兜底删除。"""
    try:
        if path is None:
            return
        _PENDING_CLEANUP.add(str(path))
    except Exception:
        pass


def _unregister_pending(path) -> None:
    """正常删除后从待清理集合移除，避免重复操作。"""
    try:
        _PENDING_CLEANUP.discard(str(path))
    except Exception:
        pass


def _track_temp_package(model_obj) -> None:
    """若 model_obj 是 coremltools 的临时包，登记其路径以便异常退出时兜底清理。"""
    if getattr(model_obj, "is_temp_package", False):
        _register_pending(getattr(model_obj, "package_path", None))


def _cleanup_pending() -> None:
    """删除所有仍登记的临时路径（幂等、忽略一切错误）。"""
    for p in list(_PENDING_CLEANUP):
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.lexists(p):
                os.remove(p)
        except Exception:
            pass
    _PENDING_CLEANUP.clear()


atexit.register(_cleanup_pending)


def _on_fatal_signal(signum, _frame):
    """收到终止信号时先清临时包，再恢复原处置让进程按预期退出。"""
    try:
        _cleanup_pending()
    finally:
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            pass


# 覆盖外部终止（kill / kill -TERM / 系统关机）；非主线程或平台不支持时跳过，
# 不影响正常转换。SIGINT 同理（保留默认 Ctrl-C 行为：清完再让 KeyboardInterrupt 抛出）。
for _sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
    if _sig is not None:
        try:
            signal.signal(_sig, _on_fatal_signal)
        except (ValueError, OSError, RuntimeError):
            pass


SEQ_LEN_CAP = 512

# checkpoint 权重名与模型期望名的最低重合比例：低于此值即判定"权重没加载上"。
# transformers 在名字对不上时不会报错，而是把权重全部新建初始化，导出的会是
# 随机权重的模型——这种静默错误必须在加载前拦下。阈值留得很宽松（正常布局是
# 100%，tie 掉的 lm_head / 缺失的 MTP 模块只会掉几个百分点）。
# ANE 执行计划构建失败（-14）时自动降长度重试的次数与长度下限。
# 图里随上下文增长的部分（pad 后的静态激活、KV/conv/rec state、因果掩码）会让
# ANE 计划构建失败，而"能构建的最大长度"随模型与机型而异（实测同一模型 512 失败、
# 256 通过），只能试出来。因此自动收敛，而不是把失败丢给用户手动试长度；每次重试
# 都是一次完整转换（含 ANE 编译，耗时可观），故次数与幅度都受限。
_MAX_SEQ_RETRIES = 3
_MIN_SEQ_RETRY = 64

_WEIGHT_MATCH_RATIO = 0.5

# 复合（多模态）checkpoint 里文本 LM 权重的常见前缀：HF 侧期望 "model.layers.*"，
# 而 MLX 的多模态布局把它们放在 "language_model." 之下（并带有 vision_tower.*）。
_TEXT_WEIGHT_PREFIXES = ("", "language_model.", "model.language_model.")

# Qwen3.5 系零中心 RMSNorm（HF 的 Qwen3_5RMSNorm 计算 1.0 + weight）的权重后缀，
# 与 mlx-lm sanitize 折叠 +1.0 的名单逐条一致（mlx_lm/models/qwen3_5.py 的
# norm_keys）。GDN 内部的 linear_attn.norm 是 Qwen3_5RMSNormGated（HF 侧本就不加
# +1），不在此列——两侧约定相同，动它才会错。
_FOLDED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_FOLDED_NORM_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")

_SWIFT_EXAMPLE_TEMPLATE = '''// 由 modconv 自动生成 — 模型 "{model_name}" 的 Swift 调用示例
// 单步 decode 模型：每次前向推进**一个** token，prefill 由循环调用完成
// （recurrent prefill）。KV cache 存于 Core ML State；混合架构的 GDN 状态
// （conv_states / recurrent_states）走普通张量，逐步传入、取回再传下一步。
// 输入: input_ids (Int32, [1, 1])、cache_position (Int32, [1])，取值 0..{max_seq_len_minus_1}
//       conv_states / recurrent_states（混合架构才有）
// 输出: {output_name} (Float, [1, 1, vocab_size]) + 下一步要回传的状态
// 注意: 需先用分词器（如 swift-transformers）把文本转成 token ids。
// 注意: 首次 prediction 会做 ANE 程序特化（大模型可能数分钟），建议 App 启动时
//       先用任意合法 token 做一次预热前向。
import CoreML
import Foundation

enum ConvertedModelError: Error {{
    case missingOutput(String)
    case emptyPrompt
    case promptTooLong(prefixCount: Int, maxSeqLen: Int)
}}

/// 载入转换产物。Core ML 只能加载**已编译**的模型：传入未编译的 .mlpackage 时
/// 先调用 compileModel 生成 .mlmodelc（把模型加入 Xcode 工程时，Xcode 会在构建期
/// 自动完成这一步，此时直接传 Bundle 里的 URL 即可，本函数会原样使用）。
func loadConvertedModel(_ modelURL: URL) throws -> MLModel {{
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let url = modelURL.pathExtension == "mlpackage"
        ? try MLModel.compileModel(at: modelURL)
        : modelURL
    return try MLModel(contentsOf: url, configuration: configuration)
}}

/// 单步前向：推进一个 token，返回该步的 logits 与下一步要回传的状态。
/// - token: 本步的 token id；position: 它在序列中的位置（0..{max_seq_len_minus_1}）
/// - gdnStates: 上一步输出的 conv/recurrent 状态（首次调用传零数组）
/// - 返回 (logits, 新的 gdn 状态字典)；KV cache 由 MLState 自动携带
func forwardWithConvertedModel(
    _ model: MLModel, _ state: MLState,
    token: Int32, position: Int32,
    gdnStates: [String: MLMultiArray]
) throws -> (logits: MLMultiArray, newStates: [String: MLMultiArray]) {{
    var dict: [String: Any] = [
        "input_ids": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: [token], shape: [1, 1]))),
        "cache_position": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: [position], shape: [1]))),
    ]
    for (name, value) in gdnStates {{ dict[name] = MLFeatureValue(multiArray: value) }}
    let output = try model.prediction(
        from: try MLDictionaryFeatureProvider(dictionary: dict), using: state)
    guard let logits = output.featureValue(for: "{output_name}")?.multiArrayValue,
          logits.count > 0 else {{
        throw ConvertedModelError.missingOutput("{output_name}")
    }}
    // 收集同名的新状态（Core ML 输出名形如 new_conv_states / conv_states）
    var newStates: [String: MLMultiArray] = [:]
    for name in gdnStates.keys {{
        if let v = output.featureValue(for: name)?.multiArrayValue {{
            newStates[name] = v
        }} else if let v = output.featureValue(for: "new_" + name)?.multiArrayValue {{
            newStates[name] = v
        }}
    }}
    return (logits, newStates)
}}

/// 取 logits 最后一行 argmax 的 token id。
///
/// 注意 vocab 取自 **shape 的最后一维**，不是 logits.count - 1：logits 是
/// [1, q, vocab_size] 三维数组，元素总数与最后一维长度是两个不同概念。
private func argmaxToken(_ logits: MLMultiArray) -> Int32 {{
    let vocab = logits.shape[logits.shape.count - 1].intValue
    var bestIdx = 0
    var bestVal = -Double.greatestFiniteMagnitude
    let base = logits.count - vocab
    for i in 0..<vocab {{
        let v = logits[base + i].doubleValue
        if v > bestVal {{
            bestVal = v
            bestIdx = i
        }}
    }}
    return Int32(bestIdx)
}}

/// 用 stateful Core ML 模型做 KV cache 自回归生成。
/// - Parameters:
///   - modelURL: 模型 URL（.mlpackage 或已编译的 .mlmodelc）
///   - promptTokens: 分词后的 prompt token ids（长度需在 1..{max_seq_len} 内）
///   - maxNewTokens: 最多生成的 token 数（总长度不超过 {max_seq_len}）
/// - Returns: 生成的 token ids（不含 prompt）
func generateWithConvertedModel(modelURL: URL, promptTokens: [Int32],
                                maxNewTokens: Int) throws -> [Int32] {{
    guard !promptTokens.isEmpty else {{ throw ConvertedModelError.emptyPrompt }}
    guard promptTokens.count <= {max_seq_len} else {{
        throw ConvertedModelError.promptTooLong(prefixCount: promptTokens.count,
                                                maxSeqLen: {max_seq_len})
    }}
    let model = try loadConvertedModel(modelURL)
    let state = model.makeState()

    // 零初始化的 GDN 状态（混合架构专用；纯注意力模型该字典为空）
    let gdnStateShapes: [String: [Int]] = {gdn_state_swift_literal}
    var gdnStates: [String: MLMultiArray] = [:]
    for (name, shape) in gdnStateShapes {{
        let arr = try MLMultiArray(shape: shape.map {{ NSNumber(value: $0) }},
                                   dataType: .float32)
        memset(arr.dataPointer, 0, arr.count * MemoryLayout<Float>.size)
        gdnStates[name] = arr
    }}

    // 1) recurrent prefill：逐 token 推进整段 prompt（同一步骤也用于 decode）
    var position: Int32 = 0
    var logits: MLMultiArray? = nil
    for token in promptTokens {{
        let out = try forwardWithConvertedModel(model, state, token: token,
                                                position: position,
                                                gdnStates: gdnStates)
        logits = out.logits
        gdnStates = out.newStates
        position += 1
    }}

    // 2) decode：继续逐 token 前向
    var generated: [Int32] = []
    for _ in 0..<maxNewTokens {{
        guard let current = logits else {{ break }}
        let next = argmaxToken(current)
        generated.append(next)
        if position >= {max_seq_len} {{ break }}
        let out = try forwardWithConvertedModel(model, state, token: next,
                                                position: position,
                                                gdnStates: gdnStates)
        logits = out.logits
        gdnStates = out.newStates
        position += 1
    }}
    return generated
}}
'''


class _Heartbeat:
    """后台心跳线程：在长时间的阻塞阶段（torch.export 追踪 / coremltools 转换）
    周期性向 UI 发"仍在运行"的日志，避免界面长时间无任何输出时用户误以为卡死。

    在 worker 线程里启动一个 daemon 线程按间隔 log，阻塞的主任务返回后 stop()。
    日志经由 hooks.on_log 发出的 Qt 信号，跨线程 emit 由 Qt 自动排队，安全。
    with_memory=True 时附带进程内存峰值（MB），便于在冻结前发现内存吃满/swap。
    """

    def __init__(self, log, interval: float = 25.0,
                 message: str = "仍在处理中，请稍候（大模型可能需数分钟，此阶段无进度条属正常）…",
                 with_memory: bool = False):
        self._log = log
        self._interval = interval
        self._message = message
        self._with_memory = with_memory
        self._stop = threading.Event()
        # 线程延迟到 start() 才创建：未启动的 Thread 调 join() 会抛 RuntimeError，
        # 一旦某个阻塞阶段在 start() 之前就失败，stop() 会二次抛错掩盖真实异常
        self._thread: threading.Thread | None = None

    @staticmethod
    def _rss_mb() -> float:
        # macOS 的 ru_maxrss 单位为字节，Linux 为 KB
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            msg = self._message
            if self._with_memory:
                try:
                    msg = f"{msg}（内存峰值 {self._rss_mb():.0f} MB）"
                except Exception:
                    pass
            self._log(msg)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _import_heavy():
    """导入重量级依赖；缺失或损坏时抛异常（上层据此回退打包后端）。"""
    import coremltools as ct
    import numpy as np
    import torch
    import transformers
    return SimpleNamespace(ct=ct, torch=torch, transformers=transformers, np=np)


def _is_quantized_source(source: Path) -> bool:
    """config.json 是否为 MLX 量化模型（含 "quantization" 字段）。"""
    try:
        config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "quantization" in config


def _make_dequant_tmp(output_dir: Optional[Path] = None) -> Path:
    """生成一个不存在的临时目录路径（mlx_lm.convert 要求目标不存在）。

    优先放在输出目录旁：反量化会写出一份完整的浮点权重（大模型可达数十 GB），
    写进系统临时目录很容易撑爆启动盘。输出目录不可用时才退回系统临时目录。
    """
    name = f"modconv-dequant-{uuid.uuid4().hex[:12]}"
    if output_dir is not None:
        try:
            base = Path(output_dir)
            base.mkdir(parents=True, exist_ok=True)
            if os.access(base, os.W_OK):
                path = base / name
                _register_pending(path)
                return path
        except OSError:
            pass
    path = Path(tempfile.gettempdir()) / name
    _register_pending(path)
    return path


def _dequantize_mlx_model(source: Path, dest: Path, log: Callable[[str], None]) -> None:
    """用 mlx-lm 把量化权重反量化为浮点 HF 格式（safetensors + config.json）。

    mlx_lm.convert(dequantize=True) 会加载量化模型、反量化并保存，
    同时从 config.json 移除 quantization 字段，transformers 可直接加载。
    """
    try:
        from mlx_lm.convert import convert as mlx_convert
    except Exception as exc:
        raise RuntimeError(f"mlx-lm 不可用，无法反量化: {exc}") from exc
    log("用 mlx-lm 加载并反量化权重（大模型较慢，属正常现象）…")
    mlx_convert(hf_path=str(source), mlx_path=str(dest), dequantize=True)


def _load_config(source: Path, libs):
    return libs.transformers.AutoConfig.from_pretrained(str(source))


def _text_config(config):
    """取真正的文本模型配置。

    多模态（复合）配置的顶层只有 vision/audio 等无关字段，max_position_embeddings、
    layer_types、num_hidden_layers 全在 text_config 上；而 AutoModelForCausalLM 用的
    也正是它。直接读顶层会全部取到 None，进而选错上下文长度、漏掉混合架构的
    最小长度约束。
    """
    return getattr(config, "text_config", None) or config


def _rope_limit(config) -> Optional[int]:
    """模型位置编码上限（max_position_embeddings）；缺失或非法时返回 None。"""
    max_pos = getattr(_text_config(config), "max_position_embeddings", None)
    try:
        return int(max_pos) if max_pos else None
    except (TypeError, ValueError):
        return None


def _shards_and_weight_map(source: Path) -> tuple[list[Path], dict]:
    """(分片文件路径, 权重名 → 分片文件名)；无 index.json 时按 glob 取全部分片。"""
    src = Path(source)
    index = src / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            return sorted({src / name for name in weight_map.values()}), weight_map
        except (OSError, ValueError, KeyError):
            pass
    return sorted(src.glob("*.safetensors")), {}


def _checkpoint_tensor_shapes(source: Path) -> dict:
    """checkpoint 的权重名 → 形状（只读 safetensors 头部，不加载权重数据）。"""
    shapes: dict[str, tuple] = {}
    try:
        from safetensors import safe_open
    except Exception:
        return shapes
    for shard in _shards_and_weight_map(Path(source))[0]:
        try:
            with safe_open(str(shard), framework="pt") as handle:
                for name in handle.keys():
                    shapes[name] = tuple(handle.get_slice(name).get_shape())
        except Exception:
            continue  # 分片损坏：交给加载阶段报错
    return shapes


def _expected_state_shapes(libs, config) -> dict:
    """模型期望的权重名 → 形状。

    在 meta 设备上搭骨架：只为拿到名字与形状，不为大模型真实分配内存
    （0.8B 约 0.2s）。取不到时返回空字典，调用方据此跳过布局检查（不阻断转换）。
    """
    try:
        with libs.torch.device("meta"):
            skeleton = libs.transformers.AutoModelForCausalLM.from_config(config)
        return {name: tuple(tensor.shape)
                for name, tensor in skeleton.state_dict().items()}
    except Exception:
        return {}


def _resolve_weight_layout(want: dict, have: dict) -> Optional[tuple[str, set]]:
    """比对 checkpoint 与模型期望的权重布局，返回 (前缀, 需转置的权重名)。

    None 表示权重名对不上：transformers 会把它们全部随机初始化，导出的模型
    是无效权重，必须在加载前拦下。

    两类 MLX → HF 的布局差异在这里一并修正：
    1. 多模态 checkpoint 把文本模型放在 `language_model.` 前缀之下；
    2. MLX 的 Conv1d 权重是 (out, kernel, in)，PyTorch 是 (out, in, kernel)
       ——mlx-lm 的 convert（含量化模型的反量化）不会转这一步，GDN/linear
       attention 的 conv1d 权重因此要么形状对不上直接报错，要么在两维相等时
       被静默当成正确权重加载。
    """
    if not want:
        return "", set()  # 拿不到期望名，不做检查
    best, best_hits = "", 0
    for prefix in _TEXT_WEIGHT_PREFIXES:
        hits = sum(1 for name in want if prefix + name in have)
        if hits > best_hits:
            best, best_hits = prefix, hits
    if best_hits < len(want) * _WEIGHT_MATCH_RATIO:
        return None
    transposed = set()
    for name, shape in want.items():
        ckpt = have.get(best + name)
        if ckpt is None or len(ckpt) != 3 or len(shape) != 3 or ckpt == shape:
            continue
        if ckpt == (shape[0], shape[2], shape[1]):
            transposed.add(name)
    return best, transposed


def _needs_norm_unfolding(config, transposed: set) -> bool:
    """checkpoint 是否为 mlx-lm 转换产物（RMSNorm 权重已预折叠 1+gain）。

    判定与 mlx-lm 的 sanitize 互为镜像：那里只在 conv1d 为 PyTorch 布局
    （未消毒）时才给 norm 权重 +1.0；这里 conv1d 需要转置（transposed 非空）
    即说明 convert 时已消过毒——权重库里存的就是 1+gain。写回 HF 布局时
    必须减掉这层折叠，否则 HF 的 Qwen3_5RMSNorm 会二次 +1（得 2+gain），
    归一化尺度整体翻倍，输出乱码。RMSNormGated（GDN 内部）两侧都不加 +1，
    不在此列。仅 Qwen3.5 系使用该零中心约定，其余模型不猜。
    """
    if not any("conv1d" in name for name in transposed):
        return False
    model_type = str(getattr(_text_config(config), "model_type", "") or "")
    return model_type in _FOLDED_NORM_MODEL_TYPES


def _write_text_checkpoint(source: Path, dest: Path, prefix: str, transposed: set,
                           unfold_norms: bool,
                           log: Callable[[str], None]) -> None:
    """把文本权重另存为 HF 布局（去掉前缀 + 转置 conv1d 权重）。

    多模态 checkpoint 把文本模型放在 `language_model.` 之下并带有 vision_tower.*；
    这里只取该前缀的权重、去掉前缀另存，输出目录因此只有文本 LM，可被
    AutoModelForCausalLM 正常加载。逐分片读写，内存占用约等于一个分片。

    unfold_norms：MLX 转换产物的 RMSNorm 权重已预折叠为 1+gain（见
    _needs_norm_unfolding），逐键减 1.0 还原成 HF 零中心约定的原始 gain。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    dest.mkdir(parents=True, exist_ok=True)
    shards, weight_map = _shards_and_weight_map(source)
    kept: dict[str, str] = {}
    for shard in shards:
        # 只物化要保留的张量：load_file 会把整个分片（含 vision_tower 等用不到的
        # 权重）全部读进内存，多模态 checkpoint 上白占一半以上。safe_open 先看
        # key 再按需 get_tensor，峰值内存从"整个分片"降到"保留的权重"。
        try:
            with safe_open(str(shard), framework="pt") as handle:
                tensors = {}
                for name in handle.keys():
                    if not name.startswith(prefix):
                        continue
                    tensor = handle.get_tensor(name)
                    key = name[len(prefix):]
                    if key in transposed:
                        tensor = tensor.transpose(1, 2)
                    if unfold_norms and key.endswith(_FOLDED_NORM_SUFFIXES):
                        # 折叠值 1+gain → 原始 gain；经 fp32 中间量计算，避免
                        # bf16 存储下 (1.2x - 1.0) 的二次舍入误差
                        tensor = (tensor.float() - 1.0).to(tensor.dtype)
                    tensors[key] = tensor
        except Exception:
            continue  # 分片损坏：交给加载阶段报错
        if not tensors:
            continue
        log(f"归一化权重分片 {shard.name}（{len(tensors)} 个张量"
            f"{f'，转置 {len(transposed)} 个 conv1d 权重' if transposed else ''}"
            f"{'，还原 RMSNorm 权重折叠' if unfold_norms else ''}）…")
        try:
            save_file(tensors, str(dest / shard.name), metadata={"format": "pt"})
        except Exception:
            # 共享存储 / 非连续（转置后）的张量不能直接落盘，克隆后再存
            save_file({k: v.clone() for k, v in tensors.items()},
                      str(dest / shard.name), metadata={"format": "pt"})
        del tensors
        if weight_map:
            kept.update({name[len(prefix):]: shard.name
                         for name in weight_map if name.startswith(prefix)})
    if kept:
        (dest / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": dict(sorted(kept.items()))}),
            encoding="utf-8")
    shutil.copyfile(Path(source) / "config.json", dest / "config.json")


def _pick_max_seq_len(config) -> int:
    """自动选取最大上下文长度：模型位置编码上限与全局上限取小。"""
    limit = _rope_limit(config) or SEQ_LEN_CAP
    return max(1, min(limit, SEQ_LEN_CAP))


def _min_max_seq_len(config) -> int:
    """该架构允许的最小上下文长度。

    含 linear attention 的混合架构要求 >= 2：GDN 分块内核的分块循环次数依赖序列
    长度，wrapper 必须把输入 pad 到静态 max_seq_len，且 seq_len==1 的 decode 快速
    路径要在追踪期静态消除（见 KvCacheWrapper）。
    """
    config = _text_config(config)
    layer_types = list(getattr(config, "layer_types", None) or [])
    if not layer_types:
        return 1
    has_linear = any(t not in ATTENTION_LAYER_TYPES for t in layer_types)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    return 2 if (has_linear and len(layer_types) == num_layers) else 1


def _resolve_max_seq_len(config, requested: Optional[int], log: Callable[[str], None]) -> int:
    """确定最终上下文长度：用户指定值 / 自动值，再按位置编码上限与架构下限收敛。"""
    max_seq_len = requested or _pick_max_seq_len(config)
    # 用户显式指定的长度也不能超过模型位置编码上限（超出后 rope 语义失效）
    rope_limit = _rope_limit(config)
    if rope_limit and max_seq_len > rope_limit:
        log(f"最大上下文长度 {max_seq_len} 超过模型位置编码上限 {rope_limit}，已截断")
        max_seq_len = rope_limit
    # 混合注意力架构要求至少 2（GDN 分块内核需静态长度，见 _min_max_seq_len）
    min_len = _min_max_seq_len(config)
    if max_seq_len < min_len:
        log(f"最大上下文长度 {max_seq_len} 小于该架构所需的最小值 {min_len}，已调整")
        max_seq_len = min_len
    return max_seq_len


class _CoremlCastNoiseFilter(logging.Filter):
    """过滤 coremltools 的 int64 → int32 常量降级告警。

    torch 的索引张量（scatter/gather 的 index）必须是 int64，而 Core ML/ANE 只
    接受 int32，前端于是逐个常量降级并告警。这些都是远小于 2^31 的索引常量，
    降级无损；但大模型动辄数万条，会把日志彻底刷满（真正的问题反而看不见）。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "might lose precision" not in record.getMessage()


def _unusable_model_message(caught) -> Optional[str]:
    """从 coremltools 告警里挑出"模型无法执行"的那条（ANE 执行计划构建失败 -14）。

    coremltools 遇到编译不过的模型并不抛异常，只发一条 RuntimeWarning 就照常
    返回对象——不拦住的话界面上会显示"转换完成"，交到用户手里却是个跑不起来的
    模型。
    """
    for item in caught:
        text = str(getattr(item, "message", item))
        if "will not be able to run predict" in text:
            return text
    return None


@contextmanager
def _capture_ane_warnings():
    """只捕获 coremltools 的"模型无法执行"告警，供 _unusable_model_message 判读。

    不能用 `simplefilter("always")`：那会记录转换期间**所有**告警——大模型转换
    动辄几十分钟，numpy 的溢出告警每条 cast 都发一次，既白占内存，也会把它们从
    stderr 里吞掉（真正该看的错误反而看不见）。按 message 精确匹配只留目标告警。
    """
    with warnings.catch_warnings(record=True) as caught:
        # 注意 catch_warnings(record=True) 会清空过滤器并记录所有告警，必须再补一条
        # catch-all ignore，否则无关告警仍会被收进来（既占内存也看不出重点）。
        # filterwarnings 插在过滤器最前面：先加 ignore、后加目标规则，目标才排在前头。
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("always", message=".*will not be able to run predict.*")
        yield caught


def _layout_mismatch_message(source: Path) -> str:
    return (f"源目录 {source} 的权重名与该架构期望的不一致：checkpoint 里找不到"
            " model.layers.* 等文本模型权重。若继续加载，transformers 会把它们"
            "全部随机初始化，导出的是无效权重的模型，因此已中止转换。"
            "请确认该目录是 mlx-lm 可读取的 MLX/HF 模型目录。")


class _LoggingBridge(logging.Handler):
    """把 transformers 的加载告警（MISSING / UNEXPECTED 权重）转发到 GUI 日志。"""

    def __init__(self, log: Callable[[str], None]):
        super().__init__(level=logging.WARNING)
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log(record.getMessage().strip())
        except Exception:  # pragma: no cover - 日志失败不能影响转换
            self.handleError(record)


def _load_torch_model(source: Path, libs, config, dtype=None, log=None):
    """按 HF 架构加载模型权重（权重布局须已与架构匹配，见 _resolve_weight_prefix）。

    dtype 为 None 时用 float32；量化模型反量化后用 float16 以减半内存。
    注意必须用 eager 注意力：sdpa 的 math 分解会引入运行时形状 expand，
    让 ANE 执行计划构建失败（-14）。
    MoE 架构（如 qwen3_5_moe）还要强制 eager 专家实现：默认的融合内核用
    torch.histc 统计每张专家的 token 数，coremltools 前端不支持该算子
    （"Unsupported fx node histc"）。
    """
    transformers, torch = libs.transformers, libs.torch
    for cfg in (config, getattr(config, "text_config", None)):
        try:
            cfg.use_cache = False
        except AttributeError:
            pass
    # transformers 5.x 用 dtype（torch_dtype 已弃用）；旧版仅认 torch_dtype 时回退。
    # low_cpu_mem_usage 让权重直接就地载入（不再先随机初始化一份再整体覆盖），
    # 大模型能省掉约一倍峰值内存；老版本不认识该参数时逐级降级。
    base = {"config": config, "dtype": dtype or torch.float32,
            "attn_implementation": "eager", "experts_implementation": "eager"}
    attempts = [
        {**base, "low_cpu_mem_usage": True},
        base,
        {"config": config, "torch_dtype": dtype or torch.float32},
    ]
    # 加载期的告警（权重缺失/多余）默认只进 stderr，界面上看不到；桥接到日志
    bridge = _LoggingBridge(log or (lambda _: None))
    logger = logging.getLogger("transformers")
    logger.addHandler(bridge)
    try:
        for kwargs in attempts:
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    str(source), **kwargs)
                break
            except TypeError:
                continue  # 该版本 transformers 不认这套参数名，换下一组
        else:  # pragma: no cover - 三组都失败说明 transformers 接口完全变了
            raise RuntimeError("AutoModelForCausalLM.from_pretrained 参数不兼容，无法加载模型")
    finally:
        logger.removeHandler(bridge)
    model.eval()
    return model


def _build_kv_wrapper(model, max_seq_len: int):
    """把 HF 模型包装为带 KV cache 状态的导出模块（懒加载重量级模块）。"""
    from modconv.kv_wrapper import KvCacheWrapper
    return KvCacheWrapper(model, max_seq_len)


def _decomposition_table(libs):
    """导出降级表：默认分解（去掉 matmul）+ alias 恒等透传。

    必须去掉 aten.matmul 分解：matmul 分解会发出以 sym_size（运行时标量）为
    目标形状的 aten.expand，coremltools 前端无法静态证明 tile reps=1，降级为
    运行时 reps 的 tile，ANE 执行计划构建失败（-14）。保留 4D matmul 原样
    进前端即可——MIL matmul 原生支持 N-D batch 广播。
    """
    table = {}
    try:
        from torch._decomp import core_aten_decompositions
        table.update(core_aten_decompositions())
        table.pop(libs.torch.ops.aten.matmul.default, None)
    except Exception:
        pass
    try:
        table[libs.torch.ops.aten.alias.default] = lambda x: x
    except AttributeError:
        pass
    return table


@contextmanager
def _fail_fast_dynamo(libs):
    """让 torch.export 追踪在陷入重追踪循环时快速失败，而不是盲 grind 几十分钟。

    Dynamo 默认 recompile_limit=64：一旦模型里有触发反复重追踪的算子/守卫，
    每次重追踪整模型要几十秒，撞满上限前界面会"卡死"数十分钟（0.8B 模型实测
    可达 40 分钟）。这里把上限调到很小并打开 verbose，使其立刻抛出清晰错误
    （含重追踪/图断裂原因），方便定位，而不是让用户无限等待。
    """
    torch = libs.torch
    cfg = getattr(torch, "_dynamo", None)
    cfg = getattr(cfg, "config", None) if cfg is not None else None
    saved = {}
    patched = False
    if cfg is not None:
        for key, val in (("recompile_limit", 4), ("error_on_recompile_limit", True),
                         ("verbose", True)):
            if hasattr(cfg, key):
                saved[key] = getattr(cfg, key)
                setattr(cfg, key, val)
                patched = True
    try:
        yield
    finally:
        if patched:
            for key, val in saved.items():
                setattr(cfg, key, val)


def _trace_torch_model(libs, model, max_seq_len: int):
    """包装为单步 decode 模块并 torch.export 导出，返回 (exported, wrapper)。

    图每次只推进**一个** token（input_ids 形状恒为 (1, 1)），prefill 由调用方
    循环调用完成。这样 GDN 的分块内核（63 步三角求解展开、fp16 下 cumsum 误差
    随步数放大）完全不需要出现在图里，算子数下降两个数量级——是 ANE 执行计划
    构建失败（-14）、预测期编译卡死与精度发散的共同根源。

    混合架构的 conv/recurrent state 作为图的**输入/输出**（不进 MLState，见
    KvCacheWrapper.gdn_state_specs），因此示例输入里要带上它们。
    """
    torch = libs.torch
    wrapper = _build_kv_wrapper(model, max_seq_len).eval()
    example = [torch.ones((1, 1), dtype=torch.long), torch.zeros((1,), dtype=torch.long)]
    for _, shape, dtype_str in wrapper.gdn_state_specs():
        example.append(torch.zeros(shape, dtype=getattr(torch, dtype_str)))
    # 全静态形状（单步 decode），无需声明动态维度
    # 包在 no_grad 里导出：模型参数 requires_grad=True 时，KV buffer 的 in-place
    # scatter_ 变异会被 aot_export 拒绝（"graph input that requires gradients,
    # and received a mutation"）。桩 torch 没有 no_grad 时退化为空上下文。
    no_grad = getattr(torch, "no_grad", nullcontext)
    with no_grad(), _fail_fast_dynamo(libs):
        exported = torch.export.export(wrapper, tuple(example))
    return exported.run_decompositions(_decomposition_table(libs)), wrapper


def _convert_to_coreml(libs, exported, state_specs, max_seq_len, gdn_state_specs=()):
    """转换为带 KV cache 状态的单步 stateful mlprogram（iOS18 目标）。

    Core ML State 后端仅支持 fp16；KV buffer 固定 fp32（前端对 read_state 无条件
    插入 fp32 cast，见 KvCacheWrapper），state 读写由前端自动双向 cast。

    只保留**一个** StateType（KV 缓存）：Core ML 在 ANE/CPU 上对多 StateType
    的图会出问题（参考 ref/CoreML-LLM 在同架构上的实测：CPU runtime 误编译
    slice_update 输出 garbage，ANE 侧 predict 期报 Error 11）。混合架构的
    GDN conv/recurrent state 因此走普通输入/输出张量。

    只依赖导出图与 state 元信息（而非 KvCacheWrapper），这样调用方在追踪结束后
    就能立刻释放 wrapper 及其持有的 HF 模型。
    """
    ct, np = libs.ct, libs.np
    np_dtype = {"float16": np.float16, "float32": np.float32}
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=shape,
                                                      dtype=np_dtype[dtype_str]),
                           name=name)
              for name, shape, dtype_str in state_specs]
    # 单步 decode：序列长度恒为 1，输入形状全静态
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
    ]
    inputs += [ct.TensorType(name=name, shape=shape, dtype=np_dtype[dtype_str])
               for name, shape, dtype_str in gdn_state_specs]
    return ct.convert(
        exported,
        inputs=inputs,
        states=states,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )


def _describe_spec(spec, short_description: str, input_descs: dict, output_descs: dict) -> None:
    """向 Core ML spec 写入元数据（Xcode/Swift 中可见），便于调用。"""
    desc = spec.description
    desc.metadata.shortDescription = short_description
    for feature in desc.input:
        if feature.name in input_descs:
            feature.shortDescription = input_descs[feature.name]
    for feature in desc.output:
        if feature.name in output_descs:
            feature.shortDescription = output_descs[feature.name]


def _nonfinite_const_names(mlmodel) -> set:
    """找出 MIL 程序中含 ±Inf/NaN 的 const 常量名。

    transformers 的 chunked linear-attention 内核用 masked_fill(mask, -inf)
    构造因果掩码，coremltools 前端把它常量折叠为 where(mask, -inf, 0) 常量
    （形状如 (1, n_heads, 1, chunk, chunk)）。这类掩码常量不是模型权重，
    而 kmeans palettization 无法处理非有限值（sklearn 直接报错），量化时按名
    跳过：掩码以 fp16 原值参与 add/exp，数学语义不变。
    """
    program = getattr(mlmodel, "_mil_program", None)
    if program is None:
        return set()
    import numpy as np

    names = set()
    for func in program.functions.values():
        for op in func.operations:
            if op.op_type != "const":
                continue
            try:
                val = np.asarray(op.val.val)
            except Exception:
                continue
            if val.dtype.kind == "f" and not np.isfinite(val).all():
                names.add(op.name)
    return names


def _compress_weights(libs, mlmodel, nbits: int, log):
    """把 mlprogram 权重压缩为 n-bit palettization（kmeans LUT），减小体积、加速端侧推理。

    含 ±Inf/NaN 的常量（注意力掩码等）不量化：kmeans 无法处理非有限值，
    这些常量保留 fp16 原值参与运算，语义不变。
    """
    ct = libs.ct
    opt = ct.optimize.coreml
    skip = _nonfinite_const_names(mlmodel)
    if skip:
        shown = ", ".join(sorted(skip)[:3]) + ("…" if len(skip) > 3 else "")
        log(f"{len(skip)} 个常量（如注意力掩码）含 ±Inf，不参与量化、保留原值：{shown}")
    config = opt.OptimizationConfig(
        global_config=opt.OpPalettizerConfig(nbits=nbits, mode="kmeans"),
        op_name_configs={name: None for name in skip})
    log(f"对模型权重做 {nbits}-bit palettization（kmeans LUT，大模型较慢）…")
    return opt.palettize_weights(mlmodel, config=config)


def _write_swift_example(out_path: Path, model_name: str, max_seq_len: int,
                         output_name: str,
                         gdn_state_specs: list[tuple[str, tuple[int, ...], str]] = ()
                         ) -> Path:
    """在输出目录生成 Swift 调用示例文件。"""
    # 混合架构的 GDN 状态形状写成 Swift 字面量，供示例里构造零初始数组
    literal = ("[" + ", ".join(
        f'"{name}": {list(shape)}' for name, shape, _ in gdn_state_specs) + "]"
        if gdn_state_specs else "[:]")  # 空字典字面量必须是 [:]
    swift_path = out_path.parent / f"{out_path.stem}-swift-example.swift"
    swift_path.write_text(
        _SWIFT_EXAMPLE_TEMPLATE.format(model_name=model_name,
                                       max_seq_len=max_seq_len,
                                       max_seq_len_minus_1=max(max_seq_len - 1, 0),
                                       output_name=output_name,
                                       gdn_state_swift_literal=literal),
        encoding="utf-8")
    return swift_path


def _save_mlmodel(libs, mlmodel, path: Path):
    mlmodel.save(str(path))


def _first_output_name(spec) -> str:
    """spec 的第一个输出名；没有输出时退回 "logits"（改名失败时的兜底名）。"""
    outputs = getattr(getattr(spec, "description", None), "output", None) or []
    return outputs[0].name if outputs else "logits"


def _drop_package(model_obj) -> None:
    """删除 MLModel 自带的临时 .mlpackage 目录，立即释放磁盘。

    `MLModel(spec, weights_dir=…)` 与 `ct.convert` 都会在系统临时目录下生成一份
    完整包（大模型上 GB 级）。coremltools 只在**进程正常退出**时（atexit）清理，
    而 GUI 是长驻进程：连续转换会一直堆积，-14 重试时甚至同时存在数份——既吃磁盘
    也干扰后续编译。重建/保存完成后旧包已无用（`save()` 是 copytree，重建出的包
    自带权重副本），显式删除；只删 coremltools 自己也标记为临时的包。
    """
    if not getattr(model_obj, "is_temp_package", False):
        return
    path = getattr(model_obj, "package_path", None)
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)
    _unregister_pending(path)


def _measure(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    if path.is_file():
        try:
            return 1, path.stat().st_size
        except OSError:
            return 0, 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                files += 1
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return files, total


class CoremltoolsBackend:
    """coremltools 真实转换后端，分阶段执行并回报进度。

    非量化模型 4 个阶段；MLX 量化模型先反量化，共 5 个阶段。
    """

    def __init__(self, source: Path, output_dir: Path, max_seq_len: Optional[int] = None,
                 quantize_nbits: Optional[int] = 4):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.max_seq_len = int(max_seq_len) if max_seq_len else None
        # 输出权重压缩位数（4-bit palettization）；None/0 表示不压缩
        self.quantize_nbits = int(quantize_nbits) if quantize_nbits else None

    def run(self, hooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success: bool, cancelled: bool, message: str,
                   output: Optional[Path], files_done: int, bytes_done: int) -> ConversionResult:
            return ConversionResult(success, cancelled, message, output,
                                    files_done, bytes_done, time.monotonic() - start)

        if hooks.should_cancel():
            return finish(False, True, "转换已取消", None, 0, 0)

        libs = _import_heavy()
        quantized = _is_quantized_source(self.source)
        total_stages = 5 if quantized else 4
        counter = {"i": 0}

        def stage(text: str):
            counter["i"] += 1
            i = counter["i"]
            hooks.on_log(f"[{i}/{total_stages}] {text}")
            hooks.on_file_progress(f"阶段 {i}/{total_stages}",
                                   i, total_stages, i, total_stages)

        tmp_dirs: list[Path] = []
        coreml_logger = logging.getLogger("coremltools")
        noise_filter = _CoremlCastNoiseFilter()
        coreml_logger.addFilter(noise_filter)
        try:
            if quantized:
                stage("MLX 量化模型反量化（mlx-lm）…")
                # 反量化产物放在输出目录旁：避免数十 GB 中间文件撑爆系统盘
                tmp_dir = _make_dequant_tmp(self.output_dir)
                tmp_dirs.append(tmp_dir)
                _dequantize_mlx_model(self.source, tmp_dir, hooks.on_log)
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)

            stage("加载模型配置与权重…")
            torch_source: Path = tmp_dirs[-1] if tmp_dirs else self.source
            config = _load_config(torch_source, libs)

            # 权重布局检查 + 归一化：checkpoint 的权重名/形状必须与架构期望的
            # 对得上。transformers 在名字对不上时不会报错，而是把权重全部"新建
            # 初始化"，结果是一个带着随机权重、表面却转换成功的 Core ML 模型，
            # 所以必须在加载前拦下（见 _resolve_weight_layout）。
            layout = _resolve_weight_layout(_expected_state_shapes(libs, config),
                                            _checkpoint_tensor_shapes(torch_source))
            if layout is None:
                return finish(False, False, _layout_mismatch_message(torch_source),
                              None, 0, 0)
            prefix, transposed = layout
            unfold_norms = _needs_norm_unfolding(config, transposed)
            if prefix or transposed:
                if len(tmp_dirs) > 1:
                    # 已经归一化过一次仍对不上，这条路径救不回来
                    return finish(False, False, _layout_mismatch_message(torch_source),
                                  None, 0, 0)
                reasons = []
                if prefix:
                    reasons.append(f"文本权重带 '{prefix}' 前缀（多模态布局）")
                if transposed:
                    reasons.append(f"{len(transposed)} 个 conv1d 权重为 MLX 布局"
                                   "（out, kernel, in）")
                if unfold_norms:
                    reasons.append("RMSNorm 权重已预折叠 1+gain，将还原为 HF 的"
                                   "零中心约定")
                hooks.on_log("checkpoint 为 " + "、".join(reasons)
                             + "，正在另存为 HF 布局…")
                # 与反量化产物同样放在输出目录旁：体积和源权重相当
                norm_dir = _make_dequant_tmp(self.output_dir)
                tmp_dirs.append(norm_dir)
                _write_text_checkpoint(torch_source, norm_dir, prefix, transposed,
                                       unfold_norms, hooks.on_log)
                torch_source = norm_dir
                config = _load_config(torch_source, libs)
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)

            max_seq_len = _resolve_max_seq_len(config, self.max_seq_len, hooks.on_log)
            dtype = libs.torch.float16 if quantized else None
            seq_attempt = max_seq_len
            requested_seq_len = max_seq_len
            quantized_bits = 0
            gdn_state_specs = ()
            # 降级方案只有 1 个且首轮就失败时（非量化 + 长度已在下限），_post_convert
            # 不会被执行，这两个变量必须预先有值，否则保存阶段直接 NameError
            output_name = "logits"

            def _next_seq_len(current: int) -> Optional[int]:
                """-14 时的下一个尝试长度；None 表示已到下限、不能再降。"""
                nxt = max(_MIN_SEQ_RETRY, _min_max_seq_len(config), current // 2)
                return None if nxt >= current else nxt

            def _fallback_plans() -> list[tuple]:
                """(是否跳过量化, 上下文长度) 的降级序列，按实测影响力排序。

                1. 用户配置原样；
                2. 开了量化时：先给"量化 + 长度减半"一次机会（有的模型靠缩短
                   上下文就能过），随后退回未量化并回到原长度——实测 4-bit 图才是
                   -14 的主导因素，而上下文砍到 64 基本没法用；
                3. 最后逐级降长度（此时量化场景已是未量化，未量化场景保持原样）。
                """
                plans = [(False, requested_seq_len)]
                if self.quantize_nbits:
                    half = _next_seq_len(requested_seq_len)
                    if half is not None:
                        plans.append((False, half))
                    plans.append((True, requested_seq_len))
                # skip_quantize 只在量化场景有意义；未量化时它恒为 False
                skip = bool(self.quantize_nbits)
                cur = requested_seq_len
                for _ in range(_MAX_SEQ_RETRIES):
                    nxt = _next_seq_len(cur)
                    if nxt is None:
                        break
                    cur = nxt
                    plans.append((skip, cur))
                return plans

            def _post_convert(model_obj, seq_len):
                """量化 + 写元数据 + 重建落盘对象；返回 (mlmodel, 位数, 输出名, -14 信息)。

                三个阶段都可能暴露 -14：palettization 内部会重建 MLModel（等于再编译
                一次），重建落盘对象时默认也会编译。skip_quantize 为真时跳过量化
                （量化后的图构建不了执行计划时的兜底，见下方重试逻辑）。
                """
                bits = 0
                if self.quantize_nbits and not skip_quantize:
                    heartbeat = _Heartbeat(hooks.on_log,
                                           message="仍在做权重量化（palettization），请稍候…")
                    heartbeat.start()
                    try:
                        with _capture_ane_warnings() as caught:
                            compressed = _compress_weights(libs, model_obj,
                                                           self.quantize_nbits,
                                                           hooks.on_log)
                    finally:
                        heartbeat.stop()
                    msg = _unusable_model_message(caught)
                    # 压缩前模型的临时权重目录（数十 GB 级）随旧对象一起释放
                    del model_obj
                    gc.collect()
                    model_obj = compressed
                    del compressed
                    bits = self.quantize_nbits
                    if msg:
                        # 输出名也要从压缩后的 spec 取：硬编码 "logits" 会让
                        # Swift 示例在改名失败时指向不存在的输出
                        name = _first_output_name(model_obj.get_spec())
                        return model_obj, bits, name, msg

                # get_spec() 返回深拷贝：在拷贝上重命名输出并写入元数据，
                # 再用 ct.models.MLModel(spec) 重建，确保落盘内容生效。
                spec = model_obj.get_spec()
                if spec.description.output:
                    old_name = spec.description.output[0].name
                    if old_name != "logits":
                        try:
                            libs.ct.utils.rename_feature(spec, old_name, "logits")
                        except Exception:
                            pass  # 个别格式不支持重命名时保留原名，示例会用真实输出名
                # 混合架构的新状态输出默认是 cat_N 之类通用名，重命名为
                # new_<state 名>：Swift 侧靠名字把它们回传为下一步输入。
                # 输出顺序与 gdn_state_specs 一致（导出图按 conv→rec 返回）。
                state_outs = [f for f in spec.description.output
                              if f.name != "logits"]
                for feature, (name, _, _) in zip(state_outs, gdn_state_specs):
                    if feature.name != f"new_{name}":
                        try:
                            libs.ct.utils.rename_feature(spec, feature.name,
                                                         f"new_{name}")
                        except Exception:
                            pass
                input_descs = {
                    "input_ids": "本步的 1 个 Token id (Int32)，形状 [1, 1]",
                    "cache_position": f"该 token 的位置 (Int32)，形状 [1]，取值 0..{seq_len - 1}",
                    "conv_states": "GDN 卷积状态 (Float)，上一步同名输出原样回传",
                    "recurrent_states": "GDN 循环状态 (Float)，上一步同名输出原样回传",
                }
                output_descs = {f.name: ("词表 logits (Float)，形状 [1, 1, vocab_size]"
                                        if f.name == "logits" else
                                        "下一步要回传的同名状态 (Float)")
                                for f in spec.description.output}
                _describe_spec(spec, f"由模型 '{self.source.name}' 转换"
                                     f"（modconv + coremltools，单步 decode + KV cache 状态）",
                               input_descs, output_descs)
                name = _first_output_name(spec)
                # mlprogram 的 MLModel 不能仅从 spec 重建，必须带上原模型权重目录。
                # 这里重建只为让改过的 spec 落盘，用 CPU_ONLY 跳过 ANE 编译：每构造
                # 一个 MLModel 就会构建一次执行计划（大模型十几分钟，长上下文下还会
                # -14），而落盘内容并不含计算单元信息，预测能力在此毫无必要。
                with _capture_ane_warnings() as caught:
                    rebuilt = libs.ct.models.MLModel(
                        spec, weights_dir=model_obj.weights_dir,
                        compute_units=libs.ct.ComputeUnit.CPU_ONLY)
                # 重建包自带权重副本，旧包（含其权重目录）到此无用，立即释放磁盘
                _drop_package(model_obj)
                _track_temp_package(rebuilt)
                return rebuilt, bits, name, _unusable_model_message(caught)

            # 追踪 + 转换可能因 ANE -14 反复重试：第一次按目标长度走正常阶段提示，
            # 后续重试只打日志（阶段计数已经报完，避免显示"第 5/4 阶段"）。
            plans = _fallback_plans()
            plan_i = 0
            while True:
                skip_quantize, seq_attempt = plans[plan_i]
                attempt = plan_i + 1
                emit = stage if attempt == 1 else hooks.on_log
                # 每轮都要重新载入：上一轮的模型已随 wrapper 一起释放（省内存），
                # 而降长度后必须重新追踪，权重目录已归一化过，重载很快
                model = _load_torch_model(torch_source, libs, config, dtype=dtype,
                                          log=hooks.on_log)
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)
                emit("追踪 torch 计算图…（大模型可能需数分钟，此阶段无进度条属正常）"
                     if attempt == 1 else f"以上下文长度 {seq_attempt} 重新追踪 torch 计算图…")
                heartbeat = _Heartbeat(hooks.on_log,
                                       message="仍在追踪 torch 计算图，请稍候（Qwen3.5 混合架构较大，通常需数分钟）…")
                heartbeat.start()
                try:
                    exported, wrapper = _trace_torch_model(libs, model, seq_attempt)
                finally:
                    heartbeat.stop()
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)

                # 追踪完成后 HF 模型与 StaticCache 都不再需要：wrapper.inner 就是
                # model，只 del 局部变量根本释放不掉（引用还挂在 wrapper 上）。这里
                # 先把转换所需的 state 元信息取出来，再一次性丢掉 wrapper + model 并
                # 回收——这是转换期能砍掉的最大一块内存，避免大模型触发 swap 而显得
                # "卡死"（coremltools 转换时会持有自己的 fp16 权重副本）。
                state_specs = wrapper.kv_state_specs()
                gdn_state_specs = wrapper.gdn_state_specs()
                traced_seq_len = wrapper.max_seq_len
                del wrapper, model
                gc.collect()

                emit("coremltools 转换为 Core ML 模型（大模型可能较慢）…"
                     if attempt == 1 else "重新转换为 Core ML 模型…")
                heartbeat = _Heartbeat(hooks.on_log,
                                       message="仍在转换为 Core ML 模型，请稍候…",
                                       with_memory=True)
                heartbeat.start()
                try:
                    # ct.convert 内部就会实例化 MLModel（即编译一次），-14 在这里
                    # 就已经暴露：此时重试只浪费一次编译，不必等量化跑完才发现。
                    with _capture_ane_warnings() as caught:
                        mlmodel = _convert_to_coreml(libs, exported, state_specs,
                                                     traced_seq_len, gdn_state_specs)
                        _track_temp_package(mlmodel)
                finally:
                    heartbeat.stop()
                ane_msg = _unusable_model_message(caught)
                # 导出图此后不再使用，尽早释放并回收，降低后续量化/保存阶段的峰值内存
                # gdn_state_specs 留到后面写 Swift 示例（只是形状元信息，几十字节）
                del exported, state_specs
                gc.collect()
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)
                if ane_msg is None:
                    mlmodel, quantized_bits, output_name, ane_msg = _post_convert(
                        mlmodel, traced_seq_len)
                if ane_msg is None:
                    break
                # 这个方案下 ANE 构建不了执行计划：换下一个降级方案
                if plan_i + 1 >= len(plans):
                    hooks.on_log("已试遍降级方案（缩短上下文 / 取消量化），仍无法构建执行计划")
                    break
                next_skip, next_seq = plans[plan_i + 1]
                changes = []
                if next_skip and not skip_quantize:
                    changes.append("取消量化（输出体积更大，但可真正运行）")
                if next_seq != seq_attempt:
                    changes.append(f"上下文长度 {seq_attempt} → {next_seq}")
                hooks.on_log(f"当前方案（长度 {seq_attempt}"
                             f"{'' if skip_quantize else '、4-bit 量化'}）下 Core ML 无法"
                             f"构建执行计划（-14）：改为{'，'.join(changes)}后重试…")
                plan_i += 1
                # 失败方案的临时包立刻删掉：重试最多可能同时积数份 GB 级包
                _drop_package(mlmodel)
                mlmodel = None
                gc.collect()
            max_seq_len = traced_seq_len
            unusable = ane_msg
            # 上下文被自动降过、或量化被退回：都直接影响产物可用性，必须在结论里说清楚
            reduced_note = (f"（注意：上下文长度已由 {requested_seq_len} 自动降到 {max_seq_len}，"
                            f"更长的图在 ANE 上无法构建执行计划）"
                            if max_seq_len != requested_seq_len else "")
            if skip_quantize and self.quantize_nbits:
                reduced_note += "（注意：量化后的模型在 ANE 上无法构建执行计划，已改为未量化输出）"

            # 量化是最长的阶段（大模型数十分钟）：这里必须再查一次取消，
            # 否则用户点了取消也会一路走到保存，白跑完整个流程
            if hooks.should_cancel():
                return finish(False, True, "转换已取消", None, 0, 0)

            stage("保存输出模型…")
            # mlprogram → <名称>.mlpackage（目录）；输出目录不存在时先创建
            # （直接 API 调用而非 GUI 时不会经过 GUI 侧的目录创建）
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"{self.source.name}.mlpackage"
            if out_path.is_dir():
                shutil.rmtree(out_path)
            elif out_path.exists():
                out_path.unlink()
            _save_mlmodel(libs, mlmodel, out_path)
            # 产物已是完整副本，临时包没有再留的必要（大模型上 GB 级）
            _drop_package(mlmodel)
            swift_path = _write_swift_example(out_path, self.source.name, max_seq_len,
                                              output_name, gdn_state_specs)
            hooks.on_log(f"已生成 Swift 调用示例: {swift_path}")
            files_done, bytes_done = _measure(out_path)

            bits_text = f", {quantized_bits}-bit 权重" if quantized_bits else ""
            message = (f"coremltools 转换完成: {out_path} "
                       f"(stateful mlprogram 格式, KV cache 状态模型{bits_text}, "
                       f"{files_done} 个文件, {format_size(bytes_done)}, "
                       f"单步 decode（每次 1 个 token，prefill 循环调用），"
                       f"最大上下文 {max_seq_len}{reduced_note}, "
                       f"耗时 {time.monotonic() - start:.1f} 秒)")
            if unusable:
                # 不能当成功交付：产物已落盘便于排查，但必须明确告知不可用
                detail = " ".join(unusable.split())
                if len(detail) > 200:
                    detail = detail[:200] + "…"
                return finish(False, False,
                              f"Core ML 无法为该模型构建执行计划（ANE 编译失败，"
                              f"常见原因是图过大或含不支持的结构）：{detail} "
                              f"模型已保存到 {out_path}，但很可能无法在设备上运行；"
                              "建议减小最大上下文长度或关闭权重量化后重试。",
                              out_path, files_done, bytes_done)
            hooks.on_log(message)
            return finish(True, False, message, out_path, files_done, bytes_done)
        finally:
            coreml_logger.removeFilter(noise_filter)
            for tmp_dir in tmp_dirs:
                shutil.rmtree(tmp_dir, ignore_errors=True)
