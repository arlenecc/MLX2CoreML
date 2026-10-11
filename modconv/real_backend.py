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
import os
import shutil
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from modconv.converter import ConversionResult
from modconv.utils import format_size

SEQ_LEN_CAP = 512

_SWIFT_EXAMPLE_TEMPLATE = '''// 由 modconv 自动生成 — 模型 "{model_name}" 的 Swift 调用示例
// stateful 模型：KV cache 存于 Core ML State，prefill 与 decode 共用同一模型。
// 输入: input_ids (Int32, [1, q])、cache_position (Int32, [q])，q ∈ 1..{max_seq_len}
// 输出: {output_name} (Float, [1, q, vocab_size])
// 注意: 需先用分词器（如 swift-transformers）把文本转成 token ids。
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

/// 单次前向：给定 token 与起始位置，返回最后一位 token 的 logits（MLMultiArray，长度 vocab）。
/// - prefill 时 tokens 为整段 prompt、startPos 为 0；
/// - decode 时 tokens 为单个 token、startPos 为已生成长度。
func forwardWithConvertedModel(_ model: MLModel, _ state: MLState,
                               tokens: [Int32], startPos: Int32) throws -> MLMultiArray {{
    let q = tokens.count
    let positions = (0..<q).map {{ startPos + Int32($0) }}
    let features = try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: tokens, shape: [1, q]))),
        "cache_position": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: positions, shape: [q]))),
    ])
    let output = try model.prediction(from: features, using: state)
    guard let logits = output.featureValue(for: "{output_name}")?.multiArrayValue,
          logits.count > 0 else {{
        throw ConvertedModelError.missingOutput("{output_name}")
    }}
    return logits
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

    // 1) prefill：整段 prompt 一次前向，KV cache 写入 state
    var position: Int32 = 0
    var logits = try forwardWithConvertedModel(model, state, tokens: promptTokens,
                                               startPos: position)
    position += Int32(promptTokens.count)

    // 2) decode：逐 token 前向，每步 cache_position = [position]，
    //    KV cache 由模型 state 自动携带，无需重算前缀
    var generated: [Int32] = []
    for _ in 0..<maxNewTokens {{
        let next = argmaxToken(logits)
        generated.append(next)
        if position >= {max_seq_len} {{ break }}
        logits = try forwardWithConvertedModel(model, state, tokens: [next],
                                               startPos: position)
        position += 1
    }}
    return generated
}}
'''


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
                return base / name
        except OSError:
            pass
    return Path(tempfile.gettempdir()) / name


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


def _rope_limit(config) -> Optional[int]:
    """模型位置编码上限（max_position_embeddings）；缺失或非法时返回 None。"""
    max_pos = getattr(config, "max_position_embeddings", None)
    try:
        return int(max_pos) if max_pos else None
    except (TypeError, ValueError):
        return None


def _pick_max_seq_len(config) -> int:
    """自动选取最大上下文长度：模型位置编码上限与全局上限取小。"""
    limit = _rope_limit(config) or SEQ_LEN_CAP
    return max(1, min(limit, SEQ_LEN_CAP))


# 既非 full/sliding/chunked 的层类型即视作 linear attention（GDN）
_ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


def _min_max_seq_len(config) -> int:
    """该架构允许的最小上下文长度。

    含 linear attention 的混合架构要求 >= 2：GDN 分块内核的分块循环次数依赖序列
    长度，wrapper 必须把输入 pad 到静态 max_seq_len，且 seq_len==1 的 decode 快速
    路径要在追踪期静态消除（见 KvCacheWrapper）。
    """
    layer_types = list(getattr(config, "layer_types", None) or [])
    if not layer_types:
        return 1
    has_linear = any(t not in _ATTENTION_LAYER_TYPES for t in layer_types)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    return 2 if (has_linear and len(layer_types) == num_layers) else 1


def _load_torch_model(source: Path, libs, config, dtype=None):
    """按 HF 架构加载模型权重（MLX 社区模型保留 HF 权重命名，可直接载入）。

    dtype 为 None 时用 float32；量化模型反量化后用 float16 以减半内存。
    注意必须用 eager 注意力：sdpa 的 math 分解会引入运行时形状 expand，
    让 ANE 执行计划构建失败（-14）。
    MoE 架构（如 qwen3_5_moe）还要强制 eager 专家实现：默认的融合内核用
    torch.histc 统计每张专家的 token 数，coremltools 前端不支持该算子
    （"Unsupported fx node histc"）。
    """
    transformers, torch = libs.transformers, libs.torch
    try:
        config.use_cache = False
    except AttributeError:
        pass
    # transformers 5.x 用 dtype（torch_dtype 已弃用）；旧版仅认 torch_dtype 时回退
    kwargs = {"config": config, "dtype": dtype or torch.float32,
              "attn_implementation": "eager",
              "experts_implementation": "eager"}
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        kwargs.pop("attn_implementation", None)
        kwargs.pop("experts_implementation", None)
        model = transformers.AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
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


def _trace_torch_model(libs, model, max_seq_len: int):
    """包装为 KV cache 状态模块并 torch.export 导出，返回 (exported, wrapper)。

    动态形状：input_ids 第 1 维与 cache_position 第 0 维共享同一 Dim
    （语义上二者长度恒相等；掩码与注意力相加也要求 q 维一致）。
    """
    torch = libs.torch
    wrapper = _build_kv_wrapper(model, max_seq_len).eval()
    # 示例长度取 min(max_seq_len, 4) 且必须 >=2：torch.export 会把尺寸为 1 的
    # 维度特化为常量（0/1 specialization），导致动态维度约束违规。
    example_len = min(max_seq_len, 4)
    example = (torch.ones((1, example_len), dtype=torch.long),
               torch.arange(example_len))
    # max_seq_len == 1 时 Dim(min=1, max=1) 非法（"inconsistent min=1, max=1"），
    # 此时序列长度恒为 1，直接导出静态形状即可
    if max_seq_len > 1:
        seq = torch.export.Dim("seq", min=1, max=max_seq_len)
        dynamic_shapes = {"input_ids": {1: seq}, "cache_position": {0: seq}}
    else:
        dynamic_shapes = None
    # 包在 no_grad 里导出：模型参数 requires_grad=True 时，KV buffer 的 in-place
    # scatter_ 变异会被 aot_export 拒绝（"graph input that requires gradients,
    # and received a mutation"）。桩 torch 没有 no_grad 时退化为空上下文。
    no_grad = getattr(torch, "no_grad", nullcontext)
    with no_grad():
        exported = torch.export.export(wrapper, example,
                                       dynamic_shapes=dynamic_shapes)
    return exported.run_decompositions(_decomposition_table(libs)), wrapper


def _convert_to_coreml(libs, exported, wrapper):
    """转换为带 KV cache 状态的 stateful mlprogram（iOS18 目标）。

    Core ML State 后端仅支持 fp16；KV buffer 固定 fp32（前端对 read_state 无条件
    插入 fp32 cast，见 KvCacheWrapper），state 读写由前端自动双向 cast。
    """
    ct, np = libs.ct, libs.np
    max_seq_len = wrapper.max_seq_len
    np_dtype = {"float16": np.float16, "float32": np.float32}
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=shape,
                                                      dtype=np_dtype[dtype_str]),
                           name=name)
              for name, shape, dtype_str in wrapper.kv_state_specs()]
    # max_seq_len == 1 时 RangeDim(default=1, upper_bound=1) 非法（上下界相同会
    # 退化成符号维度，与 EXIR 的静态形状 (1,1) 不匹配），直接用静态形状
    seq_dim = (1 if max_seq_len <= 1
               else ct.RangeDim(default=1, upper_bound=max_seq_len))
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, seq_dim), dtype=np.int32),
        ct.TensorType(name="cache_position", shape=(seq_dim,), dtype=np.int32),
    ]
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
                         output_name: str) -> Path:
    """在输出目录生成 Swift 调用示例文件。"""
    swift_path = out_path.parent / f"{out_path.stem}-swift-example.swift"
    swift_path.write_text(
        _SWIFT_EXAMPLE_TEMPLATE.format(model_name=model_name, max_seq_len=max_seq_len,
                                       output_name=output_name),
        encoding="utf-8")
    return swift_path


def _save_mlmodel(libs, mlmodel, path: Path):
    mlmodel.save(str(path))


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

        tmp_dir: Optional[Path] = None
        try:
            if quantized:
                stage("MLX 量化模型反量化（mlx-lm）…")
                # 反量化产物放在输出目录旁：避免数十 GB 中间文件撑爆系统盘
                tmp_dir = _make_dequant_tmp(self.output_dir)
                _dequantize_mlx_model(self.source, tmp_dir, hooks.on_log)
                if hooks.should_cancel():
                    return finish(False, True, "转换已取消", None, 0, 0)

            stage("加载模型配置与权重…")
            torch_source = tmp_dir or self.source
            config = _load_config(torch_source, libs)
            max_seq_len = self.max_seq_len or _pick_max_seq_len(config)
            # 用户显式指定的长度也不能超过模型位置编码上限（超出后 rope 语义失效）
            rope_limit = _rope_limit(config)
            if rope_limit and max_seq_len > rope_limit:
                hooks.on_log(f"最大上下文长度 {max_seq_len} 超过模型位置编码上限 "
                             f"{rope_limit}，已截断")
                max_seq_len = rope_limit
            # 混合注意力架构要求至少 2（GDN 分块内核需静态长度，见 _min_max_seq_len）
            min_len = _min_max_seq_len(config)
            if max_seq_len < min_len:
                hooks.on_log(f"最大上下文长度 {max_seq_len} 小于该架构所需的最小值 "
                             f"{min_len}，已调整")
                max_seq_len = min_len
            dtype = libs.torch.float16 if quantized else None
            model = _load_torch_model(torch_source, libs, config, dtype=dtype)
            if hooks.should_cancel():
                return finish(False, True, "转换已取消", None, 0, 0)

            stage("追踪 torch 计算图…")
            exported, wrapper = _trace_torch_model(libs, model, max_seq_len)
            if hooks.should_cancel():
                return finish(False, True, "转换已取消", None, 0, 0)

            stage("coremltools 转换为 Core ML 模型（大模型可能较慢）…")
            mlmodel = _convert_to_coreml(libs, exported, wrapper)
            # torch 侧大对象（模型权重 + 导出图）此后不再使用：coremltools 已持有
            # fp16 权重副本，尽早释放并回收，显著降低量化/保存阶段的峰值内存
            # （大模型可达数十 GB）。
            del exported, wrapper, model
            gc.collect()
            if hooks.should_cancel():
                return finish(False, True, "转换已取消", None, 0, 0)

            # 权重压缩为 4-bit palettization；失败直接抛异常（禁止降级为全精度输出）
            quantized_bits = 0
            if self.quantize_nbits:
                compressed = _compress_weights(libs, mlmodel, self.quantize_nbits,
                                               hooks.on_log)
                # 压缩前模型的临时权重目录（数十 GB 级）随旧对象一起释放
                del mlmodel
                gc.collect()
                mlmodel = compressed
                del compressed
                quantized_bits = self.quantize_nbits

            # get_spec() 返回深拷贝：在拷贝上重命名输出并写入元数据，
            # 再用 ct.models.MLModel(spec) 重建，确保落盘内容生效。
            spec = mlmodel.get_spec()
            if spec.description.output:
                old_name = spec.description.output[0].name
                if old_name != "logits":
                    try:
                        libs.ct.utils.rename_feature(spec, old_name, "logits")
                    except Exception:
                        pass  # 个别格式不支持重命名时保留原名，示例会用真实输出名
            input_descs = {
                "input_ids": f"Token ids (Int32)，形状 [1, q]，q ∈ 1..{max_seq_len}",
                "cache_position": f"各 token 的起始位置 (Int32)，长度 q ∈ 1..{max_seq_len}",
            }
            output_descs = {f.name: f"词表 logits (Float)，形状 [1, q, vocab_size]，"
                                    f"q ∈ 1..{max_seq_len}"
                            for f in spec.description.output}
            _describe_spec(spec, f"由模型 '{self.source.name}' 转换"
                                 f"（modconv + coremltools，KV cache 状态）",
                           input_descs, output_descs)
            output_name = (spec.description.output[0].name
                           if spec.description.output else "logits")
            # mlprogram 的 MLModel 不能仅从 spec 重建，必须带上原模型权重目录
            mlmodel = libs.ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

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
            swift_path = _write_swift_example(out_path, self.source.name, max_seq_len,
                                              output_name)
            hooks.on_log(f"已生成 Swift 调用示例: {swift_path}")
            files_done, bytes_done = _measure(out_path)

            bits_text = f", {quantized_bits}-bit 权重" if quantized_bits else ""
            message = (f"coremltools 转换完成: {out_path} "
                       f"(stateful mlprogram 格式, KV cache 状态模型{bits_text}, "
                       f"{files_done} 个文件, {format_size(bytes_done)}, "
                       f"输入长度 1..{max_seq_len}, "
                       f"耗时 {time.monotonic() - start:.1f} 秒)")
            hooks.on_log(message)
            return finish(True, False, message, out_path, files_done, bytes_done)
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
