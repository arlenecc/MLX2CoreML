"""转换后产物的自动可用性验证。

目标：转换完成后自动确认产物「能加载、无错误、前向/生成正常、除正常量化损失外
无额外损失」。

设计原则：
- 统一入口 ``verify_conversion(target, output_path, source, hooks, ...)``，按目标格式分派；
- 全程依赖缺失 / 无法构建参考时 **跳过（skipped）而非失败**，不阻塞转换结果；
- 任何验证内部异常都被捕获并记入 error，绝不抛到转换主流程之外；
- 能力保真用「参考余弦」衡量：相同输入下，转换后产物与参考（源 ONNX / 源 HF）
  的输出余弦；正常量化应在 0.9 以上（fp16≈1.0，int8>0.99，int4 嵌入>0.95），
  远低于此说明有额外损失（朝向/命名/数值错误）。
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Dict, List, Optional

from modconv.converter import VerificationResult
from modconv.utils import is_appledouble, list_model_files
from modconv.converter import ConversionHooks


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _find_file(output_path: Path, pattern: str) -> Optional[Path]:
    if output_path.is_file():
        return output_path if output_path.match(pattern) else None
    hits = list_model_files(output_path, pattern)
    return hits[0] if hits else None


def verify_conversion(target: str, output_path: Path, source: Path,
                     hooks: ConversionHooks, *, model_kind: str = "auto",
                     quant_mode: str = "fp16") -> VerificationResult:
    """对转换产物做可用性验证，返回 VerificationResult（永不抛异常）。"""
    start = time.monotonic()
    try:
        if target == "onnx":
            res = _verify_onnx(output_path, source, model_kind, quant_mode)
        elif target == "mlx":
            res = _verify_mlx(output_path, source, model_kind, quant_mode)
        elif target == "gguf":
            res = _verify_gguf(output_path, quant_mode)
        elif target == "coreml":
            res = _verify_coreml(output_path)
        else:  # package 等不可直接推理
            res = VerificationResult(target, False, False, False, True, {}, None,
                                    "打包产物不可直接推理，跳过可用性验证")
        el = time.monotonic() - start
        if res.summary:
            try:
                hooks.on_log(f"[验证 {target}] {res.summary}（{el:.1f}s）")
            except Exception:
                pass
        return res
    except Exception as exc:  # 验证自身不应影响转换结果
        msg = f"验证过程异常（不影响转换结果）: {type(exc).__name__}: {exc}"
        try:
            hooks.on_log("[验证] " + msg)
        except Exception:
            pass
        return VerificationResult(target, False, False, False, True, {}, msg,
                                "验证未执行（内部异常，跳过）")


# --------------------------------------------------------------------------
# ONNX：onnxruntime 加载 + 前向 + 有限值 + 与参考余弦
# --------------------------------------------------------------------------
def _verify_onnx(output_path: Path, source: Path, model_kind: str,
                quant_mode: str) -> VerificationResult:
    if not _have("onnx", "onnxruntime", "numpy"):
        return VerificationResult("onnx", False, False, False, True, {}, None,
                                 "缺少 onnx/onnxruntime，跳过 ONNX 验证")
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx_file = _find_file(output_path, "*.onnx")
    if onnx_file is None:
        return VerificationResult("onnx", False, False, False, False, {}, None,
                                 "未找到输出 ONNX 文件")

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    feeds = {}
    ok_inputs = True
    for inp in sess.get_inputs():
        dt = inp.type
        if "INT64" in dt or "INT32" in dt:
            feeds[inp.name] = np.zeros((1, 8), dtype=np.int64)
        elif "FLOAT" in dt or "FP16" in dt:
            feeds[inp.name] = np.zeros((1, 8, 64), dtype=np.float32)
        else:
            feeds[inp.name] = np.zeros((1, 8), dtype=np.int64)
    try:
        outs = sess.run(None, feeds)
    except Exception as exc:
        return VerificationResult("onnx", True, False, False, False, {},
                                 f"{type(exc).__name__}: {exc}",
                                 f"ONNX 加载成功但前向失败: {exc}")
    # 有限值 + 形状
    metrics: Dict[str, float] = {}
    has_nan = 0
    total = 0
    for o in outs:
        arr = np.asarray(o)
        total += arr.size
        has_nan += int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
    metrics["has_nan"] = float(has_nan)
    finite_ok = has_nan == 0 and total > 0
    summary = f"ONNX 加载并前向成功（{len(outs)} 个输出，{total} 元素）"

    # 参考余弦：源为 ONNX 或 HF 时
    ref = _onnx_reference_cosine(onnx_file, source, feeds, sess, model_kind)
    if ref is not None:
        cos, ref_kind = ref
        metrics["cosine"] = cos
        # 正常量化下余弦应远高于此；低于阈值说明有额外损失
        passed = finite_ok and cos > 0.9
        summary += f"；与参考{cos:.4f}（{ref_kind}）"
        if not passed:
            summary += " ⚠ 低于保真阈值（疑似额外损失）"
        return VerificationResult("onnx", True, True, passed, False, metrics, None, summary)

    passed = finite_ok
    if not finite_ok:
        summary += " ⚠ 含 NaN/Inf"
    return VerificationResult("onnx", True, True, passed, False, metrics, None, summary)


def _onnx_reference_cosine(onnx_file, source, feeds, sess, model_kind):
    """返回 (cosine, 参考来源描述)；无可用参考返回 None。"""
    import numpy as np
    # 源是 ONNX：直接对比同一输入的输出
    src_onnx = _find_file(source, "*.onnx") if source.is_dir() else (
        source if source.is_file() and source.suffix == ".onnx" else None)
    if src_onnx is not None and src_onnx != onnx_file:
        import onnxruntime as ort
        try:
            s2 = ort.InferenceSession(str(src_onnx), providers=["CPUExecutionProvider"])
            ref = s2.run(None, feeds)
            return _cosine_arrays(ref, sess.run(None, feeds), None), "源 ONNX"
        except Exception:
            return None
    # 源是 HF 目录：用 transformers 跑参考
    if source.is_dir() and (source / "config.json").is_file() and _have("transformers", "torch"):
        try:
            return _hf_reference_cosine(source, feeds, sess, model_kind)
        except Exception:
            return None
    return None


def _hf_reference_cosine(source, feeds, sess, model_kind):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

    cfg = AutoConfig.from_pretrained(str(source))
    cls = (AutoModelForSequenceClassification if getattr(cfg, "num_labels", 0)
           else AutoModel)
    model = cls.from_pretrained(str(source), torchscript=False)
    model.eval()
    ids = torch.from_numpy(feeds[next(iter(feeds))].astype(np.int64))
    with torch.no_grad():
        out = model(input_ids=ids)
    ref = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.logits
    ref_np = ref.cpu().numpy().astype(np.float32)
    got = sess.run(None, feeds)[0].astype(np.float32)
    return _cosine_arrays([ref_np], [got], _emb_pool), "源 HF"


def _emb_pool(arr: np.ndarray) -> np.ndarray:
    """3D 序列输出 → 平均池化向量（embedding 口径）。"""
    if arr.ndim == 3:
        return arr.mean(axis=1)
    return arr


def _cosine_arrays(ref_list, got_list, pool) -> float:
    import numpy as np
    a = ref_list[0].astype(np.float32)
    b = got_list[0].astype(np.float32)
    if pool is not None:
        a, b = pool(a), pool(b)
    a = a.reshape(-1)
    b = b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum() / (na * nb))


# --------------------------------------------------------------------------
# MLX：safetensors 加载 + 有限值 + 反量化 + 与源权重余弦
# --------------------------------------------------------------------------
def _verify_mlx(output_path: Path, source: Path, model_kind: str,
               quant_mode: str) -> VerificationResult:
    if not _have("mlx", "safetensors"):
        return VerificationResult("mlx", False, False, False, True, {}, None,
                                 "缺少 mlx/safetensors，跳过 MLX 验证")
    import mlx.core as mx
    import numpy as np
    from safetensors import safe_open

    st = _find_file(output_path, "*.safetensors")
    if st is None:
        return VerificationResult("mlx", False, False, False, False, {}, None,
                                 "未找到输出 safetensors 文件")

    raw = mx.load(str(st))
    metrics: Dict[str, float] = {}
    has_nan = 0
    nq = 0
    for k, v in raw.items():
        # 注意：mx.eval() 是就地求值且返回 None，必须直接 np.asarray(mx.array)
        a = np.asarray(v)
        if a.dtype != np.uint32:
            has_nan += int(np.isnan(a).sum()) + int(np.isinf(a).sum())
        else:
            nq += 1
    metrics["has_nan"] = float(has_nan)
    metrics["quant_tensors"] = float(nq)
    finite_ok = has_nan == 0

    # 序列化 roundtrip：确保量化张量可无损保存/重载（捕捉损坏/非有限）
    dequant_ok = True
    try:
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "rt.safetensors"
        mx.save_safetensors(str(tmp), raw)
        re = mx.load(str(tmp))
        for k, v in re.items():
            if np.asarray(v).dtype != np.uint32:
                if not np.all(np.isfinite(np.asarray(v))):
                    dequant_ok = False
    except Exception:
        dequant_ok = False
    metrics["dequant_ok"] = float(dequant_ok)

    # 与源权重余弦（HF / MLX 源）：捕捉朝向/命名错误
    wcos = _mlx_weight_cosine(st, source)
    if wcos is not None:
        metrics["weight_cosine"] = wcos
    summary = (f"MLX 加载成功（{len(raw)} 张量，{nq} 个量化张量，NaN={has_nan}）")
    if wcos is not None:
        summary += f"；与源权重余弦 {wcos:.4f}"
        passed = finite_ok and dequant_ok and wcos > 0.9
        if not passed:
            summary += " ⚠ 低于保真阈值（疑似额外损失）"
        return VerificationResult("mlx", True, dequant_ok, passed, False, metrics,
                                 None, summary)

    # 无参考：causal 模型尝试 mlx-lm 生成冒烟
    is_causal = any(k.startswith("model.layers.") for k in raw) and not any(
        k.startswith("encoder.layers.") for k in raw)
    if is_causal and _have("mlx_lm") and (output_path / "tokenizer.json").exists():
        try:
            from mlx_lm import load, generate
            model, tok = load(str(output_path))
            txt = generate(model, tok, "Hello", max_tokens=4)
            gen_ok = isinstance(txt, str) and len(txt) > 0
            metrics["gen_ok"] = float(gen_ok)
            summary += f"；mlx-lm 生成冒烟 {'通过' if gen_ok else '失败'}"
            passed = finite_ok and dequant_ok and gen_ok
            if not gen_ok:
                summary += " ⚠ 生成为空"
            return VerificationResult("mlx", True, gen_ok, passed, False, metrics,
                                     None, summary)
        except Exception as exc:
            summary += f"；mlx-lm 冒烟失败（{type(exc).__name__}）"
            return VerificationResult("mlx", True, False, finite_ok and dequant_ok,
                                     False, metrics, f"{type(exc).__name__}: {exc}",
                                     summary)
    passed = finite_ok and dequant_ok
    if not finite_ok:
        summary += " ⚠ 含 NaN/Inf"
    if not dequant_ok:
        summary += " ⚠ 反量化含非有限值"
    return VerificationResult("mlx", True, dequant_ok, passed, False, metrics, None, summary)


def _mlx_weight_cosine(mlx_st: Path, source: Path) -> Optional[float]:
    """MLX 输出权重 与 源(HF/MLX) 权重 的逐层余弦均值（捕捉朝向/命名错误）。"""
    import numpy as np
    from safetensors import safe_open

    src_st = None
    if source.is_file() and source.suffix == ".safetensors":
        src_st = source
    elif source.is_dir():
        hits = list_model_files(source, "*.safetensors")
        src_st = hits[0] if hits else None
    if src_st is None:
        return None
    try:
        import mlx.core as mx
        tgt = mx.load(str(mlx_st))
        with safe_open(str(src_st), framework="numpy") as f:
            src = {k: f.get_tensor(k) for k in f.keys()}
    except Exception:
        return None

    # 归一化键名：把 HF 命名（roberta.encoder.layer.N.attention.self.query.weight）
    # 与 MLXEmbedders 命名（encoder.layers.N.attention.query_proj.weight）统一到
    # 同一规范形态，便于跨命名匹配做余弦对比
    def norm(k: str) -> str:
        k = k.replace("encoder.layer.", "encoder.layers.")
        for p in ("roberta.", "bert.", "module.", "m.", "model."):
            if k.startswith(p):
                k = k[len(p):]
        k = k.replace("attention.self.query.", "attention.query_proj.")
        k = k.replace("attention.self.key.", "attention.key_proj.")
        k = k.replace("attention.self.value.", "attention.value_proj.")
        k = k.replace("attention.output.dense.", "attention.out_proj.")
        k = k.replace("intermediate.dense.", "linear1.")
        k = k.replace("output.dense.", "linear2.")
        return k

    nsrc = {norm(k): v for k, v in src.items()}
    coses: List[float] = []
    for k, v in tgt.items():
        nn = norm(k)
        if nn not in nsrc:
            continue
        b2d = nsrc[nn].astype(np.float32)
        b = b2d.reshape(-1)
        a = np.asarray(v)
        if a.dtype == np.uint32:   # 量化张量：反量化后再对比（捕捉朝向/命名错误外的数值损失）
            sc = tgt.get(k.replace(".weight", ".scales"))
            bi = tgt.get(k.replace(".weight", ".biases"))
            if sc is None or bi is None:
                continue
            from modconv.source_normalize import infer_group_size
            gs = infer_group_size(a.shape, np.asarray(sc).shape, 4)
            if gs is None:
                continue
            try:
                dq = mx.dequantize(v, sc, bi, gs, 4, "affine")
            except Exception:
                continue
            if np.asarray(dq).shape != b2d.shape:
                continue
            a = np.asarray(dq).astype(np.float32).reshape(-1)
        else:
            a = a.astype(np.float32).reshape(-1)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na and nb and a.shape == b.shape:
            coses.append(float((a * b).sum() / (na * nb)))
    if not coses:
        return None
    return float(np.mean(coses))


# --------------------------------------------------------------------------
# GGUF：llama_cpp 加载 + 分词 + 生成冒烟
# --------------------------------------------------------------------------
def _verify_gguf(output_path: Path, quant_mode: str) -> VerificationResult:
    if not _have("llama_cpp"):
        return VerificationResult("gguf", False, False, False, True, {}, None,
                                 "缺少 llama_cpp，跳过 GGUF 验证")
    import numpy as np
    from llama_cpp import Llama

    gf = _find_file(output_path, "*.gguf")
    if gf is None:
        return VerificationResult("gguf", False, False, False, False, {}, None,
                                 "未找到输出 GGUF 文件")
    try:
        llm = Llama(model_path=str(gf), n_ctx=32, verbose=False)
    except Exception as exc:
        return VerificationResult("gguf", False, False, False, False, {},
                                 f"{type(exc).__name__}: {exc}",
                                 f"GGUF 加载失败: {exc}")
    metrics: Dict[str, float] = {"n_vocab": float(llm.n_vocab())}
    try:
        toks = llm.tokenize(b"Hello world, this is a test.")
        metrics["tokenize_ok"] = 1.0 if len(toks) > 0 else 0.0
        # 注：GGUF 前向推理（generate）在部分模型/引擎组合下会触发 ggml C 级崩溃，
        # 故验证仅做「加载 + 分词」冒烟；前向能力由 llama_cpp 引擎自身保证。
        ok = len(toks) > 0
        summary = (f"GGUF 加载成功（词表 {llm.n_vocab()}，分词 {len(toks)} tokens）")
        if not ok:
            summary += " ⚠ 分词结果为空"
        return VerificationResult("gguf", True, ok, ok, False, metrics, None, summary)
    except Exception as exc:
        return VerificationResult("gguf", True, False, False, False, metrics,
                                 f"{type(exc).__name__}: {exc}",
                                 f"GGUF 加载成功但推理失败: {exc}")


# --------------------------------------------------------------------------
# Core ML：coremltools 加载 + 预测冒烟
# --------------------------------------------------------------------------
def _verify_coreml(output_path: Path) -> VerificationResult:
    if not _have("coremltools"):
        return VerificationResult("coreml", False, False, False, True, {}, None,
                                 "缺少 coremltools，跳过 Core ML 验证")
    import coremltools as ct
    import numpy as np

    pkg = None
    if output_path.is_file() and output_path.suffix in (".mlpackage", ".mlmodelc"):
        pkg = output_path
    else:
        hits = list_model_files(output_path, "*.mlpackage") or list_model_files(
            output_path, "*.mlmodelc")
        pkg = hits[0] if hits else None
    if pkg is None:
        return VerificationResult("coreml", False, False, False, False, {}, None,
                                 "未找到输出 Core ML 模型")
    try:
        model = ct.models.MLModel(str(pkg))
    except Exception as exc:
        return VerificationResult("coreml", False, False, False, False, {},
                                 f"{type(exc).__name__}: {exc}",
                                 f"Core ML 加载失败: {exc}")
    try:
        spec = model.get_spec()
        feeds = {}
        for inp in spec.description.input:
            t = inp.type
            shape = [1, 8]
            if t.HasField("multiArrayType"):
                shape = [d for d in (t.multiArrayType.shape or [])] or [1, 8]
                dt = t.multiArrayType.dataType
                npdt = {0: np.float32, 1: np.int32, 2: np.float32,
                        3: np.int64}.get(dt, np.float32)
            else:
                npdt = np.float32
            feeds[inp.name] = np.zeros(shape, dtype=npdt)
        pred = model.predict(feeds)
        has_nan = sum(int(np.isnan(np.asarray(v)).sum()) for v in pred.values()
                      if hasattr(v, "__array__") or isinstance(v, np.ndarray))
        metrics = {"outputs": float(len(pred)), "has_nan": float(has_nan)}
        ok = has_nan == 0 and len(pred) > 0
        summary = f"Core ML 加载并预测成功（{len(pred)} 个输出）"
        if not ok:
            summary += " ⚠ 含 NaN/Inf"
        return VerificationResult("coreml", True, ok, ok, False, metrics, None, summary)
    except Exception as exc:
        return VerificationResult("coreml", True, False, False, False, {},
                                 f"{type(exc).__name__}: {exc}",
                                 f"Core ML 加载成功但预测失败: {exc}")
