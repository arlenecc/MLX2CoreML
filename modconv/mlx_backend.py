"""ONNX→MLX 转换后端（encoder 类：BERT / RoBERTa / XLM-R / nomic-bert）。

按 tools/ 下已验证的转换逻辑实现（bge-reranker / jina-reranker-v2 实测通过）：

- 角色解析按**节点名**（语义化），不依赖权重初始化器名——bge/jina 的 Optimum 导出中
  MatMul 权重名是匿名 `onnx::MatMul_NNNN`，只有节点名是语义的
  （`/roberta/encoder/layer.0/attention/self/query/MatMul`）；自定义 torch.onnx.export
  则保留 HF 命名。两种方言统一用 `[./]` 容错正则解析。
- **朝向**（最易错点）：编码器 MatMul 权重无 Transpose 前置 → 存为 [in,out] →
  转置为 [out,in]（HF 布局）；有 Transpose 前置（torch.linear 分解）则已是 [out,in]。
  分类头 Gemm transB=1 → 已是 [out,in] 不转置；嵌入表 Gather 消费 → 不转置。
- **融合 QKV**（jina `mixer/Wqkv`，[in, 3H]）：转置后按行拆成 query/key/value_proj。
- 输出为 **MLXEmbedders roberta 命名**（encoder.layers.N.attention.query_proj 等），
  量化陪衬键是**兄弟键**（{prefix}.weight / .scales / .biases，绝不能 .weight.scales）。
- config.json = 源 config + quantization（量化时）+ num_labels（有分类头时=1）；
  源缺 model_type 时补 xlm-roberta（jina 源只有 auto_map，缺它会加载失败）。
- jina 源无 embeddings.LayerNorm，但 MLXEmbedders 的 BertEmbedding 总会创建并应用
  embeddings.norm，缺失会让 verify:[.all] 加载器抛错——补 ones/zeros 占位（经第一层
  ln1 抵消，不影响 reranker）。
- pooler 一律丢弃（MLXEmbedders reranker 的 classifier 直接消费 CLS）。

需要源 config.json 与 ONNX 同目录（tokenizer 也从该目录拷贝）。不依赖 Qt，可独立测试。
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modconv.converter import ConversionHooks, ConversionResult
from modconv.utils import is_appledouble, list_model_files
from modconv.backend import resolve_sidecar_dir

# ONNX 节点名中的分隔符在 torch 导出与 Optimum 导出间不同（/ 与 .），统一 [./] 容错。

# MatMul 节点名 → 角色（含 jina 的 mixer/mlp 方言与融合 Wqkv）
_MATMUL_ROLES = [
    (re.compile(r"encoder[./]layers?\.(\d+)[./]attention[./]self[./](query|key|value)[./]MatMul"),
     "attn"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]attention[./]output[./]dense[./]MatMul"),
     "out_proj"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]intermediate[./]dense[./]MatMul"), "linear1"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]output[./]dense[./]MatMul"), "linear2"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]mixer[./]Wqkv[./]MatMul"), "qkv_fused"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]mixer[./]out_proj[./]MatMul"), "out_proj"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]mlp[./]fc1[./]MatMul"), "linear1"),
    (re.compile(r"encoder[./]layers?\.(\d+)[./]mlp[./]fc2[./]MatMul"), "linear2"),
    # Optimum/SentenceTransformers 方言：/model/layers.0/attn/q_proj/MatMul（无 encoder. 前缀）
    (re.compile(r"layers?\.(\d+)[./]attn[./](q)_proj[./]MatMul"), "attn_opt"),
    (re.compile(r"layers?\.(\d+)[./]attn[./](k)_proj[./]MatMul"), "attn_opt"),
    (re.compile(r"layers?\.(\d+)[./]attn[./](v)_proj[./]MatMul"), "attn_opt"),
    (re.compile(r"layers?\.(\d+)[./]attn[./]o_proj[./]MatMul"), "out_proj"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./]fc1[./]MatMul"), "linear1"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./]fc2[./]MatMul"), "linear2"),
]

# Optimum 方言的 q/k/v 简写 → MLXEmbedders 完整名
_OPT_ATTN_NAMES = {"q": "query", "k": "key", "v": "value"}

# Gemm 节点名 → 分类头角色（transB=1 → 权重已是 [out,in]）
# 顺序敏感：具体的 classifier/dense|out_proj 先于 bare classifier 与 pooler
_GEMM_ROLES = [
    (re.compile(r"classifier[./](dense|out_proj)[./]Gemm"), "cls_named"),
    (re.compile(r"pooler[./]dense[./]Gemm"), "cls_pooler"),
    (re.compile(r"classifier[./]Gemm"), "cls_bare"),
]

# 1D 偏置 / LayerNorm（initializer 名）。注意顺序：更具体的先匹配
# （attention.output.LayerNorm 必须在 output.LayerNorm 之前）。
_1D_ROLES = [
    (re.compile(r"layers?\.(\d+)[./]attention[./]self[./](query|key|value)[./]bias"),
     lambda L, r: f"encoder.layers.{L}.attention.{r}_proj.bias"),
    (re.compile(r"layers?\.(\d+)[./]attention[./]output[./]dense[./]bias"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.bias"),
    (re.compile(r"layers?\.(\d+)[./]intermediate[./]dense[./]bias"),
     lambda L: f"encoder.layers.{L}.linear1.bias"),
    (re.compile(r"layers?\.(\d+)[./]output[./]dense[./]bias"),
     lambda L: f"encoder.layers.{L}.linear2.bias"),
    (re.compile(r"layers?\.(\d+)[./]mixer[./]Wqkv[./]bias"), "qkv_bias"),
    (re.compile(r"pooler[./]dense[./]bias"), "cls_pooler_bias"),
    (re.compile(r"classifier[./]bias"), "cls_bare_bias"),
    (re.compile(r"layers?\.(\d+)[./]mixer[./](out_proj)[./](weight|bias)"),
     lambda L, k, r: f"encoder.layers.{L}.attention.out_proj.{r}"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./](fc1|fc2)[./](weight|bias)"),
     lambda L, k, r: f"encoder.layers.{L}.{'linear1' if k == 'fc1' else 'linear2'}.{r}"),
    (re.compile(r"layers?\.(\d+)[./](norm1|norm2)[./](weight|bias)"),
     lambda L, k, r: f"encoder.layers.{L}.{'ln1' if k == 'norm1' else 'ln2'}.{r}"),
    (re.compile(r"layers?\.(\d+)[./]attention[./]output[./]LayerNorm[./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln1.{r}"),
    (re.compile(r"layers?\.(\d+)[./]output[./]LayerNorm[./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln2.{r}"),
    (re.compile(r"embeddings[./]LayerNorm[./](weight|bias)"),
     lambda r: f"embeddings.norm.{r}"),
    (re.compile(r"classifier[./](dense|out_proj)[./]bias"),
     lambda k: f"classifier.{k}.bias"),
    # ---- Optimum / SentenceTransformers 方言（/model/layers.0/attn/q_proj/...）----
    # 偏置名带 .Add. 中缀；LayerNorm 名带 _norm_layernorm 后缀；嵌入 LN 用 layers.-1
    (re.compile(r"layers?\.(\d+)[./]attn[./](q|k|v)_proj(?:[./]Add)?[./]bias"),
     lambda L, p: f"encoder.layers.{L}.attention.{_OPT_ATTN_NAMES[p]}_proj.bias"),
    (re.compile(r"layers?\.(\d+)[./]attn[./]o_proj(?:[./]Add)?[./]bias"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.bias"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./](fc1|fc2)(?:[./]Add)?[./]bias"),
     lambda L, k: f"encoder.layers.{L}.{'linear1' if k == 'fc1' else 'linear2'}.bias"),
    (re.compile(r"layers?\.(\d+)[./]attention_output_norm_layernorm[./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln1.{r}"),
    (re.compile(r"layers?\.(\d+)[./]mlp_output_norm_layernorm[./](weight|bias)"),
     lambda L, r: f"encoder.layers.{L}.ln2.{r}"),
    (re.compile(r"layers?\.-1[./]embeddings_norm_layernorm[./](weight|bias)"),
     lambda r: f"embeddings.norm.{r}"),
]

# 命名 2D 权重 initializer（自定义 torch 导出保留 HF 名时走这里）
_NAMED_2D_ROLES = [
    (re.compile(r"layers?\.(\d+)[./]attention[./]self[./](query|key|value)[./]weight"),
     lambda L, r: f"encoder.layers.{L}.attention.{r}_proj.weight"),
    (re.compile(r"layers?\.(\d+)[./]attention[./]output[./]dense[./]weight"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.weight"),
    (re.compile(r"layers?\.(\d+)[./]intermediate[./]dense[./]weight"),
     lambda L: f"encoder.layers.{L}.linear1.weight"),
    (re.compile(r"layers?\.(\d+)[./]output[./]dense[./]weight"),
     lambda L: f"encoder.layers.{L}.linear2.weight"),
    (re.compile(r"layers?\.(\d+)[./]mixer[./]out_proj[./]weight"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.weight"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./](fc1|fc2)[./]weight"),
     lambda L, k: f"encoder.layers.{L}.{'linear1' if k == 'fc1' else 'linear2'}.weight"),
    (re.compile(r"classifier[./](dense|out_proj)[./]weight"),
     lambda k: f"classifier.{k}.weight"),
    # bert 系 seq-cls：pooler.dense(tanh) + bare classifier(单 Linear)
    # → MLXEmbedders SequenceClassificationHead 的 dense + out_proj（等价函数）
    (re.compile(r"pooler[./]dense[./]weight"),
     lambda: "classifier.dense.weight"),
    (re.compile(r"classifier[./]weight"),
     lambda: "classifier.out_proj.weight"),
    # ---- Optimum 方言：权重名带 .MatMul. 中缀（layers.0.attn.q_proj.MatMul.weight）----
    (re.compile(r"layers?\.(\d+)[./]attn[./](q|k|v)_proj(?:[./]MatMul)?[./]weight"),
     lambda L, p: f"encoder.layers.{L}.attention.{_OPT_ATTN_NAMES[p]}_proj.weight"),
    (re.compile(r"layers?\.(\d+)[./]attn[./]o_proj(?:[./]MatMul)?[./]weight"),
     lambda L: f"encoder.layers.{L}.attention.out_proj.weight"),
    (re.compile(r"layers?\.(\d+)[./]mlp[./](fc1|fc2)(?:[./]MatMul)?[./]weight"),
     lambda L, k: f"encoder.layers.{L}.{'linear1' if k == 'fc1' else 'linear2'}.weight"),
]

_EMBED_ROLES = [
    (re.compile(r"embeddings[./](word|position|token_type)_embeddings[./]weight"),
     lambda k: f"embeddings.{k}_embeddings.weight"),
]

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "vocab.txt",
                    "vocab.json", "merges.txt")


class OnnxToMlxBackend:
    """把 ONNX（encoder 类）转换为 MLX 格式（MLXEmbedders roberta 命名）。

    quant_mode: "fp16" / "int4" / "int8"（affine 量化，group_size=64，逐张量回退）。
    输出：model.safetensors + config.json + tokenizer 文件。
    """

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16", group_size: int = 64):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.group_size = group_size

    @staticmethod
    def _import_deps():
        if (importlib.util.find_spec("onnx") is None
                or importlib.util.find_spec("mlx") is None):
            raise RuntimeError("ONNX→MLX 需要 onnx / mlx，请先安装")
        import onnx
        import mlx.core as mx
        return onnx, mx

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            onnx, mx = self._import_deps()
            return self._run(onnx, mx, hooks, finish)
        except Exception as exc:
            msg = f"ONNX→MLX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)

    # ------------------------------------------------------------------
    def _run(self, onnx, mx, hooks, finish):
        # 伴随目录：config.json 可能在 onnx 文件同级或其父目录（Optimum 布局）
        src_dir = resolve_sidecar_dir(self.source)
        onnx_path = self.source if self.source.is_file() else self._find_onnx(src_dir)
        cfg_path = src_dir / "config.json"
        if not cfg_path.is_file():
            raise RuntimeError(
                "ONNX→MLX 需要源 config.json（含 model_type/结构信息）与 ONNX 同目录。"
                "若只有 ONNX 文件，请先补齐 config.json/tokenizer.json。")
        src_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        model = onnx.load(str(onnx_path))
        bits = {"int4": 4, "int8": 8}.get(self.quant_mode)  # None → fp16 不量化
        # 源 config 带 num_labels/id2label → reranker（决定 pooler/bare classifier 的映射）
        is_reranker = bool(src_cfg.get("num_labels") or src_cfg.get("id2label"))

        state = self._extract(mx, model, bits, hooks, is_reranker)
        hidden = int(state["embeddings.word_embeddings.weight"].shape[-1])
        has_cls = any(k.startswith("classifier.") for k in state)
        self._fill_placeholders(mx, state, hidden)

        # ---- config.json ----
        cfg = dict(src_cfg)
        cfg.setdefault("model_type", "xlm-roberta")  # jina 源缺 model_type（只有 auto_map）
        if bits:
            cfg["quantization"] = {"bits": bits, "group_size": self.group_size,
                                   "mode": "affine"}
        else:
            cfg.pop("quantization", None)
        if has_cls:
            cfg["num_labels"] = 1

        # ---- 写出 ----
        self.output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.output_dir / "model.safetensors"
        mx.save_safetensors(str(weights_path), state)
        (self.output_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        copied = 0
        for fname in _TOKENIZER_FILES:
            sp = src_dir / fname
            if sp.is_file():
                shutil.copy(sp, self.output_dir / fname)
                copied += 1
        if copied:
            hooks.on_log(f"已拷贝 {copied} 个 tokenizer 文件")

        total = sum(p.stat().st_size for p in self.output_dir.glob("*"))
        kind = "reranker" if has_cls else "embedding"
        msg = (f"ONNX→MLX 完成: {weights_path.name}（{kind}, "
               f"{self.quant_mode}, {len(state)} 张量）")
        hooks.on_log(msg)
        return finish(True, False, msg, weights_path, 1, total)

    @staticmethod
    def _find_onnx(src_dir: Path) -> Path:
        cands = list_model_files(src_dir, "*.onnx")
        if not cands:
            sub = src_dir / "onnx"
            if sub.is_dir():
                cands = list_model_files(sub, "*.onnx")
        if not cands:
            raise FileNotFoundError(f"未找到 ONNX 文件: {src_dir}")
        # 优先 fp32/fp16 基线，避免误选已量化文件
        for pref in ("model.onnx", "model_fp16.onnx"):
            for c in cands:
                if c.name == pref:
                    return c
        return cands[0]

    # ------------------------------------------------------------------
    # 权重抽取（核心：角色解析 + 朝向 + 融合 QKV 拆分）
    # ------------------------------------------------------------------
    def _extract(self, mx, model, bits, hooks, is_reranker: bool) -> Dict:
        import numpy as np
        import onnx.numpy_helper as nh

        init: Dict[str, "np.ndarray"] = {
            i.name: nh.to_array(i) for i in model.graph.initializer}
        input2nodes: Dict[str, list] = {}
        for n in model.graph.node:
            for i in n.input:
                input2nodes.setdefault(i, []).append(n)

        state: Dict = {}
        consumed: set = set()

        # ---- 图遍历：把内联为 Constant 的 1D 偏置/LayerNorm 物化到 init ----
        # torch 导出（do_constant_folding=False）中 2D 权重是命名 initializer，但 bias 与
        # LayerNorm 的 scale/bias 是匿名 Constant。Add/LayerNormalization 的输入名保留
        # 原始 HF 名（m.encoder.layer.0.attention.self.query.bias 等），据此物化。
        import onnx.numpy_helper as _nh
        node_by_out = {o: n for n in model.graph.node for o in n.output}

        def value_of(name):
            if name in init:
                return init[name]
            node = node_by_out.get(name)
            if node is None:
                return None
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value" and a.t is not None:
                        return _nh.to_array(a.t)
                return None
            if node.op_type in ("Identity", "Reshape", "Transpose", "Squeeze",
                                "Unsqueeze", "Cast", "Flatten"):
                return value_of(node.input[0]) if node.input else None
            return None

        matmul_outs = {mm.output[0] for mm in model.graph.node
                       if mm.op_type == "MatMul"}
        for n2 in model.graph.node:
            if n2.op_type != "Add":
                continue
            mm_out = [i for i in n2.input if i in matmul_outs]
            if not mm_out:
                continue
            bias_name = next((i for i in n2.input if i != mm_out[0]), None)
            if bias_name is None:
                continue
            bv = value_of(bias_name)
            if bv is not None and bv.ndim == 1 and bias_name not in init:
                init[bias_name] = bv
        for node in model.graph.node:
            if node.op_type != "LayerNormalization":
                continue
            for idx in (1, 2):
                nm = node.input[idx]
                t = value_of(nm)
                if t is not None and nm not in init:
                    init[nm] = t

        def q2d(arr: "np.ndarray"):
            """2D 权重 → (weight, scales, biases)；fp16 模式返回 (arr, None, None)。"""
            a = mx.array(np.ascontiguousarray(arr), dtype=mx.float16)
            if bits is None:
                return a, None, None
            for gs in (self.group_size, 32, 16, 8):
                if arr.shape[-1] % gs == 0:
                    wq, sc, bi = mx.quantize(a, gs, bits, mode="affine")
                    if gs != self.group_size:
                        hooks.on_log(f"group_size 回退为 {gs}（维度 {arr.shape}）")
                    return wq, sc, bi
            hooks.on_log(f"无法量化（维度 {arr.shape}），保持 fp16")
            return a, None, None

        def store(prefix: str, wq, sc, bi):
            state[f"{prefix}.weight"] = wq
            if sc is not None:
                # 陪衬键是兄弟键（去 .weight 加 .scales/.biases），切勿写成 .weight.scales
                state[f"{prefix}.scales"] = sc
                state[f"{prefix}.biases"] = bi

        def w_transposed(w_name: str) -> bool:
            """编码器 MatMul 权重无 Transpose 前置 → [in,out] 需转置为 [out,in]。"""
            for c in input2nodes.get(w_name, []):
                if c.op_type == "Transpose":
                    return False
            return True

        # 1) 编码器 MatMul 2D 权重（含融合 Wqkv 拆分）
        for n in model.graph.node:
            if n.op_type != "MatMul":
                continue
            w_names = [i for i in n.input if i in init and init[i].ndim == 2]
            if not w_names:
                continue
            w_name = w_names[0]
            arr = init[w_name]
            role = None
            layer = 0
            for pat, tag in _MATMUL_ROLES:
                m = pat.search(n.name or "")
                if m:
                    layer = int(m.group(1))
                    role = tag
                    break
            if role is None:
                hooks.on_log(f"[warn] 未匹配 MatMul 节点名: {n.name}")
                continue
            w = arr.T if w_transposed(w_name) else arr  # → [out, in]
            if role == "attn":
                sub = m.group(2)
                store(f"encoder.layers.{layer}.attention.{sub}_proj", *q2d(w))
            elif role == "attn_opt":
                # Optimum 方言：group(2) 为 q/k/v 简写，映射为 query/key/value
                full = _OPT_ATTN_NAMES.get(m.group(2))
                if full is None:
                    hooks.on_log(f"[warn] 未知 attn 投影: {n.name}")
                    continue
                store(f"encoder.layers.{layer}.attention.{full}_proj", *q2d(w))
            elif role == "qkv_fused":
                h = w.shape[0] // 3
                for proj, piece in (("query_proj", w[0:h]),
                                    ("key_proj", w[h:2 * h]),
                                    ("value_proj", w[2 * h:3 * h])):
                    store(f"encoder.layers.{layer}.attention.{proj}", *q2d(piece))
            elif role == "out_proj":
                store(f"encoder.layers.{layer}.attention.out_proj", *q2d(w))
            elif role in ("linear1", "linear2"):
                store(f"encoder.layers.{layer}.{role}", *q2d(w))
            consumed.add(w_name)

        # 2) 分类头 Gemm（transB=1 → 已是 [out,in]）
        #    cls_named: classifier/{dense,out_proj}（bge/jina 方言）
        #    cls_pooler: pooler/dense（bert 系 pooler=tanh dense → classifier.dense）
        #    cls_bare:   bare classifier（bert 系单 Linear 头 → classifier.out_proj）
        for n in model.graph.node:
            if n.op_type != "Gemm":
                continue
            w_names = [i for i in n.input if i in init and init[i].ndim == 2]
            if not w_names:
                continue
            w_name = w_names[0]
            hit = None
            for pat, tag in _GEMM_ROLES:
                m = pat.search(n.name or "")
                if m:
                    hit = (m, tag)
                    break
            if hit is None:
                continue
            m, tag = hit
            if tag == "cls_named":
                dest = f"classifier.{m.group(1)}"
            elif tag == "cls_pooler":
                dest = "classifier.dense"
            else:  # cls_bare
                dest = "classifier.out_proj"
            if not is_reranker:
                consumed.add(w_name)  # 嵌入模型：pooler/classifier 丢弃
                continue
            if dest in {k[:-len(".weight")] for k in state if k.endswith(".weight")}:
                consumed.add(w_name)
                continue
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            w = init[w_name] if trans_b == 1 else init[w_name].T
            store(dest, *q2d(w))
            consumed.add(w_name)

        # 3) 嵌入表（不转置）
        for name, arr in init.items():
            if arr.ndim != 2:
                continue
            for pat, fn in _EMBED_ROLES:
                mm = pat.search(name)
                if mm:
                    store(fn(mm.group(1))[:-len(".weight")], *q2d(arr))
                    consumed.add(name)
                    break

        # 4) 命名 2D 权重 initializer（自定义导出保留 HF 名；未被节点消费的）
        w_prefixes = {k[:-len(".weight")] for k in state if k.endswith(".weight")}
        for name, arr in init.items():
            if arr.ndim != 2 or name in consumed:
                continue
            for pat, fn in _NAMED_2D_ROLES:
                mm = pat.search(name)
                if not mm:
                    continue
                dest = fn(*mm.groups())[:-len(".weight")]
                if dest.startswith("classifier."):
                    if not is_reranker or dest in w_prefixes:
                        # 嵌入模型丢弃 pooler/classifier；显式 classifier 优先于 pooler 映射
                        consumed.add(name)
                        break
                w = arr.T if w_transposed(name) else arr
                store(dest, *q2d(w))
                w_prefixes.add(dest)
                consumed.add(name)
                break

        # 5) 1D 偏置 / LayerNorm（含融合 Wqkv bias 拆分）
        for name, arr in init.items():
            if arr.ndim != 1:
                continue
            for pat, fn in _1D_ROLES:
                mm = pat.search(name)
                if not mm:
                    continue
                if isinstance(fn, str):
                    # 特殊标签：qkv_bias（融合偏置拆分）/ cls_pooler_bias / cls_bare_bias
                    if fn == "qkv_bias":
                        h = arr.shape[0] // 3
                        for proj, piece in (("query_proj", arr[0:h]),
                                            ("key_proj", arr[h:2 * h]),
                                            ("value_proj", arr[2 * h:3 * h])):
                            state[f"encoder.layers.{mm.group(1)}.attention.{proj}.bias"] = \
                                mx.array(piece, dtype=mx.float16)
                    elif fn == "cls_pooler_bias":
                        dest = "classifier.dense.bias"
                    else:  # cls_bare_bias
                        dest = "classifier.out_proj.bias"
                    if fn != "qkv_bias":
                        if is_reranker and dest not in state:
                            state[dest] = mx.array(arr, dtype=mx.float16)
                    break
                dest = fn(*mm.groups())
                if dest.startswith("classifier."):
                    if not is_reranker or dest in state:
                        break  # 嵌入模型丢弃；显式 classifier 偏置优先
                if dest not in state:
                    state[dest] = mx.array(arr, dtype=mx.float16)
                break

        if "embeddings.word_embeddings.weight" not in state:
            raise RuntimeError(
                "未从 ONNX 中恢复到嵌入表（word_embeddings）——该 ONNX 可能不是 "
                "BERT/RoBERTa/XLM-R 系 encoder，或导出格式不受支持。")
        return state

    @staticmethod
    def _fill_placeholders(mx, state: Dict, hidden: int):
        """jina 源无 embeddings.LayerNorm，但 MLXEmbedders 总会创建并应用 embeddings.norm；
        缺失权重会让 verify:[.all] 加载器抛 updateMissing。补 ones/zeros 占位
        （LayerNorm 的默认初始化，经第一层 ln1 抵消，不破坏模型）。"""
        if "embeddings.norm.weight" not in state:
            state["embeddings.norm.weight"] = mx.ones((hidden,), dtype=mx.float16)
        if "embeddings.norm.bias" not in state:
            state["embeddings.norm.bias"] = mx.zeros((hidden,), dtype=mx.float16)
