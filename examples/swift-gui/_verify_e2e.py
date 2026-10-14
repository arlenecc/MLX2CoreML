"""端到端数值验证：modconv 归一化路径加载的模型，wrapper 单步 logits 与
原始 HF 前向对齐。乱码修复的最终判据。"""
import tempfile, shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer
from modconv.real_backend import (_import_heavy, _load_config, _resolve_weight_layout,
                                  _checkpoint_tensor_shapes, _expected_state_shapes,
                                  _write_text_checkpoint, _load_torch_model,
                                  _needs_norm_unfolding, _build_kv_wrapper)

SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"
libs = _import_heavy()
tok = AutoTokenizer.from_pretrained(SRC)
PROMPT = "The capital of France is"

config = _load_config(SRC, libs)
want = _expected_state_shapes(libs, config)
have = _checkpoint_tensor_shapes(SRC)
prefix, transposed = _resolve_weight_layout(want, have)
unfold = _needs_norm_unfolding(config, transposed)

norm = Path(tempfile.mkdtemp(prefix="modconv-norm-"))
_write_text_checkpoint(SRC, norm, prefix, transposed, unfold, lambda s: None)
config2 = _load_config(norm, libs)
model = _load_torch_model(norm, libs, config2, dtype=torch.float32)

with torch.no_grad():
    ids = tok(PROMPT, return_tensors="pt").input_ids
    # 1) 原始 HF 前向（prefill 全序列）
    hf_logits = model(ids).logits[0, -1].float()
    # 2) modconv wrapper 单步循环（转换路径实际使用的前向）
    wrapper = _build_kv_wrapper(model, 256).eval()
    logits = None
    for i, t in enumerate(ids[0].tolist()):
        logits = wrapper.step(torch.tensor([[t]]), torch.tensor([i]))
    w_logits = logits[0, 0].float()

cos = torch.nn.functional.cosine_similarity(
    hf_logits.unsqueeze(0), w_logits.unsqueeze(0)).item()
print(f"prompt: {PROMPT!r}  ({ids.shape[1]} tokens)")
hf_top = hf_logits.topk(5)
w_top = w_logits.topk(5)
print(f"HF   top5: {[(tok.decode([i]), round(float(v), 3)) for v, i in zip(hf_top.values, hf_top.indices)]}")
print(f"wrap top5: {[(tok.decode([i]), round(float(v), 3)) for v, i in zip(w_top.values, w_top.indices)]}")
print(f"cos-sim(wrapper, HF) = {cos:.6f}   max|diff| = {(hf_logits - w_logits).abs().max():.2e}")
assert cos > 0.99, "wrapper 与 HF 不一致"
assert int(hf_logits.argmax()) == int(w_logits.argmax()), "top-1 不一致"
print("PASS: wrapper 与 HF 完全对齐")

shutil.rmtree(norm, ignore_errors=True)
