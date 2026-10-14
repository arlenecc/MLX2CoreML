"""Verify the fix: rewrite norm_dir/config.json as a proper Qwen3_5ForCausalLM
config (flatten text_config + pull rope_parameters to top level). If the model
then generates coherent text, the bug was the nested-config RoPE mismatch."""
import os, tempfile, shutil, json, torch
from pathlib import Path
from transformers import AutoTokenizer
from modconv.real_backend import (_import_heavy, _load_config, _resolve_weight_layout,
                                  _checkpoint_tensor_shapes, _expected_state_shapes,
                                  _write_text_checkpoint, _load_torch_model,
                                  _needs_norm_unfolding)

SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"
libs = _import_heavy()
tok = AutoTokenizer.from_pretrained(SRC)
config = _load_config(SRC, libs)
want = _expected_state_shapes(libs, config)
have = _checkpoint_tensor_shapes(SRC)
prefix, transposed = _resolve_weight_layout(want, have)
norm = Path(tempfile.mkdtemp(prefix="modconv-norm-"))
_write_text_checkpoint(SRC, norm, prefix, transposed, _needs_norm_unfolding(config, transposed), lambda s: None)

# ---- build corrected text config ----
raw = json.load(open(Path(SRC) / "config.json"))
tc = raw["text_config"]
rp = tc.get("rope_parameters", {})
newcfg = dict(tc)                      # flatten text_config to top level
for k in ("rope_theta", "partial_rotary_factor", "rope_type"):
    if k in rp:
        newcfg[k] = rp[k]
newcfg.pop("rope_parameters", None)
newcfg["model_type"] = "qwen3_5_text"
newcfg["architectures"] = ["Qwen3_5ForCausalLM"]
for bad in ("vision_config", "text_config", "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id", "tie_word_embeddings_unused"):
    newcfg.pop(bad, None)
json.dump(newcfg, open(norm / "config.json", "w"), indent=2)
print("fixed config rope_theta:", newcfg.get("rope_theta"),
      "partial_rotary_factor:", newcfg.get("partial_rotary_factor"))

config2 = _load_config(norm, libs)
model = _load_torch_model(norm, libs, config2, dtype=torch.float32).eval()
print("loaded model:", type(model).__name__)

from modconv.kv_wrapper import KvCacheWrapper
wrapper = KvCacheWrapper(model, 512).eval()

def gen(p, n=24):
    ids = tok(p, return_tensors="pt").input_ids[0].tolist()
    out = []
    for i, t in enumerate(ids):
        logits = wrapper.step(torch.tensor([[t]]), torch.tensor([i]))
    last = logits
    for _ in range(n):
        nx = int(last[0, 0].argmax())
        if nx in (tok.eos_token_id, 248044, 248045, 248046):
            break
        out.append(nx)
        last = wrapper.step(torch.tensor([[nx]]), torch.tensor([len(ids) + len(out)]))
    print(f"{p!r} -> {tok.decode(out)!r}")

with torch.no_grad():
    for p in ["The capital of France is", "Hello, my name is",
              "def fib(n):", "Qwen is a"]:
        gen(p)
shutil.rmtree(norm, ignore_errors=True)
