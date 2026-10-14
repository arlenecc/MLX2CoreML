"""End-to-end: strip prefix (as conversion does) -> write norm_dir -> load via
modconv pipeline -> run KvCacheWrapper. Reveals whether current code produces a
working model, and if not, WHERE it breaks (config? loading?)."""
import os, tempfile, shutil, time
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

config = _load_config(SRC, libs)
want = _expected_state_shapes(libs, config)
have = _checkpoint_tensor_shapes(SRC)
prefix, transposed = _resolve_weight_layout(want, have)
unfold = _needs_norm_unfolding(config, transposed)
print("detected prefix:", repr(prefix), "transposed:", len(transposed),
      "unfold_norms:", unfold)

norm = Path(tempfile.mkdtemp(prefix="modconv-norm-"))
print("writing norm_dir:", norm)
_write_text_checkpoint(SRC, norm, prefix, transposed, unfold, lambda s: print("  ", s))
print("norm_dir contents:", sorted(os.listdir(norm)))

config2 = _load_config(norm, libs)
print("norm_dir config.architectures:", getattr(config2, "architectures", None))

print("loading norm_dir via modconv pipeline...")
t0 = time.time()
model = _load_torch_model(norm, libs, config2, dtype=torch.float32)
print(f"  loaded in {time.time()-t0:.1f}s")

print("=== model introspection ===")
print("type(model):", type(model).__name__)
print("config.architectures:", getattr(config2, "architectures", None))
print("hasattr(model,'model'):", hasattr(model, "model"),
      "| type(model.model):", type(getattr(model, "model", None)).__name__ if hasattr(model, "model") else None)
print("hasattr(model,'language_model'):", hasattr(model, "language_model"))
emb = model.get_input_embeddings()
print("input_embeddings type:", type(emb).__name__, "weight:", tuple(emb.weight.shape))
lm = model.get_output_embeddings()
print("output_embeddings:", "None" if lm is None else (type(lm).__name__, tuple(lm.weight.shape)))


def generate(prompt, n=30):
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
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
    print(f"  {prompt!r} -> {tok.decode(out)!r}")


with torch.no_grad():
    # CONTROL: raw HF model BEFORE patching (ground truth)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    raw = model.generate(ids, max_new_tokens=20, do_sample=False)
    print("\n=== CONTROL raw HF model.generate (unpatched) ===")
    print("  ->", tok.decode(raw[0]))

    wrapper = _build_kv_wrapper(model, 256).eval()
    print("hybrid:", wrapper._hybrid, "gdn_slots:", wrapper._gdn_slots)

    generate("The capital of France is")
    generate("Hello, my name is")
    generate("def add(a, b):")

shutil.rmtree(norm, ignore_errors=True)
