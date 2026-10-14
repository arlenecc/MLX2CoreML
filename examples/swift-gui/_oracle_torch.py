"""Decisive test: run modconv's KvCacheWrapper in PURE torch using modconv's
OWN model-loading pipeline (so weight loading matches the real conversion).
If THIS is garbage, the bug is in the wrapper/graph logic (not coremltools)."""
import sys
import time
import torch
from transformers import AutoTokenizer

SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"
MAX_SEQ = 256

from modconv.real_backend import (_import_heavy, _load_config, _load_torch_model,
                                  _build_kv_wrapper)

print("importing heavy deps...")
libs = _import_heavy()
tok = AutoTokenizer.from_pretrained(SRC)
print("loading model via modconv pipeline (fp32)...")
t0 = time.time()
config = _load_config(SRC, libs)
model = _load_torch_model(SRC, libs, config, dtype=torch.float32)
print(f"  loaded in {time.time()-t0:.1f}s")

print("building KvCacheWrapper...")
wrapper = _build_kv_wrapper(model, MAX_SEQ).eval()
print(f"  hybrid={wrapper._hybrid} gdn_slots={wrapper._gdn_slots}")
print(f"  conv_shape={wrapper._conv_shape} rec_shape={wrapper._rec_shape}")


def generate(prompt: str, n: int = 30):
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt: {prompt!r}  ids={ids}")
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
    print("  ->", tok.decode(out))


with torch.no_grad():
    generate("The capital of France is")
    generate("Hello, my name is")
    generate("def add(a, b):")
