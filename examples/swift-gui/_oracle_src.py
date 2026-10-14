"""Disambiguate: is the source checkpoint good (load it the canonical way,
which knows the language_model. prefix) or is it itself corrupt?"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"
tok = AutoTokenizer.from_pretrained(SRC)

print("loading Qwen3_5ForConditionalGeneration (canonical, knows language_model. prefix)...")
m = Qwen3_5ForConditionalGeneration.from_pretrained(
    SRC, torch_dtype=torch.float32, attn_implementation="eager")
print("loaded:", type(m).__name__)

with torch.no_grad():
    for p in ["The capital of France is", "Hello, my name is"]:
        ids = tok(p, return_tensors="pt").input_ids
        out = m.generate(ids, max_new_tokens=20, do_sample=False)
        print(f"{p!r} -> {tok.decode(out[0])!r}")
