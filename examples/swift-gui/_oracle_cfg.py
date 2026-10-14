import os, tempfile, shutil, torch
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
config2 = _load_config(norm, libs)
model = _load_torch_model(norm, libs, config2, dtype=torch.float32).eval()
shutil.rmtree(norm, ignore_errors=True)

print("=== resolved model.config key fields ===")
c = model.config
for k in ["model_type","hidden_size","num_attention_heads","num_key_value_heads",
          "head_dim","intermediate_size","num_hidden_layers","vocab_size",
          "max_position_embeddings","rotary_base"]:
    print(f"  {k}: {getattr(c, k, '<none>')}")
print("  layer_types:", getattr(c, "layer_types", "<none>"))
print("  has text_config:", hasattr(c, "text_config"))

# full-prompt next-token prediction
with torch.no_grad():
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    print("prompt ids:", ids.tolist())
    out = model(input_ids=ids, use_cache=True)
    last = out.logits[0, -1]
    v, i = torch.topk(last, 5)
    print("FULL-PROMPT next-token top5:", list(zip(i.tolist(), [round(float(x),3) for x in v])))
    print("  ->", tok.decode(i.tolist()))

    # autoregressive 3 steps from scratch
    ctx = ids[0, 0:1]
    gen = []
    cp = torch.zeros(1, dtype=torch.long)
    for _ in range(5):
        o = model(input_ids=ctx[:, -1:], cache_position=cp, use_cache=True)
        n = int(o.logits[0, -1].argmax())
        if n in (tok.eos_token_id, 248044, 248045, 248046): break
        gen.append(n); ctx = torch.cat([ctx, torch.tensor([[n]])], 1); cp += 1
    print("AUTOREGRESSIVE (raw, full attention) ->", tok.decode(gen))
