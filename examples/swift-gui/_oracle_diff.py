import os, tempfile, shutil, time, torch
from pathlib import Path
from transformers import AutoTokenizer
from modconv.real_backend import (_import_heavy, _load_config, _resolve_weight_layout,
                                  _checkpoint_tensor_shapes, _expected_state_shapes,
                                  _write_text_checkpoint, _load_torch_model, _build_kv_wrapper,
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

# Print key config hyperparams actually used
print("=== config hyperparams (CausalLM built with) ===")
for k in ["hidden_size","num_attention_heads","num_key_value_heads","head_dim",
          "intermediate_size","num_hidden_layers","vocab_size","max_position_embeddings",
          "layer_types","out_hidden_size"]:
    print(f"  {k}: {getattr(config2, k, '<none>')}")

def topk(t, k=5):
    v, i = torch.topk(t, k)
    return list(zip(i.tolist(), [round(float(x),3) for x in v]))

# RAW single-token logits (unpatched)
with torch.no_grad():
    tok_id = tok("The", add_special_tokens=False).input_ids[0]
    print("tok_id for 'The':", tok_id)
    out_raw = model(input_ids=torch.tensor([[tok_id]]), cache_position=torch.tensor([0]),
                    use_cache=True)
    logits_raw = out_raw.logits[0, 0]
    print("RAW  top5:", topk(logits_raw))
    arg_raw = int(logits_raw.argmax()); print("RAW  argmax:", arg_raw, repr(tok.decode([arg_raw])))

    # now patch and use wrapper
    wrapper = _build_kv_wrapper(model, 256).eval()
    lw = wrapper.step(torch.tensor([[tok_id]]), torch.tensor([0]))[0, 0]
    print("WRAP top5:", topk(lw))
    arg_wrap = int(lw.argmax()); print("WRAP argmax:", arg_wrap, repr(tok.decode([arg_wrap])))
    print("RAW vs WRAP argmax match:", arg_raw == arg_wrap)
