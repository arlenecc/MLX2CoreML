import os, tempfile, shutil, json, torch
from pathlib import Path
import safetensors.torch as st
from modconv.real_backend import (_import_heavy, _load_config, _resolve_weight_layout,
                                  _checkpoint_tensor_shapes, _expected_state_shapes,
                                  _write_text_checkpoint, _needs_norm_unfolding)

SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"
libs = _import_heavy()
config = _load_config(SRC, libs)
want = _expected_state_shapes(libs, config)
have = _checkpoint_tensor_shapes(SRC)
prefix, transposed = _resolve_weight_layout(want, have)
print("transposed count:", len(transposed))
print("layer0 conv1d in transposed set:",
      any('layers.0.linear_attn.conv1d.weight' in n for n in transposed))

norm = Path(tempfile.mkdtemp(prefix="modconv-norm-"))
_write_text_checkpoint(SRC, norm, prefix, transposed, _needs_norm_unfolding(config, transposed), lambda s: None)

# raw checkpoint conv1d (6144,4,1)
raw = st.load_file(Path(SRC) / "model.safetensors")
rk = raw["language_model.model.layers.0.linear_attn.conv1d.weight"]
print("RAW  layer0 conv1d shape:", tuple(rk.shape))
# norm conv1d (should be transposed to 6144,1,4)
nk = st.load_file(norm / "model.safetensors")["model.layers.0.linear_attn.conv1d.weight"]
print("NORM layer0 conv1d shape:", tuple(nk.shape))
# expected: norm == raw transposed (dims 1,2): raw(c,k,in=1) -> (c,in=1,k) so nk[c,0,k]==raw[c,k,0]
expected = rk.permute(0, 2, 1).contiguous()  # (6144,1,4)
print("norm == raw.permute(0,2,1) ?", torch.equal(nk.float(), expected.float()))
maxerr = (nk.float() - expected.float()).abs().max().item()
print("max abs diff vs expected transpose:", maxerr)
shutil.rmtree(norm, ignore_errors=True)
