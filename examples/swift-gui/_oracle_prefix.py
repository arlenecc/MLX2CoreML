"""Test whether modconv's prefix detection + stripping works for this
checkpoint. If _resolve_weight_prefix returns ('', set()), the conversion
silently loads random weights -> garbage. That would be the conversion bug."""
import json, glob, os
SRC = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"

from modconv.real_backend import (_import_heavy, _load_config,
                                  _resolve_weight_layout, _checkpoint_tensor_shapes,
                                  _expected_state_shapes, _TEXT_WEIGHT_PREFIXES)

libs = _import_heavy()
print("config architectures:", json.load(open(f"{SRC}/config.json")).get("architectures"))
config = _load_config(SRC, libs)

print("expected (want) sample:")
want = _expected_state_shapes(libs, config)
for k in list(want)[:5]:
    print("   want:", k, want[k])
print("  total want keys:", len(want))

print("checkpoint (have) sample:")
have = _checkpoint_tensor_shapes(SRC)
for k in list(have)[:5]:
    print("   have:", k, have[k])
print("  total have keys:", len(have))

print("PREFIXES to try:", _TEXT_WEIGHT_PREFIXES)
prefix, transposed = _resolve_weight_layout(want, have)
print(">>> _resolve_weight_layout ->", repr(prefix), "transposed:", len(transposed))
