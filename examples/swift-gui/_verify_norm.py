"""验证 checkpoint 中 RMSNorm 权重的存储约定：
- 若均值≈0 → 原始 HF 约定（HF 的 1+w 正确）
- 若均值≈1 → 已消毒（MLX 约定），HF 的 1+w 会变成 2+w（bug）
"""
import mlx.core as mx
import numpy as np

CKPT = "/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16"

w = mx.load(f"{CKPT}/weights.safetensors")
keys = list(w.keys())
print(f"total keys: {len(keys)}")

norm_keys = [k for k in keys if "layernorm" in k or "norm.weight" in k or "q_norm" in k or "k_norm" in k]
print(f"norm keys: {len(norm_keys)}")
for k in sorted(norm_keys)[:8]:
    v = np.array(w[k].astype(mx.float32))
    print(f"  {k}: shape={v.shape} mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")

# conv1d 布局确认
conv_keys = [k for k in keys if "conv1d" in k]
print(f"\nconv1d keys: {len(conv_keys)}")
for k in conv_keys[:2]:
    print(f"  {k}: shape={w[k].shape}")

# 全体 norm 权重的均值汇总
all_means = [float(np.array(w[k].astype(mx.float32)).mean()) for k in norm_keys]
print(f"\nALL norm weight means: min={min(all_means):.4f} max={max(all_means):.4f} avg={np.mean(all_means):.4f}")
