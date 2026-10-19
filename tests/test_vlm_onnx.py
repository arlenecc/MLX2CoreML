"""VLM→ONNX 导出管线的单元测试（无需真实大模型）。

本地可用的 VLM 多为 MLX 或自定义架构（jina-omni），无法直接作为 HF→ONNX 源，
故用合成 VLM 模块验证「探测 forward 签名 → 构造哑输入 → 多输入 ONNX 导出 →
onnxruntime 推理」全链路；并单测 Qwen2-VL 的 pixel_values/grid_thw 形状耦合，以及
M-RoPE 的 position_ids 作为真实图输入（非烤死常量）。
"""
import sys
import numpy as np
import torch
import onnxruntime as ort

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modconv.onnx_backend import (  # noqa: E402
    _vlm_export_spec,
    _VlmWrap,
    _vlm_pixel_dummy,
    _vlm_grid_thw,
    HfToOnnxBackend,
)


class _DummyVLM(torch.nn.Module):
    """模拟 LLaVA 风格多模态：forward(input_ids, attention_mask, pixel_values)。"""

    def __init__(self):
        super().__init__()
        self.txt = torch.nn.Embedding(100, 16)
        self.vis = torch.nn.Linear(3 * 8 * 8, 16)
        self.head = torch.nn.Linear(16, 50)

    def forward(self, input_ids, attention_mask, pixel_values):
        B, S = input_ids.shape
        t = self.txt(input_ids)                      # (B, S, 16)
        v = self.vis(pixel_values.flatten(1)).mean(0, keepdim=True).expand(B, S, -1)
        h = (t + v) * attention_mask.unsqueeze(-1).float()
        return self.head(h)                          # (B, S, 50)


class _FakeConfig:
    def __init__(self, model_type="llava", vision_config=None):
        self.model_type = model_type
        self.vision_config = vision_config
        self.architectures = []


class _FakeVisionConfig:
    def __init__(self, image_size=336, patch_size=14, temporal_patch_size=2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size


def test_vlm_export_spec_detects_inputs():
    model = _DummyVLM()
    cfg = _FakeConfig(model_type="llava",
                      vision_config=_FakeVisionConfig(image_size=8))
    arg_names, dummies, fixed_seq = _vlm_export_spec(model, cfg)
    assert fixed_seq is False
    assert arg_names == ["input_ids", "attention_mask", "pixel_values"]
    assert dummies["input_ids"].shape == (1, 8)
    assert dummies["attention_mask"].shape == (1, 8)
    # 标准视觉塔：(N, 3, image_size, image_size)
    assert dummies["pixel_values"].shape == (1, 3, 8, 8)


def test_vlm_export_spec_qwen2vl_shape_coupling():
    # Qwen2-VL：pixel_values 扁平 (N, 3*tps*ps*ps)，N 须等于 grid_thw 的 t*h*w
    cfg = _FakeConfig(model_type="qwen2.5_vl",
                      vision_config=_FakeVisionConfig(image_size=None))
    pix = _vlm_pixel_dummy(cfg, "qwen2.5_vl")
    grid = _vlm_grid_thw(cfg, "qwen2.5_vl")
    assert grid.tolist() == [[1, 4, 4]]
    # grid 4x4(merge 单位) => 1*4*4 = 16 个 patch
    assert pix.shape == (16, 3 * 2 * 14 * 14)


def test_vlm_is_detected_by_architectures():
    cfg = _FakeConfig()
    cfg.architectures = ["LlavaForConditionalGeneration"]
    assert HfToOnnxBackend._is_vlm(cfg, "LlavaForConditionalGeneration", "llava") is True


def test_vlm_export_and_onnxruntime_roundtrip(tmp_path):
    model = _DummyVLM()
    cfg = _FakeConfig(model_type="llava",
                      vision_config=_FakeVisionConfig(image_size=8))
    arg_names, dummies, fixed_seq = _vlm_export_spec(model, cfg)
    wrapper = _VlmWrap(model, arg_names)

    onnx_path = tmp_path / "vlm.onnx"
    torch.onnx.export(
        wrapper, tuple(dummies[a] for a in arg_names), str(onnx_path),
        input_names=list(arg_names), output_names=["logits"],
        opset_version=17, do_constant_folding=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {n: dummies[n].numpy() for n in arg_names}
    out = sess.run(["logits"], feeds)[0]
    assert out.shape == (1, 8, 50)   # (batch, seq, vocab)


def test_vlm_export_spec_qwen3_5_uses_fixed_seq():
    # qwen3_5（混合线性注意力）必须定长导出，避免 chunk 顺序扫描被动态 num_chunks 卡死。
    class _Qwen35Vision:
        image_size = 224
        patch_size = 16
        temporal_patch_size = 2
        spatial_merge_size = 2
    cfg = _FakeConfig(model_type="qwen3_5", vision_config=_Qwen35Vision())
    cfg.image_token_id = 151655
    model = _DummyVLM()
    arg_names, dummies, fixed_seq = _vlm_export_spec(model, cfg)
    assert fixed_seq is True
    # 固定序列长度 = 1024（chunk_size 64 的倍数），且含正确数量的图像占位 token
    assert dummies["input_ids"].shape == (1, 1024)
    img_count = int((dummies["input_ids"] == 151655).sum())
    assert img_count == 4  # grid 1x4x4 经 spatial_merge(2) => 4 视觉 token
    # M-RoPE：position_ids 作为显式输入（不再烤死、不再依赖 mm_token_type_ids 图输入）
    assert "position_ids" in arg_names
    assert dummies["position_ids"].shape == (3, 1, 1024)
    # 合成模型无 get_rope_index，mm_token_type_ids 不应作为图输入
    assert "mm_token_type_ids" not in arg_names


def test_vlm_export_mrope_position_ids_is_live_input(tmp_path):
    # M-RoPE：position_ids 必须是真实图输入（运行期由消费端计算），而非烤死的常量。
    # 若被烤成常量，则更换 position_ids 输入不会改变输出；本测试断言输出随之改变。
    class _Q35Vision:
        image_size = 224
        patch_size = 16
        temporal_patch_size = 2
        spatial_merge_size = 2

    class _MropeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.txt = torch.nn.Embedding(200000, 16)
            self.head = torch.nn.Linear(16, 50)

        def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, position_ids):
            B, S = input_ids.shape
            t = self.txt(input_ids)
            pv = pixel_values.to(t.dtype).flatten().mean().expand(B, S, t.shape[-1])
            h = (t + pv) * attention_mask.unsqueeze(-1).float()
            h = h + image_grid_thw.to(t.dtype).flatten().mean().expand_as(h)  # 消费 image_grid_thw
            h = h + position_ids.float().mean(0).unsqueeze(-1)  # 真实使用 position_ids
            return self.head(h)

    cfg = _FakeConfig(model_type="qwen3_5", vision_config=_Q35Vision())
    cfg.image_token_id = 151655
    model = _MropeVLM()
    arg_names, dummies, fixed_seq = _vlm_export_spec(model, cfg)
    assert "position_ids" in arg_names
    assert "mm_token_type_ids" not in arg_names
    wrapper = _VlmWrap(model, arg_names)
    onnx_path = tmp_path / "mrope.onnx"
    torch.onnx.export(
        wrapper, tuple(dummies[a] for a in arg_names), str(onnx_path),
        input_names=list(arg_names), output_names=["logits"],
        opset_version=17, do_constant_folding=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {n: dummies[n].numpy() for n in arg_names}
    out_a = sess.run(["logits"], feeds)[0]
    # 更换 position_ids 输入，输出必须不同 → 证明是真实输入而非烤死常量
    feeds2 = dict(feeds)
    feeds2["position_ids"] = (feeds["position_ids"] + 7).astype("int64")
    out_b = sess.run(["logits"], feeds2)[0]
    assert not np.allclose(out_a, out_b), "position_ids 被烤成常量，未作为真实输入生效"
