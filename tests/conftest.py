import os

# GUI 测试在无显示环境下运行
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture
def model_dir(tmp_path):
    """构造一个模拟的 MLX 模型目录：配置/tokenizer/说明/嵌套元数据/权重分片。"""
    src = tmp_path / "tiny-llm"
    (src / "sub").mkdir(parents=True)
    (src / "meta").mkdir()
    (src / "config.json").write_bytes(b"{}" * 10)                        # 20 B
    (src / "tokenizer.json").write_bytes(b"tok" * 7)                     # 21 B
    (src / "notes.txt").write_bytes(b"x" * 500)                          # 500 B
    (src / "meta" / "params.json").write_bytes(b"p" * 64)                # 64 B
    (src / "sub" / "model.safetensors").write_bytes(b"\0" * (1 << 20))   # 1 MiB
    return src


@pytest.fixture
def hf_model_dir(tmp_path):
    """真实转换口径的模型目录：config.json 带受支持的 model_type + 权重。"""
    import json

    src = tmp_path / "hf-llm"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (src / "tokenizer.json").write_bytes(b"tok")
    (src / "model.safetensors").write_bytes(b"\0" * (4 << 20))   # 4 MiB
    return src
