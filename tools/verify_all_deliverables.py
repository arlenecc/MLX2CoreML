#!/usr/bin/env python3
"""
verify_all_deliverables.py
==========================

一键验证所有「转换交付物」是否与源模型一致（测试 / 验证工具）。

对 tools/ 下三个转换工具产出的交付物做端到端校验：
- bge-reranker-base-ONNX  -> bge-reranker-base-mlx-int4（ONNX 工具：对比 onnxruntime 原 logits）
- bge-reranker-v2-m3-mlx-fp16 -> -fp16-fixed / -int8-fixed / -int4（重命名工具：键集合对比交付物）
- jina-embeddings-v5-text-small-mlx -> int4（jina 工具：与 fp16 基线比余弦）

每个校验都重新从源模型转换到临时目录，避免依赖已落盘的交付物本身。

前置：源模型在 BASE 目录下（见 SRC 配置），依赖已安装（mlx/onnx/onnxruntime/torch/transformers）。

用法：
    python3 verify_all_deliverables.py [--base BASE_DIR] [--tmp TMP_DIR]
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# 每个交付物的校验任务：(工具脚本, 源目录后缀或绝对片段, 额外参数, 交付物参考目录或 None)
JOBS = [
    dict(
        tool="convert_onnx_reranker_to_mlx.py",
        src="bge-reranker-base-ONNX",
        extra=["--no-verify"],  # 先转换；随后单独跑 verify-only
        verify_only=True,
        ref=None,
        name="bge-reranker-base-mlx-int4",
    ),
    dict(
        tool="convert_mlx_rekey_quantize.py",
        src="bge-reranker-v2-m3-mlx-fp16",
        extra=["--no-quantize"],
        ref="bge-reranker-v2-m3-mlx-fp16-fixed",
        name="bge-reranker-v2-m3-mlx-fp16-fixed",
    ),
    dict(
        tool="convert_mlx_rekey_quantize.py",
        src="bge-reranker-v2-m3-mlx-fp16",
        extra=["--bits", "8"],
        ref="bge-reranker-v2-m3-mlx-int8-fixed",
        name="bge-reranker-v2-m3-mlx-int8-fixed",
    ),
    dict(
        tool="convert_mlx_rekey_quantize.py",
        src="bge-reranker-v2-m3-mlx-fp16",
        extra=["--bits", "4"],
        ref="bge-reranker-v2-m3-mlx-int4",
        name="bge-reranker-v2-m3-mlx-int4",
    ),
    dict(
        tool="convert_jina_mlx_to_int4.py",
        src="jina-embeddings-v5-text-small-mlx",
        extra=["--no-verify"],
        ref=None,
        name="jina-embeddings-v5-text-small-mlx-int4",
    ),
]


def run(cmd):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=False)
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="/Volumes/SSD/embedding&reranker",
                    help="源模型根目录（含各 SRC 子目录）")
    ap.add_argument("--tmp", default=None, help="临时输出目录（默认系统临时目录）")
    args = ap.parse_args()

    tmp_root = args.tmp or tempfile.mkdtemp(prefix="verify_models_")
    os.makedirs(tmp_root, exist_ok=True)
    base = args.base

    all_ok = True
    for job in JOBS:
        src = os.path.join(base, job["src"])
        if not os.path.isdir(src):
            print(f"[skip] 源目录不存在：{src}")
            continue
        dst = os.path.join(tmp_root, job["name"])
        tool = os.path.join(HERE, job["tool"])

        print(f"\n========== 校验 {job['name']} ==========")
        # 1) 转换
        conv = [sys.executable, tool, "--src", src, "--dst", dst] + job["extra"]
        ok = run(conv)
        # 2) 验证
        if job.get("verify_only"):
            ok = run([sys.executable, tool, "--src", src, "--dst", dst, "--verify-only"]) and ok
        elif job["ref"]:
            ref = os.path.join(base, job["ref"])
            ok = run([sys.executable, tool, "--src", src, "--dst", dst,
                      "--ref", ref]) and ok
        else:
            ok = run([sys.executable, tool, "--src", src, "--dst", dst, "--verify-only"]) and ok

        status = "OK" if ok else "FAIL"
        print(f"---------- {job['name']}: {status} ----------")
        all_ok = all_ok and ok

    print("\n==== 总结 ====")
    print("全部通过 ✅" if all_ok else "存在失败 ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
