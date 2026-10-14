#!/usr/bin/env bash
# ModConv Swift GUI 示例构建/运行脚本（macOS，需 Apple Silicon + Core ML）
#
# 做了什么：
#   1) 用系统 swiftc 把 main.swift 编译为 Cocoa 可执行（无 Xcode 也能编，因为只依赖
#      AppKit/CoreML/Foundation/Combine，不用 SwiftUI 的 @State 等宏）。
#   2) 打包成 .app，并把 tok_helper.py 一并拷进 Resources，这样运行时不依赖外部路径。
#   3) 用 open 启动，弹出一个「对话体验」窗口。
#
# 使用：
#   bash build.sh              # 编译并直接打开 .app
#   bash build.sh --no-open    # 只编译，不打开
#
# 自定义模型/分词器（可选，环境变量覆盖）：
#   MODCONV_TOKENIZER=/path/to/tokenizer.json
#   MODCONV_MODEL_4BIT8K=/path/to/Qwen3.5-0.8B-xxx.mlpackage
#   MODCONV_MODEL_6BIT4K=...   MODCONV_MODEL_INT4=...
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="/tmp/ModConvDemo.app"
OUT_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"

rm -rf "$APP"
mkdir -p "$OUT_DIR" "$RES_DIR"

echo "==> 编译 main.swift ..."
swiftc -O "$SRC_DIR/main.swift" \
  -o "$OUT_DIR/gui_app" \
  -framework AppKit -framework CoreML -framework Foundation -framework Combine

cp "$SRC_DIR/Info.plist" "$APP/Contents/Info.plist"
cp "$SRC_DIR/tok_helper.py" "$RES_DIR/tok_helper.py"

echo "==> 已构建: $APP"

if [[ "${1:-}" != "--no-open" ]]; then
  echo "==> 打开应用（首次加载模型约需 1–2 分钟做 ANE 特化）..."
  open "$APP"
fi
