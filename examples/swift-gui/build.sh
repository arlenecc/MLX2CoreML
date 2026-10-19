#!/usr/bin/env bash
# swift-gui 构建/运行脚本（SwiftPM 包，需 macOS 15+ / Apple Silicon）
#
# 支持三后端：CoreML / MLX（mlx-swift-lm）/ ONNX（onnxruntime-swift）；
# 分词统一用 Swift Tokenizers（swift-transformers），无需 Python 桥接。
# 依赖首次构建时由 SPM 自动拉取（见 Package.swift）。
#
# 用法：
#   bash build.sh            # swift build -c release 并运行（swift run）
#   bash build.sh --no-run   # 只构建
#   bash build.sh --app      # 额外打包成 /tmp/ModConvDemo.app 并打开
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SRC_DIR"

echo "==> swift build -c release ..."
swift build -c release
BIN=".build/release/swift-gui"

if [[ "${1:-}" == "--app" ]]; then
  APP="/tmp/ModConvDemo.app"
  rm -rf "$APP"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks"
  cp "$BIN" "$APP/Contents/MacOS/gui_app"
  cp "$SRC_DIR/Info.plist" "$APP/Contents/Info.plist"
  # SPM 动态框架（mlx/onnxruntime 等）一并拷入并加 rpath
  if [[ -d ".build/release/PackageFrameworks" ]]; then
    cp -R ".build/release/PackageFrameworks/" "$APP/Contents/Frameworks/"
  fi
  # SPM 资源包（MLX 的 Cmlx Metal 着色器、swift-transformers、swift-crypto）；
  # Bundle.module 会在 Contents/Resources 查找，缺少则 MLX 后端无法初始化。
  mkdir -p "$APP/Contents/Resources"
  cp -R .build/release/*.bundle "$APP/Contents/Resources/" 2>/dev/null || true
  install_name_tool -add_rpath "@executable_path/../Frameworks" \
    "$APP/Contents/MacOS/gui_app" 2>/dev/null || true
  echo "==> 已打包: $APP"
  open "$APP"
elif [[ "${1:-}" != "--no-run" ]]; then
  echo "==> 运行 swift-gui（首次加载模型约需 1–2 分钟做 ANE 特化）..."
  swift run swift-gui
fi
