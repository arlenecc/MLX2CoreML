#!/usr/bin/env python3
"""分词桥接：被 Swift GUI 调用，仅做 encode / decode。
依赖本地 tokenizer.json（与转换源模型同族）。可用环境变量 MODCONV_TOKENIZER 覆盖路径。

两种模式：
  - 一次性（便于测试）：
      tok_helper.py encode "<text>"      -> 打印 JSON 数组 [id,...]
      tok_helper.py decode "[id,id,...]" -> 打印解码文本
  - 常驻（Swift GUI 默认使用，避免每条消息冷启动 Python + 重新加载分词器）：
      tok_helper.py --server
    从 stdin 逐行读 JSON 请求 {"mode":"encode"|"decode","arg":...}，
    输出一行 JSON 响应 {"ok":true,"data":...} 或 {"ok":false,"error":...}。
"""
import os
import sys
import json
from tokenizers import Tokenizer

TOK_PATH = os.environ.get("MODCONV_TOKENIZER",
                          "/Users/arlenecc/Core-ML/tokenizer.json")
_T = Tokenizer.from_file(TOK_PATH)


def _encode(text):
    return _T.encode(text).ids


def _decode(ids):
    return _T.decode(ids, skip_special_tokens=True)


def _serve():
    # 常驻模式：每行一个 JSON 请求，输出一行 JSON 响应。
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            mode = req.get("mode")
            if mode == "encode":
                data = _encode(req["arg"])
            elif mode == "decode":
                data = _decode(req["arg"])
            else:
                ret = {"ok": False, "error": "unknown mode: %r" % (mode,)}
                data = None
            if data is not None:
                ret = {"ok": True, "data": data}
        except Exception as e:  # noqa: BLE001
            ret = {"ok": False, "error": str(e)}
        sys.stdout.write(json.dumps(ret, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    args = sys.argv[1:]
    if args and args[0] == "--server":
        _serve()
        return
    # 一次性模式（保留，便于独立测试）
    if len(args) < 2:
        print("usage: tok_helper.py [--server] encode|decode <arg>", file=sys.stderr)
        sys.exit(2)
    mode, arg = args[0], args[1]
    if mode == "encode":
        print(json.dumps(_encode(arg)))
    elif mode == "decode":
        print(_decode(json.loads(arg)))
    else:
        print("unknown mode", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
