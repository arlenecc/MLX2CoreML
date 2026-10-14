#!/usr/bin/env python3
"""分词桥接：被 Swift GUI 通过子进程调用，仅做 encode / decode。
依赖本地 tokenizer.json（与转换源模型同族）。可用环境变量 MODCONV_TOKENIZER 覆盖路径。

用法:
  tok_helper.py encode "<text>"      -> 打印 JSON 数组 [id,...]
  tok_helper.py decode "[id,id,...]" -> 打印解码文本
"""
import os
import sys
import json
from tokenizers import Tokenizer

TOK_PATH = os.environ.get("MODCONV_TOKENIZER",
                          "/Users/arlenecc/Core-ML/tokenizer.json")
_T = Tokenizer.from_file(TOK_PATH)


def main():
    if len(sys.argv) < 3:
        print("usage: tok_helper.py encode|decode <arg>", file=sys.stderr)
        sys.exit(2)
    mode, arg = sys.argv[1], sys.argv[2]
    if mode == "encode":
        print(json.dumps(_T.encode(arg).ids))
    elif mode == "decode":
        ids = json.loads(arg)
        print(_T.decode(ids, skip_special_tokens=True))
    else:
        print("unknown mode", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
