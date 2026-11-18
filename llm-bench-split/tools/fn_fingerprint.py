#!/usr/bin/env python3
"""输出 shell 文件里每个函数的指纹（name + 函数体 sha256 前 16 位）。

用途：重构前后各跑一次，diff 必须为空 —— 这是「只搬家、没改行为」的机器证据。

    python3 tools/fn_fingerprint.py ref/04_bench.sh.orig          # 基线（单文件）
    python3 tools/fn_fingerprint.py 04_bench.sh bench/*.sh        # 拆分后（多文件）
"""
import hashlib
import re
import sys
from pathlib import Path


def funcs(path: Path):
    lines = path.read_text(encoding="utf-8").split("\n")
    out, cur, buf, depth = {}, None, [], 0
    for ln in lines:
        m = re.match(r"^([a-zA-Z_][\w-]*)\(\)\s*\{", ln)
        if m and cur is None:
            cur, buf = m.group(1), [ln]
            depth = ln.count("{") - ln.count("}")
            if depth <= 0:                      # 单行函数 f() { ...; }
                out[cur] = "\n".join(buf)
                cur = None
            continue
        if cur is not None:
            buf.append(ln)
            depth += ln.count("{") - ln.count("}")
            if ln.startswith("}") and depth <= 0:
                out[cur] = "\n".join(buf)
                cur, buf = None, []
    return out


def main():
    seen, dup = {}, []
    for arg in sys.argv[1:]:
        for p in sorted(Path().glob(arg)) if any(c in arg for c in "*?[") else [Path(arg)]:
            if not p.is_file():
                continue
            for name, body in funcs(p).items():
                if name in seen:
                    dup.append(name)
                seen[name] = hashlib.sha256(body.encode()).hexdigest()[:16]
    for name in sorted(seen):
        print(f"{name}\t{seen[name]}")
    if dup:
        print(f"# 重复定义：{sorted(set(dup))}", file=sys.stderr)


if __name__ == "__main__":
    main()
