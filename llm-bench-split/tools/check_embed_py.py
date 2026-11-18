#!/usr/bin/env python3
"""检查 04_bench.sh / lib.sh 里所有内嵌 python heredoc 是否能编译。

bash -n 只查 shell 语法，查不出 heredoc 里的 Python 缩进/转义错误：
11-10 的 highmem_record 就因为 rec 字典多缩进 2 个空格，整段 python
抛 IndentationError，highmem.jsonl 一条都写不出来，而 shell 侧毫无报错。
"""
import ast
import re
import sys
from pathlib import Path

HEREDOC_END = {"PY", "PYCHK"}
PAT = re.compile(r"<<'?(\w+)'?\s*$")


def blocks(path: Path):
    cur = None
    start = 0
    tag = ""
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if cur is None:
            m = PAT.search(ln)
            if m:
                cur, start, tag = [], i, m.group(1)
        elif ln.strip() == tag:
            yield start, "\n".join(cur)
            cur = None
        else:
            cur.append(ln)


def main() -> int:
    total = bad = 0
    from glob import glob
    paths = []
    for x in sys.argv[1:]:
        paths += [Path(g) for g in glob(x)] if any(c in x for c in "*?[") else [Path(x)]
    for p in paths:
        if not p.is_file():
            continue
        for start, body in blocks(p):
            if not body.strip():
                continue
            total += 1
            try:
                ast.parse(body)
            except SyntaxError as e:
                bad += 1
                print(f"  [NG] {p.name} 第 {start} 行起的 python 块：line {e.lineno}: {e.msg}")
                print(f"       {e.text!r}")
    print(f"  checked {total} blocks, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
