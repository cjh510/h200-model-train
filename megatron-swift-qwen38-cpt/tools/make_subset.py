#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 swift/chinese-c4 的 .jsonl.zst 分片抽取小子集，生成两种格式供测试。

  * text 格式      {"text": "..."}            -> megatron pt  / swift pt
  * messages 格式  {"messages": [...]}        -> megatron sft 回退路径（仅链路验证用）

messages 版本把一段文本按字符对半切成 user/assistant，仅为验证多模态 VL 模型
在 pt 链路不通时的 sft 回退，不用于效果评估。

用法：
    python tools/make_subset.py --input-dir data/chinese-c4 --n 2000 \
        --out-text data/smoke_text.jsonl --out-messages data/smoke_messages.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path


def open_text(path: Path):
    if path.suffix == ".zst" or path.name.endswith(".jsonl.zst"):
        try:
            import zstandard as zstd
        except ImportError:
            sys.exit("缺少 zstandard，请先执行: pip install zstandard")
        fh = path.open("rb")
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_records(paths: list[Path]):
    for path in paths:
        with open_text(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text")
                if isinstance(text, str) and text.strip():
                    yield text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="含 *.jsonl.zst / *.jsonl 的目录")
    ap.add_argument("--shards", type=int, default=2, help="最多读取几个分片（按文件名排序）")
    ap.add_argument("--n", type=int, default=2000, help="抽取样本条数")
    ap.add_argument("--out-text", default="data/smoke_text.jsonl")
    ap.add_argument("--out-messages", default="data/smoke_messages.jsonl")
    ap.add_argument("--min-chars", type=int, default=40, help="messages 格式要求的最小字符数")
    args = ap.parse_args()

    base = Path(args.input_dir)
    candidates = sorted(list(base.rglob("*.jsonl.zst")) + list(base.rglob("*.jsonl")))
    if not candidates:
        print(f"[FATAL] {base} 下未找到 .jsonl.zst / .jsonl。"
              f"请用 RAW_DATA_DIR 指向已下载的数据目录，或先 modelscope download --dataset swift/chinese-c4")
        return 2
    paths = candidates[: args.shards]
    print(f"读取 {len(paths)} 个分片: {', '.join(p.name for p in paths)}")

    out_text = Path(args.out_text)
    out_msg = Path(args.out_messages)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_msg.parent.mkdir(parents=True, exist_ok=True)

    n_text = n_msg = total_chars = 0
    with out_text.open("w", encoding="utf-8") as ft, out_msg.open("w", encoding="utf-8") as fm:
        for text in iter_records(paths):
            if n_text >= args.n:
                break
            ft.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_text += 1
            total_chars += len(text)
            if len(text) >= args.min_chars * 2:
                cut = len(text) // 2
                fm.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": text[:cut]},
                        {"role": "assistant", "content": text[cut:]},
                    ]
                }, ensure_ascii=False) + "\n")
                n_msg += 1

    print(f"[OK] text 格式 {n_text} 条 -> {out_text}")
    print(f"[OK] messages 格式 {n_msg} 条 -> {out_msg}")
    if n_text:
        print(f"平均长度 {total_chars / n_text:.0f} 字符（用于估算 max_length 与 packing 后的 token 数）")
    return 0 if n_text > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
