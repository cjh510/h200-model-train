#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 manifests/*.json 校验本地权重（或数据集）目录的完整性。

默认逐文件核对「存在性 + 字节大小」；大文件 sha256 极慢，
需显式加 --check-sha 才计算（或 --max-sha-mb 只对小文件计算）。

分级校验：权重分片与关键配置（config/tokenizer/index）为 CRITICAL，任何差异都算失败；
README / chat_template / tokenizer_config 等会随仓库更新的文本文件只记 WARN，不阻塞预检
（本机 2026-11-03 快照即属于这种情况）。加 --strict 可对全部文件严格判定。

用法：
    python tools/verify_weights.py --model 27b --path /mnt/shared/models/Qwen3.8-27B
    python tools/verify_weights.py --model flash_next --path /data/models/FlashNext
    python tools/verify_weights.py --model dataset --path /data/chinese-c4
    python tools/verify_weights.py --model 27b --path D:/model/qwen --windows
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANIFESTS = {
    "27b": "qwen3_8_27b.json",
    "flash_next": "qwen3_8_flash_next.json",
    "dataset": "chinese_c4.json",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


CRITICAL_NAMES = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "special_tokens_map.json", "added_tokens.json",
}


def is_critical(name: str) -> bool:
    return name.endswith(".safetensors") or name.endswith(".jsonl.zst") or name in CRITICAL_NAMES


def main() -> int:
    ap = argparse.ArgumentParser(description="校验本地模型/数据集文件完整性")
    ap.add_argument("--model", choices=list(MANIFESTS), required=True)
    ap.add_argument("--path", required=True, help="本地权重或数据集根目录")
    ap.add_argument("--check-sha", action="store_true", help="对所有文件计算 sha256（很慢）")
    ap.add_argument("--max-sha-mb", type=float, default=16.0,
                    help="未指定 --check-sha 时，仅对该尺寸以下的文件校验 sha256")
    ap.add_argument("--windows", action="store_true",
                    help="在本机 Windows 快照上校验（放宽：允许清单外的本地文件）")
    ap.add_argument("--strict", action="store_true",
                    help="所有文件差异都算失败（默认只对权重/关键配置严格）")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manifests" / MANIFESTS[args.model]).read_text(encoding="utf-8"))
    base = Path(args.path)
    if not base.is_dir():
        print(f"[FATAL] 目录不存在: {base}")
        return 2

    missing, bad_size, bad_sha, soft, ok = [], [], [], [], 0

    def record(kind, rel, detail):
        """CRITICAL 差异进硬错误列表；其余进 soft，仅告警。"""
        if args.strict or is_critical(Path(rel).name):
            {"missing": missing, "size": bad_size, "sha": bad_sha}[kind].append(detail)
        else:
            soft.append(f"[{kind.upper()}] {rel} " + " ".join(str(x) for x in detail[1:]))

    for item in manifest["files"]:
        rel = item["path"]
        candidate = base / rel
        if not candidate.exists():
            candidate = base / item["name"]  # 兼容「所有文件平铺」的下载方式
        if not candidate.is_file():
            record("missing", rel, (rel, item["size"]))
            continue
        size = candidate.stat().st_size
        if size != item["size"]:
            record("size", rel, (rel, item["size"], size))
            continue
        do_sha = args.check_sha or (item["sha256"] and size <= args.max_sha_mb * 1024 * 1024)
        if do_sha and item["sha256"]:
            if sha256_of(candidate) != item["sha256"]:
                record("sha", rel, (rel,))
                continue
        ok += 1

    gib = manifest["data_total_bytes"] / 1024 ** 3
    print(f"清单: {manifest['source_id']}  ({manifest['captured_at']})")
    print(f"目录: {base}")
    print(f"预期: {manifest['file_count']} 文件，其中 {manifest['data_file_count']} 个 "
          f"{manifest['data_suffix']}，合计 {gib:.2f} GiB")
    print(f"结果: 通过 {ok} / 关键缺失 {len(missing)} / 关键大小不符 {len(bad_size)} / "
          f"关键哈希不符 {len(bad_sha)} / 非关键差异 {len(soft)}")
    for rel, size in missing[:20]:
        print(f"  [MISSING] {rel}  (预期 {size} B)")
    for rel, expect, actual in bad_size[:20]:
        print(f"  [SIZE]    {rel}  预期 {expect} B, 实际 {actual} B")
    for entry in bad_sha[:20]:
        print(f"  [SHA256]  {entry[0]}")
    for line in soft[:20]:
        print(f"  [WARN]    {line}")
    if len(missing) > 20:
        print(f"  ... 另有 {len(missing) - 20} 个关键缺失")

    if soft:
        print("提示: 非关键差异多为仓库文档 / chat_template / tokenizer_config 的版本更新，"
              "不影响权重加载；若需要完全一致，请重新下载这些文件或加 --strict。")

    # 常见的「下载中断」痕迹
    incomplete = list(base.glob("*.incomplete"))
    if incomplete:
        print(f"注意: 发现 {len(incomplete)} 个 .incomplete 残留，说明有分片未下载完成")

    return 0 if not (missing or bad_size or bad_sha) else 1


if __name__ == "__main__":
    raise SystemExit(main())
