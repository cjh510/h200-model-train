#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""容器内镜像能力探针：判定当前环境能否训练 Qwen3.8-27B / Qwen3.8-Flash-Next。

不依赖 swift 的内部 API（不同版本差异大），直接做两件事：
  1) 打印关键包版本矩阵；
  2) 扫描已安装 swift / transformers 包源码是否注册了 qwen3_8 / qwen4_exp 架构。

判定依据（2026-11-13 核实）：ms-swift v4.4.1 / v4.4.3 的源码树与支持表中
qwen3_8、qwen4_exp、Qwen3.8 出现次数均为 0；v4.5.3 起才出现。

用法：
    python tools/probe_image.py                        # 报告 + 写 logs/image_probe.json
    python tools/probe_image.py --allow-fail           # 不支持也返回 0（用于镜像 build 自检）
    python tools/probe_image.py --no-json              # 不落盘
退出码：0 = 支持，1 = 不支持，2 = 运行错误。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MIN_SWIFT = "4.5.3"
SWIFT_NEEDLES = ("qwen4_exp", "qwen3_8")
TRANSFORMERS_NEEDLES = (
    "Qwen4ExpForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
PACKAGES = (
    "torch", "transformers", "modelscope", "swift", "megatron.core",
    "mcore_bridge", "transformer_engine", "flash_attn", "fla",
    "causal_conv1d", "decord", "qwen_vl_utils", "zstandard", "vllm", "peft",
)


def numeric(v: str) -> tuple[int, int, int]:
    parts = []
    for seg in str(v).split(".")[:3]:
        digits = "".join(c for c in seg if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))  # type: ignore[return-value]


def ge(v: str | None, minimum: str) -> bool:
    return v is not None and numeric(v) >= numeric(minimum)


def version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "?"))
    except Exception:
        return None


def package_dir(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None:
        return None
    locs = spec.submodule_search_locations
    if locs:
        return Path(list(locs)[0])
    if spec.origin and spec.origin.endswith("__init__.py"):
        return Path(spec.origin).parent
    return None


def scan(root: Path | None, needles: tuple[str, ...], limit: int = 4500) -> dict:
    result = {n: 0 for n in needles}
    result["_files"] = 0
    if root is None or not root.is_dir():
        return result
    for path in root.rglob("*.py"):
        if result["_files"] >= limit:
            break
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        result["_files"] += 1
        for needle in needles:
            if needle in text:
                result[needle] += 1
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-fail", action="store_true", help="不支持时也返回 0")
    ap.add_argument("--no-json", action="store_true", help="不写 JSON 报告")
    ap.add_argument("--no-pip-check", action="store_true", help="跳过 pip check")
    ap.add_argument("--out", default="", help="JSON 报告路径")
    args = ap.parse_args()

    versions = {name: version(name) for name in PACKAGES}
    swift_dir = package_dir("swift")
    tf_dir = package_dir("transformers")
    swift_scan = scan(swift_dir, SWIFT_NEEDLES)
    tf_scan = scan(tf_dir, TRANSFORMERS_NEEDLES)

    gpu_info = {"cuda_available": None, "device_count": None}
    try:
        import torch
        gpu_info["cuda_available"] = bool(torch.cuda.is_available())
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass

    swift_ver = versions.get("swift")
    tf_ver = versions.get("transformers")
    reasons, warnings = [], []

    if not ge(swift_ver, MIN_SWIFT):
        reasons.append(f"ms-swift={swift_ver or 'MISSING'} < {MIN_SWIFT}（4.4.1/4.4.3 不含 Qwen3.8 注册）")
    if swift_scan["qwen4_exp"] == 0:
        reasons.append("swift 包内未找到 qwen4_exp（Qwen3.8-Flash-Next 架构）")
    if swift_scan["qwen3_8"] == 0:
        reasons.append("swift 包内未找到 qwen3_8（Qwen3.8 模板/注册）")
    if tf_scan["Qwen3_5ForConditionalGeneration"] == 0:
        reasons.append("transformers 包内未找到 Qwen3_5ForConditionalGeneration")
    if tf_scan["Qwen4ExpForConditionalGeneration"] == 0:
        warnings.append("transformers 包内未找到 Qwen4ExpForConditionalGeneration；Flash-Next 可能不可用")
    if not ge(tf_ver, "5.16"):
        warnings.append(f"transformers={tf_ver or 'MISSING'} < 5.16，Flash-Next 需 >=5.16")
    if versions.get("vllm"):
        warnings.append(f"镜像内 vllm={versions['vllm']}：本项目只用 Megatron 训练 + transformers 推理，不要用 vLLM 后端")
    if versions.get("fla") is None:
        warnings.append("flash-linear-attention(fla) 未安装，GDN 混合线性注意力不可用")
    if versions.get("causal_conv1d") is None:
        warnings.append("causal_conv1d 未安装，GDN 卷积算子不可用")

    pip_check = ""
    if not args.no_pip_check:
        try:
            out = subprocess.run([sys.executable, "-m", "pip", "check"],
                                 capture_output=True, text=True, timeout=30)
            pip_check = (out.stdout + out.stderr).strip()[:1200]
        except Exception as exc:  # noqa: BLE001
            pip_check = f"pip check 跳过: {exc}"

    supported = not reasons
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_hint": os.environ.get("IMAGE_NAME", ""),
        "python": sys.version.split()[0],
        "min_swift_for_qwen38": MIN_SWIFT,
        "supported": supported,
        "reasons": reasons,
        "warnings": warnings,
        "versions": versions,
        "gpu": gpu_info,
        "swift_package_dir": str(swift_dir) if swift_dir else None,
        "swift_scan": swift_scan,
        "transformers_scan": tf_scan,
        "pip_check": pip_check,
    }

    print("=" * 72)
    print("镜像能力探针 | Qwen3.8-27B / Qwen3.8-Flash-Next")
    print("=" * 72)
    print(f"image_hint : {report['image_hint'] or '(未设置 IMAGE_NAME)'}")
    print(f"python     : {report['python']}  cuda={gpu_info.get('torch_cuda')} "
          f"visible_gpus={gpu_info.get('device_count')}")
    print(f"{'package':<22} {'version':<20}")
    print("-" * 72)
    for name, ver in versions.items():
        print(f"{name:<22} {ver or 'MISSING':<20}")
    print("-" * 72)
    print(f"swift 源码扫描       ({swift_scan['_files']} 个 .py): "
          f"qwen4_exp={swift_scan['qwen4_exp']}, qwen3_8={swift_scan['qwen3_8']}")
    print(f"transformers 源码扫描 ({tf_scan['_files']} 个 .py): "
          f"Qwen4Exp={tf_scan['Qwen4ExpForConditionalGeneration']}, "
          f"Qwen3_5={tf_scan['Qwen3_5ForConditionalGeneration']}")
    if pip_check:
        print(f"pip check: {pip_check.splitlines()[0][:160]}")
    print("-" * 72)
    print(f"判定: {'SUPPORTED（可以训练两个模型）' if supported else 'UNSUPPORTED（不能原样训练）'}")
    for r in reasons:
        print(f"  [FAIL] {r}")
    for w in warnings:
        print(f"  [WARN] {w}")
    if not supported:
        print("\n修复：换官方 ms-swift >= v4.5.3 镜像（config/cluster.env 的 RUN_IMAGE），")
        print("      该镜像含 qwen3_8 / qwen4_exp 注册代码与配套 FLA，见 README 镜像章节。")
    print("=" * 72)

    if not args.no_json:
        out = Path(args.out) if args.out else Path("logs/image_probe.json")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"报告: {out}")
        except OSError as exc:
            print(f"[WARN] 报告写入失败: {exc}")

    if supported:
        return 0
    return 0 if args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
