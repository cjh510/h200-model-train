# -*- coding: utf-8 -*-
"""解析单机 8 卡基准测试结果，输出结构化 metrics.json（仅用标准库）。"""
import json
import os
import re
import statistics as st
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "2026-11-10_1827_worker-gpu-a_w8_full"
OUT = Path(__file__).resolve().parent / "metrics.json"

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)\s*\|")
FIELDS = {
    "step_ms": r"elapsed time per iteration \(ms\):\s*([\d.]+)",
    "tflops": r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)",
    "loss": r"lm loss:\s*([\d.Ee+-]+)",
    "grad": r"grad norm:\s*([\d.Ee+-]+)",
    "lr": r"learning rate:\s*([\d.Ee+-]+)",
    "gbs": r"global batch size:\s*(\d+)",
    "skipped": r"number of skipped iterations:\s*(\d+)",
    "nan": r"number of nan iterations:\s*(\d+)",
}
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def kv(path):
    d = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def parse_iterations(path):
    """返回 iteration 级记录列表。"""
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ITER_RE.search(line)
            if not m:
                continue
            rec = {"iter": int(m.group(1)), "total": int(m.group(2))}
            tsm = TS_RE.search(line)
            if tsm:
                rec["ts"] = tsm.group(1)
            for k, pat in FIELDS.items():
                fm = re.search(pat, line)
                if fm:
                    try:
                        rec[k] = float(fm.group(1))
                    except ValueError:
                        rec[k] = fm.group(1)
            rows.append(rec)
    return rows


def parse_memory(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    reserved = [float(x) for x in re.findall(r"max reserved:\s*([\d.]+)", txt)]
    allocated = [float(x) for x in re.findall(r"max allocated:\s*([\d.]+)", txt)]
    theo = re.findall(r"weight and optimizer=\s*([\d.]+)\s*MB", txt)
    return {
        "max_reserved_mb": max(reserved) if reserved else None,
        "max_allocated_mb": max(allocated) if allocated else None,
        "theoretical_mb": float(theo[0]) if theo else None,
    }


def describe(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    s = sorted(vals)
    def pct(p):
        if len(s) == 1:
            return s[0]
        idx = p / 100 * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    mean = st.fmean(vals)
    return {
        "n": len(vals), "mean": round(mean, 3), "median": round(st.median(vals), 3),
        "std": round(st.pstdev(vals), 3), "cv_pct": round(st.pstdev(vals) / mean * 100, 2) if mean else 0,
        "min": round(min(vals), 3), "max": round(max(vals), 3),
        "p50": round(pct(50), 3), "p95": round(pct(95), 3), "p99": round(pct(99), 3),
    }


def steady(rows, key, drop_head=1, drop_tail=1):
    core = rows[drop_head:len(rows) - drop_tail] if len(rows) > drop_head + drop_tail else rows
    return describe([r.get(key) for r in core])


def parse_tag(tag):
    d = {}
    for key, pat in [("tp", r"_tp(\d+)"), ("pp", r"_pp(\d+)"), ("mbs", r"_mbs(\d+)"),
                     ("seq", r"_s(\d+)"), ("gb", r"_gb(\d+)")]:
        m = re.search(pat, tag)
        if m:
            d[key] = int(m.group(1))
    d["precision"] = "fp8" if tag.endswith("fp8") else "bf16"
    return d


def parse_dmon(path):
    """nvidia-smi dmon: 返回每张卡 pwr/sm/mem/gtemp 的均值与峰值。"""
    per_gpu, samples = {}, 0
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 6 or not p[0].lstrip("-").isdigit():
                continue
            try:
                idx, pwr, gtemp, mtemp, sm, mem = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
            except (ValueError, IndexError):
                continue
            g = per_gpu.setdefault(idx, {"pwr": [], "sm": [], "mem": [], "gtemp": []})
            g["pwr"].append(pwr); g["sm"].append(sm); g["mem"].append(mem); g["gtemp"].append(gtemp)
            samples += 1
    res = {"samples": samples, "gpus": {}}
    for idx, g in sorted(per_gpu.items()):
        res["gpus"][str(idx)] = {
            "pwr_mean": round(st.fmean(g["pwr"]), 1), "pwr_max": max(g["pwr"]),
            "sm_mean": round(st.fmean(g["sm"]), 1), "sm_max": max(g["sm"]),
            "mem_mean": round(st.fmean(g["mem"]), 1), "mem_max": max(g["mem"]),
            "temp_mean": round(st.fmean(g["gtemp"]), 1), "temp_max": max(g["gtemp"]),
        }
    allpwr = [v for g in per_gpu.values() for v in g["pwr"]]
    allsm = [v for g in per_gpu.values() for v in g["sm"]]
    if allpwr:
        res["pwr_mean_all"] = round(st.fmean(allpwr), 1)
        res["pwr_max_all"] = max(allpwr)
        res["sm_mean_all"] = round(st.fmean(allsm), 1)
        res["sm_max_all"] = max(allsm)
    return res


def downsample(rows, key, buckets=120, drop_head=0, drop_tail=0):
    core = rows[drop_head:len(rows) - drop_tail] if len(rows) > drop_head + drop_tail else rows
    n = len(core)
    if n == 0:
        return []
    size = max(1, n // buckets)
    out = []
    for i in range(0, n, size):
        chunk = core[i:i + size]
        vals = [r[key] for r in chunk if r.get(key) is not None]
        if not vals:
            continue
        out.append({"iter": chunk[0]["iter"], "mean": round(st.fmean(vals), 2),
                    "min": round(min(vals), 2), "max": round(max(vals), 2)})
    return out


def main():
    data = {}
    data["run"] = kv(RUN / "00-run-info.txt")
    data["args_check"] = json.loads((RUN / "args-check.json").read_text(encoding="utf-8"))

    # ---- 阶段索引 ----
    phases = []
    for line in (RUN / "index.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| 0") or line.startswith("| 1"):
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) >= 4:
                phases.append({"phase": c[0], "duration": c[1], "status": c[2]})
    data["phases"] = phases

    # ---- 硬件 ----
    hw = read_jsonl(RUN / "04-hw/hw.jsonl")
    data["hw"] = {"gpu": next((x for x in hw if x.get("kind") == "gpu"), {}),
                  "gemm_bf16": next((x for x in hw if x.get("kind") == "gemm" and x.get("dtype") == "bf16"), {}),
                  "gemm_fp8": next((x for x in hw if x.get("kind") == "gemm" and x.get("dtype") == "fp8"), {}),
                  "hbm": [x for x in hw if x.get("kind") == "hbm"],
                  "pcie": [x for x in hw if x.get("kind") == "pcie"],
                  "collectives": [x for x in hw if x.get("kind") == "collective"],
                  "dmon": parse_dmon(RUN / "04-hw/dmon.csv")}

    # ---- 训练档位 ----
    cases = []
    for group, sub, meta_file in [("dense8B", "06-train-s8b", "results.jsonl"),
                                 ("moe", "06-train-moe", "results.jsonl")]:
        meta = {x["tag"]: x for x in read_jsonl(RUN / sub / meta_file)}
        for log in sorted((RUN / sub).glob("*.log")):
            if log.name == "engine.log":
                continue
            tag = log.stem
            rows = parse_iterations(log)
            if not rows:
                continue
            c = {"group": group, "tag": tag, **parse_tag(tag), **meta.get(tag, {})}
            c["iters_logged"] = len(rows)
            c["step_ms"] = steady(rows, "step_ms")
            c["tflops"] = steady(rows, "tflops")
            c["loss_first"] = rows[0].get("loss")
            c["loss_last"] = rows[-1].get("loss")
            c["grad_last"] = rows[-1].get("grad")
            c["skipped_total"] = sum(r.get("skipped", 0) or 0 for r in rows)
            c["nan_total"] = sum(r.get("nan", 0) or 0 for r in rows)
            c["mem"] = parse_memory(log)
            c["dmon"] = parse_dmon(RUN / sub / ("dmon-" + tag + ".csv"))
            c["series"] = [{"iter": r["iter"], "step_ms": r.get("step_ms"), "tflops": r.get("tflops"),
                            "loss": r.get("loss")} for r in rows]
            cases.append(c)
    data["cases"] = cases

    # MoE jsonl 与 limit jsonl 的补充指标
    data["moe_meta"] = read_jsonl(RUN / "06-train-moe/moe.jsonl")
    data["limit"] = read_jsonl(RUN / "06-train-limit/limit.jsonl")
    data["limit_summary"] = json.loads((RUN / "06-train-limit/limit-summary.json").read_text(encoding="utf-8"))
    for rec in data["limit"]:
        tag = "w8_LIMITMOE_B_tp1_pp1_mbs%s_s%s" % (rec["mbs"], rec["seq"])
        log = RUN / "06-train-limit" / (tag + ".log")
        rec["tag"] = tag
        rec["iters_logged"] = len(parse_iterations(log))
        rec["mem"] = parse_memory(log) if log.exists() else {}
        rec["dmon"] = parse_dmon(RUN / "06-train-limit" / ("dmon-" + tag + ".csv"))

    # ---- IO / 故障 ----
    data["io"] = read_jsonl(RUN / "07-io/io.jsonl")
    data["fault"] = read_jsonl(RUN / "08-fault/fault.jsonl")

    # ---- 长稳 soak ----
    soak_cfg = [
        ("dense_2h", "w8_8B_soak_tp8_pp1_mbs4_s4096_gb8_bf16", "dmon-w8_8B_soak_tp8_pp1_mbs4_s4096_gb8_bf16.csv"),
        ("moe_30m", "w8_MOE_soak_ep8_tp1_pp1_mbs1_s4096_alltoall", "dmon-moe-soak-w8_MOE_ep8_alltoall.csv"),
        ("highmem_1h", "w8_HIGHMEM_8B_tp8_pp1_mbs1_s16384_bf16", "dmon-highmem-w8_8B_mbs1_s16384.csv"),
    ]
    soak = {}
    for name, tag, dmonf in soak_cfg:
        log = RUN / "09-soak" / (tag + ".log")
        if not log.exists():
            continue
        rows = parse_iterations(log)
        core = rows[20:-5] if len(rows) > 30 else rows
        s = {"tag": tag, "iters_logged": len(rows),
             "first_ts": rows[0].get("ts") if rows else None,
             "last_ts": rows[-1].get("ts") if rows else None,
             "step_ms": describe([r.get("step_ms") for r in core]),
             "tflops": describe([r.get("tflops") for r in core]),
             "skipped_total": sum(r.get("skipped", 0) or 0 for r in rows),
             "nan_total": sum(r.get("nan", 0) or 0 for r in rows),
             "tflops_series": downsample(rows, "tflops", 120, 20, 5),
             "step_series": downsample(rows, "step_ms", 120, 20, 5),
             "dmon": parse_dmon(RUN / "09-soak" / dmonf),
             "finished": (RUN / "09-soak" / (".ok_" + tag.replace("w8_", "w8_", 1))).exists()
                         or (name == "moe_30m" and (RUN / "09-soak/.ok_moe_soak").exists())}
        # 分段漂移：前 10% vs 后 10%
        vals = [r.get("tflops") for r in core if r.get("tflops") is not None]
        if len(vals) > 20:
            k = max(1, len(vals) // 10)
            head, tail = st.fmean(vals[:k]), st.fmean(vals[-k:])
            s["head_tflops"] = round(head, 2)
            s["tail_tflops"] = round(tail, 2)
            s["drift_pct"] = round((tail - head) / head * 100, 2)
        # 慢步比例（> 稳态中位数 1.5 倍）
        med = s["tflops"].get("median") or 0
        slow = [r for r in core if r.get("step_ms") and r.get("step_ms") > (s["step_ms"].get("median") or 1e9) * 1.5]
        s["slow_steps"] = len(slow)
        s["slow_pct"] = round(len(slow) / max(1, len(core)) * 100, 3)
        soak[name] = s
    data["soak"] = soak
    data["soak_meta"] = read_jsonl(RUN / "09-soak/results.jsonl")
    data["moe_soak_meta"] = read_jsonl(RUN / "09-soak/moe-soak.jsonl")

    # 显存时序（highmem）
    memfile = RUN / "09-soak/memused-w8_highmem.csv"
    byts = {}
    for line in memfile.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        p = line.split(",")
        if len(p) < 3:
            continue
        try:
            byts.setdefault(p[0], []).append(float(p[2]))
        except ValueError:
            pass
    memseries = [{"ts": ts, "mean_mib": round(st.fmean(v), 1), "max_mib": max(v)} for ts, v in sorted(byts.items())]
    data["highmem_series"] = memseries[::max(1, len(memseries) // 160)]
    data["highmem_peak_mib"] = max((x["max_mib"] for x in memseries), default=None)

    # 健康
    for key, fn in [("health_delta", "health-delta.json"), ("moe_health_delta", "moe-health-delta.json")]:
        p = RUN / "09-soak" / fn
        data[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 控制台摘要 ----
    print("run:", data["run"].get("run_id"), "| cases:", len(cases), "| soak:", len(soak))
    print("\n%-62s %10s %10s %10s %9s %8s" % ("tag", "step_ms", "TFLOPS", "MFU%", "mem_MB", "iters"))
    peak = data["hw"]["gemm_bf16"].get("tflops") or 690
    for c in sorted(cases, key=lambda x: -(x["tflops"].get("median") or 0)):
        t = c["tflops"].get("median") or 0
        print("%-62s %10.1f %10.1f %10.1f %9.0f %8d" % (
            c["tag"], c["step_ms"].get("median") or 0, t, t / peak * 100,
            c["mem"].get("max_reserved_mb") or 0, c["iters_logged"]))
    print("\nlimit:", [(r["mbs"], r["seq"], r["status"], r.get("step_s")) for r in data["limit"]])
    print("\nsoak:")
    for k, s in soak.items():
        print("  %-12s iters=%-6d tflops mean=%.1f p95=%.1f cv=%.2f%% drift=%.2f%% slow=%d finished=%s" % (
            k, s["iters_logged"], s["tflops"]["mean"], s["tflops"]["p95"], s["tflops"]["cv_pct"],
            s.get("drift_pct", 0), s["slow_steps"], s["finished"]))
    print("\nhw dmon:", {k: v for k, v in data["hw"]["dmon"].items() if k != "gpus"})
    print("collectives:", len(data["hw"]["collectives"]))
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
