# -*- coding: utf-8 -*-
"""node1-e2e-56b 结果统计分析：解析训练日志 + nvidia-smi dmon CSV，输出 stats.json"""
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(r"C:/Users/26488/Desktop/ht/e2e-megatron-56b-2501/node1-e2e-56b")
OUT = ROOT / "analysis"
LOG = ROOT / "logs" / "nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node1-20261112_000626.out"
CSV = ROOT / "logs" / "nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node1-dmon-20261112_000626.csv"
HELP = ROOT / "args-help.txt"

# ---------- 工具 ----------
def summarize(vals):
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    def q(p):
        i = min(n - 1, int(round(p * (n - 1))))
        return s[i]
    m = st.fmean(s)
    sd = st.pstdev(s) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "cv": round(sd / m * 100, 3) if m else 0,
            "min": round(s[0], 4), "p05": round(q(.05), 4), "p50": round(q(.5), 4),
            "p95": round(q(.95), 4), "p99": round(q(.99), 4), "max": round(s[-1], 4)}

# ---------- 1. 训练日志 ----------
text = LOG.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

ITER_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] iteration\s+(\d+)/\s*(\d+)\s*\| consumed samples:\s*(\d+)"
    r"\s*\| elapsed time per iteration \(ms\):\s*([\d.]+)"
    r"\s*\| throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
    r"\s*\| learning rate:\s*([\d.E+-]+)\s*\| global batch size:\s*(\d+)"
    r"\s*\| lm loss:\s*([\d.E+-]+)\s*\| loss scale:\s*([\d.E+-]+)"
    r"\s*\| grad norm:\s*([\d.]+)\s*\| number of skipped iterations:\s*(\d+)"
    r"\s*\| number of nan iterations:\s*(\d+)")
VAL_RE = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([\d.E+-]+) \| lm loss PPL: ([\d.E+-]+)")
EVAL_RE = re.compile(r"evaluate \.+: \(([\d.]+), ([\d.]+)\)")

T0 = datetime(2026, 11, 11, 16, 6, 28)  # UTC, dmon 首个采样(本地 00:06:28 = UTC-8h)

def hrs(ts):
    return round((ts - T0).total_seconds() / 3600, 4)

train, vals, eval_ms = [], [], []
for ln in lines:
    m = ITER_RE.search(ln)
    if m:
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        train.append({
            "iter": int(m.group(2)), "total": int(m.group(3)), "samples": int(m.group(4)),
            "ms": float(m.group(5)), "tflops": float(m.group(6)), "lr": float(m.group(7)),
            "gbs": int(m.group(8)), "loss": float(m.group(9)), "loss_scale": float(m.group(10)),
            "gnorm": float(m.group(11)), "skipped": int(m.group(12)), "nan": int(m.group(13)),
            "t": hrs(ts), "ts": m.group(1)})
        continue
    m = VAL_RE.search(ln)
    if m:
        vals.append({"iter": int(m.group(1)), "loss": float(m.group(2)), "ppl": float(m.group(3))})
        continue
    m = EVAL_RE.search(ln)
    if m:
        eval_ms.append(float(m.group(2)))

# 验证点时间：用同 iteration 的训练点时间近似
tmap = {d["iter"]: d["t"] for d in train}
for v in vals:
    v["t"] = tmap.get(v["iter"], None)

# 日志分类与关键事件
cat = Counter()
events = {
    "traceback": 0, "oom": 0, "cuda_error": 0, "nccl_timeout": 0, "nccl_ib_timeout_env": 0, "watchdog": 0,
    "abort_complete": 0, "pg_not_destroyed": 0, "plugin_symbol_missing": 0,
    "socket_hostname_warn": 0, "rope_warning": 0, "deprecation": 0,
    "resource_warning": 0, "torchrun_warn": 0, "nan_warn": 0}
for ln in lines:
    low = ln.lower()
    if ITER_RE.search(ln): cat["训练迭代日志"] += 1
    elif VAL_RE.search(ln): cat["验证日志"] += 1
    elif "NCCL INFO" in ln: cat["NCCL 通信日志"] += 1
    elif "evaluate " in ln or "(min, max) time" in ln or set(ln.strip()) == {"-"}: cat["耗时统计日志"] += 1
    elif "DeprecationWarning" in ln: cat["Python 弃用警告"] += 1
    elif "UserWarning" in ln or "warnings.warn" in ln: cat["用户/框架警告"] += 1
    elif "ResourceWarning" in ln: cat["资源清理警告"] += 1
    elif re.search(r"\[(W|E)\d{4}|\[rank\d+\]:\[W", ln): cat["分布式/框架 Warning"] += 1
    elif ln.strip() == "" : cat["空行"] += 1
    else: cat["环境横幅/其他"] += 1

    if "Traceback" in ln: events["traceback"] += 1
    if "out of memory" in low or "oom" in low.replace("room", ""): events["oom"] += 1
    if "cuda error" in low: events["cuda_error"] += 1
    if "NCCL_IB_TIMEOUT set by environment" in ln: events["nccl_ib_timeout_env"] += 1
    if "Watchdog caught" in ln or "collective operation timeout" in low or "Connection timed out" in ln: events["nccl_timeout"] += 1
    if "watchdog" in low: events["watchdog"] += 1
    if "Abort COMPLETE" in ln: events["abort_complete"] += 1
    if "has NOT been destroyed" in ln: events["pg_not_destroyed"] += 1
    if "Failed to find nccl" in ln: events["plugin_symbol_missing"] += 1
    if "hostname of the client socket" in ln: events["socket_hostname_warn"] += 1
    if "fused RoPE" in ln: events["rope_warning"] += 1
    if "DeprecationWarning" in ln: events["deprecation"] += 1
    if "ResourceWarning" in ln: events["resource_warning"] += 1
    if "OMP_NUM_THREADS" in ln or "one_logger package is required" in ln: events["torchrun_warn"] += 1
    if "nan iterations" in low and "number of nan iterations:   0" not in low: events["nan_warn"] += 1

# Abort 按通信域规模分组
aborts = Counter(re.findall(r"nranks (\d+)", "\n".join(l for l in lines if "Abort COMPLETE" in l)))

# ---------- 2. GPU dmon ----------
COLS = ["date", "time", "gpu", "pwr", "gtemp", "mtemp", "sm", "mem", "enc", "dec",
        "jpg", "ofa", "mclk", "pclk", "pviol", "tviol", "fb", "bar1", "ccpm", "rxpci", "txpci"]
per_gpu = {g: defaultdict(list) for g in range(8)}
B30 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # bucket30[g][metric][]
B1 = defaultdict(lambda: defaultdict(list))                          # startup 1min bucket
heat = defaultdict(lambda: defaultdict(list))                        # heat[metric][gpu][bucket] 后整理
pviol_nz = tviol_nz = 0
ts_first = ts_last = None
nrows = 0
sm_dist = Counter()
pviol_dist = Counter()
pclk_dist = Counter()
fb_peak = 0
sm_ge90 = 0

with CSV.open(encoding="utf-8", errors="replace") as fh:
    for ln in fh:
        if ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) != 21 or not p[0].isdigit():
            continue
        try:
            dt = datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S") - timedelta(hours=8)  # 本地 UTC+8 -> UTC
        except ValueError:
            continue
        g = int(p[2])
        v = {c: int(x) for c, x in zip(COLS[3:], p[3:])}
        nrows += 1
        t = hrs(dt)
        ts_first = ts_first or dt
        ts_last = dt
        b30 = int(t * 2)      # 30 分钟桶
        for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "pclk", "fb", "rxpci", "txpci", "mclk", "pviol", "tviol"):
            per_gpu[g][k].append(v[k])
            B30[g][b30][k].append(v[k])
        if t < 0.5:
            B1[int(t * 60)]["fb"].append(v["fb"])
            B1[int(t * 60)]["pwr"].append(v["pwr"])
            B1[int(t * 60)]["pclk"].append(v["pclk"])
            B1[int(t * 60)]["sm"].append(v["sm"])
        pviol_nz += (v["pviol"] > 0)
        tviol_nz += (v["tviol"] > 0)
        fb_peak = max(fb_peak, v["fb"])
        sm_dist[min(v["sm"] // 10 * 10, 90)] += 1
        pviol_dist[v["pviol"]] += 1
        sm_ge90 += (v["sm"] >= 90)
        pclk_dist[v["pclk"]] += 1

# 每卡汇总
gpu_summary = {}
for g in range(8):
    d = per_gpu[g]
    gpu_summary[g] = {
        "samples": len(d["pwr"]),
        "pwr": summarize(d["pwr"]), "gtemp": summarize(d["gtemp"]), "mtemp": summarize(d["mtemp"]),
        "sm": summarize(d["sm"]), "mem": summarize(d["mem"]), "fb": summarize(d["fb"]),
        "pclk": summarize(d["pclk"]), "rxpci": summarize(d["rxpci"]), "txpci": summarize(d["txpci"]),
        "pviol_nz": sum(1 for x in d["pviol"] if x > 0), "tviol_nz": sum(1 for x in d["tviol"] if x > 0),
    }

# 30 分钟分桶（8 卡均值 + 每卡温度/功耗/显存用于热力图）
nb = max(max(B30[g].keys()) for g in range(8)) + 1
buckets = []
for b in range(nb):
    rec = {"b": b, "t": round(b * .5, 2)}
    allm = defaultdict(list)
    gp = {}
    for g in range(8):
        dd = B30[g].get(b, {})
        gp[g] = {k: round(st.fmean(dd[k]), 2) if dd.get(k) else None
                 for k in ("pwr", "gtemp", "fb", "sm", "pclk")}
        for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "pclk", "fb", "rxpci", "txpci", "pviol"):
            allm[k] += dd.get(k, [])
    for k, vv in allm.items():
        rec[k] = round(st.fmean(vv), 2) if vv else None
    rec["gtemp_max"] = round(max(x for x in [gp[g]["gtemp"] for g in range(8)] if x is not None), 2)
    rec["gtemp_min"] = round(min(x for x in [gp[g]["gtemp"] for g in range(8)] if x is not None), 2)
    rec["per_gpu"] = gp
    buckets.append(rec)

# 启动阶段 1 分钟桶
startup = []
for b in sorted(B1):
    d = B1[b]
    startup.append({"m": b, "fb": round(st.fmean(d["fb"]), 1) if d["fb"] else None,
                    "pwr": round(st.fmean(d["pwr"]), 1) if d["pwr"] else None,
                    "pclk": round(st.fmean(d["pclk"]), 1) if d["pclk"] else None,
                    "sm": round(st.fmean(d["sm"]), 1) if d["sm"] else None})

# ---------- 3. 自检 / args-help ----------
selfcheck = {}
for ln in (ROOT / "selfcheck.log").read_text(encoding="utf-8", errors="replace").splitlines():
    if re.match(r"^[A-Z][A-Z_]*=", ln):
        k, v = ln.split("=", 1)
        selfcheck[k.strip()] = v.strip()
help_txt = HELP.read_text(encoding="utf-8", errors="replace")
flags = re.findall(r"^\s*(--[a-zA-Z0-9-]+)", help_txt, re.M)
key_flags = ["attention-backend", "squared-relu", "rotary-base", "log-throughput", "tp-comm-overlap",
             "ckpt-format", "async-save", "use-precision-aware-optimizer", "recompute-granularity",
             "disable-gloo-process-groups", "exit-duration-in-mins", "fp8-format"]
flag_cover = {f: (f"--{f}" in flags) for f in key_flags}

# ---------- 4. 派生统计 ----------
stable = [d for d in train if d["iter"] >= 500]          # 稳定期
warm = [d for d in train if d["iter"] < 500]              # 预热期
derived = {
    "throughput_all": summarize([d["tflops"] for d in train]),
    "throughput_stable": summarize([d["tflops"] for d in stable]),
    "throughput_warm": summarize([d["tflops"] for d in warm]),
    "iter_ms_all": summarize([d["ms"] for d in train]),
    "iter_ms_stable": summarize([d["ms"] for d in stable]),
    "gnorm": summarize([d["gnorm"] for d in train]),
    "eval_ms": summarize(eval_ms),
    "train_loss_first": train[0]["loss"], "train_loss_last": train[-1]["loss"],
    "val_loss_first": vals[0]["loss"], "val_loss_last": vals[-1]["loss"],
    "ppl_first": vals[0]["ppl"], "ppl_last": vals[-1]["ppl"],
    "skipped_total": sum(d["skipped"] for d in train), "nan_total": sum(d["nan"] for d in train),
    "samples_total": train[-1]["samples"],
    "tokens_total": train[-1]["samples"] * 8192,
    "hours": round(train[-1]["t"], 3),
    "iter_last": train[-1]["iter"], "iter_target": train[-1]["total"],
    "progress": round(train[-1]["iter"] / train[-1]["total"] * 100, 2),
    "avg_tflops_node": round(st.fmean([d["tflops"] for d in stable]) * 8, 1),
    "startup_min": round(train[0]["t"] * 60, 1),
}
# 收敛里程碑
for thr in (1.0, 0.1, 0.01, 0.005):
    hit = next((d["iter"] for d in train if d["loss"] < thr), None)
    derived[f"iter_loss_below_{thr}"] = hit
# 异常点
ms_mean, ms_sd = derived["iter_ms_stable"]["mean"], derived["iter_ms_stable"]["sd"]
gn_mean, gn_sd = derived["gnorm"]["mean"], derived["gnorm"]["sd"]
derived["slow_iters"] = [{"iter": d["iter"], "ms": d["ms"]} for d in stable if d["ms"] > ms_mean + 3 * ms_sd]
derived["gnorm_spikes"] = [{"iter": d["iter"], "gnorm": d["gnorm"]} for d in train if d["gnorm"] > gn_mean + 3 * gn_sd]
derived["gnorm_gt8"] = [{"iter": d["iter"], "gnorm": d["gnorm"], "loss": d["loss"]}
                        for d in train if d["gnorm"] > 8]
derived["val_rebounds"] = [{"iter": vals[i]["iter"], "from": vals[i-1]["loss"], "to": vals[i]["loss"],
                            "pct": round((vals[i]["loss"] / vals[i-1]["loss"] - 1) * 100, 1)}
                           for i in range(1, len(vals)) if vals[i]["loss"] > vals[i-1]["loss"] * 1.3]
# 跨卡温度/功耗差异
even = [gpu_summary[g]["gtemp"]["mean"] for g in (0, 2, 4, 6)]
odd = [gpu_summary[g]["gtemp"]["mean"] for g in (1, 3, 5, 7)]
derived["gtemp_even_avg"] = round(st.fmean(even), 2)
derived["gtemp_odd_avg"] = round(st.fmean(odd), 2)
derived["pwr_even_avg"] = round(st.fmean([gpu_summary[g]["pwr"]["mean"] for g in (0, 2, 4, 6)]), 2)
derived["pwr_odd_avg"] = round(st.fmean([gpu_summary[g]["pwr"]["mean"] for g in (1, 3, 5, 7)]), 2)
derived["fb_peak_mb"] = fb_peak
derived["pviol_nz"] = pviol_nz
derived["tviol_nz"] = tviol_nz
derived["sm_dist"] = dict(sorted(sm_dist.items()))
derived["sm_ge90_pct"] = round(sm_ge90 / nrows * 100, 2)
derived["sm_idle_rows"] = sum(1 for k, v in sm_dist.items() if k == 0)  # 0-9% 档
derived["pviol_dist"] = {"0": pviol_dist.get(0, 0),
                         "1-49": sum(v for k, v in pviol_dist.items() if 1 <= k <= 49),
                         "50-89": sum(v for k, v in pviol_dist.items() if 50 <= k <= 89),
                         "90-99": sum(v for k, v in pviol_dist.items() if 90 <= k <= 99),
                         "100": pviol_dist.get(100, 0)}
derived["power_cap_pct"] = round(pviol_dist.get(100, 0) / nrows * 100, 2)
derived["fb_stable_min"] = next((x["m"] for x in startup if x["fb"] and x["fb"] > 110000), None)
derived["pclk_dist"] = dict(sorted(pclk_dist.items(), key=lambda kv: -kv[1])[:8])
derived["gpu_rows"] = nrows
derived["ts_first"] = ts_first.strftime("%Y-%m-%d %H:%M:%S UTC")
derived["ts_last"] = ts_last.strftime("%Y-%m-%d %H:%M:%S UTC")
derived["csv_lines"] = sum(1 for _ in CSV.open(encoding="utf-8", errors="replace"))

result = {
    "meta": {"root": str(ROOT), "log_lines": len(lines),
             "case": "Nemotron-56B / TP8 PP1 DP2 / MBS1 / SEQ8192 / GBS64 / BF16 / node1",
             "log_name": LOG.name, "csv_name": CSV.name},
    "selfcheck": selfcheck,
    "flags": {"total_lines": len(re.findall(r"^\s*--", help_txt, re.M)),
              "total": len(set(flags)), "cover": flag_cover},
    "train": train, "val": vals, "eval_ms": eval_ms,
    "gpu": {"summary": gpu_summary, "buckets": buckets, "startup": startup},
    "log": {"categories": dict(cat.most_common()), "events": events, "aborts": dict(aborts)},
    "stats": derived,
}
(OUT / "stats.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

# ---------- 摘要 ----------
print("=== 数据规模 ===")
print(f"日志 {len(lines)} 行 | 训练点 {len(train)} | 验证点 {len(vals)} | evaluate 计时 {len(eval_ms)}")
print(f"CSV 有效行 {nrows} | 8 卡各 {gpu_summary[0]['samples']} 采样 | {derived['ts_first']} ~ {derived['ts_last']}")
print(f"30min 桶 {len(buckets)} | 启动 1min 桶 {len(startup)}")
print("\n=== 训练概览 ===")
for k in ("hours", "iter_last", "iter_target", "progress", "train_loss_first", "train_loss_last",
          "val_loss_first", "val_loss_last", "ppl_first", "ppl_last", "skipped_total", "nan_total",
          "startup_min", "avg_tflops_node"):
    print(f"  {k}: {derived[k]}")
print("\n=== 吞吐 / 步时 ===")
for k in ("throughput_warm", "throughput_stable", "iter_ms_stable", "eval_ms", "gnorm"):
    print(f"  {k}: {derived[k]}")
print(f"  里程碑 loss<1/0.1/0.01/0.005 @ iter {[derived[f'iter_loss_below_{t}'] for t in (1.0,0.1,0.01,0.005)]}")
print(f"  慢步(>均值+3σ) {derived['slow_iters']}")
print(f"  grad spike 数 {len(derived['gnorm_spikes'])}, 最大 {derived['gnorm']['max']}")
print(f"  验证反弹 {derived['val_rebounds']}")
print("\n=== GPU ===")
print(f"  显存峰值 {fb_peak} MB | 稳定于第 {derived['fb_stable_min']} 分钟 | pviol {pviol_nz} | tviol {tviol_nz}")
print(f"  pviol 分布 {derived['pviol_dist']} | 功率封顶(100%) 占比 {derived['power_cap_pct']}% | SM>=90% 占比 {derived['sm_ge90_pct']}%")
print(f"  偶/奇卡 平均温度 {derived['gtemp_even_avg']}/{derived['gtemp_odd_avg']} C, 功耗 {derived['pwr_even_avg']}/{derived['pwr_odd_avg']} W")
print(f"  SM 分布 {derived['sm_dist']}")
print(f"  pclk 分布 {derived['pclk_dist']}")
for g in range(8):
    s = gpu_summary[g]
    print(f"  GPU{g}: pwr {s['pwr']['mean']}/{s['pwr']['max']}W  gtemp {s['gtemp']['mean']}/{s['gtemp']['max']}C  sm {s['sm']['mean']}%  fb {s['fb']['mean']}MB  pclk {s['pclk']['mean']}MHz")
print("\n=== 日志事件 ===")
print("  分类:", dict(cat.most_common()))
print("  事件:", events)
print("  Abort 按 nranks:", dict(aborts))
print("  自检:", selfcheck, "| flags:", len(set(flags)), "| 覆盖:", flag_cover)
print(f"\n输出 -> {OUT / 'stats.json'}")
