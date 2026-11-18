# -*- coding: utf-8 -*-
"""双机联合分析：node0 主日志 + node0 dmon + node1 stats.json + node0 历史基准日志
输出 stats_dual.json"""
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(r"C:/Users/26488/Desktop/ht/e2e-megatron-56b-2501")
N0 = BASE / "node0-e2e-56b"
N1 = BASE / "node1-e2e-56b"
OUT = N1 / "analysis"
LOG0 = N0 / "logs" / "nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node0-20261112_000625.out"
CSV0 = N0 / "logs" / "nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node0-dmon-20261112_000625.csv"
N1STATS = OUT / "stats.json"
N1CSV = N1 / "logs" / "nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node1-dmon-20261112_000626.csv"
T0 = datetime(2026, 11, 11, 16, 6, 27)  # UTC，node0 dmon 首个采样


def summarize(vals):
    if not vals:
        return {}
    s = sorted(vals); n = len(s)
    q = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    m = st.fmean(s); sd = st.pstdev(s) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 3), "sd": round(sd, 3),
            "cv": round(sd / m * 100, 3) if m else 0,
            "min": round(s[0], 3), "p50": round(q(.5), 3),
            "p95": round(q(.95), 3), "max": round(s[-1], 3)}


def hrs(ts):
    return round((ts - T0).total_seconds() / 3600, 4)


# ---------------- 1. node0 主日志 ----------------
txt = LOG0.read_text(encoding="utf-8", errors="replace")
lines = txt.splitlines()

# 配置表（形如 "  key ....... value"）
conf = {}
for ln in lines:
    m = re.match(r"^\s{2}([a-z_][a-z0-9_]*)\s+\.+\s+(.*?)\s*$", ln)
    if m:
        conf[m.group(1)] = m.group(2).strip()
KEY = ["num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups",
       "kv_channels", "vocab_size", "seq_length", "max_position_embeddings", "micro_batch_size",
       "global_batch_size", "tensor_model_parallel_size", "pipeline_model_parallel_size",
       "data_parallel_size", "world_size", "train_iters", "exit_duration_in_mins", "eval_interval",
       "eval_iters", "lr", "lr_warmup_iters", "lr_decay_iters", "lr_decay_style", "min_lr",
       "clip_grad", "adam_beta1", "adam_beta2", "weight_decay", "optimizer", "bf16", "fp8",
       "mock_data", "attention_backend", "transformer_impl", "sequence_parallel", "tp_comm_overlap",
       "overlap_grad_reduce", "overlap_param_gather", "use_distributed_optimizer",
       "recompute_granularity", "save", "save_interval", "log_interval", "distributed_backend",
       "params_dtype", "main_params_dtype", "manual_gc", "inprocess_restart"]
config = {k: conf.get(k) for k in KEY if k in conf}

# 参数量 / 里程碑 / 显存
params = {}
for ln in lines:
    m = re.search(r"Number of parameters in transformer block in billions:\s*([\d.]+)", ln)
    if m: params["transformer_B"] = float(m.group(1))
    m = re.search(r"Number of parameters in embedding layers in billions:\s*([\d.]+)", ln)
    if m: params["embedding_B"] = float(m.group(1))
    m = re.search(r"Total number of parameters in billions:\s*([\d.]+)", ln)
    if m: params["total_B"] = float(m.group(1))
    m = re.search(r"Number of parameters in most loaded shard in billions:\s*([\d.]+)", ln)
    if m: params["shard_B"] = float(m.group(1))
    m = re.search(r"number of parameters on \(tensor, pipeline\) model parallel rank \((\d+), (\d+)\): (\d+)", ln)
    if m: params.setdefault("per_tp_rank", {})[int(m.group(1))] = int(m.group(3))

milestones = []
for ln in lines:
    m = re.search(r"\[([^\]]+)\] datetime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
    if m:
        ts = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")
        milestones.append({"label": m.group(1), "ts": m.group(2), "t": hrs(ts), "t_min": round(hrs(ts) * 60, 2)})
exit_min = None
m = re.search(r"\[exiting program after ([\d.]+) minutes\]", txt)
if m: exit_min = float(m.group(1))

mem = []
for ln in lines:
    m = re.search(r"\[Rank (\d+)\] \(after (\d+) iterations\) memory \(MB\) \| allocated: ([\d.]+) \| max allocated: ([\d.]+) \| reserved: ([\d.]+) \| max reserved: ([\d.]+)", ln)
    if m:
        mem.append({"rank": int(m.group(1)), "after_iter": int(m.group(2)),
                    "allocated": float(m.group(3)), "max_allocated": float(m.group(4)),
                    "reserved": float(m.group(5)), "max_reserved": float(m.group(6))})

# 事件与 NCCL 拓扑
low = txt.lower()
ev = {
    "traceback": txt.count("Traceback"), "oom": len(re.findall(r"out of memory", low)),
    "cuda_error": len(re.findall(r"cuda error", low)),
    "nccl_timeout": len(re.findall(r"watchdog caught|collective operation timeout", low)),
    "abort_complete": txt.count("Abort COMPLETE"),
    "pg_not_destroyed": txt.count("has NOT been destroyed"),
    "rerun_state_machine_warn": txt.count("megatron.core.rerun_state_machine"),
    "plugin_symbol_missing": txt.count("Failed to find nccl"),
    "resource_warning": txt.count("ResourceWarning"),
    "deprecation": txt.count("DeprecationWarning"),
    "rope_warning": txt.count("fused RoPE"),
}
ib = [l for l in lines if "via NET/IBext" in l]
p2p = [l for l in lines if "via P2P" in l]
ib_pairs = sorted(set(re.findall(r"Channel \d+/\d+ : (\d+)\[\d+\] -> (\d+)\[\d+\]", "\n".join(ib))))
topo = {"ib_channels": len(ib), "p2p_channels": len(p2p),
        "ib_ext": sorted(set(re.findall(r"NET/(IBext_v\d+)", "\n".join(ib)))),
        "gdrdma": txt.count("GDRDMA"),
        "cross_node_pairs": len(ib_pairs),
        "pair_sample": ib_pairs[:8],
        "abort_nranks": dict(Counter(re.findall(r"Abort COMPLETE", "\n".join(lines))))}
aborts = Counter(re.findall(r"nranks (\d+)", "\n".join(l for l in lines if "Abort COMPLETE" in l)))
topo["abort_by_nranks"] = dict(aborts)
cat = Counter()
for ln in lines:
    if "NCCL INFO" in ln: cat["NCCL 通信日志"] += 1
    elif "megatron.core.rerun_state_machine" in ln: cat["RerunStateMachine 警告"] += 1
    elif re.match(r"^\s{2}[a-z_]+\s+\.+", ln): cat["超参配置表"] += 1
    elif "memory (MB)" in ln: cat["显存统计"] += 1
    elif "datetime:" in ln: cat["里程碑"] += 1
    elif "DeprecationWarning" in ln: cat["Python 弃用警告"] += 1
    elif "UserWarning" in ln or "warnings.warn" in ln: cat["框架警告"] += 1
    elif "ResourceWarning" in ln: cat["资源清理警告"] += 1
    elif re.search(r"\[(W|E)\d{4}|\[rank\d+\]:\[W", ln): cat["分布式 Warning"] += 1
    elif ln.strip() == "": cat["空行"] += 1
    else: cat["环境横幅/其他"] += 1

# ---------------- 2. node0 dmon ----------------
COLS = ["date", "time", "gpu", "pwr", "gtemp", "mtemp", "sm", "mem", "enc", "dec", "jpg",
        "ofa", "mclk", "pclk", "pviol", "tviol", "fb", "bar1", "ccpm", "rxpci", "txpci"]
per_gpu = {g: defaultdict(list) for g in range(8)}
B30 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
B1 = defaultdict(lambda: defaultdict(list))
pviol_dist = Counter(); pclk_dist = Counter(); sm_dist = Counter()
ts_by_gpu = defaultdict(list)
pviol_nz = tviol_nz = sm_ge90 = fb_peak = 0
nrows = 0
ts_first = ts_last = None
with CSV0.open(encoding="utf-8", errors="replace") as fh:
    for ln in fh:
        if ln.startswith("#"): continue
        p = ln.split()
        if len(p) != 21 or not p[0].isdigit(): continue
        try:
            dt = datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S") - timedelta(hours=8)
        except ValueError:
            continue
        g = int(p[2]); v = {c: int(x) for c, x in zip(COLS[3:], p[3:])}
        nrows += 1; t = hrs(dt)
        ts_first = ts_first or dt; ts_last = dt
        ts_by_gpu[g].append(dt)
        b30 = int(t * 2)
        for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "pclk", "fb", "rxpci", "txpci", "pviol", "tviol"):
            per_gpu[g][k].append(v[k]); B30[g][b30][k].append(v[k])
        if t < 0.5:
            for k in ("fb", "pwr", "pclk", "sm"): B1[int(t * 60)][k].append(v[k])
        pviol_nz += v["pviol"] > 0; tviol_nz += v["tviol"] > 0; sm_ge90 += v["sm"] >= 90
        fb_peak = max(fb_peak, v["fb"])
        pviol_dist[v["pviol"]] += 1; pclk_dist[v["pclk"]] += 1
        sm_dist[min(v["sm"] // 10 * 10, 90)] += 1

gpu0 = {}
for g in range(8):
    d = per_gpu[g]
    gpu0[g] = {"samples": len(d["pwr"]),
               **{k: summarize(d[k]) for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "fb", "pclk", "rxpci", "txpci")},
               "pviol_nz": sum(1 for x in d["pviol"] if x > 0), "tviol_nz": sum(1 for x in d["tviol"] if x > 0)}

nb = max(max(B30[g].keys()) for g in range(8)) + 1
buckets0 = []
for b in range(nb):
    rec = {"b": b, "t": round(b * .5, 2)}; allm = defaultdict(list); gp = {}
    for g in range(8):
        dd = B30[g].get(b, {})
        gp[g] = {k: round(st.fmean(dd[k]), 2) if dd.get(k) else None
                 for k in ("pwr", "gtemp", "fb", "sm", "pclk")}
        for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "pclk", "fb", "rxpci", "txpci", "pviol"):
            allm[k] += dd.get(k, [])
    for k, vv in allm.items(): rec[k] = round(st.fmean(vv), 2) if vv else None
    rec["per_gpu"] = gp
    buckets0.append(rec)

startup0 = [{"m": b, **{k: round(st.fmean(d[k]), 1) if d[k] else None for k in ("fb", "pwr", "pclk", "sm")}}
            for b, d in sorted(B1.items())]

# 采样完整性：时间间隔分布与空洞
gaps = []
for dt0, dt1 in zip(ts_by_gpu[0], ts_by_gpu[0][1:]):
    gaps.append((dt1 - dt0).total_seconds())
gaps_s = summarize(gaps)
holes = [(ts_by_gpu[0][i].strftime("%H:%M:%S"), ts_by_gpu[0][i + 1].strftime("%H:%M:%S"), round(g, 1))
         for i, g in enumerate(gaps) if g > 20]

# 末尾显存释放快照（最后 3 个时间点）
tail_times = sorted(set(ts_by_gpu[0]))[-3:]
tail_fb = defaultdict(dict)
with CSV0.open(encoding="utf-8", errors="replace") as fh:
    for ln in fh:
        if ln.startswith("#"): continue
        p = ln.split()
        if len(p) != 21 or not p[0].isdigit(): continue
        try: dt = datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S") - timedelta(hours=8)
        except ValueError: continue
        if dt in tail_times: tail_fb[dt.strftime("%H:%M:%S")][int(p[2])] = int(p[16])

node0_gpu = {
    "rows": nrows, "points": len(ts_by_gpu[0]), "ts_first": ts_first.strftime("%Y-%m-%d %H:%M:%S UTC"),
    "ts_last": ts_last.strftime("%Y-%m-%d %H:%M:%S UTC"),
    "fb_peak": fb_peak, "pviol_nz": pviol_nz, "tviol_nz": tviol_nz,
    "sm_ge90_pct": round(sm_ge90 / nrows * 100, 2),
    "pviol_dist": {"0": pviol_dist.get(0, 0), "1-49": sum(v for k, v in pviol_dist.items() if 1 <= k <= 49),
                   "50-89": sum(v for k, v in pviol_dist.items() if 50 <= k <= 89),
                   "90-99": sum(v for k, v in pviol_dist.items() if 90 <= k <= 99),
                   "100": pviol_dist.get(100, 0)},
    "power_cap_pct": round(pviol_dist.get(100, 0) / nrows * 100, 2),
    "pclk_dist": dict(sorted(pclk_dist.items(), key=lambda kv: -kv[1])[:8]),
    "sm_dist": dict(sorted(sm_dist.items())),
    "gap": gaps_s, "holes": holes[:12], "holes_n": len(holes),
    "tail_fb": {k: v for k, v in tail_fb.items()},
    "summary": gpu0, "buckets": buckets0, "startup": startup0,
}

# ---------------- 3. node1 数据 ----------------
n1 = json.loads(N1STATS.read_text(encoding="utf-8"))
gpu1 = {"rows": n1["stats"]["gpu_rows"], "points": 15410,
        "ts_first": n1["stats"]["ts_first"], "ts_last": n1["stats"]["ts_last"],
        "fb_peak": n1["stats"]["fb_peak_mb"], "pviol_nz": n1["stats"]["pviol_nz"],
        "tviol_nz": n1["stats"]["tviol_nz"], "sm_ge90_pct": n1["stats"]["sm_ge90_pct"],
        "pviol_dist": n1["stats"]["pviol_dist"], "power_cap_pct": n1["stats"]["power_cap_pct"],
        "summary": n1["gpu"]["summary"], "buckets": n1["gpu"]["buckets"],
        "startup": n1["gpu"]["startup"]}

# ---------------- 4. 双机对比 ----------------
def node_avg(gp, metric):
    return round(st.fmean([gp[str(g)][metric]["mean"] if str(g) in gp else gp[g][metric]["mean"]
                           for g in range(8)]), 3)

cmp = {}
for label, obj in (("node0", node0_gpu), ("node1", gpu1)):
    gp = obj["summary"]
    cmp[label] = {m: node_avg(gp, m) for m in ("pwr", "gtemp", "mtemp", "sm", "mem", "fb", "pclk", "rxpci", "txpci")}
    cmp[label]["power_cap_pct"] = obj["power_cap_pct"]
    cmp[label]["sm_ge90_pct"] = obj["sm_ge90_pct"]
    cmp[label]["rows"] = obj["rows"]
    cmp[label]["points"] = obj["points"]

per_card = []
for g in range(8):
    a = node0_gpu["summary"][g]; b = gpu1["summary"][str(g)]
    per_card.append({"gpu": g,
                     "n0_pwr": a["pwr"]["mean"], "n1_pwr": b["pwr"]["mean"],
                     "n0_gtemp": a["gtemp"]["mean"], "n1_gtemp": b["gtemp"]["mean"],
                     "n0_fb": a["fb"]["mean"], "n1_fb": b["fb"]["mean"],
                     "n0_pclk": a["pclk"]["mean"], "n1_pclk": b["pclk"]["mean"],
                     "n0_sm": a["sm"]["mean"], "n1_sm": b["sm"]["mean"]})
node0_gpu["per_card"] = per_card

dual = {
    "cmp": cmp,
    "diff": {k: round(cmp["node0"][k] - cmp["node1"][k], 3) for k in
             ("pwr", "gtemp", "mtemp", "sm", "mem", "fb", "pclk", "rxpci", "txpci", "power_cap_pct", "sm_ge90_pct")},
    "sample_gap_points": gpu1["points"] - node0_gpu["points"],
    "exit_time_n0": milestones[-1]["ts"] if milestones else None,
    "exit_time_n1": n1["train"][-1]["ts"],
    "exit_min_config": exit_min,
    "exit_min_logged": exit_min,
    "hours_config": round(exit_min / 60, 3) if exit_min else None,
    "hours_from_train": n1["stats"]["hours"],
    "grad_accum": 64 // (1 * 2),
    "total_params_B": params.get("total_B"),
    "reserved_vs_fb_MB": round(node0_gpu["summary"][0]["fb"]["mean"] - mem[0]["max_reserved"], 1) if mem else None,
    "all16_pwr": round(st.fmean([node0_gpu["summary"][g]["pwr"]["mean"] for g in range(8)] +
                                [gpu1["summary"][str(g)]["pwr"]["mean"] for g in range(8)]), 2),
    "all16_temp_range": [round(min([node0_gpu["summary"][g]["gtemp"]["mean"] for g in range(8)] +
                                   [gpu1["summary"][str(g)]["gtemp"]["mean"] for g in range(8)]), 2),
                         round(max([node0_gpu["summary"][g]["gtemp"]["mean"] for g in range(8)] +
                                   [gpu1["summary"][str(g)]["gtemp"]["mean"] for g in range(8)]), 2)],
    "all16_fb_range": [round(min([node0_gpu["summary"][g]["fb"]["mean"] for g in range(8)] +
                                 [gpu1["summary"][str(g)]["fb"]["mean"] for g in range(8)]), 1),
                       round(max([node0_gpu["summary"][g]["fb"]["mean"] for g in range(8)] +
                                 [gpu1["summary"][str(g)]["fb"]["mean"] for g in range(8)]), 1)],
}

# ---------------- 5. 历史基准日志 ----------------
history = []
for f in sorted((N0 / "logs").glob("single-*.out")):
    t = f.read_text(encoding="utf-8", errors="replace")
    kind = "8B 单机" if "single-8b" in f.name else ("MFU 基准" if "single-mfu" in f.name else "56B 单机")
    iters = [int(x) for x in re.findall(r"iteration\s+(\d+)/\s*\d+", t)]
    tfl = [float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", t)]
    errs = Counter()
    for pat, name in ((r"RuntimeError", "RuntimeError"), (r"torch\.OutOfMemoryError|CUDA out of memory", "OOM"),
                      (r"FileNotFoundError", "文件缺失"), (r"AssertionError", "断言失败"),
                      (r"ValueError", "参数错误"), (r"NCCL error|ncclInternal", "NCCL 错误"),
                      (r"unexpected pos|inline_container", "容器/检查点读取"),
                      (r"ConnectionError|Connection refused", "连接失败")):
        c = len(re.findall(pat, t))
        if c: errs[name] = c
    tb = t.count("Traceback")
    if tb == 0 and not iters: status = "初始化中断"
    elif tb > 0: status = "失败"
    else: status = "完成短跑"
    history.append({"file": f.name, "kind": kind, "size_kb": round(f.stat().st_size / 1024, 1),
                    "status": status, "iter_lines": len(iters), "max_iter": max(iters) if iters else 0,
                    "last_tflops": tfl[-1] if tfl else None, "tracebacks": tb,
                    "errors": dict(errs)})
hist_summary = {"runs": len(history), "by_status": dict(Counter(h["status"] for h in history)),
                "by_kind": dict(Counter(h["kind"] for h in history)),
                "total_tracebacks": sum(h["tracebacks"] for h in history),
                "error_types": dict(Counter(k for h in history for k in h["errors"])),
                "max_iter": max(h["max_iter"] for h in history),
                "successful_tflops": [h["last_tflops"] for h in history if h["last_tflops"]]}

# ---------------- 5b. MFU ----------------
# 硬件峰值口径：NVIDIA H200 SXM 官方规格 BFLOAT16 Tensor Core 1,979 TFLOPS（with sparsity），
# dense 取一半 = 989.5 TFLOP/s；硬件型号由功耗/显存/时钟反推（见 mfu.hardware_infer）。
PEAK = 989.5
n1s = n1["stats"]
tpi = 64 * 8192                                   # tokens / iteration
flops_6nd = 6 * params.get("total_B", 56.51) * 1e9 * tpi
tflops_6nd_gpu = flops_6nd / (n1s["iter_ms_stable"]["mean"] / 1000) / 16 / 1e12
def mfu_of(t): return round(t / PEAK * 100, 2)
mfu = {
    "peak_tflops": PEAK,
    "peak_note": "NVIDIA H200 SXM 官方规格：BFLOAT16 Tensor Core 1,979 TFLOPS（标注 with sparsity），dense 减半 = 989.5 TFLOP/s",
    "hardware_infer": [
        ("dmon 平均 / 峰值功耗", f'{round((cmp["node0"]["pwr"] + cmp["node1"]["pwr"]) / 2, 1)} W / 744 W',
         "对应 700 W TDP 的 H200 SXM；B200 为 1000 W，实测远低于此"),
        ("dmon 稳定期显存占用", f'{round(cmp["node1"]["fb"]):,} MB（≈108.2 GiB）',
         "需 ≥141 GB 显存，排除 H100/H800 的 80 GB；H200 141 GB 可容纳"),
        ("内存 / 核心时钟", "mclk 3201 MHz · pclk 1860–1965 MHz", "Hopper 代 HBM3e / boost 频率区间"),
        ("容器与互联", "NVIDIA Release 25.01 · IBext_v8 GDRDMA · NVLink/CUMEM", "与 HGX H200 8-GPU 节点的常见栈一致"),
        ("结论", "H200 SXM（141 GB HBM3e / 700 W）/ 或同代同峰值 Hopper 卡", "BF16 dense 峰值按 989.5 TFLOP/s 计算 MFU"),
    ],
    "stable_mean": mfu_of(n1s["throughput_stable"]["mean"]),
    "stable_min": mfu_of(n1s["throughput_stable"]["min"]),
    "stable_p05": mfu_of(n1s["throughput_stable"]["p05"]),
    "stable_p95": mfu_of(n1s["throughput_stable"]["p95"]),
    "stable_max": mfu_of(n1s["throughput_stable"]["max"]),
    "all_mean": mfu_of(n1s["throughput_all"]["mean"]),
    "first": mfu_of(n1s["throughput_warm"]["min"]),
    "last": mfu_of(n1["train"][-1]["tflops"]),
    "global_pflops": round(n1s["throughput_stable"]["mean"] * 16 / 1000, 2),
    "tokens_per_iter": tpi,
    "flops_6nd_per_iter_pflops": round(flops_6nd / 1e15, 2),
    "flops_6nd_per_gpu_tflops": round(tflops_6nd_gpu, 1),
    "mfu_6nd": mfu_of(tflops_6nd_gpu),
    "gap_vs_6nd_pct": round((n1s["throughput_stable"]["mean"] / tflops_6nd_gpu - 1) * 100, 2),
    "sensitivity": [
        {"basis": "H200 SXM BF16 dense（主口径）", "peak": PEAK, "tflops": n1s["throughput_stable"]["mean"],
         "mfu": mfu_of(n1s["throughput_stable"]["mean"])},
        {"basis": "若峰值按 1,000 TFLOP/s 取整", "peak": 1000, "tflops": n1s["throughput_stable"]["mean"],
         "mfu": round(n1s["throughput_stable"]["mean"] / 1000 * 100, 2)},
        {"basis": "若硬件为 B200（BF16 dense 2,250）", "peak": 2250, "tflops": n1s["throughput_stable"]["mean"],
         "mfu": round(n1s["throughput_stable"]["mean"] / 2250 * 100, 2)},
        {"basis": "6ND 理论模型 FLOPs（不含 attention 等）", "peak": PEAK,
         "tflops": round(tflops_6nd_gpu, 1), "mfu": mfu_of(tflops_6nd_gpu)},
    ],
    "history": [{"file": h["file"], "tflops": h["last_tflops"], "mfu": mfu_of(h["last_tflops"])}
                for h in history if h["last_tflops"]],
}
mfu["hist_range"] = [min(x["mfu"] for x in mfu["history"]), max(x["mfu"] for x in mfu["history"])]

# ---------------- 6. 自检与镜像 ----------------
sc0 = {}
for ln in (N0 / "selfcheck.log").read_text(encoding="utf-8", errors="replace").splitlines():
    if re.match(r"^[A-Z][A-Z_]*=", ln):
        k, v = ln.split("=", 1); sc0[k.strip()] = v.strip()
digest0 = (N0 / "image-digest.txt").read_text(encoding="utf-8").strip()
digest1 = (N1 / "image-digest.txt").read_text(encoding="utf-8").strip()

result = {
    "n1": n1,
    "n0": {"config": config, "params": params, "milestones": milestones, "mem": mem,
           "events": ev, "topo": topo, "categories": dict(cat.most_common()),
           "log_lines": len(lines), "selfcheck": sc0, "digest": digest0},
    "gpu0": node0_gpu, "gpu1": gpu1,
    "dual": dual, "history": history, "hist_summary": hist_summary, "mfu": mfu,
    "meta": {"digest1": digest1, "n0_log": LOG0.name, "n0_csv": CSV0.name,
             "n1_log": n1["meta"]["log_name"], "n1_csv": n1["meta"]["csv_name"],
             "generated": datetime.now().strftime("%Y-%m-%d %H:%M")},
}
(OUT / "stats_dual.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

# ---------------- 摘要 ----------------
print("=== node0 主日志 ===")
print(f"{len(lines)} 行 | 配置参数 {len(conf)} 项 | 里程碑 {len(milestones)} | 显存记录 {len(mem)} ranks")
print("模型:", params)
print("配置:", {k: config.get(k) for k in ("num_layers","hidden_size","ffn_hidden_size","num_attention_heads","seq_length","tensor_model_parallel_size","data_parallel_size","world_size","exit_duration_in_mins","mock_data")})
print("里程碑:", [(m["label"][:42], m["ts"]) for m in milestones])
print("退出分钟:", exit_min, "| 显存 rank0:", mem[0] if mem else None)
print("事件:", ev)
print("拓扑:", {k: v for k, v in topo.items() if k != "pair_sample"}, "| 跨节点对样例:", topo["pair_sample"][:4])
print("\n=== node0 dmon ===")
print(f"{nrows} 行 | {node0_gpu['points']} 时间点 | {node0_gpu['ts_first']} ~ {node0_gpu['ts_last']}")
print("采样间隔:", gaps_s, "| >20s 空洞:", len(holes), holes[:5])
print("pviol:", node0_gpu["pviol_dist"], "| 功率封顶:", node0_gpu["power_cap_pct"], "% | SM>=90:", node0_gpu["sm_ge90_pct"], "%")
print("末尾显存快照:", dict(list(node0_gpu["tail_fb"].items())[-2:]))
print("\n=== 双机对比 ===")
print("node0:", cmp["node0"])
print("node1:", cmp["node1"])
print("差异 (n0-n1):", dual["diff"])
print("采样点差:", dual["sample_gap_points"], "| 16 卡均功耗:", dual["all16_pwr"], "W | 温度范围:", dual["all16_temp_range"])
print("显存: PyTorch max_reserved", mem[0]["max_reserved"] if mem else None, "MB vs dmon fb", node0_gpu["summary"][0]["fb"]["mean"], "MB, 差", dual["reserved_vs_fb_MB"])
print("\n=== MFU ===")
print(f"峰值 {PEAK} TFLOP/s | 稳定期 MFU {mfu['stable_mean']}% "
      f"(区间 {mfu['stable_min']}–{mfu['stable_max']}%) | 全程 {mfu['all_mean']}% | 16 卡 {mfu['global_pflops']} PFLOP/s")
print(f"6ND 校验: {mfu['flops_6nd_per_iter_pflops']} PFLOPs/iter → {mfu['flops_6nd_per_gpu_tflops']} TFLOP/s/GPU "
      f"→ MFU {mfu['mfu_6nd']}%，Megatron 口径高 {mfu['gap_vs_6nd_pct']}%")
print("敏感性:", [(x["basis"], x["mfu"]) for x in mfu["sensitivity"]])
print("历史 MFU 范围:", mfu["hist_range"])
print("\n=== 历史基准 ===")
print(hist_summary)
print("\n输出 ->", OUT / "stats_dual.json")
