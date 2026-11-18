# -*- coding: utf-8 -*-
"""单机 56B 测试统计分析：解析 single_56b .out + dmon CSV，并与双机结果对比。
输出 stats_single.json"""
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(r"C:/Users/26488/Desktop/ht/e2e-megatron-56b-2501/node1-e2e-56b")
OUT = ROOT / "analysis"
LOG = ROOT / "logs" / "single_56b_tp8_pp1_dp1_mbs1_s4096_gbs8_bf16_lowmem1-20261112_234438.out"
CSV = ROOT / "logs" / "single_56b_tp8_pp1_dp1_mbs1_s4096_gbs8_bf16_lowmem1-dmon-20261112_234438.csv"
DUAL = OUT / "stats_dual.json"
T0 = datetime(2026, 11, 12, 15, 44, 39)   # UTC；dmon 首个采样 23:44:39 (UTC+8)


def summarize(vals):
    if not vals:
        return {}
    s = sorted(vals); n = len(s)
    q = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    m = st.fmean(s); sd = st.pstdev(s) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "cv": round(sd / m * 100, 3) if m else 0,
            "min": round(s[0], 4), "p05": round(q(.05), 4), "p50": round(q(.5), 4),
            "p95": round(q(.95), 4), "p99": round(q(.99), 4), "max": round(s[-1], 4)}


def hrs(ts):
    return round((ts - T0).total_seconds() / 3600, 4)


# ---------------- 1. 主日志 ----------------
txt = LOG.read_text(encoding="utf-8", errors="replace")
lines = txt.splitlines()

conf = {}
for ln in lines:
    m = re.match(r"^\s{2}([a-z_][a-z0-9_]*)\s+\.+\s+(.*?)\s*$", ln)
    if m:
        conf[m.group(1)] = m.group(2).strip()
KEY = ["num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups",
       "vocab_size", "seq_length", "max_position_embeddings", "micro_batch_size", "global_batch_size",
       "tensor_model_parallel_size", "pipeline_model_parallel_size", "data_parallel_size", "world_size",
       "train_iters", "train_samples", "exit_duration_in_mins", "eval_interval", "eval_iters", "lr",
       "lr_warmup_iters", "lr_decay_iters", "lr_decay_style", "min_lr", "clip_grad", "optimizer",
       "bf16", "fp8", "mock_data", "attention_backend", "transformer_impl", "sequence_parallel",
       "tp_comm_overlap", "overlap_grad_reduce", "overlap_param_gather", "use_distributed_optimizer",
       "use_precision_aware_optimizer", "recompute_granularity", "recompute_method", "save",
       "save_interval", "log_interval", "distributed_backend", "params_dtype", "manual_gc",
       "inprocess_restart"]
config = {k: conf.get(k) for k in KEY if k in conf}

params = {}
for ln in lines:
    for pat, key in ((r"Number of parameters in transformer block in billions:\s*([\d.]+)", "transformer_B"),
                     (r"Number of parameters in embedding layers in billions:\s*([\d.]+)", "embedding_B"),
                     (r"Total number of parameters in billions:\s*([\d.]+)", "total_B"),
                     (r"Number of parameters in most loaded shard in billions:\s*([\d.]+)", "shard_B")):
        m = re.search(pat, ln)
        if m: params[key] = float(m.group(1))
    m = re.search(r"number of parameters on \(tensor, pipeline\) model parallel rank \((\d+), (\d+)\): (\d+)", ln)
    if m: params.setdefault("per_tp_rank", {})[int(m.group(1))] = int(m.group(3))

milestones = []
for ln in lines:
    m = re.search(r"\[([^\]]+)\] datetime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
    if m:
        ts = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")
        milestones.append({"label": m.group(1), "ts": m.group(2), "t": hrs(ts), "t_min": round(hrs(ts) * 60, 2)})

ITER_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] iteration\s+(\d+)/\s*(\d+)\s*\| consumed samples:\s*(\d+)"
    r"\s*\| elapsed time per iteration \(ms\):\s*([\d.]+)"
    r"\s*\| throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
    r"\s*\| learning rate:\s*([\d.E+-]+)\s*\| global batch size:\s*(\d+)"
    r"\s*\| lm loss:\s*([\d.E+-]+)\s*\| loss scale:\s*([\d.E+-]+)"
    r"\s*\| grad norm:\s*([\d.]+)\s*\| number of skipped iterations:\s*(\d+)"
    r"\s*\| number of nan iterations:\s*(\d+)")
VAL_RE = re.compile(r"validation loss at iteration (\d+)(?: on (validation|test) set)?\s*\| "
                    r"lm loss value:\s*([\d.E+-]+)\s*\|\s*lm loss PPL:\s*([\d.E+-]+)")
EVAL_RE = re.compile(r"evaluate \.+: \(([\d.]+), ([\d.]+)\)")
MEM_RE = re.compile(r"\[Rank (\d+)\] \(after (\d+) iterations\) memory \(MB\) \| allocated: ([\d.]+) "
                    r"\| max allocated: ([\d.]+) \| reserved: ([\d.]+) \| max reserved: ([\d.]+)")

train, vals, eval_ms, mem = [], [], [], []
for ln in lines:
    m = ITER_RE.search(ln)
    if m:
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        train.append({"iter": int(m.group(2)), "total": int(m.group(3)), "samples": int(m.group(4)),
                      "ms": float(m.group(5)), "tflops": float(m.group(6)), "lr": float(m.group(7)),
                      "gbs": int(m.group(8)), "loss": float(m.group(9)), "loss_scale": float(m.group(10)),
                      "gnorm": float(m.group(11)), "skipped": int(m.group(12)), "nan": int(m.group(13)),
                      "t": hrs(ts), "ts": m.group(1)})
    for m in VAL_RE.finditer(ln):
        vals.append({"iter": int(m.group(1)), "split": m.group(2) or "val",
                     "loss": float(m.group(3)), "ppl": float(m.group(4))})
    m = EVAL_RE.search(ln)
    if m: eval_ms.append(float(m.group(2)))
    for m in MEM_RE.finditer(ln):
        mem.append({"rank": int(m.group(1)), "after_iter": int(m.group(2)),
                    "allocated": float(m.group(3)), "max_allocated": float(m.group(4)),
                    "reserved": float(m.group(5)), "max_reserved": float(m.group(6))})

tmap = {d["iter"]: d["t"] for d in train}
for v in vals:
    v["t"] = tmap.get(v["iter"], train[-1]["t"] if train else None)

low = txt.lower()
ev = {
    "traceback": txt.count("Traceback"),
    "oom": len(re.findall(r"torch\.OutOfMemoryError|CUDA out of memory", txt)),
    "cuda_error": len(re.findall(r"cuda error", low)),
    "nccl_timeout": len(re.findall(r"watchdog caught|collective operation timeout", low)),
    "abort_complete": txt.count("Abort COMPLETE"),
    "pg_not_destroyed": txt.count("has NOT been destroyed"),
    "rerun_state_machine_warn": txt.count("megatron.core.rerun_state_machine"),
    "plugin_symbol_missing": txt.count("Failed to find nccl"),
    "resource_warning": txt.count("ResourceWarning"),
    "deprecation": txt.count("DeprecationWarning"),
    "rope_warning": txt.count("fused RoPE"),
    "socket_hostname_warn": txt.count("hostname of the client socket"),
}
ib = [l for l in lines if "via NET/IBext" in l]
p2p = [l for l in lines if "via P2P" in l]
topo = {"ib_channels": len(ib), "p2p_channels": len(p2p), "gdrdma": txt.count("GDRDMA"),
        "abort_by_nranks": dict(Counter(re.findall(r"nranks (\d+)",
                                        "\n".join(l for l in lines if "Abort COMPLETE" in l)))),
        "nccl_debug_info": len([l for l in lines if "NCCL INFO" in l]),
        "nccl_version": (re.search(r"NCCL version ([\d.+\w]+)", txt).group(1)
                         if re.search(r"NCCL version ([\d.+\w]+)", txt) else None)}
cat = Counter()
for ln in lines:
    if ITER_RE.search(ln): cat["训练迭代日志"] += 1
    elif VAL_RE.search(ln): cat["验证日志"] += 1
    elif "NCCL INFO" in ln: cat["NCCL 通信日志"] += 1
    elif "megatron.core.rerun_state_machine" in ln: cat["RerunStateMachine 警告"] += 1
    elif re.match(r"^\s{2}[a-z_]+\s+\.+", ln): cat["超参配置表"] += 1
    elif "memory (MB)" in ln: cat["显存统计"] += 1
    elif "evaluate " in ln or "(min, max) time" in ln or set(ln.strip()) == {"-"}: cat["耗时统计日志"] += 1
    elif "DeprecationWarning" in ln: cat["Python 弃用警告"] += 1
    elif "UserWarning" in ln or "warnings.warn" in ln: cat["框架警告"] += 1
    elif "ResourceWarning" in ln: cat["资源清理警告"] += 1
    elif re.search(r"\[(W|E)\d{4}|\[rank\d+\]:\[W", ln): cat["分布式 Warning"] += 1
    elif ln.strip() == "": cat["空行"] += 1
    else: cat["环境横幅/其他"] += 1

# ---------------- 2. dmon ----------------
COLS = ["date", "time", "gpu", "pwr", "gtemp", "mtemp", "sm", "mem", "enc", "dec", "jpg",
        "ofa", "mclk", "pclk", "pviol", "tviol", "fb", "bar1", "ccpm", "rxpci", "txpci"]
per_gpu = {g: defaultdict(list) for g in range(8)}
B30 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
B1 = defaultdict(lambda: defaultdict(list))
pviol_dist, pclk_dist, sm_dist = Counter(), Counter(), Counter()
ts_by_gpu = defaultdict(list)
pviol_nz = tviol_nz = sm_ge90 = fb_peak = 0
nrows = 0; ts_first = ts_last = None
with CSV.open(encoding="utf-8", errors="replace") as fh:
    for ln in fh:
        if ln.startswith("#"): continue
        p = ln.split()
        if len(p) != 21 or not p[0].isdigit(): continue
        try: dt = datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S") - timedelta(hours=8)
        except ValueError: continue
        g = int(p[2]); v = {c: int(x) for c, x in zip(COLS[3:], p[3:])}
        nrows += 1; t = hrs(dt)
        ts_first = ts_first or dt; ts_last = dt
        ts_by_gpu[g].append(dt)
        b30 = int(t * 2)
        for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "pclk", "fb", "rxpci", "txpci", "pviol", "tviol"):
            per_gpu[g][k].append(v[k]); B30[g][b30][k].append(v[k])
        if t < .5:
            for k in ("fb", "pwr", "pclk", "sm"): B1[int(t * 60)][k].append(v[k])
        pviol_nz += v["pviol"] > 0; tviol_nz += v["tviol"] > 0; sm_ge90 += v["sm"] >= 90
        fb_peak = max(fb_peak, v["fb"])
        pviol_dist[v["pviol"]] += 1; pclk_dist[v["pclk"]] += 1
        sm_dist[min(v["sm"] // 10 * 10, 90)] += 1

gpu_summary = {}
for g in range(8):
    d = per_gpu[g]
    gpu_summary[g] = {"samples": len(d["pwr"]),
                      **{k: summarize(d[k]) for k in ("pwr", "gtemp", "mtemp", "sm", "mem", "fb", "pclk", "rxpci", "txpci")},
                      "pviol_nz": sum(1 for x in d["pviol"] if x > 0),
                      "tviol_nz": sum(1 for x in d["tviol"] if x > 0)}

nb = max(max(B30[g].keys()) for g in range(8)) + 1
buckets = []
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
    buckets.append(rec)

startup = [{"m": b, **{k: round(st.fmean(d[k]), 1) if d[k] else None for k in ("fb", "pwr", "pclk", "sm")}}
           for b, d in sorted(B1.items())]

gaps = [(b - a).total_seconds() for a, b in zip(ts_by_gpu[0], ts_by_gpu[0][1:])]
holes = [(ts_by_gpu[0][i].strftime("%H:%M:%S"), ts_by_gpu[0][i + 1].strftime("%H:%M:%S"), round(g, 1))
         for i, g in enumerate(gaps) if g > 20]

gpu = {
    "rows": nrows, "points": len(ts_by_gpu[0]),
    "ts_first": ts_first.strftime("%Y-%m-%d %H:%M:%S UTC"),
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
    "gap": summarize(gaps), "holes": holes[:12], "holes_n": len(holes),
    "summary": gpu_summary, "buckets": buckets, "startup": startup,
    "csv_lines": sum(1 for _ in CSV.open(encoding="utf-8", errors="replace")),
}

# ---------------- 3. 派生统计 ----------------
stable = [d for d in train if d["iter"] >= 500]
warm = [d for d in train if d["iter"] < 500]
PEAK = 989.5
mfu_of = lambda t: round(t / PEAK * 100, 2)
tpi = int(config.get("global_batch_size", 8)) * int(config.get("seq_length", 4096))
flops_6nd = 6 * params.get("total_B", 56.51) * 1e9 * tpi
ms_stable = st.fmean([d["ms"] for d in stable])
tflops_6nd_gpu = flops_6nd / (ms_stable / 1000) / 8 / 1e12

derived = {
    "throughput_all": summarize([d["tflops"] for d in train]),
    "throughput_stable": summarize([d["tflops"] for d in stable]),
    "throughput_warm": summarize([d["tflops"] for d in warm]),
    "iter_ms_all": summarize([d["ms"] for d in train]),
    "iter_ms_stable": summarize([d["ms"] for d in stable]),
    "gnorm": summarize([d["gnorm"] for d in train]),
    "eval_ms": summarize(eval_ms),
    "train_loss_first": train[0]["loss"], "train_loss_last": train[-1]["loss"],
    "val_loss_first": vals[0]["loss"], "val_loss_last": vals[-3]["loss"] if len(vals) >= 3 else vals[-1]["loss"],
    "ppl_first": vals[0]["ppl"], "ppl_last": vals[-3]["ppl"] if len(vals) >= 3 else vals[-1]["ppl"],
    "final_val_loss": next((v["loss"] for v in vals if v["split"] == "validation"), None),
    "final_test_loss": next((v["loss"] for v in vals if v["split"] == "test"), None),
    "final_val_ppl": next((v["ppl"] for v in vals if v["split"] == "validation"), None),
    "final_test_ppl": next((v["ppl"] for v in vals if v["split"] == "test"), None),
    "skipped_total": sum(d["skipped"] for d in train), "nan_total": sum(d["nan"] for d in train),
    "samples_total": train[-1]["samples"], "tokens_total": train[-1]["samples"] * int(config.get("seq_length", 4096)),
    "hours": round(train[-1]["t"], 3),
    "iter_last": train[-1]["iter"], "iter_target": train[-1]["total"],
    "log_iter_last": train[-1]["iter"], "completed_iters": train[-1]["total"],
    "completed": True,
    "completion_evidence": "[after training is done] + 最终 validation set / test set 评估（日志每 10 iter 打印，末条记录为 20780）",
    "progress": 100.0,
    "avg_tflops_node": round(st.fmean([d["tflops"] for d in stable]) * 8, 1),
    "startup_min": round(train[0]["t"] * 60, 1),
    "mfu_stable": mfu_of(st.fmean([d["tflops"] for d in stable])),
    "mfu_all": mfu_of(st.fmean([d["tflops"] for d in train])),
    "mfu_first": mfu_of(train[0]["tflops"]),
    "mfu_max": mfu_of(max(d["tflops"] for d in train)),
    "mfu_6nd": mfu_of(tflops_6nd_gpu),
    "flops_6nd_per_iter_pflops": round(flops_6nd / 1e15, 2),
    "flops_6nd_per_gpu_tflops": round(tflops_6nd_gpu, 1),
    "gap_vs_6nd_pct": round((st.fmean([d["tflops"] for d in stable]) / tflops_6nd_gpu - 1) * 100, 2),
    "tokens_per_iter": tpi,
    "grad_accum": int(config.get("global_batch_size", 8)) // (int(config.get("micro_batch_size", 1)) * int(config.get("data_parallel_size", 1))),
    "peak_tflops": PEAK,
}
for thr in (1.0, 0.1, 0.01, 0.005):
    derived[f"iter_loss_below_{thr}"] = next((d["iter"] for d in train if d["loss"] < thr), None)

ms_mean, ms_sd = derived["iter_ms_stable"]["mean"], derived["iter_ms_stable"]["sd"]
gn_mean, gn_sd = derived["gnorm"]["mean"], derived["gnorm"]["sd"]
derived["slow_iters"] = [{"iter": d["iter"], "ms": d["ms"]} for d in stable if d["ms"] > ms_mean + 3 * ms_sd]
derived["gnorm_spikes"] = [{"iter": d["iter"], "gnorm": d["gnorm"]} for d in train if d["gnorm"] > gn_mean + 3 * gn_sd]
derived["gnorm_gt8"] = [{"iter": d["iter"], "gnorm": d["gnorm"], "loss": d["loss"]}
                        for d in train if d["gnorm"] > 8]
reg = [v for v in vals if v["split"] == "val"]
derived["val_rebounds"] = [{"iter": reg[i]["iter"], "from": reg[i-1]["loss"], "to": reg[i]["loss"],
                            "pct": round((reg[i]["loss"] / reg[i-1]["loss"] - 1) * 100, 1)}
                           for i in range(1, len(reg)) if reg[i]["loss"] > reg[i-1]["loss"] * 1.3]
even = [gpu_summary[g]["gtemp"]["mean"] for g in (0, 2, 4, 6)]
odd = [gpu_summary[g]["gtemp"]["mean"] for g in (1, 3, 5, 7)]
derived["gtemp_even_avg"] = round(st.fmean(even), 2)
derived["gtemp_odd_avg"] = round(st.fmean(odd), 2)
derived["fb_stable_min"] = next((x["m"] for x in startup if x["fb"] and x["fb"] > 130000), None)
derived["reserved_vs_fb_MB"] = round(gpu_summary[0]["fb"]["mean"] - mem[0]["max_reserved"], 1) if mem else None
derived["gtemp_max"] = max(gpu_summary[g]["gtemp"]["max"] for g in range(8))
derived["pwr_max"] = max(gpu_summary[g]["pwr"]["max"] for g in range(8))
derived["fb_card_spread_MB"] = round(max(gpu_summary[g]["fb"]["mean"] for g in range(8))
                                     - min(gpu_summary[g]["fb"]["mean"] for g in range(8)), 2)
derived["pwr_avg"] = round(st.fmean([gpu_summary[g]["pwr"]["mean"] for g in range(8)]), 2)
derived["gtemp_avg"] = round(st.fmean([gpu_summary[g]["gtemp"]["mean"] for g in range(8)]), 2)

# ---------------- 4. 与双机对比 ----------------
d2 = json.loads(DUAL.read_text(encoding="utf-8"))
d2s, d2m, d2c = d2["n1"]["stats"], d2["mfu"], d2["dual"]["cmp"]["node1"]
comparison = {
    "config": [
        ("节点数 / 总 ranks", "1 / 8", "2 / 16"),
        ("TP / PP / DP", "8 / 1 / 1", "8 / 1 / 2"),
        ("序列长度", config.get("seq_length"), d2["n0"]["config"].get("seq_length")),
        ("全局批次 GBS", config.get("global_batch_size"), d2["n0"]["config"].get("global_batch_size")),
        ("每 iter tokens", f'{tpi:,}', f'{d2m["tokens_per_iter"]:,}'),
        ("梯度累积步数", derived["grad_accum"], d2["dual"]["grad_accum"]),
        ("train_iters", config.get("train_iters"), d2["n0"]["config"].get("train_iters")),
        ("退出时长上限 (min)", config.get("exit_duration_in_mins"), d2["n0"]["config"].get("exit_duration_in_mins")),
        ("recompute_granularity", config.get("recompute_granularity"), d2["n0"]["config"].get("recompute_granularity")),
        ("use_precision_aware_optimizer", config.get("use_precision_aware_optimizer"), d2["n0"]["config"].get("use_precision_aware_optimizer")),
        ("overlap_grad_reduce", config.get("overlap_grad_reduce"), d2["n0"]["config"].get("overlap_grad_reduce")),
        ("结束方式", "跑满 20,786 iter（含最终 val/test 评估）", "1410 min 定时退出，仅完成 3,680 iter"),
    ],
    "metrics": [
        ("运行时长 (h)", derived["hours"], d2s["hours"]),
        ("完成迭代", f'{derived["completed_iters"]:,}', f'{d2s["iter_last"]:,}'),
        ("稳定期吞吐 (TFLOP/s/GPU)", derived["throughput_stable"]["mean"], d2s["throughput_stable"]["mean"]),
        ("稳定期 MFU (%)", derived["mfu_stable"], d2m["stable_mean"]),
        ("迭代耗时 (s)", round(derived["iter_ms_stable"]["mean"] / 1000, 3), round(d2s["iter_ms_stable"]["mean"] / 1000, 3)),
        ("吞吐 CV (%)", derived["throughput_stable"]["cv"], d2s["throughput_stable"]["cv"]),
        ("最终 train loss", derived["train_loss_last"], d2s["train_loss_last"]),
        ("最终 val PPL", derived["final_val_ppl"], d2s["ppl_last"]),
        ("稳定期显存 (MB/卡)", gpu_summary[0]["fb"]["p50"], d2["gpu1"]["summary"]["0"]["fb"]["p50"]),
        ("PyTorch max_reserved (MB)", mem[0]["max_reserved"] if mem else None, d2["n0"]["mem"][0]["max_reserved"]),
        ("平均功耗 (W)", derived["pwr_avg"], d2c["pwr"]),
        ("平均温度 (°C)", derived["gtemp_avg"], d2c["gtemp"]),
        ("pviol=100 占比 (%)", gpu["power_cap_pct"], d2["gpu1"]["power_cap_pct"]),
        ("SM≥90% 占比 (%)", gpu["sm_ge90_pct"], d2["gpu1"]["sm_ge90_pct"]),
        ("GPU 采样点", gpu["rows"], d2["gpu1"]["rows"]),
    ],
}
comparison["throughput_ratio"] = round(derived["throughput_stable"]["mean"] / d2s["throughput_stable"]["mean"] * 100, 1)
comparison["mfu_gap_pp"] = round(derived["mfu_stable"] - d2m["stable_mean"], 2)
comparison["fb_gap_MB"] = round(gpu_summary[0]["fb"]["p50"] - d2["gpu1"]["summary"]["0"]["fb"]["p50"], 0)
comparison["reserved_gap_MB"] = round(mem[0]["max_reserved"] - d2["n0"]["mem"][0]["max_reserved"], 0) if mem else None

result = {
    "meta": {"log": LOG.name, "csv": CSV.name, "log_lines": len(lines),
             "config_total": len(conf),
             "case": "单机 56B / TP8 PP1 DP1 / MBS1 / SEQ4096 / GBS8 / BF16 / lowmem1",
             "generated": datetime.now().strftime("%Y-%m-%d %H:%M")},
    "config": config, "params": params, "milestones": milestones, "mem": mem,
    "events": ev, "topo": topo, "categories": dict(cat.most_common()),
    "train": train, "val": vals, "eval_ms": eval_ms, "gpu": gpu,
    "stats": derived, "comparison": comparison,
    "dual_peak": d2m["peak_tflops"],
    # 按「各自测试运行后的小时数」对齐的对比序列（单机降采样到 ~416 点）
    "series": {
        "single_tflops": [[d["t"], d["tflops"]] for d in train[::5] + [train[-1]]],
        "dual_tflops": [[d["t"], d["tflops"]] for d in d2["n1"]["train"]],
        "single_loss": [[d["t"], d["loss"]] for d in train[::5] + [train[-1]]],
        "dual_loss": [[d["t"], d["loss"]] for d in d2["n1"]["train"]],
        "single_fb": [[b["t"], b["fb"]] for b in buckets],
        "dual_fb": [[b["t"], b["fb"]] for b in d2["gpu1"]["buckets"]],
    },
}
(OUT / "stats_single.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

# ---------------- 摘要 ----------------
print("=== 单机主日志 ===")
print(f"{len(lines)} 行 | 训练点 {len(train)} | 验证 {len(vals)} | evaluate 计时 {len(eval_ms)} | 显存 rank {len(mem)}")
print("配置:", {k: config.get(k) for k in ("tensor_model_parallel_size", "data_parallel_size", "world_size", "seq_length", "global_batch_size", "train_iters", "exit_duration_in_mins", "recompute_granularity", "use_precision_aware_optimizer")})
print("里程碑:", [(m["label"][:38], m["ts"]) for m in milestones])
print("参数量:", params)
print("显存 rank0:", mem[0] if mem else None)
print("事件:", ev)
print("拓扑:", topo)
print("\n=== 训练 ===")
for k in ("hours", "iter_last", "iter_target", "progress", "train_loss_first", "train_loss_last",
          "val_loss_first", "final_val_loss", "final_test_loss", "final_val_ppl", "final_test_ppl",
          "skipped_total", "nan_total", "startup_min", "tokens_per_iter", "grad_accum"):
    print(f"  {k}: {derived[k]}")
print("  吞吐稳定:", derived["throughput_stable"])
print("  步时稳定:", derived["iter_ms_stable"])
print("  MFU:", derived["mfu_stable"], "% | 6ND:", derived["mfu_6nd"], "% | 差", derived["gap_vs_6nd_pct"], "%")
print("  里程碑:", [derived[f"iter_loss_below_{t}"] for t in (1.0, 0.1, 0.01, 0.005)])
print("  grad>8:", derived["gnorm_gt8"][:8], "| 慢步:", len(derived["slow_iters"]), "| 验证反弹:", len(derived["val_rebounds"]))
print("\n=== GPU ===")
print(f"  {gpu['rows']} 行 / {gpu['points']} 时间点 | {gpu['ts_first']} ~ {gpu['ts_last']}")
print("  pviol:", gpu["pviol_dist"], "| 封顶占比", gpu["power_cap_pct"], "% | tviol", gpu["tviol_nz"], "| SM>=90", gpu["sm_ge90_pct"], "%")
print("  显存峰值", gpu["fb_peak"], "MB | 偶奇卡均温", derived["gtemp_even_avg"], "/", derived["gtemp_odd_avg"])
print("  采样间隔", gpu["gap"], "| 空洞", gpu["holes_n"])
for g in range(8):
    s = gpu_summary[g]
    print(f"  GPU{g}: pwr {s['pwr']['mean']}W gtemp {s['gtemp']['mean']}C sm {s['sm']['mean']}% fb {s['fb']['mean']}MB pclk {s['pclk']['mean']}MHz")
print("\n=== 双机对比 ===")
print("  吞吐比:", comparison["throughput_ratio"], "% | MFU 差:", comparison["mfu_gap_pp"], "pp | 显存差:", comparison["fb_gap_MB"], "MB | reserved 差:", comparison["reserved_gap_MB"], "MB")
for r in comparison["metrics"]:
    print("   ", r)
print("\n输出 ->", OUT / "stats_single.json")
