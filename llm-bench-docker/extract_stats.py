"""Extract every chartable number from logs/ into one JSON payload."""
import argparse, json, re, statistics
from pathlib import Path

LOGDIR = Path(__file__).resolve().parent / "logs"
PEAK_BF16 = 989.4          # H200 theoretical dense bf16 TFLOP/s
PEAK_FP8 = 1978.8          # H200 theoretical dense fp8 TFLOP/s
WARM = 5                  # iterations dropped before averaging (log shows warmup ramp)
N_PARAMS = {"8B": 8.03e9, "70B": 70.55e9}

RE_TAG = re.compile(
    r"^w(?P<world>\d+)_(?P<model>8B|70B)_tp(?P<tp>\d+)_pp(?P<pp>\d+)"
    r"_mbs(?P<mbs>\d+)_s(?P<seq>\d+)_gb(?P<gbs>\d+)_(?P<prec>\w+)")
PAT = {
    "ms": re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)"),
    "tf": re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"),
    "loss": re.compile(r"(?:lm|label) loss:\s*([0-9.eE+-]+)"),
    "gn": re.compile(r"grad norm:\s*([0-9.eE+-]+)"),
    "nan": re.compile(r"number of nan iterations:\s*(\d+)"),
}
PAT_TMEM = re.compile(r"Theoretical memory footprints: weight and optimizer=([0-9.]+) MB, "
                      r"activation=([0-9.]+) MB, total=([0-9.]+) MB")
PAT_RANK = re.compile(r"memory \(MB\) \| allocated: ([0-9.]+) \| max allocated: ([0-9.]+) "
                      r"\| reserved: ([0-9.]+) \| max reserved: ([0-9.]+)")


def mean(xs):
    return statistics.mean(xs) if xs else None


def perf_from(txt):
    its = [l for l in txt.splitlines() if "elapsed time per iteration" in l]
    return {
        "its": its,
        "ms": [float(PAT["ms"].search(l).group(1)) for l in its],
        "tf": [float(PAT["tf"].search(l).group(1)) for l in its if PAT["tf"].search(l)],
        "gn": [float(PAT["gn"].search(l).group(1)) for l in its if PAT["gn"].search(l)],
        "loss": [float(PAT["loss"].search(l).group(1)) for l in its if PAT["loss"].search(l)],
        "nan": sum(int(PAT["nan"].search(l).group(1)) for l in its if PAT["nan"].search(l)),
    }


def run_row(path, peer_dir=None):
    m = RE_TAG.match(path.stem)
    if not m:
        return None
    g = m.groupdict()
    txt = path.read_text(encoding="utf-8", errors="ignore")
    p = perf_from(txt)
    src = "local"
    # PP>1 时 iteration 行只打在流水线末级所在的那台，本机日志没有 → 从对端同名日志取。
    if len(p["ms"]) <= WARM and peer_dir:
        pp = peer_dir / path.name
        if pp.exists():
            p = perf_from(pp.read_text(encoding="utf-8", errors="ignore"))
            src = "peer"
        else:
            src = "absent"

    ms, tf, gn, loss, nans = p["ms"], p["tf"], p["gn"], p["loss"], p["nan"]
    seq, gbs, world, mbs = int(g["seq"]), int(g["gbs"]), int(g["world"]), int(g["mbs"])
    n = N_PARAMS[g["model"]]
    peak = PEAK_FP8 if g["prec"] == "fp8" else PEAK_BF16

    row = {
        "tag": path.stem,
        "file": path.name, "model": g["model"], "world": world,
        "tp": int(g["tp"]), "pp": int(g["pp"]), "mbs": mbs, "seq": seq, "gbs": gbs,
        "prec": g["prec"], "iters": len(ms), "has_perf": len(ms) > WARM, "perf_src": src,
    }
    if row["has_perf"]:
        tail = ms[WARM:]
        step = mean(tail) / 1000.0
        row.update(
            step_s=round(step, 4),
            step_first=round(ms[0] / 1000.0, 4),
            step_min=round(min(tail) / 1000.0, 4),
            step_max=round(max(tail) / 1000.0, 4),
            jitter_pct=round(statistics.pstdev(tail) / mean(tail) * 100, 2),
            tflops=round(mean(tf[WARM:]), 1),
            mfu_peak=round(mean(tf[WARM:]) / peak * 100, 2),
            mfu_6n=round(6 * n * gbs * seq / (step * world * PEAK_BF16 * 1e12) * 100, 2),
            tps=round(gbs * seq / world / step, 1),
            sample_s=round(step * gbs / world, 3),
            grad_norm=round(mean(gn[WARM:]), 2),
            loss_first=loss[0] if loss else None,
            loss_last=loss[-1] if loss else None,
            nan_iters=nans,
        )
    th = PAT_TMEM.search(txt)
    ranks = [PAT_RANK.search(l) for l in txt.splitlines() if "memory (MB)" in l]
    ranks = [r.groups() for r in ranks if r]
    if th:
        row.update(mem_static_mb=round(float(th.group(1)), 0),
                   mem_act_mb=round(float(th.group(2)), 0),
                   mem_total_mb=round(float(th.group(3)), 0))
    if ranks:
        row.update(mem_alloc_mb=round(mean([float(r[0]) for r in ranks]), 0),
                   mem_resv_mb=round(mean([float(r[3]) for r in ranks]), 0),
                   mem_ranks=len(ranks))
    if "after training is done" in txt:
        row["reached_end"] = True
    return row


def main():
    ap = argparse.ArgumentParser(description="把 logs/ 里的 Megatron 压测日志抽成图表数据")
    ap.add_argument("--logs", default=str(LOGDIR), help="本机日志目录（默认 ./logs）")
    ap.add_argument("--peer", default=None,
                    help="node_rank=1 那台拷回来的同名日志目录，只用来补 70B 的 iteration 行")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "stats_payload.json"))
    a = ap.parse_args()
    logdir, peer = Path(a.logs), (Path(a.peer) if a.peer else None)
    logs = sorted(logdir.glob("w*_*.log"))
    runs = [r for r in (run_row(p, peer) for p in logs) if r]

    # results.jsonl -> attempt history per tag (first/last status, retries)
    attempts = {}
    for line in (logdir / "results.jsonl").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line.split("\t", 1)[1] if "\t" in line else line)
        attempts.setdefault(d["tag"], []).append(d["status"])
    history = [{"tag": t, "n": len(s),
                "ok": s.count("OK"), "failed": s.count("FAILED"),
                "first": s[0], "last": s[-1]} for t, s in attempts.items()]

    hwf = logdir / "hwcheck.txt"
    if not hwf.exists():          # --peer/--logs 指到对端时，硬件基线仍取本机那份
        hwf = LOGDIR / "hwcheck.txt"
    hw = hwf.read_text(encoding="utf-8", errors="ignore")
    gemm = [int(x) for x in re.search(r"bf16 GEMM \[([0-9, ]+)\]", hw).group(1).split(",")]
    hbm_line = [l for l in hw.splitlines() if "HBM" in l][0]
    hb = re.findall(r"([0-9.]+) TB/s", hbm_line)
    nccl = re.findall(r"\s*(\d+) MB  busbw\s+([0-9.]+) GB/s", hw)

    payload = {
        "meta": {
            "gpu": "NVIDIA H200 141 GiB x8/node (sm_90, NVLink NV18)",
            "image": "nvcr.io/nvidia/pytorch:25.06 (torch 2.8.0a0, CUDA 12.9, NCCL 2.27.3)",
            "te": "2.4.0+3cd6870", "megatron": "core 0.13.1",
            "peak_tflops": {"bf16": PEAK_BF16, "fp8": PEAK_FP8,
                            "bf16_measured": round(mean(gemm), 1)},
            "steps": 15, "warmup_dropped": WARM,
            "flop_model": "Megatron reported throughput / theoretical dense peak",
        },
        "runs": runs,
        "history": history,
        "hw": {
            "gemm_per_card": gemm, "gemm_mean": round(mean(gemm), 1),
            "gemm_pct_of_peak": round(mean(gemm) / PEAK_BF16 * 100, 1),
            "gemm_spread_pct": round((max(gemm) - min(gemm)) / mean(gemm) * 100, 1),
            "hbm_read_tb": float(hb[0]), "hbm_rw_tb": float(hb[1]),
            "hbm_theory_tb": 4.8,
            "nccl_busbw": [{"size": f"{s} MB", "gbps": float(b)} for s, b in nccl],
        },
    }
    out = Path(a.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    perf = [r for r in runs if r["has_perf"]]
    from_peer = [r for r in perf if r["perf_src"] == "peer"]
    print(f"runs={len(runs)} with_perf={len(perf)} (from_peer={len(from_peer)}) "
          f"history={len(history)} -> {out}")
    for r in from_peer:
        print(f"  +PEER  {r['tag']}  {r['mfu_peak']}% MFU  {r['tps']} tok/s/GPU")
    for r in runs:
        if not r["has_perf"]:
            why = {"absent": "peer has no file with this name",
                   "peer": "peer log has no iteration lines"}.get(
                r["perf_src"], "no --peer given" if peer is None else "?")
            print(f"  NO-PERF {r['tag']}  mem_alloc={r.get('mem_alloc_mb')}MB "
                  f"resv={r.get('mem_resv_mb')}MB end={r.get('reached_end', False)}  [{why}]")


if __name__ == "__main__":
    main()
