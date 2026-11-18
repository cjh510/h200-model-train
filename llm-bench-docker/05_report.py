#!/usr/bin/env python3
# =============================================================================
# [在哪跑] 目标裸机（04_bench.sh --phase report 会调它）。
# 把 Megatron 日志压成带判定的验收表，并对齐两台机器的体检结果。
#
#   python 05_report.py --logs /root/llm-bench/logs
#   python 05_report.py --compare node-a-env.json node-b-env.json
# =============================================================================
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Windows 控制台默认 GBK，表里的 † → ² 之类的字符会直接抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PEAK = 989.4          # H200 bf16 稠密峰值 TFLOPS
N_PARAMS = {"8B": 8.03e9, "70B": 70.55e9}      # 一手：meta-llama 公开 config 逐项算得

RE_TAG = re.compile(
    r"^w(?P<world>\d+)_(?P<model>\d+B)_(?P<rest>.*)?tp(?P<tp>\d+)_pp(?P<pp>\d+)"
    r"_mbs(?P<mbs>\d+)_s(?P<seq>\d+)_gb(?P<gbs>\d+)_(?P<prec>\w+)", re.I)
NUM = r"([0-9][0-9.eE+-]*)"
PAT = {
    "ms": re.compile(rf"elapsed time per iteration \(ms\):\s*{NUM}"),
    "tf": re.compile(rf"throughput per GPU \(TFLOP/s/GPU\):\s*{NUM}"),
    "gbs": re.compile(rf"global batch size:\s*([0-9]+)"),
    "loss": re.compile(rf"(?:lm loss|label loss):\s*{NUM}"),
}
WARMUP = 5           # 前几步含 kernel 编译与 NCCL 建链，不计入均值


def parse_log(path: Path):
    m = RE_TAG.match(path.stem)
    if not m:
        return None
    f = m.groupdict()
    text = path.read_text(errors="ignore")
    rows = [ln for ln in text.splitlines() if "elapsed time per iteration" in ln]
    if not rows:
        return None
    samples = []
    for ln in rows:
        s = {k: float(p.search(ln).group(1)) for k, p in PAT.items() if p.search(ln)}
        if "ms" in s:
            samples.append(s)
    keep = samples[WARMUP:] or samples[-3:]
    # 取中位数不取均值：FP8 每步重算缩放因子（fp8_interval=1）会偶发单步 1.6× 尖峰，
    # 均值被拉高、抖动虚大，中位数才是稳态。
    ms = statistics.median(s["ms"] for s in keep)
    gbs = int(keep[0].get("gbs", int(f["gbs"])))
    world, seq = int(f["world"]), int(f["seq"])
    n = N_PARAMS.get(f["model"].upper(), 0)
    tflops_6nd = 6 * n * gbs * seq / (ms / 1e3) / world / 1e12
    tf_reported = (statistics.median(s["tf"] for s in keep)
                   if all("tf" in s for s in keep) else None)
    jit = (statistics.stdev([s["ms"] for s in keep]) / ms * 100) if len(keep) > 1 else 0.0
    return {
        "tag": path.stem, "model": f["model"].upper(), "world": world,
        "tp": int(f["tp"]), "pp": int(f["pp"]), "mbs": int(f["mbs"]),
        "seq": seq, "gbs": gbs, "prec": f["prec"], "dp": world // (int(f["tp"]) * int(f["pp"])),
        "step_s": ms / 1e3, "jitter": jit,
        "tok_s_gpu": gbs * seq / (ms / 1e3) / world,
        "tflops_6nd": tflops_6nd, "tflops_megatron": tf_reported,
        "mfu": tflops_6nd / PEAK * 100,
        "mfu_megatron": (tf_reported / PEAK * 100) if tf_reported else None,
        # 一个 step 的 FLOPs = 6·N·(GBS×seq)，摊到 world 张卡；
        # 不是 6·N·mbs·seq —— 那样只算了单个 micro-batch，会漏掉 num_micro_batches 倍。
        "t_min": 6 * n * gbs * seq / (world * PEAK * 1e12),
        "loss": keep[-1].get("loss"),
    }


def dmon(logs: Path):
    f = logs / "dmon.csv"
    if not f.exists():
        return None
    hdr, rows = None, []
    for ln in f.read_text(errors="ignore").splitlines():
        if ln.startswith("#"):
            hdr = ln.lstrip("#").split()
        elif hdr and ln.strip():
            rows.append(ln.split())
    out = []
    for name, unit in (("pwr", "W"), ("gput", "°C"), ("totm", "°C")):
        try:
            i = hdr.index(name)
        except (ValueError, AttributeError):
            continue
        vals = [float(r[i]) for r in rows if len(r) > i and re.fullmatch(r"[\d.]+", r[i] or "")]
        if vals:
            out.append(f"{name} avg {statistics.mean(vals):.0f} / max {max(vals):.0f} {unit}")
    return "  |  ".join(out) if out else None


def bench(logs: Path):
    logs = Path(logs)
    done = [r for r in (parse_log(p) for p in sorted(logs.glob("*.log"))) if r]
    status = {}
    if (logs / "results.jsonl").exists():
        for ln in (logs / "results.jsonl").read_text().splitlines():
            try:
                j = json.loads(ln)
            except json.JSONDecodeError:
                continue
            status[j["tag"]] = j.get("status", "?")
    failed = {k: v for k, v in status.items() if v in ("OOM", "FAILED")}
    if not done:
        print("logs/ 里没有可解析的成功结果。先看 04_bench.sh 的输出和 *.log 末尾。")
        return 1

    print(f"\n## 训练压测结果（{len(done)} 组成功 / {len(failed)} 组失败）\n")
    print("| 配置 | world | tp/pp/dp | mbs | seq | step s | tok/s/GPU | "
          "TFLOP/s(6N) | MFU(6N) | MFU† | t_min | 抖动 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(done, key=lambda x: (x["model"], x["world"], -x["seq"])):
        mm = f"**{r['mfu_megatron']:.1f}%**" if r["mfu_megatron"] else "—"
        print(f"| {r['prec']} {r['tag'].split('_', 1)[1]} | {r['world']} "
              f"| {r['tp']}/{r['pp']}/{r['dp']} | {r['mbs']} | {r['seq']:,} "
              f"| {r['step_s']:.3f} | {r['tok_s_gpu']:,.0f} | {r['tflops_6nd']:.1f} "
              f"| {r['mfu']:.1f}% | {mm} | {r['t_min']:.3f} | {r['jitter']:.1f}% |")

    print("\n`MFU(6N)` = 6·N·tokens，只算线性层，长序列下漏掉 attention 的 O(s²)，是保守下界；"
          "`MFU†` 用 Megatron 自己按完整公式（含 attention）算的 throughput 除以 989.4 TFLOPS —— "
          "两者比值就是 attention 占比，seq 越长差得越多，跨 seq 比较只看 MFU†。")

    if failed:
        print("\n### 失败 / 溢出\n")
        for t, s in sorted(failed.items()):
            print(f"- `{t}` → **{s}**")

    # ---- 三条最有用的对比
    print("\n### 关键对比\n")
    by = {(r["tag"].split("mbs")[0], r["world"], r["seq"], r["prec"]): r for r in done}

    def one(model, world, tp, seq, mbs=None, gbs=None):
        # GBS 必须一起锁定：单机那组里 mbs 相同但 GBS 不同的两条会串味，
        # 跨机比值会算错一倍以上，而这条比值正是 D1 的判据。
        c = [r for r in done if r["model"] == model and r["world"] == world
             and r["tp"] == tp and r["seq"] == seq
             and (mbs is None or r["mbs"] == mbs) and (gbs is None or r["gbs"] == gbs)]
        if len(c) > 1:
            print(f"   ! {model} world={world} 匹配到 {len(c)} 组，取第一组 —— 过滤条件不够唯一")
        return c[0] if c else None

    s8 = one("8B", 8, 8, 4096, 2, 32)
    d8 = one("8B", 16, 8, 4096, 2, 32)
    if s8 and d8:
        ratio = d8["step_s"] / s8["step_s"]
        print(f"**D1 机间门禁（8B，TP8，GBS=32 seq=4096 两边完全同配置）**："
              f"单机 {s8['step_s']:.3f}s → 双机 {d8['step_s']:.3f}s，比值 **{ratio:.2f}**")
        print(f"  - 比值 ≈1.0-1.3：机间带宽够，数据并行成立（卡数翻倍、step 时间几乎不变 = 吞吐翻倍）")
        print(f"  - 比值 >2.0：跨机 all-reduce 是瓶颈 → 70B 只能用 TP8×PP2（跨机只传激活），"
              f"且报告里要写明『这台环境不支持跨机数据并行』")
        print(f"  - 比值 ≈2.0 就是**白加一台机器**：卡多一倍，吞吐一点没涨\n")

    pp = sorted([r for r in done if r["model"] == "70B" and r["pp"] > 1], key=lambda x: x["gbs"])
    if len(pp) >= 2:
        base = pp[0]
        print("**D2 流水线气泡（70B TP8×PP2，GBS 即 micro-batch 数）**")
        for r in pp:
            bubble = (r["pp"] - 1) / (r["gbs"] / (r["mbs"] * r["dp"])) * 100
            print(f"  GBS={r['gbs']:>3}  step {r['step_s']:.3f}s  MFU {r['mfu']:.1f}%  "
                  f"理论气泡 ≈{bubble:.0f}%"
                  + ("  ← 基准" if r is base else f"  相对 m={base['gbs']} {base['step_s']/r['step_s']:.2f}×"))
        best = max(pp, key=lambda x: x["mfu"])
        print(f"  最优 micro-batch 数：GBS={best['gbs']}（继续加大收益趋平就说明已非气泡主导）\n")

    seqs = sorted([r for r in done if r["model"] == "8B" and r["world"] == 8 and r["mbs"] == 1],
                  key=lambda x: x["seq"])
    if len(seqs) >= 2:
        print("**S3 长序列（8B TP8 mbs1）**")
        for r in seqs:
            mm = f"  MFU† {r['mfu_megatron']:5.1f}%" if r["mfu_megatron"] else ""
            print(f"  seq {r['seq']:>7,}  step {r['step_s']:>7.3f}s  "
                  f"tok/s/GPU {r['tok_s_gpu']:>8,.0f}  6N-MFU {r['mfu']:>5.1f}%{mm}"
                  + ("  ← OOM 前最后一档" if str(r['tag']) in failed else ""))
        print("  6N-MFU 随 seq 下降是公式漏了 attention 平方项，不是硬件退化；"
              "看 MFU† 是否平——平就说明算力一直是满的。真正的问题是 step 时间随 seq 超线性增长。")

    print("### 判定\n")
    # 判定用 MFU†（含 attention 的完整 FLOPs）：按 6N 排会把 128K 那种 attention 主导的
    # 档位错判成"低效"，而它其实是同一台机器上真实算力最足的负载。
    eff = lambda r: r["mfu_megatron"] or r["mfu"]
    for r in sorted(done, key=lambda x: -eff(x)):
        v = "✓" if eff(r) >= 30 else ("!" if eff(r) >= 20 else "✗")
        print(f"  {v} {r['tag']:<52} MFU† {eff(r):5.1f}%  (6N {r['mfu']:5.1f}%)")
    top = max(done, key=eff)
    print(f"\n最好成绩：{top['tag']} → **{eff(top):.1f}% MFU†**")
    print("- Megatron+TE 跑 dense Llama 在 H200 上的正常水位：**35-45%**（单机 8B）；"
          "70B 双机 TP8×PP2 **25-38%**（含流水线气泡）。")
    print("- **低于 20% 就别调参了**：回去查 hwcheck 的 allreduce busbw 和 nvidia-smi 的 power.limit。")
    print("- 抖动 >5% 的组：有卡降频或 NUMA 跨节点，配合 dmon 看功耗/时钟。")
    d = dmon(Path(logs))
    if d:
        print(f"\n长稳遥测：{d}")
        print("  确认没触发节流：nvidia-smi -q -d PERFORMANCE | grep -A8 'Clocks Event Reasons'")
    print("\n**报告必须写明**：mock 数据 + 随机初始化，不含数据管线开销和 checkpoint 落盘开销；"
          "这两项都会让真实训练更慢，所以本表是算力/通信上限，不是端到端吞吐。")
    return 0


def compare(a: str, b: str):
    A, B = json.loads(Path(a).read_text(encoding="utf-8")), json.loads(Path(b).read_text(encoding="utf-8"))
    ia = {x["item"]: x for x in A.get("items", [])}
    ib = {x["item"]: x for x in B.get("items", [])}
    keys = [k for k in ia if k in ib and ia[k]["detail"] != ib[k]["detail"]]
    print(f"\n{A.get('host')} vs {B.get('host')}")
    crit = {"驱动", "镜像建议", "GPU 数量", "显存容量一致", "GPU 型号一致", "卡间互联", "RDMA 网卡", "memlock"}
    bad = [k for k in keys if k in crit]
    if bad:
        print("!! 影响双机可比性的差异（先修再测）：")
        for k in bad:
            print(f"   {k:<14} A= {ia[k]['detail'][:78]}\n   {'':<14} B= {ib[k]['detail'][:78]}")
    for k in [x for x in keys if x not in crit]:
        print(f"   {k:<14} A= {ia[k]['detail'][:52]} | B= {ib[k]['detail'][:52]}")
    if not keys:
        print("两台环境一致 ✓")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/root/llm-bench/logs")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args()
    raise SystemExit(compare(*a.compare) if a.compare else bench(Path(a.logs)))


main()
