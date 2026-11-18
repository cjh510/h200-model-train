#!/usr/bin/env python3
# =============================================================================
# [在哪跑] 目标裸机（04_bench.sh --phase report 会调它）。
# 把 Megatron 日志压成带判定的验收表，并打印结果指纹、对齐两台机器的体检结果。
#
#   python3 05_report.py --logs /root/llm-bench/logs
#   python3 05_report.py --compare node-a-env.json node-b-env.json
#
# 相对原方案的变化（详见 变更清单.md）：
#   · 显存口径写实测 143771 MiB = 140.4 GiB，不再用「141 GiB」
#   · dmon 字段名兼容 gput/gtemp、totm/mtemp，并支持 dmon-<tag>.csv 回退
#   · 新增「长稳」段落：步数、等效时长、快照（原 soak_8B tag 解析不了，整组丢失）
#   · 打印 results.jsonl 里的环境指纹（engine / namespace / 镜像 digest / 驱动 / torch / CUDA / NCCL）
# =============================================================================
import argparse
import json
import re
import statistics
from pathlib import Path

PEAK = 989.4          # H200 SXM bf16 稠密峰值 TFLOPS（不含稀疏；稀疏是 2×）
VRAM_GIB = 140.4      # 实测 143771 MiB = 140.4 GiB。报告里一律用实测，不写 141
N_PARAMS = {"8B": 8.03e9, "70B": 70.55e9}      # 一手：meta-llama 公开 config 逐项算得


class RunTree:
    """把按次 run 目录（<run>/04-hw/、<run>/06-train-s8b/ …）当成扁平日志视图。

    按文件名递归查找，让原来的 parse/load 逻辑不用改。
    """

    def __init__(self, root):
        self.root = Path(root)

    def _hits(self, pattern):
        out = []
        for p in sorted(self.root.rglob(pattern)):
            # node-b/ 是旁证，不参与 node-a 的汇总表
            if "node-b" in p.relative_to(self.root).parts:
                continue
            out.append(p)
        return out

    def __truediv__(self, name):
        hits = self._hits(name)
        return hits[0] if hits else self.root / "__missing__" / name

    def glob(self, pattern):
        return self._hits(pattern)


def _as_logs(logs):
    """logs 仍可指向扁平目录；如果它下面有 NN-阶段 子目录，自动切换为递归视图。"""
    logs = Path(logs)
    try:
        sub = [p for p in logs.iterdir() if p.is_dir() and re.match(r"^\d\d-", p.name)]
    except OSError:
        sub = []
    return RunTree(logs) if sub else logs

# tag 形如 w8_8B_tp8_pp1_mbs2_s4096_gb32_bf16
# soak 档位是 w8_8B_soak_tp8_pp1_mbs4_s4096_gb8_bf16：模型名后面允许多一段词，
# 否则 "soak" 会被当成 model 名导致整行被丢弃（原方案的 bug）。
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
    tp, pp = int(f["tp"]), int(f["pp"])
    dp = world // (tp * pp)
    return {
        "tag": path.stem, "model": f["model"].upper(), "world": world,
        "tp": tp, "pp": pp, "mbs": int(f["mbs"]),
        "seq": seq, "gbs": gbs, "prec": f["prec"], "dp": dp,
        "accum": gbs // (int(f["mbs"]) * dp) if dp else None,
        "step_s": ms / 1e3, "jitter": jit, "iters": len(rows),
        "tok_s_gpu": gbs * seq / (ms / 1e3) / world,
        "tflops_6nd": tflops_6nd, "tflops_megatron": tf_reported,
        "mfu": tflops_6nd / PEAK * 100,
        "mfu_megatron": (tf_reported / PEAK * 100) if tf_reported else None,
        # 一个 step 的 FLOPs = 6·N·(GBS×seq)，摊到 world 张卡；
        # 不是 6·N·mbs·seq —— 那样只算了单个 micro-batch。
        "t_min": 6 * n * gbs * seq / (world * PEAK * 1e12),
        "loss": keep[-1].get("loss"),
        "is_soak": "soak" in path.stem.lower(),
    }


def _read_dmon(f: Path):
    hdr, rows = None, []
    for ln in f.read_text(errors="ignore").splitlines():
        if ln.startswith("#"):
            hdr = ln.lstrip("#").split()
        elif hdr and ln.strip():
            rows.append(ln.split())
    return hdr, rows


def dmon(logs: Path, pattern="dmon-*.csv", exclude=None):
    """逐卡序列：不做跨卡平均（8 卡平均会把离群卡与降频卡抹平）。"""
    f = logs / "dmon.csv"
    if pattern != "dmon-*.csv" or not f.exists():
        cand = sorted(logs.glob(pattern))
        if exclude:
            cand = [p for p in cand if exclude not in p.name]
        if not cand:
            return None
        f = cand[-1]
    hdr, rows = _read_dmon(f)
    if not hdr or "gpu" not in hdr:
        return None
    gi = hdr.index("gpu")
    fields = (("pwr", "W", ("pwr",)),
              ("温度", "°C", ("gput", "gtemp")),
              ("显存温度", "°C", ("totm", "mtemp")))
    per = {}
    for r in rows:
        if len(r) > gi and re.fullmatch(r"[0-9]+", r[gi]):
            per.setdefault(int(r[gi]), []).append(r)
    out = []
    for g in sorted(per):
        parts = []
        for label, unit, names in fields:
            i = next((hdr.index(n) for n in names if n in hdr), None)
            if i is None:
                continue
            vals = [float(r[i]) for r in per[g] if len(r) > i and re.fullmatch(r"[\d.]+", r[i] or "")]
            if vals:
                parts.append(f"{label} {statistics.mean(vals):.0f}/{max(vals):.0f}{unit}")
        if parts:
            out.append(f"    GPU{g}: " + "  ".join(parts))
    tag = f"（{f.name}）" if f.name != "dmon.csv" else ""
    return "\n".join(out) + (f"\n    采样文件 {tag}" if tag else "") if out else None


def dmon_stats(root: Path, pattern="dmon-*.csv", exclude=None):
    """取一份 dmon，汇总全部采样点的功耗 / 温度（用于长稳表的单行单元格）。"""
    files = sorted(root.rglob(pattern))
    if exclude:
        files = [p for p in files if exclude not in p.name]
    if not files:
        return None
    hdr, rows = _read_dmon(files[-1])
    if not hdr or "gpu" not in hdr:
        return None
    gi = hdr.index("gpu")

    def col(*names):
        i = next((hdr.index(n) for n in names if n in hdr), None)
        if i is None:
            return []
        return [float(r[i]) for r in rows
                if len(r) > i and r[gi].isdigit() and re.fullmatch(r"[\d.]+", r[i] or "")]

    return {"file": files[-1].name, "pwr": col("pwr"),
            "temp": col("gput", "gtemp"), "memtemp": col("totm", "mtemp")}


def _cell(vals):
    return f"{statistics.mean(vals):.0f}/{max(vals):.0f}" if vals else "—"


def expert_notes(logs):
    """expert-load-timeseries.csv → 每次采样的 note 列（空串也计入，用于数采样次数）。"""
    out = []
    files = list(logs.glob("expert-load-timeseries.csv")) if hasattr(logs, "glob") else []
    if isinstance(logs, RunTree):
        peer = [p for p in sorted(logs.root.rglob("expert-load-timeseries.csv"))
                if "node-b" in p.relative_to(logs.root).parts]
        if peer:
            files = peer
    for p in files:
        for ln in p.read_text(errors="ignore").splitlines()[1:]:
            parts = ln.split(",")
            if len(parts) < 6:
                continue
            out.append(parts[5])
    return out


def load_jsonl(logs, *names):
    rows = []
    for n in names:
        paths = logs.glob(n) if hasattr(logs, "glob") else [logs / n]
        for p in paths:
            if not p.exists():
                continue
            for ln in p.read_text(errors="ignore").splitlines():
                try:
                    j = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                j["_file"] = n
                rows.append(j)
    return rows


def soak_jsonl_rows(logs, name):
    """长稳 JSONL 同时收 node-b；同 tag/status 优先取 iteration 更多的末级节点记录。"""
    rows = load_jsonl(logs, name)
    if isinstance(logs, RunTree):
        for p in sorted(logs.root.rglob(name)):
            if "node-b" not in p.relative_to(logs.root).parts:
                continue
            for ln in p.read_text(errors="ignore").splitlines():
                try:
                    j = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                j["_file"] = name
                rows.append(j)
    best, keys = {}, []
    for r in rows:
        key = (r.get("tag"), r.get("status"))
        if key not in best:
            keys.append(key)
            best[key] = r
        elif (r.get("iters") or 0) > (best[key].get("iters") or 0):
            best[key] = r
    return [best[k] for k in keys]


def _sz(b):
    if b is None:
        return "—"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.0f}{u}"
        b /= 1024
    return f"{b:.0f}TB"


def probe_tables(logs: Path):
    rows = load_jsonl(logs, "hw.jsonl", "coll.jsonl", "nccl.jsonl")
    if not rows:
        return
    micro = [r for r in rows if r.get("kind") in ("gpu", "gemm", "hbm", "l2", "pcie")]
    if micro:
        print("\n### 硬件微基准（稳态 ≥5s）\n")
        print("| world | 项 | 值 | 状态 |")
        print("|---|---|---|---|")
        for r in micro:
            k = r.get("kind")
            if k == "gpu":
                v = f"容量 {r.get('cap_gib')} GiB / {','.join(r.get('names') or [])}"
            elif k == "gemm":
                v = f"{r.get('tflops', '—')} TFLOPS"
            elif k == "hbm":
                v = f"{r.get('tbs', '—')} TB/s（{r.get('op')}）"
            elif k == "l2":
                v = f"{r.get('gbps', '—')} GB/s"
            else:
                v = f"{r.get('gbs', '—')} GB/s（{r.get('dir')}）"
            print(f"| {r.get('world', '—')} | {k}:{r.get('dtype') or r.get('op') or r.get('dir') or ''} "
                  f"| {v} | {r.get('status', 'OK')} |")
    nr = [r for r in rows if r.get("kind") == "nccl"]
    if nr:
        print(f"\n  NCCL 版本：{nr[0].get('version')}（world={nr[0].get('world')}）")
    coll_table(logs)


def _root(logs):
    return logs.root if isinstance(logs, RunTree) else Path(logs)


def _jsonl(root: Path, *patterns):
    """按文件名模式跨 node-a/node-b 收集 JSONL（collective 要两台的数，不能跳过 node-b）。"""
    rows = []
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            for ln in p.read_text(errors="ignore").splitlines():
                try:
                    j = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                j["_file"] = p.name
                rows.append(j)
    return rows


def coll_table(logs):
    """Collective 带宽：op × 消息大小，单机与跨机分列（同一 tag 在 node-a/node-b 各自落盘）。"""
    # hw 阶段默认 --only all，已含四类 collective，单机 coll 降为可选后仍能出这张表
    rows = [r for r in _jsonl(_root(logs), "hw.jsonl", "coll.jsonl", "coll-node*.jsonl", "nccl.jsonl")
            if r.get("kind") == "collective"]
    if not rows:
        return
    print("\n### Collective 带宽（op × 消息大小；busbw GB/s / 延迟 us）\n")
    print("| op | 消息 | 单机 busbw | 单机延迟 | 跨机 busbw (node-a/node-b) | 跨机延迟 | "
          "all_to_all 每组 token | 状态 |")
    print("|---|---|---|---|---|---|---|---|")
    groups = {}
    for r in rows:
        groups.setdefault((r.get("op"), r.get("msg_bytes")), []).append(r)
    order = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
    for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1] or 0)):
        op, msg = key
        g = groups[key]
        single = [r for r in g if (r.get("nodes") or (2 if (r.get("world") or 0) >= 16 else 1)) == 1]
        cross = [r for r in g if r not in single]
        st = "OK" if all(r.get("status") == "OK" for r in g) else \
             next((r.get("status") or "OK") for r in g if r.get("status") != "OK")
        sb = f"{single[0].get('busbw_gbs', '—')}" if single else "—"
        sl = f"{single[0].get('latency_us', '—')}" if single else "—"
        cv = "/".join(str(r.get("busbw_gbs", "—")) for r in sorted(
            cross, key=lambda x: x.get("node_rank", 0))) if cross else "—"
        cl = "/".join(str(r.get("latency_us", "—")) for r in sorted(
            cross, key=lambda x: x.get("node_rank", 0))) if cross else "—"
        tpg = next((r.get("tokens_per_group") for r in g if r.get("tokens_per_group")), None)
        print(f"| {op} | {_sz(msg)} | {sb} | {sl} | {cv} | {cl} | "
              f"{tpg if tpg is not None else '—'} | {st} |")
    print("  跨机行按 node_rank 分列（node-a/node-b 分别落盘）；all_to_all 的「每组 token」="
          "每 rank 发往单个对端的 bf16 元素数，split_dim=0。")

def moe_table(logs):
    """MoE / EP 扩展：step 时间、每卡 token/s、激活参数 MFU、专家负载、dispatcher 对比。"""
    rows = load_jsonl(logs, "moe.jsonl")
    if not rows:
        return
    print("\n### MoE / EP 扩展\n")
    print("| topology | world | ep/tp/pp | dispatcher | attn | exp | mbs | seq | GBS | step s | "
          "tok/s/GPU | MFU(激活) | 专家负载 max/mean | 状态/备注 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (str(x.get("attention") or "mla"), x.get("world", 0),
                                         x.get("ep", 0), x.get("dispatcher") or "")):
        load = r.get("expert_load_max_over_mean")
        ltxt = "—" if load in (None, "null") else f"{load}"
        if isinstance(load, (int, float)) and load > 1.5:
            ltxt += " **WARN**"
        mfu = r.get("mfu_active")
        mtxt = "—" if mfu in (None, 0, 0.0) else f"{mfu}%"
        st = r.get("status", "—")
        attn = str(r.get("attention") or "mla")
        exp = "yes" if r.get("experimental") else "no"
        note = r.get("note") or ""
        stxt = f"{st}：{note}" if note and st in ("SKIP", "FAILED") else st
        if st in ("SKIP", "FAILED"):
            mtxt, ltxt = "—", "—"   # SKIP/失败不参与 MFU 判定
            print(f"| {r.get('topology', '—')} | {r.get('world')} | {r.get('ep')}/{r.get('tp')}/{r.get('pp')} "
                  f"| {r.get('dispatcher')} | {attn} | {exp} | {r.get('mbs')} | {r.get('seq')} | {r.get('gbs')} "
                  f"| — | — | — | — | {stxt} |")
        else:
            print(f"| {r.get('topology', '—')} | {r.get('world')} | {r.get('ep')}/{r.get('tp')}/{r.get('pp')} "
                  f"| {r.get('dispatcher')} | {attn} | {exp} | {r.get('mbs')} | {r.get('seq')} | {r.get('gbs')} "
                  f"| {r.get('step_s', '—')} | {r.get('tokens_per_gpu', '—'):,} | {mtxt} | {ltxt} | {stxt} |")
    print("\n  `attn`/`exp` 标明该档用 MLA 还是普通 GQA、是否传了 `--enable-experimental`。"
          "MLA 与 GQA 结构不同，**仅可比 EP 扩展趋势，不能横向比 step/MFU**；表中已按 attention 分组排序。")
    print("  `MFU(激活)` 算法：MFU = 6·P_active·(GBS×seq)/step_s ÷ world ÷ 989.4 TFLOPS，"
          "P_active = 2·V·H + L·(attn + shared + (topk/专家数)·每专家 FFN)，"
          "即每 token 真正参与计算的参数（不含未激活专家）。")
    print("  专家负载 max/mean >1.5 标 WARN（路由塌缩风险）；SKIP/FAILED 不参与 MFU 与最好成绩判定。")


def limit_table(logs):
    """显存 / 规模边界：成功上界与首个 OOM。OOM_BOUNDARY 不算失败。"""
    rows = load_jsonl(logs, "limit.jsonl")
    root = _root(logs)
    summaries = sorted(root.rglob("limit-summary.json"))
    data = None
    if summaries:
        try:
            data = json.loads(summaries[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
    if not rows and not data:
        return
    print("\n### 显存 / 规模边界\n")
    if rows:
        print("| 模型 | tp/pp/ep | mbs | seq | GBS | accum | step s | 静态估算 GiB/卡 | 状态 | 关键错误 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (x.get("model", ""), x.get("seq", 0), x.get("mbs", 0))):
            st = r.get("status", "—")
            mark = {"OK": "OK", "OOM_BOUNDARY": "**OOM_BOUNDARY**", "FAILED": "**FAILED**",
                    "SKIP": "SKIP", "PARTIAL": "PARTIAL"}.get(st, st)
            print(f"| {r.get('model')} | {r.get('tp')}/{r.get('pp')}/{r.get('ep')} | {r.get('mbs')} "
                  f"| {r.get('seq'):,} | {r.get('gbs')} | {r.get('accum', '—')} | {r.get('step_s', '—')} "
                  f"| {r.get('static_gb_est', '—')} | {mark} | {r.get('error_line') or '—'} |")
    if data:
        print(f"\n  **成功上界**：max_ok_mbs@4096 = **{data.get('max_ok_mbs@4096')}**，"
              f"max_ok_seq@best_mbs = **{data.get('max_ok_seq@best_mbs'):,}**；"
              f"静态估算 {data.get('static_gb_est')} GiB/卡，按 {data.get('vram_gib_per_gpu')} GiB 余量 "
              f"{data.get('static_margin_gib')} GiB")
        fo = data.get("first_oom")
        if fo:
            print(f"  首个 OOM：tp/pp/ep {fo.get('tp')}/{fo.get('pp')}/{fo.get('ep')}，"
                  f"mbs={fo.get('mbs')} seq={fo.get('seq'):,} GBS={fo.get('gbs')}，"
                  f"error_class={fo.get('error_class')}；`{fo.get('error_line')}`")
    print("  `OOM_BOUNDARY` 是预期边界：不计失败、不 dual_stop、不影响退出码，也排除在最好成绩与 MFU 之外；"
          "只有 NCCL 错误 / hang / 进程崩溃记 FAILED。")
    print("  非性能口径：70B D3 与双机 limit 都带 PP2/DP1，GBS=mbs×dp 使 nmib=1，"
          "流水线气泡约 50%；这些 step 时间只用于显存边界，不作性能结论。")


def io_table(logs: Path):
    rows = load_jsonl(logs, "io.jsonl")
    if not rows:
        return
    print("\n### 存储 / 数据管线\n")
    print("| 模型 | world | 总大小 | 写 MB/s | 读 MB/s | 写耗时 s | 读耗时 s | 状态 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") == "checkpoint":
            print(f"| {r.get('model')} | {r.get('world')} | {r.get('total_gib')} GiB "
                  f"| {r.get('write_mbs')} | {r.get('read_mbs')} | {r.get('write_s')} | {r.get('read_s')} | OK |")
        else:
            print(f"| — | — | — | — | — | — | — | SKIP: {r.get('note', '')} |")
    print("  `checkpoint` 是分片写读代理（每卡写 bf16 权重 1/world），不是 Megatron 分布式 checkpoint 格式。")


def fault_table(logs: Path):
    rows = load_jsonl(logs, "fault.jsonl")
    if not rows:
        return
    print("\n### 故障注入\n")
    print("| 用例 | 结果 | 退出码 | 说明 |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r.get('case')} | **{r.get('status')}** | {r.get('rc')} | {r.get('note', '')} |")


def health_table(root: Path, pattern: str):
    """读出该长稳自己的 health-delta：返回 (md 行, 总判定, 文件数)。

    注意这里要连 node-b/ 一起收（RunTree 汇总表会跳过 node-b，硬件增量不能跳）。
    """
    files = sorted(root.rglob(pattern))
    if not files:
        return [], None, 0
    rows = ["| 机器 | ECC修正 | ECC未修正 | retired | NVLink误码 | Xid | SXid | 判定 | 定位 |",
            "|---|---|---|---|---|---|---|---|---|"]
    overall = "PASS"

    def _g(v):
        return "GPU" + ",".join(str(x) for x in v) if isinstance(v, list) else f"GPU{v}"

    for f in files:
        try:
            j = json.loads(f.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        host = j.get("host") or ("node-b" if "node-b" in f.parts else "?")
        d = j.get("delta", {})
        rs = j.get("retired_pages_state")
        ret_cell = "SKIP（字段不支持）" if rs == "SKIP" else f"{(d.get('retired_pending') or 0):+d}"
        bad = [x for x in (j.get("bad") or [])
               if not (rs == "SKIP" and x.get("key") == "retired_pending")]
        st = "FAIL" if bad else ("SKIP" if rs == "SKIP" else "PASS")
        if st == "FAIL":
            overall = "FAIL"
        loc = "；".join(f"{_g(x['gpu'])} {x['key']} {x['delta']:+d}" for x in bad) or "—"
        rows.append(f"| {host} | {(d.get('ecc_corrected') or 0):+d} | {(d.get('ecc_uncorrected') or 0):+d} "
                    f"| {ret_cell} | {(d.get('nvlink_errors') or 0):+d} "
                    f"| {(d.get('xid') or 0):+d} | {(d.get('sxid') or 0):+d} | **{st}** | {loc} |")
    return rows, overall, len(files)


def soak_section(logs: Path, soak):
    """8B/70B Dense 长稳同表分行；每档读自己的 dmon，硬件增量分开判定。"""
    if not soak:
        return
    root = logs.root if isinstance(logs, RunTree) else Path(logs)
    print("\n### Dense 长稳（8B TP8×PP1；双机 70B TP8×PP2）\n")
    print("| 档位 | 步数 | 等价时长 | step s | 抖动 | MFU† | 功耗 均/峰 (W) | 温度 均/峰 (°C) | 判定 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(soak, key=lambda x: x["tag"]):
        # 不能全局取最新 dmon：dense70 跑完后会把 70B 的功耗/温度错套到 8B 行。
        st = dmon_stats(root, f"dmon-{r['tag']}.csv")
        dur_h = r["iters"] * r["step_s"] / 3600
        mm = f"{r['mfu_megatron']:.1f}%" if r["mfu_megatron"] else "—"
        warn = r["jitter"] > 5 or (st and st["temp"] and max(st["temp"]) >= 85)
        print(f"| `{r['tag']}` | {r['iters']} | {dur_h:.2f} h | {r['step_s']:.3f} | "
              f"{r['jitter']:.1f}% | {mm} | {_cell(st['pwr']) if st else '—'} | "
              f"{_cell(st['temp']) if st else '—'} | **{'WARN' if warn else 'OK'}** |")
    print("\n- Dense 判定：抖动 >5% 或温度 ≥85°C 记 WARN；只看抖动是否随时间放大、功耗/温度是否爬升、有无 OOM/超时。")
    print("  双机 iteration 由全局末级 rank 打印；非末级节点无 iteration 不代表未训练，步数/性能取 node-b 日志。")
    models = {r["model"] for r in soak}
    if "8B" in models:
        rows, overall, n = health_table(root, "health-delta.json")
        if rows:
            print("\n**Dense 8B 长稳硬件增量（与 70B/MoE 独立）**\n")
            print("\n".join(rows))
            print(f"\n- Dense 8B 硬件判定：**{overall}**" + ("（只收到 1 台，双机需回传 node-b）" if n < 2 else ""))
    if "70B" in models:
        rows, overall, n = health_table(root, "dense70-health-delta.json")
        if rows:
            print("\n**Dense 70B 长稳硬件增量（与 8B/MoE 独立）**\n")
            print("\n".join(rows))
            print(f"\n- Dense 70B 硬件判定：**{overall}**" + ("（只收到 1 台，双机需回传 node-b）" if n < 2 else ""))


def moe_soak_section(logs: Path):
    """MoE 长稳单独一张表：额外报专家负载 max/mean 与 drift。"""
    rows = soak_jsonl_rows(logs, "moe-soak.jsonl")
    if not rows:
        return
    root = logs.root if isinstance(logs, RunTree) else Path(logs)
    dmon_st = dmon_stats(root, "dmon-moe-soak-*.csv")
    series = expert_notes(logs)
    print("\n### MoE 长稳（配置 B，EP×TP1×PP1，alltoall）\n")
    print("| 拓扑 | attn | exp | dispatcher | 步数 | 等价时长 | step s | 抖动 | MFU† | 功耗 均/峰 (W) | "
          "温度 均/峰 (°C) | 专家负载 max/mean | drift | 判定/备注 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        dur_h = (r.get("iters") or 0) * (r.get("step_s") or 0) / 3600
        jit = r.get("jitter") or 0
        imb, dr = r.get("imbalance_max_over_mean"), r.get("imbalance_drift")
        imb_cell = f"{imb:.2f}" if isinstance(imb, (int, float)) else "null"
        dr_cell = f"{dr:+.1f}%" if isinstance(dr, (int, float)) else "null"
        attn = str(r.get("attention") or "mla")
        exp = "yes" if r.get("experimental") else "no"
        st = r.get("status", "OK")
        note = r.get("note") or ""
        if st == "SKIP":
            print(f"| EP{r.get('ep')}×TP{r.get('tp')}×PP{r.get('pp')} | {attn} | {exp} | {r.get('dispatcher')} "
                  f"| — | — | — | — | — | — | — | — | — | SKIP：{note or '未跑'} |")
            continue
        warn = ((isinstance(imb, (int, float)) and imb > 1.5)
                or (isinstance(dr, (int, float)) and abs(dr) > 20)
                or jit > 5
                or (dmon_st and dmon_st["temp"] and max(dmon_st["temp"]) >= 85))
        print(f"| EP{r.get('ep')}×TP{r.get('tp')}×PP{r.get('pp')} | {attn} | {exp} | "
              f"{r.get('dispatcher')} | {r.get('iters')} | {dur_h:.2f} h | {r.get('step_s') or 0:.3f} | "
              f"{jit:.1f}% | {r.get('mfu_active') or 0:.1f}% | {_cell(dmon_st['pwr']) if dmon_st else '—'} | "
              f"{_cell(dmon_st['temp']) if dmon_st else '—'} | {imb_cell} | {dr_cell} | "
              f"**{'WARN' if warn else 'OK'}**" + (f"（{note}）" if note else "") + " |")
    print("\n- MoE 判定：`imbalance_max_over_mean` >1.5 或 |drift| >20% 记 WARN（路由塌缩/负载漂移）；"
          "抖动 >5% 或温度 ≥85°C 同样 WARN。")
    bad = sorted({n for n in series if n})
    if bad:
        print(f"- 专家负载有 null 采样（{len(series)} 次）：{'；'.join(bad)}")
    elif series:
        print(f"- 专家负载采样 {len(series)} 次（expert-load-timeseries.csv）。")
    rows_h, overall, n = health_table(root, "moe-health-delta.json")
    if rows_h:
        print("\n**MoE 长稳硬件增量（与 Dense 独立）**\n")
        print("\n".join(rows_h))
        print(f"\n- MoE 硬件判定：**{overall}**" + ("（只收到 1 台，双机需回传 node-b）" if n < 2 else ""))


def _hm_verdict(status, peak, growth, mmax, mend, max_at, n):
    """显存高占用判定：区间 75~92% 为 PASS；只报 WARN，不影响其它两条长稳。"""
    if status == "OOM_BOUNDARY":
        return "OOM_BOUNDARY", []
    # 冒烟（HIGHMEM_MINUTES<10）：漂移还没发生，不能拿 75~92% 判它
    if status == "SMOKE":
        return "SMOKE（短跑，未判定）", []
    if status and status not in ("OK",):
        return status, []
    warns = []
    if peak is not None and peak < 75:
        warns.append("未达高占用（档位/seq 太小，见 highmem-cal.jsonl）")
    elif peak is not None and peak > 92:
        warns.append("过于贴边，长稳结论不稳")
    if growth is not None and growth > 5:
        warns.append("疑似泄漏")
    # 「曲线中途无回落」：峰值出现在中途、且末尾仍在峰值附近 → 疑似碎片；
    # 全程平稳（growth≈0）不触发，避免把正常稳态误报成碎片。
    if (isinstance(growth, (int, float)) and growth > 1 and mmax and mend is not None
            and isinstance(max_at, int) and isinstance(n, int) and max_at < n - 1
            and (mmax - mend) / mmax * 100 < 3):
        warns.append("疑似碎片")
    return ("WARN" if warns else "PASS"), warns


def mfu_section(logs: Path):
    """峰值 MFU（--phase mfu）：每 (模型, world) 一个峰值，附配置、CV、与 GEMM 天花板的差距。"""
    rows = soak_jsonl_rows(logs, "mfu.jsonl")
    if not rows:
        return
    print("\n### 峰值 MFU（--phase mfu；口径见下）\n")
    ok = [r for r in rows if r.get("status") == "OK" and r.get("step_s")]
    if not ok:
        print("- `mfu.jsonl` 里没有一档 status=OK，先看 `mfu.jsonl` 与各档日志")
        return
    # 同一配置多次复测 → 中位 + CV；CV>阈值或 throttle≠0 不进峰值
    groups = {}
    for r in ok:
        k = (r.get("model"), r.get("nodes"), r.get("tp"), r.get("pp"),
             r.get("mbs"), r.get("seq"), r.get("gbs"))
        groups.setdefault(k, []).append(r)
    per = []
    for k, rs in groups.items():
        vals = [x.get("mfu_megatron") or 0 for x in rs]
        med = statistics.median(vals)
        cv = (statistics.pstdev(vals) / statistics.mean(vals) * 100) if len(vals) > 1 and statistics.mean(vals) else None
        per.append({"model": k[0], "nodes": k[1], "tp": k[2], "pp": k[3], "mbs": k[4], "seq": k[5],
                    "gbs": k[6], "n": len(vals), "mfu": med,
                    "mfu6n": statistics.median([x.get("mfu_6n") or 0 for x in rs]),
                    "cv": cv, "throttle": any(x.get("throttle") == 1 for x in rs)})
    per.sort(key=lambda x: -x["mfu"])
    print("| 模型 | TP×PP | mbs | seq | GBS | 复测 | MFU† | MFU(6N) | CV | 判定 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in per[:12]:
        cv = "—" if c["cv"] is None else f'{c["cv"]:.2f}%'
        bad = c["throttle"] or (c["cv"] is not None and c["cv"] > 2.0)
        verdict = "**排除**（降频/CV 超限）" if bad else ("有效" if c["cv"] is not None else "单次")
        print(f'| {c["model"]} | {c["tp"]}×{c["pp"]} | {c["mbs"]} | {c["seq"]} | {c["gbs"]} | {c["n"]} | '
              f'**{c["mfu"]:.1f}%** | {c["mfu6n"]:.1f}% | {cv} | {verdict} |')
    # 每个 (模型, world) 的峰值
    peaks = {}
    for c in per:
        if c["throttle"] or (c["cv"] is not None and c["cv"] > 2.0):
            continue
        k = (c["model"], c["nodes"])
        if k not in peaks or c["mfu"] > peaks[k]["mfu"]:
            peaks[k] = c
    gemm_pct = 690.0 / PEAK * 100
    print(f"\n**峰值**（分母 {PEAK} TFLOPS；04-hw 的 bf16 GEMM 实测 690 TFLOP/s = 理论 {gemm_pct:.1f}%，训练峰值不可能超过它）\n")
    for k, c in sorted(peaks.items()):
        print(f'- **{k[0]} @ {k[1]} 节点 → MFU† {c["mfu"]:.1f}%**（MFU(6N) {c["mfu6n"]:.1f}%），'
              f'TP{c["tp"]}×PP{c["pp"]}×DP{c["world_dp"] if "world_dp" in c else (k[1] * 8 // (c["tp"] * c["pp"]))} '
              f'mbs{c["mbs"]} seq{c["seq"]} GBS{c["gbs"]}；复测 {c["n"]} 次'
              f'{("，CV %.2f%%" % c["cv"]) if c["cv"] is not None else "（未复测，标 UNCERTIFIED）"}；'
              f'约为 GEMM 天花板的 {c["mfu"] / gemm_pct * 100:.0f}%')
    print("\n- 口径：`MFU†` 用 Megatron 自报 throughput（含 attention）除以 989.5 TFLOPS，跨 seq 只比这个；"
          "`MFU(6N)` 只算线性层，是保守下界。`CV>2%` 或采样到 throttle≠0 的档不进峰值。"
          "复测次数 =1 的标 UNCERTIFIED，需 `MFU_REPEAT≥3` 才算已验证。")
    print("- 峰值明细：`mfu.jsonl`（逐档）+ `mfu-peak.json`（每 (模型, world) 一个峰值）；"
          "时钟/降频证据：`clocks-w<N>_MFU_*.csv`")


def highmem_section(logs: Path):
    """第三张长稳表：显存高占用。每条 run 一列、逐卡一行，与 Dense/MoE 不合并、不互判。"""
    rows = soak_jsonl_rows(logs, "highmem.jsonl")
    if not rows:
        return
    print("\n### 显存高占用长稳（不拉满；与 Dense/MoE 两条独立）\n")
    print("| 模型/拓扑 | mbs 请求→实际 | seq | GPU | start/peak/end (MiB) | peak_pct | "
          "growth_pct | 判定 | 边界来源 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        req, used = r.get("mbs_requested"), r.get("mbs_used")
        mbs_cell = f"{req}→{used}" if req != used else str(used)
        model = f"{r.get('model', '?')} TP{r.get('tp')}×PP{r.get('pp')}×DP{r.get('dp')}"
        src = r.get("boundary_source") or "none"
        src_cell = {"limit": "limit（同模型）", "calib": "同模型标定",
                    "manual": "手动指定", "tokens": "tokens（换算·已废弃）",
                    "infeasible": "infeasible（已废弃）",
                    "cross-model": "cross-model（不借用）",
                    "no-seq": "limit 无该 seq"}.get(src, "none（无边界数据）")
        st = r.get("status") or ""
        gpus = r.get("gpu") or []
        if not gpus:
            verdict, warns = _hm_verdict(st, None, None, None, None, None, None)
            det = f"**{verdict}**" + (f"（{'；'.join(warns)}）" if warns else "")
            print(f"| {model} | {mbs_cell} | {r.get('seq')} | — | — | — | — | {det} | {src_cell} |")
            continue
        for g in gpus:
            peak, growth = g.get("peak_pct"), g.get("growth_pct")
            verdict, warns = _hm_verdict(st, peak, growth, g.get("mem_max"), g.get("mem_end"),
                                         g.get("mem_max_at"), g.get("samples"))
            det = f"**{verdict}**" + (f"（{'；'.join(warns)}）" if warns else "")
            print(f"| {model} | {mbs_cell} | {r.get('seq')} | GPU{g.get('index')} | "
                  f"{g.get('mem_start')}/{g.get('mem_max')}/{g.get('mem_end')} | "
                  f"{peak if peak is not None else '—'}% | "
                  f"{(f'{growth:+.1f}%' if isinstance(growth, (int, float)) else '—')} | "
                  f"{det} | {src_cell} |")
    print("\n- 判定口径：`status` 由脚本按实测峰值收口——peak_pct 75~92% 记 OK；<75% 记 `WARN_LOW`"
          "（未达高占用，且会被撤销 `.ok_highmem` 迫使其重跑）；>92% 记 `WARN_HOT`（过于贴边）；"
          "`growth_pct` >5% 记 WARN（疑似泄漏）；峰值在中途、末尾仍贴峰（差 <3%）记 WARN（疑似碎片）。"
          "OOM 记 `OOM_BOUNDARY`，不计失败；`HIGHMEM_MINUTES<10` 的冒烟记 `SMOKE`，不参与判定带。")
    notes = []
    for r in rows:
        if r.get("status") == "OOM_BOUNDARY":
            notes.append(f"OOM_BOUNDARY（{r.get('error_class') or 'OOM'}）：mbs={r.get('mbs_used')}、"
                         f"seq={r.get('seq')}，不计失败，退出码 0")
        if r.get("mbs_requested") != r.get("mbs_used"):
            notes.append(f"mbs 由 limit.jsonl 边界自动下调：{r.get('mbs_requested')} → {r.get('mbs_used')}"
                         f"（seq={r.get('seq')} 边界 {r.get('boundary_mbs')}）")
        if r.get("boundary_source") == "calib":
            cal_p = r.get("calib_peak_mib")
            notes.append(f"档位由同模型标定得出：mbs={r.get('mbs_used')}"
                         + (f"（标定档实测 {cal_p} MiB）" if isinstance(cal_p, (int, float)) else "")
                         + f"，逐档曲线见 highmem-cal.jsonl；本次实测整机峰值 {r.get('peak_pct')}%")
        elif r.get("boundary_source") in ("cross-model", "no-seq"):
            notes.append("limit 阶段扫的是另一个模型（单机 MOE_B vs dense 8B），其 mbs/seq 边界"
                         "不可跨模型借用——旧逻辑据此把 mbs 压成 1、占用只有 29%（11-10 实测），"
                         "现已改为同模型标定")
        elif r.get("boundary_source") == "tokens":
            notes.append("边界按 limit OK 档最大 tokens/GPU 换算（单机 limit 与 highmem 模型不同，"
                         "该 seq 无直接 OK 档）：MoE 口径对 dense 保守")
        elif r.get("boundary_source") == "infeasible":
            notes.append("seq 超出 limit tokens/GPU 上限（换算 mbs=0）：按 mbs=1 跑，peak 会「未达高占用」，"
                         "建议 HIGHMEM_MODEL=70b 或降低 HIGHMEM_SEQ")
        elif r.get("boundary_source") != "limit":
            notes.append("无边界数据（limit.jsonl 缺失或该 seq 无 OK 记录）：按默认 mbs 跑，可能 OOM")
        tmax = r.get("torch_max_allocated_mib")
        if isinstance(tmax, (int, float)):
            notes.append(f"torch max allocated {tmax:.0f} MiB")
    for n in dict.fromkeys(notes):
        print(f"- {n}")
    print("\n**峰值占用（逐卡，分母 143771 MiB）**\n")
    for r in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        for g in r.get("gpu") or []:
            p = g.get("peak_pct") or 0
            bar = "█" * max(0, min(20, int(p // 5)))
            print(f"  GPU{g.get('index')}  {bar:<20} {p:.0f}%  "
                  f"（{r.get('model')} mbs={r.get('mbs_used')} seq={r.get('seq')}）")
    rows_h, overall, n = health_table(logs.root if isinstance(logs, RunTree) else Path(logs),
                                      "highmem-health-delta.json")
    if rows_h:
        print("\n**显存高占用长稳硬件增量（与 Dense/MoE 独立）**\n")
        print("\n".join(rows_h))
        print(f"\n- 硬件判定：**{overall}**" + ("（只收到 1 台，双机需回传 node-b）" if n < 2 else ""))


def health_delta(logs: Path, title="长稳硬件增量（跑后 - 跑前，按机器分列）",
                 pattern="health-delta.json"):
    root = logs.root if isinstance(logs, RunTree) else Path(logs)
    if not root.is_dir():
        return
    rows, overall, n = health_table(root, pattern)
    if not rows:
        return
    print(f"\n### {title}\n")
    print("\n".join(rows))
    if n < 2:
        print(f"\n- 只收到 {n} 台机器的增量：双机长稳要求每台各采一份（node-b 的结果需回传，见 sync_logs）。")
    print(f"\n- 总判定：**{overall}**（ECC/retired/NVLink/Xid/SXid 任一增量 >0 即 FAIL；Xid 请按“定位”列到卡）")


def load_results(logs: Path):
    """results.jsonl → 每 tag 最后一次状态 + 全表出现过的环境指纹。

    按次 run 时 results.jsonl 分散在各训练阶段目录，这里合并全部。
    """
    status, fp = {}, {}
    for j in load_jsonl(logs, "results.jsonl"):
        tag = j.get("tag")
        if not tag:
            continue
        status[tag] = j.get("status", "?")
        key = tuple(str(j.get(k, "")) for k in
                    ("image", "engine", "namespace", "driver", "torch", "cuda", "nccl"))
        dg = str(j.get("digest", ""))[:19]
        if any(key):
            fp[(key, dg)] = fp.get((key, dg), 0) + 1
    return status, fp


def print_fingerprint(fp):
    if not fp:
        return
    print("\n### 结果指纹（results.jsonl 内出现的环境组合）\n")
    for (key, dg), n in fp.items():
        image, engine, namespace, driver, torch, cuda, nccl = key
        print(f"- {n:>2} 组：engine={engine or '?'} namespace={namespace or '?'} "
              f"image={image or '?'} digest={dg or '?'}  "
              f"driver={driver or '?'} torch={torch or '?'} cuda={cuda or '?'} nccl={nccl or '?'}")
    if len(fp) > 1:
        print("  ! 出现多组指纹：跨引擎/跨机/跨镜像混跑过，D1 的比值不能直接比，先按 env.txt 分组重跑。")
    else:
        print("  环境指纹唯一 ✓（两台 env.txt 的 engine/namespace/digest 也要 diff 一遍，见 03_deploy.sh 最后一步）")


def bench(logs: Path):
    logs = _as_logs(logs)          # 按次 run：递归进 NN-阶段子目录；扁平目录保持原行为
    done = [r for r in (parse_log(p) for p in sorted(logs.glob("*.log"))) if r]
    # 双机只有全局末级 rank 打印 iteration，非末级节点没有 iteration 也可能是正常训练；
    # 从 node-b 补 soak 结果，覆盖 8B/70B/MoE 等所有并行形态。
    # 仅补 soak，不改变既有短跑汇总；同 tag 取 iteration 更多的一份，避免误用非末级零步日志。
    if isinstance(logs, RunTree):
        by_tag = {r["tag"]: i for i, r in enumerate(done)}
        for p in sorted(logs.root.rglob("*.log")):
            if "node-b" not in p.relative_to(logs.root).parts:
                continue
            r = parse_log(p)
            if not r or not r["is_soak"]:
                continue
            i = by_tag.get(r["tag"])
            if i is None:
                by_tag[r["tag"]] = len(done)
                done.append(r)
            elif r["iters"] > done[i]["iters"]:
                done[i] = r
    status, fp = load_results(logs)
    bl = set()
    for p in logs.glob("baseline.txt"):
        bl.update(x.strip() for x in p.read_text(errors="ignore").splitlines() if x.strip())
    failed = {k: v for k, v in status.items() if v in ("OOM", "FAILED", "PARTIAL", "ABORTED")}
    skipped = {k: v for k, v in status.items() if v == "SKIP"}
    if not done:
        probes = [n for n in ("hw.jsonl", "coll.jsonl", "nccl.jsonl", "io.jsonl", "fault.jsonl",
                              "moe.jsonl", "limit.jsonl", "moe-soak.jsonl", "highmem.jsonl")
                  if (logs / n).exists()]
        if not probes:
            print("logs/ 里既没有可解析的训练结果，也没有探针 JSON。先看 04_bench.sh 的输出。")
            return 1
        print(f"\n## 无训练日志，仅输出探针结果（{', '.join(probes)}）\n")
        probe_tables(logs)
        moe_table(logs)
        limit_table(logs)
        io_table(logs)
        # MoE/highmem 长稳的 tag 不含 8B/70B，训练日志解析不到；只跑它们时靠各自 JSONL 出表
        moe_soak_section(logs)
        mfu_section(logs)
        highmem_section(logs)
        health_delta(logs, "长稳硬件增量（Dense 8B）", "health-delta.json")
        health_delta(logs, "长稳硬件增量（Dense 70B）", "dense70-health-delta.json")
        health_delta(logs, "长稳硬件增量（MoE）", "moe-health-delta.json")
        health_delta(logs, "长稳硬件增量（显存高占用）", "highmem-health-delta.json")
        fault_table(logs)
        return 0

    normal = [r for r in done if not r["is_soak"]]
    soak = [r for r in done if r["is_soak"]]
    print(f"\n## 训练压测结果（{len(done)} 组成功 / {len(failed)} 组失败或未完成）\n")
    print("| 配置 | world | tp/pp/dp | mbs | accum | GBS | seq | step s | tok/s/GPU | "
          "TFLOP/s(6N) | MFU(6N) | MFU† | t_min | 抖动 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(normal, key=lambda x: (x["model"], x["world"], -x["seq"])):
        mm = f"**{r['mfu_megatron']:.1f}%**" if r["mfu_megatron"] else "—"
        star = "★基线 " if r["tag"] in bl else ""
        print(f"| {star}{r['prec']} {r['tag'].split('_', 1)[1]} | {r['world']} "
              f"| {r['tp']}/{r['pp']}/{r['dp']} | {r['mbs']} | {r.get('accum', '—')} | {r['gbs']} "
              f"| {r['seq']:,} | {r['step_s']:.3f} | {r['tok_s_gpu']:,.0f} | {r['tflops_6nd']:.1f} "
              f"| {r['mfu']:.1f}% | {mm} | {r['t_min']:.3f} | {r['jitter']:.1f}% |")

    print("\n`MFU(6N)` = 6·N·tokens，只算线性层，长序列下漏掉 attention 的 O(s²)，是保守下界；"
          "`MFU†` 用 Megatron 自己按完整公式（含 attention）算的 throughput 除以 989.4 TFLOPS —— "
          "两者比值就是 attention 占比，seq 越长差得越多，跨 seq 比较只看 MFU†。")
    if bl:
        print(f"\n**基线**：★ 为本轮固定基线档（{', '.join(sorted(bl))}）。"
              "本轮基线用的是 fix 的 DO/overlap/recompute 门控，"
              "与历史 docker 基线**不可直接比较**；跨轮比较必须固定 baseline.txt 里的同一档配置。")
        miss = sorted(t for t in bl if not any(r["tag"] == t for r in done))
        if miss:
            print(f" ! 基线档未产出成功日志：{', '.join(miss)} —— 本轮没有可比的基线，先修该档再下结论。")

    if failed:
        print("\n### 失败 / 未完成\n")
        for t, s in sorted(failed.items()):
            print(f"- `{t}` → **{s}**")

    if skipped:
        print("\n### 跳过（SKIP，配置不成立；不是失败）\n")
        for t in sorted(skipped):
            print(f"- `{t}` → 见 results.jsonl 的 note（tp×pp 或 GBS 不整除）")

    # ---- 长稳单列：soak 的 step 时间不能和 15 步的短跑混在一张表里比。
    # Dense 8B / Dense 70B / MoE / 显存高占用分别验证 DP 算力带宽、PP 大模型、
    # 专家路由与 alltoall、显存余量；各自读取本档遥测与硬件增量，互不合并判定。
    soak_section(logs, soak)
    moe_soak_section(logs)
    mfu_section(logs)
    highmem_section(logs)

    # ---- 三条最有用的对比
    print("\n### 关键对比\n")

    def one(model, world, tp, seq, mbs=None, gbs=None):
        # GBS 必须一起锁定：单机那组里 mbs 相同但 GBS 不同的两条会串味，
        # 跨机比值会算错一倍以上，而这条比值正是 D1 的判据。
        c = [r for r in done if r["model"] == model and r["world"] == world
             and r["tp"] == tp and r["seq"] == seq and not r["is_soak"]
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
        print(f"  口径：GBS=32，双机 per-GPU token {d8['tok_s_gpu']*d8['step_s']:.0f} = 单机的一半，"
              f"accum 单机 {s8.get('accum')} / 双机 {d8.get('accum')}（不可跨 accum 横比）")
        print("  - 理想比值 0.5（同样总工作量、卡数翻倍）；双机 accum=8、单机 accum=16，"
              "0.5 只是强扩展理想比值，非同 accum 比较")
        print("  - **≤0.6：机间带宽够，跨机数据并行成立**")
        print("  - 0.6–1.0：有扩展收益但链路是瓶颈，报告要写明")
        print("  - >1.0：加一台机器吞吐不涨；>2.0 为负收益 → 70B 只能用 TP8×PP2（跨机只传激活）\n")

    d64 = one("8B", 16, 8, 4096, 2, 64)
    if d8 and d64:
        print(f"**D1b 同负载对照（双机 GBS 32→64）**：step {d8['step_s']:.3f}s → {d64['step_s']:.3f}s，"
              f"比值 {d64['step_s']/d8['step_s']:.2f}（理想 2.0：GBS 翻倍、卡数不变）；"
              f"per-GPU token 与单机 gb32 同口径（{d64['tok_s_gpu']*d64['step_s']:.0f}）\n")

    pp = sorted([r for r in done if r["model"] == "70B" and r["pp"] > 1 and not r["is_soak"]],
                key=lambda x: x["gbs"])
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

    seqs = sorted([r for r in done if r["model"] == "8B" and r["world"] == 8
                   and r["mbs"] == 1 and not r["is_soak"]], key=lambda x: x["seq"])
    if len(seqs) >= 2:
        print("**S3 长序列（8B TP8 mbs1）**\n")
        print("> seq 变化时每步 token 数不同（GBS×seq），只看 step 时间会误判超线性；"
              "下表同时给每步 token 与 tok/s/GPU 归一值。\n")
        print("| seq | 每步 token | step s | tok/s/GPU | 6N-MFU | MFU† |")
        print("|---|---|---|---|---|---|")
        for r in seqs:
            mm = f"{r['mfu_megatron']:.1f}%" if r["mfu_megatron"] else "—"
            print(f"| {r['seq']:,} | {r['gbs']*r['seq']:,} | {r['step_s']:.3f} | "
                  f"{r['tok_s_gpu']:,.0f} | {r['mfu']:.1f}% | {mm} |"
                  + (" ← OOM 前最后一档" if r['tag'] in failed else ""))
        print("  6N-MFU 随 seq 下降是公式漏了 attention 平方项，不是硬件退化；"
              "看 MFU† 是否平——平就说明算力一直是满的。真正的问题是 step 时间随 seq 超线性增长。")

    # ---- 新阶段：微基准 / collective / MoE / 显存边界 / 存储 / 长稳硬件增量 / 故障注入
    probe_tables(logs)
    moe_table(logs)
    limit_table(logs)
    io_table(logs)
    if not soak:                       # 有长稳时硬件增量已随两条长稳各自单列，避免重复
        health_delta(logs)
    fault_table(logs)

    print("\n### 判定\n")
    # 判定用 MFU†（含 attention 的完整 FLOPs）：按 6N 排会把 128K 那种 attention 主导的
    # 档位错判成「低效」，而它其实是同一台机器上真实算力最足的负载。
    # 只对验收档 world=8/16 生效；更小的 world 是开发/冒烟，不参与达标判定。
    eff = lambda r: r["mfu_megatron"] or r["mfu"]
    judge = [r for r in normal if r["world"] in (8, 16)]
    for r in sorted(judge, key=lambda x: -eff(x)):
        v = "✓" if eff(r) >= 30 else ("!" if eff(r) >= 20 else "✗")
        print(f"  {v} {r['tag']:<52} MFU† {eff(r):5.1f}%  (6N {r['mfu']:5.1f}%)")
    if judge:
        top = max(judge, key=eff)
        print(f"\n最好成绩：{top['tag']} → **{eff(top):.1f}% MFU†**")
    print("- Megatron+TE 跑 dense Llama 在 H200 上的正常水位：**35-45%**（单机 8B）；"
          "70B 双机 TP8×PP2 **25-38%**（含流水线气泡）。")
    print("- **低于 20% 就别调参了**：回去查 hw.jsonl / coll.jsonl 的 busbw 和 nvidia-smi 的 power.limit。")
    print("- 抖动 >5% 的组：有卡降频或 NUMA 跨节点，配合 dmon 看功耗/时钟。")
    d = dmon(logs)
    if d:
        print("\n长稳遥测（逐卡，非跨卡平均）：")
        print(d)
        print("  确认没触发节流：nvidia-smi -q -d PERFORMANCE | grep -A8 'Clocks Event Reasons'")
    print_fingerprint(fp)
    print(f"\n**口径**：显存按 nvidia-smi 实测 {VRAM_GIB} GiB（143771 MiB）计，不写 141 GiB。"
          "容器内 torch 报 `total_memory` 约 139.8 GiB，差额约 0.6 GiB 是 ECC/驱动预留与 CUDA 上下文占用，"
          "不是卡有问题；比较「还剩多少显存」用 nvidia-smi，比较「torch 能分配多少」用 total_memory。"
          "docker 的 --gpus all 走 nvidia runtime，nerdctl 走 CDI/toolkit；"
          "镜像 digest 取值方式不同，跨引擎比较以 digest 为准；"
          "mock 数据 + 随机初始化，不含数据管线与 checkpoint 落盘开销 —— "
          "本表是算力/通信上限，不是端到端吞吐。")
    return 0


def compare(a: str, b: str):
    A, B = json.loads(Path(a).read_text(encoding="utf-8")), json.loads(Path(b).read_text(encoding="utf-8"))
    ia = {x["item"]: x for x in A.get("items", [])}
    ib = {x["item"]: x for x in B.get("items", [])}
    keys = [k for k in ia if k in ib and ia[k]["detail"] != ib[k]["detail"]]
    print(f"\n{A.get('host')} vs {B.get('host')}")
    print(f"驱动：A={A.get('driver', '?')}  B={B.get('driver', '?')}")
    crit = {"驱动", "镜像建议", "GPU 数量", "显存容量一致", "GPU 型号一致", "卡间互联",
            "RDMA 网卡", "memlock", "toolkit", "GPU 运行时", "ECC", "NVLink 链路数",
            "NVLink 误码", "retired_pages", "Xid"}

    def rend(x, n=78):
        if x.get("level") == "SKIP":
            return "SKIP（字段不支持）"        # 三态：字段不可用不当成 FAIL，也不当成差异
        return str(x.get("detail", ""))[:n]

    bad = [k for k in keys if k in crit
           and not (ia[k].get("level") == "SKIP" and ib[k].get("level") == "SKIP")]
    if bad:
        print("!! 影响双机可比性的差异（先修再测）：")
        for k in bad:
            print(f"   {k:<14} A= {rend(ia[k])}\n   {'':<14} B= {rend(ib[k])}")
    for k in [x for x in keys if x not in crit]:
        print(f"   {k:<14} A= {rend(ia[k], 52)} | B= {rend(ib[k], 52)}")
    if not keys:
        print("两台环境一致 ✓")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/root/llm-bench/logs")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args()
    raise SystemExit(compare(*a.compare) if a.compare else bench(Path(a.logs)))


if __name__ == "__main__":
    main()
