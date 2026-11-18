# -*- coding: utf-8 -*-
"""由 metrics.json 生成离线 HTML 图表报告 + 汇总 CSV。"""
import csv
import json
from html import escape
from pathlib import Path

from svgcharts import C, hbar, vbar, lines, scatter_status, mini_progress, nice_max

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
RUN = D["run"]
HW = D["hw"]
# MFU 分母：H200 SXM (700W / HGX 8 卡 NVSwitch) BF16 Tensor Core dense 理论峰值。
# NVIDIA 规格表列 1979 TFLOPS 为「with sparsity」，稠密训练应取其一半 989.5。
PEAK_THEORY = 989.5
PEAK_GEMM = HW["gemm_bf16"].get("tflops", 690.0)   # 04-hw 单卡 n=8192 GEMM 探针实测
PEAK = PEAK_THEORY
GEMM_RATIO = PEAK_GEMM / PEAK_THEORY * 100
CAP_GIB = HW["gpu"].get("cap_gib", [139.8])[0]

cases = D["cases"]
dense = [c for c in cases if c["group"] == "dense8B"]
moe = [c for c in cases if c["group"] == "moe"]
limit = D["limit"]
soak = D["soak"]
by_tag = {c["tag"]: c for c in cases}


def med(c, key="tflops"):
    return (c.get(key) or {}).get("median") or 0


def gib(c):
    v = (c.get("mem") or {}).get("max_reserved_mb")
    return v / 1024 if v else 0


def mfu(c):
    """标准 MFU：模型有效 FLOPs ÷ 设备理论 dense 峰值。"""
    return med(c) / PEAK_THEORY * 100


def eff_gemm(c):
    """辅助口径：相对本机实测 GEMM 可达上限的效率（不是 MFU）。"""
    return med(c) / PEAK_GEMM * 100


def short(tag):
    return tag.replace("w8_8B_", "").replace("w8_MOE_", "MOE:")


def fmt(v, n=1):
    return f"{v:,.{n}f}"


# --------------------------------------------------------------- 数据准备
ranked = sorted(dense, key=lambda c: -med(c))
best = ranked[0]

# 档位配色（按扫描维度）
def case_color(c):
    if c["tag"].endswith("fp8"):
        return C["orange"]
    if c.get("tp") in (1, 4):
        return C["purple"]
    if c.get("seq", 4096) != 4096:
        return C["cyan"]
    if c.get("mbs") == 4 and c.get("gb") == 8:
        return C["green"]
    if c.get("mbs", 1) == 1 and c.get("gb") == 8:
        return C["slate"]
    return C["blue"]


# NCCL：修正 busbw（hw_probe.py 未除以 iters）
ops = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]
msg_sizes = [8, 65536, 1048576, 67108864, 1073741824, 8589934592]
msg_labels = ["8B", "64K", "1M", "64M", "1G", "8G"]
coll = {op: {r["msg_bytes"]: r for r in HW["collectives"] if r.get("op") == op and r.get("status") == "OK"}
        for op in ops}
op_colors = {"all_reduce": C["blue"], "all_gather": C["orange"],
             "reduce_scatter": C["green"], "all_to_all": C["purple"]}
for op in ops:
    for r in coll[op].values():
        r["fixed"] = round(r["busbw_gbs"] * r["iters"], 1)

# 长稳曲线按进度归一化到 100 点
def norm_progress(series, key="mean", n=101):
    if not series:
        return [None] * n
    out = []
    for i in range(n):
        p = i / (n - 1) * (len(series) - 1)
        lo = int(p)
        hi = min(lo + 1, len(series) - 1)
        v = series[lo][key] + (series[hi][key] - series[lo][key]) * (p - lo)
        out.append(round(v, 2))
    return out


soak_names = [("dense_2h", "Dense 8B · 2h", C["blue"]),
              ("moe_30m", "MoE EP8 · 30min", C["purple"]),
              ("highmem_1h", "Highmem · 1h(未完成)", C["orange"])]

findings = [
    ("F1", "P0", "FP8 档未真正启用 FP8 计算",
     "fp8 与 bf16 两档日志均为 <code>fp8 = None</code>、<code>params_dtype = bfloat16</code>；首步 loss / grad norm 逐位相同；稳态吞吐 345.0 vs 344.1 TFLOP/s/GPU（+0.26%）。日志中 <code>fp8_recipe = delayed</code> 已写入，说明脚本侧参数注入链路已通，缺的是 TransformerEngine 层的 fp8 开关；04-hw 的 fp8 GEMM 探针同样 SKIP（<code>_scaled_mm 不可用</code>）。",
     "FP8 性能结论不成立，且缺一条可用的 FP8 硬件基线。",
     "在已生效的 <code>--fp8-recipe=delayed</code> 之上补 <code>--fp8-hybrid</code> 等真正开启 TE fp8 计算的开关（并确认 amax history 配置）；先修复 fp8 GEMM 探针，再重测两档。"),
    ("F2", "P0", "08-fault 三个故障用例全部无效，2 个 PASS 为假阳性",
     "fault-oom / fault-kill / fault-timeout 三个日志各含 4 次 <code>AssertionError: ... require setting CUDA_DEVICE_MAX_CONNECTIONS to 1</code>，进程在启动阶段即崩溃；<code>fault.jsonl</code> 仅以 rc≠0 判定 PASS，未匹配预期错误特征。",
     "OOM 注入、杀 rank、NCCL 超时三项恢复能力均未被真正验证。",
     "注入脚本导出 <code>CUDA_DEVICE_MAX_CONNECTIONS=1</code>；PASS 判定改为匹配 OOM / Watchdog / SIGKILL 等特征文本，再重跑 3 个用例。"),
    ("F3", "P1", "hw_probe.py 的 busbw 少除了迭代次数",
     "代码 <code>busbw = payload_b * factor / sec / 1e9</code>，其中 sec 是 iters 次集合通信的总耗时；修正值 = 记录值 × iters。修正后 1G~8G 消息各算子收敛到 330~370 GB/s，原始记录仅 3.3~12.2 GB/s。",
     "NCCL 带宽被低估 30~100 倍，且因不同消息下 iters 不同（200/100/30）导致曲线不可比。",
     "改为 <code>payload_b * factor * iters / sec / 1e9</code>，或直接由 <code>latency_us</code> 反推。"),
    ("F4", "P1", "09-soak 高显存长稳未完成",
     "只有 <code>.ok_moe_soak</code> 与 <code>.ok_w8_8B_soak_...</code> 两个完成标记；highmem 无 .ok，soak-progress.log 停在 <code>21:00/1:00:00（35%）</code>，results.jsonl 也仅 2 条记录。",
     "1 小时高显存长稳目标未达成，仅能作为 21 分钟的部分样本；且该档位本身也没造成显存压力（见 F11）。",
     "补跑至 60 分钟并采集结束健康快照；或在报告中显式标注为 unfinished。"),
    ("F5", "P1", "运行元数据未覆盖全部阶段",
     "<code>index.md</code> 只记录到 06-train-limit，缺 06-train-moe 与 09-soak；<code>00-run-info.txt</code> 的 end_time=19:16:44，但 soak 日志最晚到 22:09。",
     "阶段耗时与总时长统计不完整，无法据此核算端到端资源占用。",
     "收尾阶段回写 index.md 与 run-info，补齐 06-train-moe / 09-soak 的时间与状态。"),
    ("F6", "P1", "MoE 专家负载均衡指标缺失",
     "<code>expert-load-timeseries.csv</code> 7 个采样点全为 null（注明「未解析到 expert load」）；moe.jsonl / moe-soak.jsonl 的 <code>expert_load_max_over_mean</code>、<code>imbalance_max_over_mean</code>、<code>imbalance_drift</code> 均为 null。",
     "无法判断 EP8 路由倾斜，也无法解释 MoE 长稳 11.2% 的吞吐漂移。",
     "开启专家 token 计数日志或解析 router 输出，补齐 max/mean 与漂移序列。"),
    ("F7", "P2", "07-io 的 dataloader 路径被跳过",
     "<code>io.jsonl</code>：dataloader status=SKIP，原因是无 <code>data/*.bin + .idx</code>，mock-data 路径不含 dataloader/tokenizer。",
     "IO 结论只覆盖 checkpoint 读写，不覆盖真实数据加载。",
     "准备小规模真实 bin/idx 数据集后补测。"),
    ("F8", "P2", "短测仅 15 步，稳态样本 13 步",
     "每档 15 个 iteration，本报告剔除首步（初始化）与末步（收尾）后取中位数；mbs4/GBS8 短测 344.1 与 2h soak 344.3 仅差 0.06%，一致性已验证。",
     "样本量偏小，对抖动敏感的配置（MoE）估计不稳。",
     "短测提高到 30~50 步，并保留 warmup 剔除规则。"),
    ("F9", "P2", "04-hw / 短测 dmon 采样包含启动空转",
     "04-hw dmon 平均 SM 48.7%、功耗 190.6 W；短测各档 SM 均值仅 19.8~96.1%，且吞吐最高的 mbs4/GBS8 档 SM 反而只有 19.8%；soak 稳态 SM 97.3%、功耗 643.9 W。短测采样窗口被部署/初始化空转严重稀释。",
     "短测 GPU 利用率均值偏低，不宜直接引用为训练态利用率。",
     "按 iteration 时间戳裁剪 dmon 采样窗口，或仅使用 soak 段的利用率数据。"),
    ("F10", "P1", "hw_probe.py 的 HBM / L2 / PCIe 带宽多乘了 iters",
     "<code>timed()</code> 返回的是<b>平均单次耗时</b>（<code>elapsed_time / it</code>），但 hbm/l2/pcie 的公式写成 <code>m*2*it/t/1e12</code>，重复乘了 it；gemm_bf16 未乘 it 因而正确（690 TFLOP/s，记录 iters=500）。按饱和 iters 估算：HBM read 1813.26÷500≈3.63 TB/s、copy 2147.69÷500≈4.30 TB/s（符合 H200 HBM3e）；PCIe h2d 1478÷50≈29.6 GB/s、d2h 1044.9÷50≈20.9 GB/s。",
     "原始 HBM / PCIe 带宽虚高 50~500 倍，无法直接用于 IO 瓶颈判断；hw.jsonl 未记录这几项的 iters，只能估算。",
     "去掉公式中的 <code>* it</code>，并在 emit 中补 <code>iters</code> 字段；修后重跑 04-hw 探针。"),
    ("F11", "P1", "highmem 长稳的档位与判据自相矛盾：只占 29% 却被记为完成",
     "该档实际 tag 是 <code>HIGHMEM_8B_tp8_pp1_<b>mbs1</b>_s16384</code>，<code>memused-w8_highmem.csv</code> 峰值 "
     "<b>41696 MiB = 29.0%</b>（分母 143771 MiB）。成因链：<code>limit.jsonl</code> 4 行全是 "
     "<code>\"model\":\"MOE_B\"</code>（静态 84.7 GiB/卡），与 dense 8B TP8（静态约 17 GiB/卡）差 5 倍；"
     "<code>highmem_boundary()</code> 找不到 <code>seq=16384</code> 的 OK 行，回退按 <code>max_tokens/GPU=8192</code> "
     "换算 <code>8192÷16384=0</code> 判 <code>infeasible</code>，于是强制 <code>mbs=1</code>。"
     "而脚本默认的 <code>HIGHMEM_MBS=16</code> 同样不成立：16×16384=262144 tokens/卡，按实测三点拟合 "
     "<code>peak ≈ 27.5 GiB + 0.83 MiB/token</code>（16K→40.7、32K→53.1、128K→137.3 GiB）需约 240 GiB，必然 OOM。"
     "更关键的是判定只查 <code>.ok_highmem</code> 是否存在，<code>peak_pct</code> 算出来却没参与判断。",
     "该档没有制造任何显存压力（29% 甚至低于 128K 短测的 97.8%），无法回答「能否长期贴着显存上限跑」；"
     "同时「跑完 = OK」的判定把 29% 记成通过，与 RUNBOOK 自定的 75~92% 口径冲突。",
     "改为同模型标定：每档 4 步试跑、倍增爬坡、有 OK/OOM 两点时线性插值直取目标 mbs，命中 75~92% 带即停；"
     "跨模型边界只保留「模型一致」场景（双机 110B），其余记 <code>cross-model</code> 不借用；"
     "判定改用实测 <code>peak_pct</code> 收口，&lt;75% 记 <code>WARN_LOW</code> 并撤销 <code>.ok_highmem</code>。"),
]


def table(headers, rows, cls=""):
    th = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def card(title, sub, body, anchor=None, tag=""):
    return (f'<section class="card" {"id="+anchor if anchor else ""}>'
            f'<div class="card-h"><div><h2>{title}</h2><p>{sub}</p></div>{tag}</div>{body}</section>')


# --------------------------------------------------------------- 图表 1：总览
kpi = [
    ("峰值单卡算力", f"{fmt(med(best))}", "TFLOP/s/GPU · tp1/DP8, GBS32, mbs2", C["blue"]),
    ("最高 MFU", f"{fmt(mfu(best))}%", f"H200 SXM BF16 dense 理论峰值 {PEAK_THEORY} TFLOP/s 为分母", C["green"]),
    ("最长连续长稳", f"{soak['dense_2h']['iters_logged']:,}", "步 / 约 2 小时，0 个慢步", C["purple"]),
    ("2h 吞吐漂移", f"+{soak['dense_2h'].get('drift_pct',0):.2f}%", "前 10% vs 后 10% 均值", C["cyan"]),
    ("硬件健康增量", "0", "ECC / NVLINK / XID / SXID 全部为 0", C["green"]),
    ("数据可信缺口", f"{len(findings)}", "项，其中 P0 两项", C["red"]),
]
kpi_html = "".join(
    f'<div class="kpi" style="--kc:{col}"><div class="kpi-l">{t}</div>'
    f'<div class="kpi-v">{v}</div><div class="kpi-s">{s}</div></div>' for t, v, s, col in kpi)

phase_rows = [[escape(p["phase"]), escape(p["duration"]), escape(p["status"])] for p in D["phases"]]
phase_rows += [
    ["06-train-moe", "—（index 未记录）", "PASS（4 档，日志时间 18:59~19:03）"],
    ["09-soak", "—（index 未记录）", "2 完成 / 1 未完成，日志时间 19:16~22:09"],
]

status_bar = mini_progress(20, 2, 3, 1, labels=["有效通过", "预期边界", "无效/失败", "未完成"])

FIXED = {"F1", "F2", "F3", "F6", "F10", "F11"}   # 代码层已修，历史 run 需重跑才能关闭
find_rows = [[f'<b>{i}</b>', f'<span class="pri {p.lower()}">{p}</span>',
              ('<span class="fixed">已修·待重跑</span> ' if i in FIXED else '') + f"<b>{t}</b>",
              ev, imp, fix]
             for i, p, t, ev, imp, fix in findings]

# --------------------------------------------------------------- 图表 2：dense 全景
bar_rows = [{"label": short(c["tag"]), "value": med(c), "color": case_color(c),
             "tip": f'{c["tag"]} · step {med(c,"step_ms"):.1f} ms · {gib(c):.1f} GiB'}
            for c in ranked]
chart_dense = hbar(bar_rows, label_w=372, row_h=28, unit="TFLOP/s/GPU",
                   max_val=1050, ref=PEAK_THEORY,
                   ref_label=f"H200 BF16 dense 理论峰值 {PEAK_THEORY} TFLOP/s（MFU 分母）",
                   desc="Dense 8B 各档稳态吞吐")

dense_table_rows = []
for c in sorted(dense, key=lambda x: (x.get("tp", 8), x.get("seq", 4096), x.get("mbs", 1))):
    dp = 8 // (c.get("tp") or 8)
    accum = (c.get("gbs") or 8) / ((c.get("mbs") or 1) * dp)
    dense_table_rows.append([
        f'<span class="dot" style="background:{case_color(c)}"></span>{escape(short(c["tag"]))}',
        f'{c.get("tp")}/{c.get("pp")}', f'{c.get("mbs")}', f'{c.get("seq"):,}', f'{c.get("gb") or c.get("gbs")}',
        f'{dp}', f'{accum:g}', f'<b>{fmt(med(c,"step_ms"))}</b>', f'<b>{fmt(med(c))}</b>',
        f'{fmt(mfu(c))}%', f'{fmt(gib(c))}', f'{c["loss_first"]:.3g} → {c["loss_last"]:.3g}',
        '<span class="ok">OK</span>'])

mem_rows = [{"label": short(c["tag"]), "value": gib(c),
             "color": C["red"] if gib(c) > CAP_GIB * 0.95 else (C["amber"] if gib(c) > 60 else C["blue"]),
             "tip": f'{c["tag"]}: {gib(c):.1f} GiB / {CAP_GIB} GiB'}
            for c in sorted(dense, key=lambda x: -gib(x))]
chart_mem = hbar(mem_rows, label_w=372, row_h=28, unit="GiB / GPU", max_val=160,
                 ref=CAP_GIB, ref_label=f"H200 容量 {CAP_GIB} GiB", desc="各档显存峰值")

cmp_tags = ["w8_8B_tp1_pp1_mbs2_s4096_gb32_bf16",
            "w8_8B_tp8_pp1_mbs4_s4096_gb8_bf16",
            "w8_8B_tp8_pp1_mbs1_s4096_gb8_bf16",
            "w8_MOE_8ep_1tp_alltoall_mbs1_s4096"]
mfu_cmp_rows = [[escape(short(t)), fmt(med(by_tag[t])),
                 f'{eff_gemm(by_tag[t]):.1f}%（首版误用，非 MFU）',
                 f'<b>{mfu(by_tag[t]):.1f}%</b>'] for t in cmp_tags]

# --------------------------------------------------------------- 图表 3：扩展性
tp_cases = {c["tp"]: c for c in dense if c.get("seq") == 4096 and c.get("mbs") == 2 and c.get("gb") == 32}
tp_x = sorted(tp_cases)
chart_tp = lines(tp_x, [
    {"name": "吞吐 TFLOP/s/GPU", "color": C["blue"], "values": [med(tp_cases[t]) for t in tp_x]},
    {"name": "显存峰值 GiB", "color": C["orange"], "values": [gib(tp_cases[t]) for t in tp_x], "axis": "right"},
], w=840, h=300, xlabels=[f"TP{t}" for t in tp_x] + [""], x_tick_values=tp_x,
    unit="TFLOP/s/GPU", unit2="GiB", ymax=600, y2max=110, xlabel="张量并行度（固定 mbs2 / seq4096 / GBS32）",
    desc="TP 扩展性：吞吐与显存权衡")
tp_rows = [[f"TP{t}", f'{8//t}', fmt(med(tp_cases[t])), f'{fmt(mfu(tp_cases[t]))}%',
            fmt(gib(tp_cases[t])), fmt(med(tp_cases[t], "step_ms")),
            ("基准" if t == 1 else f'{med(tp_cases[t])/med(tp_cases[1])*100-100:+.1f}%')]
           for t in tp_x]

mbs_cases = [c for c in dense if c.get("tp") == 8 and c.get("seq") == 4096 and not c["tag"].endswith("fp8")]
mbs_cases = sorted(mbs_cases, key=lambda c: (c.get("gb") or 8, c.get("mbs") or 1))
chart_mbs = vbar([f'GBS{c.get("gb")} / mbs{c.get("mbs")}' for c in mbs_cases],
                 [{"name": "稳态吞吐 TFLOP/s/GPU", "color": C["blue"], "values": [med(c) for c in mbs_cases]}],
                 w=840, h=300, unit="TFLOP/s/GPU", max_val=420,
                 ylabel="TFLOP/s/GPU", desc="TP8 下 mbs / GBS 组合对比")

seq_cases = sorted([c for c in dense if c.get("tp") == 8 and c.get("mbs") == 1 and c.get("gb") == 8],
                   key=lambda c: c["seq"])
seq_x = [c["seq"] for c in seq_cases]
chart_seq = lines(seq_x, [
    {"name": "吞吐 TFLOP/s/GPU", "color": C["blue"], "values": [med(c) for c in seq_cases]},
    {"name": "单步耗时 ms（右轴，对数）", "color": C["orange"], "values": [med(c, "step_ms") for c in seq_cases], "axis": "right"},
], w=880, h=320, xlog=True, y2log=True, ymax=450, y2min=500, y2max=100000,
    x_tick_values=seq_x, xlabels=["2K", "4K", "8K", "32K", "128K"],
    unit="TFLOP/s/GPU", unit2="ms", xlabel="序列长度（对数轴）", desc="序列长度扫描")

fp8c = by_tag["w8_8B_tp8_pp1_mbs4_s4096_gb8_fp8"]
bf16c = by_tag["w8_8B_tp8_pp1_mbs4_s4096_gb8_bf16"]
chart_prec = vbar(["BF16", "FP8(名义)"],
                  [{"name": "TFLOP/s/GPU", "color": C["slate"], "values": [med(bf16c), med(fp8c)]}],
                  w=500, h=250, unit="TFLOP/s/GPU", max_val=400, legend=False,
                  desc="精度对比")

# --------------------------------------------------------------- 图表 4：MoE
moe_topo = ["ep8_tp1", "ep2_tp4"]
def moe_case(topo, disp):
    for c in moe:
        if ("8ep" in c["tag"] and topo.startswith("ep8") or "2ep" in c["tag"] and topo.startswith("ep2")) and disp in c["tag"]:
            return c
    return None
moe_groups = [
    {"name": "alltoall", "color": C["blue"], "values": [med(moe_case(t, "alltoall")) for t in moe_topo]},
    {"name": "allgather", "color": C["orange"], "values": [med(moe_case(t, "allgather")) for t in moe_topo]},
]
chart_moe = vbar(["EP8 / TP1", "EP2 / TP4"], moe_groups, w=760, h=300,
                 unit="TFLOP/s/GPU", max_val=220, desc="MoE 拓扑 × dispatcher 吞吐")
moe_meta = {(m["topology"], m["dispatcher"]): m for m in D["moe_meta"]}
moe_rows = []
for c in sorted(moe, key=lambda x: -med(x)):
    topo = "ep8_tp1" if "8ep" in c["tag"] else "ep2_tp4"
    disp = "alltoall" if "alltoall" in c["tag"] else "allgather"
    m = moe_meta.get((topo, disp), {})
    moe_rows.append([escape(short(c["tag"])), topo, disp, f'{med(c,"step_ms"):.1f}', fmt(med(c)),
                     f'{m.get("mfu_active","—")}%', fmt(gib(c)), f'{m.get("accum","—")}',
                     f'{m.get("active_params",0)/1e9:.2f}B', '<span class="ok">OK</span>'])

# --------------------------------------------------------------- 图表 5：OOM 边界
lim_points = []
for r in limit:
    lane = 0 if r["seq"] == 4096 else 1
    lim_points.append({"x": r["tokens_per_gpu"], "y": lane, "label": f'mbs{r["mbs"]}/s{r["seq"]//1024}K',
                       "status": r["status"], "color": C["green"] if r["status"] == "OK" else C["red"],
                       "lane_label": "mbs 扫描" if lane == 0 else "seq 扫描"})
chart_limit = scatter_status(lim_points, w=840, h=270, xmax=20000, desc="MoE_B 显存边界")
lim_rows = [[f'mbs{r["mbs"]} · seq{r["seq"]:,}', f'{r["tokens_per_gpu"]:,}',
             f'{r["step_s"]:.4f}' if r["step_s"] else "—",
             f'{r["static_gb_est"]}', f'{r["status"]}',
             escape(r.get("error_class", "") or "—"),
             '<span class="ok">通过</span>' if r["status"] == "OK" else '<span class="warn">预期边界</span>']
            for r in limit]

# --------------------------------------------------------------- 图表 6：NCCL
nccl_series = []
for op in ops:
    nccl_series.append({"name": op, "color": op_colors[op],
                        "values": [coll[op][m]["fixed"] if m in coll[op] else None for m in msg_sizes]})
raw_ar = [coll["all_reduce"][m]["busbw_gbs"] if m in coll["all_reduce"] else None for m in msg_sizes]
nccl_series.append({"name": "all_reduce 原始记录值(未修正)", "color": C["slate"],
                    "values": raw_ar, "dash": True})
chart_nccl = lines(msg_sizes, nccl_series, w=880, h=330, xlog=True,
                   x_tick_values=msg_sizes, xlabels=msg_labels, ymax=420,
                   unit="GB/s（busbw，修正后）", xlabel="集合通信消息大小（字节，对数轴）",
                   desc="NCCL 集合通信带宽（修正后）")
nccl_rows = []
for m, lab in zip(msg_sizes, msg_labels):
    row = [lab]
    for op in ops:
        r = coll[op].get(m)
        row.append(("—" if not r else ("<1" if r["fixed"] < 1 else f'{r["fixed"]:,.0f}')))
    ar = coll["all_reduce"].get(m)
    row.append(f'{ar["latency_us"]:,.1f}' if ar else "—")
    nccl_rows.append(row)

hbm_read = next((x for x in HW["hbm"] if x.get("op") == "read"), {})
hbm_copy = next((x for x in HW["hbm"] if x.get("op") == "copy"), {})
h2d = next((x for x in HW["pcie"] if x.get("dir") == "h2d"), {})
d2h = next((x for x in HW["pcie"] if x.get("dir") == "d2h"), {})

# --------------------------------------------------------------- 图表 7：长稳
soak_series = []
for key, name, col in soak_names:
    s = soak.get(key)
    if s:
        soak_series.append({"name": name, "color": col, "values": norm_progress(s["tflops_series"])})
chart_soak = lines(list(range(101)), soak_series, w=880, h=330, ymin=180, ymax=390,
                   xticks=4, xlabels=["0%", "25%", "50%", "75%", "100%"],
                   x_tick_values=[0, 25, 50, 75, 100], unit="TFLOP/s/GPU",
                   xlabel="长稳进度（按各自总步数归一化）", desc="长稳吞吐时序")
soak_rows = []
for key, name, col in soak_names:
    s = soak[key]
    t = s["tflops"]
    soak_rows.append([
        f'<span class="dot" style="background:{col}"></span>{name}', f'{s["iters_logged"]:,}',
        f'{s["first_ts"]} → {s["last_ts"]}', f'{fmt(t["mean"])}', f'{fmt(t["p95"])}',
        f'{t["cv_pct"]}%', f'{s.get("drift_pct",0):+.2f}%', f'{s["slow_steps"]}',
        f'{s["nan_total"]} / {s["skipped_total"]}',
        '<span class="ok">完成</span>' if s["finished"] else '<span class="warn">未完成 35%</span>'])
hms = D["highmem_series"]
chart_hmmem = lines(list(range(len(hms))), [
    {"name": "8 卡均值 GiB", "color": C["blue"], "values": [r["mean_mib"] / 1024 for r in hms]},
    {"name": "单卡峰值 GiB", "color": C["orange"], "values": [r["max_mib"] / 1024 for r in hms]},
], w=880, h=260, unit="GiB", ymax=50, xticks=4, xlabels=["0", "25%", "50%", "75%", "100%"],
    x_tick_values=[0, len(hms) // 4, len(hms) // 2, len(hms) * 3 // 4, len(hms) - 1],
    xlabel="highmem 长稳进度", desc="highmem 显存占用趋势")

# 显存占用 vs tokens/卡：解释「29% 是怎么来的、85% 需要多少 tokens」
# 取 8B TP8 各 seq 短测的 torch max reserved，再并上 highmem 档的 nvidia-smi 实测峰值
_cap = (D.get("limit_summary") or {}).get("vram_gib_per_gpu") or 140.4
_short = {}                                    # 15 步短测：torch max reserved
for _c in D["cases"]:
    _m = _c.get("mem") or {}
    if "tp8_pp1" in _c["tag"] and _c["tag"].endswith("bf16") and _m.get("max_reserved_mb"):
        _t = _c["mbs"] * _c["seq"]
        _short[_t] = max(_short.get(_t, 0), _m["max_reserved_mb"] / 1024)
_hp = D.get("highmem_peak_mib") or 0           # highmem 长稳 mbs=1 × seq=16384：nvidia-smi used
_toks = sorted(_short)
_n = len(_toks)
chart_hmscale = lines(
    _toks,
    [{"name": "15 步短测（torch max reserved）", "color": C["blue"],
      "values": [_short[t] for t in _toks]},
     {"name": "21 分钟长稳（nvidia-smi 峰值）", "color": C["purple"],
      "values": [(_hp / 1024 if t == 16384 else None) for t in _toks]},
     {"name": f"75% 下限 {_cap*0.75:.0f} GiB", "color": C["green"], "values": [_cap * 0.75] * _n, "dash": True},
     {"name": f"92% 上限 {_cap*0.92:.0f} GiB", "color": C["orange"], "values": [_cap * 0.92] * _n, "dash": True}],
    w=880, h=290, xlog=True, unit="GiB", ymax=_cap * 1.06,
    xlabels=[f"{t // 1024}K" for t in _toks], x_tick_values=_toks,
    xlabel="tokens/卡 = mbs × seq（对数轴）", desc="显存占用与 tokens 的关系")
_gap = (_hp / 1024) - _short.get(16384, 0) if _hp else 0
_need_lo = (_cap * 0.75 * 1024 - 27500) / 0.83
_need_hi = (_cap * 0.92 * 1024 - 27500) / 0.83

# --------------------------------------------------------------- 图表 8：IO / 故障
io = next((x for x in D["io"] if x.get("kind") == "checkpoint"), {})
chart_io = vbar(["顺序写 checkpoint", "顺序读 checkpoint"],
                [{"name": "MB/s", "color": C["cyan"], "values": [io.get("write_mbs", 0), io.get("read_mbs", 0)]}],
                w=620, h=270, unit="MB/s", max_val=35000, legend=False, desc="checkpoint IO 吞吐")
fault_rows = []
for f in D["fault"]:
    cls = "ok" if f["status"] == "PASS" else "bad"
    fault_rows.append([escape(f["case"]), f'<span class="{cls}">{f["status"]}</span>',
                       f'{f["rc"]}', escape(f["note"]),
                       '<span class="bad">断言失败导致进程启动即崩</span>'])

# --------------------------------------------------------------- 结论
conclusions = [
    ("性能天花板", f"最优档 <b>{short(best['tag'])}</b> 达 {fmt(med(best))} TFLOP/s/GPU（MFU {fmt(mfu(best))}%），靠 DP8 无 TP 通信 + 每卡 16K tokens 的大 GEMM；代价是显存 {fmt(gib(best))} GiB。"),
    ("TP 不是越大越好", f"同配置 TP1→TP4→TP8 吞吐 {fmt(med(tp_cases[1]))}→{fmt(med(tp_cases[4]))}→{fmt(med(tp_cases[8]))}（-39.7%），但显存从 {fmt(gib(tp_cases[1]))} 降到 {fmt(gib(tp_cases[8]))} GiB（-81.5%）。TP8 的价值在长序列/大模型的显存可行性，而非算力效率。"),
    ("序列长度拐点", f"TP8/GBS8 下 2K→32K 吞吐从 {fmt(med(seq_cases[0]))} 升到 {fmt(med(seq_cases[3]))}，128K 回落到 {fmt(med(seq_cases[4]))}（-20.5%），单步 {fmt(med(seq_cases[4],'step_ms'),0)} ms、显存 {fmt(gib(seq_cases[4]))} GiB（≈98% 容量）；32K 是吞吐拐点，128K 不建议作为生产档。"),
    ("MoE 拓扑决定一切", f"EP8/TP1 的 {fmt(med(moe_case('ep8_tp1','alltoall')))} TFLOP/s/GPU 是 EP2/TP4（{fmt(med(moe_case('ep2_tp4','alltoall')))}）的 4.7 倍；dispatcher 影响仅 2~12%。EP2/TP4 需 4 次梯度累加（accum=4）是主要损耗来源。"),
    ("长稳表现分化", f"Dense 2h 12,503 步 CV {soak['dense_2h']['tflops']['cv_pct']}%、漂移 +{soak['dense_2h']['drift_pct']}%、0 慢步、SM 97.3%、功耗 643.9 W，健康零增量；MoE 30min 漂移 +{soak['moe_30m']['drift_pct']}%、CV {soak['moe_30m']['tflops']['cv_pct']}%、20 个慢步，且缺专家负载数据，需补测后才能判定稳定性。"),
    ("显存边界", f"MOE_B 静态占用 {D['limit_summary']['static_gb_est']} GiB / 卡容量 {CAP_GIB} GiB，余量 {D['limit_summary']['static_margin_gib']} GiB；mbs4@4096 与 mbs2@8192 同为 16,384 tokens/GPU 均 OOM，边界由「每卡 token 数」而非 mbs 或 seq 单独决定，位于 8K~16K tokens/GPU。"),
]
concl_html = "".join(f'<div class="concl"><div class="concl-t">{t}</div><div>{v}</div></div>' for t, v in conclusions)

actions = [
    ["P0", "修复 FP8 参数链与 fp8 GEMM 探针，重测 BF16/FP8 对照", "预计解锁 H200 FP8 算力；当前档位差 0.26% 属无效结果"],
    ["P0", "为故障注入脚本补 <code>CUDA_DEVICE_MAX_CONNECTIONS=1</code>，并改用错误特征判定 PASS", "恢复 3 项容错能力验证的可信度"],
    ["P1", "修正 hw_probe.py 两处 iters 口径错误（busbw 少除、HBM/PCIe 多乘）并重跑 04-hw", "NCCL 带宽修正为 330~370 GB/s、HBM ≈3.6~4.3 TB/s，恢复硬件基线可用性"],
    ["P1", "补跑 highmem 1h soak、补齐 index.md / run-info 元数据", "消除未完成项与阶段统计缺口"],
    ["P1", "开启 MoE 专家负载日志，补齐 max/mean 与漂移", "定位 MoE 长稳 +11.2% 漂移与 20 个慢步的根因"],
    ["P2", "短测步数 15 → 30~50；按 iteration 时间窗裁剪 dmon；补真实数据集重测 dataloader", "提高短测统计置信度与利用率数据可用性"],
]

# --------------------------------------------------------------- HTML
CSS = """
:root{--bg:#f5f7fa;--card:#fff;--bd:#e2e8f0;--tx:#1e293b;--mu:#64748b;--blue:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
 font-size:14px;line-height:1.65}
header{background:linear-gradient(120deg,#0f172a,#1e3a8a 65%,#1d4ed8);color:#fff;padding:38px 0 30px}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
header h1{margin:0 0 8px;font-size:27px;letter-spacing:.3px}
header p{margin:3px 0;color:#c7d7f5;font-size:13px}
.hmeta{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.hmeta span{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);
 padding:4px 10px;border-radius:20px;font-size:12px}
nav{position:sticky;top:0;z-index:9;background:rgba(255,255,255,.94);backdrop-filter:blur(8px);
 border-bottom:1px solid var(--bd);padding:10px 0}
nav .wrap{display:flex;gap:6px;flex-wrap:wrap}
nav a{color:#475569;text-decoration:none;font-size:12.5px;padding:5px 10px;border-radius:6px}
nav a:hover{background:#eef2ff;color:var(--blue)}
main{padding:24px 0 60px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--kc);
 border-radius:10px;padding:14px 16px}
.kpi-l{font-size:12px;color:var(--mu)}.kpi-v{font-size:25px;font-weight:700;line-height:1.25;color:var(--kc)}
.kpi-s{font-size:11.5px;color:var(--mu);margin-top:2px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:20px 22px;margin:0 0 20px;
 box-shadow:0 1px 2px rgba(15,23,42,.04)}
.card-h{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px;
 border-bottom:1px solid var(--bd);padding-bottom:12px}
.card h2{margin:0;font-size:18px}.card-h p{margin:3px 0 0;color:var(--mu);font-size:12.5px}
.badge{background:#eef2ff;color:#3730a3;font-size:11px;padding:4px 9px;border-radius:14px;white-space:nowrap}
.chart{width:100%;height:auto;display:block;margin:8px 0 4px}
.chart text{font-family:inherit}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:12.3px}
th{background:#f8fafc;color:#475569;text-align:left;font-weight:600;padding:8px 9px;
 border-bottom:2px solid var(--bd);white-space:nowrap}
td{padding:7px 9px;border-bottom:1px solid #f1f5f9;vertical-align:top}
tr:hover td{background:#f8fbff}
table.compact td,table.compact th{padding:5px 7px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.ok{color:#15803d;font-weight:600}.bad{color:#b91c1c;font-weight:600}.warn{color:#b45309;font-weight:600}
.pri{font-size:11px;font-weight:700;padding:2px 7px;border-radius:5px}
.fixed{display:inline-block;background:#dcfce7;color:#15803d;font-size:10.5px;font-weight:700;
 padding:1px 6px;border-radius:4px;margin-right:6px;white-space:nowrap}
.banner{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;border-radius:8px;
 padding:12px 16px;margin:0 0 18px;font-size:12.8px;color:#1e3a8a}
.pri.p0{background:#fee2e2;color:#b91c1c}.pri.p1{background:#ffedd5;color:#c2410c}
.pri.p2{background:#e2e8f0;color:#475569}
code{background:#f1f5f9;padding:1px 4px;border-radius:4px;font-size:11.5px;
 font-family:Consolas,"SF Mono",monospace;color:#0f172a}
.note{background:#f8fafc;border-left:3px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;
 font-size:12.5px;color:#475569;margin:12px 0}
.concl{display:flex;gap:14px;padding:11px 0;border-bottom:1px dashed var(--bd);font-size:13px}
.concl:last-child{border:0}.concl-t{flex:0 0 118px;font-weight:700;color:#1d4ed8}
footer{color:var(--mu);font-size:12px;text-align:center;padding:0 0 40px}
h3{font-size:14px;margin:18px 0 4px;color:#334155}
@media(max-width:780px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
@media print{nav{display:none}body{background:#fff}.card{break-inside:avoid;box-shadow:none}}
"""

head = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>单机 8 卡基准测试图表统计分析 · {escape(RUN.get('run_id',''))}</title>
<style>{CSS}</style></head><body>
<header><div class="wrap"><h1>单机 8 卡基准测试 · 图表统计分析</h1>
<p>run_id：{escape(RUN.get('run_id',''))}　|　主机：{escape(RUN.get('host',''))}　|　范围：{escape(RUN.get('scope',''))} / {escape(RUN.get('phases',''))}</p>
<p>数据来源：<code>2026-11-10_1827_worker-gpu-a_w8_full/</code> 全量日志与 jsonl；统计口径见文末。</p>
<div class="hmeta"><span>8 × NVIDIA H200（139.8 GiB）</span><span>PyTorch 2.6 / CUDA 12.8 / NCCL 2.25.1</span>
<span>驱动 570.172.08</span><span>nerdctl 容器</span><span>IB 传输</span><span>引擎 25.01-py3</span></div>
</div></header>
<nav><div class="wrap">
<a href="#s0">总览与数据质量</a><a href="#s1">Dense 8B 全景</a><a href="#s2">扩展性分析</a>
<a href="#s3">MoE 对比</a><a href="#s4">显存与 OOM 边界</a><a href="#s5">硬件与 NCCL</a>
<a href="#s6">长稳 soak</a><a href="#s7">IO / 故障注入</a><a href="#s8">结论与建议</a>
</div></nav><main class="wrap">
<div class="kpis">{kpi_html}</div>
"""

sec0 = card("① 执行总览与数据质量", "阶段状态、通过率分布与证据驱动的问题排查表",
            f'''{status_bar}
<div class="note">状态口径：共 26 个执行项 —— 20 项有效通过（16 短测档 + 2 个 limit OK + 2 段完成的 soak）、2 项预期 OOM 边界（不计失败）、3 项故障注入用例无效、1 项 highmem soak 未完成，且该档位本身未达高占用（F11）。<b>08-fault 的 2 个 PASS 已被日志证据判定为假阳性</b>，详见下表 F2；另有 fp8 GEMM / dataloader / 专家负载三类指标缺失（F1 / F7 / F6），以及硬件探针的两处口径错误（F3 / F10）。</div>
<h3>阶段记录</h3>{table(["阶段", "耗时", "结果"], phase_rows, "compact")}
<h3>问题排查表（按优先级）</h3>
{table(["#", "级别", "发现", "证据", "影响", "建议"], find_rows)}''',
            anchor="s0", tag='<span class="badge">P0 × 2</span>')

sec1 = card("② Dense 8B 十二档性能全景", "稳态吞吐 = 去掉首步 warmup 与末步收尾后 13 步的中位数",
            f'''{chart_dense}
<div class="note">颜色含义：<b>灰</b>基线档 / <b>绿</b>推荐稳定档 / <b>蓝</b> mbs·GBS 扫描 / <b>青</b>序列长度扫描 / <b>紫</b>TP 扫描 / <b>橙</b>精度对照。虚线为 H200 SXM BF16 Tensor Core <b>dense 理论峰值 {PEAK_THEORY} TFLOP/s</b>（NVIDIA 规格 1979 为含稀疏值，稠密训练取其一半），即 MFU 分母。04-hw 实测 GEMM {PEAK_GEMM:.0f} TFLOP/s 只达理论峰值的 {GEMM_RATIO:.1f}%，是<b>本机可达上限</b>而非硬件峰值，不能当 MFU 分母 —— 本报告首版曾误用该口径，现已全部按 {PEAK_THEORY} 重算（旧值 × {PEAK_GEMM/PEAK_THEORY:.3f} = 新值）。</div>
<h3>MFU 口径修订对照（分母 690 → {PEAK_THEORY}）</h3>
{table(["代表档位", "TFLOP/s/GPU", "相对实测 GEMM 690", "标准 MFU ÷ 989.5"], mfu_cmp_rows, "compact")}
<h3>逐档明细（按 TP / 序列 / mbs 排序）</h3>
{table(["档位", "TP/PP", "mbs", "seq", "GBS", "DP", "accum", "step ms", "TFLOP/s/GPU", "MFU", "显存 GiB", "loss 首→末", "状态"], dense_table_rows, "compact")}
<h3>显存峰值（H200 容量 {CAP_GIB} GiB）</h3>{chart_mem}''',
            anchor="s1", tag='<span class="badge">12 / 12 OK</span>')

sec2 = card("③ 扩展性分析：TP / mbs / 序列 / 精度", "同一张图里同时看算力收益与显存代价",
            f'''<h3>3.1 张量并行（固定 mbs2 / seq4096 / GBS32）</h3>{chart_tp}
{table(["配置", "DP", "TFLOP/s/GPU", "MFU", "显存 GiB", "step ms", "相对 TP1"], tp_rows, "compact")}
<div class="note">TP 增大后每卡只保存 1/TP 的参数与优化器状态，显存下降 81.5%；但每层引入 all-reduce，吞吐下降 39.7%。TP1 的 519.5 TFLOP/s/GPU 之所以最高，是因为 DP8 无张量并行通信且每卡 GEMM 大。</div>
<h3>3.2 mbs / GBS 组合（TP8, seq4096）</h3>{chart_mbs}
<div class="note">同为 GBS8，mbs1→mbs4 吞吐 200.7→344.1（<b>+71.5%</b>）：micro-batch 从 1 增到 4，accum 从 8 降到 2，kernel 更大、流水线更饱满。同为 GBS32，mbs2→mbs4 为 313.4→354.4（+13.1%）。注意 TFLOP/s/GPU 与 GBS×seq 成正比，跨 GBS 比较时须结合 accum 一起看。</div>
<h3>3.3 序列长度扫描（TP8, mbs1, GBS8；右轴为对数刻度）</h3>{chart_seq}
<div class="note">2K→32K 吞吐单调上升至 398.6；128K 回落到 316.8，单步骤增到 61.3 s、显存 137.3 GiB（≈98% 容量）。序列放大 32 倍、单步耗时放大 62 倍，体现 attention 的超线性开销，32K 是本轮吞吐拐点。</div>
<div class="grid2"><div><h3>3.4 精度对照（名义 FP8）</h3>{chart_prec}
<div class="note" style="border-color:#dc2626">两档只差 <b>+0.26%</b>，且日志中 <code>fp8=None</code>、<code>params_dtype=bfloat16</code>、loss 与 grad norm 逐位一致 —— FP8 档实际仍跑 BF16，<b>该结论不可用</b>（见 F1）。</div></div>
<div><h3>扫描维度小结</h3>{table(["维度", "最优", "数值", "代价/风险"], [
 ["TP", "TP1 / DP8", "519.5 TFLOP/s/GPU", "显存 92.8 GiB"],
 ["mbs(GBS8)", "mbs4", "344.1", "accum=2"],
 ["序列", "32K", "398.6", "显存 53.1 GiB"],
 ["精度", "BF16", "344.1", "FP8 未生效"],
], "compact")}</div></div>''',
            anchor="s2")

sec3 = card("④ MoE 四档：拓扑与 dispatcher", "EP/TP 组合对吞吐的影响远大于通信算子选择",
            f'''{chart_moe}
{table(["档位", "拓扑", "dispatcher", "step ms", "TFLOP/s/GPU", "mfu_active", "显存 GiB", "accum", "激活参数", "状态"], moe_rows, "compact")}
<div class="note">EP8/TP1 比 EP2/TP4 快 <b>4.7×</b>（177.6 vs 37.7 TFLOP/s/GPU）。ep8_tp1 的 accum=1、ep2_tp4 的 accum=4，梯度累加与 TP4 的层内 all-reduce 是主要损耗。dispatcher 差异很小：EP8 下 alltoall 比 allgather 快 2.6%，EP2 下反而 allgather 快 11.7%。<b>口径对齐验证：</b>moe.jsonl 的 <b>mfu_active</b> 与本报告 MFU 的分母同为 {PEAK_THEORY} TFLOP/s，差别在分子 —— 官方按 3.30B 激活参数做 6ND 估算，本报告用 Megatron 日志 throughput。反推：6×3.30B×GBS×seq ÷ step ÷ 8 ÷ {PEAK_THEORY} = <b>13.99% / 13.66% / 2.96%</b>，与官方 14.0 / 13.7 / 3.0 完全吻合；若用 690 作分母会得到 20.07% / 19.59% / 4.24%，与官方值矛盾 —— 这也反证项目自身的 MFU 分母就是理论峰值。两者分子不同，数值仍不可直接横比。</div>''',
            anchor="s3", tag='<span class="badge">4 / 4 OK</span>')

sec4 = card("⑤ 显存容量与 OOM 边界", f"MOE_B 静态占用 {D['limit_summary']['static_gb_est']} GiB，卡容量 {CAP_GIB} GiB",
            f'''{chart_limit}
{table(["配置", "tokens/GPU", "step s", "静态 GiB", "状态", "错误类", "判定"], lim_rows, "compact")}
<div class="note">两种不同配置（mbs4@4096 与 mbs2@8192）都是 <b>16,384 tokens/GPU</b>，均触发 <code>CUDA out of memory. Tried to allocate 7.84 GiB ... 7.16 GiB free</code>；8,192 tokens/GPU 稳过。可见边界由<b>每卡 token 数</b>决定，位于 8K~16K tokens/GPU，而不是 mbs 或 seq 单独决定。OOM_BOUNDARY 属预期边界，不计失败、不参与最好成绩与 MFU。</div>''',
            anchor="s4", tag='<span class="badge">2 OK / 2 边界</span>')

sec5 = card("⑥ 硬件基线与 NCCL 集合通信", "原始 busbw 记录存在系统性口径错误，图中为按 iters 修正后的值",
            f'''{chart_nccl}
{table(["消息", "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "all_reduce 延迟 μs"], nccl_rows, "compact")}
<div class="note"><b>口径修正：</b><code>hw_probe.py</code> 中 <code>busbw = payload_b * factor / sec / 1e9</code> 的 sec 为 iters 次通信总耗时，未除以 iters，导致记录值被低估 30~100 倍。修正值 = 记录值 × iters（如 all_reduce@1GB：3.6×100 = 360 GB/s；all_gather@8GB：12.2×30 = 366 GB/s）。修正后四种算子在 1G~8G 消息下收敛到 <b>330~370 GB/s</b>，符合单机 8×H200 NVLink 的合理区间（见 F3）。<br>同一个文件还有反向错误：HBM / L2 / PCIe 的 <code>timed()</code> 已返回单次平均耗时，公式却多乘了 it，原始记录（HBM 1813 / 2148 TB/s、PCIe 1478 / 1045 GB/s）虚高 50~500 倍，表中给出的是按饱和 iters 的估算修正值（见 F10）。两处公式现已在 <code>probes/hw_probe.py</code> 修复，重跑后可直接采信 <code>hw.jsonl</code> 原值。</div>
<div class="grid2"><div><h3>其它硬件基线</h3>{table(["项目", "实测", "状态"], [
 ["GPU", "8 × H200 SXM / SM 9.0 / 139.8 GiB / NV18 NVSwitch", '<span class="ok">OK</span>'],
 ["BF16 理论峰值 (dense)", f"{PEAK_THEORY} TFLOP/s（规格值，稀疏 {PEAK_THEORY*2:g}）", '<span class="ok">MFU 分母</span>'],
 ["bf16 GEMM (n=8192)", f"{PEAK_GEMM} TFLOP/s（实测，达理论 {GEMM_RATIO:.1f}%）", '<span class="warn">非 MFU 分母</span>'],
 ["fp8 GEMM", "_scaled_mm 不可用", '<span class="warn">SKIP</span>'],
 ["HBM read / copy", "≈3.63 / 4.30 TB/s（估算修正）", '<span class="warn">口径存疑</span>'],
 ["PCIe h2d / d2h", "≈29.6 / 20.9 GB/s（估算修正）", '<span class="warn">口径存疑</span>'],
 ["NCCL", "2.25.1 / IB / 8 ranks", '<span class="ok">OK</span>'],
], "compact")}</div><div><h3>通信特征解读</h3>{table(["消息规模", "特征"], [
 ["8 B ~ 64 KB", "延迟主导，18~24 μs，带宽可忽略"],
 ["1 MB", "过渡区，all_gather 领先（170 GB/s）"],
 ["64 MB ~ 8 GB", "带宽主导，四算子收敛到 330~370 GB/s"],
 ["all_to_all", "MoE 的 dispatch/combine 对应算子，大消息下不劣于 all_reduce"],
], "compact")}
<div class="note">小消息延迟 ~20 μs 解释了 EP2/TP4 中大量细粒度通信为何拖慢 step：accum=4 时每层多次小集合通信叠加。</div></div></div>''',
            anchor="s5")

sec6 = card("⑦ 长稳 soak：吞吐、漂移与健康", "3 段长稳按各自进度归一化，便于直接比较稳定性",
            f'''{chart_soak}
{table(["长稳段", "步数", "时间窗（容器内）", "均值", "P95", "CV", "前后漂移", "慢步", "NaN/跳过", "完成"], soak_rows, "compact")}
<div class="note">慢步定义：step ms &gt; 该段中位数的 1.5 倍。<b>Dense 2h</b> 12,503 步、CV 2.51%、漂移 +0.12%、0 慢步，ECC/NVLINK/XID/SXID 增量全 0（health-delta status=PASS），SM 均值 97.3%、功耗 643.9 W —— 本轮最可信的稳定性结论。<b>MoE 30min</b> 漂移 +11.2%、CV 6.87%、20 个慢步（0.55%），且专家负载序列全 null，暂无法归因；moe-soak.jsonl 官方 jitter=9.62、mfu_active=17.5%（口径不同）。<b>Highmem</b> 仅运行 21 分钟（35%）即结束，无 .ok 标记，不计入完成项。</div>
<h3>highmem 档：只占 29%，不构成高显存压力（F11）</h3>{chart_hmmem}
<div class="note">8 卡显存升到 <b>{D['highmem_peak_mib']/1024:.1f} GiB（{D['highmem_peak_mib']/1437.71:.1f}%）</b> 后平台化，
未见泄漏式单调增长（growth≈0）；dmon 显示 SM 94.3%、功耗 642.4 W——就「跑 21 分钟不掉速」而言是稳定的。
但就「高显存压力」而言这档是<b>无效</b>的：峰值远低于 128K 短测的 137.3 GiB（97.8%），也低于 RUNBOOK 自定的 75% 下限，
按现行口径应判 <b><code>WARN_LOW</code>「未达高占用」</b>，不能拿来回答「能否长期贴着显存上限跑」。</div>
<h3>为什么会只有 29%：占用随 tokens/卡 线性增长</h3>{chart_hmscale}
<div class="note">8B TP8 的静态占用（权重+梯度+优化器 16 B/参数 ÷ 8，加 NCCL/上下文）约 17 GiB，此后每 1K tokens/卡 约 +0.83 GiB：
16K→40.7、32K→53.1、128K→137.3 GiB。要落进 75~92% 的判定带需要约 <b>{_need_lo/1000:.0f}K~{_need_hi/1000:.0f}K tokens/卡</b>，
而该档只有 16K（<code>mbs=1 × seq=16384</code>）。反过来，脚本原默认 <code>HIGHMEM_MBS=16</code>（262K tokens/卡）约需 240 GiB，必然 OOM ——
「猜一个 mbs」本身不可行，这才是根因；现已改为在同一个模型上标定（见 F11）。
<br><b>这里要校正一处口径</b>：16K tokens/卡 在图上有两个点——<code>torch max reserved</code>
<b>{_short.get(16384,0):.1f} GiB</b>（mbs4×seq4096 短测）与 <code>nvidia-smi</code> 峰值 <b>{_hp/1024:.1f} GiB</b>
（mbs1×seq16384 的 21 分钟长稳），<b>指标和 seq 都不同，不能相减</b>。能算漂移的只有同配置同指标：
4 步试跑 <b>35.6 GiB（25.4%）</b> vs 21 分钟长稳 <b>40.7 GiB（29.0%）</b> → <b>+5.1 GiB ≈ +3.6 个点</b>。
标定按 +8 点留保守余量（<code>HIGHMEM_DRIFT</code>，选档带 75~84%，判定带 75~92%）。</div>''',
            anchor="s6", tag='<span class="badge">1 项未完成 · 1 项未达高占用</span>')

sec7 = card("⑧ IO 与故障注入", "checkpoint 读写吞吐、dataloader 覆盖、容错用例有效性",
            f'''<div class="grid2"><div>{chart_io}
<div class="note">checkpoint 15.0 GiB：写 {fmt(io.get('write_mbs',0))} MB/s（2.91 s），读 {fmt(io.get('read_mbs',0))} MB/s（0.52 s），读为写的 5.55 倍，明显受页缓存影响，不能等同于磁盘顺序读带宽。dataloader 因缺真实数据集 SKIP。</div></div>
<div><h3>故障注入结果</h3>{table(["用例", "记录状态", "rc", "注释", "日志核验"], fault_rows, "compact")}
<div class="note" style="border-color:#dc2626">三个用例日志均出现 4 次 <code>AssertionError: CUDA_DEVICE_MAX_CONNECTIONS to 1</code>，说明进程在初始化阶段就崩溃；<code>fault.jsonl</code> 的 PASS 只依据 rc≠0，属于假阳性。<b>本阶段 3/3 用例均无效</b>（见 F2）。</div></div></div>''',
            anchor="s7")

sec8 = card("⑨ 结论与可执行建议", "性能结论 + 按优先级排列的修复清单",
            concl_html + '<h3>修复清单</h3>' +
            table(["优先级", "动作", "收益 / 验收标准"], actions),
            anchor="s8")

foot = f'''<div class="card"><h2>数据口径说明</h2>
<div class="note" style="border-color:#94a3b8">
1. <b>稳态指标</b>：解析每个训练日志的 <code>iteration N/15 | elapsed time per iteration (ms) | throughput per GPU (TFLOP/s/GPU)</code>，剔除第 1 步（初始化）与第 15 步（收尾），取中间 13 步的中位数；长稳段剔除前 20 步与末 5 步后统计。<br>
2. <b>MFU（本次修订）</b> = Megatron 日志 <code>throughput per GPU (TFLOP/s/GPU)</code>（模型有效 FLOPs，6ND 口径）÷ H200 SXM BF16 Tensor Core <b>dense 理论峰值 {PEAK_THEORY} TFLOP/s</b>。NVIDIA 规格表的 1979 TFLOPS 标注「with sparsity」，稠密训练须取一半。04-hw 实测 GEMM {PEAK_GEMM:.0f} TFLOP/s 仅为理论峰值的 {GEMM_RATIO:.1f}%，属于「本机可达上限」，以它为分母会把 MFU 放大 {PEAK_THEORY/PEAK_GEMM:.3f} 倍；首版报告误用该口径，现全部废弃，<b>旧值 × {PEAK_GEMM/PEAK_THEORY:.3f} = 新值</b>。反推 moe.jsonl 的 mfu_active 也证实项目自身分母为 {PEAK_THEORY}。注意 Megatron throughput 与 GBS×seq 成正比，跨 GBS/seq 比较须同时看 accum 与 tokens/GPU；且分子不含重计算，启用 recompute 时实际硬件利用率（HFU）会高于此处 MFU。<br>
3. <b>显存</b>：取日志 <code>max reserved (MB)</code> 的最大值并按 1024 换算为 GiB；GPU 容量来自 04-hw 的 <code>cap_gib={CAP_GIB}</code>。<br>
4. <b>硬件探针修正</b>：NCCL busbw = 原始 <code>busbw_gbs × iters</code>（少除）；HBM / PCIe = 原始记录 ÷ 饱和 iters（500 / 50，多乘，属估算）；<code>latency_us</code> 为单次通信耗时，未作修正。bf16 GEMM 公式无此问题。两处公式已修（F3 / F10），但本历史 run 的数值仍按上述修正口径展示。<br>
5. <b>状态</b>：OOM_BOUNDARY 按 limit-summary.json 约定为预期边界，不计失败；未标记 .ok 的 soak 段视为未完成。<br>
6. 所有图表由 Python 标准库直接生成 SVG，离线可看；配套 CSV 见 <code>analysis/cases.csv</code>、<code>soak_timeseries.csv</code>、<code>findings.csv</code>，复现脚本为 <code>parse.py</code> + <code>build_report.py</code>。
</div></div>
</main><footer>生成时间：由本地日志解析生成 · 不修改任何原始测试结果 · 图表与数值均来自 run 目录原始文件</footer>
</body></html>'''

fix_banner = ("""<div class="banner"><b>代码层修复已完成（2026-11-10）</b>：本报告的 """
              """F1（S4 假 FP8）、F2（故障注入假阳性）、F11（highmem 只占 29% 却记 OK）已在 <code>04_bench.sh</code> / <code>lib.sh</code> 修复；"""
              """F3 / F10（<code>hw_probe.py</code> 两处相反的 iters 口径错误）与 F6（MoE 专家负载全 null）"""
              """已修 <code>probes/hw_probe.py</code> / 新增 <code>probes/moe_load_probe.py</code> / <code>04_bench.sh</code>；"""
              """同步 RUNBOOK / TESTPLAN / 变更清单，顺带修掉 <code>highmem_record()</code> 内嵌 python 的缩进错误（会让 highmem.jsonl 一条都写不出来，<code>bash -n</code> 查不到）；本机静态验证与桩测 29/29 + 46/46 通过。"""
              """<b>本报告仍是对该历史 run 的如实记录，结论不因代码修改而改变</b> —— """
              """需把脚本同步到目标机、清除旧 <code>.ok_</code> 哨兵并重跑 """
              """<code>--phase hw / coll / s8b / moe / fault / soak</code> 才能关闭这些项。</div>""")

html = head + fix_banner + sec0 + sec1 + sec2 + sec3 + sec4 + sec5 + sec6 + sec7 + sec8 + foot
(HERE / "report.html").write_text(html, encoding="utf-8")

# --------------------------------------------------------------- CSV
def write_csv(path, headers, rows):
    """写 CSV；目标文件被本地编辑器占用时不崩、不静默丢数据。"""
    import io
    buf = io.StringIO()
    # write_text 在 Windows 上会把 \n 转成 \r\n；这里用 \n 避免 \r\r\n 产生空行
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(headers)
    wr.writerows(rows)
    text = buf.getvalue()
    try:
        old = path.read_text(encoding="utf-8-sig")
    except OSError:
        old = None
    # read_text 会把 CRLF 归一成 LF，比较时统一换行，避免「内容相同却每次都重写」
    if old is not None and old.replace("\r\n", "\n") == text.replace("\r\n", "\n"):
        print(f"[skip] {path.name} 内容未变更")
        return
    try:
        path.write_text(text, encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(path.name + ".new")
        alt.write_text(text, encoding="utf-8-sig")
        print(f"[WARN] {path.name} 被本地编辑器占用，已写入 {alt.name}（关闭后可替换）")

write_csv(HERE / "cases.csv",
          ["group", "tag", "tp", "pp", "mbs", "seq", "gbs", "precision", "status",
           "iters", "step_ms_median", "tflops_median", "tflops_mean", "tflops_cv_pct",
           "mfu_pct_theory_989_5", "eff_pct_vs_measured_gemm_690",
           "mem_max_reserved_mb", "mem_gib", "loss_first", "loss_last",
           "skipped", "nan", "sm_mean_pct", "pwr_mean_w"],
          [[c["group"], c["tag"], c.get("tp"), c.get("pp"), c.get("mbs"), c.get("seq"),
            c.get("gbs") or c.get("gb"), c.get("precision"), "OK", c["iters_logged"],
            med(c, "step_ms"), med(c), c["tflops"].get("mean"), c["tflops"].get("cv_pct"),
            round(mfu(c), 2), round(eff_gemm(c), 2),
            c["mem"].get("max_reserved_mb"), round(gib(c), 2),
            c.get("loss_first"), c.get("loss_last"), c["skipped_total"], c["nan_total"],
            (c.get("dmon") or {}).get("sm_mean_all"), (c.get("dmon") or {}).get("pwr_mean_all")]
           for c in cases] +
          [["limit", r["tag"], r["tp"], r["pp"], r["mbs"], r["seq"], r["gbs"], "bf16", r["status"],
            r["iters_logged"], r["step_s"] * 1000 if r["step_s"] else "", "", "", "", "", "",
            (r.get("mem") or {}).get("max_reserved_mb"), "", "", "", "", "",
            (r.get("dmon") or {}).get("sm_mean_all"), (r.get("dmon") or {}).get("pwr_mean_all")]
           for r in limit])

ts_rows = []
for key, name, _ in soak_names:
    for p in soak[key]["tflops_series"]:
        ts_rows.append([key, name, p["iter"], p["mean"], p["min"], p["max"]])
write_csv(HERE / "soak_timeseries.csv", ["run", "label", "iter", "tflops_mean", "tflops_min", "tflops_max"], ts_rows)

write_csv(HERE / "findings.csv",
          ["id", "priority", "fix_status", "finding", "evidence", "impact", "recommendation"],
          [[i, p, "fixed_pending_rerun" if i in FIXED else "open", t,
            ev.replace("<code>", "").replace("</code>", "").replace("<b>", "").replace("</b>", ""),
            imp, fix.replace("<code>", "").replace("</code>", "")] for i, p, t, ev, imp, fix in findings])

print("report.html", (HERE / "report.html").stat().st_size, "bytes")
print("cases.csv", len(cases) + len(limit), "rows | soak_timeseries.csv", len(ts_rows),
      "rows | findings.csv", len(findings), "rows")
print("peak:", best["tag"], fmt(med(best)), "TFLOP/s/GPU")
