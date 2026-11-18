# -*- coding: utf-8 -*-
"""生成单机 56B 测试统计分析报告：复用已校验的 CSS / SVG 绘图库，离线自包含。"""
import json
import re
from datetime import datetime
from pathlib import Path

OUT = Path(r"C:/Users/26488/Desktop/ht/e2e-megatron-56b-2501/node1-e2e-56b/analysis")
D = json.loads((OUT / "stats_single.json").read_text(encoding="utf-8"))
S, G, CFG, PAR = D["stats"], D["gpu"], D["config"], D["params"]
CMP, EV, TOPO, MEM = D["comparison"], D["events"], D["topo"], D["mem"]
BASE = (OUT / "node1-e2e-56b-统计分析报告.html").read_text(encoding="utf-8")
CSS = re.search(r"<style>([\s\S]*?)</style>", BASE).group(1)
JS = re.search(r"<script>([\s\S]*?)</script>", BASE).group(1)
LIB = JS[JS.find("const NS = "):JS.find("/* ============ 图 1")]
GEN = datetime.now().strftime("%Y-%m-%d %H:%M")
PEAK = S["peak_tflops"]


def nf(v, n=2):
    try:
        return f"{float(v):,.{n}f}"
    except (TypeError, ValueError):
        return "—" if v is None else str(v)


# ---------- KPI ----------
kpis = [
    ("运行时长", nf(S["hours"]), "h", "完整训练，非定时退出", "blue"),
    ("完成迭代", f'{S["completed_iters"]:,}', f'/ {S["iter_target"]:,}', "100% · 含最终 val/test 评估", "green"),
    ("稳定期吞吐", nf(S["throughput_stable"]["mean"], 1), "TFLOP/s/GPU", f'CV {S["throughput_stable"]["cv"]}%', "blue"),
    ("稳定期 MFU", nf(S["mfu_stable"]), "%", f"H200 BF16 dense {PEAK} TFLOP/s", "green"),
    ("最终 train loss", nf(S["train_loss_last"], 6), "", f'初始 {S["train_loss_first"]:.3f}', "purple"),
    ("最终 test PPL", nf(S["final_test_ppl"], 6), "", f'val PPL {S["final_val_ppl"]:.6f}', "purple"),
    ("单卡显存峰值", nf(G["fb_peak"] / 1024, 1), "GiB", "8 卡差异仅 3 MB", "cyan"),
    ("功率封顶占比", nf(G["power_cap_pct"], 1), "% pviol=100", "tviol 全 0，无热降频", "orange"),
    ("GPU 计算占用", nf(G["sm_ge90_pct"], 1), "% SM≥90", f'{G["rows"]:,} 个采样点', "cyan"),
    ("NaN / skipped", "0 / 0", "", "全程无数值异常", "green"),
    ("致命错误", "0", "", "无 OOM / Traceback / CUDA error", "green"),
    ("验证次数", f'{len(D["val"]):,}', "", f'每次约 {nf(S["eval_ms"]["mean"]/1000,2)} s', "blue"),
]
kpi_html = "\n".join(
    f'<div class="kpi k-{c}"><div class="kpi-t">{t}</div><div class="kpi-v">{v}<span class="kpi-u">{u}</span></div><div class="kpi-s">{s}</div></div>'
    for t, v, u, s, c in kpis)

# ---------- 配置表 ----------
cfg_rows = [
    ("并行拓扑", f'TP {CFG.get("tensor_model_parallel_size")} · PP {CFG.get("pipeline_model_parallel_size")} · DP {CFG.get("data_parallel_size")} · world_size {CFG.get("world_size")}（单机 8 卡）'),
    ("模型结构", f'{CFG.get("num_layers")} 层 · hidden {CFG.get("hidden_size")} · FFN {CFG.get("ffn_hidden_size")} · {CFG.get("num_attention_heads")} heads · vocab {CFG.get("vocab_size")}'),
    ("总参数量", f'{nf(PAR.get("total_B"),2)} B（transformer {nf(PAR.get("transformer_B"),2)} B + embedding {nf(PAR.get("embedding_B"),2)} B）· 每 TP 分片 {nf(PAR.get("shard_B"),4)} B'),
    ("批次与序列", f'seq {CFG.get("seq_length")} · MBS {CFG.get("micro_batch_size")} · GBS {CFG.get("global_batch_size")} · 梯度累积 {S["grad_accum"]} 步 · 每 iter {S["tokens_per_iter"]:,} tokens'),
    ("训练目标", f'train_iters {CFG.get("train_iters")} · exit_duration_in_mins {CFG.get("exit_duration_in_mins")}（未触发）· eval_interval {CFG.get("eval_interval")} / eval_iters {CFG.get("eval_iters")} · log_interval {CFG.get("log_interval")}'),
    ("优化器", f'{CFG.get("optimizer")} · lr {CFG.get("lr")} · warmup {CFG.get("lr_warmup_iters")} iter · cosine decay {CFG.get("lr_decay_iters")} · clip_grad {CFG.get("clip_grad")}'),
    ("精度与数据", f'bf16 {CFG.get("bf16")} · fp8 {CFG.get("fp8")} · mock_data {CFG.get("mock_data")}（合成数据）· params_dtype {CFG.get("params_dtype")}'),
    ("显存策略（lowmem1）", f'recompute_granularity <b>{CFG.get("recompute_granularity")}</b> · use_precision_aware_optimizer <b>{CFG.get("use_precision_aware_optimizer")}</b> · use_distributed_optimizer {CFG.get("use_distributed_optimizer")} · save {CFG.get("save")}'),
    ("通信与重叠", f'sequence_parallel {CFG.get("sequence_parallel")} · tp_comm_overlap {CFG.get("tp_comm_overlap")} · overlap_grad_reduce <b>{CFG.get("overlap_grad_reduce")}</b> · overlap_param_gather <b>{CFG.get("overlap_param_gather")}</b>'),
]
cfg_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in cfg_rows)

ms = D["milestones"]
ms_notes = ["启动后 34 s", "1 s", "1 s", "首个 iter 吞吐 297.7 TFLOP/s（编译/warmup）",
            f'训练跑满 {S["completed_iters"]:,} iter，随后执行最终 validation / test 评估']
tl_html = "\n".join(
    f'<tr><td><code>{m["ts"]}</code></td><td>{m["label"]}</td><td>{ms_notes[i] if i < len(ms_notes) else ""}</td></tr>'
    for i, m in enumerate(ms))

# ---------- 收敛里程碑 ----------
mile = [(1.0, S["iter_loss_below_1.0"]), (0.1, S["iter_loss_below_0.1"]),
        (0.01, S["iter_loss_below_0.01"]), (0.005, S["iter_loss_below_0.005"])]
mile_html = "\n".join(
    f'<tr><td>&lt; {a}</td><td><b>{b:,}</b></td><td>约 {nf(next((d["t"] for d in D["train"] if d["iter"] == b), 0), 2)} h</td>'
    f'<td>累计 {b * S["tokens_per_iter"] / 1e6:,.1f} M tokens</td></tr>' for a, b in mile)

# ---------- 统计量 ----------
stat_rows = [
    ("预热期吞吐 (TFLOP/s/GPU)", S["throughput_warm"]),
    ("稳定期吞吐 (TFLOP/s/GPU)", S["throughput_stable"]),
    ("稳定期迭代耗时 (s/iter)", {**S["iter_ms_stable"],
                              **{k: S["iter_ms_stable"][k] / 1000 for k in ("mean", "sd", "min", "p50", "p95", "max")}}),
    ("验证单次耗时 (s)", {**S["eval_ms"],
                       **{k: S["eval_ms"][k] / 1000 for k in ("mean", "sd", "min", "p50", "p95", "max")}}),
    ("梯度范数", S["gnorm"]),
]
stat_html = "\n".join(
    f'<tr><td><b>{a}</b></td><td>{b["n"]}</td><td>{nf(b["mean"],3)}</td><td>{nf(b["sd"],3)}</td>'
    f'<td>{nf(b["cv"],2)}%</td><td>{"/".join(nf(b[k],3) for k in ("min","p50","p95","max"))}</td></tr>'
    for a, b in stat_rows)

# ---------- 显存表 ----------
m0 = MEM[0] if MEM else {}
mem_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in [
    ("PyTorch allocated（iter 10 后）", f'{nf(m0.get("allocated"),1)} MB', "8 个 rank 完全一致"),
    ("PyTorch max allocated", f'{nf(m0.get("max_allocated"),1)} MB', "峰值张量占用"),
    ("PyTorch max reserved", f'{nf(m0.get("max_reserved"),0)} MB', "分配器缓存上限"),
    ("dmon fb 稳定期", f'{G["summary"]["0"]["fb"]["p50"]:,} MB（≈{nf(G["summary"]["0"]["fb"]["p50"]/1024,1)} GiB）', f'8 卡差异 {nf(S["fb_card_spread_MB"],2)} MB'),
    ("dmon fb 峰值", f'{G["fb_peak"]:,} MB', "接近 H200 141 GB 容量上限（约 96%）"),
    ("非 PyTorch 占用", f'{nf(S["reserved_vs_fb_MB"],0)} MB', "NCCL / CUDA context / 驱动缓冲"),
])

# ---------- 事件表 ----------
ev_rows = [
    ("致命错误", "Traceback", EV["traceback"], "无崩溃"),
    ("致命错误", "OOM", EV["oom"], "无显存溢出"),
    ("致命错误", "CUDA error", EV["cuda_error"], "无设备错误"),
    ("致命错误", "NCCL 真超时", EV["nccl_timeout"], "无集合通信超时"),
    ("退出清理", "Abort COMPLETE", EV["abort_complete"], "单机日志未开 NCCL_DEBUG=INFO，无 Abort 明细"),
    ("退出清理", "process group 未销毁警告", EV["pg_not_destroyed"], "可显式 destroy_process_group() 消除"),
    ("噪声", "RerunStateMachine 警告", EV["rerun_state_machine_warn"], f'{len(D["val"])} 次验证 × 每轮约 3 条，正常噪声'),
    ("噪声", "ResourceWarning", EV["resource_warning"], "临时目录清理"),
    ("噪声", "Python 弃用警告", EV["deprecation"], "protobuf / importlib"),
    ("噪声", "TE fused RoPE 警告", EV["rope_warning"], "transpose_output_memory 被忽略"),
]
ev_html = "\n".join(
    f'<tr><td>{a}</td><td><code>{b}</code></td><td><b>{c}</b></td><td>{d}</td></tr>' for a, b, c, d in ev_rows)

# ---------- 双机对比表 ----------
cmp_cfg_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in CMP["config"])
cmp_metric_html = "\n".join(
    f'<tr><td><b>{a}</b></td><td>{nf(b,3) if isinstance(b,(int,float)) else b}</td>'
    f'<td>{nf(c,3) if isinstance(c,(int,float)) else c}</td></tr>' for a, b, c in CMP["metrics"])
cmp_metric_html += (
    f'<tr><td><b>相对双机吞吐</b></td><td colspan="2"><b>{nf(CMP["throughput_ratio"],1)}%</b>'
    f'（MFU 差 {nf(CMP["mfu_gap_pp"],2)} pp · 显存高 {CMP["fb_gap_MB"]:,} MB）</td></tr>')

# ---------- 诊断表 ----------
diag = [
    ("单机吞吐比双机低 15.7%",
     f'{nf(S["throughput_stable"]["mean"],1)} vs 524.97 TFLOP/s/GPU；MFU {nf(S["mfu_stable"],2)}% vs 53.05%；'
     f'SM≥90% {nf(G["sm_ge90_pct"],2)}% vs 99.9%；6ND 差额 {nf(S["gap_vs_6nd_pct"],2)}% vs 双机 7.46%',
     "配置差异导致计算密度更低：seq 减半、GBS 从 64 降到 8、selective recompute 额外计算、显存接近上限", "不是故障；单机 MFU 天花板更低", "ok"),
    ("显存比双机高 28.3 GB",
     f'PyTorch max_reserved {nf(m0.get("max_reserved"),0)} MB vs 双机 102,558 MB；fb 稳定期 {G["summary"]["0"]["fb"]["p50"]:,} MB；'
     "H200 141 GB 容量占用约 96%",
     "DP=1 时 distributed optimizer 无法跨 DP 分片优化器状态，单机每卡需常驻完整分片状态；selective recompute 只抵消了一部分", "容量余量极小，增大 seq / GBS 有 OOM 风险", "warn"),
    ("pviol=100 占 96.0%，但功耗反而低 19 W",
     f'平均功耗 {nf(S["pwr_avg"],1)} W vs 双机 676.9 W；pclk {nf(G["summary"]["0"]["pclk"]["mean"],0)} MHz（高于双机 1,889）；'
     f'最高温度 {S["gtemp_max"]} °C；tviol 全 0',
     "同样处于 TDP 功率墙，但显存/重计算使 SM 利用率略低、功耗未吃满；非散热问题", "无热降频；提升空间仍在功率预算与计算密度", "info"),
    ("训练完整跑满 20,786 iter",
     f'<code>[after training is done]</code> + 最终 validation set / test set 评估；'
     f'实测 {nf(S["hours"],2)} h（1,128 min）未达 exit_duration_in_mins=1440（24 h）',
     "按迭代数正常完成，与双机按运行时长定时退出是两种不同的 E2E 设计", "单机功能验证更完整（含测试集评估）", "ok"),
    ("0 致命错误、0 NaN、0 skipped",
     "Traceback / OOM / CUDA error / NCCL 超时全为 0；2,078 个训练点、417 次验证均无 skipped / nan",
     "训练数值健康，clip_grad=1.0 生效", "无", "ok"),
    ("早期梯度尖峰与验证反弹较多",
     f'前 250 iter 多次 grad norm &gt;8（最大 113.3 @iter 230）；验证 loss 较前次上涨 &gt;30% 共 {len(S["val_rebounds"])} 次',
     "合成数据训练早期的数值波动；验证次数 417 次（双机仅 73 次），按次数看反弹比例并不更高；均自愈", "可忽略；真实数据训练前仍建议确认 lr warmup", "info"),
    ("日志无 NCCL 拓扑明细",
     f'NCCL INFO 行数 {TOPO["nccl_debug_info"]}；<code>via P2P</code> / <code>via NET/IBext</code> 均为 0；'
     f'仅记录 NCCL version {TOPO["nccl_version"]}',
     "本次未启用 NCCL_DEBUG=INFO，不能解读为「没有 P2P 通道」", "如需通信剖析请开启该环境变量重跑", "info"),
]
_suggest = [
    "先固定 seq / GBS / 重计算策略，再评估扩展性；或降低显存压力后关闭 selective recompute",
    "容量规划按 139 GB 取上界，放大 seq / GBS 前先做 OOM 评估",
    "优先提升计算密度与显存效率，加强散热无收益",
    "无需处理；可作为双机定时退出模式的对照基线",
    "无需处理",
    "真实数据训练前确认 lr warmup 与 grad clip 配置",
    "如需通信剖析，开启 NCCL_DEBUG=INFO 重跑",
]
diag = [(a, b, c, d, _suggest[i], lv) for i, (a, b, c, d, lv) in enumerate(diag)]
diag_html = "\n".join(
    f'<tr><td class="phen">{a}</td><td>{b}</td><td><span class="tag tag-{lv}">{c}</span></td><td>{d}</td><td>{e}</td></tr>'
    for a, b, c, d, e, lv in diag)

# ---------- 文件表 ----------
files_html = "\n".join(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in [
    (D["meta"]["log"], "单机训练主日志", f'{D["meta"]["log_lines"]:,} 行 / 1.1 MB', "超参表、里程碑、2,078 训练点、417 次验证、显存明细、最终 test 评估"),
    (D["meta"]["csv"], "单机 GPU 监控", f'{G["csv_lines"]:,} 行 / 16.2 MB', f'{G["rows"]:,} 有效采样（8 卡 × {G["points"]:,} 时间点）'),
    ("node1…node1-20261112_000626.out / -dmon-.csv", "双机测试对照", "514 KB / 20.3 MB", "TP8 × DP2 / SEQ8192 / GBS64 / 定时退出"),
    ("selfcheck.log · args-help.txt · image-digest.txt", "环境自检", "3 个文件", "8 卡 / CUDA 12.8 / TE 1.14 / 512 条参数行 / 镜像 SHA"),
])

DATA = json.dumps(D, ensure_ascii=False, separators=(",", ":"))

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>单机 56B 训练结果统计分析</title>
<style>__CSS__</style></head>
<body><div class="wrap">
<header class="top">
  <h1>单机 56B 训练结果统计分析</h1>
  <div class="sub">__CASE__ ｜ 2,078 个训练点 + 417 次验证 + 98,568 个 GPU 采样点的量化复盘</div>
  <div class="meta">
    <span>窗口 __TS1__ ~ __TS2__</span><span>时长 __HOURS__ h</span>
    <span>迭代 __ITERS__ / __TARGET__（100%）</span><span>8 × GPU · TP8 × PP1 × DP1</span>
    <span>报告生成 __GEN__</span>
  </div>
</header>
<nav>
  <a href="#s1">结论</a><a href="#s2">配置与时间线</a><a href="#s3">收敛</a><a href="#s4">性能与 MFU</a>
  <a href="#s5">GPU</a><a href="#s6">日志</a><a href="#s7">单机 vs 双机</a><a href="#s8">诊断</a><a href="#s9">附录</a>
</nav>
<div class="kpis">__KPIS__</div>

<section id="s1">
  <h2><span class="no">1</span>总体结论</h2>
  <div class="verdict"><div class="ic">✓</div><div>
    <b>单机 56B 训练完整跑满 20,786 iter 并正常结束，含最终 validation / test 集评估，全程无故障。</b><br>
    连续运行 __HOURS__ 小时，稳定期吞吐 __TF__ TFLOP/s/GPU（CV __CV__%）、MFU __MFU__%；
    train loss 从 __L0__ 收敛至 __L1__，最终 val PPL __VPPL__、test PPL __TPPL__；
    0 NaN / 0 skipped / 0 OOM / 0 Traceback。与双机测试不同，本次是<b>按迭代数完成</b>而非按运行时长定时退出
    （exit_duration_in_mins=1440 未触发）。
  </div></div>
  <h3>E2E 检查项</h3>
  <table><thead><tr><th>检查项</th><th>判定</th><th>证据</th></tr></thead><tbody>
    <tr><td>8 卡与组件版本</td><td><span class="tag tag-ok">通过</span></td><td>GPUS=8 · PyTorch 2.6.0a0 · CUDA 12.8 · TE 1.14.0（selfcheck.log）</td></tr>
    <tr><td>训练完整性</td><td><span class="tag tag-ok">通过</span></td><td>20,786 / 20,786 iter · <code>[after training is done]</code> · 最终 val/test 评估</td></tr>
    <tr><td>收敛质量</td><td><span class="tag tag-ok">通过</span></td><td>loss 9.767 → 0.005257 · val PPL 28,702 → 1.0046 · test PPL 1.00427</td></tr>
    <tr><td>长时稳定性</td><td><span class="tag tag-ok">通过</span></td><td>18.79 h · 吞吐 CV 0.686% · 0 NaN / 0 skipped</td></tr>
    <tr><td>无致命错误</td><td><span class="tag tag-ok">通过</span></td><td>0 Traceback / OOM / CUDA error / NCCL 超时</td></tr>
    <tr><td>显存容量余量</td><td><span class="tag tag-warn">需关注</span></td><td>fb 峰值 139,068 MB（≈135.8 GiB），约占 H200 141 GB 的 96%</td></tr>
  </tbody></table>
  <div class="note">口径说明：日志内部时间戳为 UTC，dmon 为 UTC+8，已统一到 UTC 时间轴；
    MFU 采用与双机报告相同的 H200 SXM BF16 dense 峰值 989.5 TFLOP/s。
    本次为「单机 lowmem1」配置，与双机测试的 seq / GBS / DP / 重计算策略均不同，对比结论见第 7 节。</div>
</section>

<section id="s2">
  <h2><span class="no">2</span>配置与执行时间线</h2>
  <p class="lead">单机日志自带完整超参表（__CONFTOTAL__ 项）与显存明细，这是双机 node1 日志所没有的。</p>
  <h3>关键配置</h3>
  <table><thead><tr><th style="width:22%">类别</th><th>配置</th></tr></thead><tbody>__CFG__</tbody></table>
  <h3>启动 → 完成时间线（UTC）</h3>
  <table><thead><tr><th style="width:24%">时刻</th><th style="width:42%">事件</th><th>说明</th></tr></thead><tbody>__TL__</tbody></table>
  <h3>显存构成</h3>
  <table><thead><tr><th style="width:38%">口径</th><th style="width:24%">数值</th><th>说明</th></tr></thead><tbody>__MEM__</tbody></table>
</section>

<section id="s3">
  <h2><span class="no">3</span>收敛与数值行为</h2>
  <p class="lead">2,078 个训练记录点（每 10 iter）+ 417 次验证；loss 用对数坐标以同时呈现早期下降与后期平台。</p>
  <div class="chart" id="c_loss"><div class="ct">训练 / 验证 loss 收敛曲线</div><div class="cs">对数纵轴 · 横轴 iteration</div><div class="legend" id="l_loss"></div></div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_ppl"><div class="ct">验证 PPL</div><div class="cs">对数纵轴 · 28,702 → 1.0046</div></div>
    <div class="chart" id="c_gnorm"><div class="ct">梯度范数 grad norm</div><div class="cs">对数纵轴 · 红色标注早期尖峰</div></div>
  </div>
  <h3>收敛里程碑</h3>
  <table><thead><tr><th>train loss 阈值</th><th>首次低于该值的 iteration</th><th>对应训练时长</th><th>累计 tokens</th></tr></thead><tbody>__MILE__</tbody></table>
  <div class="note"><code>mock_data=True</code>（合成数据），loss 最终收敛到 ~0.005、PPL → 1.004，属于典型的记忆化形态，
    用于 E2E 功能与稳定性验证而非真实语言模型质量评估。单机每 iter 仅 32,768 tokens（双机 524,288），
    按 iteration 口径收敛更慢，但按 token 口径其实更早。</div>
</section>

<section id="s4">
  <h2><span class="no">4</span>计算性能与 MFU</h2>
  <p class="lead">稳定期定义为 iteration ≥ 500，共 2,029 个记录点。</p>
  <div class="chart" id="c_perf"><div class="ct">每卡吞吐与迭代耗时</div><div class="cs">双纵轴 · 20,786 iter 全程稳定</div><div class="legend" id="l_perf"></div></div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_hist_tf"><div class="ct">稳定期吞吐分布</div><div class="cs">直方图 · TFLOP/s/GPU</div></div>
    <div class="chart" id="c_hist_ms"><div class="ct">稳定期迭代耗时分布</div><div class="cs">直方图 · 秒 / iteration</div></div>
  </div>
  <h3>MFU（Model FLOPs Utilization）</h3>
  <div class="kpis" style="grid-template-columns:repeat(3,1fr)">
    <div class="kpi k-green"><div class="kpi-t">稳定期 MFU</div><div class="kpi-v">__MFU__<span class="kpi-u">%</span></div><div class="kpi-s">__TF__ TFLOP/s ÷ 989.5</div></div>
    <div class="kpi k-blue"><div class="kpi-t">6ND 理论口径 MFU</div><div class="kpi-v">__MFU6__<span class="kpi-u">%</span></div><div class="kpi-s">__F6__ TFLOP/s/GPU</div></div>
    <div class="kpi k-cyan"><div class="kpi-t">8 卡有效算力</div><div class="kpi-v">__PF__<span class="kpi-u">PFLOP/s</span></div><div class="kpi-s">每 iter __TPI__ tokens</div></div>
  </div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_mfu"><div class="ct">MFU 随迭代变化</div><div class="cs">由日志每卡吞吐换算</div><div class="legend" id="l_mfu"></div></div>
    <div class="chart" id="c_mfu_hist"><div class="ct">稳定期 MFU 分布</div><div class="cs">iteration ≥ 500 的 2,029 个点</div></div>
  </div>
  <h3>关键统计量</h3>
  <table><thead><tr><th>指标</th><th>样本数</th><th>均值</th><th>标准差</th><th>CV</th><th>最小 / P50 / P95 / 最大</th></tr></thead><tbody>__STAT__</tbody></table>
  <div class="note"><b>6ND 交叉校验：</b>6 × 56.51B × 32,768 tokens/iter = __PF6__ PFLOPs/iter，
    除以稳定期 __MS__ s 与 8 卡得 __F6__ TFLOP/s/GPU（MFU __MFU6__%）；Megatron 日志口径高 __GAP__%，
    远低于双机的 7.46% —— 因为 seq 4096 下 attention 等额外 FLOPs 占比更小，两种口径更接近。
    首个 iteration 吞吐 297.7 TFLOP/s 为编译/warmup，20 个记录点内升至 428.8。</div>
</section>

<section id="s5">
  <h2><span class="no">5</span>GPU 硬件与功耗</h2>
  <p class="lead">dmon 每 ~5.5 s 采样，8 卡 × 12,321 个时间点 = 98,568 条有效记录；按 30 分钟分桶聚合。</p>
  <div class="grid2">
    <div class="chart" id="c_gpu_time"><div class="ct">功耗 / 温度时序</div><div class="cs">双纵轴 · 8 卡均值</div><div class="legend" id="l_gpu_time"></div></div>
    <div class="chart" id="c_clk"><div class="ct">核心频率与功率墙 pviol 时序</div><div class="cs">左轴 MHz（实线）· 右轴 %（虚线）</div><div class="legend" id="l_clk"></div></div>
  </div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_startup"><div class="ct">启动阶段显存与功耗爬升</div><div class="cs">前 30 分钟按 1 分钟聚合</div><div class="legend" id="l_startup"></div></div>
    <div class="chart" id="c_power"><div class="ct">功率墙 pviol 采样分布</div><div class="cs">pviol = 采样周期内功率封顶时间占比</div></div>
  </div>
  <h3>8 卡横向对比</h3>
  <div class="grid2" id="gpu_bars"></div>
  <h3>温度热力图（8 卡 × 30 分钟）</h3>
  <div class="chart" id="c_heat"><div class="ct">GPU 温度分布</div><div class="cs">偶数卡系统性高于奇数卡</div></div>
  <div class="note"><b>功率墙而非温度墙：</b>__PCAP__% 的采样点 pviol=100，tviol 全 0，最高温度仅 __TMAX__ °C；
    平均功耗 __PWR__ W、峰值 __PWRMAX__ W，核心频率在 __CLKMIN__–__CLKMAX__ MHz 间自适应。
    显存稳定在 __FB__ MB/卡（≈__FBG__ GiB），8 卡差异仅 __FBSPREAD__ MB。</div>
</section>

<section id="s6">
  <h2><span class="no">6</span>日志构成与事件</h2>
  <div class="grid2">
    <div class="chart" id="c_events"><div class="ct">主日志 6,270 行构成</div><div class="cs">按行类型分类</div></div>
    <div class="chart" id="c_cmp_tf"><div class="ct">吞吐时序对比（按各自运行时长对齐）</div><div class="cs">单机 18.8 h vs 双机 23.5 h</div><div class="legend" id="l_cmp_tf"></div></div>
  </div>
  <h3>事件计数</h3>
  <table><thead><tr><th>类别</th><th>事件</th><th>次数</th><th>性质</th></tr></thead><tbody>__EV__</tbody></table>
  <div class="note">单机日志未启用 <code>NCCL_DEBUG=INFO</code>（NCCL INFO 行数 0），因此没有 P2P / IB 通道明细与
    Abort COMPLETE 记录；日志中仅出现 <code>NCCL version __NCCL__</code> 与 8 条 ProcessGroupNCCL barrier 提示。
    这不代表没有机内 NVLink/P2P 通信，只说明日志级别不同。</div>
</section>

<section id="s7">
  <h2><span class="no">7</span>单机 vs 双机对比</h2>
  <p class="lead">两次测试是<b>不同配置</b>的 E2E 验证：seq 4096 vs 8192、GBS 8 vs 64、DP1 vs DP2、
    selective recompute vs 无重计算，因此吞吐差异不能单独归因于「单机 / 双机」。</p>
  <div class="chart" id="c_cmp"><div class="ct">关键指标归一化对比（双机 = 100%）</div><div class="cs">每组左侧单机（蓝）· 右侧双机（橙）· 显存越高代表容量压力越大</div><div class="legend" id="l_cmp"></div></div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_cmp_loss"><div class="ct">loss 收敛对比（按运行时长对齐）</div><div class="cs">对数纵轴</div><div class="legend" id="l_cmp_loss"></div></div>
    <div class="chart" id="c_cmp_fb"><div class="ct">显存占用时序对比</div><div class="cs">单机比双机高约 28 GB，全程恒定</div><div class="legend" id="l_cmp_fb"></div></div>
  </div>
  <h3>配置差异</h3>
  <table><thead><tr><th style="width:26%">项目</th><th>单机 lowmem1</th><th>双机 node1</th></tr></thead><tbody>__CMPCFG__</tbody></table>
  <h3>指标差异</h3>
  <table><thead><tr><th style="width:38%">指标</th><th>单机</th><th>双机</th></tr></thead><tbody>__CMPMETRIC__</tbody></table>
  <div class="note"><b>为什么单机显存反而更高：</b>DP=1 时 <code>use_distributed_optimizer</code> 没有可分片的数据并行维度，
    每个 TP rank 需常驻完整的优化器状态；双机 DP2 把它分到两组，单卡 max_reserved 直接少 31.9 GB。
    单机虽启用 selective recompute 与 precision-aware optimizer 来省显存，仍只抵消了一部分，最终 fb 达到 139 GB。</div>
</section>

<section id="s8">
  <h2><span class="no">8</span>根因排查与建议</h2>
  <p class="lead">按「现象 → 证据 → 判定 → 影响 → 建议」结构化呈现。</p>
  <table><thead><tr><th style="width:19%">现象</th><th style="width:34%">证据</th><th style="width:18%">判定</th><th style="width:14%">影响</th><th style="width:15%">建议</th></tr></thead>
  <tbody>__DIAG__</tbody></table>
  <h3>后续可优化项</h3>
  <ul class="rec">
    <li><b>提升单机 MFU：</b>当前瓶颈是显存压力与计算密度（seq/GBS 小 + selective recompute）。若硬件允许，先通过增加 DP 维度、CPU offload 或降低优化器状态占用腾出显存，再考虑关闭重计算、放大 seq / GBS。</li>
    <li><b>显存容量告警：</b>稳定期 fb 139,068 MB ≈ H200 141 GB 的 96%，余量极小；任何增大 seq、GBS 或开启额外缓冲的改动都需先做 OOM 评估。</li>
    <li><b>公平性能对比：</b>若要量化「单机 vs 双机」的扩展性，需固定 seq / GBS / 重计算策略，只改 DP 与世界规模，再比较每卡吞吐与 MFU。</li>
    <li><b>可观测性：</b>开启 <code>NCCL_DEBUG=INFO</code> 才能拿到通信通道与集合操作计时；本次日志缺少这部分，无法做通信剖析。</li>
    <li><b>退出清理：</b>显式调用 <code>destroy_process_group()</code> 可消除 8 条 ProcessGroupNCCL warning。</li>
  </ul>
</section>

<section id="s9">
  <h2><span class="no">9</span>数据附录</h2>
  <h3>输入文件</h3>
  <table><thead><tr><th>文件</th><th>类型</th><th>规模</th><th>用途</th></tr></thead><tbody>__FILES__</tbody></table>
  <h3>统计口径</h3>
  <table><thead><tr><th>项目</th><th>口径</th></tr></thead><tbody>
    <tr><td>时间轴</td><td>日志 UTC；dmon 为 UTC+8，已减 8 h 统一；t₀ = 2026-11-12 15:44:39 UTC</td></tr>
    <tr><td>稳定期</td><td>iteration ≥ 500 的 2,029 个记录点；预热期为 iter &lt; 500 的 49 个点</td></tr>
    <tr><td>MFU</td><td>Megatron 日志吞吐 ÷ H200 SXM BF16 dense 989.5 TFLOP/s；另列 6ND 理论口径</td></tr>
    <tr><td>完成判定</td><td>日志每 10 iter 打印，末条为 20,780；以 <code>[after training is done]</code> 与最终 val/test 评估判定跑满 20,786</td></tr>
    <tr><td>pviol / tviol</td><td>功率封顶时间占比（%）/ 是否发生热降频（bool）；两者均非故障标志</td></tr>
    <tr><td>分桶</td><td>GPU 时序 30 分钟一桶（38 桶）；启动阶段 1 分钟一桶；桶内先对 8 卡取均值</td></tr>
  </tbody></table>
  <div class="note">复现：<code>python analyze_single.py</code> → <code>stats_single.json</code>，
    <code>python report_single.py</code> → 本报告。报告为单文件、零 CDN 依赖，可离线打开。</div>
</section>
<footer>单机 56B lowmem1 · 统计分析报告 · 生成于 __GEN__ · 数据源：single_56b 训练日志 / dmon CSV / selfcheck</footer>
</div>
<script>
const D = __DATA__;
const S = D.stats, G = D.gpu, T = D.train, V = D.val, CMP = D.comparison, SER = D.series;
const PEAK = 989.5;
__LIB__
__CHARTS__
</script></body></html>"""

CHARTS = r"""
/* ============ 单机报告图表 ============ */
const regV = V.filter(v=>v.split==='val');
lineChart('c_loss',{series:[
  {n:'训练 lm loss',c:C.blue,data:T.map(d=>[d.iter,d.loss]),fmt:v=>n2(v,6)},
  {n:'验证 lm loss',c:C.purple,data:regV.map(v=>[v.iter,v.loss]),dots:false,fmt:v=>n2(v,6)}],
  xmin:0,xmax:20800,ymin:.003,ymax:12,logY:true,yfmt:v=>v>=1?n2(v,0):n2(v,3),
  xlabel:'iteration',ylabel:'loss (log)',h:360,
  markers:[{x:230,t:'早期梯度尖峰区',c:C.orange}]});
legend('l_loss',[{n:'训练 lm loss（每 10 iter）',c:C.blue},{n:'验证 lm loss（每 50 iter）',c:C.purple}]);
lineChart('c_ppl',{series:[{n:'验证 PPL',c:C.cyan,data:regV.map(v=>[v.iter,v.ppl]),fmt:v=>n2(v,4)}],
  xmin:0,xmax:20800,ymin:1,ymax:40000,logY:true,yfmt:v=>v>=10?n2(v,0):n2(v,1),
  xlabel:'iteration',ylabel:'PPL (log)',h:280});
lineChart('c_gnorm',{series:[{n:'grad norm',c:C.green,data:T.map(d=>[d.iter,d.gnorm]),fmt:v=>n2(v,3)}],
  xmin:0,xmax:20800,ymin:.01,ymax:200,logY:true,yfmt:v=>v>=1?n2(v,0):n2(v,2),
  xlabel:'iteration',ylabel:'norm (log)',h:280,
  markers:[{x:230,t:'113.3',c:C.red}]});

lineChart('c_perf',{dual:true,series:[
  {n:'吞吐',c:C.blue,data:T.map(d=>[d.iter,d.tflops]),fmt:v=>n2(v,1)+' TFLOP/s'},
  {n:'迭代耗时',c:C.orange,data:T.map(d=>[d.iter,d.ms/1000]),axis:2,fmt:v=>n2(v,3)+' s'}],
  xmin:0,xmax:20800,ymin:280,ymax:470,y2min:3.0,y2max:5.0,xlabel:'iteration',
  ylabel:'TFLOP/s/GPU',y2label:'s / iter',h:340});
legend('l_perf',[{n:'每卡吞吐（左轴）',c:C.blue},{n:'迭代耗时（右轴）',c:C.orange}]);
const stable=T.filter(d=>d.iter>=500);
histo('c_hist_tf',stable.map(d=>d.tflops),2,C.blue,'TFLOP/s/GPU');
histo('c_hist_ms',stable.map(d=>d.ms/1000),.02,C.orange,'秒 / iteration');
lineChart('c_mfu',{series:[{n:'MFU',c:C.green,data:T.map(d=>[d.iter,d.tflops/PEAK*100]),fmt:v=>n2(v,2)+'%'}],
  xmin:0,xmax:20800,ymin:28,ymax:47,xlabel:'iteration',ylabel:'MFU %',h:300,
  markers:[{x:10,t:'warmup 30.1%',c:C.orange}]});
legend('l_mfu',[{n:'MFU（每 10 iter 记录点）',c:C.green}]);
histo('c_mfu_hist',stable.map(d=>d.tflops/PEAK*100),.2,C.green,'MFU %');

const B=G.buckets, bt=B.map(b=>b.t);
const bs=k=>B.map(b=>[b.t,b[k]]);
const sfb=B.map(b=>[b.t,b.fb/1024]);
lineChart('c_gpu_time',{dual:true,series:[
  {n:'平均功耗',c:C.blue,data:bs('pwr'),fmt:v=>n2(v,1)+' W'},
  {n:'平均温度',c:C.red,data:bs('gtemp'),axis:2,fmt:v=>n2(v,1)+' °C'}],
  xmin:0,xmax:19,ymin:620,ymax:690,y2min:50,y2max:65,xfmt:v=>n2(v,0)+'h',
  xlabel:'运行时长',ylabel:'功耗 W',y2label:'温度 °C',h:300});
legend('l_gpu_time',[{n:'功耗（左轴）',c:C.blue},{n:'温度（右轴）',c:C.red}]);
lineChart('c_clk',{dual:true,series:[
  {n:'核心频率',c:C.green,data:bs('pclk'),fmt:v=>n2(v,0)+' MHz'},
  {n:'pviol',c:C.orange,data:bs('pviol'),axis:2,dash:'5 3',fmt:v=>n2(v,1)+'%'}],
  xmin:0,xmax:19,ymin:1830,ymax:1980,y2min:0,y2max:105,xfmt:v=>n2(v,0)+'h',
  xlabel:'运行时长',ylabel:'核心频率 MHz',y2label:'pviol %',h:300});
legend('l_clk',[{n:'核心频率（实线·左轴）',c:C.green},{n:'pviol（虚线·右轴）',c:C.orange}]);
lineChart('c_startup',{dual:true,series:[
  {n:'显存占用',c:C.purple,data:G.startup.map(b=>[b.m,b.fb/1024]),fmt:v=>n2(v,1)+' GiB'},
  {n:'功耗',c:C.orange,data:G.startup.map(b=>[b.m,b.pwr]),axis:2,fmt:v=>n2(v,0)+' W'}],
  xmin:0,xmax:30,ymin:0,ymax:150,y2min:0,y2max:760,xfmt:v=>n2(v,0)+'m',
  xlabel:'启动后分钟',ylabel:'显存 GiB',y2label:'功耗 W',h:280});
legend('l_startup',[{n:'显存占用（左轴）',c:C.purple},{n:'功耗（右轴）',c:C.orange}]);
const pd=G.pviol_dist, ptot=Object.values(pd).reduce((a,b)=>a+b,0);
barChart('c_power',{data:[
  {k:'0%',v:pd['0'],c:C.green},{k:'1–49%',v:pd['1-49'],c:C.cyan},
  {k:'50–89%',v:pd['50-89'],c:C.blue},{k:'90–99%',v:pd['90-99'],c:C.orange},
  {k:'100%',v:pd['100'],c:C.red}],h:280,yfmt:v=>n2(v,0),
  vfmt:v=>n2(v/ptot*100,1)+'% ('+n2(v,0)+')',xlabel:'pviol 档位'});

const metrics=[['pwr','平均功耗 (W)',C.blue,640,675],['gtemp','平均温度 (°C)',C.red,50,66],
  ['pclk','平均核心频率 (MHz)',C.green,1900,1960],['fb','平均显存 (MB)',C.purple,138900,139100]];
box('gpu_bars').innerHTML='';
metrics.forEach((m,i)=>{
  const div=document.createElement('div');div.className='chart';div.id='gb'+i;
  div.innerHTML='<div class="ct">'+m[1]+'</div><div class="cs">偶数卡深蓝 / 奇数卡浅青</div>';
  box('gpu_bars').appendChild(div);
  const arr=[];for(let g=0;g<8;g++)arr.push({k:'GPU'+g,v:G.summary[g][m[0]].mean,c:g%2===0?C.blue:C.cyan});
  barChart('gb'+i,{data:arr,h:230,ymin:m[3],ymax:m[4],vfmt:v=>n2(v,1)});
});
const heatVals=[];for(let g=0;g<8;g++)heatVals.push(B.map(b=>b.per_gpu[g].gtemp));
heatmap('c_heat',Array.from({length:8},(_,g)=>'GPU'+g),bt,heatVals,50,67,' °C');

const cats=Object.entries(D.categories).map(([k,v])=>[k==='RerunStateMachine 警告'?'RerunState 警告':k,v]);
barChart('c_events',{horizontal:true,data:cats.map(([k,v])=>({k:k,v:v,c:C.blue})),h:300,labW:170,
  xfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 行 ('+n2(v/D.meta.log_lines*100,1)+'%)'});
lineChart('c_cmp_tf',{series:[
  {n:'单机 lowmem1',c:C.blue,data:SER.single_tflops,fmt:v=>n2(v,1)+' TFLOP/s'},
  {n:'双机 node1',c:C.orange,data:SER.dual_tflops,fmt:v=>n2(v,1)+' TFLOP/s'}],
  xmin:0,xmax:24,ymin:280,ymax:560,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',
  ylabel:'TFLOP/s/GPU',h:300});
legend('l_cmp_tf',[{n:'单机 lowmem1',c:C.blue},{n:'双机 node1',c:C.orange}]);

const cmItems=[
  {k:'吞吐',a:S.throughput_stable.mean,b:524.9671},
  {k:'MFU',a:S.mfu_stable,b:53.05},
  {k:'SM 利用率',a:G.sm_ge90_pct,b:99.9},
  {k:'功耗',a:S.pwr_avg,b:676.944},
  {k:'显存',a:G.summary['0'].fb.p50,b:110796},
  {k:'功率封顶',a:G.power_cap_pct,b:76.2}];
const cmData=[];
cmItems.forEach(r=>{cmData.push({k:r.k,v:+(r.a/r.b*100).toFixed(2),c:C.blue});
                    cmData.push({k:'',v:100,c:C.orange});});
barChart('c_cmp',{data:cmData,h:320,ymin:70,ymax:135,yfmt:v=>n2(v,0)+'%',
  vfmt:v=>n2(v,1)+'%',xlabel:'双机 = 100%（每组左蓝单机 / 右橙双机）'});
legend('l_cmp',[{n:'单机相对值',c:C.blue},{n:'双机基准 100%',c:C.orange}]);
lineChart('c_cmp_loss',{series:[
  {n:'单机',c:C.blue,data:SER.single_loss,fmt:v=>n2(v,5)},
  {n:'双机',c:C.orange,data:SER.dual_loss,fmt:v=>n2(v,5)}],
  xmin:0,xmax:24,ymin:.003,ymax:12,logY:true,yfmt:v=>v>=1?n2(v,0):n2(v,3),
  xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'loss (log)',h:300});
legend('l_cmp_loss',[{n:'单机 lowmem1',c:C.blue},{n:'双机 node1',c:C.orange}]);
lineChart('c_cmp_fb',{series:[
  {n:'单机',c:C.blue,data:SER.single_fb.map(p=>[p[0],p[1]/1024]),fmt:v=>n2(v,1)+' GiB'},
  {n:'双机 node1',c:C.orange,data:SER.dual_fb.map(p=>[p[0],p[1]/1024]),fmt:v=>n2(v,1)+' GiB'}],
  xmin:0,xmax:24,ymin:90,ymax:145,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'显存 GiB',h:300});
legend('l_cmp_fb',[{n:'单机 lowmem1',c:C.blue},{n:'双机 node1',c:C.orange}]);
"""

repl = {
    "__CSS__": CSS, "__DATA__": DATA, "__LIB__": LIB, "__CHARTS__": CHARTS,
    "__KPIS__": kpi_html, "__CFG__": cfg_html, "__TL__": tl_html, "__MEM__": mem_html,
    "__MILE__": mile_html, "__STAT__": stat_html, "__EV__": ev_html,
    "__CMPCFG__": cmp_cfg_html, "__CMPMETRIC__": cmp_metric_html,
    "__DIAG__": diag_html, "__FILES__": files_html,
    "__CASE__": D["meta"]["case"],
    "__TS1__": G["ts_first"], "__TS2__": G["ts_last"],
    "__HOURS__": nf(S["hours"]), "__ITERS__": f'{S["completed_iters"]:,}',
    "__TARGET__": f'{S["iter_target"]:,}',
    "__TF__": nf(S["throughput_stable"]["mean"], 1), "__CV__": nf(S["throughput_stable"]["cv"]),
    "__MFU__": nf(S["mfu_stable"]), "__MFU6__": nf(S["mfu_6nd"]),
    "__F6__": nf(S["flops_6nd_per_gpu_tflops"], 1),
    "__PF__": nf(S["avg_tflops_node"] / 1000),
    "__PF6__": nf(S["flops_6nd_per_iter_pflops"]),
    "__MS__": nf(S["iter_ms_stable"]["mean"] / 1000, 3),
    "__GAP__": nf(S["gap_vs_6nd_pct"]), "__TPI__": f'{S["tokens_per_iter"]:,}',
    "__L0__": nf(S["train_loss_first"], 3), "__L1__": nf(S["train_loss_last"], 6),
    "__VPPL__": nf(S["final_val_ppl"], 6), "__TPPL__": nf(S["final_test_ppl"], 6),
    "__PCAP__": nf(G["power_cap_pct"], 1), "__TMAX__": str(S["gtemp_max"]),
    "__PWR__": nf(S["pwr_avg"], 1), "__PWRMAX__": str(S["pwr_max"]),
    "__CLKMIN__": "1860", "__CLKMAX__": "1965",
    "__FB__": f'{G["summary"]["0"]["fb"]["p50"]:,.0f}',
    "__FBG__": nf(G["summary"]["0"]["fb"]["p50"] / 1024, 1),
    "__FBSPREAD__": nf(S["fb_card_spread_MB"], 2),
    "__NCCL__": TOPO["nccl_version"] or "unknown",
    "__CONFTOTAL__": f'{D["meta"]["config_total"]:,}',
    "__GEN__": GEN,
}
for k, v in repl.items():
    HTML = HTML.replace(k, str(v))
left = [k for k in repl if k in HTML]
target = OUT / "单机统计分析报告.html"
target.write_text(HTML, encoding="utf-8")
print(f"生成: {target.name} | {target.stat().st_size/1024:.1f} KB | 内嵌数据 {len(DATA)/1024:.1f} KB")
print(f"残留占位符: {left}")
