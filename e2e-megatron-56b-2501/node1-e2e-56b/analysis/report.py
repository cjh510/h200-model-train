# -*- coding: utf-8 -*-
"""由 stats.json / stats_dual.json 生成离线自包含的中文图表分析报告（无 CDN 依赖）"""
import json
from datetime import datetime
from pathlib import Path
from dual_content import build as build_dual

ROOT = Path(r"C:/Users/26488/Desktop/ht/e2e-megatron-56b-2501/node1-e2e-56b")
OUT = ROOT / "analysis"
D = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
DUAL = json.loads((OUT / "stats_dual.json").read_text(encoding="utf-8"))
DC = build_dual(DUAL)
S = D["stats"]
SC = D["selfcheck"]
LG = D["log"]
GP = D["gpu"]

def f(v, n=2):
    return f"{v:,.{n}f}"

# ---------- KPI 卡片 ----------
kpis = [
    ("有效训练时长", f(S["hours"], 2), "h", "16:06:28 → 15:37:10 UTC", "blue"),
    ("完成迭代", f(S["iter_last"], 0), f"/ {S['iter_target']:,}", f"进度 {S['progress']}% · 定时退出", "blue"),
    ("稳定期吞吐", f(S["throughput_stable"]["mean"], 1), "TFLOP/s/GPU", f"CV {S['throughput_stable']['cv']}% · 8 卡 {S['avg_tflops_node']:,.0f}", "green"),
    ("迭代耗时", f(S["iter_ms_stable"]["mean"]/1000, 3), "s/iter", f"σ {S['iter_ms_stable']['sd']/1000:.3f} s · CV {S['iter_ms_stable']['cv']}%", "green"),
    ("最终验证 loss", f(S["val_loss_last"], 6), "", f"PPL {S['ppl_last']:.6f} · 初始 {S['ppl_first']:,.0f}", "purple"),
    ("训练 loss 降幅", f((1-S["train_loss_last"]/S["train_loss_first"])*100, 2), "%", f"{S['train_loss_first']:.4f} → {S['train_loss_last']:.6f}", "purple"),
    ("GPU 计算占用", f(S["sm_ge90_pct"], 1), "% SM≥90%", f"平均 ~98.9% · {S['gpu_rows']:,} 采样点", "cyan"),
    ("单卡显存峰值", f(S["fb_peak_mb"]/1024, 1), "GiB", "8 卡差异 &lt; 1 MB", "cyan"),
    ("NaN / 跳过迭代", "0 / 0", "", "全程无数值异常", "green"),
    ("致命错误", "0", "", "无 OOM / Traceback / CUDA error / NCCL 超时", "green"),
    ("功率封顶采样", f(S["power_cap_pct"], 1), "%", f"pviol=100 · tviol 全 0（无热降频）", "orange"),
    ("关键参数覆盖", f"{sum(D['flags']['cover'].values())}/{len(D['flags']['cover'])}", "", f"共 {D['flags']['total_lines']} 条参数 · MISSING_FLAGS=none", "blue"),
]
kpi_html = "\n".join(
    f'<div class="kpi k-{c}"><div class="kpi-t">{t}</div><div class="kpi-v">{v}<span class="kpi-u">{u}</span></div><div class="kpi-s">{s}</div></div>'
    for t, v, u, s, c in kpis)

# ---------- 诊断表 ----------
diag = [
    ("日志末尾 56 条 NCCL <code>Abort COMPLETE</code>，看似崩溃",
     "最后一条 iteration 为 15:36:41 UTC（iter 3680）；紧随其后是 8 条 ResourceWarning（临时目录清理）与 15:37:06 的 8 条 <code>process group has NOT been destroyed</code>；全日志 Traceback / OOM / CUDA error / NCCL 超时均为 0",
     "正常定时退出时的进程组析构，<b>非故障</b>", "无", "可显式调用 destroy_process_group() 消除 warning", "ok"),
    ("训练全程 <code>pviol</code> 高位（100% 占 76.2%，90~99% 占 22.0%）",
     f"123,280 个采样中 93,943 次 pviol=100；<code>tviol</code> 全为 0；平均功耗 {f(S['pwr_even_avg'],1)} W、峰值 744 W；pclk 在 1860–1965 MHz 间自适应浮动",
     "GPU 长期处于 <b>TDP 功率墙</b> 约束，散热未触发限频", "吞吐存在功耗上限，但 23.5h 内 CV 仅 0.32%，未造成波动", "若追求更高 MFU，评估提升 power limit 或优化供电/风道", "warn"),
    ("偶数卡比奇数卡高约 7.3 °C",
     f"GPU 0/2/4/6 均温 {f(S['gtemp_even_avg'],1)} °C，GPU 1/3/5/7 均温 {f(S['gtemp_odd_avg'],1)} °C；最高 66 °C；对应 pclk 低约 3.5%（GPU4 1862 vs GPU5 1929 MHz）",
     "机架风道 / GPU 拓扑散热差异，属物理层正常离散", "轻微；无热降频、无吞吐差异", "检查机柜进出风与 GPU 布局", "info"),
    ("iter 380/390 梯度范数爆炸并引发 loss 反弹",
     "grad norm 174.7 → 324.3（全期中位数 0.09）；训练 loss 0.056 → 0.289；iter 400 验证 loss 0.0212 → 0.0423（+99.8%）；iter 410 即回落至 0.0354",
     "训练早期数值不稳定，<b>已自愈</b>，未产生 NaN 或 skipped iteration", "短暂，约 30 个 iteration 内恢复", "换真实数据训练时建议确认 grad clip 与 lr warmup 配置", "warn"),
    ("iter 3450 验证 loss 反弹 +47%",
     "0.004999 → 0.007350，随后 3500 回到 0.005337",
     "验证集小样本波动，绝对量级仅 10⁻³", "可忽略", "无需处理", "info"),
    ("40 条 <code>Failed to find ncclNetPlugin_v9 / TunerPlugin</code>",
     "NCCL 回退 <code>internal tuner</code> 与 <code>IBext_v8</code>；通信域正常建立（nranks 2 / 8 / 16 全部 Connected）",
     "可选插件缺失，<b>非错误</b>", "可能损失少量网络调优收益", "如需定制 IB/拓扑调优可安装对应插件", "info"),
    ("args-help 512 条参数行，去重后 502 个唯一名称",
     "selfcheck.sh 用 <code>grep -cE '^ *--'</code> 计行；本报告的正则按参数名去重，两者口径不同",
     "统计口径差异", "无", "记录口径即可", "ok"),
]
diag_html = "\n".join(
    f'<tr><td class="phen">{p}</td><td>{e}</td><td><span class="tag tag-{lv}">{j}</span></td><td>{i}</td><td>{a}</td></tr>'
    for p, e, j, i, a, lv in diag)

# ---------- E2E 检查项 ----------
checks = [
    ("8 卡识别与组件版本", "通过", f"GPUS={SC.get('GPUS')} · PyTorch {SC.get('TORCH')} · CUDA {SC.get('CUDA')} · TE {SC.get('TE')}"),
    ("关键训练参数可用性", "通过", f"12/12 命中，MISSING_FLAGS={SC.get('MISSING_FLAGS')}；含 exit-duration-in-mins / fp8-format / tp-comm-overlap"),
    ("长时稳定运行", "通过", f"{f(S['hours'],2)} h · 3,680 iter · 0 NaN / 0 skipped · 吞吐 CV {S['throughput_stable']['cv']}%"),
    ("模型收敛", "通过", f"train loss {S['train_loss_first']:.3f} → {S['train_loss_last']:.6f}；val PPL {S['ppl_first']:,.0f} → {S['ppl_last']:.6f}"),
    ("无致命错误", "通过", "Traceback 0 · OOM 0 · CUDA error 0 · NCCL timeout 0 · watchdog 0"),
    ("GPU 满载且无热降频", "通过", f"SM≥90% 占 {S['sm_ge90_pct']}% · 最高 66 °C · tviol 全 0"),
    ("跑满 100,000 iteration", "不适用", f"仅完成 {S['progress']}%，符合长时 E2E 压测按运行时长退出的设计"),
]
check_html = "\n".join(
    f'<tr><td>{a}</td><td><span class="tag tag-{"ok" if b=="通过" else "info"}">{b}</span></td><td>{c}</td></tr>'
    for a, b, c in checks)

flags_html = "".join(
    f'<span class="chip {"chip-ok" if v else "chip-bad"}">--{k}</span>' for k, v in D["flags"]["cover"].items())

n_head = (S["csv_lines"] - 2 - S["gpu_rows"]) // 2
files = [
    ("logs/…20261112_000626.out", "训练主日志", f"{D['meta']['log_lines']:,} 行 / 514 KB", "iteration、validation、耗时、NCCL 事件、退出清理"),
    ("logs/…-dmon-…000626.csv", "GPU 监控", f"{S['csv_lines']:,} 行 / 20.3 MB", f"有效采样 {S['gpu_rows']:,} 行（8 卡 × 15,410 时间点），含 {n_head:,} 组重复表头"),
    ("args-help.txt", "参数清单", f"122 KB / {D['flags']['total_lines']} 条 -- 行", "pretrain_gpt.py --help 输出，去重 502 个唯一参数名"),
    ("selfcheck.log", "环境自检", "36 行", "8 卡 / Python / Torch / CUDA / TE 版本与关键 flag 检查"),
    ("selfcheck.sh", "自检脚本", "34 行", "生成 args-help 并校验 12 个关键参数"),
    ("image-digest.txt", "镜像摘要", "72 B", "sha256:e28aab520934…"),
]
file_html = "\n".join(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in files)

DATA = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
GEN = datetime.now().strftime("%Y-%m-%d %H:%M")

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>node1 · E2E 56B 训练结果统计分析</title>
<style>
:root{--bg:#f6f8fc;--card:#fff;--bd:#e5eaf2;--tx:#1e293b;--tx2:#64748b;--tx3:#94a3b8;
--blue:#2563eb;--cyan:#0891b2;--purple:#7c3aed;--green:#16a34a;--orange:#ea580c;--red:#dc2626;
--grid:#eef2f7;--shadow:0 1px 3px rgba(15,23,42,.05),0 8px 24px rgba(15,23,42,.04)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1220px;margin:0 auto;padding:28px 22px 60px}
header.top{background:linear-gradient(120deg,#1e3a8a,#2563eb 55%,#0891b2);color:#fff;border-radius:16px;padding:30px 34px;box-shadow:var(--shadow)}
header.top h1{margin:0 0 6px;font-size:27px;letter-spacing:.5px}
header.top .sub{opacity:.9;font-size:14px}
header.top .meta{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
header.top .meta span{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);padding:4px 11px;border-radius:20px;font-size:12.5px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 22px}
nav a{color:var(--tx2);text-decoration:none;background:var(--card);border:1px solid var(--bd);padding:6px 13px;border-radius:8px;font-size:13px}
nav a:hover{color:var(--blue);border-color:var(--blue)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:13px;padding:16px 18px;box-shadow:var(--shadow);border-top:3px solid var(--blue)}
.kpi-t{color:var(--tx2);font-size:12.5px;margin-bottom:6px}
.kpi-v{font-size:25px;font-weight:700;letter-spacing:-.5px;line-height:1.25}
.kpi-u{font-size:12px;font-weight:500;color:var(--tx2);margin-left:6px;letter-spacing:0}
.kpi-s{color:var(--tx3);font-size:11.5px;margin-top:5px}
.k-green{border-top-color:var(--green)}.k-purple{border-top-color:var(--purple)}
.k-cyan{border-top-color:var(--cyan)}.k-orange{border-top-color:var(--orange)}
.k-green .kpi-v{color:#15803d}.k-purple .kpi-v{color:#6d28d9}.k-cyan .kpi-v{color:#0e7490}
.k-orange .kpi-v{color:#c2410c}.k-blue .kpi-v{color:#1d4ed8}
section{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:22px 24px 26px;margin-bottom:20px;box-shadow:var(--shadow)}
h2{font-size:19px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
h2 .no{display:inline-flex;width:26px;height:26px;border-radius:7px;background:#eff6ff;color:var(--blue);font-size:13px;align-items:center;justify-content:center;font-weight:700}
h3{font-size:15px;margin:24px 0 10px;color:var(--tx)}
.lead{color:var(--tx2);margin:0 0 18px;font-size:13.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.chart{background:#fcfdff;border:1px solid var(--bd);border-radius:11px;padding:12px 12px 8px;position:relative}
.chart .ct{font-size:13.5px;font-weight:600;margin:2px 0 0 6px;color:var(--tx)}
.chart .cs{font-size:11.5px;color:var(--tx3);margin:0 0 4px 6px}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:#0f172a;color:#fff;padding:7px 10px;border-radius:7px;font-size:12px;line-height:1.6;white-space:nowrap;opacity:0;transition:opacity .1s;z-index:20;box-shadow:0 4px 14px rgba(0,0,0,.2)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 2px 8px;font-size:12px;color:var(--tx2)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
th,td{border-bottom:1px solid var(--bd);padding:9px 10px;text-align:left;vertical-align:top}
th{background:#f8fafc;color:var(--tx2);font-weight:600;font-size:12.5px;white-space:nowrap}
td code,.chip{background:#f1f5f9;color:#475569;padding:1px 5px;border-radius:4px;font-size:11.5px}
.phen{font-weight:600;min-width:190px}
.tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:600;white-space:nowrap}
.tag-ok{background:#dcfce7;color:#15803d}.tag-warn{background:#ffedd5;color:#c2410c}
.tag-info{background:#e0f2fe;color:#0369a1}.tag-err{background:#fee2e2;color:#b91c1c}
.verdict{display:flex;gap:14px;align-items:flex-start;background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid var(--green);border-radius:10px;padding:15px 18px;margin-bottom:18px}
.verdict .ic{font-size:22px;font-weight:800;color:var(--green)}
.verdict b{color:#15803d}
.note{background:#f8fafc;border:1px dashed var(--bd);border-radius:9px;padding:11px 14px;color:var(--tx2);font-size:12.5px;margin-top:14px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.chip-ok{background:#dcfce7;color:#15803d}.chip-bad{background:#fee2e2;color:#b91c1c}
ul.rec{margin:8px 0 0;padding-left:20px;color:var(--tx2);font-size:13px}
ul.rec li{margin-bottom:7px}
ul.rec b{color:var(--tx)}
footer{text-align:center;color:var(--tx3);font-size:12px;margin-top:26px}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
</style>
</head>
<body><div class="wrap">
<header class="top">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="meta">__HEAD_META__</div>
</header>
<nav>
  __NAV_DUAL__<a href="#s1">结论</a><a href="#s2">收敛曲线</a><a href="#s3">计算性能</a>
  <a href="#s4">GPU 硬件</a><a href="#s5">日志与自检</a><a href="#s6">诊断表</a><a href="#s7">附录</a>
</nav>
<div class="kpis">__KPIS__</div>
__KPI_NOTE__
__SECTION_A__
<section id="s1">
  <h2><span class="no">1</span>总体结论</h2>
  <div class="verdict"><div class="ic">✓</div><div>
    <b>长时 E2E 稳定性压测通过，训练按运行时长正常退出，未发生崩溃。</b><br>
    连续运行 __HOURS__ 小时完成 3,680 次迭代（目标的 __PROG__%），吞吐稳定在 __TF__ TFLOP/s/GPU（CV __CV__%）；
    loss 从 9.75 收敛至 0.005、验证 PPL 从 3,764 降至 1.0054；全程 0 NaN、0 skipped、0 OOM、0 CUDA error、0 NCCL 超时。
    日志末尾的 56 条 NCCL <code>Abort COMPLETE</code> 是进程组析构清理，不是故障。
  </div></div>
  <h3>E2E 检查项逐条判定</h3>
  <table><thead><tr><th>检查项</th><th>判定</th><th>证据</th></tr></thead><tbody>__CHECKS__</tbody></table>
  <div class="note">__SCOPE_NOTE__</div>
</section>

<section id="s2">
  <h2><span class="no">2</span>收敛与数值行为</h2>
  <p class="lead">368 个训练记录点（每 10 iter 打印一次）+ 73 次验证；loss 使用对数坐标以同时呈现早期下降与后期平台。</p>
  <div class="chart" id="c_loss"><div class="ct">训练 / 验证 loss 收敛曲线</div><div class="cs">对数纵轴 · 横轴 iteration</div><div class="legend" id="l_loss"></div></div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_ppl"><div class="ct">验证困惑度 PPL</div><div class="cs">对数纵轴 · 3,764 → 1.005</div></div>
    <div class="chart" id="c_gnorm"><div class="ct">梯度范数 grad norm</div><div class="cs">对数纵轴 · 红色标记为 &gt;8 的尖峰</div></div>
  </div>
  <h3>收敛里程碑</h3>
  <table><thead><tr><th>训练 loss 阈值</th><th>首次低于该值的 iteration</th><th>对应训练时长</th><th>说明</th></tr></thead><tbody id="t_mile"></tbody></table>
  <div class="note">loss 在约 800 iter 内降到 0.01 以下、2,400 iter 后进入 ~0.005 平台，属于典型的合成/记忆化数据收敛形态，符合 E2E 功能验证（而非真实预训练质量评估）的目的。</div>
</section>

<section id="s3">
  <h2><span class="no">3</span>计算性能与稳定性</h2>
  <p class="lead">稳定期定义为 iteration ≥ 500（剔除 warmup），共 319 个采样点。</p>
  <div class="chart" id="c_perf"><div class="ct">每卡吞吐与迭代耗时</div><div class="cs">双纵轴 · 吞吐越平稳说明长时运行越健康</div><div class="legend" id="l_perf"></div></div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_hist_tf"><div class="ct">稳定期吞吐分布</div><div class="cs">直方图 · TFLOP/s/GPU</div></div>
    <div class="chart" id="c_hist_ms"><div class="ct">稳定期迭代耗时分布</div><div class="cs">直方图 · 秒 / iteration</div></div>
  </div>
  <h3>关键统计量</h3>
  <table><thead><tr><th>指标</th><th>样本数</th><th>均值</th><th>标准差</th><th>CV</th><th>最小 / P50 / P95 / 最大</th></tr></thead><tbody id="t_stat"></tbody></table>
  <div class="note">73 次验证平均耗时 __EVAL__ s（首次 9.69 s，之后稳定在 8.80 s 左右），累计约 10.7 分钟，占全程 0.76%；
    首个 iteration 吞吐 432.6 TFLOP/s/GPU 为编译/warmup 阶段，10 个记录点内即升至 506+。</div>
  __MFU_BLOCK__
</section>

<section id="s4">
  <h2><span class="no">4</span>GPU 硬件与功耗</h2>
  <p class="lead">__S4_LEAD__</p>
  <div class="grid2">
    <div class="chart" id="c_gpu_time"><div class="ct">23.5 小时功耗 / 温度时序</div><div class="cs">双纵轴 · 30 分钟分桶 · 8 卡均值</div><div class="legend" id="l_gpu_time"></div></div>
    <div class="chart" id="c_clk"><div class="ct">核心频率与功率墙 pviol 时序</div><div class="cs">频率随功率预算自适应浮动</div><div class="legend" id="l_clk"></div></div>
  </div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_startup"><div class="ct">启动阶段：显存占用与功耗爬升</div><div class="cs">前 30 分钟按 1 分钟聚合 · 2 分钟内即满载</div><div class="legend" id="l_startup"></div></div>
    <div class="chart" id="c_power"><div class="ct">功率墙 pviol 采样分布</div><div class="cs">pviol = 采样周期内触发功率上限的时间占比</div></div>
  </div>
  <h3>8 卡横向对比</h3>
  <div class="grid2" id="gpu_bars"></div>
  <h3>温度热力图（8 卡 × 30 分钟）</h3>
  <div class="chart" id="c_heat"><div class="ct">GPU 温度分布</div><div class="cs">偶数卡（0/2/4/6）系统性高于奇数卡（1/3/5/7）</div></div>
  <div class="note"><b>功率墙而非温度墙：</b>__PCAP__% 的采样点 pviol=100（全程功率封顶），而 tviol 全部为 0、最高温度仅 66 °C；
    核心频率在 1860–1965 MHz 间自适应调整。显存稳定在 110,796 MB（≈108.2 GiB），8 卡间差异 &lt; 1 MB，负载均衡极佳。</div>
</section>

__SECTION_B__
<section id="s5">
  <h2><span class="no">5</span>日志构成与环境自检</h2>
  <div class="grid2">
    <div class="chart" id="c_events"><div class="ct">node1 主日志 4,387 行构成</div><div class="cs">按行类型分类 · 横向条形</div></div>
    <div class="chart" id="c_abort"><div class="ct">退出清理：Abort COMPLETE 按通信域规模</div><div class="cs">nranks 2 / 8 / 16 分别对应 DP / TP / 全局通信域</div></div>
  </div>
  <h3>事件计数</h3>
  <table><thead><tr><th>类别</th><th>事件</th><th>次数</th><th>性质</th></tr></thead><tbody id="t_events"></tbody></table>
  <h3>容器与环境自检</h3>
  <table><thead><tr><th>项目</th><th>值</th><th>项目</th><th>值</th></tr></thead><tbody id="t_self"></tbody></table>
  <h3>12 个关键训练参数覆盖</h3>
  <div class="chips">__FLAGS__</div>
  <div class="note">镜像摘要：<code>__DIGEST__</code>；selfcheck.sh 检查的 12 个参数全部命中，<code>MISSING_FLAGS=none</code>。</div>
</section>

__SECTION_C__
<section id="s6">
  <h2><span class="no">6</span>根因排查与建议</h2>
  <p class="lead">按「现象 → 证据 → 判定 → 影响 → 建议」结构化呈现，区分真实故障与正常清理/告警噪声。</p>
  <table><thead><tr><th style="width:19%">现象</th><th style="width:34%">证据</th><th style="width:18%">判定</th><th style="width:14%">影响</th><th style="width:15%">建议</th></tr></thead>
  <tbody>__DIAG__</tbody></table>
  <h3>后续可优化项</h3>
  <ul class="rec">
    <li><b>功率预算：</b>若目标是把 MFU 再往上推，优先评估提升 GPU power limit 或机柜供电/散热余量——当前瓶颈是 TDP 墙，不是散热（tviol=0）。</li>
    <li><b>数值稳定性：</b>iter 380–400 的梯度爆炸提示在真实数据训练前应确认 grad clip 与 lr warmup；本例虽自愈，但换数据分布后未必。</li>
    <li><b>退出清理：</b>在训练脚本结束处显式 <code>destroy_process_group()</code>，可消除 8 条 ProcessGroupNCCL warning 与部分 Abort 日志噪声。</li>
    <li>__OBS_ITEM__</li>
    <li><b>监控口径：</b>dmon 每 45 行重复写表头（共 2,802 组），长期采集建议加 <code>-d</code> 或用 CSV 解析器过滤 <code>#</code> 行。</li>
  </ul>
</section>

<section id="s7">
  <h2><span class="no">7</span>数据附录</h2>
  <h3>输入文件清单</h3>
  <table><thead><tr><th>文件</th><th>类型</th><th>规模</th><th>用途</th></tr></thead><tbody>__FILES____EXTRA_FILES__</tbody></table>
  <h3>统计口径</h3>
  <table><thead><tr><th>项目</th><th>口径</th></tr></thead><tbody>
    <tr><td>时间轴</td><td>日志 UTC；dmon 为 UTC+8 本地时间，已减 8 h 统一；t₀ = 2026-11-11 16:06:28 UTC</td></tr>
    <tr><td>稳定期</td><td>iteration ≥ 500 的 319 个记录点；warmup 为 iter &lt; 500 的 49 个点</td></tr>
    <tr><td>CV</td><td>总体标准差 / 均值 × 100%（pandas 口径为样本标准差，此处用 pstdev）</td></tr>
    <tr><td>分桶</td><td>GPU 时序按 30 分钟聚合（48 桶）；启动阶段按 1 分钟聚合（30 桶）；桶内先对 8 卡取均值</td></tr>
    <tr><td>pviol / tviol</td><td>nvidia-smi dmon 的功率违规 / 热违规时间占比（本例 tviol 列恒为 0）</td></tr>
    <tr><td>日志分类</td><td>按行内容正则归类，优先级：iteration → validation → NCCL INFO → timing → Warning → 其他</td></tr>
  </tbody></table>
  <div class="note">复现方式：在本目录运行 <code>python analyze.py</code> 生成 stats.json，再运行 <code>python report.py</code> 重建本报告。报告为单文件、零外部依赖，可离线打开。</div>
</section>
<footer>__FOOTER__</footer>
</div>
<script>
const DUAL = __DATA__;
const D = DUAL.n1 || DUAL;
const S = D.stats, G = D.gpu, T = D.train, V = D.val, L = D.log;
const NS = 'http://www.w3.org/2000/svg';
const C = {blue:'#2563eb',cyan:'#0891b2',purple:'#7c3aed',green:'#16a34a',orange:'#ea580c',red:'#dc2626',grid:'#eef2f7',axis:'#94a3b8',txt:'#64748b'};
function E(tag,a,p){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function n2(v,d){d=d===undefined?2:d;if(v===null||v===undefined||isNaN(v))return '-';return Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});}
function ticks(a,b,n){if(a===b){a-=1;b+=1;}const raw=(b-a)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),norm=raw/mag;let step=mag*((norm<1.5)?1:(norm<3)?2:(norm<7)?5:10);const lo=Math.floor(a/step)*step,hi=Math.ceil(b/step)*step,out=[];for(let x=lo;x<=hi+step*.001;x+=step)out.push(+x.toFixed(10));return out;}
function logticks(a,b){const out=[];for(let e=Math.floor(Math.log10(a));e<=Math.ceil(Math.log10(b));e++)out.push(Math.pow(10,e));return out;}
function box(id){return document.getElementById(id);}
function tipFor(el){let t=el.querySelector('.tip');if(!t){t=document.createElement('div');t.className='tip';el.appendChild(t);}return t;}
function legend(id,items){const e=box(id);if(!e)return;e.innerHTML=items.map(i=>'<span><i style="background:'+i.c+'"></i>'+i.n+'</span>').join('');}

/* 通用折线图：支持双轴、对数轴、标记点、hover tooltip */
function lineChart(id,cfg){
  const host=box(id),W=host.closest('.grid2')?620:1000,H=cfg.h||330,M={l:64,r:cfg.dual?64:22,t:14,b:44};
  const svg=E('svg',{viewBox:'0 0 '+W+' '+H,role:'img'},host);
  const ss=cfg.series, xmin=cfg.xmin,xmax=cfg.xmax;
  let y1min=cfg.ymin,y1max=cfg.ymax,y2min=cfg.y2min,y2max=cfg.y2max;
  if(y1min===undefined){let a=[];ss.filter(s=>s.axis!==2).forEach(s=>s.data.forEach(p=>a.push(p[1])));y1min=Math.min(...a);y1max=Math.max(...a);}
  if(cfg.dual&&y2min===undefined){let a=[];ss.filter(s=>s.axis===2).forEach(s=>s.data.forEach(p=>a.push(p[1])));y2min=Math.min(...a);y2max=Math.max(...a);}
  const lg=!!cfg.logY, eps=1e-12;
  const fy=v=>lg?Math.log10(Math.max(v,eps)):v;
  const lo1=fy(y1min),hi1=fy(y1max),lo2=cfg.dual?fy(y2min):0,hi2=cfg.dual?fy(y2max):1;
  const X=v=>M.l+(v-xmin)/(xmax-xmin)*(W-M.l-M.r);
  const Y=(v,ax)=>H-M.b-(ax===2?(fy(v)-lo2)/(hi2-lo2||1):(fy(v)-lo1)/(hi1-lo1||1))*(H-M.t-M.b);
  const yt=lg?logticks(y1min,y1max):ticks(y1min,y1max,5);
  yt.forEach(v=>{if(v<y1min*0.999||v>y1max*1.001)return;const y=Y(v,1);
    E('line',{x1:M.l,x2:W-M.r,y1:y,y2:y,stroke:C.grid,'stroke-width':1},svg);
    E('text',{x:M.l-9,y:y+4,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=cfg.yfmt?cfg.yfmt(v):n2(v,lg?(v<1?3:0):1);});
  if(cfg.dual){ticks(y2min,y2max,5).forEach(v=>{if(v<y2min||v>y2max)return;const y=Y(v,2);
    E('text',{x:W-M.r+9,y:y+4,'font-size':11,fill:C.txt},svg).textContent=cfg.y2fmt?cfg.y2fmt(v):n2(v,1);});}
  ticks(xmin,xmax,6).forEach(v=>{if(v<xmin||v>xmax)return;const x=X(v);
    E('line',{x1:x,x2:x,y1:M.t,y2:H-M.b,stroke:C.grid,'stroke-width':1},svg);
    E('text',{x:x,y:H-M.b+18,'text-anchor':'middle','font-size':11,fill:C.txt},svg).textContent=cfg.xfmt?cfg.xfmt(v):n2(v,0);});
  E('line',{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:'#cbd5e1'},svg);
  E('text',{x:W-M.r,y:H-M.b+36,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=cfg.xlabel||'';
  E('text',{x:M.l,y:11,'font-size':11,fill:C.txt},svg).textContent=cfg.ylabel||'';
  if(cfg.y2label)E('text',{x:W-M.r,y:11,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=cfg.y2label;
  ss.forEach(s=>{
    const pts=s.data.map(p=>X(p[0])+','+Y(p[1],s.axis===2?2:1)).join(' ');
    E('polyline',{points:pts,fill:'none',stroke:s.c,'stroke-width':s.w||2,'stroke-dasharray':s.dash||'','stroke-linejoin':'round','stroke-linecap':'round'},svg);
    if(s.dots)s.data.forEach(p=>E('circle',{cx:X(p[0]),cy:Y(p[1],s.axis===2?2:1),r:2.6,fill:s.c},svg));
  });
  (cfg.markers||[]).forEach((m,i)=>{const x=X(m.x),ty=M.t+11+(i%2)*14;
    E('line',{x1:x,x2:x,y1:M.t,y2:H-M.b,stroke:m.c||C.red,'stroke-width':1,'stroke-dasharray':'3 3',opacity:.7},svg);
    if(!m.t)return;
    E('rect',{x:x+3,y:ty-9,width:m.t.length*6.6+8,height:13,rx:3,fill:'#ffffff',opacity:.88},svg);
    E('text',{x:x+6,y:ty,'font-size':10.5,fill:m.c||C.red},svg).textContent=m.t;});
  const cross=E('line',{x1:0,x2:0,y1:M.t,y2:H-M.b,stroke:'#475569','stroke-width':1,opacity:0},svg);
  const tip=tipFor(host);
  svg.addEventListener('mousemove',ev=>{
    const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)/r.width*W;
    const xv=xmin+(px-M.l)/(W-M.l-M.r)*(xmax-xmin);
    let best=null,bd=1e18;
    ss[0].data.forEach(p=>{const d=Math.abs(p[0]-xv);if(d<bd){bd=d;best=p;}});
    if(!best)return;
    const cx=X(best[0]);cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.setAttribute('opacity',.6);
    let html='<b>'+(cfg.xlabel||'x')+' '+n2(best[0],0)+'</b>';
    ss.forEach(s=>{const p=s.data.reduce((a,b)=>Math.abs(b[0]-best[0])<Math.abs(a[0]-best[0])?b:a);
      html+='<br><span style="color:'+s.c+'">●</span> '+s.n+' '+(s.fmt?s.fmt(p[1]):n2(p[1],2));});
    tip.innerHTML=html;tip.style.opacity=1;
    const lx=Math.min(Math.max(ev.clientX-r.left+12,8),r.width-160);
    tip.style.left=lx+'px';tip.style.top=Math.max(ev.clientY-r.top-52,4)+'px';
  });
  svg.addEventListener('mouseleave',()=>{tip.style.opacity=0;cross.setAttribute('opacity',0);});
}

/* 柱状图 / 直方图 */
function barChart(id,cfg){
  const host=box(id),W=host.closest('.grid2')?620:1000,H=cfg.h||300,M={l:64,r:20,t:16,b:cfg.horizontal?30:48};
  const svg=E('svg',{viewBox:'0 0 '+W+' '+H},host), data=cfg.data, n=data.length;
  const vals=data.map(d=>d.v),mx=cfg.ymax||Math.max(...vals)*(cfg.horizontal?1.32:1.08),mn=cfg.ymin||0;
  if(cfg.horizontal){
    const rowH=(H-M.t-M.b)/n,labW=cfg.labW||160;
    ticks(mn,mx,4).forEach(v=>{if(v<mn-1e-9||v>mx+1e-9)return;const x=labW+(v-mn)/(mx-mn)*(W-labW-M.r);
      E('line',{x1:x,x2:x,y1:M.t,y2:H-M.b,stroke:C.grid},svg);
      E('text',{x:x,y:H-M.b+17,'text-anchor':'middle','font-size':11,fill:C.txt},svg).textContent=cfg.xfmt?cfg.xfmt(v):n2(v,0);});
    data.forEach((d,i)=>{const y=M.t+i*rowH+rowH*.16,w=(d.v-mn)/(mx-mn)*(W-labW-M.r);
      const rect=E('rect',{x:labW,y:y,width:Math.max(w,1),height:rowH*.68,rx:3,fill:d.c||C.blue,opacity:.88},svg);
      E('title',{},rect).textContent=d.k+': '+(cfg.vfmt?cfg.vfmt(d.v):n2(d.v,0));
      E('text',{x:labW-9,y:y+rowH*.52,'text-anchor':'end','font-size':11.5,fill:'#334155'},svg).textContent=d.k;
      const vt=cfg.vfmt?cfg.vfmt(d.v):n2(d.v,0);
      let tw=0;for(const ch of vt)tw+=(ch.charCodeAt(0)>127?11:6.2);
      let tx=labW+w+6,anch='start';
      if(tx+tw>W-4){tx=W-4;anch='end';}
      E('text',{x:tx,y:y+rowH*.52,'text-anchor':anch,'font-size':11,fill:C.txt},svg).textContent=vt;});
  }else{
    const bw=(W-M.l-M.r)/n;
    ticks(mn,mx,5).forEach(v=>{if(v<mn-1e-9||v>mx+1e-9)return;const y=H-M.b-(v-mn)/(mx-mn)*(H-M.t-M.b);
      E('line',{x1:M.l,x2:W-M.r,y1:y,y2:y,stroke:C.grid},svg);
      E('text',{x:M.l-9,y:y+4,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=cfg.yfmt?cfg.yfmt(v):n2(v,1);});
    data.forEach((d,i)=>{const h=(d.v-mn)/(mx-mn)*(H-M.t-M.b),x=M.l+i*bw+bw*.14;
      const rect=E('rect',{x:x,y:H-M.b-h,width:bw*.72,height:Math.max(h,1),rx:2,fill:d.c||C.blue,opacity:.88},svg);
      E('title',{},rect).textContent=d.k+': '+(cfg.vfmt?cfg.vfmt(d.v):n2(d.v,2));
      if(n<=16)E('text',{x:x+bw*.36,y:H-M.b+17,'text-anchor':'middle','font-size':10.5,fill:C.txt},svg).textContent=d.k;});
    if(cfg.xlabel)E('text',{x:W-M.r,y:H-M.b+34,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=cfg.xlabel;
  }
  return svg;
}
function histo(id,vals,bin,color,unit){
  const mn=Math.min(...vals),mx=Math.max(...vals),b0=Math.floor(mn/bin)*bin,n=Math.ceil((mx-b0)/bin)+1;
  const cs=new Array(n).fill(0);vals.forEach(v=>cs[Math.min(n-1,Math.floor((v-b0)/bin))]++);
  return barChart(id,{data:cs.map((c,i)=>({k:n2(b0+i*bin,bin<1?2:0),v:c,c:color})),h:280,
    yfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 次',xlabel:unit});
}

/* 热力图 */
function heatmap(id,rows,cols,vals,lo,hi,unit){
  const host=box(id),W=1000,H=300,M={l:52,r:74,t:20,b:40};
  const svg=E('svg',{viewBox:'0 0 '+W+' '+H},host);
  const cw=(W-M.l-M.r)/cols.length,ch=(H-M.t-M.b)/rows.length;
  const col=t=>{const x=Math.max(0,Math.min(1,(t-lo)/(hi-lo)));
    const stops=[[239,246,255],[191,219,254],[147,197,253],[253,224,71],[249,115,22],[220,38,38]];
    const p=x*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),q=p-i;
    const a=stops[i],b=stops[i+1];return 'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*q)).join(',')+')';};
  vals.forEach((row,i)=>row.forEach((v,j)=>{
    const r=E('rect',{x:M.l+j*cw,y:M.t+i*ch,width:cw+.5,height:ch+.5,fill:v===null?'#f1f5f9':col(v),stroke:'#fff','stroke-width':.5},svg);
    E('title',{},r).textContent=rows[i]+' / '+n2(cols[j],1)+' h: '+n2(v,1)+unit;}));
  rows.forEach((r,i)=>E('text',{x:M.l-8,y:M.t+i*ch+ch*.68,'text-anchor':'end','font-size':11,fill:C.txt},svg).textContent=r);
  cols.forEach((c,j)=>{if(j%6===0)E('text',{x:M.l+j*cw,y:H-M.b+17,'text-anchor':'middle','font-size':10.5,fill:C.txt},svg).textContent=n2(c,0)+'h';});
  for(let k=0;k<=5;k++){const v=lo+(hi-lo)*k/5,y=M.t+k*(H-M.t-M.b)/5;
    E('rect',{x:W-M.r+8,y:y-7,width:14,height:14,fill:col(v)},svg);
    E('text',{x:W-M.r+26,y:y+4,'font-size':10.5,fill:C.txt},svg).textContent=n2(v,0)+'°';}
}

/* ============ 图 1：loss ============ */
const tr=T.map(d=>[d.iter,d.loss]),va=V.map(d=>[d.iter,d.loss]);
lineChart('c_loss',{series:[{n:'训练 lm loss',c:C.blue,data:tr,fmt:v=>n2(v,6)},
  {n:'验证 lm loss',c:C.purple,data:va,dots:true,fmt:v=>n2(v,6)}],
  xmin:0,xmax:3700,ymin:.003,ymax:12,logY:true,yfmt:v=>v>=1?n2(v,0):n2(v,3),
  xlabel:'iteration',ylabel:'loss (log)',h:360,
  markers:[{x:395,t:'iter 380–400 梯度尖峰与验证反弹',c:C.orange}]});
legend('l_loss',[{n:'训练 lm loss（每 10 iter）',c:C.blue},{n:'验证 lm loss（每 50 iter）',c:C.purple}]);
lineChart('c_ppl',{series:[{n:'验证 PPL',c:C.cyan,data:V.map(d=>[d.iter,d.ppl]),fmt:v=>n2(v,4)}],
  xmin:0,xmax:3700,ymin:1,ymax:5000,logY:true,yfmt:v=>v>=10?n2(v,0):n2(v,1),xlabel:'iteration',ylabel:'PPL (log)',h:280});
lineChart('c_gnorm',{series:[{n:'grad norm',c:C.green,data:T.map(d=>[d.iter,d.gnorm]),fmt:v=>n2(v,3)}],
  xmin:0,xmax:3700,ymin:.01,ymax:500,logY:true,yfmt:v=>v>=1?n2(v,0):n2(v,2),xlabel:'iteration',ylabel:'norm (log)',h:280,
  markers:[{x:385,t:'iter 380/390 梯度尖峰',c:C.red}]});

box('t_mile').innerHTML=[[1,S['iter_loss_below_1.0']],[0.1,S['iter_loss_below_0.1']],[0.01,S['iter_loss_below_0.01']],[0.005,S['iter_loss_below_0.005']]]
 .map((r,i)=>'<tr><td>&lt; '+r[0]+'</td><td><b>'+n2(r[1],0)+'</b></td><td>约 '+n2(T.find(d=>d.iter===r[1]).t,2)+' h</td><td>'+
 ['从随机初始化快速进入有效学习区','数量级下降','进入低 loss 平台前段','进入最终 ~0.005 平台'][i]+'</td></tr>').join('');

/* ============ 图 2：性能 ============ */
lineChart('c_perf',{dual:true,series:[
  {n:'吞吐',c:C.blue,data:T.map(d=>[d.iter,d.tflops]),fmt:v=>n2(v,1)+' TFLOP/s'},
  {n:'迭代耗时',c:C.orange,data:T.map(d=>[d.iter,d.ms/1000]),axis:2,fmt:v=>n2(v,3)+' s'}],
  xmin:0,xmax:3700,ymin:420,ymax:540,y2min:22,y2max:28,xlabel:'iteration',
  ylabel:'TFLOP/s/GPU',y2label:'s / iter',h:340});
legend('l_perf',[{n:'每卡吞吐（左轴）',c:C.blue},{n:'迭代耗时（右轴）',c:C.orange}]);
const stable=T.filter(d=>d.iter>=500);
histo('c_hist_tf',stable.map(d=>d.tflops),2,C.blue,'TFLOP/s/GPU');
histo('c_hist_ms',stable.map(d=>d.ms/1000),.1,C.orange,'秒 / iteration');
const stRows=[
 ['预热期吞吐 (TFLOP/s/GPU)',S.throughput_warm],
 ['稳定期吞吐 (TFLOP/s/GPU)',S.throughput_stable],
 ['稳定期迭代耗时 (s/iter)',{...S.iter_ms_stable,mean:S.iter_ms_stable.mean/1000,sd:S.iter_ms_stable.sd/1000,min:S.iter_ms_stable.min/1000,p50:S.iter_ms_stable.p50/1000,p95:S.iter_ms_stable.p95/1000,max:S.iter_ms_stable.max/1000}],
 ['验证单次耗时 (s)',{...S.eval_ms,mean:S.eval_ms.mean/1000,sd:S.eval_ms.sd/1000,min:S.eval_ms.min/1000,p50:S.eval_ms.p50/1000,p95:S.eval_ms.p95/1000,max:S.eval_ms.max/1000}],
 ['梯度范数',S.gnorm]];
box('t_stat').innerHTML=stRows.map(r=>'<tr><td><b>'+r[0]+'</b></td><td>'+r[1].n+'</td><td>'+n2(r[1].mean,3)+'</td><td>'+n2(r[1].sd,3)+
 '</td><td>'+n2(r[1].cv,2)+'%</td><td>'+[r[1].min,r[1].p50,r[1].p95,r[1].max].map(v=>n2(v,3)).join(' / ')+'</td></tr>').join('');

/* ============ 图 3：GPU 时序 ============ */
const B=G.buckets, bt=B.map(b=>b.t);
function gpuSeries(metric){return B.map(b=>[b.t,b[metric]]);}
lineChart('c_gpu_time',{dual:true,series:[
  {n:'平均功耗',c:C.blue,data:gpuSeries('pwr'),fmt:v=>n2(v,1)+' W'},
  {n:'平均温度',c:C.red,data:gpuSeries('gtemp'),axis:2,fmt:v=>n2(v,1)+' °C'}],
  xmin:0,xmax:23.5,ymin:560,ymax:720,y2min:45,y2max:70,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',
  ylabel:'功耗 W',y2label:'温度 °C',h:300});
legend('l_gpu_time',[{n:'功耗（左轴）',c:C.blue},{n:'温度（右轴）',c:C.red}]);
lineChart('c_clk',{dual:true,series:[
  {n:'核心频率',c:C.green,data:gpuSeries('pclk'),fmt:v=>n2(v,0)+' MHz'},
  {n:'功率封顶占比',c:C.orange,data:gpuSeries('pviol'),axis:2,fmt:v=>n2(v,1)+'%'}],
  xmin:0,xmax:23.5,ymin:1830,ymax:1980,y2min:0,y2max:105,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',
  ylabel:'核心频率 MHz',y2label:'pviol %',h:300});
legend('l_clk',[{n:'核心频率（左轴）',c:C.green},{n:'pviol（右轴）',c:C.orange}]);
lineChart('c_startup',{dual:true,series:[
  {n:'显存占用',c:C.purple,data:G.startup.map(b=>[b.m,b.fb/1024]),fmt:v=>n2(v,1)+' GiB'},
  {n:'功耗',c:C.orange,data:G.startup.map(b=>[b.m,b.pwr]),axis:2,fmt:v=>n2(v,0)+' W'}],
  xmin:0,xmax:30,ymin:0,ymax:120,y2min:0,y2max:760,xfmt:v=>n2(v,0)+'m',xlabel:'启动后分钟',
  ylabel:'显存 GiB',y2label:'功耗 W',h:280});
legend('l_startup',[{n:'显存占用（左轴）',c:C.purple},{n:'功耗（右轴）',c:C.orange}]);
const pd=S.pviol_dist;
const v1=Object.values(pd).reduce((a,b)=>a+b,0);
barChart('c_power',{data:[{k:'0%',v:pd['0'],c:C.green},{k:'1–49%',v:pd['1-49'],c:C.cyan},
 {k:'50–89%',v:pd['50-89'],c:C.blue},{k:'90–99%',v:pd['90-99'],c:C.orange},
 {k:'100%',v:pd['100'],c:C.red}],h:280,yfmt:v=>n2(v,0),vfmt:v=>n2(v/v1*100,1)+'% ('+n2(v,0)+')',xlabel:'pviol 档位'});

/* 8 卡对比小图 */
const metrics=[['pwr','平均功耗 (W)',C.blue,660,690],['gtemp','平均温度 (°C)',C.red,50,66],
 ['pclk','平均核心频率 (MHz)',C.green,1840,1950],['fb','平均显存 (MB)',C.purple,110600,110800]];
box('gpu_bars').innerHTML='';
metrics.forEach((m,i)=>{
  const div=document.createElement('div');div.className='chart';div.id='gb'+i;
  div.innerHTML='<div class="ct">'+m[1]+'</div><div class="cs">偶数卡深蓝 / 奇数卡浅青</div>';
  box('gpu_bars').appendChild(div);
  const arr=[];for(let g=0;g<8;g++)arr.push({k:'GPU'+g,v:G.summary[g][m[0]].mean,c:g%2===0?C.blue:C.cyan});
  barChart('gb'+i,{data:arr,h:230,ymin:m[3],ymax:m[4],yfmt:v=>n2(v,m[0]==='fb'?0:0),vfmt:v=>n2(v,1)});
});
const heatVals=[];for(let g=0;g<8;g++)heatVals.push(B.map(b=>b.per_gpu[g].gtemp));
heatmap('c_heat',Array.from({length:8},(_,g)=>'GPU'+g),bt,heatVals,50,67,' °C');

/* ============ 图 4：日志 ============ */
const cats=Object.entries(L.categories);
barChart('c_events',{horizontal:true,data:cats.map(([k,v])=>({k:k,v:v,c:C.blue})),h:300,labW:140,
  xfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 行 ('+n2(v/D.meta.log_lines*100,1)+'%)'});
barChart('c_abort',{data:Object.entries(L.aborts).map(([k,v])=>({k:'nranks '+k,v:v,c:k==='16'?C.purple:(k==='8'?C.blue:C.cyan)})),
  h:300,yfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 次',xlabel:'通信域规模'});
const ev=L.events;
const evRows=[
 ['致命错误','Traceback',ev.traceback,'0 = 无崩溃'],
 ['致命错误','OOM',ev.oom,'0 = 无显存溢出'],
 ['致命错误','CUDA error',ev.cuda_error,'0 = 无设备错误'],
 ['致命错误','NCCL 真超时',ev.nccl_timeout,'0 = 无集合通信超时'],
 ['配置','NCCL_IB_TIMEOUT 环境变量',ev.nccl_ib_timeout_env,'设置为 22，非超时事件'],
 ['退出清理','Abort COMPLETE',ev.abort_complete,'进程组析构，正常退出'],
 ['退出清理','process group 未销毁警告',ev.pg_not_destroyed,'可优化项'],
 ['噪声','NCCL 插件符号缺失',ev.plugin_symbol_missing,'回退内置 tuner / IBext_v8'],
 ['噪声','Python 弃用警告',ev.deprecation,'protobuf / importlib'],
 ['噪声','TE fused RoPE 警告',ev.rope_warning,'transpose_output_memory 被忽略'],
 ['噪声','socket hostname 警告',ev.socket_hostname_warn,'c10d 无法解析客户端主机名'],
];
box('t_events').innerHTML=evRows.map(r=>'<tr><td>'+r[0]+'</td><td><code>'+r[1]+'</code></td><td><b>'+r[2]+'</b></td><td>'+r[3]+'</td></tr>').join('');
const sc=D.selfcheck,scEnt=Object.entries(sc),rows=[];
for(let i=0;i<scEnt.length;i+=2)rows.push('<tr><td><b>'+scEnt[i][0]+'</b></td><td>'+scEnt[i][1]+'</td><td><b>'+(scEnt[i+1]?scEnt[i+1][0]:'')+'</b></td><td>'+(scEnt[i+1]?scEnt[i+1][1]:'')+'</td></tr>');
box('t_self').innerHTML=rows.join('');
const MFU_PEAK=989.5;
lineChart('c_mfu',{series:[{n:'MFU',c:C.green,data:T.map(d=>[d.iter,d.tflops/MFU_PEAK*100]),fmt:v=>n2(v,2)+'%'}],
  xmin:0,xmax:3700,ymin:42,ymax:56,xlabel:'iteration',ylabel:'MFU %',h:300,
  markers:[{x:10,t:'warmup 43.7%',c:C.orange}]});
legend('l_mfu',[{n:'MFU（每 10 iter 记录点）',c:C.green}]);
histo('c_mfu_hist',stable.map(d=>d.tflops/MFU_PEAK*100),.1,C.green,'MFU %');
__DUAL_JS__
</script>
</body></html>"""

M = DUAL["mfu"]
mfu_kpis = [
    ("稳定期 MFU", f(M["stable_mean"]), "%", f"峰值按 H200 SXM BF16 dense {M['peak_tflops']} TFLOP/s", "green"),
    ("MFU 波动区间", f"{f(M['stable_min'])}–{f(M['stable_max'])}", "%", f"P05 {f(M['stable_p05'])}% · P95 {f(M['stable_p95'])}%", "blue"),
    ("16 卡有效算力", f(M["global_pflops"]), "PFLOP/s", f"全程平均 MFU {f(M['all_mean'])}%", "cyan"),
]
mfu_kpi_html = "\n".join(
    f'<div class="kpi k-{c}"><div class="kpi-t">{t}</div><div class="kpi-v">{v}<span class="kpi-u">{u}</span></div><div class="kpi-s">{s}</div></div>'
    for t, v, u, s, c in mfu_kpis)
mfu_sens_rows = "\n".join(
    f"<tr><td><b>{x['basis']}</b></td><td>{f(x['peak'],1)}</td><td>{f(x['tflops'],1)}</td><td><b>{f(x['mfu'])}%</b></td></tr>"
    for x in M["sensitivity"])
mfu_hw_rows = "\n".join(
    f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in M["hardware_infer"])
mfu_html = f"""
  <h3>MFU（Model FLOPs Utilization，模型算力利用率）</h3>
  <p class="lead">MFU = 实测模型吞吐 ÷ 硬件 BF16 dense 峰值吞吐。日志与 dmon 均不记录 GPU 型号，
    按功耗、显存与时钟反推为 H200 SXM 级，采用 {M['peak_tflops']} TFLOP/s 作为主口径。</p>
  <div class="kpis" style="grid-template-columns:repeat(3,1fr)">{mfu_kpi_html}</div>
  <div class="grid2">
    <div class="chart" id="c_mfu"><div class="ct">MFU 随迭代变化</div><div class="cs">由日志每卡吞吐换算 · 首个点为 warmup</div><div class="legend" id="l_mfu"></div></div>
    <div class="chart" id="c_mfu_hist"><div class="ct">稳定期 MFU 分布</div><div class="cs">iteration ≥ 500 的 319 个记录点</div></div>
  </div>
  <h3>口径与敏感性</h3>
  <table><thead><tr><th style="width:44%">口径</th><th>硬件峰值 TFLOP/s</th><th>模型吞吐 TFLOP/s/GPU</th><th>MFU</th></tr></thead>
  <tbody>{mfu_sens_rows}</tbody></table>
  <h3>硬件型号推断依据</h3>
  <table><thead><tr><th style="width:24%">观测项</th><th style="width:32%">观测值</th><th>判读</th></tr></thead>
  <tbody>{mfu_hw_rows}</tbody></table>
  <div class="note"><b>口径说明：</b>分子采用 Megatron 日志的 <code>throughput per GPU (TFLOP/s/GPU)</code>，
    它是框架估算的模型 FLOPs（含 attention 等），而非 6ND 理论下限。
    用 6 × N × D = 6 × 56.51B × {M['tokens_per_iter']:,} tokens/iter = {f(M['flops_6nd_per_iter_pflops'])} PFLOPs/iter 校验，
    除以稳定期 {f(S['iter_ms_stable']['mean']/1000,3)} s 与 16 卡，得 {f(M['flops_6nd_per_gpu_tflops'],1)} TFLOP/s/GPU（MFU {f(M['mfu_6nd'])}%）；
    Megatron 口径高 {f(M['gap_vs_6nd_pct'])}%，差额来自 attention / activation 等额外 FLOPs，属合理范围。
    {M['peak_note']}。若实际硬件为 B200，请按敏感性表取 {f([x['mfu'] for x in M['sensitivity'] if x['peak']==2250][0])}% 这一行。</div>
"""

DIGEST = (ROOT / "image-digest.txt").read_text(encoding="utf-8").strip()
subtitle_single = (f'{D["meta"]["case"]} ｜ 基于训练日志 + nvidia-smi dmon '
                   f'全量采样（{S["gpu_rows"]:,} 个 GPU 采样点）的量化复盘')
head_single = (f'<span>窗口 {S["ts_first"]} ~ {S["ts_last"]}</span>'
               f'<span>时长 {f(S["hours"], 2)} h</span><span>迭代 3,680 / 100,000</span>'
               f'<span>8 × GPU，全局 16 ranks（TP8 × DP2）</span>'
               f'<span>报告生成 {GEN}</span>')
footer_single = f'node1-e2e-56b · 单节点统计分析 · 生成于 {GEN} · 数据源：训练日志 / dmon CSV / selfcheck / args-help'
footer_dual = (f'E2E 56B 双机联合统计分析 · node0 + node1 · 生成于 {GEN} · '
               f'数据源：双机训练日志 / 两份 dmon CSV / selfcheck / args-help / 21 次前置基准')
kpi_note_dual = ('<div class="note" style="margin:0 0 20px">顶部 KPI 中，迭代 / loss / 吞吐 / 步时是全局 16 ranks 的训练指标（由 node1 输出）；'
                 'GPU 指标为 node1 本地 8 卡采样口径。两节点 16 卡汇总与差异见 A / B 节。</div>')
s4_lead_single = ('nvidia-smi dmon 每 ~5.5 s 采样一次，8 卡 × 15,410 个时间点 = 123,280 条有效记录；'
                  '下图按 30 分钟分桶聚合。')
s4_lead_dual = ('以下为 <b>node1 本地 8 卡</b> 的 dmon 视图：每 ~5.5 s 采样、15,410 个时间点 = 123,280 条有效记录，'
                '按 30 分钟分桶聚合；两节点 16 卡对比见 B 节。')
extra_files_dual = "".join([
    f'<tr><td><code>node0/logs/…node0-20261112_000625.out</code></td><td>node0 主日志</td>'
    f'<td>{DUAL["n0"]["log_lines"]:,} 行 / 419 KB</td><td>523 项超参表、5 个里程碑、8 个 rank 显存明细、NCCL 跨节点拓扑、定时退出记录</td></tr>',
    f'<tr><td><code>node0/logs/…node0-dmon-…000625.csv</code></td><td>node0 GPU 监控</td>'
    f'<td>20.0 MB / {DUAL["gpu0"]["rows"]:,} 有效行</td><td>8 卡 × {DUAL["gpu0"]["points"]:,} 个时间点，与 node1 dmon 时间轴对齐</td></tr>',
    '<tr><td><code>node0/selfcheck.log · args-help.txt · image-digest.txt</code></td><td>node0 自检</td>'
    '<td>3 个文件</td><td>与 node1 完全一致（同镜像 SHA、同组件版本、512 条参数行）</td></tr>',
    f'<tr><td><code>node0/logs/single-*.out · dmon-*.csv</code></td><td>前置基准运行</td>'
    f'<td>{DUAL["hist_summary"]["runs"]} 次 short-run + 对应 dmon</td><td>8B / 56B / MFU 调参轨迹，见 C 节</td></tr>',
])

common = {
    "__KPIS__": kpi_html, "__CHECKS__": check_html, "__DIAG__": diag_html,
    "__FLAGS__": flags_html, "__FILES__": file_html, "__MFU_BLOCK__": mfu_html,
    "__TS1__": S["ts_first"], "__TS2__": S["ts_last"],
    "__HOURS__": f(S["hours"], 2), "__PROG__": f(S["progress"], 2),
    "__TF__": f(S["throughput_stable"]["mean"], 1),
    "__CV__": f(S["throughput_stable"]["cv"], 2),
    "__EVAL__": f(S["eval_ms"]["mean"] / 1000, 2),
    "__PCAP__": f(S["power_cap_pct"], 1),
    "__DIGEST__": DIGEST, "__GEN__": GEN, "__CASE__": D["meta"]["case"],
}

for mode, out_name, payload in (
    ("dual", "双机联合统计分析报告.html", DUAL),
    ("single", "node1-e2e-56b-统计分析报告.html", D),
):
    dual = mode == "dual"
    repl = dict(common)
    repl.update({
        "__DATA__": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "__TITLE__": DC["title_dual"] if dual else DC["title_single"],
        "__SUBTITLE__": DC["subtitle_dual"] if dual else subtitle_single,
        "__HEAD_META__": DC["head_meta_dual"] if dual else head_single,
        "__NAV_DUAL__": DC["nav_dual"] if dual else "",
        "__SECTION_A__": DC["section_a"] if dual else "",
        "__SECTION_B__": DC["section_b"] if dual else "",
        "__SECTION_C__": DC["section_c"] if dual else "",
        "__DUAL_JS__": DC["dual_js"] if dual else "",
        "__SCOPE_NOTE__": DC["scope_dual"] if dual else DC["scope_single"],
        "__OBS_ITEM__": DC["obs_dual"] if dual else DC["obs_single"],
        "__FOOTER__": footer_dual if dual else footer_single,
        "__KPI_NOTE__": kpi_note_dual if dual else "",
        "__S4_LEAD__": s4_lead_dual if dual else s4_lead_single,
        "__EXTRA_FILES__": extra_files_dual if dual else "",
    })
    html = HTML
    for k, v in repl.items():
        html = html.replace(k, str(v))
    left = [k for k in repl if k in html]
    target = OUT / out_name
    target.write_text(html, encoding="utf-8")
    print(f"[{mode}] {target.name}: {target.stat().st_size / 1024:.1f} KB | 残留占位符 {left}")
