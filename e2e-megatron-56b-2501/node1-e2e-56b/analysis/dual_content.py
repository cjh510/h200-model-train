# -*- coding: utf-8 -*-
"""生成双机联合报告的新增章节 HTML / JS 片段（供 report.py 引用）"""


def nf(v, n=2):
    try:
        return f"{float(v):,.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def build(DUAL):
    n0, g0, g1 = DUAL["n0"], DUAL["gpu0"], DUAL["gpu1"]
    cm, hs, hist, n1 = DUAL["dual"], DUAL["hist_summary"], DUAL["history"], DUAL["n1"]
    mfu = DUAL["mfu"]
    pv0v = [b["pviol"] for b in g0["buckets"] if b.get("pviol") is not None]
    pv1v = [b["pviol"] for b in g1["buckets"] if b.get("pviol") is not None]
    pv0_avg = sum(pv0v) / len(pv0v) if pv0v else 0
    pv1_avg = sum(pv1v) / len(pv1v) if pv1v else 0
    cfg, par, mem, ev, topo = n0["config"], n0["params"], n0["mem"], n0["events"], n0["topo"]
    S, ev1 = n1["stats"], n1["log"]["events"]
    c0, c1, diff = cm["cmp"]["node0"], cm["cmp"]["node1"], cm["diff"]

    # ---------- 头部 ----------
    title = "E2E 56B 双机联合统计分析报告"
    subtitle = (f"node0 + node1 ｜ 56.51B 参数 · TP8 × PP1 × DP2 / 16 ranks · "
                f"两份 dmon 合计 {g0['rows'] + g1['rows']:,} 个 GPU 采样点的量化复盘")
    head_meta = (
        f'<span>窗口 {g0["ts_first"]} ~ {g1["ts_last"]}</span>'
        f'<span>时长 {nf(S["hours"])} h（配置 {cfg.get("exit_duration_in_mins")} min）</span>'
        f'<span>迭代 {S["iter_last"]:,} / {S["iter_target"]:,}</span>'
        f'<span>2 节点 × 8 GPU = 16 ranks</span>'
        f'<span>报告生成 {DUAL["meta"]["generated"]}</span>')
    nav_dual = ('<a href="#sA">双机总览</a><a href="#sB">双机 GPU</a>'
                '<a href="#sC">前置基准</a>')

    # ---------- A 双机总览 ----------
    ak = [
        ("模型总参数", nf(par.get("total_B"), 2), "B", f"transformer {nf(par.get('transformer_B'),2)} B + embedding {nf(par.get('embedding_B'),2)} B", "purple"),
        ("并行拓扑", "8×1×2", "", "TP8 / PP1 / DP2 · world_size 16", "blue"),
        ("每卡参数分片", nf(par.get("shard_B"), 4), "B", "8 个 TP rank 完全一致（7,064,510,464）", "blue"),
        ("梯度累积步数", str(cm["grad_accum"]), "步", "GBS 64 / (MBS 1 × DP 2)", "cyan"),
        ("双机采样点", f"{(g0['rows'] + g1['rows']) / 1000:,.1f}", "K", f"node0 {g0['rows']:,} + node1 {g1['rows']:,}", "cyan"),
        ("16 卡均功耗", nf(cm["all16_pwr"], 1), "W", f"温度范围 {cm['all16_temp_range'][0]}–{cm['all16_temp_range'][1]} °C", "orange"),
        ("稳定期 MFU", nf(mfu["stable_mean"]), "%", f"H200 BF16 dense 峰值 {mfu['peak_tflops']} TFLOP/s", "green"),
        ("16 卡有效算力", nf(mfu["global_pflops"], 2), "PFLOP/s", "16 ranks × 每卡 524.97 TFLOP/s", "green"),
        ("每迭代 tokens", f'{mfu["tokens_per_iter"]:,}', "", "GBS 64 × seq 8192", "cyan"),
    ]
    akpi = "\n".join(
        f'<div class="kpi k-{c}"><div class="kpi-t">{t}</div><div class="kpi-v">{v}<span class="kpi-u">{u}</span></div><div class="kpi-s">{s}</div></div>'
        for t, v, u, s, c in ak)

    consistency = [
        ("容器镜像 SHA", n0["digest"][:22] + "…", DUAL["meta"]["digest1"][:22] + "…", "完全一致", "ok"),
        ("selfcheck 版本", f"GPUS 8 / CUDA {n0['selfcheck'].get('CUDA')}", f"GPUS 8 / CUDA {n1['selfcheck'].get('CUDA')}", "完全一致", "ok"),
        ("日志角色", "超参表 / 里程碑 / 每 rank 显存 / 跨节点拓扑", "训练迭代 / 验证 / 吞吐 / 步时", "互补，合起来是完整视图", "info"),
        ("torchrun 启动", "16:06:32 UTC", "16:06:37 UTC", "相差 5 s，同步拉起", "ok"),
        ("初始化完成", "16:07:04（34 s）", "—", "模型/优化器 16:07:05、数据集 16:07:06", "info"),
        ("训练起点", "16:07:06 before training step", "16:11:42 首条 iteration 日志", "相差 4.6 min ≈ 10 个 iter 的日志间隔", "info"),
        ("定时退出", "15:37:04，exiting after 1410.2568 min", "15:36:41 最后一条 iteration", "相差 23 s，同批退出", "ok"),
        ("NCCL Abort COMPLETE", str(ev["abort_complete"]), str(ev1["abort_complete"]), "56 / 56 完全对称", "ok"),
        ("process group 未销毁警告", str(ev["pg_not_destroyed"]), str(ev1["pg_not_destroyed"]), "8 / 8 完全对称", "info"),
        ("NCCL 插件符号缺失", str(ev["plugin_symbol_missing"]), str(ev1["plugin_symbol_missing"]), "40 / 40，回退内置 tuner", "info"),
        ("IB / GDRDMA 通道", "224 / 224", "224 / 224", "跨节点 Ring 通道完全对称", "ok"),
        ("机内 P2P 通道", "1,280", "1,280", "TP 组内 NVLink/CUMEM 对称", "ok"),
        ("致命错误", "0 Traceback / OOM / CUDA / 超时", "0 Traceback / OOM / CUDA / 超时", "双机均无故障", "ok"),
    ]
    cons_html = "\n".join(
        f'<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td><span class="tag tag-{lv}">{d}</span></td></tr>'
        for a, b, c, d, lv in consistency)

    cfg_rows = [
        ("模型结构", f'{cfg.get("num_layers")} 层 · hidden {cfg.get("hidden_size")} · FFN {cfg.get("ffn_hidden_size")} · {cfg.get("num_attention_heads")} heads / {cfg.get("num_query_groups")} query groups'),
        ("词表与序列", f'vocab {cfg.get("vocab_size")} · seq {cfg.get("seq_length")} · max_pos {cfg.get("max_position_embeddings")}'),
        ("并行与批次", f'TP {cfg.get("tensor_model_parallel_size")} · PP {cfg.get("pipeline_model_parallel_size")} · DP {cfg.get("data_parallel_size")} · world {cfg.get("world_size")} · MBS {cfg.get("micro_batch_size")} · GBS {cfg.get("global_batch_size")} · 梯度累积 {cm["grad_accum"]}'),
        ("训练目标", f'train_iters {cfg.get("train_iters")} · exit_duration_in_mins {cfg.get("exit_duration_in_mins")} · eval_interval {cfg.get("eval_interval")} / eval_iters {cfg.get("eval_iters")} · log_interval {cfg.get("log_interval")}'),
        ("优化器", f'{cfg.get("optimizer")} · lr {cfg.get("lr")} · warmup {cfg.get("lr_warmup_iters")} iters · cosine decay {cfg.get("lr_decay_iters")} · clip_grad {cfg.get("clip_grad")} · β {cfg.get("adam_beta1")}/{cfg.get("adam_beta2")} · wd {cfg.get("weight_decay")}'),
        ("精度与数据", f'bf16 {cfg.get("bf16")} · fp8 {cfg.get("fp8")} · params_dtype {cfg.get("params_dtype")} · mock_data {cfg.get("mock_data")}（合成数据，非真实语料）'),
        ("并行优化", f'sequence_parallel {cfg.get("sequence_parallel")} · tp_comm_overlap {cfg.get("tp_comm_overlap")} · overlap_grad_reduce {cfg.get("overlap_grad_reduce")} · overlap_param_gather {cfg.get("overlap_param_gather")} · distributed_optimizer {cfg.get("use_distributed_optimizer")}'),
        ("显存策略", f'recompute_granularity {cfg.get("recompute_granularity")}（无重计算）· save {cfg.get("save")}（不落 checkpoint）· transformer_impl {cfg.get("transformer_impl")}'),
        ("通信后端", f'{cfg.get("distributed_backend")} · attention_backend {cfg.get("attention_backend")} · inprocess_restart {cfg.get("inprocess_restart", "False")}'),
    ]
    cfg_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in cfg_rows)

    ms = n0["milestones"]
    tl = [
        ("16:06:27 / 16:06:28", "dmon 开始采样", "node0 / node1 本地 00:06 UTC+8", "两机监控同时启动"),
        ("16:06:32 / 16:06:37", "torchrun 拉起 16 ranks", "node0 / node1", "相差 5 s"),
        (ms[0]["ts"] if ms else "16:07:04", "Megatron 初始化完成", "node0 里程碑", "启动后 34 s"),
        (ms[1]["ts"] if len(ms) > 1 else "16:07:05", "模型 / 优化器 / LR scheduler 构建完成", "node0 里程碑", "1 s"),
        (ms[2]["ts"] if len(ms) > 2 else "16:07:06", "训练 / 验证 / 测试数据集构建完成", "node0 里程碑", "1 s；validation 128,064 samples"),
        (ms[3]["ts"] if len(ms) > 3 else "16:07:06", "进入第一个 training step", "node0 里程碑", "首个 iter 吞吐 432.6 TFLOP/s"),
        ("16:11:42", "首条 iteration 日志（iter 10）", "node1 输出", "启动后 5.2 min"),
        ("15:36:41", "最后一条 iteration 日志（iter 3,680）", "node1 输出", "完成 3.68% 目标"),
        (ms[-1]["ts"] if ms else "15:37:04", "定时退出：exiting after 1410.2568 min", "node0 里程碑", f'与配置 {cfg.get("exit_duration_in_mins")} min 一致'),
        ("15:37:06 ~ 15:37:12", "进程组析构 / 显存释放", "双机 dmon + 日志", "node0 15:37:12 的 fb 已降至 1,286 MB"),
    ]
    tl_html = "\n".join(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in tl)

    m0 = mem[0] if mem else {}
    mem_rows = [
        ("PyTorch allocated（rank 0，iter 10 后）", f'{nf(m0.get("allocated"), 1)} MB', "8 个 rank 完全相同"),
        ("PyTorch max allocated", f'{nf(m0.get("max_allocated"), 1)} MB', "峰值张量占用"),
        ("PyTorch reserved / max reserved", f'{nf(m0.get("max_reserved"), 1)} MB', "分配器缓存上限"),
        ("node0 dmon fb（稳定期）", f'{nf(g0["summary"]["0"]["fb"]["p50"], 0)} MB', "≈105.5 GiB"),
        ("node1 dmon fb（稳定期）", f'{nf(g1["summary"]["0"]["fb"]["p50"], 0)} MB', "≈108.2 GiB"),
        ("两节点显存差", f'{nf(abs(diff["fb"]), 0)} MB（{nf(abs(diff["fb"]) / 1024, 2)} GiB）', "全程恒定，p50 = p95 = max"),
        ("非 PyTorch 占用（node0 / node1）", f'{nf(cm["reserved_vs_fb_MB"], 0)} / {nf(g1["summary"]["0"]["fb"]["p50"] - m0.get("max_reserved", 0), 0)} MB', "NCCL / CUDA context / 驱动缓冲；差异几乎全部来自这一层"),
    ]
    mem_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in mem_rows)

    topo_rows = [
        ("跨节点网络", ", ".join(topo["ib_ext"]) + " / GDRDMA", "NCCL 使用 IB RDMA 直通 GPU 显存"),
        ("跨节点 IB 通道", f'{topo["ib_channels"]} 条（node0）/ 224 条（node1）', "两机完全对称"),
        ("机内 P2P 通道", f'{topo["p2p_channels"]:,} 条 / 1,280 条', "TP 组内 NVLink / CUMEM"),
        ("跨节点 rank 对", f'{topo["cross_node_pairs"]} 组', "16 ranks Ring 拓扑中相邻节点互联"),
        ("三类通信域", "nranks 2（DP）/ 8（TP）/ 16（全局）", "Abort COMPLETE 分别 16 / 16 / 24 次，node0 与 node1 一致"),
        ("PCIe 吞吐均值", f'node0 rx/tx {nf(c0["rxpci"],0)}/{nf(c0["txpci"],0)} MB/s · node1 {nf(c1["rxpci"],0)}/{nf(c1["txpci"],0)} MB/s', "跨节点流量经 PCIe 到 NIC，node0 略高 3.5%"),
    ]
    topo_html = "\n".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in topo_rows)

    section_a = f"""
<section id="sA">
  <h2><span class="no">A</span>双机联合总览</h2>
  <p class="lead">node0 是主进程配置 / 里程碑日志（本地 ranks 0–7），node1 输出训练指标（ranks 8–15）；两份日志互补，共同构成 16 ranks 的完整视图。</p>
  <div class="kpis" style="grid-template-columns:repeat(3,1fr)">{akpi}</div>
  <div class="chart" id="c_dual_rel"><div class="ct">node0 相对 node1 的关键指标（node1 = 100%）</div><div class="cs">5 项指标归一化对比 · 显存差与功率墙差异最显著</div><div class="legend" id="l_dual_rel"></div></div>
  <h3>双机一致性证据</h3>
  <table><thead><tr><th style="width:22%">项目</th><th style="width:27%">node0</th><th style="width:27%">node1</th><th>判定</th></tr></thead><tbody>{cons_html}</tbody></table>
  <h3>模型与并行配置（node0 超参表，共 {len(cfg)} 项关键参数）</h3>
  <table><thead><tr><th style="width:20%">类别</th><th>配置</th></tr></thead><tbody>{cfg_html}</tbody></table>
  <h3>启动 → 退出时间线（UTC）</h3>
  <table><thead><tr><th style="width:22%">时刻</th><th style="width:32%">事件</th><th style="width:20%">来源</th><th>说明</th></tr></thead><tbody>{tl_html}</tbody></table>
  <div class="note"><b>退出原因已被 node0 日志直接证实：</b><code>[exiting program after 1410.256824962298 minutes]</code>，
    与参数 <code>exit_duration_in_mins=1410</code> 精确对应；两节点的 56 条 NCCL Abort COMPLETE 均为定时退出后的进程组析构，不是故障。</div>
  <h3>显存：模型状态 vs GPU 实测</h3>
  <table><thead><tr><th style="width:38%">口径</th><th style="width:24%">数值</th><th>说明</th></tr></thead><tbody>{mem_html}</tbody></table>
  <h3>跨节点通信拓扑</h3>
  <table><thead><tr><th style="width:22%">项目</th><th style="width:38%">观测</th><th>解读</th></tr></thead><tbody>{topo_html}</tbody></table>
</section>
"""

    # ---------- B 双机 GPU ----------
    diff_rows = [
        ("平均功耗", f'{nf(c0["pwr"],2)} W', f'{nf(c1["pwr"],2)} W', f'{diff["pwr"]:+.2f} W', "node0 略高，均在 ~700 W 功率墙附近"),
        ("平均核心温度", f'{nf(c0["gtemp"],2)} °C', f'{nf(c1["gtemp"],2)} °C', f'{diff["gtemp"]:+.2f} °C', "双机散热水平一致"),
        ("平均显存温度", f'{nf(c0["mtemp"],2)} °C', f'{nf(c1["mtemp"],2)} °C', f'{diff["mtemp"]:+.2f} °C', "基本一致"),
        ("SM 利用率", f'{nf(c0["sm"],2)} %', f'{nf(c1["sm"],2)} %', f'{diff["sm"]:+.2f} %', "两机均近乎满载"),
        ("显存占用（均值）", f'{nf(c0["fb"],0)} MB', f'{nf(c1["fb"],0)} MB', f'{diff["fb"]:+,.0f} MB', "稳定期恒定差，见下方诊断"),
        ("核心频率", f'{nf(c0["pclk"],1)} MHz', f'{nf(c1["pclk"],1)} MHz', f'{diff["pclk"]:+.1f} MHz', "功率墙下自适应"),
        ("pviol=100 占比", f'{nf(c0["power_cap_pct"],2)} %', f'{nf(c1["power_cap_pct"],2)} %', f'{diff["power_cap_pct"]:+.2f} pp', "node0 功率封顶更频繁"),
        ("tviol 热降频", "0", "0", "0", "两机均无热降频"),
        ("PCIe rx / tx", f'{nf(c0["rxpci"],0)} / {nf(c0["txpci"],0)} MB/s', f'{nf(c1["rxpci"],0)} / {nf(c1["txpci"],0)} MB/s', f'{diff["rxpci"]:+.0f} / {diff["txpci"]:+.0f} MB/s', "node0 跨节点流量略高"),
        ("采样点数", f'{g0["points"]:,}', f'{g1["points"]:,}', f'{cm["sample_gap_points"]:+,}', "间隔 5.54 s vs 5.49 s，无 >20 s 空洞"),
    ]
    diff_html = "\n".join(
        f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in diff_rows)

    section_b = f"""
<section id="sB">
  <h2><span class="no">B</span>双机 GPU 对比（16 卡）</h2>
  <p class="lead">node0 {g0['rows']:,} 条 + node1 {g1['rows']:,} 条有效采样，按 30 分钟分桶叠加；两机监控时间轴已统一到 UTC。</p>
  <div class="grid2">
    <div class="chart" id="c_dual_fb"><div class="ct">显存占用时序</div><div class="cs">两条平台相差固定 2,640 MB</div><div class="legend" id="l_dual_fb"></div></div>
    <div class="chart" id="c_dual_pwr"><div class="ct">平均功耗时序</div><div class="cs">node0 略高约 3.5 W</div><div class="legend" id="l_dual_pwr"></div></div>
  </div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_dual_temp"><div class="ct">平均温度时序</div><div class="cs">两机均温几乎重合</div><div class="legend" id="l_dual_temp"></div></div>
    <div class="chart" id="c_dual_pci"><div class="ct">PCIe 双向吞吐时序</div><div class="cs">rx + tx，MB/s</div><div class="legend" id="l_dual_pci"></div></div>
  </div>
  <div class="chart" id="c_dual_clkpv" style="margin-top:18px">
    <div class="ct">核心频率与功率墙 pviol 时序（双机叠加）</div>
    <div class="cs">左轴：核心频率 MHz（实线）· 右轴：pviol 功率封顶占比 %（虚线）· 30 分钟分桶</div>
    <div class="legend" id="l_dual_clkpv"></div>
  </div>
  <div class="note">核心频率均值 node0 {nf(c0['pclk'],1)} MHz / node1 {nf(c1['pclk'],1)} MHz；
    pviol 分桶均值 node0 {nf(pv0_avg,1)}% / node1 {nf(pv1_avg,1)}%，且 {nf(c0['power_cap_pct'],1)}% / {nf(c1['power_cap_pct'],1)}% 的采样点达到 pviol=100。
    两条 pviol 曲线长期贴近 100%，对应频率在 1860–1965 MHz 间随功率预算自适应浮动；两机 <code>tviol</code> 全为 0，
    说明限制来自 <b>TDP 功率墙而非散热限频</b>。</div>
  <div class="chart" id="c_dual_cards" style="margin-top:18px"><div class="ct">16 张 GPU 平均温度排序</div><div class="cs">蓝色 = node0（ranks 0–7） · 橙色 = node1（ranks 8–15）</div></div>
  <div class="chart" id="c_dual_pcap" style="margin-top:18px"><div class="ct">功率墙 pviol 采样分布对比</div><div class="cs">每档左侧 node0 / 右侧 node1</div></div>
  <h3>双机指标差异表</h3>
  <table><thead><tr><th style="width:20%">指标</th><th>node0</th><th>node1</th><th>差值</th><th>判读</th></tr></thead><tbody>{diff_html}</tbody></table>
  <div class="note"><b>值得留意的一点：</b>两节点稳定期显存相差恒定 {nf(abs(diff['fb']),0)} MB（{nf(abs(diff['fb'])/1024,2)} GiB），
    node0 为 {nf(g0['summary']['0']['fb']['p50'],0)} MB、node1 为 {nf(g1['summary']['0']['fb']['p50'],0)} MB，且各自 8 卡内部差异 &lt; 1 MB、p50=p95=max。
    由于 node0 记录的 PyTorch <code>max_reserved</code> 为 {nf(m0.get('max_reserved'),0)} MB 且 8 个 rank 完全一致，这 {nf(abs(diff['fb'])/1024,2)} GiB 差异几乎全部落在
    <b>NCCL / CUDA context / 驱动缓冲等非 PyTorch 层</b>，而非模型或优化器状态。全程无 OOM、吞吐 CV 仅 {nf(S['throughput_stable']['cv'],2)}%，不影响稳定性；若做容量规划，建议按 node1 的 108.2 GiB 作为上界。</div>
</section>
"""

    # ---------- C 历史基准 ----------
    st_counts = hs["by_status"]
    runs_sorted = sorted(hist, key=lambda h: (h["status"] != "失败", h["file"]))
    run_html = "\n".join(
        f'<tr><td><code>{h["file"]}</code></td><td>{h["kind"]}</td><td><span class="tag tag-{"ok" if h["status"]=="完成短跑" else ("warn" if h["status"]=="失败" else "info")}">{h["status"]}</span></td>'
        f'<td>{h["max_iter"]}</td><td>{nf(h["last_tflops"],1) if h["last_tflops"] else "—"}</td>'
        f'<td>{h["tracebacks"]}</td><td>{", ".join(f"{k}×{v}" for k,v in h["errors"].items()) or "—"}</td></tr>'
        for h in runs_sorted)
    err_html = "\n".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in
        sorted(hs["error_types"].items(), key=lambda kv: -kv[1]))

    section_c = f"""
<section id="sC">
  <h2><span class="no">C</span>双机测试前的前置调参 / 基准运行</h2>
  <p class="lead">node0/logs 下另有 {hs['runs']} 次早于双机测试的 short-run（8B 单机、56B 单机、MFU 基准），是本次 56B 双机配置定型的调参轨迹。</p>
  <div class="grid2">
    <div class="chart" id="c_hist_status"><div class="ct">运行结局分布</div><div class="cs">共 {hs['runs']} 次 · 失败 {st_counts.get('失败',0)} / 完成短跑 {st_counts.get('完成短跑',0)} / 初始化中断 {st_counts.get('初始化中断',0)}</div></div>
    <div class="chart" id="c_hist_tflops"><div class="ct">短跑 MFU 分布</div><div class="cs">末次吞吐 ÷ 989.5 TFLOP/s · 双机稳定期基线 {nf(mfu['stable_mean'])}%</div></div>
  </div>
  <h3>失败原因汇总</h3>
  <table><thead><tr><th style="width:40%">错误类型</th><th>出现次数（按运行文件累计）</th></tr></thead><tbody>{err_html}</tbody></table>
  <h3>{hs['runs']} 次运行明细</h3>
  <table><thead><tr><th>文件</th><th>类型</th><th>结局</th><th>最大 iter</th><th>末次吞吐</th><th>Traceback</th><th>错误</th></tr></thead><tbody>{run_html}</tbody></table>
  <div class="note">调参期累计 {hs['total_tracebacks']} 个 Traceback、最深跑到 iter {hs['max_iter']}；
    短跑 MFU 分布在 {nf(mfu['hist_range'][0])}%–{nf(mfu['hist_range'][1])}%（同一 H200 峰值口径）。
    最终双机配置（TP8 × DP2、seq 8192、GBS 64、BF16、mock_data）连续稳定运行 {nf(S['hours'],2)} h 无致命错误，
    吞吐稳定在 {nf(S['throughput_stable']['mean'],1)} TFLOP/s/GPU、MFU {nf(mfu['stable_mean'])}%，说明前置试错已收敛到可用配置。</div>
</section>
"""

    # ---------- 双机 JS ----------
    dual_js = r"""
/* ============ 双机联合图表 ============ */
if (DUAL.gpu0) {
  const G0 = DUAL.gpu0, G1 = DUAL.gpu1, CM = DUAL.dual, HS = DUAL.hist_summary, HR = DUAL.history;
  const N0 = 'node0', N1 = 'node1';
  const rel = [
    {k:'功耗', a:CM.cmp.node0.pwr, b:CM.cmp.node1.pwr, c:C.blue},
    {k:'核心温度', a:CM.cmp.node0.gtemp, b:CM.cmp.node1.gtemp, c:C.blue},
    {k:'显存占用', a:CM.cmp.node0.fb, b:CM.cmp.node1.fb, c:C.blue},
    {k:'核心频率', a:CM.cmp.node0.pclk, b:CM.cmp.node1.pclk, c:C.blue},
    {k:'功率封顶占比', a:CM.cmp.node0.power_cap_pct, b:CM.cmp.node1.power_cap_pct, c:C.blue}];
  const relData = [];
  rel.forEach(r=>{relData.push({k:r.k, v:+(r.a/r.b*100).toFixed(2), c:C.blue});
                  relData.push({k:'', v:100, c:C.orange});});
  barChart('c_dual_rel',{data:relData,h:300,ymin:90,ymax:106,yfmt:v=>n2(v,0)+'%',
    vfmt:v=>n2(v,2)+'%',xlabel:'node1 = 100%（每组左蓝 node0 / 右橙 node1）'});
  legend('l_dual_rel',[{n:'node0 相对值',c:C.blue},{n:'node1 基准 100%',c:C.orange}]);

  function bseries(metric,f){return {n0:G0.buckets.map(b=>[b.t,f?f(b[metric]):b[metric]]),
                                   n1:G1.buckets.map(b=>[b.t,f?f(b[metric]):b[metric]])};}
  const sfb=bseries('fb',v=>v/1024), spw=bseries('pwr'), st=bseries('gtemp');
  const spci={n0:G0.buckets.map(b=>[b.t,(b.rxpci||0)+(b.txpci||0)]),
              n1:G1.buckets.map(b=>[b.t,(b.rxpci||0)+(b.txpci||0)])};
  lineChart('c_dual_fb',{series:[{n:N0,c:C.blue,data:sfb.n0,fmt:v=>n2(v,2)+' GiB'},
    {n:N1,c:C.orange,data:sfb.n1,fmt:v=>n2(v,2)+' GiB'}],
    xmin:0,xmax:23.5,ymin:90,ymax:115,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'显存 GiB',h:300});
  legend('l_dual_fb',[{n:N0,c:C.blue},{n:N1,c:C.orange}]);
  lineChart('c_dual_pwr',{series:[{n:N0,c:C.blue,data:spw.n0,fmt:v=>n2(v,1)+' W'},
    {n:N1,c:C.orange,data:spw.n1,fmt:v=>n2(v,1)+' W'}],
    xmin:0,xmax:23.5,ymin:640,ymax:710,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'功耗 W',h:300});
  legend('l_dual_pwr',[{n:N0,c:C.blue},{n:N1,c:C.orange}]);
  lineChart('c_dual_temp',{series:[{n:N0,c:C.blue,data:st.n0,fmt:v=>n2(v,1)+' °C'},
    {n:N1,c:C.orange,data:st.n1,fmt:v=>n2(v,1)+' °C'}],
    xmin:0,xmax:23.5,ymin:45,ymax:70,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'温度 °C',h:300});
  legend('l_dual_temp',[{n:N0,c:C.blue},{n:N1,c:C.orange}]);
  lineChart('c_dual_pci',{series:[{n:N0,c:C.blue,data:spci.n0,fmt:v=>n2(v,0)+' MB/s'},
    {n:N1,c:C.orange,data:spci.n1,fmt:v=>n2(v,0)+' MB/s'}],
    xmin:0,xmax:23.5,ymin:0,ymax:3200,xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'rx+tx MB/s',h:300});
  legend('l_dual_pci',[{n:N0,c:C.blue},{n:N1,c:C.orange}]);

  const clk0=G0.buckets.filter(b=>b.pclk!==null&&b.pviol!==null).map(b=>[b.t,b.pclk]);
  const clk1=G1.buckets.filter(b=>b.pclk!==null&&b.pviol!==null).map(b=>[b.t,b.pclk]);
  const pv0=G0.buckets.filter(b=>b.pviol!==null).map(b=>[b.t,b.pviol]);
  const pv1=G1.buckets.filter(b=>b.pviol!==null).map(b=>[b.t,b.pviol]);
  lineChart('c_dual_clkpv',{dual:true,series:[
    {n:'node0 核心频率',c:C.green,data:clk0,fmt:v=>n2(v,0)+' MHz'},
    {n:'node1 核心频率',c:C.cyan,data:clk1,fmt:v=>n2(v,0)+' MHz'},
    {n:'node0 pviol',c:C.orange,data:pv0,axis:2,dash:'5 3',fmt:v=>n2(v,1)+'%'},
    {n:'node1 pviol',c:C.red,data:pv1,axis:2,dash:'5 3',fmt:v=>n2(v,1)+'%'}],
    xmin:0,xmax:23.5,ymin:1830,ymax:1980,y2min:0,y2max:105,
    xfmt:v=>n2(v,0)+'h',xlabel:'运行时长',ylabel:'核心频率 MHz',y2label:'pviol %',h:360});
  legend('l_dual_clkpv',[{n:'node0 频率（实线·左轴）',c:C.green},{n:'node1 频率（实线·左轴）',c:C.cyan},
    {n:'node0 pviol（虚线·右轴）',c:C.orange},{n:'node1 pviol（虚线·右轴）',c:C.red}]);

  const cardData=[];
  for(let g=0;g<8;g++)cardData.push({k:'N0-'+g,v:G0.summary[g].gtemp.mean,c:C.blue});
  for(let g=0;g<8;g++)cardData.push({k:'N1-'+g,v:G1.summary[g].gtemp.mean,c:C.orange});
  cardData.sort((a,b)=>a.v-b.v);
  barChart('c_dual_cards',{data:cardData,h:320,ymin:48,ymax:66,yfmt:v=>n2(v,0)+'°',
    vfmt:v=>n2(v,2)+' °C',xlabel:'GPU（按均温升序）'});

  const p0=G0.pviol_dist,p1=G1.pviol_dist;
  const pkeys=[['0','0%'],['1-49','1–49%'],['50-89','50–89%'],['90-99','90–99%'],['100','100%']];
  const pdata=[];
  pkeys.forEach(([k,lab])=>{pdata.push({k:lab,v:p0[k],c:C.blue});pdata.push({k:'',v:p1[k],c:C.orange});});
  barChart('c_dual_pcap',{data:pdata,h:300,yfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 次',xlabel:'pviol 档位（每组左 node0 / 右 node1）'});

  barChart('c_hist_status',{data:Object.entries(HS.by_status).map(([k,v])=>({k:k,v:v,
    c:k==='完成短跑'?C.green:(k==='失败'?C.red:C.orange)})),h:280,
    yfmt:v=>n2(v,0),vfmt:v=>n2(v,0)+' 次',xlabel:'运行结局'});
  const hmf=DUAL.mfu.history.map(x=>x.mfu);
  histo('c_hist_tflops',hmf,5,C.purple,'MFU %（H200 BF16 dense 989.5 TFLOP/s 基准）');
}
"""

    # ---------- 单节点模式对应文案 ----------
    scope_single = ("口径说明：日志内部时间戳为 UTC，dmon 为 UTC+8（本地），本报告已统一到 UTC 时间轴；"
                    "<code>throughput per GPU</code> 是全局 16 ranks 的每卡口径；本次采集仅覆盖 node1 的 8 张本地 GPU。")
    scope_dual = ("口径说明：两份日志内部时间戳为 UTC，dmon 为 UTC+8（本地），已统一到 UTC 时间轴；"
                  "<code>throughput per GPU</code> 是全局 16 ranks 的每卡口径；"
                  "训练指标序列取自 node1 输出，配置 / 里程碑 / 每 rank 显存取自 node0，GPU 指标覆盖两节点共 16 张卡。")
    obs_single = ("<b>观测完整性：</b>补充 node2 的 dmon 与 IB/NVLink 计数器，才能评估跨节点 DP 通信开销；当前结论仅覆盖 node1 本地 8 卡。")
    obs_dual = ("<b>观测完整性：</b>本报告已覆盖两节点 16 张 GPU 与 IB/GDRDMA 通道统计；若需精确评估跨节点 DP 通信耗时，建议补充 NCCL 计时或 IB 计数器。")

    return {
        "title_dual": title, "title_single": "node1 · E2E 56B 训练结果统计分析",
        "subtitle_dual": subtitle, "head_meta_dual": head_meta,
        "nav_dual": nav_dual,
        "section_a": section_a, "section_b": section_b, "section_c": section_c,
        "dual_js": dual_js,
        "scope_dual": scope_dual, "scope_single": scope_single,
        "obs_dual": obs_dual, "obs_single": obs_single,
    }
