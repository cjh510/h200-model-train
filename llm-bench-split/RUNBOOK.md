# RUNBOOK —— llm-bench-fix 执行命令清单

> 本文件所有命令都以「脚本目录」为当前目录，默认放在 `/root/llm-bench`。
> 命令与脚本实现逐项核对过；以脚本为准。旧的 `TESTPLAN` / `03_deploy.sh` 末尾提示没有列
> `coll`、`io`、`fault` 三个阶段，但 `04_bench.sh` 实际支持这些 `--phase`，本 RUNBOOK 以脚本为准。
>
> 重要：多数阶段的 shell 退出码只反映「脚本有没有跑完」，不反映「每个档位是否通过」。
> `s8b` / `d70b` / `soak` 请以本次 run 目录里 `06-train-*/results.jsonl` 的 `status` 与同目录 `*.log` 为准；
> `hw` / `coll` / `nccl` / `io` / `fault` 请以对应阶段目录的 JSONL 与 `.txt` 日志为准。
>
> 所有产物按次归档：`$ROOT/runs/<RUN_ID>/`，每个阶段一个带序号子目录（见第 4 节）。
> 先执行 `preflight.sh` 会创建本次 run，后续阶段自动续用（也可以 `RUN_ID=<id>` 显式续跑）。

## 0. 前置

### 0.1 目录放置

两台机器使用**完全相同**的路径（脚本把 `$ROOT` 挂进容器，双机路径不同会直接找不到文件）：

```bash
# 两台都执行：把 llm-bench-fix 目录内容放到 /root/llm-bench
mkdir -p /root/llm-bench
# 将 llm-bench-fix 下的脚本、probes/、05_report.py 等复制到 /root/llm-bench
```

`ROOT` 默认 `/root/llm-bench`，可用 `ROOT=<路径>` 覆盖；覆盖后两台仍必须一致。
每次运行的结果写到 `$ROOT/runs/<RUN_ID>/`；`$ROOT/logs` 会自动做成指向 `runs/latest`
的符号链接（文件系统不支持符号链接时只打印替代路径），旧命令里的 `logs/...` 仍然可用。

目录应包含：

| 位置 | 内容 |
|---|---|
| `$ROOT/lib.sh`、`preflight.sh`、`02_precheck.sh`、`03_deploy.sh`、`04_bench.sh`、`05_report.py` | 本目录脚本 |
| `$ROOT/bench/*.sh` | **`04_bench.sh` 的实现模块**（11 个文件，按依赖编号）。入口只做参数解析与分发，正文全在这里；**缺一个文件入口会直接报「同步不完整」退出** |
| `$ROOT/probes/` | `hw_probe.py`、`io_probe.py`、`gpu_health.sh` |
| `$ROOT/pytorch*.tar` 或 `pytorch*.tar.gz` | 镜像归档（`03_deploy.sh` 自动 load） |
| `$ROOT/Megatron-LM*.tar.gz` / `*.tgz` / `*.zip` | Megatron 源码包，解出 `Megatron-LM/pretrain_gpt.py` |
| `$ROOT/SHA256SUMS` | 可选；有则强校验，没有则只做文件头弱校验 |
| `$ROOT/data/*.bin` + `*.idx` | 可选；有则 `--phase io` 跑真实 dataloader，否则 SKIP |
| `$ROOT/runs/<RUN_ID>/` | 自动生成，一次运行的全部产物（见第 4 节） |
| `$ROOT/logs` | 自动生成的符号链接 → `runs/latest`（兼容旧路径） |

`04_bench.sh` 已按职责拆分，入口 84 行只做「加载模块 → 解析阶段 → 分发」：

```
04_bench.sh          入口：# ---- 模块 ---- / parse_phase / boot_env / 分发表 / 收尾
bench/00-config.sh   唯一的可调面：默认值、PEAK/VRAM 常量、LLAMA_*/MOE_B、阶段清单 ALL_PHASES
bench/10-boot.sh     双机门禁、GID 解析、GPU 排空检查、dmon、cleanup、boot_env()
bench/20-net.sh      RoCE 门禁：逐口 GID、8MB allreduce 探针、rdma_gate_precheck()
bench/30-record.sh   results/moe/health 记账、step/err/expert 解析、--help 自检
bench/40-run.sh      run_train / fault_train / 进度 / 专家负载采样（唯一起训练容器的两处）
bench/50-phase-train.sh  argscheck hw coll nccl s8b dx8b d70b moe moe-dp io fault report
bench/60-phase-soak.sh   长稳 dense / dense70 / moe + phase_soak
bench/62-highmem.sh      显存高占用长稳：采样、从 mfu.jsonl 取峰值档、泄漏/碎片判定
bench/70-phase-mfu.sh    峰值 MFU 全家族
```

> **同步单位变了**：以前同步「`04_bench.sh` + `lib.sh` + `05_report.py`」三个文件，
> 现在要同步 **`04_bench.sh` + 整个 `bench/` 目录**（再加 `lib.sh` / `05_report.py`）。
> 入口对每个模块做了 `|| exit 2`，少文件会立刻报错而不是跑一半才 `command not found`。

### 0.2 镜像导入位置

把 `pytorch*.tar`（或 `.tar.gz`）放在 `$ROOT` 根目录即可，`03_deploy.sh` 会优先复用引擎里已有镜像，
没有再 `load -i`。已有镜像可直接指定：

```bash
IMAGE=<repository:tag> bash 03_deploy.sh
```

### 0.3 引擎说明

- 默认 **docker 优先**：`docker version` 可用才采用；不可用自动回落 **nerdctl**。
- 目标机当前只有 nerdctl，不需要额外设置，会自动选 `使用 nerdctl（namespace=default）`。
- 强制指定：`ENGINE=nerdctl bash <脚本>`（docker 可用时也可强制走 nerdctl）。
- nerdctl 路径固定 `CONTAINERD_NAMESPACE=default`，脚本拒绝 `k8s.io` / `moby`；
  docker 路径忽略该变量。
- 启动时打印引擎行，并写入本次 run 的阶段目录（如 `01-preflight/engine.log`）。

### 0.4 按次运行（run）

- `preflight.sh` 每次启动默认新开一次 run：
  `<YYYY-MM-DD>_<HHMM>_<hostname>_<scope>_<phases>`，例 `2026-11-05_2257_worker-gpu-a_w8_full`。
- `scope` 由 `NPROC × NNODES` 推断：`w8` / `w16`；`phases`：单阶段用阶段名，多阶段 `full`，长稳单跑 `soak`。
- 后续脚本默认续用 `runs/latest`；显式续跑：`RUN_ID=<id> bash 04_bench.sh --phase hw`。
- 指定全新目录：`RUN_ID=<自定义名>`（同时避免覆盖）；整体换根目录用 `OUT_DIR=<路径>`。
- 同名冲突自动追加 `-2`、`-3`，不覆盖已有目录。

## 1. 单机完整链路

```bash
# 进入脚本目录（两台路径必须一致）
cd /root/llm-bench

# 1) 开跑前一次性自检：GPU 排空 / 引擎 / 镜像 / toolkit / runtime / 内核 / rmem / 容器内 8 卡
#    退出码 0=全过，1=有 WARN（可跑，写进报告），2=有 FAIL（先修）
bash preflight.sh

# 2) 环境体检（只读）：产出 <hostname>-env.json 与 <hostname>-topo.txt
#    退出码 0=干净，1=有 WARN，2=有 FAIL
bash 02_precheck.sh

# 3) 校验归档 → 复用/导入镜像 → 解 Megatron → 容器内 8 卡自检 → 写 03-deploy/env.txt
#    通过判据：日志出现 "必需参数全部存在"、8 卡断言通过、03-deploy/env.txt 生成
bash 03_deploy.sh

# 4) 硬件基线：bf16/FP8 GEMM、HBM、L2、PCIe、collective（稳态 PROBE_SECS=5）
#    产物 04-hw/hw.jsonl、hw.txt、dmon.csv；判据：hw.jsonl 有记录且 hw.txt 无 !! / 大面积 SKIP
bash 04_bench.sh --phase hw

# 5) 单机 collective 全类型扫描（已并入 hw；仅 OPTIONAL=1 时单独跑）
#    产物 05-coll/coll.jsonl、coll.txt
OPTIONAL=1 bash 04_bench.sh --phase coll

# 6) 单机 8B 矩阵 S1–S4（batch 上界 / 并行策略 / 长序列 / FP8 vs BF16），共 12 档（精简矩阵）
#    产物 06-train-s8b/w8_8B_*.log、results.jsonl；判据：results.jsonl 中 status=OK 的档位数
bash 04_bench.sh --phase s8b

# 7) checkpoint 分片写读 + 数据管线（无 $ROOT/data/*.bin+.idx 时 dataloader 记 SKIP）
#    产物 07-io/io.jsonl、io-dataloader.log（若有真实数据）
bash 04_bench.sh --phase io

# 8) 故障注入：F1 单机 70B 预期 OOM / F2 kill 一个 rank / F3 SIGSTOP + NCCL 超时
#    产物 08-fault/fault.jsonl、fault-*.log；判据：三个 case 均为 PASS
bash 04_bench.sh --phase fault

# 9) 单机长稳（NNODES 默认 1，自动起 dmon，跑前跑后采 ECC/NVLink/Xid）
#    默认 SOAK_MODEL=dense：8B TP8×PP1 mbs4 GBS8，SOAK_MINUTES=120
#    产物 09-soak/results.jsonl 的 soak 行、health-before/after/delta.json、dmon-w8_8B_soak_*.csv
#    判据：status=OK 且 health-delta.json 的 status=PASS
SOAK_MINUTES=120 bash 04_bench.sh --phase soak

# 9b) 再加一条 MoE 长稳（配置 B，EP8×TP1×PP1，alltoall；单机默认 30 分钟）
#    产物 moe-soak.jsonl、expert-load-timeseries.csv、moe-health-before/after/delta.json、
#         dmon-moe-soak-w8_MOE_ep8_alltoall.csv，哨兵 .ok_moe_soak / .fail_moe_soak
#    两条都跑：先 dense 后 MoE，任一失败整阶段非 0（互不影响对方结论）
SOAK_MODEL=both MOE_SOAK_MINUTES=30 bash 04_bench.sh --phase soak

# 9c) 第三条：显存高占用长稳（2026-11 改版：档位直接取 MFU 峰值档）
#    ★ 不再自己标定 mbs 去凑占用率 —— 那样选出来的是「为压显存而跑的配置」，吞吐没有参考价值。
#      现在取本 run 里 --phase mfu 中 MFU† 最高的那一档（model/TP/PP/mbs/seq/GBS 全沿用），
#      跑满 HIGHMEM_MINUTES，回答「我们要用的最优配置，连跑一小时显存稳不稳、有没有泄漏」。
#    ★ 前置：本 run 必须已有 mfu.jsonl；没有就报错退出（提示先跑 --phase mfu），不会瞎跑。
#    ★ 判定：没有占用率通过线。跑满且无异常 = OK；任一卡增长 ≥5% = WARN_LEAK；
#      峰值在前 20% 且末尾仍贴峰 = WARN_FRAG；OOM = OOM_BOUNDARY（不计失败）；
#      <10 分钟 = SMOKE（冒烟，不判定、不写哨兵）。peak_pct 只记录，用来回答「这配置还剩多少余量」。
#    产物 highmem.jsonl、memused-w<N>_highmem.csv、highmem-health-*.json、dmon-highmem-w<N>_*.csv
SOAK_MODEL=highmem HIGHMEM_MINUTES=60 bash 04_bench.sh --phase soak

# 9d) 峰值 MFU：回答「单机/双机最高能到多少」，不是「某一档的 MFU」
#     A 粗扫并行度×GBS → B 细扫 mbs×accum → C seq 扫描 → D 前 3 名复测取中位
#     先看计划（不跑，几秒）：MFU_PLAN_ONLY=1 ...
MFU_PLAN_ONLY=1 bash 04_bench.sh --phase mfu
#     正式跑：单机 8B 约 30 档 ≈ 75 分钟；双机 8B+70B 约 40 档 ≈ 2.5 小时
bash 04_bench.sh --phase mfu
#     产物：mfu.jsonl（逐档）、mfu-peak.json（每 (模型, world) 一个峰值）、
#           clocks-w<N>_MFU_*.csv（SM 时钟/功耗/throttle 取证）
# 单机三条全跑（顺序 dense → moe → highmem；dense70 仅双机，单机 all 不启用）
# 各自哨兵/遥测/硬件增量互不干扰
SOAK_MODEL=all bash 04_bench.sh --phase soak

# 10) 汇总报告：读整次 run（各阶段目录递归收集），报告同时落到 10-report/
bash 04_bench.sh --phase report

# 11) 可选：单独跑报告脚本，指定某次 run 目录
python3 05_report.py --logs /root/llm-bench/runs/latest
```

### 1.1 本轮新增阶段（单机：hw → mfu → moe；coll 可选）

```bash
# 0) 参数自检（必做，也被下列阶段自动触发）：容器内跑 pretrain_gpt.py --help
#    产物 <run>/args-help.txt、args-check.json；缺失 flag → 剔除 + WARN；关键 4 项缺失 → MoE 阶段 SKIP
bash 04_bench.sh --phase argscheck

# 5) collective 四类（all_reduce / all_gather / reduce_scatter / all_to_all）已并入 hw
#    字段含 op、busbw_gbs、latency_us、world、nodes；all_to_all 另记 groups/tokens_per_group/split_dim
#    只有需要单独扫描时才跑本阶段
OPTIONAL=1 bash 04_bench.sh --phase coll

# 6) MoE / MLA（配置 B：24 层 / 32 专家 + 1 共享 / topk2 / seq4096 / GBS8）
#    单机两档 EP8×TP1、EP2×TP4 × dispatcher 两种（alltoall / allgather）
#    MLA 路径必须在参数里带 --enable-experimental（fused_apply_mla_rope_for_q 被实验性开关挡住）
#    产物 06-train-moe/moe.jsonl；判据：status=OK，专家负载 max/mean >1.5 标 WARN
#    ★ 修复后必须用新的 RUN_ID 复跑；旧 run 的 FAILED 记录不要与新结果合并
RUN_ID=<新的 run 名> bash 04_bench.sh --phase moe

# 6b) MoE 的 MLA 开关：MOE_MLA=auto（默认，--help 有 --enable-experimental 才跑）| 1 强制 | 0 跳过
#    依赖缺失（--multi-latent-attention/q-lora-rank/kv-lora-rank/qk-head-dim/
#    qk-pos-emb-head-dim/v-head-dim/--enable-experimental 任一）→ 该档 SKIP，不是 FAILED
MOE_MLA=1 RUN_ID=<新的 run 名> bash 04_bench.sh --phase moe

# 7) 峰值 MFU（替代原来的 limit：不再扫「撞墙边界」，而是扫「最高吞吐」）
#    粗扫并行度×GBS → 细扫 mbs×accum → seq → 前 3 名复测取中位；产物 mfu.jsonl + mfu-peak.json
#    先用 MFU_PLAN_ONLY=1 看网格，几秒出结果
MFU_PLAN_ONLY=1 bash 04_bench.sh --phase mfu
bash 04_bench.sh --phase mfu
```

耗时粗估（单机 8 卡，`STEPS=15`，精简矩阵）：`argscheck` <1 分钟；`hw` 5–10 分钟（含四类 collective）；
`coll` 仅在 `OPTIONAL=1` 时跑，约 6 分钟；`moe` 4 档 × 约 2–4 分钟；`mfu` 单机约 30 档（≈75 分钟）、双机约 41 档（≈2.5 小时）。

### 1.2 长稳：单机三条 / 双机四条独立长稳

`soak` 阶段由 `SOAK_MODEL` 选形态，默认 `dense`（与历史行为完全一致）；支持逗号组合，
按列出顺序执行。`both` 兼容旧语义 = dense+moe；`all` = 全部适用项：
单机 dense→moe→highmem，双机 dense→**dense70**→moe→highmem。`dense70` 仅双机可用。

> **版本识别**：双机最新脚本的横幅应显示 `长稳计划：dense dense70 moe highmem`，并列出各段**实际**分钟数。
> 若仍显示 `dense moe highmem`，说明目标机运行的是未同步的旧版 `04_bench.sh`，也可能仍带「非末级节点 0 步误判 PARTIAL」问题，应停止本轮并同步后再跑。

| SOAK_MODEL | 跑什么 | 哨兵 | 主要产物 |
|---|---|---|---|
| `dense`（默认） | 8B TP8×PP1 mbs4 GBS8，`SOAK_MINUTES`（默认 120） | `.ok_<tag>` / `.fail_<tag>` | `results.jsonl` soak 行、`health-*.json`、`dmon-w<N>_8B_soak_*.csv` |
| `dense70`（双机） | Llama-3.1-70B TP8×PP2×DP1，mbs1 seq4096 GBS16，num_micro_batches=16（PP 气泡约 6.25%），`DENSE70_SOAK_MINUTES`（默认 60） | `.ok_w16_70B_soak_*` / `.fail_w16_70B_soak_*` | `results.jsonl` soak 行、`dense70-health-*.json`、`dmon-w16_70B_soak_*.csv`；性能步数取全局末级 rank（node-b）日志 |
| `moe` | 配置 B（24 层/32 专家/topk2/MLA），EP×TP1×PP1，alltoall，`MOE_SOAK_MINUTES`（双机 60 / 单机 30） | `.ok_moe_soak` / `.fail_moe_soak` | `moe-soak.jsonl`、`expert-load-timeseries.csv`、`moe-health-*.json`、`dmon-moe-soak-w<N>_*.csv` |
| `highmem` | 显存高占用长稳：**档位 = 本 run `mfu.jsonl` 里 MFU† 最高的那一档**（model/TP/PP/mbs/seq/GBS 全沿用），`HIGHMEM_MINUTES` 默认 60；**前置 = 已有 mfu.jsonl** | `.ok_highmem`（仅 OK）/ `.fail_highmem` | `highmem.jsonl`、`memused-w<N>_highmem.csv`、`highmem-health-*.json`、`dmon-highmem-w<N>_*.csv` |
| `both` | 先 dense，再 MoE | 两套互不覆盖 | 以上前两条 |
| `all` | 单机 dense→moe→highmem；双机 dense→dense70→moe→highmem | 各套互不覆盖 | 对应机器全部适用项 |
| 逗号组合 | 如 `dense,dense70,moe`、`highmem,moe`，按书写顺序；`dense70` 在单机是配置错误 | 各自哨兵 | 对应条目 |

- **档位来源（highmem，2026-11 改版）**：直接取本 run `mfu.jsonl` 里 `status=OK` 且 `nodes` 匹配、
  `MFU†` 最高的一档，原样复用它的 `model / TP / PP / mbs / seq / GBS`。
  为什么不再自己标定：旧版为了让占用率落进 75~92%，会挑一个**不是最优**的配置（实测 11-10 单机落到
  mbs=1/seq=16384，只占 29%），长稳跑出来的吞吐没有参考价值；而为了凑占用率去改 mbs，本质上是在测
  一个没人会用的配置。前置依赖从「limit 边界」换成了「MFU 峰值」。
  没有 `mfu.jsonl`、或里面没有可用档（全 OOM/FAILED）、或该档 TP×PP 超过当前 world 时，**直接报错退出**
  并提示先跑 `--phase mfu`——不会退回「猜一个 mbs」。
- 显存观测：独立 `mem_sampler` 每 `MEM_SAMPLE_SECS`（默认 5s）采一次逐卡 used，
  写 `memused-w<N>_highmem.csv`（ts,gpu,used_mib）。
- `highmem.jsonl` 逐卡写 `mem_start/mem_max/mem_end`、`growth_pct`、`peak_pct`（分母 143771 MiB）、
  `mem_max_at_pct`（峰值出现在时间轴的百分之几）、`end_over_peak_pct`；整档另有
  `peak_mib` / `peak_pct` / `growth_pct_max` / `frag_suspect` / `source=mfu` / `source_mfu_pct`。
  **判定（没有占用率通过线）**：
  - 跑满 `HIGHMEM_MINUTES` 且无异常 → `OK`，写 `.ok_highmem`；
  - 任一卡增长 ≥ `HIGHMEM_GROWTH_MAX`（默认 5%）→ `WARN_LEAK`（疑似泄漏，必须看曲线），**不写哨兵**；
  - 峰值出现在前 20% 且末尾仍 ≥ 峰值−`HIGHMEM_FRAG_DIFF`（默认 3%）→ `WARN_FRAG`（疑似碎片），不写哨兵；
  - OOM → `OOM_BOUNDARY` + `error_class`，不计失败、不 dual_stop、退出码 0
    （MFU 是 40 步短跑，60 分钟的碎片更高，这本身就是「最优配置的显存上限」结论）；
  - `HIGHMEM_MINUTES<10` → `SMOKE`，不判定、不写哨兵。
  `peak_pct` **只记录不判定**：档位是按吞吐选的，占用率高低不代表通过与否，只用来回答「还剩多少余量」。
- `MOE_SOAK_MIN_ITERS` / `DENSE70_SOAK_MIN_ITERS` 不填时沿用 `SOAK_MIN_ITERS`；MoE 档每 `EXPERT_SAMPLE_SECS`（默认 300s）采样一次专家负载。
  **数据源（11-10 P1 修订）**：core_v0.13.1 根本不向 stdout 打印逐专家 token 计数（`--moe-per-layer-logging` 只打 aux / z loss，`tokens_per_expert` 仅进 tensorboard），所以单纯改正则永远修不好 null。`moe)` / `soak_moe` 现在调用 `moe_probe_enable()`，把 `probes/moe_load_probe.py` 挂成 `$LOGDIR/sitecustomize.py` 并透传 `MOE_LOAD_PROBE=1`，训练时按 `MOE_LOAD_LOG_EVERY`（默认 200 次 routing）打印 `[expert load] step=N max/mean=x/y counts=[...]`，`expert_load_pair()` 解析该行写 CSV。探针缺失或 Megatron API 不匹配会打印 `[expert load probe] install failed|disabled` 并降级，此时 CSV 写 `null` 且 `note` 明确指向该原因，不再是静默的「未解析到 expert load」。
- `dense70` 是**正常训练口径**，不是 highmem 边界档：GBS16 / mbs1 / DP1 产生 16 个 micro-batches；
  TP8×PP2 用于验证双机 70B 的跨机流水线与高显存常态。Megatron 用 `print_rank_last`，
  双机只有全局末级 rank（node-b）打印 iteration；8B DP2、MoE EP16、70B PP2 都遵守这一规则。
  非末级节点 `rc=0` 但本机 0 行不再误判 PARTIAL，实际步数由末级节点判定；报告从 node-b 回传日志/JSONL 取性能。
  8B 与 70B 的 dmon、`health-delta.json` / `dense70-health-delta.json` 分别判定，不共用。
- **各条长稳验证的是不同子系统，不能互相替代**：8B Dense 看 DP2 持续算力/带宽；
  70B Dense 看 PP2 大模型跨机流水线；MoE 看专家路由与 alltoall；highmem 看显存余量、
  增长趋势与碎片，回答「能不能长期贴着显存上限跑」。
- **峰值 MFU（`--phase mfu`）的口径**——不给「唯一一个 MFU」，按 (模型, world) 各给一个：
  - 头条 `MFU† = Megatron 自报 throughput per GPU ÷ 989.5 TFLOPS`（H200 SXM bf16 **稠密**峰值；
    Megatron 的 FLOPs 公式含 attention，所以长序列会偏高，跨 seq 只比这个）；
  - 同时记 `MFU(6N) = 6·N·tokens/step ÷ world ÷ 989.5`，只算线性层，是**保守下界**；
  - 每档 `MFU_STEPS` 步（默认 40，报告丢前 5 步），前 3 名复测 `MFU_REPEAT` 次取**中位**；
    `CV > MFU_CV_MAX(2%)` 或采样到 `throttle≠0` 的档**不进峰值**；复测次数=1 的结论标 `UNCERTIFIED`；
  - **硬天花板**：`04-hw` 的 bf16 GEMM 实测 690 TFLOP/s = 理论 69.7%，训练峰值**不可能超过它**；
    报告里会同时给出「峰值占天花板的百分比」，剩下的差距才是并行/气泡/优化器开销。
  - 先跑 `MFU_PLAN_ONLY=1` 看计划与档数，再决定是否开跑（双机 auto 是 25 档）。
- 失败语义：任一条失败不影响其它条目的结论，但整个阶段返回非 0；重跑规则与各自哨兵一致（`FORCE=1` 强制重跑）。
- **进度显示**：长稳训练是前台阻塞的 `lc_run`，脚本另起后台循环只读训练日志，每 `PROGRESS_SECS`（默认 60s）往终端 **stderr** 打一行，同时追加到 `09-soak/soak-progress.log`：

  ```text
  2026-11-09 21:35:00  [soak dense 42:00/2:00:00 35%  ETA 1:18:00] step 1683/>=100  last 1.24s  159.9 TFLOP/s/GPU
  ```

  含已用 / 目标时长、百分比、ETA、步数与 `SOAK_MIN_ITERS` 门槛、最近一次 step 与吞吐。双机额外带 `node<rank>`；只有全局末级 rank 打印 iteration，非末级节点步数恒 0 时会标注「步数由全局末级 rank 打印」，**不是卡死**。`SOAK_PROGRESS=0` 关闭；短档（15 步）不启用；输出走 stderr 以免污染参数拼接的 stdout。
- 进度启动前会先清空同名训练日志，避免首行读到上一轮残留的 iteration；训练结束后补打一条收尾进度。容器 stdout 可能批量刷盘，60 秒周期行的步数可能滞后，**收尾行和 `results.jsonl` 才是最终口径**。
- **短时冒烟**：只验证脚本编排时，可把四段都压到 1 分钟并同步调低步数门槛；这不是性能/稳定性验收：

  ```bash
  SOAK_MODEL=all SOAK_MINUTES=1 DENSE70_SOAK_MINUTES=1 \
  MOE_SOAK_MINUTES=1 HIGHMEM_MINUTES=1 SOAK_MIN_ITERS=1 \
  bash 04_bench.sh --phase soak
  ```

  双机两台变量必须完全一致。若只把 `DENSE70_SOAK_MINUTES` 改为 1，却保留默认 `SOAK_MIN_ITERS=100`，70B 在 1 分钟内通常只能跑几步，末级节点会按 PARTIAL 中止，这是门槛配置问题，不是训练失败。
- 单机命令见第 1 节 9/9b/9c，双机命令见 2.2 末尾（两台都要给 `SOAK_MODEL`，必须一致）。

## 2. 双机完整链路


两台先各自确认路径与网卡名完全一致；以下变量在**两台**都设置：

```bash
# 两台都执行：公共变量（按现场替换 IP / 网卡名）
cd /root/llm-bench
export NODE_A_IP=192.0.2.101                 # node-a 管理 IP（不能是回环）
export NODE_B_IP=192.0.2.11                # node-b 管理 IP（PEER_ADDR 用：逐口门禁 ssh 对端逐口表 + node-b 日志回传）
export NCCL_IF=bond0                           # 管理/数据网卡名，按 ip a 实际值
export NCCL_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8,mlx5_bond_9
```

> **双机新增的硬性要求（脚本自动处理，宿主必须满足）**
>
> 1. **IB 设备**：宿主 `/dev/infiniband/` 下要有 `uverbs*` 与 `rdma_cm` 字符设备。
>    `NNODES>1` 时 `lib.sh` 的 `lc_run` 会逐个 `--device` 注入；nerdctl 的 `--device` **不接受目录**，
>    不要手工传 `/dev/infiniband`。缺设备会打印告警，容器内 NCCL 退回 socket。
> 2. **Gloo**：`NNODES>1` 时自动追加 `--disable-gloo-process-groups`。容器主机名是随机 ID、
>    容器内无 DNS/对端 hosts，Gloo 的 TCP 进程组会退化成 `127.0.0.1`（日志里
>    `loopback` / `connectFullMesh` 警告就是它）；NCCL 组用 IP，不受影响。单机不加，保持已验证路径。
> 3. **退出码**：双机任一档 `OOM/PARTIAL/FAILED` → 写 `results.jsonl` 的 `ABORTED` 并 `exit 2`，
>    两台会同步中止（避免档位错位）；单机不中止，逐档看 `results.jsonl` 的 `status`。
> 4. **判定**：双机 `rc=0` 但本机无 iteration 也算 PASS（Megatron 只有全局末级 rank 打日志，
>    与 PP 是否启用无关），对端回传的 `node-b/**.jsonl` 里有该 tag 的 `OK` 同样算 PASS。

### 2.1 两台都要执行（可先后执行，不要求同时）

```bash
# node-a 先执行：自检并生成本次 run 名（保持同一个 shell，RUN_ID 要传给 node-b）
bash preflight.sh
export RUN_ID=$(basename "$(readlink -f /root/llm-bench/runs/latest)"); echo "$RUN_ID"

# node-b 用 node-a 的 RUN_ID 跑自检：强制同名，node-b 的 01–03 也落在同结构目录
export RUN_ID=<node-a 上面打印的 run 名>
bash preflight.sh

# 两台都执行：环境体检，产出各自的 <hostname>-env.json
bash 02_precheck.sh

# 两台都执行：镜像 + Megatron + 容器内 8 卡自检 + 03-deploy/env.txt
bash 03_deploy.sh

# 两台都执行：比较结果指纹（engine/namespace/digest/版本必须一致）
# 在 node-a 上执行：
diff <(grep -v -E 'host=|time=' /root/llm-bench/runs/latest/03-deploy/env.txt) \
     <(ssh node-b grep -v -E 'host=|time=' /root/llm-bench/runs/$RUN_ID/03-deploy/env.txt)
```

### 2.2 node-a 执行：单机基线 + 双机训练（双机段必须与 node-b 同时执行）

```bash
# 仅 node-a：单机基线（node-b 不执行）
bash 04_bench.sh --phase hw
bash 04_bench.sh --phase coll
bash 04_bench.sh --phase s8b
bash 04_bench.sh --phase io
bash 04_bench.sh --phase fault

# 双机：8B TP8×DP2 训练对照（GBS=32；跨机 collective 由 nccl 阶段承担）
# ★ 与下一节 node-b 的同一条命令同时执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase dx8b

# 双机：collective 全类型 × 8B–8G（不跑训练）
# ★ 与 node-b 同时执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase nccl

# 双机：70B TP8×PP2
# ★ 与 node-b 同时执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase d70b

# 本轮新增：MoE EP16×TP1×PP1（GBS16，dispatcher 两种）
# ★ 与 node-b 同时执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase moe

# 本轮新增：EP8×DP2（专家并行留机内，梯度跨机 all-reduce）—— 与 EP16 档对照
# ★ 与 node-b 同时执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase moe-dp

# 本轮新增：显存 / 规模墙（配置 A：80 层 / hidden 10240 / 80 heads / FFN 35840 / 词表 128256，TP8×PP2）
#   先看网格：MFU_PLAN_ONLY=1，只看不跑（双机 auto 是 25 档）
# ★ 与 node-b 同时执行；两台的 MFU_* 参数必须一致，否则档位序列会错位
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  bash 04_bench.sh --phase mfu

# 双机长稳：Dense 8B 120 + Dense 70B 60 + MoE 60 + 高占用 60 ≈ 5 小时
#   SOAK_MODEL=all 顺序 dense→dense70→moe→highmem
#   Dense 8B ：TP8×PP1 mbs4 GBS8
#   Dense 70B：TP8×PP2 mbs1 seq4096 GBS16，≥100 步，默认 60 分钟
#   MoE      ：配置 B，EP16×TP1×PP1，alltoall，GBS16（NNODES=2 自动 60 分钟）
#   highmem  ：档位取本次 run mfu.jsonl 里 nodes=2、MFU† 最高的那一档（两台读同一份数据，
#              档位天然一致，不再需要「双机不标定」的特殊规则）；没有 mfu.jsonl 就直接报错退出
# ★ 与 node-b 同时执行，两台 SOAK_MODEL / SOAK_MINUTES / DENSE70_SOAK_MINUTES /
#   MOE_SOAK_MINUTES / HIGHMEM_MINUTES 必须一致（highmem 的档位由 mfu.jsonl 决定，不用手工对齐）
NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_B_IP \
  SOAK_MODEL=all SOAK_MINUTES=120 DENSE70_SOAK_MINUTES=60 MOE_SOAK_MINUTES=60 HIGHMEM_MINUTES=60 \
  bash 04_bench.sh --phase soak
```

### 2.3 node-b 执行（与 node-a 的双机命令逐条对应、同时启动）

```bash
# node-b：只需自检 / 体检 / 部署（2.1 已执行），然后逐条对应下面的双机命令

# ★ 与 node-a 的 dx8b 同时执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  bash 04_bench.sh --phase dx8b

# ★ 与 node-a 的 nccl 同时执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  bash 04_bench.sh --phase nccl

# ★ 与 node-a 的 d70b 同时执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  bash 04_bench.sh --phase d70b

# ★ 与 node-a 的 moe 同时执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  bash 04_bench.sh --phase moe

# ★ 与 node-a 的 moe-dp 同时执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  bash 04_bench.sh --phase moe-dp

# ★ limit 阶段已删除（见 §5.1 变更说明）：显存边界改由 --phase mfu + highmem 覆盖

# ★ 与 node-a 的 soak 同时执行
#   SOAK_MODEL / SOAK_MINUTES / DENSE70_SOAK_MINUTES / MOE_SOAK_MINUTES / HIGHMEM_* 与 node-a 逐字一致
NNODES=2 NODE_RANK=1 MASTER_ADDR=$NODE_A_IP \
  NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA PEER_ADDR=$NODE_A_IP RUN_ID=$RUN_ID \
  SOAK_MODEL=all SOAK_MINUTES=120 DENSE70_SOAK_MINUTES=60 MOE_SOAK_MINUTES=60 HIGHMEM_MINUTES=60 \
  bash 04_bench.sh --phase soak
```

> `PEER_ADDR` = 对端管理 IP，**双机两台都要给**（node-a 给 node-b、node-b 给 node-a）：
> ① 逐口门禁用它 BatchMode ssh 拿「对端逐口 HCA IP 表」，同名 HCA（mlx5_bond_2..9）一一对应，
> 拿不到表再退回「已学偏移 → ±10 候选」推导（11-07 实测：只按 ±10 推导在低位侧会整阶段 RDMA_GATE_FAIL）；
> ② node-b 回传日志：有 ssh/rsync 时自动把 node-b 的整个 run 目录
> `rsync` 到 node-a 同名的 `runs/<RUN_ID>/node-b/`，失败只打印命令、不阻断测试。

### 2.4 只在 node-a 执行的收尾

```bash
cd /root/llm-bench
RID=$(basename "$(readlink -f runs/latest)")

# 拉回 node-b 的环境体检结果（node-b 自己也有同名 run 目录）
scp node-b:/root/llm-bench/runs/$RID/02-precheck/<node-b-hostname>-env.json \
    /root/llm-bench/runs/$RID/node-b-env.json

# 若 node-b 日志没有自动回传，手工把整个 run 目录拉进 node-b/ 子目录
rsync -a node-b:/root/llm-bench/runs/$RID/ /root/llm-bench/runs/$RID/node-b/

# 汇总报告（读 node-a 的整次 run；node-b/ 是旁证，不参与训练表）
bash 04_bench.sh --phase report

# 双机环境一致性对比（node-a 自己的 <hostname>-env.json vs node-b-env.json）
python3 05_report.py --compare \
  /root/llm-bench/runs/$RID/02-precheck/<node-a-hostname>-env.json \
  /root/llm-bench/runs/$RID/node-b-env.json
```

> `soak` 在双机下用 `NNODES=2`；`io` / `fault` 脚本内部固定单机（`--nnodes=1`），
> 因此只在 node-a 执行，且不要和双机阶段并发。

### 2.5 本轮新增阶段的约束

**参数自检（argscheck）**：`moe` / `moe-dp` 在拼参数前先读
`<run>/args-check.json`；不存在就自动先跑一次容器内 `pretrain_gpt.py --help`（落盘 `args-help.txt`）。
- 校验 13 个 flag：`multi-latent-attention`、`num-experts`、`moe-ffn-hidden-size`、`moe-router-topk`、
  `moe-router-enable-expert-bias`、`moe-shared-expert-intermediate-size`、`expert-model-parallel-size`、
  `moe-token-dispatcher-type`、`moe-aux-loss-coeff`、`mtp-num-layers`、`fp8-recipe`、
  `no-create-attention-mask-in-dataloader`、`enable-experimental`。
- 存在性按 `--help` 原文精确匹配（`--flag` 与 `--flag.sub` 区分）；缺失的从候选参数剔除并 WARN。
- 关键 4 项（`num-experts`、`multi-latent-attention`、`expert-model-parallel-size`、
  `moe-token-dispatcher-type`）任一缺失 → 该 MoE 阶段直接 **SKIP** 并打印原因，不带错参数硬跑。
- **MLA 依赖门禁**：除上述 4 项，MLA 还要求 `q-lora-rank`、`kv-lora-rank`、`qk-head-dim`、
  `qk-pos-emb-head-dim`、`v-head-dim` 与 `--enable-experimental` 全部存在（按 `args-help.txt` 原文匹配）；
  缺任一项 → 该档 **SKIP**（不是 FAILED），原因写入 `moe.jsonl` 的 `note`，退出码仍为 0。
- 本机 `args-help.txt` 第 416 行出现 `[--enable-experimental] [--spec [SPEC ...]]`，第 1886 行为其独立说明；
  `--spec` 只是同一用法行里的相邻参数（spec 解码），与 MLA 无关，不加入门禁。
- 阶段退出码：`moe` / `moe-dp` 任一档 FAILED → 非 0；全 SKIP/全 OK → 0，`dual_stop` 语义不变。
- dispatcher 取值也查 `--help` choices：`alltoall` / `allgather` 不支持的记 SKIP。

**FP8 限制（11-10 P0 修订）**：本方案用 **`--fp8-format=e4m3` + `--fp8-recipe=delayed` 两个参数一起传**（镜像 25.01-py3 / CUDA 12.8 / TE 2.1）。

`--fp8-recipe` 只选缩放策略，**单独传不会把 `args.fp8` 从 `None` 打开**：11-10 现场 S4 只传了 `delayed`，日志里 `fp8 = None`、`params_dtype=bfloat16`、loss / grad norm 与 BF16 档逐位相同、吞吐只差 0.26% —— 是「名义 FP8」。`--fp8-format` 才是 TE 进入 `fp8_autocast`、真正启用 FP8 GEMM 的入口。判定顺序：

1. `ensure_args_check` + `flag_in_help` 读 `--help` 原文；
2. 有 `--fp8-format` → 传 `--fp8-format=e4m3`；有 `--fp8-recipe` 再附 `--fp8-recipe=delayed`；
3. **只有 recipe、没有 format → 整组 SKIP**，绝不跑假 FP8 档；BF16 基准档照跑；
4. FP8 档跑完自检日志的 `args.fp8`：仍为 `None` 则打印 `[FAIL]`、写 `.fp8_not_active` 标记并记入 `index.md` 口径说明（不中断整轮，避免 `lc_run_stage_done` 缺失）。

`delayed` 要求 FP8 参数带 amax history，且每步重算缩放因子会带来单步尖峰；`tensorwise` / `blockwise` / `mxfp8` 不在本轮 matrix 内，别混跑比较。

**`OOM_BOUNDARY` 的语义**：只出现在「预期间界」的档位上（`ALLOW_OOM=1` + `OOM_AS_BOUNDARY=1`，
目前只有 highmem 用）。它不 dual_stop、不判 FAILED、不影响退出码；只有 NCCL 错误 / hang / 进程崩溃才 FAILED。
highmem 里出现 `OOM_BOUNDARY` 的物理含义是「MFU 峰值档撑不住 60 分钟」——MFU 只跑 40 步，
长稳的碎片更高，这本身可以作为「该配置的显存上限」结论。


## 3. 环境变量总表

> **训练矩阵的并行/批次参数**集中在 `bench/00-config.sh` 的「并行与批次」一节
> （`S8B_*` / `DX8B_*` / `D70B_*`），默认值就是原先写死在 `phase_*` 调用里的那些数字。
> 改之前建议先干跑看网格：`PLAN_ONLY=1 bash 04_bench.sh --phase s8b`（不启动容器，几秒出结果）。
> 长稳/高占用/MFU 各有自己的旋钮（`SOAK_*` / `HIGHMEM_*` / `MFU_*`），不在这张表里。

| 变量 | 默认值 | 何时必须设置 | 影响阶段 |
|---|---|---|---|
| `ENGINE` | `docker`（不可用回落 `nerdctl`） | 想强制走某个引擎时 | 所有脚本 / `lib.sh` |
| `CONTAINERD_NAMESPACE` | `default` | nerdctl 路径要换 namespace 时；禁止 `k8s.io`/`moby` | 所有脚本（仅 nerdctl 生效） |
| `ROOT` | `/root/llm-bench` | 脚本不在该目录时；两台必须一致 | `02/03/04` |
| `RUN_ID` | 空（自动） | 显式续跑某次 run / 双机强制同名 / 指定自定义目录名（禁止 `/` 与 `..`） | 所有脚本 |
| `RUN_PHASES` | `full` | 想改 run 名末尾的 phases 标签（如 `soak` / `hw`） | `preflight.sh` 新建 run 时 |
| `OUT_DIR` | 本次 run 的阶段目录 | 把 `02_precheck` 产物放别处时 | `02_precheck.sh` |
| `IMAGE` | 自动挑 `pytorch` 镜像 | 镜像名不含 pytorch，或想指定版本时 | `preflight` / `02` / `03` / `04` |
| `TAR_IMG` | `$ROOT/pytorch*.tar $ROOT/pytorch*.tar.gz` | 归档名/位置不同时 | `03_deploy.sh` |
| `MEGATRON_PKG` | `$ROOT/Megatron-LM*.tar.gz` / `.tgz` / `.zip` | 包名/位置不同时 | `03_deploy.sh` |
| `NPROC` | `8` | 不要改；8 卡断言按 8 张判 | `04_bench.sh` |
| `NNODES` | `1` | 双机所有阶段设为 `2` | `04_bench.sh` |
| `NODE_RANK` | `0` | 双机：node-a=`0`、node-b=`1` | `04_bench.sh` |
| `MASTER_ADDR` | `127.0.0.1` | 双机必须设为 node-a 可达 IP | `04_bench.sh` |
| `MASTER_PORT` | `29500` | 端口冲突时改；两台必须相同 | `04_bench.sh` |
| `STEPS` | `15` | 想拉长每档步数时；`soak` 会内部覆盖为 100000 | `04_bench.sh` |
| `NCCL_IF` | 空 | 双机必须设（如 `bond0`） | `04_bench.sh` 双机阶段 |
| `NCCL_HCA` | 空（自动探测 GID） | 双机必须设（本机 RoCE：`mlx5_bond_2..9` 逗号列表） | `04_bench.sh` 双机阶段 |
| `NCCL_GID` | 空，自动探测 RoCEv2 | 自动探测不对时显式指定 | `04_bench.sh` |
| `NCCL_DEBUG` | `INFO` | 排障时改 `DEBUG` | `04_bench.sh` |
| `PEER_ADDR` | 空 | 对端管理 IP；双机**两台都设**（node-a=node-b IP，node-b=node-a IP）。逐口门禁 ssh 对端逐口表 + node-b 日志回传 | `04_bench.sh` |
| `PEER_ROCE_BASE` | 空（回落 `PEER_ADDR`） | 对端任一可 ssh 的 IP，显式覆盖逐口门禁/GID 对照的 ssh 目标 | `04_bench.sh` |
| `PROBE_SECS` | `5` | 想加长/缩短稳态微基准时 | `hw` / `coll` / `nccl` / `dx8b` |
| `MFU_STEPS` | `40` | 峰值 MFU 每档步数；报告侧丢前 5 步，<40 步结论不稳 | `mfu` |
| `MFU_REPEAT` | `3` | 前 3 名各复测几次；=1 时结论标 UNCERTIFIED | `mfu` |
| `MFU_CV_MAX` | `2.0` | 复测 CV 上限（%），超限的档不进峰值 | `mfu` |
| `MFU_GBS_LIST` | `32,64,128,256` | A 段的 GBS 扫描点（只保留能被 mbs×DP 整除的） | `mfu` |
| `MFU_TP_LIST` | `1,4,8` | 单机 A 段的 TP 扫描点（双机固定 1,2,4,8 + PP2 对照） | `mfu` |
| `MFU_MBS_LIST` / `MFU_ACCUM_LIST` | `1,2,4,8` / `2,4` | B 段：GBS 由 `mbs×DP×accum` 反推 | `mfu` |
| `MFU_SEQ_LIST` | `2048,4096,8192` | C 段 seq 扫描点（该档已被 B 段跑过的会跳过） | `mfu` |
| `MFU_MAX_GBS` | `512` | B 段 GBS 上限，防止扫出荒唐配置 | `mfu` |
| `MFU_MAX_RUNS` | `48` | A 段最多跑几档（双机 auto 是 25 档，一般不用改） | `mfu` |
| `MFU_PLAN_ONLY` | `0` | `1` = 只打印计划不跑，用来估算耗时 | `mfu` |
| `MFU_MODELS` | `auto` | `auto` = 单机 8B / 双机 8B,70B；也可写 `8B` 或 `8B,70B` | `mfu` |
| `MFU_CLOCKS` / `MFU_CLOCKS_SECS` | `1` / `2` | 时钟/降频取证开关与采样间隔；没有 nvidia-smi 时自动跳过 | `mfu` |
| `COLL_SIZES` | `8,65536,1048576,67108864,1073741824,8589934592` | 想裁剪 collective 消息大小时 | `coll` / `nccl` / `dx8b` |
| `COLL_MAX_MSG` | `1073741824` | 跨机总传输量上限，按 `size×NNODES` 过滤（allgather 输出放大 nodes 倍；双机默认剔除 1G/8G 档——无 PFC 的 RoCE 上大突发会丢包卡死）；`0` = 全量放开 | `coll` / `nccl` |
| `MOE_GBS` | 空（单机 8 / 双机 16） | 想让 MoE 档位 GBS 不做默认值时 | `moe` / `moe-dp` |
| `MOE_MLA` | `auto` | `auto`=有 `--enable-experimental` 才跑 MLA；`1`=强制（缺依赖则 SKIP）；`0`=主动跳过 MLA。本机已确认 flag 存在，按方案 A 保留 MLA，不做 GQA 替换 | `moe` / `moe-dp` / `soak`(moe) |
| `PLAN_ONLY` | `0` | `1` = 只打印每档的 tag/TP/PP/DP/mbs/seq/GBS/accum，不启动容器 | 所有训练阶段 |
| `S8B_TP` / `S8B_PP` | `8` / `1` | S1/S3/S4 的固定并行度 | `s8b` |
| `S8B_MBS_LIST` | `1 2 4` | S1 扫哪些 mbs；GBS 由 `mbs × S8B_GBS_MULT` 得出 | `s8b` |
| `S8B_GBS_MULT` | `8` | S1 的 GBS 倍数（= DP；单机 8 卡即 8） | `s8b` |
| `S8B_SEQ` | `4096` | S1/S4 的序列长度 | `s8b` |
| `S8B_TOPO_LIST` | `1:1 4:1 8:1` | S2 扫的 `tp:pp` 对；GBS 固定才可比 | `s8b` |
| `S8B_TOPO_MBS` / `S8B_TOPO_SEQ` / `S8B_TOPO_GBS` | `2` / `4096` / `32` | S2 每档的固定 mbs/seq/GBS | `s8b` |
| `S8B_SEQ_LIST` | `2048 8192 32768 131072` | S3 扫哪些 seq（单机唯一逼显存上限的方向） | `s8b` |
| `S8B_SEQ_MBS` / `S8B_SEQ_GBS` | `1` / `8` | S3 每档的固定 mbs/GBS | `s8b` |
| `S8B_FP8_MBS` / `S8B_FP8_SEQ` / `S8B_FP8_GBS` | `4` / `4096` / `8` | S4 BF16/FP8 对照档的参数 | `s8b` |
| `DX8B_TP` / `DX8B_PP` | `8` / `1` | D1 双机 8B 的并行度 | `dx8b` |
| `DX8B_MBS` / `DX8B_SEQ` / `DX8B_GBS` | `2` / `4096` / `32` | D1 的批次参数 | `dx8b` |
| `D70B_TP` / `D70B_PP` | `8` / `2` | D2/D3 双机 70B 的并行度 | `d70b` |
| `D70B_MBS` / `D70B_SEQ` | `1` / `2048` | D2 的固定 mbs/seq | `d70b` |
| `D70B_BUBBLE_GBS_LIST` | `4 16` | D2 扫哪些 GBS（看 PP 气泡） | `d70b` |
| `D70B_SEQ_LIST` / `D70B_SEQ_GBS` | `4096 8192` / `12` | D3 扫哪些 seq、固定 GBS | `d70b` |
| `FORCE` | `0` | `1` = 忽略 `.ok_` 哨兵重跑已完成档位 | `04_bench.sh` 训练阶段 |
| `OPTIONAL` | `0` | `1` = 跑精简矩阵中降级的档位（单机 `coll`、`io` 70B） | `04_bench.sh` |
| `ALLOW_BUSY` | `0` | `1` = GPU 有占用也强跑（结果不可验收） | `04_bench.sh` |
| `TELEMETRY` | `0` | `1` = 每档起 `nvidia-smi dmon`；`hw` / `soak` 内部强制开 | `04_bench.sh` |
| `SOAK_PROGRESS` | `1` | `0` = 关闭长稳进度显示 | `soak` |
| `PROGRESS_SECS` | `60` | 进度刷新间隔（秒）；非法值 / 0 自动回落到 60 | `soak` |
| `SOAK_MODEL` | `dense` | `dense` / `dense70` / `moe` / `highmem` 的逗号组合；`both`=dense+moe（旧语义）；`all`=全部适用项（双机含 dense70）；各条长稳不可互相替代 | `soak` |
| `SOAK_MINUTES` | `120` | 想改 8B Dense 长稳时长时 | `soak` |
| `SOAK_MIN_ITERS` | `100` | 想改长稳默认最少步数时（8B Dense 默认，MoE/70B/highmem 未单独指定时沿用）；1 分钟冒烟应设为 `1` | `soak` |
| `DENSE70_SOAK_MINUTES` | `60` | 想改双机 70B Dense 长稳时长时 | `soak`（`dense70` / 双机 `all`） |
| `DENSE70_SOAK_MIN_ITERS` | 空（= `SOAK_MIN_ITERS`） | 70B 单独要求步数时；由全局末级 rank（node-b）判定 | `soak`（`dense70` / 双机 `all`） |
| `MOE_SOAK_MINUTES` | 空（双机 60 / 单机 30） | 想改 MoE 长稳时长时 | `soak`（`SOAK_MODEL=moe`/`both`/`all`） |
| `MOE_SOAK_MIN_ITERS` | 空（= `SOAK_MIN_ITERS`） | MoE 单独要求步数时 | `soak`（`SOAK_MODEL=moe`/`both`/`all`） |
| `EXPERT_SAMPLE_SECS` | `300` | 想改专家负载采样间隔时 | `soak`（MoE 档） |
| `MOE_LOAD_PROBE` | `1`（MoE 档自动设） | 想关掉专家负载探针时设 `0`；关掉后 `expert-load-timeseries.csv` 全 null | `moe`、`soak`（MoE 档） |
| `MOE_LOAD_LOG_EVERY` | `200` | 想改 `[expert load]` 打印频率（每 N 次 routing 一次）；短测步数少时调小 | `moe`、`soak`（MoE 档） |
| `HIGHMEM_MINUTES` | `60` | 想改显存高占用长稳时长时（标定耗时另计，约 3~5 分钟）。**<10 分钟按冒烟处理**：不参与 75~92% 判定、不写 `.ok_highmem`、步数门槛降到 1，`status=SMOKE` | `soak`（highmem） |
| `HIGHMEM_HEALTH` | `1` | `0` = 不采跑前/跑后硬件快照 | `soak`（highmem） |
| `HIGHMEM_GROWTH_MAX` | `5` | 任一卡显存增长 ≥ 此百分比 → `WARN_LEAK`（疑似泄漏） | `soak`（highmem） |
| `HIGHMEM_FRAG_DIFF` | `3` | 峰值在前 20% 且末尾仍 ≥ 峰值−此值 → `WARN_FRAG`（疑似碎片） | `soak`（highmem） |
| `MEM_SAMPLE_SECS` | `5` | 显存逐卡采样间隔（`memused-*.csv`） | `soak`（highmem） |
| `DATA_PATH` | `$ROOT/data` | 真实数据集不在默认位置时 | `io` |

## 4. 结果目录结构

每次运行一个目录，所有阶段产物都在里面，不再散落在 `logs/`：

```
/root/llm-bench/runs/<RUN_ID>/
  00-run-info.txt          # 引擎 / namespace / 镜像 digest / 驱动 / 节点数 / 起止时间 / 关键环境变量
  index.md                 # 每次运行的文件清单：阶段、耗时、结果、产出（每个阶段结束追加一行）
  args-help.txt            # pretrain_gpt.py --help 原文（参数自检）
  args-check.json          # 12 个 flag 的存在性、status（OK/DEGRADED/ERROR）、missing / critical_missing
  00-argscheck/            # args-help.txt、args-check.json 的副本（阶段产物）
  01-preflight/            # engine.log
  02-precheck/<host>-env.json、<host>-topo.txt
  03-deploy/               # selfcheck.txt、env.txt、megatron_args.txt
  04-hw/                   # hw.jsonl、hw.txt、dmon.csv
  05-coll/                 # coll.jsonl、coll.txt
  06-train-s8b/            # w8_8B_*.log、results.jsonl、dmon-*.csv
  06-train-dx8b/           # 双机 8B：dx8b 日志、results.jsonl
  06-train-nccl/           # 双机 collective 直测
  06-train-d70b/           # 双机 70B：results.jsonl、*.log
  06-train-moe/            # MoE：moe.jsonl、w<world>_MOE_*.log、dmon-*.csv
  06-train-moe-dp/         # 双机 EP8×DP2：moe.jsonl（topology=ep8_dp2）
  06-train-mfu/            # 峰值 MFU：mfu.jsonl、mfu-peak.json、clocks-*.csv（highmem 的档位来源）
  07-io/                   # io.jsonl、io-dataloader.log
  08-fault/                # fault.jsonl、fault-*.log
  09-soak/                 # 8B Dense：results.jsonl、health-before/after/delta.json、dmon-*.csv
                           # 70B Dense：results.jsonl、dense70-health-before/after/delta.json、
                           #            dmon-w16_70B_soak_*.csv、soak-progress.log
                           # MoE  ：moe-soak.jsonl、expert-load-timeseries.csv、
                           #        moe-health-before/after/delta.json、dmon-moe-soak-*.csv
                           # highmem：highmem.jsonl、memused-w<N>_highmem.csv、
                           #        highmem-health-before/after/delta.json、dmon-highmem-*.csv
  10-report/               # report.txt、report.md（内容相同，md 便于贴评审）
  node-b/                  # 双机时 node-b 回传的同结构目录
```

`runs/latest` 指向最近一次 run（不支持符号链接时退化为 `runs/latest.path` 文本指针）；
`$ROOT/logs` 是 → `runs/latest` 的兼容链接，因此旧的 `logs/xxx` 写法仍能读到最近一次。

### 4.1 回看某次运行

```bash
cd /root/llm-bench

# 本次 run 的概览：目录、索引、环境
cat runs/latest/00-run-info.txt
cat runs/latest/index.md

# 列所有历史运行（按时间）
ls -1dt runs/2026-* runs/2* | grep -v latest

# 用某次运行显式续跑（例如补跑报告，不新建目录）
RUN_ID=<历史 run 名> bash 04_bench.sh --phase report

# 对某次运行重新生成报告
python3 05_report.py --logs runs/<RUN_ID>
```

### 4.2 对比两次运行

```bash
cd /root/llm-bench
# 环境一致性（先比指纹，再比数字）
python3 05_report.py --compare \
  runs/<RUN_A>/02-precheck/<host>-env.json \
  runs/<RUN_B>/02-precheck/<host>-env.json

# 训练结果对比：直接 diff 两次的 results.jsonl（tag 相同才可比）
diff runs/<RUN_A>/06-train-s8b/results.jsonl runs/<RUN_B>/06-train-s8b/results.jsonl

# 报告对比：两次各生成一份，再 diff
python3 05_report.py --logs runs/<RUN_A> > /tmp/A.txt
python3 05_report.py --logs runs/<RUN_B> > /tmp/B.txt
diff -u /tmp/A.txt /tmp/B.txt
```

- 训练结果在各自阶段的 `results.jsonl`，每行一个档位，`status` 为 `OK` / `OOM` / `PARTIAL` / `FAILED`；
  `OK` 才写 `.ok_<tag>` 哨兵，失败写 `.fail_<tag>`，重跑不会误跳过。报告会合并所有训练阶段的 `results.jsonl`。
- 报告命令：

```bash
cd /root/llm-bench
bash 04_bench.sh --phase report      # 终端输出 + 10-report/report.txt + report.md
```

报告包含：环境指纹（`engine` / `namespace` / `image` / `digest` / 驱动 / torch / CUDA / NCCL）、
硬件微基准表、collective busbw/延迟表、checkpoint IO 表、故障注入表、长稳硬件增量、
训练验收表（D1 跨机比值判据 `≤0.6` 通过）。
本次为精简矩阵（不含已删除的 mbs8/16、tp2、seq16K/64K、D2 gb12、D4、EP4×TP2、moe-dp allgather、dx8b probe/gb64），完整档位见 `变更清单.md`。

双机环境差异用第 2.4 节的 `--compare`；有 `FAIL` 级差异时退出码非 0，先修再测。

## 5. 常见失败处置

| 现象 | 判据/位置 | 处置 |
|---|---|---|
| 引擎起不来 | `preflight.sh` / `03_deploy.sh` 报 `docker info` Runtimes 不含 nvidia | docker：`nvidia-ctk runtime configure --runtime=docker && systemctl restart docker`；nerdctl：检查 containerd 与 `/etc/containerd/config.toml` 的 nvidia runtime |
| toolkit 版本 <1.14 | `preflight.sh` / `03_deploy.sh` 打印 `[WARN] nvidia-container-cli ...` | **警告，可继续**：不影响 rc、不阻断部署；容器内 8 卡自检能过就不用处理。仅当 `ENVF gpus≠8` 时，再升 `libnvidia-container1/-tools`、`nvidia-container-toolkit-base`、`nvidia-container-toolkit` 四包到 1.17.x 后重试 |
| 容器内看不到 8 卡 | `preflight` 第 7 步 / `03` 自检 `ENVF gpus≠8` | 硬门禁（仍 `exit 1`）。先分两类：宿主 `nvidia-smi -L` 有 8 张而容器内没有 → 运行时/toolkit 注入问题（按上一条升 1.17.x 四包）；宿主本身不足 8 张 → 驱动/设备/掉卡/MIG/`CUDA_VISIBLE_DEVICES`；`nvidia-container-cli -k -d /dev/tty info` 看注入报错 |
| NCCL 选错网卡 | `06-train-nccl/nccl.txt` 没有 `NET/IB`，或 `NET/Socket` | 检查 `NCCL_IF` / `NCCL_HCA`；`show_gids | grep v2` 核对 `NCCL_GID`；`ibv_devinfo` 看 `state` / `link_layer` |
| 逐口门禁 RDMA_GATE_FAIL | `06-train-*/rdma-gate.json` 的 `per_hca[].verdict=drop`，原因「对端候选 … 均 ping 不通」或「无法 … 推导对端 IP」 | ① 确认两台 `PEER_ADDR` 都已设（node-a=node-b IP、node-b=node-a IP）且 BatchMode ssh 互通（脚本会先 ssh 拿对端逐口 HCA IP 表）；② ssh 不通时按「已学偏移→±10」推导，核对两台 RoCE 口末位是否恰好差一个固定偏移（11-07 实测 .2-.9 ↔ .12-.19）；③ RoCE 口之间手工 `ping` 验证（ICMP 被禁也会 drop，需放通或改用 `PEER_ROCE_BASE`+ssh 表）；④ 修复后 `FORCE=1` 只重跑 `--phase nccl` 探针即可复验 |
| 逐口全 ok 但探针秒败（无 `rdma-gate.txt`） | `rdma-gate.json` 的 `per_hca` 全 `ok`、`busbw_gbs:null`、阶段耗时仅几秒、目录里没有 `rdma-gate.txt`；`probe_error` 有归因 | ① 看 `probe_error`：`missing_probe`/`no_output` = 本机没有 `probes/hw_probe.py` 或拷贝失败 —— **把整个 `probes/` 目录同步到两台**（11-08 两台即此因，且报错曾被命令替换吞掉，已改为走 stderr 必现在终端）；② 有 `rdma-gate.txt` 时看末尾：引擎报错（nerdctl `--device`/镜像）、rendezvous（对端没同时起）、timeout 137（卡 60s 被杀，两台启动时间差太多） |
| 显存 OOM | `06-train-*/results.jsonl` 中 `status=OOM`；同目录 `*.log` 有 `CUDA out of memory` | 属正常的上界信号：终端会打印关键行 + 静态显存估算（16B/参数，按 tp×pp×dp 分摊）；降 `seq`/`mbs`/`GBS` 或换并行切分；F1 的 70B 单机 OOM 是**预期 PASS** |
| 显存数字不一致 | nvidia-smi 143771 MiB=140.4 GiB，容器内 torch `total_memory`≈139.8 GiB | 差额约 0.6 GiB 是 ECC/驱动预留与 CUDA 上下文：看余量用 nvidia-smi，看 torch 可分配用 `total_memory`；报告已按两套口径注明 |
| 双机 Gloo 挂住 | 日志 `ProcessGroupGloo`/`loopback`/`connectFullMesh`，或只挂 dp>1 的档 | 确认已自动带 `--disable-gloo-process-groups`（`megatron_args.txt` 可查）；仍挂则查容器内 `/etc/hosts`、`NCCL_SOCKET_IFNAME` |
| d70b 报 world 不足 | `d70b` 启动即打印「并行度/机器数配置错误」 | 不是硬件问题：`d70b` 需 `NNODES=2`（world=16）；world=8 跑 70B 必然 OOM，报告不可归因显卡 |
| 基线不可比 | 报告表里 `★基线` 档，或 `baseline.txt` | 本轮基线与历史 docker 基线（DO/overlap/recompute 门控不同）**不可横比**；跨轮只比同一 `baseline.txt` 配置 |
| 长稳 fail（Dense） | `09-soak/` 里 `.fail_w8_8B_soak_*`、`status=PARTIAL`、`health-delta.json` `status=FAIL` | 看 `dmon-w8_8B_soak_*.csv` 是否降频/贴顶；查 ECC/NVLink/Xid 增量；修好后直接重跑（失败不会写 `.ok_`）；已完成的档位用 `FORCE=1` 重跑 |
| 长稳 fail（MoE） | `09-soak/` 里 `.fail_moe_soak`、`moe-soak.jsonl` 的 `status` 非 OK、`moe-health-delta.json` `status=FAIL` | 先看 `expert-load-timeseries.csv` 的 max/mean 与 `note`（null=日志里解析不到）；再看 `dmon-moe-soak-*.csv` 温度/功耗；两条长稳哨兵独立，重跑 MoE 不会动 Dense 结论 |
| 长稳异常（highmem） | `highmem.jsonl` 的 `status=WARN_LEAK/WARN_FRAG`、`.fail_highmem`、`highmem-health-delta.json` `status=FAIL` | `WARN_LEAK`（增长 ≥5%）= 疑似泄漏，对照 `memused-*.csv` 看是否单调上升，这条不能忽略；`WARN_FRAG`（峰值在前 20% 且末尾贴峰）= 疑似碎片，只影响长时间稳定性判断；OOM 属 `OOM_BOUNDARY` 不计失败，说明「MFU 峰值档撑不住 60 分钟」，可直接作为显存上限结论 |
| 跑不起来（highmem） | 终端提示「需要 MFU 阶段的峰值档，但没有可用的 mfu.jsonl」 | 先跑 `bash 04_bench.sh --phase mfu`（本 run 内），highmem 会复用它的峰值档；limit 阶段已删除，不再需要边界数据 |
| retired_pages 报字段不存在 | H200 上 `Field "retired_pages.correctable" is not a valid field` | 三态判定：`SKIP（字段不支持）`≠坏卡，不影响 preflight rc；需要确证时手动 `dcgmi diag -r 1`，或 `DCGM_DIAG=1 bash preflight.sh` 让其执行并计入 PASS/FAIL；只有 `pending=Yes` 或 Remapped Rows 非 0 才 FAIL |
| fault 用例全 INVALID | `08-fault/fault.jsonl` 出现 `status=INVALID`；三个日志各含 `AssertionError: ... CUDA_DEVICE_MAX_CONNECTIONS to 1` | `fault_train` 漏传该环境变量（TP=8 时 Megatron 强制断言），8 个 rank 启动即崩；11-10 已修，重跑前核对 `fault_train` 的 `-e` 行与 `run_train` 一致。`INVALID` 已计入 `lib.sh` 阶段汇总的「未过」，不会再被当成 PASS |
| 专家负载全 null | `expert-load-timeseries.csv` 的 max/mean 列为 `null`，`moe-soak.jsonl` 的 `imbalance_*` 为 null | 先 `grep -F '[expert load probe]' <moe 日志>`：有 `installed` 但无 `[expert load]` → 步数不够或采样间隔太大（调小 `MOE_LOAD_LOG_EVERY`）；有 `install failed` / `disabled` → Megatron API 不匹配，按 `warn` 原因核对 `TopKRouter.routing` 返回值；完全没有探针行 → 确认 `probes/moe_load_probe.py` 已同步、`sitecustomize.py` 已落 LOGDIR、`MOE_LOAD_PROBE=1` 已透传进容器 |
| hw 带宽数值异常 | HBM 上千 TB/s、PCIe 上千 GB/s、collective 只有几 GB/s | 11-10 前的 `hw_probe.py` 有两处相反的 iters 错误（HBM/L2/PCIe 多乘、busbw 少除）。已修；重跑 `--phase hw` 后应见 HBM read ≈3~4 TB/s、PCIe h2d ≈30 GB/s、大消息 busbw ≈330~370 GB/s，并可用新增的 `iters`/`secs`/`algbw_gbs` 复核 |
| FP8 SKIP / 假 FP8 | `04-hw/hw.jsonl` 的 FP8 记录 `status=SKIP`；`s8b` 打印「容器 --help 里没有 --fp8-format（据 …/args-help.txt）」；或阶段目录出现 `.fp8_not_active` | 提示必须带 `args-help.txt` 路径（老脚本「（见 ）」括号为空 = `$MA` 为空、假 SKIP）。**`.fp8_not_active` 表示 `args.fp8=None`，该档加速比不可用**，核对 `--fp8-format` 是否拼进命令行、TE 是否接受；`hw.jsonl` 的 FP8 SKIP 现在会带逐签名失败原因，按 `note` 定位 `_scaled_mm` 的 B 转置 / scale 维度问题 |
| 参数被误裁 / 误判不存在 | 启动日志出现 `· 裁掉（容器不支持）: --xxx`，或 S4/soak 报某个 flag 缺失 | 先看该行前后有没有「暂无 Megatron --help 原文」提示。`$MA` 为空时任何 grep 都恒失败；跑一次 `--phase argscheck` 生成 `args-help.txt` 后复核 `grep -E '^ *--<flag>' <run>/args-help.txt` |
| data pipeline SKIP | `07-io/io.jsonl` 有 `{"kind":"dataloader","status":"SKIP"}` | 放 `$ROOT/data/*.bin` + `*.idx`，或 `DATA_PATH=<目录>` 指向真实数据；mock-data 不含 dataloader |
| run 磁盘不足 | `preflight` 报「run 磁盘余量 < 10 GB」 | 删旧 run 目录（`du -sh runs/*` 找大的）；或把 `ROOT` 放到大容量盘（两台都要改） |
| 磁盘不足 | `preflight` / `02` 报镜像磁盘 <60G | `lc_rmi` 删旧镜像；或把 `ROOT` 放到大容量盘（两台都要改） |
| 镜像 load 失败 | `03` 报导入失败 | docker：`journalctl -u docker -n 50`；nerdctl：`journalctl -u containerd -n 50`；确认 `load` 与 `run` 同一 namespace |

## 6. 预计耗时（8 卡，`PROBE_SECS=5`，`SOAK_MINUTES=120`）

| 阶段 | 粗估 | 说明 |
|---|---|---|
| `preflight.sh` | 2–5 分钟 | 含容器内 8 卡实测 |
| `02_precheck.sh` | <1 分钟 | 只读采集 |
| `03_deploy.sh` | 5–10 分钟 | 已复用镜像时更快；首次 load 约 30GB |
| `--phase hw` | 5–10 分钟 | 稳态 ≥5s × GEMM/HBM/L2/PCIe/collective（已含四类 collective） |
| `--phase coll` | 约 6 分钟（可选） | 仅 `OPTIONAL=1`；单机已并入 `hw` |
| `--phase s8b` | 20–30 分钟 | 12 档 × 15 步；长序列档最慢 |
| `--phase dx8b` | 8–12 分钟 | 跨机训练对照 1 档（GBS=32） |
| `--phase nccl` | 5–10 分钟 | 仅 collective 直测 |
| `--phase d70b` | 20–35 分钟 | 4 档；seq8192 最慢 |
| `--phase soak`（dense） | 约 130 分钟 | 120 分钟训练 + 前后健康采集（默认 `SOAK_MODEL=dense`，行为不变） |
| `--phase soak`（dense70） | 约 65 分钟 | 双机 60 分钟训练 + 前后健康采集；单机是配置错误 |
| `--phase soak`（moe） | 约 40 分钟 | 单机 30 分钟训练 + 采集；双机约 70 分钟（60 分钟训练） |
| `--phase soak`（both） | 单机约 170 分钟 / 双机约 190 分钟 | 先 dense 后 MoE，不含 dense70；双机 dense 120 + MoE 60 ≈ 3 小时 |
| `--phase mfu` | 单机约 75 分钟 / 双机约 2.5~3 小时 | A 段 12~25 档 + B/C/D 约 15~20 档，每档 40 步 + 容器启动；单档约 2~3 分钟（双机 3~4） |
| `--phase soak`（highmem） | 约 62 分钟 | 60 分钟训练 + 前后健康采集；档位来自 mfu.jsonl，不再有标定耗时 |
| `--phase soak`（all） | 单机约 215 分钟 / 双机约 305 分钟 | 单机 dense 120 + MoE 30 + highmem 60 ≈3.5 小时；双机再加 dense70 60，合计 ≈5 小时 |
| `--phase io` | 5–10 分钟 | 默认只跑 8B；70B 需 `OPTIONAL=1`，无真实数据时 dataloader 直接 SKIP |
| `--phase fault` | 10–20 分钟 | F3 最长等 5 分钟看门狗 |
| `--phase report` | <1 分钟 | 纯解析 |
| **单机全链路（不含 soak）** | **约 140 分钟** | 精简矩阵：hw + s8b + mfu + moe + io + fault（mfu 约 75 分钟是大头） |
| **双机全链路（不含 soak）** | **约 3 小时** | nccl + dx8b + d70b + moe + moe-dp + mfu（约 2.5 小时），两台同时执行 |
