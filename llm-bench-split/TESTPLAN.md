# H200 单机 / 双机大模型训练压测方案（llm-bench-fix 实施版）

> **适用范围**：`llm-bench-fix/` 下的 `02_precheck.sh`、`03_deploy.sh`、`04_bench.sh`、`05_report.py`、`lib.sh`、`probes/`。
> **与其他文档的分工**：本文件定义「测什么 / 为什么测 / 怎么判 / 什么算完成」；`RUNBOOK.md` 定义「命令怎么敲」；`变更清单.md` 记录相对旧版的修复；`TESTPLAN修订说明.md` 列出口径勘误。
> **依据**：所有阶段、档位、判据均按当前脚本逐项核对；文档与脚本冲突时**以脚本为准**。原 `llm-bench-docker/TESTPLAN.md` 不改动，本文件是它的实施替代版，已吸收修订说明中的全部勘误。
> **版本**：2026-11-09。

---

## 0. 测试目标与结论边界

### 0.1 目标

| 编号 | 目标 | 对应阶段 | 交付证据 |
|---|---|---|---|
| T1 | 验收硬件是否适合做基准：无掉卡、无误码、无降频、GPU 独占 | `02_precheck.sh`、长稳健康快照 | `<host>-env.json`、`<host>-topo.txt`、`health-*-delta.json` |
| T2 | 测出算力 / 通信上限：GEMM、HBM、L2、PCIe、collective 与 dense 模型 MFU | `hw`、`coll`、`nccl`、`s8b` | `hw.jsonl`、`coll.jsonl`、`nccl.jsonl`、`results.jsonl` |
| T3 | 验证跨机扩展是否成立：RDMA 门禁、DP 扩展比值、PP 流水线、MoE EP 对照 | `dx8b`、`nccl`、`d70b`、`moe`、`moe-dp` | `rdma-gate.json`、`nccl.txt`、`results.jsonl`、`moe.jsonl` |
| T4 | 验证长稳与容错：持续算力、显存增长、专家路由、故障自愈 | `soak`、`fault`、`io`、`mfu` | `*-soak.jsonl`、`dmon-*.csv`、`fault.jsonl`、`mfu-peak.json` |
| T5 | 结果可复现、可归因、可审计 | 全阶段 | `00-run-info.txt`、`index.md`、`03-deploy/env.txt`、`results.jsonl` 指纹字段、`10-report/report.md` |

### 0.2 能声称 / 不能声称

| 结论 | 成立条件 |
|---|---|
| ✅「训练可用、跨机链路满足 PP 扩展」 | 单机 + 双机训练矩阵跑完，D1 比值达标，`moe` / `d70b` 无非预期失败 |
| ✅「整机达标」 | 还需 `hw`（8 卡全 collective）+ `nccl`（NET/IB）+ 全部适用长稳（含 8B Dense 2 h、双机 70B Dense 60 min）硬件增量 + `io` + `fault` **全部通过**，见 5.5 |
| ❌ 端到端业务吞吐、真实收敛效果 | 全程 `--mock-data` + `NullTokenizer` + 随机初始化，不含数据管线与 checkpoint 落盘开销 |
| ❌ 跨引擎 / 跨镜像 / 跨轮次横比 | 指纹不唯一时比值无意义；本轮基线与历史 docker 基线的 DO / overlap / recompute 门控不同，不可直接比较 |
| ❌ 窗口外的长期可靠性 | 长稳窗口为 8B dense 120 min + 双机 70B dense 60 min + MoE 30/60 min + highmem 60 min |

### 0.3 不在范围内

真实语料与权重加载、VPP（虚拟流水线，脚本未实现）、参数服务器 / 推理性能、跨 major CUDA 兼容镜像（25.10 / CUDA 13.0.2 未验证，不进首轮）、K8s CRI 路径。

---

## 1. 被测对象与基线

### 1.1 环境基线

| 项 | 基线 | 校验位置 |
|---|---|---|
| 硬件 | 2 × 8 卡 H200 SXM；单卡实测 **143771 MiB = 140.4 GiB**；bf16 稠密峰值 **989.4 TFLOPS/卡** | `02_precheck.sh` / `05_report.py` 常量 |
| 机内互联 | NVLink NV18，链路数须为 8 × 18 = **144**，误码合计 0 | `nvidia-smi topo -m` / `nvlink -e` |
| 机间互联 | **RoCE**（`link_layer=Ethernet`），`mlx5_bond_2..9`，400 Gb/s，MTU 4200；不是真 IB | `02_precheck.sh` 的 RDMA 网卡项（WARN） |
| 容器引擎 | docker 优先（`docker version` 可用才采用），否则回落 nerdctl（固定 `CONTAINERD_NAMESPACE=default`） | `lib.sh` / `engine.log` |
| 镜像 | 首发固定 `pytorch:25.01-py3`（CUDA 12.8.0）；25.06 仅作 minor 兼容备选；两台 **digest 必须一致** | `03-deploy/env.txt` |
| 框架 | Megatron-LM **core_v0.13.1** + 容器内 TransformerEngine；`pretrain_gpt.py` | `selfcheck.txt` / `args-help.txt` |
| 数据 | `--mock-data` + `NullTokenizer` + 随机初始化 | 训练命令行 |
| 路径 | `ROOT=/root/llm-bench`，两台必须完全一致；产物在 `$ROOT/runs/<RUN_ID>/` | `lib.sh` |

### 1.2 显存硬账（决定能测什么）

按 `16 B/参数`（bf16 权重 + 梯度 + 优化器状态）估算，TP 分片下每卡静态占用 ≈ 一份完整 bf16 权重：

| 模型 | 参数 | bf16 权重 | 单机 8 卡 | 双机 16 卡 |
|---|---|---|---|---|
| Llama-3.1-8B | 8.03 B | 15.0 GiB | ≈16.1 GiB/卡 ✅ | ≈8.0 GiB/卡 ✅ |
| Llama-3.1-70B | 70.55 B | 131.4 GiB | **≈141.1 GiB/卡 ❌** | ≈70.6 GiB/卡 ✅ |

**结论**：70B 单机零余量、任何 `mbs` / `seq` 都 OOM，所以 70B 训练只出现在双机 `d70b` 与双机 `soak(dense70)`；单机 70B OOM 被设计为 `fault` 阶段 F1 的**预期通过**用例，不是硬件故障。显存一律按 nvidia-smi 的 140.4 GiB 写；容器内 torch `total_memory` ≈139.8 GiB 的 0.6 GiB 差额是 ECC / 驱动预留与 CUDA 上下文，不是坏卡。

---

## 2. 准入条件（G0）

### 2.1 硬门禁

`bash 02_precheck.sh` 两台各跑一次（主机只读 + 容器内实测）。退出码：`0`=全过，`1`=有 WARN（可继续，必须写进报告限制声明），`2`=有 FAIL（**先修，不得进入下一阶段**）。

| 类别 | 检查项 | 通过判据 |
|---|---|---|
| 驱动 | 三处版本一致 | `nvidia-smi` = `/proc/driver/nvidia/version` = `modinfo nvidia` |
| GPU | 数量 / 独占 / 容量 / 型号 | 8 张；无计算进程且单卡显存 <500 MiB；`memory.total` 全为 143771 MiB；型号唯一 |
| 硬件验收 | ECC / retired_pages / NVLink / Xid | ECC Enabled 且 uncorrected=0；pending 无退役页（字段不支持记 SKIP）；链路 144 条且误码 0；`NVRM.*Xid` 计数 0 |
| 状态 | 功耗 / 性能态 / 降频事件 | P0、Persistence Enabled、无实质节流位；功耗上限等于标称 TGP |
| 容器 | 引擎 / runtime / toolkit / 容器内实测 | 引擎可用；docker 的 Runtimes 含 nvidia；容器内 `device_count=8`、memlock=unlimited、shm≈64G |
| 系统 | memlock / rmem_max / 内核冻结 / 磁盘 | memlock unlimited；`rmem_max ≥16 MiB`；镜像存储可用 >60 G；run 归档余量 >10 G |
| 双机附加 | 指纹一致性 | 两台 `03-deploy/env.txt` 去掉 `host=` / `time=` 后 `diff` 为空；`<host>-env.json --compare` 无关键项差异；`NCCL_IF` / `NCCL_HCA` / `MASTER_ADDR` 显式给定 |

WARN 不阻断的项：toolkit <1.14（仅当容器内 8 卡自检失败才升级四包）、RoCE 非 IB、NUMA 多 node、无 `chronyc`、run 磁盘 <10 G、nerdctl 未注册 nvidia runtime。

### 2.2 结果指纹（每档结果必须自带）

`results.jsonl` 每条记录写入 `image` / `digest` / `engine` / `namespace` / `driver` / `torch` / `cuda` / `nccl` / `nodes` / `port` / `iters`。`05_report.py` 打印出现过的指纹组合，**多于一组即判定为跨机 / 跨镜像混跑，比值不可比**。

### 2.3 关键口径：退出码 ≠ 档位通过

多数阶段的 shell 退出码只反映「脚本有没有跑完」。判读规则：

| 阶段 | 以什么为准 |
|---|---|
| `s8b` / `d70b` / `soak` / `mfu` | `06-train-*/results.jsonl` 的 `status` 与同目录 `*.log`（`mfu` 另看 `mfu.jsonl`） |
| `hw` / `coll` / `nccl` / `io` / `fault` | 对应阶段目录的 `*.jsonl` 与 `*.txt` |
| `02` / `03` | 退出码 + `<host>-env.json` / `selfcheck.txt` 内容 |

---

## 3. 执行编排

### 3.1 分层与顺序

| 层 | 阶段 | 目的 | 前置 | 产物目录 |
|---|---|---|---|---|
| G0 | `02_precheck.sh` | 环境门禁，创建本次 run | 物料就位 | `02-precheck/` |
| G1 | `03_deploy.sh` | 镜像 / 解包 / 8 卡自检 / 结果指纹 | G0 无 FAIL | `03-deploy/` |
| G2 | `--phase argscheck` | `--help` 参数表 + 12 个 flag 存在性 | G1 | run 根 + 阶段副本 |
| G3 | `hw` → `coll`(可选) → `nccl`(双机) | 硬件与通信基线 | G2 | `04-hw/` `05-coll/` `06-train-nccl/` |
| G4 | `s8b` / `dx8b` / `d70b` / `moe` / `moe-dp` / `mfu` | 训练与峰值矩阵 | 对应单机 / 双机门禁 | `06-train-*` |
| G5 | `io` / `fault` / `soak` | 存储、容错、长稳 | G4 关键档位通过 | `07-io/` `08-fault/` `09-soak/` |
| G6 | `--phase report` | 汇总验收表 | 全部阶段 | `10-report/` |

`argscheck` 会被后续阶段自动触发；`soak(highmem)` 需要本 run 已有 `mfu.jsonl`（档位来源），没有会直接报错退出。

### 3.2 单机 / 双机 / 收尾

- **单机链路**：`02` → `03` → `argscheck` → `hw` → `s8b` → `mfu` → `moe` → `io` → `fault` → `soak` → `report`。
- **双机链路**：两台各自完成 `02` / `03`（node-b 用 node-a 的 `RUN_ID` 强制同名），之后 `dx8b` / `nccl` / `d70b` / `moe` / `moe-dp` / `mfu` / `soak` **两条命令必须同时启动**，仅 `NODE_RANK` 不同；`SOAK_MODEL`、`SOAK_MINUTES`、`DENSE70_SOAK_*`、`MOE_SOAK_*`、`HIGHMEM_*`、`MFU_*` 两台逐字一致。
- **收尾**：node-b 的 run 目录由 `sync_logs` 回传到 node-a 的 `<run>/node-b/`（无 ssh/rsync 时手工回传），`05_report.py --compare` 比对两台体检结果，最后在 node-a 出报告。
- 单机 `moe` 建议用独立 `RUN_ID`，避免 `w8` 档位混进 `w16` 双机 run 影响口径。

### 3.3 耗时预算（`PROBE_SECS=5`、`STEPS=15`）

| 段 | 粗估 | 段 | 粗估 |
|---|---|---|---|
| `02` + `03` | 7–15 min | `d70b` | 20–35 min |
| `hw` | 5–10 min | `mfu` | 单机 ≈75 min / 双机 ≈2.5 h |
| `s8b`（12 档） | 20–30 min | `io` + `fault` | 15–30 min |
| `dx8b` + `nccl` | 13–22 min | `soak(all)` | 单机 ≈215 min / 双机 ≈245 min |
| 单机全链路（不含 soak） | **≈70 min** | 双机全链路（含 soak all） | **≈4 h 长稳另计** |

### 3.4 最小命令骨架

完整命令（含全部环境变量、双机逐条对应）见 `RUNBOOK.md` 第 1、2 节。骨架如下：

```bash
bash 02_precheck.sh && bash 03_deploy.sh
bash 04_bench.sh --phase argscheck
bash 04_bench.sh --phase hw
bash 04_bench.sh --phase s8b
bash 04_bench.sh --phase soak          # SOAK_MODEL=all 正式验收
bash 04_bench.sh --phase report
```

---

## 4. 测试用例矩阵

### 4.1 环境与自检（G0–G2）

| 用例 | 内容 | 通过判据 | 证据 |
|---|---|---|---|
| TC-01 | 主机侧只读体检（驱动三处一致、GPU 独占、ECC / retired / NVLink / Xid、容器引擎与 runtime、系统资源） | 无 FAIL；WARN 逐条登记 | `02-precheck/<host>-env.json` |
| TC-02 | 拓扑与 RoCE 探测 | `topo -m` 见 NV18；HCA 8 口可见 | `<host>-topo.txt` |
| TC-03 | 容器内 8 卡 / memlock / shm / NCCL 实测 | `device_count=8` | `02` 终端 + `selfcheck.txt` |
| TC-04 | 镜像与 Megatron 自检、参数表 dump | 「必需参数全部存在」、8 卡断言通过、`env.txt` 生成 | `03-deploy/selfcheck.txt` / `env.txt` / `megatron_args.txt` |
| TC-05 | 参数自检 + MoE 门禁 | `args-check.json` 的 `status=OK/DEGRADED`；关键 4 flag 齐备 | `args-help.txt` / `args-check.json` |

关键 flag 缺失时 `moe` / `moe-dp` 记 **SKIP**（不是 FAILED），不拼参数硬跑。

### 4.2 硬件微基准与通信（G3）

| 用例 | 内容 | 通过判据 | 证据 |
|---|---|---|---|
| TC-10 | `hw`：bf16 / FP8 GEMM、HBM、L2、PCIe、collective 四类 × 6 尺寸，稳态 ≥5 s | `hw.jsonl` 有记录；`hw.txt` 无 `!!` / 大面积 SKIP；FP8 缺失标 SKIP 不算失败 | `04-hw/hw.jsonl`、`hw.txt`、`dmon.csv` |
| TC-11 | `coll` 单机（已并入 hw，`OPTIONAL=1` 才单跑） | `coll.jsonl` 含 op × size 的 `busbw_gbs` / `latency_us` | `05-coll/coll.jsonl` |
| TC-12 | `nccl` 双机直测 | `nccl.txt` 出现 `NET/IB`，HCA 为 `mlx5_bond_*`；无 `NET/Socket` | `06-train-nccl/nccl.txt`、`nccl.jsonl` |
| TC-13 | RDMA 门禁（`dx8b`/`d70b`/`moe`/`moe-dp`/`soak` 前置，8 MB allreduce，60 s 超时） | `rdma-gate.json` 的 `verdict=PASS`、`busbw_gbs` 非 null、逐口 `verdict` 无异常 drop | `rdma-gate.json` / `.jsonl` / `.txt` |

> TC-10/11 带宽口径（11-10 P1 修订）：`timed()` 返回的是**单次平均耗时**，HBM / L2 / PCIe 公式不得再乘 `it`；collective 的 `sec` 是 `it` 次**总**耗时，busbw 必须按单次耗时算（`algbw = payload_b/(sec/it)/1e9`、`busbw = algbw*factor`）。`hw.jsonl` 现补 `iters` / `secs` / `algbw_gbs` 便于复核。修正后应见 HBM read ≈3.6 TB/s、PCIe h2d ≈30 GB/s、1G~8G 消息 busbw ≈330~370 GB/s；旧记录的 1813 TB/s / 1478 GB/s / 3.6 GB/s 均为口径错误。

`COLL_MAX_MSG` 默认 1 GiB，按 `size × NNODES` 剔除跨机大消息档（无 PFC 的 RoCE 上 allgather 输出放大 nodes 倍会丢包卡死），`0` 可全量放开但需现场确认。

### 4.3 单机 8B 矩阵 S1–S4（12 档，`--phase s8b`）

| 组 | 档位（`w8_8B_*`） | 变量 | 观测目的 |
|---|---|---|---|
| S1 | `tp8_pp1_mbs{1,2,4}_s4096_gb{8,16,32}_bf16` | micro-batch 上界 | 算力饱和点 + 显存余量；`mbs1` 为**基线档**，写入 `baseline.txt` |
| S2 | `tp{1,4,8}_pp1_mbs2_s4096_gb32_bf16` | 并行策略（GBS 固定 32 才可比） | TP 越多通信越重 |
| S3 | `tp8_pp1_mbs1_s{2048,8192,32768,131072}_gb8_bf16` | 序列长度 | 长上下文与显存上限；看 `tok/s/GPU` 归一值，不能只看 step 时间 |
| S4 | `tp8_pp1_mbs4_s4096_gb8_{bf16,fp8}` | 数值精度 | FP8 真实收益；按 `--help` 判定，**`--fp8-format=e4m3`（启用入口）+ `--fp8-recipe=delayed`（缩放策略）同传**；只有 recipe 时整组 SKIP（11-10 P0：只传 recipe 会跑出 `args.fp8=None` 的假 FP8 档）。跑完自检日志 `args.fp8`，未启用则写 `.fp8_not_active`；BF16 基准档始终照跑 |

通过判据：`results.jsonl` 中每个预期档位 `status=OK`；`*.log` 有 `elapsed time per iteration` 且步数达标；MFU† 见 5.2；OOM / NCCL / HANG / CRASH 由日志分类归因。

### 4.4 双机矩阵（G4）

| 组 | 阶段 | 档位 | 判据 |
|---|---|---|---|
| D1 | `dx8b` | 8B TP8×DP2，`mbs2 s4096 GBS32` | 与单机同配置 step 时间比值 **≤0.6** 视为跨机链路可用；0.6–1.0 有收益但链路瓶颈；>1.0 加机器不涨；>2.0 负收益，70B 只能靠 PP |
| D2 | `d70b` | 70B TP8×PP2，`mbs1 s2048 GBS{4,16}` | 流水线气泡 `≈(pp-1)/num_micro`，GBS=4 时约 25%；加大 GBS 后 MFU 应上升 |
| D3 | `d70b` | 70B TP8×PP2，`mbs1 s{4096,8192} GBS12`，`OOM_AS_BOUNDARY=1` | 序列长度与 OOM 边界；OOM 记 `OOM_BOUNDARY`，**不计失败、不 dual_stop、不影响退出码** |

D3 的 `OOM_BOUNDARY` 只落 `.oom_<tag>` 哨兵、**不进 `results.jsonl`**，判读时别只盯 `results.jsonl`。`d70b` 要求 `world≥16`；在 8 卡上跑 70B 必然 OOM，属配置错误，不得归因显卡。

### 4.5 MoE / MLA（配置 B：24 层 / 32 专家 + 1 共享 / topk2 / MLA）

| 阶段 | 拓扑 | 档数 | 判据 |
|---|---|---|---|
| `moe` 单机 | EP8×TP1、EP2×TP4，各 × `alltoall` / `allgather` | 4 | `moe.jsonl` 的 `status=OK`；专家负载 `max/mean >1.5` 标 WARN（路由塌缩） |
| `moe` 双机 | EP16×TP1×PP1 × 两种 dispatcher | 2 | 同上，另看跨机 all-to-all 成本 |
| `moe-dp` 双机 | EP8×DP2，`alltoall` | 1 | 与 EP16 对照：跨机 all-to-all vs 跨机 all-reduce 的成本差 |

> 专家负载数据源（11-10 P1 修订）：该版 Megatron 不向 stdout 打印逐专家 token 计数，只改正则拿不到数据。`moe)` / `soak_moe` 自动挂 `probes/moe_load_probe.py`（复制成 `sitecustomize.py` + `MOE_LOAD_PROBE=1`），训练时打印 `[expert load] step=N max/mean=x/y counts=[...]`，供 `expert-load-timeseries.csv`（`expert_tokens` / `max` / `mean`）与 `moe-soak.jsonl`（`imbalance_max_over_mean` / `imbalance_drift`）采集。探针不可用时明确降级并在 `note` 写明原因；短测步数少时可把 `MOE_LOAD_LOG_EVERY` 调到 20~50 以保证至少采到一次。

MLA 路径必须带 `--enable-experimental`（`MOE_MLA=auto` 默认）；配置 B 已剥离 GQA 两项，避免 TE 断言崩溃。观测：激活参数 MFU、`expert_load_max_over_mean`、`step_s`、`transport`。

### 4.6 峰值 MFU（`mfu`）

`--phase mfu` 回答「单机/双机最高能到多少」，不是「某一档的 MFU 是多少」：
A 段粗扫并行度 × GBS → B 段在最优拓扑上扫 mbs × accum → C 段扫 seq → D 段前 3 名各复测 `MFU_REPEAT` 次取中位。
每档 `MFU_STEPS` 步、采样时钟与 `clocks_throttle_reasons`；`CV>MFU_CV_MAX` 或降频的档不进峰值。

产出 `mfu.jsonl`（逐档）+ `mfu-peak.json`（每 (模型, world) 一个峰值）。头条口径
`MFU† = Megatron 自报 throughput ÷ 989.5 TFLOPS`（含 attention），`MFU(6N)` 作保守下界；
硬天花板是 `04-hw` 的 bf16 GEMM 实测 690 TFLOP/s（69.7%），报告同时给「占天花板百分比」。
先跑 `MFU_PLAN_ONLY=1` 看网格（不启动容器）。

> 原 `limit` 阶段（扫 mbs→seq 撞 OOM、产出 `limit.jsonl`）已删除：它原本只用来给 highmem 提供
> 跨模型边界，而跨模型借边界正是把 mbs 压成 1、占用只剩 29% 的根因。highmem 现在直接复用 MFU 峰值档。

### 4.7 存储与数据管线（`io`）

| 用例 | 内容 | 判据 |
|---|---|---|
| IO-1 | 8B checkpoint 分片写读代理 | `io.jsonl` 有 `write_mbs` / `read_mbs` / `write_s` / `read_s` |
| IO-2 | 70B checkpoint（`OPTIONAL=1`） | 同上；不跑不算失败 |
| IO-3 | 真实 dataloader | 有 `$ROOT/data/*.bin` + `*.idx` 才跑，看 `io-dataloader.log` 的 throughput 行；否则 `status=SKIP` 并注明 mock-data 不含 dataloader / tokenizer |

checkpoint 是**分片写读代理**（每卡写 bf16 权重 1/world），不是 Megatron 分布式 checkpoint 格式。

### 4.8 故障注入（`fault`）

| 用例 | 注入 | 预期 / 判据 |
|---|---|---|
| F1 | 单机 8 卡跑 70B | 日志命中 `CUDA out of memory` → PASS；未 OOM → FAIL；启动期断言/参数错误 → **INVALID** |
| F2 | 训练进入 iteration 后 `kill -9` 一个 rank | rc≠0 **且**容器已清理 **且**日志命中 `SIGKILL` / `exitcode: -9` / `ChildFailedError` → PASS；缺任一 → FAIL |
| F3 | 训练进入 iteration 后 `SIGSTOP` 一个 rank + `--distributed-timeout-minutes=2` | rc≠0 **且**日志命中 `Watchdog caught collective operation timeout` 等超时特征 → PASS；缺任一 → FAIL |

共同前置（11-10 P0 修订）：

- `fault_train` 必须带 `-e CUDA_DEVICE_MAX_CONNECTIONS=1`（与 `run_train` 一致），否则 TP=8 触发 Megatron 启动断言，8 个 rank 全崩 —— 现场 3/3 用例因此无效。
- 注入时机由固定 `sleep 20/15` 改为 `fault_wait_ready`：等日志出现第一个 `elapsed time per iteration`（上限 120s）再注入，避免「训练没起来」或「已跑完」时误判。
- 任何日志出现 `AssertionError` / `unrecognized arguments` 等启动期错误 → `status=INVALID`（用例未执行），**不再用 rc≠0 直接判 PASS**；`INVALID` 计入 `lib.sh` 阶段汇总的未过项。
- 三个 case 均需 PASS 才算全绿；FAIL / INVALID 不中断后续阶段，但必须写进报告。

### 4.9 长稳（`soak`，单机三条 / 双机四条互相独立、不可互相替代）

| 形态 | 配置 | 时长 | 判据 | 主要证据 |
|---|---|---|---|---|
| `dense` | 8B TP8×PP1 mbs4 GBS8 | 120 min，≥100 iters | 双机以全局末级 rank 的 `status=OK` 且 ≥100 iters 为准；非末级节点 `rc=0` 且无 iteration 不算失败；抖动 ≤5%；温度 <85 °C；`health-delta.json` PASS | `results.jsonl` soak 行、`dmon-w<N>_8B_soak_*.csv`、`health-*.json`、node-b 末级 rank 日志 |
| `dense70`（仅双机） | Llama-3.1-70B TP8×PP2×DP1，mbs1 seq4096 GBS16；num_micro_batches=16，PP2 理论气泡约 6.25% | 60 min，≥100 iters | 全局末级 rank `status=OK` 且 ≥100 iters；抖动 ≤5%；温度 <85 °C；`dense70-health-delta.json` PASS | `results.jsonl` soak 行、`dmon-w16_70B_soak_*.csv`、`dense70-health-*.json`、node-b 末级 rank 日志 |
| `moe` | 配置 B，EP×TP1×PP1，alltoall | 单机 30 / 双机 60 min | `moe-soak.jsonl` `status=OK`；`max/mean ≤1.5`、`|drift| ≤20%`；`moe-health-delta.json` PASS | `expert-load-timeseries.csv`、`dmon-moe-soak-*.csv` |
| `highmem` | **档位 = 本 run `mfu.jsonl` 里 MFU† 最高的那一档**（model/TP/PP/mbs/seq/GBS 全沿用，单机双机同一套逻辑；没有 mfu.jsonl 直接报错） | 60 min（无标定耗时） | **无占用率通过线**：跑满且无异常 → `OK`；任一卡增长 ≥5% → `WARN_LEAK`；峰值在前 20% 且末尾贴峰 → `WARN_FRAG`；OOM → `OOM_BOUNDARY`（不计失败）；<10 min → `SMOKE` | `highmem.jsonl`、`memused-w<N>_highmem.csv`、`highmem-health-*.json` |
| `mfu` | 单机 8B：TP{1,4,8}×GBS{32,64,128,256} 粗扫 → 最优拓扑上 mbs{1,2,4,8}×accum{2,4} → seq{2048,4096,8192} → 前 3 名各复测 3 次；双机再加 8B TP1/2/4/8×DP 与 TP8×PP2 对照、70B TP8×PP2 | 单机 ≈75 min / 双机 ≈2.5~3 h | 每 (模型, world) 一个峰值：`MFU†` 取复测中位，`CV>2%` 或 `throttle≠0` 排除，复测=1 标 UNCERTIFIED；与 GEMM 天花板（690 TFLOP/s = 69.7%）对比 | `mfu.jsonl`、`mfu-peak.json`、`clocks-w<N>_MFU_*.csv` |

`dense70` 是正常训练口径：验证跨机 PP 流水线的持续稳定性；highmem 验证「最优配置连续跑一小时」的显存稳定性与泄漏，不能相互替代。双机 `SOAK_MODEL=all` 顺序为 `dense → dense70 → moe → highmem`；单机 `all` 不含 dense70。

各条有独立哨兵（`.ok_` / `.fail_`），一条失败不影响其它条目结论，但整个 `soak` 阶段返回非 0。Megatron `training_log` 使用 `print_rank_last`，双机只有 `global rank = world-1` 打印 iteration；这与 PP 是否启用无关，8B DP2、MoE EP16、70B PP2 的非末级节点都可能 0 行。非末级节点仅在 `rc=0` 时免于本机步数门槛，最少步数由末级节点判定；末级节点不足步数仍为 PARTIAL。8B 与 70B 各有独立 dmon 和硬件增量，报告不把后跑档位的功耗/温度套到前档。MoE / highmem 的 JSONL 若 node-a 与 node-b 同 tag，报告优先采用 node-b 的更多 iteration 记录。跑前 / 跑后各采一次 ECC、retired_pages、NVLink 误码、Xid，**任一增量 >0 即 FAIL**；H200 不暴露 `retired_pages.correctable` 时按 SKIP 处理，不算失败。

**进度证据**：长稳期间后台循环每 `PROGRESS_SECS`（默认 60 s）把一行进度打到终端 stderr 并追加到 `09-soak/soak-progress.log`（已用 / 目标时长、百分比、ETA、步数 vs `SOAK_MIN_ITERS`、最近 step 与吞吐、双机 node rank）。进度启动前清空同名训练日志，避免读到上一轮残留；结束时补打一条收尾进度。容器 stdout 可能批量刷盘，周期行步数可滞后，最终以收尾行与 `results.jsonl` 为准。该文件随 `sync_logs` 一并回传，用于判断「卡在第几步 / 何时开始掉速」；`SOAK_PROGRESS=0` 可关闭。双机只有全局末级 rank 打印 iteration，非末级节点步数恒 0 属正常，日志会标注「步数由全局末级 rank 打印」，不得据此判为卡死；报告从 node-b 回传日志/JSONL 取性能数据。

**短时冒烟边界**：若将 `SOAK_MINUTES`、`DENSE70_SOAK_MINUTES`、`MOE_SOAK_MINUTES`、`HIGHMEM_MINUTES` 都设为 1 分钟做编排验证，必须同步设 `SOAK_MIN_ITERS=1` 或分别设各段 `*_MIN_ITERS`。70B 首步含编译、建链，1 分钟通常只有几步；保留默认 ≥100 步会让末级节点按 `PARTIAL` 中止。此类结果只能证明脚本链路可运行，不能作为性能、抖动或长稳验收证据。

### 4.10 汇总（`report`）

`python3 05_report.py --logs <run>` 递归收集各阶段目录，输出：训练压测结果表、关键对比（D1 / D1b / D2 / S3）、硬件微基准、collective 带宽、MoE / EP、峰值 MFU、存储、故障注入、8B/70B Dense 长稳、MoE 长稳、highmem 与各自硬件增量、判定、结果指纹、口径声明。报告由 `tee` 同时落 `10-report/report.txt` 与 `report.md`。

---

## 5. 判定标准

### 5.1 指标口径（统一，避免两套口径混写）

| 指标 | 公式 / 定义 |
|---|---|
| 每卡每步 token | `GBS × seq ÷ world` |
| 理论最短 step `t_min` | `6 × N × GBS × seq ÷ (world × 989.4 TFLOPS)`（用 GBS 不是 mbs） |
| 6N-MFU | `t_min ÷ t_actual`；长序列下漏掉 attention 平方项，会系统性偏低 |
| MFU† | Megatron 自报 `throughput per GPU (TFLOP/s/GPU) ÷ 989.4`；**验收判定以 MFU† 为准** |
| 通信 / 计算 | `4 × 989e12 ÷ (6 × 带宽bytes/s × 每卡token)`（**分子不含 `world`**）；跨机按实测 `busbw` 填，不套机内 300 GB/s |
| PP 气泡 | `≈(pp-1) ÷ num_micro_batches`，`num_micro_batches = GBS ÷ (mbs × dp)`（派生量，不是命令行参数） |
| 显存 | nvidia-smi 143771 MiB = 140.4 GiB；torch `total_memory` ≈139.8 GiB 用于「能分配多少」 |

### 5.2 数值阈值

| 指标 | 通过 | 警告 | 失败 / 停手 | 来源 |
|---|---|---|---|---|
| MFU†（验收档 world=8/16） | ≥30% | 20–30% | <20%（回去查 `hw.jsonl` busbw 与 `power.limit`） | `05_report.py` 判定段 |
| MFU 正常水位 | 单机 8B 35–45%；双机 70B 25–38%（含气泡） | 明显低于水位 | — | 同上 / 修订说明 |
| D1 双机 / 单机 step 比值 | ≤0.6 | 0.6–1.0 | >1.0 无收益；>2.0 负收益 | `dx8b` + 报告关键对比 |
| step 抖动 | ≤5% | >5% | — | 长稳 / `dmon-*.csv` |
| 温度 / 功耗 | 稳定、不贴顶 | 抖动随时间放大、温度 ≥85 °C、功耗爬升 | 不能写「长稳通过」 | 修订说明 §7 |
| 硬件增量 | ECC / retired / NVLink / Xid / SXid 全为 0 | — | 任一 >0 → FAIL | `health-delta.json` |
| highmem `peak_pct` | 75–92%（即 `HIGHMEM_BAND_LO~HI`） | <75% → `WARN_LOW`（未达高占用，撤销哨兵）/ >92% → `WARN_HOT` / `growth_pct >5%` | OOM → `OOM_BOUNDARY`（不算失败） | `highmem.jsonl` |
| MoE 专家负载 | `max/mean ≤1.5`、`|drift| ≤20%` | 超阈值 WARN | `status=FAILED` | `moe-soak.jsonl` |
| 结果指纹 | 全 run 唯一 | — | 多组指纹 → 比值不可比 | `print_fingerprint` |

### 5.3 状态字典

| 状态 | 含义 | 是否计入失败 |
|---|---|---|
| `OK` | 步数达标、日志有 iteration、无错误 | 否 |
| `OOM` | 非预期显存溢出 | 是 |
| `OOM_BOUNDARY` | D3 / highmem 的预期边界；**只落 `.oom_<tag>` 与 `highmem.jsonl`，不进 `results.jsonl`** | 否，排除在最好成绩与 MFU 之外 |
| `PARTIAL` | 长稳步数不足 `SOAK_MIN_ITERS` | 是，不写 `.ok_` 哨兵 |
| `FAILED` | 参数错误 / NCCL / CRASH / HANG | 是 |
| `SKIP` | 前置条件不具备（缺 flag、无真数据、EP 不整除） | 否，必须注明原因 |
| `ABORTED` | 双机一侧失败后整阶段中止，避免档位错位 | 是 |

### 5.4 分层准出

| 层 | 准出条件 |
|---|---|
| G0 | `02` 无 FAIL，WARN 已登记；双机指纹 diff 为空 |
| G1 | `03` 自检通过，`env.txt` 生成，两台逐项一致 |
| G2 | `args-check.json` `status=OK/DEGRADED`，关键 MoE flag 齐备（否则相关阶段 SKIP 并声明） |
| G3 | `hw.jsonl` 有记录；双机 `nccl.txt` 见 `NET/IB`；RDMA gate PASS |
| G4 | 预期档位 `status=OK`，非预期失败已归因；D1 比值达标；OOM 均为预期边界 |
| G5 | `fault` 三例全 PASS；单机三条 / 双机四条长稳各自 `status=OK` 且硬件增量 PASS；`io` 有记录或显式 SKIP |
| G6 | 报告生成，指纹唯一，限制声明齐备，验收清单逐项签核 |

### 5.5 「整机达标」的必要条件（缺一不可）

`hw`（8 卡全 collective）+ `nccl`（NET/IB 实测）+ 单机/双机训练矩阵 + **8B Dense 2 h 及其余适用长稳（含双机 70B 60 min）硬件增量 PASS** + `io` + `fault` 全部通过。只跑完单机 + 双机训练矩阵，最多只能声称「训练可用、跨机链路满足 PP 扩展」。

---

## 6. 证据链与复跑规则

### 6.1 一档一证据

每个训练档位同时留下三类文件，缺一不可用于验收：① `w<world>_<tag>.log` 原始 stdout/stderr；② `results.jsonl` 一行结构化结论（含指纹）；③ `.ok_` / `.fail_` / `.oom_` 哨兵。探针类留 `*.jsonl` + `*.txt`；长稳另留 `dmon-*.csv` 与本档 `health-*.json`（70B 为 `dense70-health-*.json`）。PP2 的 70B 性能日志以 node-b 末级节点为准。

### 6.2 归档位置

```
$ROOT/runs/<RUN_ID>/
  00-run-info.txt  index.md  args-help.txt  args-check.json  baseline.txt
  02-precheck/   03-deploy/   00-argscheck/
  04-hw/  05-coll/  06-train-{s8b,dx8b,nccl,d70b,moe,moe-dp,mfu}/
  07-io/  08-fault/  09-soak/  10-report/  node-b/
```

`runs/latest` → 最近一次 run；`$ROOT/logs` → `runs/latest` 的兼容链接。容器内 `/workspace/logs/` 通过 `-v "$LOGDIR":/workspace/logs` 映射到宿主当前阶段目录。`OUT_DIR` 只改 `02` 的两个文件落点。

### 6.3 复跑与回归

| 场景 | 规则 |
|---|---|
| 修复后复验 | **必须用新的 `RUN_ID`**；旧 run 的 FAILED 记录不要与新结果合并 |
| 已完成档位重跑 | `FORCE=1` 忽略 `.ok_` 哨兵；失败档位本身不会写 `.ok_`，可直接重跑 |
| 补跑报告 | `RUN_ID=<历史 run 名> bash 04_bench.sh --phase report`，不新建目录 |
| 双机 | 两台 `RUN_ID` 同名，双机命令同时启动；`sync_logs` 回传 node-b 目录作为旁证 |
| 跨轮对比 | 只比同一 `baseline.txt` 配置；先 diff `env.txt` 指纹，再 diff `results.jsonl`（tag 相同才可比） |

---

## 7. 已知差异、风险与回退

### 7.1 风险与回退

| 风险 | 症状 / 定位 | 回退 |
|---|---|---|
| 容器内看不到 8 卡 | `ENVF gpus≠8` | 宿主有 8 张 → 运行时 / toolkit 注入问题，升 1.17.x 四包；宿主不足 → 驱动 / 掉卡 / MIG |
| NCCL 退回 socket | `nccl.txt` 无 `NET/IB` 或有 `NET/Socket` | 核对 `NCCL_IF` / `NCCL_HCA` / `show_gids` 的 v2 GID / `ibv_devinfo` |
| 逐口门禁 `RDMA_GATE_FAIL` | `per_hca[].verdict=drop`，对端 ping 不通 | 两台都设 `PEER_ADDR`、BatchMode ssh 互通；必要时 `PEER_ROCE_BASE`；修完 `FORCE=1` 只复验 `nccl` |
| 探针秒败、无 `rdma-gate.txt` | `probe_error=missing_probe` / `no_output` | 把整个 `probes/` 目录同步到两台；再看引擎 / rendezvous / timeout 137 |
| 跨机大消息卡死 | 1 GiB allgather 输出 16 GiB/卡、600 s 看门狗 | 保留 `COLL_MAX_MSG` 上限；需全量时先确认 PFC 配置 |
| 双机 Gloo 挂住 | `ProcessGroupGloo` / `connectFullMesh` | 确认自动带 `--disable-gloo-process-groups`，查容器内 `/etc/hosts` |
| MLA backward 崩溃 | `view size is not compatible` | `03_deploy.sh` 已打 `.view → .reshape` 补丁，两台各执行一次 |
| 长稳失败 | `.fail_*`、`status=PARTIAL`、硬件增量 FAIL | 看 `dmon-*.csv` 降频/温度、ECC/NVLink/Xid 增量；修好直接重跑 |
| 显存 OOM | `results.jsonl` `status=OOM` | 属上界信号：降 `seq` / `mbs` / `GBS` 或换并行切分；F1 单机 70B OOM 是预期 PASS |
| FP8 / dataloader SKIP | `hw.jsonl` FP8=SKIP、`io.jsonl` dataloader=SKIP | 不是失败，写进限制声明；`hw_probe.py` 的 FP8 GEMM 已按 `b.t()` + 0 维 scale 修好，若仍 SKIP 看 `note` 里的逐签名失败原因；需真实数据则放 `$ROOT/data/*.bin`+`*.idx` |
| S4 假 FP8 | 阶段目录出现 `.fp8_not_active`，或 FP8 档日志 `fp8 ..... None` | 该档 `args.fp8=None`、实际跑 BF16，加速比不可用；核对 `--fp8-format=e4m3` 是否拼进命令行（11-10 P0 已修参数链），重跑 `--phase s8b` 前先清掉旧 `.ok_` 哨兵 |
| 专家负载全 null | `expert-load-timeseries.csv` 全 `null`、`moe-soak.jsonl` 的 `imbalance_*` 为 null | 查训练日志 `[expert load probe]` 行：`install failed` / `disabled` = Megatron API 不匹配；无探针行 = 探针未同步或 `MOE_LOAD_PROBE` 未透传；有 `installed` 但无 `[expert load]` = 调小 `MOE_LOAD_LOG_EVERY`。11-10 P1 已补探针，历史 run 需重跑 |
| hw 带宽量级异常 | HBM 上千 TB/s、PCIe 上千 GB/s、collective 个位数 GB/s | 11-10 P1 已修 `hw_probe.py` 两处相反的 iters 错误；重跑 `--phase hw` 并用新增的 `iters`/`secs`/`algbw_gbs` 复核 |
| 磁盘不足 | `02` 报镜像存储 <60 G / run 余量 <10 G | 清旧 run / 旧镜像（nerdctl 只动 `default`，别碰 `k8s.io` 业务镜像），或把 `ROOT` 换大容量盘（两台都改） |

### 7.2 审计发现：方案与脚本的已知差异（需评审确认）

| 项 | 现状 | 影响 / 建议 |
|---|---|---|
| **D1b（双机 GBS=64 同每卡 token 对照）** | `05_report.py` 有 `D1b` 解析与打印分支，但 `--phase dx8b` 当前**只跑 GBS=32 一档**，没有产出 GBS=64 的档位 | 报告中 D1b 行不会出现。若验收需要「同 per-GPU token 口径」对照，须补一档 `8B_tp8_pp1_mbs2_s4096_gb64_bf16` 的训练用例（可手工 `run_train` 或扩展 `dx8b`），否则在报告里声明该对照未执行 |
| 旧 TESTPLAN 的 VPP / `num-micro-batches` 扫参 | 脚本未实现 VPP；`num-micro-batches` 是派生量 | 本方案已删除，改为加大 `--global-batch-size`；VPP 列为后续可选项 |
| 旧 TESTPLAN 的 141 GiB / `core_r0.12` / 通信占比带 `world` / D1 阈值 >2.0 | 均为已勘误项 | 本方案按 140.4 GiB / `core_v0.13.1` / 去 `world` / ≤0.6 写 |
| 单机 `coll` 与 `io` 70B 档 | 已并入 `hw` / 降为 `OPTIONAL=1` | 不作为必测项，避免重复计时 |
| 精简矩阵 | S1 3 档、S2 3 档、S3 4 档（2K/8K/32K/128K）、S4 2 档 | 与旧计划 5 档长序列不同，以脚本为准 |

---

## 8. 交付物与验收清单

### 8.1 报告结构

`05_report.py` 自动生成：训练压测结果表、关键对比（D1 / D1b / D2 / S3）、硬件微基准、collective 带宽、MoE / EP、显存边界、存储、故障注入、8B/70B Dense 长稳、MoE 长稳、highmem 与各自硬件增量、判定、结果指纹、口径声明。

需人工补：① 执行摘要（硬件是否达标、8B 最好 MFU†、双机 70B 是否成立）；② 两台环境与物料（driver / CUDA / 镜像 digest / Megatron commit / NCCL）；③ 拓扑原文与机间 RDMA 实测；④ 限制声明；⑤ 结论与建议（能否收机、若要 DP 扩展需什么网络）。

### 8.2 验收清单

- [ ] G0 两台无 FAIL，WARN 已逐条登记；双机 `env.txt` diff 为空
- [ ] `results.jsonl` 指纹唯一；`index.md` 各阶段结果非 `ABORTED`
- [ ] G3：`hw.jsonl` 有记录、`nccl.txt` 见 `NET/IB`、所有 RDMA gate PASS
- [ ] G4：S1–S4 12 档、D1/D2/D3、MoE 各档 `status=OK` 或显式 SKIP；非预期 OOM 已归因
- [ ] D1 比值 ≤0.6（或已写明链路瓶颈结论）；最好成绩 MFU† ≥30%
- [ ] `mfu-peak.json` 给出每个 (模型, world) 的峰值 MFU† 与复测 CV
- [ ] `fault` 三例全 PASS；`io` 有记录或显式 SKIP 并说明原因
- [ ] `soak(all)` 单机三条 / 双机四条各自 `status=OK`；`health-delta.json`、`dense70-health-delta.json`（双机）、`moe-health-delta.json`、`highmem-health-delta.json` 均 `status=PASS`
- [ ] `10-report/report.md` 生成，人工补的 5 个段落齐备
- [ ] 7.2 的 D1b 差异已处置：补跑或写进限制声明

### 8.3 限制声明（可直接引用）

> 本轮使用 `--mock-data` + `NullTokenizer` + 随机初始化，不含真实数据管线与分布式 checkpoint 落盘开销，测得的是**算力 / 通信上限**，不是端到端训练吞吐；真实训练会更慢。显存按 nvidia-smi 实测 143771 MiB（140.4 GiB）计，容器内 torch 可分配约 139.8 GiB，差额为 ECC / 驱动预留与 CUDA 上下文。机间为 RoCE（非 IB），跨机带宽按实测 `busbw` 判定。docker 与 nerdctl 的 GPU 注入路径不同，跨引擎比较以镜像 digest 为准。本轮基线与历史 docker 基线的性能优化门控不同，不做横向对比。
