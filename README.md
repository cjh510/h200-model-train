# H200 大模型训练压测工程集

一套在 **8×H200（143GB）单机 / 双机**环境上做大规模语言模型**训练性能压测**与**继续预训练**落地的工程集合。

覆盖从镜像与源码准备、硬件门禁自检、压测用例矩阵、长稳与显存墙探测、MFU 峰值搜索，到日志解析与统计报告的完整链路；每个子项目都保留了当时真实的用例结果与问题清单（`analysis/`）。

> 仓库内所有内网地址已替换为 **RFC 5737 文档保留地址**，路径与主机名已做别名化，详见[脱敏说明](#脱敏说明)。

---

## 子项目

| 目录 | 定位 | 入口 |
|---|---|---|
| `llm-bench-docker/` | **基线版**：Docker 路线从拉镜像到出报告的首版工程骨架 | `01_fetch.sh` → `05_report.py` |
| `llm-bench-fix/` | **修复版**：单入口多 phase 的压测矩阵，修正首版的口径与门禁问题，带 `probes/` 硬件探针与 `analysis/` 问题清单 | `04_bench.sh --phase <name>` |
| `llm-bench-split/` | **拆分版**：把 `fix` 的巨型脚本按阶段拆成 `bench/00-config` ~ `70-phase-mfu`，配置与逻辑分离，便于复用与二次开发 | `bench/40-run.sh` |
| `e2e-megatron-56b-2501/` | **端到端压测**：2×8 H200 上 56B 级模型的单机/双机对照，含单机报告与双机联合统计报告 | `run_56b_2x8h200.sh` |
| `megatron-swift-qwen38-cpt/` | **继续预训练**：`megatron pt` + Mcore-Bridge，在已下载权重上用 `swift/chinese-c4` 做 CPT | `scripts/03_continue_pt.sh` |

### 演进关系

```
llm-bench-docker  ──修复口径/门禁──>  llm-bench-fix  ──按阶段拆分──>  llm-bench-split
                                            │
                                            └──同一套采集口径──>  e2e-megatron-56b-2501
                                                                  megatron-swift-qwen38-cpt
```

---

## 目录结构

```
.
├── llm-bench-docker/          # 基线版（含 TESTPLAN、环境/拓扑快照、统计数据）
├── llm-bench-fix/             # 修复版
│   ├── 02_precheck.sh         #   环境与硬件门禁
│   ├── 03_deploy.sh           #   起容器 / 挂载 / 环境注入
│   ├── 04_bench.sh            #   压测矩阵总入口（多 phase）
│   ├── 05_report.py           #   汇总报告
│   ├── lib.sh                 #   公共函数（容器引擎探测、指标解析、指纹）
│   ├── probes/                #   硬件探针：GPU 健康 / IO / MoE 负载
│   ├── analysis/              #   解析 → metrics.json / cases.csv / findings.csv / report.html
│   ├── RUNBOOK.md             #   逐条执行命令清单
│   └── TESTPLAN.md            #   测试方案：目标、门禁、用例矩阵、判定口径
├── llm-bench-split/           # 拆分版（bench/ 分阶段脚本 + ref/ + tools/）
├── e2e-megatron-56b-2501/     # 56B 端到端压测
│   ├── 00_prepare.sh          #   镜像 tar + Megatron 源码包落地与自检
│   ├── run_56b_2x8h200.sh     #   双机 2×8 卡
│   ├── run_1x8h200_cases.sh   #   单机用例集
│   └── node{0,1}-e2e-56b/     #   各节点 args-help、镜像 digest、自检脚本、analysis 报告
└── megatron-swift-qwen38-cpt/ # 继续预训练
    ├── config/                #   cluster.env / env.sh / image.env / model_*.env
    ├── scripts/               #   00_preflight → 02_smoke → 03_continue_pt → 04_resume → 05_verify
    ├── docs/                  #   AUDIT（证据链）/ TESTPLAN（用例矩阵）
    ├── manifests/             #   权重与数据清单（分片数、大小、sha256）
    └── tools/                 #   权重校验 / 镜像探针 / 数据子集切分
```

---

## 通用设计

**门禁优先，不静默降级。** 双机场景下 `NCCL_IF` / `NCCL_HCA` / 非回环 `MASTER_ADDR` 缺一即 `exit 2`，不会悄悄退回 TCP socket 导致测出来的数字失真。

**每档结果自带指纹。** 用例结果同时记录镜像 digest、TP/PP/MBS/SEQ/GBS、精度、迭代数，保证数字可复现、可追溯。

**退出码不等于档位通过。** OOM 被显式记为 `OOM_BOUNDARY` 作为结论，而不是失败；只有门禁失败才是 `FAIL`。

**证据链落盘。** 每轮跑测产出 `metrics.json`（原始结构化指标）、`cases.csv`（全部用例）、`findings.csv`（带优先级的问题清单）、`report.html`（可读报告）。

---

## 快速开始

以 `llm-bench-fix` 为例（在目标裸机上执行，脚本内部自己起容器）：

```bash
cd llm-bench-fix

# 1. 环境与硬件门禁
bash 02_precheck.sh

# 2. 部署：起容器、挂载 Megatron 源码、注入 NCCL 环境
bash 03_deploy.sh

# 3. 单机：硬件基线 → 8B 用例矩阵
bash 04_bench.sh --phase hw
bash 04_bench.sh --phase s8b

# 4. 双机（两台执行同一条命令，只有 NODE_RANK 不同）
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 NCCL_IF=bond0 \
NCCL_HCA=mlx5_bond_2,mlx5_bond_3 bash 04_bench.sh --phase d70b

# 5. 汇总
python 05_report.py
```

长稳与峰值：

```bash
SOAK_MINUTES=30 bash 04_bench.sh --phase soak        # 长稳
bash 04_bench.sh --phase mfu                          # MFU 峰值粗扫→细扫→复测
bash 04_bench.sh --phase limit                        # 显存/规模墙，OOM 记边界
```

### 压测阶段一览（`04_bench.sh --phase`）

| phase | 内容 |
|---|---|
| `argscheck` | 容器内 `pretrain_gpt.py --help` 落 `args-check.json`，确认参数面支持情况 |
| `hw` | 单机硬件基线：FP8 / L2 / PCIe / collective，稳态 ≥5s |
| `coll` | 单机 collective 扫描（已并入 `hw`，`OPTIONAL=1` 时单跑） |
| `s8b` | 单机 8 卡 Llama-3.1-8B 四组用例 S1–S4 |
| `dx8b` | 双机 8B TP8×DP2 训练对照（GBS=32） |
| `nccl` | 双机 collective 全类型 × 8B–8G（不跑训练） |
| `d70b` | 双机 16 卡 70B TP8×PP2 |
| `moe` | MoE/MLA：单机 EP8×TP1、EP2×TP4 + 双机 EP16，两种 dispatcher |
| `moe-dp` | 仅双机：EP8×DP2（alltoall + 跨机 all-reduce），与 EP16 对照 |
| `limit` | 显存/规模墙：扫 mbs→seq 至 OOM，记录 `OOM_BOUNDARY` |
| `soak` | 长稳：dense / dense70 / moe / highmem 可组合 |
| `io` | checkpoint 写读与数据管线（无真实数据则 SKIP） |
| `fault` | 故障注入：OOM / kill rank / NCCL 超时 |
| `mfu` | 峰值 MFU：并行度×GBS 粗扫 → mbs×accum 细扫 → seq → 前三名复测取中位 |
| `report` | 汇总成表 |

---

## 指标口径

| 指标 | 定义 |
|---|---|
| `tflops` | Megatron 自报吞吐换算的 TFLOPS（bf16） |
| `MFU†` | `throughput / 989.5`，989.5 为 H200 SXM **BF16 dense** 峰值（带稀疏时官方标 1979） |
| `step_ms` | 单步耗时中位数，稳定期取 `iter >= 500` |
| `tflops_cv_pct` | 稳态 TFLOPS 变异系数，用于判定抖动 |
| `mem_max_reserved_mb` | 显存峰值保留量，用于定位显存墙 |

`e2e-megatron-56b-2501/node1-e2e-56b/analysis/` 下还有单机与双机联合的统计分析报告（HTML，含吞吐、线性度、通信占比）。

---

## 环境要求

| 项 | 要求 |
|---|---|
| GPU | 8×H200 143GB / 节点 |
| 驱动 | ≥ 580 原生；≥ 570 依赖镜像自带 `cuda-compat-13-0` 前向兼容（已实测） |
| 容器引擎 | `docker` 优先，缺失时回落 `nerdctl`（`lib.sh` 内自动探测） |
| 镜像 | NVIDIA PyTorch 容器；继续预训练用 ms-swift v4.5.3 官方镜像 |
| Python | 3.10+（`05_report.py` 与 `analysis/*.py` 仅用标准库） |

---

## 脱敏说明

仓库内容来自真实集群跑测，发布前已做如下处理：

| 类型 | 原值 | 现值为 |
|---|---|---|
| 管理网 / 集群地址 | 内网三段地址 | `192.0.2.x`（RFC 5737 TEST-NET-1） |
| RoCE 网段 | 内网私有段 | `198.51.100.x`（RFC 5737 TEST-NET-2） |
| 项目路径 | `/root/<user>/...` | `/root/...` |
| 模型根目录 | `/<user>/Qwen3.8-27B` | `/data/Qwen3.8-27B` |
| 主机名 | `worker-gpu-<编号>` | `worker-gpu-a` |
| 环境快照文件 | `<user>-env.json` / `<user>-topo.txt` | `env-snapshot.json` / `topo-snapshot.txt` |

替换后所有 shell 脚本（`bash -n`）、Python 文件（`py_compile`）与 JSON 配置均可正常解析，**配置项仍保持可运行的 IP 字面量**，不会因为占位符变成语法错误。实际使用时请把 `192.0.2.x` / `198.51.100.x` 换回你的真实地址。

镜像仓库地址（ModelScope / NGC）为公开仓库，未做处理。

### 未包含的内容

- `run/`、`logs/` 下的原始压测日志与 `dmon` GPU 采样 CSV（数百 MB，逐行含内网地址）
- 模型权重、数据集、Megatron 源码压缩包
- 临时脚本与备份目录（`.tmp/`）

需要复现具体数字时，按 `RUNBOOK.md` 的步骤重跑即可；`analysis/` 里保留了当次的结构化结果与问题清单。

---

## 已知限制

- `analysis/` 中的结果对应当时的脚本版本，后续修改脚本不会回填历史数据。
- `llm-bench-docker` 为早期基线，部分口径问题已在 `llm-bench-fix` 的 `findings.csv` 中记录（如 FP8 档实际未启用 FP8 计算）。
- 双机场景的结论强依赖 RoCE 链路质量，换机器后需重跑 `--phase nccl` 建立基线。

## License

未指定。如需开源授权，可在仓库中补充 LICENSE 文件。
