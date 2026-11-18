# H200 大模型训练压测工程集

一套在 **8×H200（143GB）单机 / 双机**环境上做大规模语言模型**训练性能压测**与**继续预训练**落地的工程集合。

覆盖从镜像与源码准备、硬件门禁自检、压测用例矩阵、长稳与显存墙探测、MFU 峰值搜索，到日志解析与统计报告的完整链路。

> 仓库只收录**脚本、配置与文档**，不含跑测产物。地址、路径、主机名均为占位值，**直接运行会在门禁阶段失败**，请先按[使用前需要修改哪些地方](#使用前需要修改哪些地方)改一遍。

---

## 子项目

| 目录 | 定位 | 入口 |
|---|---|---|
| `llm-bench-docker/` | **基线版**：Docker 路线从拉镜像到出报告的首版工程骨架 | `01_fetch.sh` → `05_report.py` |
| `llm-bench-fix/` | **修复版**：单入口多 phase 的压测矩阵，修正首版的口径与门禁问题，带 `probes/` 硬件探针 | `04_bench.sh --phase <name>` |
| `llm-bench-split/` | **拆分版**：把 `fix` 的巨型脚本按阶段拆成 `bench/00-config` ~ `70-phase-mfu`，配置与逻辑分离，便于复用与二次开发 | `bench/40-run.sh` |
| `e2e-megatron-56b-2501/` | **端到端压测**：2×8 H200 上 56B 级模型的单机/双机对照 | `run_56b_2x8h200.sh` |
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
│   ├── RUNBOOK.md             #   逐条执行命令清单
│   └── TESTPLAN.md            #   测试方案：目标、门禁、用例矩阵、判定口径
├── llm-bench-split/           # 拆分版（bench/ 分阶段脚本 + tools/）
├── e2e-megatron-56b-2501/     # 56B 端到端压测
│   ├── 00_prepare.sh          #   镜像 tar + Megatron 源码包落地与自检
│   ├── run_56b_2x8h200.sh     #   双机 2×8 卡
│   └── run_1x8h200_cases.sh   #   单机用例集
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

**证据链落盘。** 每轮跑测在 `ROOT` 下产出 `metrics.json`（原始结构化指标）、`cases.csv`（全部用例）、`findings.csv`（带优先级的问题清单）、`report.html`（可读报告）——这些是**运行产物**，不在仓库内。

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

---

## 环境要求

| 项 | 要求 |
|---|---|
| GPU | 8×H200 143GB / 节点 |
| 驱动 | ≥ 580 原生；≥ 570 依赖镜像自带 `cuda-compat-13-0` 前向兼容（已实测） |
| 容器引擎 | `docker` 优先，缺失时回落 `nerdctl`（`lib.sh` 内自动探测） |
| 镜像 | NVIDIA PyTorch 容器；继续预训练用 ms-swift v4.5.3 官方镜像 |
| Python | 3.10+（`05_report.py` 等报表脚本仅用标准库，无需额外依赖） |

---

## 使用前需要修改哪些地方

仓库里的地址、路径、主机名都是占位值，**直接跑会在门禁阶段 `exit 2`**。按下面 5 类改完即可跑通。

### 1. 网络地址

| 占位值 | 含义 | 改成 |
|---|---|---|
| `192.0.2.x` | 节点管理网 IP | 你的管理 IP |
| `198.51.100.x` | RoCE 网段 | 你的 RoCE IP |

涉及位置：

| 文件 | 需要改的项 |
|---|---|
| `megatron-swift-qwen38-cpt/config/cluster.env` | `MASTER_ADDR`（默认 `192.0.2.10`）、`NODE_IPS`（示例 `192.0.2.21,192.0.2.22`） |
| `megatron-swift-qwen38-cpt/scripts/13_multinode.sh` | 多机用法示例里的 `MASTER_ADDR` / `NODE_IPS` |
| `llm-bench-fix/RUNBOOK.md`、`llm-bench-split/RUNBOOK.md` | `NODE_A_IP` / `NODE_B_IP` 示例值 |
| `llm-bench-fix/04_bench.sh` | 报错文案里的 `198.51.100.x` / `192.0.2.x`（只是提示语，不影响逻辑） |
| `llm-bench-split/bench/00-config.sh`、`10-boot.sh`、`20-net.sh` | 同上 |
| `e2e-megatron-56b-2501/run_56b_2x8h200.sh` | 注释中的 `MASTER_ADDR` 示例 |

双机硬性要求（缺任一项直接 `exit 2`，不会静默退回 TCP socket）：

```bash
# 两台执行同一条，只有 NODE_RANK 不同
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-0 管理 IP> \
  NCCL_IF=bond0 NCCL_HCA=mlx5_bond_2,mlx5_bond_3 bash 04_bench.sh --phase d70b
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-0 管理 IP> \
  NCCL_IF=bond0 NCCL_HCA=mlx5_bond_2,mlx5_bond_3 bash 04_bench.sh --phase d70b
```

`NCCL_IF` 填承载 RoCE 的 bond，`NCCL_HCA` 填本机实际的 `mlx5_bond_*` 列表，**两台必须一致**；`MASTER_ADDR` 不能是回环。

### 2. 路径

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ROOT` | `/root/llm-bench` | 宿主上挂进容器的根目录；Megatron 源码与所有产物都必须在这下面 |
| `MODEL_ROOT` | `cluster.env` 里是 `/data/Qwen3.8-27B` | **语义是模型根目录，但当前默认值多带了一层模型名**：`env.sh` 会再拼 `Qwen3.8-27B`，结果变成 `/data/Qwen3.8-27B/Qwen3.8-27B`。要么把 `MODEL_ROOT` 改成 `/data`，要么显式指定 `MODEL_27B_DIR` / `MODEL_FLASH_NEXT_DIR` |
| `MODELSCOPE_CACHE` | `/mnt/shared/modelscope_cache` | 多机必须落在共享存储上 |
| `PROJECT_REMOTE_DIR` | `/root/megatron-swift-qwen38-cpt` | 代码在目标机上的实际路径 |

### 3. 镜像

| 位置 | 默认值 | 要改什么 |
|---|---|---|
| `e2e-megatron-56b-2501/` 三个脚本 | `nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3` | 换成你能拉取该镜像的地址（本地 registry 或 `nvcr.io`） |
| `llm-bench-docker/01_fetch.sh` | `nvcr.io/nvidia/pytorch:$NGC_PYTORCH_TAG`（默认 `25.06-py3`）、`MEGATRON_REF=core_v0.13.1` | 设置 `NV_API_KEY`（NGC API Key），按驱动版本选 tag |
| `megatron-swift-qwen38-cpt/config/image.env` | ModelScope 官方 `swift4.5.3` | 版本必须 ≥ v4.5.3，否则 T0 探针直接判 `UNSUPPORTED` 并中止 |

镜像 tar 与源码包需按 `00_prepare.sh` 注释的布局就位：`$ROOT/pytorch-*.tar`、`$ROOT/Megatron-LM-core_v0.13.1.zip`，解包目录必须正好叫 `Megatron-LM`（`PYTHONPATH` 依赖它，改名后 8 个 rank 会全部 import 失败）。

### 4. 权重与数据（`megatron-swift-qwen38-cpt`）

| 项 | 默认 | 说明 |
|---|---|---|
| `MODEL_27B_DIR` | `$MODEL_ROOT/Qwen3.8-27B` | 已下载的 18 分片权重目录 |
| `MODEL_FLASH_NEXT_DIR` | `$MODEL_ROOT/Qwen3.8-Flash-Next` | 131 分片权重目录 |
| `DATASET_ID` / `DATASET_FULL_SPEC` | `swift/chinese-c4` | 需先下载到 `DATA_ROOT`（96 个 `.jsonl.zst`） |

分片数与 sha256 可对照 `manifests/*.json`：`scripts/00_preflight.sh` 会校验，`tools/verify_weights.py` 可单独跑。

### 5. 运行环境与权限

- 容器引擎：`docker` 优先，缺失时回落 `nerdctl`；`megatron-swift` 全程用 `nerdctl --namespace default`，`load` / `run` / `images` 必须同一 namespace
- 双机需两节点 root 免密 ssh（`PEER_ADDR` 的逐口门禁要 ssh 到对端取网卡表）
- 驱动 ≥ 580 原生可用；≥ 570 依赖镜像自带 `cuda-compat-13-0` 前向兼容

### 建议的自检顺序

```bash
# llm-bench-fix：环境门禁 → 单机硬件基线 → 双机通信基线
cd llm-bench-fix
bash 02_precheck.sh
bash 04_bench.sh --phase hw
bash 04_bench.sh --phase nccl        # 双机时两台都要执行

# megatron-swift-qwen38-cpt：镜像 / 权重 / 数据预检
cd ../megatron-swift-qwen38-cpt
bash scripts/00_preflight.sh
```

逐条命令与判读标准见各项目的 `RUNBOOK.md` 与 `TESTPLAN.md`。

### 仓库里没有包含的东西

- `run/`、`logs/` 下的原始压测日志与 `dmon` GPU 采样 CSV
- 跑测产出的 `metrics.json` / `cases.csv` / `findings.csv` / `*.html` 报告
- 模型权重、数据集、Megatron 源码压缩包
- 临时脚本与备份目录（`.tmp/`）

---

## 已知限制

- 仓库不含历史跑测结果，文档里提到的数字需自行重跑复现。
- `llm-bench-docker` 为早期基线，存在口径问题：FP8 档实际未启用 FP8 计算（日志中 `fp8=None`、`params_dtype=bfloat16`，与 bf16 档首步 loss / grad norm 逐位相同）。`llm-bench-fix` 已用 `--phase argscheck` 在跑测前拦截此类情况。
- 双机场景的结论强依赖 RoCE 链路质量，换机器后需重跑 `--phase nccl` 建立基线。

## License

未指定。如需开源授权，可在仓库中补充 LICENSE 文件。
