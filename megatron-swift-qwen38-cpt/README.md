# Megatron-SWIFT 继续预训练（Qwen3.8-27B / Qwen3.8-Flash-Next）

用 `megatron pt` + Mcore-Bridge，在**已下载的权重**上，用 `swift/chinese-c4` 做继续预训练（CPT）。
本文档只讲**怎么配、怎么跑**；版本依据与证据链见 [docs/AUDIT.md](docs/AUDIT.md)，用例矩阵见 [docs/TESTPLAN.md](docs/TESTPLAN.md)。

---

## 0. 前置条件

| 项 | 要求 | 自查命令 |
| --- | --- | --- |
| 镜像 | 官方 ms-swift **v4.5.3**（已拉取，不构建派生镜像） | `nerdctl --namespace default images \| grep swift4.5.3` |
| **驱动** | **≥ 580** 原生可用；**≥ 570** 靠镜像自带的 `cuda-compat-13-0` 前向兼容（已实测通过） | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| GPU | 每节点 ≥ `GPUS_PER_NODE` 张，本流程按 H200 143GB 设计 | `nvidia-smi -L` |
| 权重 | 27B：18 分片 / 51.75 GiB；Flash-Next：131 分片 / 335.28 GiB | `ls /data/Qwen3.8-27B` |
| 数据 | `swift/chinese-c4`（96 个 `.jsonl.zst`，压缩 21.3 GiB） | `ls data/chinese-c4/data \| head` |
| 引擎 | `nerdctl` + `--namespace default`（load/run/images 必须一致） | `nerdctl --namespace default ps` |

> 驱动那条最容易踩：`nvidia-smi` 能看到 8 张 H200 **不代表**容器里 torch 能用。
> 容器里的用户态驱动 `libcuda.so` 是 **nvidia-container-toolkit 从宿主注入的**，
> 版本恒等于宿主驱动，与镜像的 CUDA 版本无关——这就是 `nvidia-smi` 正常但
> `torch.cuda.is_available()=False` 的原因。
>
> 镜像自带 `cuda-compat-13-0`（一套 580 的用户态 `libcuda`），`lib.sh` 会在
> 检测到宿主驱动 < 580 时**自动**把它前置到 `LD_LIBRARY_PATH`，实现前向兼容。
> 现场实测：宿主 `570.172.08` + compat → `torch.cuda.is_available()=True`。

---

## 1. 配置

### 1.1 `config/cluster.env`（现场唯一必改）

加载顺序是 `cluster.env → env.sh → image.env`，后两者全部用 `${VAR:-default}`，
所以**本文件里设的值优先**，命令行同名变量再覆盖它。

| 变量 | 含义 | 当前示例 | 说明 |
| --- | --- | --- | --- |
| `RUN_IMAGE` | 训练镜像 | `...swift4.5.3` | 必须 ≥ v4.5.3，否则不认识 `qwen3_8`/`qwen4_exp` |
| `SKIP_PULL` | 跳过拉取检查 | `1` | 镜像已在各节点时保持 1 |
| `MODEL_ROOT` | 模型**根目录** | `/data` | 下面会自动拼 `Qwen3.8-27B`、`Qwen3.8-Flash-Next` |
| `MODEL_27B_DIR` | 27B 权重目录 | `$MODEL_ROOT/Qwen3.8-27B` | 若 27B 就是根目录本身，**显式写全路径** |
| `MODEL_FLASH_NEXT_DIR` | Flash-Next 目录 | `$MODEL_ROOT/Qwen3.8-Flash-Next` | 同上 |
| `MODELSCOPE_CACHE` | 下载缓存 | `/mnt/shared/modelscope_cache` | **多机必须共享** |
| `GPUS_PER_NODE` | 每节点卡数 | `8` | 改小会自动重算 `CUDA_VISIBLE_DEVICES` |
| `NNODES` | 节点数 | `1` | 单机保持 1，13 会自动退化为直调 12 |
| `MASTER_ADDR` / `MASTER_PORT` | 通信地址 | `192.0.2.10` / `29500` | 单机不影响；多机填 node0 的 IP |
| `NODE_IPS` | 各节点 IP（逗号分隔） | 空 | **顺序即 `NODE_RANK`**，个数必须等于 `NNODES` |
| `PROJECT_REMOTE_DIR` | 各节点项目路径 | `/root/megatron-swift-qwen38-cpt` | 多机时各节点必须一致 |
| `CONTAINER_ENGINE` / `CONTAINER_NAMESPACE` | 容器引擎 | `nerdctl` / `default` | docker 用户改成 `docker` |

> `MODEL_ROOT` 是**根目录**，不是模型目录。填 `/data/Qwen3.8-27B` 会得到
> `/data/Qwen3.8-27B/Qwen3.8-27B`。填 `/data`，或把 `MODEL_27B_DIR` 写成完整路径。

### 1.2 训练开关（`config/env.sh` 默认值，命令行可覆盖）

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `TRAIN_ITERS` | `100` | 训练步数上限 |
| `DURATION_MINS` | `0` | >0 时按**墙钟分钟**收口（`--exit_duration_in_mins`）；步数自动抬到 `DURATION_ITERS_CAP` 当上限 |
| `DURATION_ITERS_CAP` | `100000` | 时长模式下的步数上限，只在用户没显式给 `TRAIN_ITERS` 时生效 |
| `LOG_THROUGHPUT` | `true` | 是否尝试传 `--log_throughput`。**swift 4.5.3 的 CLI 不暴露该参数**，会自动跳过；吞吐改由 `05_verify.sh` 用步时换算 |
| `LOGGING_STEPS` | `1` | 日志打印间隔（步）。**必须 ≤ 总步数**，否则连 `iteration` 行都不打印，步时/吞吐全部拿不到 |
| `DMON` | `1` | 训练全程在**宿主机**跑 `nvidia-smi dmon` |
| `DMON_INTERVAL` | `5` | 采样间隔秒；24h×8 卡约 10 MB |
| `DMON_METRICS` | `pucvmt` | dmon 的 `-s`：功耗温度/利用率/时钟/违规/显存/链路 |
| `CUDA_COMPAT` | `auto` | `auto` 宿主驱动 <580 时自动启用镜像的 `cuda-compat-13-0`；`force` 强制；`off` 关闭 |
| `CUDA_COMPAT_DIR` | 空 | 留空自动探测 `/usr/local/cuda/compat`，非默认布局时手工指定 |
| `CUDA_COMPAT_MIN_DRIVER` | `580` | 低于此大版本才启用 compat，即 `auto` 的判定阈值 |
| `MAX_LENGTH` | `8192` | 序列长度 |
| `GLOBAL_BATCH_SIZE` | `16` | 全局批；必须能被 `MICRO_BATCH_SIZE × DP` 整除 |
| `LR` / `MIN_LR` / `LR_WARMUP_ITERS` | `1e-5` / `1e-6` / `30` | 学习率与 warmup |
| `SAVE_STEPS` | `50` | 每多少步存一次 checkpoint |
| `STREAMING` / `PACKING` | `true` / `true` | 官方 `pretrain.sh` 基线 |
| `EXTRA_ARGS` | 空 | 追加任意 Megatron 参数，如 `'--optimizer_cpu_offload true'` |

### 1.3 模型档案

`config/model_27b.env`（27B，全参）：

| 项 | 值 | 说明 |
| --- | --- | --- |
| `MODEL_TYPE` / `ARCHITECTURE` | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | 与 `config.json` 一致，T0 会校验 |
| `TP` / `PP` / `CP` | `4` / `1` / `1` | 8 卡 → DP=2 |
| `TUNER_TYPE` | `full` | 全参 CPT |
| `RECOMPUTE_GRANULARITY` | `full` | 省显存 |
| `ATTENTION_BACKEND` | `flash` | |

`config/model_flash_next.env`（Flash-Next，全参）：

| 项 | 值 | 说明 |
| --- | --- | --- |
| `MODEL_TYPE` / `ARCHITECTURE` | `qwen4_exp` / `Qwen4ExpForConditionalGeneration` | |
| `TUNER_TYPE` | `full` | 全参 CPT。显存不够时改 `lora` |
| `TP` / `PP` / `EP` / `ETP` | `1` / `4` / `2` / `1` | DP 随节点数变：单机 DP=2、2 节点 DP=4、4 节点 DP=8；**EP 必须整除 DP，2 节点以上建议 `EP=4`**；48 层/4 段=12 层 |
| `SP` | 由 TP 自动决定 | TP=1 时强制 `false`（sequence parallel 依赖 TP>1） |
| `OPTIMIZER_CPU_OFFLOAD` | `1` | **全参必需**：888 GB 优化器状态挪到主机内存（另需约 0.9 TB DRAM） |
| `PLE_CPU_OFFLOAD` | `1` | N-gram 表（≈95 GiB）卸载到 CPU，**会被冻结** |
| `MODEL_TOTAL_PARAMS_B` / `_FROZEN_` | `125` / `51` | 03 据此做全参显存预检 |
| `MOE_ARGS` | `--moe_permute_fusion true --decoder_first_pipeline_num_layers 12` | ETP 由 03 统一按 `ETP` 变量传，这里不重复写 |

### 1.4 容器（`config/image.env`）

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `PROJECT_IN` | `/workspace/qwen38-cpt` | 项目在容器内的路径（与宿主不同，`12` 自动映射） |
| `MODEL_ROOT_BIND` / `CACHE_BIND` | `1` / `1` | 把权重与缓存 bind 回**相同路径**（现场踩过的坑） |
| `CONTAINER_EXTRA_MOUNTS` | 空 | 额外挂载，如 `'-v /pfs/x:/pfs/x'` |
| `CONTAINER_GPUS` / `CONTAINER_SHM` | `all` / `64g` | |

---

## 2. 脚本清单

| 脚本 | 在哪跑 | 作用 |
| --- | --- | --- |
| `lib.sh` | 被 source | 配置加载、日志、`run_logged`、`ctr()`、**CUDA 前向兼容自动启用**、dmon 启停与汇总 |
| `00_preflight.sh` | 容器内 | **T0 预检**：驱动/CUDA 匹配、依赖版本、GPU 数、权重完整性、架构一致性、镜像能力、`megatron pt --help` |
| `02_smoke.sh` | 容器内 | **T2 冒烟**：缺失时自动抽 2000 条子集 → 调 03 跑 2 步 → 检查 checkpoint |
| `03_continue_pt.sh` | 容器内 | **T3 正式 CPT**，全项目唯一的真实训练命令（含时长/吞吐/MoE/LoRA 分支） |
| `04_resume.sh` | 容器内 | **T4 断点续训**：找最近 ckpt，`--finetune false` 连优化器状态一起恢复 |
| `05_verify.sh` | 容器内 | **T5 验收**：checkpoint 分片、loss、NaN、步时、tokens/s、TFLOP/s；可选推理 |
| `12_run_in_container.sh` | 宿主机 | **容器入口**：拼 `nerdctl run`、挂载、路径映射、变量透传、dmon 包裹；`--shell` 进容器 |
| `13_multinode.sh` | 宿主机 | **多机编排**：`--check` 自检、ssh 并行启动、按 IP 顺序注入 `NODE_RANK` |

`tools/`：

| 文件 | 作用 |
| --- | --- |
| `probe_image.py` | 15 个包版本 + 源码注册扫描，判定镜像是否支持 Qwen3.8 |
| `verify_weights.py` | 按 `manifests/*.json` 校验权重（存在/大小/小文件 sha256），分级 CRITICAL / WARN |
| `make_subset.py` | 从 `.jsonl(.zst)` 抽冒烟子集，产出 text 与 messages 两种格式 |

---

## 3. 使用流程

### Step 0 · 改配置

```bash
cd /data/megatron-swift-qwen38-cpt
vi config/cluster.env        # 权重路径、GPU 数、IP
```

### Step 1 · 预检（T0）

```bash
bash scripts/12_run_in_container.sh scripts/00_preflight.sh 27b
```

预期结尾：`[OK] 架构与本档案一致` → `=== T0 通过 ===`。
失败会明确告诉你是驱动、依赖、权重还是镜像的问题，**不要跳过这一步直接训练**。

### Step 2 · 冒烟（T2）

```bash
bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b
```

首次会自动从 `data/chinese-c4` 抽 2000 条到 `data/smoke_text.jsonl`；
2 步跑完并产出 `checkpoint-*` 即通过。数据在别处时：

```bash
RAW_DATA_DIR=/mnt/shared/datasets/chinese-c4 \
  bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b
```

### Step 3 · 正式 CPT（T3）

```bash
# 按步数
bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# 按墙钟（24 小时）
DURATION_MINS=1440 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# 只看命令不执行
CONTAINER_DRY_RUN=1 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b
DRY_RUN=1 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b   # 容器内 dry-run
```

### Step 4 · 断点续训（T4）

```bash
bash scripts/12_run_in_container.sh scripts/04_resume.sh 27b
RESUME_FROM=/data/megatron-swift-qwen38-cpt/outputs/qwen38_27b/xxx/checkpoint-50 \
  bash scripts/12_run_in_container.sh scripts/04_resume.sh 27b
```

### Step 5 · 验收（T5）

```bash
bash scripts/12_run_in_container.sh scripts/05_verify.sh 27b
RUN_INFER=1 bash scripts/12_run_in_container.sh scripts/05_verify.sh 27b   # 带推理冒烟
```

### 多机

```bash
bash scripts/13_multinode.sh --check                                  # ssh + 目录 + 镜像
DRY_RUN=1 bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b  # 只打印每节点命令
bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b            # 真跑
```

前提：各节点项目/权重/缓存路径一致且共享、`NODE_IPS` 个数 = `NNODES`、ssh 免密。
`NNODES=1` 时 13 自动直接调 12，不用单独配置。

**Flash-Next 全参必须多机**（8 卡放不下，见 1.3 的显存账）。2 节点示例：

```bash
# config/cluster.env
#   NNODES=2
#   NODE_IPS=192.0.2.10,192.0.2.11
#   MASTER_ADDR=192.0.2.10      # node0 的 IP
#   PROJECT_REMOTE_DIR=/data/megatron-swift-qwen38-cpt

bash scripts/13_multinode.sh --check
DRY_RUN=1 EP=4 bash scripts/13_multinode.sh scripts/03_continue_pt.sh flash_next
bash scripts/13_multinode.sh scripts/03_continue_pt.sh flash_next
```

启动后每个节点都会打印显存账，2 节点应是：

```
全参显存账: 可训练 74B → 状态 1184 GB（权重+梯度 296 + 优化器 888）
            单卡需求: 权重+梯度 74 GB（÷TP×PP=4）+ 优化器 222 GB（÷DP=4）= 296 GB，可用 140 GB
            CPU 卸载已开：每节点需约 444 GB 主机内存（888 GB ÷ 2 节点）
```

**`OPTIMIZER_CPU_OFFLOAD=1` 在 2 节点时仍然必需**——不开的话每卡要 296 GB，而只有 143.7 GB。
要到 64 卡（8 节点）才降到 129 GB/卡，可以不卸载。

### 进容器排查

```bash
bash scripts/12_run_in_container.sh --shell
# 容器内：nvidia-smi / python -c "import torch;print(torch.cuda.is_available())"
# 退出后清理：nerdctl --namespace default rm -f qwen38-cpt
```

---

## 4. 数据集与权重怎么放

`DATASET_SPEC` 三种写法，**写宿主机路径即可**（`12` 自动转容器路径并补挂载）：

| 放法 | 路径 | 怎么写 | 适用 |
| --- | --- | --- | --- |
| 数据集 ID（默认） | 不落地，缓存在 `MODELSCOPE_CACHE` | `DATASET_SPEC=swift/chinese-c4` | 正式 CPT |
| 项目内 | `$PROJECT_ROOT/data/<任意>/` | `DATASET_SPEC=$PWD/data/chinese-c4-jsonl` | 单机离线 |
| 共享存储 | `/mnt/shared/datasets/<任意>/` | `DATASET_SPEC=/mnt/shared/datasets/chinese-c4` | 多机 |

三条规则：

1. **`.jsonl.zst` 不能直接喂 `--dataset`**（只认 csv/json/jsonl/txt/目录）。要用本地全量先解压：
   ```bash
   mkdir -p data/chinese-c4-jsonl
   for f in data/chinese-c4/data/*.jsonl.zst; do
     zstd -dc "$f" > "data/chinese-c4-jsonl/$(basename "${f%.zst}")"
   done       # 96 片约 46 GB，先确认磁盘
   ```
2. **目录会递归扫描**，同目录里混着 `.zst` 会被忽略，正式跑建议单独放纯 `.jsonl` 目录。
3. **多机必须共享**：ID 模式靠 `MODELSCOPE_CACHE`，本地目录靠数据所在路径。

权重只需放在 `MODEL_27B_DIR` / `MODEL_FLASH_NEXT_DIR`，不转格式——Mcore-Bridge 直读 safetensors，
`--save_safetensors true` 直接存回 HF 格式。

---

## 5. 常用操作

```bash
# 显存吃紧（2048 上下文 + 优化器 offload）
MAX_LENGTH=2048 EXTRA_ARGS='--optimizer_cpu_offload true --use_precision_aware_optimizer true' \
  bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# VL 架构 + 纯文本 pt 不兼容时退回 sft
MEGATRON_SUBCMD=sft DATASET_SPEC=$PWD/data/smoke_messages.jsonl PACKING=false \
EXTRA_ARGS='--freeze_vit true --freeze_aligner true' \
  bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b

# 改并行（8 卡 → TP2/PP2）
TP=2 PP=2 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# 关掉吞吐行 / GPU 采样
LOG_THROUGHPUT=false bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b
DMON=0 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# 加密采样
DMON_INTERVAL=1 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b

# 驱动已 >=580、不需要前向兼容时显式关掉
CUDA_COMPAT=off bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b
```

---

## 6. 排障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available=False`，但 `nvidia-smi` 有卡 | 容器内用户态 `libcuda` 是宿主注入的，版本恒等于宿主驱动；宿主 r570 撑不起镜像的 cu130 | 确认 `CUDA_COMPAT=auto` 且镜像有 `/usr/local/cuda/compat`；否则升驱动到 580+ |
| 启用 compat 后 `nvidia-smi` 仍报宿主驱动版本 | 正常现象：`nvidia-smi` 走 NVML（宿主注入），不是 `libcuda` | 只看 `torch.cuda.is_available()`，别用 nvidia-smi 判断 |
| `megatron pt --help` 长时间不返回 | CUDA 初始化阻塞（已加 120s 超时） | 先确认 `torch.cuda.is_available()` 为 True |
| `[c10d] The hostname of the client socket cannot be retrieved. err=-3` | 容器内 `/etc/hosts` 没有本机 hostname，`getaddrinfo` 失败（EAI_AGAIN） | **可忽略**：`MASTER_ADDR` 是裸 IP 时不影响通信；想去掉在 `12` 里加 `--hostname $(hostname)` |
| 日志停在 c10d 之后长时间无输出 | ① 27B 权重 18 分片 / 51.75 GiB，TP 切分 + 初始化需要数分钟；② 更常见的真因是 `MASTER_ADDR` 填了本机不可达地址，rendezvous 重试到 600s 超时 | 用 `nvidia-smi` 看显存是否在涨；`NNODES=1` 时 `lib.sh` 已自动改用 `127.0.0.1`，要保留原值设 `MASTER_ADDR_KEEP=1` |
| `pt_help_*.txt` 里只有启动日志、没有参数列表 | `--help` 也走 `torch.distributed.run`，rendezvous 卡住导致没走到打印参数 | 同上；T0 已改为单进程 `NPROC_PER_NODE=1` 探测并做内容校验 |
| `assert lr_warmup_steps < lr_decay_steps` 后 8 个 rank 全退 | `lr_decay_steps` 就是 `--train_iters`，总步数 ≤ warmup 时 Megatron 直接断言失败（报错只有一行 assert，极易误判成模型/显存问题） | `03` 已自动钳制为 `min(原值, train_iters-1)` 并打印 WARN；冒烟 `02` 显式设 `LR_WARMUP_ITERS=1` |
| 日志里完全没有吞吐 / 步时 | 两个独立原因：① `--log_throughput` 不在 swift CLI 里（自动跳过，属正常）；② `LOGGING_STEPS` 大于总步数，连 `iteration` 行都不打印 | `LOGGING_STEPS` 已默认 1；吞吐看 `05_verify.sh` 的 tok/s |
| 参数量表里 `Trainable` 只有 93% 左右 | Qwen3.5 是多模态架构，ViT 视觉塔默认冻结，纯文本 CPT 不需要它 | 正常现象，不是配置错误 |
| `nerdctl: image not found`，但 `images` 能看到 | namespace 不一致 | 所有命令带 `--namespace default` |
| 容器内看不到权重/数据 | 宿主目录没 bind 回相同路径 | `MODEL_ROOT_BIND=1 CACHE_BIND=1`；数据集用 `DATA_ROOT` 或 `DATASET_SPEC` 指到真实目录 |
| 容器名 `qwen38-cpt` 已存在 | 上次 `--shell` 退出未清理 | `nerdctl --namespace default rm -f qwen38-cpt` |
| OOM | 序列太长 / 并行不匹配 | 降 `MAX_LENGTH`、开 `RECOMPUTE_GRANULARITY=full`、调 TP/PP |
| Flash-Next 全参启动就被判死「训练状态 > 总显存」 | 125B 全参约需 1184 GB 训练状态，8×H200 只有 1123 GB | 保持 `OPTIMIZER_CPU_OFFLOAD=1`（需约 0.9 TB 主机内存）；或扩到 2 节点；或 `TUNER_TYPE=lora` |
| loss 出现 NaN | lr 过大 / dtype / 数据异常 | 降 `LR`，检查 `05_verify.sh` 的 NaN 判定 |
| 脚本跑一半**突然断掉、没有任何报错** | `set -euo pipefail` + `cmd \| head -1` 触发 SIGPIPE（退出码 141），SIGPIPE 不产生 stderr | 已修（`-i 0` / `\|\| true` 兜底）；现在触发会打印 `[FATAL] xxx.sh 第 N 行失败` |
| `--help` 里没有 `--load` / `--exit_duration_in_mins` | 版本差异 | 03/04 会自动跳过并 WARN；续训改 `RESUME_MODE=mcore` |

---

## 7. 产出物

```
logs/
├── <tag>[_nodeN]_<时间戳>.log       # 训练全量 stdout（iteration/loss/lr/grad_norm/步时/吞吐）
├── <tag>[_nodeN]_latest.path        # 指向最近一次日志
├── dmon_<tag>[_nodeN]_<时间戳>.csv  # GPU 采样原始数据（功耗/温度/SM/显存/链路）
├── image_probe_<key>.json           # 镜像能力报告
├── pt_help_<key>.txt                # megatron pt --help 快照（参数能力以它为准）
└── multinode/<时间戳>/nodeN_ip.log  # 多机各节点日志

outputs/<模型 key>/
└── <时间戳>_cpt/checkpoint-<step>/  # HF safetensors + 优化器状态（视 NO_SAVE_OPTIM）
```

`05_verify.sh` 会汇总四类指标：

1. **loss 曲线**（末 10 条，兼容 Megatron 原生与 swift dict 两种格式）+ NaN 判定
2. **步时**：均值/最小/最大，以及 `tokens/s` 全局与每卡
3. **耗时分解**：训练循环总时长、checkpoint 存盘耗时与占比、扣除存盘后的真实步时（算 MFU 要用这个）、GPU 高负荷占比（dmon）
4. `TFLOP/s/GPU`（仅当镜像暴露 `--log_throughput` 时才有，本镜像没有）

耗时分解的三个来源与精度：

| 数字 | 来源 | 精度 |
| --- | --- | --- |
| 训练循环总时长 | tqdm 末尾的 `[MM:SS` | 精确 |
| checkpoint 存盘 | `Saving:` 进度条完成时长（每次存盘 2 个：Megatron + safetensors） | 精确 |
| GPU 高负荷占比 | dmon 的 `sm` 列，按 `End time` 反推循环区间后过滤 | 粗估（5s 采样） |

> 注意：存盘耗时与「GPU 低负荷」区间**部分重叠**（存盘期间仍有少量 copy kernel），两项不能相加。
