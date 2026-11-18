# 现状排查与证据清单（Audit）

> 采集时间：2026-11-13。所有数值均来自官方页面 / 官方开放 API / 本机文件系统实测，
> 未使用推测值。`manifests/*.json` 是可复现的完整文件清单（含每个文件的 Size / Sha256）。

## 1. 框架侧证据

| 来源 | 关键结论 | 依据 |
| --- | --- | --- |
| [Megatron-SWIFT 快速开始](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Quick-start.html) | 继续预训练用 `megatron pt`（不是 `sft`）；使用生成式 template | "若要进行预训练，你可以使用 `megatron pt` 替代 `megatron sft`" |
| 同上（Mcore-Bridge 章节） | `--model` + `--save_safetensors true` 可直接读 / 存 safetensors，**免 HF↔mcore 权重转换** | 这是本项目"在已有权重下继续训练"的核心机制 |
| [Mcore-Bridge 文档](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Mcore-Bridge.html) | `--model` 走 bridge 读 safetensors；`--mcore_model` 走 mcore 默认读取；`--no_save_optim false` 会额外存一份 mcore 权重用于断点续训 | 决定 T3 / T4 的参数分工 |
| [examples/megatron/pretrain.sh](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/pretrain.sh) | **官方就是用 `swift/chinese-c4` 做 Megatron 预训练**：4×80GiB、`streaming true`、`packing true`、TP4、`global_batch_size 16`、`max_length 8192`、`finetune true`、`attention_backend flash` | 本项目的参数基线 |
| [支持模型表](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html) | `Qwen/Qwen3.8-27B` → model_type `qwen3_5`，template `qwen3_8`，**Support Megatron ✔**；`Qwen/Qwen3.8-Flash-Next` → `qwen4_exp`，**Support Megatron ✔** | 两个模型都已被官方接入，不是"未支持的新架构"盲测 |
| [Qwen3.8 最佳实践](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/Qwen3_8-Best-Practice.md) | Qwen3.5/3.6/3.8 结构相同；Qwen3.8 用独立模板 `qwen3_8`；需 `transformers>=5.9`、`flash-linear-attention>=0.4.2`、`causal_conv1d`、`flash-attn==2.8.3` | 版本门槛 |
| [Qwen3.8-Flash-Next 最佳实践](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/Qwen3_8-Flash-Next-Best-Practice.md) | 125B 总参 / ~6B 激活，512 experts，51B N-gram 表（bf16 约 95 GiB）；`PLE_CPU_OFFLOAD=1`；官方 8 卡配置为 **TP2 / EP4 / ETP1 / PP2 + LoRA**，8×70G | 并行与显存基线 |
| 同上 | `PLE_CPU_OFFLOAD=1` 后 N-gram 表**被冻结**，全参训练时该部分静默不更新 | 全参 CPT 的重要语义风险 |
| [examples/models/qwen3_5/packing.sh](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/packing.sh) | 全参 + packing 的 MoE 参考：TP4/EP8、`--freeze_vit true --freeze_aligner true`、optimizer CPU offload | sft 回退路径与显存优化参考 |
| 源码 `MegatronBaseArguments.__post_init__` | `if self.packing: self.padding_free = True`；`if self.streaming: dataloader_num_workers = 1` | 解释了官方为何 streaming + packing 可同开 |

### 版本要求（官方 Quick-start 推荐环境）

| 组件 | 推荐 / 门槛 |
| --- | --- |
| Python | 3.12（>=3.10） |
| CUDA / torch | 12.8 或 13.0 / 2.8.0 或 2.11.0 |
| megatron-core | >=0.18,<0.21（0.20.0）或 dev 分支（CP 支持 GDN 需 dev） |
| mcore-bridge | >=1.6.5 |
| transformer-engine | >=2.3（2.16.0） |
| flash-attn | 2.8.3 / 4.0.0b15 |
| transformers | 27B: >=5.9；Flash-Next: **>=5.16** |
| flash-linear-attention | 27B: >=0.4.2；Flash-Next: **>=0.5.2** |

> 注意：`transformers 5.16` 与 `5.9` 的比较必须用数值分段，不能用字符串字典序
> （`"5.16" < "5.9"` 在字符串比较下成立，是错的）。`00_preflight.sh` 已按数值实现。

## 2. 模型侧证据（ModelScope 开放 API + 本地 config.json）

| 项 | Qwen3.8-27B | Qwen3.8-Flash-Next |
| --- | --- | --- |
| `architectures` | `Qwen3_5ForConditionalGeneration` | `Qwen4ExpForConditionalGeneration` |
| `model_type` | `qwen3_5` | `qwen4_exp` |
| template | `qwen3_8` | `qwen3_8` |
| 层数 / hidden | 64 / 5120 | 48 / 2560 |
| 注意力 | `linear_attention ×3 : full_attention ×1`（GDN 混合） | GDN + QSA（`indexer_budget 2048`）+ Gated Residual（`hc_count 4`） |
| MoE | Dense | 512 experts，`num_experts_per_tok 10`，`moe_intermediate_size 640` |
| 其他 | ViT（depth 27）、MTP 1 层、max_pos 262144 | 51B N-gram 表、MTP（hybrid）、ViT、max_pos 262144 |
| dtype | bfloat16 | bfloat16 |
| vocab | 248320 | 248320 |
| safetensors 分片 | **18** | **131** |
| 权重合计 | **55,563,006,776 B = 51.75 GiB** | **360,000,192,888 B = 335.28 GiB** |
| `model.safetensors.index.json` | 112,216 B | 170,726 B |
| 仓库文件总数 | 33 | 145 |

**本机实测**：`D:\model\qwen` 存在 18 个分片，合计 55,563,006,776 B，与官方清单逐字节一致；
本地 `config.json` 为 `Qwen3_5ForConditionalGeneration` / `qwen3_5`，即 **Qwen3.8-27B 完整权重**。
本机 `D:\model` 下未发现 Flash-Next（应为 `model-00001-of-00131.safetensors` 等 131 个分片），
需确认集群上的实际路径。

**快照差异**（本机 11-03 下载 vs 官方 master 11 月更新）：权重分片完全一致，但若干文本配置不同，
例如 `tokenizer_config.json` 本机 17,928 B / 官方 9,984 B、`chat_template.jinja` 8,952 B / 9,004 B。
`verify_weights.py` 会把它们标成 SIZE 不符。权重训练不受影响，但** tokenizer / chat template 建议同步到官方当前版本**，
以免 `qwen3_8` 模板行为不一致。

## 3. 数据集侧证据

| 项 | 值 | 来源 |
| --- | --- | --- |
| ID | `swift/chinese-c4` | 用户指定 |
| 用途 | 中文 Common Crawl 清洗语料，**纯文本预训练** | 数据集 README |
| 格式 | `data/chinese-c4-XXXX-of-0096.jsonl.zst`，每行为 `{"text": "..."}` | repo tree API |
| 分片 | **96** 个，每个约 238 MB | repo tree API |
| 压缩体积 | **21.30 GiB**（22,875,770,910 B） | repo tree API / 数据集详情 `StorageSize` |
| 解压体积 | 46.29 GB | 数据集 README |
| 划分 | 只有 train 分片（无 validation/test） | repo tree API |
| 许可 | cc-by-4.0 | 数据集详情 API |

> 因此本项目不需要自己造数据；数据由使用者预先下载（`modelscope download --dataset swift/chinese-c4`），
> `02_smoke.sh` 在冒烟子集缺失时从 `RAW_DATA_DIR` 自动抽 2000 条，
> 并额外提供 `messages` 格式仅用于 sft 回退链路验证，不用于效果评估。

## 4. 本机 / 集群环境事实

| 项 | 实测 | 影响 |
| --- | --- | --- |
| 当前机器 | Windows，`Python 3.13.14`，**无 nvidia-smi（无 GPU）** | 只能编辑与静态校验；训练必须放到 Linux GPU 集群 |
| 工作区既有资源 | `e2e-megatron-56b-2501/`（H200 单/双机 Megatron-LM 基准脚本与日志） | 说明用户的训练环境是 H200 集群，脚本按 Linux/bash 编写 |
| 既有约定 | 镜像 `pytorch:25.01-py3`、`$ROOT` 挂进容器 `/workspace`、8 卡/节点 | 新项目默认 `NPROC_PER_NODE=8`、`CUDA_VISIBLE_DEVICES=0..7` |
| 已知集群坑 | 容器路径与宿主路径不一致会导致 dataset cache 读不到 | 新项目强制 `MODELSCOPE_CACHE` 可配置，且多机必须共享 |

## 5. 结论

1. **可行性已确认**：框架支持两个模型，且官方有针对 chinese-c4 的 `megatron pt` 实例；
   Mcore-Bridge 让"已有权重 → 继续训练 → 直接存 safetensors"无需任何格式转换。
2. **权重现状不对等**：27B 本机完整；Flash-Next 未在本机发现，且体积 335 GiB，必须先确认集群路径与磁盘。
3. **真正的未知量**：`megatron pt` 对「VL ConditionalGeneration 架构 + 纯文本语料」组合没有官方实例；
   混合线性注意力 / PLE / MTP 在 pt 链路的行为也需实测。因此测试矩阵把链路冒烟（T0/T2）放在长跑（T3）之前。

---

## 6. 镜像环境证据（用户指定，2026-11-13 追加）

### 6.1 指定镜像

```
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:
  ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1
```

来自官方 [SWIFT 安装文档 - 镜像](https://github.com/modelscope/ms-swift/blob/main/docs/source/GetStarted/SWIFT-installation.md) 的
`swift4.4.1` 条目（同一 tag 在 cn-hangzhou / cn-beijing / us-west-1 三个 registry 均有）。
属于 Quick-start 的 **cu130 / fp8 training** 镜像系列。

### 6.2 版本分界（硬证据：GitHub tag 源码树 + 支持表）

采集方式：`GET /repos/modelscope/ms-swift/git/trees/<sha>?recursive=1` 与
`GET /repos/modelscope/ms-swift/contents/docs/source/Instruction/Supported-models-and-datasets.md?ref=<tag>`，
再按关键字计数。`Qwen3_8` 指 `docs/source/BestPractices/Qwen3_8-*.md`。

| tag | commit 日期 (UTC) | 文件数 | `qwen4_exp` | `Qwen3_8` 文档 | `qwen3_8` | 支持表中 `Qwen3.8` 行数 |
| --- | --- | --- | --- | --- | --- | --- |
| **v4.4.1（镜像）** | 2026-07-13T02:43:45Z | 1367 | **0** | **0** | **0** | **0** |
| v4.4.3 | 2026-08-03T12:35:44Z | 1383 | 0 | 0 | 0 | 0 |
| **v4.5.3** | 2026-09-07T12:31:14Z | 1426 | 2 | 2 | 6 | 有 |
| v4.6.1 | 2026-11-13T08:30:47Z | 1609 | 2 | 2 | 6 | 有 |
| main (4.7.0.dev0) | — | 1617 | 3 | 2 | 8 | 有 |

对照时间线（UTC+8，由 ModelScope 仓库 `CommittedDate` 换算）：

| 事件 | 时间 |
| --- | --- |
| Qwen3.8-27B 权重首次提交 | 2026-08-13 16:45:58 |
| Qwen3.8-Flash-Next 首次提交（release） | 2026-08-26 20:22:36 |
| 官方 chat_template / 模型卡更新 | 2026-11-07 21:49:44 |

**v4.4.1 比两个模型早 1 个月以上**；其支持表最新条目止于 `Qwen3.6-27B`（model_type `qwen3_5`，
template `qwen3_5`），完全没有 `qwen3_8` / `qwen4_exp`。
虽然 27B 的 config.json 架构名 `Qwen3_5ForConditionalGeneration` 在 4.4.1 里存在，
但 `transformers_version: 5.8.0.dev0` 的新字段与 `qwen3_8` 模板均不在该版本内，
不能视为可用；Flash-Next 的 `qwen4_exp` 则是**完全不存在**。

### 6.3 v4.5.3 支持表原文（首个可用版本）

```
| Qwen/Qwen3.8-27B        | qwen3_5  | qwen3_8   | qwen3_5 | transformers>=5.0.0.dev, qwen_vl_utils>=0.0.14, decord | ✔ |
| Qwen/Qwen3.8-Flash-Next | qwen4_exp| qwen4_exp | qwen3_5 | transformers>=5.16.0,    qwen_vl_utils>=0.0.14, decord | ✔ |
```

注意 v4.5.3 中 Flash-Next 的 template 列是 `qwen4_exp`，而 main/4.7 文档里是 `qwen3_8` ——
**版本间模板名不同**，所以脚本一律不手动传 `--template`，由框架按 `--model` 自动匹配。

v4.5.3 最佳实践的依赖命令：27B `transformers>=5.9` + `flash-linear-attention>=0.4.2`；
Flash-Next `transformers>=5.16` + `flash-linear-attention>=0.5.2` + `mcore-bridge -U` +
`transformer_engine[pytorch]` + `causal_conv1d`，并注明 vLLM 需源码 `>0.28.0`。
本项目不用 vLLM 后端，因此镜像里的 vllm 版本不构成阻塞。

### 6.4 本项目使用的镜像

```
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.13.0-vllm0.28.0-modelscope1.40.0-swift4.5.3
```
（同系列还有 `swift4.4.3`：torch2.11.0 / vllm0.25.1 / modelscope1.39.0 —— **仍不支持 Qwen3.8**。）

### 6.5 结论与处置

1. 最初指定的 swift4.4.1 镜像**不能原样训练这两个模型**，这是版本时间线造成的能力缺口，不是配置错误。
2. 处置：直接使用官方 **swift4.5.3** 镜像（`RUN_IMAGE`），依赖已由官方配对
   （torch2.13 / vllm0.28 / modelscope1.40），**不构建派生镜像、不在容器内 pip 补装**。
3. 判定固化在 `tools/probe_image.py`：扫描已安装包源码里的 `qwen3_8` / `qwen4_exp` 注册，
   并检查 `transformers` 是否含两个 `ForConditionalGeneration` 类，输出 JSON 报告。
   `00_preflight.sh` 会把不支持直接判为失败，避免把"版本不支持"误当成调参问题。

### 6.6 驱动栈：容器里的 libcuda 来自宿主（2026-11-16 实测结论）

**现象**：宿主 `nvidia-smi` 正常看到 8×H200，容器内 `torch.cuda.is_available()=False`，
`megatron pt --help` 长时间阻塞。最初判断为"驱动 12.8 < cu130 所需 580，必须升驱动"，
**该判断不完整**——升驱动不是唯一解。

**根因（NVIDIA 容器驱动分层）**：

| 层 | 所在位置 | 是否在镜像内 |
| --- | --- | --- |
| 内核态驱动 `nvidia.ko` | 宿主内核（容器共享） | 不可能有 |
| 用户态驱动 `libcuda.so` / `libnvidia-ml.so` / `nvidia-smi` | **宿主**，由 nvidia-container-toolkit 在 `--gpus all` 时注入 | 默认没有 |
| CUDA runtime（`libcudart` / cuBLAS / NCCL） | 镜像 | 有 |

所以**容器内的 `libcuda` 版本恒等于宿主驱动版本**，与镜像 tag 里的 `cuda13.0.3` 无关。

**现场实测数据（worker-gpu-a）**：

```
cuda-compat-13-0            580.178.04-1ubuntu1   （镜像内已安装）
/usr/local/cuda/compat/libcuda.so.580.178.04       （镜像自带）
/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08    （宿主注入）
nvidia-smi 报告驱动          570.172.08
原生 torch.cuda.is_available                  → False
LD_LIBRARY_PATH=<compat> 后 torch.cuda.is_available → True
```

**解法**：把 `/usr/local/cuda/compat` 前置到 `LD_LIBRARY_PATH`，让用户态走 580、
内核态保持 570 —— 即 NVIDIA 的 **forward compatibility**。

**官方依据**（CUDA Compatibility 文档 Table 3，`cuda-compat-13-0` 行）：

| forward-compat 包 | 535+(12.2) | 550+(12.4) | 570+(12.8) | 580+(13.0) |
| --- | --- | --- | --- | --- |
| `cuda-compat-13-0` | C | C | **C** | N/A |
| `cuda-compat-12-9` | C | C | C | X |

C=Compatible，X=不兼容。570 分支在兼容范围内，实测吻合。

**落点**：`scripts/lib.sh:ensure_cuda_compat()` 在 source 时自动判定
（宿主驱动大版本 < 580 且找到 compat 目录 → 自动前置 `LD_LIBRARY_PATH`），
`00_preflight.sh` 在 CUDA 不可用时用 `CUDA_COMPAT=force` 再兜一次。
可用 `CUDA_COMPAT=off|auto|force` 控制。

**两个易错点**：

1. **不能用容器内 `nvidia-smi` 判断 compat 是否生效**：`nvidia-smi` 走 NVML
   （`libnvidia-ml.so`，仍是宿主注入的 570），启用 compat 后它照样报 570。
   唯一可信的验证是 `torch.cuda.is_available()`。
2. **compat 模式下有功能限制**（官方 Table 4）：CUDA-OpenGL/Vulkan 互操作不可用，
   但 `cuMemMap` 系列可用。训练只走 compute，不受影响。

---

## 7. 最终选型与现场配置（用户确认，2026-11-13 追加）

| 决策项 | 用户确认 | 项目落点 |
| --- | --- | --- |
| 镜像 | 官方 swift4.5.3 镜像，且**该镜像已在集群各节点拉取** | `RUN_IMAGE` / `OFFICIAL_453_IMAGE`（`config/image.env`） |
| 构建派生镜像 | **不需要**（相关脚本与 Dockerfile 已从项目移除） | 镜像只通过 `RUN_IMAGE` 指定 |
| 容器引擎 | **nerdctl**（namespace=default） | `CONTAINER_ENGINE=nerdctl`，`CTR_CMD=(nerdctl --namespace default)` |
| 权重路径 | 可指定 | `config/cluster.env`：`MODEL_ROOT` / `MODEL_27B_DIR` / `MODEL_FLASH_NEXT_DIR` |
| GPU 台数 | 可指定 | `GPUS_PER_NODE` × `NNODES`，`CUDA_VISIBLE_DEVICES` 按卡数自动生成 |
| 节点 IP | 可指定 | `NODE_IPS`（逗号分隔，顺序即 `NODE_RANK`）、`MASTER_ADDR` / `MASTER_PORT` |
| 拉取镜像 | 已拉取，`SKIP_PULL=1` | `00_preflight.sh` 内部的 `probe_image.py` 直接判能力；`13_multinode.sh --check` 核验镜像存在 |

### 新增的现场配置边界

`config/cluster.env` 由 **`scripts/lib.sh` 最先加载**（顺序：cluster.env → env.sh → image.env）。
因为 `env.sh` / `image.env` 全部用 `${VAR:-default}`，现场值优先于项目默认值；
若放在最后加载，`${VAR:-...}` 会保留 env.sh 已写入的默认值而失效（首轮自检实测到
`MODELSCOPE_CACHE` 仍是项目内路径、`NPROC_PER_NODE` 仍是 8），故顺序是本配置的关键。
`env.sh` 的 `mkdir -p` 对 `/mnt/shared/...` 这类本机不可写路径静默跳过，
保证 Windows 静态自检不被阻塞。容器内同样用 `${VAR:-...}` 保留 `12` 透传的 `-e` 值。

多机前提（写进 `13_multinode.sh` 与 README）：所有节点的项目目录 / 权重 / 缓存
**路径一致且共享**、镜像已拉取、`MASTER_PORT` 可达、ssh 免密；`NODE_IPS` 个数必须等于
`NNODES`；`MASTER_ADDR` 未填时取 `NODE_IPS` 第一个 IP。

> 关于"该镜像已拉取"的口径：项目按**官方 swift4.5.3 镜像**已拉取处理。
> 若现场实际拉取的仍是 swift4.4.1，只需重新拉取/改 `RUN_IMAGE` tag；本项目不做 pip 补装。
> `13_multinode.sh --check` 会在每个节点核验 `RUN_IMAGE` 是否真实存在。


