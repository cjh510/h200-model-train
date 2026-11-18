# 继续预训练测试计划（用例矩阵）

> 目标不是"把命令跑起来"，而是**分层证伪**：先证明新架构能被 Megatron 正确加载与反向，
> 再证明真实数据能喂进去，最后才做长一点的训练与断点续训。
> 每个用例都有明确的通过标准与失败后的下一步，避免把不兼容误判成调参问题。

## 用例总表

| CASE | 名称 | 命令 | 输入 | 通过标准 | 失败时 |
| --- | --- | --- | --- | --- | --- |
| T0 | 环境与架构预检 | `bash scripts/00_preflight.sh 27b` | 官方清单 + 本地权重 | 驱动/CUDA 匹配、依赖版本达标、GPU 数够、权重 0 缺失 0 大小不符、config.json 架构匹配、CLI 参数已探测 | 先补依赖／升驱动／改路径，禁止跳过直接训练 |
| T1 | 冒烟子集（自动） | 无需单独执行，`02_smoke.sh` 会自动抽取 | 已下载的原始数据（`RAW_DATA_DIR`） | `data/smoke_text.jsonl`（默认 2000 条）与 `smoke_messages.jsonl` 被生成 | 检查 `RAW_DATA_DIR`、`zstandard`、磁盘 |
| T2 | 链路冒烟 | `bash scripts/02_smoke.sh 27b` | T1 子集 + 已有权重 | 2 iter 完成、loss 为有限值、产出 `checkpoint-*` 且含 safetensors | 按下方"失败分流表"定位 |
| T3 | 正式 CPT | `bash scripts/03_continue_pt.sh 27b` | 完整 `swift/chinese-c4` | `train_iters` 跑满（或 `DURATION_MINS` 到点）、loss 稳定、`--save_steps` 落盘；日志含 `throughput per GPU` 行；`logs/dmon_*.csv` 非空且每卡点数≈`时长/采样间隔` | 降 `MAX_LENGTH` / 开重计算 / 调并行 |
| T4 | 断点续训 | `bash scripts/04_resume.sh 27b` | T3 的 checkpoint | 从 checkpoint 恢复、iteration 接续、优化器状态可恢复 | 检查 `RESUME_MODE` 与 `pt_help_*.txt` |
| T5 | 产物与推理 | `RUN_INFER=1 bash scripts/05_verify.sh 27b` | T3/T4 产物 | checkpoint 含 safetensors、日志无 NaN、步时/吞吐汇总有输出、单条推理可出文本 | 检查是否 LoRA adapter（需带 `--adapters`） |

用例按顺序手跑即可（T1 已并入 T2，无需单独执行）。

## 两个模型的差异（同一个脚本，不同档案）

| 项 | 27B（`config/model_27b.env`） | Flash-Next（`config/model_flash_next.env`） |
| --- | --- | --- |
| 默认训练类型 | `full`（全参 CPT） | `lora`（官方只给 8 卡 LoRA 配置） |
| 并行 | TP4 / PP1 / CP1 → 8 卡时 DP2 | TP2 / EP4 / ETP1 / PP2 → 8 卡时 DP2 |
| `max_length` | 8192（冒烟 2048） | 8192（冒烟 2048） |
| `global_batch_size` | 16（冒烟 8） | 8 |
| 专属 | — | `PLE_CPU_OFFLOAD=1`、`--moe_permute_fusion true`、`--decoder_first_pipeline_num_layers 12` |
| 权重 | 51.75 GiB，本机已有 | 335.28 GiB，本机未发现 |

### 显存账（为什么 Flash-Next 默认不跑全参）

| 模型 | bf16 权重 | 全参训练状态（权重+梯度+fp32 master+Adam m/v，粗估） | 8×H200(141 GiB/卡) |
| --- | --- | --- | --- |
| 27B | 51.75 GiB | 约 414 GiB，TP4 分片 + DP2 的 distributed optimizer 后约 65 GiB/卡 + 激活 | **可行**，这也是 27B 默认 `full` 的依据 |
| Flash-Next | 335.28 GiB（含 51B N-gram 表 ≈95 GiB） | 约 2 TiB 量级，8 卡约 250 GiB/卡 | **不可行**；需 ≥2 机或优化器 CPU offload，故默认 LoRA |

官方实测（LoRA rank8、TP2/EP4/PP2、seq8192、PLE offload）：显存 **77.7 GiB/卡**；
不开 PLE offload 时 93.3 GiB/卡。冒烟用 seq2048 会更宽松。

## T2 失败分流表（证据 → 根因 → 动作）

| 现象 | 最可能根因 | 动作 |
| --- | --- | --- |
| `KeyError` / `NotImplementedError` 含 `qwen4_exp`、`GDN`、`PLE`、`mtp` | 框架 / mcore-bridge 版本落后于模型 | `pip install -U ms-swift mcore-bridge transformers`；确认达到 `docs/AUDIT.md` 的版本门槛 |
| `transformers` 报 config 字段不认识 | transformers < 5.16（Flash-Next） | 升级 transformers，不要改模型 config |
| 权重加载报 shape/size mismatch | 权重快照与 config 不同版本 / 分片缺失 | 用 `verify_weights.py --check-sha` 全量校验；同步 tokenizer 配置 |
| 纯文本数据报 `messages` / `images` 缺失 | `megatron pt` 对 VL 模型走了对话 template | 用 sft 回退：`MEGATRON_SUBCMD=sft DATASET_SPEC=data/smoke_messages.jsonl PACKING=false EXTRA_ARGS='--freeze_vit true --freeze_aligner true' bash scripts/03_continue_pt.sh 27b` |
| CUDA OOM（27B） | seq/MBS 过大或 DP=1 无优化器分片 | 先降 `MAX_LENGTH=2048`、`MICRO_BATCH_SIZE=1`；加 `--recompute_granularity full`；必要时 `EXTRA_ARGS='--optimizer_cpu_offload true --use_precision_aware_optimizer true'` |
| CUDA OOM（Flash-Next） | 忘记 `PLE_CPU_OFFLOAD=1` 或用了全参 | 确认档案里 `PLE_CPU_OFFLOAD=1`；保持 `TUNER_TYPE=lora` |
| loss = NaN | lr 过大 / bf16 溢出 / 数据异常 | 降 `LR`、检查 `data/smoke_text.jsonl` 内容、`grep NaN` 日志 |
| 卡在数据加载 | 多机 `MODELSCOPE_CACHE` 不共享 | 按官方要求把缓存放共享存储；冒烟阶段用本地子集可绕过 |

## 判定口径

- **CPT 起点 vs 续训**：T3 用 `--finetune true`（读权重、重置 iteration/优化器）；T4 用 `--finetune false`（连优化器状态一起恢复）。两者语义不同，不要混用。
- **权重产物**：`--save_safetensors true` 直接产出 HF 格式；若 T4 需要 mcore 续训，03 里已设 `NO_SAVE_OPTIM=false`，会额外保存 mcore 权重。
- **有效性证据**：不看"命令退出码 0"就算通过，必须有 loss 数值 + checkpoint 文件 + 可选推理输出三者之一为证。
- **未在 Windows 本机实跑**：本机无 GPU，所有 shell 脚本只做 `bash -n` 静态校验；真实判定以集群日志为准。

---

## 镜像能力判定（已并入 T0）

> 要求：ms-swift **≥ v4.5.3**（Qwen3.8 支持从该版本起才有）。`00_preflight.sh` 内部的
> `tools/probe_image.py` 会做版本矩阵 + 源码注册扫描，不满足直接 `die`；
> 不需要单独跑镜像探针脚本。

| CASE | 命令 | 通过标准 | 失败时 |
| --- | --- | --- | --- |
| T0 | `bash scripts/12_run_in_container.sh scripts/00_preflight.sh 27b` | 驱动/CUDA 可用、依赖版本达标、权重完整、`probe_image.py` 判 **SUPPORTED**（`qwen3_8` / `qwen4_exp` 命中 >0） | 见下方分流 |

### 判定口径

| 检查项 | 通过条件 | 为什么 |
| --- | --- | --- |
| `swift` 版本 | 数值 ≥ 4.5.3 | v4.4.1 / v4.4.3 不含 Qwen3.8 注册 |
| swift 包源码 `qwen4_exp` 命中 | > 0 | Flash-Next 的 model_type 注册 |
| swift 包源码 `qwen3_8` 命中 | > 0 | Qwen3.8 模板 / 注册 |
| transformers 包 `Qwen3_5ForConditionalGeneration` | > 0 | 27B 架构类 |
| transformers 包 `Qwen4ExpForConditionalGeneration` | WARN（不阻断） | 部分版本按需动态导入 |
| `transformers >= 5.16` / `fla` / `causal_conv1d` | WARN 汇总 | 训练前必须齐备（27B 只要 transformers≥5.9）；`pip check` 结果入报告 |
| vLLM | 仅 WARN | 本项目不用 vLLM 后端 |

### 失败分流

| 现象 | 根因 | 动作 |
| --- | --- | --- |
| `torch.cuda.is_available=False` 但 nvidia-smi 有卡 | 容器内 `libcuda` 由宿主注入（版本=宿主驱动），r570 撑不起镜像的 cu130 | 确认 `CUDA_COMPAT=auto` 且镜像有 `/usr/local/cuda/compat`；否则升驱动到 580+（见 AUDIT 6.6） |
| `megatron pt --help` 超时 | CUDA 初始化阻塞 | 先确认 `torch.cuda.is_available()` 为 True；`PT_HELP_TIMEOUT` 可调 |
| 判 `UNSUPPORTED` + `swift=4.4.1` | `RUN_IMAGE` 指到了旧 tag | 改回官方 4.5.3 tag；不要靠改 `--model` 绕过 |
| `swift=4.5.3` 但 `qwen4_exp=0` | 镜像内包被 pip 覆盖 | 重新拉取镜像，不要就地升级 |
| `nerdctl: image not found` 但 `images` 里能看到 | load/run namespace 不一致 | 全部命令固定 `--namespace default` |
| 容器内看不到权重 / 缓存 | 宿主目录没 bind 回相同路径 | `MODEL_ROOT_BIND=1 CACHE_BIND=1`（默认开） |
| `fla` / `causal_conv1d` import 失败 | 镜像不完整 | 重新拉取镜像 |
| `--help` 里没有 `--load` | 版本差异 | T4 用 `RESUME_MODE=mcore` 或 `RESUME_ARGS` |

### 执行顺序

```
[12 包装] T0 → T2 → T3 → T4 → T5
```
T1（抽冒烟子集）已并入 T2，缺失时自动生成。

---

## 现场执行顺序（nerdctl + 官方 swift4.5.3 镜像）

| 步骤 | 命令 | 通过标准 |
| --- | --- | --- |
| S0 | 编辑 `config/cluster.env`（权重路径、GPU 台数、节点 IP、master） | 所有节点项目/权重/缓存路径一致且共享 |
| S1 | `bash scripts/13_multinode.sh --check` | 每个节点 ssh 通、目录存在、`RUN_IMAGE` 在 `nerdctl --namespace default images` 里 |
| S2 | `bash scripts/12_run_in_container.sh scripts/00_preflight.sh 27b` | 驱动/CUDA 可用、判定 **SUPPORTED**、权重 0 缺失 |
| S3 | `CONTAINER_DRY_RUN=1 bash scripts/12_run_in_container.sh scripts/03_continue_pt.sh 27b` | 打印的 `nerdctl ... run` 命令里挂载与 `-e` 正确 |
| S4 | `DRY_RUN=1 bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b` | 每个节点一条命令，`NODE_RANK` 从 0 递增、`MASTER_ADDR` 一致 |
| S5 | `bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b` | 各节点 rc=0，日志在 `logs/multinode/<时间戳>/` |
| S6 | 换 `flash_next` 复跑 T2/T3 | 两模型都通过冒烟 |

### 单机退化

`NNODES=1` 或 `NODE_IPS` 为空时，`13_multinode.sh` 直接调用 `12_run_in_container.sh`，
不启 ssh —— 便于先在单节点跑通 T0~T2，再扩到多机。

### 多机失败分流

| 现象 | 根因 | 动作 |
| --- | --- | --- |
| T0-I 判 UNSUPPORTED | `RUN_IMAGE` 不是 4.5.3，或镜像实际未拉取 | `13_multinode.sh --check` 核验；把 `RUN_IMAGE` 改成官方 4.5.3 tag |
| `--check` 某节点 FAIL | 目录缺失 / 镜像未拉 / ssh 未免密 / namespace 不一致 | 按 `SKIP_PULL=1` 前提补齐；所有引擎命令固定 `--namespace default` |
| `NODE_IPS 个数 != NNODES` | 配置不一致，脚本主动 die | 对齐两者；IP 顺序即 NODE_RANK |
| 只有 node0 起来、其余卡住 | `MASTER_ADDR=127.0.0.1` 未改 | 填 master IP；未填时脚本会自动取第一个 IP |
| 各节点数据进度不一致 | `MODELSCOPE_CACHE` 不是共享存储 | 指向共享目录（Quick-start 明确警告会卡住） |
| 容器内读不到权重 | 宿主共享目录未 bind 回相同路径 | `MODEL_ROOT_BIND=1 CACHE_BIND=1`（默认开） |


