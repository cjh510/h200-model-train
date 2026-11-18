#!/usr/bin/env bash
# =============================================================================
# 全局配置：所有脚本经 scripts/lib.sh 间接 source 本文件。
#
# 【运行环境】训练必须在 Linux + NVIDIA GPU 集群上执行；本机 Windows 无 GPU，
# 只用于编写、静态校验与权重清点。所有变量都可用环境变量覆盖，例如：
#   MODEL_ROOT=/data/models NPROC_PER_NODE=8 bash scripts/00_preflight.sh
#
# 【证据】本机 D:\model\qwen 已存在完整 Qwen3.8-27B 快照
#   （18 个 safetensors / 55,563,006,776 B，config.json = qwen3_5）。
#   把它挂载或拷贝到集群后，用 MODEL_27B_DIR 指向该目录。
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# --- 模型权重路径 -----------------------------------------------------------
export MODEL_ROOT="${MODEL_ROOT:-/mnt/shared/models}"
export MODEL_27B_DIR="${MODEL_27B_DIR:-$MODEL_ROOT/Qwen3.8-27B}"
export MODEL_FLASH_NEXT_DIR="${MODEL_FLASH_NEXT_DIR:-$MODEL_ROOT/Qwen3.8-Flash-Next}"

# --- 数据集 -----------------------------------------------------------------
# swift/chinese-c4：96 个 .jsonl.zst，压缩 21.30 GiB / 原始 46.29 GB，字段 text。
export DATASET_ID="${DATASET_ID:-swift/chinese-c4}"
export DATASET_FULL_SPEC="${DATASET_FULL_SPEC:-$DATASET_ID}"
# 冒烟子集由 02_smoke.sh 在缺失时从 RAW_DATA_DIR 自动抽取（默认 2000 条），
# 走本地文件、关闭 streaming，可复现。
export SMOKE_TEXT_JSONL="${SMOKE_TEXT_JSONL:-$PROJECT_ROOT/data/smoke_text.jsonl}"
export SMOKE_MESSAGES_JSONL="${SMOKE_MESSAGES_JSONL:-$PROJECT_ROOT/data/smoke_messages.jsonl}"

# --- 缓存与产物 -------------------------------------------------------------
# 官方 Quick-start 明确要求：多机训练时 MODELSCOPE_CACHE 必须是共享存储，
# 否则各节点数据集缓存不一致会导致训练卡住。
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PROJECT_ROOT/data/modelscope_cache}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
export LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"

# --- 硬件 / 分布式 ----------------------------------------------------------
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# 单机（NNODES=1）时 scripts/lib.sh 会把 MASTER_ADDR 归一化成 127.0.0.1：
# 若 cluster.env 里填的是本机不可达的地址，rendezvous 会一直重试到 600s 超时，
# 表现为「日志停在 c10d 之后不动、无任何报错」。确实要用真实 IP（模拟多机首节点、
# 验证网卡绑定）时设 MASTER_ADDR_KEEP=1。
export MASTER_ADDR_KEEP="${MASTER_ADDR_KEEP:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- CUDA 前向兼容（宿主驱动 < 580 时走镜像自带的 cuda-compat-13-0） --------
# 容器里的用户态驱动 libcuda.so 由 nvidia-container-toolkit 从宿主注入，
# 版本恒等于宿主驱动，与镜像的 CUDA 版本无关。宿主停在 r570（12.8）分支时，
# cu130 的 torch 会 is_available=False —— 此时把镜像自带的
# /usr/local/cuda/compat（580 用户态 libcuda）前置到 LD_LIBRARY_PATH 即可。
# 实现见 scripts/lib.sh:ensure_cuda_compat，被 source 时自动判定一次。
#   auto  仅当注入的 libcuda < CUDA_COMPAT_MIN_DRIVER 且找到 compat 目录时启用
#   force 无条件启用（auto 判定失误时的手动兜底）
#   off   禁用
export CUDA_COMPAT="${CUDA_COMPAT:-auto}"
export CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-}"                  # 留空=自动探测 /usr/local/cuda/compat
export CUDA_COMPAT_MIN_DRIVER="${CUDA_COMPAT_MIN_DRIVER:-580}"

# --- 版本门槛（官方支持表 / 最佳实践，见 docs/AUDIT.md） --------------------
# 文档为 swift 4.7.0.dev0，但 Qwen3.8 支持从 4.5.3 起就有（首个正式版）。
# 本项目运行镜像为官方 swift 4.5.3，满足门槛；镜像 tag 见 config/image.env。
export MIN_MS_SWIFT="${MIN_MS_SWIFT:-4.5.3}"
export MIN_SWIFT_FOR_QWEN38="${MIN_SWIFT_FOR_QWEN38:-4.5.3}"
export MIN_MCORE_BRIDGE="${MIN_MCORE_BRIDGE:-1.6.5}"
export MIN_TRANSFORMERS_27B="${MIN_TRANSFORMERS_27B:-5.9}"
export MIN_TRANSFORMERS_FLASH_NEXT="${MIN_TRANSFORMERS_FLASH_NEXT:-5.16}"
export MIN_FLA_27B="${MIN_FLA_27B:-0.4.2}"
export MIN_FLA_FLASH_NEXT="${MIN_FLA_FLASH_NEXT:-0.5.2}"

# --- 训练规模与时长（03 使用） ----------------------------------------------
# Megatron 只有「步数上限」和「墙钟上限」两种收口方式，谁先到谁生效：
#   TRAIN_ITERS     → --train_iters，按步数停
#   DURATION_MINS>0 → --exit_duration_in_mins，按墙钟停
# 因此按时长跑必须同时给足步数上限，否则步数先耗尽、提前退出。
# 下面这个 CAP 只在 DURATION_MINS>0 且用户没显式设 TRAIN_ITERS 时生效。
export DURATION_MINS="${DURATION_MINS:-0}"                 # 0 = 按步数；>0 = 跑满 N 分钟
export DURATION_ITERS_CAP="${DURATION_ITERS_CAP:-100000}"  # 时长模式下的步数上限

# --- 日志指标 ---------------------------------------------------------------
# Megatron 的吞吐字段只在 --log_throughput 打开时才打印：
#   throughput per GPU (TFLOP/s/GPU): ...
# 注意该值基于 Megatron 自己对 FLOPs 的估算，对 GDN/MoE/ViT 架构仅作参考。
export LOG_THROUGHPUT="${LOG_THROUGHPUT:-true}"
# 日志打印间隔（步）。Megatron 的 `iteration N/M | ... | elapsed time per iteration`
# 行按这个节奏输出，05_verify 的步时 / tokens/s 汇总全靠它。**设得比总步数还大会
# 导致一行都不打印**——冒烟只跑 2 步时必须是 1，否则日志里看不到任何训练指标。
export LOGGING_STEPS="${LOGGING_STEPS:-1}"

# --- GPU 采样（nvidia-smi dmon，宿主机侧，由 12 启停） ----------------------
# DMON=1 时训练全程采集功耗/温度/SM利用率/显存/PCIe-NVLink，落 logs/dmon_*.csv，
# 结束后按卡汇总峰值与均值。跑在宿主机：镜像内不一定有 nvidia-smi。
export DMON="${DMON:-1}"
export DMON_INTERVAL="${DMON_INTERVAL:-5}"     # 采样间隔（秒）；24h×8 卡约 10MB
export DMON_METRICS="${DMON_METRICS:-pucvmt}"  # p功耗温度 u利用率 c时钟 v违规 m显存 t PCIe/NVLink

# --- 行为开关 ---------------------------------------------------------------
export MODEL_PROFILE="${MODEL_PROFILE:-27b}"   # 27b | flash_next
export DRY_RUN="${DRY_RUN:-0}"                 # 1 = 只打印命令，不执行

# cluster.env 可能把 MODELSCOPE_CACHE 指到 /mnt/shared/...；本机 Windows 静态自检时
# 该路径不存在/不可写，静默跳过即可，避免阻塞 source（集群上目录由现场保证存在）。
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" 2>/dev/null || true
mkdir -p "$MODELSCOPE_CACHE" 2>/dev/null || true
