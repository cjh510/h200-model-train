#!/usr/bin/env bash
# =============================================================================
# Nemotron-H-56B 复现训练脚本（E2E Networks 49% MFU 基准的本地适配版）
#
# 参考：https://www.e2enetworks.com/blog/multi-node-megatron-lm-training-slurm-h200
#       原文 = 2 节点 × 8 H200、Megatron-LM core_v0.15.2、NGC pytorch:25.08-py3、
#       Slurm + Pyxis/Enroot + /pfs + torchrun(c10d)，TP8 / PP1 / DP2 / GBS64 /
#       MBS1 / seq8192 / BF16，调优后 487~489 TFLOP/s/GPU、49% MFU。
#
# 本脚本只保留原文的「测试参数与脚本结构」，把运行层换成你的实际环境：
#   镜像 nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3 (torch 2.6.0a0 / CUDA 12.8 /
#   NCCL 2.25.1 / TE 1.14) + Megatron-LM core_v0.13.1 源码包（zip 解到
#   $ROOT/Megatron-LM，PYTHONPATH 方式，不 pip install）+ docker/nerdctl + torchrun。
#   依据：本机 2026-11-10 run 的 03-deploy/env.txt 与 megatron_args.txt（真实 --help）。
#
# -----------------------------------------------------------------------------
# 差异对照（「原文」→「本脚本」），每条都在下面代码里有对应处理
# -----------------------------------------------------------------------------
# | 项            | 原文 core_v0.15.2 / 25.08      | 本脚本 core_v0.13.1 / 25.01    |
# |---------------|---------------------------------|--------------------------------|
# | 调度/容器     | sbatch + srun + Pyxis .sqsh     | 每节点各跑一次；docker/nerdctl   |
# |               |                                 | --network host（也认 SLURM_*）  |
# | 共享存储      | /pfs/nemotron-h                 | $ROOT 挂到 /workspace，          |
# |               |                                 | 产物在 /workspace/e2e-56b       |
# | 源码          | git clone core_v0.15.2 + make   | 已解好的 core_v0.13.1 源码包    |
# | rendezvous    | --rdzv-backend=c10d             | 默认静态 --master_addr/--       |
# |               |                                 | node_rank；RENDEZVOUS=c10d 可切 |
# | IB 网卡       | NCCL_IB_HCA 写死 8 条 IB rails  | 不写死；本机是 RoCE              |
# |               | (=mlx5_0:1,...:mlx5_11:1)      | mlx5_bond_*，用 NCCL_HCA 覆盖   |
# | Gloo          | 无                              | 双机必加                        |
# |               |                                 | --disable-gloo-process-groups   |
# |               |                                 | （容器内无对端 DNS，已实测）    |
# | attention     | --attention-backend flash       | 默认 auto（0.13.1 枚举取值      |
# |               | + NVTE_FUSED_ATTN=0             | flash/fused/unfused/local/auto；|
# |               |                                 | TE 1.14 自适应更稳）            |
# | save-retain   | --save-retain-interval 840      | 0.13.1 无此 flag → 脚本内       |
# |               |                                 | ckpt_roll_once() 实现：只留最近 |
# |               |                                 | CKPT_KEEP 套（默认 1），新的    |
# |               |                                 | 完整落盘后删旧的                |
# | FP8           | --fp8-format hybrid            | 本机验证过的是                  |
# |               |                                 | e4m3 + --fp8-recipe delayed     |
# | --use-mcore-  | 正常                            | 0.13.1 标 DEPRECATED 但仍接受   |
# |  models       |                                 | 保留，保证与原文一致            |
# | 其它 79 个    | 同                              | 全部在 0.13.1 --help 中存在，   |
# | flag          |                                 | 启动前自动按 --help 裁剪        |
# -----------------------------------------------------------------------------
#
# 【前置条件】两台机器都要：8 张 H200、同一镜像 digest、$ROOT/Megatron-LM/pretrain_gpt.py、
#             同一份 $ROOT 内容、29500 端口互通。
#
# 【共享盘】两机跑只有一样东西必须共享，但它决定成败：data_cache 与 checkpoints。
#   data_cache  —— MockGPTDataset 的索引只由 **global rank 0**（永远是 node0）生成，
#                  barrier 之后其它 rank 去**自己盘上**读；node1 没有 builder，
#                  冷缓存必崩：FileNotFoundError → "MockGPTDataset failed to build ..."
#   checkpoints —— torch_dist 每个 rank 写自己的分片；两机各写本地盘 → 目录不完整
#   tensorboard —— 只有 rank0 写，放本地也行
# 准备（node0 当 server，node1 挂客户端）：
#   # node0
#   mkdir -p /mnt/shared/e2e-56b && chmod 777 /mnt/shared/e2e-56b
#   echo '/mnt/shared/e2e-56b  <node1_ip>(rw,sync,no_subtree_check,no_root_squash)' >> /etc/exports
#   exportfs -ra && systemctl enable --now nfs-server
#   # node1（hard 保证不半途静默失败；vers=4.1 + 1M 读写块）
#   mkdir -p /mnt/shared/e2e-56b
#   mount -t nfs -o vers=4.1,rsize=1M,wsize=1M,hard,timeo=600,retrans=2,_netdev \
#         <node0_ip>:/mnt/shared/e2e-56b /mnt/shared/e2e-56b
# 两台都这么跑（SHARED_ROOT 必须一致）：
#   SHARED_ROOT=/mnt/shared/e2e-56b MASTER_ADDR=<node0_ip> NODE_RANK=0 bash run_56b_2x8h200.sh
#   SHARED_ROOT=/mnt/shared/e2e-56b MASTER_ADDR=<node0_ip> NODE_RANK=1 bash run_56b_2x8h200.sh
#   脚本会把 data_cache/checkpoints/tensorboard 自动 bind 回容器内的
#   /workspace/e2e-56b/{data_cache,checkpoints,tensorboard}，Megatron 参数不用改。
# 网络盘慢：一套 0.8 TB 经 25GbE 约 7~13 分钟，记得 CKPT_KEEP_GRACE=900（默认 300s 偏紧）。
#
# 【用法】两机同时启动（torchrun 静态 rendezvous 会互等）：
#   # node-a（rank0），node-b 用 NODE_RANK=1、MASTER_ADDR 填 node-a 的 IP
#   MASTER_ADDR=192.0.2.20 NODE_RANK=0 bash run_56b_2x8h200.sh
#   MASTER_ADDR=192.0.2.20 NODE_RANK=1 bash run_56b_2x8h200.sh
#
#   # 先看完整命令行，不启动容器（本机可离线验证参数）
#   DRY_RUN=1 ARGS_HELP=/path/to/megatron_args.txt bash run_56b_2x8h200.sh
#   # 冒烟：50 步、不写 checkpoint、不限时
#   SMOKE=1 MASTER_ADDR=192.0.2.20 NODE_RANK=0 bash run_56b_2x8h200.sh
#   # 只跑单机 8 卡（DP=1，用于先验证模型能起来）
#   NNODES=1 MASTER_ADDR=127.0.0.1 NODE_RANK=0 SMOKE=1 bash run_56b_2x8h200.sh
#   # Slurm 下用 srun 拉起（每节点一个 task，脚本自动读 SLURM_NODEID/NNODES）
#   sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --wrap='srun bash run_56b_2x8h200.sh'
#
# 【可调项】全部是「环境变量 + 默认值」，与原文脚本同风格；常用：
#   TP / PP / MBS / GBS / SEQ / TRAIN_ITERS / EXIT_DURATION_MINS / SAVE_INTERVAL
#   ATTN_BACKEND / TP_COMM_OVERLAP / FP8 / DEBUG_LOGGING / ENABLE_CKPT
#   CKPT_KEEP（默认 1 = 只留最新一套，存新删旧）/ CKPT_KEEP_GRACE / CKPT_TB
#   NCCL_HCA / NCCL_SOCKET_IFNAME / NCCL_DEBUG / IMAGE / ROOT / RENDEZVOUS
#   DMON / DMON_INTERVAL / DMON_METRICS（nvidia-smi dmon 采样，见 9.5 节）
#
# 【日志命名】$LOGDIR_HOST/<node N 子目录，共享盘时自动加>/nemotron56b_tp.._.._node<N>-<时间戳>.out
#   文件名永远带 _node<N>；若 logs/ 在共享盘上，还会再分一层 node0/ node1/（见 6.5 节，
#   LOG_PER_NODE=auto|1|0 可强制）。
#   nemotron56b_tp8_pp1_dp2_mbs1_s8192_gbs64_bf16_node0-20261111_143021.out
#   两机日志混在一起时，光看文件名就能确认是不是同一个实验；dmon CSV 同前缀加 -dmon。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- 0. 路径与镜像
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="${ROOT:-/root/llm-bench}"                       # 宿主上挂进容器的根目录
WORKSPACE_HOST="${WORKSPACE_HOST:-$ROOT/e2e-56b}"     # 宿主：日志/ckpt/tb/cache
WORKSPACE_IN="${WORKSPACE_IN:-/workspace/e2e-56b}"    # 容器内：同一份目录
MEGATRON_IN="${MEGATRON_IN:-/workspace/Megatron-LM}"  # core_v0.13.1 源码（PYTHONPATH）
IMAGE="${IMAGE:-nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3}"
# 共享盘：两节点同一路径的共享文件系统（NFS/Lustre/GPFS/CephFS…）。
# 给了 SHARED_ROOT 就把 data_cache / checkpoints / tensorboard 放过去（见文件头【共享盘】）。
# 单独覆盖优先级更高：CHECKPOINT_HOST=... 会盖掉 $SHARED_ROOT/checkpoints。
SHARED_ROOT="${SHARED_ROOT:-}"
CHECKPOINT_HOST="${CHECKPOINT_HOST:-${SHARED_ROOT:+$SHARED_ROOT/checkpoints}}"
CHECKPOINT_HOST="${CHECKPOINT_HOST:-$WORKSPACE_HOST/checkpoints}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-${SHARED_ROOT:+$SHARED_ROOT/tensorboard}}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-$WORKSPACE_HOST/tensorboard}"
DATA_CACHE_HOST="${DATA_CACHE_HOST:-${SHARED_ROOT:+$SHARED_ROOT/data_cache}}"
DATA_CACHE_HOST="${DATA_CACHE_HOST:-$WORKSPACE_HOST/data_cache}"
LOGDIR_HOST="${LOGDIR_HOST:-$WORKSPACE_HOST/logs}"   # 日志留在本地盘即可
CHECKPOINT_IN="$WORKSPACE_IN/checkpoints"
TENSORBOARD_IN="$WORKSPACE_IN/tensorboard"
DATA_CACHE_IN="$WORKSPACE_IN/data_cache"
ARGS_HELP="${ARGS_HELP:-}"   # 默认值在 NODE_RANK 确定后才给（见第 8 节前，文件名带节点号）
RUN_ID="${RUN_ID:-}"                       # 留到拓扑确定后再给默认值（见第 1 节）
RUN_STAMP=$(date +%Y%m%d_%H%M%S)          # 只用于日志文件名，不进 rendezvous id

mkdir -p "$CHECKPOINT_HOST" "$TENSORBOARD_HOST" "$DATA_CACHE_HOST" "$LOGDIR_HOST"

# 写权限探针：共享盘没挂上 / exports 少了 no_root_squash 时，这里 10 秒内报清楚，
# 而不是等 40 秒后 rank8 崩在 numpy.load。容器内进程是 root，NFS 端必须 no_root_squash。
for _d in "$CHECKPOINT_HOST" "$TENSORBOARD_HOST" "$DATA_CACHE_HOST" "$LOGDIR_HOST"; do
  if ! ( : > "$_d/.wtest.$$" ) 2>/dev/null; then
    echo "!! [node${NODE_RANK:-?}] 目录不可写：$_d" >&2
    echo "   共享盘没挂载 / 权限不对 / NFS 少了 no_root_squash（容器里以 root 写）" >&2
    exit 2
  fi
  rm -f -- "$_d/.wtest.$$" 2>/dev/null || true
done

# 容器只挂了 $ROOT。上面三个目录若被指到 $ROOT 之外（典型就是共享盘），必须再 bind 回
# 容器内的原路径；否则 --save / --data-cache-path / --tensorboard-dir 指向的是容器里的空目录。
# 嵌套 bind 是合法的：后挂的覆盖先挂的对应子树，$ROOT 其它部分不受影响。
EXTRA_MOUNTS=()
add_extra_mount() {   # $1=宿主路径 $2=容器内路径
  case "$1" in "$ROOT"|"$ROOT"/*) return 0 ;; esac
  EXTRA_MOUNTS+=( -v "$1:$2" )
  echo "   bind 共享盘：$1 → $2"
}
add_extra_mount "$DATA_CACHE_HOST" "$DATA_CACHE_IN"
add_extra_mount "$CHECKPOINT_HOST" "$CHECKPOINT_IN"
add_extra_mount "$TENSORBOARD_HOST" "$TENSORBOARD_IN"

# 引擎：目标服务器是 nerdctl，默认即 nerdctl（ENGINE=docker 可切回）。
# nerdctl 的所有命令都必须带同一个 --namespace，否则 load 与 run 会落进不同命名空间，
# 表现为「镜像明明 load 了，run 却说 not found」。沿用现有 lib.sh 的 default 约定，
# 拒绝 k8s.io / moby（会与集群业务镜像混用）。
ENGINE="${ENGINE:-nerdctl}"
# 强制 namespace=default：现场 nerdctl 的 CONTAINERD_NAMESPACE 可能被集群设成
# k8s.io/moby，而镜像是用 -n default 导入的；不带 -n 的 `nerdctl image inspect`
# 会报「镜像不存在」。显式 --namespace 优先级高于环境变量，且再 export 一次双保险。
# 确需换命名空间时用 NERDCTL_NAMESPACE=<ns> 覆盖（不要用 CONTAINERD_NAMESPACE）。
NS="${NERDCTL_NAMESPACE:-default}"
if [ -n "${CONTAINERD_NAMESPACE:-}" ] && [ "$CONTAINERD_NAMESPACE" != "$NS" ]; then
  echo "   ! 检测到 CONTAINERD_NAMESPACE=$CONTAINERD_NAMESPACE，已忽略，固定 namespace=$NS" >&2
fi
case "$ENGINE" in
  nerdctl)
    # DRY_RUN 允许在没有 nerdctl 的机器上离线打印命令
    if [ "${DRY_RUN:-0}" != 1 ]; then
      command -v nerdctl >/dev/null 2>&1 || { echo "!! 找不到 nerdctl"; exit 2; }
    fi
    ENGINE_ARGS=(--namespace "$NS")
    export CONTAINERD_NAMESPACE="$NS"
    ;;
  docker) ENGINE_ARGS=() ;;          # docker 没有 containerd namespace 概念
  *) echo "!! ENGINE 只能是 nerdctl | docker（收到 $ENGINE）"; exit 2 ;;
esac
ENGINE_CMD=("$ENGINE" "${ENGINE_ARGS[@]}")

# ---------------------------------------------------------------- 1. 拓扑/rendezvous
# Slurm 下由 srun 每节点拉起一次时，自动从 Slurm 变量推导
if [ -n "${SLURM_JOB_ID:-}" ]; then
  NNODES="${NNODES:-${SLURM_NNODES:-2}}"
  NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
  if [ -z "${MASTER_ADDR:-}" ] && command -v scontrol >/dev/null 2>&1; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  fi
fi
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RENDEZVOUS="${RENDEZVOUS:-static}"      # static | c10d（c10d 对齐原文，static 贴合本机）
WORLD=$((NNODES * GPUS_PER_NODE))        # 16
# c10d 要求两机 --rdzv_id 完全一致：不能用各自的时间戳，取拓扑+端口做确定性 id。
# Slurm 下用 SLURM_JOB_ID（与原文一致）；需要区分多次提交时显式给同一个 RUN_ID。
if [ -z "$RUN_ID" ]; then
  RUN_ID="${SLURM_JOB_ID:-nemotron56b-${NNODES}x${GPUS_PER_NODE}-p${MASTER_PORT}}"
fi

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"                      # 1 = 50 步、不保存、不加退出时长

if [ "$DRY_RUN" != 1 ]; then
  # GPU 数量断言：原文每节点 8 卡，少于 8 卡 TP=8 直接不成立
  gpu_n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  [ "${gpu_n:-0}" -eq "$GPUS_PER_NODE" ] || {
    echo "!! 本机 GPU 数 ${gpu_n:-0} != $GPUS_PER_NODE，退出"; exit 2; }
  [ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ] || {
    echo "!! 缺 $ROOT/Megatron-LM/pretrain_gpt.py（请先解包 Megatron-LM-core_v0.13.1.zip 并改名为 Megatron-LM）"; exit 2; }
  if [ "$ENGINE" = nerdctl ] && command -v systemctl >/dev/null 2>&1 \
     && ! systemctl is-active containerd >/dev/null 2>&1; then
    echo "!! containerd 未运行：systemctl enable --now containerd"; exit 2
  fi
  # 用 images 列表精确匹配，不用 image inspect：部分 nerdctl 版本没有该子命令，
  # 会把「命令不存在」误报成「镜像不存在」。grep 不加 -q，避免 pipefail 下的 SIGPIPE 误判。
  if ! "${ENGINE_CMD[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
       | grep -Fx "$IMAGE" >/dev/null; then
    echo "!! 在 $ENGINE (namespace=$NS) 的镜像列表里找不到 $IMAGE"
    echo "   nerdctl load 进 containerd 的镜像 docker 看不见，反之亦然；"
    echo "   确认用同一引擎、同一 namespace 导入（见 00_prepare.sh）"; exit 2
  fi
  if [ "$NNODES" -gt 1 ] && [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    echo "!! 双机必须显式给 MASTER_ADDR=<node-a 的 IP>"; exit 2
  fi
fi

# ---------------------------------------------------------------- 2. 容器内设备/环境
IB_DEVICES=()
if [ "$NNODES" -gt 1 ] && [ -d /dev/infiniband ]; then
  for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
    [ -c "$d" ] && IB_DEVICES+=(--device "$d")
  done
  [ "${#IB_DEVICES[@]}" -gt 0 ] || echo "   ! /dev/infiniband 无 uverbs 字符设备，双机 NCCL 可能退回 socket" >&2
fi

ENV_ARGS=(
  -e "PYTHONPATH=$MEGATRON_IN"
  -e "CUDA_DEVICE_MAX_CONNECTIONS=1"       # TP 通信重叠的硬要求，缺了启动断言会崩
  -e "TRITON_CACHE_DIR=/tmp/triton-cache"  # JIT cache 放节点本地，不落共享盘
  -e "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache"
  -e "TORCHINDUCTOR_WORKER_START=fork"
  -e "TORCH_CUDA_ARCH_LIST=9.0"            # 原文建议：缩短首次 checkpoint 的一次性编译
  -e "NVTE_FWD_LAYERNORM_SM_MARGIN=16"
  -e "NVTE_BWD_LAYERNORM_SM_MARGIN=16"
  -e "TORCH_NCCL_AVOID_RECORD_STREAMS=1"
  -e "TORCH_NCCL_HIGH_PRIORITY=1"
  -e "UCX_MEM_MMAP_HOOK_MODE=none"
  -e "UCX_MEM_CUDA_HOOK_MODE=none"
  -e "UCX_MEM_MALLOC_HOOKS=n"
  -e "UCX_ERROR_SIGNALS="
)

# ---------------------------------------------------------------- 3. 网络（RoCE 而非原文 IB）
# 原文的 =mlx5_0:1,...mlx5_11:1 是 E2E TIR 的 8 条 400G IB rails，本机是 RoCE
# mlx5_bond_*，写死会挑不到卡。默认不注入 NCCL_IB_HCA，交给 NCCL 自选；
# 需要精确绑定时用 NCCL_HCA（值可带前导 = 表示精确匹配）。
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"     # 例 bond0 / eth0；空 = 不注入
NCCL_HCA="${NCCL_HCA:-}"                         # 例 =mlx5_bond_2,...,mlx5_bond_9
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"                 # 首次跑留 INFO 看 NET/IB vs NET/Socket
[ -n "$NCCL_SOCKET_IFNAME" ] && ENV_ARGS+=(-e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
[ -n "$NCCL_HCA" ]          && ENV_ARGS+=(-e "NCCL_IB_HCA=$NCCL_HCA")
[ -n "$NCCL_DEBUG" ]        && ENV_ARGS+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
if [ "$NNODES" -gt 1 ]; then
  # 本机双机 RoCE 已验证的 QP 建链参数
  ENV_ARGS+=(-e "NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}" -e "NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}")
fi

# attention backend：0.13.1 的 --attention-backend 取 AttnBackend 枚举名
# {flash,fused,unfused,local,auto}；TE 1.14 下 auto 最稳，flash 需镜像自带 flash-attn。
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
if [ "$ATTN_BACKEND" = fused ]; then
  ENV_ARGS+=(-e "NVTE_FUSED_ATTN=1")
fi
# 注意：原文在 flash 时显式 NVTE_FUSED_ATTN=0（为强选 FA2）；TE 1.14 上不注入，
# 避免禁用 cuDNN fused 后又找不到 flash-attn。

# ---------------------------------------------------------------- 4. 并行 / batch（原文调优后的值）
TP="${TENSOR_PARALLELISM:-8}"     # PP=1 时 56B 模型必须 TP=8，TP=4 约 126GB/卡放不下
PP="${PIPELINE_PARALLELISM:-1}"   # DP = 16 / (8*1) = 2
MBS="${MICRO_BATCH_SIZE:-1}"
GBS="${GLOBAL_BATCH_SIZE:-64}"    # 每 DP rank 32 个 microbatch
SEQ="${SEQ:-8192}"                # 原文 8192；显存不够时第一个降的就是它
DP=$((WORLD / TP / PP))
[ $((DP * TP * PP)) -eq "$WORLD" ] || { echo "!! TP×PP 无法整除 world=$WORLD"; exit 2; }
[ $((GBS % (MBS * DP))) -eq 0 ]   || { echo "!! GBS=$GBS 不能被 mbs×dp=$((MBS * DP)) 整除"; exit 2; }

# ---------------------------------------------------------------- 5. 时长 / checkpoint
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
EXIT_DURATION_MINS="${EXIT_DURATION_MINS:-1410}"  # 23h30m 后写最终 ckpt 并退出
SAVE_INTERVAL="${SAVE_INTERVAL:-140}"              # 3600/25.5s ≈ 141，取整 140
ENABLE_CKPT="${ENABLE_CKPT:-0}"    # 默认关：两机须共享 checkpoint 目录，否则 torch_dist
                                   # 并行保存会各写各的，恢复时结构不一致
CKPT_TB="${CKPT_TB:-0.80}"         # 单套 checkpoint 体积（TB）。56.5B×(2B bf16权重
                                   # +4B fp32 master+8B Adam m/v)=14 B/参数 ≈ 0.79 TB
# 滚动保留：只留最近 CKPT_KEEP 套，新的一套完整落盘后删掉更老的（见 ckpt_roll_once）。
# 0.13.1 没有 --save-retain-interval，只能脚本外部做；这套是 24h 长跑不炸盘的唯一办法。
CKPT_KEEP="${CKPT_KEEP:-1}"                    # 1 = 只留最新一套；0 = 不删（只统计）
CKPT_KEEP_GRACE="${CKPT_KEEP_GRACE:-300}"      # 秒；目录 mtime 在此之内视为还在写，不删
CKPT_ROLL_POLL="${CKPT_ROLL_POLL:-60}"         # 秒；巡检间隔
CKPT_DISK_FORCE="${CKPT_DISK_FORCE:-0}"        # 1 = 空间不够也强行跑（不推荐）
if [ "$SMOKE" = 1 ]; then
  TRAIN_ITERS="${SMOKE_ITERS:-50}"
  EXIT_DURATION_MINS=""            # 冒烟不传 --exit-duration-in-mins
  ENABLE_CKPT=0
  echo ">> SMOKE 模式：${TRAIN_ITERS} 步、不保存 checkpoint、不限时"
fi

# ---------------------------------------------------------------- 6. 性能开关
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-1}"   # TE userbuffers 把 TP all-gather/RS 藏进 GEMM
FP8="${FP8:-0}"
DEBUG_LOGGING="${DEBUG_LOGGING:-0}"       # 1 = params-norm / num-zeros / straggler（约 1~2% 损耗）
CHECK_HASH="${CHECK_HASH:-1}"             # DP 副本权重哈希校验；双机若报 gloo 相关错误设 0
SEED="${SEED:-1234}"

echo "==================== Nemotron-H-56B / core_v0.13.1 / 25.01-py3 ===================="
echo "RUN_ID=$RUN_ID  engine=$ENGINE namespace=${NS:-none}  image=$IMAGE"
echo "NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD=$WORLD NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT RENDEZVOUS=$RENDEZVOUS"
echo "TP=$TP PP=$PP DP=$DP MBS=$MBS GBS=$GBS accum=$((GBS / (MBS * DP))) SMOKE=$SMOKE"
echo "ATTN_BACKEND=$ATTN_BACKEND TP_COMM_OVERLAP=$TP_COMM_OVERLAP FP8=$FP8 DEBUG_LOGGING=$DEBUG_LOGGING ENABLE_CKPT=$ENABLE_CKPT"
echo "TRAIN_ITERS=$TRAIN_ITERS EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-<none>} SAVE_INTERVAL=$SAVE_INTERVAL"
echo "CKPT: ENABLE_CKPT=$ENABLE_CKPT CKPT_KEEP=$CKPT_KEEP（留最近 N 套，0=不删）单套约 ${CKPT_TB} TB"
[ "$ENABLE_CKPT" = 1 ] && [ "$CKPT_KEEP" -gt 0 ] && \
  echo "      稳态占用 ≈ $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" 'BEGIN{printf "%.2f", k*t}') TB；" \
       "写新一套的那几分钟需要 $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" 'BEGIN{printf "%.2f", (k+1)*t}') TB 可用"

# ---------------------------------------------------------------- 6.5 日志目录（按节点分离）
# 文件名里已经有 _node<N>（RUN_STEM 带的），但两台写同一个共享 logs/ 时 ls 还是混在一起。
# 共享盘上再按节点分一层子目录 logs/node0/、logs/node1/，最直观；本地盘各写各的，不分更省事。
# 所以默认 auto：LOGDIR 落在网络文件系统上才分目录。强制用 LOG_PER_NODE=1 / 0。
LOG_PER_NODE="${LOG_PER_NODE:-auto}"
if [ "$LOG_PER_NODE" = auto ]; then
  _logfs=$(stat -f -c %T "$(dirname "$LOGDIR_HOST")" 2>/dev/null || echo unknown)
  case "$_logfs" in
    nfs*|cifs|smb*|lustre*|gpfs*|ceph*|9p|fuse*|glusterfs*|beegfs*|ocfs2) LOG_PER_NODE=1 ;;
    *) LOG_PER_NODE=0 ;;
  esac
  echo ">> logs 所在文件系统：$_logfs → LOG_PER_NODE=$LOG_PER_NODE（auto 判定）"
fi
case "$LOG_PER_NODE" in
  0|1) ;;
  *) echo "!! LOG_PER_NODE 只能是 0/1/auto（收到 '$LOG_PER_NODE'），按 0 处理" >&2; LOG_PER_NODE=0 ;;
esac
if [ "$LOG_PER_NODE" = 1 ]; then
  LOGDIR_HOST="${LOGDIR_HOST%/}/node${NODE_RANK}"
  mkdir -p "$LOGDIR_HOST"
fi
echo ">> 日志目录：$LOGDIR_HOST（LOG_PER_NODE=$LOG_PER_NODE；文件名另带 _node${NODE_RANK}）"

# ---------------------------------------------------------------- 7. Megatron 参数（原文逐项搬过来）
# 有值的 flag 一律写成 --flag=value，便于下面按 --help 逐项裁剪（布尔 flag 保持裸写）。
MEGATRON_ARGS=(
  --attention-backend="$ATTN_BACKEND"
  --use-mcore-models                 # 0.13.1 标 DEPRECATED 但仍接受
  --no-mmap-bin-files
  --sequence-parallel
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --init-method-std=0.014
  --position-embedding-type=rope
  --rotary-base=1000000
  --rotary-percent=1.0
  --squared-relu                    # Nemotron-H 用 squared-relu；勿与 --swiglu 同传
  --group-query-attention
  --kv-channels=128
  --normalization=RMSNorm
  --attention-dropout=0.0
  --hidden-dropout=0.0
  --distributed-timeout-minutes=230
  --train-iters="$TRAIN_ITERS"
  --lr-decay-iters=1830030
  --lr=6e-4
  --min-lr=6e-6
  --weight-decay=0.1
  --clip-grad=1.0
  --lr-decay-style=cosine
  --lr-warmup-iters=5              # 生产训练应提高到几百，短 warmup 会有 grad-norm 尖峰
  --eval-iters=1
  --eval-interval=50
  --log-interval=10
  --tokenizer-type=NullTokenizer
  --vocab-size=131072
  --mock-data                      # 原文是纯算力基准，不接数据管道
  --num-workers=1
  --no-create-attention-mask-in-dataloader
  --log-progress
  --timing-log-option=minmax
  --log-throughput
  --bf16
  --adam-beta1=0.9
  --adam-beta2=0.95
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
  --manual-gc
  --num-layers=79
  --hidden-size=8192
  --ffn-hidden-size=32768
  --num-attention-heads=64
  --seq-length="$SEQ"
  --max-position-embeddings="$SEQ"
  --num-query-groups=8
  --tensor-model-parallel-size="$TP"
  --pipeline-model-parallel-size="$PP"
  --micro-batch-size="$MBS"
  --global-batch-size="$GBS"
  --data-cache-path="$DATA_CACHE_IN"
  --tensorboard-dir="$TENSORBOARD_IN"
  --seed="$SEED"
)
[ "$CHECK_HASH" = 1 ] && MEGATRON_ARGS+=(--check-weight-hash-across-dp-replicas-interval=20000)
[ -n "$EXIT_DURATION_MINS" ] && MEGATRON_ARGS+=(--exit-duration-in-mins="$EXIT_DURATION_MINS")
[ "$TP_COMM_OVERLAP" = 1 ] && MEGATRON_ARGS+=(--tp-comm-overlap)
# 双机容器内无对端 DNS：Gloo 组会退化成 127.0.0.1 并 connectFullMesh 拒连
[ "$NNODES" -gt 1 ] && MEGATRON_ARGS+=(--disable-gloo-process-groups)

if [ "$FP8" = 1 ]; then
  # 原文是 --fp8-format hybrid；本机 25.01 + TE 1.14 实测要用 e4m3 + 显式 delayed，
  # 只传 recipe 会跑出 args.fp8=None 的「假 FP8」。
  MEGATRON_ARGS+=(--fp8-format=e4m3 --fp8-recipe=delayed
                  --fp8-amax-history-len=1024 --fp8-amax-compute-algo=max --fp8-param-gather)
fi
if [ "$DEBUG_LOGGING" = 1 ]; then
  MEGATRON_ARGS+=(--log-params-norm --log-num-zeros-in-grad --log-straggler
                  --disable-straggler-on-startup --straggler-minmax-count=16)
fi
if [ "$ENABLE_CKPT" = 1 ]; then
  # --save-retain-interval 是 core_v0.15 新增、0.13.1 没有；「存新的删旧的」由
  # 第 9 节的 ckpt_roll_once() 在外部做（CKPT_KEEP 控制保留几套）。
  MEGATRON_ARGS+=(--save="$CHECKPOINT_IN" --save-interval="$SAVE_INTERVAL"
                  --ckpt-format=torch_dist --ckpt-fully-parallel-save --ckpt-fully-parallel-load
                  --ckpt-assume-constant-structure --async-save)
  # 双机写 ckpt 必须共享目录：torch_dist 每个 rank 写自己的分片，两机各写本地盘时
  # node0 只有一半分片、latest_checkpointed_iteration.txt 也只落在 node0，RESUME 必然失败。
  ck_fstype=$(stat -f -c %T "$CHECKPOINT_HOST" 2>/dev/null || echo unknown)
  case "$ck_fstype" in
    nfs*|cifs|smb*|lustre*|gpfs*|ceph*|9p|fuse*|glusterfs*|beegfs*|ocfs2)
      echo ">> checkpoint 目录在 $ck_fstype 上（网络文件系统，两机共享）" ;;
    *)
      if [ "$NNODES" -gt 1 ]; then
        echo "!! NNODES=$NNODES，但 $CHECKPOINT_HOST 看起来是本地盘（fstype=$ck_fstype）" >&2
        echo "   dp_rank=1 的分片会写到 node1 本地：node0 只有一半，RESUME 必然失败。" >&2
        echo "   ① 用共享盘：SHARED_ROOT=/mnt/shared/e2e-56b（见文件头【共享盘】）" >&2
        echo "   ② 或 ENABLE_CKPT=0 只测吞吐" >&2
        echo "   ③ 明知后果仍要跑：CKPT_LOCAL_OK=1" >&2
        [ "${CKPT_LOCAL_OK:-0}" = 1 ] || exit 2
      fi ;;
  esac
  # 长跑被调度器 SIGTERM 时先存再退（KillWait 常只给 30s，800GB ckpt 写不完，
  # 但至少不会「什么都没留下」）。只在开了 checkpoint 时才有意义。
  MEGATRON_ARGS+=(--exit-signal-handler)
  if [ -f "$CHECKPOINT_HOST/latest_checkpointed_iteration.txt" ]; then
    echo ">> 从 $CHECKPOINT_HOST 第 $(cat "$CHECKPOINT_HOST/latest_checkpointed_iteration.txt") 步恢复"
    MEGATRON_ARGS+=(--load="$CHECKPOINT_IN")
  else
    echo ">> $CHECKPOINT_HOST 无 checkpoint，从头开始"
  fi
  # 磁盘代价：56.5B × (2B bf16 权重 + 4B fp32 master + 8B Adam m/v) = 14 B/参数 ≈ 0.8 TB/套。
  # 滚动保留上线后，占用不再随时间线性增长，但「写新一套」期间新旧并存：
  #   CKPT_KEEP=1 → 稳态 1 套 ≈0.8TB，峰值 2 套 ≈1.6TB；删除只在新套完整落盘后才发生。
  if [ -n "$EXIT_DURATION_MINS" ]; then
    ITER_S_EST="${ITER_S_EST:-25.5}"   # 原文双机口径 ~24.5s/步，留余量
    n_save=$(awk -v m="$EXIT_DURATION_MINS" -v s="$SAVE_INTERVAL" -v t="$ITER_S_EST" \
      'BEGIN{printf "%d", m*60/(s*t) + 1}')
    if [ "$CKPT_KEEP" -gt 0 ]; then
      echo ">> 本跑 ${EXIT_DURATION_MINS} 分钟共存约 ${n_save} 套（每 ${SAVE_INTERVAL} 步一套，单套 ${CKPT_TB} TB）"
      echo "   保留最近 ${CKPT_KEEP} 套，其余自动删 → 稳态 $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" \
           'BEGIN{printf "%.2f", k*t}') TB，峰值 $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" \
           'BEGIN{printf "%.2f", (k+1)*t}') TB"
    else
      echo ">> 本跑 ${EXIT_DURATION_MINS} 分钟约 ${n_save} 套 ckpt ≈ $(awk -v a="$n_save" -v b="$CKPT_TB" \
           'BEGIN{printf "%.1f", a*b}') TB（CKPT_KEEP=0，不删旧盘，盘不够别长跑）"
    fi
  fi
  # 开机前先量盘：写满之后 checkpoint 会以 inline_container.cc 的
  # 「unexpected pos 448 vs 342」形式炸掉，且 tee 也会报 No space left on device，
  # 真正的首错很难看出来。这里提前拦。
  # df -Pk 给的是 KB，所以 TB→KB 要乘 1024^3（写成 1024*1024 会差 1024 倍）
  ck_need_kb=$(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" -v n="${n_save:-1}" \
    'BEGIN{ if (k>0) printf "%d", (k+1)*t*1024*1024*1024*1.05; else printf "%d", n*t*1024*1024*1024*1.05 }')
  ck_avail_kb=$(df -Pk "$CHECKPOINT_HOST" 2>/dev/null | awk 'NR==2{print $4+0}')
  ck_need_gb=$(awk -v a="$ck_need_kb" 'BEGIN{printf "%.0f", a/1048576}')
  ck_avail_gb=$(awk -v a="${ck_avail_kb:-0}" 'BEGIN{printf "%.0f", a/1048576}')
  if [ "${ck_avail_kb:-0}" -lt "$ck_need_kb" ]; then
    echo "!! $CHECKPOINT_HOST 可用 ${ck_avail_gb} GB < 需要 ${ck_need_gb} GB" >&2
    echo "   （CKPT_KEEP=$CKPT_KEEP 单套 ${CKPT_TB} TB；峰值按 (KEEP+1) 套算）" >&2
    echo "   三条路：① 换到大盘 CHECKPOINT_HOST=... ② 关掉 ENABLE_CKPT=0 只测吞吐" >&2
    echo "           ③ 缩小 SAVE_INTERVAL 之外的其它变量无效；CKPT_DISK_FORCE=1 可强行继续" >&2
    if [ "$CKPT_DISK_FORCE" != 1 ]; then exit 2; fi
    echo "   → CKPT_DISK_FORCE=1，继续（写满会崩，后果自负）" >&2
  else
    echo ">> 盘位 OK：$CHECKPOINT_HOST 可用 ${ck_avail_gb} GB ≥ 需要 ${ck_need_gb} GB"
  fi
fi

# ---------------------------------------------------------------- 8. 参数表自检 + 自动裁剪
# 用容器里那份 core_v0.13.1 的真实 --help 兜底：小版本差异不会让脚本启动即崩。
# 参数表文件名带节点号：$ROOT 放在共享盘时，两台同时 dump 同一个 args-help.txt
# 会互相截断，随后「按 --help 裁剪参数」就会读到半截表，把合法 flag 当成不存在裁掉。
ARGS_HELP="${ARGS_HELP:-$WORKSPACE_HOST/args-help-node${NODE_RANK}.txt}"

# 参数表必须是真正的 --help 输出。容器启动失败日志（CNI/iptables、GPU 注入等）
# 也会被重定向进 ARGS_HELP，若把它当参数表，所有 flag 会被裁光。
help_table_ok() {
  [ -s "$1" ] || return 1
  local n
  n=$(grep -cE '^ *--' "$1" 2>/dev/null || true)
  [ "${n:-0}" -ge 400 ]
}
HAVE_HELP=0
if help_table_ok "$ARGS_HELP"; then
  HAVE_HELP=1
elif [ "$DRY_RUN" != 1 ] && [ "${ARGS_CHECK:-1}" = 1 ]; then
  echo ">> dump core_v0.13.1 的 pretrain_gpt.py --help → $ARGS_HELP"
  # 先写临时文件、成功后再 mv：dump 中途被 Ctrl-C / SIGTERM 打断，也不会留下半截
  # 参数表被下一次运行当成有效表（共享盘上尤其容易踩）。
  _ah_tmp="$ARGS_HELP.tmp.$$"
  # 必须带 --network host：默认 bridge 走 CNI firewall 插件，主机
  # iptables(nf_tables) 与 nft 不兼容时容器根本起不来（11-11 现场实测）。
  if ! "${ENGINE_CMD[@]}" run --rm --gpus all --network host \
      --shm-size=64g --ulimit memlock=-1:-1 --ulimit stack=67108864 \
      -v "$ROOT:/workspace" -w "$MEGATRON_IN" \
      -e "PYTHONPATH=$MEGATRON_IN" -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
      "$IMAGE" python pretrain_gpt.py --help > "$_ah_tmp" 2>&1; then
    echo "!! --help dump 失败，$_ah_tmp 前 6 行：" >&2
    head -6 "$_ah_tmp" >&2
    rm -f -- "$_ah_tmp" 2>/dev/null || true
    if [ -f "$SCRIPT_DIR/args-help.core_v0.13.1.txt" ] \
       && help_table_ok "$SCRIPT_DIR/args-help.core_v0.13.1.txt"; then
      echo "   → 改用随脚本附带的 core_v0.13.1 参数表（同一 25.01 镜像实测 --help）"
      ARGS_HELP="$SCRIPT_DIR/args-help.core_v0.13.1.txt"
      HAVE_HELP=1
    else
      echo "   常见原因：容器启动失败（CNI/iptables、GPU 注入）、源码路径不对、TE import 报错" >&2
      echo "   绕过：ARGS_HELP=<合法参数表> / ARGS_CHECK=0 / 先跑 00_prepare.sh" >&2
      exit 2
    fi
  elif ! help_table_ok "$_ah_tmp"; then
    echo "!! dump 出来的不是有效参数表（flag 行 < 400），见 $_ah_tmp"; exit 2
  else
    if ! mv -f -- "$_ah_tmp" "$ARGS_HELP" 2>/dev/null; then
      echo "!! 无法写入参数表 $ARGS_HELP（目录不可写？）" >&2; exit 2
    fi
    HAVE_HELP=1
  fi
fi
if [ "$HAVE_HELP" = 1 ]; then
  n=$(grep -cE '^ *--' "$ARGS_HELP" || true)
  echo ">> 参数表 $ARGS_HELP（$n 行 flag）；逐项核对中"
  pruned=()
  for e in "${MEGATRON_ARGS[@]}"; do
    if [[ "$e" == --* ]]; then
      f=${e%%=*}
      if ! grep -qE "^ *--${f#--}([ =]|$)" "$ARGS_HELP"; then
        echo "   [裁剪] $f 不在 core_v0.13.1 的 --help 中"
        continue
      fi
    fi
    pruned+=("$e")
  done
  MEGATRON_ARGS=("${pruned[@]}")
else
  echo "   ! 无 --help 参数表，跳过裁剪（DRY_RUN 可给 ARGS_HELP=/path/to/megatron_args.txt）" >&2
fi

# ---------------------------------------------------------------- 9. checkpoint 滚动保留
# 需求：「第二次保存时把第一次删掉」= 只保留最近 N 套（CKPT_KEEP，默认 1）。
# 0.13.1 没有 --save-retain-interval（args-help 里只有 --save-interval /
# --non-persistent-save-interval），也没有任何「保留最近 N 个」的开关，只能在外部做。
# 单套 56.5B ≈ 0.8 TB（bf16 权重 113 GB + fp32 master 226 GB + Adam m/v 453 GB），
# 24h 逐小时存就是十几 TB —— 11-11 单机第 720 步就是这么写满盘的。
#
# 判定与安全边界（每条都对应一次真实翻车）：
#   ① 一套 ckpt「可用」= 有完成标记 + 体积≥期望的 1/3。标记：torch_dist 是
#      .metadata/common.pt，legacy torch 是 mp_rank_*/model_optim_rng.pt。只看标记不够：
#      盘写满时 .metadata 可能已落、分片却截断，此时它不算数（否则「删了旧的留个坏的」）。
#   ② latest_checkpointed_iteration.txt 指向的那个永不删（RESUME 就靠它）。
#   ③ 比「最新可用 ckpt」还新的目录一律不碰 —— 可能就是正在写的那一套。
#      没有可用 ckpt 时整轮跳过，宁可多占一套也不删到一半。
#   ④ 年龄按 stat 精确到秒：可用套要 ≥ CKPT_KEEP_GRACE（默认 300s）；
#      残套要 ≥ max(4×grace, 30min)（0.8TB 落盘本身要 5~10min）。
#   ⑤ 先 mv 成 iter_x.deleting.<pid> 再 rm -rf。mv 原子，双机同时判定要删时只有一个成功，
#      不会「两边各删一半」；残留墓碑下一轮再清。
#   ⑥ 每台机器都跑：不共享盘时各删各的；共享盘时靠 ⑤ 保证幂等。
CKPT_ROLL_LOG="$LOGDIR_HOST/ckpt-retention-${RUN_ID:-$RUN_STAMP}-node${NODE_RANK}.log"
CKPT_TB_MB=$(awk -v t="$CKPT_TB" 'BEGIN{printf "%d", t*1048576}')   # 体积校验用（MB）

# 判定一套 ckpt 是否「能用」：① 有完成标记 ② 体积站得住
#   ① torch_dist 落 .metadata / common.pt；legacy torch 落 mp_rank_*/model_optim_rng.pt。
#   ② 只靠标记不够：盘写满时 .metadata 可能已落、分片却截断（11-11 就是这么炸的）。
#      所以再要求体积 ≥ 期望的 1/3；expect_mb=0 时不校验（单测用）。
ckpt_usable() {   # $1=dir $2=期望体积 MB
  local mb
  [ -f "$1/.metadata" ] || [ -f "$1/common.pt" ] || \
    [ -n "$(find "$1" -maxdepth 3 -name model_optim_rng.pt -print -quit 2>/dev/null)" ] || return 1
  [ "${2:-0}" -gt 0 ] || return 0
  mb=$(du -sm "$1" 2>/dev/null | awk '{print $1+0}')
  awk -v a="${mb:-0}" -v b="$2" 'BEGIN{exit !(a*3 >= b)}'
}

file_age() {   # 目录/文件的年龄（秒）。取不到时间就返回 0 = 当作刚写完，不删（保守）
  local t
  t=$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "")
  [ -n "$t" ] || { echo 0; return 0; }
  echo $(( $(date +%s) - t ))
}

ckpt_roll_once() {
  local root="$1" keep="$2" expect_mb="${3:-0}"
  # 墓碑名带主机标识：共享盘上两台同时判定要删时，若 PID 恰好相同，
  # 纯 .deleting.$$ 会让 mv 把目录塞进对方正在删的墓碑里。
  local hosttag; hosttag=$(hostname -s 2>/dev/null || echo h)
  local d it sz age n=0 i j best tmp nfull=0 kept=0 del=0 freed=0 latest=0
  local newest_ok=-1 partial_age
  [ -d "$root" ] || return 0

  # 残套的最小年龄：4×grace，但不少于 30 分钟（--async-save 落 0.8TB 要 5~10 分钟）
  partial_age=$(( CKPT_KEEP_GRACE * 4 )); [ "$partial_age" -lt 1800 ] && partial_age=1800

  # 收尾上次没删完的墓碑
  for d in "$root"/iter_*.deleting.*; do
    [ -d "$d" ] || continue
    if [ "$(file_age "$d")" -ge $(( partial_age + 60 )) ]; then rm -rf -- "$d" 2>/dev/null || true; fi
  done

  if [ -f "$root/latest_checkpointed_iteration.txt" ]; then
    latest=$(tr -cd '0-9' < "$root/latest_checkpointed_iteration.txt" 2>/dev/null || true)
    latest=$(( 10#${latest:-0} ))
  fi

  # 收集 iter_*：两个平行数组 iters[]/dirs[]，不拼 "数字<TAB>路径" 字符串
  #（不拼就不用解析分隔符，tab 在某些 shell/环境下会被吃掉）
  local iters=() dirs=()
  for d in "$root"/iter_*; do
    [ -d "$d" ] || continue
    case "${d##*/}" in *.deleting.*) continue ;; esac
    it="${d##*/}"; it="${it#iter_}"
    case "$it" in ''|*[!0-9]*) continue ;; esac     # iter_720 / iter_720_tmp 只认纯数字
    iters+=( "$((10#$it))" ); dirs+=( "$d" )
  done
  n=${#iters[@]}
  [ "$n" -gt 0 ] || return 0

  # 从新到旧：选择排序（n 只有个位数~几十，O(n²) 无所谓）
  for (( i=0; i<n; i++ )); do
    best=$i
    for (( j=i+1; j<n; j++ )); do
      [ "${iters[j]}" -gt "${iters[best]}" ] && best=$j
    done
    if [ "$best" -ne "$i" ]; then
      tmp=${iters[i]}; iters[i]=${iters[best]}; iters[best]=$tmp
      tmp=${dirs[i]};  dirs[i]=${dirs[best]};   dirs[best]=$tmp
    fi
  done

  # 最新的「可用」ckpt：只有比它更旧的目录才允许删
  for (( i=0; i<n; i++ )); do
    if ckpt_usable "${dirs[i]}" "$expect_mb"; then newest_ok=${iters[i]}; break; fi
  done
  if [ "$newest_ok" -lt 0 ]; then
    echo "[$(date '+%F %T')] 还没有可用 ckpt，跳过" >> "$CKPT_ROLL_LOG"
    return 0
  fi

  # 保留集 = 最新 keep 个「可用」的 + latest 指针指向的（latest 永远不删）
  local keep_list=" "
  for (( i=0; i<n; i++ )); do
    [ "$nfull" -lt "$keep" ] || break
    if ckpt_usable "${dirs[i]}" "$expect_mb"; then
      keep_list+="${iters[i]} "; nfull=$((nfull+1))
    fi
  done
  case "$keep_list" in *" $latest "*) ;; *) keep_list+="$latest " ;; esac

  for (( i=0; i<n; i++ )); do
    it=${iters[i]}; d=${dirs[i]}
    case "$keep_list" in *" $it "*) kept=$((kept+1)); continue ;; esac
    # 比最新的可用 ckpt 还新 → 可能正在写，绝不碰（宁可多占一套，也不能删正在写的）
    [ "$it" -lt "$newest_ok" ] || continue
    age=$(file_age "$d")
    if ckpt_usable "$d" "$expect_mb"; then
      [ "$age" -ge "$CKPT_KEEP_GRACE" ] || continue
    else
      [ "$age" -ge "$partial_age" ] || continue       # 残套：等更久，确认是废弃的
    fi
    sz=$(du -sm "$d" 2>/dev/null | awk '{print $1+0}')
    if mv -- "$d" "$d.deleting.$hosttag.$$" 2>/dev/null; then
      rm -rf -- "$d.deleting.$hosttag.$$" 2>/dev/null || true
      del=$((del+1)); freed=$((freed+sz))
      echo "[$(date '+%F %T')] removed $(basename "$d") ${sz} MB" >> "$CKPT_ROLL_LOG"
    fi
  done
  if [ "$del" -gt 0 ]; then
    echo "[$(date '+%F %T')] 本轮 removed=$del freed=${freed} MB kept=$kept" >> "$CKPT_ROLL_LOG"
  fi
  return 0
}

ckpt_roll_loop() {
  echo "[$(date '+%F %T')] start keep=$CKPT_KEEP grace=${CKPT_KEEP_GRACE}s poll=${CKPT_ROLL_POLL}s root=$CHECKPOINT_HOST" \
    >> "$CKPT_ROLL_LOG"
  while true; do
    ckpt_roll_once "$CHECKPOINT_HOST" "$CKPT_KEEP" "$CKPT_TB_MB" || true
    sleep "$CKPT_ROLL_POLL"
  done
}
CKPT_ROLL_PID=""
if [ "$DRY_RUN" != 1 ] && [ "$ENABLE_CKPT" = 1 ] && [ "$CKPT_KEEP" -gt 0 ]; then
  ckpt_roll_loop & CKPT_ROLL_PID=$!
fi

# ---------------------------------------------------------------- 9.4 运行标识
# 日志名自带并行/batch 配置，和截图里 w8_..._tp1_pp1_mbs2_s4096_gbs256 一个风格：
# 两机的日志混在一起时，光看文件名就能确认是不是同一个实验。
# dtype 在裁剪之后取，避免 --fp8-* 被裁掉时文件名还标 fp8。
if printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--fp8-format'; then
  DTYPE=fp8e4m3
else
  DTYPE=bf16
fi
RUN_STEM="nemotron56b_tp${TP}_pp${PP}_dp${DP}_mbs${MBS}_s${SEQ}_gbs${GBS}_${DTYPE}_node${NODE_RANK}"
LOG="$LOGDIR_HOST/${RUN_STEM}-${RUN_STAMP}.out"

# ---------------------------------------------------------------- 9.5 GPU 采样（dmon）
# 两节点各采各的：dmon 读本机物理卡，跨机聚合没意义，文件名带 node rank 区分。
# 回答「24h 有没有热降频 / 功耗撞墙 / 显存爬升 / NVLink 打满但 step 没变快」。
DMON="${DMON:-1}"
DMON_INTERVAL="${DMON_INTERVAL:-5}"     # 秒；24h×8 卡在 5s 间隔下约 10 MB
DMON_METRICS="${DMON_METRICS:-pucvmt}"  # p功耗温度 u利用率 c时钟 v违规 m显存 t PCIe/NVLink
DMON_PID=""
DMON_LOG="$LOGDIR_HOST/${RUN_STEM}-dmon-${RUN_STAMP}.csv"

dmon_probe() { nvidia-smi dmon -s "$1" -c 1 -d 1 >/dev/null 2>&1; }

start_dmon() {
  [ "$DMON" = 1 ] || { echo "   ! DMON=0，跳过 GPU 采样"; return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "   ! 找不到 nvidia-smi，跳过 GPU 采样" >&2; return 0; }
  case "$DMON_INTERVAL" in
    ''|*[!0-9]*) echo "   ! DMON_INTERVAL 必须是整数秒，已按 5 处理" >&2; DMON_INTERVAL=5 ;;
  esac
  [ "$DMON_INTERVAL" -ge 1 ] || DMON_INTERVAL=1
  # 先探一次（-c 1 采一个点即退）：驱动/卡不支持 t(PCIe/NVLink) 或 e(ECC) 时
  # 直接跑会启动失败，采样整段静默丢失，事后才发现 CSV 只有两行表头。
  if ! dmon_probe "$DMON_METRICS"; then
    echo "   ! dmon -s $DMON_METRICS 不可用，退回 -s pucvm" >&2
    DMON_METRICS=pucvm
    dmon_probe "$DMON_METRICS" || { echo "   ! dmon 完全不可用，跳过 GPU 采样" >&2; return 0; }
  fi
  nvidia-smi dmon -s "$DMON_METRICS" -d "$DMON_INTERVAL" -o DT > "$DMON_LOG" 2>&1 &
  DMON_PID=$!
  echo ">> GPU 采样 → $DMON_LOG（每 ${DMON_INTERVAL}s，-s $DMON_METRICS）"
}

stop_dmon() {
  [ -n "${DMON_PID:-}" ] || return 0
  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true    # 回收子进程；被 kill 后 wait 非 0，set -e 下必须兜住
  DMON_PID=""
}

# dmon CSV 解析：按**列名**取值，不按固定列号。不同驱动会在 sm/mem 后面多插
# jpg/ofa 之类的列，写死列号一旦错位，功耗温度全会串列且不会报错。
summarize_dmon() {
  local f=$1
  if [ ! -s "$f" ]; then
    echo "（无 GPU 采样数据：DMON=0，或 dmon 在本机不可用）"; return 0
  fi
  echo "  采样文件 $f（$(awk '!/^#/{n++} END{print n+0}' "$f") 行数据）"
  echo "  GPU  采样点  功耗峰值  温度峰值  SM均值  SM峰值   Mem峰值   NVLink TX   NVLink RX  pviol  tviol"
  awk '
    NR==1 {
      for (i=1; i<=NF; i++) { h=$i; sub(/^#/, "", h); idx[h]=i }
      # 只强制要求这 4 列；显存/NVLink 按可用性降级，-s pucv 这种精简组合也能出统计
      n=split("gpu pwr gtemp sm", need, " ")
      for (k=1; k<=n; k++) if (!(need[k] in idx)) {
        printf "   ! dmon 表头缺少字段 %s，跳过统计\n     首行原文：%s\n", need[k], $0
        bad=1; exit
      }
      # 「显存占用」是 fb（framebuffer, MB）；mem 是显存**控制器利用率 %**，别混用
      if ("fb" in idx) fbk = "fb"; else if ("mem" in idx) fbk = "mem"; else fbk = ""
      # NVLink 吞吐要 -s 里带 t 才有；降级到 pucvm 时整列是 N/A
      nvl = (("nvltx" in idx) && ("nvlrx" in idx))
      next
    }
    /^#/ { next }
    NF < 5 { next }
    {
      g = $(idx["gpu"])
      if (g == "") next
      npt[g]++
      v = $(idx["pwr"]);   if (v ~ /^[0-9.]+$/) { if (v > pmax[g]) pmax[g]=v }
      v = $(idx["gtemp"]); if (v ~ /^[0-9.]+$/) { if (v > tmax[g]) tmax[g]=v }
      v = $(idx["sm"]);    if (v ~ /^[0-9.]+$/) { usum[g]+=v; ucnt[g]++; if (v > umax[g]) umax[g]=v }
      if ("mem" in idx) { v = $(idx["mem"]); if (v ~ /^[0-9.]+$/) { if (v > memmax[g]) memmax[g]=v } }
      if (fbk != "")    { v = $(idx[fbk]);   if (v ~ /^[0-9.]+$/) { if (v > fbmax[g]) fbmax[g]=v } }
      # NVLink：dmon 的单位是 MB/s，汇总换 GB/s（÷1024）。均值看稳态负载，峰值看有没有打满
      if (nvl) {
        v = $(idx["nvltx"]); if (v ~ /^[0-9.]+$/) { txsum[g]+=v; txc[g]++; if (v > txmax[g]) txmax[g]=v }
        v = $(idx["nvlrx"]); if (v ~ /^[0-9.]+$/) { rxsum[g]+=v; rxc[g]++; if (v > rxmax[g]) rxmax[g]=v }
      }
      if ("pviol" in idx) { v = $(idx["pviol"]); if (v ~ /^[0-9.]+$/ && v+0 > 0) pv[g]++ }
      if ("tviol" in idx) { v = $(idx["tviol"]); if (v ~ /^[0-9.]+$/ && v+0 > 0) tv[g]++ }
    }
    END {
      if (bad) exit
      for (g in npt) {
        # 一卡 143771 MB，直接打 MB 比换 GiB 更好对 nvidia-smi
        if (fbk == "fb")      memstr = sprintf("%.0fMB", fbmax[g])
        else if (fbk == "mem") memstr = sprintf("%.0f%%", memmax[g])
        else                   memstr = "N/A"
        if (nvl) {
          txstr = txc[g] ? sprintf("%.0f/%.0f", txsum[g]/txc[g]/1024, txmax[g]/1024) : "N/A"
          rxstr = rxc[g] ? sprintf("%.0f/%.0f", rxsum[g]/rxc[g]/1024, rxmax[g]/1024) : "N/A"
        } else { txstr = "N/A"; rxstr = "N/A" }
        printf "  %-4s %6d  %7.0fW  %7.0fC  %5.1f%%  %5.1f%%  %10s  %10s  %10s  %5d  %5d\n",
          g, npt[g], pmax[g], tmax[g], (ucnt[g] ? usum[g]/ucnt[g] : 0), umax[g],
          memstr, txstr, rxstr, pv[g]+0, tv[g]+0
      }
    }
  ' "$f" 2>/dev/null | sort -n
  if head -1 "$f" 2>/dev/null | grep -q 'nvltx'; then
    echo "  NVLink 列 = 均值/峰值，单位 GB/s；H200 单向容量约 450 GB/s"
    if [ "${DMON_INTERVAL:-5}" -gt 2 ] 2>/dev/null; then
      echo "  采样间隔 ${DMON_INTERVAL}s，step 更长时只够看趋势，追突发要 DMON_INTERVAL=1"
    fi
  fi
  # 判读：pviol/tviol 非 0 = 撞到功耗墙/温度墙，MFU 会掉但 step 时间不一定炸；
  #       gtemp 长时间 >85C、SM 均值明显低于另一节点，都是降频特征。
  #       NVLink 峰值接近 450 = 单向打满；均值高但 SM 也高 = overlap 正常；
  #       均值低、峰值高 = 通信是突发式的，盯 SM 有没有掉 0 比盯流量有用。
  #       双机还要对比两节点的 TX/RX：一边高一边低，通常是 RoCE 链路不对称。
}

# 所有后台任务统一在这里收；trap 只能有一个 EXIT，重复安装会覆盖前一个。
cleanup() {
  if [ -n "${CKPT_ROLL_PID:-}" ]; then kill "$CKPT_ROLL_PID" 2>/dev/null || true; fi
  stop_dmon
}
trap 'cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------- 10. 启动
if [ "$RENDEZVOUS" = c10d ]; then
  # 对齐原文 c10d；torchrun 的正式 option 是下划线 --rdzv_backend/--rdzv_endpoint
  TORCHRUN_ARGS=(--nproc_per_node="$GPUS_PER_NODE" --nnodes="$NNODES"
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id="$RUN_ID")
else
  TORCHRUN_ARGS=(--nproc_per_node="$GPUS_PER_NODE" --nnodes="$NNODES"
    --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

# data_cache 也是同一个道理：本地盘时 node1 没有 builder（只有 global rank 0 建），
# 只能靠「两边内容完全一致」——要么共享盘，要么 node0 烤好后 rsync 到 node1 同一路径。
# 这里只告警不拦：rsync 方案本来就能跑，拦下来会误伤。
if [ "$NNODES" -gt 1 ]; then
  dc_fstype=$(stat -f -c %T "$DATA_CACHE_HOST" 2>/dev/null || echo unknown)
  case "$dc_fstype" in
    nfs*|cifs|smb*|lustre*|gpfs*|ceph*|9p|fuse*|glusterfs*|beegfs*|ocfs2) ;;
    *) echo "!! data_cache 在本地盘（fstype=$dc_fstype）：node1 不会自己建索引" >&2
       echo "   两节点内容必须完全一致，否则 node1 的 rank8 会崩在 numpy.load" >&2
       echo "   → 共享盘 SHARED_ROOT=...，或 node0 烤热后 rsync 到 node1 同一路径" >&2 ;;
  esac
fi

CMD=("${ENGINE_CMD[@]}" run --rm --gpus all --network host --shm-size=64g
     --ulimit memlock=-1:-1 --ulimit stack=67108864
     "${IB_DEVICES[@]}"
     -v "$ROOT:/workspace" -w "$MEGATRON_IN"
     ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}
     "${ENV_ARGS[@]}" "$IMAGE"
     torchrun "${TORCHRUN_ARGS[@]}" pretrain_gpt.py "${MEGATRON_ARGS[@]}")

echo ">> 日志：$LOG"
if [ "$DRY_RUN" = 1 ]; then
  echo ">> DRY_RUN：只打印命令，不启动容器"
  echo "   GPU 采样 → $DMON_LOG（每 ${DMON_INTERVAL}s，-s $DMON_METRICS）"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

start_dmon
echo ">> 启动 torchrun（world=$WORLD，另一节点需同时启动）"
# set -o pipefail 下若 torchrun 失败，整个 pipeline 会先触发 set -e，
# 导致 rc/日志提示走不到；这里临时关掉 errexit，显式回收 PIPESTATUS。
set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo ">> torchrun 退出码 rc=$rc，日志 $LOG"
stop_dmon        # 先停采样，保证 CSV 刷完再统计；EXIT trap 再调一次是空操作
echo "---- GPU 采样（dmon，本节点）----"
summarize_dmon "$DMON_LOG"
echo "   另一节点的采样在对端同名文件里（dmon-*-node1.csv），降频要两机一起看"
exit "$rc"
