#!/usr/bin/env bash
# =============================================================================
# 单机 8×H200 测试脚本（配套 run_56b_2x8h200.sh）
#
# 镜像 nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3 (torch 2.6 / CUDA 12.8 / TE 1.14)
# 源码 Megatron-LM core_v0.13.1（$ROOT/Megatron-LM，PYTHONPATH 方式）
# 容器 docker/nerdctl --network host + torchrun 单节点，NNODES 固定 1。
#
# 与双机脚本的差别：默认不传 checkpoint / --load、不注入 IB 设备与双机 NCCL 超时、
# 不加 --disable-gloo-process-groups、DP=1 时自动关掉只在 DP>1 有意义的开关；
# 训练结束解析日志出摘要。长跑（DURATION_MINS）见下方【时长】一节。
#
# -----------------------------------------------------------------------------
# 三档参数（CASE=8b|mfu|56b，默认 8b）；MFU 分母一律 989.5 TFLOP/s。
# 基线来源两处：8b 与旧的 gbs32 行来自 llm-bench-split/analysis/cases.csv 的 w8 档；
# mfu 现用 2026-11-11 GBS 扫描截图里的最优档（GBS256），细节见下方注释。
# -----------------------------------------------------------------------------
# | CASE | 模型                | TP/PP/DP | seq  | MBS | GBS | steps | 用途           |
# |------|---------------------|----------|------|-----|-----|-------|----------------|
# | 8b   | Llama-3 8B          | 8/1/1    | 4096 | 1   | 8   | 15    | 链路冒烟/基线  |
# | mfu  | Llama-3 8B          | 1/1/8    | 4096 | 2   | 256 | 20    | 单机峰值对标   |
# | 56b  | Nemotron-H-56B      | 8/1/1    | 4096 | 1   | 8   | 20    | 单机可行性/显存|
#
# mfu 档 = 2026-11-11 GBS 扫描里的最优组合（GBS 256 > 128 > 64）：
#   TP1×PP1 mbs2 seq4096 GBS256，dp=8 → num_micro_batches=16
#   截图三次复测：554.1 / 556.7 TFLOP/s/GPU，MFU 56.0% / 56.3%，CV 0.25%，SM 4/1MHz，3201W
#   （分母 989.5 时 GEMM 天花板 69.7%，即已达天花板的 80%）
#
# | 实测基线（同一 25.01 + core_v0.13.1）                | step ms | TFLOP/s/GPU | MFU@989.5 | 显存 GiB |
# |-------------------------------------------------------|---------|-------------|-----------|----------|
# | w8_8B_tp8_pp1_mbs1_s4096_gb8_bf16                    | 985.1   | 200.7       | 20.3%     | 17.21    |
# | w8_8B_tp1_pp1_mbs2_s4096_gb32_bf16（旧 mfu 默认）      | 1521.9  | 519.5       | 52.5%     | 92.79    |
# | w8_MFU_R_8B_tp1_pp1_mbs2_s4096_gbs256（新 mfu 默认）   | ~11360  | 554.1~556.7 | 56.0~56.3 | 未单测   |
# | E2E 文章（双机 DP=2，不是单机口径）                   | 24500   | 487~489     | 49.2~49.4 | 100.4 GB |
#
# gbs256 那行的 11360ms 是**反推值**不是实测：1521.9ms × (256/32) ÷ (556.7/519.5) = 11363ms。
# 显存未单测——峰值激活只由 MBS×SEQ 决定，GBS 只改梯度累积次数，故应仍是 92.79 GiB。
# 想要与截图完全同协议：STEPS=40（截图每档 40 步）。
#
# 基线源日志 = llm-bench-split/2026-11-10_1827_worker-gpu-a_w8_full/06-train-s8b/*.log，
# 不是 llm-bench-docker/logs 下 11-05 的旧日志（两批数据不同，别混着对数）。
# 本脚本 summarize() 用「跳过前 2 步取中位数」的同一口径，回放这两份日志可逐值复现上表。
#
# 【56B 单机显存风险 —— 必须知道】
# 文章的 81.6 GB allocated / 100.4 GB peak 是 **DP=2 + distributed optimizer** 的口径，
# 优化器状态按 2 个 DP rank 分片。单机 DP=1 时没有分片收益，粗算：
#   56.5B × (2B bf16 权重 + 2B 梯度 + 4B fp32 master + 8B Adam m/v) ÷ TP8 ≈ 113 GB/卡
#   + seq4096 激活约 10 GB + 缓冲 ≈ 125 GB / 140.4 GiB —— 极贴边，容易 OOM。
# 因此 CASE=56b 默认 LOW_MEM=1：加 --use-precision-aware-optimizer（压优化器状态）
# 与 --recompute-granularity=selective（压激活），seq 降到 4096。
# 【2026-11-11 现场更正】core_v0.13.1 的
#   megatron/core/optimizer/optimizer_config.py:199
# 有硬断言：use_precision_aware_optimizer 必须为 use_distributed_optimizer 配合使用。
# 之前只核了 arguments.py（那里的断言确实只关 optimizer_cpu_offload），漏了这个文件；
# 单机 DP=1 时条件分支不会加 DO，8 个 rank 会在 setup_model_and_optimizer 全崩
# （现场 single-56b-*.out 报错原文：
#  AssertionError: --use-precision-aware-optimizer only supported with distributed optimizer）。
# 现在 LOW_MEM=1 会在 DP=1 时自动补 --use-distributed-optimizer。
# LOW_MEM 三种取值：
#   1   = precision-aware optimizer + DO + selective recompute（贴文章语义）
#   sgd = --optimizer=sgd + selective recompute（不分配 Adam m/v，最省显存，
#         完全避开该断言；只作可行性验证，loss 曲线不可与文章 Adam 对比）
#   0   = 文章原生配置，无低显存开关（高 OOM 风险）
# 56b 档只用于验证「单机能不能把 56B 架构跑起来」，吞吐不能与文章 DP=2 数据直接比。
#
# -----------------------------------------------------------------------------
# 【用法】（都在同一台 8 卡机上跑）
#   PLAN_ONLY=1 bash run_1x8h200_cases.sh                 # 只打印三档参数表
#   DRY_RUN=1 ARGS_HELP=/path/to/megatron_args.txt bash run_1x8h200_cases.sh
#   bash run_1x8h200_cases.sh                             # 默认 CASE=8b，约 1 分钟
#   CASE=mfu bash run_1x8h200_cases.sh                    # 单机峰值档（显存约 93 GiB）
#   CASE=56b bash run_1x8h200_cases.sh                    # 56B 单机可行性档
#   CASE=56b LOW_MEM=sgd bash run_1x8h200_cases.sh        # 最省显存、避开 optimizer 断言
#   CASE=56b SEQ=8192 LOW_MEM=0 bash run_1x8h200_cases.sh # 试文章原生配置（高 OOM 风险）
#   CASE=8b STEPS=5 SEQ=2048 bash run_1x8h200_cases.sh    # 更快冒烟
#   DURATION_MINS=1440 CASE=mfu bash run_1x8h200_cases.sh # 按墙钟跑 24 小时
#   DURATION_MINS=1440 ENABLE_CKPT=1 CASE=mfu bash ...    # 24 小时 + 每小时存 ckpt
#   DMON_INTERVAL=2 CASE=mfu bash run_1x8h200_cases.sh    # 加密 GPU 采样（默认 5s 一采）
#   DMON=0 CASE=mfu bash run_1x8h200_cases.sh             # 不要 GPU 采样
#
# 【可调】CASE / STEPS / DURATION_MINS / ITER_S_EST / LOG_INTERVAL /
#         ENABLE_CKPT / SAVE_INTERVAL / RESUME /
#         DMON / DMON_INTERVAL / DMON_METRICS /
#         TP / PP / MBS / GBS / SEQ / LOW_MEM / ATTN_BACKEND /
#         TP_COMM_OVERLAP / FP8 / NCCL_DEBUG / ROOT / IMAGE / ARGS_HELP / DRY_RUN
#
# -----------------------------------------------------------------------------
# 【时长】两套口径，二选一，脚本按哪个给了就走哪个
#   STEPS=15                 → --train-iters=15，跑满 15 步自然结束（默认，冒烟/取 MFU）
#   DURATION_MINS=1440       → --exit-duration-in-mins=1440，跑满 1440 分钟退出
#     · Megatron 没有「训练多长时间」这个参数：--train-iters 是**步数**上限，
#       --exit-duration-in-mins 才是**墙钟**上限。两个都给时谁先满足谁生效。
#     · 所以长跑必须给足 train-iters，否则步数先耗尽、提前退出。脚本按 ITER_S_EST
#       反推步数并留 20% + 50 步余量，保证是「时间到」而不是「步数到」。
#     · 到点后 Megatron 走 save_checkpoint_and_time 存一次再退出，日志里能看到
#       "exiting after 1440 minutes"（不一定每个 rank 都打）；**只有给了 --save
#       才真落盘**（ENABLE_CKPT=1）。没给 --save 时该调用直接返回，日志照样有那行，
#       别误以为存了。判定在每个 iteration 结束时做，粒度是一步。
#     · 与调度器的关系：宿主机/K8s 的 24h 是硬杀，KillWait 常给 30s，不够写 800GB
#       checkpoint。脚本默认给 --exit-signal-handler，收到 SIGTERM 先存再退；
#       真要 24h 请设 DURATION_MINS=1410（23h30m），留 30 分钟给最后一次落盘。
#     · DURATION_MINS>0 时与双机同策略：train-iters 固定给 DURATION_ITERS（默认 100000）
#       只作上限，真正收口靠 --exit-duration-in-mins，不再用 ITER_S_EST 反推步数
#       （那是「估大就少跑一半」的隐患）。日志因而也是 iteration N/ 100000，和双机对得上。
#     · ITER_S_EST 现在只用于两件事：ckpt 间隔（3600/ITER_S_EST 步）与「预计几套」的估算。
#   ENABLE_CKPT=1            → 每 SAVE_INTERVAL 步存一次 torch_dist checkpoint
#     · 默认关。56B 单机一套 checkpoint（权重+优化器）约 0.8 TB。
#     · SAVE_INTERVAL 默认按 ITER_S_EST 反推「约 1 小时一次」。
#     · CKPT_KEEP=N（默认 1）→ 只保留最近 N 套，新的完整落盘后自动删更老的
#       （= 「存第二次时删第一次」）。稳态 N×0.8TB，写新那套时峰值 (N+1)×0.8TB。
#       CKPT_KEEP=0 关掉删除。开跑前会先量 CHECKPOINT_HOST 可用空间，不够直接退出。
#     · 删除记录写在 $LOGDIR_HOST/ckpt-retention-<stamp>.log。
#     · RESUME=1 时若目录里有 latest_checkpointed_iteration.txt 就 --load 续跑。
#
# -----------------------------------------------------------------------------
# 【GPU 采样】nvidia-smi dmon，训练全程记录功耗/温度/利用率/显存/PCIe-NVLink 时间序列
#   DMON=1（默认开）   → 训练期间后台采样，CSV 落在 $LOGDIR_HOST/dmon-<case>-<stamp>.csv
#   DMON_INTERVAL=5    → 采样间隔秒（-d）。24h 长跑会存约 17k 点/卡，约 10 MB
#   DMON_METRICS=pucvmt→ -s 取值：p功耗温度 u利用率 c时钟 v违规 m显存 t PCIe/NVLink
#     · 先探测一次（-c 1），驱动/卡不支持 t 或 e 时自动退回 pucvm，不会静默丢数据
#     · 采样跑在宿主机，不在容器内（容器里不一定有 nvidia-smi）
#     · 结束时用 summarize_dmon() 汇总：功耗/温度峰值、SM 利用率均值、显存峰值、
#       NVLink TX/RX 均值与峰值（GB/s）、降频计数（pviol/tviol）
#   DMON=0             → 关掉
#
# -----------------------------------------------------------------------------
# 【日志命名】$LOGDIR_HOST/single_<case>_tp.._pp.._dp.._mbs.._s.._gbs.._<dtype>[-lowmemX]-<时间戳>.out
#   single_mfu_tp1_pp1_dp8_mbs2_s4096_gbs256_bf16-20261111_143021.out
#   single_56b_tp8_pp1_dp1_mbs1_s4096_gbs8_bf16_lowmem1-20261111_143021.out
#   · dtype 取的是**裁剪后**的参数（--fp8-* 被裁掉时不会还标 fp8）
#   · 56b 的 LOW_MEM 有 1 / sgd / 0 三种，优化器行为完全不同，必须进文件名
#   · dmon 的 CSV 与训练日志同前缀，只多一个 -dmon，ls 时天然成对
#   · checkpoint 目录同样带 _lowmemX 后缀，避免 RESUME=1 时跨优化器加载
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- 0. 路径/引擎
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="${ROOT:-/root/llm-bench}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$ROOT/e2e-56b}"
WORKSPACE_IN="${WORKSPACE_IN:-/workspace/e2e-56b}"
MEGATRON_IN="${MEGATRON_IN:-/workspace/Megatron-LM}"
IMAGE="${IMAGE:-nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3}"
LOGDIR_HOST="${LOGDIR_HOST:-$WORKSPACE_HOST/logs}"
DATA_CACHE_HOST="${DATA_CACHE_HOST:-$WORKSPACE_HOST/data_cache}"
DATA_CACHE_IN="$WORKSPACE_IN/data_cache"
ARGS_HELP="${ARGS_HELP:-$WORKSPACE_HOST/args-help.txt}"   # 与双机脚本共用同一份 --help 缓存
mkdir -p "$LOGDIR_HOST" "$DATA_CACHE_HOST"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

# 引擎：目标服务器是 nerdctl，默认即 nerdctl（ENGINE=docker 可切回）。
# nerdctl 的 load / run 必须带同一个 --namespace，否则镜像「load 了却 run 不到」；
# 沿用现有 lib.sh 的 default 约定，拒绝 k8s.io / moby。
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
    # DRY_RUN / PLAN_ONLY 允许在没有 nerdctl 的机器上只看命令或参数表
    if [ "${DRY_RUN:-0}" != 1 ] && [ "${PLAN_ONLY:-0}" != 1 ]; then
      command -v nerdctl >/dev/null 2>&1 || { echo "!! 找不到 nerdctl"; exit 2; }
    fi
    ENGINE_ARGS=(--namespace "$NS")
    export CONTAINERD_NAMESPACE="$NS"
    ;;
  docker) ENGINE_ARGS=() ;;
  *) echo "!! ENGINE 只能是 nerdctl | docker（收到 $ENGINE）"; exit 2 ;;
esac
ENGINE_CMD=("$ENGINE" "${ENGINE_ARGS[@]}")

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES=1
WORLD=$((NNODES * GPUS_PER_NODE))    # 8
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DRY_RUN="${DRY_RUN:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PEAK_TFLOPS=989.5                    # H200 SXM BF16 dense 理论峰值（MFU 唯一分母）
VRAM_GIB=140.4                       # nvidia-smi 实测 143771 MiB

# ---------------------------------------------------------------- 1. 档位参数表
CASE="${CASE:-8b}"
ATTN_BACKEND="${ATTN_BACKEND:-auto}"   # 0.13.1 枚举：flash/fused/unfused/local/auto
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-1}"
FP8="${FP8:-0}"                        # 只测 BF16 基线；1 = e4m3 + delayed（本机验证过的组合）
LOW_MEM="${LOW_MEM:-1}"                # 仅 CASE=56b 生效
SEED="${SEED:-1234}"

case "$CASE" in
  8b)
    LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; VOCAB=128256
    TP="${TP:-8}"; PP="${PP:-1}"; SEQ="${SEQ:-4096}"; MBS="${MBS:-1}"; GBS="${GBS:-8}"
    STEPS="${STEPS:-15}"; MASTER_PORT="${MASTER_PORT:-29501}"
    ITER_S_EST="${ITER_S_EST:-1.0}"   # 实测 985.1ms，留 2% 余量
    CKPT_TB=0.11                      # 8B × ~14B/参数
    ACT=swiglu; NOTE="链路冒烟：实测 985.1ms / 200.7 TFLOP/s / 17.21 GiB" ;;
  mfu)
    LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; VOCAB=128256
    # GBS 扫描最优档（256 > 128 > 64）：dp=8、mbs=2 → accum=16，与截图 num_micro_batches 一致
    TP="${TP:-1}"; PP="${PP:-1}"; SEQ="${SEQ:-4096}"; MBS="${MBS:-2}"; GBS="${GBS:-256}"
    STEPS="${STEPS:-20}"; MASTER_PORT="${MASTER_PORT:-29502}"
    # 11.4s 由 gbs32 的 1521.9ms 按 token×8 / 吞吐×1.072 反推；实测偏离时用 ITER_S_EST= 覆盖
    ITER_S_EST="${ITER_S_EST:-11.4}"
    CKPT_TB=0.11                      # 8B × ~14B/参数（DO 分片，总量不变）
    ACT=swiglu; NOTE="单机峰值 GBS256：554.1~556.7 TFLOP/s/GPU / MFU 56.0~56.3%（截图 3 次复测）" ;;
  56b)
    LAYERS=79; HIDDEN=8192; FFN=32768; HEADS=64; VOCAB=131072
    TP="${TP:-8}"; PP="${PP:-1}"; SEQ="${SEQ:-4096}"; MBS="${MBS:-1}"; GBS="${GBS:-8}"
    STEPS="${STEPS:-20}"; MASTER_PORT="${MASTER_PORT:-29503}"
    ITER_S_EST="${ITER_S_EST:-5.0}"   # 无实测：按 44% MFU + selective recompute 粗估
    CKPT_TB=0.80                      # 56.5B × ~14B/参数
    ACT=squared_relu; NOTE="文章 56B 架构单机可行性档（非性能对标）" ;;
  *)
    echo "!! CASE 只能是 8b | mfu | 56b，收到 '$CASE'"; exit 2 ;;
esac

DP=$((WORLD / TP / PP))
[ $((DP * TP * PP)) -eq "$WORLD" ] || { echo "!! TP=$TP PP=$PP 无法整除 world=$WORLD"; exit 2; }
[ $((GBS % (MBS * DP))) -eq 0 ]   || { echo "!! GBS=$GBS 不能被 mbs×dp=$((MBS * DP)) 整除"; exit 2; }
ACCUM=$((GBS / (MBS * DP)))

# ---------------------------------------------------------------- 1.1 时长口径
# Megatron 里 --train-iters 是**步数**上限，--exit-duration-in-mins 是**墙钟**上限。
# 长跑必须两个都给，且步数留余量：只给步数 → 吞吐估错就提前结束；只给时间 → 步数
# 默认为 None 时会一直跑。谁先满足谁生效。
DURATION_MINS="${DURATION_MINS:-0}"
case "$DURATION_MINS" in
  ''|*[!0-9]*) echo "!! DURATION_MINS 必须是非负整数分钟（收到 '$DURATION_MINS'）"; exit 2 ;;
esac
if [ "$DURATION_MINS" -gt 0 ]; then
  # 与双机同策略（run_56b_2x8h200.sh 是 TRAIN_ITERS=100000 + EXIT_DURATION_MINS=1410）：
  # 步数只当「不会被触及的上限」，真正收口靠 --exit-duration-in-mins。
  # 好处：不再依赖 ITER_S_EST 反推步数 —— 估大（步数算少）也不会「步数先到 → 只跑了半程」，
  # 日志也变成双机那种 iteration N/ 100000 的样子，两边能直接对表。
  TRAIN_ITERS="${TRAIN_ITERS:-${DURATION_ITERS:-100000}}"
  LOG_INTERVAL="${LOG_INTERVAL:-10}"   # 长跑别每步打一行
  MODE_DESC="墙钟 ${DURATION_MINS} 分钟（train-iters=${TRAIN_ITERS} 仅作上限，由 --exit-duration-in-mins 收口）"
  # mock 数据集的 num_samples = train_iters × GBS，上限给太大会把索引撑大（只是启动慢几秒）
  _samples=$(( TRAIN_ITERS * GBS ))
  if [ "$_samples" -gt 20000000 ]; then
    echo "   ! train-iters×GBS=${_samples} 会让 mock 索引偏大（约 ${_samples} 条样本）；" >&2
    echo "     嫌启动慢可 DURATION_ITERS=20000（对 ${DURATION_MINS} 分钟来说仍远跑不完）" >&2
  fi
else
  TRAIN_ITERS="$STEPS"
  LOG_INTERVAL="${LOG_INTERVAL:-1}"
  MODE_DESC="固定 ${STEPS} 步"
fi
[ "$TRAIN_ITERS" -ge 1 ] || { echo "!! 算出 train-iters=$TRAIN_ITERS，检查 DURATION_MINS/ITER_S_EST"; exit 2; }
# Megatron 断言 lr_warmup_iters < lr_decay_iters 且短跑时不能 >= train_iters
WARMUP=5; (( WARMUP >= TRAIN_ITERS )) && WARMUP=$(( TRAIN_ITERS > 1 ? TRAIN_ITERS - 1 : 0 ))

echo "==================== 单机 8×H200 / CASE=$CASE ===================="
echo "engine=$ENGINE namespace=${NS:-none}  image=$IMAGE"
echo "model=${LAYERS}L/${HIDDEN}/${FFN}/${HEADS}H vocab=$VOCAB act=$ACT"
echo "TP=$TP PP=$PP DP=$DP  MBS=$MBS GBS=$GBS accum=$ACCUM  SEQ=$SEQ"
echo "时长=$MODE_DESC  log_interval=$LOG_INTERVAL"
echo "ATTN_BACKEND=$ATTN_BACKEND TP_COMM_OVERLAP=$TP_COMM_OVERLAP FP8=$FP8 LOW_MEM=$LOW_MEM"
echo "说明：$NOTE"

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<'PLAN'

CASE  模型            TP PP DP  seq  mbs gbs steps  实测基线(step/TFLOP/s/GPU/MFU/GiB)
----  --------------  -- -- --  ---- --- --- -----  ----------------------------------------
8b    Llama-3 8B      8  1  1  4096 1   8   15     985.1 / 200.7 / 20.3% / 17.21
mfu   Llama-3 8B      1  1  8  4096 2   256 20     ~11360 / 556.7 / 56.3% / ~92.8(未单测)
56b   Nemotron-H-56B  8  1  1  4096 1   8   20     无单机实测；DP=1 无优化器分片，默认 LOW_MEM=1

mfu 档 GBS 扫描（截图，40 步/档，MFU 分母 989.5）：
  GBS 256 → 554.1 / 556.7 TFLOP/s/GPU，MFU 56.0% / 56.3%   ← 默认取这档
  GBS 128 → 550.2 / 550.4                ，MFU 55.6% / 55.6%
  GBS  64 → 539.8 / 538.3                ，MFU 54.6% / 54.4%
  GBS 256 的 step ms 是反推值：1521.9 × 8 ÷ (556.7/519.5) = 11363ms，不是实测。

56b 显存估算（非实测）：56.5B×(2+2+4+8)B ÷ TP8 ≈ 113 GB/卡 + 激活 ~10 GB
  ≈ 125 GB / 140.4 GiB，贴边。
LOW_MEM=1   precision-aware optimizer + 自动补 distributed optimizer + selective recompute
LOW_MEM=sgd --optimizer=sgd + selective recompute（最省显存，避开 optimizer 断言）
LOW_MEM=0   文章原生配置，无低显存开关，高 OOM 风险
注意：core_v0.13.1 的 optimizer_config.py:199 要求 precision-aware optimizer 必须配 DO。

24 小时长跑（DURATION_MINS=1440）的换算与磁盘代价：
  CASE  ITER_S_EST  1 小时约几步  24h 折算 train-iters  单套 ckpt 约
  8b    1.0s        3600          103730                0.11 TB
  mfu   11.4s       315           9144                  0.11 TB
  56b   5.0s(估)    720           20786                 0.8 TB
  （ckpt 按 每参 2B bf16 权重 + 4B fp32 master + 8B Adam m/v ≈ 14B 估；
    56b 单机 24h 逐小时存 = 24 × 0.8TB ≈ 19 TB，务必先确认盘）
  ITER_S_EST 与实际吞吐偏离较多时用 ITER_S_EST= 覆盖，脚本只靠它换算步数。
PLAN
  exit 0
fi

# ---------------------------------------------------------------- 2. 前置检查
if [ "$DRY_RUN" != 1 ]; then
  gpu_n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  [ "${gpu_n:-0}" -eq "$GPUS_PER_NODE" ] || { echo "!! 本机 GPU 数 ${gpu_n:-0} != $GPUS_PER_NODE"; exit 2; }
  [ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ] || {
    echo "!! 缺 $ROOT/Megatron-LM/pretrain_gpt.py（core_v0.13.1 源码包）"; exit 2; }
  if [ "$ENGINE" = nerdctl ] && command -v systemctl >/dev/null 2>&1 \
     && ! systemctl is-active containerd >/dev/null 2>&1; then
    echo "!! containerd 未运行：systemctl enable --now containerd"; exit 2
  fi
  # 用 images 列表精确匹配，不用 image inspect：部分 nerdctl 版本没有该子命令，
  # 会把「命令不存在」误报成「镜像不存在」。grep 不加 -q，避免 pipefail 下的 SIGPIPE 误判。
  if ! "${ENGINE_CMD[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
       | grep -Fx "$IMAGE" >/dev/null; then
    echo "!! 在 $ENGINE (namespace=$NS) 的镜像列表里找不到 $IMAGE"
    echo "   nerdctl load 的镜像 docker 看不见；确认同一引擎+同一 namespace（见 00_prepare.sh）"
    exit 2
  fi
fi
if [ "$CASE" = 56b ]; then
  case "$LOW_MEM" in
    1) echo ">> 56b LOW_MEM=1：precision-aware optimizer + distributed optimizer + selective recompute"
       echo "   （DP=1 时自动补 DO，避开 optimizer_config.py:199 断言）" ;;
    sgd) echo ">> 56b LOW_MEM=sgd：--optimizer=sgd + selective recompute，不分配 Adam m/v，最省显存"
         echo "   （只验能否跑起来，loss 曲线不可与文章 Adam 对比）" ;;
    0) echo "!! 警告：56b LOW_MEM=0 接近文章原生配置，单机 DP=1 无优化器分片，约 125 GB/卡，OOM 风险高" >&2
       echo "   建议 LOW_MEM=sgd 或 LOW_MEM=1；若仍 OOM，降 SEQ=2048 或改 CASE=8b" >&2 ;;
  esac
fi

# ---------------------------------------------------------------- 3. 容器环境
ENV_ARGS=(
  -e "PYTHONPATH=$MEGATRON_IN"
  -e "CUDA_DEVICE_MAX_CONNECTIONS=1"
  -e "TRITON_CACHE_DIR=/tmp/triton-cache"
  -e "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache"
  -e "TORCH_CUDA_ARCH_LIST=9.0"
  -e "NVTE_FWD_LAYERNORM_SM_MARGIN=16"
  -e "NVTE_BWD_LAYERNORM_SM_MARGIN=16"
  -e "TORCH_NCCL_AVOID_RECORD_STREAMS=1"
  -e "TORCH_NCCL_HIGH_PRIORITY=1"
  -e "UCX_MEM_MMAP_HOOK_MODE=none"
  -e "UCX_MEM_CUDA_HOOK_MODE=none"
  -e "UCX_ERROR_SIGNALS="
)
# 与双机一致，默认 INFO：首次跑要能从日志里确认「机内 P2P vs SHM、跨机 NET/IB vs
# NET/Socket、有没有 GDRDMA」。INFO 只在建链阶段多打约 1k 行，长跑可忽略。
# 24h 稳态跑嫌吵可 NCCL_DEBUG=WARN；要追时序用 NCCL_DEBUG_SUBSYS=COLL,NET。
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
[ -n "$NCCL_DEBUG" ] && ENV_ARGS+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
echo ">> NCCL_DEBUG=$NCCL_DEBUG（日志里搜 'via ' 可确认传输方式：P2P/CUMEM vs SHM、NET/IB vs NET/Socket）"
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
  ENV_ARGS+=(-e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi
[ "$ATTN_BACKEND" = fused ] && ENV_ARGS+=(-e "NVTE_FUSED_ATTN=1")

# ---------------------------------------------------------------- 4. Megatron 参数
# 有值一律 --flag=value，便于按 --help 逐项裁剪；布尔 flag 裸写。
MEGATRON_ARGS=(
  --use-mcore-models
  --no-mmap-bin-files
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --group-query-attention
  --num-query-groups=8
  --kv-channels=128
  --normalization=RMSNorm
  --attention-dropout=0.0
  --hidden-dropout=0.0
  --attention-backend="$ATTN_BACKEND"
  --tokenizer-type=NullTokenizer
  --vocab-size="$VOCAB"
  --mock-data
  --num-workers=1
  --no-create-attention-mask-in-dataloader
  --log-progress
  --timing-log-option=minmax
  --log-throughput
  --bf16
  --manual-gc
  --num-layers="$LAYERS"
  --hidden-size="$HIDDEN"
  --ffn-hidden-size="$FFN"
  --num-attention-heads="$HEADS"
  --seq-length="$SEQ"
  --max-position-embeddings="$SEQ"
  --tensor-model-parallel-size="$TP"
  --pipeline-model-parallel-size="$PP"
  --micro-batch-size="$MBS"
  --global-batch-size="$GBS"
  --data-cache-path="$DATA_CACHE_IN"
  --train-iters="$TRAIN_ITERS"
  --log-interval="$LOG_INTERVAL"
  --seed="$SEED"
  --distributed-timeout-minutes=60
  --clip-grad=1.0
  --weight-decay=0.1
  --lr-decay-style=cosine
  --lr-warmup-iters="$WARMUP"
)

if [ "$ACT" = swiglu ]; then
  # 与本机 8B 实测 COMMON 完全一致（Llama-3 结构）
  MEGATRON_ARGS+=(
    --swiglu --norm-epsilon=1e-5
    --use-rotary-position-embeddings --rotary-base=500000
    --position-embedding-type=rope
    --make-vocab-size-divisible-by=128
    --split="${SPLIT:-100,0,0}"
    --init-method-std=0.02
    --lr=1e-4 --min-lr=1e-5
  )
else
  # 文章 Nemotron-H-56B 结构：squared-relu、rope base 1e6、131072 词表
  MEGATRON_ARGS+=(
    --squared-relu
    --position-embedding-type=rope
    --rotary-base=1000000
    --rotary-percent=1.0
    --init-method-std=0.014
    --lr=6e-4 --min-lr=6e-6
    --lr-decay-iters=1830030
    --adam-beta1=0.9 --adam-beta2=0.95
  )
fi

# 条件开关：只在真正有收益的拓扑下开，避免 DP=1 的额外 bucket 开销
(( TP > 1 )) && MEGATRON_ARGS+=(--sequence-parallel)
(( TP > 1 )) && [ "$TP_COMM_OVERLAP" = 1 ] && MEGATRON_ARGS+=(--tp-comm-overlap)
if (( DP > 1 )); then
  MEGATRON_ARGS+=(--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather)
fi
if [ "$CASE" = 56b ]; then
  case "$LOW_MEM" in
    1)
      # optimizer_config.py:199 硬断言：precision-aware optimizer 必须配 DO。
      # DP=1 时上面的分支不会加 DO，这里必须补上，否则 8 rank 全崩（11-11 现场实测）。
      (( DP == 1 )) && MEGATRON_ARGS+=(--use-distributed-optimizer)
      MEGATRON_ARGS+=(--use-precision-aware-optimizer --recompute-granularity=selective)
      ;;
    sgd)
      # 不建 Adam m/v 状态，彻底绕开 precision-aware optimizer 的 DO 断言
      MEGATRON_ARGS+=(--optimizer=sgd --recompute-granularity=selective)
      ;;
    0) : ;;
    *) echo "!! LOW_MEM 只能是 1 | 0 | sgd（收到 $LOW_MEM）"; exit 2 ;;
  esac
fi
# 静态防呆：以后任何组合都不允许出现「有 precision-aware 但没 DO」
if printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--use-precision-aware-optimizer'; then
  printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--use-distributed-optimizer' || {
    echo "!! --use-precision-aware-optimizer 必须同时给 --use-distributed-optimizer"
    echo "   （core_v0.13.1 optimizer_config.py:199 的硬断言）"; exit 2; }
fi
if [ "$FP8" = 1 ]; then
  # 25.01 + TE 1.14 实测：format 与 recipe 必须同传，只传 recipe 是假 FP8
  MEGATRON_ARGS+=(--fp8-format=e4m3 --fp8-recipe=delayed
                  --fp8-amax-history-len=1024 --fp8-amax-compute-algo=max --fp8-param-gather)
fi

# --- 时长：墙钟上限（0.13.1 有该参数，type=int，见 args-help 第 763 行）
if [ "$DURATION_MINS" -gt 0 ]; then
  MEGATRON_ARGS+=(--exit-duration-in-mins="$DURATION_MINS")
  # 被调度器 SIGTERM 时先存 ckpt 再退，避免 24h 硬杀（KillWait 30s）丢进度。
  # 注意它只在训练循环里响应，卡在初始化/编译阶段时仍然会被直接杀掉。
  MEGATRON_ARGS+=(--exit-signal-handler)
fi

# --- 评估（eval）：与双机同口径 —— 每 EVAL_INTERVAL 步打一次
#     "| validation loss at iteration N | lm loss value: ... | lm loss PPL: ... |"；
#     eval 前后还会带出 "(min, max) time across ranks (ms)" 那段 timer 汇总
#     （你截图里第 250 步那一段就是，只在 eval 边界出现，log-interval 处不出现）。
#     双机模板（run_56b_2x8h200.sh）是 --eval-iters=1 --eval-interval=50。
#     注意：--split=100,0,0 的档位（8b/mfu）valid 集为 0，评估只会空跑 → 自动关掉；
#     要给它们也开评估，用 SPLIT=969,30,1 CASE=8b ...
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
case "$EVAL_ITERS$EVAL_INTERVAL" in
  ''|*[!0-9]*) echo "!! EVAL_ITERS/EVAL_INTERVAL 必须是非负整数，已按 1/50 处理" >&2
               EVAL_ITERS=1; EVAL_INTERVAL=50 ;;
esac
if [ "$EVAL_ITERS" -le 0 ] || [ "$EVAL_INTERVAL" -le 0 ]; then
  MEGATRON_ARGS+=(--eval-iters=0)
  echo ">> 评估：关闭（EVAL_ITERS=$EVAL_ITERS EVAL_INTERVAL=$EVAL_INTERVAL）"
elif printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--split=100,0,0'; then
  MEGATRON_ARGS+=(--eval-iters=0)
  echo ">> 评估：关闭（--split=100,0,0 的 valid 集为 0；要评估请 SPLIT=969,30,1）"
else
  MEGATRON_ARGS+=(--eval-iters="$EVAL_ITERS" --eval-interval="$EVAL_INTERVAL")
  echo ">> 评估：每 ${EVAL_INTERVAL} 步 1 次（--eval-iters=$EVAL_ITERS），与双机同口径"
fi

# --- checkpoint：只落盘、不自动续训（续训要显式 RESUME=1）
# 滚动保留：只留最近 CKPT_KEEP 套，新的一套「完整落盘」后删掉更老的（实现见 5.6 节）。
# 0.13.1 没有 --save-retain-interval，也没有任何「保留最近 N 个」的开关，只能脚本外部做。
CKPT_KEEP="${CKPT_KEEP:-1}"                 # 1 = 只留最新一套；0 = 不删（只留清单）
CKPT_KEEP_GRACE="${CKPT_KEEP_GRACE:-300}"   # 秒；目录 mtime 在此之内视为还在写，不删
CKPT_ROLL_POLL="${CKPT_ROLL_POLL:-60}"      # 秒；巡检间隔
CKPT_DISK_FORCE="${CKPT_DISK_FORCE:-0}"     # 1 = 空间不够也强行跑（不推荐）
ENABLE_CKPT="${ENABLE_CKPT:-0}"
if [ "$ENABLE_CKPT" = 1 ]; then
  # 不同 LOW_MEM 的优化器状态互不兼容（precision-aware / sgd / 纯 Adam），
  # 分目录存，免得 RESUME=1 把 sgd 的 ckpt 加载进 Adam 里。
  if [ "$CASE" = 56b ]; then CK_SUFFIX="_lowmem${LOW_MEM}"; else CK_SUFFIX=""; fi
  CHECKPOINT_HOST="${CHECKPOINT_HOST:-$WORKSPACE_HOST/checkpoints/single-${CASE}${CK_SUFFIX}}"
  CHECKPOINT_IN="$WORKSPACE_IN/checkpoints/single-${CASE}${CK_SUFFIX}"
  # 默认「约 1 小时一存」：3600s ÷ 每步秒数
  SAVE_INTERVAL="${SAVE_INTERVAL:-$(awk -v t="$ITER_S_EST" 'BEGIN{printf "%d", 3600/t}')}"
  [ "${SAVE_INTERVAL:-0}" -ge 1 ] 2>/dev/null || SAVE_INTERVAL=1
  mkdir -p "$CHECKPOINT_HOST"
  MEGATRON_ARGS+=(--save="$CHECKPOINT_IN" --save-interval="$SAVE_INTERVAL"
                  --ckpt-format=torch_dist --ckpt-fully-parallel-save
                  --ckpt-fully-parallel-load --ckpt-assume-constant-structure --async-save)
  if [ "${RESUME:-0}" = 1 ] && [ -f "$CHECKPOINT_HOST/latest_checkpointed_iteration.txt" ]; then
    echo ">> RESUME=1：从第 $(cat "$CHECKPOINT_HOST/latest_checkpointed_iteration.txt") 步续跑"
    MEGATRON_ARGS+=(--load="$CHECKPOINT_IN")
  else
    echo ">> checkpoint → $CHECKPOINT_HOST（每 $SAVE_INTERVAL 步），单套约 $CKPT_TB TB" >&2
    if [ "$DURATION_MINS" -gt 0 ]; then
      # 实际存几次取决于「到点退出」前跑了多少步，用时间反推更靠谱
      n_save=$(awk -v m="$DURATION_MINS" -v s="$SAVE_INTERVAL" -v t="$ITER_S_EST" \
        'BEGIN{printf "%d", m*60/(s*t) + 1}')
      if [ "$CKPT_KEEP" -gt 0 ]; then
        echo "   ${DURATION_MINS} 分钟共存约 ${n_save} 套；保留最近 ${CKPT_KEEP} 套，其余自动删" \
             "→ 稳态 $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" 'BEGIN{printf "%.2f", k*t}') TB，" \
             "峰值 $(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" 'BEGIN{printf "%.2f", (k+1)*t}') TB" >&2
      else
        echo "   ${DURATION_MINS} 分钟量级约 ${n_save} 套 ≈ $(awk -v a="$n_save" -v b="$CKPT_TB" \
             'BEGIN{printf "%.1f", a*b}') TB；CKPT_KEEP=0 不删旧盘，盘不够就别开 ENABLE_CKPT" >&2
      fi
    fi
  fi
  # 开机前先量盘：写满之后 checkpoint 会以「inline_container.cc: unexpected pos 448 vs 342」
  # 的形式炸掉（11-11 单机 56B 第 720 步实测），真正的首错被后面的 No space left 埋掉。
  # df -Pk 给的是 KB，所以 TB→KB 要乘 1024^3（写成 1024*1024 会差 1024 倍）
  ck_need_kb=$(awk -v k="$CKPT_KEEP" -v t="$CKPT_TB" -v n="${n_save:-1}" \
    'BEGIN{ if (k>0) printf "%d", (k+1)*t*1024*1024*1024*1.05; else printf "%d", n*t*1024*1024*1024*1.05 }')
  ck_avail_kb=$(df -Pk "$CHECKPOINT_HOST" 2>/dev/null | awk 'NR==2{print $4+0}')
  ck_need_gb=$(awk -v a="$ck_need_kb" 'BEGIN{printf "%.0f", a/1048576}')
  ck_avail_gb=$(awk -v a="${ck_avail_kb:-0}" 'BEGIN{printf "%.0f", a/1048576}')
  if [ "${ck_avail_kb:-0}" -lt "$ck_need_kb" ]; then
    echo "!! $CHECKPOINT_HOST 可用 ${ck_avail_gb} GB < 需要 ${ck_need_gb} GB" >&2
    echo "   （CKPT_KEEP=$CKPT_KEEP 单套 $CKPT_TB TB；峰值按 (KEEP+1) 套算）" >&2
    echo "   三条路：① 换大盘 CHECKPOINT_HOST=... ② ENABLE_CKPT=0 只测吞吐" >&2
    echo "           ③ CKPT_DISK_FORCE=1 强行继续（写满会崩，后果自负）" >&2
    [ "$CKPT_DISK_FORCE" = 1 ] || exit 2
    echo "   → CKPT_DISK_FORCE=1，继续" >&2
  else
    echo ">> 盘位 OK：$CHECKPOINT_HOST 可用 ${ck_avail_gb} GB ≥ 需要 ${ck_need_gb} GB" >&2
  fi
fi

# ---------------------------------------------------------------- 5. 参数表自检 + 裁剪
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
  echo ">> dump core_v0.13.1 的 --help → $ARGS_HELP"
  # 必须带 --network host：默认 bridge 走 CNI firewall 插件，主机
  # iptables(nf_tables) 与 nft 不兼容时容器根本起不来（11-11 现场实测）。
  if ! "${ENGINE_CMD[@]}" run --rm --gpus all --network host \
      --shm-size=64g --ulimit memlock=-1:-1 --ulimit stack=67108864 \
      -v "$ROOT:/workspace" -w "$MEGATRON_IN" \
      -e "PYTHONPATH=$MEGATRON_IN" -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
      "$IMAGE" python pretrain_gpt.py --help > "$ARGS_HELP" 2>&1; then
    echo "!! --help dump 失败，$ARGS_HELP 前 6 行：" >&2
    head -6 "$ARGS_HELP" >&2
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
  elif ! help_table_ok "$ARGS_HELP"; then
    echo "!! dump 出来的不是有效参数表（flag 行 < 400），见 $ARGS_HELP"; exit 2
  else
    HAVE_HELP=1
  fi
fi
if [ "$HAVE_HELP" = 1 ]; then
  n=$(grep -cE '^ *--' "$ARGS_HELP" || true)
  echo ">> 参数表 $ARGS_HELP（$n 行 flag）；逐项核对中"
  kept=()
  for e in "${MEGATRON_ARGS[@]}"; do
    if [[ "$e" == --* ]]; then
      f=${e%%=*}
      if ! grep -qE "^ *--${f#--}([ =]|$)" "$ARGS_HELP"; then
        echo "   [裁剪] $f 不在 core_v0.13.1 的 --help 中"; continue
      fi
    fi
    kept+=("$e")
  done
  MEGATRON_ARGS=("${kept[@]}")
  # 56b 低显存档的关键开关若被裁掉，必须显性告知（否则会按高危配置跑）
  if [ "$CASE" = 56b ] && [ "$LOW_MEM" = 1 ] \
     && ! printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--use-precision-aware-optimizer'; then
    echo "!! 警告：--use-precision-aware-optimizer 被裁剪，LOW_MEM 保护失效，OOM 风险高" >&2
  fi
else
  echo "   ! 无 --help 参数表，跳过裁剪（DRY_RUN 可给 ARGS_HELP=...）" >&2
fi

# ---------------------------------------------------------------- 5.4 运行标识
# 日志名自带并行/batch 配置，和截图里 w8_..._tp1_pp1_mbs2_s4096_gbs256 一个风格：
# 事后 ls 一眼能看出每份日志跑的是什么，不用回头翻当时的命令行。
# 命名放在裁剪之后取 dtype，是为了别在 --fp8-* 被裁掉时还标成 fp8。
if printf '%s\n' "${MEGATRON_ARGS[@]}" | grep -q -- '--fp8-format'; then
  DTYPE=fp8e4m3
else
  DTYPE=bf16
fi
RUN_STEM="single_${CASE}_tp${TP}_pp${PP}_dp${DP}_mbs${MBS}_s${SEQ}_gbs${GBS}_${DTYPE}"
# 56b 的三种省显存档行为差别大，必须进文件名，否则没法区分是不是同一个实验
[ "$CASE" = 56b ] && RUN_STEM="${RUN_STEM}_lowmem${LOW_MEM}"
LOG="$LOGDIR_HOST/${RUN_STEM}-${RUN_STAMP}.out"

# ---------------------------------------------------------------- 5.5 GPU 采样（dmon）
# 回答「24h 有没有热降频 / 功耗撞墙 / 显存缓慢爬升 / NVLink 打满」这类只看训练日志
# 看不出来的问题。跑在宿主机而不是容器内：镜像里不一定有 nvidia-smi，而且 dmon
# 读的是物理卡，容器内外看到的是同一块。
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
  # 先探一次（-c 1 采一个点即退）：某些驱动/卡不支持 t(PCIe/NVLink) 或 e(ECC)，
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

# ---------------------------------------------------------------- 5.6 checkpoint 滚动保留
# 需求：「第二次保存时把第一次删掉」= 只保留最近 N 套（CKPT_KEEP，默认 1）。
# core_v0.13.1 既没有 --save-retain-interval，也没有任何「保留最近 N 个」的开关，
# 而单套 56.5B checkpoint 约 0.8 TB（bf16 权重 113 GB + fp32 master 226 GB
# + Adam m/v 453 GB），24h 逐小时存就是十几 TB，必然写满（11-11 第 720 步已实测）。
#
# 判定与安全边界（每条都对应一次真实翻车）：
#   ① 一套 ckpt「可用」= 有完成标记 + 体积≥期望的 1/3。标记：torch_dist 是
#      .metadata/common.pt，legacy torch 是 mp_rank_*/model_optim_rng.pt。只看标记不够：
#      盘写满时 .metadata 可能已落、分片却截断，此时它不算数。
#   ② latest_checkpointed_iteration.txt 指向的那个永不删（RESUME 靠它）。
#   ③ 比「最新可用 ckpt」还新的目录一律不碰 —— 可能就是正在写的那一套。
#   ④ 年龄按 stat 精确到秒：可用套要 ≥ CKPT_KEEP_GRACE（默认 300s）；
#      残套要 ≥ max(4×grace, 30min)（0.8TB 落盘本身要 5~10min）。
#   ⑤ 先 mv 成 iter_x.deleting.<pid> 再 rm -rf：mv 原子，重复触发也只有一个成功；
#      残留墓碑下一轮再清。
CKPT_ROLL_LOG="$LOGDIR_HOST/ckpt-retention-${RUN_STAMP}.log"
CKPT_TB_MB=$(awk -v t="${CKPT_TB:-0}" 'BEGIN{printf "%d", t*1048576}')   # 体积校验用（MB）

# 判定一套 ckpt 是否「能用」：① 有完成标记 ② 体积站得住
#   ① torch_dist 落 .metadata / common.pt；legacy torch 落 mp_rank_*/model_optim_rng.pt。
#   ② 只靠标记不够：盘写满时 .metadata 可能已落、分片却截断（11-11 第 720 步就是这么炸的）。
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
    if mv -- "$d" "$d.deleting.$$" 2>/dev/null; then
      rm -rf -- "$d.deleting.$$" 2>/dev/null || true
      del=$((del+1)); freed=$((freed+sz))
      echo "[$(date '+%F %T')] removed $(basename "$d") ${sz} MB" >> "$CKPT_ROLL_LOG"
    fi
  done
  if [ "$del" -gt 0 ]; then
    echo "[$(date '+%F %T')] 本轮 removed=$del freed=${freed} MB kept=$kept" >> "$CKPT_ROLL_LOG"
  fi
  return 0
}

stop_ckpt_roll() {
  [ -n "${CKPT_ROLL_PID:-}" ] || return 0
  kill "$CKPT_ROLL_PID" 2>/dev/null || true
  wait "$CKPT_ROLL_PID" 2>/dev/null || true
  CKPT_ROLL_PID=""
}

# ---------------------------------------------------------------- 6. 启动
# 容器只挂了 $ROOT。CHECKPOINT_HOST / DATA_CACHE_HOST 若被指到 $ROOT 之外（大盘、共享盘），
# 必须再 bind 回容器内的原路径；否则 --save / --data-cache-path 指向的是容器里那个空目录，
# 日志看着在存，文件其实写到别处 —— 事后才发现盘没变大。
EXTRA_MOUNTS=()
add_extra_mount() {   # $1=宿主路径 $2=容器内路径
  case "$1" in "$ROOT"|"$ROOT"/*) return 0 ;; esac
  EXTRA_MOUNTS+=( -v "$1:$2" )
  echo "   bind 外部目录：$1 → $2"
}
add_extra_mount "$DATA_CACHE_HOST" "$DATA_CACHE_IN"
if [ "$ENABLE_CKPT" = 1 ]; then add_extra_mount "$CHECKPOINT_HOST" "$CHECKPOINT_IN"; fi

TORCHRUN_ARGS=(--nproc_per_node="$GPUS_PER_NODE" --nnodes=1
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
CMD=("${ENGINE_CMD[@]}" run --rm --gpus all --network host --shm-size=64g
  --ulimit memlock=-1:-1 --ulimit stack=67108864
  -v "$ROOT:/workspace" -w "$MEGATRON_IN"
  ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}
  "${ENV_ARGS[@]}" "$IMAGE"
  torchrun "${TORCHRUN_ARGS[@]}" pretrain_gpt.py "${MEGATRON_ARGS[@]}")

if [ "$DRY_RUN" = 1 ]; then
  echo ">> DRY_RUN：只打印命令，不启动容器"
  echo "   训练日志 → $LOG"
  echo "   GPU 采样 → $DMON_LOG（每 ${DMON_INTERVAL}s，-s $DMON_METRICS）"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

echo ">> 日志：$LOG"
start_dmon
# checkpoint 滚动保留：开了 ckpt 才需要。第一轮立即跑（清理上次残留），之后每 CKPT_ROLL_POLL 秒一次。
CKPT_ROLL_PID=""
if [ "$ENABLE_CKPT" = 1 ] && [ "$CKPT_KEEP" -gt 0 ]; then
  echo "[$(date '+%F %T')] start keep=$CKPT_KEEP grace=${CKPT_KEEP_GRACE}s poll=${CKPT_ROLL_POLL}s root=$CHECKPOINT_HOST" \
    >> "$CKPT_ROLL_LOG"
  ( while true; do
      ckpt_roll_once "$CHECKPOINT_HOST" "$CKPT_KEEP" "$CKPT_TB_MB" || true
      sleep "$CKPT_ROLL_POLL"
    done ) &
  CKPT_ROLL_PID=$!
  echo ">> checkpoint 保留：只留最近 $CKPT_KEEP 套（删除记录 → $CKPT_ROLL_LOG）"
fi
# dmon / 滚动删除都是后台子进程，任何退出路径都得收掉，否则训练结束它们还挂着。
# INT/TERM 走 exit，由 EXIT 统一清理，避免只 trap EXIT 时信号被忽略。
cleanup() { stop_ckpt_roll; stop_dmon; }
trap 'cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e    # pipefail 下要显式回收 torchrun 的退出码，不能被 set -e 提前截断
"${CMD[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------- 7. 结果摘要
summarize() {
  local log=$1 n ms med_ms t med_t mfu mem
  # 跳过前 2 步（首次编译/预热），取稳定段中位数
  grep -oE 'elapsed time per iteration \(ms\): [0-9.]+' "$log" 2>/dev/null \
    | awk '{print $NF}' | tail -n +3 | sort -n > "$LOGDIR_HOST/.ms.$$" || true
  grep -oE 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" 2>/dev/null \
    | awk '{print $NF}' | tail -n +3 | sort -n > "$LOGDIR_HOST/.tf.$$" || true
  median() { awk '{a[NR]=$1} END{if(NR==0) print "NA"; else if(NR%2) print a[(NR+1)/2]; else printf "%.1f",(a[NR/2]+a[NR/2+1])/2}' "$1"; }
  n=$(wc -l < "$LOGDIR_HOST/.ms.$$")
  if [ "${n:-0}" -gt 0 ]; then
    med_ms=$(median "$LOGDIR_HOST/.ms.$$")
    med_t=$(median "$LOGDIR_HOST/.tf.$$")
    mfu=$(awk -v t="$med_t" -v p="$PEAK_TFLOPS" 'BEGIN{printf "%.1f", t/p*100}')
    mem=$(grep -oE 'max reserved: [0-9.]+' "$log" 2>/dev/null | awk '{print $NF}' | sort -n | tail -1)
    echo "----------------------------------------------------------------"
    echo "样本 ${n} 步（跳过前 2 步）| step 中位 ${med_ms} ms | TFLOP/s/GPU 中位 ${med_t}"
    echo "MFU@${PEAK_TFLOPS} = ${mfu}%"
    if [ -n "${mem:-}" ]; then
      echo "峰值显存 max reserved ≈ $(awk -v m="$mem" 'BEGIN{printf "%.1f",m/1024}') GiB / ${VRAM_GIB} GiB"
    else
      echo "峰值显存：日志里没有 max reserved 行（可加 --timing-log-level=2 或用 nvidia-smi 采样）"
    fi
    echo "对照基线：$NOTE"
  else
    echo "!! 日志里没有 iteration 行，训练未进入稳态；看 $LOG 末尾定位" >&2
  fi
  rm -f "$LOGDIR_HOST/.ms.$$" "$LOGDIR_HOST/.tf.$$"
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
  # 判读：pviol/tviol 非 0 = 撞到功耗墙/温度墙，MFU 会掉但日志里的 step 时间不一定炸；
  #       gtemp 长时间 >85C、SM 均值明显低于基线档，都是降频特征。
  #       NVLink 峰值接近 450 = 单向打满；均值高但 SM 也高 = overlap 正常；
  #       均值低、峰值高 = 通信是突发式的，盯 SM 有没有掉 0 比盯流量有用。
}
echo "==================== 单机测试结果 CASE=$CASE ===================="
stop_ckpt_roll    # 先停滚动删除，避免统计时还在删
stop_dmon         # 再停采样，保证 CSV 刷完；后面 EXIT trap 再调一次是空操作
summarize "$LOG"
echo "---- GPU 采样（dmon）----"
summarize_dmon "$DMON_LOG"
if [ "$ENABLE_CKPT" = 1 ] && [ -f "${CKPT_ROLL_LOG:-/nonexistent}" ]; then
  echo "---- checkpoint 保留（CKPT_KEEP=$CKPT_KEEP）----"
  echo "  删除记录 → $CKPT_ROLL_LOG"
  grep -c "removed iter_" "$CKPT_ROLL_LOG" 2>/dev/null | \
    awk '{print "  本次已删除 " $1 " 套旧 ckpt"}'
fi
echo ">> torchrun 退出码 rc=$rc"
exit "$rc"
