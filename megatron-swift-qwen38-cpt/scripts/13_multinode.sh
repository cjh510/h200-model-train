#!/usr/bin/env bash
# =============================================================================
# 多机编排：通过 ssh 在每台节点上启动同一个容器命令，并注入各自的 NODE_RANK。
#
#   # 1) 先填现场配置（权重路径 / GPU 台数 / IP / master），见 config/cluster.env
#   # 2) 连通性与路径自检（推荐先跑）
#   bash scripts/13_multinode.sh --check
#   # 3) 只打印每个节点将执行的命令
#   DRY_RUN=1 bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b
#   # 4) 真正启动（所有节点并行，日志落 logs/multinode/<时间戳>/）
#   bash scripts/13_multinode.sh scripts/03_continue_pt.sh 27b
#   bash scripts/13_multinode.sh scripts/02_smoke.sh 27b
#
# 也可不编辑文件，直接用环境变量覆盖：
#   MODEL_ROOT=/data/models NNODES=2 GPUS_PER_NODE=8 \
#   NODE_IPS=192.0.2.21,192.0.2.22 MASTER_ADDR=192.0.2.21 \
#   PROJECT_REMOTE_DIR=/root/megatron-swift-qwen38-cpt \
#   DRY_RUN=1 bash scripts/13_multinode.sh scripts/00_preflight.sh 27b
#
# 【前提】所有节点：项目目录 / 权重 / MODELSCOPE_CACHE 路径一致且共享、
#         镜像已拉取（SKIP_PULL=1 默认）、MASTER_PORT 可达、ssh 免密已配好。
# =============================================================================
set -euo pipefail
# 编排层不启用 CUDA compat（只对容器内有意义）；原值经 REMOTE_ENV 透传到各节点
export SKIP_CUDA_COMPAT_AUTO=1
source "$(dirname "$0")/lib.sh"   # lib.sh 已最先加载 config/cluster.env

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  echo "用法: bash scripts/13_multinode.sh <scripts/xxx.sh|--check> [args...]" >&2
  exit 2
fi
shift || true

IFS=',' read -r -a NODES <<< "$NODE_IPS"
for i in "${!NODES[@]}"; do NODES[$i]="${NODES[$i]// /}"; done
read -r -a SSH_OPTS_ARR <<< "$SSH_OPTS"

MASTER="$MASTER_ADDR"
if [[ "$NNODES" -gt 1 && ( -z "$MASTER" || "$MASTER" == "127.0.0.1" ) && ${#NODES[@]} -gt 0 ]]; then
  MASTER="${NODES[0]}"
  warn "MASTER_ADDR 未指定，自动取第一个节点: $MASTER"
fi

# --- --check：ssh / 目录 / 镜像自检 -----------------------------------------
if [[ "$ACTION" == "--check" ]]; then
  log "现场自检：引擎=$CONTAINER_ENGINE namespace=$CONTAINER_NAMESPACE 镜像=$RUN_IMAGE"
  log "  MODEL_ROOT=$MODEL_ROOT"
  log "  MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
  log "  PROJECT_REMOTE_DIR=$PROJECT_REMOTE_DIR"
  log "  NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE MASTER_ADDR=$MASTER PORT=$MASTER_PORT"
  if [[ ${#NODES[@]} -eq 0 ]]; then
    warn "NODE_IPS 为空；单机请直接跑 12_run_in_container.sh（不需要 13）"
    exit 0
  fi
  rc=0
  for i in "${!NODES[@]}"; do
    ip="${NODES[$i]}"
    remote="test -d $(printf '%q' "$PROJECT_REMOTE_DIR") && test -d $(printf '%q' "$MODEL_ROOT") && test -d $(printf '%q' "$MODELSCOPE_CACHE") && ${CTR_CMD[*]} images --format '{{.Repository}}:{{.Tag}}' | grep -Fq $(printf '%q' "$RUN_IMAGE")"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      log "  [DRY]  node$i $ip: ssh $SSH_USER@$ip \"$remote\""
      continue
    fi
    if ssh "${SSH_OPTS_ARR[@]}" "$SSH_USER@$ip" "$remote" >/dev/null 2>&1; then
      log "  [OK]   node$i $ip：目录 + 镜像齐备"
    else
      log "  [FAIL] node$i $ip：目录缺失 / 镜像未拉取 / ssh 不通（SKIP_PULL=1 时请先在各节点拉镜像）"
      rc=1
    fi
  done
  exit "$rc"
fi

# --- 单机快捷路径 ------------------------------------------------------------
if [[ "$NNODES" -le 1 || ${#NODES[@]} -eq 0 ]]; then
  log "单机模式：NNODES=$NNODES，NODE_IPS=${NODE_IPS:-<空>}，直接调用 12"
  bash "$PROJECT_ROOT/scripts/12_run_in_container.sh" "$ACTION" "$@"
  exit $?
fi

if [[ ${#NODES[@]} -ne "$NNODES" ]]; then
  die "NODE_IPS 有 ${#NODES[@]} 个节点，但 NNODES=$NNODES；两者必须一致（顺序即 NODE_RANK）"
fi

# --- 组装每节点命令 ---------------------------------------------------------
SCRIPT_ARG="$(printf '%q' "$ACTION")"
ARGS_STR=""
for a in "$@"; do ARGS_STR+=" $(printf '%q' "$a")"; done

REMOTE_ENV="NODE_RANK=__RANK__ NNODES=$NNODES MASTER_ADDR=$MASTER MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE GPUS_PER_NODE=$GPUS_PER_NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
MODEL_ROOT=$MODEL_ROOT MODEL_27B_DIR=$MODEL_27B_DIR MODEL_FLASH_NEXT_DIR=$MODEL_FLASH_NEXT_DIR \
MODELSCOPE_CACHE=$MODELSCOPE_CACHE RUN_IMAGE=$RUN_IMAGE PROJECT_REMOTE_DIR=$PROJECT_REMOTE_DIR \
SKIP_PULL=$SKIP_PULL CONTAINER_ENGINE=$CONTAINER_ENGINE CONTAINER_NAMESPACE=$CONTAINER_NAMESPACE \
DURATION_MINS=$DURATION_MINS LOG_THROUGHPUT=$LOG_THROUGHPUT LOGGING_STEPS=${LOGGING_STEPS:-} \
DMON=$DMON DMON_INTERVAL=$DMON_INTERVAL DMON_METRICS=$DMON_METRICS \
CUDA_COMPAT=${CUDA_COMPAT:-auto} CUDA_COMPAT_MIN_DRIVER=${CUDA_COMPAT_MIN_DRIVER:-580} \
TP=${TP:-} PP=${PP:-} EP=${EP:-} ETP=${ETP:-} CP=${CP:-} SP=${SP:-} \
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-} \
TRAIN_ITERS=${TRAIN_ITERS:-} TUNER_TYPE=${TUNER_TYPE:-} \
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-} SAVE_STEPS=${SAVE_STEPS:-}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$LOG_ROOT/multinode/$TS"

log "任务: $ACTION $*"
log "拓扑: $NNODES 节点 × $GPUS_PER_NODE 卡, master=$MASTER:$MASTER_PORT"
log "镜像: $RUN_IMAGE (SKIP_PULL=$SKIP_PULL)"
log "时长: DURATION_MINS=$DURATION_MINS（0=按步数） | GPU 采样: DMON=$DMON/${DMON_INTERVAL}s"

if [[ "${DRY_RUN:-0}" != "1" ]]; then mkdir -p "$LOG_DIR"; fi
PIDS=(); LABELS=(); LOGS=()
for i in "${!NODES[@]}"; do
  ip="${NODES[$i]}"
  env_str="${REMOTE_ENV//__RANK__/$i}"
  remote_cmd="cd $(printf '%q' "$PROJECT_REMOTE_DIR") && env $env_str bash scripts/12_run_in_container.sh $SCRIPT_ARG$ARGS_STR"
  ssh_cmd=(ssh "${SSH_OPTS_ARR[@]}" "$SSH_USER@$ip" "$remote_cmd")
  LOGF="$LOG_DIR/node${i}_${ip}.log"
  log "node$i $ip -> ${ssh_cmd[*]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then continue; fi
  "${ssh_cmd[@]}" > "$LOGF" 2>&1 &
  PIDS+=("$!"); LABELS+=("node$i $ip"); LOGS+=("$LOGF")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  log "DRY_RUN=1，未实际启动任何节点"
  exit 0
fi

rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    log "[OK]   ${LABELS[$i]}  日志 ${LOGS[$i]}"
  else
    log "[FAIL] ${LABELS[$i]}  日志 ${LOGS[$i]}"
    rc=1
  fi
done
log "多机任务结束 rc=$rc，日志目录 $LOG_DIR"
exit "$rc"
