#!/usr/bin/env bash
# =============================================================================
# 容器执行入口：把项目脚本放进镜像里跑（默认 RUN_IMAGE = 官方 swift4.5.3 镜像）
#
#   bash scripts/12_run_in_container.sh scripts/00_preflight.sh 27b
#   bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b
#   bash scripts/12_run_in_container.sh --shell        # 起交互容器，进去手动跑 00~05
#   CONTAINER_DRY_RUN=1 bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b  # 只打印容器命令
#   DRY_RUN=1 bash scripts/12_run_in_container.sh scripts/02_smoke.sh 27b          # 容器内 dry-run
#
# 【现场配置】config/cluster.env 由 lib.sh 最先加载（权重路径、GPU 台数、IP、镜像），
# 本脚本把这些值透传进容器；容器内 config/env.sh 用 ${VAR:-...} 保留它们。
#
# 【挂载约定】沿用现场 e2e-megatron-56b-2501 的坑位经验：
#   MODEL_ROOT / MODELSCOPE_CACHE 都 bind 回容器「相同路径」，
#   否则容器内看不到 /mnt/shared/... 会导致权重与 dataset cache 读不到。
# =============================================================================
set -euo pipefail
# 宿主侧不启用 CUDA compat（只对容器内有意义）；CUDA_COMPAT 原值仍通过 -e 透传
export SKIP_CUDA_COMPAT_AUTO=1
source "$(dirname "$0")/lib.sh"   # lib.sh 已最先加载 config/cluster.env

IMAGE="${IMAGE:-$RUN_IMAGE}"
require_cmd "$CONTAINER_ENGINE"

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "--shell" || "$MODE" == "shell" ]]; then
  TARGET=(bash)
  INTERACTIVE=1
else
  SCRIPT="$MODE"
  shift
  if [[ "$SCRIPT" == /* ]]; then SCRIPT_IN="$SCRIPT"; else SCRIPT_IN="$PROJECT_IN/$SCRIPT"; fi
  TARGET=(bash "$SCRIPT_IN" "$@")
  INTERACTIVE=0
fi

# --- 挂载 -------------------------------------------------------------------
# dry-run 时目录在本机不一定存在，但仍把挂载参数完整打印出来。
DRY_SHOW="${CONTAINER_DRY_RUN:-0}"
MOUNTS=()
MOUNT_SEEN="|"
add_mount() {   # $1 = host:container，重复的挂载只保留一次
  local spec="$1"
  case "$MOUNT_SEEN" in *"|$spec|"*) return 0 ;; esac
  MOUNTS+=(-v "$spec"); MOUNT_SEEN+="$spec|"
}
add_mount "$PROJECT_HOST:$PROJECT_IN"
if [[ "$MODEL_ROOT_BIND" == "1" ]] && { [[ -d "$MODEL_ROOT" ]] || [[ "$DRY_SHOW" == "1" ]]; }; then
  add_mount "$MODEL_ROOT:$MODEL_ROOT"
fi
if [[ "$CACHE_BIND" == "1" ]] && { [[ -d "$MODELSCOPE_CACHE" ]] || [[ "$DRY_SHOW" == "1" ]]; }; then
  add_mount "$MODELSCOPE_CACHE:$MODELSCOPE_CACHE"
fi
# 数据集放在项目目录之外时（如 /mnt/shared/datasets），必须同样路径 bind 进容器，
# 否则容器里看不到，--dataset 直接报路径不存在。放项目内则已被上面的项目挂载覆盖。
if [[ -n "${DATA_ROOT:-}" && "$DATA_ROOT" != "${PROJECT_HOST%/}"/* ]] \
   && { [[ -d "$DATA_ROOT" ]] || [[ "$DRY_SHOW" == "1" ]]; }; then
  add_mount "$DATA_ROOT:$DATA_ROOT"
fi
# DATASET_SPEC 直接给了容器外的绝对路径时，把它（或所在目录）一起挂进去
_ds_path="${DATASET_SPEC:-}"; _ds_path="${_ds_path%%#*}"   # 去掉 '#5000' 这类后缀
if [[ "$_ds_path" == /* && "$_ds_path" != "${PROJECT_HOST%/}"/* ]]; then
  if [[ -d "$_ds_path" ]]; then add_mount "$_ds_path:$_ds_path"
  elif [[ -d "$(dirname "$_ds_path")" ]]; then add_mount "$(dirname "$_ds_path"):$(dirname "$_ds_path")"
  fi
fi
unset _ds_path
if [[ -n "$CONTAINER_EXTRA_MOUNTS" ]]; then
  read -r -a extra_mounts <<< "$CONTAINER_EXTRA_MOUNTS"
  MOUNTS+=("${extra_mounts[@]}")
fi

# --- 宿主路径 → 容器路径 -----------------------------------------------------
# 项目目录在容器里是 $PROJECT_IN（如 /workspace/qwen38-cpt），与宿主路径不同。
# 若把宿主的 $PROJECT_ROOT/data/xxx.jsonl 原样 -e 进去，容器里根本不存在该路径。
# 只改写以 $PROJECT_HOST 开头的绝对路径，其余（dataset id、/mnt/shared/...）原样保留。
_to_in() {
  local val="$1"
  case "$val" in
    "${PROJECT_HOST%/}"/*) printf '%s%s' "$PROJECT_IN" "${val#"${PROJECT_HOST%/}"}" ;;
    *)                     printf '%s' "$val" ;;
  esac
}

# --- 环境变量透传（只传已设置的，避免把容器内默认值覆盖成空） --------------
PASS_VARS=(
  MODEL_PROFILE MODEL_27B_DIR MODEL_FLASH_NEXT_DIR
  DATASET_ID DATASET_SPEC DATASET_FULL_SPEC SMOKE_TEXT_JSONL SMOKE_MESSAGES_JSONL DATA_ROOT
  GPUS_PER_NODE NPROC_PER_NODE CUDA_VISIBLE_DEVICES NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  PYTORCH_CUDA_ALLOC_CONF PLE_CPU_OFFLOAD
  TP PP EP ETP CP SP MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_LENGTH TRAIN_ITERS
  LR MIN_LR LR_WARMUP_ITERS SAVE_STEPS EVAL_STEPS RECOMPUTE_GRANULARITY
  RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS TUNER_TYPE LORA_RANK LORA_ALPHA LORA_TARGET
  STREAMING PACKING ATTENTION_BACKEND FINETUNE NO_SAVE_OPTIM NO_SAVE_RNG
  DURATION_MINS DURATION_ITERS_CAP LOG_THROUGHPUT LOGGING_STEPS DMON DMON_INTERVAL DMON_METRICS
  CUDA_COMPAT CUDA_COMPAT_DIR CUDA_COMPAT_MIN_DRIVER
  OPTIMIZER_CPU_OFFLOAD
  MOE_ARGS EXTRA_ARGS RESUME_ARGS RESUME_MODE RESUME_FROM MCORE_MODEL MEGATRON_SUBCMD
  OUT_DIR RUN_TAG DRY_RUN RUN_INFER
)
ENVS=(-e "IMAGE_NAME=$IMAGE"
      -e "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
      -e "MODEL_ROOT=$MODEL_ROOT"
      -e "PROJECT_IN=$PROJECT_IN")
for v in "${PASS_VARS[@]}"; do
  if [[ -n "${!v:-}" ]]; then
    ENVS+=(-e "$v=$(_to_in "${!v}")")
  fi
done

RUN_OPTS=(--gpus "$CONTAINER_GPUS" --network host --shm-size "$CONTAINER_SHM"
          --ulimit memlock=-1 --ulimit stack=67108864 -w "$PROJECT_IN")
if [[ "$INTERACTIVE" == "1" ]]; then
  RUN_OPTS=(-it --name "$CONTAINER_NAME" "${RUN_OPTS[@]}")
else
  RUN_OPTS=(--rm "${RUN_OPTS[@]}")
fi

CMD=("${CTR_CMD[@]}" run "${RUN_OPTS[@]}" "${MOUNTS[@]}" "${ENVS[@]}" "$IMAGE" "${TARGET[@]}")

case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*) warn "当前是 Windows/Git Bash，路径无法作为 Linux 容器挂载；请在 Linux GPU 集群上执行本脚本" ;;
esac

log "镜像: $IMAGE"
log "命令: ${CMD[*]}"
if [[ "${CONTAINER_DRY_RUN:-0}" == "1" ]]; then
  log "CONTAINER_DRY_RUN=1，跳过执行（DRY_RUN 仍会透传进容器）"
  exit 0
fi

# --- GPU 采样（宿主机侧） ---------------------------------------------------
# dmon 读的是物理卡，容器内外同一块；镜像里不一定有 nvidia-smi，所以放在这里。
# 训练结束（含异常退出）都要收掉，否则后台进程会一直写 CSV。
DMON_TAG="${RUN_TAG:-$(basename "${SCRIPT:-interactive}" .sh)}"
start_dmon "$DMON_TAG"
trap 'stop_dmon' EXIT INT TERM

set +e
"${CMD[@]}"
rc=$?
set -e

stop_dmon
summarize_dmon
exit "$rc"
