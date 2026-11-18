#!/usr/bin/env bash
# =============================================================================
# T4 断点续训：从 03 产出的 checkpoint 继续，而不是重新从基座权重开始
#
# 【为什么要单独一个脚本】
# 03 用 --finetune true：加载权重但重置 iteration / 优化器状态，是「从已有权重开始 CPT」。
# 04 用 --finetune false：连 iteration、lr schedule、优化器状态一起恢复，是「严格续跑」。
# 因此 03 默认 --no_save_optim true（省磁盘），而 04 必须设 NO_SAVE_OPTIM=false，
# 否则下一次没有优化器状态可恢复（Mcore-Bridge 文档明确说明了这一点）。
#
# 【两种 checkpoint 加载方式，按现场版本选一个】
#   RESUME_MODE=bridge（默认）：保留 --model（safetensors 基座）+ --load <ckpt>
#   RESUME_MODE=mcore        ：改用 --mcore_model <ckpt>（03 里设 MCORE_MODEL）
# 具体支持哪个参数，以 T0 生成的 logs/pt_help_<key>.txt 为准；
# 也可用 RESUME_ARGS 完全覆盖（例如 RESUME_ARGS='--load /xxx --finetune false'）。
#
# 用法：
#   bash scripts/04_resume.sh 27b
#   RESUME_FROM=/xxx/checkpoint-50 bash scripts/04_resume.sh 27b
#   DRY_RUN=1 RESUME_MODE=mcore bash scripts/04_resume.sh flash_next
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
install_err_trap
PROFILE="${1:-$MODEL_PROFILE}"
load_profile "$PROFILE"

RESUME_MODE="${RESUME_MODE:-bridge}"
RESUME_FROM="${2:-${RESUME_FROM:-}}"

# 自动定位最近的 checkpoint（跳过冒烟产物）
if [[ -z "$RESUME_FROM" ]]; then
  RESUME_FROM="$(find "$OUTPUT_ROOT/$MODEL_KEY" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort | tail -1 || true)"
fi
[[ -n "$RESUME_FROM" && -d "$RESUME_FROM" ]] || die "未找到 checkpoint，请显式传 RESUME_FROM=<目录>"

# 让 04 的输出与 03 分开，避免覆盖原 run
export OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/${MODEL_KEY}/resume_$(date '+%Y%m%d_%H%M%S')}"
export RUN_TAG="${RUN_TAG:-resume_$MODEL_KEY}"
export FINETUNE=false
export NO_SAVE_OPTIM=false
export NO_SAVE_RNG=false
export SAVE_STEPS="${SAVE_STEPS:-50}"
# 时长模式下不要预设步数：03 需要「用户没给 TRAIN_ITERS」才能把它抬到上限，
# 否则续训会先跑完步数、时长看起来没生效。
if (( ${DURATION_MINS:-0} > 0 )); then
  log "时长模式：TRAIN_ITERS 交由 03 抬到 ${DURATION_ITERS_CAP:-100000} 上限"
else
  export TRAIN_ITERS="${TRAIN_ITERS:-$(( ${SAVE_STEPS:-50} * 2 ))}"
fi

# 默认续训参数；与 03 的 --finetune 会重复，故由 RESUME_ARGS 之外的 FINETUNE 统一控制
case "$RESUME_MODE" in
  bridge)
    export RESUME_ARGS="${RESUME_ARGS:---load $RESUME_FROM}"
    export MCORE_MODEL=""
    ;;
  mcore)
    export RESUME_ARGS="${RESUME_ARGS:-}"
    export MCORE_MODEL="$RESUME_FROM"
    ;;
  *) die "未知 RESUME_MODE=$RESUME_MODE（可选 bridge / mcore）" ;;
esac

# 版本兜底：确认 --help 里真的有我们要用的参数
HELP_FILE="$LOG_ROOT/pt_help_${MODEL_KEY}.txt"
if [[ -f "$HELP_FILE" ]]; then
  if [[ "$RESUME_MODE" == "bridge" ]] && ! awk '/--load/{f=1} END{exit !f}' "$HELP_FILE"; then
    warn "当前版本 megatron pt --help 未出现 --load；请改用 RESUME_MODE=mcore 或自定义 RESUME_ARGS"
  fi
  if [[ "$RESUME_MODE" == "mcore" ]] && ! awk '/--mcore_model/{f=1} END{exit !f}' "$HELP_FILE"; then
    warn "当前版本未出现 --mcore_model；请检查 $HELP_FILE"
  fi
else
  warn "未找到 $HELP_FILE，建议先跑 00_preflight.sh 探测 CLI 参数"
fi

log "=== T4 断点续训 | mode=$RESUME_MODE | from=$RESUME_FROM ==="
log "iters=$TRAIN_ITERS finetune=$FINETUNE no_save_optim=$NO_SAVE_OPTIM"
bash "$PROJECT_ROOT/scripts/03_continue_pt.sh" "$PROFILE"
log "=== T4 执行结束 ==="
