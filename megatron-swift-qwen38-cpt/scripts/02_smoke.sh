#!/usr/bin/env bash
# =============================================================================
# T2 冒烟测试：真实数据的本地抽样子集 + 2 个 iteration + 2048 上下文
#
# 目的：在烧掉长跑时间之前，先验证
#   1) Mcore-Bridge 能否直接加载已有 safetensors 权重（新架构是否真的接进来了）
#   2) 混合线性注意力 / MoE / PLE / MTP 在 Megatron 侧能否前向 + 反向
#   3) 纯文本 CPT 数据能否喂进 VL（ConditionalGeneration）模型
#   4) checkpoint 能否以 safetensors 落盘
#
# 前置：数据已下载。若还没有冒烟子集，本脚本会从 RAW_DATA_DIR 自动抽 2000 条。
#   RAW_DATA_DIR 默认 $DATA_ROOT/chinese-c4（目录内有 .jsonl 或 .jsonl.zst 即可，递归查找）
#
# 用法：
#   bash scripts/02_smoke.sh [27b|flash_next]
#   RAW_DATA_DIR=/data/chinese-c4 SAMPLES=500 bash scripts/02_smoke.sh 27b
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
install_err_trap
PROFILE="${1:-$MODEL_PROFILE}"

# --- 冒烟子集：缺失则从已下载的数据里抽 -------------------------------------
if [[ ! -f "$SMOKE_TEXT_JSONL" ]]; then
  RAW_DATA_DIR="${RAW_DATA_DIR:-$DATA_ROOT/chinese-c4}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "DRY_RUN=1：冒烟子集不存在，将从 $RAW_DATA_DIR 抽取（跳过实际生成）"
  else
    [[ -d "$RAW_DATA_DIR" ]] || die "找不到原始数据目录: $RAW_DATA_DIR
       用 RAW_DATA_DIR=<含 .jsonl 或 .jsonl.zst 的目录> 指定；整库下载可执行：
       modelscope download --dataset $DATASET_ID --local_dir $DATA_ROOT/chinese-c4"
    require_cmd python
    log "生成冒烟子集：$RAW_DATA_DIR -> $SMOKE_TEXT_JSONL（${SAMPLES:-2000} 条）"
    python "$PROJECT_ROOT/tools/make_subset.py" \
      --input-dir "$RAW_DATA_DIR" \
      --shards "${SMOKE_SHARDS:-2}" \
      --n "${SAMPLES:-2000}" \
      --out-text "$SMOKE_TEXT_JSONL" \
      --out-messages "$SMOKE_MESSAGES_JSONL"
  fi
fi

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$SMOKE_TEXT_JSONL" ]]; then
  die "冒烟数据集仍不存在: $SMOKE_TEXT_JSONL"
fi

# 这些变量会覆盖 model_*.env 中的默认值（profile 全部使用 ${VAR:-default}）
export DATASET_SPEC="$SMOKE_TEXT_JSONL"
export STREAMING=false          # 本地小文件，关流式以便 packing 与复现
export PACKING=true
export MAX_LENGTH=2048
export TRAIN_ITERS=2
# 冒烟只有 2 步，warmup 必须 < decay，否则 Megatron 的 OptimizerParamScheduler
# 会 assert 失败（03 里也有钳制兜底，这里显式给出以保持语义清晰）
export LR_WARMUP_ITERS=1
# 只有 2 步，日志频率必须 <= 2 才看得见 iteration 行（默认 5 会一行不打印）
export LOGGING_STEPS=1
export SAVE_STEPS=2
export DURATION_MINS=0          # 冒烟按步数收口，不受现场时长变量影响
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=8
export EVAL_STEPS=""
export NO_SAVE_OPTIM=true
export RUN_TAG="smoke_$PROFILE"
export OUT_DIR="$OUTPUT_ROOT/smoke_${PROFILE}_$(date '+%Y%m%d_%H%M%S')"

log "=== T2 冒烟 | profile=$PROFILE | data=$SMOKE_TEXT_JSONL ==="
bash "$PROJECT_ROOT/scripts/03_continue_pt.sh" "$PROFILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  log "DRY_RUN=1，跳过产物检查"
  exit 0
fi
CKPT_COUNT="$(find "$OUT_DIR" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | wc -l | tr -d ' ')"
[[ "$CKPT_COUNT" -gt 0 ]] || die "冒烟结束但没有 checkpoint-* 产出，检查日志 $LOG_ROOT/smoke_$PROFILE*"
log "=== T2 通过：产出 $CKPT_COUNT 个 checkpoint ==="
