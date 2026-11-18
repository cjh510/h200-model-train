#!/usr/bin/env bash
# =============================================================================
# T3 继续预训练（CPT）主脚本：megatron pt + Mcore-Bridge + swift/chinese-c4
#
# 参数依据（详见 docs/AUDIT.md）：
#   · examples/megatron/pretrain.sh —— pt 基线命令、streaming+packing、TP、GBS、lr
#   · Megatron-SWIFT/Quick-start.md —— megatron pt、--model、--save_safetensors
#   · Megatron-SWIFT/Mcore-Bridge.md —— 直接读/存 safetensors，免 HF<->mcore 转换
#   · BestPractices/Qwen3_8-Flash-Next —— PLE_CPU_OFFLOAD、TP2/EP4/PP2、LoRA
#   · examples/models/qwen3_5/packing.sh —— 全参 + packing 的 MoE 参数组合
#
# 用法：
#   bash scripts/03_continue_pt.sh 27b
#   DATASET_SPEC='swift/chinese-c4' TRAIN_ITERS=1000 bash scripts/03_continue_pt.sh 27b
#   DRY_RUN=1 bash scripts/03_continue_pt.sh flash_next     # 只打印命令
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
install_err_trap
# 记录用户是否显式给了步数：profile 里是 ${TRAIN_ITERS:-100}，
# 只有「用户没给」时才允许被时长模式抬到上限。
_USER_TRAIN_ITERS="${TRAIN_ITERS:-}"
PROFILE="${1:-$MODEL_PROFILE}"
load_profile "$PROFILE"

DATASET_SPEC="${DATASET_SPEC:-$DATASET_FULL_SPEC}"
OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/$MODEL_KEY/$(date '+%Y%m%d_%H%M%S')_cpt}"
RUN_TAG="${RUN_TAG:-continue_pt_$MODEL_KEY}"
EVAL_STEPS="${EVAL_STEPS:-}"
RESUME_ARGS="${RESUME_ARGS:-}"
NO_SAVE_RNG="${NO_SAVE_RNG:-true}"

# --- 训练时长（墙钟） -------------------------------------------------------
# Megatron 的两个收口参数互不替代：--train_iters 是步数，--exit_duration_in_mins
# 是墙钟。只给时长而不给足步数，会先跑完步数提前退出，看起来像「时长没生效」。
DURATION_MINS="${DURATION_MINS:-0}"
if ! [[ "$DURATION_MINS" =~ ^[0-9]+$ ]]; then
  die "DURATION_MINS 必须是非负整数分钟（当前 '$DURATION_MINS'）"
fi
if (( DURATION_MINS > 0 )) && [[ -z "$_USER_TRAIN_ITERS" ]]; then
  TRAIN_ITERS="${DURATION_ITERS_CAP:-100000}"
  log "时长模式：TRAIN_ITERS 抬到上限 $TRAIN_ITERS（到点收口靠 --exit_duration_in_mins $DURATION_MINS）"
fi

# CLI 能力查询：0=help 里有，1=help 里没有，2=help 文件不存在（未知）
_help_has() {
  local f="$LOG_ROOT/pt_help_${MODEL_KEY}.txt"
  [[ -f "$f" ]] || return 2
  grep -qE -- "$1" "$f"
}   # 严格续训时应设为 false

require_cmd megatron
# 默认是 Mcore-Bridge 路径（--model 直读已有 safetensors）；
# 断点续训若改用 mcore checkpoint，可设 MCORE_MODEL=<ckpt> 切换到 --mcore_model。
MCORE_MODEL="${MCORE_MODEL:-}"
if [[ -z "$MCORE_MODEL" ]]; then
  [[ -d "$MODEL_PATH" ]] || die "权重目录不存在: $MODEL_PATH（先改 config/env.sh 或用 MODEL_ROOT 覆盖）"
else
  [[ -d "$MCORE_MODEL" ]] || die "mcore checkpoint 不存在: $MCORE_MODEL"
fi

# --- 并行合法性自检 ---------------------------------------------------------
WORLD=$((NPROC_PER_NODE * NNODES))
PAR=$((TP * PP * CP))
if (( WORLD % PAR != 0 )); then
  die "world_size=$WORLD 不能被 TP*PP*CP=$PAR 整除，请调整并行参数"
fi
DP=$((WORLD / PAR))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
  die "global_batch_size=$GLOBAL_BATCH_SIZE 必须能被 micro_batch_size*DP=$((MICRO_BATCH_SIZE * DP)) 整除"
fi
# MoE 特有约束：专家并行是从 DP 组里划分的，DP 必须是 EP 的整数倍。
# 例如 TP2/PP2/EP4 在 8 卡上 DP 只剩 2，4 不能整除 2，Megatron 会直接报
# data_parallel_size must be divisible by expert_model_parallel_size。
if [[ "${IS_MOE:-0}" == "1" ]] && (( EP > 1 )); then
  if (( DP % EP != 0 )); then
    die "DP=$DP 不能被 EP=$EP 整除（MoE 要求 data_parallel_size % expert_model_parallel_size == 0）。请调小 EP（<= $DP）或增大 TP/PP 之外的 DP 空间"
  fi
fi
log "并行: world=$WORLD TP=$TP PP=$PP CP=$CP EP=$EP -> DP=$DP, 梯度累积=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))"

# --- 全参可行性预检（显存账）-------------------------------------------------
# 训练状态 ≈ 可训练参数 × 16 字节：
#   bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + Adam m 4 + Adam v 4 = 16 B/参数
# 判据必须按**每卡**算，不能拿总量比总显存：权重/梯度按 TP×PP 切，优化器状态按 DP
# 分片，两者的乘积才等于 world_size。只看总量会在 16 卡时误判通过（1184 < 2246），
# 实际每卡仍需 296 GB。这是不含 activation 的下限。
if [[ "$TUNER_TYPE" == "full" ]] && (( ${MODEL_TOTAL_PARAMS_B:-0} > 0 )); then
  _train_b=$(( MODEL_TOTAL_PARAMS_B - ${MODEL_FROZEN_PARAMS_B:-0} ))
  _wg_total=$(( _train_b * 4 ))
  _opt_total=$(( _train_b * 12 ))
  _state_total=$(( _train_b * 16 ))
  _gpu_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ "$_gpu_mb" =~ ^[0-9]+$ ]]; then
    _gpu_gb=$(( _gpu_mb / 1024 ))
    _pp_factor=$(( TP * PP ))
    if (( _pp_factor < 1 )); then _pp_factor=1; fi
    _dp_div="${DP:-1}"
    if (( _dp_div < 1 )); then _dp_div=1; fi
    _wg_per_gpu=$(( _wg_total / _pp_factor ))
    _opt_per_gpu=$(( _opt_total / _dp_div ))
    _need_per_gpu=$(( _wg_per_gpu + _opt_per_gpu ))

    _nodes="${NNODES:-1}"
    if (( _nodes < 1 )); then _nodes=1; fi
    _cpu_per_node=$(( _opt_total / _nodes ))

    log "全参显存账: 可训练 ${_train_b}B → 状态 ${_state_total} GB（权重+梯度 ${_wg_total} + 优化器 ${_opt_total}）"
    log "            单卡需求: 权重+梯度 ${_wg_per_gpu} GB（÷TP×PP=${_pp_factor}）+ 优化器 ${_opt_per_gpu} GB（÷DP=${_dp_div}）= ${_need_per_gpu} GB，可用 ${_gpu_gb} GB"

    case "${OPTIMIZER_CPU_OFFLOAD:-0}" in
      1|true|True|TRUE|yes|YES)
        log "            CPU 卸载已开：每节点需约 ${_cpu_per_node} GB 主机内存（${_opt_total} GB ÷ ${_nodes} 节点）"
        if (( _wg_per_gpu * 100 / _gpu_gb > 85 )); then
          warn "权重+梯度 ${_wg_per_gpu} GB 已占单卡 ${_gpu_gb} GB 的 85% 以上，activation 未必够；考虑加大 TP/PP"
        fi
        ;;
      *)
        if (( _need_per_gpu > _gpu_gb )); then
          warn "全参放不下：单卡需 ${_need_per_gpu} GB（权重+梯度 ${_wg_per_gpu} + 优化器 ${_opt_per_gpu}），只有 ${_gpu_gb} GB"
          die "三选一：① OPTIMIZER_CPU_OFFLOAD=1（每节点需约 ${_cpu_per_node} GB 内存）；② 加大 PP 或增加节点；③ TUNER_TYPE=lora"
        elif (( _wg_per_gpu * 100 / _gpu_gb > 85 )); then
          warn "权重+梯度已占单卡显存 85% 以上，activation 可能不够"
        fi
        ;;
    esac
  fi
fi

# --- 学习率 schedule 合法性 -------------------------------------------------
# Megatron 的 OptimizerParamScheduler.__init__ 断言
#     lr_warmup_steps < lr_decay_steps
# 而 lr_decay_steps 取的就是 --train_iters。冒烟这类「总步数很小」的场景如果沿用
# 档案里的 warmup=30，会直接 AssertionError 让 8 个 rank 全部退出（报错信息只有
# 一行 assert，很容易误判成模型或显存问题）。
if ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]]; then
  die "TRAIN_ITERS='$TRAIN_ITERS' 不是非负整数"
fi
if (( LR_WARMUP_ITERS >= TRAIN_ITERS )); then
  _clamped=$(( TRAIN_ITERS - 1 ))
  if (( _clamped < 0 )); then _clamped=0; fi
  warn "LR_WARMUP_ITERS=$LR_WARMUP_ITERS >= TRAIN_ITERS=$TRAIN_ITERS（Megatron 要求 warmup < decay），已钳制为 $_clamped"
  LR_WARMUP_ITERS=$_clamped
fi
if (( LR_WARMUP_ITERS > 0 )); then
  log "lr schedule: warmup=$LR_WARMUP_ITERS decay=$TRAIN_ITERS lr=$LR min_lr=$MIN_LR"
fi

# --- 组装命令 ---------------------------------------------------------------
# 正常走 CPT（pt）。若现场发现 VL 架构不兼容纯文本 pt，
# 可设 MEGATRON_SUBCMD=sft + DATASET_SPEC=<messages jsonl> 走 sft 回退路径。
MEGATRON_SUBCMD="${MEGATRON_SUBCMD:-pt}"
ARGS=(
  megatron "$MEGATRON_SUBCMD"
)

if [[ -n "$MCORE_MODEL" ]]; then
  ARGS+=(--mcore_model "$MCORE_MODEL")
else
  ARGS+=(--model "$MODEL_PATH")
fi

ARGS+=(
  --save_safetensors true
  --dataset "$DATASET_SPEC"
  --streaming "$STREAMING"
  --packing "$PACKING"
  --tensor_model_parallel_size "$TP"
  --pipeline_model_parallel_size "$PP"
  --context_parallel_size "$CP"
  --sequence_parallel "$SP"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --global_batch_size "$GLOBAL_BATCH_SIZE"
  --train_iters "$TRAIN_ITERS"
  --finetune "$FINETUNE"
  --cross_entropy_loss_fusion true
  --lr "$LR"
  --lr_warmup_iters "$LR_WARMUP_ITERS"
  --min_lr "$MIN_LR"
  --max_length "$MAX_LENGTH"
  --save_steps "$SAVE_STEPS"
  --output_dir "$OUT_DIR"
  --recompute_granularity "$RECOMPUTE_GRANULARITY"
  --recompute_method "$RECOMPUTE_METHOD"
  --recompute_num_layers "$RECOMPUTE_NUM_LAYERS"
  --attention_backend "$ATTENTION_BACKEND"
  --dataloader_num_workers 4
  --dataset_num_proc 8
  --no_save_optim "$NO_SAVE_OPTIM"
  --no_save_rng "$NO_SAVE_RNG"
)

# MoE 专属并行参数（dense 模型不传，避免无意义的参数组合）
if [[ "$IS_MOE" == "1" ]]; then
  ARGS+=(--expert_model_parallel_size "$EP" --expert_tensor_parallel_size "$ETP")
fi

# 优化器 CPU 卸载：把 fp32 主权重与 Adam m/v 放到主机内存，靠 DRAM 换显存。
# 125B 量级的全参必需，代价是每步多一轮 CPU<->GPU 搬运，吞吐明显下降。
# use_precision_aware_optimizer 是它的配套开关（Megatron 要求成对出现）。
case "${OPTIMIZER_CPU_OFFLOAD:-0}" in
  1|true|True|TRUE|yes|YES)
    ARGS+=(--optimizer_cpu_offload true --use_precision_aware_optimizer true)
    ;;
esac

# LoRA（Flash-Next 单机默认路径）
if [[ "$TUNER_TYPE" == "lora" ]]; then
  read -r -a lora_targets <<< "$LORA_TARGET"
  ARGS+=(--tuner_type lora --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA"
         --target_modules "${lora_targets[@]}")
else
  ARGS+=(--tuner_type full)
fi

# 模型专属 / 用户追加参数
split_args MOE_ARGS moe_arr
split_args EXTRA_ARGS extra_arr
split_args RESUME_ARGS resume_arr
ARGS+=("${moe_arr[@]}" "${extra_arr[@]}" "${resume_arr[@]}")

if [[ -n "$EVAL_STEPS" ]]; then
  ARGS+=(--eval_steps "$EVAL_STEPS")
fi

# --- 日志频率 ---------------------------------------------------------------
# Megatron 的 `iteration N/M | ... | elapsed time per iteration (ms)` 行按
# logging_steps 输出。05_verify 的步时 / tokens/s 汇总完全依赖它——**设得比总步数
# 还大就会一行都不打印**（冒烟 2 步 + 默认 5 正是这种情况，日志里因此没有任何指标）。
#
# 这里自带默认值，不依赖 config/env.sh 是否已同步到含 LOGGING_STEPS 的新版：
# 脚本在 set -u 下引用未定义变量会直接以「未绑定的变量」退出。
LOGGING_STEPS="${LOGGING_STEPS:-1}"
if ! [[ "$LOGGING_STEPS" =~ ^[0-9]+$ ]] || (( LOGGING_STEPS < 1 )); then
  warn "LOGGING_STEPS='$LOGGING_STEPS' 无效，按 1 处理"
  LOGGING_STEPS=1
fi
_rc=0; _help_has 'logging.steps' || _rc=$?
if (( _rc == 1 )); then
  warn "当前版本没有 --logging_steps，无法调整日志频率"
else
  ARGS+=(--logging_steps "$LOGGING_STEPS")
  if (( LOGGING_STEPS > TRAIN_ITERS )); then
    warn "LOGGING_STEPS=$LOGGING_STEPS > TRAIN_ITERS=$TRAIN_ITERS：本次训练不会打印任何 iteration 行（也就拿不到步时/吞吐）"
  fi
fi

# --- 吞吐日志（默认开） -----------------------------------------------------
# Megatron 只在 log_throughput 打开时打印 "throughput per GPU (TFLOP/s/GPU)"。
# 该值由 Megatron 自己的 FLOPs 估算得出，对 GDN 线性注意力 / MoE / ViT 有系统性偏差，
# 只适合同模型不同并行配置之间横向比较；对外报数请用 05_verify.sh 的 tokens/s/卡。
#
# 注意：swift 的 CLI 只暴露自身参数，**不含 Megatron 原生的 --log_throughput**，
# 所以多数镜像里这里会走「跳过」分支——这是正常的，不是配置错误。此时吞吐由
# 05_verify.sh 用步时反算（GBS × seq / 步时），对外报数用那个更准。
case "${LOG_THROUGHPUT:-true}" in
  0|false|False|FALSE|no|NO) _log_tp=0 ;;
  *)                         _log_tp=1 ;;
esac
if (( _log_tp == 1 )); then
  _rc=0; _help_has 'log.throughput' || _rc=$?
  if (( _rc == 1 )); then
    log "该版本 CLI 不暴露 --log_throughput，跳过；吞吐改由 05_verify.sh 用步时换算 tok/s"
  else
    ARGS+=(--log_throughput true)
    if (( _rc == 2 )); then
      warn "未找到 pt_help，按 >=4.5.3 假设支持 --log_throughput；若启动报未知参数，先跑 00_preflight.sh 生成 help"
    fi
  fi
fi

# --- 墙钟上限（DURATION_MINS>0 时） -----------------------------------------
if (( DURATION_MINS > 0 )); then
  _rc=0; _help_has 'exit.duration' || _rc=$?
  if (( _rc == 1 )); then
    warn "当前版本没有 exit_duration_in_mins，DURATION_MINS 不生效，只能用 TRAIN_ITERS 收口"
  else
    ARGS+=(--exit_duration_in_mins "$DURATION_MINS")
  fi
fi

# --- 环境变量 ---------------------------------------------------------------
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF MODELSCOPE_CACHE
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT
if [[ "$PLE_CPU_OFFLOAD" == "1" ]]; then
  export PLE_CPU_OFFLOAD=1
  warn "PLE_CPU_OFFLOAD=1：N-gram 表被卸载到 CPU 并被冻结，全参训练时该部分不会更新"
fi

log "模型: $MODEL_ID ($ARCHITECTURE, $MODEL_TYPE)"
log "数据: $DATASET_SPEC | streaming=$STREAMING packing=$PACKING max_length=$MAX_LENGTH"
log "训练: iters=$TRAIN_ITERS lr=$LR tuner=$TUNER_TYPE finetune=$FINETUNE"
log "时长: ${DURATION_MINS} 分钟（0=按步数） | log_throughput=$_log_tp | dmon=${DMON}（宿主机由 12 启停）"
log "输出: $OUT_DIR"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "$OUT_DIR"
fi
run_logged "$RUN_TAG" "${ARGS[@]}"
log "=== T3 命令执行结束（产物验证请跑 05_verify.sh）==="
