# ============================================================================
# 60-phase-soak —— 长稳编排：dense / dense70 / moe 三种形态 + `--phase soak` 的总调度。
# 显存高占用那条在 62/63。
# ============================================================================

moe_soak_record() {                      # status tag ep tp pp disp mbs seq gbs iters jitter active
  local st=$1 tag=$2 ep=$3 tp=$4 pp=$5 disp=$6 mbs=$7 seq=$8 gbs=$9 it=${10} jit=${11} act=${12}
  local log="$LOGDIR/w${WORLD}_${tag}.log" ss tok mfu imb="" drift="" csv="$LOGDIR/expert-load-timeseries.csv"
  ss=$(step_seconds "$log"); tok=$(( gbs * seq / WORLD ))
  mfu=$(mfu_active "$act" "$gbs" "$seq" "$ss")
  if [ -f "$csv" ]; then
    read -r imb drift < <(awk -F, 'NR>1 && $4!="null" && $5!="null" && $5+0>0 {
        r=$4/$5; n++; if(n==1) fr=r; lr=r }
        END{if(n) printf "%.2f", fr; if(n>=2) printf " %.2f", (lr-fr)/fr*100}' "$csv")
  fi
  local note="${MOE_NOTE:-}"
  [ -n "$imb" ] || { imb=null; note="${note:+$note；}专家负载未采集到（查训练日志的 [expert load probe] 行）"; }
  [ -n "$drift" ] || drift=null
  printf '{"world":%d,"nodes":%d,"tag":"%s","model":"moe","ep":%d,"tp":%d,"pp":%d,"dispatcher":"%s","mbs":%d,"seq":%d,"gbs":%d,"accum":%d,"step_s":%s,"tokens_per_gpu":%d,"iters":%d,"jitter":%s,"mfu_active":%s,"imbalance_max_over_mean":%s,"imbalance_drift":%s,"attention":"%s","experimental":%s,"note":"%s","status":"%s","ts":"%s"}\n' \
    "$WORLD" "$NNODES" "$tag" "$ep" "$tp" "$pp" "$disp" "$mbs" "$seq" "$gbs" \
    "$(( gbs / (mbs * (WORLD / tp / pp)) ))" "${ss:-0}" "$tok" "$it" "${jit:-0}" "$mfu" \
    "$imb" "$drift" "${MOE_ATTN:-mla}" "$([ "${MOE_EXPERIMENTAL:-0}" = 1 ] && echo true || echo false)" \
    "$note" "$st" "$(date -Is)" >> "$LOGDIR/moe-soak.jsonl"
}

soak_dense() {                           # dense 长稳：与历史行为完全一致
  local MINS=$1 TAG="8B_soak_tp8_pp1_mbs4_s4096_gb8_bf16"
  SOAK_TARGET_MINS="$MINS" SOAK_LABEL=dense
  echo "########## Dense 长稳 $MINS 分钟（NNODES=$NNODES，要求 ≥ $SOAK_MIN_ITERS 步）##########"
  SOAK_PREFIX=""
  health_snap "$LOGDIR/health-before.json"
  TELEMETRY=1
  STEPS=100000
  SOAK_REQUIRE_ITERS="$SOAK_MIN_ITERS" \
    run_train "$TAG" LLAMA_8B 8 1 4 4096 8 --exit-duration-in-mins="$MINS"
  local rc=$?
  health_snap "$LOGDIR/health-after.json"
  health_delta_json
  echo "  遥测：$LOGDIR/dmon-w${WORLD}_${TAG}.csv（逐卡一行，报告按单卡序列，不做跨卡平均）"
  sync_logs
  return "$rc"
}

soak_dense70() {                         # 双机 70B 正常训练长稳：与 highmem 的显存边界口径分开
  if (( NNODES < 2 || WORLD < 16 )); then
    echo "!! dense70 需要双机 world≥16（当前 NNODES=$NNODES、world=$WORLD），不能用单机 8 卡硬跑 70B"
    return 2
  fi
  local MINS="$DENSE70_SOAK_MINUTES" MITERS="${DENSE70_SOAK_MIN_ITERS:-$SOAK_MIN_ITERS}"
  local TAG="70B_soak_tp8_pp2_mbs1_s4096_gb16_bf16"
  SOAK_TARGET_MINS="$MINS" SOAK_LABEL=dense70
  echo "########## Dense 70B 长稳 $MINS 分钟（TP8×PP2×DP1，mbs=1，seq=4096，GBS=16，要求 ≥ $MITERS 步）##########"
  echo "    · num_micro_batches=16，PP2 理论气泡约 6.25%；node-b（全局末级 rank）的 iteration 是步数判据"
  if [ "$FORCE" != 1 ] && [ -f "$LOGDIR/.ok_w${WORLD}_${TAG}" ]; then
    echo "→ 跳过 Dense 70B 长稳（已完成哨兵 .ok_w${WORLD}_${TAG}）"; return 0
  fi
  SOAK_PREFIX="dense70-"
  health_snap "$LOGDIR/dense70-health-before.json"
  TELEMETRY=1
  STEPS=100000
  SOAK_REQUIRE_ITERS="$MITERS" \
    run_train "$TAG" LLAMA_70B 8 2 1 4096 16 --exit-duration-in-mins="$MINS"
  local rc=$?
  health_snap "$LOGDIR/dense70-health-after.json"
  health_delta_json dense70-
  echo "  遥测：$LOGDIR/dmon-w${WORLD}_${TAG}.csv；硬件增量：dense70-health-delta.json（与 8B/MoE 独立）"
  sync_logs
  SOAK_PREFIX=""
  return "$rc"
}

soak_moe() {                             # MoE 长稳：配置 B，EP8（单机）/ EP16（双机），alltoall
  local MINS="$MOE_SOAK_MINUTES"
  [ -n "$MINS" ] || { (( NNODES > 1 )) && MINS=60 || MINS=30; }
  SOAK_TARGET_MINS="$MINS" SOAK_LABEL=moe
  local MITERS="${MOE_SOAK_MIN_ITERS:-$SOAK_MIN_ITERS}"
  local EP=$WORLD GBS="$MOE_GBS"
  [ -n "$GBS" ] || { (( NNODES > 1 )) && GBS=16 || GBS=8; }
  local TAG="MOE_soak_ep${EP}_tp1_pp1_mbs1_s4096_alltoall"
  local LOG="$LOGDIR/w${WORLD}_${TAG}.log"
  if [ "$FORCE" != 1 ] && [ -f "$LOGDIR/.ok_moe_soak" ]; then
    echo "→ 跳过 MoE 长稳（已完成哨兵 .ok_moe_soak）$TAG"; return 0
  fi
  ensure_args_check
  MOE_ACTIVE_PARAMS=$(moe_est "${MOE_B_EST[@]}" 1 1 "$EP" "$MOE_B_VOCAB" | awk '{print $2}')
  moe_resolve_attn; gate_rc=$?
  if [ "$gate_rc" -eq 2 ]; then echo "   !! $MOE_NOTE"; return 2; fi
  if [ "$gate_rc" -ne 0 ]; then
    echo "   !! SKIP：$MOE_NOTE（MoE 长稳未跑）"
    MOE_NOTE="$MOE_NOTE" moe_soak_record SKIP "$TAG" "$EP" 1 1 alltoall 1 4096 "$GBS" 0 0 "$MOE_ACTIVE_PARAMS"
    return 0
  fi
  if ! args_gate_moe; then
    MOE_NOTE="关键 MoE flag 缺失（见 $ARGS_CHECK）"
    echo "   !! SKIP：MoE 关键 flag 缺失，MoE 长稳未跑"
    moe_soak_record SKIP "$TAG" "$EP" 1 1 alltoall 1 4096 "$GBS" 0 0 "$MOE_ACTIVE_PARAMS"
    return 0
  fi
  if ! disp_supported alltoall; then
    echo "   !! SKIP：该核心的 --moe-token-dispatcher-type 不支持 alltoall"; return 0
  fi
  local -a MARGS=( "${MOE_B[@]}" )
  [ "$MOE_EXPERIMENTAL" = 1 ] && MARGS+=( --enable-experimental )
  mapfile -t MARGS < <(filter_args "${MARGS[@]}" --expert-model-parallel-size="$EP" \
                       --moe-token-dispatcher-type=alltoall)
  echo "########## MoE 长稳 $MINS 分钟（EP${EP}×TP1×PP1，alltoall，要求 ≥ $MITERS 步）##########"
  SENTINEL_KEY=moe_soak
  DMON_TAG="moe-soak-w${WORLD}_MOE_ep${EP}_alltoall"
  SOAK_PREFIX="moe-"
  health_snap "$LOGDIR/moe-health-before.json"
  TELEMETRY=1
  STEPS=100000
  moe_probe_enable || true
  expert_sampler_start "$LOG" "$LOGDIR/expert-load-timeseries.csv"
  SOAK_REQUIRE_ITERS="$MITERS" \
    run_train "$TAG" MARGS 1 1 1 4096 "$GBS" --exit-duration-in-mins="$MINS"
  local rc=$?
  expert_sampler_stop
  moe_probe_disable                     # 后续 highmem/dense 长稳不挂 MoE 探针
  health_snap "$LOGDIR/moe-health-after.json"
  health_delta_json moe-
  local njit jit st="OK" nonlast_no_iter=0
  dual_nonlast_no_iter "$LOG" "$rc" && nonlast_no_iter=1
  njit=$(grep -c "elapsed time per iteration" "$LOG" 2>/dev/null); njit=${njit:-0}
  jit=$(step_jitter "$LOG")
  if [ ! -f "$LOGDIR/.ok_moe_soak" ]; then
    if [ "$njit" -lt "$MITERS" ] && [ "$njit" -gt 0 ]; then st=PARTIAL; else st=FAILED; fi
  fi
  if [ "$nonlast_no_iter" -eq 1 ] && [ "$st" = OK ]; then
    MOE_NOTE="非末级节点不打印 iteration；步数/性能以 node-b 全局末级 rank 为准"
  fi
  moe_soak_record "$st" "$TAG" "$EP" 1 1 alltoall 1 4096 "$GBS" "$njit" "$jit" "$MOE_ACTIVE_PARAMS"
  echo "    · 记录 → moe-soak.jsonl / expert-load-timeseries.csv / dmon-${DMON_TAG}.csv"
  echo "  遥测：$LOGDIR/dmon-${DMON_TAG}.csv；硬件增量：moe-health-delta.json（与 dense 独立）"
  SENTINEL_KEY=""; DMON_TAG=""; SOAK_PREFIX=""
  sync_logs
  [ "$rc" -eq 0 ] && [ "$st" = OK ]
}

phase_soak() {          # --phase soak
  MINS="${SOAK_MINUTES:-120}"
  # SOAK_MODEL 支持逗号组合：dense / dense70 / moe / highmem；both=dense+moe。
  # all=全部适用项：双机 dense→dense70→moe→highmem；单机没有 dense70，保持原三条。
  ORDER=()
  add_soak() { case " ${ORDER[*]:-} " in *" $1 "*) ;; *) ORDER+=( "$1" ) ;; esac; }
  for tok in ${SOAK_MODEL//,/ }; do
    case "$tok" in
      all)
        add_soak dense
        (( NNODES > 1 && WORLD >= 16 )) && add_soak dense70
        add_soak moe; add_soak highmem ;;
      both) add_soak dense; add_soak moe ;;
      dense70)
        if (( NNODES < 2 || WORLD < 16 )); then
          echo "!! SOAK_MODEL=dense70 只适用于双机 world≥16（当前 NNODES=$NNODES、world=$WORLD）"; exit 2
        fi
        add_soak dense70 ;;
      dense|moe|highmem) add_soak "$tok" ;;
      *) echo "!! SOAK_MODEL=$SOAK_MODEL 不合法：只接受 dense / dense70 / moe / highmem 的逗号组合"
         echo "   （兼容旧值 both=dense+moe；all=全部适用项，dense70 仅双机）"; exit 2 ;;
    esac
  done
  [ "${#ORDER[@]}" -gt 0 ] || ORDER=( dense )
  # 参数存在性以容器 --help 为准：megatron_args.txt 只在跑过 03_deploy 的 run 里有，
  # 直接用 04 创建的 run 会误判成「容器没有该参数」（11-08 实测）。
  # 回退链：megatron_args.txt → 现场跑一次 --help（ensure_args_check，幂等）→ flag_in_help 判定。
  # 注意 $MA 可能为空：空串必须交给 flag_in_help 兜底，不能直接 grep。
  if [ -n "$MA" ] && grep -qE '^ *--exit-duration-in-mins([ =]|$)' "$MA" 2>/dev/null; then
    :
  else
    ensure_args_check
    if ! flag_in_help exit-duration-in-mins; then
      echo "!! 容器里的 Megatron 没有 --exit-duration-in-mins，长稳无法按时收尾。"
      echo "   要么换 core_v0.13.1（本方案已验证存在该参数），要么改用手工 kill + SOAK_MIN_ITERS 判定。"
      exit 2
    fi
    if [ -n "$MA" ]; then
      echo "    · $MA 里没有 --exit-duration-in-mins，已按容器 --help 原文确认存在（该参数表可能来自旧镜像）"
    else
      echo "    · 暂无 megatron_args.txt（本 run 未跑 03），已按容器 --help 确认 --exit-duration-in-mins 存在"
    fi
  fi
  # highmem 的档位来自 --phase mfu 的峰值档（见 62-highmem.sh），不再需要 limit 边界。
  # 这里只提前提示一句：没有 mfu.jsonl 时 highmem 会直接报错退出，别等跑到那一步才发现。
  case " ${ORDER[*]} " in
    *" highmem "*)
      if ! ls "$LC_RUN_DIR"/*/mfu.jsonl >/dev/null 2>&1; then
        echo "   ! [WARN] 本 run 还没有 mfu.jsonl：highmem 需要 MFU 峰值档才能定档，"
        echo "     请先跑  bash 04_bench.sh --phase mfu （或把它加进 SOAK_MODEL 之前）"
      fi
      ;;
  esac
  DENSE70_PLAN=""
  case " ${ORDER[*]} " in
    *" dense70 "*)
      DENSE70_PLAN="；dense70 ${DENSE70_SOAK_MINUTES} 分钟"
      if awk -v m="$DENSE70_SOAK_MINUTES" \
             -v i="${DENSE70_SOAK_MIN_ITERS:-$SOAK_MIN_ITERS}" \
             'BEGIN{exit !(m+0 < 10 && i+0 > 10)}'; then
        echo "   ! [WARN] dense70 仅 ${DENSE70_SOAK_MINUTES} 分钟却要求 ≥ ${DENSE70_SOAK_MIN_ITERS:-$SOAK_MIN_ITERS} 步："
        echo "     1 分钟冒烟通常达不到，末级节点会按 PARTIAL 中止；请设 DENSE70_SOAK_MIN_ITERS=1 或恢复正式时长。"
      fi ;;
  esac
  MOE_PLAN="${MOE_SOAK_MINUTES:-}"
  [ -n "$MOE_PLAN" ] || { (( NNODES > 1 )) && MOE_PLAN=60 || MOE_PLAN=30; }
  echo "长稳计划：${ORDER[*]}（dense ${MINS} 分钟${DENSE70_PLAN}；MoE ${MOE_PLAN} 分钟；highmem ${HIGHMEM_MINUTES:-60} 分钟）"
  if [ "${SOAK_PROGRESS:-1}" = 1 ]; then
    echo "    · 进度每 ${PROGRESS_SECS}s 刷新（终端 stderr），记录 → $LOGDIR/soak-progress.log"
    echo "    · 关闭用 SOAK_PROGRESS=0；改频率用 PROGRESS_SECS=<秒>"
  fi
  RC=0
  for m in "${ORDER[@]}"; do
    case "$m" in
      dense)   soak_dense "$MINS" || RC=1 ;;
      dense70) soak_dense70 || RC=1 ;;
      moe)     soak_moe || RC=1 ;;
      highmem) soak_highmem || RC=1 ;;
    esac
  done
  [ "$RC" -eq 0 ] || exit 1
}
