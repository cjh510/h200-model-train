# ============================================================================
# 40-run —— 执行核心：run_train / fault_train（唯一真正起训练容器的两个函数），
# 加上进度显示、专家负载采样、MoE 探针开关。
# ============================================================================

# ---- MoE 专家负载探针（P1 11-10）--------------------------------------------
# core_v0.13.1 的 pretrain_gpt.py 根本不向 stdout 打印逐专家 token 计数：
# --moe-per-layer-logging 只打 aux/z loss，tokens_per_expert 仅进 tensorboard。
# 所以光改正则永远修不好 null —— 必须让日志里真的出现专家计数。
# 这里把 probes/moe_load_probe.py 挂成 sitecustomize.py（LOGDIR 已挂 /workspace/logs），
# 由容器内 Python 自动导入，训练时按间隔打印：
#   [expert load] step=200 max/mean=12.000/9.000 counts=[12,9,8,...]
moe_probe_enable() {
  local src="$HERE_PROBES/moe_load_probe.py"
  if [ ! -f "$src" ]; then
    echo "   ! 专家负载探针缺失：$src（同步整个 probes/ 目录），本阶段指标仍会是 null" >&2
    return 1
  fi
  cp "$src" "$LOGDIR/sitecustomize.py" || return 1
  EXTRA_PYTHONPATH="/workspace/logs"
  MOE_LOAD_PROBE=1
  MOE_LOAD_LOG_EVERY="${MOE_LOAD_LOG_EVERY:-200}"
  echo "→ 专家负载探针已启用（每 ${MOE_LOAD_LOG_EVERY} 次 routing 采样一次，见 [expert load] 行）"
}

moe_probe_disable() {
  EXTRA_PYTHONPATH=""; MOE_LOAD_PROBE=""; MOE_LOAD_LOG_EVERY=""
}

# 专家负载：优先探针格式 "max/mean=x/y" 或 "counts=[...]"，兜底旧格式
# "expert load ... 1.83 / 1.07"。返回 "max mean"（绝对 token 计数，不是比值）。
expert_load_pair() {
  local log=$1 line pair counts p
  line=$(grep -i '\[expert load\]' "$log" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    pair=$(printf '%s\n' "$line" | grep -oE 'max/mean=[0-9.]+/[0-9.]+' | tail -1 | cut -d= -f2)
    if [ -n "$pair" ]; then
      p=$(awk -v p="$pair" 'BEGIN{split(p,a,"/"); if(a[2]+0>0) printf "%.1f %.1f", a[1], a[2]}')
      if [ -n "$p" ]; then printf '%s' "$p"; return; fi
    fi
    counts=$(printf '%s\n' "$line" | grep -oE 'counts=\[[0-9, .-]+\]' | tail -1 \
             | sed -E 's/.*\[([0-9, .-]+)\].*/\1/' | tr ',' ' ')
    if [ -n "$counts" ]; then
      awk -v c="$counts" 'BEGIN{n=split(c,a," "); s=0; mx=0;
        for(i=1;i<=n;i++){s+=a[i]; if(a[i]>mx)mx=a[i]} if(n>0) printf "%.1f %.1f", mx, s/n}'
      return
    fi
  fi
  grep -ioE 'expert load[^0-9]*[0-9.]+[[:space:]]*[/,][[:space:]]*[0-9.]+' "$log" 2>/dev/null | tail -1 \
    | grep -oE '[0-9.]+[[:space:]]*[/,][[:space:]]*[0-9.]+' \
    | awk -F'[/,]' '{gsub(/ /,"",$1); gsub(/ /,"",$2); printf "%.1f %.1f", $1, $2}'
}

expert_load_max_over_mean() {
  local p; p=$(expert_load_pair "$1")
  [ -n "$p" ] || { echo null; return; }
  awk -v p="$p" 'BEGIN{split(p,a," "); if(a[2]>0) printf "%.2f", a[1]/a[2]; else print "null"}'
}

# MFU(激活) = 6·P_active·(GBS×seq) ÷ world ÷ step_s ÷ 峰值。
# 没有 P_active 或 step_s<=0 时输出整数 0（不是 0.0）：JSONL 里 mfu_active 的取值形态要保持一致。
mfu_active() {                           # mfu_active <active_params> <gbs> <seq> <step_s>
  local p=${1:-} s=${4:-0}
  [ -n "$p" ] || { echo 0; return 0; }
  awk -v p="$p" -v t="$(( $2 * $3 ))" -v s="$s" -v pk="${PEAK_TFLOPS}e12" -v w="$WORLD" \
    'BEGIN{if (s>0) printf "%.1f", 6*p*t/w/s/pk*100; else print 0}'
}

# Megatron core_v0.13.1 的 training_log 用 print_rank_last 打 iteration 行：
# 只有 global rank=world-1 打印，与 PP 是否大于 1 无关。双机 8B DP2 / MoE EP16
# 的非末级节点同样可能 0 行，但确实参与了训练；需由末级节点判最少步数。
dual_nonlast_no_iter() {               # dual_nonlast_no_iter <log> <rc>
  local log=$1 rc=$2
  (( NNODES > 1 && NODE_RANK < NNODES - 1 )) || return 1
  [ "$rc" -eq 0 ] || return 1
  ! grep -q "elapsed time per iteration" "$log" 2>/dev/null
}

# --------------------------------------------------------------- 一次训练
# run_train <tag> <model数组名> <tp> <pp> <mbs> <seq> <gbs> [额外 flag...]
run_train() {
  local tag=$1 modelname=$2 tp=$3 pp=$4 mbs=$5 seq=$6 gbs=$7; shift 7
  local -n MODEL=$modelname
  tag="w${WORLD}_${tag}"
  local log="$LOGDIR/$tag.log"
  # 哨兵键可单独指定：MoE 长稳用 moe_soak，绝不与 dense 长稳混用
  local sk="${SENTINEL_KEY:-$tag}"
  # PLAN_ONLY=1：只打印这一档的网格，不启动容器（几秒跑完整个阶段）。
  # 放在哨兵检查之前，这样已经跑过的档也会照样列出来。
  if [ "${PLAN_ONLY:-0}" = 1 ]; then
    local _dp=$(( WORLD / tp / pp ))
    local _acc=0; (( mbs > 0 && _dp > 0 )) && _acc=$(( gbs / (mbs * _dp) ))
    printf 'PLAN %-46s %-9s TP%-2s PP%-2s DP%-2s mbs%-2s seq%-7s GBS%-4s accum=%s\n' \
      "$tag" "$modelname" "$tp" "$pp" "$_dp" "$mbs" "$seq" "$gbs" "$_acc"
    return 0
  fi

  local port; port=$(det_port "$tag")

  # 跳过判据用哨兵文件，不再「日志里有 iteration 就算跑过」：
  # 半截日志（跑 3 步被杀）也有 iteration，原逻辑会永久跳过这档，长稳尤其致命。
  if [ "$FORCE" != "1" ] && [ -f "$LOGDIR/.ok_$sk" ]; then
    echo "→ 跳过（已完成哨兵 .ok_$sk）$tag  端口 $port"; return 0
  fi

  local dp=$(( WORLD / tp / pp ))
  if (( dp * tp * pp != WORLD )); then
    echo "   !! TP=$tp PP=$pp 无法整除 world=$WORLD，跳过（SKIP，非 FAILED）"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$(lc_env_json_fields)" 0 SKIP "tp×pp 不整除 world"
    return 1
  fi
  if (( gbs % (mbs * dp) != 0 )); then
    echo "   !! GBS=$gbs 不能被 mbs×dp=$(( mbs * dp )) 整除，跳过（SKIP，非 FAILED）"
    echo "      Megatron 的 num_micro_batches = GBS ÷ (mbs × dp)，必须整除"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$(lc_env_json_fields)" 0 SKIP "GBS 不整除 mbs×dp"
    return 1
  fi
  local nmib=$(( gbs / (mbs * dp) ))
  echo "→ $tag   mbs=$mbs dp=$dp → num_micro_batches=$nmib   port=$port"
  if (( pp > 1 )) && (( nmib < 2 * pp )); then
    echo "   ! micro-batch 太少（$nmib），PP=$pp 流水线气泡约 $(awk -v p=$pp -v m=$nmib 'BEGIN{printf "%.0f%%", (p-1)/m*100}')；加大 GBS 再比"
  fi

  local par=( --tensor-model-parallel-size="$tp" --pipeline-model-parallel-size="$pp" )
  # Megatron 断言 lr_warmup_steps < lr_decay_steps：短跑档（STEPS=5）写死 warmup=5 会触发
  # AssertionError（11-08 单机实测），按实际步数收紧
  local warmup_iters=5
  (( warmup_iters >= STEPS )) && warmup_iters=$(( STEPS > 1 ? STEPS - 1 : 0 ))
  # 只按「这一档是否真的有收益」开开关：
  #   SP 需要 TP>1；DO 与两个 overlap 需要 DP>1（DP=1 时反而多一轮 bucket 处理）；
  #   recompute 只在 seq>=64K 打开，避免基线档因重算掩盖真实算力。
  local candidates=() f
  for f in "${OPTIONAL_FLAGS[@]}"; do
    case "$f" in
      --sequence-parallel)                 (( tp > 1 )) || continue ;;
      --use-distributed-optimizer|--overlap-grad-reduce|--overlap-param-gather)
                                           (( dp > 1 )) || continue ;;
      --recompute-granularity=selective)   (( seq >= 65536 )) || continue
                                           # highmem 档禁止 recompute：本档就是要吃满激活内存，
                                           # 开了重算等于把激活压回去，峰值占用会被系统性低估
                                           [ "${NO_RECOMPUTE:-0}" = 1 ] && continue ;;
    esac
    candidates+=( "$f" )
  done
  # 双机禁用 Gloo 进程组：Megatron 给 dp/cp 建的 Gloo 组走纯 TCP，靠主机名互相寻址，
  # 而容器主机名是随机 ID 且容器内无 DNS/对端 hosts（getaddrinfo err=-3）→ 退化成
  # 127.0.0.1 → connectFullMesh 拒连。NCCL 组用 IP 不受影响；单机保持已验证路径不变。
  (( NNODES > 1 )) && candidates+=( --disable-gloo-process-groups )
  mapfile -t opts < <(prune_optional "${candidates[@]}" "$@")

  nccl_env_for "$tag"

  # progress_start 会立即读一次 $log；先清空，避免读到同一 RUN_ID 上一轮的旧 iteration 行。
  # lc_run 随后的 > "$log" 会再次截断，不影响训练日志。
  : > "$log"
  dmon_start "$LOGDIR/dmon-${DMON_TAG:-$tag}.csv"
  progress_start "$log"

  lc_run -v "$ROOT":/workspace -v "$LOGDIR":/workspace/logs -w "$MEGATRON" \
    -e PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$MEGATRON" -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    ${MOE_LOAD_PROBE:+-e MOE_LOAD_PROBE=$MOE_LOAD_PROBE} \
    ${MOE_LOAD_LOG_EVERY:+-e MOE_LOAD_LOG_EVERY=$MOE_LOAD_LOG_EVERY} \
    "${NCCL_ENV[@]}" \
    "$IMAGE" torchrun --nproc_per_node="$NPROC" --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
      --master_port="$port" \
    pretrain_gpt.py \
      "${MODEL[@]}" "${par[@]}" "${opts[@]}" \
      --seq-length="$seq" --max-position-embeddings="$seq" \
      --micro-batch-size="$mbs" --global-batch-size="$gbs" \
      --bf16 --init-method-std=0.02 --clip-grad=1.0 --weight-decay=0.1 \
      --lr=1e-4 --min-lr=1e-5 --lr-decay-style=cosine --lr-warmup-iters="$warmup_iters" \
      --train-iters="$STEPS" --log-interval=1 --eval-iters=0 --seed="${SEED:-1234}" \
      > "$log" 2>&1
  local rc=$?
  progress_stop
  dmon_stop

  local njit; njit=$(grep -c "elapsed time per iteration" "$log" 2>/dev/null || true)
  njit=${njit:-0}
  # 双机所有并行形态都可能只有全局末级 rank 打印 iteration；非末级节点 0 行
  # 且 rc=0 时，步数门槛交给末级节点。末级节点自己 0 行仍必须 PARTIAL。
  local nonlast_no_log=0
  dual_nonlast_no_iter "$log" "$rc" && nonlast_no_log=1
  local fingerprint; fingerprint=$(lc_env_json_fields)
  if grep -qiE "CUDA out of memory|OutOfMemoryError" "$log"; then
    if [ "${OOM_AS_BOUNDARY:-0}" = 1 ] && [ "${ALLOW_OOM:-0}" = 1 ]; then
      local oom_line
      oom_line=$(grep -m1 -iE 'CUDA out of memory|OutOfMemoryError' "$log" | tr -d '"\\' | tr '\n' ' ' | cut -c1-180)
      echo "   ○ OOM_BOUNDARY  $tag（预期边界，不 dual_stop、不影响退出码）"
      echo "      $oom_line"
      : > "$LOGDIR/.oom_$tag"
      return 0
    fi
    echo "   ✗ OOM  $tag（FAILED，不是 SKIP）"
    grep -m1 -iE 'CUDA out of memory|OutOfMemoryError' "$log" | cut -c1-160 | sed 's/^/       /'
    local N=8 shard=1; [ "$modelname" = LLAMA_70B ] && N=70; (( dp > 1 )) && shard=$dp
    echo "      显存估算：静态（权重+梯度+优化器 16B/参数）约 $(awk -v n="$N" -v s=$(( tp * pp * shard )) 'BEGIN{printf "%.1f", 16*n*1e9/(s*1073741824)}') GiB/卡；seq=$seq mbs=$mbs 的激活另算，先降 mbs / 缩短 seq / 开 recompute 再试"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" OOM "CUDA out of memory；显存估算见终端"
    : > "$LOGDIR/.fail_$sk"
    dual_stop "$tag" OOM
  elif [ -n "${SOAK_REQUIRE_ITERS:-}" ] && [ "${njit:-0}" -lt "$SOAK_REQUIRE_ITERS" ] \
       && [ "$nonlast_no_log" -eq 0 ]; then
    echo "   ✗ 长稳只跑了 ${njit:-0} 步（要求 ≥ $SOAK_REQUIRE_ITERS），不算完成（PARTIAL，不是 SKIP）"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" PARTIAL "步数不足 $SOAK_REQUIRE_ITERS"
    : > "$LOGDIR/.fail_$sk"
    dual_stop "$tag" PARTIAL
  elif grep -q "elapsed time per iteration" "$log" || \
       { (( NNODES > 1 )) && [ "$rc" -eq 0 ]; } || \
       { (( NNODES > 1 )) && peer_ok "$tag"; }; then
    local why=""
    (( NNODES > 1 )) && [ "${njit:-0}" -eq 0 ] && \
      why="（全局末级 rank 在 NODE_RANK=$(( NNODES - 1 )) 打印 iteration；本机 rc=0 → 双机 PASS，性能看 node-b）"
    echo "   ✓ $tag   $(grep -oE 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" | tail -1)  ${njit} 步 ${why}"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" OK
    : > "$LOGDIR/.ok_$sk"
  else
    echo "   ✗ FAILED（rc=$rc，本机未产出 iteration）$tag"
    echo "      可疑行（已排除 Copyright / Ronan Collobert 等日志首部噪声）："
    grep -nE 'unrecognized arguments|invalid choice|ValueError|AssertionError|RuntimeError|ImportError|ModuleNotFoundError|KeyError|CUDA error|out of memory|NCCL error|ncclInvalidUsage|Segmentation fault|Killed|terminate called|assert ' "$log" 2>/dev/null \
      | grep -viE 'Copyright|Ronan Collobert' | head -8 | sed 's/^/      /'
    awk '/Traceback \(most recent call last\)/{f=1} f{print; if(++n>30) exit}' "$log" | sed 's/^/      /'
    echo "      末 20 行："
    tail -20 "$log" | sed 's/^/      /'
    echo "      完整日志：$log"
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" FAILED "见日志末尾 20 行"
    : > "$LOGDIR/.fail_$sk"
    dual_stop "$tag" FAILED
    return 1
  fi
}

# 故障注入专用：最小 8 卡训练，可命名容器（便于 kill/stop 单个 rank）。
fault_train() {                          # fault_train <log> <model> [容器名] [额外参数...]
  local log=$1 modelname=$2; shift 2
  local name="${1:-}"; [ -n "$name" ] && shift
  local -n FM=$modelname
  local nm=(); [ -n "$name" ] && nm=( --name "$name" )
  # P0（11-10）：这里漏了 CUDA_DEVICE_MAX_CONNECTIONS=1（run_train 一直有）。
  # 本函数硬编码 TP=8，Megatron 断言直接让 8 个 rank 在启动阶段 AssertionError 崩溃，
  # 三个 fault 用例全没跑起来；而判据只看 rc≠0，把 2 个崩溃记成了 PASS。
  lc_run "${nm[@]}" -v "$ROOT":/workspace -v "$LOGDIR":/workspace/logs -w "$MEGATRON" \
    -e PYTHONPATH="$MEGATRON" -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e NCCL_DEBUG="$NCCL_DBG" \
    ${NCCL_HCA:+-e NCCL_IB_HCA=$NCCL_HCA} ${NCCL_IF:+-e NCCL_SOCKET_IFNAME=$NCCL_IF} \
    "$IMAGE" torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port="$(det_port "$log")" \
      pretrain_gpt.py "${FM[@]}" --tensor-model-parallel-size=8 --pipeline-model-parallel-size=1 \
        --seq-length=4096 --max-position-embeddings=4096 --micro-batch-size=1 --global-batch-size=8 \
        --bf16 --train-iters=30 --log-interval=1 --eval-iters=0 --seed=1234 "$@" > "$log" 2>&1
}

# ------------------------------------------------------------ 长稳（dense / MoE / 显存高占用）
# 专家负载采样：每 EXPERT_SAMPLE_SECS 秒把当前日志的 step、逐专家计数（若有）、max/mean 落 CSV
expert_sample() {                        # expert_sample <log> <csv>
  local log=$1 csv=$2 ts step pair mx mn toks note=""
  ts=$(date -Is)
  step=$(grep -c "elapsed time per iteration" "$log" 2>/dev/null); step=${step:-0}
  pair=$(expert_load_pair "$log")
  if [ -n "$pair" ]; then mx=${pair% *}; mn=${pair#* }
  else
    mx=null; mn=null
    note="日志无专家计数；确认 MOE_LOAD_PROBE=1 且日志里没有 [expert load probe] install failed"
  fi
  # P1 11-10：探针格式是 "[expert load] ... counts=[12,9,...]"，旧正则匹配不到
  toks=$(grep -oE 'counts=\[[0-9, .-]+\]' "$log" 2>/dev/null | tail -1 \
         | sed -E 's/.*\[([0-9, .-]+)\].*/[\1]/' | tr -d ' ')
  if [ -z "$toks" ]; then
    toks=$(grep -oE 'expert load[^[]*\[[0-9, .-]+\]' "$log" 2>/dev/null | tail -1 \
           | sed -E 's/.*\[([0-9, .-]+)\].*/[\1]/' | tr -d ' ')
  fi
  # 逐专家列表里的逗号会把 CSV 列切碎 → 换成竖线（1|2|3），仍是「每个专家的 token 数」
  toks=${toks//,/|}
  if [ -z "$toks" ]; then
    toks=null
    [ -n "$note" ] || note="日志无逐专家计数（探针只打印了 max/mean）"
  fi
  note=${note//,/;}                      # 逗号会破坏 CSV
  [ -f "$csv" ] || echo "ts,step,expert_tokens,max,mean,note" > "$csv"
  printf '%s,%s,%s,%s,%s,%s\n' "$ts" "$step" "$toks" "$mx" "$mn" "$note" >> "$csv"
}

# ---------------------------------------------------------------- 长稳进度
# 训练是前台阻塞的 lc_run，进度由独立后台循环只读 $log 打印；输出走 stderr，
# 避免污染 filter_args / mapfile 这些「stdout 即参数」的命令替换路径。
# 只在 SOAK_REQUIRE_ITERS 存在（长稳档）时启用，15 步短档完全不受影响。
_fmt_progress_secs() {                   # <秒> → H:MM:SS / MM:SS
  awk -v s="${1:-0}" 'BEGIN{ if(s<0)s=0; h=int(s/3600); m=int((s%3600)/60); x=s%60;
    if(h>0) printf "%d:%02d:%02d", h,m,x; else printf "%02d:%02d", m,x }'
}

progress_tick() {                        # <log> <目标秒> <最少步数> <标签>
  local log=$1 total=${2:-0} minit=${3:-0} label=${4:-soak}
  local now elapsed pct eta tstr msg njit ms tf ts
  now=$(date +%s); elapsed=$(( now - ${PROGRESS_T0:-$now} )); [ "$elapsed" -lt 0 ] && elapsed=0
  njit=$(grep -c "elapsed time per iteration" "$log" 2>/dev/null || true); njit=${njit:-0}
  ms=$(grep -oE 'elapsed time per iteration \(ms\): [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
  tf=$(grep -oE 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
  if [ "${total:-0}" -gt 0 ]; then
    pct=$(( elapsed * 100 / total )); [ "$pct" -gt 100 ] && pct=100
    eta=$(( total - elapsed )); [ "$eta" -lt 0 ] && eta=0
    tstr="$(_fmt_progress_secs "$elapsed")/$(_fmt_progress_secs "$total") ${pct}%  ETA $(_fmt_progress_secs "$eta")"
  else
    tstr="$(_fmt_progress_secs "$elapsed")"
  fi
  msg=$(printf '[soak %s %s] step %s/>=%s' "$label" "$tstr" "$njit" "$minit")
  [ -n "$ms" ] && msg="$msg  last $(awk -v v="$ms" 'BEGIN{printf "%.2f", v/1000}')s"
  [ -n "$tf" ] && msg="$msg  ${tf} TFLOP/s/GPU"
  (( NNODES > 1 )) && msg="$msg  node${NODE_RANK}"
  # 双机只有全局末级 rank 打印 iteration，非末级节点步数恒 0 属正常，别误读成卡死
  if (( NNODES > 1 )) && [ "${njit:-0}" -eq 0 ] && [ "$elapsed" -gt 30 ]; then
    msg="$msg  (步数由全局末级 rank 打印)"
  fi
  ts=$(date '+%F %T')
  [ -n "${PROGRESS_FILE:-}" ] && printf '%s  %s\n' "$ts" "$msg" >> "$PROGRESS_FILE"
  printf '%s  %s\n' "$ts" "$msg" >&2
}

progress_start() {                       # progress_start <log>
  [ "${SOAK_PROGRESS:-1}" = 1 ] || return 0
  [ -n "${SOAK_REQUIRE_ITERS:-}" ] || return 0
  local log=$1 total mins label minit secs
  case "${PROGRESS_SECS:-60}" in ''|*[!0-9]*|0) PROGRESS_SECS=60 ;; esac
  mins="${SOAK_TARGET_MINS:-0}"
  case "$mins" in ''|*[!0-9]*) mins=0 ;; esac
  total=$(( mins * 60 ))
  label="${SOAK_LABEL:-soak}"; minit="${SOAK_REQUIRE_ITERS:-0}"
  secs="$PROGRESS_SECS"
  PROGRESS_FILE="$LOGDIR/soak-progress.log"
  PROGRESS_LOG="$log"
  PROGRESS_T0=$(date +%s)
  printf '%s  === %s 开始：目标 %s 分钟，最少 %s 步，node%s ===\n' \
    "$(date '+%F %T')" "$label" "$mins" "$minit" "${NODE_RANK:-0}" >> "$PROGRESS_FILE"
  ( while :; do sleep "$secs"; progress_tick "$log" "$total" "$minit" "$label"; done ) &
  PROGRESS_PID=$!
  printf '    · 进度每 %ss 刷新（终端 stderr）→ %s\n' "$secs" "$PROGRESS_FILE" >&2
  progress_tick "$log" "$total" "$minit" "$label"   # 立即给一行，不用等首个间隔
  return 0
}

progress_stop() {
  [ -n "${PROGRESS_PID:-}" ] || return 0
  kill "$PROGRESS_PID" 2>/dev/null
  wait "$PROGRESS_PID" 2>/dev/null || true
  PROGRESS_PID=""
  # 容器 stdout 可能到进程退出才批量刷盘；收尾再读一次，给出最终步数而非 60s 周期点的滞后值。
  if [ -n "${PROGRESS_LOG:-}" ] && [ "${SOAK_PROGRESS:-1}" = 1 ]; then
    local mins="${SOAK_TARGET_MINS:-0}" total
    case "$mins" in ''|*[!0-9]*) mins=0 ;; esac
    total=$(( mins * 60 ))
    progress_tick "$PROGRESS_LOG" "$total" "${SOAK_REQUIRE_ITERS:-0}" "${SOAK_LABEL:-soak}"
  fi
  [ -n "${PROGRESS_FILE:-}" ] && printf '%s  === %s 结束 ===\n' \
    "$(date '+%F %T')" "${SOAK_LABEL:-soak}" >> "$PROGRESS_FILE"
  PROGRESS_FILE=""; PROGRESS_LOG=""
  return 0
}

expert_sampler_start() {                 # expert_sampler_start <log> <csv>
  EXPERT_SAMPLER_PID=""
  ( while :; do expert_sample "$1" "$2"; sleep "$EXPERT_SAMPLE_SECS"; done ) &
  EXPERT_SAMPLER_PID=$!
  echo "    · 专家负载采样 → $2（每 ${EXPERT_SAMPLE_SECS}s）"
}

expert_sampler_stop() {
  [ -n "${EXPERT_SAMPLER_PID:-}" ] || return 0
  kill "$EXPERT_SAMPLER_PID" 2>/dev/null
  wait "$EXPERT_SAMPLER_PID" 2>/dev/null || true
  EXPERT_SAMPLER_PID=""
}

# ---- 顶层常量（原脚本散落在函数之间，值不变）----
EXPERT_SAMPLER_PID=""
