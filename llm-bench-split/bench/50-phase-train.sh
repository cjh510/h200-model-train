# ============================================================================
# 50-phase-train —— 训练类阶段编排：argscheck / hw / coll / nccl / s8b / dx8b / d70b /
# moe / moe-dp / io / fault / report。每个 phase_* 就是 `--phase X` 的全部逻辑。
# ============================================================================

phase_argscheck() {          # --phase argscheck
  ensure_args_check
  echo "    · 产物：$ARGS_HELP（--help 原文）、$ARGS_CHECK"
  grep -oE '"status": "[A-Z]+"|"missing": \[[^]]*\]|"critical_missing": \[[^]]*\]' "$ARGS_CHECK" | sed 's/^/      /'
  if args_gate_moe; then
    echo "    · 关键 flag 齐备，moe / moe-dp 可执行"
  else
    echo "    · 关键 flag 缺失：moe / moe-dp 会直接 SKIP，不会带错参数硬跑"
  fi
}

phase_hw() {          # --phase hw
  # ------------------------------------------------------------------ 硬件基线
  TELEMETRY=1 dmon_start "$LOGDIR/dmon.csv"
  run_probe hw_probe.py hw --out /workspace/logs/hw.jsonl --secs "${PROBE_SECS:-5}"
  dmon_stop
  echo "    · 指标 → $LOGDIR/hw.jsonl（05_report 的微基准表读它）"
}

phase_coll() {          # --phase coll
  # ------------------------------------------------- 单机 collective 全类型扫描
  if (( NNODES == 1 )) && [ "$OPTIONAL" != 1 ]; then
    echo "   · 单机 coll 已并入 hw（hw_probe 默认 all，含四类 collective × 6 尺寸）"
    echo "     需要单独扫描 collective 时用 OPTIONAL=1 再跑本阶段。"
    exit 0
  fi
  ensure_args_check
  run_probe hw_probe.py coll --only coll --out /workspace/logs/coll.jsonl \
    --secs "${PROBE_SECS:-5}" --sizes "$(coll_sizes)"; rc=$?
  # 双机：两台都跑同一命令，各自落在自己的 run 目录（文件名带 node_rank）
  if (( NNODES > 1 )); then
    cp "$LOGDIR/coll.jsonl" "$LOGDIR/coll-node${NODE_RANK}.jsonl" 2>/dev/null || true
  fi
  echo "    · 指标 → $LOGDIR/coll.jsonl（op × size；all_to_all 含 groups/tokens_per_group/split_dim）"
  if (( NNODES > 1 && rc != 0 )); then
    echo "   !! coll 探针失败（rc=$rc），见 $LOGDIR/coll.txt"
    sync_logs
    exit "$rc"
  fi
  sync_logs
}

phase_nccl() {          # --phase nccl
  # ------------------------------------------------- 双机 collective 直测（nccl-tests 等价）
  [ "$NNODES" -gt 1 ] || { echo "!! nccl 阶段是双机探针，用 NNODES=2 跑"; exit 2; }
  run_probe hw_probe.py nccl --only coll --out /workspace/logs/nccl.jsonl \
    --secs "${PROBE_SECS:-5}" --sizes "$(coll_sizes)"; rc=$?
  echo "    · 指标 → $LOGDIR/nccl.jsonl"
  echo "    · NCCL 实际通道（应为 NET/IB 且 HCA 为 mlx5_bond_*）："
  grep -m6 -E 'NET/IB|NET/Socket|Using network|NCCL_IB_HCA' "$LOGDIR/nccl.txt" 2>/dev/null | sed 's/^/      /' || true
  grep -qE 'NET/IB' "$LOGDIR/nccl.txt" 2>/dev/null \
    || echo "      ! 没看到 NET/IB：检查 NCCL_IB_HCA / GID=$NCCL_GID / ibv_devinfo state"
  sync_logs
  # 非法访存后 CUDA 上下文已污染，继续跑后续阶段只会把失败记错阶段（本次就被误当成 moe）
  if [ "$rc" -ne 0 ]; then
    echo "   !! nccl 探针失败（rc=$rc），见 $LOGDIR/nccl.txt；后续阶段未执行"
    exit "$rc"
  fi
}

phase_s8b() {          # --phase s8b
  # ------------------------------------------------------ 单机 8B：S1–S4
  local b_mbs=${S8B_MBS_LIST%% *}                    # 基线档 = mbs 列表首个
  local b_gbs=$(( b_mbs * S8B_GBS_MULT ))
  echo "########## S1 micro batch 上界（TP${S8B_TP}/PP${S8B_PP}，num_micro_batches 固定让算力持续饱和）##########"
  mark_baseline "w${WORLD}_8B_tp${S8B_TP}_pp${S8B_PP}_mbs${b_mbs}_s${S8B_SEQ}_gb${b_gbs}_bf16" \
    "S1 首档：mbs${b_mbs}/seq${S8B_SEQ}/GBS${b_gbs}，每次原样复跑做基线"
  for mbs in $S8B_MBS_LIST; do
    run_train "8B_tp${S8B_TP}_pp${S8B_PP}_mbs${mbs}_s${S8B_SEQ}_gb$(( mbs * S8B_GBS_MULT ))_bf16" \
      LLAMA_8B "$S8B_TP" "$S8B_PP" "$mbs" "$S8B_SEQ" $(( mbs * S8B_GBS_MULT ))
  done
  echo "########## S2 并行策略（GBS 固定 32 才可比）##########"
  local spec tp pp
  for spec in $S8B_TOPO_LIST; do
    IFS=':' read -r tp pp <<< "$spec"
    run_train "8B_tp${tp}_pp${pp}_mbs${S8B_TOPO_MBS}_s${S8B_TOPO_SEQ}_gb${S8B_TOPO_GBS}_bf16" \
      LLAMA_8B "$tp" "$pp" "$S8B_TOPO_MBS" "$S8B_TOPO_SEQ" "$S8B_TOPO_GBS"
  done
  echo "########## S3 长序列（TP8 mbs1 GBS8：单机唯一能逼出显存上限的方向）##########"
  for seq in $S8B_SEQ_LIST; do
    run_train "8B_tp${S8B_TP}_pp${S8B_PP}_mbs${S8B_SEQ_MBS}_s${seq}_gb${S8B_SEQ_GBS}_bf16" \
      LLAMA_8B "$S8B_TP" "$S8B_PP" "$S8B_SEQ_MBS" "$seq" "$S8B_SEQ_GBS"
  done
  echo "########## S4 FP8 vs BF16 ##########"
  # 判定只能走容器 --help 原文（上午已修 $MA 为空导致的假 SKIP，这里保持 ensure_args_check）。
  # P0（11-10 下午）：**--fp8-recipe 只选缩放策略，单独传不会把 args.fp8 从 None 打开**。
  # 现场 fp8 档日志 `fp8 ..... None` / params_dtype=bfloat16 / loss 与 grad norm 与 BF16
  # 档逐位相同 / 吞吐只差 0.26% —— 是「名义 FP8」。真正的入口是 --fp8-format
  # （TE 才会进 fp8_autocast），必须和 recipe 一起传：
  #   --fp8-format=e4m3  → args.fp8=e4m3，启用 FP8 GEMM
  #   --fp8-recipe=delayed → 缩放策略（RUNBOOK §2.5 锁定 delayed）
  # 老版本没有 recipe 时，只有 format 仍能启用 FP8；只有 recipe 没有 format 时必须 SKIP。
  ensure_args_check
  FP8_OPTS=()
  FP8_ENTRY=""
  if flag_in_help fp8-format; then
    FP8_ENTRY="--fp8-format=e4m3"
    FP8_OPTS+=( "$FP8_ENTRY" )
    flag_in_help fp8-recipe && FP8_OPTS+=( --fp8-recipe=delayed )
  fi
  if [ -n "$FP8_ENTRY" ]; then
    echo "FP8 入口：${FP8_OPTS[*]}（据 $ARGS_HELP）"
    local s4="8B_tp${S8B_TP}_pp${S8B_PP}_mbs${S8B_FP8_MBS}_s${S8B_FP8_SEQ}_gb${S8B_FP8_GBS}"
    run_train "${s4}_bf16" LLAMA_8B "$S8B_TP" "$S8B_PP" "$S8B_FP8_MBS" "$S8B_FP8_SEQ" "$S8B_FP8_GBS"
    run_train "${s4}_fp8"  LLAMA_8B "$S8B_TP" "$S8B_PP" "$S8B_FP8_MBS" "$S8B_FP8_SEQ" "$S8B_FP8_GBS" "${FP8_OPTS[@]}"
    # 防回归自检：Megatron 参数打印里 args.fp8 必须是 e4m3/hybrid，不能还是 None。
    # 不中断整轮（exit 会让 lc_run_stage_done 不写 index.md，制造新的元数据缺口），
    # 改为落标记 + index 口径说明 + 醒目 FAIL，重跑后一眼可见。
    local s4t="8B_tp${S8B_TP}_pp${S8B_PP}_mbs${S8B_FP8_MBS}_s${S8B_FP8_SEQ}_gb${S8B_FP8_GBS}"
    FP8_LOG="$LOGDIR/w${WORLD}_${s4t}_fp8.log"
    if grep -qE '^[[:space:]]*fp8 .* (e4m3|hybrid)([[:space:]]|$)' "$FP8_LOG" 2>/dev/null; then
      echo "  ✔ FP8 已真正启用（args.fp8 ≠ None）"
      rm -f "$LOGDIR/.fp8_not_active"
    else
      echo "  !! [FAIL] FP8 档 args.fp8 仍为 None：名义 FP8，本档加速比不可用" >&2
      echo "     核对 $FP8_LOG 的 'fp8 ..... None'，以及 TE 是否接受 --fp8-format" >&2
      : > "$LOGDIR/.fp8_not_active"
      note_index "S4 假 FP8" "FP8 档未启用（args.fp8=None），加速比不可用，标记见 .fp8_not_active"
    fi
  else
    echo "  跳过：容器 --help 里没有 --fp8-format（据 $ARGS_HELP）"
    echo "        只有 --fp8-recipe 不能启用 FP8，整组 SKIP，不跑假 FP8 档"
    echo "        核对：grep -iE '^ *--fp8' $ARGS_HELP"
    echo "        BF16 基准档照跑，S4 只缺 FP8 对照："
    run_train "8B_tp${S8B_TP}_pp${S8B_PP}_mbs${S8B_FP8_MBS}_s${S8B_FP8_SEQ}_gb${S8B_FP8_GBS}_bf16" \
      LLAMA_8B "$S8B_TP" "$S8B_PP" "$S8B_FP8_MBS" "$S8B_FP8_SEQ" "$S8B_FP8_GBS"
  fi
}

phase_dx8b() {          # --phase dx8b
  # ------------------------------------- 双机 8B TP8×DP2：机间带宽门禁
  echo "########## D1 训练对照：GBS=32（8 卡口径）##########"
  run_train "8B_tp${DX8B_TP}_pp${DX8B_PP}_mbs${DX8B_MBS}_s${DX8B_SEQ}_gb${DX8B_GBS}_bf16" \
    LLAMA_8B "$DX8B_TP" "$DX8B_PP" "$DX8B_MBS" "$DX8B_SEQ" "$DX8B_GBS"
  echo "判据：与单机同 GBS 的 gb32 比，理想比值 0.5；≤0.6 视为机间链路可用。"
  echo "      1.0 左右 = 加一台机器吞吐不涨；>2.0 = 负收益。"
  echo "      口径：双机 accum=8 / 单机 accum=16（mbs2×dp1），0.5 只是强扩展理想比值，非同 accum 比较。"
  note_index "D1 accum" "D1（8B GBS32）：双机 accum=8、单机 accum=16，理想 0.5 只是强扩展比值，非同 accum 比较。"
  sync_logs
}

phase_d70b() {          # --phase d70b
  # ------------------------------------- 双机 70B TP8×PP2（跨机只传激活）
  if (( WORLD < 16 )); then
    echo "!! d70b 需要 world≥16（2 机 × 8 卡），当前 world=$WORLD、NNODES=$NNODES。"
    echo "   这是并行度/机器数配置错误，不是硬件 OOM：70B 在 8 卡上必然 OOM，且容易被误读成显卡故障。"
    exit 2
  fi
  echo "########## D2 流水线气泡（m=4 时 bubble≈25%）##########"
  local b_gbs=${D70B_BUBBLE_GBS_LIST%% *}
  mark_baseline "w${WORLD}_70B_tp${D70B_TP}_pp${D70B_PP}_mbs${D70B_MBS}_s${D70B_SEQ}_gb${b_gbs}_bf16" \
    "D2 首档：mbs${D70B_MBS}/seq${D70B_SEQ}/GBS${b_gbs}，每次原样复跑做基线"
  for m in $D70B_BUBBLE_GBS_LIST; do
    run_train "70B_tp${D70B_TP}_pp${D70B_PP}_mbs${D70B_MBS}_s${D70B_SEQ}_gb${m}_bf16" \
      LLAMA_70B "$D70B_TP" "$D70B_PP" "$D70B_MBS" "$D70B_SEQ" "$m"
  done
  echo "########## D3 序列长度与 OOM 边界（OOM 记 OOM_BOUNDARY，不 dual_stop）##########"
  ALLOW_OOM=1 OOM_AS_BOUNDARY=1
  for seq in $D70B_SEQ_LIST; do
    run_train "70B_tp${D70B_TP}_pp${D70B_PP}_mbs${D70B_MBS}_s${seq}_gb${D70B_SEQ_GBS}_bf16" \
      LLAMA_70B "$D70B_TP" "$D70B_PP" "$D70B_MBS" "$seq" "$D70B_SEQ_GBS"
  done
  ALLOW_OOM=0 OOM_AS_BOUNDARY=0
  sync_logs
}

phase_moe() {          # --phase moe
  # ------------------------------------- MoE / MLA：单机 EP8×TP1 / EP2×TP4 + 双机 EP16
  ensure_args_check
  moe_probe_enable || true              # 缺失只降级并告警，不阻断训练
  MOE_GBS_USE=${MOE_GBS:-}
  [ -n "$MOE_GBS_USE" ] || { (( NNODES > 1 )) && MOE_GBS_USE=16 || MOE_GBS_USE=8; }
  moe_resolve_attn; gate_rc=$?
  if [ "$gate_rc" -eq 2 ]; then exit 2; fi
  if [ "$gate_rc" -ne 0 ]; then
    echo "   !! SKIP：$MOE_NOTE"
    MOE_TOPO="mla" MOE_ATTN=mla MOE_EXPERIMENTAL=0 \
      moe_record SKIP 0 0 0 "-" 1 4096 "$MOE_GBS_USE"
    sync_logs
    exit 0
  fi
  if ! args_gate_moe; then
    MOE_NOTE="关键 MoE flag 缺失（见 $ARGS_CHECK）"
    MOE_TOPO="gate" MOE_ATTN=mla MOE_EXPERIMENTAL=0 \
      moe_record SKIP 0 0 0 "-" 1 4096 "$MOE_GBS_USE"
    echo "    → SKIP：关键 MoE flag 缺失，本阶段不拼参数、不硬跑（见 $ARGS_CHECK）"
    sync_logs
    exit 0
  fi
  MOE_BASE=( "${MOE_B[@]}" )
  [ "$MOE_EXPERIMENTAL" = 1 ] && MOE_BASE+=( --enable-experimental )
  MOE_ACTIVE_PARAMS=$(moe_est "${MOE_B_EST[@]}" 8 1 8 "$MOE_B_VOCAB" | awk '{print $2}')
  MOE_STATIC_GB=$(moe_est "${MOE_B_EST[@]}" 1 1 8 "$MOE_B_VOCAB" | awk '{print $1}')
  echo "→ 配置 B：24 层 / 32 专家 + 1 共享 / topk2 / MLA（experimental=$MOE_EXPERIMENTAL）；GBS=$MOE_GBS_USE；激活参数≈${MOE_ACTIVE_PARAMS}；静态≈${MOE_STATIC_GB} GiB/卡(TP1 PP1 EP8)"
  RC=0
  if (( NNODES > 1 )); then COMBOS=( "16 1 1" ); else COMBOS=( "8 1 1" "2 4 1" ); fi
  for spec in "${COMBOS[@]}"; do
    read -r ep tp pp <<< "$spec"
    dp=$(( WORLD / tp / pp ))
    if (( (dp * tp) % ep != 0 )); then
      echo "   !! EP=$ep 不能整除 dp×tp=$(( dp * tp ))，跳过该档（SKIP，非 FAILED）"
      MOE_TOPO="ep${ep}_tp${tp}" moe_record SKIP "$ep" "$tp" "$pp" "-" 1 4096 "$MOE_GBS_USE"
      continue
    fi
    for disp in alltoall allgather; do
      if ! disp_supported "$disp"; then
        echo "   ! dispatcher=$disp 不在 --help choices 中，跳过（SKIP）"
        MOE_TOPO="ep${ep}_tp${tp}" moe_record SKIP "$ep" "$tp" "$pp" "$disp" 1 4096 "$MOE_GBS_USE"
        continue
      fi
      mapfile -t MOE_ARGS < <(filter_args "${MOE_BASE[@]}" \
        --expert-model-parallel-size="$ep" --moe-token-dispatcher-type="$disp")
      MOE_TOPO="ep${ep}_tp${tp}"
      echo "########## EP${ep}×TP${tp}×PP${pp}  dispatcher=$disp ##########"
      run_train "MOE_${ep}ep_${tp}tp_${disp}_mbs1_s4096" MOE_ARGS "$tp" "$pp" 1 4096 "$MOE_GBS_USE"
      rc=$?
      if [ "$rc" -eq 0 ]; then
        MOE_TOPO="ep${ep}_tp${tp}" moe_record OK "$ep" "$tp" "$pp" "$disp" 1 4096 "$MOE_GBS_USE"
      else
        MOE_TOPO="ep${ep}_tp${tp}" moe_record FAILED "$ep" "$tp" "$pp" "$disp" 1 4096 "$MOE_GBS_USE"
        RC=1
      fi
    done
  done
  echo "    · 指标 → $LOGDIR/moe.jsonl（step_s / 每卡 token/s / 激活参数 MFU / 专家负载；attention=$MOE_ATTN experimental=$MOE_EXPERIMENTAL）"
  sync_logs
  exit "$RC"
}

phase_moe_dp() {          # --phase moe-dp
  # ------------------------------------- 仅双机：EP8×DP2（专家并行留机内，梯度跨机 all-reduce）
  (( NNODES > 1 )) || { echo "!! moe-dp 只用于双机对照，用 NNODES=2 跑"; exit 2; }
  ensure_args_check
  MOE_GBS_USE=${MOE_GBS:-16}
  moe_resolve_attn; gate_rc=$?
  if [ "$gate_rc" -eq 2 ]; then exit 2; fi
  if [ "$gate_rc" -ne 0 ]; then
    echo "   !! SKIP：$MOE_NOTE"
    MOE_TOPO="mla" MOE_ATTN=mla MOE_EXPERIMENTAL=0 \
      moe_record SKIP 0 0 0 "-" 1 4096 "$MOE_GBS_USE"
    sync_logs
    exit 0
  fi
  if ! args_gate_moe; then
    MOE_NOTE="关键 MoE flag 缺失（见 $ARGS_CHECK）"
    MOE_TOPO="gate" MOE_ATTN=mla MOE_EXPERIMENTAL=0 \
      moe_record SKIP 0 0 0 "-" 1 4096 "$MOE_GBS_USE"
    echo "    → SKIP：关键 MoE flag 缺失（见 $ARGS_CHECK）"
    sync_logs
    exit 0
  fi
  MOE_BASE=( "${MOE_B[@]}" )
  [ "$MOE_EXPERIMENTAL" = 1 ] && MOE_BASE+=( --enable-experimental )
  MOE_ACTIVE_PARAMS=$(moe_est "${MOE_B_EST[@]}" 8 1 8 "$MOE_B_VOCAB" | awk '{print $2}')
  echo "→ EP8×DP2（world=$WORLD，EP 留在机内，DP 梯度跨机）；GBS=$MOE_GBS_USE；MLA experimental=$MOE_EXPERIMENTAL"
  RC=0
  for disp in alltoall; do
    if ! disp_supported "$disp"; then
      echo "   ! dispatcher=$disp 不在 --help choices 中，跳过（SKIP）"
      MOE_TOPO=ep8_dp2 moe_record SKIP 8 1 1 "$disp" 1 4096 "$MOE_GBS_USE"
      continue
    fi
    mapfile -t MOE_ARGS < <(filter_args "${MOE_BASE[@]}" \
      --expert-model-parallel-size=8 --moe-token-dispatcher-type="$disp")
    echo "########## EP8×DP2  dispatcher=$disp ##########"
    run_train "MOE_8ep_1tp_${disp}_mbs1_s4096" MOE_ARGS 1 1 1 4096 "$MOE_GBS_USE"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      MOE_TOPO=ep8_dp2 moe_record OK 8 1 1 "$disp" 1 4096 "$MOE_GBS_USE"
    else
      MOE_TOPO=ep8_dp2 moe_record FAILED 8 1 1 "$disp" 1 4096 "$MOE_GBS_USE"
      RC=1
    fi
  done
  echo "    · 与 moe 阶段 EP16 档对照：跨机 all-to-all vs 跨机 all-reduce 的成本差"
  sync_logs
  exit "$RC"
}

phase_io() {          # --phase io
  # ---------------------------------------------------------------- 存储与数据管线
  : > "$LOGDIR/io.jsonl"
  echo "########## 存储：checkpoint 写读代理（8B；70B 为 OPTIONAL）##########"
  run_probe io_probe.py io_8b --model 8B --out /workspace/logs/io.jsonl
  if [ "$OPTIONAL" = 1 ]; then
    run_probe io_probe.py io_70b --model 70B --out /workspace/logs/io.jsonl
  else
    echo "   · 70B checkpoint 写读已降为可选（OPTIONAL=1 才跑）；8B 档已覆盖存储路径"
  fi
  DP="${DATA_PATH:-$ROOT/data}"
  if ls "$DP"/*.bin >/dev/null 2>&1 && ls "$DP"/*.idx >/dev/null 2>&1; then
    echo "########## 数据管线：检测到 $DP（去掉 --mock-data）##########"
    COMMON_REAL=()
    for a in "${COMMON[@]}"; do
      case "$a" in --mock-data|--split=*) ;; *) COMMON_REAL+=("$a") ;; esac
    done
    ARCH8=( --num-layers=32 --hidden-size=4096 --ffn-hidden-size=14336 --num-attention-heads=32 )
    lc_run -v "$ROOT":/workspace -v "$LOGDIR":/workspace/logs -w "$MEGATRON" -e PYTHONPATH="$MEGATRON" \
      "$IMAGE" torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port="$(det_port io-data)" \
        pretrain_gpt.py "${COMMON_REAL[@]}" "${ARCH8[@]}" \
          --tensor-model-parallel-size=8 --pipeline-model-parallel-size=1 \
          --data-path "$DP" --split=969,30,1 \
          --seq-length=4096 --max-position-embeddings=4096 --micro-batch-size=1 --global-batch-size=8 \
          --bf16 --train-iters=20 --log-interval=1 --eval-iters=0 --seed=1234 \
      > "$LOGDIR/io-dataloader.log" 2>&1
    echo "    · tokens/s 见 $LOGDIR/io-dataloader.log 的 throughput 行（含 dataloader/tokenizer）"
  else
    printf '{"kind":"dataloader","status":"SKIP","note":"无真实数据集 %s/*.bin+.idx；mock-data 路径不含 dataloader/tokenizer"}\n' "$DP" >> "$LOGDIR/io.jsonl"
    echo "    · 数据管线 SKIP：$DP 下没有 .bin/.idx；mock-data 不经过 dataloader，不能代表真实吞吐"
  fi
  echo "    · 指标 → $LOGDIR/io.jsonl（存储为分片写读代理，不是 Megatron 分布式 checkpoint 格式）"
}

phase_fault() {          # --phase fault
  # ---------------------------------------------------------------- 故障注入
  FR="$LOGDIR/fault.jsonl"; : > "$FR"
  rec() { printf '{"case":"%s","status":"%s","rc":%s,"note":"%s"}\n' "$1" "$2" "${3:-0}" "$4" >> "$FR"; }

  # P0（11-10）：fault_train 漏了 CUDA_DEVICE_MAX_CONNECTIONS=1，三个用例都在启动断言处
  # 崩溃；旧判据只看 rc≠0，于是 2 个「启动失败」被记成 PASS。现在做两层校验：
  #   ① 启动健全性：出现 AssertionError / 参数错误 → INVALID（用例根本没执行，不参与 PASS）
  #   ② 预期特征：日志必须命中被注入故障的证据，任意非 0 退出不再算通过
  # 另外把固定 sleep 改成「等日志出现 iteration 再注入」，避免训练没起来/已跑完时误判。
  fault_startup_bad() {                 # <log>：进程根本没进入训练
    grep -qiE 'AssertionError|unrecognized arguments|invalid choice|ModuleNotFoundError|ImportError|NameError|SyntaxError' "$1" 2>/dev/null
  }
  fault_kill_evidence() {               # <log>：torchrun 报出 rank 被 SIGKILL
    grep -qiE 'SIGKILL|signal: 9|exitcode: -9|exitcode: 137|ChildFailedError|Killed' "$1" 2>/dev/null
  }
  fault_timeout_evidence() {            # <log>：NCCL watchdog 真的超时
    grep -qiE 'Watchdog caught collective operation timeout|before timing out|[Tt]imed out|[Tt]imeout.*[Ww]atchdog|ProcessGroupNCCL.*[Tt]imeout' "$1" 2>/dev/null
  }
  fault_wait_ready() {                  # <log> <秒> <后台PID>：等到第一个 iteration
    local log=$1 lim=${2:-120} pid=$3 i
    for ((i=0; i<lim; i++)); do
      grep -q "elapsed time per iteration" "$log" 2>/dev/null && return 0
      kill -0 "$pid" 2>/dev/null || return 1
      sleep 1
    done
    return 1
  }

  echo "########## F1 预期 OOM：70B 在单机 8 卡 ##########"
  fault_train "$LOGDIR/fault-oom.log" LLAMA_70B
  if fault_startup_bad "$LOGDIR/fault-oom.log"; then
    rec oom INVALID 0 "启动期错误（AssertionError/参数错误），用例未执行；见 fault-oom.log"
  elif grep -qiE 'CUDA out of memory|OutOfMemoryError' "$LOGDIR/fault-oom.log"; then
    rec oom PASS 0 "70B world=8 按预期 OOM"
  else
    rec oom FAIL 0 "未出现预期 OOM，查 fault-oom.log"
  fi

  echo "########## F2 kill 一个 rank：应非 0 退出且容器自动清理 ##########"
  CNAME="bench-fault-kill"
  lc rm -f "$CNAME" >/dev/null 2>&1
  fault_train "$LOGDIR/fault-kill.log" LLAMA_8B "$CNAME" &
  BP=$!
  if fault_wait_ready "$LOGDIR/fault-kill.log" 120 "$BP"; then
    lc exec "$CNAME" bash -c 'kill -9 $(pgrep -f pretrain_gpt.py | head -1)' >/dev/null 2>&1
  else
    echo "  ! F2 训练未在 120s 内进入 iteration（或已退出），仍按日志实际证据判定" >&2
  fi
  wait "$BP"; RC=$?
  FKLOG="$LOGDIR/fault-kill.log"
  CLEAN=0; lc ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CNAME" || CLEAN=1
  if fault_startup_bad "$FKLOG"; then
    rec kill_rank INVALID "$RC" "启动期错误（进程未进入训练），非 0 退出不算容错通过"
  elif [ "$RC" -ne 0 ] && [ "$CLEAN" = 1 ] && fault_kill_evidence "$FKLOG"; then
    rec kill_rank PASS "$RC" "进入 iteration 后 kill -9；日志命中 SIGKILL 特征，容器已清理"
  else
    rec kill_rank FAIL "$RC" "rc=$RC / 容器清理=$CLEAN / SIGKILL 特征未命中；见 fault-kill.log"
  fi
  lc rm -f "$CNAME" >/dev/null 2>&1

  echo "########## F3 NCCL 超时：SIGSTOP 一个 rank + 2 分钟看门狗 ##########"
  CNAME2="bench-fault-stop"
  lc rm -f "$CNAME2" >/dev/null 2>&1
  fault_train "$LOGDIR/fault-timeout.log" LLAMA_8B "$CNAME2" --distributed-timeout-minutes=2 &
  BP=$!
  if fault_wait_ready "$LOGDIR/fault-timeout.log" 120 "$BP"; then
    lc exec "$CNAME2" bash -c 'kill -STOP $(pgrep -f pretrain_gpt.py | head -1)' >/dev/null 2>&1
  else
    echo "  ! F3 训练未在 120s 内进入 iteration（或已退出），仍按日志实际证据判定" >&2
  fi
  for _ in $(seq 1 60); do kill -0 "$BP" 2>/dev/null || break; sleep 5; done
  wait "$BP"; RC=$?
  FTLOG="$LOGDIR/fault-timeout.log"
  if fault_startup_bad "$FTLOG"; then
    rec nccl_timeout INVALID "$RC" "启动期错误（进程未进入训练），非 0 退出不算看门狗生效"
  elif [ "$RC" -ne 0 ] && fault_timeout_evidence "$FTLOG"; then
    rec nccl_timeout PASS "$RC" "日志命中 NCCL watchdog 超时特征，进程组在窗口内被终止"
  else
    rec nccl_timeout FAIL "$RC" "rc=$RC 或未见 watchdog 超时特征；可能卡死已人工清理"
  fi
  lc rm -f "$CNAME2" >/dev/null 2>&1
  echo "    · 用例结果 → $FR（PASS/FAIL/INVALID 不中断后续阶段）"
}

phase_report() {          # --phase report
  RP="$HERE/05_report.py"; [ -f "$RP" ] || RP="$ROOT/05_report.py"
  # 读整次 run（各阶段目录由 05_report.py 递归收集），报告落到 10-report/
  python3 "$RP" --logs "$LC_RUN_DIR" 2>&1 | tee "$LOGDIR/report.txt" "$LOGDIR/report.md" \
    || echo "!! 05_report.py 不在 $HERE 也不在 $ROOT"
  echo "    · 报告 → $LOGDIR/report.txt / report.md"
  echo "    · run 目录：$LC_RUN_DIR"
}
