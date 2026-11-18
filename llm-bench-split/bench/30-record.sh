# ============================================================================
# 30-record —— 记账与解析：results/moe/health 的写入，step/err/expert 的解析，
# 以及 --help 参数自检与 MoE 前置门。都不启动容器。
# ============================================================================

health_snap() {                          # health_snap <输出json>
  bash "$HERE_PROBES/gpu_health.sh" "$1" >/dev/null 2>&1 || true
}

# 长稳前后快照 → 增量（逐卡 + 按机器）。soak 正常收尾与 dual_stop 提前中止都走这里，
# 保证「失败也要留下硬件证据」。
health_delta_json() {                    # health_delta_json [前缀]（moe- = MoE 长稳独立增量）
  local pfx="${1:-}"
  command -v python3 >/dev/null 2>&1 || { echo "   ! 宿主机没有 python3，请人工比对 health-before/after.json"; return 0; }
  [ -f "$LOGDIR/${pfx}health-before.json" ] && [ -f "$LOGDIR/${pfx}health-after.json" ] || return 0
  python3 - "$LOGDIR/${pfx}health-before.json" "$LOGDIR/${pfx}health-after.json" "$LOGDIR/${pfx}health-delta.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); a = json.load(open(sys.argv[2]))
keys = ("ecc_corrected", "ecc_uncorrected", "retired_pending", "nvlink_errors", "xid", "sxid")
d = {k: (a.get(k) or 0) - (b.get(k) or 0) for k in keys}
bg = {g.get("index"): g for g in (b.get("per_gpu") or [])}
ag = {g.get("index"): g for g in (a.get("per_gpu") or [])}
gk = ("ecc_corrected", "ecc_uncorrected", "retired_pending", "nvlink_errors")
gd, bad = [], []
for i in sorted(set(ag) | set(bg), key=lambda x: (x is None, x)):
    row = {"index": i}
    row.update({k: (ag.get(i, {}).get(k) or 0) - (bg.get(i, {}).get(k) or 0) for k in gk})
    gd.append(row)
    for k in gk:
        if row[k] > 0:
            bad.append({"gpu": i, "key": k, "delta": row[k]})
for k in ("xid", "sxid"):
    if d.get(k, 0) > 0:
        bad.append({"gpu": a.get("xid_gpus") or "未定位", "key": k, "delta": d[k]})
def gpu_tag(v):
    return "GPU" + ",".join(str(x) for x in v) if isinstance(v, list) else "GPU%s" % v
rs = a.get("retired_pages_state") or b.get("retired_pages_state") or ""
if rs == "SKIP":                        # 字段不支持 → 三态里的 SKIP，不能算硬件失败
    bad = [x for x in bad if x["key"] != "retired_pending"]
json.dump({"host": a.get("host") or b.get("host") or "?",
           "before": b, "after": a, "delta": d, "gpu_delta": gd, "bad": bad,
           "retired_pages_state": rs,
           "status": "FAIL" if bad else "PASS"}, open(sys.argv[3], "w"), ensure_ascii=False)
print("健康增量（%s）：" % (a.get("host") or "?"), d,
      "retired=SKIP（字段不支持）" if rs == "SKIP" else "", "→",
      ("FAIL " + ",".join("%s@%s" % (x["key"], gpu_tag(x["gpu"])) for x in bad)) if bad else "PASS")
PY
}

# 基线档：固定同一 mbs/seq/GBS/开关组合，写 baseline.txt 给报告标★，并在 index.md 里注明。
mark_baseline() {                        # mark_baseline <带 w 前缀的完整 tag> <说明>
  local tag=$1 note=$2
  [ -n "${LOGDIR:-}" ] && [ -d "$LOGDIR" ] || return 0
  grep -qxF "$tag" "$LOGDIR/baseline.txt" 2>/dev/null || echo "$tag" >> "$LOGDIR/baseline.txt"
  [ -n "${LC_RUN_DIR:-}" ] && [ -f "$LC_RUN_DIR/index.md" ] || return 0
  grep -qF "$tag" "$LC_RUN_DIR/index.md" 2>/dev/null && return 0
  printf '\n>> **基线档（本次）**：`%s` —— %s\n>> 本轮用 fix 的 DO/overlap/recompute 门控，与历史 docker 基线**不可直接比较**。\n' \
    "$tag" "$note" >> "$LC_RUN_DIR/index.md"
}

# 给 index.md 追加一条口径说明（幂等：同一个键只写一次）
note_index() {                           # note_index <键> <说明>
  [ -n "${LC_RUN_DIR:-}" ] && [ -f "$LC_RUN_DIR/index.md" ] || return 0
  grep -qF "口径（$1）" "$LC_RUN_DIR/index.md" 2>/dev/null && return 0
  printf '\n>> **口径（%s）**：%s\n' "$1" "$2" >> "$LC_RUN_DIR/index.md"
}

# 统一写 results.jsonl：状态只有 OK/OOM/PARTIAL/FAILED/SKIP/ABORTED，不再混用。
rec_result() {                           # rec_result <tag> <tp> <pp> <mbs> <seq> <gbs> <port> <fp> <iters> <status> [note]
  # highmem 标定的试跑档不算正式结果：不写 results.jsonl，否则档位表会被 4 步试跑行污染
  [ "${NO_RESULT_REC:-0}" = 1 ] && return 0
  local tag=$1 tp=$2 pp=$3 mbs=$4 seq=$5 gbs=$6 port=$7 fp=$8 njit=${9:-0} st=${10} note=${11:-}
  local extra=""
  [ -n "$note" ] && extra=",\"note\":\"$note\""
  printf '{"tag":"%s","tp":%s,"pp":%s,"mbs":%s,"seq":%s,"gbs":%s,"world":%s,"nodes":%s,"port":%s,%s,"transport":"%s","iters":%s,"status":"%s"%s}\n' \
    "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$WORLD" "$NNODES" "$port" "$fp" "${TRANSPORT:-ib}" "$njit" "$st" "$extra" \
    >> "$LOGDIR/results.jsonl"
}

# 双机任一模失败立即整阶段中止：两台必须跑同一档位序列，否则后续所有对比都错位。
dual_stop() {                            # dual_stop <tag> <status>
  (( NNODES > 1 )) || return 0
  echo "   ✗ 双机模式中止本阶段（$1 → $2），避免两台档位错位；另一台请同样退出。" >&2
  printf '{"tag":"%s","status":"ABORTED","aborted_by":"%s","reason":"%s","ts":"%s"}\n' \
    "$1" "$(hostname)" "$2" "$(date -Iseconds)" >> "$LOGDIR/results.jsonl"
  # 长稳中止也要把跑后快照与增量留下，否则硬件证据缺失
  if [ "${PHASE:-}" = soak ] && [ -f "$LOGDIR/${SOAK_PREFIX:-}health-before.json" ] \
     && [ ! -f "$LOGDIR/${SOAK_PREFIX:-}health-after.json" ]; then
    health_snap "$LOGDIR/${SOAK_PREFIX:-}health-after.json"
    health_delta_json "${SOAK_PREFIX:-}"
  fi
  sync_logs
  lc_run_stage_done "$PHASE" "$START_TS" "ABORTED"
  exit 2
}

# 双机日志汇总：node-b 把整个 run 目录回传到 node-a 同名 RUN_DIR/node-b/；
# 无 ssh/rsync 时只打印命令，不阻断测试。
sync_logs() {
  [ "$NNODES" -gt 1 ] || return 0
  [ "$NODE_RANK" != 0 ] || return 0
  local target="$PEER_ADDR:${LC_RUNS_ROOT}/${RUN_ID}/node-b/"
  if [ -n "$PEER_ADDR" ] && command -v rsync >/dev/null 2>&1 \
     && ssh -o BatchMode=yes -o ConnectTimeout=3 "$PEER_ADDR" true 2>/dev/null; then
    rsync -a "$LC_RUN_DIR/" "$target" \
      && echo "    · 本次 run 已回传 node-a:${LC_RUNS_ROOT}/${RUN_ID}/node-b/"
  else
    echo "    · 日志汇总（在 node-b 执行）：rsync -a $LC_RUN_DIR/ <node-a>:${LC_RUNS_ROOT}/${RUN_ID}/node-b/"
  fi
}

# 对端已回传的 results.jsonl 里该 tag 是否已有 OK（双机只有全局末级 rank 打印 iteration 时用）
peer_ok() {                              # peer_ok <tag>
  local tag=$1
  [ -n "${LC_RUN_DIR:-}" ] && [ -d "$LC_RUN_DIR/node-b" ] || return 1
  grep -RqsE "\"tag\":\"$tag\".*\"status\":\"OK\"" "$LC_RUN_DIR/node-b" 2>/dev/null
}

arg_has() { grep -qs "\"$1\": true" "$ARGS_CHECK"; }

arg_check_ok() { [ -s "$ARGS_CHECK" ] && grep -qs '"status": "OK"\|"status": "DEGRADED"' "$ARGS_CHECK"; }

ensure_args_check() {                    # 不存在/为空则容器内跑一次 --help
  if [ -s "$ARGS_CHECK" ] && [ -s "$ARGS_HELP" ]; then
    cp "$ARGS_HELP" "$LOGDIR/args-help.txt" 2>/dev/null || true
    cp "$ARGS_CHECK" "$LOGDIR/args-check.json" 2>/dev/null || true
    return 0
  fi
  echo "→ 参数自检：容器内 python pretrain_gpt.py --help → $ARGS_HELP"
  lc_run -v "$ROOT":/workspace -w "$MEGATRON" -e PYTHONPATH="$MEGATRON" \
    "$IMAGE" python pretrain_gpt.py --help > "$ARGS_HELP" 2>&1
  local rc=$? missing=() crit=() first=1 f
  for f in "${ARG_FLAGS[@]}"; do
    if ! grep -qE "^[[:space:]]*--$f([[:space:]=]|$)" "$ARGS_HELP"; then
      missing+=( "$f" ); [[ " ${ARG_CRIT[*]} " == *" $f "* ]] && crit+=( "$f" )
    fi
  done
  local status=OK
  [ "$rc" -ne 0 ] && status=ERROR
  [ "${#missing[@]}" -gt 0 ] && [ "$status" = OK ] && status=DEGRADED
  {
    printf '{\n  "generated_at": "%s",\n  "help_file": "%s",\n  "status": "%s",\n  "flags": {' \
      "$(date -Is)" "$ARGS_HELP" "$status"
    for f in "${ARG_FLAGS[@]}"; do
      # 注意：这里不能调 arg_has —— 此时 JSON 正在写、尚未落盘，读到的永远是 false
      if grep -qE "^[[:space:]]*--$f([[:space:]=]|$)" "$ARGS_HELP"; then
        printf '%s "%s": true' "$([ "$first" = 1 ] || printf ',')" "$f"
      else
        printf '%s "%s": false' "$([ "$first" = 1 ] || printf ',')" "$f"
      fi
      first=0
    done
    printf ' },\n  "missing": ['
    first=1
    for f in "${missing[@]}"; do printf '%s"%s"' "$([ "$first" = 1 ] || printf ',')" "$f"; first=0; done
    printf ' ],\n  "critical_missing": ['
    first=1
    for f in "${crit[@]}"; do printf '%s"%s"' "$([ "$first" = 1 ] || printf ',')" "$f"; first=0; done
    printf ' ]\n}\n'
  } > "$ARGS_CHECK"
  cp "$ARGS_HELP" "$LOGDIR/args-help.txt" 2>/dev/null || true
  cp "$ARGS_CHECK" "$LOGDIR/args-check.json" 2>/dev/null || true
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "   ! [WARN] --help 中缺失，已从候选参数剔除：${missing[*]}"
  fi
  [ -n "${crit[*]}" ] && echo "   ! [WARN] 关键 flag 缺失：${crit[*]}"
  return 0
}

args_gate_moe() {                        # 关键 flag 缺失 → 该 MoE 阶段 SKIP
  local c
  for c in "${ARG_CRIT[@]}"; do
    arg_has "$c" || { echo "   !! SKIP：关键 flag --$c 在 core_v0.13.1 的 --help 中不存在（见 $ARGS_CHECK）"; return 1; }
  done
  # MLA 路径还依赖 RoPE 融合所需的实验性开关（fused_apply_mla_rope_for_q）
  if [ "${MOE_ATTN:-mla}" = mla ] && ! moe_mla_missing; then
    echo "   !! SKIP：MLA 依赖缺失：${MOE_MLA_MISSING[*]}（见 $ARGS_HELP）"
    return 1
  fi
  return 0
}

flag_in_help() {                         # flag_in_help <name>：以 --help 原文为准（args-check 可能是旧 run 的）
  # 用法行里布尔 flag 写作 [--enable-experimental]，说明行写作 "  --enable-experimental"，
  # 两种都要认；`]` 收尾是为了不放过分组里的布尔项，同时避免把 --flag-x 当成 --flag。
  grep -qE "^[[:space:]]*\[?--$1([[:space:]=,]|\]|$)" "$ARGS_HELP" 2>/dev/null
}

moe_mla_missing() {                      # 0=依赖齐全；1=缺；缺项在 MOE_MLA_MISSING
  MOE_MLA_MISSING=()
  local f
  for f in multi-latent-attention q-lora-rank kv-lora-rank qk-head-dim \
            qk-pos-emb-head-dim v-head-dim enable-experimental; do
    flag_in_help "$f" || MOE_MLA_MISSING+=( "--$f" )
  done
  [ "${#MOE_MLA_MISSING[@]}" -eq 0 ]
}

moe_resolve_attn() {                     # 0=跑 MLA；1=SKIP（原因在 MOE_NOTE）；2=MOE_MLA 非法
  MOE_ATTN=mla; MOE_EXPERIMENTAL=0; MOE_NOTE=""
  case "${MOE_MLA:-auto}" in
    auto|1|"") ;;
    0) MOE_NOTE="MOE_MLA=0，主动跳过 MLA"; return 1 ;;
    *) MOE_NOTE="MOE_MLA=$MOE_MLA 非法（auto|1|0）"; return 2 ;;
  esac
  [ -s "$ARGS_HELP" ] || ensure_args_check
  if ! moe_mla_missing; then
    MOE_NOTE="MLA 依赖缺失：${MOE_MLA_MISSING[*]}"
    return 1
  fi
  MOE_EXPERIMENTAL=1
  return 0
}

# moe / moe-dp 两个阶段的公共前置：--help 自检 → MLA 依赖门禁 → 关键 flag 门禁 → 拼 MOE_BASE。
# 返回：0=可以跑（MOE_BASE / MOE_ACTIVE_PARAMS / MOE_EXPERIMENTAL / MOE_GBS_USE 已就绪）
#       1=SKIP（原因在 MOE_NOTE，该记哪个拓扑看 MOE_SKIP_TOPO）
#       2=MOE_MLA 取值非法，调用方直接 exit 2
# 这里不 exit、不 sync_logs、不写 moe_record：两个阶段的 SKIP 提示文案不同，留在各自阶段体里。
moe_gate() {
  ensure_args_check
  MOE_GBS_USE=${MOE_GBS:-}
  [ -n "$MOE_GBS_USE" ] || { (( NNODES > 1 )) && MOE_GBS_USE=16 || MOE_GBS_USE=8; }
  moe_resolve_attn
  local rc=$?
  if [ "$rc" -ne 0 ]; then MOE_SKIP_TOPO=mla; return "$rc"; fi
  if ! args_gate_moe; then
    MOE_NOTE="关键 MoE flag 缺失（见 $ARGS_CHECK）"
    MOE_SKIP_TOPO=gate
    return 1
  fi
  MOE_BASE=( "${MOE_B[@]}" )
  [ "$MOE_EXPERIMENTAL" = 1 ] && MOE_BASE+=( --enable-experimental )
  MOE_ACTIVE_PARAMS=$(moe_est "${MOE_B_EST[@]}" 8 1 8 "$MOE_B_VOCAB" | awk '{print $2}')
  return 0
}

# ------------------------------------------------------------ 指标/记录小工具
err_class() {                            # err_class <log>
  local log=$1
  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$log"; then echo OOM
  elif grep -qiE 'NCCL error|ncclInvalidUsage|ncclInternalError|Watchdog caught' "$log"; then echo NCCL
  elif grep -qiE 'timeout|Timed out|hang' "$log"; then echo HANG
  elif grep -qiE 'Segmentation fault|Killed|terminate called|core dumped' "$log"; then echo CRASH
  else echo OTHER; fi
}

# Megatron 的 iteration 行以 " | " 结尾，$NF 是 "|" 而不是耗时的毫秒数。
# 旧写法用 $NF 取耗时，step_seconds 恒返回 0.0000、step_jitter 恒返回 0，
# results.jsonl / moe-soak.jsonl / highmem.jsonl 里的 step_s、jit 全是假的。
# 改为按正则抓 (ms) 数值，并与 05_report.py 的 WARMUP=5 对齐：丢弃前 5 步
# （kernel 编译 / NCCL 建链 / lr warmup），样本不足时回退最后 3 步。
step_samples() {                        # step_samples <log>：逐行输出稳态 ms
  local out n
  out=$(grep -oE 'elapsed time per iteration \(ms\): [0-9.]+' "$1" 2>/dev/null | awk '{print $NF}')
  [ -n "$out" ] || return 0
  n=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
  if [ "${n:-0}" -gt 5 ]; then printf '%s\n' "$out" | tail -n +6
  else printf '%s\n' "$out" | tail -n 3
  fi
}

step_seconds() {                        # 稳态中位数的秒数（无则 0）
  step_samples "$1" | sort -n | awk '{a[NR]=$1}
       END{if(!NR){print 0; exit}
           m=(NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
           printf "%.4f", m/1000}'
}

step_jitter() {                         # 稳态 step 时间变异系数（%），<2 样本则 0
  # 与 05_report.py 同口径：样本标准差（n-1）÷ 中位数，中位数抗 FP8 单步尖峰
  step_samples "$1" | sort -n | awk '{a[NR]=$1; s+=$1; q+=$1*$1; n=NR}
       END{if(n<2){print 0; exit}
           med=(n%2) ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2
           m=s/n; sd=sqrt((q-n*m*m)/(n-1))
           if(med>0) printf "%.2f", sd/med*100; else print 0}'
}


moe_record() {                           # status ep tp pp dispatcher mbs seq gbs
  local st=$1 ep=$2 tp=$3 pp=$4 disp=$5 mbs=$6 seq=$7 gbs=$8
  local log="$LOGDIR/w${WORLD}_MOE_${ep}ep_${tp}tp_${disp}_mbs${mbs}_s${seq}.log"
  local ss tok mfu load="null"
  ss=$(step_seconds "$log")
  tok=$(( gbs * seq / WORLD ))
  mfu=$(mfu_active "${MOE_ACTIVE_PARAMS:-}" "$gbs" "$seq" "$ss")
  load=$(expert_load_max_over_mean "$log")
  [ -n "$load" ] || load="null"
  printf '{"world":%d,"nodes":%d,"topology":"%s","ep":%d,"tp":%d,"pp":%d,"dispatcher":"%s","mbs":%d,"seq":%d,"gbs":%d,"accum":%d,"step_s":%s,"tokens_per_gpu":%d,"mfu_active":%s,"expert_load_max_over_mean":%s,"active_params":%s,"attention":"%s","experimental":%s,"transport":"%s","note":"%s","status":"%s","ts":"%s"}\n' \
    "$WORLD" "$NNODES" "${MOE_TOPO:-ep${ep}_tp${tp}}" "$ep" "$tp" "$pp" "$disp" "$mbs" "$seq" "$gbs" \
    "$(( gbs / (mbs * (WORLD / tp / pp)) ))" "${ss:-0}" "$tok" "$mfu" "$load" "${MOE_ACTIVE_PARAMS:-0}" \
    "${MOE_ATTN:-mla}" "$([ "${MOE_EXPERIMENTAL:-0}" = 1 ] && echo true || echo false)" "${TRANSPORT:-ib}" "${MOE_NOTE:-}" \
    "$st" "$(date -Is)" >> "$LOGDIR/moe.jsonl"
}

# 静态/激活参数估算（16B/参数 = bf16 权重+梯度 + Adam fp32 状态）
moe_est() {                              # layers experts hidden moe_ffn shared topk tp pp ep vocab
  awk -v L="$1" -v E="$2" -v H="$3" -v F="$4" -v S="$5" -v K="$6" -v T="$7" -v P="$8" -v EP="$9" -v V="${10}" '
    BEGIN{ kv=1024; attn=2*H*H+2*kv*H; shared=3*H*S; perex=3*H*F;
      non=L*(attn+shared)+2*V*H; ex=L*E*perex;
      static=16*(non/(T*P)+ex/(T*P*EP))/1073741824;
      active=2*V*H+L*(attn+shared+(K/E)*perex);
      printf "%.1f %.0f", static, active }'
}

dense_est() {                            # params tp pp → GiB/卡
  awk -v n="$1" -v s="$2" -v p="$3" 'BEGIN{printf "%.1f", 16*n/(s*p)/1073741824}'
}

# 跨机 collective 尺寸：按「总传输量 = size × NNODES」与 COLL_MAX_MSG 比较。
# allgather/reduce_scatter 的实际流量会放大 nodes 倍（1G 档 allgather = 16GiB/卡跨机突发），
# 11-08 实测：这类大突发在无 PFC 的 RoCE 上丢包 → RC 重传卡死 600s 被看门狗杀；
# 且 COLL_MAX_MSG 只按 size 过滤时 1G 档漏网。单机保持原列表不变。
coll_sizes() {
  local s out=() dropped=()
  local IFS=,
  for s in $COLL_SIZES; do
    if (( NNODES > 1 && COLL_MAX_MSG > 0 && s * NNODES > COLL_MAX_MSG )); then
      dropped+=( "$s" )
    else
      out+=( "$s" )
    fi
  done
  if (( ${#dropped[@]} )); then
    echo "    ! [SKIP] 跨机总传输量（size×nodes）超过 ${COLL_MAX_MSG}B，已剔除：${dropped[*]}（COLL_MAX_MSG=0 可放开）" >&2
  fi
  (( ${#out[@]} )) || out=( 8 65536 1048576 )
  local joined; joined=$(printf '%s,' "${out[@]}")
  echo "${joined%,}"
}

# run_probe 与 run_train 共用的 NCCL 环境变量。
# 结果放全局数组 NCCL_ENV，不用 $(...) 捕获：子 shell 里的数组赋值会全部丢失（同 BUSBW_OUT 的理由）。
# 用法：nccl_env_for <tag> [额外的 -e 键值对…]，额外项插在 TOPO_DUMP 之后、GID 之前。
nccl_env_for() {
  local tag=$1; shift
  NCCL_ENV=( -e NCCL_DEBUG="$NCCL_DBG"
             -e NCCL_TOPO_DUMP_FILE="/workspace/logs/nccl-topo-$tag.xml" "$@" )
  # GID：默认不注入，交 NCCL 逐 HCA 自选；只有双机且显式给了 NCCL_GID 才注入。
  # 写死 index 注入全部口会在未验证的口上选到错 GID → ibv_modify_qp errno 110。
  (( NNODES > 1 )) && [ -n "$NCCL_GID" ] && NCCL_ENV+=( -e NCCL_IB_GID_INDEX="$NCCL_GID" )
  # 建链超时与重试只在双机注入；单机沿用 NCCL 默认，行为不变
  if (( NNODES > 1 )); then
    NCCL_ENV+=( -e NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT" -e NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT" )
    [ "${NCCL_IB_DISABLE:-0}" = 1 ] && NCCL_ENV+=( -e NCCL_IB_DISABLE=1 )
  fi
  # 不显式指 HCA 时 NCCL 可能挑到 bond0（50G 管理口）甚至 socket，
  # 双机表现就是「卡住 30 分钟后超时」。
  [ -n "$NCCL_HCA" ] && NCCL_ENV+=( -e NCCL_IB_HCA="$NCCL_HCA" )
  [ -n "$NCCL_IF"  ] && NCCL_ENV+=( -e NCCL_SOCKET_IFNAME="$NCCL_IF" )
  return 0
}

# ---- 顶层常量（原脚本散落在函数之间，值不变）----
ARG_FLAGS=(multi-latent-attention num-experts moe-ffn-hidden-size moe-router-topk
           moe-router-enable-expert-bias moe-shared-expert-intermediate-size
           expert-model-parallel-size moe-token-dispatcher-type moe-aux-loss-coeff
           mtp-num-layers fp8-recipe no-create-attention-mask-in-dataloader
           enable-experimental)
ARG_CRIT=(num-experts multi-latent-attention expert-model-parallel-size moe-token-dispatcher-type)
