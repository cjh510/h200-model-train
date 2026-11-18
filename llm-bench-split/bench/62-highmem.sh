# ============================================================================
# 62-highmem —— 显存高占用长稳（2026-11 改版）
#
# 改版原因：旧版先自己标定 mbs（短程试跑 → 倍增 → 插值 → seq 细化）去凑 75~92% 的占用带，
# 结果是为了压显存选了一个不是最优的配置，跑出来的吞吐没有参考价值。
# 现在改为：**直接沿用 --phase mfu 里 MFU† 最高的那一档配置**，跑满 HIGHMEM_MINUTES，
# 回答的是「实际要用的最优配置，连续跑一小时显存稳不稳、有没有泄漏」。
#
# 判定（不再有 75~92% 的通过线 —— 档位不是为压显存选的，硬套那条线只会天天误报）：
#   OOM                    → OOM_BOUNDARY（预期边界，退出码 0）
#   跑不满 / 失败           → PARTIAL / FAILED
#   < 10 分钟              → SMOKE（冒烟：不判定、不写完成哨兵）
#   任一卡显存增长 ≥ 5%     → WARN_LEAK（疑似泄漏）
#   峰值在前 20% 且末尾仍贴峰 → WARN_FRAG（疑似碎片）
#   其余                    → OK
#
# 前置：本 run 必须已有 MFU 阶段的 mfu.jsonl；没有就报错并提示先跑 --phase mfu。
# ============================================================================

# ------------------------------------------------------------ 显存采样
mem_sample() {                           # mem_sample <csv>：每卡一行 ts,gpu,used_mib
  local csv=$1 ts gpu used line
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  ts=$(date -Is)
  [ -f "$csv" ] || echo "ts,gpu,used_mib" > "$csv"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null \
    | while IFS= read -r line; do
        gpu=${line%%,*}; used=${line#*,}
        gpu=${gpu//[!0-9]/}; used=${used//[!0-9]/}
        [ -n "$gpu" ] && [ -n "$used" ] || continue
        printf '%s,%s,%s\n' "$ts" "$gpu" "$used" >> "$csv"
      done
}

mem_sampler_start() {                    # mem_sampler_start <csv>
  MEM_SAMPLER_PID=""
  ( while :; do mem_sample "$1"; sleep "$MEM_SAMPLE_SECS"; done ) &
  MEM_SAMPLER_PID=$!
  echo "    · 显存采样 → $1（每 ${MEM_SAMPLE_SECS}s，pid $MEM_SAMPLER_PID）"
}

mem_sampler_stop() {
  [ -n "${MEM_SAMPLER_PID:-}" ] || return 0
  kill "$MEM_SAMPLER_PID" 2>/dev/null
  wait "$MEM_SAMPLER_PID" 2>/dev/null || true
  MEM_SAMPLER_PID=""
}

mem_peak_mib() {                         # mem_peak_mib <csv> → 所有卡里的最高占用（MiB；无数据 0）
  local csv=$1
  [ -s "$csv" ] || { echo 0; return 0; }
  awk -F, 'NR>1 && $3 ~ /^[0-9]+$/ { if ($3+0 > m) m = $3+0 } END { printf "%d\n", m+0 }' "$csv"
}

mem_gpu_max_mib() {                      # 当前 nvidia-smi 各卡 memory.used 的最高值（MiB）
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return 0; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{ if ($1+0 > m) m = $1+0 } END { printf "%d\n", m+0 }'
}

# 容器退出后显存不会立刻回落到基线（CUDA context 释放 + 页缓存），直接开下一档会把
# 上一档的残留算进本档峰值 → 峰值偏大。等到占用 ≤5% 或超时；超时只 WARN。
mem_wait_idle() {                        # mem_wait_idle [最多等几秒]
  local max=${1:-30} t0 used lim
  lim=$(( VRAM_MIB / 20 ))
  t0=$(date +%s)
  while :; do
    used=$(mem_gpu_max_mib)
    [ "$used" -le "$lim" ] && return 0
    if (( $(date +%s) - t0 >= max )); then
      echo "   ! [WARN] 等待显存回落超时（当前 ${used} MiB > 阈值 ${lim} MiB）；本档峰值可能偏高"
      return 1
    fi
    sleep 2
  done
}

# ------------------------------------------------------------ 档位来源：MFU 峰值
# 从本 run 的 mfu.jsonl 里取「同节点数、status=OK、MFU† 最高」的一档，原样复用它的
# model / TP / PP / mbs / seq / GBS。退出码：0 成功；3 没有 mfu.jsonl；4 没有可用档。
highmem_pick_from_mfu() {                # → "arr|model|tp|pp|mbs|seq|gbs|mfu_pct"
  python3 - "$LC_RUN_DIR" "$NNODES" <<'PYEOF'
import glob, json, os, sys
run_dir, nodes = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(run_dir, "*", "mfu.jsonl")))
if not files:
    sys.exit(3)
rows = []
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("status") != "OK":
                    continue
                if int(r.get("nodes") or 0) != nodes:
                    continue
                rows.append(r)
    except OSError:
        continue
if not rows:
    sys.exit(4)
b = max(rows, key=lambda r: r.get("mfu_megatron") or 0)
print("|".join(str(b[k]) for k in ("arr", "model", "tp", "pp", "mbs", "seq", "gbs", "mfu_megatron")))
PYEOF
}

# ------------------------------------------------------------ 记录与判定
# status 传 AUTO 时由这里按「增长 / 碎片」定状态，并把最终状态打到 stdout 供调用方使用。
highmem_record() {                       # tag model tp pp mbs seq gbs iters step_s jitter status src_mfu
  python3 - "$LOGDIR/memused-w${WORLD}_highmem.csv" "$LOGDIR/highmem.jsonl" \
           "$WORLD" "$NNODES" "$@" "$VRAM_MIB" "$HIGHMEM_GROWTH_MAX" "$HIGHMEM_FRAG_DIFF" <<'PYEOF'
import csv, json, sys, datetime

(mpath, out, world, nodes, tag, model, tp, pp, mbs, seq, gbs,
 iters, step_s, jitter, status, src_mfu, peak_cap, grow_max, frag_diff) = sys.argv[1:21]
world, nodes, tp, pp, mbs, seq, gbs = map(int, (world, nodes, tp, pp, mbs, seq, gbs))
iters, peak_cap = int(iters), int(peak_cap)
grow_max, frag_diff = float(grow_max), float(frag_diff)

per = {}
try:
    with open(mpath, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                g, u = int(row.get("gpu")), int(row.get("used_mib"))
            except (TypeError, ValueError):
                continue
            per.setdefault(g, []).append(u)
except FileNotFoundError:
    pass

gpus = []
for g in sorted(per):
    v = per[g]
    mx = max(v)
    thr = 0.2 * mx
    idx = [i for i, x in enumerate(v) if x >= thr]
    first, last = (idx[0], idx[-1]) if idx else (0, len(v) - 1)
    start, end = v[first], v[last]
    growth = round((end - start) / start * 100, 1) if start > 0 else None
    gpus.append({"index": g, "mem_start": start, "mem_max": mx, "mem_end": end,
                 "growth_pct": growth, "peak_pct": round(mx / peak_cap * 100, 1),
                 "mem_max_at_pct": round(v.index(mx) / max(len(v) - 1, 1) * 100, 1),
                 "end_over_peak_pct": round(end / mx * 100, 1) if mx else None,
                 "samples": len(v)})

peak = max((x["mem_max"] for x in gpus), default=0)
growth = max((x["growth_pct"] for x in gpus if x["growth_pct"] is not None), default=None)
# 峰值出现在前 20% 且末尾仍贴着峰值 → 全程贴边，疑似碎片
frag = any(x["mem_max_at_pct"] <= 20 and (x["end_over_peak_pct"] or 0) >= (100 - frag_diff)
           for x in gpus)

final = status
if status == "AUTO":
    if growth is not None and growth >= grow_max:
        final = "WARN_LEAK"
    elif frag:
        final = "WARN_FRAG"
    else:
        final = "OK"

dp = world // (tp * pp) if tp * pp else 0
rec = {"world": world, "nodes": nodes, "tag": tag, "model": model.upper(),
       "tp": tp, "pp": pp, "dp": dp, "mbs": mbs, "seq": seq, "gbs": gbs,
       "accum": gbs // (mbs * dp) if mbs and dp else 0,
       "tokens_per_gpu": gbs * seq // world if world else 0,
       "iters": iters, "step_s": float(step_s or 0), "jitter": float(jitter or 0),
       "peak_mib": peak, "peak_pct": round(peak / peak_cap * 100, 1),
       "growth_pct_max": growth, "frag_suspect": frag,
       "source": "mfu", "source_mfu_pct": None if src_mfu in ("", None) else float(src_mfu),
       "gpu": gpus, "status": final,
       "ts": datetime.datetime.now().isoformat(timespec="seconds")}
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(final)
PYEOF
}

# ------------------------------------------------------------ 主流程
soak_highmem() {
  local MINS="${HIGHMEM_MINUTES:-60}"
  SOAK_TARGET_MINS="$MINS"; SOAK_LABEL=highmem
  # 冒烟（<10 分钟）不判定：一小时的泄漏/碎片趋势在 2 分钟里看不出来
  local SMOKE=0
  (( MINS < 10 )) && SMOKE=1

  if [ "$FORCE" != 1 ] && [ -f "$LOGDIR/.ok_highmem" ]; then
    echo "→ 跳过显存高占用长稳（已完成哨兵 .ok_highmem；FORCE=1 可重跑）"; return 0
  fi

  # ---- 档位：MFU 峰值档 ----
  local PICK rc
  PICK=$(highmem_pick_from_mfu); rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! 显存高占用长稳需要 MFU 阶段的峰值档，但没有可用的 mfu.jsonl"
    if [ "$rc" = 3 ]; then
      echo "   本 run 还没有 MFU 数据：先跑  bash 04_bench.sh --phase mfu"
    else
      echo "   mfu.jsonl 里没有 status=OK 且 nodes=$NNODES 的档：先重跑 --phase mfu"
    fi
    echo "   （旧版是自己标定 mbs 去凑占用率；现在改为直接复用最优配置，必须先有 MFU 结果）"
    return 1
  fi
  local ARR NAME TP PP MBS SEQ GBS MFU_PCT
  IFS='|' read -r ARR NAME TP PP MBS SEQ GBS MFU_PCT <<< "$PICK"
  if (( TP * PP > WORLD )); then
    echo "!! MFU 峰值档是 TP${TP}×PP${PP}，当前 world=$WORLD 放不下（换机器了？先重跑 --phase mfu）"
    return 1
  fi
  local DP=$(( WORLD / TP / PP ))
  if (( MBS * DP == 0 || GBS % (MBS * DP) != 0 )); then
    echo "!! MFU 记录的 GBS=${GBS} 不能被 mbs${MBS}×DP${DP} 整除，mfu.jsonl 可能被改过"
    return 1
  fi

  local TAG="HIGHMEM_${NAME}_tp${TP}_pp${PP}_mbs${MBS}_s${SEQ}_bf16"
  local LOG="$LOGDIR/w${WORLD}_${TAG}.log"
  rm -f "$LOGDIR/memused-w${WORLD}_highmem.csv"

  echo "########## 显存高占用长稳 $MINS 分钟 ##########"
  echo "##########   档位来源=MFU 峰值档（MFU† ${MFU_PCT}%）：$NAME TP${TP}×PP${PP}×DP${DP}"
  echo "##########   mbs=$MBS seq=$SEQ GBS=$GBS → 不是为压显存选的，占用率只记录不判定"
  SENTINEL_KEY=highmem
  DMON_TAG="highmem-w${WORLD}_${NAME}_mbs${MBS}_s${SEQ}"
  SOAK_PREFIX="highmem-"
  local H="$HIGHMEM_HEALTH" HAVE_H=0
  if [ "$H" = 1 ]; then
    health_snap "$LOGDIR/highmem-health-before.json"
    HAVE_H=1
  fi
  TELEMETRY=1
  STEPS=100000
  NO_RECOMPUTE=1                 # 不开重算：开重算等于把激活压回去，峰值会失真
  mem_wait_idle 30
  mem_sampler_start "$LOGDIR/memused-w${WORLD}_highmem.csv"
  local REQUIRE="$SOAK_MIN_ITERS"
  (( SMOKE == 1 )) && REQUIRE=1
  ALLOW_OOM=1 OOM_AS_BOUNDARY=1 SOAK_REQUIRE_ITERS="$REQUIRE" \
    run_train "$TAG" "$ARR" "$TP" "$PP" "$MBS" "$SEQ" "$GBS" --exit-duration-in-mins="$MINS"
  local run_rc=$?
  mem_sampler_stop
  NO_RECOMPUTE=0
  if [ "$HAVE_H" = 1 ]; then
    health_snap "$LOGDIR/highmem-health-after.json"
    health_delta_json highmem-
  fi

  # ---- 结论 ----
  local njit ss jit st ec="" final
  njit=$(grep -c "elapsed time per iteration" "$LOG" 2>/dev/null); njit=${njit:-0}
  ss=$(step_seconds "$LOG"); jit=$(step_jitter "$LOG")
  if [ -f "$LOGDIR/.oom_w${WORLD}_${TAG}" ]; then
    st=OOM_BOUNDARY; ec=OOM; run_rc=0
  elif [ "$run_rc" -ne 0 ] || [ "$njit" -eq 0 ]; then
    st=FAILED; ec=$(err_class "$LOG"); run_rc=1
  elif [ "$njit" -lt "$REQUIRE" ]; then
    st=PARTIAL; run_rc=1
  elif (( SMOKE == 1 )); then
    st=SMOKE
  else
    st=AUTO                        # 交给 highmem_record 按增长/碎片判定
  fi

  final=$(highmem_record "$TAG" "$NAME" "$TP" "$PP" "$MBS" "$SEQ" "$GBS" \
                          "$njit" "$ss" "$jit" "$st" "$MFU_PCT")
  final=${final:-$st}
  local PEAK PCT
  PEAK=$(mem_peak_mib "$LOGDIR/memused-w${WORLD}_highmem.csv")
  PCT=$(awk -v p="$PEAK" -v v="$VRAM_MIB" 'BEGIN{printf "%.1f", (v>0 ? p/v*100 : 0)}')

  case "$final" in
    OK)   echo "   ✓ 长稳完成：峰值 ${PEAK} MiB = ${PCT}%，无泄漏/碎片迹象" ;;
    WARN_LEAK) echo "   ! [WARN] 疑似泄漏：显存增长 ≥ ${HIGHMEM_GROWTH_MAX}%（峰值 ${PEAK} MiB / ${PCT}%）"
                echo "     对照 memused-w${WORLD}_highmem.csv 看是否单调上升；这条不能忽略" ;;
    WARN_FRAG) echo "   ! [WARN] 疑似碎片：峰值在前 20% 且末尾仍贴峰（峰值 ${PEAK} MiB / ${PCT}%）"
                echo "     只影响长时间稳定性判断，不代表泄漏" ;;
    SMOKE) echo "   · 冒烟 ${MINS} 分钟：峰值 ${PEAK} MiB = ${PCT}%（不判定、不写哨兵）"
           rm -f "$LOGDIR/.ok_highmem" ;;
    OOM_BOUNDARY) echo "   ○ OOM_BOUNDARY：MFU 峰值档在长稳里 OOM（峰值 ${PEAK} MiB），不计失败"
                  echo "     MFU 是 40 步短跑；长稳 60 分钟的碎片更高，这本身是个有价值的结论" ;;
    *)     echo "   ✗ $final（峰值 ${PEAK} MiB / ${PCT}%）" ;;
  esac
  [ "$final" = OK ] && : > "$LOGDIR/.ok_highmem"
  echo "    · 记录 → highmem.jsonl；显存曲线 → memused-w${WORLD}_highmem.csv"
  echo "  遥测：$LOGDIR/dmon-${DMON_TAG}.csv；硬件增量：highmem-health-delta.json"
  SENTINEL_KEY=""; DMON_TAG=""; SOAK_PREFIX=""
  sync_logs
  [ "$run_rc" -eq 0 ]
}
