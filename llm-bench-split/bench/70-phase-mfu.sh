# ============================================================================
# 70-phase-mfu —— 峰值 MFU：A 粗扫 / B 细扫 / C seq / D 复测 + 时钟取证 + 峰值汇总。
# ============================================================================

mfu_params() {                            # 模型名 → 参数量（算 6N-MFU 用）
  case "$1" in
    8B)   echo 8.03e9 ;;
    70B)  echo 70.55e9 ;;
    110B) echo 1.10e11 ;;
    *)    echo 0 ;;
  esac
}

# 时钟/降频取证：dmon 只采 pwr/sm%，没有 clocks 和 throttle。
# 峰值 MFU 必须能证明「全程 boost 时钟、没有降频」，否则测到的只是当时的状态。
mfu_clocks_start() {                      # mfu_clocks_start <csv>
  MFU_CLK_PID=""
  [ "$MFU_CLOCKS" = 1 ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || { echo "   ! [WARN] 无 nvidia-smi，跳过时钟取证" >&2; return 0; }
  : > "$1"
  ( while :; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r l; do printf '%s,%s\n' "$ts" "$l"; done
      sleep "$MFU_CLOCKS_SECS"
    done ) >> "$1" &
  MFU_CLK_PID=$!
}

mfu_clocks_stop() {
  [ -n "${MFU_CLK_PID:-}" ] || return 0
  kill "$MFU_CLK_PID" 2>/dev/null
  wait "$MFU_CLK_PID" 2>/dev/null || true
  MFU_CLK_PID=""
}

mfu_clocks_agg() {                        # mfu_clocks_agg <csv> → "sm_avg sm_min pwr_avg thr_any"
  local csv=$1
  [ -s "$csv" ] || { echo "- - - -"; return 0; }
  awk -F',' '
    { gsub(/ /,"",$0); n=split($0,a,","); if (n < 6) next;
      if (a[2] ~ /^[0-9]+$/ && a[2]+0 > 0) { sm+=a[2]; if (mn==0 || a[2]+0<mn) mn=a[2]+0; ns++ }
      if (a[4] ~ /^[0-9.]+$/) { pw+=a[4]; np++ }
      if (a[6] !~ /^(0x0+)?$/ && a[6] != "N/A") thr=1 }
    END { printf "%s %s %s %s\n",
          (ns?int(sm/ns):"-"), (ns?mn:"-"), (np?int(pw/np):"-"), (thr?"1":"0") }' "$csv"
}

mfu_record() {                            # stage arr model tp pp mbs seq gbs step_s jitter iters tf mfu6n clock_csv status repeat
  python3 - "$LOGDIR/mfu.jsonl" "$WORLD" "$NNODES" "$@" <<'PYEOF'
import json, sys, datetime
(out, world, nodes, stage, arr, model, tp, pp, mbs, seq, gbs,
 ss, jit, iters, tf, mfu6n, clkcsv, st, rep) = sys.argv[1:20]
tp, pp, mbs, seq, gbs = int(tp), int(pp), int(mbs), int(seq), int(gbs)
dp = int(world) // (tp * pp)
clk = {"sm_avg": None, "sm_min": None, "pwr_avg": None, "throttle": None}
try:
    parts = open(clkcsv, encoding="utf-8").read().split()
    if len(parts) == 4:
        clk = {"sm_avg": None if parts[0] == "-" else int(parts[0]),
               "sm_min": None if parts[1] == "-" else int(parts[1]),
               "pwr_avg": None if parts[2] == "-" else int(parts[2]),
               "throttle": None if parts[3] == "-" else int(parts[3])}
except OSError:
    pass
peak = 989.5
rec = {"world": int(world), "nodes": int(nodes), "stage": stage, "model": model,
       "arr": arr, "tp": tp, "pp": pp, "dp": dp, "mbs": mbs, "seq": seq, "gbs": gbs,
       "accum": gbs // (mbs * dp) if mbs * dp else 0,
       "tokens_per_gpu": gbs * seq // int(world),
       "step_s": float(ss or 0), "jitter": float(jit or 0), "iters": int(iters or 0),
       "tflops_megatron": float(tf or 0), "mfu_megatron": round(float(tf or 0) / peak * 100, 1),
       "mfu_6n": round(float(mfu6n or 0), 1),
       "clk_sm_avg": clk["sm_avg"], "clk_sm_min": clk["sm_min"],
       "pwr_avg": clk["pwr_avg"], "throttle": clk["throttle"],
       "repeat": int(rep or 1), "status": st, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PYEOF
}

mfu_tag() {                               # stage model tp pp mbs seq gbs
  echo "MFU_${1}_${2}_tp${3}_pp${4}_mbs${5}_s${6}_gbs${7}"
}

mfu_run_one() {                           # spec="stage|arr|model|tp|pp|mbs|seq|gbs" [repeat]
  local spec=$1 rep=${2:-1}
  local stage arr model tp pp mbs seq gbs
  IFS='|' read -r stage arr model tp pp mbs seq gbs <<< "$spec"
  local tag; tag=$(mfu_tag "$stage" "$model" "$tp" "$pp" "$mbs" "$seq" "$gbs")
  [ "$rep" -gt 1 ] && tag="${tag}_r${rep}"
  local log="$LOGDIR/w${WORLD}_${tag}.log"
  local clk="$LOGDIR/clocks-w${WORLD}_${tag}.csv"
  local ss_save=$STEPS
  STEPS=$MFU_STEPS
  mfu_clocks_start "$clk"
  NO_RESULT_REC=1
  ALLOW_OOM=1 OOM_AS_BOUNDARY=1 SOAK_REQUIRE_ITERS="" \
    run_train "$tag" "$arr" "$tp" "$pp" "$mbs" "$seq" "$gbs"
  local rc=$?
  NO_RESULT_REC=""
  mfu_clocks_stop
  STEPS=$ss_save

  local njit ss jit tf m6 st note=""
  njit=$(grep -c "elapsed time per iteration" "$log" 2>/dev/null); njit=${njit:-0}
  ss=$(step_seconds "$log")
  jit=$(step_jitter "$log")
  tf=$(grep -oE 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  tf=${tf:-0}
  if [ -f "$LOGDIR/.oom_w${WORLD}_${tag}" ]; then
    st=OOM_BOUNDARY
  elif [ "$rc" -ne 0 ] || [ "$njit" -eq 0 ]; then
    st=FAILED; note=$(err_class "$log")
  elif [ "$njit" -lt "$MFU_STEPS" ]; then
    st=PARTIAL; note="只跑了 ${njit}/${MFU_STEPS} 步"
  else
    st=OK
  fi
  m6=$(awk -v n="$(mfu_params "$model")" -v g="$gbs" -v s="$seq" -v t="$ss" -v w="$WORLD" \
        'BEGIN{ if (t>0) printf "%.1f", 6*n*g*s/t/w/989.4e12*100; else print 0 }')
  mfu_record "$stage" "$arr" "$model" "$tp" "$pp" "$mbs" "$seq" "$gbs" \
             "$ss" "$jit" "$njit" "$tf" "$m6" "$clk" "$st" "$rep"
  local cagg ca cm cp ct extra=""
  cagg=$(mfu_clocks_agg "$clk")
  set -- $cagg; ca=${1:--}; cm=${2:--}; cp=${3:--}; ct=${4:--}
  [ "$ct" = 1 ] && extra="  [WARN throttle!=0]"
  if [ "$st" = OK ]; then
    echo "   OK $(printf '%-44s' "$tag") MFU† $(awk -v t="$tf" 'BEGIN{printf "%.1f", t/989.5*100}')%  ${njit} 步  SM ${ca}/${cm}MHz  ${cp}W${extra}"
  else
    echo "   -- $(printf '%-44s' "$tag") $st${note:+（$note）}"
  fi
  return 0
}

mfu_best() {                              # <stage|any> <model> → "arr|model|tp|pp|mbs|seq|gbs"
  python3 - "$LOGDIR/mfu.jsonl" "$1" "$2" <<'PYEOF'
import json, sys
path, stage, model = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
try:
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("status") != "OK" or not r.get("step_s") or r.get("model") != model:
            continue
        if stage != "any" and r.get("stage") != stage:
            continue
        rows.append(r)
except OSError:
    pass
if rows:
    b = max(rows, key=lambda r: r["mfu_megatron"])
    print("|".join(str(b[k]) for k in ("arr", "model", "tp", "pp", "mbs", "seq", "gbs")))
PYEOF
}

mfu_top3() {                              # <model> → 最多 3 行 "arr|model|tp|pp|mbs|seq|gbs"（按 MFU† 降序、去重）
  python3 - "$LOGDIR/mfu.jsonl" "$1" <<'PYEOF'
import json, sys
path, model = sys.argv[1], sys.argv[2]
rows, seen = [], set()
try:
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("status") != "OK" or not r.get("step_s") or r.get("model") != model:
            continue
        rows.append(r)
except OSError:
    pass
for r in sorted(rows, key=lambda x: -x["mfu_megatron"]):
    k = (r["arr"], r["model"], r["tp"], r["pp"], r["mbs"], r["seq"], r["gbs"])
    if k in seen:
        continue
    seen.add(k)
    print("|".join(str(x) for x in k))
    if len(seen) >= 3:
        break
PYEOF
}

mfu_models_list() {
  if [ "$MFU_MODELS" = auto ]; then
    (( NNODES > 1 )) && echo "8B 70B" || echo "8B"
  else
    echo "${MFU_MODELS//,/ }"
  fi
}

mfu_print_plan() {
  local spec stage arr model tp pp mbs seq gbs dp
  for spec in "${MFU_SPECS[@]}"; do
    IFS='|' read -r stage arr model tp pp mbs seq gbs <<< "$spec"
    dp=$(( WORLD / tp / pp ))
    printf '    %s  %-4s TP%-2s PP%-2s DP%-2s mbs%-2s seq%-6s GBS%-4s accum=%s\n' \
      "$stage" "$model" "$tp" "$pp" "$dp" "$mbs" "$seq" "$gbs" "$(( gbs / (mbs * dp) ))"
  done
}

mfu_build_plan() {                        # 生成 A 段 MFU_SPECS
  MFU_SPECS=()
  local tp gbs name arr dp mbs
  for name in $(mfu_models_list); do
    case "$name" in
      8B)  arr=LLAMA_8B ;;
      70B) arr=LLAMA_70B ;;
      *)   echo "   ! 跳过未知模型 $name"; continue ;;
    esac
    if [ "$name" = 70B ]; then
      # 70B 只有 TP8×PP2 一条路（world=16）：扫 mbs × GBS，看 PP 气泡与 accum 的平衡
      dp=$(( WORLD / 16 ))
      (( dp >= 1 )) || continue
      for mbs in 1 2; do
        for gbs in 8 16 32 64; do
          (( gbs % (mbs * dp) == 0 )) && MFU_SPECS+=( "A|$arr|$name|8|2|$mbs|4096|$gbs" )
        done
      done
    elif (( NNODES > 1 )); then
      for tp in 1 2 4 8; do
        dp=$(( WORLD / tp ))
        for gbs in ${MFU_GBS_LIST//,/ }; do
          (( gbs % (2 * dp) == 0 )) && MFU_SPECS+=( "A|$arr|$name|$tp|1|2|4096|$gbs" )
        done
      done
      dp=$(( WORLD / 16 ))
      (( dp >= 1 && 64 % (2 * dp) == 0 )) && MFU_SPECS+=( "A|$arr|$name|8|2|2|4096|64" )
    else
      for tp in ${MFU_TP_LIST//,/ }; do
        (( tp <= WORLD )) || continue
        dp=$(( WORLD / tp ))
        for gbs in ${MFU_GBS_LIST//,/ }; do
          (( gbs % (2 * dp) == 0 )) && MFU_SPECS+=( "A|$arr|$name|$tp|1|2|4096|$gbs" )
        done
      done
    fi
  done
  if (( ${#MFU_SPECS[@]} > MFU_MAX_RUNS )); then
    echo "   ! 计划 ${#MFU_SPECS[@]} 档超过 MFU_MAX_RUNS=$MFU_MAX_RUNS，只跑前 $MFU_MAX_RUNS 档"
    MFU_SPECS=( "${MFU_SPECS[@]:0:MFU_MAX_RUNS}" )
  fi
}

mfu_stage_bcd() {                         # <model>：细扫 + seq 扫描 + 复测
  local name=$1 best arr model tp pp mbs seq gbs dp bseq bgbs spec acc rep k top
  best=$(mfu_best A "$name"); best=${best%$'\r'}
  if [ -z "$best" ]; then
    echo "   ! $name：A 段没有任何一档跑通，跳过 B/C/D"
    return 0
  fi
  IFS='|' read -r arr model tp pp mbs seq gbs <<< "$best"
  dp=$(( WORLD / tp / pp ))
  bseq=$seq
  bgbs=$gbs
  echo "  A 段最优：$name TP${tp}×PP${pp}×DP${dp} mbs${mbs} seq${seq} GBS${gbs}"

  # ---- B 段：固定拓扑，扫 mbs × accum（GBS = mbs·DP·accum）----
  MFU_SPECS=()
  for mbs in ${MFU_MBS_LIST//,/ }; do
    for acc in ${MFU_ACCUM_LIST//,/ }; do
      gbs=$(( mbs * dp * acc ))
      (( gbs <= MFU_MAX_GBS )) || continue
      MFU_SPECS+=( "B|$arr|$name|$tp|$pp|$mbs|$bseq|$gbs" )
    done
  done
  echo "  B 段（TP${tp}×PP${pp}，mbs × accum）：${#MFU_SPECS[@]} 档"
  mfu_print_plan
  if [ "${MFU_PLAN_ONLY:-0}" != 1 ]; then
    for spec in "${MFU_SPECS[@]}"; do mfu_run_one "$spec"; done
  fi

  # ---- C 段：最优 mbs/accum 上扫 seq（attention 占比 + 是否触发重算）----
  best=$(mfu_best B "$name"); best=${best%$'\r'}
  if [ -n "$best" ]; then
    IFS='|' read -r arr model tp pp mbs seq gbs <<< "$best"   # mfu_best 不含 stage 前缀
    bgbs=$gbs
  fi
  MFU_SPECS=()
  for seq in ${MFU_SEQ_LIST//,/ }; do
    [ "$seq" = "$bseq" ] && continue          # 该 seq 已在 A/B 段跑过
    MFU_SPECS+=( "C|$arr|$name|$tp|$pp|$mbs|$seq|$bgbs" )
  done
  echo "  C 段（TP${tp}×PP${pp} mbs${mbs} GBS${bgbs}，seq 扫描）：${#MFU_SPECS[@]} 档"
  mfu_print_plan
  if [ "${MFU_PLAN_ONLY:-0}" != 1 ]; then
    for spec in "${MFU_SPECS[@]}"; do mfu_run_one "$spec"; done
  fi

  # ---- D 段：前 3 名各复测 MFU_REPEAT 次（第 1 次已跑过，这里补 repeat-1 次）----
  echo "  D 段（前 3 名 × 复测，CV≤${MFU_CV_MAX}% 才算有效）"
  k=0
  for top in $(mfu_top3 "$name"); do
    top=${top%$'\r'}
    k=$((k + 1))
    IFS='|' read -r arr model tp pp mbs seq gbs <<< "$top"
    echo "     #$k TP${tp}×PP${pp} mbs${mbs} seq${seq} GBS${gbs}"
    for (( rep = 2; rep <= MFU_REPEAT; rep++ )); do
      [ "${MFU_PLAN_ONLY:-0}" = 1 ] && continue
      mfu_run_one "R|$top" "$rep"
    done
  done
}

mfu_peak_summary() {                      # 汇总：每个 (model, nodes) 一个峰值 + CV + 天花板差距
  python3 - "$LOGDIR/mfu.jsonl" "$LOGDIR/mfu-peak.json" "$MFU_CV_MAX" <<'PYEOF'
import json, sys, statistics, datetime
path, out, cvmax = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
try:
    for ln in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
except OSError:
    rows = []
groups = {}
for r in rows:
    if r.get("status") != "OK":
        continue
    key = (r["model"], r["nodes"], r["tp"], r["pp"], r["mbs"], r["seq"], r["gbs"])
    groups.setdefault(key, []).append(r)

per_cfg = []
for key, rs in groups.items():
    vals = [x["mfu_megatron"] for x in rs]
    med = statistics.median(vals)
    cv = (statistics.pstdev(vals) / statistics.mean(vals) * 100) if len(vals) > 1 and statistics.mean(vals) else None
    thr = any(x.get("throttle") == 1 for x in rs)
    per_cfg.append({"cfg": dict(zip(("model", "nodes", "tp", "pp", "mbs", "seq", "gbs"), key)),
                    "n": len(vals), "mfu_megatron": round(med, 1),
                    "mfu_6n": round(statistics.median([x["mfu_6n"] for x in rs]), 1),
                    "cv_pct": None if cv is None else round(cv, 2),
                    "throttled": thr,
                    "step_s": round(statistics.median([x["step_s"] for x in rs]), 3)})

peak, rejected = {}, []
for c in per_cfg:
    k = (c["cfg"]["model"], c["cfg"]["nodes"])
    ok = (not c["throttled"]) and (c["cv_pct"] is None or c["cv_pct"] <= cvmax)
    if ok:
        cur = peak.get(k)
        if cur is None or c["mfu_megatron"] > cur["mfu_megatron"]:
            peak[k] = {**c, "certified": c["n"] > 1 and c["cv_pct"] is not None}
    else:
        rejected.append(c)

gemm = 690.0
doc = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "definition": "MFU† = Megatron 自报 throughput per GPU / 989.5 TFLOPS（dense bf16 理论峰值，含 attention）；"
                     "MFU(6N) 只算线性层，是保守下界",
       "ceiling": {"gemm_bf16_tflops": gemm, "gemm_pct": round(gemm / 989.5 * 100, 1)},
       "cv_max_pct": cvmax,
       "peak": {f"{k[0]}@{k[1]}node": v for k, v in peak.items()},
       "rejected": rejected,
       "configs": sorted(per_cfg, key=lambda x: -x["mfu_megatron"])}
for v in doc["peak"].values():
    v["pct_of_gemm_ceiling"] = round(v["mfu_megatron"] / doc["ceiling"]["gemm_pct"] * 100, 1)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=2)

print("\n=== MFU† 峰值（分母 989.5 TFLOPS；GEMM 天花板 %.1f%%）===" % doc["ceiling"]["gemm_pct"])
if not doc["peak"]:
    print("  没有任何一档同时满足 status=OK / 未降频 / CV<=%.1f%%" % cvmax)
for _key, v in doc["peak"].items():
    c = v["cfg"]
    cv = "-" if v["cv_pct"] is None else "%.2f%%" % v["cv_pct"]
    print("  %-4s@%d节点  MFU† %5.1f%%  (6N %5.1f%%)  CV %-6s TP%s×PP%s  mbs%s seq%s GBS%s  -> 天花板的 %.0f%%"
          % (c["model"], c["nodes"], v["mfu_megatron"], v["mfu_6n"], cv,
             c["tp"], c["pp"], c["mbs"], c["seq"], c["gbs"], v["pct_of_gemm_ceiling"]))
    print("        复测 %d 次  %s" % (v["n"], "已验证" if v["certified"] else "单次（未复测，标 UNCERTIFIED）"))
for c in rejected:
    print("  ! %s@%s节点 TP%s×PP%s mbs%s seq%s GBS%s 被排除（throttle=%s cv=%s）"
          % (c["cfg"]["model"], c["cfg"]["nodes"], c["cfg"]["tp"], c["cfg"]["pp"], c["cfg"]["mbs"],
             c["cfg"]["seq"], c["cfg"]["gbs"], c["throttled"], c["cv_pct"]))
print("\n  明细 -> %s" % out)
PYEOF
}

mfu_phase() {                             # --phase mfu 入口
  echo "########## MFU 峰值测量（目标：给出单机/双机的峰值，不是某一档的 MFU）##########"
  echo "  口径：MFU† = Megatron 自报 throughput ÷ 989.5 TFLOPS（含 attention）；MFU(6N) 为保守下界"
  echo "  采样：每档 ${MFU_STEPS} 步、复测 ${MFU_REPEAT} 次、CV≤${MFU_CV_MAX}%、时钟取证 $( [ "$MFU_CLOCKS" = 1 ] && echo 开 || echo 关 )"
  echo "  天花板：04-hw 的 bf16 GEMM 实测 690 TFLOP/s = 理论 69.7%，训练峰值不可能超过它"
  : > "$LOGDIR/mfu.jsonl"

  mfu_build_plan
  echo "  A 段（并行度 × GBS，mbs=2 seq=4096）：${#MFU_SPECS[@]} 档"
  mfu_print_plan
  if [ "${MFU_PLAN_ONLY:-0}" = 1 ]; then
    echo "  MFU_PLAN_ONLY=1：只打印计划，不跑"
    return 0
  fi
  local i=0 spec
  for spec in "${MFU_SPECS[@]}"; do
    i=$((i + 1)); echo "  [A $i/${#MFU_SPECS[@]}]"
    mfu_run_one "$spec"
  done

  local name
  for name in $(mfu_models_list); do
    echo "########## $name 细扫 ##########"
    mfu_stage_bcd "$name"
  done

  mfu_peak_summary
}

# ---- 顶层常量（原脚本散落在函数之间，值不变）----
MFU_STEPS="${MFU_STEPS:-40}"
MFU_REPEAT="${MFU_REPEAT:-3}"
MFU_CV_MAX="${MFU_CV_MAX:-2.0}"
MFU_GBS_LIST="${MFU_GBS_LIST:-32,64,128,256}"
MFU_TP_LIST="${MFU_TP_LIST:-1,4,8}"
MFU_MBS_LIST="${MFU_MBS_LIST:-1,2,4,8}"
MFU_ACCUM_LIST="${MFU_ACCUM_LIST:-2,4}"
MFU_SEQ_LIST="${MFU_SEQ_LIST:-2048,4096,8192}"
MFU_MAX_GBS="${MFU_MAX_GBS:-512}"
MFU_MAX_RUNS="${MFU_MAX_RUNS:-48}"
MFU_PLAN_ONLY="${MFU_PLAN_ONLY:-0}"
MFU_CLOCKS_SECS="${MFU_CLOCKS_SECS:-2}"
MFU_CLOCKS="${MFU_CLOCKS:-1}"
MFU_MODELS="${MFU_MODELS:-auto}"          # auto = 单机 8B / 双机 8B,70B
MFU_CLK_PID=""
MFU_SPECS=()

phase_mfu() {          # --phase mfu
  mfu_phase
}
