#!/usr/bin/env bash
# =============================================================================
# T5 产物验证：checkpoint 文件、训练 loss 记录、可选推理冒烟
# 用法：bash scripts/05_verify.sh [27b|flash_next] [checkpoint_dir]
#       RUN_INFER=1 bash scripts/05_verify.sh 27b
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
install_err_trap
PROFILE="${1:-$MODEL_PROFILE}"
load_profile "$PROFILE"

RUN_DIR="${2:-$(ls -1d "$OUTPUT_ROOT/$MODEL_KEY" 2>/dev/null | tail -1 || true)}"
[[ -n "$RUN_DIR" ]] || die "未找到输出目录：$OUTPUT_ROOT/$MODEL_KEY，请先跑 03_continue_pt.sh"
log "输出根目录: $RUN_DIR"

# 按 step 数字排序取最大（字典序会把 checkpoint-100 排在 checkpoint-20 前面）
CKPT="$(find "$RUN_DIR" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2- || true)"
[[ -n "$CKPT" ]] || die "未找到 checkpoint-* 目录（save_steps 是否大于 train_iters？）"
log "最新 checkpoint: $CKPT"

# save_safetensors=true 时产物应为 HF 格式
SHARD_COUNT="$(find "$CKPT" -maxdepth 1 -name '*.safetensors' | wc -l | tr -d ' ')"
log "safetensors 分片数: $SHARD_COUNT"
[[ "$SHARD_COUNT" -gt 0 ]] || die "checkpoint 中没有 safetensors，权重未成功导出"
if [[ -f "$CKPT/model.safetensors.index.json" ]]; then
  log "存在 model.safetensors.index.json（HF 格式权重完整）"
else
  warn "没有 index json；单分片或 LoRA adapter 属正常，全参数权重请复核"
fi
if [[ -f "$CKPT/adapter_config.json" ]]; then
  log "检测到 LoRA adapter（adapter_config.json），需配合基座模型推理"
fi

# 训练日志中的 loss / 步时 / 吞吐（Megatron 在 last rank 打印）
# 依次尝试 T3 / T4 / T2 的日志指针，兼容多机时的 _nodeX 后缀
LOG_FILE=""
for tag in "continue_pt_$MODEL_KEY" "resume_$MODEL_KEY" "smoke_$MODEL_PROFILE"; do
  p="$LOG_ROOT/$(log_prefix "$tag")_latest.path"
  [[ -f "$p" ]] || p="$LOG_ROOT/${tag}_latest.path"
  if [[ -f "$p" ]]; then LOG_FILE="$(cat "$p")"; break; fi
done

# 对应的 dmon GPU 采样 CSV（用于把「GPU 在算」与「GPU 在等」区分开）。
# 排除 verify/preflight/smoke 这几类短任务产生的文件，取最近一次正式训练的。
DMON_CSV="${DMON_CSV:-}"
if [[ -z "$DMON_CSV" ]]; then
  DMON_CSV="$(ls -t "$LOG_ROOT"/dmon_*.csv 2>/dev/null | grep -vE 'dmon_(05_verify|00_preflight|02_smoke)' | head -1 || true)"
fi

if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
  log "训练日志: $LOG_FILE"
  log "训练 loss 记录（末 10 条）："
  # 两种日志格式都要认：
  #   Megatron 原生 : iteration N/M | ... | lm loss: 2.84 | ...
  #   swift callback: {'loss': 2.84, 'grad_norm': ..., 'iteration': '1/2', ...}
  # 只按 "loss:" 匹配会漏掉后者（引号里的 'loss':），表现为「一行都没打印」。
  if grep -q "lm loss" "$LOG_FILE" 2>/dev/null; then
    awk '/lm loss/ {print}' "$LOG_FILE" | tail -10 || true
  else
    grep -oE "'loss': [-+0-9.eE]+.*'iteration': '[0-9]+/[0-9]+'" "$LOG_FILE" 2>/dev/null \
      | sed -E "s|.*'loss': ([-+0-9.eE]+).*'iteration': '([0-9]+)/([0-9]+)'.*|  iter \2/\3    loss \1|" \
      | tail -10 || true
    if ! grep -qE "'loss':" "$LOG_FILE" 2>/dev/null; then
      warn "  日志里既没有 lm loss 也没有 'loss':，训练可能未进入迭代（检查 LOGGING_STEPS 与 TRAIN_ITERS）"
    fi
  fi

  # NaN 只在「loss 本身是 nan」或「nan iterations > 0」时才算问题；
  # Megatron 每行都带 "number of nan iterations: 0"，按 /nan/ 粗搜会永远误报。
  if awk "/lm loss: *[-+]?nan|'loss': *[-+]?nan|nan iterations: *[1-9]/ {found=1} END {exit !found}" "$LOG_FILE"; then
    warn "日志中出现 NaN，请检查 lr / dtype / 数据"
  else
    log "日志中未发现 NaN"
  fi

  # 步时与吞吐。三个来源按优先级：
  #   1) Megatron 原生 `elapsed time per iteration (ms)` —— 纯计算时间，最准
  #   2) swift print callback 的 `train_speed(s/it)` —— 秒/步，含数据加载/存盘开销
  #   3) `throughput per GPU (TFLOP/s/GPU)` —— 仅 LOG_THROUGHPUT 生效时才有
  # 冒烟/小步数场景下 Megatron 的 log_interval 可能大于总步数，一行原生行都没有，
  # 所以必须有 (2) 兜底，否则永远算不出吞吐。
  read -r STEP_AVG STEP_MIN STEP_MAX STEP_N STEP_SRC TPUT_AVG TPUT_MIN TPUT_MAX TPUT_N <<< "$(awk '
    /elapsed time per iteration \(ms\):/ {
      s = $0; sub(/.*\(ms\): */, "", s); sub(/ .*/, "", s)
      if (s ~ /^[0-9.]+$/) { n++; t += s; if (n == 1 || s+0 < mn) mn = s+0; if (s+0 > mx) mx = s+0 }
    }
    /train_speed\(s\/it\)'"'"': *[0-9.]+/ {
      s = $0; sub(/.*train_speed\(s\/it\)'"'"': */, "", s); sub(/[,}].*/, "", s)
      if (s ~ /^[0-9.]+$/) { k++; ms = s*1000; u += ms; if (k == 1 || ms < un) un = ms; if (ms > ux) ux = ms }
    }
    /throughput per GPU \(TFLOP\/s\/GPU\):/ {
      s = $0; sub(/.*GPU\): */, "", s); sub(/ .*/, "", s)
      if (s ~ /^[0-9.]+$/) { tn++; tt += s; if (tn == 1 || s+0 < tmn) tmn = s+0; if (s+0 > tmx) tmx = s+0 }
    }
    END {
      if (n > 0)      { ag=t/n; lo=mn; hi=mx; cnt=n; src=1 }
      else if (k > 0) { ag=u/k; lo=un; hi=ux; cnt=k; src=2 }
      else            { ag=0;   lo=0;  hi=0;  cnt=0; src=0 }
      printf "%.1f %.1f %.1f %d %d %.1f %.1f %.1f %d\n",
        ag, lo, hi, cnt, src,
        (tn ? tt/tn : 0), (tn ? tmn : 0), (tn ? tmx : 0), tn
    }' "$LOG_FILE")"

  if (( ${STEP_N:-0} > 0 )); then
    TOK_GLOBAL=$(awk -v g="$GLOBAL_BATCH_SIZE" -v s="$MAX_LENGTH" -v ms="$STEP_AVG" \
                 'BEGIN{printf "%.0f", g*s*1000/ms}')
    TOK_GPU=$(awk -v t="$TOK_GLOBAL" -v w="$((NPROC_PER_NODE * NNODES))" \
              'BEGIN{printf "%.0f", t/w}')
    if [[ "${STEP_SRC:-0}" == "2" ]]; then
      log "步时(ms): 均值 $STEP_AVG / 最小 $STEP_MIN / 最大 $STEP_MAX（$STEP_N 点，取自 swift train_speed，含数据加载与存盘开销）"
    else
      log "步时(ms): 均值 $STEP_AVG / 最小 $STEP_MIN / 最大 $STEP_MAX（$STEP_N 步）"
    fi
    log "           最大值通常是第 1 步的图编译开销（torch._dynamo），判断稳态看均值与最小"
    log "吞吐估算: 全局 ≈${TOK_GLOBAL} tok/s，每卡 ≈${TOK_GPU} tok/s（GBS=$GLOBAL_BATCH_SIZE × seq=$MAX_LENGTH）"
    if (( ${TPUT_N:-0} > 0 )); then
      log "Megatron TFLOP/s/GPU: 均值 $TPUT_AVG / 最大 $TPUT_MAX（$TPUT_N 点，仅作同模型横向参考）"
    else
      log "无 Megatron throughput 行：该版本 CLI 不暴露 --log_throughput，属正常；以上 tok/s 即对外口径"
    fi
  else
    warn "日志中既没有 elapsed time per iteration 也没有 train_speed，拿不到步时"
    warn "  多半是 LOGGING_STEPS 大于总步数（03 现在默认 1），或训练未进入稳态"
  fi

  # --- 耗时分解 -------------------------------------------------------------
  # train_speed 把数据加载、checkpoint 存盘都摊进「每步耗时」，直接拿它算 MFU 会偏低。
  # 三个独立来源交叉核对：
  #   ① tqdm 末尾累计时长                          → 训练循环总时长
  #   ② Saving 进度条（每次存盘 2 个：Megatron + safetensors）→ 存盘耗时（精确）
  #   ③ dmon 的 SM 利用率（按循环区间过滤）         → GPU 真正在算的比例
  # ③ 必须过滤到循环区间：模型加载阶段 GPU 大量空转，按全程算会把忙时占比压低十几个点。
  _loop_secs="$(grep -oE 'Train: +100%[^[]*\[[0-9]+:[0-9]+' "$LOG_FILE" 2>/dev/null \
                | tail -1 | sed -E 's/.*\[([0-9]+):([0-9]+)/\1 \2/' \
                | awk '{printf "%.0f", $1*60+$2}')"
  if [[ "${_loop_secs:-}" =~ ^[0-9]+$ ]] && (( _loop_secs > 0 )) && (( ${STEP_N:-0} > 0 )); then
    _save_raw="$(grep -oE 'Saving: *100%[^[]*\[[0-9]+:[0-9]+' "$LOG_FILE" 2>/dev/null \
                 | sed -E 's/.*\[([0-9]+):([0-9]+)/\1 \2/' \
                 | awk '{s+=$1*60+$2; n++} END{printf "%d %d", s, n}')"
    _save_secs=0; _save_n=0
    read -r _save_secs _save_n <<< "${_save_raw:-0 0}"

    log "=== 耗时分解（训练循环 ${_loop_secs}s / ${STEP_N} 步）==="
    log "含存盘: $(awk -v a="$_loop_secs" -v b="$STEP_N" 'BEGIN{printf "%.2f", a/b}') s/步"
    if (( _save_secs > 0 )); then
      _eff=$(( _loop_secs - _save_secs ))
      log "checkpoint 存盘: ${_save_secs}s（${_save_n} 个进度条 ≈ $(( _save_n / 2 )) 次 × Megatron+safetensors 两种格式），占 $(awk -v a="$_save_secs" -v b="$_loop_secs" 'BEGIN{printf "%.1f", a*100/b}')%"
      log "扣除存盘后: $(awk -v a="$_eff" -v b="$STEP_N" 'BEGIN{printf "%.2f", a/b}') s/步　← 算 MFU 用这个"
    else
      log "checkpoint 存盘: 未找到 Saving 进度条（本次可能未触发 SAVE_STEPS）"
    fi

    if [[ -n "$DMON_CSV" && -f "$DMON_CSV" ]]; then
      _end_raw="$(grep -oE 'End time of running main: [0-9-]+ [0-9:.]+' "$LOG_FILE" 2>/dev/null | tail -1 | sed -E 's/.*main: //')"
      _end_epoch=0
      if [[ -n "$_end_raw" ]]; then
        _end_epoch="$(date -d "$_end_raw" +%s 2>/dev/null || echo 0)"
      fi
      if (( _end_epoch > 0 )); then
        _loop_start="$(date -d "@$(( _end_epoch - _loop_secs ))" '+%Y%m%d %H:%M:%S')"
        read -r _sm _busy _idle _npts <<< "$(awk -v t0="$_loop_start" '
          NR==1 { for (i=1;i<=NF;i++) { h=$i; sub(/^#/,"",h); idx[h]=i } ; next }
          /^#/ { next }
          {
            if ($1 " " $2 < t0) next
            v = $(idx["sm"])
            if (v ~ /^[0-9]+$/) { n++; s+=v; if (v+0>=30) b++; if (v+0<5) i2++ }
          }
          END { if (n) printf "%.1f %d %d %d\n", s/n, b, i2, n; else print "0 0 0 0" }
        ' "$DMON_CSV")"
        if (( ${_npts:-0} > 0 )); then
          _per="${NPROC_PER_NODE:-8}"
          if (( _per < 1 )); then _per=8; fi
          _tt=$(( _npts / _per ))
          log "GPU 高负荷占比: $(awk -v b="$_busy" -v n="$_npts" 'BEGIN{printf "%.1f", b*100/n}')%（SM≥30%，${_tt} 个时间点，仅循环区间）"
          log "            SM 均值 ${_sm}%　高=数据加载已与计算重叠、没让 GPU 空等；低=卡在数据或通信"
          log "            采样 ${DMON_INTERVAL:-5}s 属粗估，存盘时仍有少量 copy kernel，不要与上一行相加"
        else
          warn "dmon 里没有落在训练循环区间内的采样点（$DMON_CSV）"
        fi
      else
        warn "日志缺 End time of running main，无法把 dmon 对齐到训练区间"
      fi
    else
      log "（未找到 dmon CSV，跳过 GPU 忙闲统计；DMON=0 时属正常）"
    fi
  fi
else
  warn "未找到训练日志文件"
fi

if [[ "${RUN_INFER:-0}" == "1" ]]; then
  require_cmd swift
  INFER_MODEL="$CKPT"
  if [[ -f "$CKPT/adapter_config.json" ]]; then
    log "LoRA 权重推理：swift infer --model $MODEL_PATH --adapters $CKPT"
    run_logged "infer_${MODEL_KEY}" \
      swift infer --model "$MODEL_PATH" --adapters "$CKPT" \
        --stream false --max_new_tokens 64 --enable_thinking false
  else
    log "全参权重推理：swift infer --model $CKPT"
    run_logged "infer_${MODEL_KEY}" \
      swift infer --model "$CKPT" --stream false --max_new_tokens 64 --enable_thinking false
  fi
fi

log "=== T5 完成 ==="
