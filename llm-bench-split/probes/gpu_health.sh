#!/usr/bin/env bash
# =============================================================================
# [宿主机] 采集 ECC / retired_pages / NVLink 误码 / Xid 快照，输出一行 JSON。
# 用法：bash probes/gpu_health.sh <输出文件>
# 长稳前后各一次（双机每台各采一份），05_report.py 按机器/逐卡比增量；
# ECC/retired/NVLink/Xid 任一增量 >0 判 FAIL，报告里给出机器名与 GPU 编号。
#
# 输出兼容旧字段（总量）+ 新增 per_gpu 明细与 xid_gpus（PCI 映射到卡号）。
# =============================================================================
set -uo pipefail
OUT=${1:?用法: gpu_health.sh <out.json>}

kernel_log() { (dmesg -T 2>/dev/null || journalctl -k --no-pager 2>/dev/null) || true; }
q()  { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null; }
sum_of() { q "$1" | awk '{s+=$1} END{printf "%d", s+0}'; }
num() { tr -dc '0-9'; }                  # "[N/A]" → 空 → 下面统一补 0

# retired_pages.pending 是 Yes/No；H200 上部分子字段不可查 → SKIP（不能当 FAIL）
retired_count() {
  nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader,nounits 2>/dev/null \
    | awk 'tolower($0) ~ /yes/ {n++} END{printf "%d", n+0}'
}
retired_state() {
  local out
  out=$(nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader,nounits 2>&1)
  if printf '%s' "$out" | grep -qiE 'not a valid field|N/A'; then echo SKIP
  elif printf '%s' "$out" | grep -qi 'Yes'; then echo FAIL
  elif printf '%s' "$out" | grep -qi 'No'; then echo PASS
  else echo SKIP; fi
}

per_gpu_json() {
  local idx i c u r n out="" sep=""
  mapfile -t idx < <(q index)
  for i in "${idx[@]}"; do
    i=${i// /}
    c=$(nvidia-smi --query-gpu=ecc.errors.corrected.aggregate.total   -i "$i" --format=csv,noheader,nounits 2>/dev/null | num); c=${c:-0}
    u=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total -i "$i" --format=csv,noheader,nounits 2>/dev/null | num); u=${u:-0}
    r=$(nvidia-smi --query-gpu=retired_pages.pending -i "$i" --format=csv,noheader,nounits 2>/dev/null \
        | awk 'tolower($0) ~ /yes/ {print 1; exit} {print 0; exit}'); r=${r:-0}
    n=$(nvidia-smi nvlink -e -i "$i" 2>/dev/null | awk -F: '/Errors/{gsub(/[ \t]/,"",$2); s+=$2} END{printf "%d", s+0}')
    out+="${sep}{\"index\":${i},\"ecc_corrected\":${c},\"ecc_uncorrected\":${u},\"retired_pending\":${r},\"nvlink_errors\":${n}}"
    sep=","
  done
  printf '%s' "$out"
}

# dmesg 的 Xid 行里只有 PCI 地址，这里映射回 GPU 编号，便于报告直接定位到卡。
xid_gpus_json() {
  local map line pci gpu u out=""
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits 2>/dev/null)
  while read -r line; do
    pci=$(printf '%s' "$line" | grep -oE 'PCI:[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}' | head -1 | cut -d: -f2-)
    [ -n "$pci" ] || continue
    gpu=$(printf '%s\n' "$map" | awk -F', ' -v p="$pci" '{gsub(/[ \t]/,"",$2); if (index($2,p)==1) {print $1; exit}}')
    [ -n "$gpu" ] && out+="$gpu,"
  done < <(kernel_log | grep -E 'NVRM.*Xid' | head -20)
  u=$(printf '%s' "$out" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, - 2>/dev/null || true)
  printf '%s' "$u"
}

printf '{"time":"%s","host":"%s","ecc_corrected":%s,"ecc_uncorrected":%s,"retired_pending":%s,"retired_pages_state":"%s","nvlink_errors":%s,"xid":%s,"sxid":%s,"xid_gpus":[%s],"per_gpu":[%s]}\n' \
  "$(date -Iseconds)" "$(hostname)" \
  "$(sum_of ecc.errors.corrected.aggregate.total)" \
  "$(sum_of ecc.errors.uncorrected.aggregate.total)" \
  "$(retired_count)" "$(retired_state)" \
  "$(nvidia-smi nvlink -e 2>/dev/null | awk -F: '/Errors/{gsub(/[ \t]/,"",$2); s+=$2} END{printf "%d", s+0}')" \
  "$(kernel_log | grep -cE 'NVRM.*Xid' || true)" \
  "$(kernel_log | grep -cE 'SXid' || true)" \
  "$(xid_gpus_json)" "$(per_gpu_json)" > "$OUT"
cat "$OUT"
