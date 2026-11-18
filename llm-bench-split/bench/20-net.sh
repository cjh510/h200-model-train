# ============================================================================
# 20-net —— RoCE/NCCL 门禁：逐口 GID、对端 GID 对照、8MB allreduce 探针、失败诊断，
# 以及每个训练阶段开跑前必须过的 rdma_gate_precheck()。
# ============================================================================

# 容器内 probe 统一入口：run_probe <文件名> <tag> [probe 参数...]
# 提前 return 的报错走 stderr：探针没同步/cp 失败这类原因必须显式出现在终端，不能只留在日志里。
run_probe() {
  local py=$1 tag=$2; shift 2
  if [ ! -f "$HERE_PROBES/$py" ]; then echo "!! 缺少 $HERE_PROBES/$py（把整个 probes/ 目录同步到本机）" >&2; return 2; fi
  cp "$HERE_PROBES/$py" "$LOGDIR/$py" 2>/dev/null || { echo "!! 拷贝探针失败：$HERE_PROBES/$py → $LOGDIR/（查权限/磁盘）" >&2; return 2; }
  nccl_env_for "$tag" -e COLL_MAX_MSG="$COLL_MAX_MSG"
  # RUN_PROBE_TIMEOUT：探针整体超时（秒），仅门禁这类短探针使用
  local to=()
  [ -n "${RUN_PROBE_TIMEOUT:-}" ] && to=( timeout -s KILL "$RUN_PROBE_TIMEOUT" )
  lc_run -v "$ROOT":/workspace -v "$LOGDIR":/workspace/logs -w /workspace \
    -e PYTHONPATH=/workspace/Megatron-LM -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
    "${NCCL_ENV[@]}" \
    "$IMAGE" "${to[@]}" torchrun --nproc_per_node="$NPROC" --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
      --master_port="$(det_port "$tag")" \
      "/workspace/logs/$py" "$@" 2>&1 | tee "$LOGDIR/$tag.txt"
  return "${PIPESTATUS[0]}"
}

gate_needed() {
  (( NNODES > 1 )) || return 1
  case " $RDMA_GATE_PHASES " in *" $PHASE "*) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------- 逐口门禁
hca_state() {                             # sysfs 内容是 "4: ACTIVE" / "1: DOWN"
  local f="$NCCL_SYSFS_IB/$1/ports/1/state"
  [ -r "$f" ] || { echo ""; return 1; }
  awk '{print ($2 == "" ? $1 : $2)}' "$f" 2>/dev/null
}

hca_netdev() { local d=${1#mlx5_}; echo "${d//_/}"; }   # mlx5_bond_2 → bond2

ipv4_of_dev() {                          # <netdev> → 首个 IPv4（去掉掩码）
  ip -4 -o addr show "$1" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1
}

ipv4_last() { local x=${1##*.}; [ -n "$x" ] && echo "$x" || true; }   # 198.51.100.12 → 12

peer_hca_ips() {                         # 对端逐口 IP 表：每行 hca<TAB>ip（BatchMode ssh 可用才行）
  local host="$PEER_ROCE_BASE"           # 两台 HCA 同名（mlx5_bond_2..9），同名直配，不用猜偏移
  [ -n "$host" ] || host="$PEER_ADDR"
  [ -n "$host" ] || return 1
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" sh -s -- "$NCCL_SYSFS_IB" <<'REMOTE' 2>/dev/null
for p in "$1"/*; do
  [ -d "$p" ] || continue
  h=${p##*/}
  nd=$(cat "$p/ports/1/gid_attrs/ndevs/0" 2>/dev/null)
  case "$h" in mlx5_bond_*) [ -n "$nd" ] || nd="bond${h#mlx5_bond_}" ;; esac
  [ -n "$nd" ] || continue
  ip -4 -o addr show "$nd" 2>/dev/null | awk -v h="$h" '{sub(/\/.*/,"",$4); if ($4 != "") { print h "\t" $4; exit }}'
done
REMOTE
}

peer_ip_candidates() {                   # <本机IP> [偏好偏移…] → 同网段候选（已知偏移优先，其次 ±10）
  local ip=$1 a b c d p off
  IFS=. read -r a b c d <<< "$ip"
  [ -n "${d:-}" ] || return 1
  for off in "${@:2}"; do
    [ -n "$off" ] || continue
    p=$(( d + off ))
    (( p >= 1 && p <= 254 )) && echo "$a.$b.$c.$p"
  done
  p=$(( d - 10 ))
  (( p >= 1 && p <= 254 )) && echo "$a.$b.$c.$p"
  p=$(( d + 10 ))
  (( p >= 1 && p <= 254 )) && echo "$a.$b.$c.$p"
}

peer_ip_for() {                          # 两台 RoCE 网段相差 10，谁在低位不可预设（.2↔.12 必须对称成立）：
  local ip=$1 cand                       # ssh 表直配 → 已学偏移 → ±10 候选，每个候选都要 ping 通过
  for cand in $(peer_ip_candidates "$ip" ${PEER_LEARNED_OFF:+"$PEER_LEARNED_OFF"}); do
    ping -c1 -W1 "$cand" >/dev/null 2>&1 && { echo "$cand"; return 0; }
  done
  return 1
}

rdma_hca_filter() {                      # 0=可继续（可能已剔除坏口）；2=无可用口或 STRICT FAIL
  local hca dev st ip peer pg vd reason keep=() total=0 bad=0 e tab pmap_host
  local -A PMAP=()
  GATE_PER_HCA=(); PEER_LEARNED_OFF=""; PEER_MAP_OK=0
  pmap_host=${PEER_ROCE_BASE:-$PEER_ADDR}
  # 11-07 双机实测（logs0/logs1）：只按 ±10 推导时低位侧（本机 .2→对端 .12）整阶段 RDMA_GATE_FAIL。
  # 对端解析改三层：① ssh 逐口表（同名 HCA 直配）→ ② 已学偏移 → ③ ±10 候选，全部 ping 验证。
  if tab=$(peer_hca_ips) && [ -n "$tab" ]; then
    PEER_MAP_OK=1
    while IFS=$'\t' read -r e peer; do [ -n "$e" ] && PMAP[$e]=$peer; done <<< "$tab"
    echo "   对端 HCA IP 表（ssh ${pmap_host}）：$(printf '%s' "$tab" | tr '\t' '=' | tr '\n' ' ')"
  else
    echo "   ! 对端逐口表不可用（ssh ${pmap_host:-未设} 不通）——退回「已学偏移 → ±10 候选」推导；"
    echo "     要确定性解析请设 PEER_ROCE_BASE=<对端可达 IP> 且打通 BatchMode ssh"
  fi
  for hca in ${NCCL_HCA//,/ }; do
    [ -z "$hca" ] && continue
    total=$(( total + 1 ))
    dev=$(hca_netdev "$hca"); st=$(hca_state "$hca")
    ip=$(ipv4_of_dev "$dev"); peer=""; pg="n/a"; vd=ok; reason=""
    case "$dev" in
      bond0|bond1) vd=excluded; reason="非数据面/管理口，显式排除" ;;
      *) case "$st" in
           ACTIVE|4:*) ;;
           *) vd=drop; reason="state=${st:-不可读}，非 ACTIVE" ;;
         esac ;;
    esac
    if [ "$vd" = ok ] && [ -z "$ip" ]; then vd=drop; reason="无 IPv4"; fi
    if [ "$vd" = ok ]; then
      peer=${PMAP[$hca]:-}
      # ssh 表给的候选也要 ping 通过；不通就退回候选推导，不直接信表
      [ -z "$peer" ] || ping -c1 -W1 "$peer" >/dev/null 2>&1 || peer=""
      [ -n "$peer" ] || peer=$(peer_ip_for "$ip") || peer=""
      if [ -n "$peer" ]; then
        pg=ok
        e=$(ipv4_last "$peer") && tab=$(ipv4_last "$ip") \
          && PEER_LEARNED_OFF=$(( e - tab ))
      else
        pg=fail; vd=drop
        tab=$(peer_ip_candidates "$ip" ${PEER_LEARNED_OFF:+"$PEER_LEARNED_OFF"} | paste -sd/ -)
        reason="对端候选 ${tab:-无} 均 ping 不通$([ "$PEER_MAP_OK" = 1 ] || echo "；ssh 逐口表也不可用")"
      fi
    fi
    GATE_PER_HCA+=( "$hca|${st:-}|${ip:-}|${peer:-}|$pg|$vd|${reason:-}" )
    if [ "$vd" = ok ]; then keep+=( "$hca" ); else bad=$(( bad + 1 )); fi
  done

  echo "   逐口门禁（HCA / state / 本机IPv4 / 对端 / ping / 判定 / 原因）："
  for e in ${GATE_PER_HCA[@]+"${GATE_PER_HCA[@]}"}; do printf '     %s\n' "${e//|/  }"; done

  if (( NCCL_HCA_STRICT == 1 && bad > 0 )); then
    echo "   ✗ NCCL_HCA_STRICT=1：$bad/$total 个口不可用，不剔除，直接 FAIL"
    USED_HCA=""; return 2
  fi
  if (( ${#keep[@]} == 0 )); then
    echo "   ✗ 全部 $total 个 HCA 不可用，双机 RDMA 无法建立"
    USED_HCA=""; return 2
  fi
  (( bad > 0 )) && echo "   ! 已剔除 $bad 个不可用口，保留 ${#keep[@]} 个继续"
  USED_HCA=$(printf '%s,' "${keep[@]}"); USED_HCA=${USED_HCA%,}
  NCCL_HCA=$USED_HCA
  echo "   使用 NCCL_IB_HCA=$USED_HCA"
  return 0
}

                                          # 不能用 busbw=$(rdma_gate_once ...) —— 子 shell 里的
                                          # PROBE_ERR/IB_* 赋值会全部丢失，门禁失败就没了归因
rdma_gate_once() {                        # <tag> → 0=成功（BUSBW_OUT=busbw）；0≠失败（PROBE_ERR 已归因）
  # 注意：local 是一条命令，bash 先展开整条命令的所有词再执行赋值 —— 同一条 local 里
  # 后面的赋值引用前面刚赋的变量，在 set -u 下会报 "tag: unbound variable"（11-08 双机
  # 探针秒败、probe_error 为空的根因）。必须拆成两条 local。
  local tag=$1
  local log="$LOGDIR/$tag.txt" rc busbw
  BUSBW_OUT=""
  rm -f "$LOGDIR/rdma-gate.jsonl" "$log"
  # 探针文件没同步过来时 run_probe 会在 tee 之前就 return 2：目录里连 .txt 都不会有，
  # 表现成「NCCL 一条网络日志都没有」，很容易被误判成网络不通。这里先点明。
  if [ ! -f "$HERE_PROBES/hw_probe.py" ]; then
    PROBE_ERR="missing_probe:$HERE_PROBES/hw_probe.py"
    echo "   ✗ 探针未启动：缺少 $HERE_PROBES/hw_probe.py（把整个 probes/ 目录同步到两台）" >&2
    IB_USED=0; IBEXT=0; GDRDMA=0; SOCKET=0
    return 1
  fi
  RUN_PROBE_TIMEOUT=60 run_probe hw_probe.py "$tag" --only coll --ops all_reduce \
    --sizes 8388608 --secs 1 --out /workspace/logs/rdma-gate.jsonl
  rc=$?
  IB_USED=0; IBEXT=0; GDRDMA=0; SOCKET=0
  [ -f "$log" ] || PROBE_ERR="no_output:${tag}.txt（容器未启动或 $ENGINE 报错）"
  grep -q 'NET/IB : Using' "$log" 2>/dev/null && IB_USED=1
  grep -q 'Using network IBext' "$log" 2>/dev/null && IBEXT=1
  grep -q 'GDRDMA' "$log" 2>/dev/null && GDRDMA=1
  grep -q 'NET/Socket' "$log" 2>/dev/null && SOCKET=1
  busbw=$(grep -oE '"busbw_gbs": [0-9.]+' "$LOGDIR/rdma-gate.jsonl" 2>/dev/null | head -1 | awk '{print $2}')
  # 失败原因分类：探针起来了但没拿到 busbw 时，不分类就只能写「未走 RoCE」，分不清是
  # 路径/依赖/建链/对端没起。写进 rdma-gate.json 的 probe_error，便于事后一眼归因。
  if (( rc != 0 )) && [ -z "$PROBE_ERR" ]; then
    if   (( rc == 137 )); then
      PROBE_ERR="timeout:${RUN_PROBE_TIMEOUT:-60}s 被 KILL（对端没同时起，卡在 rendezvous）"
    elif grep -q "can't open file" "$log" 2>/dev/null; then
      PROBE_ERR="probe_path:/workspace/logs/hw_probe.py 打不开（LOGDIR 挂载或拷贝失败）"
    elif grep -qE "ModuleNotFoundError|ImportError" "$log" 2>/dev/null; then
      PROBE_ERR="import:python 依赖缺失（torch 导入失败）"
    elif grep -qE "Rendezvous|Connection refused|Failed to connect" "$log" 2>/dev/null; then
      PROBE_ERR="rendezvous:对端未就绪（两台必须同时起）"
    elif grep -qE "ibv_modify_qp|ncclSystemError|ncclInternalError|NCCL error|NCCL WARN" "$log" 2>/dev/null; then
      PROBE_ERR="nccl:建链失败（见 ${tag}.txt）"
    elif grep -q "Traceback (most recent call last)" "$log" 2>/dev/null; then
      PROBE_ERR="python:traceback（见 ${tag}.txt）"
    else
      PROBE_ERR="rc=$rc（无已知特征，见 ${tag}.txt）"
    fi
  elif (( rc == 0 )) && [ -z "$busbw" ]; then
    PROBE_ERR="no_busbw:${tag}.jsonl 无 busbw_gbs（rank0 未写或没跑到 coll）"
  fi
  if (( rc == 0 )) && [ -n "$busbw" ]; then BUSBW_OUT=$busbw; return 0; fi
  return 1
}

gate_json() {                             # gate_json <PASS|FAIL> [busbw]
  local verdict=$1 busbw=${2:-null} entries="" e h st ip peer pg vd rs peers="${PEER_ADDR:-$MASTER_ADDR}" ping="n/a"
  [ -n "${PEER_ADDR:-}" ] && { ping -c 1 -W 1 "$PEER_ADDR" >/dev/null 2>&1 && ping=ok || ping=fail; }
  for e in ${GATE_PER_HCA[@]+"${GATE_PER_HCA[@]}"}; do
    [ -z "$e" ] && continue
    IFS='|' read -r h st ip peer pg vd rs <<< "$e"
    entries+="${entries:+,}{\"hca\":\"$h\",\"state\":\"$st\",\"ip\":\"$ip\",\"peer\":\"$peer\",\"ping\":\"$pg\",\"verdict\":\"$vd\",\"reason\":\"$rs\"}"
  done
  printf '{"phase":"%s","verdict":"%s","transport":"%s","used_hca":"%s","per_hca":[%s],"gid":"%s","gid_source":"%s","ib_used":%s,"ibext":%s,"gdrdma":%s,"socket":%s,"peer":"%s","ping":"%s","busbw_gbs":%s,"probe_error":"%s","ts":"%s"}\n' \
    "$PHASE" "$verdict" "$TRANSPORT" "${USED_HCA:-}" "$entries" "${NCCL_GID:-auto}" "$GID_SOURCE" \
    "$IB_USED" "$IBEXT" "$GDRDMA" "$SOCKET" "$peers" "$ping" "$busbw" "${PROBE_ERR:-}" "$(date -Iseconds)" \
    > "$LOGDIR/rdma-gate.json"
}

gate_diag() {                             # 门禁失败四行诊断
  echo "   ✗ RDMA gate FAIL（$PHASE 不启动）诊断："
  local ibv hca i list="" ips peer="${PEER_ADDR:-}"
  ibv=$(ibv_devinfo 2>/dev/null | awk '/hca_id:/{id=$2} /state:/{st=$2} /link_layer:/{printf "%s state=%s link=%s; ", id, st, $2}' | cut -c1-200)
  echo "     1) ibv_devinfo：${ibv:-不可读}"
  for hca in ${NCCL_HCA//,/ }; do
    [ -z "$hca" ] && continue
    for i in "$NCCL_SYSFS_IB/$hca/ports/1/gid_attrs/types"/*; do
      [ -f "$i" ] || continue
      list+="$hca/$(basename "$i")=$(tr -d '\n' < "$i") "
    done
  done
  echo "     2) GID 类型：${list:-sysfs 不可读}"
  ips=$(ip -4 -o addr show 2>/dev/null | awk '{print $2"="$4}' | tr '\n' ' ' | cut -c1-160)
  echo "     3) 本机 IPv4：${ips:-?}；peer=${peer:-未设置} ping=$(if [ -n "$peer" ] && ping -c 1 -W 1 "$peer" >/dev/null 2>&1; then echo ok; else echo fail; fi)"
  echo "     4) NCCL 片段：$(grep -E 'NET/IB|NET/Socket|Using network|ibv_modify_qp|Bootstrap|error' "$LOGDIR/rdma-gate.txt" 2>/dev/null | tail -4 | tr '\n' '|' | cut -c1-220)"
  [ -n "${PROBE_ERR:-}" ] && echo "     5) 探针：${PROBE_ERR}"
}

gate_rerun_hint() {                      # 可直接照抄的复跑命令（只跑探针，不跑训练）
  local hca="${USED_HCA:-$NCCL_HCA}"
  echo "   复跑（只跑 8MB 探针，不跑训练）："
  echo "   NNODES=2 NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR NCCL_IF=${NCCL_IF:-bond0} \\"
  echo "     NCCL_HCA=$hca${NCCL_GID:+ NCCL_GID=$NCCL_GID} NCCL_DEBUG=INFO COLL_SIZES=8388608 FORCE=1 \\"
  echo "     bash 04_bench.sh --phase nccl"
}

rdma_gate() {                             # 0=PASS（含 socket 回退）；2=FAIL
  local busbw ok=0
  USED_HCA=""
  PROBE_ERR=""
  # 探针文件缺失时下面所有网络判定都无意义，直接点明，别让它伪装成「网络不通」
  if [ ! -f "$HERE_PROBES/hw_probe.py" ]; then
    PROBE_ERR="missing_probe:$HERE_PROBES/hw_probe.py"
    echo "!! 缺少探针 $HERE_PROBES/hw_probe.py：本阶段（$PHASE）不启动（把整个 probes/ 目录同步到两台）"
    gate_rerun_hint; gate_json FAIL
    return 2
  fi
  if ! rdma_hca_filter; then
    echo "!! 逐口门禁未通过：本阶段（$PHASE）不启动"
    gate_diag; gate_rerun_hint; gate_json FAIL
    return 2
  fi
  if [ -n "$NCCL_GID" ] && ! gid_cross_check; then
    echo "!! 双端 GID 不一致：本阶段（$PHASE）不启动（改 NCCL_GID 或用 NCCL_HCA 缩口）"
    gate_rerun_hint; gate_json FAIL
    return 2
  fi
  busbw=""
  if rdma_gate_once rdma-gate; then ok=1; busbw=$BUSBW_OUT; fi
  if (( ok )) && (( ! SOCKET )); then
    TRANSPORT=ib
    echo "RDMA gate PASS（8MB allreduce，busbw=${busbw} GB/s，IBext=$IBEXT GDRDMA=$GDRDMA，HCA=$NCCL_HCA）"
    gate_json PASS "$busbw"
    return 0
  fi
  # NET/IB=1 却失败 ≠ 未走 RoCE：是「走了 RoCE 但传输中报错」（11-08 实测 rank15 ncclSystemError）
  if (( IB_USED || IBEXT )) && (( ! SOCKET )); then
    echo "   ! 8MB allreduce 已走 RoCE（NET/IB=$IB_USED IBext=$IBEXT GDRDMA=$GDRDMA）但传输失败：${PROBE_ERR:-未知}——查 rdma-gate.txt 的 NCCL WARN 与 dmesg 链路抖动"
  else
    echo "   ! 8MB allreduce 未走 RoCE（NET/IB=$IB_USED IBext=$IBEXT GDRDMA=$GDRDMA socket=$SOCKET）"
  fi
  if [ "${NCCL_FALLBACK}" = socket ]; then
    echo "   ! NCCL_FALLBACK=socket → NCCL_IB_DISABLE=1 重试一次"
    NCCL_IB_DISABLE=1
    if rdma_gate_once rdma-gate-socket; then
      busbw=$BUSBW_OUT
      TRANSPORT=socket
      echo "RDMA gate PASS（socket 回退，busbw=${busbw} GB/s）"
      echo "   ! transport=socket：非 RoCE 数据，不可用于验收"
      gate_json PASS "$busbw"
      return 0
    fi
    NCCL_IB_DISABLE=0
    echo "   ✗ socket 回退仍失败"
  fi
  # 探针阶段失败必有归因；能走到这里而 PROBE_ERR 为空的唯一情形：
  # rc=0 且有 busbw 但走了 socket（未走 RoCE）
  [ -n "$PROBE_ERR" ] || PROBE_ERR="socket:8MB allreduce 有 busbw 但走了 socket（非 RoCE，不用于验收）"
  gate_diag; gate_rerun_hint; gate_json FAIL
  return 2
}

# ---- 顶层常量（原脚本散落在函数之间，值不变）----
RDMA_GATE_PHASES="dx8b d70b moe moe-dp soak"
PEER_LEARNED_OFF=""                       # 第一个解析成功的口学到的末位偏移（对端末位−本机末位），供其余口复用
PEER_MAP_OK=0                             # 1 = 已从对端 ssh 拿到逐口 IP 表
BUSBW_OUT=""                              # rdma_gate_once 成功时的 busbw；必须经全局变量带回，

rdma_gate_precheck() {  # 每个训练阶段开跑前过一遍 8MB allreduce 门禁
  if gate_needed; then
    echo "########## RDMA gate：8MB allreduce，timeout 60s ##########"
    if ! rdma_gate; then
      echo "!! RDMA 门禁未通过：本阶段（$PHASE）不启动，rc=2（档位未开跑，不触发 dual_stop）"
      lc_run_stage_done "$PHASE" "$START_TS" "RDMA_GATE_FAIL"
      exit 2
    fi
  fi
}
