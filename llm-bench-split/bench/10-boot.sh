# ============================================================================
# 10-boot —— 起来之前：双机门禁、RoCEv2 GID 解析、GPU 排空检查、dmon、清理、
# 参数裁剪、确定性端口，以及 boot_env()（原脚本顶部那段可执行流程）。
# ============================================================================

# ------------------------------------------------------------------ 双机门禁
need_dual() {
  [ "$NNODES" -gt 1 ] || return 0
  local err=0
  if [ -z "$NCCL_IF" ]; then
    echo "!! 双机必须显式 NCCL_IF（bond0，或 198.51.100.x 那段 RoCE 网卡名）"; err=1
  fi
  if [ -z "$NCCL_HCA" ]; then
    echo "!! 双机必须显式 NCCL_HCA（本机是 RoCE：mlx5_bond_2..9，逗号列表）"; err=1
  fi
  if [ -z "$MASTER_ADDR" ] || echo "$MASTER_ADDR" | grep -qiE '^(127\.|::1$|localhost$)'; then
    echo "!! MASTER_ADDR=$MASTER_ADDR 不能是空/回环；写对端可达 IP（推荐 192.0.2.x）"; err=1
  fi
  [ "$err" = 0 ] || {
    echo "   缺配置时跑起来只会表现为「卡住/超时」，所以这里直接退出。"
    echo "   自查一台机器：nc -zv $MASTER_ADDR $MASTER_PORT；ibv_devinfo | grep -E 'state|link_layer'"
    exit 2
  }
}

# --------------------------------------------------------- RoCEv2 GID（实测）
# 选 GID：sysfs 读 gid_attrs/types + gids，只认 RoCE v2 且非 fe80，优先 IPv4 映射；
# 选完再做双端一致性校验：同一 index 在两台必须指向同一类型。
gid_candidates() {                        # gid_candidates <hca> → 每行 index<TAB>gid<TAB>类型
  local hca=$1 d i idx t g
  d="$NCCL_SYSFS_IB/$hca/ports/1"
  [ -d "$d/gid_attrs/types" ] || return 1
  for i in "$d"/gid_attrs/types/*; do
    [ -f "$i" ] || continue
    idx=${i##*/}
    t=$(cat "$i" 2>/dev/null) || continue
    case "$t" in *"RoCE v2"*|*RoCEv2*) ;; *) continue ;; esac
    g=$(cat "$d/gids/$idx" 2>/dev/null) || continue
    [ -n "$g" ] || continue
    case "$g" in fe80*|FE80*) continue ;; esac   # 链路本地 IPv6 不能跨机
    printf '%s\t%s\t%s\n' "$idx" "$g" "$t"
  done
}

gid_ipv4() { case "$1" in 0000:0000:0000:0000:0000:ffff:*) return 0 ;; *) return 1 ;; esac; }

gid_first_ipv4() {                         # 仅用于提示：本机第一个 IPv4 映射的 RoCEv2 index
  local hca idx g t
  for hca in ${NCCL_HCA//,/ }; do
    [ -z "$hca" ] && continue
    while IFS=$'\t' read -r idx g t; do
      gid_ipv4 "$g" && { echo "$idx"; return 0; }
    done < <(gid_candidates "$hca" 2>/dev/null)
  done
  return 1
}

resolve_gid() {
  (( NNODES > 1 )) || return 0
  if [ -n "$NCCL_GID" ]; then
    GID_SOURCE=env
    echo "GID=$NCCL_GID 来源=env（手工指定，注入 HCA=${NCCL_HCA}）"
    local ipv4
    ipv4=$(gid_first_ipv4) || true
    if [ -n "${ipv4:-}" ] && [ "$ipv4" != "$NCCL_GID" ]; then
      echo "   ! 本机 sysfs 的 IPv4 RoCEv2 GID 是 $ipv4，与 NCCL_GID=$NCCL_GID 不同（按 NCCL_GID 注入）"
    fi
    return 0
  fi
  GID_SOURCE=auto
  echo "GID=auto 来源=auto（不注入 NCCL_IB_GID_INDEX，由 NCCL 逐 HCA 自选 RoCEv2 GID）"
}

gid_types_local() {                       # 本机：每行 hca|index|类型（全量 index）
  local hca f
  for hca in ${NCCL_HCA//,/ }; do
    [ -z "$hca" ] && continue
    for f in "$NCCL_SYSFS_IB/$hca/ports/1/gid_attrs/types"/*; do
      [ -f "$f" ] || continue
      printf '%s|%s|%s\n' "$hca" "${f##*/}" "$(tr -d '\n' < "$f")"
    done
  done
}

peer_gid_table() {                        # 对端：每行 hca|index|类型（ssh 可用才行）
  local host="$PEER_ROCE_BASE"
  [ -n "$host" ] || host="$PEER_ADDR"
  [ -n "$host" ] || return 1
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" sh -s -- "$NCCL_HCA" "$NCCL_SYSFS_IB" <<'REMOTE' 2>/dev/null
for h in $(echo "$1" | tr ',' ' '); do
  for f in "$2/$h/ports/1/gid_attrs/types"/*; do
    [ -f "$f" ] || continue
    echo "$h|${f##*/}|$(tr -d '\n' < "$f")"
  done
done
REMOTE
}

gid_cross_check() {                       # 0=一致/无法校验；1=不一致（门禁 FAIL）
  local lt pt h i t key peer_type bad=0 n=0 me
  me=$(hostname 2>/dev/null || echo local)
  lt=$(gid_types_local)
  [ -n "$lt" ] || { GID_CROSS=unknown; return 0; }
  if ! pt=$(peer_gid_table); then
    GID_CROSS=unknown
    echo "   ! 无法获取对端 GID 表（对端 ssh 不可达），跳过双端一致性校验"
    return 0
  fi
  local -A LT=() PT=()
  while IFS='|' read -r h i t; do [ -n "$h" ] && LT["$h/$i"]="$t"; done <<< "$lt"
  while IFS='|' read -r h i t; do [ -n "$h" ] && PT["$h/$i"]="$t"; done <<< "$pt"
  for key in "${!LT[@]}"; do
    n=$(( n + 1 ))
    h=${key%/*}; i=${key#*/}
    peer_type=${PT[$key]:-缺失}
    if [ "$peer_type" = 缺失 ] || [ "$peer_type" != "${LT[$key]}" ]; then
      bad=$(( bad + 1 ))
      echo "   ✗ GID 类型不一致：$me/$h index=$i 本机=${LT[$key]} 对端=${peer_type}"
    fi
  done
  if (( bad )); then
    GID_CROSS=mismatch
    echo "   两端口径不一致 $bad/$n 项：改用 NCCL_GID=<两端同为 RoCE v2 IPv4 的 index>，或用 NCCL_HCA 缩到一致的口"
    return 1
  fi
  GID_CROSS=ok
  echo "GID=$NCCL_GID 双端一致（$n 个 index 类型对照通过）"
  return 0
}

dmon_start() {                               # dmon_start <输出文件>
  [ "${TELEMETRY:-0}" = 1 ] || return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "   ! [WARN] 宿主机没有 nvidia-smi，本档遥测跳过" >&2
    return 0
  fi
  DMON_FILE="$1"; DMON_ERR="$1.err"
  rm -f "$DMON_FILE" "$DMON_ERR"
  # `-o` 是输出选项（D/T/csv/nounit/noheader），必须跟取值；旧写法 `-o -f FILE`
  # 会把 -f 当成 -o 的取值、FILE 变成多余位置参数 → dmon 秒退或只写 stdout，
  # 文件永远不会生成。默认格式（# gpu pwr gtemp mtemp ...）正是 05_report.py 解析的格式，
  # 不要加 `-o csv`（会去掉表头，报告读不到 hdr）。
  nvidia-smi dmon -s put -d 1 -f "$DMON_FILE" >/dev/null 2>"$DMON_ERR" &
  DMON_PID=$!
  sleep 1
  if ! kill -0 "$DMON_PID" 2>/dev/null; then
    wait "$DMON_PID" 2>/dev/null
    echo "   !! [WARN] nvidia-smi dmon 启动即退出，遥测未启动：$DMON_FILE" >&2
    [ -s "$DMON_ERR" ] && sed 's/^/       /' "$DMON_ERR" >&2
    rm -f "$DMON_ERR"
    DMON_PID=""; DMON_FILE=""
    return 1
  fi
  echo "    · 遥测 dmon → $DMON_FILE (pid $DMON_PID)"
  return 0
}

dmon_stop() {
  [ -n "${DMON_PID:-}" ] || return 0
  # 用 SIGTERM：非交互 shell 里 & 启动的后台作业默认忽略 SIGINT，
  # 发 -INT 会等不到退出、最后只能 KILL，反而更容易丢文件缓冲。
  kill "$DMON_PID" 2>/dev/null
  for _ in 1 2 3 4 5; do
    kill -0 "$DMON_PID" 2>/dev/null || break
    sleep 0.2
  done
  kill -0 "$DMON_PID" 2>/dev/null && kill -KILL "$DMON_PID" 2>/dev/null
  wait "$DMON_PID" 2>/dev/null
  DMON_PID=""
  # 兜底：无论启动检查是否漏判，收尾时一定确认文件真的有内容
  if [ -n "${DMON_FILE:-}" ] && [ ! -s "$DMON_FILE" ]; then
    echo "   !! [WARN] 遥测文件未生成或为空：$DMON_FILE（报告的长稳遥测表会缺数据）" >&2
    [ -s "${DMON_ERR:-}" ] && sed 's/^/       /' "$DMON_ERR" >&2
  fi
  [ -n "${DMON_ERR:-}" ] && rm -f "$DMON_ERR"
  DMON_FILE=""; DMON_ERR=""
  return 0
}

cleanup() {
  # 函数定义在后面；脚本若在定义前就 exit，type 保护可避免 trap 报 command not found
  type progress_stop >/dev/null 2>&1 && progress_stop
  dmon_stop
  # 专家负载采样也是独立后台循环（原先 Ctrl-C 会残留）
  type expert_sampler_stop >/dev/null 2>&1 && expert_sampler_stop
  # 显存采样是独立后台循环，Ctrl-C / OOM 退出也要收掉（否则 nvidia-smi 会一直挂着）
  if [ -n "${MEM_SAMPLER_PID:-}" ]; then
    kill "$MEM_SAMPLER_PID" 2>/dev/null
    wait "$MEM_SAMPLER_PID" 2>/dev/null || true
    MEM_SAMPLER_PID=""
  fi
  return 0
}

prune_optional() {
  local table="$MA" out=() t name
  # 提示走 stderr：调用方用 mapfile 捕 stdout，stdout 里任何文字都会被当成参数拼进命令行。
  for t in "$@"; do
    [ -n "$t" ] || continue
    if [ -s "$table" ] && [ "${t:0:2}" = "--" ]; then
      name=${t%%=*}; name=${name#--}
      grep -qE "^ *--$name([ =]|$)" "$table" || { echo "    · 裁掉（容器不支持）: $t" >&2; continue; }
      if [ "$t" = "--$name" ] && grep -qE "^ *--$name (\{[^ ]*\}|[A-Z0-9_]+)([ ]|$)" "$table"; then
        echo "    · 裁掉（裸 flag 需要取值，写法不对）: $t" >&2; continue
      fi
    fi
    out+=( "$t" )
  done
  printf '%s\n' "${out[@]}"
}

# 新阶段的模型数组逐个参数按 --help 过滤：缺失的剔除并 WARN（stdout 只出可用参数）
filter_args() {
  local a name
  for a in "$@"; do
    [ -n "$a" ] || continue
    if [ "${a:0:2}" = "--" ]; then
      name=${a%%=*}; name=${name#--}
      if ! grep -qE "^[[:space:]]*--$name([[:space:]=]|$)" "$ARGS_HELP" 2>/dev/null; then
        echo "   ! [WARN] 剔除（--help 无此参数）：$a" >&2
        continue
      fi
    fi
    printf '%s\n' "$a"
  done
}

# dispatcher 取值是否被 --help 列出（choices 可能在同一行或下一行）
disp_supported() {
  grep -A2 -E "^[[:space:]]*--moe-token-dispatcher-type" "$ARGS_HELP" 2>/dev/null | grep -qw "$1"
}

# 确定性端口：两台对同一个 tag 必须算出同一个号，否则 torchrun 会各连各的。
# 原方案 $(( MASTER_PORT + RANDOM % 200 )) 在双机上必然分叉 → 两端都等，直到超时。
det_port() {
  local h
  h=$(printf '%s' "$1" | cksum | awk '{print $1}')
  echo $(( MASTER_PORT + h % 200 ))
}

# ---- 顶层常量（原脚本散落在函数之间，值不变）----
DMON_PID=""
DMON_FILE=""
DMON_ERR=""

parse_phase() {       # --phase X / X 两种写法；未知阶段直接退出并打用法
  if [ "${1:-}" = "--phase" ]; then set -- "${2:-}"; fi
  PHASE="${1:-}"
  export PHASE
  case " $ALL_PHASES " in                 # 阶段清单只在 bench/00-config.sh 维护一份
    *" $PHASE "*) ;;
    *) echo "!! 未知或缺失阶段：'${PHASE}'"; sed -n '1,30p' "$0"; exit 2 ;;
  esac
}

boot_env() {          # 原顶部可执行流程：run 目录 → 引擎 → 镜像 → run-info → 门禁
  START_TS=$(date +%s)
  # 不指定 RUN_ID 时续用 runs/latest；没有则新建（标签用阶段名，soak 单跑即 ..._soak）
  lc_run_resume "${PHASE:-full}" || exit 2
  LOGDIR=$(lc_run_stage_dir "$PHASE") || exit 2
  export LOGDIR
  lc_engine_log
  MA=$(lc_megatron_args)
  # MA 是「参数存在性」的唯一依据文件；为空说明本 run 与自己都还没拿到容器 --help。
  # 这里只提示不中断——真正需要判定的地方一律 ensure_args_check + flag_in_help 兜底，
  # 绝不能拿空串直接 grep（那样恒失败，会把「有该参数」误报成「容器没有」）。
  [ -n "$MA" ] || echo "   · 暂无 Megatron --help 原文（本 run 未跑 03_deploy）；需要时现场生成"

  IMAGE="${IMAGE:-$(lc_pick_image 'pytorch' '' || true)}"
  [ -n "$IMAGE" ] || { echo "!! 当前引擎的镜像列表里没有 pytorch 镜像，先跑 03_deploy.sh"; exit 2; }
  lc_image_exists "$IMAGE" || { echo "!! 镜像 $IMAGE 在 $ENGINE${NS:+ / namespace=$NS} 下不存在（load/run 必须同一引擎与命名空间）"; exit 2; }
  echo "IMAGE=$IMAGE（引擎 $ENGINE${NS:+ / namespace=$NS}）"
  lc_run_info_set image "$IMAGE"
  lc_run_info_set digest "$(lc_digest "$IMAGE")"
  lc_run_info_set engine "$ENGINE"
  lc_run_info_set namespace "${NS:-none}"
  lc_run_info_set nodes "$NNODES"
  lc_run_info_set nproc "$NPROC"
  lc_run_info_set world "$WORLD"
  lc_run_info_set nccl_if "$NCCL_IF"
  lc_run_info_set nccl_hca "$NCCL_HCA"
  echo "run 目录：$LC_RUN_DIR"
  need_dual
  resolve_gid

  GID_CROSS="unknown"                       # ok|mismatch|unknown
  if REASON=$(lc_gpu_busy_reason); then
    :
  else
    echo "$REASON"
    if [ "${ALLOW_BUSY:-0}" = 1 ]; then
      echo "   ALLOW_BUSY=1：继续，但结果只能当探索值，不能进验收报告。"
    else
      echo "!! GPU 未排空。排空后重跑，或 ALLOW_BUSY=1 强行跑（结果不可用于验收）。"
      exit 2
    fi
  fi
  # 参数自检的产物路径（依赖上面 lc_run_stage_dir 产生的 LC_RUN_DIR）
  ARGS_CHECK="$LC_RUN_DIR/args-check.json"
  ARGS_HELP="$LC_RUN_DIR/args-help.txt"
}
