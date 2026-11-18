#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机（脚本内部自己起容器）。压测矩阵总入口。
# 容器引擎：docker 优先，没有可用 docker 时回落 nerdctl（探测与封装见 lib.sh）。
#
#   bash 04_bench.sh --phase argscheck # 参数自检：容器内 pretrain_gpt.py --help → args-check.json
#   bash 04_bench.sh --phase hw      # G3 单机硬件基线（含 FP8/L2/PCIe/collective，稳态 ≥5s）
#   bash 04_bench.sh --phase coll    # 单机 collective 扫描（已并入 hw；仅 OPTIONAL=1 时单跑）
#   bash 04_bench.sh --phase s8b     # 单机 8 卡 Llama-3.1-8B：S1 S2 S3 S4
#   bash 04_bench.sh --phase dx8b    # 双机 8B TP8×DP2 训练对照（GBS=32）
#   bash 04_bench.sh --phase nccl    # 双机 collective 全类型 × 8B–8G（不跑训练）
#   bash 04_bench.sh --phase d70b    # 双机 16 卡 Llama-3.1-70B：TP8×PP2
#   bash 04_bench.sh --phase moe     # MoE/MLA：单机 EP8×TP1 / EP2×TP4 + 双机 EP16；dispatcher 两种
#   bash 04_bench.sh --phase moe-dp  # 仅双机：EP8×DP2（alltoall；梯度跨机 all-reduce），与 EP16 对照
#   bash 04_bench.sh --phase limit   # 显存/规模墙：扫 mbs→seq 到 OOM，OOM 记为 OOM_BOUNDARY
#   bash 04_bench.sh --phase soak    # 长稳（SOAK_MODEL=dense / dense70 / moe / highmem，可逗号组合；
#                                    #   both=dense+moe；all 在双机含 dense70，单机保持三条；默认 dense）
#   bash 04_bench.sh --phase io      # checkpoint 写读 + 数据管线（无真数据则 SKIP）
#   bash 04_bench.sh --phase fault   # 故障注入：OOM / kill rank / NCCL 超时
#   bash 04_bench.sh --phase mfu     # 峰值 MFU：粗扫并行度×GBS → 细扫 mbs×accum → seq → 前3名复测取中位
#                                    #   头条 MFU† = Megatron throughput / 989.5；含时钟/降频取证
#   bash 04_bench.sh --phase report  # 汇总成表
#
# 双机：两台都执行同一条命令，只有 NODE_RANK 不同，且 NCCL_IF / NCCL_HCA 必须显式给：
#   node-a: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> NCCL_IF=bond0 \
#           NCCL_HCA=mlx5_bond_2,mlx5_bond_3,... bash 04_bench.sh --phase d70b
#   node-b: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> ...（同上）
# 缺 NCCL_IF / NCCL_HCA / MASTER_ADDR 是回环地址时，脚本直接 exit 2 —— 不再静默退回 socket。
#
# 可调：IMAGE / ROOT / NPROC / STEPS / FORCE=1 / OPTIONAL=1 / TELEMETRY=1
#       SOAK_MINUTES / SOAK_MIN_ITERS / SOAK_MODEL / MOE_SOAK_MINUTES / MOE_SOAK_MIN_ITERS
#       DENSE70_SOAK_MINUTES / DENSE70_SOAK_MIN_ITERS（dense70 仅双机）
#       HIGHMEM_MINUTES / HIGHMEM_MBS（留空=同模型自动标定）/ HIGHMEM_SEQ / HIGHMEM_MODEL / HIGHMEM_HEALTH
#       HIGHMEM_TARGET_PCT / HIGHMEM_BAND_LO / HIGHMEM_BAND_HI / HIGHMEM_CAL / HIGHMEM_CAL_ITERS
#       HIGHMEM_CAL_TRIES / HIGHMEM_CAL_START
#       MFU_STEPS / MFU_REPEAT / MFU_CV_MAX / MFU_GBS_LIST / MFU_TP_LIST / MFU_MBS_LIST
#       MFU_ACCUM_LIST / MFU_SEQ_LIST / MFU_MAX_GBS / MFU_MAX_RUNS / MFU_PLAN_ONLY / MFU_MODELS
#       COLL_SIZES / MOE_GBS
#       LIMIT_MAX_MBS / LIMIT_MAX_SEQ / LIMIT_STEPS
# OPTIONAL=1 才跑精简矩阵里降级的档位（单机 coll、io 的 70B 档）。
#
# 带值 flag 一律 --name=value：prune_optional 才能「一项一裁」。
# =============================================================================

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
lc_init || exit 2

ROOT="${ROOT:-/root/llm-bench}"
MEGATRON=/workspace/Megatron-LM

NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"          # 档位端口 = 基址 + cksum(tag) % 200，确定性
STEPS="${STEPS:-15}"
NCCL_IF="${NCCL_IF:-}"                       # 例 bond0 / 198.51.100.x 所属网卡
NCCL_HCA="${NCCL_HCA:-}"                     # 本机 RoCE：mlx5_bond_2..9
NCCL_GID="${NCCL_GID:-${NCCL_IB_GID_INDEX:-}}"  # 空 = 不注入，由 NCCL 逐 HCA 自选 RoCEv2 GID
NCCL_DBG="${NCCL_DEBUG:-INFO}"
NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"     # 仅双机注入；QP 建链超时（小于对端重试窗口）
NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"  # 仅双机注入
NCCL_FALLBACK="${NCCL_FALLBACK:-none}"       # none|socket：门禁失败时是否用 NCCL_IB_DISABLE=1 重试
TRANSPORT="${TRANSPORT:-ib}"                 # ib|socket；socket = 非 RoCE 数据，报告标 WARN
GID_SOURCE="auto"                            # auto|env：auto = 不注入 NCCL_IB_GID_INDEX
NCCL_HCA_STRICT="${NCCL_HCA_STRICT:-0}"      # 1 = 任一口不可用直接 FAIL，不剔除
PEER_ROCE_BASE="${PEER_ROCE_BASE:-}"         # 对端任一可 ssh 的 IP（优先对端管理 IP；逐口门禁 ssh 拿对端逐口表、GID 对照）
NCCL_SYSFS_IB="${NCCL_SYSFS_IB:-/sys/class/infiniband}"   # 桩测可指向假目录
USED_HCA=""                                  # 逐口门禁剔除后实际使用的 HCA 列表
GATE_PER_HCA=()                              # 逐口明细：hca|state|ip|peer|ping|verdict
IB_USED=0; IBEXT=0; GDRDMA=0; SOCKET=0       # 8MB 探针的传输判定
PEER_ADDR="${PEER_ADDR:-}"                   # 双机日志回传目标（node-a 的管理 IP）
HERE_PROBES="$HERE/probes"
TELEMETRY="${TELEMETRY:-0}"
FORCE="${FORCE:-0}"
OPTIONAL="${OPTIONAL:-0}"                    # 1 = 跑精简矩阵里降级的档位（单机 coll / io 70B）
SOAK_MIN_ITERS="${SOAK_MIN_ITERS:-100}"
SOAK_MODEL="${SOAK_MODEL:-dense}"             # dense / dense70 / moe / highmem；both=dense+moe；all=全部适用项
SOAK_PROGRESS="${SOAK_PROGRESS:-1}"          # 长稳进度显示开关（1=开，0=关）
PROGRESS_SECS="${PROGRESS_SECS:-60}"         # 进度刷新间隔（秒）
SOAK_TARGET_MINS=""                          # 当前长稳目标分钟数，由 soak_* 设置
SOAK_LABEL=""                                # 当前长稳形态 dense / dense70 / moe / highmem
PROGRESS_PID=""                              # 进度循环 PID
PROGRESS_FILE=""                             # 进度日志（09-soak/soak-progress.log）
PROGRESS_LOG=""                              # 当前训练日志，供收尾进度行读取
PROGRESS_T0=""                               # 本条长稳起始 epoch
MOE_SOAK_MINUTES="${MOE_SOAK_MINUTES:-}"      # 空 = 双机 60 / 单机 30
MOE_SOAK_MIN_ITERS="${MOE_SOAK_MIN_ITERS:-}"  # 空 = 沿用 SOAK_MIN_ITERS
DENSE70_SOAK_MINUTES="${DENSE70_SOAK_MINUTES:-60}"  # 双机 70B 正常训练长稳时长
DENSE70_SOAK_MIN_ITERS="${DENSE70_SOAK_MIN_ITERS:-}"  # 空 = 沿用 SOAK_MIN_ITERS
EXPERT_SAMPLE_SECS="${EXPERT_SAMPLE_SECS:-300}"   # 专家负载采样间隔（默认 5 分钟）
HIGHMEM_MINUTES="${HIGHMEM_MINUTES:-60}"      # 显存高占用长稳时长
# 显存高占用的档位不再靠「猜一个 mbs」，而是跑前在同模型上标定（见 highmem_calibrate）：
#   HIGHMEM_MBS 留空 = 自动标定到 HIGHMEM_TARGET_PCT；显式给值 = 跳过标定、按给定值跑（仍做峰值判定）。
# 旧默认「单机 16 @ seq16384」= 262144 tokens/GPU，按实测拟合（peak ≈ 27 GiB + 0.83 MiB/token）
# 需要约 240 GiB，必然 OOM；它之所以没炸，是因为跨模型边界守卫先把 mbs 压成了 1。
HIGHMEM_MBS="${HIGHMEM_MBS:-}"                # 空 = 自动标定；显式值 = 跳过标定
HIGHMEM_SEQ="${HIGHMEM_SEQ:-}"                # 空 = 单机 16384 / 双机 4096
HIGHMEM_MODEL="${HIGHMEM_MODEL:-auto}"        # auto|8b|70b|110b；auto = 单机 8b、双机 110b
HIGHMEM_HEALTH="${HIGHMEM_HEALTH:-1}"         # 1 = 跑前/跑后各采一次硬件快照
HIGHMEM_TARGET_PCT="${HIGHMEM_TARGET_PCT:-80}"   # 标定目标占用率（分母 VRAM_MIB）；取带内偏中下，
                                                 # 给 mbs 取整向上（+1 mbs ≈ +9 个点）和长稳漂移都留余量
# 漂移余量：标定只跑 4 步，长稳要跑 60 分钟，分配器碎片会让峰值继续涨。
# 口径要对齐才算数——同配置、同指标（nvidia-smi memory.used）实测：
#   4 步试跑 36,494 MiB（25.4%） vs 21 分钟长稳 41,696 MiB（29.0%）→ +5,202 MiB ≈ +3.6 个点。
# （早先写的「+11.9 GiB ≈ 8.5 个点」是拿 torch max reserved 28.8 GiB 和 nvidia-smi used 40.7 GiB
#  相减，两个指标不同、seq 也不同，那个数作废。）
# 这里仍取 8 作保守余量：mbs 越大激活越多，碎片漂移通常比 mbs=1 更明显。
HIGHMEM_DRIFT="${HIGHMEM_DRIFT:-8}"           # 标定选档上限 = HIGHMEM_BAND_HI - HIGHMEM_DRIFT
HIGHMEM_BAND_LO="${HIGHMEM_BAND_LO:-75}"      # 判定下限：低于此值记 WARN_LOW「未达高占用」
HIGHMEM_BAND_HI="${HIGHMEM_BAND_HI:-92}"      # 判定上限：高于此值记 WARN_HOT「过于贴边」
HIGHMEM_CAL="${HIGHMEM_CAL:-1}"               # 1 = HIGHMEM_MBS 留空时自动标定；0 = 不标定（沿用 HIGHMEM_MBS）
HIGHMEM_CAL_ITERS="${HIGHMEM_CAL_ITERS:-4}"   # 标定每档步数（够让分配器涨到稳态即可）
HIGHMEM_CAL_TRIES="${HIGHMEM_CAL_TRIES:-6}"   # 标定最多试几档
HIGHMEM_CAL_START="${HIGHMEM_CAL_START:-1}"   # 标定起始 mbs
MEM_SAMPLE_SECS="${MEM_SAMPLE_SECS:-5}"       # 显存采样间隔
NO_RECOMPUTE="${NO_RECOMPUTE:-0}"             # 1 = 即使 seq>=64K 也不开 recompute（highmem 必须置 1，否则激活被压缩）
COLL_SIZES="${COLL_SIZES:-8,65536,1048576,67108864,1073741824,8589934592}"
COLL_MAX_MSG="${COLL_MAX_MSG:-1073741824}"   # 跨机总传输量上限（0=不限），按 size×NNODES 比较：allgather/reduce_scatter
                                             # 的实际流量会放大 nodes 倍（11-08 实测：1G 档 allgather 输出 16GiB/卡，
                                             # 无 PFC 的 RoCE 上大突发丢包 → RC 重传卡死 600s 被看门狗杀）。单机不受限
MOE_GBS="${MOE_GBS:-}"                       # 空 = 单机 8 / 双机 16
MOE_MLA="${MOE_MLA:-auto}"                   # auto|1|0；auto = 容器支持 --enable-experimental 时跑 MLA，否则 SKIP；0 = 主动跳过 MLA
MOE_ATTN="gqa"                               # 运行期实际注意力（gqa|mla），由 moe_resolve_attn 决定
MOE_EXPERIMENTAL=0                           # 本次 MoE 配置是否传了 --enable-experimental
MOE_NOTE=""                                  # 写进 moe.jsonl 的备注（SKIP 原因等）
MOE_SKIP_TOPO=""                             # SKIP 时该记哪个拓扑：mla=MLA 依赖缺失，gate=关键 flag 缺失
LIMIT_MAX_MBS="${LIMIT_MAX_MBS:-32}"
LIMIT_MAX_SEQ="${LIMIT_MAX_SEQ:-131072}"
LIMIT_STEPS="${LIMIT_STEPS:-5}"

# 硬件口径常量：全部取 H200 SXM 实测值，05_report.py 里有一份同值的 PEAK / VRAM_GIB。
PEAK_TFLOPS=989.4                            # bf16 稠密峰值 TFLOPS（不含稀疏；稀疏是 2×）
VRAM_MIB=143771                              # nvidia-smi 实测每卡显存
VRAM_GIB=140.4                               # = 143771/1024，报告一律用实测，不写 141

WORLD=$(( NPROC * NNODES ))
export ROOT NPROC NNODES WORLD
echo "world=$WORLD (nproc=$NPROC × nodes=$NNODES) rank=$NODE_RANK master=$MASTER_ADDR steps=$STEPS"

# 归一化：`--phase hw` 和 `hw` 两种写法都接受。先定 PHASE，再决定 run 的阶段目录。
if [ "${1:-}" = "--phase" ]; then set -- "${2:-}"; fi
PHASE="${1:-}"
export PHASE
case "$PHASE" in
  argscheck|hw|coll|nccl|s8b|dx8b|d70b|moe|moe-dp|limit|io|fault|mfu|soak|report) ;;
  *) echo "!! 未知或缺失阶段：'${PHASE}'"; sed -n '1,30p' "$0"; exit 2 ;;
esac

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
need_dual

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
resolve_gid

GID_CROSS="unknown"                       # ok|mismatch|unknown
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

# 空闲门禁：别人的进程在跑，step 时间和功耗全失真，测了也白测。
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

# ------------------------------------------------------------------ 遥测
DMON_PID=""
DMON_FILE=""
DMON_ERR=""
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
trap cleanup EXIT INT TERM

# ------------------------------------------------------------- Llama 模型定义
# 一手来源：meta-llama Llama-3.1-8B / 70B 公开 config。
#   8B ：32 层 / hidden 4096 / ffn 14336 / 32 heads / 8 kv / head_dim 128 → 8.03B
#   70B：80 层 / hidden 8192 / ffn 28672 / 64 heads / 8 kv / head_dim 128 → 70.55B
COMMON=(
  --swiglu --normalization=RMSNorm --norm-epsilon=1e-5
  --use-rotary-position-embeddings --rotary-base=500000
  --disable-bias-linear --position-embedding-type=rope
  --untie-embeddings-and-output-weights
  --group-query-attention --num-query-groups=8 --kv-channels=128
  --attention-dropout=0.0 --hidden-dropout=0.0
  --vocab-size=128256 --make-vocab-size-divisible-by=128
  --tokenizer-type=NullTokenizer
  --mock-data --split=100,0,0
)
LLAMA_8B=( "${COMMON[@]}" --num-layers=32 --hidden-size=4096 --ffn-hidden-size=14336 --num-attention-heads=32 )
LLAMA_70B=( "${COMMON[@]}" --num-layers=80 --hidden-size=8192 --ffn-hidden-size=28672 --num-attention-heads=64 )
# 配置 A（双机 limit 用）：80 层 / hidden 10240 / 80 heads（head_dim 128）/ 8 KV / FFN 35840
LIMIT_110B=( "${COMMON[@]}" --num-layers=80 --hidden-size=10240 --ffn-hidden-size=35840 --num-attention-heads=80 )
# MLA 配置（配置 B）不能带 GQA：num-query-groups=8 会让 TE 断言
# 「Keys and values must have num_gqa_group = 8 heads!」——MLA 的 KV 是压缩潜变量
#（1 组），与 8 组 GQA 互斥。11-08 单机 moe 实测 8 rank 全崩，配置 B 剥离 GQA 两项。
#（dense 的 LLAMA_8B/70B/LIMIT_110B 仍用 COMMON 的 GQA=8，Llama-3.1 结构如此。）
MOE_COMMON=()
for _a in "${COMMON[@]}"; do
  case "$_a" in --group-query-attention|--num-query-groups=*) continue ;; esac
  MOE_COMMON+=( "$_a" )
done
# 配置 B（moe / limit 单机用）：24 层 / hidden 4096 / 32 heads / MLA / 32 专家 + 1 共享
MOE_B=( "${MOE_COMMON[@]}" --num-layers=24 --hidden-size=4096 --ffn-hidden-size=2048 --num-attention-heads=32
        --num-experts=32 --moe-ffn-hidden-size=2048 --moe-router-topk=2
        --moe-shared-expert-intermediate-size=4096 --moe-aux-loss-coeff=1e-3
        --multi-latent-attention --q-lora-rank=1536 --kv-lora-rank=512
        --qk-head-dim=128 --qk-pos-emb-head-dim=64 --v-head-dim=128 )
# 配置 B 的 moe_est 入参：层数 / 专家数 / hidden / moe_ffn / 共享 intermediate / topk。
# 与上面 MOE_B 的形状参数同源，改模型时两处一起改；调用时再接 <tp> <pp> <ep> <vocab>。
MOE_B_EST=( 24 32 4096 2048 4096 2 )
MOE_B_VOCAB=128256

# 只允许这些「性能开关」缺失后自动裁掉：它们只影响优化路径，不改模型形状。
OPTIONAL_FLAGS=(
  --sequence-parallel --use-distributed-optimizer --overlap-grad-reduce
  --overlap-param-gather --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --recompute-granularity=selective
  --timing-log-level=2 --log-throughput
  --no-create-attention-mask-in-dataloader
)

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

# ------------------------------------------------------------ 参数自检 argscheck
ARGS_CHECK="$LC_RUN_DIR/args-check.json"
ARGS_HELP="$LC_RUN_DIR/args-help.txt"
ARG_FLAGS=(multi-latent-attention num-experts moe-ffn-hidden-size moe-router-topk
           moe-router-enable-expert-bias moe-shared-expert-intermediate-size
           expert-model-parallel-size moe-token-dispatcher-type moe-aux-loss-coeff
           mtp-num-layers fp8-recipe no-create-attention-mask-in-dataloader
           enable-experimental)
ARG_CRIT=(num-experts multi-latent-attention expert-model-parallel-size moe-token-dispatcher-type)

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

limit_record() {                         # status tag tp pp mbs seq gbs step_s tokens err_class
  local st=$1 tag=$2 tp=$3 pp=$4 mbs=$5 seq=$6 gbs=$7 ss=$8 tok=$9 ec=${10}
  local dp=$(( WORLD / tp / pp )) acc=$(( gbs / (mbs * dp) ))
  printf '{"world":%d,"nodes":%d,"model":"%s","tp":%d,"pp":%d,"dp":%d,"ep":%d,"mbs":%d,"seq":%d,"gbs":%d,"accum":%d,"tokens_per_gpu":%d,"step_s":%s,"status":"%s","transport":"%s","static_gb_est":%s,"error_class":"%s","error_line":"%s","ts":"%s"}\n' \
    "$WORLD" "$NNODES" "${LIMIT_MODEL:-unknown}" "$tp" "$pp" "$dp" "${LIMIT_EP:-1}" \
    "$mbs" "$seq" "$gbs" "$acc" "$tok" "${ss:-0}" "$st" "${TRANSPORT:-ib}" "${LIMIT_STATIC_GB:-0}" "$ec" \
    "${LIMIT_ERR_LINE:-}" "$(date -Is)" >> "$LOGDIR/limit.jsonl"
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

# limit.jsonl → limit-summary.json（成功上界 / 首个 OOM / 静态余量）
limit_summary() {
  local f="$LOGDIR/limit.jsonl" out="$LOGDIR/limit-summary.json"
  python3 - "$f" "$out" "${LIMIT_STATIC_GB:-0}" "$WORLD" "$NNODES" "${LIMIT_MODEL:-unknown}" \
          "$VRAM_GIB" <<'PY'
import datetime, json, sys
src, dst, static, world, nodes, model, vram = (
    sys.argv[1], sys.argv[2], float(sys.argv[3] or 0),
    int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], float(sys.argv[7]))
rows = []
try:
    with open(src, encoding="utf-8") as fh:
        for ln in fh:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
except FileNotFoundError:
    pass
ok = [r for r in rows if r.get("status") == "OK"]
seq0 = [r for r in ok if r.get("seq") == 4096]
oom = [r for r in rows if r.get("status") == "OOM_BOUNDARY"]
oom.sort(key=lambda r: (r.get("seq", 0), r.get("mbs", 0)))
data = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": model, "world": world, "nodes": nodes,
    "static_gb_est": static, "vram_gib_per_gpu": vram,
    "static_margin_gib": round(vram - static, 1) if static else None,
    "max_ok_mbs@4096": max([r["mbs"] for r in seq0], default=0),
    "max_ok_seq@best_mbs": max([r["seq"] for r in ok], default=0),
    "max_ok_mbs": max([r["mbs"] for r in ok], default=0),
    "first_oom": ({k: oom[0].get(k) for k in
                   ("tp", "pp", "ep", "mbs", "seq", "gbs", "error_class", "error_line")} if oom else None),
    "oom_boundaries": len(oom),
    "failed": len([r for r in rows if r.get("status") == "FAILED"]),
    "note": "OOM_BOUNDARY 是预期边界：不计失败、不参与最好成绩与 MFU；只有 NCCL/hang/崩溃才 FAILED",
}
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
print(json.dumps(data, ensure_ascii=False))
PY
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

# ------------------------------------------------------------- 双机 RDMA 门禁
# 训练前先跑 8MB allreduce：QP 建链问题在 60s 内暴露，不等到 d70b 跑到一半才报 ibv_modify_qp。
RDMA_GATE_PHASES="dx8b d70b moe moe-dp limit soak"
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
PEER_LEARNED_OFF=""                       # 第一个解析成功的口学到的末位偏移（对端末位−本机末位），供其余口复用
PEER_MAP_OK=0                             # 1 = 已从对端 ssh 拿到逐口 IP 表
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

BUSBW_OUT=""                              # rdma_gate_once 成功时的 busbw；必须经全局变量带回，
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
  local port; port=$(det_port "$tag")

  # 跳过判据用哨兵文件，不再「日志里有 iteration 就算跑过」：
  # 半截日志（跑 3 步被杀）也有 iteration，原逻辑会永久跳过这档，长稳尤其致命。
  if [ "$FORCE" != "1" ] && [ -f "$LOGDIR/.ok_$sk" ]; then
    echo "→ 跳过（已完成哨兵 .ok_$sk）$tag  端口 $port"; return 0
  fi

  local dp=$(( WORLD / tp / pp ))
  if (( dp * tp * pp != WORLD )); then
    echo "   !! TP=$tp PP=$pp 无法整除 world=$WORLD，跳过（SKIP，非 FAILED）"
    [ "${LIMIT_JSONL:-0}" = 1 ] && limit_record SKIP "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" 0 0 SKIP
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$(lc_env_json_fields)" 0 SKIP "tp×pp 不整除 world"
    return 1
  fi
  if (( gbs % (mbs * dp) != 0 )); then
    echo "   !! GBS=$gbs 不能被 mbs×dp=$(( mbs * dp )) 整除，跳过（SKIP，非 FAILED）"
    echo "      Megatron 的 num_micro_batches = GBS ÷ (mbs × dp)，必须整除"
    [ "${LIMIT_JSONL:-0}" = 1 ] && limit_record SKIP "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" 0 0 SKIP
    rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$(lc_env_json_fields)" 0 SKIP "GBS 不整除 mbs×dp"
    return 1
  fi
  local nmib=$(( gbs / (mbs * dp) ))
  echo "→ $tag   mbs=$mbs dp=$dp → num_micro_batches=$nmib   port=$port"
  if (( pp > 1 )) && (( nmib < 2 * pp )); then
    echo "   ! micro-batch 太少（$nmib），PP=$pp 流水线气泡约 $(awk -v p=$pp -v m=$nmib 'BEGIN{printf "%.0f%%", (p-1)/m*100}')；加大 GBS 再比"
  fi

  local par=( --tensor-model-parallel-size="$tp" --pipeline-model-parallel-size="$pp" )
  # Megatron 断言 lr_warmup_steps < lr_decay_steps：limit/短跑档 STEPS=5 时写死 warmup=5
  # 会触发 AssertionError（11-08 单机 limit 实测），按步数收紧
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
    if { [ "${LIMIT_JSONL:-0}" = 1 ] || [ "${OOM_AS_BOUNDARY:-0}" = 1 ]; } \
       && [ "${ALLOW_OOM:-0}" = 1 ]; then
      LIMIT_ERR_LINE=$(grep -m1 -iE 'CUDA out of memory|OutOfMemoryError' "$log" | tr -d '"\\' | tr '\n' ' ' | cut -c1-180)
      echo "   ○ OOM_BOUNDARY  $tag（预期边界，不 dual_stop、不影响退出码）"
      echo "      $LIMIT_ERR_LINE"
      # 只有 limit 阶段才往 limit.jsonl 记；highmem 的 OOM 是它自己的边界，
      # 由 highmem_record 记进 highmem.jsonl，混进 limit.jsonl 会污染边界扫描的 OK/OOM 统计。
      [ "${LIMIT_JSONL:-0}" = 1 ] && limit_record OOM_BOUNDARY "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" 0 "$(( gbs * seq / WORLD ))" OOM
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
    [ "${LIMIT_JSONL:-0}" = 1 ] && limit_record PARTIAL "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$(step_seconds "$log")" "$(( gbs * seq / WORLD ))" PARTIAL
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
    if [ "${LIMIT_JSONL:-0}" = 1 ]; then
      limit_record OK "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$(step_seconds "$log")" "$(( gbs * seq / WORLD ))" ""
    else
      rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" OK
    fi
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
    if [ "${LIMIT_JSONL:-0}" = 1 ]; then
      LIMIT_ERR_LINE=$(grep -m1 -E 'unrecognized arguments|invalid choice|ValueError|AssertionError|RuntimeError|NCCL error|Segmentation fault|Killed' "$log" 2>/dev/null | tr -d '"\\' | tr '\n' ' ' | cut -c1-180)
      limit_record FAILED "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$(step_seconds "$log")" 0 "$(err_class "$log")"
    else
      rec_result "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$port" "$fingerprint" "$njit" FAILED "见日志末尾 20 行"
    fi
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

EXPERT_SAMPLER_PID=""
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

# ------------------------------------------------------------ 显存高占用长稳（highmem）
# 峰值基准取实测每卡 MiB（H200 143771 MiB = 140.4 GiB），与报告一致
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

MEM_SAMPLER_PID=""
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
# 上一档的残留算进本档峰值 → 标定结果偏大。等到占用 ≤5% 或超时；超时只 WARN，
# 因为残留只会让峰值偏高、档位偏保守，方向是安全的。
mem_wait_idle() {                        # mem_wait_idle [最多等几秒]
  local max=${1:-30} t0 used lim
  lim=$(( VRAM_MIB / 20 ))
  t0=$(date +%s)
  while :; do
    used=$(mem_gpu_max_mib)
    [ "$used" -le "$lim" ] && return 0
    if (( $(date +%s) - t0 >= max )); then
      echo "   ! [WARN] 等待显存回落超时（当前 ${used} MiB > 阈值 ${lim} MiB）；下一档峰值可能偏高"
      return 1
    fi
    sleep 2
  done
}

# 显存边界（只在**同一个模型**上才可作为硬上限）
# 背景：limit 阶段扫的是 LIMIT_*（单机 MOE_B，每卡静态 84.7 GiB；双机 110B），
#       highmem 跑的是 LLAMA_8B / 70B / 110B。单机两者根本不是同一个模型，
#       把 MoE 的 tokens/GPU 上限套到 dense 8B 上会算出「seq=16384 只能 mbs=1」，
#       实测占用只有 41696/143771 = 29%（11-10 单机），高显存长稳名不副实。
# 因此：模型一致（双机 110B 对 110B）→ 取该 seq 下最大 OK mbs 作为上限；
#       模型不一致 → 返回 cross-model，档位改由 highmem_calibrate 在同模型上试跑决定。
# 返回 "<mbs上限> <limit|cross-model|no-seq|none>"。
highmem_boundary() {                     # highmem_boundary <seq> <model> → "<mbs上限> <来源>"
  local seq=$1 model=${2:-} f="$LC_RUN_DIR/06-train-limit/limit.jsonl" out
  [ -s "$f" ] || { echo "0 none"; return 0; }
  out=$(python3 - "$f" "$seq" "$NNODES" "$model" <<'PY'
import json, sys
path, seq, nodes, want = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].upper()
rows, models = [], set()
try:
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            m_model = str(r.get("model", "")).upper()
            if m_model:
                models.add(m_model)
            if r.get("status") != "OK":
                continue
            if nodes > 1 and m_model != "110B":
                continue
            m, s = int(r.get("mbs") or 0), int(r.get("seq") or 0)
            if m > 0 and s > 0:
                rows.append((m, s, m_model))
except FileNotFoundError:
    pass
if not rows:
    print(0, "none")
    sys.exit()
same = [(m, s) for m, s, mm in rows if mm == want]
if not same:
    print(0, "cross-model")
    sys.exit()
best = max((m for m, s in same if s == seq), default=0)
print(best, "limit" if best else "no-seq")
PY
)
  echo "${out:-0 none}"
}

highmem_record() {                       # status tag model tp pp req used seq gbs iters step_s jitter src bmbs ec [cal_mbs cal_peak]
  local st=$1 tag=$2 model=$3 tp=$4 pp=$5 req=$6 used=$7 seq=$8 gbs=$9 it=${10}
  local ss=${11} jit=${12} src=${13} bmbs=${14} ec=${15}
  local cmbs=${16:-0} cpeak=${17:-0}
  local csv="$LOGDIR/memused-w${WORLD}_highmem.csv" log="$LOGDIR/w${WORLD}_${tag}.log"
  [ -f "$csv" ] || csv="$LOGDIR/memused-w${WORLD}_highmem-cal.csv"
  python3 - "$csv" "$log" "$LOGDIR/highmem.jsonl" "$WORLD" "$NNODES" "$tag" "$model" \
           "$tp" "$pp" "$req" "$used" "$seq" "$gbs" "$it" "${ss:-0}" "${jit:-0}" \
           "$st" "$src" "$bmbs" "$ec" "$VRAM_MIB" "${cmbs:-0}" "${cpeak:-0}" <<'PY'
import csv, json, re, sys, datetime

(mpath, lpath, out, world, nodes, tag, model, tp, pp, req, used, seq,
 gbs, iters, step_s, jitter, status, src, bmbs, ec) = sys.argv[1:21]
peak_cap = int(sys.argv[21])
cal_mbs = int(sys.argv[22] or 0)
cal_peak = int(sys.argv[23] or 0)

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
    thr = 0.2 * mx                     # 去掉容器启动前/退出后的低占用样本，只在「模型已加载」窗口内取首末
    idx = [i for i, x in enumerate(v) if x >= thr]
    first, last = (idx[0], idx[-1]) if idx else (0, len(v) - 1)
    start, end = v[first], v[last]
    growth = round((end - start) / start * 100, 1) if start > 0 else None
    gpus.append({"index": g, "mem_start": start, "mem_max": mx, "mem_end": end,
                 "growth_pct": growth, "peak_pct": round(mx / peak_cap * 100, 1),
                 "mem_max_at": v.index(mx), "samples": len(v)})

tmax = None
try:
    txt = open(lpath, errors="ignore").read()
    hits = re.findall(r"max allocated[^0-9]*([0-9.]+)", txt)
    if hits:
        tmax = round(float(hits[-1]), 1)
except FileNotFoundError:
    pass

rec = {"world": int(world), "nodes": int(nodes), "tag": tag, "model": model.upper(),
       "tp": int(tp), "pp": int(pp), "dp": int(world) // (int(tp) * int(pp)),
       "mbs_requested": int(req), "mbs_used": int(used), "seq": int(seq), "gbs": int(gbs),
       "iters": int(iters), "step_s": float(step_s), "jitter": float(jitter), "status": status,
       "boundary_source": src, "boundary_mbs": int(bmbs) if bmbs not in ("", "none") else None,
       # 本次长稳的整机峰值（取最热的一张卡）：判定「是否真的高占用」看这个数
       "peak_mib": max((g["mem_max"] for g in gpus), default=0),
       "peak_pct": round(max((g["mem_max"] for g in gpus), default=0) / peak_cap * 100, 1),
       # 标定档位（boundary_source=calib 时有效），便于回查「这个 mbs 是怎么来的」
       "calib_mbs": cal_mbs or None, "calib_peak_mib": cal_peak or None,
       "torch_max_allocated_mib": tmax, "error_class": ec or "", "gpu": gpus,
       "ts": datetime.datetime.now().isoformat(timespec="seconds")}
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

# ------------------------------------------------------------ 同模型显存标定
# 为什么必须同模型：limit 阶段扫的是 LIMIT_*（单机 MOE_B，每卡静态 84.7 GiB），
# highmem 跑的是 LLAMA_8B dense TP8（每卡静态约 17 GiB）。两者 tokens/GPU 上限差 5 倍以上，
# 把 MoE 的上限套给 dense 会得出「seq=16384 只能 mbs=1」→ 实测占用 41696/143771 = 29%
# （11-10 单机），高显存长稳完全名不副实。所以档位不再跨模型推导，而是在**同一个模型**
# 上短程试跑，直接用 peak_pct 收口：倍增爬坡 → 命中带内即停；有 OK 点与 OOM 点时按
# 「峰值≈静态 + a×tokens」线性插值直接算出目标 mbs，通常 2~3 档就到位。
highmem_cal_record() {                   # tag model tp pp mbs seq gbs peak_mib pct status step_s
  python3 - "$LOGDIR/highmem-cal.jsonl" "$WORLD" "$NNODES" "$@" <<'PY'
import json, sys, datetime
(out, world, nodes, tag, model, tp, pp, mbs, seq, gbs,
 peak, pct, st, ss) = sys.argv[1:15]
rec = {"world": int(world), "nodes": int(nodes), "tag": tag, "model": model.upper(),
       "tp": int(tp), "pp": int(pp), "mbs": int(mbs), "seq": int(seq), "gbs": int(gbs),
       "peak_mib": int(peak), "peak_pct": round(float(pct), 1), "status": st,
       "step_s": float(ss or 0),
       "ts": datetime.datetime.now().isoformat(timespec="seconds")}
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

# 结果写全局：CAL_OK / CAL_MBS / CAL_PEAK_MIB / CAL_PCT / CAL_TRIED
# 不用 $(...) 调用——run_train 依赖当前 shell 的 STEPS、哨兵键等状态。
highmem_calibrate() {                    # <NAME> <ARR> <TP> <PP> <SEQ> [上限 mbs，0=不限]
  CAL_OK=0; CAL_MBS=0; CAL_PEAK_MIB=0; CAL_PCT=0; CAL_TRIED=0; CAL_SUGGEST_SEQ=0; CAL_SEQ=$5
  local NAME=$1 ARR=$2 TP=$3 PP=$4 SEQ=$5 CAP=${6:-0}
  local dp=$(( WORLD / TP / PP ))
  local target=$HIGHMEM_TARGET_PCT lo=$HIGHMEM_BAND_LO hi=$HIGHMEM_BAND_HI
  # 选档上限要比判定上限更低：4 步试跑量不到 60 分钟的碎片漂移
  local acc_hi=$(( hi - ${HIGHMEM_DRIFT:-8} ))
  (( acc_hi > lo + 2 )) || acc_hi=$(( lo + 2 ))
  local tgt_mib; tgt_mib=$(awk -v v="$VRAM_MIB" -v p="$target" 'BEGIN{printf "%.0f", v*p/100}')
  local m=$HIGHMEM_CAL_START
  (( CAP >= 1 && m > CAP )) && m=$CAP
  (( m >= 1 )) || m=1
  local -a ok_m=() ok_p=()
  local oom_m=0 oom_p=0
  # 循环变量用 _ci 而不是 i：bash 是动态作用域，被调函数里一个裸 `i` 就会改掉调用方的
  # 局部 i（真机模拟时 stub 的 `for i in $(seq 1 120)` 直接把「最多 6 档」跑成了 120 档）。
  local _ci tag csv peak pct ss trc ss_save

  # 每次标定都重写档位表：同一个 run 目录里重跑时，旧档位留着会让报告里的逐档曲线串轮次
  rm -f "$LOGDIR/highmem-cal.jsonl"
  # 每次标定都重写档位表：同一个 run 目录里重跑时，旧档位留着会让报告里的逐档曲线串轮次
  rm -f "$LOGDIR/highmem-cal.jsonl"
  echo "########## highmem 标定：$NAME TP${TP}×PP${PP} seq=$SEQ，目标 ${target}%（≈${tgt_mib} MiB）##########"
  echo "    · 选档带 ${lo}~${acc_hi}%（判定带 ${lo}~${hi}%，另留 ${HIGHMEM_DRIFT:-8} 个点给 60 分钟碎片漂移）"
  echo "    · 每档 ${HIGHMEM_CAL_ITERS} 步试跑，最多 ${HIGHMEM_CAL_TRIES} 档；档位表 → $LOGDIR/highmem-cal.jsonl"
  for (( _ci = 1; _ci <= HIGHMEM_CAL_TRIES; _ci++ )); do
    tag="HIGHMEMCAL_${NAME}_tp${TP}_pp${PP}_mbs${m}_s${SEQ}"
    csv="$LOGDIR/memused-w${WORLD}_highmem-cal-t${_ci}-mbs${m}.csv"
    echo "   · 第 $_ci/${HIGHMEM_CAL_TRIES} 档：mbs=$m，GBS=$(( m * dp ))"
    mem_wait_idle 30
    : > "$csv"
    rm -f "$LOGDIR/.ok_highmem_cal_mbs${m}" "$LOGDIR/.oom_w${WORLD}_${tag}"
    ss_save=$STEPS; STEPS=$HIGHMEM_CAL_ITERS
    SENTINEL_KEY="highmem_cal_mbs${m}"
    mem_sampler_start "$csv"
    # 试跑不带 SOAK_REQUIRE_ITERS（4 步必然 <100，会被判 PARTIAL 并写 .fail_）；
    # rec_result 也不写，避免标定档污染 results.jsonl 的正式结果表。
    NO_RESULT_REC=1
    ALLOW_OOM=1 OOM_AS_BOUNDARY=1 SOAK_REQUIRE_ITERS="" \
      run_train "$tag" "$ARR" "$TP" "$PP" "$m" "$SEQ" "$(( m * dp ))"
    trc=$?
    NO_RESULT_REC=""
    mem_sampler_stop
    STEPS=$ss_save; SENTINEL_KEY=""
    CAL_TRIED=$_ci
    peak=$(mem_peak_mib "$csv")
    ss=$(step_seconds "$LOGDIR/w${WORLD}_${tag}.log")
    pct=$(awk -v p="$peak" -v v="$VRAM_MIB" 'BEGIN{printf "%.1f", p/v*100}')
    if [ -f "$LOGDIR/.oom_w${WORLD}_${tag}" ]; then
      echo "   · mbs=$m → OOM（作为上界，不选）；崩溃前峰值 ${peak} MiB（${pct}%）"
      highmem_cal_record "$tag" "$NAME" "$TP" "$PP" "$m" "$SEQ" "$(( m * dp ))" \
                         "$peak" "$pct" OOM "$ss"
      oom_m=$m; oom_p=$peak
      if (( ${#ok_m[@]} == 0 && m <= 1 )); then
        echo "   ✗ mbs=1 就 OOM：$NAME seq=$SEQ 连最小档都放不下，请降低 HIGHMEM_SEQ（如 8192）后重跑"
        return 1
      fi
    elif [ "$trc" -ne 0 ]; then
      echo "   ! [WARN] mbs=$m 试跑失败（rc=$trc，不是 OOM），本档不参与选档；日志 $LOGDIR/w${WORLD}_${tag}.log"
      highmem_cal_record "$tag" "$NAME" "$TP" "$PP" "$m" "$SEQ" "$(( m * dp ))" \
                         "$peak" "$pct" FAILED "$ss"
      (( ${#ok_m[@]} > 0 )) && break      # 已有 OK 点就按已有结果定档，否则继续换档
    else
      # 峰值必须随 mbs 单调不减。11-10 真机实测出现过 mbs=6 → 141692 MiB（98.6%）后
      # mbs=7 反而只有 99138 MiB（69.0%）且吞吐腰斩（198 vs 390 TFLOP/s）——那不是
      # "更省显存"，是没跑到稳态（分配器退避/建链失败）。这种点若被当有效值参与选档，
      # 会让结果随机漂，所以直接排除。
      local pmax=0 k
      for k in "${ok_p[@]}"; do [ "$k" -gt "$pmax" ] && pmax=$k; done
      if (( pmax > 0 )) && awk -v a="$peak" -v b="$pmax" 'BEGIN{exit !(a < b * 0.95)}'; then
        echo "   ! [WARN] mbs=$m 峰值 ${peak} MiB（${pct}%）低于更小档的 ${pmax} MiB，差 $(( 100 - peak * 100 / pmax ))%"
        echo "     判为「未达稳态」：不参与选档，并停止继续加大（更大的 mbs 只会更糟）"
        highmem_cal_record "$tag" "$NAME" "$TP" "$PP" "$m" "$SEQ" "$(( m * dp ))" \
                           "$peak" "$pct" SUSPECT "$ss"
        break
      fi
      echo "   · mbs=$m → 峰值 ${peak} MiB（${pct}%）"
      highmem_cal_record "$tag" "$NAME" "$TP" "$PP" "$m" "$SEQ" "$(( m * dp ))" \
                         "$peak" "$pct" OK "$ss"
      ok_m+=( "$m" ); ok_p+=( "$peak" )
      if awk -v p="$pct" -v a="$lo" -v b="$acc_hi" 'BEGIN{exit !(p>=a && p<=b)}'; then
        echo "   · [命中] mbs=$m 的峰值 ${pct}% 落在 ${lo}~${acc_hi}% 选档带内，标定结束"
        break
      fi
      if (( CAP >= 1 && m >= CAP )); then
        echo "   · [封顶] mbs=$CAP 是同模型 limit 的硬上限，占比只有 ${pct}%；标定结束"
        break
      fi
      # 已把目标夹在中间就停：再往上试只会更贴边（11-10 真机就是因为没停，一路试到 98.6%）
      if awk -v p="$pct" -v t="$target" 'BEGIN{exit !(p>=t)}'; then
        local below=0 q
        for q in "${ok_p[@]}"; do
          awk -v x="$q" -v v="$VRAM_MIB" -v t="$target" 'BEGIN{exit !(x / v * 100 < t)}' && below=1
        done
        if (( below == 1 )); then
          echo "   · [夹住] 本档 ${pct}% 已越过目标 ${target}%，且已有更低档 → 停止爬坡，按「离目标最近」选档"
          break
        fi
      fi
    fi
    # ---------- 决定下一档 ----------
    local n=${#ok_m[@]} next=0
    if (( n == 0 )); then
      next=$(( m > 1 ? m / 2 : 1 ))                 # 首档就 OOM：往下退
    else
      local m_lo=${ok_m[n-1]} p_lo=${ok_p[n-1]} m_hi=0 p_hi=0
      local have_hi=0 has_oom=0
      if (( oom_m > 0 && oom_m > m_lo )); then
        m_hi=$oom_m; p_hi=$oom_p; have_hi=1; has_oom=1
      elif (( n >= 2 )); then
        m_lo=${ok_m[n-2]}; p_lo=${ok_p[n-2]}
        m_hi=${ok_m[n-1]}; p_hi=${ok_p[n-1]}; have_hi=1
      fi
      if (( have_hi == 1 )); then
        # 两个锚点：
        #   ① 上锚是 OOM 档 → 只在 (下锚, 上锚-1] 内插值，绝不越过 OOM 档；
        #   ② 上锚是另一个 OK 档 → 允许外推到上锚之上（目标往往就在这一段），
        #      旧写法无条件夹到 him-1，会在「两个低档点」时把目标硬压回下锚+1，导致选档偏小。
        next=$(awk -v m1="$m_lo" -v p1="$p_lo" -v m2="$m_hi" -v p2="$p_hi" \
                    -v t="$tgt_mib" -v lom="$m_lo" -v him="$m_hi" -v iso="${has_oom}" 'BEGIN{
          a = (m2 > m1) ? (p2 - p1) / (m2 - m1) : 0;
          x = (a > 1) ? (m1 + (t - p1) / a) : (m1 * 2);   # 斜率过小=已吃满静态，直接翻倍
          x = int(x + 0.5);
          if (x > lom * 8) x = lom * 8;                   # 单步最多 8 倍，避免外推失控跑出荒唐 mbs
          if (x <= lom) x = lom + 1;
          if (iso == 1 && x >= him) x = him - 1;          # 只有 OOM 档才夹
          if (x < 1) x = 1;
          print x }')
      else
        next=$(( m_lo * 2 ))
      fi
    fi
    (( next >= 1 )) || next=1
    if (( CAP >= 1 && next > CAP )); then next=$CAP; fi
    # 防止原地打转：与已试档位重号就 +1
    local guard=0 dup x
    while (( guard < 10 )); do
      dup=0
      for x in "${ok_m[@]}"; do [ "$x" = "$next" ] && dup=1; done
      [ "$oom_m" = "$next" ] && dup=1
      [ "$dup" = 0 ] && break
      next=$(( next + 1 )); guard=$(( guard + 1 ))
    done
    if (( next == m )); then next=$(( m + 1 )); fi
    if (( CAP >= 1 && next > CAP )); then next=$CAP; fi
    if (( next == m )); then
      echo "   · [收敛] 下一档仍是 mbs=$m（受限 mbs 粒度/上限），标定结束"
      break
    fi
    m=$next
  done

  # ---------- 选档：在「不超过判定上限」的档里，取离目标最近的 ----------
  # 旧兜底写的是「全部高于选档带 → 取最低」，但它没判断是不是真的「全部高于」：
  # 11-10 真机 6 档是 25.4 / 42.7 / 74.1 / 89.9 / 98.6 / 69.0，既有低于带也有高于带，
  # 这一条把 25.4% 的 mbs=1 选成了结论 —— 于是长稳又只占 40 GB。改为按距离选，且
  # 硬性排除超过判定上限（>acc_hi）的档：那种档 60 分钟必定 OOM 或 WARN_HOT。
  local n=${#ok_m[@]}
  if (( n == 0 )); then
    echo "   ✗ 标定失败：没有任何一档跑通（见 $LOGDIR/highmem-cal.jsonl 与对应日志）"
    return 1
  fi
  local bi
  bi=$(awk -v v="$VRAM_MIB" -v t="$target" -v lim="$acc_hi" -v list="${ok_p[*]}" 'BEGIN{
    n = split(list, a, " "); bi = 0; bestd = 1e18; bestp = 0;
    for (i = 1; i <= n; i++) {
      p = a[i] / v * 100;
      if (p > lim) continue;                       # 超过判定上限的档一律不选
      d = (p > t) ? (p - t) : (t - p);
      if (d < bestd - 1e-9 || (d < bestd + 1e-9 && p < bestp)) { bestd = d; bi = i; bestp = p }
    }
    if (bi > 0) { print bi; exit }
    # 全部超过上限：取最低的那档（最接近可用），由调用方 WARN 说明
    mni = 1; mn = 1e18;
    for (i = 1; i <= n; i++) { p = a[i] / v * 100; if (p < mn) { mn = p; mni = i } }
    print mni }')
  (( bi >= 1 )) || bi=$n
  CAL_MBS=${ok_m[bi-1]}; CAL_PEAK_MIB=${ok_p[bi-1]}
  CAL_PCT=$(awk -v p="$CAL_PEAK_MIB" -v v="$VRAM_MIB" 'BEGIN{printf "%.1f", p/v*100}')
  CAL_OK=1
  local seq_pin=${7:-0}
  # 处方：按最高两档的斜率（MiB/token）反推「要让试跑值直接落进目标需要多大 seq」。
  # 只提示、不自动改，避免猜错的 seq 把整轮带偏；未达标时该值会写进 .warn_highmem_low，
  # 下次没显式给 HIGHMEM_SEQ 时自动套用（见 soak_highmem）。
  CAL_SUGGEST_SEQ=0
  if (( n >= 2 )); then
    CAL_SUGGEST_SEQ=$(awk -v t="$tgt_mib" -v m="$CAL_MBS" -v seq="$SEQ" \
        -v m1="${ok_m[n-2]}" -v p1="${ok_p[n-2]}" -v m2="${ok_m[n-1]}" -v p2="${ok_p[n-1]}" 'BEGIN{
      if (m2 <= m1 || p2 <= p1) { print 0; exit }
      a = (p2 - p1) / ((m2 - m1) * seq);            # MiB / token
      if (a <= 0) { print 0; exit }
      tok = m1 * seq + (t - p1) / a;
      s = int((tok / m) / 256 + 0.5) * 256;          # 256 粒度：1024 粒度在 mbs=4 时会偏 1.4 个点
      if (s < seq) s = seq;
      if (s > seq * 2) s = seq * 2;                 # 最多翻倍，不给荒唐值
      print s }')
  fi
  # ---- 用 seq 细化：mbs 粒度太粗时，把落点挪进选档带 ----
  # 典型情形：seq=16384 下 mbs=4→74.1%、mbs=5→89.9%，整段跨过 75~84%，
  # 靠 mbs 永远落不进带内。此时按斜率反推一个 seq（同一 mbs），试一档就能补上。
  # 只在「用户没钉 seq」「当前点不在带内」「处方合法」「还有档数」时做一次。
  if [ "$seq_pin" = 0 ] && (( n >= 2 )) \
     && [ "${CAL_SUGGEST_SEQ:-0}" -gt 0 ] && [ "$CAL_SUGGEST_SEQ" != "$SEQ" ] \
     && (( CAL_TRIED < HIGHMEM_CAL_TRIES )) \
     && ! awk -v p="$CAL_PCT" -v a="$lo" -v b="$acc_hi" 'BEGIN{exit !(p>=a && p<=b)}'; then
    local rm=$CAL_MBS rs=$CAL_SUGGEST_SEQ rtag rcsv rpeak rpct rrc
    CAL_TRIED=$(( CAL_TRIED + 1 ))
    rtag="HIGHMEMCAL_${NAME}_tp${TP}_pp${PP}_mbs${rm}_s${rs}"
    rcsv="$LOGDIR/memused-w${WORLD}_highmem-cal-t${CAL_TRIED}-mbs${rm}-s${rs}.csv"
    echo "   · 细化第 ${CAL_TRIED} 档：mbs=$rm / seq=$rs（mbs 粒度跨过选档带，用 seq 补）"
    mem_wait_idle 30
    : > "$rcsv"
    rm -f "$LOGDIR/.ok_highmem_cal_mbs${rm}_s${rs}" "$LOGDIR/.oom_w${WORLD}_${rtag}"
    ss_save=$STEPS; STEPS=$HIGHMEM_CAL_ITERS
    SENTINEL_KEY="highmem_cal_mbs${rm}_s${rs}"
    mem_sampler_start "$rcsv"
    NO_RESULT_REC=1
    ALLOW_OOM=1 OOM_AS_BOUNDARY=1 SOAK_REQUIRE_ITERS="" \
      run_train "$rtag" "$ARR" "$TP" "$PP" "$rm" "$rs" "$(( rm * dp ))"
    rrc=$?
    NO_RESULT_REC=""
    mem_sampler_stop
    STEPS=$ss_save; SENTINEL_KEY=""
    rpeak=$(mem_peak_mib "$rcsv")
    rpct=$(awk -v p="$rpeak" -v v="$VRAM_MIB" 'BEGIN{printf "%.1f", p/v*100}')
    if [ -f "$LOGDIR/.oom_w${WORLD}_${rtag}" ] || [ "$rrc" -ne 0 ]; then
      echo "   ! [WARN] 细化档 mbs=$rm/seq=$rs 失败（rc=$rrc），仍用 seq=$SEQ"
      highmem_cal_record "$rtag" "$NAME" "$TP" "$PP" "$rm" "$rs" "$(( rm * dp ))" \
                         "$rpeak" "$rpct" FAILED "$(step_seconds "$LOGDIR/w${WORLD}_${rtag}.log")"
    else
      echo "   · mbs=$rm/seq=$rs → 峰值 ${rpeak} MiB（${rpct}%）"
      highmem_cal_record "$rtag" "$NAME" "$TP" "$PP" "$rm" "$rs" "$(( rm * dp ))" \
                         "$rpeak" "$rpct" OK "$(step_seconds "$LOGDIR/w${WORLD}_${rtag}.log")"
      # 只在「不超过选档上限」且「比原档更接近目标」时采纳
      if awk -v a="$rpct" -v b="$CAL_PCT" -v t="$target" -v lim="$acc_hi" 'BEGIN{
           if (a > lim) exit 1;
           da = (a > t) ? (a - t) : (t - a); db = (b > t) ? (b - t) : (t - b);
           exit !(da < db) }'; then
        echo "   · [细化采纳] 落点 ${CAL_PCT}% → ${rpct}%，seq 由 $SEQ 改为 $rs"
        CAL_PCT=$rpct; CAL_PEAK_MIB=$rpeak; CAL_SEQ=$rs
      fi
    fi
  fi
  local drift="${HIGHMEM_DRIFT:-8}" dlo=$(( ${HIGHMEM_DRIFT:-8} / 2 ))
  local est_lo est_hi
  est_lo=$(awk -v p="$CAL_PCT" -v d="$dlo" 'BEGIN{printf "%.0f", p + d}')
  est_hi=$(awk -v p="$CAL_PCT" -v d="$drift" 'BEGIN{printf "%.0f", p + d}')
  echo "   标定结论：mbs=$CAL_MBS / seq=$CAL_SEQ（试了 $CAL_TRIED 档），该档实测峰值 ${CAL_PEAK_MIB} MiB = ${CAL_PCT}%"
  echo "   预计长稳峰值 ${est_lo}~${est_hi}%（4 步试跑比 60 分钟少算 ${dlo}~${drift} 个点的碎片漂移，判定带 ${lo}~${hi}%）"
  if awk -v p="$CAL_PCT" -v a="$lo" -v b="$acc_hi" 'BEGIN{exit !(p>=a && p<=b)}'; then
    echo "   ✓ 落在 ${lo}~${acc_hi}% 选档带内"
  elif awk -v p="$CAL_PCT" -v a="$lo" 'BEGIN{exit !(p<a)}'; then
    if (( est_hi >= lo )); then
      echo "   · 试跑值 ${CAL_PCT}% 低于选档带，但按漂移估计能进 ${lo}~${hi}% 判定带，先按此档跑"
    else
      echo "   ! [WARN] 最高只能到 ${CAL_PCT}%（预计长稳 ${est_lo}~${est_hi}%，仍低于 ${lo}%）"
      echo "     该模型/seq 在可行 mbs 内填不满显存：调大 HIGHMEM_SEQ 或换更大模型（HIGHMEM_MODEL=70b）"
    fi
  else
    echo "   ! [WARN] ${CAL_PCT}% 已超过 ${acc_hi}% 选档上限：长稳 60 分钟可能 OOM，建议把 HIGHMEM_TARGET_PCT 调低"
  fi
  if [ "$CAL_SEQ" != "$SEQ" ]; then
    echo "   · 已自动把 seq 从 $SEQ 细化到 $CAL_SEQ（只改 seq、mbs 不变，就是为了把落点挪进目标占用）"
  elif [ "${CAL_SUGGEST_SEQ:-0}" -gt 0 ] && [ "$CAL_SUGGEST_SEQ" != "$CAL_SEQ" ]; then
    echo "   · 处方：想让试跑值直接落进 ${lo}~${acc_hi}%，用 HIGHMEM_SEQ=$CAL_SUGGEST_SEQ（mbs 仍 $CAL_MBS）；"
    echo "     若本轮长稳判 WARN_LOW，该值会写进 .warn_highmem_low，下次没显式给 HIGHMEM_SEQ 时自动套用"
  fi
  return 0
}

soak_highmem() {                         # 显存高占用（不拉满）：TP8，单机 8B/双机 110B（或 70B）
  local MINS="${HIGHMEM_MINUTES:-60}"
  SOAK_TARGET_MINS="$MINS" SOAK_LABEL=highmem
  # 冒烟（<10 分钟）不参与 75~92% 判定：判定带是按 60 分钟定的，短跑时碎片漂移还没发生，
  # 拿它判 WARN_LOW 会把「配置对了、只是跑得短」误报成未达标，还会撤掉哨兵。
  local SMOKE=0
  (( MINS < 10 )) && SMOKE=1
  # 哨兵检查放在最前面：标定要花 3~5 分钟试跑，跑完再发现「已完成」就白烧了
  if [ "$FORCE" != 1 ] && [ -f "$LOGDIR/.ok_highmem" ]; then
    echo "→ 跳过显存高占用长稳（已完成哨兵 .ok_highmem；FORCE=1 可重跑）"; return 0
  fi
  local MODE="${HIGHMEM_MODEL:-auto}" NAME ARR TP PP
  [ "$MODE" = auto ] && { (( NNODES > 1 )) && MODE=110b || MODE=8b; }
  case "$MODE" in
    8b)   NAME=8B   ARR=LLAMA_8B   TP=8 PP=1 ;;
    70b)  NAME=70B  ARR=LLAMA_70B  TP=8 PP=2 ;;
    110b) NAME=110B ARR=LIMIT_110B TP=8 PP=2 ;;
    *) echo "!! HIGHMEM_MODEL=$MODE 不合法，只能是 auto|8b|70b|110b"; return 1 ;;
  esac
  local SEQ_GIVEN=0
  [ -n "${HIGHMEM_SEQ:-}" ] && SEQ_GIVEN=1
  local SEQ="${HIGHMEM_SEQ:-}"
  [ -n "$SEQ" ] || { (( NNODES > 1 )) && SEQ=4096 || SEQ=16384; }
  # 上一轮判了 WARN_LOW 时会留下 suggest_seq（按实测斜率算的处方）。没显式给 HIGHMEM_SEQ
  # 就自动套用它 —— 否则同样的参数会再跑出同样偏低的占用，形成「重跑还是 40 GB」的死循环。
  if [ -z "${HIGHMEM_SEQ:-}" ] && [ -s "$LOGDIR/.warn_highmem_low" ]; then
    local sug
    sug=$(sed -n 's/.*suggest_seq=\([0-9]\+\).*/\1/p' "$LOGDIR/.warn_highmem_low" 2>/dev/null | head -1)
    if [ -n "${sug:-}" ] && [ "$sug" -gt 0 ] && [ "$sug" != "$SEQ" ]; then
      echo "    · 上轮 WARN_LOW 留下处方 suggest_seq=$sug，本次自动改用 seq=$sug（显式给 HIGHMEM_SEQ 可覆盖）"
      SEQ=$sug
    fi
  fi
  local DP=$(( WORLD / TP / PP ))
  if (( TP * PP > WORLD )); then
    echo "!! HIGHMEM $NAME TP${TP}×PP${PP} 超过 world=$WORLD"; return 1
  fi

  # ---- 档位来源：同模型 limit 硬上限（仅模型一致时才有）+ 同模型标定 ----
  # 旧逻辑在这里把「limit 阶段（单机 MOE_B）的 tokens/GPU 上限 ÷ seq」当 mbs，
  # 单机必然换算成 0 → 强制 mbs=1 → 占用只有 29%。现已改为同模型标定。
  local BMBS BSRC CAP=0
  read -r BMBS BSRC < <(highmem_boundary "$SEQ" "$NAME")
  { [ "$BSRC" = limit ] && [ "${BMBS:-0}" -ge 1 ]; } && CAP=$BMBS
  if [ "$CAP" -ge 1 ]; then
    echo "    · 边界来源=limit（同模型）：seq=$SEQ 实测最大 OK mbs=$CAP，作为硬上限"
  else
    echo "    · 边界来源=$BSRC：limit 阶段扫的是 ${LIMIT_MODEL:-MOE_B}，与本次 $NAME 不是同一模型 → 不借用其上限，改由同模型标定定档"
  fi

  local REQ="${HIGHMEM_MBS:-}" USED=0 SRC="" CAL_PEAK=0
  if [ -n "$REQ" ]; then
    USED=$REQ; SRC="manual"
    if [ "$CAP" -ge 1 ] && [ "$REQ" -gt "$CAP" ]; then
      USED=$CAP
      echo "   ! [WARN] HIGHMEM_MBS=$REQ 超过同模型 limit 边界（seq=$SEQ 最大 OK mbs=$CAP），已下调为 $USED"
    fi
    echo "    · mbs=$USED 由 HIGHMEM_MBS 指定（跳过标定）；峰值判定仍生效"
  elif (( NNODES > 1 )); then
    # 双机不能各自标定：两台机器独立试跑会选出不同 mbs（显存余量、碎片状态不同），
    # 档位一错位，两个 torchrun 的 --micro-batch-size 不一致 → 直接 hang 或被看门狗杀。
    if [ "$CAP" -ge 1 ]; then
      USED=$CAP; SRC="limit"
      echo "    · 双机不自动标定（两台必须同档位）：取同模型 limit 边界 mbs=$CAP（seq=$SEQ 实测 OK）"
    else
      USED=1; SRC="none"
      echo "   ! [WARN] 双机不自动标定，且没有同模型 limit 边界 → 按 mbs=1 跑；占用会偏低，预期 WARN_LOW"
      echo "     要固定档位：两台机器上同时给同一个 HIGHMEM_MBS=<mbs>"
    fi
  elif [ "${HIGHMEM_CAL:-1}" = 1 ]; then
    highmem_calibrate "$NAME" "$ARR" "$TP" "$PP" "$SEQ" "$CAP" "$SEQ_GIVEN" || return 1
    USED=$CAL_MBS; CAL_PEAK=$CAL_PEAK_MIB; SRC="calib"
    if [ "${CAL_SEQ:-0}" -gt 0 ] && [ "$CAL_SEQ" != "$SEQ" ]; then
      echo "    · 采纳标定细化的 seq=$CAL_SEQ（原 $SEQ），长稳按 mbs=$USED / seq=$CAL_SEQ 跑"
      SEQ=$CAL_SEQ
    fi
  else
    echo "!! HIGHMEM_CAL=0 但没给 HIGHMEM_MBS：档位没有来源"
    echo "   要么留空让脚本标定（HIGHMEM_CAL 默认 1），要么显式指定 HIGHMEM_MBS"
    return 1
  fi
  if (( USED < 1 )); then echo "!! 档位解析失败（mbs=$USED）"; return 1; fi

  # 显式钉了 mbs、没给 seq 时，seq 会落到默认值。这一档的落点是可以先算出来的：
  # 静态（权重+梯度+优化器 16 B/参数 ÷ TP）+ 11-10 真机拟合斜率 1.3822 MiB/token（仅 8B TP8 bf16）。
  # 11-10 实测 mbs=5/seq=16384 = 89.9%，离 92% 判定上限只剩 2.1 个点 → 60 分钟必越界。
  # 这里只提示、不改配置：改了 seq 就不是用户点的那一档了。
  if [ "$SEQ_GIVEN" = 0 ] && [ "$SRC" = manual ] && [ "$NAME" = 8B ] && [ "$TP" = 8 ] \
     && (( NNODES == 1 )) && [ "${HIGHMEM_SEQ_QUIET:-0}" != 1 ]; then
    local st_gib pred pct_sug seq_sug tgt_mib cap_hi
    cap_hi=$(( HIGHMEM_BAND_HI - ${HIGHMEM_DRIFT:-8} ))     # 选档上限，与 highmem_calibrate 内一致
    st_gib=$(dense_est 8.03e9 8 1)
    tgt_mib=$(( VRAM_MIB * HIGHMEM_TARGET_PCT / 100 ))
    pred=$(awk -v g="$st_gib" -v m="$USED" -v s="$SEQ" -v a=1.3822 -v v="$VRAM_MIB" \
      'BEGIN{printf "%.1f", (g*1024 + a*m*s)/v*100}')
    echo "    · 落点估算：静态 ${st_gib} GiB + 1.3822 MiB/token × $(( USED * SEQ )) = ${pred}%（经验斜率，仅 8B TP8 bf16）"
    if awk -v p="$pred" -v b="$cap_hi" 'BEGIN{exit !(p>b)}'; then
      seq_sug=$(awk -v g="$st_gib" -v m="$USED" -v t="$tgt_mib" -v a=1.3822 -v s0="$SEQ" 'BEGIN{
        tok = (t - g*1024) / a; if (tok <= 0) { print s0; exit }
        x = int((tok / m) / 1024 + 0.5) * 1024;      # 就近取 1024 的整数倍：向上取整会多漂 6 GiB
        if (x < 1024) x = 1024; if (x > s0) x = s0;
        print x }')
      pct_sug=$(awk -v g="$st_gib" -v m="$USED" -v s="$seq_sug" -v a=1.3822 -v v="$VRAM_MIB" \
        'BEGIN{printf "%.1f", (g*1024 + a*m*s)/v*100}')
      echo "   ! [WARN] 估算 ${pred}% 已超过选档上限 ${cap_hi}%：60 分钟大概率越过 ${HIGHMEM_BAND_HI}% 判定上限（贴边严重时会 OOM）"
      echo "     想钉住 mbs=$USED 又要落在带内：加 HIGHMEM_SEQ=$seq_sug（估算 ${pct_sug}%）"
    fi
  fi
  local GBS=$(( USED * DP )); [ "$GBS" -gt 0 ] || GBS=$USED
  local TAG="HIGHMEM_${NAME}_tp${TP}_pp${PP}_mbs${USED}_s${SEQ}_bf16"
  local LOG="$LOGDIR/w${WORLD}_${TAG}.log"

  # 标定会留下 memused-*highmem-cal-*.csv；正式档用 memused-*highmem.csv，
  # 跑前清掉同名旧文件，避免峰值判定读到上一次运行的曲线。
  rm -f "$LOGDIR/memused-w${WORLD}_highmem.csv" "$LOGDIR/.warn_highmem_low" "$LOGDIR/.warn_highmem_hot"

  echo "########## 显存高占用长稳 $MINS 分钟（$NAME TP${TP}×PP${PP}×DP${DP}，mbs=$USED，seq=$SEQ，"
  echo "##########   GBS=$GBS，不带 recompute；档位来源=$SRC，目标占用 ${HIGHMEM_TARGET_PCT}%（判定带 ${HIGHMEM_BAND_LO}~${HIGHMEM_BAND_HI}%）##########"
  SENTINEL_KEY=highmem
  DMON_TAG="highmem-w${WORLD}_${NAME}_mbs${USED}_s${SEQ}"
  SOAK_PREFIX="highmem-"
  local H="$HIGHMEM_HEALTH" HAVE_H=0
  if [ "$H" = 1 ]; then
    health_snap "$LOGDIR/highmem-health-before.json"
    HAVE_H=1
  fi
  TELEMETRY=1
  STEPS=100000
  # 本档必须真的把激活内存吃满，所以关掉 recompute（seq≥64K 时 run_train 默认会开，
  # 开了等于把激活压回去，峰值占用会被系统性低估，横幅里「不带 recompute」也就成了假话）
  NO_RECOMPUTE=1
  mem_wait_idle 30
  mem_sampler_start "$LOGDIR/memused-w${WORLD}_highmem.csv"
  # OOM 在本档是预期边界：记 OOM_BOUNDARY、不 dual_stop、退出码 0（ALLOW_OOM 只作用于这一条命令）
  local REQUIRE="$SOAK_MIN_ITERS"
  (( SMOKE == 1 )) && REQUIRE=1          # 冒烟只验显存能不能上去，不验步数
  ALLOW_OOM=1 OOM_AS_BOUNDARY=1 SOAK_REQUIRE_ITERS="$REQUIRE" \
    run_train "$TAG" "$ARR" "$TP" "$PP" "$USED" "$SEQ" "$GBS" --exit-duration-in-mins="$MINS"
  local rc=$?
  mem_sampler_stop
  NO_RECOMPUTE=0
  if [ "$HAVE_H" = 1 ]; then
    health_snap "$LOGDIR/highmem-health-after.json"
    health_delta_json highmem-
  fi

  local njit ss jit st ec=""
  njit=$(grep -c "elapsed time per iteration" "$LOG" 2>/dev/null); njit=${njit:-0}
  ss=$(step_seconds "$LOG")
  jit=$(step_jitter "$LOG")
  if [ -f "$LOGDIR/.oom_w${WORLD}_${TAG}" ]; then
    st=OOM_BOUNDARY; ec=OOM; rc=0          # OOM 是本档的预期边界：不 dual_stop、退出码 0
  elif [ -f "$LOGDIR/.ok_highmem" ]; then
    st=OK
  else
    ec=$(err_class "$LOG")
    if [ "$njit" -lt "$SOAK_MIN_ITERS" ] && [ "$njit" -gt 0 ]; then st=PARTIAL; else st=FAILED; fi
    rc=1
  fi

  # ---- 峰值判定：本档唯一的实质判据 ----
  # 旧逻辑只看 .ok_highmem 在不在，于是「跑完了但只占 29%」被记成 OK（11-10 单机实测
  # 41696/143771 = 29.0%，而 RUNBOOK 自己写的是 75~92% 才算 PASS）。现在按实测峰值收口，
  # 且 WARN_LOW 会撤掉 .ok_highmem，让下一次 --phase soak 重跑而不是静默通过。
  if [ "$st" = OK ] && (( SMOKE == 1 )); then
    st=SMOKE
    rm -f "$LOGDIR/.ok_highmem"        # 冒烟不写完成哨兵，免得挡住正式 60 分钟长稳
  fi
  local PEAK PCT
  PEAK=$(mem_peak_mib "$LOGDIR/memused-w${WORLD}_highmem.csv")
  PCT=$(awk -v p="$PEAK" -v v="$VRAM_MIB" 'BEGIN{printf "%.1f", (v>0 ? p/v*100 : 0)}')
  if [ "$st" = OK ]; then
    if awk -v p="$PCT" -v a="$HIGHMEM_BAND_LO" 'BEGIN{exit !(p<a)}'; then
      st=WARN_LOW
      rm -f "$LOGDIR/.ok_highmem"
      printf 'peak_pct=%s peak_mib=%s visible_mib=%s band=%s-%s mbs=%s seq=%s tokens_per_gpu=%s source=%s suggest_seq=%s\n' \
        "$PCT" "$PEAK" "$VRAM_MIB" "$HIGHMEM_BAND_LO" "$HIGHMEM_BAND_HI" "$USED" "$SEQ" \
        "$(( USED * SEQ ))" "$SRC" "${CAL_SUGGEST_SEQ:-0}" > "$LOGDIR/.warn_highmem_low"
      echo "   ! [WARN] 未达高占用：实测峰值 ${PEAK} MiB = ${PCT}%（判定下限 ${HIGHMEM_BAND_LO}%）"
      echo "     本档不能作为「高显存压力」结论：mbs=$USED × seq=$SEQ = $(( USED * SEQ )) tokens/卡 太少"
      if [ "${CAL_SUGGEST_SEQ:-0}" -gt 0 ] && [ "$CAL_SUGGEST_SEQ" != "$SEQ" ]; then
        echo "     下次会自动改用 HIGHMEM_SEQ=$CAL_SUGGEST_SEQ（已写入哨兵）；显式给 HIGHMEM_SEQ 可覆盖"
      else
        echo "     调法：放开 HIGHMEM_MBS 让脚本标定，或调大 HIGHMEM_SEQ；已撤销 .ok_highmem，下次会重跑"
      fi
    elif awk -v p="$PCT" -v b="$HIGHMEM_BAND_HI" 'BEGIN{exit !(p>b)}'; then
      st=WARN_HOT
      printf 'peak_pct=%s peak_mib=%s visible_mib=%s band=%s-%s mbs=%s seq=%s tokens_per_gpu=%s source=%s\n' \
        "$PCT" "$PEAK" "$VRAM_MIB" "$HIGHMEM_BAND_LO" "$HIGHMEM_BAND_HI" "$USED" "$SEQ" \
        "$(( USED * SEQ ))" "$SRC" > "$LOGDIR/.warn_highmem_hot"
      echo "   ! [WARN] 过于贴边：实测峰值 ${PEAK} MiB = ${PCT}%（判定上限 ${HIGHMEM_BAND_HI}%）"
      echo "     60 分钟长稳可能因碎片 OOM，结论不稳；建议 HIGHMEM_TARGET_PCT 下调（如 80）"
    else
      echo "   ✓ 峰值占用 ${PEAK} MiB = ${PCT}%，落在 ${HIGHMEM_BAND_LO}~${HIGHMEM_BAND_HI}% 判定带内"
    fi
  elif [ "$st" = SMOKE ]; then
    echo "   · 冒烟 ${MINS} 分钟：峰值 ${PEAK} MiB = ${PCT}%（漂移未发生，不做 ${HIGHMEM_BAND_LO}~${HIGHMEM_BAND_HI}% 判定）"
    echo "     正式判定请跑 HIGHMEM_MINUTES=60；本轮未写 .ok_highmem"
  elif [ "$st" = OOM_BOUNDARY ]; then
    echo "   ○ OOM_BOUNDARY：峰值 ${PEAK} MiB（${PCT}%）只是崩溃前的瞬时值，不作为占用结论"
  fi

  local CALM=0; [ "$SRC" = calib ] && CALM=$USED
  [ -n "$REQ" ] || REQ=$USED
  highmem_record "$st" "$TAG" "$NAME" "$TP" "$PP" "$REQ" "$USED" "$SEQ" "$GBS" \
                 "$njit" "$ss" "$jit" "$SRC" \
                 "$([ "$CAP" -ge 1 ] && echo "$CAP" || echo none)" "$ec" "$CALM" "${CAL_PEAK:-0}"
  echo "    · 记录 → highmem.jsonl（status=$st，peak_pct=${PCT}%）；显存曲线 → memused-w${WORLD}_highmem.csv"
  echo "  遥测：$LOGDIR/dmon-${DMON_TAG}.csv；硬件增量：highmem-health-delta.json（与另两条独立）"
  SENTINEL_KEY=""; DMON_TAG=""; SOAK_PREFIX=""
  sync_logs
  [ "$rc" -eq 0 ]
}

# ================================================================ MFU 峰值测量（--phase mfu）
# 目标：回答「单机 / 双机的峰值 MFU 是多少」，不是「某一个配置的 MFU 是多少」。
# 阶段：A 粗扫（并行度 × GBS）→ B 细扫（mbs × accum）→ C seq 扫描 → D 复测（前 3 名 × MFU_REPEAT）。
#
# 口径先定死，否则数字没法比：
#   · 头条 MFU† = Megatron 自报 throughput per GPU ÷ 989.5 TFLOPS
#     （989.5 = H200 SXM bf16 稠密峰值；Megatron 的 FLOPs 公式含 attention，故长序列偏高）
#   · 同时记 MFU(6N) = 6·N·tokens/step ÷ world ÷ 989.5，只算线性层，是保守下界
#   · 每档 MFU_STEPS 步（默认 40），重复 MFU_REPEAT 次（默认 3）取中位
#   · CV > MFU_CV_MAX(2%) 或 throttle≠0 的档不进峰值
#   · 硬天花板：04-hw 的 bf16 GEMM 实测 690 TFLOP/s = 理论 69.7%，训练峰值不可能超过它
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

# ------------------------------------------------ 显存边界扫描（limit 主体，limit 阶段与 soak/highmem 前置共用）
# 全部副作用封在子 shell：LIMIT_JSONL/ALLOW_OOM/STEPS/LOGDIR 不会泄漏到调用方
#（soak 的 run_train 依赖自己设置的这些变量；LOGDIR 必须指回 06-train-limit 而非调用方的阶段目录）。
# rc：0=正常（含 args 跳过）；2=阶段目录创建失败。
run_limit_scan() {
  (
    LOGDIR=$(lc_run_stage_dir limit) || exit 2
    export LOGDIR
    echo "    · limit 产物 → $LOGDIR"
    ensure_args_check
    LIMIT_JSONL=1 ALLOW_OOM=1
    STEPS="${LIMIT_STEPS:-5}"
    if (( NNODES > 1 )); then
      LIMIT_MODEL=110B; LIMIT_EP=1; LIMIT_TP=8; LIMIT_PP=2
      LIMIT_CUR=( "${LIMIT_110B[@]}" )
      LIMIT_STATIC_GB=$(dense_est 110e9 "$LIMIT_TP" "$LIMIT_PP")
      SEQ_LIST="8192 16384 32768"
      echo "########## 配置 A（合成 110B）TP8×PP2×DP1；静态≈${LIMIT_STATIC_GB} GiB/卡，余量≈$(awk -v s="$LIMIT_STATIC_GB" -v v="$VRAM_GIB" 'BEGIN{printf "%.1f", v-s}') GiB ##########"
      echo "    · 口径：GBS=mbs×dp、PP2 时 nmib=1，气泡约 50%，仅用于显存边界，不作性能结论"
    else
      MOE_ATTN=mla; MOE_EXPERIMENTAL=0
      args_gate_moe || { echo "    → SKIP：单机 limit 用配置 B（MLA-MoE），关键/实验性 flag 缺失"; exit 0; }
      # 单机 limit 用配置 B 原版（24 层/32 专家，静态≈84.7 GiB/卡）：
      # 原设计用配置 B 放大版（48 层/64 专家），但 EP8 下静态≈225.7 GiB 已超 140.4 GiB 显存
      #（11-08 实测 static_margin=-85.3），任何 mbs 都必然 OOM，扫不出任何边界数据。
      # 改用配置 B：84.7 GiB 起步，靠激活扫 mbs/seq 到 OOM 才有意义的边界。
      LIMIT_MODEL=MOE_B; LIMIT_EP=8; LIMIT_TP=1; LIMIT_PP=1
      LDISP=alltoall; disp_supported alltoall || LDISP=allgather
      LIMIT_CUR=( "${MOE_B[@]}" )
      if flag_in_help enable-experimental; then
        LIMIT_CUR+=( --enable-experimental ); MOE_EXPERIMENTAL=1
      else
        echo "   ! [WARN] 单机 limit 的 MLA 配置缺 --enable-experimental，预计会失败（继续，日志会给出原因）"
      fi
      mapfile -t LIMIT_CUR < <(filter_args "${LIMIT_CUR[@]}" \
        --expert-model-parallel-size=8 --moe-token-dispatcher-type="$LDISP")
      LIMIT_STATIC_GB=$(moe_est "${MOE_B_EST[@]}" 1 1 8 "$MOE_B_VOCAB" | awk '{print $1}')
      SEQ_LIST="8192 16384 32768 65536 131072"
      echo "########## 配置 B（24 层 / 32 专家，EP8×TP1，dispatcher=$LDISP）；静态≈${LIMIT_STATIC_GB} GiB/卡，余量≈$(awk -v s="$LIMIT_STATIC_GB" -v v="$VRAM_GIB" 'BEGIN{printf "%.1f", v-s}') GiB ##########"
    fi
    BEST_MBS=0
    echo "########## 第一段：seq=4096，mbs 1→…→LIMIT_MAX_MBS=$LIMIT_MAX_MBS ##########"
    for mbs in 1 2 4 8 16 32; do
      (( mbs <= LIMIT_MAX_MBS )) || break
      TAG="LIMIT${LIMIT_MODEL}_tp${LIMIT_TP}_pp${LIMIT_PP}_mbs${mbs}_s4096"
      GBS=$(( mbs * (WORLD / LIMIT_TP / LIMIT_PP) ))
      run_train "$TAG" LIMIT_CUR "$LIMIT_TP" "$LIMIT_PP" "$mbs" 4096 "$GBS"; rc=$?
      if [ -f "$LOGDIR/.oom_w${WORLD}_${TAG}" ]; then
        echo "   → mbs=$mbs 触及显存边界（OOM_BOUNDARY），停止放大 mbs"
        break
      fi
      if [ "$rc" -ne 0 ]; then
        echo "   → mbs=$mbs 非 OOM 失败（rc=$rc），停止；看日志，不掩盖"
        break
      fi
      BEST_MBS=$mbs
    done
    echo "########## 第二段：mbs=$BEST_MBS 固定，seq 8192→…→LIMIT_MAX_SEQ=$LIMIT_MAX_SEQ ##########"
    if [ "$BEST_MBS" -gt 0 ]; then
      for seq in $SEQ_LIST; do
        if (( seq > LIMIT_MAX_SEQ )); then echo "   · seq=$seq > LIMIT_MAX_SEQ，停止"; break; fi
        TAG="LIMIT${LIMIT_MODEL}_tp${LIMIT_TP}_pp${LIMIT_PP}_mbs${BEST_MBS}_s${seq}"
        GBS=$(( BEST_MBS * (WORLD / LIMIT_TP / LIMIT_PP) ))
        run_train "$TAG" LIMIT_CUR "$LIMIT_TP" "$LIMIT_PP" "$BEST_MBS" "$seq" "$GBS"; rc=$?
        if [ -f "$LOGDIR/.oom_w${WORLD}_${TAG}" ]; then
          echo "   → seq=$seq 触及显存边界（OOM_BOUNDARY），停止"
          break
        fi
        if [ "$rc" -ne 0 ]; then
          echo "   → seq=$seq 非 OOM 失败（rc=$rc），停止；看日志，不掩盖"
          break
        fi
      done
    else
      echo "   ! 首档 mbs=1 未成功（原因见上方：OOM 边界或其它失败），跳过 seq 段"
    fi
    limit_summary
    echo "    · 明细 → $LOGDIR/limit.jsonl；汇总 → $LOGDIR/limit-summary.json"
    sync_logs
  )
}

# ------------------------------------------------ 双机 RDMA 门禁（训练前）
if gate_needed; then
  echo "########## RDMA gate：8MB allreduce，timeout 60s ##########"
  if ! rdma_gate; then
    echo "!! RDMA 门禁未通过：本阶段（$PHASE）不启动，rc=2（档位未开跑，不触发 dual_stop）"
    lc_run_stage_done "$PHASE" "$START_TS" "RDMA_GATE_FAIL"
    exit 2
  fi
fi

case "${PHASE:-}" in
# ------------------------------------------------------------ 参数自检
argscheck)
  ensure_args_check
  echo "    · 产物：$ARGS_HELP（--help 原文）、$ARGS_CHECK"
  grep -oE '"status": "[A-Z]+"|"missing": \[[^]]*\]|"critical_missing": \[[^]]*\]' "$ARGS_CHECK" | sed 's/^/      /'
  if args_gate_moe; then
    echo "    · 关键 flag 齐备，moe / moe-dp / limit(单机 MoE) 可执行"
  else
    echo "    · 关键 flag 缺失：moe / moe-dp / limit(单机 MoE) 会直接 SKIP，不会带错参数硬跑"
  fi
  ;;

# ------------------------------------------------------------------ 硬件基线
hw)
  TELEMETRY=1 dmon_start "$LOGDIR/dmon.csv"
  run_probe hw_probe.py hw --out /workspace/logs/hw.jsonl --secs "${PROBE_SECS:-5}"
  dmon_stop
  echo "    · 指标 → $LOGDIR/hw.jsonl（05_report 的微基准表读它）"
  ;;

# ------------------------------------------------- 单机 collective 全类型扫描
coll)
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
  ;;

# ------------------------------------------------- 双机 collective 直测（nccl-tests 等价）
nccl)
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
  ;;

# ------------------------------------------------------ 单机 8B：S1–S4
s8b)
  echo "########## S1 micro batch 上界（TP8，num_micro_batches 固定 8 让算力持续饱和）##########"
  mark_baseline "w${WORLD}_8B_tp8_pp1_mbs1_s4096_gb8_bf16" "S1 首档：mbs1/seq4096/GBS8，每次原样复跑做基线"
  for mbs in 1 2 4; do
    run_train "8B_tp8_pp1_mbs${mbs}_s4096_gb$(( mbs * 8 ))_bf16" LLAMA_8B 8 1 "$mbs" 4096 $(( mbs * 8 ))
  done
  echo "########## S2 并行策略（GBS 固定 32 才可比）##########"
  for spec in "1 1" "4 1" "8 1"; do
    set -- $spec
    run_train "8B_tp$1_pp1_mbs2_s4096_gb32_bf16" LLAMA_8B "$1" 1 2 4096 32
  done
  echo "########## S3 长序列（TP8 mbs1 GBS8：单机唯一能逼出显存上限的方向）##########"
  for seq in 2048 8192 32768 131072; do
    run_train "8B_tp8_pp1_mbs1_s${seq}_gb8_bf16" LLAMA_8B 8 1 1 "$seq" 8
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
    run_train "8B_tp8_pp1_mbs4_s4096_gb8_bf16" LLAMA_8B 8 1 4 4096 8
    run_train "8B_tp8_pp1_mbs4_s4096_gb8_fp8"  LLAMA_8B 8 1 4 4096 8 "${FP8_OPTS[@]}"
    # 防回归自检：Megatron 参数打印里 args.fp8 必须是 e4m3/hybrid，不能还是 None。
    # 不中断整轮（exit 会让 lc_run_stage_done 不写 index.md，制造新的元数据缺口），
    # 改为落标记 + index 口径说明 + 醒目 FAIL，重跑后一眼可见。
    FP8_LOG="$LOGDIR/w${WORLD}_8B_tp8_pp1_mbs4_s4096_gb8_fp8.log"
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
    run_train "8B_tp8_pp1_mbs4_s4096_gb8_bf16" LLAMA_8B 8 1 4 4096 8
  fi
  ;;

# ------------------------------------- 双机 8B TP8×DP2：机间带宽门禁
dx8b)
  echo "########## D1 训练对照：GBS=32（8 卡口径）##########"
  run_train "8B_tp8_pp1_mbs2_s4096_gb32_bf16" LLAMA_8B 8 1 2 4096 32
  echo "判据：与单机同 GBS 的 gb32 比，理想比值 0.5；≤0.6 视为机间链路可用。"
  echo "      1.0 左右 = 加一台机器吞吐不涨；>2.0 = 负收益。"
  echo "      口径：双机 accum=8 / 单机 accum=16（mbs2×dp1），0.5 只是强扩展理想比值，非同 accum 比较。"
  note_index "D1 accum" "D1（8B GBS32）：双机 accum=8、单机 accum=16，理想 0.5 只是强扩展比值，非同 accum 比较。"
  sync_logs
  ;;

# ------------------------------------- 双机 70B TP8×PP2（跨机只传激活）
d70b)
  if (( WORLD < 16 )); then
    echo "!! d70b 需要 world≥16（2 机 × 8 卡），当前 world=$WORLD、NNODES=$NNODES。"
    echo "   这是并行度/机器数配置错误，不是硬件 OOM：70B 在 8 卡上必然 OOM，且容易被误读成显卡故障。"
    exit 2
  fi
  echo "########## D2 流水线气泡（m=4 时 bubble≈25%）##########"
  mark_baseline "w${WORLD}_70B_tp8_pp2_mbs1_s2048_gb4_bf16" "D2 首档：mbs1/seq2048/GBS4，每次原样复跑做基线"
  for m in 4 16; do
    run_train "70B_tp8_pp2_mbs1_s2048_gb${m}_bf16" LLAMA_70B 8 2 1 2048 "$m"
  done
  echo "########## D3 序列长度与 OOM 边界（OOM 记 OOM_BOUNDARY，不 dual_stop）##########"
  ALLOW_OOM=1 OOM_AS_BOUNDARY=1
  LIMIT_MODEL=70B LIMIT_EP=1
  LIMIT_STATIC_GB=$(dense_est 70.55e9 8 2)
  for seq in 4096 8192; do
    run_train "70B_tp8_pp2_mbs1_s${seq}_gb12_bf16" LLAMA_70B 8 2 1 "$seq" 12
  done
  ALLOW_OOM=0 OOM_AS_BOUNDARY=0
  sync_logs
  ;;

# ------------------------------------- MoE / MLA：单机 EP8×TP1 / EP2×TP4 + 双机 EP16
moe)
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
  ;;

# ------------------------------------- 仅双机：EP8×DP2（专家并行留机内，梯度跨机 all-reduce）
moe-dp)
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
  ;;

# ------------------------------------- 显存 / 规模墙：扫 mbs→seq 到 OOM（OOM 记为边界）
limit)
  run_limit_scan
  ;;

# ---------------------------------------------------------------- 存储与数据管线
io)
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
  ;;

# ---------------------------------------------------------------- 故障注入
fault)
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
  ;;

soak)
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
  # 直接用 04 创建的 run（如先 limit 再 soak）会误判成「容器没有该参数」（11-08 实测）。
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
  # highmem 依赖本 run 的 limit 边界（highmem_boundary 读 06-train-limit/limit.jsonl）：
  # 缺失时自动先扫一轮 limit（同 RUN_ID），不再要求单独手工跑 --phase limit（11-09 合并）。
  case " ${ORDER[*]} " in
    *" highmem "*)
      if [ -s "$LC_RUN_DIR/06-train-limit/limit.jsonl" ]; then
        echo "    · 已有 limit 边界数据（06-train-limit/limit.jsonl），highmem 直接续用"
      else
        echo "########## highmem 前置：本 run 尚无 limit.jsonl，先自动扫显存边界 ##########"
        run_limit_scan || { echo "!! limit 前置扫描失败（rc=$?），highmem 没有边界保护，中止 soak"; exit 2; }
        lc_run_stage_done limit "$(date +%s)" "PASS(边界前置)"
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
  ;;

mfu)
  mfu_phase
  ;;

report)
  RP="$HERE/05_report.py"; [ -f "$RP" ] || RP="$ROOT/05_report.py"
  # 读整次 run（各阶段目录由 05_report.py 递归收集），报告落到 10-report/
  python3 "$RP" --logs "$LC_RUN_DIR" 2>&1 | tee "$LOGDIR/report.txt" "$LOGDIR/report.md" \
    || echo "!! 05_report.py 不在 $HERE 也不在 $ROOT"
  echo "    · 报告 → $LOGDIR/report.txt / report.md"
  echo "    · run 目录：$LC_RUN_DIR"
  ;;

esac

# 每个阶段结束写一条到 index.md（阶段、耗时、结果、产出）
lc_run_stage_done "$PHASE" "$START_TS"
echo "run 目录：$LC_RUN_DIR"
