#!/usr/bin/env bash
# 公共函数库：被 scripts/*.sh source。
set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 现场配置必须最先加载：config/env.sh / image.env 全部用 ${VAR:-default}，
# 因此 cluster.env 先设的值会被保留，实现「现场配置优先于项目默认值」。
# shellcheck source=../config/cluster.env
source "$LIB_DIR/../config/cluster.env"
# shellcheck source=../config/env.sh
source "$LIB_DIR/../config/env.sh"
# shellcheck source=../config/image.env
source "$LIB_DIR/../config/image.env"

# =============================================================================
# 关键变量兜底（兼容旧版 config/env.sh）
#
# 现场升级脚本时常见「scripts/ 已更新、config/env.sh 还是旧的」——而所有脚本都在
# set -u 下运行，一旦引用未定义变量就直接以「未绑定的变量」退出（例如
# 03 报 `LOGGING_STEPS: 未绑定的变量`、13 拼 REMOTE_ENV 时报 `CUDA_COMPAT`）。
# 这里统一补齐新增开关的默认值，让旧配置也能跑起来。
# =============================================================================
export CUDA_COMPAT="${CUDA_COMPAT:-auto}"
export CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-}"
export CUDA_COMPAT_MIN_DRIVER="${CUDA_COMPAT_MIN_DRIVER:-580}"
export MASTER_ADDR_KEEP="${MASTER_ADDR_KEEP:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export LOG_THROUGHPUT="${LOG_THROUGHPUT:-true}"
export DMON="${DMON:-1}"
export DMON_INTERVAL="${DMON_INTERVAL:-5}"
export DMON_METRICS="${DMON_METRICS:-pucvmt}"
export DURATION_MINS="${DURATION_MINS:-0}"
export DURATION_ITERS_CAP="${DURATION_ITERS_CAP:-100000}"

# 容器命令统一入口：现场 nerdctl 必须带 --namespace，docker 不需要。
# CTR_CMD 用于拼装可直接复制执行的命令（日志里不出现内部函数名）。
if [[ "$CONTAINER_ENGINE" == "nerdctl" ]]; then
  CTR_CMD=(nerdctl --namespace "$CONTAINER_NAMESPACE")
else
  CTR_CMD=(docker)
fi

ctr() {
  "${CTR_CMD[@]}" "$@"
}

log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '[%s] [WARN] %s\n' "$(date '+%F %T')" "$*" >&2; }
die()  { printf '[%s] [ERROR] %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

# 未捕获失败的兜底诊断。
# set -e 会让脚本在任一命令非 0 时退出，但**不打印任何信息**——尤其是
# `cmd | head -1` 这类 SIGPIPE（退出码 141）场景，表现为「日志突然断了、没有报错」，
# 极难定位。装上 ERR trap 后至少能拿到文件名、行号和那条命令。
install_err_trap() {
  set -E
  trap 'rc=$?; printf "[%s] [FATAL] %s 第 %s 行失败（退出码 %s）：%s\n" \
        "$(date "+%F %T")" "${0##*/}" "$LINENO" "$rc" "${BASH_COMMAND:-未知}" >&2' ERR
}

# =============================================================================
# 单机 MASTER_ADDR 归一化
#
# PyTorch 的 rendezvous 以 MASTER_ADDR:MASTER_PORT 为汇聚点：node_rank 0 在该地址
# 上 listen，其余 rank 连它。单机若 cluster.env 里填的是**本机不可达**的地址
# （别的机器、已下线的网卡、容器里看不到的网段），客户端会一直重试到 rendezvous
# 超时（默认 600s）——表现就是「日志停在 c10d 之后不动，且没有任何报错」。
# `megatron pt --help` 同样走 torch.distributed.run，所以也会一起卡住。
#
# 因此 NNODES=1 统一改用 127.0.0.1。确实要用真实 IP（模拟多机首节点、验证网卡）
# 时设 MASTER_ADDR_KEEP=1。
# =============================================================================
if [[ "${NNODES:-1}" == "1" ]] && [[ "${MASTER_ADDR_KEEP:-0}" != "1" ]] \
   && [[ "${MASTER_ADDR:-}" != "127.0.0.1" ]]; then
  export MASTER_ADDR_ORIG="${MASTER_ADDR:-}"
  export MASTER_ADDR=127.0.0.1
  log "单机训练：MASTER_ADDR ${MASTER_ADDR_ORIG:-<空>} -> 127.0.0.1（多机才需真实 IP；保留原值设 MASTER_ADDR_KEEP=1）"
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1（请先按 docs/AUDIT.md 准备 Megatron-SWIFT 环境）"
}

# 载入模型档案。profile 内一律使用 ${VAR:-default}，因此外部环境变量优先。
load_profile() {
  local profile="${1:-${MODEL_PROFILE:-27b}}"
  case "$profile" in
    27b|qwen38_27b)            source "$LIB_DIR/../config/model_27b.env" ;;
    flash_next|qwen38_flash_next) source "$LIB_DIR/../config/model_flash_next.env" ;;
    *) die "未知 MODEL_PROFILE=$profile（可选：27b / flash_next）" ;;
  esac
  export MODEL_PROFILE="$profile"
}

# 把字符串安全地拆成 argv 数组（用于 MOE_ARGS / EXTRA_ARGS / RESUME_ARGS）。
split_args() {
  local var_name="$1" out_name="$2"
  local raw="${!var_name:-}"
  local -a arr=()
  if [[ -n "$raw" ]]; then
    read -r -a arr <<< "$raw"
  fi
  eval "$out_name=(\"\${arr[@]}\")"
}

# =============================================================================
# CUDA 前向兼容（forward compatibility）
#
# 背景：镜像内的 torch 按 cu130 编译，但容器里的用户态驱动 libcuda.so 是
# nvidia-container-toolkit 从**宿主**注入的，版本恒等于宿主驱动。宿主停在
# r570（CUDA 12.8）分支时，cu130 的 runtime 直接失败：
#     nvidia-smi 正常，但 torch.cuda.is_available() == False
#
# 解法：镜像自带 cuda-compat-13-0（一套 580 的用户态 libcuda + ptxjitcompiler），
# 装在 /usr/local/cuda/compat，默认**不在** loader 路径上。把该目录前置到
# LD_LIBRARY_PATH，就能让「新用户态 + 旧内核态」共存 —— 这正是 NVIDIA 的
# forward compatibility，官方兼容矩阵里 cuda-compat-13-0 × 570+ 驱动 = Compatible。
#
#   CUDA_COMPAT=auto（默认）  仅当注入的 libcuda < CUDA_COMPAT_MIN_DRIVER 且找到 compat 目录
#   CUDA_COMPAT=force         无条件启用（auto 判定失误时的手动兜底）
#   CUDA_COMPAT=off           禁用
#
# 注意：启用后 nvidia-smi 仍会报宿主驱动版本，这是正常的——它走 NVML
# （libnvidia-ml.so，宿主注入），而 torch 用的是 libcuda.so（已被 compat 覆盖）。
# 唯一可信的验证只有 torch.cuda.is_available()。
# =============================================================================

# 从宿主注入的 libcuda.so.<major>.<minor> 反推驱动大版本；拿不到返回 1
# 注意两点：
#   1. 必须匹配「两段数字」，否则会先命中符号链接 libcuda.so.1（版本号取不出来）
#   2. 目录列表可覆盖，兼容 aarch64 / GB200 等非 x86_64 布局
_CUDA_LIB_DIRS="${_CUDA_LIB_DIRS:-/usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib/aarch64-linux-gnu /usr/lib}"
_cuda_injected_major() {
  local d f=""
  for d in $_CUDA_LIB_DIRS; do
    # || true 兜住 SIGPIPE：pipefail 下 head 提前关管道会让整个赋值非 0，set -e 直接退出
    f="$(ls -1 "$d"/libcuda.so.[0-9]*.[0-9]* 2>/dev/null | head -1 || true)"
    if [[ -n "$f" ]]; then break; fi
  done
  if [[ -z "$f" ]]; then return 1; fi
  basename "$f" | sed -nE 's/^libcuda\.so\.([0-9]+)\.[0-9]+.*/\1/p'
}

# 定位 compat 目录：显式指定优先，否则按常见布局探测
_find_cuda_compat_dir() {
  local d
  if [[ -n "${CUDA_COMPAT_DIR:-}" ]]; then
    if [[ -f "$CUDA_COMPAT_DIR/libcuda.so.1" ]]; then printf '%s' "$CUDA_COMPAT_DIR"; return 0; fi
    return 1
  fi
  for d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
    if [[ -f "$d/libcuda.so.1" ]]; then printf '%s' "$d"; return 0; fi
  done
  return 1
}

ensure_cuda_compat() {
  if [[ "${CUDA_COMPAT:-auto}" == "off" ]]; then return 0; fi
  # 上一层脚本已启用过（02→03 这类嵌套调用），子进程会继承环境，直接复用不再重复打印
  if [[ "${CUDA_COMPAT_ACTIVE:-0}" == "1" ]]; then return 0; fi

  local dir="" major="" min="${CUDA_COMPAT_MIN_DRIVER:-580}"
  if ! dir="$(_find_cuda_compat_dir)"; then
    if [[ "${CUDA_COMPAT:-auto}" == "force" ]]; then
      warn "CUDA_COMPAT=force 但未找到 cuda-compat 目录（可试 CUDA_COMPAT_DIR=/usr/local/cuda/compat）"
    fi
    return 0
  fi

  major="$(_cuda_injected_major || true)"

  if [[ "${CUDA_COMPAT:-auto}" != "force" ]]; then
    # 没有注入的 libcuda = 不在 GPU 容器里（宿主侧脚本），不动环境
    if [[ -z "$major" ]]; then return 0; fi
    # 驱动已足够新，原生路径即可
    if (( major >= min )); then return 0; fi
  fi

  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;                                   # 已包含，保持幂等
    *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  export CUDA_COMPAT_ACTIVE=1
  export CUDA_COMPAT_USED_DIR="$dir"
  log "CUDA 前向兼容：宿主注入 libcuda=${major:-unknown} < ${min}，已把 $dir 前置到 LD_LIBRARY_PATH"
}

# 多机共用一个 LOG_ROOT（共享存储）时，日志名必须带节点号，
# 否则各节点在同一秒启动会写同一个文件、互相截断。
_node_suffix() {
  if [[ "${NNODES:-1}" -gt 1 ]]; then printf '_node%s' "${NODE_RANK:-0}"; fi
}
# 当前节点的日志前缀，供需要拼自己文件名的脚本引用
log_prefix() { printf '%s%s' "$1" "$(_node_suffix)"; }

# 统一日志落盘执行；DRY_RUN=1 时只打印命令。
run_logged() {
  local tag="$1"; shift
  local name; name="$(log_prefix "$tag")"
  local log_file="$LOG_ROOT/${name}_$(date '+%Y%m%d_%H%M%S').log"
  log "日志文件: $log_file"
  log "命令: $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "DRY_RUN=1，跳过执行"
    return 0
  fi
  mkdir -p "$(dirname "$log_file")"
  set -o pipefail
  "$@" 2>&1 | tee "$log_file"
  log "完成，日志: $log_file"
  echo "$log_file" > "$LOG_ROOT/${name}_latest.path"
}

# =============================================================================
# GPU 采样：nvidia-smi dmon（宿主机侧）
#
# 为什么放这里：训练日志能看出 step 变慢，但看不出「热降频 / 功耗撞墙 / 显存缓慢
# 爬升 / NVLink 是否打满」。dmon 直接读物理卡，容器内外是同一块，因此在宿主机跑。
#
#   DMON=1（默认） 训练全程后台采样 → $LOG_ROOT/dmon_<tag>[_nodeX]_<ts>.csv
#   DMON=0         关闭
# 结束时由调用方执行 summarize_dmon，按**列名**解析（不同驱动会在 sm/mem 后插列，
# 写死列号会串列且不报错）。
# =============================================================================
DMON_PID=""
DMON_LOG=""

_dmon_probe() { nvidia-smi dmon -s "$1" -c 1 -d 1 >/dev/null 2>&1; }

start_dmon() {   # $1 = 文件名 tag（如 continue_pt_qwen38_27b）
  DMON_PID=""; DMON_LOG=""
  local tag="${1:-dmon}"
  if [[ "${DMON:-1}" != "1" ]]; then log "DMON=0，跳过 GPU 采样"; return 0; fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "找不到 nvidia-smi，跳过 GPU 采样（训练机上应存在）"; return 0
  fi
  local metrics="${DMON_METRICS:-pucvmt}" interval="${DMON_INTERVAL:-5}"
  if ! [[ "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
    warn "DMON_INTERVAL='$interval' 不是正整数秒，按 5 处理"; interval=5
  fi
  # 先探一个点：某些驱动/卡不支持 t(PCIe/NVLink)/e(ECC)，直接跑会整段静默丢失
  if ! _dmon_probe "$metrics"; then
    warn "dmon -s $metrics 不可用，退回 -s pucvm"
    metrics=pucvm
    if ! _dmon_probe "$metrics"; then
      warn "dmon 在本机不可用，跳过 GPU 采样"; return 0
    fi
  fi
  DMON_LOG="$LOG_ROOT/dmon_$(log_prefix "$tag")_$(date '+%Y%m%d_%H%M%S').csv"
  mkdir -p "$LOG_ROOT"
  nvidia-smi dmon -s "$metrics" -d "$interval" -o DT > "$DMON_LOG" 2>&1 &
  DMON_PID=$!
  log "GPU 采样已启动: $DMON_LOG（每 ${interval}s，-s $metrics）"
}

stop_dmon() {
  [[ -n "${DMON_PID:-}" ]] || return 0
  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true   # 被 kill 后 wait 非 0，必须兜住 set -e
  DMON_PID=""
}

summarize_dmon() {   # $1 = csv 路径（默认 $DMON_LOG）
  local f="${1:-${DMON_LOG:-}}"
  if [[ -z "$f" || ! -s "$f" ]]; then
    log "（无 GPU 采样数据：DMON=0，或 dmon 在本机不可用）"; return 0
  fi
  log "GPU 采样汇总: $f"

  # 先按列名判定链路列：-s 里带 t 时，不同驱动输出 NVLink（nvltx/nvlrx 或 txn/rxn）
  # 或只有 PCIe（txpci/rxpci）。列名判定一次，表头与取列都用它，避免错位。
  local link rxk txk
  read -r link rxk txk <<< "$(awk 'NR==1 {
      for (i=1; i<=NF; i++) { h=$i; sub(/^#/, "", h); idx[h]=i }
      if (("nvltx" in idx) && ("nvlrx" in idx))       print "NVLink nvlrx nvltx";
      else if (("txn" in idx) && ("rxn" in idx))      print "NVLink rxn txn";
      else if (("txpci" in idx) && ("rxpci" in idx))  print "PCIe rxpci txpci";
      else                                            print "- - -"
    }' "$f")"

  printf '  %-4s %6s %9s %9s %7s %7s %10s %14s %5s %5s\n' \
         "GPU" "点数" "功耗峰" "温度峰" "SM均值" "SM峰" "显存峰" "${link}TX(均/峰)" "pviol" "tviol"

  awk -v link="$link" -v rxk="$rxk" -v txk="$txk" '
    NR==1 {
      for (i=1; i<=NF; i++) { h=$i; sub(/^#/, "", h); idx[h]=i }
      n=split("gpu pwr gtemp sm", need, " ")
      for (k=1; k<=n; k++) if (!(need[k] in idx)) {
        printf "   ! dmon 表头缺少字段 %s，跳过统计\n     首行原文：%s\n", need[k], $0
        bad=1; exit
      }
      # 「显存占用」是 fb（framebuffer, MB）；mem 是显存控制器利用率 %
      if ("fb" in idx) fbk="fb"; else if ("mem" in idx) fbk="mem"; else fbk=""
      next
    }
    /^#/ { next }
    NF < 5 { next }
    {
      g=$(idx["gpu"]); if (g == "") next
      npt[g]++
      v=$(idx["pwr"]);   if (v ~ /^[0-9.]+$/) { if (v+0 > pmax[g]) pmax[g]=v+0 }
      v=$(idx["gtemp"]); if (v ~ /^[0-9.]+$/) { if (v+0 > tmax[g]) tmax[g]=v+0 }
      v=$(idx["sm"]);    if (v ~ /^[0-9.]+$/) { usum[g]+=v; ucnt[g]++; if (v+0 > umax[g]) umax[g]=v+0 }
      if (fbk != "") { v=$(idx[fbk]); if (v ~ /^[0-9.]+$/) { if (v+0 > fbmax[g]) fbmax[g]=v+0 } }
      if (link != "-") {
        v=$(idx[txk]); if (v ~ /^[0-9.]+$/) { txsum[g]+=v; txc[g]++; if (v+0 > txmax[g]) txmax[g]=v+0 }
      }
      if ("pviol" in idx) { v=$(idx["pviol"]); if (v ~ /^[0-9.]+$/ && v+0 > 0) pv[g]++ }
      if ("tviol" in idx) { v=$(idx["tviol"]); if (v ~ /^[0-9.]+$/ && v+0 > 0) tv[g]++ }
    }
    END {
      if (bad) exit
      for (g in npt) {
        if (fbk == "fb")       memstr = sprintf("%.0fMB", fbmax[g])
        else if (fbk == "mem") memstr = sprintf("%.0f%%", fbmax[g])
        else                   memstr = "N/A"
        if (link != "-" && txc[g])
          txstr = sprintf("%.0f/%.0f", txsum[g]/txc[g]/1024, txmax[g]/1024)
        else if (link != "-") txstr = "0/0"
        else                  txstr = "N/A"
        printf "  %-4s %6d %8.0fW %8.0fC %6.1f%% %6.1f%% %10s %14s %5d %5d\n",
               g, npt[g], pmax[g], tmax[g], (ucnt[g] ? usum[g]/ucnt[g] : 0), umax[g],
               memstr, txstr, pv[g]+0, tv[g]+0
      }
    }
  ' "$f" 2>/dev/null | sort -n

  case "$link" in
    NVLink*) log "  链路列取自 NVLink（均值/峰值，GB/s；H200 单向容量约 450）" ;;
    PCIe)    log "  链路列取自 PCIe（均值/峰值，GB/s；不含 NVLink，驱动未输出 nvl 列时属正常）" ;;
    *)       log "  链路列不可用（该驱动未输出 t 相关列）" ;;
  esac
  log "  pviol/tviol>0 = 撞功耗墙/温度墙的次数，MFU 会掉但日志里的步时不一定炸"
  if [[ "${DMON_INTERVAL:-5}" -gt 2 ]] 2>/dev/null; then
    log "  采样间隔 ${DMON_INTERVAL}s：看趋势够用，追突发要 DMON_INTERVAL=1"
  fi
}

# -----------------------------------------------------------------------------
# 被 source 时自动执行一次（幂等）。
# 宿主侧脚本（12/13）设 SKIP_CUDA_COMPAT_AUTO=1 跳过：compat 只对容器内有意义，
# 免得在宿主 shell 里误改 LD_LIBRARY_PATH。CUDA_COMPAT 的原值仍会经 -e 透传进容器，
# 由容器内的 lib.sh 重新判定。
# -----------------------------------------------------------------------------
if [[ "${SKIP_CUDA_COMPAT_AUTO:-0}" != "1" ]]; then
  ensure_cuda_compat || true
fi
