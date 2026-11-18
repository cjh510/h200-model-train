#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机，两台【都要各跑一次】。起容器之前的门禁（G0）。
#          主机侧只读系统信息：不装东西、不改配置、不重启任何服务。
#
#   bash 02_precheck.sh                    # 完整检查（含容器内实测）
#   bash 02_precheck.sh --no-container     # 还没决定镜像，只查主机
#   → 终端 PASS/WARN/FAIL，并写出 <hostname>-env.json（给 05_report.py --compare）
#
# 这是开跑前唯一的前置门禁（原 preflight.sh 已并入本脚本）。它的 27 个判定项里 19 项
# 与本脚本同名、8 项是本脚本某一项的旧名或已并入细节（如 镜像磁盘→镜像存储磁盘、
# nvidia runtime→GPU 运行时），真正独有的只有 7 项：GPU 发现方式、toolkit 升级命令、
# run 磁盘余量、容器内 GPU/memlock/shm/NCCL。
# 两处口径原本已经打架——驱动一致性 preflight 只查两处（nvidia-smi/modinfo）、本脚本查三处
# （另加 /proc/driver/nvidia/version），同一台机器可能 preflight PASS 而这里 FAIL。
# 合并后只剩一份口径，判定项从 41 项增至 49 项。
#
# 相对原方案的变化（详见 变更清单.md）：
#   · 引擎自动探测：docker 优先，没有可用 docker 时回落 nerdctl（见 lib.sh）
#   · GPU 占用判定改为「逐卡 <500 MiB 且无计算进程」（原来按 8 卡合计 >1000 判）
#   · 补 ECC / retired_pages / NVLink 逐卡误码 / Xid 四项硬件验收
#   · 显存按实测 143771 MiB（140.4 GiB）呈报，不再写 141 GiB
#   · 补 toolkit >=1.14、rmem_max、内核冻结、镜像是否已在当前引擎可见；
#     docker 另查 Runtimes 含 nvidia 与 Cgroup Driver，nerdctl 另查 containerd 与 namespace
#   · 补 run 归档目录余量预警、容器内 8 卡 / memlock / shm / NCCL 实测（原 preflight 独有）
#
# 退出码：0=可以往下走；1=有 WARN（写进报告后继续）；2=有 FAIL（先修）
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"

NO_CONT=0
for a in "$@"; do
  case "$a" in
    --no-container) NO_CONT=1 ;;
    -h|--help) sed -n '1,28p' "$0"; exit 0 ;;
    *) echo "!! 未知参数：$a（只支持 --no-container）"; exit 2 ;;
  esac
done

if ! lc_init; then
  echo "  → 引擎探测结果：失败（docker 与 nerdctl 均不可用）"
  echo "  → 修复路径 1：安装 docker + nvidia-container-toolkit，nvidia-ctk runtime configure --runtime=docker"
  echo "  → 修复路径 2：安装 containerd + nerdctl（nerdctl-full-*.tar.gz 解到 /usr/local），固定 namespace=default"
  exit 2
fi

ROOT="${ROOT:-/root/llm-bench}"
NPROC="${NPROC:-8}"; NNODES="${NNODES:-1}"; WORLD=$(( NPROC * NNODES ))
export ROOT NPROC NNODES WORLD
START_TS=$(date +%s)
# 本脚本是开跑入口，沿用原 preflight 的归档行为：默认新开一次 run，显式 RUN_ID= 才续跑。
# 03/04/05 都是 lc_run_resume，会自动接到这次 run 上。
if [ -n "${RUN_ID:-}" ]; then
  lc_run_resume "${RUN_PHASES:-full}" || exit 2
else
  lc_run_new "${RUN_PHASES:-full}" || exit 2
fi
LOGDIR=$(lc_run_stage_dir precheck) || exit 2
export LOGDIR
OUT_DIR="${OUT_DIR:-$LOGDIR}"                # 默认写进本次 run 的 02-precheck/
export OUT_DIR
lc_engine_log

R=(); add() { R+=("$1|$2|$3"); }            # level|item|detail
HOST=$(hostname)
CEIL=""; DRV=""

# ------------------------------------------------------------------ 系统与驱动
add INFO "操作系统" "$(grep -m1 PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '"') / glibc $(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')"
add INFO "内核" "$(uname -r)"
if [ "$ENGINE" = docker ]; then
  add INFO "容器运行时" "$(lc_engine_version) / namespace=不适用（docker 忽略 CONTAINERD_NAMESPACE）"
else
  add INFO "容器运行时" "$(lc_engine_version) / namespace=$NS"
fi
add INFO "引擎探测" "docker 优先；本次 ${LC_ENGINE_LINE}（理由：${LC_ENGINE_REASON}）"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  add FAIL "nvidia-smi" "命令不存在 —— 没有驱动就没有一切，先装驱动"
else
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  CEIL=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | head -1 | awk '{print $3}')
  add INFO "驱动" "NVIDIA $DRV → 原生 CUDA 上限 ${CEIL:-未知}"

  case "${CEIL:-0}" in
    12.8*) add INFO "镜像建议" "选 CUDA≤12.8 最稳：pytorch:25.01-py3(12.8.0)；12.9 的 25.06 靠 minor 兼容也可，但 12.9 的 PTX 在 R570 上 JIT 会报 UNSUPPORTED_PTX_VERSION" ;;
    12.9*) add INFO "镜像建议" "pytorch:25.06-py3(12.9.1) 原生；25.10(13.0.2) 跨 major 需 cuda-compat，慎用" ;;
    13.1*) add INFO "镜像建议" "25.10-py3(13.0.2) 原生可用；别越过 13.1" ;;
    *)     add WARN "镜像建议" "CUDA 上限 ${CEIL:-?} 不在已知档位，人工去 NGC release notes 核对" ;;
  esac

  # 三处版本必须一致：nvidia-smi 读驱动、/proc 是内核模块自报、modinfo 是包元数据。
  # 三者不一致的典型症状是「nvidia-smi 能跑，容器里 CUDA 初始化失败」。
  PDRV=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
  MDRV=$(modinfo nvidia 2>/dev/null | awk '/^version/{print $2}')
  if [ "$DRV" = "${PDRV:-$DRV}" ] && [ "$DRV" = "${MDRV:-$DRV}" ]; then
    add PASS "驱动三处一致" "nvidia-smi=$DRV / proc=${PDRV:-NA} / modinfo=${MDRV:-NA}"
  else
    add FAIL "驱动三处一致" "nvidia-smi=$DRV / proc=${PDRV:-NA} / modinfo=${MDRV:-NA} → 版本错配，先重装驱动再谈其他"
  fi

  CNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$CNT" = "8" ] && add PASS "GPU 数量" "8 张可见" \
                  || add FAIL "GPU 数量" "只看到 $CNT 张（预期 8）—— 掉卡 / MIG / CUDA_VISIBLE_DEVICES"

  # 占用：逐卡判，不按合计。8 卡合计 1000 MiB 在单卡 140 GiB 上仍属正常，
  # 但单卡 >500 MiB 就一定有东西驻留（驱动预留通常几十~几百 MiB）。
  BUSY=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  MAXU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [ -n "$BUSY" ] || [ "${MAXU:-0}" -gt 500 ]; then
    add FAIL "GPU 被占用" "计算进程 $(echo "$BUSY" | grep -c .) 个 / 单卡最大显存 ${MAXU:-0} MiB（阈值 500）—— 压测计时全部作废，先要独占窗口"
  else
    add PASS "GPU 空闲" "无计算进程，单卡最大显存 ${MAXU:-0} MiB（阈值 500）"
  fi

  # 容量：H200 实测 143771 MiB = 140.4 GiB。报告一律用实测值，别写「141 GiB」。
  MEMS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u)
  NUNIQ=$(echo "$MEMS" | grep -c .)
  if [ "$NUNIQ" = "1" ]; then
    GIB=$(awk -v m="${MEMS:-0}" 'BEGIN{printf "%.1f", m/1024}')
    add PASS "显存容量一致" "${MEMS} MiB × $CNT = ${GIB} GiB/卡（口径按实测，不写 141）"
    [ "${MEMS:-0}" = "143771" ] || add WARN "显存口径" "实测 ${MEMS} MiB ≠ H200 标准 143771 MiB，报告里要写实测值"
  else
    add FAIL "显存容量不一致" "$MEMS —— 有卡被换过/共享/MIG，这种机器上跑出的 MFU 无法归因"
  fi

  NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ';')
  [ "$(echo "$NAMES" | tr ';' '\n' | grep -c .)" = "1" ] \
    && add PASS "GPU 型号一致" "$NAMES" || add FAIL "GPU 混型" "$NAMES"

  LIM=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | tr '\n' ' ')
  PERS=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
  THRO=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
  PST=$(nvidia-smi --query-gpu=pstate --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | tr '\n' ' ')
  add INFO "功耗上限" "$LIM（核对是否等于机型标称 TGP；被改小就是硬件/前一位用户留下的问题）"
  case "$PERS" in Enabled*) add PASS "Persistence" "$PERS";; *) add WARN "Persistence" "$PERS → 空闲后驱动卸载上下文，首步有额外开销";; esac
  case "$PST" in *P0*) add PASS "性能状态" "$PST";; *) add WARN "性能状态" "$PST → 非 P0，省电档下测不出真实算力";; esac

  TM=${THRO%% *}; TM=${TM#0x}; TM=$((16#${TM:-0}))
  if   [ "$TM" -eq 0 ]; then add PASS "降频事件" "无原因位"
  elif [ $(( TM & ~(1|16) )) -eq 0 ]; then add PASS "降频事件" "仅 GpuIdle/SyncedClocks（空闲读数，正常）"
  else
    BAD=""
    [ $(( TM & 2 )) -ne 0 ]   && BAD="$BAD 应用时钟锁定"
    [ $(( TM & 4 )) -ne 0 ]   && BAD="$BAD SW功率封顶"
    [ $(( TM & 8 )) -ne 0 ]   && BAD="$BAD HW降速"
    [ $(( TM & 64 )) -ne 0 ]  && BAD="$BAD 热限"
    [ $(( TM & 128 )) -ne 0 ] && BAD="$BAD 功率制动"
    add WARN "降频事件" "$THRO → 有实质节流位:$BAD。训练中会掉速，先查功耗墙/散热"
  fi
  add INFO "降频复采" "空闲读数看不出热降频；04_bench.sh 已在脚本内后台起 dmon 并在退出时 trap 杀掉"

  # ---------------------------------------------------------- 硬件验收（新增）
  ECCM=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | tr '\n' ' ')
  ECCC=$(nvidia-smi --query-gpu=ecc.errors.corrected.aggregate.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}')
  ECCU=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}')
  case "$ECCM" in
    *Enabled*) 
      if [ "${ECCU:-0}" -gt 0 ]; then
        add FAIL "ECC" "mode=$ECCM corrected=$ECCC uncorrected=$ECCU → 有不可纠正错误，这块卡不能再进基准测试"
      elif [ "${ECCC:-0}" -gt 0 ]; then
        add WARN "ECC" "mode=$ECCM corrected=$ECCC uncorrected=$ECCU → 纠错在起作用，但记录偏高要跟踪"
      else
        add PASS "ECC" "mode=$ECCM corrected=0 uncorrected=0"
      fi ;;
    *Disabled*) add WARN "ECC" "ECC 被关（$ECCM）—— H200 上关 ECC 等于放弃显存校验，不要这样跑基准" ;;
    *) add WARN "ECC" "取不到（$ECCM）—— 驱动字段可能改名，用 nvidia-smi -q | grep -A6 'ECC Mode' 人工核对" ;;
  esac

  # retired_pages 三态：字段不支持=SKIP（H200 的 correctable 字段直接报错），只有确证才 FAIL
  RET=$(lc_retired_pages)
  case "${RET%%|*}" in
    PASS) add PASS "retired_pages" "${RET#*|}" ;;
    FAIL) add FAIL "retired_pages" "${RET#*|} → 确认有待重映射/已退役页，换卡后再测" ;;
    *)    add SKIP "retired_pages" "${RET#*|}" ;;
  esac

  # NVLink：逐卡看链路速率 + 误码汇总。原方案只抽 GPU0 且被 200 行截断，
  # 这里改成全量计数，不做 head。
  NVL=$(nvidia-smi nvlink -s 2>/dev/null | grep -cE '^\s+Link [0-9]+')
  NVSPD=$(nvidia-smi nvlink -s 2>/dev/null | grep -oE '[0-9.]+ GB/s' | sort -u | tr '\n' ' ')
  NVE_LINES=$(nvidia-smi nvlink -e 2>/dev/null | grep -c 'Errors' || true)
  NVE=$(nvidia-smi nvlink -e 2>/dev/null | awk -F: '/Errors/{gsub(/[ \t]/,"",$2); s+=$2} END{print s+0}')
  NVLINKS_PER_GPU=18
  if [ "${NVL:-0}" -lt $(( CNT * NVLINKS_PER_GPU )) ]; then
    add FAIL "NVLink 链路数" "只报到 ${NVL:-0} 条（预期 $(( CNT * NVLINKS_PER_GPU )) = ${CNT}×${NVLINKS_PER_GPU}）→ 有卡掉链路，TP8 会退化成 PCIe"
  else
    add PASS "NVLink 链路数" "${NVL} 条（${NVLINKS_PER_GPU}/卡），速率 $NVSPD"
  fi
  if [ "${NVE_LINES:-0}" -eq 0 ]; then
    add INFO "NVLink 误码" "驱动不支持 nvidia-smi nvlink -e，改用 nvidia-smi -q -d NVLink | grep -A3 Errors 人工核对"
  elif [ "${NVE:-0}" -gt 0 ]; then
    add FAIL "NVLink 误码" "全部链路 Errors 合计 ${NVE} —— 逐卡核对：nvidia-smi nvlink -e | grep -B4 Errors"
  else
    add PASS "NVLink 误码" "全部链路 CRC/Replay/Recovery 计数为 0"
  fi

  XIDS=$( (dmesg -T 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -cE 'NVRM.*Xid' )
  SXIDS=$( (dmesg -T 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -cE 'SXid' )
  if [ "${XIDS:-0}" -gt 0 ]; then
    add FAIL "Xid" "内核日志出现 ${XIDS} 条 NVRM Xid（GPU 级错误）—— dmesg -T | grep -E 'NVRM.*Xid' 逐条看"
  elif [ "${SXIDS:-0}" -gt 0 ]; then
    add PASS "Xid" "无 NVRM Xid（另有 ${SXIDS} 条 NVSwitch SXid，属交换机侧记录，低优先级）"
  else
    add PASS "Xid" "无 NVRM Xid，也无 SXid"
  fi
fi

# ---------------------------------------------------------------------- 拓扑
TOPO=$(nvidia-smi topo -m 2>/dev/null)
if echo "$TOPO" | grep -q "NV[0-9]"; then
  NVMAX=$(echo "$TOPO" | grep -oE "NV[0-9]+" | sort -u | sed 's/NV//' | sort -n | tail -1)
  [ "${NVMAX:-0}" -ge 18 ] && add PASS "卡间互联" "NVLink NV$NVMAX（H200 8 卡 NVSwitch 应见 NV18）" \
                           || add WARN "卡间互联" "最大只有 NV$NVMAX → 部分卡可能走 PCIe，TP8 的数字会失真"
else
  add FAIL "卡间互联" "topo 里没有任何 NV 链接 —— 全走 PCIe，不要报 TP8 的结果"
fi
echo "$TOPO" > "$OUT_DIR/$HOST-topo.txt"

# ------------------------------------------------------------------ 机间网络
# 这台机器实际是 RoCE（Ethernet 链路层），不是真 IB。nvidia_peermem 已加载，
# 但 NCCL 必须显式指到 mlx5_bond_2..9 并从 RoCEv2 的 GID 走，否则会退回 socket。
if [ -d /sys/class/infiniband ] && ls /sys/class/infiniband >/dev/null 2>&1; then
  IB=$(ls /sys/class/infiniband | tr '\n' ' ')
  RATE=$(for d in /sys/class/infiniband/*/ports/1; do cat "$d/rate" 2>/dev/null; done | tr '\n' ' ')
  LL=$(for d in /sys/class/infiniband/*/ports/1; do cat "$d/link_layer" 2>/dev/null; done | sort -u | tr '\n' ' ')
  case "$LL" in
    *Ethernet*) add WARN "RDMA 网卡" "$IB | $RATE | link_layer=$LL → 这是 RoCE，不是 IB。04_bench 必须显式 NCCL_IB_HCA=mlx5_bond_2..9 + NCCL_IB_GID_INDEX" ;;
    *)          add PASS "RDMA 网卡" "$IB | $RATE | link_layer=$LL" ;;
  esac
else
  add FAIL "RDMA 网卡" "无 /sys/class/infiniband —— 双机只能走 TCP socket，70B 的 TP8×PP2 跨机通信会非常慢"
fi
for n in /sys/class/net/*; do
  i=$(basename "$n"); [ "$i" = lo ] && continue
  st=$(cat "$n/operstate" 2>/dev/null); sp=$(cat "$n/speed" 2>/dev/null); mt=$(cat "$n/mtu" 2>/dev/null)
  [ "$st" = up ] && add INFO "网卡 $i" "speed=${sp:-?} Mb/s mtu=${mt:-?}"
done
[ "$(ulimit -l)" != "unlimited" ] && [ "$(ulimit -l)" != "-1" ] \
  && add FAIL "memlock" "$(ulimit -l) → IB/RDMA 注册大 buffer 会失败，设 LimitMEMLOCK=infinity" \
  || add PASS "memlock" "unlimited"

RM=$(lc_rmem)
if [ "${RM:-0}" -ge 16777216 ]; then
  add PASS "rmem_max" "$RM B"
else
  add WARN "rmem_max" "${RM:-0} B（默认 212992）→ NCCL 回退 socket 时会被压速；sysctl -w net.core.rmem_max=67108864 并落 /etc/sysctl.d/99-nccl.conf"
fi

# --------------------------------------------------------- 容器引擎 / toolkit
if [ "$ENGINE" = docker ]; then
  add PASS "容器引擎" "docker $(lc_server_version)（docker 优先，--gpus all 走 nvidia runtime）"
  CG=$(lc_cgroup_driver)
  add INFO "Cgroup Driver" "${CG:-取不到}（cgroup v2 建议 systemd；须与节点既有配置一致）"
  if lc_docker_has_nvidia_runtime; then
    add PASS "GPU 运行时" "docker info 的 Runtimes 含 nvidia"
  else
    add FAIL "GPU 运行时" "docker info 的 Runtimes 不含 nvidia → --gpus all 必定失败；nvidia-ctk runtime configure --runtime=docker 后 systemctl restart docker"
  fi
  add INFO "namespace" "docker 路径忽略 CONTAINERD_NAMESPACE（不涉及 containerd namespace）"
else
  ACT=$(systemctl is-active containerd 2>/dev/null)
  if [ "$ACT" = active ]; then add PASS "containerd" "active $(containerd --version 2>/dev/null | awk '{print $3}')"; \
  else add FAIL "containerd" "状态 ${ACT:-unknown} —— nerdctl 全靠它"; fi
  if lc_has_nvidia_runtime; then
    add PASS "GPU 运行时" "/etc/containerd/config.toml 已注册 nvidia runtime"
  else
    # nerdctl 的 --gpus 走 nvidia-container-cli 的 OCI hook，不强制要求 runtime 注册；
    # 但 CRI 路径（K8s）需要它。容器内实测才是最终判据，见 03_deploy.sh。
    add WARN "GPU 运行时" "config.toml 未注册 nvidia runtime；nerdctl --gpus 走 nvidia-container-cli OCI hook，靠容器实测兜底"
  fi
  DEFN=$(lc_images | grep -c . || true)
  K8SN=$(lc_k8s_images_count)
  add INFO "namespace 现状" "default=$DEFN 个镜像，k8s.io=$K8SN 个 → 本次全程固定 $NS"
  [ "${K8SN:-0}" -gt 0 ] && add WARN "namespace 提醒" "这是 K8s 节点：k8s.io 里是业务镜像，本次只动 default，别混用也别 rmi 业务镜像"
fi

TK=$(lc_toolkit_state)
add "${TK%%|*}" "toolkit" "${TK#*|}"
if [ "${TK%%|*}" != PASS ]; then
  add INFO "GPU 发现方式" "$(lc_gpu_discovery_hint)"
  add INFO "toolkit 升级命令" "离线四包同版本：rpm -Uvh libnvidia-container1-1.17.*.rpm libnvidia-container-tools-1.17.*.rpm nvidia-container-toolkit-base-1.17.*.rpm nvidia-container-toolkit-1.17.*.rpm（容器内 8 卡能过就不用装）"
fi

IMG=""
if [ -n "${IMAGE:-}" ]; then
  IMG="$IMAGE"
else
  IMG=$(lc_pick_image 'pytorch|vllm' '' || true)
fi
if [ -z "$IMG" ]; then
  add WARN "镜像" "当前引擎的镜像列表里没有 pytorch 镜像 → 先 bash 03_deploy.sh 导入 tar，或 IMAGE=<镜像> 指定"
else
  add PASS "镜像" "$IMG  digest=$(lc_digest "$IMG")"
fi

DATA_DIR=$(lc_data_dir)
# 删镜像的命令按引擎给：lc_rmi 是 shell 函数，用户在宿主机上敲不到。
if [ "$ENGINE" = docker ]; then
  RMI_HINT="docker rmi <repository:tag>"
else
  RMI_HINT="nerdctl --namespace $NS rmi <repository:tag>（只动 $NS，别删 k8s.io 的业务镜像）"
fi
FREE=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4}')
case "$FREE" in
  *T*) add PASS "镜像存储磁盘" "$DATA_DIR 可用 $FREE" ;;
  *)   N="${FREE%G}"; if [ -n "$FREE" ] && [ "${N:-0}" -gt 60 ] 2>/dev/null; then
         add PASS "镜像存储磁盘" "$DATA_DIR 可用 $FREE（镜像解包约 30GB）"
       else
         add FAIL "镜像存储磁盘" "$DATA_DIR 可用 ${FREE:-未知} —— 装不下镜像层；删旧镜像：$RMI_HINT；或把 ROOT 换到大容量盘（两台都要一致）"
       fi ;;
esac

# run 归档按次累积（一次运行一个目录），余量不足 10 GB 时预警。
# 用 df -k 而不是上面的 -h：-h 的 "9.5G" 没法直接比大小。
FREE_KB=$(df -k "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${FREE_KB:-}" ] && [ "$FREE_KB" -lt 10485760 ] 2>/dev/null; then
  add WARN "run 磁盘余量" "$DATA_DIR 可用 $(( FREE_KB / 1048576 )) GB < 10 GB —— 多次运行会累积，先清理旧 run 或换盘"
fi

if lc_kernel_frozen; then
  add PASS "内核冻结" "已配置 dnf exclude / versionlock"
else
  add WARN "内核冻结" "open 内核模块且无 dkms：内核一升级 nvidia.ko 就没了 → dnf versionlock add kernel kernel-core kernel-modules"
fi

# ------------------------------------------------------------------ 主机资源
add INFO "CPU/内存" "$(nproc) 核 / $(awk '/MemTotal/{printf "%.0f GB", $2/1048576}' /proc/meminfo)"
NUMA=$(ls -d /sys/devices/system/node/node* 2>/dev/null | grep -c .)
[ "${NUMA:-0}" -gt 1 ] && add WARN "NUMA" "$NUMA 个 node → 建议 numactl 绑核，跨 node 访存能吃掉几个百分点吞吐"
HP=$(grep -m1 HugePages_Total /proc/meminfo 2>/dev/null | awk '{print $2}')
add INFO "hugepage" "${HP:-0}（Megatron 训练不需要，0 属正常）"
command -v chronyc >/dev/null && add INFO "时钟" "$(chronyc tracking 2>/dev/null | head -2 | tr '\n' ' ')" \
                             || add WARN "时钟" "没有 chronyc → 双机时间戳不同源，跨机延迟统计失真"

# -------------------------------------------------------------- 容器内实测
# 前面查的都是主机侧「看起来配好了」，这一项才是 GPU 注入通道的最终判据：
# 用 lc_run 起一个一次性容器（--gpus all / --shm-size=64g / memlock 都由它注入），
# 8 卡看不到就是引擎或 toolkit 的问题，后面所有阶段都跑不起来。
# 已有 FAIL 时不再起容器：主机侧没修好，容器里的结论没有参考价值。
FAIL_SO_FAR=$(printf '%s\n' "${R[@]}" | grep -c "^FAIL|" || true)
if [ "$NO_CONT" = 1 ]; then
  add SKIP "容器内实测" "--no-container：只查了主机侧，GPU 注入通道未验证（03_deploy.sh 的自检仍会兜一次）"
elif [ -z "$IMG" ]; then
  add SKIP "容器内实测" "当前引擎里没有可用镜像 → 先 bash 03_deploy.sh 导入 tar，或 IMAGE=<镜像> 指定"
elif [ "${FAIL_SO_FAR:-0}" -gt 0 ]; then
  add SKIP "容器内实测" "主机侧已有 $FAIL_SO_FAR 条 FAIL，先修再验容器"
else
  echo
  echo "-- 容器内实测（$IMG）--"
  COUT=$(lc_run -v /tmp:/tmp "$IMG" bash -c '
    echo "GPUS=$(nvidia-smi -L | grep -c ^GPU)";
    echo "MEMLOCK=$(ulimit -l)";
    echo "SHM=$(df -h /dev/shm | awk "NR==2{print \$2}")";
    python -c "import torch;print(\"TORCH=\"+torch.__version__);print(\"CUDA=\"+str(torch.version.cuda));print(\"NCCL=\"+\".\".join(map(str,torch.cuda.nccl.version())))" 2>&1
  ' 2>&1)
  echo "$COUT" | sed 's/^/    /'
  G=$(echo "$COUT"  | sed -n 's/^GPUS=//p'    | tail -1)
  ML=$(echo "$COUT" | sed -n 's/^MEMLOCK=//p' | tail -1)
  SH=$(echo "$COUT" | sed -n 's/^SHM=//p'     | tail -1)
  NCL=$(echo "$COUT"| sed -n 's/^NCCL=//p'    | tail -1)
  if [ "${G:-0}" = 8 ]; then
    add PASS "容器内 GPU" "8 张（引擎 $ENGINE）"
  else
    add FAIL "容器内 GPU" "看到 ${G:-0} 张 —— $ENGINE 的 --gpus all 没生效。$(lc_gpu_fail_hint)"
  fi
  [ "$ML" = unlimited ] && add PASS "容器内 memlock" "unlimited" \
    || add WARN "容器内 memlock" "${ML:-未知}（应为 unlimited，否则 RDMA 注册大 buffer 失败）"
  case "$SH" in
    64G|6[0-9]G) add PASS "容器内 shm" "$SH" ;;
    *)           add WARN "容器内 shm" "${SH:-未知}（应约 64G）" ;;
  esac
  if [ -n "$NCL" ]; then
    add INFO "容器内 NCCL" "$NCL / torch $(echo "$COUT" | sed -n 's/^TORCH=//p' | tail -1) / CUDA $(echo "$COUT" | sed -n 's/^CUDA=//p' | tail -1)"
  else
    add WARN "容器内 NCCL" "容器里跑不起 python/torch，先解决镜像"
  fi
fi

# ---------------------------------------------------------------------- 输出
{
  printf '{\n  "host": "%s",\n  "time": "%s",\n  "driver": "%s",\n  "cuda_ceiling": "%s",\n  "items": [\n' \
    "$HOST" "$(date -Is)" "${DRV:-}" "${CEIL:-}"
  first=1
  for line in "${R[@]}"; do
    lv=${line%%|*}; rest=${line#*|}; it=${rest%%|*}; dt=${rest#*|}
    dt=$(echo "$dt" | tr '\n' ' ' | sed 's/"/\\"/g')
    [ $first = 0 ] && printf ',\n' ; first=0
    printf '    {"level":"%s","item":"%s","detail":"%s"}' "$lv" "$it" "$dt"
  done
  printf '\n  ]\n}\n'
} > "$OUT_DIR/$HOST-env.json"

echo
echo "==================== 结果（$HOST）===================="
for line in "${R[@]}"; do
  lv=${line%%|*}; rest=${line#*|}; it=${rest%%|*}; dt=${rest#*|}
  case "$lv" in FAIL) m="✗";; WARN) m="!";; PASS) m="✓";; *) m="-";; esac
  printf '  %s [%-4s] %-16s %s\n' "$m" "$lv" "$it" "$(echo "$dt" | head -1)"
  echo "$dt" | tail -n +2 | sed 's/^/          /'
done

F=$(printf '%s\n' "${R[@]}" | grep -c "^FAIL|" || true)
W=$(printf '%s\n' "${R[@]}" | grep -c "^WARN|" || true)
echo
echo "FAIL=$F  WARN=$W   → $OUT_DIR/$HOST-env.json / $OUT_DIR/$HOST-topo.txt"
lc_run_info_set driver "${DRV:-}"
lc_run_info_set cuda_ceiling "${CEIL:-}"
lc_run_stage_done precheck "$START_TS"
echo "run 目录：$LC_RUN_DIR"
[ "$F" -gt 0 ] && { echo "有 FAIL：先修，否则后面所有数字都无法归因。"; exit 2; }
[ "$W" -gt 0 ] && { echo "有 WARN：可以继续，但每条都要写进报告的限制声明。"; exit 1; }
echo "环境干净，可以 bash 03_deploy.sh。"; exit 0
