#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机，两台【都要各跑一次】。起容器之前的一道门禁（G0）。
#          全程只读取系统信息，不装东西、不改配置。
#
#   bash 02_precheck.sh
#   → 终端打印 PASS/WARN/FAIL，并写出 <hostname>-env.json（留给 05_report.py --compare）
#
# 退出码：0 = 可以往下走；1 = 有 WARN 但可继续（记录进报告）；2 = 有 FAIL，先修环境。
# =============================================================================

set -uo pipefail

R=(); add() { R+=("$1|$2|$3"); }        # level|item|detail
sh_out() { timeout 25 "$@" 2>&1 | head -8; }

# ------------------------------------------------------------------ 系统与驱动
OS=$(sh_out cat /etc/os-release | head -1)
GLIBC=$(ldd --version 2>/dev/null | head -1)
add INFO "操作系统" "$OS / glibc $GLIBC"

if ! command -v nvidia-smi >/dev/null; then
  add FAIL "nvidia-smi" "命令不存在 —— 没有驱动就没有一切，先装驱动"
else
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  CEIL=$(sh_out nvidia-smi | grep -oE "CUDA Version: [0-9.]+" | head -1 | awk '{print $3}')
  add INFO "驱动" "NVIDIA $DRV → 原生 CUDA 上限 ${CEIL:-未知}"

  # 自动给镜像建议：容器 CUDA 必须 ≤ 这个上限（同 major 内 minor 兼容可以跨小版本）
  case "${CEIL:-0}" in
    12.8*) add INFO "镜像建议" "选 CUDA≤12.8 最稳：pytorch:25.01-py3(12.8.0)；12.9 的 25.06 靠 minor 兼容也可" ;;
    12.9*) add INFO "镜像建议" "pytorch:25.06-py3(12.9.1) 原生；25.10(13.0.2) 跨 major 需 cuda-compat，慎用" ;;
    13.1*) add INFO "镜像建议" "25.10-py3(13.0.2) 原生可用；也别越过 13.1" ;;
    *)     add WARN "镜像建议" "CUDA 上限 $CEIL 不在已知档位，人工去 NGC release notes 核对" ;;
  esac

  CNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
  [ "$CNT" = "8" ] && add PASS "GPU 数量" "8 张可见" \
                  || add FAIL "GPU 数量" "只看到 $CNT 张（预期 8）—— 掉卡或 MIG 或 CUDA_VISIBLE_DEVICES 设过"

  # 共享占用检查：有别人的进程在跑时，step 时间/功耗/降频全部失真，测了也白测。
  BUSY=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  MEMU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [ -n "$BUSY" ] || [ "${MEMU:-0}" -gt 1000 ]; then
    add FAIL "GPU 被占用" "检测到计算进程或显存占用 ${MEMU} MiB —— 压测计时全部作废，先要独占窗口"
    echo "$BUSY" | head -6 | sed 's/^/            /'
  else
    add PASS "GPU 空闲" "无计算进程，8 卡合计显存 ${MEMU} MiB"
  fi

  # 容量一致性
  MEMS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | tr -d ' ' | sort -u)
  NUNIQ=$(echo "$MEMS" | grep -c .)
  if [ "$NUNIQ" = "1" ]; then
    add PASS "显存容量一致" "$MEMS × $CNT"
  else
    add FAIL "显存容量不一致" "$MEMS —— 有卡被换过/共享/MIG，这种机器上跑出的 MFU 无法归因"
  fi

  NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ';')
  [ "$(echo "$NAMES" | tr ';' '\n' | grep -c .)" = "1" ] \
    && add PASS "GPU 型号一致" "$NAMES" || add FAIL "GPU 混型" "$NAMES"

  # 功耗墙：租来的机器被前一位用户 -pl 限过很常见，症状是"GEMM 莫名只有 600 TFLOPS"
  LIM=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ' ')
  PERS=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | tr -d ' ' | sort -u)
  THRO=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader | sort -u | tr '\n' ' ')
  PST=$(nvidia-smi --query-gpu=pstate --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ' ')
  add INFO "功耗上限" "$LIM（核对是否等于该机型标称 TGP；被改小就是硬件问题）"
  case "$PERS" in Enabled*) add PASS "Persistence" "$PERS";; *) add WARN "Persistence" "$PERS → 空闲后驱动卸载上下文，首步有额外开销";; esac
  case "$PST" in *P0*) add PASS "性能状态" "$PST";; *) add WARN "性能状态" "$PST → 非 P0，省电档下测不出真实算力";; esac
  # clocks_event_reasons.active 是位掩码，不能按前缀字符串判：
  #   0x1 GpuIdle、0x10 SyncedClocks 空闲时出现属正常；
  #   0x2 应用时钟锁定、0x4 SW功率封顶、0x8/0x40/0x80 HW降速/热限/功率制动 才是真问题。
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
  echo "  · 降频位要在【训练跑起来时】复采才算数：空闲读数看不出热降频"
  echo "    跑 hw/s8b 时另开终端： nvidia-smi dmon -s put -d 1"
fi

# ---------------------------------------------------------------------- 拓扑
TOPO=$(sh_out nvidia-smi topo -m)
if echo "$TOPO" | grep -q "NV[0-9]"; then
  NVMAX=$(echo "$TOPO" | grep -oE "NV[0-9]+" | sort -u | sed 's/NV//' | sort -n | tail -1)
  [ "${NVMAX:-0}" -ge 18 ] && add PASS "卡间互联" "NVLink NV$NVMAX（H200 8 卡 NVSwitch 应见 NV18）" \
                           || add WARN "卡间互联" "最大只有 NV$NVMAX → 部分卡可能走 PCIe，TP8 的数字会失真"
else
  add FAIL "卡间互联" "topo 里没有任何 NV 链接 —— 全走 PCIe，不要报 TP8 的结果"
fi
echo "$TOPO" > "$(hostname)-topo.txt"

# ------------------------------------------------------------------ 机间网络
if [ -d /sys/class/infiniband ] && ls /sys/class/infiniband >/dev/null 2>&1; then
  IB=$(ls /sys/class/infiniband | tr '\n' ' ')
  RATE=$(for d in /sys/class/infiniband/*/ports/1; do cat "$d/rate" 2>/dev/null; done | tr '\n' ' ')
  add PASS "RDMA 网卡" "$IB | $RATE —— 双机可走 IB，70B 的 TP8×DP2 才有意义"
else
  add WARN "RDMA 网卡" "无 /sys/class/infiniband —— 双机只能走 TCP socket。
       后果：70B 必须用 TP8×PP2（跨机只传激活），且不要指望 DP 扩展。见 TESTPLAN 第 3 节"
fi
for n in /sys/class/net/*; do
  i=$(basename "$n"); [ "$i" = lo ] && continue
  st=$(cat "$n/operstate" 2>/dev/null); sp=$(cat "$n/speed" 2>/dev/null); mt=$(cat "$n/mtu" 2>/dev/null)
  [ "$st" = up ] && add INFO "网卡 $i" "speed=${sp:-?} Mb/s mtu=${mt:-?}"
done
[ "$(ulimit -l)" != "unlimited" ] && [ "$(ulimit -l)" != "-1" ] \
  && add WARN "memlock" "$(ulimit -l) → IB 注册大 buffer 会失败，设 LimitMEMLOCK=infinity" \
  || add PASS "memlock" "unlimited"

# ------------------------------------------------- 容器引擎（docker 或 nerdctl 都行）
ENGINE=""
command -v docker  >/dev/null 2>&1 && ENGINE="docker"
command -v nerdctl >/dev/null 2>&1 && ENGINE="${ENGINE:-nerdctl}"
NS="${CONTAINERD_NAMESPACE:-default}"
DATA_DIR=/var/lib/docker
[ "$ENGINE" = nerdctl ] && DATA_DIR=/var/lib/containerd

if [ -z "$ENGINE" ]; then
  if command -v containerd >/dev/null 2>&1; then
    add WARN "容器引擎" "有 containerd 但没有 nerdctl。nerdctl 是单个静态二进制，"
    add WARN "容器引擎" "从 VM 的 GitHub release 下 nerdctl-<ver>-linux-amd64.tar.gz，解出放 /usr/local/bin 即可，不用联网装"
  else
    add FAIL "容器引擎" "既没 docker 也没 containerd/nerdctl → 容器路线走不了，改用裸机 uv 方案（llm-bench-bare）"
  fi
else
  add INFO "容器引擎" "$ENGINE  $($ENGINE --version 2>&1 | head -1)"

  if [ "$ENGINE" = nerdctl ]; then
    ACT=$(systemctl is-active containerd 2>/dev/null)
    if [ "$ACT" = active ]; then add PASS "containerd" "active"
    else add FAIL "containerd" "状态 $ACT —— nerdctl 全靠它，先 systemctl status containerd 查"; fi

    # namespace 是 nerdctl 最容易踩的坑：K8s/ACK 节点的镜像都在 k8s.io，
    # 默认 namespace 是 default。两边不一致就会出现「明明 load 了却说找不到镜像」。
    DEFN=$(nerdctl --namespace default images 2>/dev/null | tail -n +2 | grep -c .)
    K8SN=$(nerdctl --namespace k8s.io images 2>/dev/null | tail -n +2 | grep -c .)
    add INFO "namespace 现状" "default=$DEFN 个镜像，k8s.io=$K8SN 个 → 全程固定 CONTAINERD_NAMESPACE（当前 $NS）"
    [ "$K8SN" -gt 0 ] && add WARN "namespace 提醒" \
      "这像是 K8s 节点。要么统一用 default 并自己管镜像，要么全程 --namespace k8s.io，别混"

    if [ -s /etc/containerd/config.toml ] && grep -qiE 'nvidia' /etc/containerd/config.toml; then
      add PASS "GPU 运行时" "containerd config 里有 nvidia runtime 配置（CRI 路径）"
    elif command -v nvidia-container-cli >/dev/null 2>&1; then
      # 关键：nerdctl 的 --gpus 不查 containerd 的 runtime 注册表，它自己在 OCI hook 里
      # 调 nvidia-container-cli 注入设备与库。所以 config.toml 干净≠不能用，
      # 缺 nvidia-ctk 也≠不能用（toolkit 1.2.x 本来就没有 nvidia-ctk 这个命令）。
      # 唯一权威判据是实测一条 --gpus all —— 见 03_deploy.sh 的容器自检。
      add INFO "GPU 注入通道" "有 nvidia-container-cli（$(nvidia-container-cli --version 2>&1 | head -1)）"
      echo "            → nerdctl --gpus 应该可用。用实测确认（03_deploy.sh 的自检就会跑一条）："
      echo "              nerdctl --namespace default run --rm --gpus all --network host <镜像> nvidia-smi -L"
    elif command -v nvidia-container-runtime >/dev/null 2>&1; then
      add FAIL "GPU 运行时" "只有 nvidia-container-runtime，没有 nvidia-container-cli 也没注册 handler →
        新版 toolkit：sudo nvidia-ctk runtime configure --runtime=containerd && sudo systemctl restart containerd
        但 toolkit 1.2.x 没有 nvidia-ctk → 先看 rpm -qa | grep nvidia-container-toolkit。
        改 config.toml 必须重启 containerd，节点上有业务（nerdctl --namespace k8s.io ps）就要维护窗口，别自己重启"
    else
      add FAIL "GPU 运行时" "既没注册 nvidia runtime，也没有 nvidia-container-cli / -runtime →
        容器里看不到 GPU。离线装 nvidia-container-toolkit 的 rpm，
        或者直接放弃容器路线走裸机 envdir 方案（不需要容器运行时）"
    fi
  else
    if command -v nvidia-ctk >/dev/null 2>&1; then
      RT=$(docker info --format '{{json .Runtimes}}' 2>/dev/null)
      case "$RT" in *nvidia*) add PASS "GPU 容器运行时" "nvidia runtime 已注册：$RT" ;;
                      *) add FAIL "GPU 容器运行时" "装了 nvidia-ctk 但 runtime 没注册 →
          跑：sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" ;;
      esac
    else
      add FAIL "GPU 容器运行时" "没有 nvidia-ctk → 容器里看不到 GPU。
          无外网就得离线装 rpm：nvidia-container-toolkit + nvidia-container-toolkit-base + libnvidia-container-tools，
          rpm -ivh 后 nvidia-ctk runtime configure --runtime=docker && systemctl restart docker"
    fi
  fi

  FREE=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4}')
  case "$FREE" in
    *T*) add PASS "镜像存储磁盘" "$DATA_DIR 可用 $FREE" ;;
    *)   N="${FREE%G}"; if [ -n "$FREE" ] && [ "${N:-0}" -gt 60 ] 2>/dev/null; then
           add PASS "镜像存储磁盘" "$DATA_DIR 可用 $FREE（9GB tar + 解包约 30GB）"
         else
           add FAIL "镜像存储磁盘" "$DATA_DIR 可用 ${FREE:-未知} —— 装不下 9GB 镜像解包后的层"
         fi ;;
  esac
fi

# ------------------------------------------------------------------ 主机资源
add INFO "CPU/内存" "$(nproc) 核 / $(awk '/MemTotal/{printf "%.0f GB", $2/1048576}' /proc/meminfo)（建模瞬时要把完整权重放 host 时要用到）"
NUMA=$(ls -d /sys/devices/system/node/node* 2>/dev/null | grep -c .)
[ "${NUMA:-0}" -gt 1 ] && add WARN "NUMA" "$NUMA 个 node → 建议 numactl 绑核，跨 node 访存能吃掉几个百分点吞吐"
command -v chronyc >/dev/null && add INFO "时钟" "$(chronyc tracking 2>/dev/null | head -2 | tr '\n' ' ')" \
                             || add WARN "时钟" "没有 chronyc → 双机时间戳可能不同源，跨机延迟统计会失真"

# ---------------------------------------------------------------------- 输出
HOST=$(hostname)
printf '{\n  "host": "%s",\n  "time": "%s",\n  "items": [\n' "$HOST" "$(date -Is)" > "$HOST-env.json"
first=1
for line in "${R[@]}"; do
  lv=${line%%|*}; rest=${line#*|}; it=${rest%%|*}; dt=${rest#*|}
  dt=$(echo "$dt" | tr '\n' ' ' | sed 's/"/\\"/g')
  [ $first = 0 ] && printf ',\n' >> "$HOST-env.json"; first=0
  printf '    {"level":"%s","item":"%s","detail":"%s"}' "$lv" "$it" "$dt" >> "$HOST-env.json"
done
printf '\n  ]\n}\n' >> "$HOST-env.json"

echo
echo "==================== 结果（$HOST）===================="
for line in "${R[@]}"; do
  lv=${line%%|*}; rest=${line#*|}; it=${rest%%|*}; dt=${rest#*|}
  case "$lv" in FAIL) m=x;; WARN) m=!;; PASS) m=+;; *) m=-;; esac
  printf '  %s [%-4s] %-16s %s\n' "$m" "$lv" "$it" "$(echo "$dt" | head -1)"
  echo "$dt" | tail -n +2 | sed 's/^/          /'
done

F=$(printf '%s\n' "${R[@]}" | grep -c "^FAIL|")
W=$(printf '%s\n' "${R[@]}" | grep -c "^WARN|")
echo
echo "FAIL=$F  WARN=$W   → $HOST-env.json / $HOST-topo.txt"
[ "$F" -gt 0 ] && { echo "有 FAIL：先修，否则后面所有数字都无法归因。"; exit 2; }
[ "$W" -gt 0 ] && { echo "有 WARN：可以继续，但每条都要写进报告的限制声明。"; exit 1; }
echo "环境干净，可以开始传物料。"; exit 0
