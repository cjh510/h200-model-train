#!/usr/bin/env bash
# =============================================================================
# lib.sh —— llm-bench-fix 共用库
# 被 02/03/04 以 source 方式加载；不单独执行。
#
# 引擎策略：**docker 优先，没有可用 docker 时回落 nerdctl**。
#   · 默认顺序 docker → nerdctl；ENGINE=nerdctl 可覆盖默认顺序（显式指定）。
#   · 只有 `docker version` 真正可用（daemon 通）才采用 docker；
#     否则回落 nerdctl，两者都没有就报错并给两条修复路径。
#   · nerdctl 路径强制 CONTAINERD_NAMESPACE=default，并拒绝 k8s.io/moby。
#   · docker 路径忽略 CONTAINERD_NAMESPACE（docker 没有 containerd namespace 概念）。
#
# 约定：所有脚本只调用本文件的 lc / lc_* 封装，不再直接写 docker / nerdctl
#（lc 负责按引擎分派并补 --namespace；lc_run 另外统一注入 GPU / shm / ulimit / IB 设备）。
# =============================================================================

lc_engine_try_docker() {
  command -v docker >/dev/null 2>&1 || return 1
  docker version >/dev/null 2>&1
}

lc_engine_try_nerdctl() {
  command -v nerdctl >/dev/null 2>&1 || return 1
  nerdctl --version >/dev/null 2>&1
}

lc_init() {
  local want="${ENGINE:-docker}" picked="" reason="" override=0
  case "$want" in
    nerdctl) override=1 ;;
    *)       want=docker ;;
  esac

  if [ "$want" = nerdctl ]; then
    if lc_engine_try_nerdctl; then
      picked=nerdctl; reason="ENGINE=nerdctl 显式指定且 nerdctl 可用"
    elif lc_engine_try_docker; then
      picked=docker; reason="ENGINE=nerdctl 指定但 nerdctl 不可用，回落 docker"
    fi
  else
    if lc_engine_try_docker; then
      picked=docker; reason="docker version 可用（默认优先）"
    elif lc_engine_try_nerdctl; then
      picked=nerdctl; reason="docker 未安装或 daemon 不通，回落 nerdctl"
    fi
  fi

  if [ -z "$picked" ]; then
    echo "!! 没有可用的容器引擎：docker 与 nerdctl 都不可用（docker version / nerdctl --version 均失败）"
    echo "   修复路径 1（推荐，docker 为主）："
    echo "     dnf install -y docker-ce nvidia-container-toolkit"
    echo "     systemctl enable --now docker"
    echo "     nvidia-ctk runtime configure --runtime=docker && systemctl restart docker"
    echo "   修复路径 2（不装 docker，用 nerdctl）："
    echo "     解压 nerdctl-full-*.tar.gz 到 /usr/local，systemctl enable --now containerd"
    echo "     固定 CONTAINERD_NAMESPACE=default（禁止 k8s.io/moby）"
    LC_ENGINE_REASON="docker version 与 nerdctl --version 均不可用"
    LC_ENGINE_LINE="!! 未找到容器引擎"
    ENGINE=none
    ENGINE_BIN=""
    NS=""
    return 2
  fi

  ENGINE=$picked
  ENGINE_BIN=$(command -v "$picked")

  if [ "$ENGINE" = docker ]; then
    # docker 忽略 CONTAINERD_NAMESPACE：不设、不校验、不导出。
    NS=""
    LC_ENGINE_LINE="使用 docker（引擎探测：docker version 可用）"
    [ "$override" = 1 ] && LC_ENGINE_LINE="使用 docker（ENGINE=nerdctl 但不可用，已回落）"
  else
    NS="${CONTAINERD_NAMESPACE:-default}"
    case "$NS" in
      k8s.io|moby)
        echo "!! CONTAINERD_NAMESPACE=$NS 会与集群业务镜像混用；nerdctl 路径固定 default"
        LC_ENGINE_REASON="nerdctl 命名空间被设为 $NS，已拒绝"
        LC_ENGINE_LINE="!! nerdctl namespace 不合法"
        ENGINE=none
        ENGINE_BIN=""
        return 2
        ;;
    esac
    export CONTAINERD_NAMESPACE="$NS"
    if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active containerd >/dev/null 2>&1; then
      echo "!! containerd 未运行（nerdctl 依赖它）：systemctl status containerd"
      LC_ENGINE_REASON="nerdctl 已安装但 containerd 未运行"
      LC_ENGINE_LINE="!! containerd 未运行"
      ENGINE=none
      ENGINE_BIN=""
      return 2
    fi
    LC_ENGINE_LINE="使用 nerdctl（namespace=$NS）"
  fi

  LC_ENGINE_REASON="$reason"
  export ENGINE ENGINE_BIN NS
  echo "$LC_ENGINE_LINE"
  return 0
}

lc_engine_log() {                  # 把引擎探测结果写进当前阶段目录的 engine.log（覆盖写）
  [ -n "${LOGDIR:-}" ] && [ -d "$LOGDIR" ] || return 0
  {
    echo "$LC_ENGINE_LINE"
    echo "reason=$LC_ENGINE_REASON"
    echo "engine=$ENGINE"
    echo "namespace=${NS:-none}"
  } > "$LOGDIR/engine.log"
}

lc_engine_version() { "${ENGINE_BIN:-docker}" --version 2>/dev/null | head -1; }

lc_server_version() {
  if [ "$ENGINE" = docker ]; then
    lc version --format '{{.Server.Version}}' 2>/dev/null
  else
    lc --version 2>/dev/null | awk '{print $NF}'
  fi
}

# 统一分发：docker 直接执行；nerdctl 追加 --namespace。
lc() {
  if [ "${ENGINE:-}" = docker ]; then
    "$ENGINE_BIN" "$@"
  else
    "$ENGINE_BIN" --namespace "$NS" "$@"
  fi
}

# ---------------------------------------------------------------- 镜像
lc_image_exists() { lc image inspect "$1" >/dev/null 2>&1; }

lc_images() {
  lc images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -v '<none>' | grep -v '^:' || true
}

lc_pick_image() {                       # lc_pick_image <正则> <默认表达式>
  local pat="${1:-pytorch}" def="${2:-}"
  local hit
  hit=$(lc_images | grep -Ei "$pat" | head -1)
  [ -n "$hit" ] && { echo "$hit"; return 0; }
  [ -n "$def" ] && { echo "$def"; return 0; }
  return 1
}

# docker / nerdctl 的 images --digests 输出格式不同：
#   docker：REPOSITORY:TAG DIGEST（.Digest 可空）
#   nerdctl：可能没有 .Digest 占位符，退化成表格
# 统一归一化成每行「repository:tag sha256:...」。
lc_images_digests() {
  local raw i d
  raw=$(lc images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}}' 2>/dev/null || true)
  if printf '%s\n' "$raw" | grep -qE 'sha256:'; then
    printf '%s\n' "$raw" | awk '
      {
        repo=$1
        if (repo ~ /^<none>/ || repo !~ /:.*/) next
        for (i=1; i<=NF; i++) if ($i ~ /sha256:[0-9a-f]+/) { print repo, $i; break }
      }'
    return 0
  fi
  # 回退：逐镜像 inspect（两边 JSON 字段不同，统一取第一个 sha256）
  while IFS= read -r i; do
    [ -n "$i" ] || continue
    d=$(lc_image_repo_digest "$i")
    [ -n "$d" ] || d=$(lc_inspect_id "$i")
    [ -n "$d" ] && printf '%s %s\n' "$i" "$d"
  done < <(lc_images)
}

lc_inspect_id() {                       # 镜像 config ID；docker 用 {{.Id}}，nerdctl 兼容 {{.ID}}
  local img=$1 out
  out=$(lc image inspect --format '{{.Id}}' "$img" 2>/dev/null | head -1)
  if [ -z "$out" ] || [ "$out" = "<no value>" ]; then
    out=$(lc image inspect --format '{{.ID}}' "$img" 2>/dev/null | head -1)
  fi
  if [ -z "$out" ] || [ "$out" = "<no value>" ]; then
    out=$(lc image inspect "$img" 2>/dev/null | grep -m1 -oE 'sha256:[0-9a-f]{64}')
  fi
  echo "${out:-unknown}"
}

lc_image_repo_digest() {                # RepoDigests 里的 registry digest（有才返回）
  lc image inspect --format '{{index .RepoDigests 0}}' "$1" 2>/dev/null \
    | grep -m1 -oE 'sha256:[0-9a-f]{64}' || true
}

lc_digest() {                           # 报告以 digest 为准；没有 registry digest 则退 config ID
  local img=$1 d=""
  d=$(lc_images_digests 2>/dev/null | awk -v i="$img" '$1==i && $2 ~ /sha256:/ {print $2; exit}')
  [ -n "$d" ] || d=$(lc_image_repo_digest "$img")
  [ -n "$d" ] || d=$(lc_inspect_id "$img")
  echo "${d:-unknown}"
}

lc_load() { lc load -i "$1"; }

# ---------------------------------------------------------------- 容器
# 双机必须把 IB 字符设备显式带进容器，否则容器内看不到 verbs → NCCL 退回 socket。
# 只展开宿主真实存在的字符设备：nerdctl 的 --device 不接受目录，传 /dev/infiniband 会直接失败。
lc_ib_devices() {
  local d out=()
  [ "${NNODES:-1}" -gt 1 ] || return 0
  for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
    [ -c "$d" ] && out+=( --device "$d" )
  done
  if [ "${#out[@]}" -gt 0 ]; then
    printf '%s\n' "${out[@]}"
  else
    echo "   ! /dev/infiniband 下没有可用字符设备（uverbs*/rdma_cm），双机 NCCL 可能退回 socket" >&2
  fi
}
lc_run() {
  local ib=()
  mapfile -t ib < <(lc_ib_devices)
  lc run --rm --gpus all --network host --shm-size=64g \
     --ulimit memlock=-1:-1 --ulimit stack=67108864 \
     ${ib[@]+"${ib[@]}"} "$@"
}

# ---------------------------------------------------------------- GPU 状态
lc_gpu_count() { nvidia-smi -L 2>/dev/null | grep -c '^GPU'; }

lc_gpu_busy_reason() {                  # 返回 0=空闲，1=忙；忙时把原因写到 stdout
  local apps maxu
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  maxu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [ -n "$apps" ]; then
    echo "  GPU 上仍有计算进程（前 6 条）："
    echo "$apps" | head -6 | sed 's/^/    /'
    return 1
  fi
  if [ "${maxu:-0}" -gt 500 ]; then
    echo "  单卡最大显存占用 ${maxu} MiB > 500 MiB（驱动预留通常只有几百 MiB）"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- retired pages
# 三态：PASS / FAIL / SKIP。H200 不认 retired_pages.correctable（直接报
# "Field ... is not a valid field"），旧逻辑把「字段不支持」当「有页面退役」判 FAIL，属假警报。
lc_retired_pages() {                     # 输出 "<PASS|FAIL|SKIP>|<说明>"
  local out n good ids="" q c u p gpus="" got=0 i dg
  out=$(nvidia-smi --query-gpu=index,retired_pages.pending --format=csv,noheader,nounits 2>&1)
  n=$(printf '%s\n' "$out" | grep -cE '^[0-9]+' || true)
  good=$(printf '%s\n' "$out" | grep -cE '^[0-9]+[[:space:]]*,[[:space:]]*(Yes|No)[[:space:]]*$' || true)
  if [ "${n:-0}" -gt 0 ] && [ "${good:-0}" -eq "${n:-0}" ]; then
    ids=$(printf '%s\n' "$out" | awk -F',' 'tolower($2) ~ /yes/ {gsub(/[ \t]/,"",$1); printf "%s ", $1}')
    if [ -n "$ids" ]; then
      echo "FAIL|retired_pages.pending=Yes，GPU ${ids% }"
    else
      echo "PASS|retired_pages.pending=No（$n 张卡全通过）"
    fi
    return 0
  fi

  # 字段不支持时退到 nvidia-smi -q 的 Remapped Rows 段，只有取到数字才采信
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); do
    q=$(nvidia-smi -q -i "$i" 2>/dev/null | awk '/Remapped Rows/{f=1} f{print} f&&/^[[:space:]]*$/{exit}')
    [ -n "$q" ] || continue
    c=$(printf '%s\n' "$q" | awk -F: '/Correctable Error/{gsub(/[^0-9]/,"",$2); print $2; exit}')
    u=$(printf '%s\n' "$q" | awk -F: '/Uncorrectable Error/{gsub(/[^0-9]/,"",$2); print $2; exit}')
    p=$(printf '%s\n' "$q" | awk -F: '/Pending/{gsub(/[ \t]/,"",$2); print $2; exit}')
    [ -n "$c" ] && got=1
    [ -n "$u" ] && got=1
    if [ "$p" = "Yes" ] || [ "${c:-0}" -gt 0 ] || [ "${u:-0}" -gt 0 ]; then
      gpus+="$i "
    fi
  done
  if [ -n "$gpus" ]; then
    echo "FAIL|Remapped Rows 非零，GPU ${gpus% }（pending=Yes 或 corrected/uncorrected >0）"
    return 0
  fi
  if [ "$got" = 1 ]; then
    echo "PASS|Remapped Rows pending=No、corrected/uncorrected 全 0"
    return 0
  fi

  if command -v dcgmi >/dev/null 2>&1; then
    if [ "${DCGM_DIAG:-0}" = "1" ]; then
      dg=$(dcgmi diag -r 1 2>&1 || true)
      if printf '%s' "$dg" | grep -qiE 'Overall Result.*(FAIL|Fail)|\[FAIL\]'; then
        echo "FAIL|DCGM diag -r 1 判 FAIL，详见 dcgmi diag 输出"
      elif printf '%s' "$dg" | grep -qiE 'Overall Result.*PASS|\[PASS\]'; then
        echo "PASS|DCGM diag -r 1 通过"
      else
        echo "SKIP|dcgmi 已执行但结果无法解析（人工看 dcgmi diag -r 1）"
      fi
      return 0
    fi
    echo "SKIP|H200 不通过 nvidia-smi 暴露 retired_pages；未自动跑重诊断，可执行 dcgmi diag -r 1（或 DCGM_DIAG=1 复跑本脚本）"
    return 0
  fi
  echo "SKIP|该架构不通过 nvidia-smi 暴露 retired_pages，需 DCGM（本机未安装 dcgmi）"
}

# ---------------------------------------------------------------- toolkit / runtime
lc_toolkit_ver() { nvidia-container-cli --version 2>/dev/null | grep -m1 -oE '[0-9]+\.[0-9]+\.[0-9]+'; }

lc_ver_ge() {                           # lc_ver_ge <实际> <最低>
  local a=${1:-0} b=${2:-0}
  [ "$(printf '%s\n%s\n' "$b" "$a" | sort -V | tail -1)" = "$a" ]
}

lc_ver_gap() {                          # lc_ver_gap <实际> <目标>：差几个 minor（不足记 0）
  awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{split(a,x,".");split(b,y,".");
    d=(y[1]-x[1])*100+(y[2]-x[2]); if (d<0) d=0; printf "%d", d}'
}

lc_toolkit_path() { command -v nvidia-container-cli 2>/dev/null || true; }

lc_gpu_discovery_hint() {               # 当前引擎怎么把 GPU 带进容器
  case "${ENGINE:-}" in
    docker)  echo "引擎=docker：--gpus all 走 nvidia runtime（docker info 的 Runtimes 需含 nvidia）" ;;
    nerdctl) echo "引擎=nerdctl：--gpus all 走 CDI（依赖 nvidia-container-cli 注入）" ;;
    *)       echo "引擎=${ENGINE:-未探测}：GPU 发现方式未知" ;;
  esac
}

# toolkit 三态判定，输出 "<PASS|WARN>|<说明>"。版本只作警告、不阻断：
# 真正的硬门禁是容器内 8 卡自检（02 的容器实测 / 03 的 selfcheck）。
# 02_precheck.sh 用它填 add PASS/WARN，lc_toolkit_warn 用它打印，避免三处各写一份。
lc_toolkit_state() {                     # lc_toolkit_state [已测到的版本]
  local tv=${1:-$(lc_toolkit_ver)} p gap
  p=$(lc_toolkit_path)
  if [ -z "$tv" ]; then
    printf 'WARN|nvidia-container-cli 未找到（PATH=%s）—— 仅警告，不阻断；容器内 8 卡自检失败再离线装 >=1.14（libnvidia-container1/-tools、toolkit-base、toolkit 四个包同版本）' "${p:-无}"
  elif ! lc_ver_ge "$tv" 1.14; then
    gap=$(lc_ver_gap "$tv" 1.14.0)
    printf 'WARN|nvidia-container-cli %s（%s）< 1.14，差 %s 个 minor —— 仅警告，不阻断；容器内 8 卡自检失败再升到 1.17.x 四个包' "$tv" "${p:-未找到}" "${gap:-?}"
  else
    printf 'PASS|nvidia-container-cli %s（%s，>=1.14）' "$tv" "${p:-未找到}"
  fi
}

lc_toolkit_warn() {                     # lc_toolkit_warn [已测到的版本]；恒 return 0，不阻断部署
  local tk; tk=$(lc_toolkit_state "${1:-}")
  if [ "${tk%%|*}" = PASS ]; then
    echo "${tk#*|}"
  else
    echo "!! [WARN] ${tk#*|}"
    echo "   $(lc_gpu_discovery_hint)"
  fi
  return 0
}

lc_docker_has_nvidia_runtime() {
  [ "$ENGINE" = docker ] || return 1
  lc info 2>/dev/null | grep -E '^[[:space:]]*Runtimes:' | grep -qi nvidia
}

lc_cgroup_driver() {
  lc info 2>/dev/null | sed -n 's/^[[:space:]]*Cgroup Driver:[[:space:]]*//p'
}

lc_has_nvidia_runtime() {               # 按引擎分别判
  if [ "$ENGINE" = docker ]; then
    lc_docker_has_nvidia_runtime
  else
    [ -s /etc/containerd/config.toml ] && grep -qi 'nvidia' /etc/containerd/config.toml
  fi
}

lc_gpu_fail_hint() {                    # 容器内看不到 GPU 时，区分「引擎问题 / toolkit 问题」
  if [ "$ENGINE" = docker ]; then
    echo "先分两类：引擎问题看 docker info 的 Runtimes 是否含 nvidia（不含→nvidia-ctk runtime configure --runtime=docker 后重启）；toolkit 问题跑 nvidia-container-cli -k -d /dev/tty info 看注入报错与版本（若 <1.14，升级 1.17.x 四个包）"
  else
    echo "先分两类：引擎问题看 containerd 是否运行、/etc/containerd/config.toml 是否注册 nvidia runtime；toolkit 问题跑 nvidia-container-cli -k -d /dev/tty info 看注入报错与版本（若 <1.14，升级 1.17.x 四个包）"
  fi
}

lc_k8s_images_count() {                 # 仅 nerdctl：k8s.io 命名空间镜像数（只读探测）
  [ "$ENGINE" = nerdctl ] || { echo 0; return 0; }
  "$ENGINE_BIN" --namespace k8s.io images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -c . || true
}

lc_data_dir() {
  if [ "$ENGINE" = docker ]; then
    lc info --format '{{.DockerRootDir}}' 2>/dev/null | grep . || echo /var/lib/docker
  else
    echo /var/lib/containerd
  fi
}

lc_kernel_frozen() {                    # 返回 0=已冻结/有保护
  grep -qE 'exclude=kernel' /etc/dnf/dnf.conf 2>/dev/null && return 0
  dnf versionlock list 2>/dev/null | grep -q 'kernel' && return 0
  grep -qE 'kernel' /etc/yum/pluginconf.d/versionlock.list 2>/dev/null && return 0
  return 1
}

lc_rmem() { sysctl -n net.core.rmem_max 2>/dev/null || echo 0; }

# ---------------------------------------------------------------- 结果指纹
# env.txt 由 03_deploy.sh 生成，key=value 一行一个。
# 按次归档后 env.txt 在 <run>/03-deploy/，当前阶段目录里通常没有，按顺序回退查找。
lc_envval() {
  local f
  for f in "${LOGDIR:-}/env.txt" "${LC_RUN_DIR:-}/03-deploy/env.txt" "${ROOT:-/root/llm-bench}/logs/env.txt"; do
    [ -n "$f" ] && [ -s "$f" ] || continue
    sed -n "s/^${1}=//p" "$f" | head -1
    return 0
  done
  return 0
}

# Megatron --help 原文的落盘路径。**它是「某个 flag 存不存在」的唯一依据**。
# 03_deploy.sh 在容器内 `python pretrain_gpt.py --help > /workspace/logs/megatron_args.txt`，
# 经 $ROOT:/workspace 映射后宿主侧就在 $ROOT/logs/；按次归档后更常见的是本 run 的 03-deploy/。
# 回退顺序：本 run 03-deploy → 本 run args-help.txt（ensure_args_check 现场生成，格式等价）
#           → $ROOT/logs → runs/latest/03-deploy。
# 全都找不到时返回空串，**调用方必须自己兜底**：拿空串去 grep 恒返回非 0，
# 会被误读成「容器不支持该参数」（11-10 现场：S4 FP8 整组被假 SKIP，提示还印着「见 」）。
# 正确兜底是 ensure_args_check（幂等，容器内现跑一次 --help）+ flag_in_help。
lc_megatron_args() {
  local f
  for f in "${LC_RUN_DIR:-}/03-deploy/megatron_args.txt" \
           "${LC_RUN_DIR:-}/args-help.txt" \
           "${ROOT:-}/logs/megatron_args.txt" \
           "${ROOT:-}/runs/latest/03-deploy/megatron_args.txt"; do
    [ -n "$f" ] && [ -s "$f" ] || continue
    printf '%s\n' "$f"
    return 0
  done
  return 0
}

lc_env_json_fields() {                  # 给 results.jsonl 用的一小段 JSON（无逗号结尾）
  printf '"image":"%s","digest":"%s","engine":"%s","namespace":"%s","driver":"%s","torch":"%s","cuda":"%s","nccl":"%s"' \
    "$(lc_envval image)" "$(lc_envval digest)" "${ENGINE}" "${NS:-none}" \
    "$(lc_envval driver)" "$(lc_envval torch)" "$(lc_envval cuda)" "$(lc_envval nccl)"
}

# =============================================================================
# 按次运行归档（run）
#   $ROOT/runs/<YYYY-MM-DD>_<HHMM>_<host>_<scope>_<phases>/<NN>-<阶段>/
#   预检/体检/部署写各自阶段目录，04_bench 每个 phase 写自己的阶段目录。
#   runs/latest 指向最近一次（优先符号链接，失败写 latest.path）；
#   $ROOT/logs 在不存在时做成 → runs/latest 的符号链接（兼容旧习惯）。
# 进入条件：调用方已设置 ROOT / NPROC / NNODES / WORLD 与 ENGINE。
# =============================================================================

lc_run_sanitize() {
  printf '%s' "${1:-}" | tr -c 'A-Za-z0-9-' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-*$//' | cut -c1-40
}

lc_run_scope() { printf 'w%s' "${WORLD:-8}"; }

lc_run_phase_dir() {
  case "${1:-}" in
    argscheck) echo 00-argscheck ;;
    precheck)  echo 02-precheck ;;
    deploy)    echo 03-deploy ;;
    hw)        echo 04-hw ;;
    coll)      echo 05-coll ;;
    s8b)       echo 06-train-s8b ;;
    dx8b)      echo 06-train-dx8b ;;
    nccl)      echo 06-train-nccl ;;
    d70b)      echo 06-train-d70b ;;
    moe)       echo 06-train-moe ;;
    moe-dp)    echo 06-train-moe-dp ;;
    io)        echo 07-io ;;
    fault)     echo 08-fault ;;
    mfu)       echo 06-train-mfu ;;
    soak)      echo 09-soak ;;
    report)    echo 10-report ;;
    *)         echo "99-$(lc_run_sanitize "${1:-unknown}")" ;;
  esac
}

lc_run_paths() {
  LC_RUNS_ROOT="${ROOT:-/root/llm-bench}/runs"
  export LC_RUNS_ROOT
}

lc_run_latest_id() {
  local p="$LC_RUNS_ROOT/latest" t=""
  if [ -L "$p" ]; then t=$(readlink "$p" 2>/dev/null) || t=""; fi
  if [ -n "$t" ]; then basename "$t"; return 0; fi
  if [ -s "$LC_RUNS_ROOT/latest.path" ]; then
    t=$(head -1 "$LC_RUNS_ROOT/latest.path" 2>/dev/null) || t=""
    [ -n "$t" ] && [ -d "$t" ] && { basename "$t"; return 0; }
  fi
  return 1
}

lc_run_set_dir() {
  LC_RUN_DIR="$LC_RUNS_ROOT/$RUN_ID"
  export RUN_ID LC_RUN_DIR
}

lc_run_id_new() {
  local label base id n=1
  label=$(lc_run_sanitize "${1:-full}")
  base="$(date +%Y-%m-%d)_$(date +%H%M)_$(lc_run_sanitize "$(hostname)")_$(lc_run_scope)_${label}"
  id="$base"
  while [ -e "$LC_RUNS_ROOT/$id" ]; do n=$((n + 1)); id="${base}-${n}"; done
  echo "$id"
}

lc_run_info_init() {
  local f="$LC_RUN_DIR/00-run-info.txt"
  [ -f "$f" ] && return 0
  {
    echo "run_id=$RUN_ID"
    echo "host=$(hostname)"
    echo "scope=$(lc_run_scope)"
    echo "phases=${LC_RUN_PHASES:-full}"
    echo "engine=${ENGINE:-}"
    echo "namespace=${NS:-none}"
    echo "image=${IMAGE:-}"
    echo "digest=${DIGEST:-}"
    echo "driver=${DRIVER:-}"
    echo "cuda_ceiling=${CUDA_CEILING:-}"
    echo "nodes=${NNODES:-1}"
    echo "nproc=${NPROC:-8}"
    echo "world=${WORLD:-8}"
    echo "root=${ROOT:-/root/llm-bench}"
    echo "nccl_if=${NCCL_IF:-}"
    echo "nccl_hca=${NCCL_HCA:-}"
    echo "start_time=$(date -Is)"
    echo "end_time="
  } > "$f"
}

lc_run_info_set() {                     # lc_run_info_set <key> <value>
  local f="$LC_RUN_DIR/00-run-info.txt"
  [ -f "$f" ] || return 0
  LCV="$2" awk -v k="$1" 'BEGIN{FS="=";OFS="="}
    $1==k{print k"=" ENVIRON["LCV"]; hit=1; next}
    {print}
    END{if(!hit) print k"=" ENVIRON["LCV"]}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

lc_run_index_init() {
  local f="$LC_RUN_DIR/index.md"
  [ -f "$f" ] && return 0
  {
    echo "# ${RUN_ID} 运行索引"
    echo
    echo "| 阶段目录 | 耗时 | 结果 | 产出文件 |"
    echo "|---|---|---|---|"
  } > "$f"
}

lc_run_stage_status() {                 # 从阶段目录里的产物推断一句话结果
  local dir=$1 bad=0 ok=0 inv=0 suffix=""
  if ls "$dir"/.fail_* >/dev/null 2>&1; then echo "FAIL"; return 0; fi
  if [ -s "$dir/results.jsonl" ]; then
    bad=$(grep -c '"status":"\(OOM\|FAILED\|PARTIAL\)"' "$dir/results.jsonl" 2>/dev/null || true)
    ok=$(grep -c '"status":"OK"' "$dir/results.jsonl" 2>/dev/null || true)
    if [ "${bad:-0}" -gt 0 ]; then echo "PARTIAL(${ok:-0} OK / ${bad} 未过)"; return 0; fi
    if [ "${ok:-0}" -gt 0 ]; then echo "PASS(${ok} 档)"; return 0; fi
  fi
  if [ -s "$dir/fault.jsonl" ]; then
    bad=$(grep -c '"status":"FAIL"' "$dir/fault.jsonl" 2>/dev/null || true)
    inv=$(grep -c '"status":"INVALID"' "$dir/fault.jsonl" 2>/dev/null || true)
    ok=$(grep -c '"status":"PASS"' "$dir/fault.jsonl" 2>/dev/null || true)
    # INVALID = 用例没真正执行（启动期断言/参数错误），既不是 PASS 也不能被当成普通 FAIL
    # 悄悄忽略，否则阶段会被误判成 PASS(0 用例) / DONE。
    bad=$(( ${bad:-0} + ${inv:-0} ))
    [ "${inv:-0}" -gt 0 ] && suffix="（含 ${inv} INVALID）"
    if [ "${bad:-0}" -gt 0 ]; then echo "PARTIAL(${ok:-0} PASS / ${bad} 未过${suffix})"; return 0; fi
    [ "${ok:-0}" -gt 0 ] && { echo "PASS(${ok} 用例)"; return 0; }
  fi
  if [ -s "$dir/mfu.jsonl" ]; then
    # 峰值阶段的结论不是「跑了几档」，而是「峰值多少、是否有效」：OK/OOM 都正常，
    # 只有 FAILED/PARTIAL 才说明有档位没跑成。
    # 冒号后可能有空格（python json.dumps 默认带），两种写法都要认
    ok=$(grep -cE '"status":[[:space:]]*"OK"' "$dir/mfu.jsonl" 2>/dev/null || true)
    bad=$(grep -cE '"status":[[:space:]]*"(FAILED|PARTIAL)"' "$dir/mfu.jsonl" 2>/dev/null || true)
    oom=$(grep -cE '"status":[[:space:]]*"OOM_BOUNDARY"' "$dir/mfu.jsonl" 2>/dev/null || true)
    best=$(python3 - "$dir/mfu.jsonl" <<'PYEOF'
import json, sys
rows = []
try:
    for ln in open(sys.argv[1], encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("status") == "OK":
            rows.append(r)
except OSError:
    pass
if rows:
    b = max(rows, key=lambda r: r["mfu_megatron"])
    print(f'{b["model"]}@{b["nodes"]}node {b["mfu_megatron"]:.1f}%')
PYEOF
    ) 2>/dev/null || true
    if [ "${bad:-0}" -gt 0 ]; then echo "PARTIAL(${ok:-0} OK / ${bad} 未跑成)"; return 0; fi
    if [ -n "${best:-}" ]; then
      if [ "${oom:-0}" -gt 0 ]; then echo "PEAK(${best}, ${oom} 档 OOM)"; else echo "PEAK(${best})"; fi
      return 0
    fi
    [ "${oom:-0}" -gt 0 ] && { echo "BOUNDARY(${oom} 档 OOM)"; return 0; }
  fi
  if [ -s "$dir/highmem.jsonl" ]; then
    # highmem 的实质判据是 peak_pct，不是「跑完了」：WARN_LOW（<下限）与 WARN_HOT（>上限）
    # 都算未过，否则「只占 29% 也记 OK」的假通过会一路走到阶段汇总里。
    # 注意 highmem.jsonl 由 python json.dumps 写出，冒号后有空格；这里两种写法都认。
    bad=$(grep -cE '"status":[[:space:]]*"(WARN_LOW|WARN_HOT|FAILED|PARTIAL)"' "$dir/highmem.jsonl" 2>/dev/null || true)
    ok=$(grep -cE '"status":[[:space:]]*"OK"' "$dir/highmem.jsonl" 2>/dev/null || true)
    oom=$(grep -cE '"status":[[:space:]]*"OOM_BOUNDARY"' "$dir/highmem.jsonl" 2>/dev/null || true)
    smoke=$(grep -cE '"status":[[:space:]]*"SMOKE"' "$dir/highmem.jsonl" 2>/dev/null || true)
    low=$(grep -cE '"status":[[:space:]]*"WARN_LOW"' "$dir/highmem.jsonl" 2>/dev/null || true)
    [ "${low:-0}" -gt 0 ] && suffix="（含 ${low} 未达高占用）"
    if [ "${bad:-0}" -gt 0 ]; then echo "PARTIAL(${ok:-0} OK / ${bad} 未过${suffix})"; return 0; fi
    [ "${ok:-0}" -gt 0 ] && { echo "PASS(${ok} 档)"; return 0; }
    [ "${oom:-0}" -gt 0 ] && { echo "BOUNDARY(${oom} 档 OOM)"; return 0; }
    # 冒烟（HIGHMEM_MINUTES<10）不参与判定带，也不代表长稳通过，单列出来
    [ "${smoke:-0}" -gt 0 ] && { echo "SMOKE(${smoke} 档冒烟)"; return 0; }
  fi
  if [ -s "$dir/health-delta.json" ] && grep -q '"status": "FAIL"' "$dir/health-delta.json" 2>/dev/null; then
    echo "FAIL(健康增量)"; return 0
  fi
  echo "DONE"
}

lc_run_stage_done() {                   # <阶段> <起始 epoch> [结果字符串]
  local stage=$1 start=${2:-0} st=${3:-} dir elapsed files
  [ -n "${LC_RUN_DIR:-}" ] || return 0
  dir="$LC_RUN_DIR/$(lc_run_phase_dir "$stage")"
  [ -d "$dir" ] || dir="$LC_RUN_DIR"
  elapsed=$(( $(date +%s) - start ))
  files=$(cd "$dir" 2>/dev/null && ls -1 2>/dev/null | head -20 | tr '\n' ' ')
  [ -n "$st" ] || st=$(lc_run_stage_status "$dir")
  printf '| %s | %ss | %s | %s |\n' "$(lc_run_phase_dir "$stage")" "$elapsed" "$st" "${files:-—}" >> "$LC_RUN_DIR/index.md"
  lc_run_info_set end_time "$(date -Is)"
}

lc_run_update_latest() {
  ln -sfn "$RUN_ID" "$LC_RUNS_ROOT/latest" 2>/dev/null || true
  if [ -L "$LC_RUNS_ROOT/latest" ] && [ -d "$LC_RUNS_ROOT/latest" ]; then
    rm -f "$LC_RUNS_ROOT/latest.path"
  else
    echo "$LC_RUN_DIR" > "$LC_RUNS_ROOT/latest.path"
  fi
  if [ -L "$ROOT/logs" ] || [ ! -e "$ROOT/logs" ]; then
    ln -sfn "runs/latest" "$ROOT/logs" 2>/dev/null || true
  else
    echo "    · 提示：$ROOT/logs 是真实目录，未改动；本次结果在 $LC_RUN_DIR"
  fi
}

lc_run_prepare() {                      # 建目录 + index + run-info + latest
  mkdir -p "$LC_RUN_DIR" || return 2
  lc_run_index_init
  lc_run_info_init
  lc_run_update_latest
  return 0
}

# 新开一次运行。02_precheck.sh 用它（每次预检 = 一次新运行，除非显式 RUN_ID）。
lc_run_new() {
  local label="${1:-full}"
  lc_run_paths
  case "${RUN_ID:-}" in
    */*|*..*) echo "!! RUN_ID 不能包含 / 或 ..：$RUN_ID"; return 2 ;;
  esac
  LC_RUN_PHASES="$label"
  RUN_ID=$(lc_run_id_new "$label")
  lc_run_set_dir
  lc_run_prepare || return 2
  echo "    · 新建 run：$LC_RUN_DIR"
}

# 续用：RUN_ID 显式指定 > runs/latest > 新建。02/03/04/05 用它避免每阶段一个新目录。
lc_run_resume() {
  local label="${1:-full}" cur
  lc_run_paths
  case "${RUN_ID:-}" in
    */*|*..*) echo "!! RUN_ID 不能包含 / 或 ..：$RUN_ID"; return 2 ;;
  esac
  if [ -n "${RUN_ID:-}" ]; then
    lc_run_set_dir
    if [ -f "$LC_RUN_DIR/00-run-info.txt" ]; then
      echo "    · 续跑：$LC_RUN_DIR"
    else
      LC_RUN_PHASES="$label"
      lc_run_prepare || return 2
      echo "    · 新建 run（指定 RUN_ID）：$LC_RUN_DIR"
    fi
    lc_run_update_latest
    return 0
  fi
  if cur=$(lc_run_latest_id); then
    RUN_ID="$cur"
    lc_run_set_dir
    lc_run_index_init
    lc_run_info_init
    lc_run_update_latest
    echo "    · 续跑（runs/latest）：$LC_RUN_DIR"
    return 0
  fi
  lc_run_new "$label"
}

lc_run_stage_dir() {                    # 返回并创建当前 run 的阶段目录
  local d
  d="$LC_RUN_DIR/$(lc_run_phase_dir "${1:-$PHASE}")"
  mkdir -p "$d" || return 1
  echo "$d"
}
