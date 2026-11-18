#!/usr/bin/env bash
# =============================================================================
# 00_prepare —— 在每台服务器上把「镜像 + Megatron 源码包」放好并自检（nerdctl 版）
#
# 两名训练脚本（run_56b_2x8h200.sh / run_1x8h200_cases.sh）的硬约定：
#   · $ROOT 整体挂进容器 /workspace，所以源码与产物都必须在 $ROOT 下面
#   · PYTHONPATH=/workspace/Megatron-LM → 源码目录名必须恰好是 Megatron-LM
#   · ARGS_HELP=$ROOT/e2e-56b/args-help.txt → 本脚本负责生成（两个训练脚本共用）
#   · 引擎 nerdctl，且 load / run 都用同一个 --namespace default
#
# -----------------------------------------------------------------------------
# 目录布局（两台机器都要建，内容一致）
# -----------------------------------------------------------------------------
#   /root/llm-bench/                        ← ROOT（可改，两个训练脚本用同一个 ROOT）
#   ├── pytorch-25.01-py3.tar               ← 镜像 tar，nerdctl load -i 用
#   │                                         （也可以是 .tar.gz）
#   ├── Megatron-LM-core_v0.13.1.zip        ← 源码包，放这里即可，本脚本负责解
#   ├── Megatron-LM/                        ← 解包后必须改名为这个（GitHub 归档带
#   │   ├── pretrain_gpt.py                   core_v0.13.1 后缀，不改名 PYTHONPATH
#   │   └── megatron/...                     找不到包，8 个 rank 全部 import 失败）
#   ├── 00_prepare.sh                       ← 建议三个脚本也放这里，一机拷一个目录
#   ├── run_56b_2x8h200.sh
#   ├── run_1x8h200_cases.sh
#   └── e2e-56b/                           ← 运行产物，本脚本自动建
#       ├── args-help.txt                   ← core_v0.13.1 的 pretrain_gpt.py --help
#       ├── selfcheck.sh / selfcheck.log
#       ├── image-digest.txt                ← 两台必须一致
#       └── logs/ data_cache/ checkpoints/ tensorboard/
#
# 容器内对应路径：/workspace/Megatron-LM、/workspace/e2e-56b
#
# 【为什么必须用 nerdctl 的 namespace】
#   nerdctl load 把镜像放进 containerd 的某个 namespace；run 默认也是 default，
#   但只要环境变量 CONTAINERD_NAMESPACE 被设成 k8s.io / moby，load 与 run 就会
#   落进不同命名空间 → 「镜像明明 load 了，run 说 not found」。本脚本与两个训练
#   脚本都固定 --namespace default，并拒绝 k8s.io / moby（沿用项目 lib.sh 约定）。
#   nerdctl 与 docker 的镜像库互不相通：docker load 的镜像 nerdctl 看不见，反之亦然。
#
# 【用法】两台各跑一次
#   scp pytorch-25.01-py3.tar Megatron-LM-core_v0.13.1.zip *.sh node-a:/root/llm-bench/
#   cd /root/llm-bench && bash 00_prepare.sh
#   DRY_RUN=1 bash 00_prepare.sh                 # 只看计划，不导入不启容器
#   SKIP_SELFCHECK=1 bash 00_prepare.sh          # 只放物料，跳过容器内 8 卡自检
#   ROOT=/data/llm-bench IMAGE=xxx bash 00_prepare.sh
#
# 【已有镜像但没有 tar】脚本会直接复用；也可从已有机器导出：
#   nerdctl --namespace default save -o pytorch-25.01-py3.tar <已有镜像名>
# 【没有 tar 但有外网】直接 pull 到目标 tag：
#   nerdctl --namespace default pull nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3
# =============================================================================
set -euo pipefail

ROOT="${ROOT:-/root/llm-bench}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$ROOT/e2e-56b}"
IMAGE="${IMAGE:-nvcr.nju.edu.cn/nvidia/pytorch:25.01-py3}"
IMAGE_TAR="${IMAGE_TAR:-}"                     # 显式指定；空 = 在 $ROOT 下找 pytorch*.tar*
MEGATRON_PKG="${MEGATRON_PKG:-}"               # 显式指定；空 = 找 Megatron-LM*.zip/tar.gz/tar
# 强制 namespace=default，不受集群注入的 CONTAINERD_NAMESPACE 影响。
# 现场症状：nerdctl -n default images 能看到镜像，但 nerdctl image inspect（不带 -n）
# 报「镜像不存在」→ 默认 namespace 不是 default。显式 --namespace 优先级最高。
NS="${NERDCTL_NAMESPACE:-default}"
if [ -n "${CONTAINERD_NAMESPACE:-}" ] && [ "$CONTAINERD_NAMESPACE" != "$NS" ]; then
  echo "   ! 检测到 CONTAINERD_NAMESPACE=$CONTAINERD_NAMESPACE，已忽略，固定 namespace=$NS" >&2
fi
export CONTAINERD_NAMESPACE="$NS"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SELFCHECK="${SKIP_SELFCHECK:-0}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"

NC=(nerdctl --namespace "$NS")

echo "==================== 00_prepare / nerdctl ===================="
echo "ROOT=$ROOT"
echo "WORKSPACE=$WORKSPACE_HOST"
echo "IMAGE=$IMAGE"
echo "ENGINE=nerdctl  namespace=$NS"
echo "镜像 tar: ${IMAGE_TAR:-<在 \$ROOT 下自动找 pytorch*.tar*>}"
echo "源码包  : ${MEGATRON_PKG:-<在 \$ROOT 下自动找 Megatron-LM*.zip|tar.gz|tar>}"

if [ "$DRY_RUN" = 1 ]; then
  cat <<'PLAN'

计划（DRY_RUN，未执行）：
  1. 校验 nerdctl / containerd / namespace
  2. 镜像：已在 nerdctl(namespace=default) 就复用；否则 nerdctl --namespace default load -i <tar>
         若 load 出来的 tag 与 IMAGE 不同，自动 nerdctl tag 补一个别名
  3. 源码：解包到 $ROOT，把 Megatron-LM-core_v0.13.1 之类改名成 Megatron-LM
  4. 建 $ROOT/e2e-56b/{logs,data_cache,checkpoints,tensorboard}
  5. 容器内自检（--gpus all）：8 卡 / torch / CUDA / TE / dump --help 到 args-help.txt
  6. 打印指纹，两台用 image-digest.txt 对齐
PLAN
  exit 0
fi

# ---------------------------------------------------------------- 1. 引擎
command -v nerdctl >/dev/null 2>&1 || { echo "!! 找不到 nerdctl"; exit 2; }
nerdctl --version >/dev/null 2>&1 || { echo "!! nerdctl --version 失败"; exit 2; }
if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active containerd >/dev/null 2>&1; then
  echo "!! containerd 未运行：systemctl enable --now containerd"; exit 2
fi
mkdir -p "$WORKSPACE_HOST/logs" "$WORKSPACE_HOST/data_cache" \
         "$WORKSPACE_HOST/checkpoints" "$WORKSPACE_HOST/tensorboard"

# ---------------------------------------------------------------- 2. 镜像
echo
echo "---------------- 2/4 镜像 ----------------"
IMG_IN_LIST() { nerdctl --namespace "$1" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
                  | grep -Fx "$IMAGE" >/dev/null; }
if IMG_IN_LIST "$NS"; then
  echo "  复用已有镜像：$IMAGE（不重复 load）"
else
  if [ -z "$IMAGE_TAR" ]; then
    IMAGE_TAR=$(find "$ROOT" -maxdepth 1 -type f \
      \( -name 'pytorch*.tar' -o -name 'pytorch*.tar.gz' -o -name 'pytorch*.tgz' \) \
      -print 2>/dev/null | head -1)
  fi
  if [ -z "$IMAGE_TAR" ] || [ ! -f "$IMAGE_TAR" ]; then
    echo "!! 当前 nerdctl(namespace=$NS) 里没有 $IMAGE，$ROOT 下也没有 pytorch*.tar"
    echo "   两条路：① 从已有机器导出：nerdctl --namespace $NS save -o $ROOT/pytorch-25.01-py3.tar <镜像>"
    echo "           ② 有外网直接拉：nerdctl --namespace $NS pull $IMAGE"
    echo "           ③ 用 IMAGE=<实际镜像名> 或 IMAGE_TAR=<路径> 显式指定"
    exit 2
  fi
  echo "  导入：$IMAGE_TAR"
  BEFORE=$("${NC[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | sort)
  "${NC[@]}" load -i "$IMAGE_TAR" || {
    echo "!! nerdctl load 失败：journalctl -u containerd -n 50"; exit 2; }
  AFTER=$("${NC[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | sort)
  NEW=$(comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") | head -1)
  if [ -z "$NEW" ]; then
    NEW=$("${NC[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
          | grep -i pytorch | head -1)
  fi
  [ -n "$NEW" ] || { echo "!! load 完仍找不到镜像名，手动查：nerdctl --namespace $NS images"; exit 2; }
  if [ "$NEW" != "$IMAGE" ]; then
    echo "  load 出来的 tag 是 $NEW，与脚本默认 $IMAGE 不同 → 补一个别名"
    "${NC[@]}" tag "$NEW" "$IMAGE"
  fi
  echo "  镜像就绪：$IMAGE（原 tag ${NEW}）"
fi

DIGEST=$("${NC[@]}" image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo unknown)
echo "$DIGEST" > "$WORKSPACE_HOST/image-digest.txt"
echo "  image id: $DIGEST"
# 现场诊断：不带 -n 能否看到镜像，直接定位「镜像明明在却报不存在」的根因
if ! nerdctl images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
     | grep -Fx "$IMAGE" >/dev/null; then
  echo "  ! 不带 --namespace 时看不到该镜像 → 本机 nerdctl 默认命名空间不是 $NS"
  echo "    （环境变量 CONTAINERD_NAMESPACE=${CONTAINERD_NAMESPACE:-<unset>}，"
  echo "     或 /etc/nerdctl/nerdctl.toml / ~/.config/nerdctl/nerdctl.toml 配了 namespace）"
  echo "    这是旧版脚本报『镜像不存在』的原因；本脚本与两个训练脚本都显式带 --namespace $NS"
fi
echo "  ★ 两台必须完全一致（image id + engine + namespace），否则容器内 NCCL/TE 版本不同，双机会以 hang 形式失败"

# ---------------------------------------------------------------- 3. 源码包
echo
echo "---------------- 3/4 Megatron-LM 源码包 ----------------"
if [ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ]; then
  echo "  已有 $ROOT/Megatron-LM/pretrain_gpt.py，跳过解包"
else
  if [ -z "$MEGATRON_PKG" ]; then
    MEGATRON_PKG=$(find "$ROOT" -maxdepth 1 -type f \
      \( -name 'Megatron-LM*.zip' -o -name 'Megatron-LM*.tar.gz' -o -name 'Megatron-LM*.tgz' \
         -o -name 'Megatron-LM*.tar' \) -print 2>/dev/null | head -1)
  fi
  [ -n "$MEGATRON_PKG" ] && [ -f "$MEGATRON_PKG" ] || {
    echo "!! 没找到源码包。把 Megatron-LM-core_v0.13.1.zip 放 $ROOT，或 MEGATRON_PKG=<路径>"; exit 2; }
  echo "  解包：$MEGATRON_PKG"
  case "$MEGATRON_PKG" in
    *.zip)
      if command -v unzip >/dev/null 2>&1; then unzip -qo "$MEGATRON_PKG" -d "$ROOT"
      else python3 -m zipfile -e "$MEGATRON_PKG" "$ROOT/"; fi ;;
    *.tar.gz|*.tgz) tar xzf "$MEGATRON_PKG" -C "$ROOT" ;;
    *.tar)          tar xf  "$MEGATRON_PKG" -C "$ROOT" ;;
    *) echo "!! 不认识的归档：$MEGATRON_PKG"; exit 2 ;;
  esac
  # GitHub/归档顶层目录带 ref 后缀（Megatron-LM-core_v0.13.1），PYTHONPATH 固定写 Megatron-LM
  if [ ! -d "$ROOT/Megatron-LM" ]; then
    CAND=$(find "$ROOT" -maxdepth 1 -type d -iname 'Megatron-LM*' ! -name 'Megatron-LM' | head -1)
    [ -n "$CAND" ] && mv "$CAND" "$ROOT/Megatron-LM" \
      && echo "  目录改名：$(basename "$CAND") → Megatron-LM"
  fi
  [ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ] || {
    echo "!! 解包后仍没有 $ROOT/Megatron-LM/pretrain_gpt.py；实际位置："
    find "$ROOT" -maxdepth 3 -name pretrain_gpt.py; exit 2; }
fi

# ---------------------------------------------------------------- 4. 容器内自检
echo
echo "---------------- 4/4 容器内自检 ----------------"
SELF="$WORKSPACE_HOST/selfcheck.sh"
cat > "$SELF" <<'SELF'
#!/usr/bin/env bash
# 在容器内执行：验证 8 卡 / 组件版本，并把 core_v0.13.1 的 --help 落到共享目录。
set -uo pipefail
export PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH:-}
echo "--- 组件版本 ---"
g=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
echo "GPUS=${g:-0}"
python - <<'PY'
import sys, torch
print("PY=" + sys.version.split()[0])
print("TORCH=" + torch.__version__)
print("CUDA=" + str(torch.version.cuda))
try:
    import transformer_engine as te
    print("TE=" + te.__version__)
except Exception as e:
    print("TE=unknown " + str(e)[:100])
PY
cd /workspace/Megatron-LM || exit 1
python pretrain_gpt.py --help > /workspace/e2e-56b/args-help.txt 2>&1
n=$(grep -cE '^ *--' /workspace/e2e-56b/args-help.txt 2>/dev/null || true)
echo "ARGS_FLAGS=${n:-0}"
MISS=""
for f in attention-backend squared-relu rotary-base log-throughput tp-comm-overlap \
         ckpt-format async-save use-precision-aware-optimizer recompute-granularity \
         disable-gloo-process-groups exit-duration-in-mins fp8-format; do
  grep -qE "^ *--$f([ =]|$)" /workspace/e2e-56b/args-help.txt || MISS="$MISS --$f"
done
if [ -n "$MISS" ]; then
  echo "MISSING_FLAGS=$MISS"
else
  echo "MISSING_FLAGS=none"
fi
SELF
chmod +x "$SELF"

if [ "$SKIP_SELFCHECK" = 1 ]; then
  echo "  SKIP_SELFCHECK=1，跳过容器内自检（args-help.txt 尚未生成，训练脚本首次会自行 dump）"
else
  # 与两个训练脚本完全相同的容器参数：--gpus all / --network host / shm / ulimit
  set +e    # pipefail 下显式回收容器退出码，不能被 set -e 提前截断
  "${NC[@]}" run --rm --gpus all --network host --shm-size=64g \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -v "$ROOT:/workspace" -w /workspace/Megatron-LM \
    -e PYTHONPATH=/workspace/Megatron-LM -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    "$IMAGE" bash /workspace/e2e-56b/selfcheck.sh 2>&1 | tee "$WORKSPACE_HOST/selfcheck.log"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || { echo "!! 容器自检失败，见 $WORKSPACE_HOST/selfcheck.log"; exit 2; }
  grep -qE "^GPUS=${EXPECTED_GPUS}$" "$WORKSPACE_HOST/selfcheck.log" || {
    echo "!! 容器内不是 $EXPECTED_GPUS 张卡；检查 nvidia-container-toolkit / CDI（nerdctl --gpus all 依赖它）"; exit 2; }
  grep -q "MISSING_FLAGS=none" "$WORKSPACE_HOST/selfcheck.log" || {
    echo "!! 关键 flag 缺失，源码可能不是 core_v0.13.1，见 selfcheck.log 的 MISSING_FLAGS"; exit 2; }
  n=$(sed -n 's/^ARGS_FLAGS=//p' "$WORKSPACE_HOST/selfcheck.log" | tail -1)
  [ "${n:-0}" -ge 400 ] || { echo "!! --help 只有 ${n:-0} 条 flag，参数表没抓全"; exit 2; }
  echo "  args-help.txt 已生成（$n 行 flag），两个训练脚本会直接复用"
fi

# ---------------------------------------------------------------- 5. 下一步
echo
echo "==================== 准备完成 ===================="
echo "  ROOT           : $ROOT"
echo "  引擎           : nerdctl / namespace=$NS"
echo "  镜像           : $IMAGE"
echo "  image id       : $DIGEST"
echo "  源码           : $ROOT/Megatron-LM"
echo "  参数表         : $WORKSPACE_HOST/args-help.txt"
echo
echo "两台对齐指纹（应无输出差异）："
echo "  diff <(cat $WORKSPACE_HOST/image-digest.txt) \\"
echo "       <(ssh node-b cat $WORKSPACE_HOST/image-digest.txt)"
echo
echo "单机先跑："
echo "  cd $ROOT && bash run_1x8h200_cases.sh                 # CASE=8b 冒烟"
echo "双机跑（两机同时启动，node-a 填自己的 IP）："
echo "  MASTER_ADDR=<node-a IP> NODE_RANK=0 bash run_56b_2x8h200.sh"
echo "  MASTER_ADDR=<node-a IP> NODE_RANK=1 bash run_56b_2x8h200.sh"
