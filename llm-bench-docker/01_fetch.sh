#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 一台【能访问 nvcr.io 和 github】的机器 —— 不是目标裸机。
#          目标机无外网，所有物料在这里落地成 tar，再 scp 过去。
#
# 产出三件东西（TESTPLAN.md 第 4 节的清单）：
#   out/pytorch-<tag>.tar        NGC 镜像，~9 GB
#   out/Megatron-LM.tar.gz       框架源码，~60 MB
#   out/SHA256SUMS               传输后校验用
#
# 权重和数据集都不需要：压测走 --mock-data + NullTokenizer + 随机初始化。
#
# 用法：
#   NV_API_KEY=nvapi-xxxx bash 01_fetch.sh
#   NV_API_KEY=... NGC_PYTORCH_TAG=25.01-py3 MEGATRON_REF=core_v0.12.3 bash 01_fetch.sh
#   （25.01 是 TE 1.14，配 core_v0.12.3；默认 25.06 + core_v0.13.1）
#   （tag 必须按 G0 在两台裸机上采到的【较低】CUDA 上限来选，见 TESTPLAN 第 4 节）
# =============================================================================

set -euo pipefail

OUT="${OUT:-$PWD/out}"
NGC_PYTORCH_TAG="${NGC_PYTORCH_TAG:-25.06-py3}"     # CUDA 12.9.1 / TE 2.4，R570 靠 minor 兼容可跑
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-core_v0.13.1}"        # tag 是 core_v0.x.y；分支才叫 core_r0.x.0。
# 25.06 容器是 TE 2.4：core_v0.13.1 不 pin TE 版本可用；core_v0.14.0 要 TE>=2.6,<2.8；
# core_v0.15.x 要 TE>=2.9,<2.10 —— 都会撞。别用 main。
IMAGE="nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}"

step() { echo; echo "==================== $* ===================="; }
die() { echo "!! $*" >&2; exit 1; }

command -v docker >/dev/null || die "本机没有 docker"
mkdir -p "$OUT"; cd "$OUT"

# ---------------------------------------------------------------- 0 选对版本
step "0/4 确认 tag 存在并看清它的 CUDA / TE 版本"
# manifest inspect 只读元数据，不会下 9GB
docker manifest inspect "$IMAGE" >/dev/null 2>&1 \
  || die "$IMAGE 拉不到：tag 不存在、没登录，或这台机器也上不了 nvcr.io
   → docker login nvcr.io   Username: \$oauthtoken   Password: <NGC API Key>
   → tag 列表 https://catalog.ngc.nvidia.com/containers/nvidia/pytorch/tags"
echo "  ✓ $IMAGE 存在"

if [ -n "${NV_API_KEY:-}" ]; then
  echo "$NV_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
else
  echo "  没给 NV_API_KEY，先试匿名拉取；报 unauthorized 再带上 key 重跑"
fi

# ---------------------------------------------------------------------- 1 镜像
step "1/4 docker pull + save（~9GB，磁盘要留 25GB 以上）"
df -h . | awk 'NR==2{print "  当前可用: "$4}'
docker pull "$IMAGE"
docker save "$IMAGE" -o "pytorch-${NGC_PYTORCH_TAG}.tar"
ls -lh "pytorch-${NGC_PYTORCH_TAG}.tar"

# ------------------------------------------------------------------ 2 框架源码
step "2/4 Megatron-LM 源码（按 tag 锁版本）"
command -v git >/dev/null || die "本机没有 git"
rm -rf Megatron-LM
# --branch 直接拉那个 tag，避免 clone main 再 checkout（main 与容器 TE 不配对会 undefined symbol）
git clone --depth 1 --branch "$MEGATRON_REF" --single-branch "$MEGATRON_REPO" Megatron-LM
( cd Megatron-LM && echo "  commit: $(git rev-parse --short HEAD)  $(git describe --tags --always)" )
tar czf Megatron-LM.tar.gz Megatron-LM
rm -rf Megatron-LM
ls -lh Megatron-LM.tar.gz

# ------------------------------------------------------------ 3 校验与打包清单
step "3/4 生成校验"
# 9GB 走网络传输出错不算罕见，到了裸机一定比一次
sha256sum pytorch-*.tar Megatron-LM.tar.gz > SHA256SUMS
cat SHA256SUMS

step "4/4 完成 —— 传到裸机"
cat <<DONE

在【这台准备机】上执行（Windows 侧改用 PowerShell 的 scp 即可）：

  scp pytorch-${NGC_PYTORCH_TAG}.tar Megatron-LM.tar.gz SHA256SUMS \\
      root@<node-a>:/root/llm-bench/ && \\
  scp pytorch-${NGC_PYTORCH_TAG}.tar Megatron-LM.tar.gz SHA256SUMS \\
      root@<node-b>:/root/llm-bench/

  # 本方案的 5 个脚本单独传（它们在 Windows 上，别混进 out/）
  scp 02_precheck.sh 03_deploy.sh 04_bench.sh 05_report.py TESTPLAN.md \\
      root@<node-a>:/root/llm-bench/

两台都要传同一份 —— 双机必须用【同一个镜像 ID】，否则 NCCL 版本不一致，
失败形态是各种看不懂的超时和 hang，很难想到是镜像不同。

到了裸机先验完整性（在【目标机】上）：
  cd /root/llm-bench && sha256sum -c SHA256SUMS
  → 两行都要输出 OK；任何一行 FAILED 就重传，别硬着往下走。

DONE
