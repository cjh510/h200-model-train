#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机，两台都要各跑一次。把物料变成"能用的容器"。
#
#   cd /root/llm-bench && sha256sum -c SHA256SUMS && bash 03_deploy.sh
#
# 做四件事：校验 → docker load → 解 Megatron-LM → 容器内一次性自检（含参数名核对）。
# 自检不过就别去跑 04_bench.sh，会白等。
# =============================================================================

set -uo pipefail

ROOT="${ROOT:-/root/llm-bench}"
# 归档候选名放宽：docker save 出来的常常是未压缩 tar，只是被命名成 .tar.gz；
# Megatron 那边你传的可能是 GitHub 的 zip 而不是我脚本原先假设的 tar.gz。
TAR_IMG="${TAR_IMG:-$ROOT/pytorch*.tar $ROOT/pytorch*.tar.gz}"
MEGATRON_PKG="${MEGATRON_PKG:-$ROOT/Megatron-LM*.tar.gz $ROOT/Megatron-LM*.tgz $ROOT/Megatron-LM*.zip}"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"; cd "$ROOT" || exit 1

step() { echo; echo "==================== $* ===================="; }

# 引擎自动选：有 docker 用 docker，只有 containerd 就用 nerdctl。
# 两者的 tar 归档格式互通（OCI / docker-archive），准备机上用哪个 save 都行。
ENGINE="${ENGINE:-}"
if [ -z "$ENGINE" ]; then
  if   command -v docker  >/dev/null 2>&1; then ENGINE=docker
  elif command -v nerdctl >/dev/null 2>&1; then ENGINE=nerdctl
  else echo "!! 既没有 docker 也没有 nerdctl —— 见 02_precheck.sh 的容器引擎一项"; exit 1; fi
fi
command -v "$ENGINE" >/dev/null 2>&1 || { echo "!! ENGINE=$ENGINE 不可用"; exit 1; }
if [ "$ENGINE" = nerdctl ]; then
  # containerd 是多 namespace 的：load 和 run 用不同 namespace 会「明明导入了却说找不到镜像」
  export CONTAINERD_NAMESPACE="${CONTAINERD_NAMESPACE:-default}"
  echo "引擎 nerdctl，CONTAINERD_NAMESPACE=$CONTAINERD_NAMESPACE（全流程固定同一个，别中途改）"
else
  echo "引擎 docker"
fi

# ------------------------------------------------------------------ 0 完整性
step "0/4 完整性检查"
if [ -f SHA256SUMS ]; then
  sha256sum -c SHA256SUMS || { echo "!! 校验失败：传输中损坏。重传，别 load 半截镜像。"; exit 1; }
else
  echo "  没带 SHA256SUMS —— 只做弱校验（读文件头判格式，不保证尾部完整）："
  echo "  ★ 弱校验挡不住截断。务必和准备机 ls -l 的字节数逐个比对，差一个字节都别 load。"
  ok=1
  # shellcheck disable=SC2086
  for f in $TAR_IMG; do
    [ -e "$f" ] || continue
    # 不能用 `tar tf|head`：脚本开了 pipefail，head 提前关管道会让 tar 以 141 退出，
    # 任何正常归档都会被误判成损坏。改成直接读文件头魔数。
    b0=$(head -c 2 "$f" 2>/dev/null | od -An -tx1 2>/dev/null | tr -d ' \n')
    sig=$(dd if="$f" bs=1 skip=257 count=5 2>/dev/null | tr -d '\0')
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if   [ "$b0" = "1f8b" ]; then fmt="gzip 压缩流"
    elif [ "$sig" = "ustar" ]; then fmt="未压缩 tar（docker save 的默认产物）"
    else fmt=""; fi
    if [ -n "$fmt" ]; then
      echo "    ✓ $f  $fmt  $sz 字节"
    else
      echo "    ✗ $f  头部既非 gzip 也非 tar（magic='$b0' sig='$sig' size=$sz）"; ok=0
    fi
  done
  [ "$ok" = 1 ] || { echo "!! 头部格式就不对：传输坏了，或传上来的不是这个文件。查：file $TAR_IMG"; exit 1; }
  echo "  弱校验挡不住中途截断。回准备机补一份再传："
  echo "    sha256sum pytorch*.tar* Megatron-LM*.* > SHA256SUMS"
fi

# ------------------------------------------------------------------ 1 镜像
step "1/4 $ENGINE load"
# shellcheck disable=SC2086
ls $TAR_IMG >/dev/null 2>&1 || { echo "!! 找不到 $TAR_IMG"; exit 1; }
IMG=$($ENGINE load -i "$(ls $TAR_IMG | head -1)" | sed -n 's/^Loaded image: //p' | tail -1)
[ -n "$IMG" ] || IMG=$($ENGINE images --format '{{.Repository}}:{{.Tag}}' | grep -m1 pytorch)
[ -n "$IMG" ] || { echo "!! load 之后仍拿不到镜像名。手动查：$ENGINE images"; exit 1; }
IMAGE_ID=$($ENGINE image inspect "$IMG" --format '{{.Id}}' | tr -d '[]"')
echo "  镜像: $IMG"
echo "  ID  : $IMAGE_ID"
echo "  ★ 记进报告，并且【两台必须完全一致】。镜像不同 → 容器内 NCCL 版本不同"
echo "    → 双机会以超时/hang 的形式失败，症状完全看不出是镜像问题。"

# ------------------------------------------------------------------ 2 框架
step "2/4 解包 Megatron-LM"
# shellcheck disable=SC2086
PKG=$(ls $MEGATRON_PKG 2>/dev/null | head -1)
[ -n "$PKG" ] || { echo "!! 没找到 Megatron 包。显式指定：MEGATRON_PKG=$ROOT/Megatron-LM-core_v0.13.1.zip bash $0"; exit 1; }
echo "  包：$PKG"
case "$PKG" in
  *.zip)  if command -v unzip >/dev/null 2>&1; then unzip -qo "$PKG" -d "$ROOT"
          else echo "    无 unzip，改用 python 标准库解包"; python3 -m zipfile -e "$PKG" "$ROOT/"; fi ;;
  *.tar.gz|*.tgz) tar xzf "$PKG" -C "$ROOT" ;;
  *.tar)          tar xf   "$PKG" -C "$ROOT" ;;
  *) echo "!! 不认识的归档类型：$PKG"; exit 1 ;;
esac

# GitHub 导出的归档顶层目录带 ref 后缀（Megatron-LM-core_v0.13.1），
# 而 selfcheck / bench.sh 的 PYTHONPATH 固定写的是 Megatron-LM，必须归一化。
if [ ! -d "$ROOT/Megatron-LM" ]; then
  cand=$(find "$ROOT" -maxdepth 1 -type d -iname 'Megatron-LM*' ! -name 'Megatron-LM' | head -1)
  [ -n "$cand" ] && mv "$cand" "$ROOT/Megatron-LM" \
    && echo "  目录已改名：$(basename "$cand") → Megatron-LM"
fi
[ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ] || {
  echo "!! 解出来没有 Megatron-LM/pretrain_gpt.py，实际找到的位置："; find "$ROOT" -maxdepth 3 -name pretrain_gpt.py; exit 1; }
grep -m1 -oE 'core_version = "[^"]+"' "$ROOT/Megatron-LM/megatron/core/package_info.py" \
  || echo "  （没取到 core 版本，进容器用 python -c 'import megatron.core as m; print(m.__version__)' 查）"

# ------------------------------------------------- 3 自检脚本（落地成文件再执行）
# 不在这里用 bash -lc '…' 内嵌：外层单引号会和 python heredoc 的引号打架，
# 而且改一行就要重跑整个 docker run。写成文件，出问题直接编辑它。
cat > "$ROOT/selfcheck.sh" <<'SELF'
#!/usr/bin/env bash
# [在容器内跑] 由 03_deploy.sh 生成。验证组件版本 + kernel 能否真加载 + 参数名对齐。
set -uo pipefail
export PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH:-}

echo "--- 组件版本 ---"
python - <<'PY'
import torch
print("torch        ", torch.__version__, " CUDA", torch.version.cuda)
print("GPU          ", torch.cuda.device_count(), "张", torch.cuda.get_device_name(0))
print("NCCL         ", ".".join(map(str, torch.cuda.nccl.version())))
try:
    import transformer_engine as te
    print("TE           ", te.__version__)
except Exception as e:
    print("TE           导入失败:", type(e).__name__, e)
try:
    import megatron.core as m
    print("megatron-core", m.__version__)
except Exception as e:
    print("megatron-core 导入失败:", type(e).__name__, e)
    print("   → 多半是 Megatron-LM 版本与容器内 TE 不配对（undefined symbol）。")
    print("     按 TE 版本换成对应的 core_rX.Y tag 重新拉源码，别硬跑。")
PY

echo
echo "--- kernel 能否真加载（import 成功不代表能跑）---"
python - <<'PY'
import torch
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
for _ in range(5):
    c = a @ a
torch.cuda.synchronize()
print("bf16 matmul  OK")
try:
    from transformer_engine.pytorch import Linear
    Linear(4096, 4096).cuda().to(torch.bfloat16)
    print("TE Linear    OK  ← Llama 走 TE fused attention 的前提")
except Exception as e:
    print("TE Linear    不可用:", type(e).__name__, str(e)[:120])
PY

echo
echo "--- dump Megatron 实际参数表 ---"
cd /workspace/Megatron-LM || exit 1
python pretrain_gpt.py --help > /workspace/logs/megatron_args.txt 2>&1
N=$(grep -cE '^ *--' /workspace/logs/megatron_args.txt)
if [ "$N" -lt 50 ]; then
  echo "!! --help 没抓到参数表，看 logs/megatron_args.txt 开头："
  head -20 /workspace/logs/megatron_args.txt
  exit 1
fi
echo "参数表 $N 条 → logs/megatron_args.txt"
echo "--- 本方案需要的参数是否存在 ---"
MISS=""
for f in num-layers hidden-size ffn-hidden-size num-attention-heads \
         group-query-attention num-query-groups kv-channels swiglu normalization \
         norm-epsilon use-rotary-position-embeddings rot-base disable-bias-linear \
         position-embedding-type untie-embeddings-and-output-weights vocab-size \
         mock-data tokenizer-type tensor-model-parallel-size pipeline-model-parallel-size \
         sequence-parallel use-distributed-optimizer recompute-granularity \
         micro-batch-size global-batch-size train-iters log-throughput num-micro-batches \
         distributed-optimizer-checkpoint-shape-checks; do
  grep -qE "^ *--$f([ =]|$)" /workspace/logs/megatron_args.txt || MISS="$MISS $f"
done
if [ -n "$MISS" ]; then
  echo "缺失:$MISS"
  echo "→ 按 logs/megatron_args.txt 找改名后的等价项，改 04_bench.sh 里的 LLAMA_8B / LLAMA_70B。"
  echo "  常见改名：rot-base↔rotary-base、num-micro-batches↔global-batch-size÷(mbs×dp)、"
  echo "  recompute-granularity 的取值 selective|full、fp8-format 可能整条不存在。"
else
  echo "全部存在 ✓ 可以开始 04_bench.sh --phase hw"
fi
SELF

step "3/4 容器内自检"
# --network host：bridge 下 NCCL 会挑到 docker0，带宽掉一个数量级，
# 而且表现是"能跑但很慢"，很容易被当成模型/软件问题。
$ENGINE run --rm --gpus all --network host --shm-size=64g \
  --ulimit memlock=-1:-1 \
  -v "$ROOT":/workspace -w /workspace \
  "$IMG" bash /workspace/selfcheck.sh 2>&1 | tee "$LOGDIR/selfcheck.txt"

echo "$IMAGE_ID" > "$ROOT/.image_id"

step "4/4 下一步"
cat <<GO
两台都重复到这里，然后逐台对比：
  cat /root/llm-bench/logs/selfcheck.txt | head -20
  → torch / CUDA / TE / megatron-core / NCCL 五个版本两台必须完全一致

开始压测（都在裸机上敲，脚本内部会自己起容器）：
  bash 04_bench.sh --phase hw        # G3 硬件基线
  bash 04_bench.sh --phase s8b       # 单机 8B：并行/batch/长序列/FP8
  bash 04_bench.sh --phase dx8b      # 双机 8B：机间带宽门禁
  bash 04_bench.sh --phase d70b      # 双机 70B：TP8×PP2
GO
