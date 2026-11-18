#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机，两台都要各跑一次。把物料变成「能用的容器」。
#
#   cd /root/llm-bench && sha256sum -c SHA256SUMS && bash 03_deploy.sh
#   IMAGE=<已存在镜像> bash 03_deploy.sh          # 镜像已在引擎里，不导入 tar
#
# 相对原方案的变化（详见 变更清单.md）：
#   · 引擎自动探测：docker 优先，没有可用 docker 时回落 nerdctl（见 lib.sh）
#   · 镜像优先复用当前引擎已有的，没有再 load -i
#   · 记录 digest 到 03-deploy/env.txt（结果指纹的一部分）
#   · 自检里 8 卡断言改成「非 8 张就 exit 1」（原来只 print）
#   · 删掉假缺失的 num-micro-batches / distributed-optimizer-checkpoint-shape-checks
#   · 自检输出 ENVF 机器可读行，由本脚本汇总成 03-deploy/env.txt
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
lc_init || exit 2

ROOT="${ROOT:-/root/llm-bench}"
NPROC="${NPROC:-8}"; NNODES="${NNODES:-1}"; WORLD=$(( NPROC * NNODES ))
export ROOT NPROC NNODES WORLD
TAR_IMG="${TAR_IMG:-$ROOT/pytorch*.tar $ROOT/pytorch*.tar.gz}"
MEGATRON_PKG="${MEGATRON_PKG:-$ROOT/Megatron-LM*.tar.gz $ROOT/Megatron-LM*.tgz $ROOT/Megatron-LM*.zip}"
START_TS=$(date +%s)
lc_run_resume deploy || exit 2
LOGDIR=$(lc_run_stage_dir deploy) || exit 2
mkdir -p "$LOGDIR"; cd "$ROOT" || exit 1
export LOGDIR
lc_engine_log

step() { echo; echo "==================== $* ===================="; }

TV=$(lc_toolkit_ver)
lc_toolkit_warn "$TV"          # 版本只警告、恒 return 0：继续 load / 解包 / 8 卡自检
echo "engine=$ENGINE${NS:+  namespace=$NS}"

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
    # 不用 `tar tf|head`：pipefail 下 head 提前关管道会让 tar 以 141 退出，归档会被误判成损坏。
    b0=$(head -c 2 "$f" 2>/dev/null | od -An -tx1 2>/dev/null | tr -d ' \n')
    sig=$(dd if="$f" bs=1 skip=257 count=5 2>/dev/null | tr -d '\0')
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if   [ "$b0" = "1f8b" ]; then fmt="gzip 压缩流"
    elif [ "$sig" = "ustar" ]; then fmt="未压缩 tar（docker/nerdctl save 的默认产物）"
    else fmt=""; fi
    if [ -n "$fmt" ]; then
      echo "    ✓ $f  $fmt  $sz 字节"
    else
      echo "    ✗ $f  头部既非 gzip 也非 tar（magic='$b0' sig='$sig' size=$sz）"; ok=0
    fi
  done
  [ "$ok" = 1 ] || { echo "!! 头部格式就不对：传输坏了，或传上来的不是这个文件。"; exit 1; }
fi

# ------------------------------------------------------------------ 1 镜像
step "1/4 镜像（$ENGINE${NS:+ / namespace=$NS}）"
IMG="${IMAGE:-$(lc_pick_image 'pytorch' '' || true)}"
if [ -n "$IMG" ] && lc_image_exists "$IMG"; then
  echo "  复用已有镜像：$IMG（不重复 load）"
else
  # shellcheck disable=SC2086
  TARF=$(ls $TAR_IMG 2>/dev/null | head -1)
  [ -n "$TARF" ] || { echo "!! 当前引擎里没有 pytorch 镜像，也没有 $TAR_IMG"; echo "   → IMAGE=<镜像> 指定，或把 pytorch*.tar 放 $ROOT"; exit 1; }
  echo "  $ENGINE_BIN load -i $TARF"
  BEFORE=$(lc_images | sort)
  lc_load "$TARF" || {
    echo "!! 镜像导入失败。docker：journalctl -u docker -n 50；nerdctl：journalctl -u containerd -n 50"
    exit 1
  }
  AFTER=$(lc_images | sort)
  NEW=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)
  IMG="${NEW:-$(lc_pick_image 'pytorch' '' || true)}"
  [ -n "$IMG" ] || { echo "!! load 完仍拿不到镜像名。手动查：$(lc_engine_version) 的 lc images 列表"; exit 1; }
fi
DIGEST=$(lc_digest "$IMG")
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
CEIL=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | head -1 | awk '{print $3}')
echo "  镜像  : $IMG"
echo "  digest: $DIGEST"
echo "  ★ 两台必须完全一致（含 digest 与 engine）。引擎/镜像不同 → 容器内 NCCL 版本与注入路径不同 → 双机以超时/hang 形式失败。"

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

# GitHub 归档顶层目录带 ref 后缀（Megatron-LM-core_v0.13.1），而 PYTHONPATH 固定写 Megatron-LM
if [ ! -d "$ROOT/Megatron-LM" ]; then
  cand=$(find "$ROOT" -maxdepth 1 -type d -iname 'Megatron-LM*' ! -name 'Megatron-LM' | head -1)
  [ -n "$cand" ] && mv "$cand" "$ROOT/Megatron-LM" \
    && echo "  目录已改名：$(basename "$cand") → Megatron-LM"
fi
[ -f "$ROOT/Megatron-LM/pretrain_gpt.py" ] || {
  echo "!! 解出来没有 Megatron-LM/pretrain_gpt.py，实际找到的位置："; find "$ROOT" -maxdepth 3 -name pretrain_gpt.py; exit 1; }

# ------------------------------ 2.1 Megatron 源码补丁（core_v0.13.1 已知 bug）
# fused_mla_yarn_rope_apply.py backward 对非连续梯度张量用 .view()：mbs>1 时梯度带跨维度
# stride → "view size is not compatible ... Use .reshape(...) instead"（11-08 单机实测
# mbs=2 实测 8 rank 全崩）。.reshape 与 .view 语义等价（兼容时零拷贝），对 autograd 安全。
MLA_FUSE="$ROOT/Megatron-LM/megatron/core/fusions/fused_mla_yarn_rope_apply.py"
if [ -f "$MLA_FUSE" ] && grep -q '\.view(-1, nheads' "$MLA_FUSE"; then
  sed -i 's/\.view(-1, nheads/.reshape(-1, nheads/g' "$MLA_FUSE"
  echo "  已打补丁：fused_mla_yarn_rope_apply.py 的 .view(-1, nheads → .reshape（MLA backward 非连续梯度）"
fi

# ------------------------------------------------- 3 自检脚本（落地成文件再执行）
cat > "$ROOT/selfcheck.sh" <<'SELF'
#!/usr/bin/env bash
# [在容器内跑] 由 03_deploy.sh 生成。验证组件版本 + kernel 能否加载 + 参数名对齐。
# 8 卡断言：不是 8 张就 exit 1（原方案只 print，后面 04 会拿着坏容器跑一整夜）。
set -uo pipefail
export PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH:-}
RC=0

echo "--- 组件版本 ---"
python - <<'PY' || RC=1
import sys, torch
print("ENVF torch=" + torch.__version__)
print("ENVF cuda=" + str(torch.version.cuda))
print("ENVF nccl=" + ".".join(map(str, torch.cuda.nccl.version())))
n = torch.cuda.device_count()
print("ENVF gpus=" + str(n))
print("CUDA 可见 %d 张卡；第一张：%s" % (n, torch.cuda.get_device_name(0) if n else "无"))
if n != 8:
    print("!! 容器内只看到 %d 张卡（要求 8）—— 宿主有 8 张则是运行时/toolkit 注入问题；宿主也不足则是驱动/设备问题" % n)
    sys.exit(1)
try:
    import transformer_engine as te
    print("ENVF te=" + te.__version__)
except Exception as e:
    print("ENVF te=unknown")
    print("TE 导入失败:", type(e).__name__, e)
try:
    import megatron.core as m
    print("ENVF megatron=" + m.__version__)
except Exception as e:
    print("ENVF megatron=unknown")
    print("megatron-core 导入失败:", type(e).__name__, e)
    print("   → 多半是 Megatron-LM 版本与容器内 TE 不配对（undefined symbol）。")
    print("     按 TE 版本换成对应的 core_rX.Y tag 重新拉源码，别硬跑。")
PY

echo
echo "--- 容器限额 ---"
echo "ENVF memlock=$(ulimit -l)"
echo "ENVF shm=$(df -h /dev/shm | awk 'NR==2{print $2}')"
[ "$(ulimit -l)" = "unlimited" ] || echo "!! memlock=$(ulimit -l) 不是 unlimited：RDMA 注册大 buffer 会失败"

echo
echo "--- kernel 能否真加载（import 成功不代表能跑）---"
python - <<'PY' || RC=1
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

echo "--- 必需参数（缺一个就说明源码/分支不对，直接失败）---"
MISS=""
for f in num-layers hidden-size ffn-hidden-size num-attention-heads \
         group-query-attention num-query-groups kv-channels swiglu normalization \
         norm-epsilon use-rotary-position-embeddings rotary-base disable-bias-linear \
         position-embedding-type untie-embeddings-and-output-weights vocab-size \
         seq-length max-position-embeddings mock-data tokenizer-type bf16 \
         tensor-model-parallel-size pipeline-model-parallel-size \
         micro-batch-size global-batch-size train-iters; do
  grep -qE "^ *--$f([ =]|$)" /workspace/logs/megatron_args.txt || MISS="$MISS $f"
done
if [ -n "$MISS" ]; then
  echo "!! 必需参数缺失:$MISS"
  echo "   → 源码不是 core_v0.13.1，或解包层级不对。别继续。"
  RC=1
else
  echo "必需参数全部存在 ✓"
fi

echo "--- 可选性能开关（缺了会被 04_bench.sh 自动裁掉，不影响启动）---"
for f in sequence-parallel use-distributed-optimizer recompute-granularity \
         log-throughput timing-log-level exit-duration-in-mins \
         no-create-attention-mask-in-dataloader overlap-grad-reduce overlap-param-gather \
         accumulate-allreduce-grads-in-fp32 attention-softmax-in-fp32; do
  grep -qE "^ *--$f([ =]|$)" /workspace/logs/megatron_args.txt \
    || echo "   · 不存在：--$f（04_bench.sh 会自动裁掉；soak 档位实测缺 --exit-duration-in-mins 会拒绝跑）"
done

echo
echo "--- 说明 ---"
echo "num-micro-batches 不是命令行参数，它 = GBS ÷ (mbs × dp)，由 04_bench.sh 自己算并打印；"
echo "distributed-optimizer-checkpoint-shape-checks 在 core_v0.13.1 里也不存在，原方案把它列进"
echo "「必需参数」是误报，已删除。"
exit $RC
SELF

step "3/4 容器内自检"
# --network host：bridge 下 NCCL 会挑到 docker0/nerdctl0，带宽掉一个数量级，且症状是「能跑但很慢」。
# 两个引擎都由 lib.sh 的 lc_run 统一注入 --gpus all / --shm-size / --ulimit / --network host。
if ! lc_run -v "$ROOT":/workspace -v "$LOGDIR":/workspace/logs -w /workspace \
      "$IMG" bash /workspace/selfcheck.sh 2>&1 | tee "$LOGDIR/selfcheck.txt"; then
  echo
  echo "!! 自检失败（见上）。最常见三种："
  echo "   1) 8 卡断言失败（容器内 device_count≠8）→ 先分两类："
  echo "      a. 容器运行时/toolkit 问题：宿主 nvidia-smi -L 有 8 张、容器内看不到；$ENGINE 的 GPU 注入未生效"
  echo "         本机 toolkit=${TV:-未知}（要求 >=1.14）；如确认 toolkit 过旧，升级到 1.17.x 四个包后重试"
  echo "         （libnvidia-container1 / libnvidia-container-tools / nvidia-container-toolkit-base / nvidia-container-toolkit）"
  echo "      b. 驱动/设备问题：宿主 nvidia-smi -L 就不足 8 张，或掉卡 / MIG / CUDA_VISIBLE_DEVICES 限制"
  echo "     判定方法：${ENGINE} 内 device_count≠8 而宿主有 8 张 → 按 a；宿主不足 8 → 按 b"
  echo "     参考：$(lc_gpu_fail_hint)"
  echo "   2) 必需参数缺失 → Megatron 源码版本不对（要 core_v0.13.1）"
  echo "   3) TE Linear 不可用 → 容器 TE 与源码不配对"
  exit 1
fi

# ------------------------------------------------------------- 4 结果指纹
step "4/4 写 03-deploy/env.txt（结果指纹）"
envf() { sed -n "s/^ENVF ${1}=//p" "$LOGDIR/selfcheck.txt" | tail -1; }
GPUS=$(envf gpus)
[ "${GPUS:-0}" = 8 ] || { echo "!! 自检报 ${GPUS:-0} 张卡，不是 8。停止。"; exit 1; }
{
  echo "host=$(hostname)"
  echo "time=$(date -Is)"
  echo "image=$IMG"
  echo "digest=$DIGEST"
  echo "engine=$ENGINE"
  echo "namespace=${NS:-none}"
  echo "driver=$DRV"
  echo "cuda_ceiling=$CEIL"
  echo "torch=$(envf torch)"
  echo "cuda=$(envf cuda)"
  echo "nccl=$(envf nccl)"
  echo "te=$(envf te)"
  echo "megatron=$(envf megatron)"
  echo "memlock=$(envf memlock)"
  echo "shm=$(envf shm)"
} > "$LOGDIR/env.txt"
echo "$DIGEST" > "$ROOT/.image_id"
lc_run_info_set image "$IMG"
lc_run_info_set digest "$DIGEST"
lc_run_info_set engine "$ENGINE"
lc_run_info_set namespace "${NS:-none}"
lc_run_info_set driver "$DRV"
lc_run_info_set cuda_ceiling "$CEIL"
cat "$LOGDIR/env.txt" | sed 's/^/  /'

step "5/5 下一步"
cat <<GO
两台都重复到这里，先对齐指纹（engine / namespace / 五个版本 / digest 两台必须一致）：
  diff <(grep -v -E 'host=|time=' $LOGDIR/env.txt) \\
       <(ssh node-b grep -v -E 'host=|time=' $LOGDIR/env.txt)

开始压测（都在裸机上敲，脚本内部自己起容器；不指定 RUN_ID 时自动续用 runs/latest）：
  bash 04_bench.sh --phase hw     # G3 硬件基线
  bash 04_bench.sh --phase coll   # collective 扫描
  bash 04_bench.sh --phase s8b    # 单机 8B：并行 / batch / 长序列 / FP8
  bash 04_bench.sh --phase dx8b   # 双机 8B：机间带宽门禁（必须给 NCCL_IF / NCCL_HCA）
  bash 04_bench.sh --phase d70b   # 双机 70B：TP8×PP2
  bash 04_bench.sh --phase mfu    # 峰值 MFU（highmem 长稳的档位来源，必须先跑）
  bash 04_bench.sh --phase soak   # 长稳（脚本内自动起 dmon，退出时 trap 清理）
  bash 04_bench.sh --phase report # 汇总到本次 run 的 10-report/

本次结果目录：$LC_RUN_DIR
GO
lc_run_stage_done deploy "$START_TS"
exit 0
