#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机（脚本内部自己起容器）。压测矩阵的总入口。
#
#   bash 04_bench.sh --phase hw        # G3 硬件/通信基线（单机）
#   bash 04_bench.sh --phase s8b       # 单机 8卡 Llama-3.1-8B：S1 S2 S3 S4
#   bash 04_bench.sh --phase dx8b      # 双机 8B TP8×DP2 —— 机间带宽门禁
#   bash 04_bench.sh --phase d70b      # 双机 16卡 Llama-3.1-70B：TP8×PP2
#   bash 04_bench.sh --phase soak      # 长稳（另开终端采 dmon）
#   bash 04_bench.sh --phase report    # 汇总成表
#
# 双机：两台都执行同一条命令，只有 NODE_RANK 不同。MASTER_ADDR 两台必须填
# 同一个值 = NODE_RANK=0 那台机器的 IP（不是各自的 IP！填错就是两台各等各的 store，
# 表现成"卡住不动"，要等到超时才报错）。
#   node-a: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> NCCL_IF=bond0 NCCL_IB=0 \
#           NCCL_HCA=mlx5_bond_2,mlx5_bond_3 bash 04_bench.sh --phase d70b
#   node-b: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> NCCL_IF=bond0 NCCL_IB=0 \
#           NCCL_HCA=mlx5_bond_2,mlx5_bond_3 bash 04_bench.sh --phase d70b
#   先起 node-a，1 分钟之内起 node-b；两边都等对端，只起一台不会报错只会等。
#
# 可调：IMAGE / ROOT / NPROC / NCCL_IF / NCCL_HCA / NCCL_IB / NCCL_DEBUG / STEPS / FORCE=1
#
# 所有带值 flag 一律写成 --name=value：这样 prune_optional 能"一项一裁"，
# 不会因为删掉一个 flag 而连带把它后面的取值喂给下一个参数。
# =============================================================================

set -uo pipefail

ROOT="${ROOT:-/root/llm-bench}"
LOGDIR="$ROOT/logs"
MEGATRON=/workspace/Megatron-LM
# 引擎自动选：裸机只有 containerd 时就走 nerdctl。注意 load 和 run 必须同一个
# CONTAINERD_NAMESPACE，否则报「镜像不存在」—— 这是 nerdctl 最常见的假故障。
ENGINE="${ENGINE:-}"
if [ -z "$ENGINE" ]; then
  if   command -v docker  >/dev/null 2>&1; then ENGINE=docker
  elif command -v nerdctl >/dev/null 2>&1; then ENGINE=nerdctl
  else echo "!! 既没有 docker 也没有 nerdctl"; exit 1; fi
fi
if [ "$ENGINE" = nerdctl ]; then
  export CONTAINERD_NAMESPACE="${CONTAINERD_NAMESPACE:-default}"
  echo "引擎 nerdctl / namespace=$CONTAINERD_NAMESPACE（要和 03_deploy.sh 导入时用的保持一致）"
else
  echo "引擎 docker"
fi
IMAGE="${IMAGE:-$($ENGINE images --format '{{.Repository}}:{{.Tag}}' | grep -m1 pytorch)}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
STEPS="${STEPS:-15}"
NCCL_IF="${NCCL_IF:-}"        # 例 bond0 / eth0；取 02_precheck 里 state=up 且 speed>=10000 的那张
NCCL_IB="${NCCL_IB:-}"        # 0=用 IB，1=禁用；02_precheck 有 /sys/class/infiniband 就填 0
FORCE="${FORCE:-0}"
mkdir -p "$LOGDIR"
[ -n "$IMAGE" ] || { echo "!! 没找到 pytorch 镜像，先跑 03_deploy.sh"; exit 1; }

WORLD=$(( NPROC * NNODES ))
echo "IMAGE=$IMAGE  world=$WORLD (nproc=$NPROC × nodes=$NNODES) rank=$NODE_RANK master=$MASTER_ADDR steps=$STEPS"

# RoCE/IB 设备节点必须挂进容器，否则跨机 NCCL 只能退回 socket 走 bond0（慢一个数量级，
# 而且会把"这台机器跨机性能差"记成验收结论）。--network host 只给网络命名空间，不给 verbs。
IB_DEV=""
# 只在双机时注入：单机走 NVLink/SHM，压根不碰 verbs，不该让新参数牵连已经验证过的单机档。
# nerdctl 的 --device 只收设备节点，给目录直接 FATA "not a device node"（docker 才允许目录），
# 所以这里逐个枚举字符设备：uverbs*/rdma_cm/umad* 都得进去，缺一个 verbs 就起不来。
if [ "$NNODES" -gt 1 ] && [ -d /dev/infiniband ]; then
  # 只挂 uverbs* 和 rdma_cm：issm*/umad* 是子网管理/MAD 接口，NCCL 用不到，
  # 生产 K8s 节点上不该多给这份权限。
  for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
    [ -c "$d" ] && IB_DEV="$IB_DEV --device $d"
  done
  [ -n "$IB_DEV" ] && echo "RDMA 设备注入：$(echo "$IB_DEV" | grep -o '/dev/infiniband/[^ ]*' | paste -sd, -)"
fi

# 双机前置门禁。表现是"卡住不动"，实际是两台各连各的 store，要等到超时才吐错。
if [ "$NNODES" -gt 1 ]; then
  case "$MASTER_ADDR" in
    ""|127.*|localhost) echo "!! 双机必须 MASTER_ADDR=<NODE_RANK=0 那台的实际 IP>，且两台的值一模一样（现在=$MASTER_ADDR）"; exit 2;;
  esac
  [ "$NODE_RANK" = 0 ] || [ "$NODE_RANK" = 1 ] || { echo "!! NODE_RANK 只能是 0 或 1（现在=$NODE_RANK）"; exit 2; }
  [ -n "$IB_DEV" ] || { echo "!! 宿主机没有 /dev/infiniband —— 跨机只能走 socket，数字不可用"; exit 2; }
  [ -n "$NCCL_HCA" ] || echo "   ! 没给 NCCL_HCA：NCCL 会自己挑 HCA，可能挑到不通的那张。RoCE 机建议 NCCL_HCA=$(ls /sys/class/infiniband 2>/dev/null | paste -sd, -)"
  if [ "$NODE_RANK" != 0 ]; then
    ping -c1 -W2 "$MASTER_ADDR" >/dev/null 2>&1 \
      || { echo "!! 本机 ping 不通 MASTER_ADDR=$MASTER_ADDR —— 先解决路由/防火墙，否则只会等到超时"; exit 2; }
    echo "    · MASTER_ADDR=$MASTER_ADDR 可达；等对端起 torchrun 后端口才打开，用 nc -z $MASTER_ADDR <port> 验"
  fi
fi

# 每档端口 = 基址 + cksum(tag)%200。必须确定性强：两台对同一个 tag 要算出同一个号。
# （原来用 RANDOM，两台各摇各的 → 端口分叉 → 双机永远连不上，而且重跑无法复现）
det_port() {
  local h
  h=$(printf '%s' "$1" | cksum | awk '{print $1}')
  echo $(( MASTER_PORT + h % 200 ))
}

# ------------------------------------------------------------- Llama 模型定义
# 一手来源：meta-llama Llama-3.1-8B / 70B 公开 config。
#   8B ：32 层 / hidden 4096 / ffn 14336 / 32 heads / 8 kv / head_dim 128 → 8.03B
#   70B：80 层 / hidden 8192 / ffn 28672 / 64 heads / 8 kv / head_dim 128 → 70.55B
# vocab 128256 本身已是 128 的倍数，make-vocab-size-divisible-by 不改变尺寸。
# 结构参数【不参与自动裁剪】—— 缺了会静默把模型变成另一个形状，测出的数字就是假的。
COMMON=(
  --swiglu --normalization=RMSNorm --norm-epsilon=1e-5
  --use-rotary-position-embeddings --rotary-base=500000
  --disable-bias-linear --position-embedding-type=rope
  --untie-embeddings-and-output-weights
  --group-query-attention --num-query-groups=8 --kv-channels=128
  --attention-dropout=0.0 --hidden-dropout=0.0
  --vocab-size=128256 --make-vocab-size-divisible-by=128
  --tokenizer-type=NullTokenizer        # 不挂词表文件，离线环境的关键
  --mock-data --split=100,0,0           # 合成数据：压测不需要语料
)
LLAMA_8B=( "${COMMON[@]}" --num-layers=32 --hidden-size=4096 --ffn-hidden-size=14336 --num-attention-heads=32 )
LLAMA_70B=( "${COMMON[@]}" --num-layers=80 --hidden-size=8192 --ffn-hidden-size=28672 --num-attention-heads=64 )

# 只有这些"性能开关类"允许缺失后自动裁掉 —— 它们只影响优化路径，不改变模型形状。
# 裸 flag 必须是 store_true；带值的（如 --exit-interval）一律写全 --name=value，
# 否则 argparse 会吃掉下一个 token 当值，报 "expected one argument"。
# --no-create-attention-mask-in-dataloader：causal mask 不在 dataloader 里实体化，
# 交给 TE 的 AttnMaskType.causal。mask 是 [1,seq,seq] 的 bool，131072² ×1B=17.2 GB/卡，
# 8 卡共用容器 /dev/shm(64 GB) 要 137 GB → worker 直接 SIGBUS；而且长序列下每步生成
# 这个张量会盖住真实算力，测出来的数字没法用。
OPTIONAL_FLAGS=(
  --sequence-parallel --use-distributed-optimizer --overlap-grad-reduce
  --overlap-param-gather --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --recompute-granularity=selective
  --timing-log-level=2 --log-throughput
  --no-create-attention-mask-in-dataloader
)

prune_optional() {
  local table="$LOGDIR/megatron_args.txt" out=() t name
  # 提示必须走 stderr：调用方用 mapfile 捕 stdout，stdout 里的任何文字都会被当成
  # 一个参数拼进 pretrain_gpt.py 的命令行。
  for t in "$@"; do
    [ -n "$t" ] || continue
    if [ -s "$table" ] && [ "${t:0:2}" = "--" ]; then
      name=${t%%=*}; name=${name#--}
      grep -qE "^ *--$name([ =]|$)" "$table" || { echo "    · 裁掉（容器不支持）: $t" >&2; continue; }
      # 裸 flag 但帮助里同一行紧跟 metavar（EXIT_INTERVAL / INT / {a,b}）→ 它要取值，
      # 传进去必定炸 argparse，这里主动裁掉并说明，别让整条矩阵白跑。
      # metavar 必须全大写或被 {} 包住，且后面紧跟空格/行尾。argparse 的同行说明是
      # "If set, ..." 这种带小写的句子，靠这点把 store_true 的注释和 metavar 分开。
      if [ "$t" = "--$name" ] && grep -qE "^ *--$name (\{[^ ]*\}|[A-Z0-9_]+)([ ]|$)" "$table"; then
        echo "    · 裁掉（裸 flag 需要取值，写法不对）: $t" >&2; continue
      fi
    fi
    out+=( "$t" )
  done
  printf '%s\n' "${out[@]}"
}

# 双机时任何一档真失败就整阶段停：脚本没有跨节点同步机制，一边停一边继续会永久错位，
# 之后每一档都在跟对端不同 tag 的端口对撞，症状是"后面全部莫名其妙挂掉"。
dual_stop() {
  [ "$NNODES" -gt 1 ] || return 0
  echo "   !! 双机模式：这一档失败，整阶段停止。两台必须同时重启，从同一档开始。"
  exit 2
}

# --------------------------------------------------------------- 一次训练
# run_train <tag> <model数组名> <tp> <pp> <mbs> <seq> <gbs> [额外 flag...]
run_train() {
  local tag=$1 modelname=$2 tp=$3 pp=$4 mbs=$5 seq=$6 gbs=$7; shift 7
  local -n MODEL=$modelname
  tag="w${WORLD}_${tag}"
  local log="$LOGDIR/$tag.log"

  # 跳过判据两台必须一致：PP>1 时只有承载最后一级的那台日志里有 iteration 行，
  # 只看日志会让 rank0 那台重跑、rank1 那台跳过 → 永久错位。所以双机再认 results.jsonl 的 OK。
  if [ "$FORCE" != "1" ] && { grep -q "elapsed time per iteration" "$log" 2>/dev/null \
       || { [ "$NNODES" -gt 1 ] && grep -q "\"tag\":\"$tag\".*\"status\":\"OK\"" "$LOGDIR/results.jsonl" 2>/dev/null; }; }; then
    echo "→ 跳过（已有结果）$tag"; return 0
  fi

  local dp=$(( WORLD / tp / pp ))
  if (( dp * tp * pp != WORLD )); then
    echo "   !! TP=$tp PP=$pp 无法整除 world=$WORLD，跳过"; return 1
  fi
  if (( gbs % (mbs * dp) != 0 )); then
    echo "   !! GBS=$gbs 不能被 mbs×dp=$(( mbs * dp )) 整除，跳过"
    echo "      Megatron 的 num_micro_batches = GBS ÷ (mbs × dp)，必须整除"; return 1
  fi
  local nmib=$(( gbs / (mbs * dp) ))
  local port
  port=$(det_port "$tag")
  echo "→ $tag   mbs=$mbs dp=$dp → num_micro_batches=$nmib   rendezvous 端口=$port（两台必须相同）"
  # PP 的气泡 ≈ (pp-1)/nmib：nmib=4 且 pp=2 就是 25%，白测
  if (( pp > 1 )) && (( nmib < 2 * pp )); then
    echo "   ! micro-batch 太少（$nmib），PP=$pp 流水线气泡约 $(awk -v p=$pp -v m=$nmib 'BEGIN{printf "%.0f%%", (p-1)/m*100}')；加大 GBS 再比"
  fi

  local par=( --tensor-model-parallel-size="$tp" --pipeline-model-parallel-size="$pp" )
  # --sequence-parallel 只在 TP>1 时有意义，TP=1 传了 Megatron 会 assert 失败
  local candidates=() f
  for f in "${OPTIONAL_FLAGS[@]}"; do
    [ "$f" = "--sequence-parallel" ] && (( tp <= 1 )) && continue
    candidates+=( "$f" )
  done
  # 双机额外禁用 Gloo 进程组：Megatron 为 dp/cp 建的 Gloo 组是纯 TCP，靠主机名互相广播，
  # 而容器主机名是随机 ID 且容器内无 DNS/对端 hosts（getaddrinfo err=-3）→ 退化成
  # 127.0.0.1 → connectFullMesh 拒连。NCCL 组用 IP，不受影响。dp=1 的档碰不到这条路径，
  # 所以只有 D4 那种 dp>1 的档会挂。单机不加，保持已验证矩阵的数据路径一字不改。
  (( NNODES > 1 )) && candidates+=( --disable-gloo-process-groups )
  mapfile -t opts < <(prune_optional "${candidates[@]}" "$@")

  $ENGINE run --rm --gpus all --network host --shm-size=64g \
    ${IB_DEV:+$IB_DEV} \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -v "$ROOT":/workspace -w "$MEGATRON" \
    -e PYTHONPATH="$MEGATRON" -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    ${NCCL_IF:+-e NCCL_SOCKET_IFNAME=$NCCL_IF} \
    ${NCCL_HCA:+-e NCCL_IB_HCA=$NCCL_HCA} \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    ${NCCL_IB:+-e NCCL_IB_DISABLE=$NCCL_IB} \
    "$IMAGE" torchrun --nproc_per_node="$NPROC" --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
      --master_port="$port" \
    pretrain_gpt.py \
      "${MODEL[@]}" "${par[@]}" "${opts[@]}" \
      --seq-length="$seq" --max-position-embeddings="$seq" \
      --micro-batch-size="$mbs" --global-batch-size="$gbs" \
      --bf16 --init-method-std=0.02 --clip-grad=1.0 --weight-decay=0.1 \
      --lr=1e-4 --min-lr=1e-5 --lr-decay-style=cosine --lr-warmup-iters=5 \
      --train-iters="$STEPS" --log-interval=1 --eval-iters=0 --seed=1234 \
      > "$log" 2>&1
  local rc=$?

  if grep -qiE "CUDA out of memory|OutOfMemoryError" "$log"; then
    echo "   ✗ OOM  $tag"
    printf '{"tag":"%s","tp":%s,"pp":%s,"mbs":%s,"seq":%s,"gbs":%s,"world":%s,"status":"OOM"}\n' \
      "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$WORLD" >> "$LOGDIR/results.jsonl"
    dual_stop
  elif grep -q "elapsed time per iteration" "$log"; then
    echo "   ✓ $tag   $(grep -oE 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" | tail -1)"
    # 成功也要落一行：05_report 按 tag 取最后一次状态，不写的话早期失败会一直挂着。
    printf '{"tag":"%s","tp":%s,"pp":%s,"mbs":%s,"seq":%s,"gbs":%s,"world":%s,"status":"OK"}\n' \
      "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$WORLD" >> "$LOGDIR/results.jsonl"
  elif [ "$NNODES" -gt 1 ] && [ "$rc" = 0 ]; then
    # PP>1 时 Megatron 用 print_rank_last 打 iteration 行，那是最后一个流水线级的 last
    # rank，落在 NODE_RANK 最大的那台机器上。本机容器正常退出即成功，数字要去对端取。
    # （原来只看"日志里有没有 iteration"，于是 rank0 那台每一档都误报失败）
    echo "   ✓ $tag（本机没有 iteration 行是正常的：PP 最后一级在 node_rank=$(( NNODES - 1 )) 那台，数字看它的同名日志）"
    printf '{"tag":"%s","tp":%s,"pp":%s,"mbs":%s,"seq":%s,"gbs":%s,"world":%s,"status":"OK"}\n' \
      "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$WORLD" >> "$LOGDIR/results.jsonl"
  else
    echo "   ✗ 没跑出 iteration（容器退出码 $rc）。真错误候选 + 日志末尾："
    # 模式必须精准：裸 nan 会撞上 "Ronan Collobert" 和 check_for_nan_in_grad，
    # Copyright 行也会占满名额，导致真错误反而打不出来（双机 70B 那次就是这样）。
    grep -vE 'Copyright|^\s*$' "$log" 2>/dev/null \
      | grep -nE 'unrecognized arguments|invalid choice|Error|error:|Assertion|out of memory|NCCL WARN|ibv_|GID|exitcode|Killed|core dumped|Signal|number of nan iterations: [1-9]' \
      | head -12 | sed 's/^/      /'
    awk '/Traceback \(most recent call last\)/{f=1} f{print; if(++n>30) exit}' "$log" | sed 's/^/      /'
    # 上面两条都可能空（进程被 SIGKILL、或 NCCL 在超时前先退出），那就只看结尾
    echo "      —— 末尾 20 行 ——"
    tail -20 "$log" | sed 's/^/      /'
    echo "      完整日志：$log"
    printf '{"tag":"%s","tp":%s,"pp":%s,"mbs":%s,"seq":%s,"gbs":%s,"world":%s,"status":"FAILED"}\n' \
      "$tag" "$tp" "$pp" "$mbs" "$seq" "$gbs" "$WORLD" >> "$LOGDIR/results.jsonl"
    [ "$NNODES" -gt 1 ] && echo "      双机时真错误常在对端的同名日志里，两台都要看"
    dual_stop
  fi
}

# 归一化：`--phase hw` 和 `hw` 两种写法都接受。
# （之前 case 用 ${1}:${2} 直接匹配，而 $1 是字面量 "--phase"，
#   于是 "--phase:hw" 永远对不上 hw:* 分支，所有阶段都会掉进 usage）
if [ "${1:-}" = "--phase" ]; then set -- "${2:-}"; fi

case "${1:---phase}:${2:-}" in
# ------------------------------------------------------------------ 硬件基线
hw:*)
  cat > "$LOGDIR/hwcheck.py" <<'PY'
# [容器内] G3 硬件基线。任一项离群就先修硬件——用离群卡跑出的训练 MFU 无法归因。
# 参考：GEMM 70-85% 峰值 / HBM 读 3.6-4.2 TB/s / 8卡 NVLink allreduce 250-400 GB/s
import os, statistics, torch, torch.distributed as dist
R = int(os.environ.get("RANK", 0)); L = int(os.environ.get("LOCAL_RANK", 0))
W = int(os.environ.get("WORLD_SIZE", 1)); torch.cuda.set_device(L)
if W > 1: dist.init_process_group("nccl")
def p(*a):
    if R == 0: print(*a, flush=True)

def timed(fn, it=20, wu=5):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def dist_vals(v):
    if W < 2: return [v]
    t = torch.tensor([v], device="cuda"); g = [torch.zeros_like(t) for _ in range(W)]
    dist.all_gather(g, t); return [x.item() for x in g]

p("=" * 74)
props = [torch.cuda.get_device_properties(i) for i in range(W)]
cap = sorted({round(x.total_memory / 1024**3, 1) for x in props})
p(f"[1] 可见 GPU {W} 张  容量 {cap} GiB  {'✓ 一致' if len(cap)==1 else '✗ 容量不齐：有卡被换过/共享/MIG'}")
p(f"    型号 {sorted({x.name for x in props})}  sm_{'/'.join(sorted({str(x.major)+str(x.minor) for x in props}))}")

n = 8192; a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
tf = 2.0 * n**3 / (timed(lambda: a @ b) / 1e3) / 1e12
v = dist_vals(tf)
p(f"[2] bf16 GEMM {[round(x) for x in v]} TFLOPS/卡  达成峰值 {statistics.mean(v)/989*100:.0f}%"
  f"  卡间离散 {(max(v)-min(v))/max(v)*100:.1f}%"
  + ("  ✗ >8% 查功耗墙/降频" if (max(v)-min(v))/max(v) > .08 else ""))
del a, b; torch.cuda.empty_cache()

m = 1_200_000_000; x = torch.randn(m, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
p(f"[3] HBM 纯读 {m*2/timed(lambda: x.sum(), 10)/1e9:.2f} TB/s   读写 {m*4/timed(lambda: y.copy_(x), 10)/1e9:.2f} TB/s（理论 4.8）")
del x, y; torch.cuda.empty_cache()

if W > 1:
    p(f"[4] NCCL allreduce {W} 卡  NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    for mb in (64, 256, 1024):
        t = torch.randn(mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
        alg = t.numel() * 2 / (timed(lambda: dist.all_reduce(t), 30, 10) / 1e3) / 1e9
        b = dist_vals(alg * 2 * (W - 1) / W)
        p(f"    {mb:>5} MB  busbw {statistics.mean(b):6.1f} GB/s  (min {min(b):.0f} / max {max(b):.0f})")
        del t; torch.cuda.empty_cache()
    p("    <100 GB/s 基本说明没走 NVLink → 回去看 nvidia-smi topo -m")
p("=" * 74)
if W > 1: dist.destroy_process_group()
PY
  $ENGINE run --rm --gpus all --network host --shm-size=64g \
    ${IB_DEV:+$IB_DEV} \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -v "$ROOT":/workspace "$IMAGE" \
    torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port=29600 \
      /workspace/logs/hwcheck.py 2>&1 | tee "$LOGDIR/hwcheck.txt"
  ;;

# ------------------------------------------------------ 单机 8B：S1–S4
s8b:*)
  echo "########## S1 micro batch 上界（TP8，num_micro_batches 固定 8 让算力持续饱和）##########"
  # GBS = mbs × dp × num_micro，这里 dp=1、num_micro=8。GBS 太小会让 kernel 吃不饱，
  # 测出来的 MFU 偏低却不是硬件的问题，是负载没给够。
  for mbs in 1 2 4 8 16; do
    run_train "8B_tp8_pp1_mbs${mbs}_s4096_gb$(( mbs * 8 ))_bf16" LLAMA_8B 8 1 "$mbs" 4096 $(( mbs * 8 ))
  done
  echo "########## S2 并行策略（GBS 固定 32 才可比）##########"
  for spec in "1 1" "2 1" "4 1" "8 1"; do
    set -- $spec
    run_train "8B_tp$1_pp1_mbs2_s4096_gb32_bf16" LLAMA_8B "$1" 1 2 4096 32
  done
  echo "########## S3 长序列（TP8 mbs1 GBS8：单机唯一能逼出显存上限的方向）##########"
  for seq in 2048 8192 16384 32768 65536 131072; do
    run_train "8B_tp8_pp1_mbs1_s${seq}_gb8_bf16" LLAMA_8B 8 1 1 "$seq" 8
  done
  echo "########## S4 FP8 vs BF16 ##########"
  if grep -qE '^ *--fp8-format' "$LOGDIR/megatron_args.txt" 2>/dev/null; then
    run_train "8B_tp8_pp1_mbs4_s4096_gb8_bf16" LLAMA_8B 8 1 4 4096 8
    run_train "8B_tp8_pp1_mbs4_s4096_gb8_fp8"  LLAMA_8B 8 1 4 4096 8 --fp8-format=e4m3
  else
    echo "  跳过：容器里这版 Megatron 没有 --fp8-format（见 logs/megatron_args.txt）"
  fi
  ;;

# ------------------------------------- 双机 8B TP8×DP2：机间带宽门禁
dx8b:*)
  echo "########## D1 机间 all-reduce 探针 ##########"
  echo "和单机 w8_8B_tp8_pp1_mbs2_s4096_gb32_bf16 比 step 时间："
  echo "  比值 > 2.0 → 机间撑不起数据并行，70B 只能用 PP 把跨机流量压到最小（见 TESTPLAN 第 3 节）"
  run_train "8B_tp8_pp1_mbs2_s4096_gb32_bf16" LLAMA_8B 8 1 2 4096 32
  ;;

# ------------------------------------- 双机 70B TP8×PP2（跨机只传激活）
d70b:*)
  # 70B 单机 8 卡放不下：TP8×PP1 → dp=1，优化器状态没有 DP 组可分片，
  # 静态 ≈ (2 bf16 + 4 fp32主 + 8 Adam + 4 fp32梯度) B × 70.55e9 ÷ 8 = 159 GB/卡 > 140.4 GiB。
  # 不拦的话会记成一条 OOM，看起来像硬件到顶，其实是并行度写错（常见于 NNODES 打成 NODES）。
  if [ "$WORLD" -lt 16 ]; then
    echo "!! d70b 要 16 卡（NNODES=2 × 每机 8 卡），现在 world=$WORLD。"
    echo "   70B 在单机 8 卡上静态显存就要 159 GB/卡，超过 140.4 GiB，必然 OOM。"
    echo "   检查一下是不是把 NNODES=2 写成了 NODES=2。"; exit 2
  fi
  echo "########## D2 流水线气泡（m=4 时 bubble≈25%）##########"
  for m in 4 12 16; do
    run_train "70B_tp8_pp2_mbs1_s2048_gb${m}_bf16" LLAMA_70B 8 2 1 2048 "$m"
  done
  echo "########## D3 序列长度与 OOM 边界 ##########"
  for seq in 4096 8192; do
    run_train "70B_tp8_pp2_mbs1_s${seq}_gb12_bf16" LLAMA_70B 8 2 1 "$seq" 12
  done
  echo "########## D4 TP8×DP2 对照：量化第 3 节那个判断 ##########"
  run_train "70B_tp8_pp1_mbs1_s2048_gb16_bf16" LLAMA_70B 8 1 1 2048 16
  ;;

soak:*)
  MINS="${SOAK_MINUTES:-120}"
  echo "########## 长稳 $MINS 分钟 ##########"
  echo "★ 另开终端在【宿主机】采遥测，否则看不出热降频："
  echo "  nvidia-smi dmon -s put -d 1 -o -f $LOGDIR/dmon.csv &"
  STEPS=100000
  run_train "soak_8B_tp8_pp1_mbs4_s4096_gb8" LLAMA_8B 8 1 4 4096 8 --exit-duration-in-mins="$MINS"
  ;;

report:*)
  python "$ROOT/05_report.py" --logs "$LOGDIR" || echo "!! 05_report.py 不在 $ROOT"
  ;;

*:|*)
  sed -n '1,30p' "$0"; exit 2 ;;
esac
