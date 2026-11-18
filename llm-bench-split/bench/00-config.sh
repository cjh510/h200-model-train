# ============================================================================
# 00-config —— 唯一的「可调面」：默认值、硬件常量、模型定义、阶段清单。
# 只做赋值与数组构造，不做探测、不启动容器；日常改参数只动这个文件。
# ============================================================================

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
# 显存高占用长稳（2026-11 改版）：档位不再自己标定，直接复用 --phase mfu 的 MFU† 峰值档，
# 所以这里只剩「跑多久 / 什么算异常」两个问题。
HIGHMEM_MINUTES="${HIGHMEM_MINUTES:-60}"      # 长稳时长；<10 分钟按冒烟处理（不判定）
HIGHMEM_HEALTH="${HIGHMEM_HEALTH:-1}"         # 1 = 跑前/跑后各采一次硬件快照
HIGHMEM_GROWTH_MAX="${HIGHMEM_GROWTH_MAX:-5}" # 任一卡显存增长 ≥ 此百分比 → WARN_LEAK（疑似泄漏）
HIGHMEM_FRAG_DIFF="${HIGHMEM_FRAG_DIFF:-3}"   # 峰值在前 20% 且末尾仍 ≥ 峰值-此值 → WARN_FRAG（疑似碎片）
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
# ---------------------------------------------------------------- 并行与批次（训练矩阵的网格）
# 这些数字以前直接写死在 phase_* 的 run_train 调用里（"8B_tp8_pp1_mbs${mbs}_s4096_..."），
# 改一个 mbs 要翻代码，也没法从命令行试。现在集中在这里：默认值 = 原硬编码值（行为不变），
# 全部环境变量可覆盖。想先看网格再决定跑不跑：PLAN_ONLY=1 bash 04_bench.sh --phase s8b
#   · s8b  = 单机 8B：S1 扫 mbs / S2 扫 TP / S3 扫 seq / S4 FP8 对照
#   · dx8b = 双机 8B 机间带宽门禁（D1）
#   · d70b = 双机 70B TP8×PP2（D2 扫 GBS、D3 扫 seq）
# 长稳（soak）、显存高占用、峰值 MFU、边界扫描各有自己的旋钮：
#   SOAK_* / HIGHMEM_* / MFU_*，不在下面这张表里。
S8B_TP="${S8B_TP:-8}"; S8B_PP="${S8B_PP:-1}"                     # S1/S3/S4 的固定拓扑
S8B_MBS_LIST="${S8B_MBS_LIST:-1 2 4}"                            # S1：扫 mbs，GBS = mbs × S8B_GBS_MULT
S8B_GBS_MULT="${S8B_GBS_MULT:-8}"
S8B_SEQ="${S8B_SEQ:-4096}"
S8B_TOPO_LIST="${S8B_TOPO_LIST:-1:1 4:1 8:1}"                    # S2：tp:pp 成对；GBS 固定才可比
S8B_TOPO_MBS="${S8B_TOPO_MBS:-2}"
S8B_TOPO_SEQ="${S8B_TOPO_SEQ:-4096}"
S8B_TOPO_GBS="${S8B_TOPO_GBS:-32}"
S8B_SEQ_LIST="${S8B_SEQ_LIST:-2048 8192 32768 131072}"           # S3：长序列（显存上限方向）
S8B_SEQ_MBS="${S8B_SEQ_MBS:-1}"
S8B_SEQ_GBS="${S8B_SEQ_GBS:-8}"
S8B_FP8_MBS="${S8B_FP8_MBS:-4}"                                  # S4：FP8 vs BF16 同参数对照
S8B_FP8_SEQ="${S8B_FP8_SEQ:-4096}"
S8B_FP8_GBS="${S8B_FP8_GBS:-8}"
DX8B_TP="${DX8B_TP:-8}"; DX8B_PP="${DX8B_PP:-1}"                 # D1：双机 8B TP8×DP2
DX8B_MBS="${DX8B_MBS:-2}"; DX8B_SEQ="${DX8B_SEQ:-4096}"; DX8B_GBS="${DX8B_GBS:-32}"
D70B_TP="${D70B_TP:-8}"; D70B_PP="${D70B_PP:-2}"                 # D2/D3：双机 70B TP8×PP2
D70B_MBS="${D70B_MBS:-1}"; D70B_SEQ="${D70B_SEQ:-2048}"
D70B_BUBBLE_GBS_LIST="${D70B_BUBBLE_GBS_LIST:-4 16}"             # D2：扫 GBS 看 PP 气泡
D70B_SEQ_LIST="${D70B_SEQ_LIST:-4096 8192}"                      # D3：扫 seq，GBS 固定 12
D70B_SEQ_GBS="${D70B_SEQ_GBS:-12}"

# 硬件口径常量：全部取 H200 SXM 实测值，05_report.py 里有一份同值的 PEAK / VRAM_GIB。
PEAK_TFLOPS=989.4                            # bf16 稠密峰值 TFLOPS（不含稀疏；稀疏是 2×）
VRAM_MIB=143771                              # nvidia-smi 实测每卡显存
VRAM_GIB=140.4                               # = 143771/1024，报告一律用实测，不写 141

WORLD=$(( NPROC * NNODES ))

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
# MLA 配置（配置 B）不能带 GQA：num-query-groups=8 会让 TE 断言
# 「Keys and values must have num_gqa_group = 8 heads!」——MLA 的 KV 是压缩潜变量
#（1 组），与 8 组 GQA 互斥。11-08 单机 moe 实测 8 rank 全崩，配置 B 剥离 GQA 两项。
#（dense 的 LLAMA_8B/70B 仍用 COMMON 的 GQA=8，Llama-3.1 结构如此。）
MOE_COMMON=()
for _a in "${COMMON[@]}"; do
  case "$_a" in --group-query-attention|--num-query-groups=*) continue ;; esac
  MOE_COMMON+=( "$_a" )
done
# 配置 B（moe 用）：24 层 / hidden 4096 / 32 heads / MLA / 32 专家 + 1 共享
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

# 阶段清单：parse_phase 与入口分发表都查这一份；新增阶段改这里 + 入口 case
ALL_PHASES="argscheck hw coll nccl s8b dx8b d70b moe moe-dp io fault soak mfu report"
