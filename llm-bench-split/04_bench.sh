#!/usr/bin/env bash
# =============================================================================
# [在哪跑] 目标裸机（脚本内部自己起容器）。压测矩阵总入口。
# 容器引擎：docker 优先，没有可用 docker 时回落 nerdctl（探测与封装见 lib.sh）。
#
#   bash 04_bench.sh --phase argscheck # 参数自检：容器内 pretrain_gpt.py --help → args-check.json
#   bash 04_bench.sh --phase hw      # G3 单机硬件基线（含 FP8/L2/PCIe/collective，稳态 ≥5s）
#   bash 04_bench.sh --phase coll    # 单机 collective 扫描（已并入 hw；仅 OPTIONAL=1 时单跑）
#   bash 04_bench.sh --phase s8b     # 单机 8 卡 Llama-3.1-8B：S1 S2 S3 S4
#   bash 04_bench.sh --phase dx8b    # 双机 8B TP8×DP2 训练对照（GBS=32）
#   bash 04_bench.sh --phase nccl    # 双机 collective 全类型 × 8B–8G（不跑训练）
#   bash 04_bench.sh --phase d70b    # 双机 16 卡 Llama-3.1-70B：TP8×PP2
#   bash 04_bench.sh --phase moe     # MoE/MLA：单机 EP8×TP1 / EP2×TP4 + 双机 EP16；dispatcher 两种
#   bash 04_bench.sh --phase moe-dp  # 仅双机：EP8×DP2（alltoall；梯度跨机 all-reduce），与 EP16 对照
#   bash 04_bench.sh --phase soak    # 长稳（SOAK_MODEL=dense / dense70 / moe / highmem，可逗号组合；
#                                    #   highmem 前置 = 本 run 已有 mfu.jsonl，档位取 MFU† 峰值档）
#                                    #   both=dense+moe；all 在双机含 dense70，单机保持三条；默认 dense）
#   bash 04_bench.sh --phase io      # checkpoint 写读 + 数据管线（无真数据则 SKIP）
#   bash 04_bench.sh --phase fault   # 故障注入：OOM / kill rank / NCCL 超时
#   bash 04_bench.sh --phase mfu     # 峰值 MFU：粗扫并行度×GBS → 细扫 mbs×accum → seq → 前3名复测取中位
#                                    #   头条 MFU† = Megatron throughput / 989.5；含时钟/降频取证
#   bash 04_bench.sh --phase report  # 汇总成表
#
# 双机：两台都执行同一条命令，只有 NODE_RANK 不同，且 NCCL_IF / NCCL_HCA 必须显式给：
#   node-a: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-a-ip> NCCL_IF=bond0 \
#           NCCL_HCA=mlx5_bond_2,mlx5_bond_3,... bash 04_bench.sh --phase d70b
#   node-b: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node-a-ip> ...（同上）
# 缺 NCCL_IF / NCCL_HCA / MASTER_ADDR 是回环地址时，脚本直接 exit 2 —— 不再静默退回 socket。
#
# 可调：IMAGE / ROOT / NPROC / STEPS / FORCE=1 / OPTIONAL=1 / TELEMETRY=1
#       SOAK_MINUTES / SOAK_MIN_ITERS / SOAK_MODEL / MOE_SOAK_MINUTES / MOE_SOAK_MIN_ITERS
#       DENSE70_SOAK_MINUTES / DENSE70_SOAK_MIN_ITERS（dense70 仅双机）
#       MFU_STEPS / MFU_REPEAT / MFU_CV_MAX / MFU_GBS_LIST / MFU_TP_LIST / MFU_MBS_LIST
#       MFU_ACCUM_LIST / MFU_SEQ_LIST / MFU_MAX_GBS / MFU_MAX_RUNS / MFU_PLAN_ONLY / MFU_MODELS
#       HIGHMEM_MINUTES / HIGHMEM_HEALTH / HIGHMEM_GROWTH_MAX / HIGHMEM_FRAG_DIFF
#       COLL_SIZES / MOE_GBS
#       —— 训练矩阵的并行与批次（默认值 = 各阶段原来的硬编码值，见 bench/00-config.sh）——
#       S8B_TP / S8B_PP / S8B_MBS_LIST / S8B_GBS_MULT / S8B_SEQ
#       S8B_TOPO_LIST(tp:pp 列表) / S8B_TOPO_MBS / S8B_TOPO_SEQ / S8B_TOPO_GBS
#       S8B_SEQ_LIST / S8B_SEQ_MBS / S8B_SEQ_GBS
#       S8B_FP8_MBS / S8B_FP8_SEQ / S8B_FP8_GBS
#       DX8B_TP / DX8B_PP / DX8B_MBS / DX8B_SEQ / DX8B_GBS
#       D70B_TP / D70B_PP / D70B_MBS / D70B_SEQ / D70B_BUBBLE_GBS_LIST
#       D70B_SEQ_LIST / D70B_SEQ_GBS
#       PLAN_ONLY=1 → 只打印每档 tag/TP/PP/DP/mbs/seq/GBS/accum，不启动容器
# OPTIONAL=1 才跑精简矩阵里降级的档位（单机 coll、io 的 70B 档）。
#
# 带值 flag 一律 --name=value：prune_optional 才能「一项一裁」。
# =============================================================================

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
lc_init || exit 2

# ---- 模块（顺序即依赖顺序；只做函数定义与赋值，无副作用）----
for _m in 00-config 10-boot 20-net 30-record 40-run 50-phase-train 60-phase-soak 62-highmem 70-phase-mfu; do
  . "$HERE/bench/$_m.sh" || { echo "!! bench/$_m.sh 缺失：同步不完整（需要整个 bench/ 目录）"; exit 2; }
done

# ---- 启动流程（函数本体在 bench/10-boot.sh）----
parse_phase "$@"
export ROOT NPROC NNODES WORLD
echo "world=$WORLD (nproc=$NPROC × nodes=$NNODES) rank=$NODE_RANK master=$MASTER_ADDR steps=$STEPS"
boot_env
trap cleanup EXIT INT TERM
rdma_gate_precheck

# ---- 分发表：一个阶段一行，正文在 bench/ 对应模块 ----
case "$PHASE" in
  argscheck)  phase_argscheck ;;
  hw)         phase_hw ;;
  coll)       phase_coll ;;
  nccl)       phase_nccl ;;
  s8b)        phase_s8b ;;
  dx8b)       phase_dx8b ;;
  d70b)       phase_d70b ;;
  moe)        phase_moe ;;
  moe-dp)     phase_moe_dp ;;
  io)         phase_io ;;
  fault)      phase_fault ;;
  soak)       phase_soak ;;
  mfu)        phase_mfu ;;
  report)     phase_report ;;
esac

# 每个阶段结束写一条到 index.md（阶段、耗时、结果、产出）
lc_run_stage_done "$PHASE" "$START_TS"
echo "run 目录：$LC_RUN_DIR"
