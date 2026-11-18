# TESTPLAN 修订说明

> 原文件 `llm-bench-docker/TESTPLAN.md` **未改动**。本文件只列需要替换的表述，
> 以及替换后报告里应该怎么写。行号均指原 TESTPLAN.md。

## 1. 显存口径：141 GiB → 140.4 GiB（143771 MiB）

| 位置 | 原文 | 改为 |
|---|---|---|
| `:18` | `141.1 GiB/卡 ❌`、`70.6 GiB/卡 ✅` | 单卡可用容量按 **140.4 GiB（143771 MiB）** 写。70B 每卡静态 141.1 GiB，仍然 ❌，结论不变 |
| `:20` | 「一张 H200 是 141 GiB」 | 「一张 H200 实测 **143771 MiB = 140.4 GiB**（nvidia-smi 口径）」 |
| `:85` | 「逐卡显存容量必须都是 141 GiB」 | 「逐卡 `memory.total` 必须都是 **143771 MiB**；有一张不同就是换过卡/MIG/共享实例」 |

理由：`nvidia-smi --query-gpu=memory.total` 返回 143771 MiB，143771 MiB ÷ 1024 = 140.4 GiB。
原文把 143771 MiB 当成 143771 GiB 再换算，才得出 141 GiB。差 0.6 GiB 会让 70B 的显存
边界判断失真，也和 `05_report.py` 的容量声明对不上。

脚本侧已落实：`02_precheck.sh` 的「显存容量一致」项、`05_report.py` 的 `VRAM_GIB`。

## 2. 镜像/CUDA 矩阵：首发固定 25.01-py3（12.8.0）

- `:148-167` 的矩阵保留，但执行顺序改成：**先 25.01-py3（CUDA 12.8.0）**。
- 25.06（12.9.1）只能作为「minor 兼容」备选：容器里的 cubin 能在 R570 上跑，
  但如果镜像内包含比驱动更新的 **PTX**，JIT 会报 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`。
  这不是「跑不起来」，而是「某个 kernel 首次触达时才失败」，长稳里最容易踩。
- 25.10（13.0.2）跨 major，需要 `cuda-compat`，本机没有验证过，不进首轮矩阵。
- 两台机器的镜像 **digest 必须一致**（`03_deploy.sh` 写进 `logs/env.txt`）。

## 3. 通信占比公式：去掉多余的 `world`

原文 `:208`：

```
通信/计算 ≈ 4 × world × 989e12 ÷ (6 × 带宽bytes/s × GBS × seq)
```

而 `:213` 的表头用的是 `GBS×seq/world`（每卡每步 token）。两套口径混写。

改为单一口径：

```
每卡token = GBS × seq ÷ world
通信/计算 ≈ 4 × 989e12 ÷ (6 × 带宽bytes/s × 每卡token)
```

数值与 `:213-217` 的表一致（4096 → 54%，16384 → 13.5%，65536 → 3.4%，按 300 GB/s）。
写成前一种形式时，读者容易把 `GBS×seq` 当全局 token 再乘一次 `world`，把通信占比
放大 8 倍（单机）或 16 倍（双机），从而误判「通信不是瓶颈」。

补充：这台机器是 RoCE（`mlx5_bond_2..9`，400 Gb/s，MTU 4200），不是 NVLink 那 300 GB/s
的 intra-node 场景。跨机档位的带宽上限按实际 `busbw` 填，不要套 300。

## 4. 版本号：core_r0.12 → core_v0.13.1

- `:71`「TE 1.14 ↔ core_r0.12 这一代」→ 按实际物料写 **`core_v0.13.1`**。
- `:143` `git clone --branch core_r0.12` → `core_v0.13.1`。

理由：目录里的物料是 `Megatron-LM-core_v0.13.1.zip`，且已验证
`--mock-data`、`--exit-duration-in-mins` 存在，而 `--num-micro-batches`、
`--distributed-optimizer-checkpoint-shape-checks` 两个名字 **不存在**。
照原文 clone 会拿到别的版本，容器内 import 时会以 undefined symbol 失败。

## 5. VPP：文档说扫了，矩阵里没有

- `:134`「或者开交错式调度 `--num-layers-per-virtual-pipeline-stage`(VPP)……矩阵里两者都扫」
  → 改为「VPP 本轮 **未实现**，矩阵里只有 PP2/PP1 对照；VPP 作为后续可选项」。
- `:275` 的「加大 `--num-micro-batches` 或开 VPP」同理：`num-micro-batches` 不是命令行参数，
  它是 `GBS ÷ (mbs × dp)` 的派生量；实际做法是加大 `--global-batch-size`。

理由：`04_bench.sh` 的 d70b 只有 `PP=2` 固定值，没有 VPP 参数。文档说扫过、报告里没有，
评审时会被直接指出自相矛盾。

## 6. 硬件验收：补 ECC / retired_pages / NVLink 逐卡误码 / Xid

原计划 `02_precheck.sh` 只有型号、显存、功耗、拓扑。新增四项（已进 fix 的 `02_precheck.sh`）：

| 项 | 判据 | 不合格含义 |
|---|---|---|
| ECC 模式 + 计数 | mode 必须 Enabled；`ecc.errors.uncorrected.aggregate.total` 必须 0 | 有不可纠正错误 → 这块卡不能进基准 |
| retired_pages | pending/correctable/uncorrectable 全 0（机型不支持该查询则跳过） | 有页面被退役 → 卡已降级 |
| NVLink 链路 + 误码 | 链路数 = 8×18；CRC/Replay/Recovery 合计 0 | 掉链路 → TP8 退化成 PCIe；误码 → 训练中随机 hang |
| Xid | `dmesg -T | grep NVRM.*Xid` 为 0（NVSwitch SXid 单独记录，低优先级） | 有 Xid → 驱动/硬件已报过 GPU 级错误 |

注意：采集时 `nvlink.txt` 被 200 行截断、`retire.txt` 是 N/A，这两项本轮 **没有真实结论**，
必须在目标机上重跑 `02_precheck.sh` 才算数。

## 7. 长稳与遥测的写法

- 「长稳另开终端采 dmon」→ 「脚本内后台启动，退出（含 Ctrl-C/OOM）时 trap 清理」。
- soak 的 tag 必须是 `w<world>_8B_soak_...` 形式，`05_report.py` 才能解析。
- 判定门槛：`iters ≥ SOAK_MIN_ITERS`（默认 100 步）且无 OOM，才写 `.ok_` 哨兵；
  否则记 `PARTIAL`，下次重跑不会跳过。
- 报告里长稳单列：步数 × step 时间 = 等效时长、MFU†、抖动。
  「抖动随时间放大 / 温度贴顶 / 功耗爬升」三项任一成立，就不能写「长稳通过」。

## 8. 结果指纹

`results.jsonl` 每条结果新增：`image`、`digest`、`driver`、`torch`、`cuda`、`nccl`、`nodes`、`port`、`iters`。
两台机器的 `logs/env.txt` 去掉 `host=`/`time=` 后必须 `diff` 为空，否则双机比值（D1）不可用。
`05_report.py` 会打印出现过的所有指纹组合；多于一组就直接提示「跨机/跨镜像混跑」。

## 9. 不变的部分

- 70B 单机不可行的结论不变（每卡静态 141.1 GiB > 140.4 GiB）。
- `t_min` 公式、MFU 阈值（<20% 停手、35-45% 正常水位）不变。
- 「mock 数据 + 随机初始化，不含数据管线/checkpoint 开销」的声明不变。

## 10. 本轮补全后的判定与阶段

- D1 判据修正：同 GBS=32 扩到 16 卡，理想 step 时间比是 **0.5**；**≤0.6 视为跨机链路可用**，
  0.6–1.0 为「有收益但链路是瓶颈」，>1.0 为加机器不涨吞吐。另跑 GBS=64 一组做每卡 token 同口径对照。
- 单机新增 `--phase coll`（collective 全类型 × 8B–8G），双机新增 `--phase nccl`（直测 busbw/延迟 + NET/IB 判定）。
- 长稳改为 `--phase soak`，默认 120 分钟，支持 `NNODES=1/2`；跑前跑后 ECC/retired/NVLink/Xid 增量必须为 0。
- 新增 `--phase io`（checkpoint 写读代理 + 数据管线）与 `--phase fault`（OOM / kill rank / NCCL 超时）。
- 硬件微基准改为稳态 ≥5 s（`PROBE_SECS`），并补 FP8、L2、PCIe、8B–8G 消息扫描。
- 结论边界：单机 + 双机全跑完，只能声称「训练可用、跨机链路满足 PP 扩展」；
  宣称「整机达标」仍需 `hw`（8 卡全 collective）+ `nccl` + 2 h `soak` 硬件增量 + `io` + `fault` 全部通过。
