# H200 单机/双机 Llama 训练压测方案(Docker,目标机无外网)

## 0. 目标与产出

在同一套硬件上,用 Docker 里的 Megatron-LM + TransformerEngine 跑 **Llama-3.1-8B(单机 8 卡)** 和 **Llama-3.1-70B(双机 16 卡)** 全参数训练,产出可对外交付的验收报告。

不加载权重、不用真实语料:压测只依赖形状、计算图、通信模式。用 Megatron 的 `--mock-data` + `NullTokenizer`,所以**不需要下载 15GB/130GB 权重,也不需要预处理数据集**。

---

## 1. 显存硬账(决定能测什么)

按 config 逐项算的参数量(`weights_gib` 同款公式,对 Qwen3-30B-A3B 实测误差 0.03%):

| 模型 | 参数 | bf16 权重 | 全参静态 `16B/参数` | 单机 8 卡 | 双机 16 卡 |
|---|---|---|---|---|---|
| Llama-3.1-8B | 8.03B | 15.0 GiB | 128.5 GB | **16.1 GiB/卡 ✅** | 8.0 GiB/卡 ✅ |
| Llama-3.1-70B | 70.55B | 131.4 GiB | 1128.8 GB | **141.1 GiB/卡 ❌** | **70.6 GiB/卡 ✅** |

一张 H200 是 141 GiB。70B 单机每卡静态就吃掉 141.1 GiB —— **零余量,连 CUDA context 和 NCCL 缓冲都放不下,任何 `mbs`/`seq` 都会 OOM**。这不是配置问题,所以方案里 70B 只出现在双机阶段。

> FULL_SHARD / TP 分片下有个好记的结论:**每卡静态显存 = 一份完整 bf16 权重的量**(16÷8 = 2 字节/参数)。看权重列就能判断可行性。

---

## 2. 阶段与门禁

每个阶段有明确的"通过标准"。**门禁不过就不要进下一阶段** —— 否则最后拿到一堆无法归因的数字。

### G0 · 两台机器分别核验(裸机上跑,不进容器)

```bash
bash 02_precheck.sh
```

必须采到的六项,以及为什么:

| 检查 | 为什么是门禁 |
|---|---|
| `nvidia-smi` 的 **Driver + CUDA Version** | 决定镜像能选哪个 CUDA。**两台不一致时按较低那台选**,而且双机 NCCL 要求两端容器完全相同 |
| `nvidia-smi topo -m` | 确认 8 卡是 NVLink 全互联(NV18);有卡走 PCIe 会让 TP8 的数字失去意义 |
| `/sys/class/infiniband/*` | 决定双机走 RDMA 还是 TCP。没有 IB,双机 70B 的结论就得改写 |
| 网卡 `operstate` / `speed` / `mtu` | 双机通信上限;非 9000 MTU 会影响 TCP 路径 |
| `docker run --gpus all` 是否成功 | **无外网裸机上最常见的失败点**:缺 nvidia-container-toolkit。装它本身需要 repo,可能得离线 rpm |
| `power.limit` / `persistence_mode` / `Clocks Event Reasons` | 功耗墙被前一个用户改小、持久模式关闭,都会伪装成"软件栈慢" |

`02_precheck.sh` 对每一项给 PASS/WARN/FAIL,并把结果写成 `$(hostname)-env.json`,两台的两份留给 G5 对齐。

### G1 · 物料落地(准备机有网 → 传输 → 裸机无网)

见第 4 节清单,逐件标明在哪取。

### G2 · 容器与框架自检

```bash
bash 03_deploy.sh          # 裸机
```

容器内必须验证四件事,任何一件不过就不要开始压测:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformer_engine as te;print(te.__version__)"
python -c "import megatron.core as m;print(m.__version__)"
# Llama 并行必需：TE fused kernel 与 grouped GEMM 是否真能加载（不是 import 成功就算）
python -c "from transformer_engine.pytorch.attention import DotProductAttention; print('TE ok')"
# 关键：把容器内 Megatron 实际支持的参数名落地,后面所有命令按它核对
cd /workspace/Megatron-LM && python pretrain_gpt.py --help > /workspace/logs/megatron_args.txt
```

**Megatron-LM 从 NGC 25.01 起不再内置于容器**,所以必须外挂源码。版本要与容器内 TE 配对(TE 1.14 ↔ core_r0.12 这一代),拿 `main` 分支去配旧 TE 会在 import 时报符号缺失。

`megatron_args.txt` 是防坑的关键:参数名跨版本改得很频繁(`--rot-base` vs `--rotary-base`、`--fp8-format` 有无 `--fp8-recipe`)。压测脚本启动时会自动按这份表裁剪不认识的可选 flag,主参数名仍需人工对一次。

### G3 · 硬件与通信基线(容器内)

```bash
bash 04_bench.sh --phase hw          # 单机 8 卡
```

先确认算力/带宽达标,再谈 MFU:
- 单卡 bf16 GEMM:989 TFLOPS 峰值的 **70–85%**(即 690–840),卡间离散 <8%
- HBM 带宽:理论 4.8 TB/s,实测读 **3.6–4.2**
- NVLink allreduce busbw:8 卡全互联大消息 **250–400 GB/s**
- 逐卡显存容量必须都是 141 GiB —— 有一张偏小就是换过卡/MIG/共享实例

**任一项离群就停。** 用离群卡跑出来的训练 MFU 无法归因,报告也不该交付。

### G4 · 训练压测矩阵

```bash
bash 04_bench.sh --phase s8b          # 单机 8B：并行策略 + batch 上界 + 长序列
bash 04_bench.sh --phase d70b         # 双机 70B：先通信门禁，再训练
```

### G5 · 两机对齐 + 长稳

```bash
python 05_report.py --compare node-a-env.json node-b-env.json
bash 04_bench.sh --phase soak
```

驱动版本、容器镜像 ID、NCCL 版本、GPU 数量任何一项不一致,合并数字就没有可比性 —— 先修环境再出报告。

### G6 · 出报告

```bash
bash 04_bench.sh --phase report
```

---

## 3. 双机的核心设计决策:70B 用 `TP8 × PP2`,不是 `TP8 × DP2`

这一条决定双机能不能成立,值得单独讲。

`TP8 × DP2`(16 卡 = 8 TP × 2 DP)每步要做**跨机梯度 all-reduce / reduce-scatter**,量级是每卡分片梯度 ~17 GB bf16。假设机间只有 25GbE(≈3 GB/s),这一步就要 **5.7 秒**,而 70B 每步计算下界只有 3.5 秒 —— 通信比超过 160%,训练时间直接翻倍以上。

`TP8 × PP2`(16 卡 = TP8 机内 × PP2 跨机)把流水线切在**两台机器之间**:跨机只传 micro-batch 的激活值,单张量是 `mbs × seq × hidden × 2字节` = 1×2048×8192×2 = 34 MB,前向+反向各一轮,总量在 GB 级而非 34 GB 级。**跨机流量降低约 1000 倍。**

所以:

- **70B 双机固定 `TP=8, PP=2, DP=1`**,让跨机边只剩 P2P 激活传递。
- **8B 双机反过来用 `TP=8, DP=2`**,把它专门当作**机间带宽的探针** —— 8B 通信量小、算得快,能敏感地暴露跨机 all-reduce 的上限。这一项的目的不是测算力,是量化你的网络。

PP=2 有个必须处理的代价:**流水线气泡** ≈ `(p-1)/m`,`p`=stage 数、`m`=micro-batch 数。所以 70B 那组一定要给足 micro-batch:

```
m=4  → bubble 25%     ← 不可接受
m=12 → bubble 8%      ← 推荐下限
m=16 → bubble 6%
```

或者开交错式调度 `--num-layers-per-virtual-pipeline-stage`(VPP),把 bubble 压到 `1/(2·v·m)` 量级。矩阵里两者都扫,这样报告里能说清"这台机器跑 70B 时 PP 调度的损失有多大"。

---

## 4. 无外网物料清单:每件都标明在哪取

| # | 产物 | 大小 | 在哪取 | 落到哪 |
|---|---|---|---|---|
| 1 | NGC PyTorch 镜像 tar | ~9 GB | **有网的准备机** `docker pull` + `docker save`。tag 按 G0 采到的较低 CUDA 上限选,见下 | 裸机 `docker load` |
| 2 | Megatron-LM 源码 tar | ~60 MB | 准备机 `git clone --branch core_r0.12`(与容器 TE 配对)。**不要 clone main** | 裸机 `/root/llm-bench/` |
| 3 | 本方案 5 个脚本 | <100 KB | 你这台 Windows(已生成) | 裸机 `/root/llm-bench/` |
| 4 | 权重 | **不需要** | — | — |
| 5 | 数据集 | **不需要** | `--mock-data` + `NullTokenizer` | — |

### 镜像 tag 怎么选(按 G0 结果,不要照抄)

驱动 570 → `nvidia-smi` 报 CUDA **12.8**;驱动 590 → CUDA 13.1。规则:

- **同一 CUDA major 内的 minor 兼容**:R570 原生 12.8,可以跑 CUDA 12.9 的容器,不需要额外动作。
- **跨 major(12.x → 13.x)** 要靠容器里的 `cuda-compat`,NGC 自带、nvidia-container-toolkit 会自动挂载 —— 能用,但报错形态难查,无外网机器上我不建议赌它。

| 候选 | CUDA | 对 R570 | 建议 |
|---|---|---|---|
| `nvcr.io/nvidia/pytorch:25.01-py3` | 12.8.0 | ✅ 精确匹配 | **最保守**,TE 1.14 |
| `nvcr.io/nvidia/pytorch:25.06-py3` | 12.9.1 | ✅ minor 兼容 | **推荐**,TE 2.4(MoE/FP8 路径更成熟) |
| `nvcr.io/nvidia/pytorch:25.10-py3` | 13.0.2 | ⚠️ 跨 major,依赖 compat | 除非 G2 实测通过,否则不选 |

**两台驱动不一致时,按较低那台选同一个 tag** —— 双机必须用同一镜像 ID,否则 NCCL 版本不一致会以极难懂的方式失败。

拉取前先验 tag 存在(不下载 9GB):
```bash
docker manifest inspect nvcr.io/nvidia/pytorch:25.06-py3 >/dev/null && echo EXISTS
```
NGC 拉取要登录:`docker login nvcr.io`,Username 填 `$oauthtoken`,Password 填 NGC 个人页生成的 API key。

### 备选:阿里云内网镜像

系统是 Alibaba Cloud Linux 3,若机器在阿里云 VPC 内,`training-nv-pytorch` 有 **VPC 内网 registry**(`acs-registry-vpc.{region}.cr.aliyuncs.com/egslingjun/...`),不需要外网,而且自带 TE + flash-attn + grouped_gemm + megatron-core。

但 25.02 版对齐 NGC 25.01、`megatron-core 0.9.0` / `transformers 4.48.3` 都太旧。**Megatron 路线不受 transformers 版本影响**(它用自己的 Llama 实现),但 0.9.0 的 core 缺少较新的 `--use-precision-aware-optimizer` 等特性。用之前必须跑 `03_deploy.sh` 里的版本自检。

**上一轮我在你 Windows 机器上测的国内源可达性不能外推到裸机** —— 网络域不同。必须在裸机上重新 curl 一次。

---

## 5. 判定标准

### 先算下界,再谈 MFU

一个 step 的总功 = `6 × N × (GBS × seq)`,摊到 world 张卡：

```
t_min = 6 × N × GBS × seq ÷ (world × 989 TFLOPS)
MFU   = t_min ÷ t_actual
```

注意是 **GBS 不是 mbs** —— 用 mbs 只算了单个 micro-batch，会漏掉 `num_micro_batches` 倍，判定表整体偏低一个量级。TP 不减少每卡的计算份额（它切的是每层权重，不是数据），所以公式里只有 world 起作用。

| 配置 | GBS×seq/world（每卡每步 token） | t_min | 合理落点 |
|---|---|---|---|
| 8B `tp8 mbs2 gb16 s4096` | 8192 | 0.399 s | 35–45% → 0.9–1.1 s |
| 8B `tp8 mbs2 gb32 s4096` | 16384 | 0.798 s | 40–49% → 1.6–2.0 s |
| 8B `tp8 mbs8 gb64 s4096` | 32768 | 1.596 s | 40–50% → 3.2–4.0 s |
| 8B `tp8 mbs1 gb8 s65536` | 65536 | 3.19 s | 40% → 8.0 s |
| 70B `tp8pp2 gb12 s2048` | 1536 | 0.657 s | 30–35% → 1.9–2.2 s |
| 70B `tp8pp2 gb16 s2048` | 2048 | 0.876 s | 30–38% → 2.3–2.9 s |

70B 那两行还要再乘 `(1 + bubble)`，`bubble ≈ (pp-1)/num_micro`；`gb12` 时约 +8%。

**硬阈值：低于 20% MFU 就不要调参了**，回去查 G3 的通信基线和 `power.limit`。Megatron+TE 的 dense 模型在 H200 上做到 35–45% 是正常水位；拿不到就是环境有问题。

### 通信占比

```
通信/计算 ≈ 4 × world × 989e12 ÷ (6 × 带宽bytes/s × GBS × seq)
```

它**与模型参数量无关**（N 被约掉），只与每卡每步的 token 数成反比：

| GBS×seq/world | 通信/计算（NVLink 按 300 GB/s） |
|---|---|
| 4096 | 54% |
| 16384 | 13.5% |
| 65536 | 3.4% |

结论：**要让通信不成为瓶颈就加大 `GBS × seq`，换大模型没用。** 这也是矩阵里把 `num_micro_batches` 固定成 8、并把 S3 的 GBS 设到 8 的原因 —— GBS 给小了，测出来的低 MFU 是负载不足，不是硬件。

### 双机通信门禁(G4 的前置)

跑 70B 训练之前,先量机间带宽:

```bash
bash 04_bench.sh --phase dx8b     # 8B 双机 TP8×DP2,专门暴露跨机 all-reduce
```

判据:双机 step 时间 / 单机 step 时间 > **2.0** → 机间网络撑不住 DP 扩展。此时 70B 只能靠 PP 把跨机流量压到最小(第 3 节),报告里要写明"这台环境不支持跨机数据并行"。

---

## 6. 压测矩阵

| 阶段 | 模型 | 拓扑 | 并行 | 扫什么 | 目的 |
|---|---|---|---|---|---|
| S1 | 8B | 单机8 | TP8 + SP | `mbs` 1/2/4/8/16 | 算力饱和点 + 显存余量验证 |
| S2 | 8B | 单机8 | TP4/DP2,TP2/DP4,TP1/DP8 | — | 并行策略选择(TP 越多通信越重) |
| S3 | 8B | 单机8 | TP8 + SP | `seq` 4K/8K/16K/32K/128K | 长上下文训练能力 + 逼显存上限 |
| S4 | 8B | 单机8 | TP8 | bf16 vs FP8 | H200 上 FP8 的真实收益 |
| D1 | 8B | 双机16 | TP8 × DP2 | — | **机间带宽探针**(门禁) |
| D2 | 70B | 双机16 | TP8 × PP2 | `m` = 4/12/16 | 流水线气泡损失 |
| D3 | 70B | 双机16 | TP8 × PP2 | `seq` 2K/4K/8K | 70B 长序列可行性 + OOM 边界 |
| L1 | 8B / 70B | 各自最优 | — | 连续 2 小时 + dmon 遥测 | 热降频、掉卡、稳定性 |

单机四阶段(S1-S4)约 40-70 分钟;70B 每步 3-7 秒,`--train-iters 12` 就够,加建模和编译,双机全套 1-2 小时。

---

## 7. 报告结构

1. **执行摘要**:三句话 —— 硬件是否达标、8B 能到什么 MFU、双机 70B 是否成立
2. **环境与物料**:两台 driver/CUDA、镜像 ID(同一 tag 也要记 digest)、Megatron-LM commit、NCCL 版本
3. **硬件基线**(G3):逐卡容量、GEMM、HBM、NVLink 带宽 + 离群卡结论
4. **拓扑实测**:`nvidia-smi topo -m` 原文 + 机间 RDMA 有无
5. **单机结果**:S1–S4 表 + 每张表一句结论
6. **双机结果**:D1 门禁结论 + D2/D3 表 + 跨机流量分析
7. **长稳**:2 小时 step 时间分布、功耗/温度/时钟、有无 XID
8. **限制声明**:mock 数据 + 随机初始化,所以不含数据管线开销和 checkpoint 落盘开销;**这两项都会让真实训练更慢**,报告必须写清楚测的是算力/通信上限而非端到端吞吐
9. **结论与建议**:能不能收这台机器;70B 若要 DP 扩展需要什么网络

`05_report.py` 生成 3-6 节的表格和判定,前 2 节和 7-9 节人工补。

---

## 8. 已知风险与回退

| 风险 | 症状 | 回退 |
|---|---|---|
| 裸机缺 nvidia-container-toolkit | `--gpus all` 报 could not select device driver | 离线 rpm:`nvidia-ctk runtime configure --runtime=docker && systemctl restart docker` |
| NGC 拉取失败 | TLS 超时 / 401 | 换一台有 nvcr 出口的机器只做 `docker save`,tar 走内网传 |
| Megatron 版本与 TE 不匹配 | import 时报 undefined symbol | 按容器 TE 版本选对应 `core_rX.Y` tag;两边都记进报告 |
| `--help` 里参数名不同 | 每个 case 都挂在 argparse | `04_bench.sh` 启动时 dump `megatron_args.txt` 并自动裁剪可选 flag;主参数人工对一次 |
| 双机无 IB | D1 比值 > 2 | 70B 只测 PP;报告写明"不支持跨机 DP",不再浪费时间 |
| PP=2 气泡吃掉收益 | D2 中 `m=4` 明显差 | 加大 `--num-micro-batches` 或开 VPP |
| 功耗墙被改小 | GEMM 只有 500–600 TFLOPS | `nvidia-smi -pl <标称>` 需要 root;或确认是机房约束并写进报告 |
| 两台驱动不同 | 双机 NCCL 诡异失败 | 按低的那台选镜像;必要时申请统一驱动 |
