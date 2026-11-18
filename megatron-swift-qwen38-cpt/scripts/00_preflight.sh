#!/usr/bin/env bash
# =============================================================================
# T0 预检：镜像能力、环境、版本、GPU、权重完整性、CLI 参数能力
# 【必须在容器内执行】bash scripts/12_run_in_container.sh scripts/00_preflight.sh 27b
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
install_err_trap
PROFILE="${1:-$MODEL_PROFILE}"
load_profile "$PROFILE"

log "=== T0 预检 | profile=$MODEL_PROFILE | model=$MODEL_ID ==="
log "权重路径: $MODEL_PATH"
log "数据集  : $DATASET_ID"
log "缓存    : $MODELSCOPE_CACHE （多机必须为共享存储）"

require_cmd python
require_cmd megatron
require_cmd swift
require_cmd modelscope

# --- GPU 数量 ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  log "可见 GPU: $GPU_COUNT 张 / NPROC_PER_NODE=$NPROC_PER_NODE"
  [[ "$GPU_COUNT" -ge "$NPROC_PER_NODE" ]] || die "GPU 数不足：$GPU_COUNT < $NPROC_PER_NODE"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -"$NPROC_PER_NODE" || true

  # --- 驱动 / CUDA 匹配性 ----------------------------------------------------
  # nvidia-smi 能看到卡 ≠ torch 能用。容器内的用户态驱动 libcuda.so 由
  # nvidia-container-toolkit 从宿主注入（版本恒等于宿主驱动），而镜像里的 torch
  # 是按 cu130 编译的。宿主停在 r570（12.8）分支时会失败——此时靠镜像自带的
  # cuda-compat-13-0 做前向兼容（lib.sh:ensure_cuda_compat，已在 source 时自动尝试）。
  # 必须限定 -i 0 且兜住退出码：nvidia-smi 是逐卡查询、逐行输出，若用
  # `| head -1` 读走第一行就关管道，nvidia-smi 写第二行时会收到 SIGPIPE（退出码
  # 141），在 set -o pipefail + set -e 下会让整个 T0 **静默退出、无任何报错**。
  DRV_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d ' ' || true)"
  _probe_cuda() {
    python - <<'PY'
import torch
print(f"{torch.__version__}|{torch.version.cuda}|{int(torch.cuda.is_available())}")
PY
  }
  CUDA_STATE="$(_probe_cuda || true)"
  if [[ -z "$CUDA_STATE" ]]; then
    die "import torch 失败（拿不到 torch 版本）；请检查依赖与 CUDA 前向兼容是否生效"
  fi
  IFS='|' read -r TORCH_VER TORCH_CUDA CUDA_OK <<< "$CUDA_STATE"
  log "CUDA 可用性: torch=$TORCH_VER (cu$TORCH_CUDA) 宿主驱动=$DRV_VER torch.cuda.is_available=$CUDA_OK"

  if [[ "$CUDA_OK" != "1" ]] && [[ "${CUDA_COMPAT_ACTIVE:-0}" != "1" ]]; then
    log "torch.cuda 不可用，尝试强制启用镜像自带的 CUDA 前向兼容包 ..."
    CUDA_COMPAT=force ensure_cuda_compat
    if [[ "${CUDA_COMPAT_ACTIVE:-0}" == "1" ]]; then
      CUDA_STATE="$(_probe_cuda || true)"
      IFS='|' read -r TORCH_VER TORCH_CUDA CUDA_OK <<< "$CUDA_STATE"
      log "启用 compat 后: torch.cuda.is_available=$CUDA_OK"
    fi
  fi

  if [[ "$CUDA_OK" != "1" ]]; then
    warn "nvidia-smi 能看到 GPU，但 torch.cuda 不可用"
    warn "镜像按 CUDA $TORCH_CUDA 编译；容器内用户态驱动来自宿主注入（libcuda.so = 宿主驱动版本 $DRV_VER）"
    if _find_cuda_compat_dir >/dev/null 2>&1; then
      warn "找到 compat 目录 $(_find_cuda_compat_dir)，但启用后仍不可用——该包可能与本机驱动分支不兼容"
    else
      warn "镜像内未找到 /usr/local/cuda/compat（cuda-compat-13-0）"
    fi
    die "三条路选一条：① 升宿主驱动到 >= ${CUDA_COMPAT_MIN_DRIVER:-580}；② 换带 cuda-compat-13-0 的镜像（需 >= swift4.5.3）；③ 换 cu12x 且 swift>=4.5.3 的镜像"
  fi

  if [[ "${CUDA_COMPAT_ACTIVE:-0}" == "1" ]]; then
    log "本次运行处于 CUDA 前向兼容模式：libcuda 取自 ${CUDA_COMPAT_USED_DIR:-/usr/local/cuda/compat}（镜像内 $(_cuda_injected_major || echo '?') 已被覆盖为 compat 版本）"
    log "  nvidia-smi 仍会报宿主驱动 ${DRV_VER}，属正常——它走 NVML，不是 libcuda"
  fi
else
  warn "未找到 nvidia-smi（本机 Windows 无 GPU 属预期；训练机上必须存在）"
fi

# --- Python 依赖与版本门槛 ---------------------------------------------------
python - "$MODEL_TYPE" "$MIN_TRANSFORMERS_27B" "$MIN_TRANSFORMERS_FLASH_NEXT" \
         "$MIN_FLA_27B" "$MIN_FLA_FLASH_NEXT" <<'PY'
import importlib, sys

def version(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, '__version__', '?')
    except Exception:
        return None

def numeric(v):
    out = []
    for part in str(v).split('.')[:3]:
        digits = ''.join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))

def ge(v, minimum):
    return v is not None and numeric(v) >= numeric(minimum)

model_type, t27, tfl, f27, ffl = sys.argv[1:6]
min_t, min_f = (tfl, ffl) if model_type == 'qwen4_exp' else (t27, f27)
checks = [
    ('torch',                  None, True),
    ('transformers',           min_t, True),
    ('modelscope',             None, True),
    ('swift',                  None, True),
    ('megatron.core',          None, True),
    ('mcore_bridge',           None, True),
    ('transformer_engine',     None, True),
    ('flash_attn',             None, True),
    ('fla',                    min_f, True),   # flash-linear-attention
    ('causal_conv1d',          None, True),
    ('decord',                 None, True),
    ('qwen_vl_utils',          None, True),
    ('zstandard',              None, False),  # 仅本地抽子集需要
]
print(f'{"package":<22} {"version":<18} {"require":<12} result')
print('-' * 66)
failed = []
for name, minimum, required in checks:
    v = version(name)
    ok = (v is not None) and (not minimum or ge(v, minimum))
    flag = 'OK' if ok else ('FAIL' if required else 'WARN')
    print(f'{name:<22} {str(v or "MISSING"):<18} {str(minimum or "-"):<12} {flag}')
    if not ok and required:
        failed.append((name, minimum))
print('-' * 66)
if failed:
    print('以下内容必须补齐（版本对比用数值分段，避免 5.16 < 5.9 的字符串误判）：')
    for name, minimum in failed:
        print(f'  - {name}: 需要 >= {minimum}' if minimum else f'  - {name}: 未安装')
    sys.exit(1)
print('[OK] 必需依赖与版本门槛全部通过')
PY

# --- 镜像能力判定（必须 >= ms-swift v4.5.3，否则不认识 Qwen3.8） -------------
log "镜像能力探针：版本矩阵 + qwen3_8/qwen4_exp 注册扫描 ..."
PROBE_OUT="$LOG_ROOT/image_probe_${MODEL_KEY}.json"
python "$PROJECT_ROOT/tools/probe_image.py" --out "$PROBE_OUT" --no-pip-check \
  || die "当前镜像不支持 Qwen3.8（报告: $PROBE_OUT）。请换官方 swift4.5.3 镜像（config/cluster.env 的 RUN_IMAGE），见 docs/AUDIT.md"
log "镜像探针通过，报告: $PROBE_OUT"

# --- 权重完整性（清单来自 ModelScope 官方 API） ------------------------------
log "校验权重完整性（按字节大小 + 小文件 sha256）..."
python "$PROJECT_ROOT/tools/verify_weights.py" --model "$MODEL_PROFILE" --path "$MODEL_PATH"

# --- config.json 架构一致性 --------------------------------------------------
python - "$MODEL_PATH/config.json" "$ARCHITECTURE" "$MODEL_TYPE" <<'PY'
import json, sys
path, expect_arch, expect_type = sys.argv[1:4]
try:
    cfg = json.load(open(path, encoding='utf-8'))
except Exception as e:
    sys.exit(f'[FAIL] 无法读取 {path}: {e}')
arch = (cfg.get('architectures') or ['?'])[0]
mtype = cfg.get('model_type')
tc = cfg.get('text_config', {})
print(f'config.json: architectures={arch}, model_type={mtype}, '
      f'layers={tc.get("num_hidden_layers")}, hidden={tc.get("hidden_size")}, '
      f'experts={tc.get("num_experts", "dense")}, dtype={tc.get("dtype")}')
if arch != expect_arch or mtype != expect_type:
    sys.exit(f'[FAIL] 架构不匹配：期望 {expect_arch}/{expect_type}，实得 {arch}/{mtype}')
print('[OK] 架构与本档案一致')
PY

# --- CLI 实际支持的参数（版本差异的兜底） ------------------------------------
# 两个坑：
#   1. `megatron pt --help` 实际走 torch.distributed.run，会按 NPROC_PER_NODE 起进程
#      并做 rendezvous。参数列表与进程数无关，所以强制 NPROC_PER_NODE=1——否则
#      8 个进程各自初始化，慢且容易被 MASTER_ADDR 拖死。
#   2. 即使如此仍可能因 CUDA / 网络初始化阻塞，必须加超时，否则 T0 会
#      「卡在最后一步」且没有任何输出。
HELP_FILE="$LOG_ROOT/pt_help_${MODEL_KEY}.txt"
HELP_TIMEOUT="${PT_HELP_TIMEOUT:-120}"
log "探测 CLI 参数：megatron pt --help（单进程 + 超时 ${HELP_TIMEOUT}s）..."
if command -v timeout >/dev/null 2>&1; then
  NPROC_PER_NODE=1 timeout "$HELP_TIMEOUT" megatron pt --help > "$HELP_FILE" 2>&1 \
    || warn "megatron pt --help 超时(${HELP_TIMEOUT}s)或失败，CLI 参数能力未知；见 $HELP_FILE"
else
  NPROC_PER_NODE=1 megatron pt --help > "$HELP_FILE" 2>&1 || warn "megatron pt --help 执行失败，请检查安装"
fi
log "已保存 CLI 帮助 -> $HELP_FILE"
if ! grep -q -- "--train_iters" "$HELP_FILE" 2>/dev/null; then
  warn "帮助文本里没有 --train_iters，说明 --help 未正常输出（文件只有启动日志）。"
  warn "  多半是 rendezvous 连不上 MASTER_ADDR；当前 MASTER_ADDR=$MASTER_ADDR（单机应为 127.0.0.1）"
fi
log "与断点续训/权重加载/指标相关的实际参数："
awk '/(--load|--mcore_model|--finetune|--no_load_optim|--no_save_optim|--save_steps|--train_iters|--streaming|--packing|--save_safetensors|--log_throughput|--log-throughput|--exit_duration_in_mins|--exit-duration-in-mins)/ {print "   " $0}' "$HELP_FILE" | head -30 || true

log "=== T0 通过 ==="
