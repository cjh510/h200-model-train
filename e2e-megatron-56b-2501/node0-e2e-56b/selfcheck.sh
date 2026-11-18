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
