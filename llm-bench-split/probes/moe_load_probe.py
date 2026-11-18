#!/usr/bin/env python3
# =============================================================================
# [容器内] MoE 专家负载采样探针（P1 11-10）
#
# 根因：core_v0.13.1 的 pretrain_gpt.py 不向 stdout 打印逐专家 token 计数。
#   --moe-per-layer-logging 只打 aux loss / z loss；tokens_per_expert 仅进
#   tensorboard / wandb。04_bench.sh 的 expert_load_pair() 却只认
#   "expert load ... max / mean" 这种从未出现过的格式 → 采样全 null。
#
# 用法：04_bench.sh 的 moe_probe_enable() 把本文件复制成 $LOGDIR/sitecustomize.py
#   （LOGDIR 已挂载为 /workspace/logs），并把它加进 PYTHONPATH、设 MOE_LOAD_PROBE=1。
#   Python 启动时自动 import sitecustomize → 装一个延迟 import hook → 在 Megatron
#   的 TopKRouter 首次可用时包裹 routing()，按间隔把每专家计数打进训练日志：
#       [expert load] step=200 max/mean=12.000/9.000 counts=[12,9,8,...]
#   expert_load_pair() 解析该行得到 max/mean；expert_sample() 取 counts 列表。
#
# 设计约束：
#   * 环境变量未开启时本文件几乎零开销（不碰 builtins）；
#   * 不在启动阶段 import megatron（避免 CUDA/TE 初始化时机问题），用 import
#     hook 延迟到 Megatron 自己加载后再 patch；
#   * 任何导入失败 / 签名不匹配都打印明确原因并降级，绝不影响训练本身；
#   * 只在 RANK=0 打印，避免 8 个 rank 重复刷屏。
# =============================================================================
import builtins
import os
import sys

ENABLED = os.environ.get("MOE_LOAD_PROBE", "0") == "1"
RANK = int(os.environ.get("RANK", 0))
try:
    LOG_EVERY = max(1, int(os.environ.get("MOE_LOAD_LOG_EVERY", "200")))
except ValueError:
    LOG_EVERY = 200
TAG = "[expert load probe]"


def _log(msg):
    if RANK == 0:
        print(f"{TAG} {msg}", flush=True)


if ENABLED:
    _real_import = builtins.__import__
    _calls = 0            # routing() 调用计数（跨所有 MoE 层）
    _tries = 0            # 延迟安装尝试次数
    _patched = False
    _disabled = False
    _installing = False   # 防重入：_install() 内部的 from ... import 会再次触发 _hook

    def _num_experts(router):
        for obj in (router, getattr(router, "config", None)):
            if obj is None:
                continue
            for key in ("num_moe_experts", "num_experts"):
                v = getattr(obj, key, None)
                if isinstance(v, int) and v > 0:
                    return v
        return None

    def _counts_from(result, num_experts):
        """从 routing() 返回值里提取每专家 token 计数（兼容多种版本签名）。"""
        import torch
        parts = result if isinstance(result, (tuple, list)) else (result,)
        for obj in parts:
            if not torch.is_tensor(obj):
                continue
            if obj.dtype == torch.bool and obj.dim() >= 2:
                # routing_map: [..., tokens, experts] → 按 experts 维求和
                return obj.reshape(-1, obj.shape[-1]).sum(dim=0)
            if obj.dtype in (torch.int32, torch.int64) and obj.dim() >= 1:
                # topk indices: [tokens, topk]
                return torch.bincount(obj.reshape(-1).long(), minlength=num_experts or 1)
        # 有些版本把计数挂在 router / dispatcher 属性上
        owner = result
        if isinstance(result, (tuple, list)):
            owner = result[0]
        for attr in ("tokens_per_expert", "local_tokens_per_expert"):
            v = getattr(owner, attr, None)
            if v is not None:
                return torch.as_tensor(v, dtype=torch.float32).reshape(-1)
        return None

    def _install():
        global _patched, _installing
        if _installing:
            return False
        _installing = True
        try:
            from megatron.core.transformer.moe import router as router_mod
            cls = getattr(router_mod, "TopKRouter", None)
            if cls is None or not hasattr(cls, "routing"):
                return False
            orig = cls.routing
            if getattr(orig, "_moe_load_probe", False):
                _patched = True
                return True

            def wrapped(self, *args, **kwargs):
                global _calls
                result = orig(self, *args, **kwargs)
                try:
                    _calls += 1
                    if _calls % LOG_EVERY != 0:
                        return result
                    counts = _counts_from(result, _num_experts(self))
                    if counts is None:
                        counts = _counts_from(self, _num_experts(self))
                    if counts is None or counts.numel() == 0:
                        _log(f"warn: 无法从 routing() 提取专家计数（返回类型 {type(result).__name__}）")
                        return result
                    c = counts.detach().float().cpu()
                    mx, mean = c.max().item(), c.mean().item()
                    vals = ",".join(str(int(x)) for x in c.tolist())
                    if RANK == 0:
                        print(f"[expert load] step={_calls} max/mean={mx:.3f}/{mean:.3f} "
                              f"counts=[{vals}]", flush=True)
                except Exception as e:  # 采样失败绝不能影响训练
                    _log(f"warn: 采样异常 {type(e).__name__}: {e}")
                return result

            wrapped._moe_load_probe = True
            cls.routing = wrapped
            _patched = True
            _log(f"installed: TopKRouter.routing, every {LOG_EVERY} calls, RANK={RANK}")
            return True
        except Exception as e:
            _log(f"install failed: {type(e).__name__}: {e}")
            return False
        finally:
            _installing = False

    def _hook(name, globals=None, locals=None, fromlist=(), level=0):
        global _tries, _disabled
        mod = _real_import(name, globals, locals, fromlist, level)
        if (not _patched and not _disabled and not _installing
                and isinstance(name, str) and name.startswith("megatron")):
            _tries += 1
            if not _install() and _tries >= 30:
                _disabled = True
                _log("disabled: 30 次尝试后仍未找到可 patch 的 TopKRouter.routing；"
                     "expert-load-timeseries.csv 将保持 null，请核对 Megatron 版本")
        return mod

    builtins.__import__ = _hook
