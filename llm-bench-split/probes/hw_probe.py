#!/usr/bin/env python3
# =============================================================================
# [容器内] 硬件微基准 + collective 全类型扫描。由 04_bench.sh --phase hw / coll / nccl 调起。
# 输出 JSONL（每行一个指标），供 05_report.py 汇总。
#   torchrun --nproc_per_node=8 hw_probe.py --out /workspace/logs/coll.jsonl
# 参数：--out 输出；--secs 每项稳态时长（默认 5s）；--sizes 消息字节列表；
#       --only gemm|hbm|l2|pcie|coll|all（默认 all）
# =============================================================================
import argparse
import json
import os
import sys

import torch

R = int(os.environ.get("RANK", 0))
L = int(os.environ.get("LOCAL_RANK", 0))
W = int(os.environ.get("WORLD_SIZE", 1))
NODES = int(os.environ.get("NNODES", 1))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
# 跨机单消息上限：8GB bf16 = 2^32 个元素，跨机路径会在计数/偏移上越界，
# 实测触发 "CUDA error: an illegal memory access"。0 = 不限制（单机默认不限）。
MAX_MSG = int(os.environ.get("COLL_MAX_MSG", str(2 * 1024 ** 3) if NODES > 1 else "0"))
torch.cuda.set_device(L)


def emit(rec):
    # 每个节点的 local rank0 各写一份（OUT 在各自节点的 LOGDIR 挂载里，互不冲突）。
    # 不能用全局 R!=0 过滤：全局 rank0 永远只在 node-a，node-b 的 rdma-gate.jsonl
    # 会永远为空 → 双机门禁在 node-b 上永远 FAIL（11-08 实测）。
    # 记录里已带 node_rank / NODE_RANK，多节点文件可区分来源。
    if L != 0:
        return
    rec.setdefault("world", W)
    rec.setdefault("nodes", NODES)
    rec.setdefault("node_rank", NODE_RANK)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def barrier():
    if W > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


def timed(fn, secs=5.0, min_it=3, max_it=500):
    """先估一次耗时，再按秒数跑够稳态；返回 (平均单次秒, 迭代数)。

    P1（11-10）：t 已经是「单次平均耗时」，不是 it 次总耗时。所有基于 timed()
    的带宽公式都不得再乘 it；只有自己用 CUDA Event 测总时长的 collective
    才需要乘 it。历史 bug 正是这两类混用：HBM/L2/PCIe 多乘 it（虚高 50~500 倍），
    collective busbw 少乘 it（低估 30~100 倍）。
    """
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    one = max(s.elapsed_time(e) / 1e3, 1e-6)
    it = max(min_it, min(max_it, int(secs / one)))
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / it, it


def gib(x):
    return round(x / 1024 ** 3, 1)


# ---------------------------------------------------------------- 单卡
def dev_info():
    # 用本机可见卡数，不用 WORLD_SIZE：双机时 WORLD=16 但每台只有 8 张。
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    emit({"kind": "gpu", "world": W, "local_gpus": len(props),
          "cap_gib": sorted({gib(p.total_memory) for p in props}),
          "names": sorted({p.name for p in props}),
          "sm": sorted({f"{p.major}.{p.minor}" for p in props})})


def gemm_bf16(secs):
    n = 8192
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    t, it = timed(lambda: a @ b, secs)
    emit({"kind": "gemm", "dtype": "bf16", "n": n, "tflops": round(2 * n ** 3 / t / 1e12, 1),
          "secs": round(t * it, 2), "iters": it})
    del a, b
    torch.cuda.empty_cache()


def gemm_fp8(secs):
    """FP8 GEMM 基线（P0 11-10）：原探针恒 SKIP，因为 B 未转置、scale 是 1 维张量。

    PyTorch 2.6 / Hopper 的 torch._scaled_mm 要求：
      * B 必须是列主序的转置视图（b.t()），否则报 "expected B to be transposed"；
      * scale_a / scale_b 必须是 0 维标量张量，1 维 [1.0] 会被拒；
      * out_dtype 关键字按版本可缺省。
    三种签名全部抛异常时原代码只写「_scaled_mm 不可用」，无法诊断，这里保留逐项原因。
    """
    n = 8192
    try:
        assert hasattr(torch, "float8_e4m3fn"), "torch 无 float8_e4m3fn"
        a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn).t()
        sa = torch.tensor(1.0, device="cuda")
        fn, why = None, []
        for label, call in (
            ("kw", lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sa, out_dtype=torch.bfloat16)),
            ("pos", lambda: torch._scaled_mm(a, b, sa, sa, out_dtype=torch.bfloat16)),
            ("no-out", lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sa)),
            ("no-scale", lambda: torch._scaled_mm(a, b, out_dtype=torch.bfloat16)),
        ):
            try:
                call()
                torch.cuda.synchronize()
                fn = call
                break
            except Exception as e:
                why.append(f"{label}:{type(e).__name__}:{str(e)[:70]}")
        if fn is None:
            raise RuntimeError("_scaled_mm 全部签名失败 → " + " | ".join(why)[:180])
        t, it = timed(fn, secs)
        emit({"kind": "gemm", "dtype": "fp8", "n": n,
              "tflops": round(2 * n ** 3 / t / 1e12, 1), "secs": round(t * it, 2), "iters": it,
              "status": "OK"})
        del a, b
    except Exception as e:
        emit({"kind": "gemm", "dtype": "fp8", "n": n, "status": "SKIP",
              "note": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()


def hbm(secs):
    m = 1_200_000_000                     # bf16 ≈ 2.4 GB，远大于 50 MB L2
    x = torch.randn(m, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    t, it = timed(lambda: x.sum(), secs, 5)
    emit({"kind": "hbm", "op": "read", "tbs": round(m * 2 / t / 1e12, 2),
          "secs": round(t * it, 2), "iters": it})
    t, it = timed(lambda: y.copy_(x), secs, 5)
    emit({"kind": "hbm", "op": "copy", "tbs": round(m * 4 / t / 1e12, 2),
          "secs": round(t * it, 2), "iters": it})
    del x, y
    torch.cuda.empty_cache()


def l2(secs):
    m = 16 * 1024 * 1024                  # 32 MB 张量，落在 L2（≈50 MB）
    x = torch.randn(m, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    t, it = timed(lambda: y.copy_(x), secs, 10)
    emit({"kind": "l2", "gbps": round(m * 8 / t / 1e9, 1), "bytes": m * 4,
          "secs": round(t * it, 2), "iters": it})
    del x, y
    torch.cuda.empty_cache()


def pcie(secs):
    n = 256 * 1024 * 1024                 # float32 1 GB
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, dtype=torch.float32, device="cuda")
    h.normal_()
    d.copy_(h); torch.cuda.synchronize()   # 预热
    t, it = timed(lambda: d.copy_(h, non_blocking=True), secs, 3, 50)
    emit({"kind": "pcie", "dir": "h2d", "gbs": round(n * 4 / t / 1e9, 1),
          "secs": round(t * it, 2), "iters": it})
    t, it = timed(lambda: h.copy_(d, non_blocking=True), secs, 3, 50)
    emit({"kind": "pcie", "dir": "d2h", "gbs": round(n * 4 / t / 1e9, 1),
          "secs": round(t * it, 2), "iters": it})
    del h, d
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- collective
def one_collective(op, msg_bytes, secs):
    n = max(1, msg_bytes // 2)            # bf16
    extra = {}
    if op in ("all_to_all", "reduce_scatter"):   # 这两个要求元素数能被 world 整除
        n = ((n + W - 1) // W) * W
    if op == "all_reduce":
        t = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        t.fill_(1.0)
        fn, payload_b, factor = (lambda: torch.distributed.all_reduce(t)), n * 2, 2 * (W - 1) / W
    elif op == "all_gather":
        if W * msg_bytes > 80 * 1024 ** 3:
            emit({"kind": "collective", "op": op, "msg_bytes": msg_bytes, "status": "SKIP",
                  "note": "输出 >80GB，显存放不下"})
            return
        i = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(W * n, dtype=torch.bfloat16, device="cuda")
        fn, payload_b, factor = (lambda: torch.distributed.all_gather_into_tensor(o, i)), n * 2 * W, (W - 1) / W
    elif op == "reduce_scatter":
        i = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(n // W if n >= W else 1, dtype=torch.bfloat16, device="cuda")
        fn, payload_b, factor = (lambda: torch.distributed.reduce_scatter_tensor(o, i)), n * 2, (W - 1) / W
    else:                                  # all_to_all
        i = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        fn, payload_b, factor = (lambda: torch.distributed.all_to_all_single(o, i)), n * 2, (W - 1) / W
        # MoE 每 rank 把本组 token 按 expert 切给 world 个 rank：均衡切分时每组 = n/world
        extra = {"groups": W, "tokens_per_group": n // W, "split_dim": 0,
                 "group_bytes": (n // W) * 2}

    fn(); torch.cuda.synchronize(); barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    one = max(s.elapsed_time(e) / 1e3, 1e-6)
    it = 200 if msg_bytes <= 64 * 1024 else max(3, min(100, int(secs / one)))
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    # P1（11-10）：sec 是 it 次通信的**总**耗时，原公式 payload_b*factor/sec/1e9
    # 少除了 it，导致记录值被低估 it 倍（且 it 随消息大小在 200/100/30 之间变化，
    # 曲线因此不可比）。先算单次耗时与算法带宽，再乘 factor 得总线带宽。
    sec = max(s.elapsed_time(e) / 1e3, 1e-9)
    per_call = sec / it
    algbw = payload_b / per_call / 1e9
    busbw = algbw * factor
    rec = {"kind": "collective", "op": op, "msg_bytes": msg_bytes, "iters": it,
           "secs": round(sec, 4), "algbw_gbs": round(algbw, 1),
           "busbw_gbs": round(busbw, 1), "latency_us": round(per_call * 1e6, 1),
           "status": "OK"}
    rec.update(extra)
    emit(rec)
    barrier()


def oversize_note(msg_bytes):
    """跨机单消息过大 → 返回 SKIP 原因；None 表示照常跑（单机 MAX_MSG=0，不受影响）。"""
    if NODES > 1 and MAX_MSG > 0 and msg_bytes > MAX_MSG:
        return f"跨机单消息 {msg_bytes}B > 上限 {MAX_MSG}B（规避跨机大消息非法访存）"
    return None


def collectives(secs, sizes, ops=("all_reduce", "all_gather", "reduce_scatter", "all_to_all")):
    emit({"kind": "nccl", "world": W, "version": ".".join(map(str, torch.cuda.nccl.version()))})
    for op in ops:
        for b in sizes:
            note = oversize_note(b)
            if note:
                emit({"kind": "collective", "op": op, "msg_bytes": b, "status": "SKIP", "note": note})
                continue
            try:
                one_collective(op, b, secs)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:90]}"
                emit({"kind": "collective", "op": op, "msg_bytes": b, "status": "FAIL", "note": msg})
                torch.cuda.empty_cache()
                # 非法访存后 CUDA 上下文已污染，继续跑只会级联报错并掩盖首个失败点
                if "CUDA error" in msg or "illegal memory access" in msg:
                    raise
                barrier()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/logs/coll.jsonl")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--sizes", default="8,65536,1048576,67108864,1073741824,8589934592")
    ap.add_argument("--only", default="all")
    ap.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,all_to_all")
    a = ap.parse_args()
    OUT = a.out
    if R == 0:
        open(OUT, "w").close()
    barrier()
    if W > 1:
        torch.distributed.init_process_group("nccl")
    sizes = [int(x) for x in a.sizes.split(",") if x]

    if a.only in ("all", "gemm"):
        dev_info()
        gemm_bf16(a.secs)
        gemm_fp8(a.secs)
    if a.only in ("all", "hbm"):
        hbm(a.secs)
    if a.only in ("all", "l2"):
        l2(a.secs)
    if a.only in ("all", "pcie"):
        pcie(a.secs)
    if a.only in ("all", "coll") and W > 1:
        collectives(a.secs, sizes, [x for x in a.ops.split(",") if x])
    if W > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
