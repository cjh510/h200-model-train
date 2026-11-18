#!/usr/bin/env python3
# =============================================================================
# [容器内] checkpoint 写/读吞吐代理。由 04_bench.sh --phase io 调起。
# 每个 rank 写自己那一片（总字节 = 模型 bf16 权重），聚合后报 MB/s。
# 这是存储带宽代理，不是 Megatron 的分布式 checkpoint 格式（报告里要写明）。
#   torchrun --nproc_per_node=8 io_probe.py --model 8B --out /workspace/logs/io.jsonl
# =============================================================================
import argparse
import json
import os
import sys
import time

import torch

R = int(os.environ.get("RANK", 0))
L = int(os.environ.get("LOCAL_RANK", 0))
W = int(os.environ.get("WORLD_SIZE", 1))
N_PARAMS = {"8B": 8.03e9, "70B": 70.55e9}
torch.cuda.set_device(L)
if W > 1:
    torch.distributed.init_process_group("nccl")


def emit(rec):
    if R == 0:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8B")
    ap.add_argument("--out", default="/workspace/logs/io.jsonl")
    ap.add_argument("--dir", default="/workspace/ckpt")
    a = ap.parse_args()
    OUT = a.out
    if R == 0 and not os.path.exists(OUT):
        open(OUT, "w").close()
    total = int(N_PARAMS[a.model] * 2)          # bf16 字节
    shard = total // W
    n = max(1, shard // 2)
    os.makedirs(a.dir, exist_ok=True)
    f = os.path.join(a.dir, f"{a.model}-rank{R}.bin")

    torch.cuda.synchronize()
    if W > 1:
        torch.distributed.barrier()
    t0 = time.time()
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    torch.save(x.cpu(), f)
    if W > 1:
        torch.distributed.barrier()
    tw = time.time() - t0

    t0 = time.time()
    y = torch.load(f, map_location="cpu")
    if W > 1:
        torch.distributed.barrier()
    tr = time.time() - t0
    del x, y
    try:
        os.remove(f)
    except OSError:
        pass

    if R == 0:
        emit({"kind": "checkpoint", "model": a.model, "world": W,
              "total_gib": round(total / 1024 ** 3, 1),
              "write_mbs": round(total / tw / 1e6, 1), "read_mbs": round(total / tr / 1e6, 1),
              "write_s": round(tw, 2), "read_s": round(tr, 2)})
    if W > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
