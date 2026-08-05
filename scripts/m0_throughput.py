"""M0 — throughput probe. Run this first, before writing anything real.

Two things to establish:
  1. actual steps/sec for a GPT-2-small-shaped model on this machine
  2. whether any op is silently falling back to CPU

The second is the dangerous one. PYTORCH_ENABLE_MPS_FALLBACK=1 masks unsupported
ops by running them on CPU — no error, just a collapse in throughput that you
discover on day three.

Usage:
    python scripts/m0_throughput.py --layers 12 --d-model 768 --steps 100
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


def build_probe(n_layers: int, d_model: int, n_heads: int) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = build_probe(args.layers, args.d_model, args.heads).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=dtype)

    for _ in range(10):  # warmup
        opt.zero_grad()
        model(x).mean().backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad()
        model(x).mean().backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device        {device}  dtype {args.dtype}")
    print(f"params        {n_params / 1e6:.1f}M")
    print(f"steps/sec     {args.steps / elapsed:.2f}")
    print(f"ms/step       {elapsed / args.steps * 1000:.1f}")
    print(f"tokens/sec    {args.steps * args.batch * args.seq_len / elapsed:.0f}")
    print()
    print("Now check for silent CPU fallbacks:")
    print("  PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/m0_throughput.py")
    print("If that errors, an op is unsupported and was running on CPU.")


if __name__ == "__main__":
    main()
