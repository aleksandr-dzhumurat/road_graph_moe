"""
Capacity planning: read Gold Parquet shards and estimate training time.

Reads actual token statistics from data/gold/, then computes estimated
training duration for a given GPU (default: Nvidia L40S, 48 GB).

Usage:
    python scripts/capacity_planning.py
    python scripts/capacity_planning.py --gpu-tflops 362 --gpu-mem-gb 48
    python scripts/capacity_planning.py --steps 300000 --batch 32
"""

import argparse
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD = REPO_ROOT / "data" / "gold"

# ---------------------------------------------------------------------------
# GPU presets
# ---------------------------------------------------------------------------
GPU_PRESETS = {
    "L40S":   {"tflops_bf16": 362,  "mem_gb": 48},
    "A100-40":{"tflops_bf16": 312,  "mem_gb": 40},
    "A100-80":{"tflops_bf16": 312,  "mem_gb": 80},
    "H100":   {"tflops_bf16": 989,  "mem_gb": 80},
    "RTX4090":{"tflops_bf16": 165,  "mem_gb": 24},
}

# ---------------------------------------------------------------------------
# Model config (from phase_01.md)
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
    "n_layers":    12,
    "d_model":     768,
    "n_heads":     12,
    "ffn_mult":    4,
    "vocab_size":  30_004,
    "max_seq_len": 1024,
}


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------
def read_gold_stats() -> dict:
    stats = {}
    for region in ["porto", "beijing"]:
        region_dir = GOLD / region
        if not region_dir.exists():
            print(f"  [!] {region_dir} not found — skipping")
            continue
        shards = sorted(region_dir.glob("shard_*.parquet"))
        if not shards:
            print(f"  [!] No shards in {region_dir} — skipping")
            continue
        table = pa.concat_tables([pq.read_table(s, columns=["n_tokens"]) for s in shards])
        n_tokens = table.column("n_tokens").to_pylist()
        stats[region] = {
            "n_trajectories": len(n_tokens),
            "mean_len":       sum(n_tokens) / len(n_tokens),
            "median_len":     sorted(n_tokens)[len(n_tokens) // 2],
            "min_len":        min(n_tokens),
            "max_len":        max(n_tokens),
            "p95_len":        sorted(n_tokens)[int(0.95 * len(n_tokens))],
            "total_tokens":   sum(n_tokens),
        }
    return stats


# ---------------------------------------------------------------------------
# Model parameter count
# ---------------------------------------------------------------------------
def count_params(cfg: dict) -> int:
    L, D, V, F = cfg["n_layers"], cfg["d_model"], cfg["vocab_size"], cfg["ffn_mult"]
    embedding   = V * D                          # token embedding
    pos_embed   = cfg["max_seq_len"] * D         # positional embedding
    per_layer   = (4 * D * D +                  # QKV + out projections
                   2 * D * F * D)               # FFN up + down
    ln_bias     = L * 2 * 2 * D                 # 2 LayerNorms per block
    heads       = 2 * V * D                     # MTM + AR heads (weight-tied = 1x)
    return embedding + pos_embed + L * per_layer + ln_bias + heads


# ---------------------------------------------------------------------------
# FLOPs per forward pass (approximate, from Chinchilla / PaLM analyses)
# 6 * N * T per step for forward+backward where N=params, T=seq_len
# ---------------------------------------------------------------------------
def flops_per_step(n_params: int, seq_len: int, batch_size: int) -> float:
    return 6 * n_params * seq_len * batch_size


# ---------------------------------------------------------------------------
# Memory estimate (model + optimizer + activations, bf16)
# ---------------------------------------------------------------------------
def estimate_memory_gb(n_params: int, batch_size: int, seq_len: int, d_model: int) -> dict:
    bytes_per_param_bf16 = 2
    bytes_per_param_fp32 = 4

    model_gb       = n_params * bytes_per_param_bf16 / 1e9
    # AdamW: 2 fp32 momentum buffers
    optimizer_gb   = n_params * bytes_per_param_fp32 * 2 / 1e9
    # Activations: rough estimate — 2 * batch * seq * d_model * n_layers bytes
    activations_gb = (2 * batch_size * seq_len * d_model * MODEL_DEFAULTS["n_layers"]
                      * bytes_per_param_bf16 / 1e9)
    total_gb       = model_gb + optimizer_gb + activations_gb
    return {
        "model_gb":       round(model_gb, 2),
        "optimizer_gb":   round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb":       round(total_gb, 2),
    }


# ---------------------------------------------------------------------------
# Training time estimate
# ---------------------------------------------------------------------------
def estimate_training_time(
    n_params: int,
    total_tokens: int,
    batch_size: int,
    seq_len: int,
    gpu_tflops: float,
    mfu: float = 0.40,        # typical achieved MFU on a single GPU
    n_steps: int | None = None,
) -> dict:
    steps = n_steps or (total_tokens // (batch_size * seq_len))
    flops = flops_per_step(n_params, seq_len, batch_size)
    achieved_tflops = gpu_tflops * 1e12 * mfu
    secs_per_step   = flops / achieved_tflops
    total_secs      = steps * secs_per_step
    return {
        "steps":           steps,
        "flops_per_step":  flops,
        "secs_per_step":   round(secs_per_step, 3),
        "total_hours":     round(total_secs / 3600, 1),
        "total_days":      round(total_secs / 86400, 1),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  CAPACITY PLANNING REPORT")
    print("=" * 60)

    # ── Dataset ──────────────────────────────────────────────────
    print("\n── Dataset (data/gold/) ─────────────────────────────────")
    stats = read_gold_stats()
    total_trajectories = 0
    total_tokens_all   = 0
    for region, s in stats.items():
        print(f"\n  {region}")
        print(f"    Trajectories : {s['n_trajectories']:>12,}")
        print(f"    mean / median: {s['mean_len']:>7.1f} / {s['median_len']:>7.1f}  tokens")
        print(f"    min / max    : {s['min_len']:>7} / {s['max_len']:>7}  tokens")
        print(f"    p95          : {s['p95_len']:>7.1f}  tokens")
        print(f"    Total tokens : {s['total_tokens']:>12,}")
        total_trajectories += s["n_trajectories"]
        total_tokens_all   += s["total_tokens"]

    print(f"\n  COMBINED")
    print(f"    Trajectories : {total_trajectories:>12,}")
    print(f"    Total tokens : {total_tokens_all:>12,}")

    # ── Model ────────────────────────────────────────────────────
    cfg = MODEL_DEFAULTS.copy()
    n_params = count_params(cfg)
    print(f"\n── Model (phase_01.md spec) ─────────────────────────────")
    print(f"    Layers × dim : {cfg['n_layers']} × {cfg['d_model']}")
    print(f"    Heads        : {cfg['n_heads']}")
    print(f"    Vocab        : {cfg['vocab_size']:,}")
    print(f"    Parameters   : {n_params / 1e6:.1f} M")

    # ── Memory ───────────────────────────────────────────────────
    mem = estimate_memory_gb(n_params, args.batch, args.seq_len, cfg["d_model"])
    print(f"\n── Memory estimate (batch={args.batch}, seq={args.seq_len}) ──")
    print(f"    Model (bf16)     : {mem['model_gb']:>6.2f} GB")
    print(f"    Optimizer (fp32) : {mem['optimizer_gb']:>6.2f} GB")
    print(f"    Activations      : {mem['activations_gb']:>6.2f} GB")
    print(f"    ─────────────────────────────")
    print(f"    Total            : {mem['total_gb']:>6.2f} GB  (GPU has {args.gpu_mem_gb} GB)")
    fits = mem["total_gb"] <= args.gpu_mem_gb
    print(f"    Fits on 1 GPU?   : {'YES' if fits else 'NO — reduce batch or use FSDP'}")

    # ── Training time ────────────────────────────────────────────
    print(f"\n── Training time estimate (1× {args.gpu}, MFU={args.mfu:.0%}) ──")
    timing = estimate_training_time(
        n_params      = n_params,
        total_tokens  = total_tokens_all,
        batch_size    = args.batch,
        seq_len       = args.seq_len,
        gpu_tflops    = args.gpu_tflops,
        mfu           = args.mfu,
        n_steps       = args.steps,
    )
    print(f"    Steps            : {timing['steps']:>12,}")
    print(f"    FLOPs / step     : {timing['flops_per_step']:.2e}")
    print(f"    Seconds / step   : {timing['secs_per_step']:>9.3f} s")
    print(f"    Total            : {timing['total_hours']:>9.1f} h  "
          f"({timing['total_days']:.1f} days)")

    if timing["total_days"] > 7:
        n_gpus_week = math.ceil(timing["total_days"] / 7)
        print(f"\n    → To finish in 1 week: {n_gpus_week}× {args.gpu} in parallel")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate training capacity from Gold shards.")
    parser.add_argument("--gpu",        default="L40S",
                        choices=list(GPU_PRESETS),
                        help="GPU preset (default: L40S)")
    parser.add_argument("--gpu-tflops", type=float, default=None,
                        help="Override GPU bf16 TFLOPS (default: from preset)")
    parser.add_argument("--gpu-mem-gb", type=float, default=None,
                        help="Override GPU memory GB (default: from preset)")
    parser.add_argument("--batch",      type=int,   default=32,
                        help="Batch size per GPU (default: 32)")
    parser.add_argument("--seq-len",    type=int,   default=512,
                        dest="seq_len",
                        help="Sequence length for FLOP calculation (default: 512)")
    parser.add_argument("--mfu",        type=float, default=0.40,
                        help="Model FLOP utilisation 0–1 (default: 0.40)")
    parser.add_argument("--steps",      type=int,   default=None,
                        help="Override step count (default: derived from total tokens)")
    args = parser.parse_args()

    preset = GPU_PRESETS[args.gpu]
    if args.gpu_tflops is None:
        args.gpu_tflops = preset["tflops_bf16"]
    if args.gpu_mem_gb is None:
        args.gpu_mem_gb = preset["mem_gb"]

    return args


if __name__ == "__main__":
    print_report(parse_args())
