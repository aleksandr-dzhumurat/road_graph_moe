"""
Parse a Nebius job log and plot training curves.

Supports two log formats automatically:

  Phase 1 — backbone pre-training (JSONL or plain text):
    step 0 | loss 10.47 | ar 10.46 (ppl 34754.8) | mtm 10.49 | grad 2.30 | lr 0.00e+00

  Phase 2 — geographic expert training (plain text):
    Rank 0 Expert porto Step 100 | ar_loss: 3.21 | aux_loss: 1.98 | ...

Usage:
    python scripts/train_curve.py data/train_log_10.txt
    python scripts/train_curve.py data/train_log_10.txt --out data/expert_curves.png
    python scripts/train_curve.py data/train_log_10.txt --smooth 20

    # Fetch from Nebius then plot in one shot:
    nebius ai job logs <job-id> --since 2026-06-07 --tail 100000 --profile nebius \\
      | python scripts/train_curve.py /dev/stdin --out data/expert_curves.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Matches lines like:
# "2026-06-04 08:28:37 step      0 | loss 10.4685 | ar 10.4561 (ppl 34754.8) | mtm 10.4871 | grad 2.298 | lr 0.00e+00"
_STEP_RE = re.compile(
    r"step\s+(\d+)\s*\|"
    r"\s*loss\s+([\d.]+)\s*\|"
    r"\s*ar\s+([\d.]+)\s*\(ppl\s+([\d.]+)\)\s*\|"
    r"\s*mtm\s+([\d.]+)\s*\|"
    r"\s*grad\s+([\d.]+)\s*\|"
    r"\s*lr\s+([\deE.+-]+)"
)


def parse_log(path: Path) -> dict[str, list]:
    records: dict[str, list] = {
        "step": [], "loss": [], "ar": [], "ppl": [], "mtm": [], "grad": [], "lr": []
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msg = obj.get("message", "")
            except json.JSONDecodeError:
                msg = line
            m = _STEP_RE.search(msg)
            if not m:
                continue
            records["step"].append(int(m.group(1)))
            records["loss"].append(float(m.group(2)))
            records["ar"].append(float(m.group(3)))
            records["ppl"].append(float(m.group(4)))
            records["mtm"].append(float(m.group(5)))
            records["grad"].append(float(m.group(6)))
            records["lr"].append(float(m.group(7)))
    return records


def smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values)
    k = np.ones(window) / window
    return np.convolve(values, k, mode="same")


def plot(records: dict[str, list], out: Path, window: int) -> None:
    steps = np.array(records["step"])
    loss  = np.array(records["loss"])
    ar    = np.array(records["ar"])
    mtm   = np.array(records["mtm"])
    ppl   = np.array(records["ppl"])
    grad  = np.array(records["grad"])
    lr    = np.array(records["lr"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training curves", fontsize=14)

    def _plot(ax, y_raw, label, color, ylabel, log=False, smooth_y=True):
        ax.plot(steps, y_raw, color=color, alpha=0.2, linewidth=0.6)
        if smooth_y and window > 1:
            ax.plot(steps, smooth(y_raw, window), color=color, linewidth=1.5, label=label)
        else:
            ax.plot(steps, y_raw, color=color, linewidth=1.5, label=label)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel 1: combined + AR + MTM loss ───────────────────────────────────
    ax = axes[0, 0]
    ax.plot(steps, loss, color="gray",   alpha=0.15, linewidth=0.6)
    ax.plot(steps, ar,   color="steelblue", alpha=0.15, linewidth=0.6)
    ax.plot(steps, mtm,  color="tomato", alpha=0.15, linewidth=0.6)
    ax.plot(steps, smooth(loss, window), color="gray",      linewidth=1.5, label="combined")
    ax.plot(steps, smooth(ar,   window), color="steelblue", linewidth=1.5, label="AR")
    ax.plot(steps, smooth(mtm,  window), color="tomato",    linewidth=1.5, label="MTM")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: AR perplexity (log scale) ──────────────────────────────────
    ax = axes[0, 1]
    _plot(ax, ppl, "AR perplexity", "steelblue", "perplexity", log=True)
    ax.set_title("AR Perplexity (log scale)")

    # ── Panel 3: gradient norm ───────────────────────────────────────────────
    ax = axes[1, 0]
    _plot(ax, grad, "grad norm", "darkorange", "grad norm")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="clip=1.0")
    ax.legend(fontsize=8)
    ax.set_title("Gradient Norm")

    # ── Panel 4: learning rate ───────────────────────────────────────────────
    ax = axes[1, 1]
    _plot(ax, lr, "lr", "seagreen", "learning rate", log=True, smooth_y=False)
    ax.set_title("Learning Rate Schedule")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Phase-2 expert log parser + plotter
# ---------------------------------------------------------------------------

# Rank 0 Expert porto Step  100 | ar_loss: 3.21 | aux_loss: 1.98 | aux_weight: 0.00 |
# total_loss: 3.21 | route_entropy: 0.05 | porto_prob: 0.99 | beijing_prob: 0.01
_EXPERT_STEP_RE = re.compile(
    r"Rank\s+(\d+)\s+Expert\s+(\w+)\s+Step\s+(\d+)\s+\|(.+)"
)

_EXPERT_METRICS = ["ar_loss", "aux_loss", "total_loss", "route_entropy",
                   "porto_prob", "beijing_prob"]


def parse_expert_log(path: Path) -> dict[str, dict[str, list]]:
    """
    Parse a phase-2 expert training log.

    Returns a dict keyed by expert name, each value a dict of metric lists:
      { "porto":   {"step": [...], "ar_loss": [...], ...},
        "beijing": {"step": [...], "ar_loss": [...], ...} }
    """
    records: dict[str, dict[str, list]] = {}

    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                msg = obj.get("message", "")
            except json.JSONDecodeError:
                msg = line
            m = _EXPERT_STEP_RE.search(msg)
            if not m:
                continue
            _, expert, step, metrics_str = m.groups()
            if expert not in records:
                records[expert] = {"step": [], **{k: [] for k in _EXPERT_METRICS}}
            records[expert]["step"].append(int(step))
            for part in metrics_str.split("|"):
                k, _, v = part.strip().partition(":")
                k = k.strip()
                if k in _EXPERT_METRICS:
                    try:
                        records[expert][k].append(float(v.strip()))
                    except ValueError:
                        pass

    return records


def plot_experts(records: dict[str, dict[str, list]], out: Path, window: int) -> None:
    """Plot phase-2 expert training curves: ar_loss, routing probs, route entropy."""
    colors = {"porto": "steelblue", "beijing": "tomato"}
    experts = sorted(records.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Phase-2 Expert Training Curves", fontsize=14)

    max_steps = {"porto": 50_000, "beijing": 135_000}

    # ── Panel 1: ar_loss per expert ─────────────────────────────────────────
    ax = axes[0, 0]
    for exp in experts:
        d = records[exp]
        steps = np.array(d["step"])
        loss  = np.array(d["ar_loss"])
        c = colors.get(exp, "gray")
        ax.plot(steps, loss, color=c, alpha=0.2, linewidth=0.6)
        ax.plot(steps, smooth(loss, window), color=c, linewidth=1.5,
                label=f"{exp}  final={loss[-1]:.4f}")
        if exp in max_steps:
            ax.axvline(max_steps[exp], color=c, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("ar_loss")
    ax.set_title("AR Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: routing probabilities ──────────────────────────────────────
    ax = axes[0, 1]
    for exp in experts:
        d = records[exp]
        steps = np.array(d["step"])
        prob_key = f"{exp}_prob"
        if prob_key in d and d[prob_key]:
            prob = np.array(d[prob_key])
            c = colors.get(exp, "gray")
            ax.plot(steps, prob, color=c, alpha=0.2, linewidth=0.6)
            ax.plot(steps, smooth(prob, window), color=c, linewidth=1.5,
                    label=f"{exp}_prob  avg={prob.mean():.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel("routing probability")
    ax.set_ylim(0, 1)
    ax.set_title("Routing Probability (should approach 1.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: route entropy ───────────────────────────────────────────────
    ax = axes[1, 0]
    for exp in experts:
        d = records[exp]
        steps = np.array(d["step"])
        if d["route_entropy"]:
            ent = np.array(d["route_entropy"])
            c = colors.get(exp, "gray")
            ax.plot(steps, ent, color=c, alpha=0.2, linewidth=0.6)
            ax.plot(steps, smooth(ent, window), color=c, linewidth=1.5,
                    label=f"{exp}  final={ent[-1]:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("route entropy")
    ax.set_title("Route Entropy (lower = more confident routing)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: summary stats table ────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    rows = [["expert", "steps", "ar_loss\nstart→end", "route\nentropy"]]
    for exp in experts:
        d = records[exp]
        n = len(d["step"])
        if n == 0:
            continue
        ar = d["ar_loss"]
        ent = d["route_entropy"]
        target = max_steps.get(exp, "?")
        pct = f"{d['step'][-1]/target*100:.0f}%" if isinstance(target, int) else "?"
        rows.append([
            exp,
            f"{d['step'][-1]:,} / {target:,}\n({pct})",
            f"{ar[0]:.3f} → {ar[-1]:.3f}",
            f"{sum(ent)/len(ent):.4f}" if ent else "n/a",
        ])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    ax.set_title("Summary", fontsize=10)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def _is_expert_log(path: Path) -> bool:
    """Peek at first 50 non-empty lines to detect phase-2 expert log format."""
    checked = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line).get("message", "")
            except json.JSONDecodeError:
                msg = line
            if _EXPERT_STEP_RE.search(msg):
                return True
            if _STEP_RE.search(msg):
                return False
            checked += 1
            if checked >= 50:
                break
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from Nebius job log.")
    parser.add_argument("log", type=Path, help="Path to log file (JSONL or plain text)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <log>.png)")
    parser.add_argument("--smooth", type=int, default=100,
                        help="Smoothing window in steps (default: 100, set 1 to disable)")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"ERROR: file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    out = args.out or args.log.with_suffix(".png")
    print(f"Parsing {args.log} …")

    if _is_expert_log(args.log):
        records = parse_expert_log(args.log)
        if not records:
            print("ERROR: no expert step lines found in log.", file=sys.stderr)
            sys.exit(1)
        for exp, d in records.items():
            n = len(d["step"])
            print(f"  {exp}: {n} records  steps {d['step'][0]}–{d['step'][-1]}"
                  f"  ar_loss {d['ar_loss'][0]:.4f}→{d['ar_loss'][-1]:.4f}")
        plot_experts(records, out, window=args.smooth)
    else:
        records_p1 = parse_log(args.log)
        n = len(records_p1["step"])
        if n == 0:
            print("ERROR: no step lines found in log.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {n} step records  |  "
              f"steps {records_p1['step'][0]}–{records_p1['step'][-1]}  |  "
              f"final loss {records_p1['loss'][-1]:.4f}  |  "
              f"final AR ppl {records_p1['ppl'][-1]:.1f}")
        plot(records_p1, out, window=args.smooth)


if __name__ == "__main__":
    main()
