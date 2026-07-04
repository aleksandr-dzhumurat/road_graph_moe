---
license: mit
language:
  - en
tags:
  - trajectory
  - geospatial
  - anomaly-detection
  - h3
  - transformer
datasets:
  - porto-taxi
  - t-drive
---

# Geospatial Trajectory Foundation Model

A 132 M-parameter Transformer pretrained on 1.68 M public taxi GPS trajectories
(Porto + Beijing) using two self-supervised objectives:

- **Masked Trajectory Modeling (MTM)** — bidirectional, for trajectory representations
- **Autoregressive (AR) next-location prediction** — causal, for anomaly scoring via perplexity

GPS coordinates are tokenized into H3 hexagonal-grid cells at resolution 9 (~0.1 km²).
After pretraining, the AR head assigns a calibrated perplexity score to any trajectory —
detours, unusual routes, and driver fraud surface as high-perplexity sequences.

## Architecture

| Parameter | Value |
|---|---|
| Layers | 12 Transformer blocks |
| d_model | 768 |
| Attention heads | 12 |
| Vocab size | 30,004 (30k spatial H3 sub-hashes + PAD/BOS/EOS/MASK) |
| Max sequence length | 512 tokens |
| Parameters | 132.8 M |
| Attention | Flash Attention (`is_causal=True`) |

Embeddings: spatial H3 token + log-spaced Δt bucket + minute-of-day + day-of-week.
The MTM and AR heads share the same weight-tied projection matrix.

## Training

| Metric | Value |
|---|---|
| Training steps | 300,000 |
| AR perplexity: initial → final | 34,754 → **2.1** |
| Final combined loss | 0.963 |
| Hardware | 1 × NVIDIA H100-SXM 80 GB (preemptible) |
| Wall time | ~12 h 43 min |
| Cost | ~$28 |

## Usage

Download `backbone.py` and `config.json` from this repo alongside the checkpoint, then:

```python
import json
import sys
import torch

sys.path.insert(0, ".")          # backbone.py must be in the working directory
from backbone import TrajectoryBackbone

with open("config.json") as f:
    cfg = json.load(f)["tokenizer"]

model = TrajectoryBackbone(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    max_seq_len=cfg["max_seq_len"],
)

ckpt = torch.load("ckpt_final.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
```