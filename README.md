# Geospatial Trajectory Foundation Model

A 132 M-parameter Transformer pretrained on 1.68 M public taxi GPS trajectories using a
two-headed self-supervised objective: **Masked Trajectory Modeling (MTM)** for representation
quality and **Autoregressive (AR) next-location prediction** for anomaly scoring.
Trained end-to-end on **Nebius Serverless AI Jobs** — Phase 1 on a preemptible H100-SXM, Phase 2 on a 2×L40S node for parallel expert training.

**Model on Hugging Face:** [aleksandr-dzhumurat/geospatial-trajectory-transformer](https://huggingface.co/aleksandr-dzhumurat/geospatial-trajectory-transformer)


---

## Why Nebius Serverless

This project uses [Nebius Serverless AI Jobs](https://nebius.com/services/serverless-gpu) throughout.

- **Remote image builds** — submits the Docker build as a Nebius job on a remote VM with the correct linux/amd64 arch, then pushes directly to the Nebius Container Registry. No local daemon, no cross-compilation, no slow layer uploads.

- **S3-mounted code — deploy without rebuilding** — scripts are mounted into the container from S3. A code change is live after a single sync command; no heavy Docker image rebuild needed.

- **S3 for all artifacts** — checkpoints are written to a read-write S3 mount and survive job restarts. Logs are fetched via the Nebius CLI — no SSH required.

- **Preemptible instances** — both training jobs use `--preemptible` to cut GPU-hour cost (~$28 vs ~$84 on-demand for Phase 1). A `SIGTERM` handler in the training loop saves a checkpoint on eviction so the next run resumes cleanly.

---

## What it does

GPS trajectories are tokenized into H3 hexagonal-grid cells (resolution 9, ~0.1 km²) and fed
to a shared Transformer trunk. Two heads are trained jointly:

| Head | Attention | Task | Used for |
|---|---|---|---|
| MTM | Bidirectional | Predict masked H3 tokens | Trajectory representations |
| AR | Causal | Predict next H3 token | Anomaly scoring via perplexity |

After pretraining, the AR head gives a calibrated perplexity score for any trajectory —
detours, unusual routes, and driver fraud surface as high-perplexity sequences.


The map below shows model inference on a real Porto route (Boavista → Praça da Liberdade).
Each hexagon is an H3 cell at resolution 9 (~0.1 km²). Given the first 8 cells, the AR head
predicts the next most-likely cell; the MTM head reconstructs a masked position from context.

![Inference — Porto trajectory](docs/img/inference_trajectory.png)


---

## Architecture

**Phase 1 — Backbone** (`backbone.py`)

- **Model:** 12-layer Transformer, d_model=768, 12 heads, Flash Attention (`is_causal=True`)
- **Embedding:** spatial H3 token + dt_bucket (log-spaced Δt) + minute-of-day + day-of-week
- **Vocab:** 30,004 tokens (30,000 spatial sub-hash buckets + PAD/BOS/EOS/MASK)
- **Max sequence:** 512 tokens
- **Parameters:** 132.8 M
- **Weight tying:** MTM head and AR head share the same projection matrix

The H3 sub-hash maps any geographic cell to the same vocab regardless of city — keeping the embedding table fixed-size and enabling cross-city pretraining without per-city vocabularies.

**Phase 2 — Geographic Experts** (`experts.py`)

City-specific adapters that sit on top of the frozen backbone and refine its hidden states using road graph structure, producing predictions in the same H3 token space:

```
input tokens
  → backbone.trunk()        # frozen Phase 1 weights, city-agnostic
  → h  [B, L, 768]
  → GeographicExpert (GAT)  # city-specific, trainable
  → fused_h  [B, L, 768]    # road context fused in
  → tok_head @ tok_emb.T    # weight-tied to backbone embedding
  → logits  [vocab_size]    # same output space as backbone
```

- **GeographicExpert:** Graph Attention Network over the city's OSM road graph; fuses road entity features (roundabouts, signals, motorways, bridges) into backbone hidden states
- **GeographicRouter:** top-1 routing with geographic prior — selects Porto or Beijing expert per trajectory
- **Weight tying:** expert `tok_head` shares the backbone's token embedding matrix — same vocab, no extra parameters for the output projection

---

## Datasets

Both datasets are publicly available and license-clean.

| Dataset | Region | Trajectories | Sampling | Source |
|---|---|---|---|---|
| Porto Taxi (ECML/PKDD 2015) | Porto, Portugal | 1,605,409 | 15 s | Kaggle |
| T-Drive | Beijing, China | 79,042 | ~60 s | Microsoft Research |
| **Total** | | **1,684,451** | | |

Download instructions: see `scripts/download_datasets.py`.

---

## Hardware configuration

**Phase 1 — backbone pretraining**

| Parameter | Value |
|---|---|
| Instance | `gpu-h100-sxm` / `1gpu-16vcpu-200gb` |
| GPU | 1 × NVIDIA H100-SXM 80 GB |
| Compute mode | Preemptible (spot) |
| Storage | S3 mount — data read-only, checkpoints read-write |

GPU memory breakdown at batch=64, seq_len=512:

| Component | Memory |
|---|---|
| Model weights (bf16) | ~0.3 GB |
| AdamW optimizer (fp32 × 2) | ~1.1 GB |
| Activations (12 layers × 2 passes) | ~10 GB |
| Logits × 2 heads | ~2 GB |
| **Total used** | **~43 GB / 80 GB (54%)** |

**Phase 2 — geographic expert training**

| Parameter | Value |
|---|---|
| Instance | `gpu-l40s-d` / `2gpu-64vcpu-384gb` |
| GPUs | 2 × NVIDIA L40S 48 GB |
| Compute mode | Preemptible (spot) |
| Storage | S3 mounts — gold shards + road graphs read-only, checkpoints read-write |

Two experts train in parallel via `torchrun --nproc-per-node=2`: Rank 0 → Porto expert, Rank 1 → Beijing expert. A single GPU is not sufficient — Beijing's GAT operates over a 163,287-node road graph and requires its own dedicated GPU throughout training.

---

## Expected outputs and training results

| Metric | Value |
|---|---|
| Training steps | 300,000 |
| Wall time | ~12 h 43 min |
| Cost (preemptible H100) | ~$28 |
| AR perplexity: initial → final | 34,754 → **2.1** |
| Final combined loss | 0.963 |
| Checkpoints saved | 150 (every 2,000 steps + final) |

![Training curves](docs/img/train_curves.png)

**Key training dynamics:**
- **Fast convergence phase (0–10k steps):** All losses drop sharply as the model learns token frequencies
- **Structural learning phase (10k–300k steps):** Slow, steady improvement with no plateau — more compute would help
- **Final AR perplexity of 2.1:** Model chooses between ~2 equally-plausible next hexes, matching road intersection structure
- **MTM consistently harder:** Masked reconstruction loss runs 0.5–1.0 nats above autoregressive loss throughout training
- **Stable gradients:** Gradient norm stays below clip threshold (1.0) except for occasional spikes from long Beijing sequences

Find more details in [training report](./train_report_summary.md)

---

## Setup and reproduction

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- AWS CLI
- A Nebius account with Serverless AI Jobs access
- SSH key pair for VM access: `ssh-keygen -t ed25519 -f ~/.ssh/nebius -C "nebius-build" -N ""`

### Local setup

```bash
make setup          # create venv, install requirements
source .venv/bin/activate
```

### Configure AWS CLI for Nebius S3

Create the AWS CLI profile for accessing Nebius Object Storage:

```bash
# Set your Nebius service account credentials (found in Nebius Console > IAM > Service accounts)
export ACCESS_KEY_AWS_ID="your_access_key_id"
export SECRET_ACCESS_KEY="your_secret_access_key"

# Configure AWS CLI with nebius profile
aws configure set aws_access_key_id $ACCESS_KEY_AWS_ID --profile nebius
aws configure set aws_secret_access_key $SECRET_ACCESS_KEY --profile nebius

# Test the connection
aws s3 ls --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

**Note**: You can find/regenerate your access key in the Nebius console under IAM → Service accounts → your account → Access keys. The secret is only shown once at creation — if you didn't save it, delete and recreate the access key.

### Create Nebius Container Registry

Create a container registry to store the Docker image:

```bash
# Create a new container registry (replace with your preferred name)
nebius container registry create \
  --parent-id <your-project-id> \
  --name trajectory-registry

# Note the registry ID from the output and update REGISTRY in Makefile:
# REGISTRY = cr.eu-north1.nebius.cloud/<your-registry-id>
```

**Important**: Update the `REGISTRY` variable in the Makefile with your actual registry ID before running `make build-remote`.


### Debug / overfit check (local CPU)

```bash
python scripts/backbone.py --debug   # 50 trips, 300 steps — both losses must converge
```

### Data preparation

The ETL pipeline runs in four stages (`scripts/etl.py --stage <stage>`):

| Stage | Output | Description |
|---|---|---|
| `bronze` | raw parquet | GPS coordinate extraction from downloaded files |
| `silver` | filtered parquet | Trajectory segmentation, filtering, validation |
| `gold` | sharded parquet | H3 tokenization, temporal bucketing, sharding |
| `road-gold` | enhanced parquet + `road_graph.pt` | OSM road graph download, map-matching, `seg_id` per token |

```bash
# Download raw datasets (Porto Taxi + T-Drive Beijing)
python scripts/download_datasets.py

# Phase 1: Bronze → Silver → Gold
python scripts/etl.py --stage gold --regions porto beijing

# Upload gold shards to S3 (Phase 1 training data)
aws s3 cp data/gold s3://geo-trajectories/ --recursive \
  --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud

# Phase 2: Road-enhanced Gold (requires osmnx, geopandas, shapely, networkx)
pip install osmnx geopandas shapely networkx scipy
python scripts/etl.py --stage road-gold --regions porto beijing

# Validate the output before uploading
python scripts/etl.py --stage check-road-gold --regions porto beijing

# Upload road-enhanced data to S3 (Phase 2 training data)
make upload-road-gold
```

### Run training on Nebius (full reproduction)

**Phase 1 — backbone pretraining**

```bash
make build-remote       # build Docker image on Nebius VM and push to registry
make upload-code        # sync scripts/ → s3://geo-trajectories-code/ (deploy without rebuild)
make create-s3-secret   # store S3 credentials as a Nebius secret
make run-cloud          # submit a preemptible H100-SXM job (~$28, ~12h 43m)
```

Reads from `s3://geo-trajectories`, writes checkpoints to `s3://geo-trajectories-checkpoints`.
On SIGTERM (spot preemption) the current checkpoint is flushed automatically.

**Phase 2 — geographic expert training**

```bash
make upload-road-gold           # push road-enhanced gold to s3://geo-trajectories-road-enhanced
make run-cloud-phase02-multi    # submit a preemptible 2×L40S job (torchrun, both experts)
```

Both experts train simultaneously: Porto (50k steps, ~30 min) and Beijing (135k steps, ~3h 30m).
The S3 volume `s3://geo-trajectories-road-enhanced` is FUSE-mounted at
`/app/data/road_enhanced_gold` before the container starts — no data staging code required.

**Monitor a running job**

```bash
nebius ai job list --profile nebius   # find job ID

# Full log from job start (default cap is 100 lines — use --tail to raise it)
nebius ai job logs <job-id> --since 2026-06-07 --tail 100000 --profile nebius \
  > data/train_log_full.txt

# Plot training curves (auto-detects Phase 1 / Phase 2 log format)
uv run python scripts/train_curve.py data/train_log_full.txt
```


### Download checkpoints

```bash
make download-ckpts  # fetches ckpt_final.pt + latest expert_porto_*.pt + latest expert_beijing_*.pt
```

`make download-ckpt` (singular) fetches only the backbone `ckpt_final.pt` if you don't need the experts.

### Run inference on a sample Porto trajectory

**Backbone only** (AR next-token + MTM masked reconstruction):

```bash
make inference
```

Saves a map visualization to `data/inference_trajectory.png`.

**With geographic expert** (backbone + GAT road-context fusion):

```bash
python scripts/experts.py --infer --expert porto
```

Prints geographic routing probabilities and top-5 AR next-token predictions from the expert-fused head.

---

## Repository structure

```
scripts/
  etl.py              — Bronze → Silver → Gold → Road-enhanced Gold (H3 + OSM map-matching)
  backbone.py         — Phase 1: Transformer model, training loop, inference
  experts.py          — Phase 2: Geographic GAT experts + multi-tenant scheduler
  train_curve.py      — Plot training curves (auto-detects Phase 1 / Phase 2 log format)
  download_datasets.py
  config.json         — vocab size, H3 resolution, model dimensions
data/
  gold/               — tokenized trajectory shards (Porto + Beijing)
  road_enhanced_gold/ — gold shards + road_graph.pt with OSM seg_ids (Phase 2)
    porto/road_graph.pt   — PyG graph, 5,038 nodes
    beijing/road_graph.pt — PyG graph, 163,287 nodes
  checkpoints/        — local checkpoint cache
docs/
  nebius_cli.md       — Nebius CLI reference: S3, jobs, log fetching, cost estimation
  implementation/     — phase-by-phase design docs
  train_report_summary.md
Makefile
requirements.txt
requirements-train.txt  — minimal deps for training container (no OSM/GDAL)
Dockerfile
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Backbone pretraining | **Done** | Two-headed Transformer on Porto + Beijing; AR perplexity 34,754 → 2.1 |
| 2 — Geographic experts | **Done** | Per-city GAT adapters over OSM road graphs; Porto (ar_loss 0.61) and Beijing trained on 2×L40S via torchrun |
| 3 — New city road graphs | Planned | Add experts for new cities (e.g. London, NYC) by running `etl.py --stage road-gold` on any OSM region and training a new `GeographicExpert` without retraining the backbone |
| 4 — Anomaly head | Planned | Perplexity scorer with threshold calibration; PR-AUC vs. GM-VSAE baseline |
| 5 — Serving | Planned | Batch scoring endpoint; trips-per-dollar benchmark |

---

## License

MIT
