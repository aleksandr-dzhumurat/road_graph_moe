"""
Two-headed Transformer backbone pretraining (phase_01.md §3–6).

Modes
-----
Debug / overfit-tiny-batch  (phase_01_metrics.md §7.2 check 1)
    python scripts/backbone.py --debug
    Trains on 50 trips from porto shard_0000.  Both losses must reach near-zero.
    Catches model / loss wiring bugs before committing GPU time.

Full single-GPU run
    python scripts/backbone.py

Resume from checkpoint
    python scripts/backbone.py --resume data/checkpoints/ckpt_step_002000.pt

Multi-GPU via torchrun  (FSDP path, future)
    torchrun --nproc-per-node=N scripts/backbone.py
"""

import argparse
import datetime
import json
import math
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

REPO_ROOT  = Path(__file__).resolve().parent.parent
GOLD       = REPO_ROOT / "data" / "gold"
CKPT_DIR   = REPO_ROOT / "data" / "checkpoints"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Special token IDs (must match etl.py)
PAD, BOS, EOS, MASK = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _load_shard(shard: Path) -> tuple[Path, list[dict]]:
    table = pq.read_table(shard)
    # column-wise to_pylist() is ~10x faster than per-cell .as_py()
    cols = {col: table.column(col).to_pylist() for col in table.schema.names}
    rows = [{col: cols[col][i] for col in cols} for i in range(table.num_rows)]
    return shard, rows


class TrajectoryDataset(Dataset):
    """Loads tokenised trajectories from Gold Parquet shards."""

    def __init__(self, regions: list[str], max_samples: int | None = None,
                 data_dir: Path | None = None):
        base = data_dir if data_dir is not None else GOLD
        all_shards: list[Path] = []
        for region in regions:
            region_dir = base / region
            shards = sorted(region_dir.glob("shard_*.parquet"))
            if not shards:
                print(f"{ts()} [dataset] WARNING: no shards found in {region_dir}", flush=True)
                continue
            print(f"{ts()} [dataset] {region}: {len(shards)} shards found", flush=True)
            all_shards.extend(shards)

        rows: list[dict] = []
        n_threads = min(len(all_shards), os.cpu_count() or 1)
        print(f"{ts()} [dataset] loading {len(all_shards)} shards with {n_threads} threads", flush=True)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {pool.submit(_load_shard, s): s for s in all_shards}
            for future in as_completed(futures):
                shard, shard_rows = future.result()
                rows.extend(shard_rows)
                label = f"{shard.parent.name}/{shard.name}"
                print(f"{ts()} [dataset] {label}: {len(shard_rows)} rows, total={len(rows):,}", flush=True)

        if max_samples:
            rows = rows[:max_samples]
        self.rows = rows
        print(f"{ts()} [dataset] Loaded {len(self.rows):,} trajectories from {regions}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad (and truncate) sequences to the same length within a batch."""
    MAX_LEN = CFG["tokenizer"]["max_seq_len"]  # hard cap from config (1024)
    max_len = min(max(len(r["h3_tokens"]) for r in batch), MAX_LEN)

    tok_batch, dt_batch, mod_batch, dow_batch = [], [], [], []
    has_seg_ids = "seg_ids" in batch[0]
    seg_batch = [] if has_seg_ids else None

    for r in batch:
        toks = r["h3_tokens"][:MAX_LEN]
        dts  = r["dt_buckets"][:MAX_LEN]
        mods = r["min_of_day"][:MAX_LEN]
        dows = r["day_of_week"][:MAX_LEN]
        L = len(toks)
        pad = max_len - L
        tok_batch.append(toks + [PAD] * pad)
        dt_batch.append(dts  + [0]   * pad)
        mod_batch.append(mods + [0]   * pad)
        dow_batch.append(dows + [0]   * pad)
        if seg_batch is not None:
            segs = r["seg_ids"][:MAX_LEN]
            seg_batch.append(segs + [0] * pad)

    result = {
        "tok": torch.tensor(tok_batch, dtype=torch.long),
        "dt":  torch.tensor(dt_batch,  dtype=torch.long),
        "mod": torch.tensor(mod_batch, dtype=torch.long),
        "dow": torch.tensor(dow_batch, dtype=torch.long),
    }
    if seg_batch is not None:
        result["seg_ids"] = torch.tensor(seg_batch, dtype=torch.long)
    return result


# ---------------------------------------------------------------------------
# Model  (phase_01.md §3)
# ---------------------------------------------------------------------------
class STEmbedding(nn.Module):
    """Spatial token + decomposed temporal embeddings, summed."""
    def __init__(self, vocab_size: int, d_model: int, n_dt_buckets: int):
        super().__init__()
        self.tok        = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.dt         = nn.Embedding(n_dt_buckets, d_model)
        self.min_of_day = nn.Embedding(1440, d_model)
        self.day_of_week= nn.Embedding(7, d_model)

    def forward(self, tok, dt, mod, dow):
        return self.tok(tok) + self.dt(dt) + self.min_of_day(mod) + self.day_of_week(dow)


class TimeAwareBlock(nn.Module):
    """Pre-norm Transformer block with Flash Attention via is_causal.
    Passing an explicit float attn_mask to SDPA disables Flash Attention and
    still materialises [B, H, L, L] in the backward pass — same OOM as MHA.
    Using is_causal=True keeps memory O(L) for both forward and backward.
    Padding is handled by ignore_index=-100 in the loss, not by key_padding_mask."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.ln1        = nn.LayerNorm(d_model)
        self.qkv        = nn.Linear(d_model, 3 * d_model)
        self.proj       = nn.Linear(d_model, d_model)
        self.ln2        = nn.LayerNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop       = nn.Dropout(dropout)
        self._attn_drop = dropout

    def forward(self, x, is_causal: bool = False):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)          # each [B, H, L, d_head]
        a = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,          # triggers Flash Attention; O(L) memory
            dropout_p=self._attn_drop if self.training else 0.0,
        )
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class TrajectoryBackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, n_dt_buckets: int = 64,
                 max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed  = STEmbedding(vocab_size, d_model, n_dt_buckets)
        self.pos    = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TimeAwareBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f    = nn.LayerNorm(d_model)
        self.mtm_head = nn.Linear(d_model, vocab_size)
        self.ar_head  = nn.Linear(d_model, vocab_size)
        # weight tying: both heads share the same projection matrix
        self.ar_head.weight = self.mtm_head.weight

    def trunk(self, tok, dt, mod, dow, is_causal: bool = False):
        B, L = tok.shape
        pos = torch.arange(L, device=tok.device).unsqueeze(0).expand(B, L)
        x = self.embed(tok, dt, mod, dow) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, is_causal=is_causal)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Objective  (phase_01.md §4)
# ---------------------------------------------------------------------------

def apply_span_masking(tok: torch.Tensor, mask_id: int = MASK,
                       point_rate: float = 0.15, span_len: int = 4,
                       pad_id: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT-style 15% point masking + short span extension."""
    labels = tok.clone()
    rand   = torch.rand_like(tok, dtype=torch.float)
    masked = (rand < point_rate) & (tok != pad_id)
    for off in range(1, span_len):
        masked[:, off:] |= masked[:, :-off]
    masked &= (tok != pad_id)
    labels[~masked] = -100
    inp = tok.clone()
    inp[masked] = mask_id
    return inp, labels


def two_headed_loss(model: TrajectoryBackbone, batch: dict,
                    lambda_ar: float = 0.6) -> tuple[torch.Tensor, dict]:
    tok, dt, mod, dow = batch["tok"], batch["dt"], batch["mod"], batch["dow"]
    B, L = tok.shape

    # MTM pass — bidirectional; padding excluded from loss via ignore_index=-100
    masked_inp, mtm_labels = apply_span_masking(tok)
    h_mtm      = model.trunk(masked_inp, dt, mod, dow, is_causal=False)
    mtm_logits = model.mtm_head(h_mtm)
    mtm_loss   = F.cross_entropy(mtm_logits.view(-1, mtm_logits.size(-1)),
                                  mtm_labels.view(-1), ignore_index=-100)

    # AR pass — causal; Flash Attention via is_causal=True (O(L) memory)
    h_ar      = model.trunk(tok, dt, mod, dow, is_causal=True)
    ar_logits = model.ar_head(h_ar)[:, :-1, :]
    ar_labels = tok[:, 1:].clone()
    ar_labels[ar_labels == PAD] = -100
    ar_loss = F.cross_entropy(ar_logits.reshape(-1, ar_logits.size(-1)),
                               ar_labels.reshape(-1), ignore_index=-100)

    loss = (1 - lambda_ar) * mtm_loss + lambda_ar * ar_loss
    return loss, {"mtm": mtm_loss.item(), "ar": ar_loss.item()}


# ---------------------------------------------------------------------------
# LR schedule — linear warmup + cosine decay  (phase_01.md §6)
# ---------------------------------------------------------------------------
def lr_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, optim: torch.optim.Optimizer,
                    step: int, tag: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"ckpt_{tag}.pt"
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step}, path)
    print(f"{ts()} [ckpt] Saved → {path}")


def load_checkpoint(path: Path, model: nn.Module,
                    optim: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    print(f"{ts()} [ckpt] Resumed from {path} at step {ckpt['step']}")
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Init-loss assertion  (phase_01_metrics.md §7.2 check 2)
# ---------------------------------------------------------------------------
def assert_init_loss(ar_loss: float, vocab_size: int, tol: float = 1.5) -> None:
    expected = math.log(vocab_size)
    if abs(ar_loss - expected) > tol:
        print(
            f"{ts()} [WARNING] AR loss at init is {ar_loss:.3f}, expected ≈ {expected:.3f} "
            f"(ln {vocab_size}). Check masking, ignore_index, and label shift."
        )
    else:
        print(f"{ts()} [check] Init AR loss {ar_loss:.3f} ≈ ln({vocab_size})={expected:.3f} ✓")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} [train] Device: {device}")

    tok_cfg = CFG["tokenizer"]
    vocab_size = tok_cfg["vocab_size"] if "vocab_size" in tok_cfg else 30_004

    # ── Debug overrides (overfit-tiny-batch check) ──────────────────────────
    if args.debug:
        print(f"\n{ts()}[debug] Overfit-tiny-batch mode: 50 trips, small model, no dropout")
        model_kwargs = dict(
            vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4,
            n_dt_buckets=tok_cfg["n_dt_buckets"], max_seq_len=tok_cfg["max_seq_len"],
            dropout=0.0,
        )
        train_kwargs = dict(
            regions=["porto"], max_samples=50,
            batch_size=16, total_steps=300, warmup=10,
            base_lr=1e-3, ckpt_every=0, log_every=10, lambda_ar=0.6,
        )
    else:
        model_kwargs = dict(
            vocab_size=vocab_size,
            d_model=tok_cfg.get("d_model", 768),
            n_layers=tok_cfg.get("n_layers", 12),
            n_heads=tok_cfg.get("n_heads", 12),
            n_dt_buckets=tok_cfg["n_dt_buckets"],
            max_seq_len=tok_cfg["max_seq_len"],
            dropout=0.1,
        )
        train_kwargs = dict(
            regions=["porto", "beijing"], max_samples=None,
            batch_size=args.batch, total_steps=args.steps, warmup=5_000,
            base_lr=3e-4, ckpt_every=2_000, log_every=50, lambda_ar=0.6,
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    print(f"{ts()} [dataset] Loading regions={train_kwargs['regions']} max_samples={train_kwargs['max_samples']}")
    dataset = TrajectoryDataset(
        regions=train_kwargs["regions"],
        max_samples=train_kwargs["max_samples"],
    )
    n_workers = 0 if args.debug else min(8, os.cpu_count() or 1)
    loader = DataLoader(
        dataset,
        batch_size=train_kwargs["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=n_workers > 0,
        drop_last=True,
    )
    print(f"{ts()} [dataset] DataLoader ready: batch={train_kwargs['batch_size']} workers={n_workers}")

    # ── Model + optimiser ───────────────────────────────────────────────────
    model = TrajectoryBackbone(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{ts()} [model] Parameters: {n_params / 1e6:.1f} M")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_kwargs["base_lr"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), model, optim)

    # preemption hook — flush checkpoint on SIGTERM (spot termination notice)
    state = {"step": start_step}
    def _sigterm_handler(signum, frame):
        print(f"\n{ts()}[preempt] SIGTERM received — saving checkpoint")
        save_checkpoint(model, optim, state["step"], tag=f"preempt_step_{state['step']:06d}")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Training loop ───────────────────────────────────────────────────────
    total   = train_kwargs["total_steps"]
    warmup  = train_kwargs["warmup"]
    base_lr = train_kwargs["base_lr"]

    model.train()
    step = start_step

    print(f"\n{ts()}[train] Starting at step {step}, target {total} steps\n")

    while step < total:
        for batch in loader:
            if step >= total:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # LR schedule
            lr = lr_schedule(step, warmup, total, base_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            # Forward + backward
            use_amp = device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, parts = two_headed_loss(model, batch, train_kwargs["lambda_ar"])

            # Init-loss check at step 0
            if step == start_step:
                assert_init_loss(parts["ar"], vocab_size)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optim.step()
            optim.zero_grad(set_to_none=True)

            state["step"] = step

            # Logging
            if step % train_kwargs["log_every"] == 0:
                ar_ppl = math.exp(min(parts["ar"], 20))
                print(
                    f"{ts()} step {step:6d} | loss {loss.item():.4f} | "
                    f"ar {parts['ar']:.4f} (ppl {ar_ppl:.1f}) | "
                    f"mtm {parts['mtm']:.4f} | "
                    f"grad {grad_norm:.3f} | lr {lr:.2e}",
                    flush=True,
                )

            # Periodic checkpoint
            if train_kwargs["ckpt_every"] and step > 0 and step % train_kwargs["ckpt_every"] == 0:
                save_checkpoint(model, optim, step, tag=f"step_{step:06d}")

            step += 1

    save_checkpoint(model, optim, step, tag="final")
    print(f"\n{ts()}[train] Done. Final checkpoint saved.")

    # Debug: confirm overfit succeeded
    if args.debug:
        print(f"\n{ts()}[debug] Final combined loss: {loss.item():.4f}")
        if loss.item() < 0.5:
            print(f"{ts()} [debug] PASS — model overfit the tiny batch as expected.")
        else:
            print(f"{ts()} [debug] FAIL — loss did not converge. Check model/loss wiring.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_png(cells: list[str], path: str = "hexes.png",
             color: str = "crimson", dpi: int = 150) -> str:
    """Render H3 cells as filled hexagons over an OSM basemap and save to PNG."""
    import contextily as ctx
    import h3 as h3lib
    fig, ax = plt.subplots(figsize=(8, 8))
    for cell in cells:
        boundary = h3lib.cell_to_boundary(cell)      # [(lat, lng), ...]
        xs = [lng for _, lng in boundary] + [boundary[0][1]]
        ys = [lat for lat, _ in boundary] + [boundary[0][0]]
        ax.fill(xs, ys, facecolor=color, edgecolor="black", alpha=0.5, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def inference() -> None:
    """Load ckpt_final.pt and run both AR and MTM inference on a sample trajectory.

    Sample trajectory: BOS + 8 spatial H3 tokens, recorded at 08:00 on a Tuesday.
    H3 token ids start at 4 (0=PAD, 1=BOS, 2=EOS, 3=MASK).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / "ckpt_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tok_cfg = CFG["tokenizer"]
    vocab_size = tok_cfg["vocab_size"]

    model = TrajectoryBackbone(
        vocab_size=vocab_size,
        d_model=tok_cfg.get("d_model", 768),
        n_layers=tok_cfg.get("n_layers", 12),
        n_heads=tok_cfg.get("n_heads", 12),
        n_dt_buckets=tok_cfg["n_dt_buckets"],
        max_seq_len=tok_cfg["max_seq_len"],
        dropout=0.0,
    ).to(device)

    print(f"{ts()} [inference] Started loading weights from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{ts()} [inference] Device      : {device}")
    print(f"{ts()} [inference] Checkpoint  : {ckpt_path}")
    print(f"{ts()} [inference] Train step  : {ckpt.get('step', '?')}")
    print(f"{ts()} [inference] Model size  : {n_params / 1e6:.1f} M parameters")
    print(f"{ts()} [inference] Vocab size  : {vocab_size}  (PAD=0 BOS=1 EOS=2 MASK=3)")

    # Real Porto trajectory: Boavista → Aliados, 08:00 Tuesday, 15-s GPS interval
    # Token IDs are computed with the same hash used in etl.py:h3_to_token
    import h3 as h3lib
    tok_cfg_full = CFG["tokenizer"]
    h3_res   = tok_cfg_full["h3_resolution"]       # 9
    subhash  = tok_cfg_full["subhash_vocab_size"]  # 30000

    def _latlng_to_token(lat: float, lng: float) -> int:
        cell   = h3lib.latlng_to_cell(lat, lng, h3_res) if hasattr(h3lib, "latlng_to_cell") \
                 else h3lib.geo_to_h3(lat, lng, h3_res)
        h3_int = int(cell, 16)
        return 4 + (h3_int ^ (h3_int >> 17)) % subhash

    # 8 GPS waypoints along Av. da Boavista → Praça da Liberdade, Porto
    waypoints = [
        (41.1579, -8.6291),  # Rotunda da Boavista
        (41.1570, -8.6265),
        (41.1558, -8.6238),
        (41.1545, -8.6210),
        (41.1532, -8.6183),
        (41.1518, -8.6155),
        (41.1504, -8.6128),
        (41.1495, -8.6108),  # Praça da Liberdade
    ]
    spatial_tokens = [_latlng_to_token(lat, lng) for lat, lng in waypoints]
    sample_tokens  = [BOS] + spatial_tokens
    L = len(sample_tokens)

    _SPECIAL = {0: "PAD", 1: "BOS", 2: "EOS", 3: "MASK"}
    def _label(tid: int) -> str:
        return _SPECIAL.get(tid, f"h3:{tid}")

    tok = torch.tensor([sample_tokens], dtype=torch.long, device=device)
    # Porto records GPS at 15-second intervals → dt_bucket(15) = int(log2(15))+1 = 4
    dt_vals = [0] + [4] * (L - 1)
    dt  = torch.tensor([dt_vals], dtype=torch.long, device=device)
    mod = torch.full((1, L), 480, dtype=torch.long, device=device)  # 08:00 = 480 min
    dow = torch.full((1, L), 1,   dtype=torch.long, device=device)  # Tuesday

    print(f"\n{ts()} [inference] Porto trajectory (len={L}, time=08:00 Tue, dt_bucket=4)")
    print("  route   : Rotunda da Boavista → Praça da Liberdade")
    for i, ((lat, lng), tid) in enumerate(zip(waypoints, spatial_tokens)):
        print(f"  [{i+1}] ({lat:.4f}, {lng:.4f})  → token_id={tid}")

    with torch.no_grad():
        # AR: predict the next spatial token after the sequence
        h_ar        = model.trunk(tok, dt, mod, dow, is_causal=True)
        ar_logits   = model.ar_head(h_ar)           # [1, L, vocab_size]
        last_logits = ar_logits[0, -1, :]           # logits for the next position
        ar_probs    = F.softmax(last_logits, dim=-1)
        topk        = torch.topk(last_logits, 5)
        topk_ids    = topk.indices.tolist()
        topk_probs  = ar_probs[topk.indices].tolist()
        greedy_id   = last_logits.argmax().item()
        entropy     = -(ar_probs * ar_probs.log().clamp(min=-1e9)).sum().item()

        print(f"\n{ts()} [inference] AR head — next token prediction (causal)")
        print(f"  greedy prediction : {_label(greedy_id)} (id={greedy_id})")
        print(f"  distribution entropy: {entropy:.2f} nats  (max={math.log(vocab_size):.2f})")
        print(f"  {'rank':<6} {'id':<8} {'label':<12} {'prob':>8}")
        print(f"  {'-'*38}")
        for rank, (tid, prob) in enumerate(zip(topk_ids, topk_probs), 1):
            print(f"  {rank:<6} {tid:<8} {_label(tid):<12} {prob:>8.4%}")

        # MTM: mask position 4 and reconstruct the original token
        masked_tok  = tok.clone()
        true_token  = sample_tokens[4]
        masked_tok[0, 4] = MASK
        h_mtm       = model.trunk(masked_tok, dt, mod, dow, is_causal=False)
        mtm_logits  = model.mtm_head(h_mtm)         # [1, L, vocab_size]
        pos_logits  = mtm_logits[0, 4, :]
        mtm_probs   = F.softmax(pos_logits, dim=-1)
        pred_token  = pos_logits.argmax().item()
        true_rank   = (mtm_probs > mtm_probs[true_token]).sum().item() + 1
        topk_mtm    = torch.topk(pos_logits, 5)
        topk_mtm_ids   = topk_mtm.indices.tolist()
        topk_mtm_probs = mtm_probs[topk_mtm.indices].tolist()

        print(f"\n{ts()} [inference] MTM head — reconstruct masked position 4 (bidirectional)")
        print(f"  context  : {[_label(t) for t in masked_tok[0].tolist()]}")
        print(f"  true token      : {_label(true_token)} (id={true_token})")
        print(f"  predicted token : {_label(pred_token)} (id={pred_token})")
        print(f"  true token rank : {true_rank} / {vocab_size}")
        print(f"  true token prob : {mtm_probs[true_token].item():.4%}")
        print(f"  {'rank':<6} {'id':<8} {'label':<12} {'prob':>8}")
        print(f"  {'-'*38}")
        for rank, (tid, prob) in enumerate(zip(topk_mtm_ids, topk_mtm_probs), 1):
            marker = " <-- true" if tid == true_token else ""
            print(f"  {rank:<6} {tid:<8} {_label(tid):<12} {prob:>8.4%}{marker}")

    # Visualize the trajectory hexes
    h3_cells = [
        (h3lib.latlng_to_cell(lat, lng, h3_res) if hasattr(h3lib, "latlng_to_cell")
         else h3lib.geo_to_h3(lat, lng, h3_res))
        for lat, lng in waypoints
    ]
    png_path = str(REPO_ROOT / "data" / "inference_trajectory.png")
    save_png(h3_cells, path=png_path, color="steelblue")
    print(f"\n{ts()} [inference] Trajectory map saved → {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 backbone pretraining.")
    parser.add_argument("--debug",  action="store_true",
                        help="Overfit-tiny-batch mode: 50 trips, small model, 300 steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch",  type=int, default=32,
                        help="Batch size (default: 32, ignored in --debug)")
    parser.add_argument("--steps",  type=int, default=300_000,
                        help="Total training steps (default: 300000, ignored in --debug)")
    parser.add_argument("--infer",  action="store_true",
                        help="Run inference on a sample trajectory using ckpt_final.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.infer:
        inference()
    else:
        train(args)
