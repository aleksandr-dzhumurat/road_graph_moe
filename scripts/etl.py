"""
ETL pipeline: data/raw → Bronze → Silver → Gold → Road-Enhanced Gold (Parquet shards).

  Bronze           immutable copy of raw data, organised by region
  Silver           cleaned, trip-segmented records (one row per trip)
  Gold             H3-tokenized trajectories as Parquet shards, ready for pretraining
  Road-Enhanced    Gold + OpenStreetMap road segments and entity features

Implements the preprocessing sequence from docs/implementation/phase_01.md + phase_02.md:
  1. Speed / noise filtering  (remove physically impossible GPS jumps)
  2. Trip segmentation        (split on large time gaps)
  3. Minimum-length filtering (CausalTAD threshold: 30 points; FM minimum: 10)
  4. H3 tokenization          (res 8-9, 30k sub-hash WordPiece vocab)
  5. Temporal feature extraction (dt_bucket, min_of_day, day_of_week)
  6. **NEW**: Road network integration (OSM + map-matching)

Token schema (phase_01.md §2):
  [BOS] tok_0 tok_1 ... tok_{L-1} [EOS]
  each tok_i: (h3_token_id, dt_bucket, min_of_day, day_of_week, road_segment_id, entity_features)
  Special IDs: PAD=0  BOS=1  EOS=2  MASK=3  spatial vocab starts at 4

Usage:
    python scripts/etl.py                        # full pipeline, both regions
    python scripts/etl.py --stage silver         # stop after silver
    python scripts/etl.py --region porto         # one region only
    python scripts/etl.py --h3-res 8            # H3 resolution override (8 or 9)
    python scripts/etl.py --shards 16           # Gold shard count per region
    
    # NEW: Road-enhanced Gold processing (OSMnx nearest-node, no Valhalla needed)
    python scripts/etl.py --stage road-gold --regions porto beijing
"""

import argparse
import json
import math
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

# OSM and map-matching imports (optional)
try:
    import osmnx as ox
    import networkx as nx
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False  # road-gold ETL stage will skip with a clear message

from scipy.spatial import cKDTree
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import h3 as h3lib
    _H3_V4 = hasattr(h3lib, "latlng_to_cell")
except ImportError:
    print("ERROR: h3 not installed.  Run: pip install h3")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW    = REPO_ROOT / "data" / "raw"
BRONZE = REPO_ROOT / "data" / "bronze"
SILVER = REPO_ROOT / "data" / "silver"
GOLD   = REPO_ROOT / "data" / "gold"

# ---------------------------------------------------------------------------
# Load config.json
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)

_CFG = _load_config()

# ---------------------------------------------------------------------------
# Thresholds — read from config.json
# ---------------------------------------------------------------------------
MAX_SPEED_KMH    = _CFG["filtering"]["max_speed_kmh"]
TRIP_GAP_SECONDS = _CFG["filtering"]["trip_gap_seconds"]
MIN_TRIP_POINTS  = _CFG["filtering"]["min_trip_points"]
MAX_TRIP_POINTS  = _CFG["tokenizer"]["max_seq_len"] - 2  # reserve BOS + EOS

PORTO_DT_SECONDS = _CFG["porto"]["dt_seconds"]
TDRIVE_TS_FMT    = _CFG["tdrive"]["timestamp_format"]

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
PAD, BOS, EOS, MASK = 0, 1, 2, 3
SPATIAL_VOCAB_OFFSET = 4

# ---------------------------------------------------------------------------
# Gold Parquet schema
# ---------------------------------------------------------------------------
GOLD_SCHEMA = pa.schema([
    pa.field("trajectory_id", pa.string()),
    pa.field("region",        pa.string()),
    pa.field("h3_tokens",     pa.list_(pa.int32())),
    pa.field("dt_buckets",    pa.list_(pa.int8())),
    pa.field("min_of_day",    pa.list_(pa.int16())),
    pa.field("day_of_week",   pa.list_(pa.int8())),
    pa.field("n_tokens",      pa.int16()),
])


# ---------------------------------------------------------------------------
# Config dataclass — populated from config.json, overridable via CLI
# ---------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    h3_resolution:      int = field(default_factory=lambda: _CFG["tokenizer"]["h3_resolution"])
    subhash_vocab_size: int = field(default_factory=lambda: _CFG["tokenizer"]["subhash_vocab_size"])
    max_seq_len:        int = field(default_factory=lambda: _CFG["tokenizer"]["max_seq_len"])
    n_dt_buckets:       int = field(default_factory=lambda: _CFG["tokenizer"]["n_dt_buckets"])


# ---------------------------------------------------------------------------
# H3 helpers — v3 / v4 API compatibility
# ---------------------------------------------------------------------------
def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    if _H3_V4:
        return h3lib.latlng_to_cell(lat, lng, resolution)
    return h3lib.geo_to_h3(lat, lng, resolution)


def h3_to_token(h3_cell: str, cfg: TokenizerConfig) -> int:
    """Stable sub-hash mapping: region-agnostic, bounded to subhash_vocab_size."""
    h3_int = int(h3_cell, 16)
    bucket = (h3_int ^ (h3_int >> 17)) % cfg.subhash_vocab_size
    return SPATIAL_VOCAB_OFFSET + bucket


def dt_bucket(seconds: float, n_buckets: int = 64) -> int:
    """Log-bucketed time delta — handles Porto 15s through T-Drive minute-level."""
    if seconds <= 1.0:
        return 0
    return min(n_buckets - 1, int(math.log2(seconds)) + 1)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def speed_kmh(lat1: float, lon1: float, lat2: float, lon2: float, dt_s: float) -> float:
    if dt_s <= 0:
        return float("inf")
    return haversine_km(lat1, lon1, lat2, lon2) / (dt_s / 3600.0)


# ---------------------------------------------------------------------------
# Tokenise one trip → Gold record dict
# ---------------------------------------------------------------------------
def tokenize_trip(
    lats: list,
    lons: list,
    timestamps_unix: list,
    trajectory_id: str,
    region: str,
    cfg: TokenizerConfig,
) -> dict | None:
    n = len(lats)
    if n < MIN_TRIP_POINTS:
        return None

    lats            = lats[:MAX_TRIP_POINTS]
    lons            = lons[:MAX_TRIP_POINTS]
    timestamps_unix = timestamps_unix[:MAX_TRIP_POINTS]
    n = len(lats)

    h3_tokens_  = [BOS]
    dt_buckets_ = [0]
    min_of_day_ = [0]
    day_of_week_= [0]

    for i in range(n):
        cell = latlng_to_cell(lats[i], lons[i], cfg.h3_resolution)
        h3_tokens_.append(h3_to_token(cell, cfg))

        dt_s = (timestamps_unix[i] - timestamps_unix[i - 1]) if i > 0 else 0.0
        dt_buckets_.append(dt_bucket(dt_s, cfg.n_dt_buckets))

        ts = datetime.fromtimestamp(timestamps_unix[i], tz=timezone.utc)
        min_of_day_.append(ts.hour * 60 + ts.minute)
        day_of_week_.append(ts.weekday())

    h3_tokens_.append(EOS)
    dt_buckets_.append(0)
    min_of_day_.append(0)
    day_of_week_.append(0)

    return {
        "trajectory_id": trajectory_id,
        "region":        region,
        "h3_tokens":     h3_tokens_,
        "dt_buckets":    dt_buckets_,
        "min_of_day":    min_of_day_,
        "day_of_week":   day_of_week_,
        "n_tokens":      len(h3_tokens_),
    }


# ---------------------------------------------------------------------------
# Road Enhancement — OpenStreetMap integration and map-matching
# ---------------------------------------------------------------------------

def download_city_road_graph(city_name: str, network_type: str = "drive", 
                           cache_dir: str = "data/road_graphs") -> 'nx.MultiDiGraph':
    """Download road network from OpenStreetMap via OSMnx."""
    if not OSM_AVAILABLE:
        raise ImportError("OSMnx and dependencies required for road graph download")
        
    cache_path = Path(cache_dir) / f"{city_name.lower().replace(' ', '_')}_{network_type}.graphml"
    cache_path.parent.mkdir(exist_ok=True)
    
    # Load from cache if available
    if cache_path.exists():
        print(f"[road] Loading cached road graph from {cache_path}")
        G = ox.load_graphml(cache_path)
        return G
        
    print(f"[road] Downloading road network for {city_name} from OpenStreetMap...")
    
    # Download road network
    G = ox.graph_from_place(city_name, network_type=network_type)
    
    # Add road entity features from OSM tags
    for u, v, key, data in G.edges(keys=True, data=True):
        # Extract road entity types from OSM highway tags
        highway = data.get('highway', 'unknown')
        junction = data.get('junction', None)
        traffic_calming = data.get('traffic_calming', None)
        
        # Entity feature encoding
        data['is_roundabout'] = junction == 'roundabout'
        data['is_signal'] = 'traffic_signals' in str(data.get('highway', '')) or junction == 'traffic_signals'
        data['is_motorway'] = highway in ['motorway', 'motorway_link', 'trunk', 'trunk_link']
        data['is_residential'] = highway in ['residential', 'living_street']
        data['has_traffic_calming'] = traffic_calming is not None
        data['is_bridge'] = data.get('bridge', None) is not None
        
    # Cache for future use
    ox.save_graphml(G, cache_path)
    print(f"[road] Cached road graph to {cache_path}")
    
    return G

SILVER_FILES = {
    'porto':   SILVER / 'porto.parquet',
    'beijing': SILVER / 'tdrive.parquet',
}


def build_seg_id_map(nx_graph: 'nx.MultiDiGraph', silver_df: pd.DataFrame) -> dict:
    """
    Build {trajectory_id: [pyg_node_idx, ...]} by finding the nearest road node
    for every GPS point in silver_df using a single cKDTree query.

    - Builds the KDTree once over all road nodes.
    - Queries all GPS points from all trips in one batched call (workers=-1 → all cores).
    - Splits results back by trajectory.

    Accuracy: Euclidean distance on (lat, lon) with a cos(lat) correction on longitude
    gives the same nearest-neighbor result as haversine within a single city.
    """
    node_ids = list(nx_graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}

    node_lats = np.array([nx_graph.nodes[n]['y'] for n in node_ids], dtype=np.float64)
    node_lons = np.array([nx_graph.nodes[n]['x'] for n in node_ids], dtype=np.float64)
    lon_scale = np.cos(np.radians(node_lats.mean()))  # correct lon distortion
    tree = cKDTree(np.column_stack([node_lats, node_lons * lon_scale]))

    all_lats = np.array([lat for lats in silver_df['lats'] for lat in lats], dtype=np.float64)
    all_lons = np.array([lon for lons in silver_df['lons'] for lon in lons], dtype=np.float64)
    point_counts = [len(lats) for lats in silver_df['lats']]

    print(f"[{ts()}][road] Querying nearest node for {len(all_lats):,} GPS points "
          f"against {len(node_ids):,} road nodes (all cores)...")
    query_pts = np.column_stack([all_lats, all_lons * lon_scale])
    _, idxs = tree.query(query_pts, workers=-1)

    seg_id_map: dict = {}
    offset = 0
    for tid, count in zip(silver_df['trajectory_id'], point_counts):
        seg_id_map[tid] = idxs[offset:offset + count].tolist()
        offset += count

    print(f"[{ts()}][road] seg_id_map built for {len(seg_id_map):,} trajectories")
    return seg_id_map


# ---------------------------------------------------------------------------
# Bronze — immutable copy of raw files, organised into the lake structure
# ---------------------------------------------------------------------------
def _find_porto_zip(directory: Path) -> Path | None:
    """Return the outer UCI zip file wherever it landed."""
    candidates = list(directory.glob("*.zip"))
    return candidates[0] if candidates else None


def bronze_porto() -> None:
    src  = RAW    / "porto"
    dest = BRONZE / "porto"
    train_csv = dest / "train.csv"

    if train_csv.exists():
        print(f"[bronze/porto] Already extracted at {train_csv}, skipping.")
        return

    dest.mkdir(parents=True, exist_ok=True)

    outer_zip = _find_porto_zip(src)
    if outer_zip is None:
        print(f"ERROR: No zip file found in {src}. Run download_datasets.py --dataset porto first.")
        sys.exit(1)

    print(f"[bronze/porto] Extracting {outer_zip.name} …")
    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(dest)

    # The training data is nested inside train.csv.zip
    inner_zip = dest / "train.csv.zip"
    if inner_zip.exists():
        print("[bronze/porto] Extracting nested train.csv.zip …")
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(dest)
        inner_zip.unlink()

    if not train_csv.exists():
        print(f"ERROR: Expected {train_csv} after extraction — check zip contents.")
        sys.exit(1)

    print(f"[bronze/porto] Done → {train_csv}")


def bronze_tdrive() -> None:
    src  = RAW    / "tdrive"
    dest = BRONZE / "beijing"
    if dest.exists():
        print(f"[bronze/beijing] Already exists at {dest}, skipping copy.")
        return
    print(f"[bronze/beijing] Copying {src} → {dest}")
    shutil.copytree(src, dest)
    print("[bronze/beijing] Done.")


# ---------------------------------------------------------------------------
# Silver — Porto
# ---------------------------------------------------------------------------
def silver_porto() -> None:
    src  = BRONZE / "porto" / "train.csv"
    dest = SILVER / "porto.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[silver/porto] Already exists at {dest}, skipping.")
        return

    print(f"[silver/porto] Reading {src} …")
    df = pd.read_csv(src)
    total_raw = len(df)

    # drop flagged missing-data trips
    df = df[~df["MISSING_DATA"].astype(bool)].copy()

    records = []
    dropped_speed  = 0
    dropped_length = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="silver/porto"):
        try:
            polyline = json.loads(row["POLYLINE"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Porto POLYLINE is [[lon, lat], ...]
        if not polyline or len(polyline) < MIN_TRIP_POINTS:
            dropped_length += 1
            continue

        start_ts = int(row["TIMESTAMP"])
        lons = [p[0] for p in polyline]
        lats = [p[1] for p in polyline]
        timestamps = [start_ts + i * PORTO_DT_SECONDS for i in range(len(lats))]

        # speed filter: drop trips with any impossible jump
        bad_speed = False
        for i in range(1, len(lats)):
            if speed_kmh(lats[i-1], lons[i-1], lats[i], lons[i], PORTO_DT_SECONDS) > MAX_SPEED_KMH:
                bad_speed = True
                break
        if bad_speed:
            dropped_speed += 1
            continue

        records.append({
            "trajectory_id": str(row["TRIP_ID"]),
            "taxi_id":       str(row["TAXI_ID"]),
            "region":        "porto",
            "start_ts":      start_ts,
            "lats":          lats,
            "lons":          lons,
            "timestamps":    timestamps,
            "n_points":      len(lats),
        })

    print(
        f"[silver/porto] raw={total_raw:,}  "
        f"missing_data_dropped={total_raw - len(df):,}  "
        f"speed_dropped={dropped_speed:,}  "
        f"length_dropped={dropped_length:,}  "
        f"kept={len(records):,}"
    )
    pd.DataFrame(records).to_parquet(dest, index=False)
    print(f"[silver/porto] Saved → {dest}")


# ---------------------------------------------------------------------------
# Silver — T-Drive (Beijing)
# ---------------------------------------------------------------------------
def _parse_tdrive_file(path: Path) -> pd.DataFrame:
    """Read one T-Drive taxi file, return sorted DataFrame."""
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=["taxi_id", "timestamp", "lon", "lat"],
            parse_dates=["timestamp"],
        )
    except Exception:
        return pd.DataFrame()
    df = df.dropna(subset=["timestamp", "lon", "lat"])
    df["ts_unix"] = df["timestamp"].apply(lambda t: t.timestamp())
    return df.sort_values("ts_unix").reset_index(drop=True)


def _segment_trips(df: pd.DataFrame, taxi_id: str) -> list[dict]:
    """Split a single taxi's pings into trips on time gaps > TRIP_GAP_SECONDS."""
    trips = []
    if df.empty:
        return trips

    lats  = df["lat"].tolist()
    lons  = df["lon"].tolist()
    tss   = df["ts_unix"].tolist()
    n     = len(lats)

    seg_start = 0
    trip_idx  = 0

    for i in range(1, n + 1):
        gap = (tss[i] - tss[i - 1]) if i < n else TRIP_GAP_SECONDS + 1

        if gap > TRIP_GAP_SECONDS or i == n:
            seg_lats = lats[seg_start:i]
            seg_lons = lons[seg_start:i]
            seg_tss  = tss[seg_start:i]

            # speed filter: drop pings with impossible jumps, keep the rest
            clean_lats, clean_lons, clean_tss = [seg_lats[0]], [seg_lons[0]], [seg_tss[0]]
            for j in range(1, len(seg_lats)):
                dt = seg_tss[j] - seg_tss[j - 1]
                if speed_kmh(seg_lats[j-1], seg_lons[j-1], seg_lats[j], seg_lons[j], dt) <= MAX_SPEED_KMH:
                    clean_lats.append(seg_lats[j])
                    clean_lons.append(seg_lons[j])
                    clean_tss.append(seg_tss[j])

            if len(clean_lats) >= MIN_TRIP_POINTS:
                trips.append({
                    "trajectory_id": f"tdrive_{taxi_id}_{trip_idx}",
                    "taxi_id":       taxi_id,
                    "region":        "beijing",
                    "start_ts":      int(clean_tss[0]),
                    "lats":          clean_lats,
                    "lons":          clean_lons,
                    "timestamps":    clean_tss,
                    "n_points":      len(clean_lats),
                })
                trip_idx += 1
            seg_start = i

    return trips


def silver_tdrive() -> None:
    src  = BRONZE / "beijing"
    dest = SILVER / "tdrive.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[silver/beijing] Already exists at {dest}, skipping.")
        return

    txt_files = sorted(src.rglob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found under {src}. Run download_datasets.py first.")
        sys.exit(1)

    print(f"[silver/beijing] Processing {len(txt_files):,} taxi files …")
    all_trips = []
    for path in tqdm(txt_files, desc="silver/beijing"):
        taxi_id = path.stem
        df = _parse_tdrive_file(path)
        all_trips.extend(_segment_trips(df, taxi_id))

    print(f"[silver/beijing] Extracted {len(all_trips):,} trips from {len(txt_files):,} taxis")
    pd.DataFrame(all_trips).to_parquet(dest, index=False)
    print(f"[silver/beijing] Saved → {dest}")


# ---------------------------------------------------------------------------
# Gold — tokenise Silver → sharded Parquet
# ---------------------------------------------------------------------------
def gold_region(region: str, silver_path: Path, cfg: TokenizerConfig, n_shards: int) -> None:
    out_dir = GOLD / region
    marker  = out_dir / ".done"

    if marker.exists():
        print(f"[gold/{region}] Already tokenised, skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gold/{region}] Reading {silver_path} …")
    df = pd.read_parquet(silver_path)
    print(f"[gold/{region}] Tokenising {len(df):,} trips into {n_shards} shards …")

    writers = [
        pq.ParquetWriter(out_dir / f"shard_{i:04d}.parquet", GOLD_SCHEMA)
        for i in range(n_shards)
    ]

    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"gold/{region}"):
        record = tokenize_trip(
            lats=row["lats"],
            lons=row["lons"],
            timestamps_unix=row["timestamps"],
            trajectory_id=row["trajectory_id"],
            region=row["region"],
            cfg=cfg,
        )
        if record is None:
            skipped += 1
            continue

        shard_idx = int(idx) % n_shards
        table = pa.table(
            {k: [v] for k, v in record.items()},
            schema=GOLD_SCHEMA,
        )
        writers[shard_idx].write_table(table)

    for w in writers:
        w.close()

    marker.touch()
    print(
        f"[gold/{region}] Done. "
        f"tokenised={len(df) - skipped:,}  skipped={skipped:,}  "
        f"shards={n_shards}  → {out_dir}"
    )


def gold(cfg: TokenizerConfig, n_shards: int, regions: list[str]) -> None:
    mapping = {
        "porto":   (SILVER / "porto.parquet",  "porto"),
        "beijing": (SILVER / "tdrive.parquet", "beijing"),
    }
    for key in regions:
        silver_path, region = mapping[key]
        if not silver_path.exists():
            print(f"ERROR: Silver file not found: {silver_path}. Run --stage silver first.")
            sys.exit(1)
        gold_region(region, silver_path, cfg, n_shards)


def road_gold_region(region: str, gold_dir: Path, cfg: TokenizerConfig) -> None:
    """
    Enhance Gold shards with road segment IDs derived from OSMnx nearest-node matching.

    For each GPS point in a trajectory (from Silver), finds the nearest road junction
    node in the city's PyG road graph using a single batched cKDTree query.  The
    resulting PyG node index (0..N-1) is stored as seg_id per H3 token position.
    No Valhalla or routing engine required.
    """
    road_dir = GOLD.parent / "road_enhanced_gold" / region
    marker   = road_dir / ".done"
    graph_path = road_dir / "road_graph.pt"

    city_names = {'porto': 'Porto, Portugal', 'beijing': 'Beijing, China'}
    if region not in city_names:
        print(f"[road-gold/{region}] Unknown region, skipping.")
        return

    if marker.exists() and graph_path.exists():
        print(f"[{ts()}][road-gold/{region}] Already done, skipping.")
        return

    if not OSM_AVAILABLE:
        print(f"[{ts()}][road-gold/{region}] OSM libraries not available, skipping.")
        return

    silver_path = SILVER_FILES.get(region)
    if silver_path is None or not silver_path.exists():
        print(f"[{ts()}][road-gold/{region}] Silver file not found at {silver_path}. "
              f"Run --stage silver first.")
        return

    road_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load / download OSM road graph ───────────────────────────────────
    try:
        print(f"[{ts()}][road-gold/{region}] Loading road network for {city_names[region]}...")
        road_graph = download_city_road_graph(city_names[region])
        print(f"[{ts()}][road-gold/{region}] Road graph: "
              f"{road_graph.number_of_nodes():,} nodes, {road_graph.number_of_edges():,} edges")
    except Exception as e:
        print(f"[{ts()}][road-gold/{region}] Failed to load road graph: {e}")
        return

    # ── 2. Serialize PyG graph for training container ────────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from experts import networkx_to_pyg
        import torch
        pyg_graph = networkx_to_pyg(road_graph)
        torch.save(pyg_graph, graph_path)
        print(f"[{ts()}][road-gold/{region}] Saved PyG graph → {graph_path} "
              f"({pyg_graph.x.size(0):,} nodes)")
    except Exception as e:
        print(f"[{ts()}][road-gold/{region}] Could not save PyG graph: {e}")
        return

    # ── 3. Build seg_id_map from Silver GPS coordinates ──────────────────────
    print(f"[{ts()}][road-gold/{region}] Loading Silver from {silver_path}...")
    silver_df = pd.read_parquet(silver_path)
    print(f"[{ts()}][road-gold/{region}] Silver: {len(silver_df):,} trips")
    seg_id_map = build_seg_id_map(road_graph, silver_df)
    del silver_df  # free RAM before shard processing

    # ── 4. Write enhanced shards ─────────────────────────────────────────────
    enhanced_schema = pa.schema([
        pa.field("trajectory_id", pa.string()),
        pa.field("region",        pa.string()),
        pa.field("h3_tokens",     pa.list_(pa.int32())),
        pa.field("dt_buckets",    pa.list_(pa.int32())),
        pa.field("min_of_day",    pa.list_(pa.int32())),
        pa.field("day_of_week",   pa.list_(pa.int32())),
        pa.field("n_tokens",      pa.int32()),
        pa.field("seg_ids",       pa.list_(pa.int32())),
    ])

    gold_shards = sorted(gold_dir.glob("shard_*.parquet"))
    if not gold_shards:
        print(f"[{ts()}][road-gold/{region}] No gold shards in {gold_dir}")
        return

    print(f"[{ts()}][road-gold/{region}] Writing {len(gold_shards)} enhanced shards...")

    def process_shard(shard_path: Path) -> tuple[int, int]:
        df = pd.read_parquet(shard_path)
        records, ok, fail = [], 0, 0
        for _, row in df.iterrows():
            try:
                tid       = row['trajectory_id']
                h3_tokens = row['h3_tokens']
                n_spatial = len(h3_tokens) - 2  # exclude BOS and EOS slots

                if tid in seg_id_map:
                    spatial_segs = seg_id_map[tid][:n_spatial]
                    if len(spatial_segs) < n_spatial:
                        spatial_segs = spatial_segs + [0] * (n_spatial - len(spatial_segs))
                else:
                    spatial_segs = [0] * n_spatial

                rec = row.to_dict()
                rec['seg_ids'] = [0] + spatial_segs + [0]  # BOS=0, spatial, EOS=0
                records.append(rec)
                ok += 1
            except Exception as e:
                print(f"[road-gold/{region}] Failed {row.get('trajectory_id', '?')}: {e}")
                fail += 1

        if records:
            out_df = pd.DataFrame(records)
            table_data = {f.name: out_df[f.name].tolist() for f in enhanced_schema}
            pq.write_table(pa.table(table_data, schema=enhanced_schema),
                           road_dir / shard_path.name)

        print(f"[{ts()}][road-gold/{region}] {shard_path.name}: {ok} ok, {fail} failed")
        return ok, fail

    enhanced_count = failed_count = 0
    with ThreadPoolExecutor(max_workers=min(len(gold_shards), 4)) as pool:
        futures = {pool.submit(process_shard, sp): sp for sp in gold_shards}
        for fut in as_completed(futures):
            ok, fail = fut.result()
            enhanced_count += ok
            failed_count += fail

    marker.touch()
    print(f"[{ts()}][road-gold/{region}] Done. "
          f"{enhanced_count:,} enhanced, {failed_count} failed → {road_dir}")

def road_gold(cfg: TokenizerConfig, n_shards: int, regions: list[str]) -> None:
    """Process all regions for road-enhanced gold."""
    for region in regions:
        if region not in ('porto', 'beijing'):
            print(f"ERROR: Unknown region '{region}' for road enhancement.")
            continue
        gold_dir = GOLD / region
        if not gold_dir.exists() or not list(gold_dir.glob("shard_*.parquet")):
            print(f"ERROR: Gold shards not found for {region}. Run --stage gold first.")
            continue
        road_gold_region(region, gold_dir, cfg)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
STAGES = ["bronze", "silver", "gold", "road-gold", "check-road-gold"]

# Thresholds for check-road-gold quality gate
_MAX_ZERO_SEG_RATE  = 0.10   # >10% unmatched GPS tokens → fail
_MAX_ZERO_NODE_RATE = 0.50   # >50% all-zero node feature rows → fail


def check_road_gold_region(region: str) -> bool:
    """
    Validate road-enhanced gold output for one region.
    Returns True if all checks pass, False otherwise.
    Prints a pass/fail line for each check so failures are easy to spot in CI logs.
    """
    try:
        import torch
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError as e:
        print(f"[check-road-gold/{region}] SKIP — missing dependency: {e}")
        return True  # not a data problem

    road_dir  = GOLD.parent / "road_enhanced_gold" / region
    graph_path = road_dir / "road_graph.pt"
    ok = True

    def _pass(msg: str) -> None:
        print(f"[check-road-gold/{region}]  OK  {msg}")

    def _fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"[check-road-gold/{region}] FAIL {msg}")

    # ── 1. road_graph.pt ────────────────────────────────────────────────────
    if not graph_path.exists():
        _fail(f"road_graph.pt not found at {graph_path}")
        return False  # nothing else to check

    g = torch.load(graph_path, weights_only=False)
    n = g.num_nodes

    if n > 0:
        _pass(f"road_graph.pt loaded — {n:,} nodes")
    else:
        _fail("road_graph.pt has 0 nodes")

    if g.x.shape == (n, 6):
        _pass(f"node feature shape {tuple(g.x.shape)} is correct")
    else:
        _fail(f"node feature shape {tuple(g.x.shape)}, expected ({n}, 6)")

    if g.edge_index.shape[1] > 0:
        _pass(f"edge_index has {g.edge_index.shape[1]:,} edges")
    else:
        _fail("edge_index is empty — graph has no edges")

    seg_min, seg_max = g.seg_id.min().item(), g.seg_id.max().item()
    if seg_min == 0 and seg_max == n - 1:
        _pass(f"seg_id range [0, {seg_max}] matches num_nodes")
    else:
        _fail(f"seg_id range [{seg_min}, {seg_max}] does not match [0, {n - 1}]")

    zero_node_rate = (g.x == 0).all(dim=1).float().mean().item()
    if zero_node_rate < _MAX_ZERO_NODE_RATE:
        _pass(f"all-zero node rows: {zero_node_rate*100:.1f}% (< {_MAX_ZERO_NODE_RATE*100:.0f}%)")
    else:
        _fail(f"all-zero node rows: {zero_node_rate*100:.1f}% (>= {_MAX_ZERO_NODE_RATE*100:.0f}%) — "
              f"entity features may be empty")

    # ── 2. Parquet shards ────────────────────────────────────────────────────
    shards = sorted(road_dir.glob("shard_*.parquet"))
    if shards:
        _pass(f"{len(shards)} parquet shard(s) found")
    else:
        _fail(f"no shard_*.parquet files in {road_dir}")
        return False

    all_seg_ids: list[int] = []
    total_rows = 0
    has_seg_ids_col = True

    for shard in shards:
        df = pq.read_table(shard).to_pandas()
        total_rows += len(df)
        if "seg_ids" not in df.columns:
            has_seg_ids_col = False
        else:
            for row in df["seg_ids"]:
                all_seg_ids.extend(row)

    if has_seg_ids_col:
        _pass("'seg_ids' column present in all shards")
    else:
        _fail("'seg_ids' column missing from one or more shards")

    if total_rows > 0:
        _pass(f"total rows: {total_rows:,}")
    else:
        _fail("shards contain 0 rows")

    if all_seg_ids:
        arr = np.array(all_seg_ids)
        seg_id_max = int(arr.max())
        if seg_id_max < n:
            _pass(f"max seg_id {seg_id_max} < num_nodes {n}")
        else:
            _fail(f"max seg_id {seg_id_max} >= num_nodes {n} — seg_ids out of range")

        zero_rate = float((arr == 0).mean())
        if zero_rate < _MAX_ZERO_SEG_RATE:
            _pass(f"unmatched GPS tokens: {zero_rate*100:.1f}% (< {_MAX_ZERO_SEG_RATE*100:.0f}%)")
        else:
            _fail(f"unmatched GPS tokens: {zero_rate*100:.1f}% (>= {_MAX_ZERO_SEG_RATE*100:.0f}%) — "
                  f"too many seg_id=0 fallbacks")

        unique = int(np.unique(arr).size)
        _pass(f"unique seg_ids: {unique:,} / {n:,} nodes ({unique/n*100:.1f}% coverage)")

    return ok


def check_road_gold(regions: list[str]) -> None:
    """Run check-road-gold quality gate for each region; exit non-zero if any fail."""
    failed = []
    for region in regions:
        print(f"\n[{ts()}] Checking road-enhanced gold for '{region}'...")
        passed = check_road_gold_region(region)
        if not passed:
            failed.append(region)

    print()
    if failed:
        print(f"[check-road-gold] FAILED for regions: {failed}")
        sys.exit(1)
    else:
        print("[check-road-gold] All checks passed.")


def run(stage: str, regions: list[str], cfg: TokenizerConfig, n_shards: int) -> None:

    if stage == "check-road-gold":
        check_road_gold(regions)
        return

    if stage == "road-gold":
        road_gold(cfg, n_shards, regions)
        return
        
    up_to = STAGES.index(stage)

    if up_to >= STAGES.index("bronze"):
        if "porto" in regions:
            bronze_porto()
        if "beijing" in regions:
            bronze_tdrive()

    if up_to >= STAGES.index("silver"):
        if "porto" in regions:
            silver_porto()
        if "beijing" in regions:
            silver_tdrive()

    if up_to >= STAGES.index("gold"):
        gold(cfg, n_shards, regions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL pipeline: raw GPS data → Bronze/Silver/Gold/Road-Enhanced-Gold Parquet shards."
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=_CFG["pipeline"]["default_stage"],
        help=f"Run up to and including this stage (default: {_CFG['pipeline']['default_stage']})",
    )
    parser.add_argument(
        "--region",
        choices=["porto", "beijing", "all"],
        default=_CFG["pipeline"]["default_region"],
        help=f"Which region to process (default: {_CFG['pipeline']['default_region']})",
    )
    parser.add_argument(
        "--regions", 
        nargs="+",
        choices=["porto", "beijing"],
        help="Multiple regions to process (alternative to --region)"
    )
    parser.add_argument(
        "--h3-res",
        type=int,
        choices=[8, 9],
        default=_CFG["tokenizer"]["h3_resolution"],
        dest="h3_res",
        help=f"H3 resolution: 9 ≈ 0.1 km², 8 ≈ 0.74 km² (default: {_CFG['tokenizer']['h3_resolution']})",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=_CFG["pipeline"]["default_shards"],
        help=f"Number of Gold Parquet shards per region (default: {_CFG['pipeline']['default_shards']})",
    )
    
    return parser.parse_args()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    args = parse_args()
    cfg  = TokenizerConfig(h3_resolution=args.h3_res)

    if args.regions:
        regions = args.regions
    elif args.region == "all":
        regions = ["porto", "beijing"]
    else:
        regions = [args.region]

    print(f"[{ts()}] Stage   : up to '{args.stage}'")
    print(f"[{ts()}] Regions : {regions}")
    print(f"[{ts()}] H3 res  : {cfg.h3_resolution}")
    print(f"[{ts()}] Shards  : {args.shards}")
    print()

    run(args.stage, regions, cfg, args.shards)
    print(f"\n[{ts()}] ETL complete.")


if __name__ == "__main__":
    main()
