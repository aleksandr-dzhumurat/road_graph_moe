#!/usr/bin/env python3
"""
Phase 2: Geographic Expert Training for Trajectory Foundation Model

Implements:
- Geographic experts (GAT road-adapters) for Porto and Beijing  
- Multi-expert routing with geographic priors
- Multi-tenant scheduling for unequal data volumes
- Anomaly-importance-weighted graph partitioning
- Expert-parallel training with frozen backbone
- **NEW**: Real OpenStreetMap integration with road entity features

Features:
- Downloads city road networks from OpenStreetMap via OSMnx
- Map-matches GPS trajectories to actual road segments
- Extracts road entity features (roundabouts, signals, bridges, etc.)
- Supports both real OSM data and mock graphs for development
- Optional Valhalla integration for production map-matching

Usage:
    # Basic training with mock road graphs
    python scripts/experts.py --expert porto --workers 4
    
    # Training with real OpenStreetMap data
    python scripts/experts.py --expert porto --use-osm --workers 4
    
    # Multi-tenant training across experts
    python scripts/experts.py --multi-tenant --use-osm
    
    # Debug mode
    python scripts/experts.py --debug --expert porto --use-osm

Dependencies for OSM integration:
    osmnx>=1.6.0, geopandas>=0.14.0, shapely>=2.0.0, networkx>=3.0, torch-geometric>=2.4.0
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Try to import PyTorch Geometric - fallback if not available
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import add_self_loops
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch Geometric not available. Road graph functionality disabled.")
    TORCH_GEOMETRIC_AVAILABLE = False

# OSM libraries are only needed for ETL (road graph download/build).
# The training container loads pre-built road_graph.pt from S3 and never calls OSMnx.
try:
    import osmnx as ox
    import networkx as nx
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False

# Import from existing scripts
from backbone import TrajectoryBackbone, TrajectoryDataset, collate_fn, load_config, ts

# ========================================================================================
# Configuration and Constants
# ========================================================================================

EXPERTS = ["porto", "beijing"]
PAD, BOS, EOS, MASK = 0, 1, 2, 3

# Per-expert step budgets scaled to approximate equal wall-clock time.
# Porto GAT (5k nodes) runs ~2.7x faster than Beijing GAT (163k nodes).
MAX_STEPS_PER_EXPERT = {
    "porto":   50_000,
    "beijing": 135_000,  # 50_000 * 2.7 ≈ equal wall-clock on L40S
}

@dataclass
class ExpertConfig:
    """Configuration for geographic experts."""
    d_model: int = 768
    entity_dim: int = 6  # roundabout, signal, motorway, etc.
    gat_hidden: int = 128
    gat_heads: int = 4
    prior_strength: float = 4.0
    aux_loss_weight: float = 0.0  # disabled: each rank trains on its own region's data,
                                  # so there is nothing to load-balance; geographic prior suffices
    lr: float = 2e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0

@dataclass 
class ExpertJob:
    """Multi-tenant scheduler job specification."""
    name: str
    n_samples: int
    graph_parts: int
    workers: int = 0
    remaining_work: int = 0

# ========================================================================================
# Geographic Expert (GAT Road-Adapter)
# ========================================================================================

class GeographicExpert(nn.Module):
    """Per-city road-aware refinement of backbone hidden states."""
    
    def __init__(self, d_model: int, entity_dim: int = 6, gat_hidden: int = 128, 
                 gat_heads: int = 4, n_seg: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.gat_hidden = gat_hidden
        
        # Road segment embedding (city-specific vocabulary)
        if n_seg is not None:
            self.seg_emb = nn.Embedding(n_seg, gat_hidden)
        else:
            # Fallback for testing without real road graphs
            self.seg_emb = nn.Embedding(10000, gat_hidden)
            
        # Entity feature projection (roundabout, signal, motorway, etc.)
        self.entity_proj = nn.Linear(entity_dim, gat_hidden)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Graph Attention Network layers (PyTorch Geometric)
            self.gat1 = GATConv(gat_hidden, gat_hidden // gat_heads, heads=gat_heads, dropout=0.1)
            self.gat2 = GATConv(gat_hidden, gat_hidden, heads=1, dropout=0.1)
        else:
            # Fallback: simple MLPs when PyG not available
            self.gat1 = nn.Sequential(
                nn.Linear(gat_hidden, gat_hidden * gat_heads),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.gat2 = nn.Linear(gat_hidden * gat_heads, gat_hidden)
            
        # Fusion layer: road context + backbone hidden state -> refined state
        self.fuse = nn.Sequential(
            nn.Linear(d_model + gat_hidden, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # AR pretext head: predict next H3 token from fused states (same vocab as backbone).
        # Kept separate so the backbone tok_emb weight can be tied at training time.
        self.tok_head = nn.Linear(d_model, d_model, bias=False)  # projection before tying
        
    def road_context(self, g=None):
        """Run GAT over city road graph -> node embeddings [num_nodes, gat_hidden]."""
        device = next(self.parameters()).device
        if not TORCH_GEOMETRIC_AVAILABLE or g is None or isinstance(g, MockRoadGraph):
            n = g.num_nodes() if isinstance(g, MockRoadGraph) else (g.num_nodes if g is not None else 32)
            return torch.zeros(n, self.gat_hidden, device=device)

        # g is a PyG Data object: g.x [num_nodes, entity_dim], g.edge_index [2, num_edges]
        seg_ids = g.seg_id if hasattr(g, 'seg_id') else torch.arange(g.x.size(0), device=device)
        x = self.seg_emb(seg_ids.to(device)) + self.entity_proj(g.x.to(device))
        x = torch.relu(self.gat1(x, g.edge_index.to(device)))
        x = self.gat2(x, g.edge_index.to(device))
        return x  # [num_nodes, gat_hidden]
        
    def forward(self, h: torch.Tensor, seg_ids_per_token: torch.Tensor, 
                node_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuse road context into backbone hidden states.
        
        Args:
            h: [B, L, d_model] backbone hidden states
            seg_ids_per_token: [B, L] map-matched road segment per token
            node_emb: [num_nodes, gat_hidden] precomputed road embeddings
            
        Returns:
            [B, L, d_model] road-aware refined hidden states
        """
        B, L, D = h.shape
        
        # Handle case where node_emb might be smaller than expected seg_ids
        max_seg_id = seg_ids_per_token.max().item() if seg_ids_per_token.numel() > 0 else 0
        if max_seg_id >= node_emb.size(0):
            # Pad node_emb or clamp seg_ids to valid range
            seg_ids_per_token = seg_ids_per_token.clamp(0, node_emb.size(0) - 1)
            
        # Gather road context per token
        ctx = node_emb[seg_ids_per_token]  # [B, L, gat_hidden]
        
        # Fuse backbone state with road context
        fused_input = torch.cat([h, ctx], dim=-1)  # [B, L, d_model + gat_hidden]
        
        return self.fuse(fused_input)  # [B, L, d_model]

# ========================================================================================
# Geographic Router (MoE Gate)
# ========================================================================================

class GeographicRouter(nn.Module):
    """Top-1 routing with geographic priors."""
    
    def __init__(self, d_model: int, n_experts: int = 2, prior_strength: float = 4.0):
        super().__init__()
        self.n_experts = n_experts
        self.prior_strength = prior_strength
        self.gate = nn.Linear(d_model, n_experts)
        
    def forward(self, pooled_h: torch.Tensor, region_prior_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route trajectories to experts based on learned gate + geographic prior.
        
        Args:
            pooled_h: [B, d_model] trajectory summary (e.g., mean-pooled)
            region_prior_onehot: [B, n_experts] one-hot region indicators
            
        Returns:
            top1: [B] expert assignments (0=porto, 1=beijing)
            probs: [B, n_experts] routing probabilities  
            aux_loss: scalar load-balancing auxiliary loss
        """
        # Learned gate logits + strong geographic prior
        logits = self.gate(pooled_h) + self.prior_strength * region_prior_onehot
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 expert selection
        top1 = probs.argmax(dim=-1)
        
        # Load-balancing auxiliary loss (Switch Transformer style)
        importance = probs.mean(0)  # Average probability per expert
        load = torch.bincount(top1, minlength=self.n_experts).float() / probs.size(0)
        aux_loss = (importance * load).sum() * self.n_experts
        
        return top1, probs, aux_loss

# ========================================================================================
# Multi-Tenant Scheduler  
# ========================================================================================

def allocate_workers(jobs: List[ExpertJob], total_workers: int, min_per_expert: int = 1) -> List[ExpertJob]:
    """
    Proportional allocation of workers based on remaining work with minimum guarantees.
    
    Args:
        jobs: List of expert jobs with work remaining
        total_workers: Total available workers
        min_per_expert: Minimum workers per expert (prevents starvation)
        
    Returns:
        Updated jobs with worker allocations
    """
    # Ensure minimum allocation
    for job in jobs:
        job.workers = min_per_expert
        
    remaining_workers = total_workers - min_per_expert * len(jobs)
    
    if remaining_workers > 0:
        total_work = sum(job.remaining_work for job in jobs)
        
        if total_work > 0:
            # Proportional allocation based on remaining work
            for job in jobs:
                additional = int(remaining_workers * job.remaining_work / total_work)
                job.workers += additional
                
            # Distribute leftover workers to jobs with most remaining work
            assigned = sum(job.workers for job in jobs)
            leftover = total_workers - assigned
            
            jobs_sorted = sorted(jobs, key=lambda x: -x.remaining_work)
            for i in range(leftover):
                jobs_sorted[i % len(jobs_sorted)].workers += 1
                
    return jobs

def expert_for_rank(rank: int, jobs: List[ExpertJob]) -> str:
    """Determine which expert this rank should handle."""
    current_rank = 0
    for job in jobs:
        if current_rank <= rank < current_rank + job.workers:
            return job.name
        current_rank += job.workers
    return jobs[0].name  # Fallback

# ========================================================================================
# OpenStreetMap Integration and Map-Matching
# ========================================================================================

def download_city_road_graph(city_name: str, network_type: str = "drive", cache_dir: str = "data/road_graphs") -> 'nx.MultiDiGraph':
    """Download road network from OpenStreetMap via OSMnx."""
    if not OSM_AVAILABLE:
        raise ImportError("OSMnx and dependencies required for real road graph download")
        
    cache_path = Path(cache_dir) / f"{city_name.lower().replace(' ', '_')}_{network_type}.graphml"
    cache_path.parent.mkdir(exist_ok=True)
    
    # Load from cache if available
    if cache_path.exists():
        logging.info(f"Loading cached road graph from {cache_path}")
        G = ox.load_graphml(cache_path)
        return G
        
    logging.info(f"Downloading road network for {city_name} from OpenStreetMap...")
    
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
    logging.info(f"Cached road graph to {cache_path}")
    
    return G

def map_match_trajectory_simple(trajectory_coords: List[Tuple[float, float]], 
                               road_graph: 'nx.MultiDiGraph') -> List[dict]:
    """Simple nearest-neighbor map-matching for development."""
    if not OSM_AVAILABLE or not trajectory_coords:
        return []
        
    matched_segments = []
    
    for lat, lon in trajectory_coords:
        try:
            # Find nearest road segment  
            nearest_edge = ox.nearest_edges(road_graph, lon, lat, return_dist=False)
            u, v, key = nearest_edge
            
            # Get edge data for entity features
            edge_data = road_graph[u][v][key]
            
            matched_segments.append({
                'edge_id': f"{u}_{v}_{key}",
                'u': u, 'v': v, 'key': key,
                'osm_id': edge_data.get('osmid', None),
                'highway': edge_data.get('highway', 'unknown'),
                'is_roundabout': edge_data.get('is_roundabout', False),
                'is_signal': edge_data.get('is_signal', False),
                'is_motorway': edge_data.get('is_motorway', False),
                'is_residential': edge_data.get('is_residential', False),
                'has_traffic_calming': edge_data.get('has_traffic_calming', False),
                'is_bridge': edge_data.get('is_bridge', False)
            })
        except Exception as e:
            logging.warning(f"Map-matching failed for point ({lat}, {lon}): {e}")
            # Fallback segment
            matched_segments.append({
                'edge_id': 'unknown',
                'is_roundabout': False, 'is_signal': False, 'is_motorway': False,
                'is_residential': False, 'has_traffic_calming': False, 'is_bridge': False
            })
            
    return matched_segments

def _bool_attr(val, default=False) -> float:
    """Convert OSM graph attribute to float, handling string booleans from GraphML."""
    if val is None:
        return float(default)
    if isinstance(val, str):
        return 1.0 if val.strip().lower() in ('true', '1', 'yes') else 0.0
    return float(bool(val))


def networkx_to_pyg(nx_graph: 'nx.MultiDiGraph') -> 'Data':
    """Convert NetworkX road graph to a PyG Data object for GAT processing."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric required for graph conversion")

    nodes = list(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    src_nodes, dst_nodes, edge_features = [], [], []
    for u, v, _, data in nx_graph.edges(keys=True, data=True):
        src_nodes.append(node_to_idx[u])
        dst_nodes.append(node_to_idx[v])
        edge_features.append([
            _bool_attr(data.get('is_roundabout')),
            _bool_attr(data.get('is_signal')),
            _bool_attr(data.get('is_motorway')),
            _bool_attr(data.get('is_residential')),
            _bool_attr(data.get('has_traffic_calming')),
            _bool_attr(data.get('is_bridge')),
        ])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)

    # Aggregate edge entity features to nodes (mean of incident edges)
    if edge_features:
        edge_feats = torch.tensor(edge_features, dtype=torch.float32)
        node_feats = torch.zeros(n, 6)
        for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
            node_feats[s] += edge_feats[i]
            node_feats[d] += edge_feats[i]
        deg = torch.bincount(torch.tensor(src_nodes), minlength=n).float().clamp(min=1).unsqueeze(1)
        node_feats = node_feats / deg
    else:
        node_feats = torch.zeros(n, 6)

    data = Data(x=node_feats, edge_index=edge_index)
    data.seg_id = torch.arange(n)
    data.degree = torch.bincount(edge_index[1], minlength=n).float().unsqueeze(1)
    return data

# ========================================================================================
# Road Graph Utilities (Enhanced with OSM Integration)
# ========================================================================================

class MockRoadGraph:
    """Mock road graph used when PyTorch Geometric or OSM data is unavailable."""

    def __init__(self, n_nodes: int = 1000):
        self.n_nodes = n_nodes
        self.x = torch.randn(n_nodes, 6)
        self.seg_id = torch.arange(n_nodes)
        self.edge_index = torch.zeros(2, 0, dtype=torch.long)
        self.degree = torch.randint(1, 10, (n_nodes, 1)).float()

    def num_nodes(self):
        return self.n_nodes

def load_city_graph(city: str, fallback_nodes: int = 1000, use_osm: bool = True):
    """
    Load city road graph with real OSM data or fallback to mock.
    
    Args:
        city: City name ('porto' or 'beijing')
        fallback_nodes: Number of nodes for mock graph
        use_osm: Whether to attempt OSM download
        
    Returns:
        PyG Data object or MockRoadGraph fallback
    """
    # City name mapping for OSM queries
    city_names = {
        'porto': 'Porto, Portugal',
        'beijing': 'Beijing, China'
    }
    
    # Prefer pre-built graph saved by ETL (no OSM/network access needed at training time)
    prebuilt = Path(f"data/road_enhanced_gold/{city}/road_graph.pt")
    if prebuilt.exists() and TORCH_GEOMETRIC_AVAILABLE:
        try:
            pyg_graph = torch.load(prebuilt, weights_only=False)
            logging.info(f"Loaded pre-built road graph for {city}: {pyg_graph.x.size(0)} nodes")
            return pyg_graph
        except Exception as e:
            logging.warning(f"Failed to load pre-built graph {prebuilt}: {e}")

    if use_osm and OSM_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE and city in city_names:
        try:
            logging.info(f"Downloading OSM road graph for {city}")
            nx_graph = download_city_road_graph(city_names[city])
            pyg_graph = networkx_to_pyg(nx_graph)
            logging.info(f"Downloaded {city} road graph: {pyg_graph.x.size(0)} nodes")
            return pyg_graph
        except Exception as e:
            logging.warning(f"Failed to load OSM graph for {city}: {e}")

    if not TORCH_GEOMETRIC_AVAILABLE:
        logging.warning(f"PyTorch Geometric unavailable, using mock graph for {city}")
    else:
        logging.warning(f"No road graph found for {city}, using mock")
    return MockRoadGraph(fallback_nodes)

def enhance_trajectory_with_roads(trajectory_data: dict, city: str) -> dict:
    """
    Enhance trajectory data with road segment information.
    
    Args:
        trajectory_data: Dict with 'h3_tokens', 'lat', 'lon' fields
        city: City name for road graph lookup
        
    Returns:
        Enhanced trajectory with 'seg_ids' and 'entity_features' fields
    """
    enhanced = trajectory_data.copy()
    
    # Try to get real road data
    if OSM_AVAILABLE and 'lat' in trajectory_data and 'lon' in trajectory_data:
        try:
            coords = list(zip(trajectory_data['lat'], trajectory_data['lon']))
            city_names = {'porto': 'Porto, Portugal', 'beijing': 'Beijing, China'}
            
            if city in city_names:
                nx_graph = download_city_road_graph(city_names[city])
                matched_segments = map_match_trajectory_simple(coords, nx_graph)
                
                # Extract segment IDs and entity features
                seg_ids = []
                entity_features = []
                
                for seg in matched_segments:
                    # Create segment ID hash (for embedding lookup)
                    seg_id = hash(seg['edge_id']) % 10000  # Map to reasonable range
                    seg_ids.append(seg_id)
                    
                    entity_features.append([
                        seg['is_roundabout'],
                        seg['is_signal'], 
                        seg['is_motorway'],
                        seg['is_residential'],
                        seg['has_traffic_calming'],
                        seg['is_bridge']
                    ])
                    
                enhanced['seg_ids'] = seg_ids
                enhanced['entity_features'] = entity_features
                
                logging.debug(f"Enhanced trajectory with {len(seg_ids)} road segments")
                return enhanced
                
        except Exception as e:
            logging.warning(f"Road enhancement failed: {e}")
    
    # Fallback: create mock segment data based on H3 tokens
    if 'h3_tokens' in trajectory_data:
        h3_tokens = trajectory_data['h3_tokens']
        # Use H3 token hash as segment ID
        seg_ids = [hash(str(token)) % 10000 for token in h3_tokens]
        # Random entity features as fallback
        entity_features = [[False] * 6 for _ in h3_tokens]
        
        enhanced['seg_ids'] = seg_ids
        enhanced['entity_features'] = entity_features
        
    return enhanced

# ========================================================================================
# Training Loop
# ========================================================================================

def expert_loss(model: GeographicExpert, router: GeographicRouter, batch: dict,
                node_emb: torch.Tensor, config: ExpertConfig,
                backbone_tok_emb: torch.Tensor = None, step: int = 0,
                world_size: int = 1) -> Tuple[torch.Tensor, dict]:
    """
    Compute expert training loss.
    
    Args:
        model: Geographic expert model
        router: Geographic router
        batch: Training batch
        node_emb: Precomputed road node embeddings
        config: Expert configuration
        
    Returns:
        loss: Total training loss
        metrics: Dictionary of training metrics
    """
    h = batch["backbone_features"]  # [B, L, d_model] from frozen backbone
    B, L, D = h.shape
    
    # seg_ids should always be present when training on road-enhanced gold shards.
    # Zeros fallback silently zeroes road context — visible as fuse weights near zero.
    if "seg_ids" not in batch:
        logging.warning("seg_ids missing from batch — road context zeroed. "
                        "Run etl.py --stage road-gold to generate road-enhanced shards.")
        batch["seg_ids"] = torch.zeros(B, L, dtype=torch.long, device=h.device)
    if "region_prior" not in batch:
        raise ValueError("region_prior must be set in the training loop per expert — see train_expert()")
        
    # Geographic routing
    pooled_h = h.mean(dim=1)  # [B, d_model] trajectory summary
    top1, probs, aux_loss = router(pooled_h, batch["region_prior"])
    
    # Road-aware refinement
    fused_h = model(h, batch["seg_ids"], node_emb)
    
    # Phase 2 pretext: AR next-H3-token prediction on fused states.
    # Same objective as Phase 1 backbone pretraining, but now conditioned on road context.
    # The expert should improve on the backbone's token predictions by knowing road structure.
    # Weight-tied to backbone tok_emb for efficiency (vocab_size → shared embedding matrix).
    if "target_tokens" in batch and backbone_tok_emb is not None:
        targets = batch["target_tokens"][:, 1:]         # [B, L-1] next H3 token IDs
        proj = model.tok_head(fused_h[:, :-1])          # [B, L-1, d_model]
        logits = proj @ backbone_tok_emb.T              # [B, L-1, vocab_size] weight-tied
        ar_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  targets.reshape(-1), ignore_index=PAD)
    else:
        ar_loss = torch.tensor(0.0, device=h.device)
        
    # Aux loss warmup: suppress load-balancing for first 1000 steps.
    # Also skip entirely when world_size == 1: Switch-style load balancing is only
    # meaningful when multiple experts compete across ranks — with a single GPU/expert
    # there is nothing to balance and the aux gradient only corrupts routing.
    aux_weight = config.aux_loss_weight if (step >= 1000 and world_size > 1) else 0.0
    total_loss = ar_loss + aux_weight * aux_loss
    
    metrics = {
        "ar_loss": ar_loss.item(),
        "aux_loss": aux_loss.item(),
        "aux_weight": aux_weight,
        "total_loss": total_loss.item(),
        "route_entropy": -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item(),
        "porto_prob": probs[:, 0].mean().item(),
        "beijing_prob": probs[:, 1].mean().item(),
    }
    
    return total_loss, metrics

def train_expert(args: argparse.Namespace) -> None:
    """Main training loop for geographic experts."""
    
    # Setup distributed training if specified
    if args.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank = 0
        world_size = 1
        
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config_data = load_config()
    expert_config = ExpertConfig()
    
    logging.info(f"[{ts()}] Rank {rank}: Starting Phase 2 expert training")
    
    # Multi-tenant scheduling
    if args.multi_tenant and args.expert is None:
        jobs = [
            ExpertJob("porto", n_samples=1_700_000, graph_parts=4, remaining_work=1_700_000),
            ExpertJob("beijing", n_samples=120_000, graph_parts=1, remaining_work=120_000)
        ]
        jobs = allocate_workers(jobs, world_size)
        my_expert = expert_for_rank(rank, jobs)
        logging.info(f"[{ts()}] Rank {rank}: Multi-tenant assigned to expert '{my_expert}'")

        # Log worker allocation
        if rank == 0:
            for job in jobs:
                logging.info(f"[{ts()}] Expert '{job.name}': {job.workers} workers, {job.remaining_work} samples")

        # Build per-expert rank groups so workers for the same expert use DDP
        expert_to_ranks: Dict[str, List[int]] = {}
        current = 0
        for job in jobs:
            expert_to_ranks[job.name] = list(range(current, current + job.workers))
            current += job.workers

        my_ranks = expert_to_ranks[my_expert]
        expert_rank = my_ranks.index(rank)
        expert_world_size = len(my_ranks)
        is_group_leader = (expert_rank == 0)
        expert_group = dist.new_group(ranks=my_ranks) if (args.distributed and expert_world_size > 1) else None
    else:
        my_expert = args.expert or "porto"
        expert_rank = 0
        expert_world_size = 1
        is_group_leader = (rank == 0)
        expert_group = None

    effective_max_steps = MAX_STEPS_PER_EXPERT.get(my_expert, args.max_steps)
    logging.info(f"[{ts()}] Rank {rank}: expert='{my_expert}' max_steps={effective_max_steps}")

    # Load frozen backbone
    backbone = TrajectoryBackbone(
        vocab_size=config_data["tokenizer"]["vocab_size"],
        d_model=config_data["tokenizer"]["d_model"],
        n_layers=config_data["tokenizer"]["n_layers"],
        n_heads=config_data["tokenizer"]["n_heads"],
        max_seq_len=config_data["tokenizer"]["max_seq_len"],
    ).to(device)
    
    # Load Phase 1 checkpoint if available
    checkpoint_path = Path("data/checkpoints/ckpt_final.pt")
    if checkpoint_path.exists():
        logging.info(f"[{ts()}] Loading Phase 1 backbone from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            backbone.load_state_dict(checkpoint["model"], strict=False)
            logging.info(f"[{ts()}] Successfully loaded backbone checkpoint from {checkpoint_path}")
        except Exception as e:
            logging.warning(f"[{ts()}] Failed to load checkpoint: {e}")
            logging.info(f"[{ts()}] Continuing with randomly initialized backbone")
    else:
        logging.warning(f"[{ts()}] No Phase 1 checkpoint found, using random backbone")
        
    # Freeze backbone 
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
        
    # Load city road graph
    city_graph = load_city_graph(my_expert, use_osm=args.use_osm)

    # Fail fast if --use-road-enhanced was requested but road_graph.pt is missing.
    # A MockRoadGraph here means the prebuilt file was not found, which would silently
    # zero-out all road context and waste the entire training run.
    if args.use_road_enhanced and isinstance(city_graph, MockRoadGraph):
        raise RuntimeError(
            f"--use-road-enhanced requires a pre-built road graph for '{my_expert}' "
            f"at data/road_enhanced_gold/{my_expert}/road_graph.pt, but the file was "
            f"not found or failed to load. "
            f"Run: python scripts/etl.py --stage road-gold --regions {my_expert}"
        )

    # PyG Data.num_nodes is a property (int), MockRoadGraph.num_nodes() is a method
    if isinstance(city_graph, MockRoadGraph):
        n_segments = city_graph.num_nodes()
    else:
        n_segments = city_graph.num_nodes  # PyG Data property
    
    # Initialize expert and router
    expert = GeographicExpert(
        d_model=expert_config.d_model,
        entity_dim=expert_config.entity_dim, 
        gat_hidden=expert_config.gat_hidden,
        gat_heads=expert_config.gat_heads,
        n_seg=n_segments
    ).to(device)
    
    router = GeographicRouter(
        d_model=expert_config.d_model,
        n_experts=len(EXPERTS),
        prior_strength=expert_config.prior_strength
    ).to(device)
    
    # Wrap in DDP when this expert has multiple workers so gradients are reduced
    # across ranks in the same expert group. Single-worker experts skip DDP.
    if expert_group is not None:
        expert = DDP(expert, device_ids=[rank % torch.cuda.device_count()], process_group=expert_group)
        router = DDP(router, device_ids=[rank % torch.cuda.device_count()], process_group=expert_group)

    # Optimizer
    trainable_params = list(expert.parameters()) + list(router.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=expert_config.lr,
        weight_decay=expert_config.weight_decay
    )
    
    # Dataset and dataloader with road enhancement
    road_enhanced_dir = Path("data/road_enhanced_gold")
    data_dir = road_enhanced_dir if args.use_road_enhanced else None
    if args.use_road_enhanced:
        logging.info(f"Using road-enhanced gold shards from {road_enhanced_dir}")
    logging.info(f"[{ts()}] Building TrajectoryDataset for expert '{my_expert}'...")
    dataset = TrajectoryDataset(regions=[my_expert], data_dir=data_dir)
    logging.info(f"[{ts()}] Dataset ready: {len(dataset)} samples")

    # Enhance dataset with road information if available
    if hasattr(dataset, 'trips') and OSM_AVAILABLE:
        logging.info(f"[{ts()}] Enhancing {len(dataset.trips)} trips with road segment data...")
        enhanced_trips = []
        for trip in dataset.trips:
            enhanced_trip = enhance_trajectory_with_roads(trip, my_expert)
            enhanced_trips.append(enhanced_trip)
        dataset.trips = enhanced_trips
        logging.info(f"[{ts()}] Road enhancement completed")

    logging.info(f"[{ts()}] Creating DataLoader (batch_size={args.batch_size}, num_workers=4)...")
    if expert_world_size > 1:
        sampler: Optional[DistributedSampler] = DistributedSampler(
            dataset, num_replicas=expert_world_size, rank=expert_rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
    logging.info(f"[{ts()}] DataLoader ready: {len(dataloader)} batches per epoch")

    # Training loop
    expert.train()
    router.train()

    step = 0
    metrics_sum = {}

    # Unwrap DDP for direct method calls and checkpoint saving.
    # Gradients still flow through the underlying parameters and are all-reduced
    # by DDP's backward hooks; the DDP-wrapped objects are used for forward/loss.
    expert_module = expert.module if isinstance(expert, DDP) else expert
    router_module = router.module if isinstance(router, DDP) else router

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        expert_idx = EXPERTS.index(my_expert)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                logging.info(f"[{ts()}] Epoch {epoch}: first batch received, starting training loop")
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Set region prior from the actual expert assigned to this rank
            B = batch["tok"].size(0)
            region_prior = torch.zeros(B, len(EXPERTS), device=device)
            region_prior[:, expert_idx] = 1.0
            batch["region_prior"] = region_prior

            # Compute road node embeddings with gradients each step so GAT layers learn.
            # Graph structure is static but seg_emb/entity_proj/GATConv weights are not.
            node_emb = expert_module.road_context(city_graph)

            # Forward pass through frozen backbone
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = backbone.trunk(
                    batch["tok"], batch["dt"], batch["mod"], batch["dow"],
                    is_causal=True
                )
                batch["backbone_features"] = h
                batch["target_tokens"] = batch["tok"]  # AR target

            # Expert forward pass and loss
            loss, metrics = expert_loss(
                expert, router, batch, node_emb, expert_config,
                backbone_tok_emb=backbone.embed.tok.weight.detach(),
                step=step,
                world_size=world_size,
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, expert_config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Accumulate metrics
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0) + value
                
            step += 1
            
            # Logging
            if step % args.log_interval == 0:
                avg_metrics = {k: v / args.log_interval for k, v in metrics_sum.items()}
                
                log_msg = f"[{ts()}] Rank {rank} Expert {my_expert} Step {step:6d}"
                for key, value in avg_metrics.items():
                    log_msg += f" | {key}: {value:.4f}"
                logging.info(log_msg)
                
                metrics_sum.clear()
                
            # Checkpointing — save from each expert's group leader, not just global rank 0
            if step % args.save_interval == 0:
                save_expert_checkpoint(expert_module, router_module, optimizer, step, my_expert, is_group_leader)
                
            if step >= effective_max_steps:
                break

        if step >= effective_max_steps:
            break
            
    logging.info(f"[{ts()}] Rank {rank}: Expert training completed after {step} steps")

def save_expert_checkpoint(expert: nn.Module, router: nn.Module, optimizer: torch.optim.Optimizer,
                          step: int, expert_name: str, is_group_leader: bool = True) -> None:
    """Save expert checkpoint. Only the group leader for each expert saves."""
    if not is_group_leader:
        return
        
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        "expert_state_dict": expert.state_dict(),
        "router_state_dict": router.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "expert_name": expert_name
    }
    
    checkpoint_path = checkpoint_dir / f"expert_{expert_name}_step_{step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"[{ts()}] Saved checkpoint to {checkpoint_path}")

# ========================================================================================
# Inference
# ========================================================================================

S3_CHECKPOINTS_BUCKET = "s3://geo-trajectories-checkpoints"


def _s3_cmd_base() -> list:
    """Build base aws s3 command with Nebius profile and endpoint."""
    cmd = ["aws", "s3", "--profile", "nebius"]
    endpoint = os.environ.get("S3_ENDPOINT_URL", "")
    if endpoint:
        cmd += ["--endpoint-url", endpoint]
    return cmd


def _ensure_local(local_path: Path, s3_uri: str) -> None:
    """Download a single file from S3 if it does not exist locally."""
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{ts()} [s3] Downloading {s3_uri} → {local_path} ...")
    subprocess.run(_s3_cmd_base() + ["cp", s3_uri, str(local_path)], check=True)


def _ensure_expert_ckpts(expert_name: str, ckpt_dir: Path) -> None:
    """Sync expert_<name>_step_*.pt from S3 if none exist locally."""
    if list(ckpt_dir.glob(f"expert_{expert_name}_step_*.pt")):
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"{ts()} [s3] Syncing expert '{expert_name}' checkpoints from {S3_CHECKPOINTS_BUCKET} ...")
    subprocess.run(
        _s3_cmd_base() + [
            "sync", S3_CHECKPOINTS_BUCKET, str(ckpt_dir),
            "--exclude", "*",
            "--include", f"expert_{expert_name}_step_*.pt",
        ],
        check=True,
    )


def inference(args: argparse.Namespace) -> None:
    """Load expert + backbone checkpoints and run inference on a sample Porto trajectory.

    Prints:
    - Geographic routing probabilities
    - Top-5 next-token predictions from the expert-fused AR head
    """
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_name = args.expert or "porto"

    config_data = load_config()
    tok_cfg = config_data["tokenizer"]
    vocab_size = tok_cfg["vocab_size"]
    expert_config = ExpertConfig()

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone_ckpt_path = Path("data/checkpoints/ckpt_final.pt")
    _ensure_local(backbone_ckpt_path, f"{S3_CHECKPOINTS_BUCKET}/ckpt_final.pt")
    if not backbone_ckpt_path.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {backbone_ckpt_path}. "
            f"Run: make download-ckpt"
        )

    backbone = TrajectoryBackbone(
        vocab_size=vocab_size,
        d_model=tok_cfg["d_model"],
        n_layers=tok_cfg["n_layers"],
        n_heads=tok_cfg["n_heads"],
        max_seq_len=tok_cfg["max_seq_len"],
    ).to(device)
    backbone_ckpt = torch.load(backbone_ckpt_path, map_location=device, weights_only=True)
    backbone.load_state_dict(backbone_ckpt["model"])
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    # ── Expert checkpoint ─────────────────────────────────────────────────────
    ckpt_dir = Path("data/checkpoints")
    _ensure_expert_ckpts(expert_name, ckpt_dir)
    expert_ckpts = sorted(ckpt_dir.glob(f"expert_{expert_name}_step_*.pt"))
    if not expert_ckpts:
        raise FileNotFoundError(
            f"No expert checkpoint found for '{expert_name}' in {ckpt_dir} or S3. "
            f"Run: python scripts/experts.py --expert {expert_name}"
        )
    expert_ckpt_path = expert_ckpts[-1]  # latest step
    expert_ckpt = torch.load(expert_ckpt_path, map_location=device, weights_only=False)

    # ── Road graph + expert models ────────────────────────────────────────────
    city_graph = load_city_graph(expert_name, use_osm=False)
    n_segments = city_graph.num_nodes() if isinstance(city_graph, MockRoadGraph) else city_graph.num_nodes

    expert = GeographicExpert(
        d_model=expert_config.d_model,
        entity_dim=expert_config.entity_dim,
        gat_hidden=expert_config.gat_hidden,
        gat_heads=expert_config.gat_heads,
        n_seg=n_segments,
    ).to(device)
    router = GeographicRouter(
        d_model=expert_config.d_model,
        n_experts=len(EXPERTS),
        prior_strength=expert_config.prior_strength,
    ).to(device)
    expert.load_state_dict(expert_ckpt["expert_state_dict"])
    router.load_state_dict(expert_ckpt["router_state_dict"])
    expert.eval()
    router.eval()

    n_params = sum(p.numel() for p in expert.parameters()) + sum(p.numel() for p in router.parameters())
    print(f"{ts()} [inference] Device       : {device}")
    print(f"{ts()} [inference] Expert       : {expert_name}")
    print(f"{ts()} [inference] Expert ckpt  : {expert_ckpt_path} (step {expert_ckpt.get('step', '?')})")
    print(f"{ts()} [inference] Backbone     : {backbone_ckpt_path} (step {backbone_ckpt.get('step', '?')})")
    print(f"{ts()} [inference] Road nodes   : {n_segments}  (mock={isinstance(city_graph, MockRoadGraph)})")
    print(f"{ts()} [inference] Expert+router: {n_params / 1e6:.1f} M parameters")

    # ── Sample: Boavista → Aliados, Porto, 08:00 Tuesday ─────────────────────
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

    try:
        import h3 as h3lib
        h3_res  = tok_cfg["h3_resolution"]
        subhash = tok_cfg["subhash_vocab_size"]
        def _latlng_to_token(lat: float, lng: float) -> int:
            cell   = h3lib.latlng_to_cell(lat, lng, h3_res) if hasattr(h3lib, "latlng_to_cell") \
                     else h3lib.geo_to_h3(lat, lng, h3_res)
            h3_int = int(cell, 16)
            return 4 + (h3_int ^ (h3_int >> 17)) % subhash
        spatial_tokens = [_latlng_to_token(lat, lng) for lat, lng in waypoints]
    except ImportError:
        spatial_tokens = list(range(4, 4 + len(waypoints)))  # synthetic fallback

    sample_tokens = [BOS] + spatial_tokens
    L = len(sample_tokens)

    tok     = torch.tensor([sample_tokens], dtype=torch.long, device=device)
    dt      = torch.tensor([[0] + [4] * (L - 1)], dtype=torch.long, device=device)
    mod     = torch.full((1, L), 480, dtype=torch.long, device=device)  # 08:00 = 480 min
    dow     = torch.full((1, L), 1,   dtype=torch.long, device=device)  # Tuesday
    seg_ids = torch.zeros(1, L, dtype=torch.long, device=device)

    expert_idx   = EXPERTS.index(expert_name)
    region_prior = torch.zeros(1, len(EXPERTS), device=device)
    region_prior[:, expert_idx] = 1.0

    _SPECIAL = {0: "PAD", 1: "BOS", 2: "EOS", 3: "MASK"}
    def _label(tid: int) -> str:
        return _SPECIAL.get(tid, f"h3:{tid}")

    print(f"\n{ts()} [inference] Porto trajectory (len={L}, 08:00 Tue, dt_bucket=4)")
    print("  route : Rotunda da Boavista → Praça da Liberdade")

    with torch.no_grad():
        # Backbone hidden states
        h = backbone.trunk(tok, dt, mod, dow, is_causal=True)   # [1, L, d_model]

        # Geographic routing
        pooled_h = h.mean(dim=1)                                 # [1, d_model]
        top1, route_probs, _ = router(pooled_h, region_prior)

        # Road-aware fusion
        node_emb = expert.road_context(city_graph)               # [n_nodes, gat_hidden]
        fused_h  = expert(h, seg_ids, node_emb)                  # [1, L, d_model]

        # AR next-token (weight-tied to backbone tok_emb)
        proj     = expert.tok_head(fused_h[:, -1])               # [1, d_model]
        logits   = (proj @ backbone.embed.tok.weight.T)[0]       # [vocab_size]
        ar_probs = F.softmax(logits, dim=-1)
        topk     = torch.topk(logits, 5)
        topk_ids   = topk.indices.tolist()
        topk_probs = ar_probs[topk.indices].tolist()
        greedy_id  = logits.argmax().item()
        entropy    = -(ar_probs * ar_probs.log().clamp(min=-1e9)).sum().item()

    print(f"\n{ts()} [inference] Geographic routing")
    print(f"  selected expert : {EXPERTS[top1.item()]} (idx={top1.item()})")
    for i, name in enumerate(EXPERTS):
        print(f"  P({name:<8}) : {route_probs[0, i].item():.4%}")

    print(f"\n{ts()} [inference] AR head — expert-fused next-token prediction")
    print(f"  input  : {[_label(t) for t in sample_tokens]}")
    print(f"  greedy : {_label(greedy_id)} (id={greedy_id})")
    print(f"  entropy: {entropy:.2f} nats  (max={math.log(vocab_size):.2f})")
    print(f"  {'rank':<6} {'id':<8} {'label':<12} {'prob':>8}")
    print(f"  {'-'*38}")
    for rank, (tid, prob) in enumerate(zip(topk_ids, topk_probs), 1):
        print(f"  {rank:<6} {tid:<8} {_label(tid):<12} {prob:>8.4%}")


# ========================================================================================
# CLI and Main
# ========================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2: Geographic Expert Training")
    
    # Expert selection
    parser.add_argument("--expert", choices=EXPERTS, help="Which expert to train")
    parser.add_argument("--multi-tenant", action="store_true", 
                       help="Use multi-tenant scheduling across all experts")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    
    # Distributed training
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    
    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-interval", type=int, default=2000, help="Checkpoint save frequency")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Debug mode with minimal data")
    
    # OSM and road graph options
    parser.add_argument("--use-osm", action="store_true", default=True,
                       help="Use real OpenStreetMap data (default: True)")
    parser.add_argument("--no-osm", dest="use_osm", action="store_false",
                       help="Disable OSM integration, use mock road graphs")
    parser.add_argument("--use-road-enhanced", action="store_true",
                       help="Load pre-computed road-enhanced gold shards from data/road_enhanced_gold/")

    # Inference
    parser.add_argument("--infer", action="store_true",
                       help="Run inference on a sample trajectory using saved checkpoints")

    return parser.parse_args()

def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.debug:
        args.max_steps = 100
        args.log_interval = 10
        logging.info("Debug mode: reduced steps and frequent logging")
        
    # SIGTERM handler for preemptible training
    def sigterm_handler(signum, frame):
        logging.info(f"[{ts()}] Received SIGTERM, saving checkpoint and exiting...")
        # TODO: Add checkpoint saving logic here
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    if args.infer:
        inference(args)
        return

    try:
        train_expert(args)
    except KeyboardInterrupt:
        logging.info(f"[{ts()}] Training interrupted by user")
    except Exception as e:
        logging.error(f"[{ts()}] Training failed: {e}")
        raise

if __name__ == "__main__":
    main()