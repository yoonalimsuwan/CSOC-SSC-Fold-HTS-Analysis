#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v29.1 — Scalable Multi‑GPU Multimer Folding Engine (N > 100,000)
#                PyG‑accelerated Sparse Graph + Equivariant Geometry & Physics
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# V29.1 replaces the custom chunked cdist neighbour search with
# PyTorch Geometric’s highly optimised radius_graph / radius functions,
# achieving near‑O(N) scaling while retaining the full physics and
# SOC dynamics of V28/V29.
# =============================================================================

import os, math, time, random, argparse, logging, urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# PyTorch Geometric – accelerated neighbour search
from torch_cluster import radius_graph, radius

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(name="CSOC‑SSC_V29.1", local_rank=-1):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '[%(asctime)s] [Rank %(process)d] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)
    return logger

# ──────────────────────────────────────────────────────────────────────────────
# Constants (same as v28, full dictionaries)
# ──────────────────────────────────────────────────────────────────────────────
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    'X': 0.0
}

RESIDUE_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5}

RAMACHANDRAN_PRIORS = {
    'general': {'phi': -60.0, 'psi': -45.0, 'width': 25.0},
    'G':       {'phi': -75.0, 'psi': -60.0, 'width': 40.0},
    'P':       {'phi': -65.0, 'psi': -30.0, 'width': 20.0},
}

AA_3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
}

LJ_PARAMS = {
    'C':   (1.9080, 0.0860),   'CA':  (1.9080, 0.0860),   'CB':  (1.9080, 0.0860),
    'CG':  (1.9080, 0.0860),   'CD':  (1.9080, 0.0860),   'CE':  (1.9080, 0.0860),
    'CZ':  (1.9080, 0.0860),   'CH2': (1.9080, 0.0860),
    'N':   (1.8240, 0.1700),   'ND':  (1.8240, 0.1700),   'NE':  (1.8240, 0.1700),
    'NH1': (1.8240, 0.1700),   'NH2': (1.8240, 0.1700),
    'O':   (1.6612, 0.2100),   'OD':  (1.6612, 0.2100),   'OE':  (1.6612, 0.2100),
    'OH':  (1.6612, 0.2100),
    'S':   (2.0000, 0.2500),   'SG':  (2.0000, 0.2500),
}

RESIDUE_TOPOLOGY = {
    'G': [],
    'A': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0)],
    'S': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('OG', 'OH', 1, 1.43, 109.5, (-2,-1,0), 0.0)],
    'C': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('SG', 'SG', 1, 1.81, 109.5, (-2,-1,0), 0.0)],
    'V': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG1','CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0)],
    'T': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('OG1','OH', 1, 1.43, 109.5, (-2,-1,0), 0.0),
          ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0)],
    'L': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CD1','CB', 2, 1.53, 109.5, (-1,0,1), 0.0),
          ('CD2','CB', 2, 1.53, 109.5, (-1,0,1), 2.0)],
    'I': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG1','CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0),
          ('CD1','CB', 2, 1.53, 109.5, (-1,0,1), 0.0)],
    'M': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('SD', 'S',  2, 1.81, 109.5, (-1,0,1), 0.0),
          ('CE', 'CB', 3, 1.81, 109.5, (-2,-1,0), 0.0)],
    'F': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
          ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
          ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
          ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
          ('CZ', 'CB', 5, 1.40, 120.0, (3,2,1), 0.0)],
    'Y': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
          ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
          ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
          ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
          ('CZ', 'CB', 5, 1.40, 120.0, (3,2,1), 0.0),
          ('OH', 'OH', 6, 1.36, 120.0, (4,3,2), 0.0)],
    'W': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
          ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
          ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
          ('NE1','N',  3, 1.38, 120.0, (2,1,0), 0.0),
          ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
          ('CE3','CB', 5, 1.40, 120.0, (2,1,0), 2.0),
          ('CZ2','CB', 6, 1.40, 120.0, (3,2,1), 0.0),
          ('CZ3','CB', 7, 1.40, 120.0, (5,4,2), 0.0),
          ('CH2','CB', 8, 1.40, 120.0, (6,3,2), 0.0)],
    'D': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'C',  1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('OD1','O',  2, 1.25, 120.0, (-1,0,1), 0.0),
          ('OD2','O',  2, 1.25, 120.0, (-1,0,1), 2.0)],
    'E': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('CD', 'C',  2, 1.52, 109.5, (-1,0,1), 0.0),
          ('OE1','O',  3, 1.25, 120.0, (2,1,0), 0.0),
          ('OE2','O',  3, 1.25, 120.0, (2,1,0), 2.0)],
    'N': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'C',  1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('OD1','O',  2, 1.25, 120.0, (-1,0,1), 0.0),
          ('ND2','N',  2, 1.33, 120.0, (-1,0,1), 2.0)],
    'Q': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('CD', 'C',  2, 1.52, 109.5, (-1,0,1), 0.0),
          ('OE1','O',  3, 1.25, 120.0, (2,1,0), 0.0),
          ('NE2','N',  3, 1.33, 120.0, (2,1,0), 2.0)],
    'K': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('CD', 'CB', 2, 1.52, 109.5, (-1,0,1), 0.0),
          ('CE', 'CB', 3, 1.52, 109.5, (-2,-1,0), 0.0),
          ('NZ', 'N',  4, 1.47, 109.5, (-1,0,1), 0.0)],
    'R': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
          ('CD', 'CB', 2, 1.52, 109.5, (-1,0,1), 0.0),
          ('NE', 'N',  3, 1.46, 109.5, (-2,-1,0), 0.0),
          ('CZ', 'C',  4, 1.33, 125.0, (-1,0,1), 0.0),
          ('NH1','N',  5, 1.33, 120.0, (4,3,2), 0.0),
          ('NH2','N',  5, 1.33, 120.0, (4,3,2), 2.0)],
    'H': [('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.50, 109.5, (-2,-1,0), 0.0),
          ('ND1','N',  2, 1.38, 120.0, (-1,0,1), 0.0),
          ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
          ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
          ('NE2','N',  4, 1.38, 120.0, (2,1,0), 2.0)],
    'P': [('CB', 'CB', 0, 1.53, 104.5, (-1,-2,-3), 0.0),
          ('CG', 'CB', 1, 1.50, 104.5, (-2,-1,0), 0.0),
          ('CD', 'CB', 2, 1.50, 104.5, (-1,0,1), 0.0)],
}

MAX_CHI = 4
RESIDUE_NCHI = {
    'A':0,'G':0,'S':1,'C':1,'V':1,'T':1,'L':2,'I':2,'M':3,'F':2,'Y':2,'W':2,
    'D':2,'E':3,'N':2,'Q':3,'K':4,'R':4,'H':2,'P':2
}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (V29.1)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class V29Config:
    # Distributed
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Architecture
    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    # EGNN decoder
    egnn_layers: int = 4
    egnn_hidden: int = 128
    egnn_edge_dim: int = 32
    egnn_cutoff: float = 15.0

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # Refinement
    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    w_avalanche: float = 0.2

    # Backbone energy weights
    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0

    # Full‑atom energy weights
    w_lj: float = 50.0
    w_coulomb: float = 5.0
    w_torsion: float = 10.0

    # SOC
    w_soc_contact: float = 0.3
    kernel_lambda: float = 12.0
    rebuild_interval: int = 100

    # Alpha modifiers
    alpha_mod_bond: float = 0.1
    alpha_mod_angle: float = 0.05
    alpha_mod_rama: float = 0.2
    alpha_mod_clash: float = 0.1
    alpha_mod_hbond: float = 0.1
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1

    # Multimer
    chain_break_weight: float = 1.0

    # Scalable neighbour search (V29.1 – PyG)
    sparse_cutoff: float = 12.0
    max_neighbors: int = 64

    # Utilities
    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200
    checkpoint_dir: str = "./v29_ckpt"
    out_pdb: str = "refined_v29.pdb"

# ──────────────────────────────────────────────────────────────────────────────
# PDB Fetcher (Multimer‑aware) – same as V28
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Backbone:
    ca: np.ndarray
    seq: str
    chain_id: str = ""
    native_coords: Optional[np.ndarray] = None

class MultimerPDBFetcher:
    @staticmethod
    def fetch(pdb_id: str) -> Tuple[List[Backbone], List[str]]:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={'User-Agent': 'CSOC-SSC_V29.1'})
        chains_data = {}
        chain_order = []
        try:
            with urllib.request.urlopen(req) as response:
                lines = response.read().decode('utf-8').split('\n')
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    chain = line[21].strip()
                    res_name = line[17:20].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if chain not in chains_data:
                        chains_data[chain] = {'res':[], 'coords':[]}
                        chain_order.append(chain)
                    chains_data[chain]['res'].append(res_name)
                    chains_data[chain]['coords'].append([x,y,z])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PDB {pdb_id}: {e}")

        backbones = []
        for chain in chain_order:
            seq = "".join([AA_3_TO_1.get(r, 'X') for r in chains_data[chain]['res']])
            coords = np.array(chains_data[chain]['coords'], dtype=np.float32)
            rand_coords = coords + np.random.randn(*coords.shape) * 10.0
            backbones.append(Backbone(ca=rand_coords, seq=seq, chain_id=chain, native_coords=coords))
        return backbones, chain_order

# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder (same as V28)
# ──────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class FlashGeometryBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Linear(dim * ff_mult, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                      dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        h = self.proj(attn_out)
        x = self.norm1(x + self.dropout(h))
        y = self.ffn(x)
        x = self.norm2(x + self.dropout(y))
        return x

class FlashSequenceEncoder(nn.Module):
    def __init__(self, dim, depth, heads, ff_mult):
        super().__init__()
        self.embed = nn.Embedding(len(AA_VOCAB), dim)
        self.pos_enc = SinusoidalPositionalEncoding(dim)
        self.layers = nn.ModuleList([FlashGeometryBlock(dim, heads, ff_mult) for _ in range(depth)])

    def forward(self, seq_ids):
        x = self.embed(seq_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x

class AdaptiveAlphaField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, latent):
        a = torch.sigmoid(self.net(latent))
        a = 0.5 + 2.5 * a.squeeze(-1)
        return torch.clamp(a, 0.5, 3.0)

# ──────────────────────────────────────────────────────────────────────────────
# EGNN Decoder (Equivariant) – same as V28
# ──────────────────────────────────────────────────────────────────────────────
class EGNNLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, h, x, edge_index, edge_dist):
        src, dst = edge_index
        edge_attr = self.edge_mlp(edge_dist.unsqueeze(-1))
        m_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        m = self.node_mlp(m_input)
        h_agg = torch.zeros_like(h)
        h_agg = h_agg.index_add(0, dst, m)
        coord_weight = self.coord_mlp(m_input)
        dir_vec = x[src] - x[dst]
        coord_update = coord_weight * dir_vec
        x_agg = torch.zeros_like(x)
        x_agg = x_agg.index_add(0, dst, coord_update)
        return h + h_agg, x + x_agg

class EquivariantDecoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim, num_layers, cutoff):
        super().__init__()
        self.layers = nn.ModuleList([EGNNLayer(node_dim, hidden_dim, edge_dim) for _ in range(num_layers)])
        self.cutoff = cutoff
        self.init_coord = nn.Linear(node_dim, 3)

    def build_edges(self, x):
        # EGNN decoder still uses dense cdist for smaller systems.
        # For huge systems replace with PyG radius_graph as needed.
        D = torch.cdist(x, x)
        triu = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        mask = (D < self.cutoff) & triu
        src, dst = torch.where(mask)
        edge_dist = D[src, dst]
        src_full = torch.cat([src, dst])
        dst_full = torch.cat([dst, src])
        edge_dist_full = torch.cat([edge_dist, edge_dist])
        return torch.stack([src_full, dst_full], dim=0), edge_dist_full

    def forward(self, h, initial_coords=None):
        B, N, _ = h.shape
        if initial_coords is None:
            x = self.init_coord(h)
        else:
            x = initial_coords
        x_out = []
        for b in range(B):
            hi = h[b]
            xi = x[b]
            ei, ed = self.build_edges(xi)
            for layer in self.layers:
                hi, xi = layer(hi, xi, ei, ed)
            x_out.append(xi)
        return torch.stack(x_out, dim=0)

# ──────────────────────────────────────────────────────────────────────────────
# CSOC Controller (same)
# ──────────────────────────────────────────────────────────────────────────────
class CSOCController:
    def __init__(self):
        self.prev_coords = None

    def sigma(self, coords):
        if self.prev_coords is None:
            self.prev_coords = coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta = torch.norm(coords - self.prev_coords, dim=-1).mean()
        self.prev_coords = coords.detach().clone()
        return delta

    def temperature(self, sigma, base_T, target):
        dev = (sigma - target) / 0.5
        T = base_T + 2000.0 * torch.sigmoid(dev)
        return torch.clamp(T, base_T * 0.5, 3000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RG (same)
# ──────────────────────────────────────────────────────────────────────────────
class DiffRGRefiner:
    def __init__(self, factor=4):
        self.factor = factor
    def forward(self, coords):
        L = coords.shape[0]
        f = self.factor
        m = L // f * f
        if m == 0: return coords
        x = coords[:m].permute(1,0).unsqueeze(0)
        pooled = F.avg_pool1d(x, kernel_size=f, stride=f)
        up = F.interpolate(pooled, size=L, mode='linear', align_corners=True)
        return up.squeeze(0).permute(1,0)

# ──────────────────────────────────────────────────────────────────────────────
# V29.1 Sparse Graph Builders – PyG accelerated
# ──────────────────────────────────────────────────────────────────────────────
def sparse_edges(coords: torch.Tensor, cutoff: float, k: int,
                 exclude_self: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build undirected sparse edge list using PyG's radius_graph.
    Returns (edge_index (2, E), edge_dist (E,)).
    """
    if coords.shape[0] == 0:
        return torch.empty((2,0), dtype=torch.long, device=coords.device), \
               torch.empty((0,), device=coords.device)

    # radius_graph returns source->target edges (no self loops by default)
    edge_index = radius_graph(coords, r=cutoff, max_num_neighbors=k,
                              flow='source_to_target')
    src, dst = edge_index[0], edge_index[1]
    edge_dist = torch.norm(coords[src] - coords[dst], dim=-1)
    return edge_index, edge_dist

def cross_sparse_edges(coords1: torch.Tensor, coords2: torch.Tensor,
                       cutoff: float, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sparse edges between coords1 and coords2 using PyG's radius.
    Returns (edge_index (2, E), edge_dist (E,)).
    """
    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        return torch.empty((2,0), dtype=torch.long, device=coords1.device), \
               torch.empty((0,), device=coords1.device)

    row, col = radius(coords1, coords2, r=cutoff, max_num_neighbors=k)
    edge_index = torch.stack([row, col], dim=0)
    edge_dist = torch.norm(coords1[row] - coords2[col], dim=-1)
    return edge_index, edge_dist

# ──────────────────────────────────────────────────────────────────────────────
# Full‑atom builder (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def build_sidechain_atoms(ca, seq, chi_angles):
    device = ca.device
    L = ca.shape[0]
    v = ca[1:] - ca[:-1]
    v_norm = F.normalize(v, dim=-1, eps=1e-8)
    N = torch.zeros_like(ca)
    C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm
    N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm
    C[-1] = ca[-1] + 1.52 * v_norm[-1]

    all_coords = []
    all_types = []
    for i, aa in enumerate(seq):
        if aa == 'G':
            res_atoms = torch.stack([N[i], ca[i], C[i]], dim=0)
            types = ['N', 'CA', 'C']
            all_coords.append(res_atoms)
            all_types.append(types)
            continue
        n_i, ca_i, c_i = N[i], ca[i], C[i]
        v1 = n_i - ca_i
        v2 = c_i - ca_i
        cb_dir = -(v1 + v2)
        cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        cb_pos = ca_i + 1.53 * cb_dir
        local_atoms = [n_i, ca_i, c_i, cb_pos]
        local_types = ['N', 'CA', 'C', 'CB']
        topo = RESIDUE_TOPOLOGY.get(aa, [])
        chi_idx = 0
        for (atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral_ang0) in topo:
            parent_abs = 3 + parent_idx
            def map_ref(idx):
                if idx == -1: return 0
                if idx == -2: return 1
                if idx == -3: return 2
                if idx == -4: return 3
                return 3 + idx
            a_idx, b_idx, c_idx = map_ref(ref_tuple[0]), map_ref(ref_tuple[1]), map_ref(ref_tuple[2])
            p_a, p_b, p_c = local_atoms[a_idx], local_atoms[b_idx], local_atoms[c_idx]
            bc = p_c - p_b
            bc_norm = F.normalize(bc, dim=-1, eps=1e-8)
            ref_vec = torch.tensor([1.0,0.0,0.0], device=device)
            if torch.abs(torch.dot(bc_norm, ref_vec)) > 0.9:
                ref_vec = torch.tensor([0.0,1.0,0.0], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-8)
            chi_val = chi_angles[i, chi_idx]
            total_angle = dihedral_ang0 + chi_val
            cos_a, sin_a = torch.cos(total_angle), torch.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            ang = torch.tensor(bond_ang_deg * math.pi / 180.0, device=device)
            bond_dir = torch.cos(ang) * bc_norm + torch.sin(ang) * rotated_perp
            new_pos = p_c + bond_len * bond_dir
            local_atoms.append(new_pos)
            local_types.append(atom_type)
            chi_idx += 1
        all_coords.append(torch.stack(local_atoms, dim=0))
        all_types.append(local_types)
    return all_coords, all_types

def get_full_atom_coords_and_types(ca, seq, chi_angles):
    res_coords, res_types = build_sidechain_atoms(ca, seq, chi_angles)
    coords_list, types_list, res_idx_list = [], [], []
    for i, (rc, rt) in enumerate(zip(res_coords, res_types)):
        coords_list.append(rc)
        types_list.extend(rt)
        res_idx_list.append(torch.full((rc.shape[0],), i, dtype=torch.long, device=ca.device))
    all_coords = torch.cat(coords_list, dim=0)
    res_indices = torch.cat(res_idx_list, dim=0)
    return all_coords, types_list, res_indices

# ──────────────────────────────────────────────────────────────────────────────
# Physics energy terms (adapted for PyG sparse graphs)
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_backbone(ca):
    L = ca.shape[0]
    v = ca[1:] - ca[:-1]
    v_norm = F.normalize(v, dim=-1, eps=1e-8)
    N = torch.zeros_like(ca)
    C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm
    N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm
    C[-1] = ca[-1] + 1.52 * v_norm[-1]
    offset = torch.tensor([0.0, 1.24, 0.0], device=ca.device)
    O = torch.zeros_like(ca)
    for i in range(L):
        if i < L-1:
            ca_c = C[i] - ca[i]
            ca_n = N[i] - ca[i]
            perp = torch.cross(ca_c, ca_n, dim=-1)
            perp_norm = torch.norm(perp)
            if perp_norm > 1e-6:
                perp = perp / perp_norm
            O[i] = C[i] + 1.24 * perp
        else:
            O[i] = C[i] + offset
    return {'N': N, 'CA': ca, 'C': C, 'O': O}

def dihedral_angle(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = torch.cross(b1n, v, dim=-1)
    y = (y * w).sum(-1)
    return torch.atan2(y + 1e-8, x + 1e-8)

def compute_phi_psi(atoms):
    N, CA, C = atoms['N'], atoms['CA'], atoms['C']
    L = CA.shape[0]
    phi = torch.zeros(L, device=CA.device)
    psi = torch.zeros(L, device=CA.device)
    if L > 2:
        phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi * 180.0 / math.pi, psi * 180.0 / math.pi

def energy_bond(ca, alpha, cfg):
    target = 3.8 * (1.0 + cfg.alpha_mod_bond * (alpha - 1.0))
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return cfg.w_bond * ((d - target_pair) ** 2).mean()

def energy_angle(ca, alpha, cfg):
    if len(ca) < 3: return torch.tensor(0.0, device=ca.device)
    v1 = ca[:-2] - ca[1:-1]
    v2 = ca[2:] - ca[1:-1]
    v1n = F.normalize(v1, dim=-1, eps=1e-8)
    v2n = F.normalize(v2, dim=-1, eps=1e-8)
    cos_ang = (v1n * v2n).sum(-1)
    target_angle = cfg.angle_target_rad * (1.0 + cfg.alpha_mod_angle * (alpha[1:-1] - 1.0))
    cos_target = torch.cos(target_angle)
    return cfg.w_angle * ((cos_ang - cos_target) ** 2).mean()

def energy_rama_vectorized(phi, psi, seq, alpha, cfg):
    L = len(seq)
    device = phi.device
    phi0, psi0, width = torch.zeros(L, device=device), torch.zeros(L, device=device), torch.zeros(L, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0[i], psi0[i], width[i] = prior['phi'], prior['psi'], prior['width']
    width_eff = width * (1.0 + cfg.alpha_mod_rama * (alpha - 1.0))
    dphi = (phi - phi0) / (width_eff + 1e-8)
    dpsi = (psi - psi0) / (width_eff + 1e-8)
    mask = torch.ones(L, device=device, dtype=torch.bool)
    mask[0], mask[-1] = False, False
    loss = (dphi**2 + dpsi**2) * mask.float()
    return cfg.w_rama * loss.sum() / max(1, mask.sum())

def energy_clash_sparse(ca, alpha, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    idx_i, idx_j = edge_index[0], edge_index[1]
    seq_dist = torch.abs(idx_i - idx_j)
    mask = seq_dist > 2
    if not mask.any():
        return torch.tensor(0.0, device=ca.device)
    src, dst = idx_i[mask], idx_j[mask]
    di = edge_dist[mask]
    radius_i = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha[src] - 1.0))
    radius_j = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha[dst] - 1.0))
    radius = 0.5 * (radius_i + radius_j)
    clash = torch.relu(radius - di)
    return cfg.w_clash * (clash ** 2).mean()

def energy_electro_sparse(ca, seq, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=ca.device)
    qi, qj = q[edge_index[0]], q[edge_index[1]]
    r = torch.clamp(edge_dist, min=1e-6)
    E = qi * qj * torch.exp(-0.1 * r) / (80.0 * r)
    return cfg.w_electro * E.mean()

def energy_solvent_sparse(ca, seq, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    src = edge_index[0]
    counts = torch.zeros(ca.shape[0], device=ca.device)
    counts = counts.index_add(0, src, torch.ones_like(src, dtype=torch.float))
    burial = 1.0 - torch.exp(-counts / 20.0)
    hydro = torch.tensor([HYDROPHOBICITY.get(a, 0.0) for a in seq], device=ca.device)
    exposed = torch.where(hydro > 0, hydro * (1.0 - burial), torch.zeros_like(burial))
    buried = torch.where(hydro <= 0, -hydro * burial, torch.zeros_like(burial))
    return cfg.w_solvent * (exposed + buried).mean()

def energy_rotamer_sparse(ca, atoms, seq, edge_index, cfg):
    L = ca.shape[0]
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    mask = torch.tensor([(aa != 'G' and i > 0 and i < L-1) for i, aa in enumerate(seq)], device=ca.device)
    if not mask.any(): return torch.tensor(0.0, device=ca.device)
    N, C = atoms['N'], atoms['C']
    ca_m = ca[mask]; n_m = N[mask]; c_m = C[mask]
    v1 = n_m - ca_m; v2 = c_m - ca_m
    cb_dir = -(v1 + v2); cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
    ideal_cb = ca_m + 1.8 * cb_dir
    global_idx = torch.where(mask)[0]
    edge_src, edge_dst = edge_index[0], edge_index[1]
    is_masked_src = mask[edge_src]
    src_masked_idx = torch.where(is_masked_src)[0]
    src_global = edge_src[src_masked_idx]; dst_global = edge_dst[src_masked_idx]
    global_to_masked = torch.full((L,), -1, device=ca.device, dtype=torch.long)
    global_to_masked[mask] = torch.arange(mask.sum(), device=ca.device)
    src_masked = global_to_masked[src_global]
    ca_dst = ca[dst_global]; cb_src = ideal_cb[src_masked]
    dists = torch.norm(cb_src - ca_dst, dim=-1)
    not_self = (src_global != dst_global)
    dists = dists[not_self]; src_masked = src_masked[not_self]
    min_per_masked = torch.full((mask.sum(),), float('inf'), device=ca.device)
    min_per_masked = torch.scatter_reduce(min_per_masked, 0, src_masked, dists, reduce='amin')
    min_per_masked[min_per_masked == float('inf')] = 10.0
    penalty = torch.relu(4.0 - min_per_masked)
    return cfg.w_rotamer * penalty.mean()

def hbond_energy_from_sparse(O, N, C, alpha, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=O.device)
    src = edge_index[0]  # O index
    dst = edge_index[1]  # N index
    vec_co = O[src] - C[src]
    vec_no = N[dst] - O[src]
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    ideal_dist = 2.9 * (1.0 + cfg.alpha_mod_hbond * (alpha[src].unsqueeze(1) - 1.0)).squeeze()
    E = -alignment * torch.exp(-((edge_dist - ideal_dist) / 0.3) ** 2)
    return E.mean()

def energy_lj_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    sigmas = torch.zeros(len(all_types), device=all_coords.device)
    epsilons = torch.zeros(len(all_types), device=all_coords.device)
    for i, t in enumerate(all_types):
        s, e = LJ_PARAMS.get(t, (1.9, 0.1))
        sigmas[i] = s; epsilons[i] = e
    sigma_src, sigma_dst = sigmas[src], sigmas[dst]
    eps_src, eps_dst = epsilons[src], epsilons[dst]
    sigma = 0.5 * (sigma_src + sigma_dst)
    eps = torch.sqrt(eps_src * eps_dst)
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0 / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    lj = 4.0 * eps * ((sigma * inv_r) ** 12 - (sigma * inv_r) ** 6)
    return cfg.w_lj * lj.mean()

def energy_coulomb_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    charge_map = {'N':-0.5,'CA':0.0,'C':0.5,'O':-0.5,'CB':0.0,'OH':-0.5,'OD':-0.5,'OE':-0.5,
                  'ND':-0.5,'NE':-0.5,'NH1':-0.5,'NH2':-0.5,'SG':-0.2,'S':-0.2}
    q = torch.tensor([charge_map.get(t, 0.0) for t in all_types], device=all_coords.device)
    qi, qj = q[src], q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r
    coulomb = 332.0637 * qi * qj / (dielectric * r)
    return cfg.w_coulomb * coulomb.mean()

def energy_torsion_chi(chi_angles, seq, cfg):
    L = len(seq); device = chi_angles.device; max_chi = chi_angles.shape[1]
    energy = 0.0
    for i, aa in enumerate(seq):
        nchi = RESIDUE_NCHI.get(aa, 0)
        for c in range(min(nchi, max_chi)):
            chi = chi_angles[i, c]
            energy = energy + 0.5 * (1.0 - torch.cos(3.0 * chi))
    return cfg.w_torsion * energy / max(1, L)

def sparse_soc_energy(ca, alpha, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    src, dst = edge_index[0], edge_index[1]
    ai, aj = alpha[src], alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    E = -K * torch.exp(-edge_dist / 8.0)
    return cfg.w_soc_contact * E.mean()

def avalanche_loss_vec(coords, alpha, edge_index, edge_dist, cfg):
    if coords.grad is None: return torch.tensor(0.0, device=coords.device)
    L = coords.shape[0]
    stress = torch.norm(coords.grad, dim=-1)
    stressed = stress > cfg.avalanche_threshold
    if not stressed.any(): return torch.tensor(0.0, device=coords.device)
    src, dst = edge_index[0], edge_index[1]
    ai, aj = alpha[src], alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K_edge = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    direction = torch.zeros_like(coords)
    stressed_idx = torch.where(stressed)[0]
    grad_stressed = coords.grad[stressed_idx]
    norm = torch.norm(grad_stressed, dim=-1, keepdim=True)
    direction[stressed_idx] = -grad_stressed / (norm + 1e-8)
    src_stressed = stressed[src]
    if not src_stressed.any(): return torch.tensor(0.0, device=coords.device)
    edge_K = K_edge[src_stressed]; edge_dst = dst[src_stressed]; edge_src = src[src_stressed]
    direction_src = direction[edge_src]; coord_dst = coords[edge_dst]
    dot = (coord_dst * direction_src).sum(dim=-1)
    loss = - (edge_K * dot).mean()
    return cfg.w_avalanche * loss

def alpha_regularisation(alpha, cfg):
    entropy = -(alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

def chain_break_energy(ca, chain_boundaries, cfg):
    if not chain_boundaries: return torch.tensor(0.0, device=ca.device)
    energy = 0.0
    for start in chain_boundaries:
        dist = torch.norm(ca[start] - ca[start-1], dim=-1)
        energy = energy + torch.relu(dist - 5.0)
    return cfg.chain_break_weight * energy

# ──────────────────────────────────────────────────────────────────────────────
# Total physics energy (V29.1)
# ──────────────────────────────────────────────────────────────────────────────
def total_physics_energy_v29(ca, seq, alpha, chi_angles,
                            edge_index_ca, edge_dist_ca,
                            edge_index_hbond, edge_dist_hbond,
                            chain_boundaries, cfg):
    atoms = reconstruct_backbone(ca)
    phi, psi = compute_phi_psi(atoms)
    e = 0.0
    e += energy_bond(ca, alpha, cfg)
    e += energy_angle(ca, alpha, cfg)
    e += energy_rama_vectorized(phi, psi, seq, alpha, cfg)
    e += energy_clash_sparse(ca, alpha, edge_index_ca, edge_dist_ca, cfg)
    e += cfg.w_hbond * hbond_energy_from_sparse(atoms['O'], atoms['N'], atoms['C'], alpha,
                                                edge_index_hbond, edge_dist_hbond, cfg)
    e += energy_electro_sparse(ca, seq, edge_index_ca, edge_dist_ca, cfg)
    e += energy_solvent_sparse(ca, seq, edge_index_ca, edge_dist_ca, cfg)
    e += energy_rotamer_sparse(ca, atoms, seq, edge_index_ca, cfg)
    e += alpha_regularisation(alpha, cfg)
    e += chain_break_energy(ca, chain_boundaries, cfg)

    if chi_angles is not None:
        all_coords, all_types, res_indices = get_full_atom_coords_and_types(ca, seq, chi_angles)
        edge_idx_full, edge_dist_full = sparse_edges(all_coords, cfg.sparse_cutoff, cfg.max_neighbors)
        e += energy_lj_full(all_coords, all_types, res_indices, edge_idx_full, edge_dist_full, cfg)
        e += energy_coulomb_full(all_coords, all_types, res_indices, edge_idx_full, edge_dist_full, cfg)
        e += energy_torsion_chi(chi_angles, seq, cfg)

    if edge_index_ca is not None and edge_index_ca.numel() > 0:
        e += sparse_soc_energy(ca, alpha, edge_index_ca, edge_dist_ca, cfg)
        e += avalanche_loss_vec(ca, alpha, edge_index_ca, edge_dist_ca, cfg)
    return e

# ──────────────────────────────────────────────────────────────────────────────
# Dataset (same)
# ──────────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        seq, coords = self.data[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a,20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        return seq_ids, coords

def synthetic_dataset(num_samples=200, min_len=50, max_len=150):
    data = []
    for _ in range(num_samples):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choices(AA_VOCAB[:-1], k=L))
        coords = np.zeros((L,3), dtype=np.float32)
        d = np.random.randn(3).astype(np.float32)
        d /= np.linalg.norm(d)+1e-8
        for i in range(1,L):
            d += 0.2*np.random.randn(3).astype(np.float32)
            d /= np.linalg.norm(d)+1e-8
            coords[i] = coords[i-1] + d*3.8
        coords -= coords.mean(axis=0)
        data.append((seq, coords))
    return data

# ──────────────────────────────────────────────────────────────────────────────
# V29.1 Core Model
# ──────────────────────────────────────────────────────────────────────────────
class CSOCSSC_V29_1(nn.Module):
    def __init__(self, cfg: V29Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = FlashSequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = EquivariantDecoder(cfg.dim, cfg.egnn_hidden, cfg.egnn_edge_dim,
                                          cfg.egnn_layers, cfg.egnn_cutoff)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids, initial_coords=None):
        latent = self.encoder(seq_ids)
        coords = self.decoder(latent, initial_coords)
        alpha = self.alpha_field(latent)
        return coords, alpha

    def predict_multimer(self, sequences: List[str], initial_coords_list=None):
        self.eval()
        all_seq = "".join(sequences)
        ids = torch.tensor([AA_TO_ID.get(a,20) for a in all_seq],
                           dtype=torch.long, device=self.cfg.device).unsqueeze(0)
        with torch.no_grad():
            latent = self.encoder(ids)
            init = None
            if initial_coords_list:
                init = torch.cat([torch.tensor(c, device=self.cfg.device) for c in initial_coords_list], dim=0)
                init = init.unsqueeze(0)
            coords, alpha = self.decoder(latent, init)
        coords_np = coords.squeeze(0).cpu().numpy()
        alpha_np = alpha.squeeze(0).cpu().numpy()
        chains_ca = []
        idx = 0
        for s in sequences:
            l = len(s)
            chains_ca.append(coords_np[idx:idx+l])
            idx += l
        return chains_ca, alpha_np

    def refine_multimer(self, sequences: List[str], init_coords_list=None,
                        steps=None, logger=None):
        if steps is None: steps = self.cfg.refine_steps
        self.eval()
        device = torch.device(self.cfg.device)
        all_seq = "".join(sequences)
        L = len(all_seq)

        boundaries = []
        idx = 0
        for s in sequences[:-1]:
            idx += len(s)
            boundaries.append(idx)

        if init_coords_list is not None:
            ca_np = np.concatenate([c - c.mean(axis=0) for c in init_coords_list], axis=0)
            ca = torch.tensor(ca_np, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a,20) for a in all_seq],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                chains_ca, alpha_np = self.predict_multimer(sequences)
            ca_np = np.concatenate(chains_ca, axis=0)
            ca = torch.tensor(ca_np, device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, device=device)

        max_chi = MAX_CHI
        chi = nn.Parameter(torch.zeros((L, max_chi), device=device).uniform_(-math.pi, math.pi))

        # Build initial sparse graphs with PyG
        edge_index_ca, edge_dist_ca = sparse_edges(ca, self.cfg.sparse_cutoff,
                                                   self.cfg.max_neighbors)
        atoms = reconstruct_backbone(ca)
        edge_index_hbond, edge_dist_hbond = cross_sparse_edges(
            atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors
        )

        opt = torch.optim.Adam([ca, chi], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)
        energy_history = []

        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                e_phys = total_physics_energy_v29(ca, all_seq, alpha, chi,
                                                  edge_index_ca, edge_dist_ca,
                                                  edge_index_hbond, edge_dist_hbond,
                                                  boundaries, self.cfg)
                loss = e_phys

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([ca, chi], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            sigma = self.csoc.sigma(ca.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                ca.add_(torch.randn_like(ca) * noise_scale)
                chi.data.add_(torch.randn_like(chi) * noise_scale * 0.5)

            if step > 0 and step % self.cfg.rebuild_interval == 0:
                edge_index_ca, edge_dist_ca = sparse_edges(
                    ca.detach(), self.cfg.sparse_cutoff, self.cfg.max_neighbors
                )
                atoms = reconstruct_backbone(ca.detach())
                edge_index_hbond, edge_dist_hbond = cross_sparse_edges(
                    atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors
                )

            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                ca.data = self.rg.forward(ca.data)

            if step % 50 == 0 and logger:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f} "
                            f"σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return ca.detach().cpu().numpy(), chi.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Training loop (DDP‑ready)
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, dataloader, cfg, logger):
    device = torch.device(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)
    for epoch in range(cfg.epochs):
        if cfg.is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (seq_ids, target_coords) in enumerate(dataloader):
            seq_ids, target_coords = seq_ids.to(device), target_coords.to(device)
            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred_coords, pred_alpha = model(seq_ids)
                coord_loss = F.mse_loss(pred_coords, target_coords)
                alpha_reg = 0.001 * ((pred_alpha[:,1:] - pred_alpha[:,:-1])**2).mean()
                loss = (coord_loss + alpha_reg) / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * cfg.gradient_accumulation_steps

        if cfg.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  MSE={total_loss/len(dataloader):.4f}")

    if cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, "v29.1_pretrained.pt")
        state_dict = model.module.state_dict() if cfg.is_distributed else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Full‑atom PDB writer (same as V28)
# ──────────────────────────────────────────────────────────────────────────────
def write_full_pdb_multimer(ca, seq, chi_angles, filename, chain_ids=None):
    device = 'cpu'
    ca_t = torch.tensor(ca, dtype=torch.float32)
    chi_t = torch.tensor(chi_angles, dtype=torch.float32)
    all_coords, all_types, res_indices = get_full_atom_coords_and_types(ca_t, seq, chi_t)
    atoms = reconstruct_backbone(ca_t)
    N_coords, C_coords, O_coords = atoms['N'], atoms['C'], atoms['O']

    with open(filename, 'w') as f:
        atom_serial = 1
        for i, aa in enumerate(seq):
            chain = chain_ids[i] if chain_ids else 'A'
            f.write(f"ATOM  {atom_serial:5d}  N   {aa:3s} {chain}{i+1:4d}    "
                    f"{N_coords[i,0]:8.3f}{N_coords[i,1]:8.3f}{N_coords[i,2]:8.3f}  1.00  0.00           N\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  CA  {aa:3s} {chain}{i+1:4d}    "
                    f"{ca[i,0]:8.3f}{ca[i,1]:8.3f}{ca[i,2]:8.3f}  1.00  0.00           C\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  C   {aa:3s} {chain}{i+1:4d}    "
                    f"{C_coords[i,0]:8.3f}{C_coords[i,1]:8.3f}{C_coords[i,2]:8.3f}  1.00  0.00           C\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  O   {aa:3s} {chain}{i+1:4d}    "
                    f"{O_coords[i,0]:8.3f}{O_coords[i,1]:8.3f}{O_coords[i,2]:8.3f}  1.00  0.00           O\n")
            atom_serial += 1

            res_atom_mask = res_indices == i
            res_coords = all_coords[res_atom_mask]
            res_types = [all_types[j] for j in range(len(all_types)) if res_indices[j]==i]
            if aa != 'G':
                for k in range(3, res_coords.shape[0]):
                    at_name = res_types[k]
                    x,y,z = res_coords[k]
                    f.write(f"ATOM  {atom_serial:5d}  {at_name:4s}{aa:3s} {chain}{i+1:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {at_name[0]}\n")
                    atom_serial += 1
        f.write("END\n")
    print(f"Full‑atom PDB written to {filename}")

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI (DDP & multimer)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC V29.1 Scalable Multimer Engine (PyG)")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--samples', type=int, default=1000)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)

    refine_parser = sub.add_parser('refine')
    refine_parser.add_argument('--seq', nargs='+', type=str, help='Sequences for each chain')
    refine_parser.add_argument('--pdb', type=str, default=None)
    refine_parser.add_argument('--init', nargs='+', type=str, help='Initial CA PDB files per chain')
    refine_parser.add_argument('--out', type=str, default='refined_v29.1.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--checkpoint', type=str, default='v29.1_pretrained.pt')

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = V29Config(local_rank=local_rank, is_distributed=is_distributed, device=device_str,
                    epochs=getattr(args,'epochs',80), batch_size=getattr(args,'batch_size',8),
                    refine_steps=getattr(args,'steps',600))

    torch.manual_seed(cfg.seed + (local_rank if local_rank>0 else 0))
    np.random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    logger = setup_logger("CSOC-SSC_V29.1", local_rank)

    if local_rank in [-1,0]:
        logger.info("="*60)
        logger.info("CSOC-SSC V29.1 – PyG‑Accelerated Scalable Multimer Engine")
        logger.info(f"Distributed: {is_distributed} | Device: {device_str}")
        logger.info("="*60)

    if args.command == 'train':
        if local_rank in [-1,0]:
            logger.info("Generating synthetic training data...")
        data = synthetic_dataset(num_samples=args.samples)
        dataset = ProteinDataset(data)
        if is_distributed:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        model = CSOCSSC_V29_1(cfg).to(torch.device(device_str))
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_model(model, dataloader, cfg, logger)

    elif args.command == 'refine':
        model = CSOCSSC_V29_1(cfg).to(torch.device(device_str))
        ckpt = args.checkpoint
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device_str))
            logger.info(f"Loaded weights from {ckpt}")
        else:
            logger.warning("Checkpoint not found; using random weights.")

        if args.pdb:
            logger.info(f"Fetching multimer PDB {args.pdb}...")
            backbones, chain_ids = MultimerPDBFetcher.fetch(args.pdb)
            sequences = [b.seq for b in backbones]
            init_coords_list = [b.ca for b in backbones]
            native_ca_list = [b.native_coords for b in backbones if b.native_coords is not None]
            chain_labels = [b.chain_id for b in backbones]
        elif args.seq:
            sequences = args.seq
            init_coords_list = []
            if args.init:
                for init_file in args.init:
                    coords_list = []
                    with open(init_file) as f:
                        for line in f:
                            if line.startswith('ATOM') and line[12:16].strip()=='CA':
                                coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    init_coords_list.append(np.array(coords_list, dtype=np.float32))
            else:
                init_coords_list = None
            chain_labels = None
            native_ca_list = None
        else:
            raise ValueError("Must provide --seq or --pdb.")

        start_time = time.time()
        refined_ca, refined_chi, _ = model.refine_multimer(sequences, init_coords_list,
                                                           steps=cfg.refine_steps, logger=logger)
        all_seq = "".join(sequences)
        if chain_labels:
            res_chain = []
            for seq, lbl in zip(sequences, chain_labels):
                res_chain.extend([lbl]*len(seq))
        else:
            res_chain = ['A'] * len(all_seq)
        write_full_pdb_multimer(refined_ca, all_seq, refined_chi, args.out, res_chain)
        logger.info(f"Full‑atom refined structure saved to {args.out}")

        if native_ca_list:
            native_all = np.concatenate(native_ca_list, axis=0)
            def rmsd(a,b):
                a = a - a.mean(axis=0); b = b - b.mean(axis=0)
                H = a.T @ b
                U,_,Vt = np.linalg.svd(H)
                rot = Vt.T @ U.T
                if np.linalg.det(rot) < 0:
                    Vt[-1,:]*=-1; rot = Vt.T @ U.T
                return np.sqrt(np.mean(np.sum((a @ rot - b)**2, axis=1)))
            rmsd_val = rmsd(refined_ca, native_all)
            logger.info(f"Final CA RMSD vs Native: {rmsd_val:.4f} Å")
        logger.info(f"Compute Time: {time.time()-start_time:.2f} seconds")

    if is_distributed:
        dist.destroy_process_group()
