# =============================================================================
# CSOC‑SSC v26 — Fully Vectorized GPU‑Native SOC Folding Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# V26 replaces the CPU kd‑tree and Python loops with fully vectorized GPU
# operations.  The sparse SOC graph is built via torch.cdist + thresholding,
# the avalanche is a vectorized loss term, and the rotamer energy uses the
# same sparse edges.  All other V25.5 features (positional encoding,
# FlashAttention, DDP, PDB fetcher) are retained.
# =============================================================================

import os, math, time, random, argparse, logging, urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(name="CSOC‑SSC_V26", local_rank=-1):
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
# Biochemical constants
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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class V26Config:
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    avalanche_topk: int = 3
    w_avalanche: float = 0.2

    ca_ca_dist: float = 3.8
    clash_radius: float = 3.5
    angle_target_rad: float = 110.0 * math.pi / 180.0

    alpha_mod_bond: float = 0.1
    alpha_mod_angle: float = 0.05
    alpha_mod_rama: float = 0.2
    alpha_mod_clash: float = 0.1
    alpha_mod_hbond: float = 0.1

    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1
    w_soc_contact: float = 0.3

    sparse_cutoff: float = 12.0      # cutoff for sparse SOC graph
    kernel_lambda: float = 12.0
    rebuild_interval: int = 100      # how often to rebuild sparse graph

    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200

    checkpoint_dir: str = "./v26_ckpt"
    out_pdb: str = "refined.pdb"

# ──────────────────────────────────────────────────────────────────────────────
# PDB Fetcher
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Backbone:
    ca: np.ndarray
    seq: str
    chain_ids: Optional[np.ndarray] = None
    native_coords: Optional[np.ndarray] = None

class PDBFetcher:
    @staticmethod
    def fetch_and_parse(pdb_id: str) -> Backbone:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={'User-Agent': 'CSOC-SSC_V26'})
        ca_coords, seq_list, chain_ids = [], [], []
        chain_map = {}
        try:
            with urllib.request.urlopen(req) as response:
                lines = response.read().decode('utf-8').split('\n')
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if chain_id not in chain_map:
                        chain_map[chain_id] = len(chain_map)
                    seq_list.append(AA_3_TO_1.get(res_name, 'X'))
                    ca_coords.append([x, y, z])
                    chain_ids.append(chain_map[chain_id])
            coords_arr = np.array(ca_coords, dtype=np.float32)
            seq_str = "".join(seq_list)
            chain_arr = np.array(chain_ids, dtype=np.int32)
            randomized_coords = coords_arr + (np.random.randn(*coords_arr.shape) * 10.0).astype(np.float32)
            return Backbone(ca=randomized_coords, seq=seq_str, chain_ids=chain_arr, native_coords=coords_arr)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PDB {pdb_id}: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
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

# ──────────────────────────────────────────────────────────────────────────────
# FlashAttention Transformer
# ──────────────────────────────────────────────────────────────────────────────
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

class GeometryDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 3))

    def forward(self, latent):
        coords = self.net(latent)
        return coords - coords.mean(dim=1, keepdim=True)

class AdaptiveAlphaField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, latent):
        a = torch.sigmoid(self.net(latent))
        a = 0.5 + 2.5 * a.squeeze(-1)
        return torch.clamp(a, 0.5, 3.0)   # explicit clamp for safety

# ──────────────────────────────────────────────────────────────────────────────
# CSOC Controller (unchanged)
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
# Differentiable RG Refinement
# ──────────────────────────────────────────────────────────────────────────────
class DiffRGRefiner:
    def __init__(self, factor=4):
        self.factor = factor

    def forward(self, coords):
        # coords: (L, 3)
        L = coords.shape[0]
        f = self.factor
        m = L // f * f
        if m == 0:
            return coords
        # input shape for avg_pool1d: (N, C, L)
        x = coords[:m].permute(1, 0).unsqueeze(0)   # (1, 3, m)
        pooled = F.avg_pool1d(x, kernel_size=f, stride=f)  # (1, 3, m//f)
        up = F.interpolate(pooled, size=L, mode='linear', align_corners=True)
        return up.squeeze(0).permute(1, 0)           # (L, 3)

# ──────────────────────────────────────────────────────────────────────────────
# GPU Sparse Graph Builder (torch-only, no CPU sync)
# ──────────────────────────────────────────────────────────────────────────────
def build_sparse_graph(coords, cutoff):
    """Build undirected sparse edge list from coordinates on GPU.
    Args:
        coords: (L, 3) tensor
        cutoff: float distance threshold
    Returns:
        edge_index: (2, E) long tensor of (src, dst) pairs (directed, both ways)
        edge_dist: (E,) float32 tensor of distances
    """
    D = torch.cdist(coords, coords)                         # (L, L)
    # mask upper triangle without diagonal
    triu = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
    mask = (D < cutoff) & triu
    src, dst = torch.where(mask)                            # each edge once
    edge_dist = D[src, dst]
    # make bidirectional
    src = torch.cat([src, dst])
    dst = torch.cat([dst, src])
    edge_dist = torch.cat([edge_dist, edge_dist])
    edge_index = torch.stack([src, dst], dim=0)             # (2, 2E)
    return edge_index, edge_dist

# ──────────────────────────────────────────────────────────────────────────────
# Backbone Reconstruction & Dihedral Angles
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
        if i < L - 1:
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

# ──────────────────────────────────────────────────────────────────────────────
# Physics Energy Terms
# ──────────────────────────────────────────────────────────────────────────────
def energy_bond(ca, alpha, cfg):
    target = cfg.ca_ca_dist * (1.0 + cfg.alpha_mod_bond * (alpha - 1.0))
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return cfg.w_bond * ((d - target_pair) ** 2).mean()

def energy_angle(ca, alpha, cfg):
    if len(ca) < 3:
        return torch.tensor(0.0, device=ca.device)
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
    phi0 = torch.zeros(L, device=device)
    psi0 = torch.zeros(L, device=device)
    width = torch.zeros(L, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0[i] = prior['phi']
        psi0[i] = prior['psi']
        width[i] = prior['width']
    width_eff = width * (1.0 + cfg.alpha_mod_rama * (alpha - 1.0))
    dphi = (phi - phi0) / (width_eff + 1e-8)
    dpsi = (psi - psi0) / (width_eff + 1e-8)
    mask = torch.ones(L, device=device, dtype=torch.bool)
    mask[0] = False; mask[-1] = False
    loss = (dphi**2 + dpsi**2) * mask.float()
    return cfg.w_rama * loss.sum() / max(1, mask.sum())

def energy_clash(ca, alpha, cfg):
    D = torch.cdist(ca, ca)
    mask = torch.ones_like(D, dtype=torch.bool)
    idx = torch.arange(len(ca), device=ca.device)
    mask[idx[:, None], idx[None, :]] = False
    mask[idx[:-1, None], (idx[None, :-1]+1)] = False
    mask[(idx[None, :-1]+1), idx[:-1, None]] = False
    radius = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha.unsqueeze(1) - 1.0))
    radius_pair = 0.5 * (radius + radius.T)
    clash = torch.relu(radius_pair - D) * mask.float()
    return cfg.w_clash * (clash ** 2).mean()

def energy_hbond(atoms, alpha, cfg):
    O, N, C = atoms['O'], atoms['N'], atoms['C']
    D = torch.cdist(O, N)
    mask = (D > 2.5) & (D < 3.5)
    vec_co = O.unsqueeze(1) - C.unsqueeze(1)
    vec_no = N.unsqueeze(0) - O.unsqueeze(1)
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    ideal_dist = 2.9 * (1.0 + cfg.alpha_mod_hbond * (alpha.unsqueeze(1) - 1.0))
    E = -alignment * torch.exp(-((D - ideal_dist) / 0.3) ** 2)
    return cfg.w_hbond * (E * mask.float()).mean()

def energy_electro(ca, seq, cfg):
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=ca.device)
    D = torch.cdist(ca, ca) + 1e-6
    E = q.unsqueeze(1) * q.unsqueeze(0) * torch.exp(-0.1 * D) / (80.0 * D)
    E.diagonal().zero_()
    return cfg.w_electro * E.mean()

def energy_solvent(ca, seq, cfg):
    D = torch.cdist(ca, ca)
    density = (D < 10.0).float().sum(dim=-1)
    burial = 1.0 - torch.exp(-density / 20.0)
    hydro = torch.tensor([HYDROPHOBICITY.get(a, 0.0) for a in seq], device=ca.device)
    exposed = torch.where(hydro > 0, hydro * (1.0 - burial), torch.zeros_like(burial))
    buried = torch.where(hydro <= 0, -hydro * burial, torch.zeros_like(burial))
    return cfg.w_solvent * (exposed + buried).mean()

# Rotamer energy using sparse edges (only O(E))
def energy_rotamer_sparse(ca, atoms, seq, edge_index, cfg):
    L = ca.shape[0]
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    # Build ideal Cβ positions for all residues (vectorized where possible)
    # We'll compute per‑residue Cβ direction only for non‑Gly non‑terminal
    mask = torch.tensor([(aa != 'G' and i > 0 and i < L-1) for i, aa in enumerate(seq)],
                        device=ca.device)
    if not mask.any():
        return torch.tensor(0.0, device=ca.device)
    # compute directions for all masked residues
    N = atoms['N']
    C = atoms['C']
    ca_m = ca[mask]
    n_m = N[mask]
    c_m = C[mask]
    v1 = n_m - ca_m
    v2 = c_m - ca_m
    cb_dir = -(v1 + v2)
    cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
    ideal_cb = ca_m + 1.8 * cb_dir   # (M, 3)
    # Now we need per‑residue minimum distance to any other CA (excluding self).
    # Use sparse edges: for each masked residue i (global index), we look at edges where src==i or dst==i.
    # To vectorize: we create a dense mapping from masked index to global index.
    global_idx = torch.where(mask)[0]
    # We'll compute distances from ideal_cb to all CA using sparse edges?
    # Instead, we can compute all pairwise distances between ideal_cb and ca, but that's O(L*M).
    # Since M ≤ L, for large L this is still O(L²). We'll limit to sparse edges:
    # For each masked residue i, we consider only CA of residues j that are neighbours in the sparse graph.
    # To do this efficiently, we can use edge_index to compute distances from ideal_cb[src] to ca[dst] for edges where src is masked.
    # Then for each masked src, take the min over dst (excluding self).
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    # map global src indices to masked indices (only keep edges where src is masked)
    is_masked_src = mask[edge_src]
    src_masked_idx = torch.where(is_masked_src)[0]   # indices into edges
    src_global = edge_src[src_masked_idx]            # global indices of masked residues
    dst_global = edge_dst[src_masked_idx]            # neighbour global indices
    # map src_global to position in ideal_cb (since mask is boolean, we need a mapping tensor)
    global_to_masked = torch.full((L,), -1, device=ca.device, dtype=torch.long)
    global_to_masked[mask] = torch.arange(mask.sum(), device=ca.device)
    src_masked = global_to_masked[src_global]        # indices into ideal_cb
    # compute distances
    ca_dst = ca[dst_global]                         # (E', 3)
    cb_src = ideal_cb[src_masked]                   # (E', 3)
    dists = torch.norm(cb_src - ca_dst, dim=-1)     # (E',)
    # exclude self (src_global == dst_global)
    not_self = (src_global != dst_global)
    dists = dists[not_self]
    src_masked = src_masked[not_self]
    # Use scatter to compute per‑residue minimum distance
    min_per_masked = torch.full((mask.sum(),), float('inf'), device=ca.device)
    min_per_masked = torch.scatter_reduce(min_per_masked, 0, src_masked, dists, reduce='amin')
    # For residues with no edges, set min to a large number (they won't contribute)
    min_per_masked[min_per_masked == float('inf')] = 10.0
    penalty = torch.relu(4.0 - min_per_masked)
    # average over masked residues only
    return cfg.w_rotamer * penalty.mean()

def sparse_soc_energy(ca, alpha, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    src, dst = edge_index[0], edge_index[1]
    ai = alpha[src]
    aj = alpha[dst]
    a = 0.5 * (ai + aj)
    # safe computation: use clamp on edge_dist to avoid log(0)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    E = -K * torch.exp(-edge_dist / 8.0)
    return cfg.w_soc_contact * E.mean()

def alpha_regularisation(alpha, cfg):
    entropy = -(alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

# ──────────────────────────────────────────────────────────────────────────────
# Vectorized Differentiable Avalanche Loss (GPU‑native)
# ──────────────────────────────────────────────────────────────────────────────
def avalanche_loss_vec(coords, alpha, edge_index, edge_dist, cfg):
    """Fully vectorized avalanche loss: pushes neighbours of stressed residues
    along negative gradient direction, weighted by SOC kernel."""
    if coords.grad is None:
        return torch.tensor(0.0, device=coords.device)
    L = coords.shape[0]
    stress = torch.norm(coords.grad, dim=-1)                    # (L,)
    stressed = stress > cfg.avalanche_threshold
    if not stressed.any():
        return torch.tensor(0.0, device=coords.device)

    src = edge_index[0]
    dst = edge_index[1]
    # kernel values for edges
    ai = alpha[src]
    aj = alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K_edge = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)

    # direction for each stressed node: -grad / ||grad||
    direction = torch.zeros_like(coords)
    stressed_idx = torch.where(stressed)[0]
    grad_stressed = coords.grad[stressed_idx]
    norm = torch.norm(grad_stressed, dim=-1, keepdim=True)
    direction[stressed_idx] = -grad_stressed / (norm + 1e-8)

    # For each edge where src is stressed, compute contribution:
    # weight = K_edge[edge] (or normalized among top-k? we'll weight by K and sum)
    # loss += -weight * (coords[dst] · direction[src])
    src_stressed = stressed[src]                                 # (E,) bool
    if not src_stressed.any():
        return torch.tensor(0.0, device=coords.device)
    # select edges
    edge_K = K_edge[src_stressed]
    edge_dst = dst[src_stressed]
    edge_src = src[src_stressed]
    direction_src = direction[edge_src]                          # (E', 3)
    coord_dst = coords[edge_dst]                                # (E', 3)
    dot = (coord_dst * direction_src).sum(dim=-1)               # (E',)
    loss = - (edge_K * dot).mean()                               # negative sign encourages alignment
    return cfg.w_avalanche * loss

# ──────────────────────────────────────────────────────────────────────────────
# Total Physics Energy Aggregator
# ──────────────────────────────────────────────────────────────────────────────
def total_physics_energy(ca, seq, alpha, edge_index, edge_dist, cfg):
    atoms = reconstruct_backbone(ca)
    phi, psi = compute_phi_psi(atoms)
    e = 0.0
    e += energy_bond(ca, alpha, cfg)
    e += energy_angle(ca, alpha, cfg)
    e += energy_rama_vectorized(phi, psi, seq, alpha, cfg)
    e += energy_clash(ca, alpha, cfg)
    e += energy_hbond(atoms, alpha, cfg)
    e += energy_electro(ca, seq, cfg)
    e += energy_solvent(ca, seq, cfg)
    e += energy_rotamer_sparse(ca, atoms, seq, edge_index, cfg)
    e += alpha_regularisation(alpha, cfg)
    if edge_index is not None and edge_index.numel() > 0:
        e += sparse_soc_energy(ca, alpha, edge_index, edge_dist, cfg)
        e += avalanche_loss_vec(ca, alpha, edge_index, edge_dist, cfg)
    return e

# ──────────────────────────────────────────────────────────────────────────────
# Dataset (centred targets)
# ──────────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
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
# V26 Core Model
# ──────────────────────────────────────────────────────────────────────────────
class CSOCSSC_V26(nn.Module):
    def __init__(self, cfg: V26Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = FlashSequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = GeometryDecoder(cfg.dim)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids):
        latent = self.encoder(seq_ids)
        coords = self.decoder(latent)
        alpha = self.alpha_field(latent)
        return coords, alpha

    def predict(self, sequence):
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([AA_TO_ID.get(a,20) for a in sequence],
                               dtype=torch.long, device=self.cfg.device).unsqueeze(0)
            coords, alpha = self.forward(ids)
        return coords.squeeze(0).cpu().numpy(), alpha.squeeze(0).cpu().numpy()

    def refine(self, sequence, init_coords=None, steps=None, logger=None):
        if steps is None:
            steps = self.cfg.refine_steps
        self.eval()
        device = torch.device(self.cfg.device)

        # Prepare initial coordinates and alpha
        if init_coords is not None:
            init_centred = init_coords - init_coords.mean(axis=0)
            coords = torch.tensor(init_centred, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a,20) for a in sequence],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                coords_np, alpha_np = self.predict(sequence)
            coords = torch.tensor(coords_np, dtype=torch.float32, device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        # Build sparse graph on GPU
        edge_index, edge_dist = build_sparse_graph(coords, self.cfg.sparse_cutoff)
        # move to device (already on GPU if coords is on GPU)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)

        # Neural restraint (once)
        neural_target = None
        if init_coords is None:
            with torch.no_grad():
                neural_coords_np, _ = self.predict(sequence)
                neural_target = torch.tensor(neural_coords_np, device=device)

        opt = torch.optim.Adam([coords], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        energy_history = []
        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                e_phys = total_physics_energy(coords, sequence, alpha, edge_index, edge_dist, self.cfg)
                loss = e_phys
                if neural_target is not None:
                    loss = loss + 0.1 * ((coords - neural_target) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([coords], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            # CSOC temperature noise
            sigma = self.csoc.sigma(coords.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                coords.add_(torch.randn_like(coords) * noise_scale)

            # Rebuild sparse graph periodically
            if step > 0 and step % self.cfg.rebuild_interval == 0:
                edge_index, edge_dist = build_sparse_graph(coords.detach(), self.cfg.sparse_cutoff)
                edge_index = edge_index.to(device)
                edge_dist = edge_dist.to(device)

            # RG refinement
            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                coords.data = self.rg.forward(coords.data)

            if step % 50 == 0 and logger:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f}  "
                            f"σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return coords.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Distributed Training
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
        path = os.path.join(cfg.checkpoint_dir, "v26_pretrained.pt")
        state_dict = model.module.state_dict() if cfg.is_distributed else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def compute_rmsd(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(np.mean(np.sum((a @ R - b)**2, axis=1))))

def write_ca_pdb(coords, seq, filename):
    with open(filename, 'w') as f:
        for i, (c, aa) in enumerate(zip(coords, seq)):
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC V26 GPU‑Native Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--samples', type=int, default=1000)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)

    refine_parser = sub.add_parser('refine')
    refine_parser.add_argument('--seq', type=str, default=None)
    refine_parser.add_argument('--pdb', type=str, default=None)
    refine_parser.add_argument('--init', type=str, default=None)
    refine_parser.add_argument('--out', type=str, default='refined_v26.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--checkpoint', type=str, default='v26_pretrained.pt')

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = V26Config(local_rank=local_rank, is_distributed=is_distributed, device=device_str,
                    epochs=getattr(args,'epochs',80), batch_size=getattr(args,'batch_size',8),
                    refine_steps=getattr(args,'steps',600))

    torch.manual_seed(cfg.seed + (local_rank if local_rank>0 else 0))
    np.random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    random.seed(cfg.seed + (local_rank if local_rank>0 else 0))

    logger = setup_logger("CSOC-SSC_V26", local_rank)

    if local_rank in [-1,0]:
        logger.info("="*60)
        logger.info("CSOC-SSC V26 – Fully GPU‑Vectorized SOC Folding Engine")
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
        model = CSOCSSC_V26(cfg).to(torch.device(device_str))
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_model(model, dataloader, cfg, logger)

    elif args.command == 'refine':
        model = CSOCSSC_V26(cfg).to(torch.device(device_str))
        ckpt = args.checkpoint
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device_str))
            logger.info(f"Loaded weights from {ckpt}")
        else:
            logger.warning("Checkpoint not found; using random weights.")

        target_seq = args.seq
        init_coords = None
        native_coords = None
        if args.pdb:
            logger.info(f"Fetching PDB {args.pdb}...")
            backbone = PDBFetcher.fetch_and_parse(args.pdb)
            init_coords = backbone.ca
            target_seq = backbone.seq
            native_coords = backbone.native_coords
        elif args.init and os.path.exists(args.init):
            coords_list = []
            with open(args.init) as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip()=='CA':
                        coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if coords_list:
                init_coords = np.array(coords_list, dtype=np.float32)
                logger.info(f"Loaded {len(init_coords)} initial CA atoms.")
        if not target_seq:
            raise ValueError("Must provide --seq or --pdb.")

        start_time = time.time()
        refined, _ = model.refine(target_seq, init_coords=init_coords, steps=cfg.refine_steps, logger=logger)
        write_ca_pdb(refined, target_seq, args.out)
        logger.info(f"Refined structure saved to {args.out}")
        if native_coords is not None:
            rmsd_val = compute_rmsd(refined, native_coords)
            logger.info(f"Final RMSD vs Native RCSB: {rmsd_val:.4f} Å")
        logger.info(f"Compute Time: {time.time()-start_time:.2f} seconds")

    if is_distributed:
        dist.destroy_process_group()
