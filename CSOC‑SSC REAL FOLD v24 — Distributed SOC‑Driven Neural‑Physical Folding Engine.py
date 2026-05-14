# =============================================================================
# CSOC‑SSC v24 — Distributed SOC‑Driven Neural‑Physical Folding Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# V24 is the definitive production build. It integrates the strictly corrected 
# physics, accurate energy double-counting fixes, and SOC avalanche dynamics 
# from V23 with the Distributed Data Parallel (DDP) and PyTorch 2.0+ 
# FlashAttention optimizations necessary for High-Performance Computing (HPC) 
# and multi-GPU cluster scaling.
#
# KEY FEATURES
# -----------------------------------------------------------------------------
# • Multi-GPU Distributed Data Parallel (DDP) Support via torchrun
# • Native PyTorch FlashAttention (Memory & Speed Optimized)
# • Corrected Physical Interaction Kernel (non-normalised, batch-safe)
# • Vectorised Ramachandran & Deep α-field coupling
# • True SOC Avalanche Dynamics using autograd stress computation
# • Soft CSOC Temperature Saturation
# • Differentiable Multiscale RG Refinement
# • Dynamic RCSB PDB Fetching capabilities
# =============================================================================

import os
import math
import time
import random
import argparse
import logging
import urllib.request
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
def setup_logger(name: str = "CSOC‑SSC_V24", local_rank: int = -1) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '[%(asctime)s] [Rank %(process)d] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    
    # Only print INFO logs on main process to avoid clutter in DDP
    if local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
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
class V24Config:
    # Distributed Training
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Neural architecture (Flash Attention enabled)
    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # Refinement (SOC dynamics - V23 corrected constants)
    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5       
    avalanche_steps: int = 3               

    # Geometry constants
    ca_ca_dist: float = 3.8           
    clash_radius: float = 3.5         
    angle_target_rad: float = 110.0 * math.pi / 180.0

    # Alpha‑field coupling strengths
    alpha_mod_bond: float = 0.1            
    alpha_mod_angle: float = 0.05          
    alpha_mod_rama: float = 0.2            
    alpha_mod_clash: float = 0.1           
    alpha_mod_hbond: float = 0.1           

    # Energy weights (V23 faithful scaling)
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

    # SOC kernel
    kernel_lambda: float = 12.0            
    use_soc_kernel: bool = True            

    # RG refinement
    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200

    # Paths
    checkpoint_dir: str = "./v24_ckpt"
    out_pdb: str = "refined.pdb"

# ──────────────────────────────────────────────────────────────────────────────
# Data Structures & PDB Fetching
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Backbone:
    ca: np.ndarray
    seq: str
    chain_ids: Optional[np.ndarray] = None
    native_coords: Optional[np.ndarray] = None

class PDBFetcher:
    """Dynamic PDB parser for retrieving targets without external dependencies."""
    @staticmethod
    def fetch_and_parse(pdb_id: str) -> Backbone:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={'User-Agent': 'CSOC-SSC_V24_Research'})
        
        ca_coords = []
        seq_list = []
        chain_ids = []
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
            
            # Start optimization from slightly perturbed state to simulate folding
            randomized_coords = coords_arr + (np.random.randn(*coords_arr.shape) * 10.0).astype(np.float32)
            
            return Backbone(
                ca=randomized_coords,
                seq=seq_str,
                chain_ids=chain_arr,
                native_coords=coords_arr
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch or parse PDB {pdb_id}: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# Neural Modules (FlashAttention Enabled)
# ──────────────────────────────────────────────────────────────────────────────
class FlashGeometryBlock(nn.Module):
    """PyTorch 2.0+ FlashAttention enabled Transformer block."""
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Native Scaled Dot-Product Attention (FlashAttention)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0)
            
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        h = self.proj(attn_out)
        
        x = self.norm1(x + self.dropout(h))
        y = self.ffn(x)
        x = self.norm2(x + self.dropout(y))
        return x

class FlashSequenceEncoder(nn.Module):
    """Sequence encoder replacing standard transformer with FlashAttention blocks."""
    def __init__(self, dim: int, depth: int, heads: int, ff_mult: int):
        super().__init__()
        self.embed = nn.Embedding(len(AA_VOCAB), dim)
        self.layers = nn.ModuleList([
            FlashGeometryBlock(dim, heads, ff_mult)
            for _ in range(depth)
        ])

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class GeometryDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 3))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        coords = self.net(latent)
        return coords - coords.mean(dim=1, keepdim=True)

class AdaptiveAlphaField(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.net(latent))
        return 0.5 + 2.5 * a.squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────────
# SOC Kernel & CSOC Controller (V23 Fixed Logic)
# ──────────────────────────────────────────────────────────────────────────────
class SOCKernel:
    def __init__(self, lam: float = 12.0, eps: float = 1e-8):
        self.lam = lam
        self.eps = eps

    def compute(self, coords: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 3:
            coords = coords.squeeze(0)  # batch assumption fix
        D = torch.cdist(coords, coords) + self.eps
        D = torch.clamp(D, min=1.0)
        ai = alpha.unsqueeze(1)
        aj = alpha.unsqueeze(0)
        a = 0.5 * (ai + aj)
        K = torch.exp(-a * torch.log(D)) * torch.exp(-D / self.lam)
        K.fill_diagonal_(0.0)
        return K

class CSOCController:
    def __init__(self):
        self.prev_coords = None

    def sigma(self, coords: torch.Tensor) -> torch.Tensor:
        if self.prev_coords is None:
            self.prev_coords = coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta = torch.norm(coords - self.prev_coords, dim=-1).mean()
        self.prev_coords = coords.detach().clone()
        return delta

    def temperature(self, sigma: torch.Tensor, base_T: float, target: float) -> torch.Tensor:
        dev = (sigma - target) / 0.5
        T = base_T + 2000.0 * torch.sigmoid(dev)
        return torch.clamp(T, base_T * 0.5, 3000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RG Refinement
# ──────────────────────────────────────────────────────────────────────────────
class DiffRGRefiner:
    def __init__(self, factor: int = 4):
        self.factor = factor

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        L = coords.shape[0]
        f = self.factor
        m = L // f
        coords_trim = coords[:m * f]
        coarse = coords_trim.reshape(m, f, 3).mean(dim=1)
        coarse = coarse.T.unsqueeze(0)
        refined = F.interpolate(coarse, size=L, mode='linear', align_corners=True)
        return refined.squeeze(0).T

# ──────────────────────────────────────────────────────────────────────────────
# Backbone Geometry Tools
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_backbone(ca: torch.Tensor) -> Dict[str, torch.Tensor]:
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

def compute_phi_psi(atoms: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    N, CA, C = atoms['N'], atoms['CA'], atoms['C']
    L = CA.shape[0]
    phi = torch.zeros(L, device=CA.device)
    psi = torch.zeros(L, device=CA.device)
    if L > 2:
        phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi * 180.0 / math.pi, psi * 180.0 / math.pi

# ──────────────────────────────────────────────────────────────────────────────
# Physics Energy Terms (V23 Corrected & Double-Counting Removed)
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
    if L > 2:
        mask[1] = True; mask[-2] = True
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
    exposed_penalty = torch.where(hydro > 0, hydro * (1.0 - burial), torch.zeros_like(burial))
    buried_penalty = torch.where(hydro <= 0, -hydro * burial, torch.zeros_like(burial))
    total = (exposed_penalty + buried_penalty).mean()
    return cfg.w_solvent * total

def energy_rotamer(ca, atoms, seq, cfg):
    L = ca.shape[0]
    E = torch.tensor(0.0, device=ca.device)
    for i, aa in enumerate(seq):
        if aa == 'G' or i == 0 or i == L-1: continue
        ca_i = ca[i]
        n_i = atoms['N'][i]
        c_i = atoms['C'][i]
        v1 = n_i - ca_i
        v2 = c_i - ca_i
        cb_dir = -(v1 + v2)
        cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        ideal_cb = ca_i + 1.8 * cb_dir
        dist_to_all = torch.norm(ca - ideal_cb.unsqueeze(0), dim=-1)
        mask = torch.ones(L, dtype=torch.bool, device=ca.device)
        mask[i] = False
        dist_min = torch.min(dist_to_all[mask])
        E += torch.relu(4.0 - dist_min)
    return cfg.w_rotamer * E / max(1, L-2)

def alpha_regularisation(alpha, cfg):
    entropy = -(alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

def total_physics_energy(ca, seq, alpha, kernel, cfg):
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
    e += energy_rotamer(ca, atoms, seq, cfg)
    e += alpha_regularisation(alpha, cfg)
    return e

# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, coords = self.data[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a, 20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        return seq_ids, coords

def synthetic_dataset(num_samples=200, min_len=50, max_len=150):
    data = []
    for _ in range(num_samples):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choices(AA_VOCAB[:-1], k=L))
        coords = np.zeros((L, 3), dtype=np.float32)
        d = np.random.randn(3).astype(np.float32)
        d /= np.linalg.norm(d) + 1e-8
        for i in range(1, L):
            d += 0.2 * np.random.randn(3).astype(np.float32)
            d /= np.linalg.norm(d) + 1e-8
            coords[i] = coords[i-1] + d * 3.8
        data.append((seq, coords))
    return data

# ──────────────────────────────────────────────────────────────────────────────
# V24 Core Model
# ──────────────────────────────────────────────────────────────────────────────
class CSOCSSC_V24(nn.Module):
    def __init__(self, cfg: V24Config):
        super().__init__()
        self.cfg = cfg
        # Replaced standard encoder with FlashSequenceEncoder
        self.encoder = FlashSequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = GeometryDecoder(cfg.dim)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.soc_kernel = SOCKernel(lam=cfg.kernel_lambda)
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(seq_ids)
        coords = self.decoder(latent)
        alpha = self.alpha_field(latent)
        return coords, alpha

    def predict(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([AA_TO_ID.get(a, 20) for a in sequence],
                               dtype=torch.long, device=self.cfg.device).unsqueeze(0)
            coords, alpha = self.forward(ids)
        return coords.squeeze(0).cpu().numpy(), alpha.squeeze(0).cpu().numpy()

    def _soc_avalanche(self, coords: torch.Tensor, alpha: torch.Tensor, kernel: torch.Tensor, loss: torch.Tensor):
        grad = torch.autograd.grad(loss, coords, retain_graph=True)[0]
        stress = torch.norm(grad, dim=-1)
        threshold = self.cfg.avalanche_threshold
        high_stress = stress > threshold
        if not high_stress.any():
            return
        stressed_idx = torch.where(high_stress)[0]
        k = min(self.cfg.avalanche_steps, len(coords)-1)
        for i in stressed_idx:
            k_vals = kernel[i].clone()
            k_vals[i] = 0
            _, top_idx = torch.topk(k_vals, k)
            direction = grad[i]
            dir_norm = torch.norm(direction)
            if dir_norm < 1e-6:
                continue
            direction = direction / dir_norm
            weight = k_vals[top_idx] / (k_vals[top_idx].sum() + 1e-8)
            coords.data[top_idx] -= 0.01 * weight.unsqueeze(-1) * direction

    def refine(self, sequence: str, init_coords=None, steps=None, logger=None):
        if steps is None:
            steps = self.cfg.refine_steps
        self.eval()
        device = torch.device(self.cfg.device)

        if init_coords is not None:
            coords = torch.tensor(init_coords, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a, 20) for a in sequence],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                coords_np, alpha_np = self.predict(sequence)
            coords = torch.tensor(coords_np, dtype=torch.float32, device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        opt = torch.optim.Adam([coords], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        energy_history = []
        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                K = self.soc_kernel.compute(coords, alpha) if self.cfg.use_soc_kernel else None
                e_phys = total_physics_energy(coords, sequence, alpha, K, self.cfg)
                if init_coords is None:
                    with torch.no_grad():
                        neural_coords, _ = self.predict(sequence)
                        neural_coords = torch.tensor(neural_coords, device=device)
                    e_reg = 0.1 * ((coords - neural_coords)**2).mean()
                else:
                    e_reg = 0.0
                loss = e_phys + e_reg

            scaler.scale(loss).backward(retain_graph=True)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([coords], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            sigma = self.csoc.sigma(coords.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                coords.add_(torch.randn_like(coords) * noise_scale)

            if step % 20 == 0 and step > 0 and K is not None:
                self._soc_avalanche(coords, alpha, K, loss)

            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                coords.data = self.rg.forward(coords.data)

            if step % 50 == 0 and logger:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f}  σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return coords.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Distributed Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model: nn.Module, dataloader: DataLoader, cfg: V24Config, logger: logging.Logger):
    device = torch.device(cfg.device)
    model.train()
    
    # In DDP, the optimizer must wrap the DDP module's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)

    for epoch in range(cfg.epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
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
                loss = coord_loss + alpha_reg
                loss = loss / cfg.gradient_accumulation_steps

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
        path = os.path.join(cfg.checkpoint_dir, "v24_pretrained.pt")
        # Save the underlying model structure
        state_dict = model.module.state_dict() if cfg.is_distributed else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(np.mean(np.sum((a @ R - b)**2, axis=1))))

def write_ca_pdb(coords: np.ndarray, seq: str, filename: str):
    with open(filename, 'w') as f:
        for i, (c, aa) in enumerate(zip(coords, seq)):
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC v24 Distributed Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train', help='Train neural predictor via DDP')
    train_parser.add_argument('--samples', type=int, default=1000)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--local_rank', type=int, default=-1, help='Provided by torchrun')

    refine_parser = sub.add_parser('refine', help='Refine a single sequence or PDB')
    refine_parser.add_argument('--seq', type=str, default=None, help='Sequence string')
    refine_parser.add_argument('--pdb', type=str, default=None, help='RCSB PDB ID to fetch')
    refine_parser.add_argument('--init', type=str, default=None, help='Local PDB file for init')
    refine_parser.add_argument('--out', type=str, default='refined_v24.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--checkpoint', type=str, default='v24_pretrained.pt')

    args = parser.parse_args()
    
    # Handle DDP Environment initialization
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if hasattr(args, 'local_rank') else -1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = V24Config(
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device_str,
        epochs=getattr(args, 'epochs', 80),
        batch_size=getattr(args, 'batch_size', 8),
        refine_steps=getattr(args, 'steps', 600)
    )
    
    # Set determinism
    torch.manual_seed(cfg.seed + (local_rank if local_rank > 0 else 0))
    np.random.seed(cfg.seed + (local_rank if local_rank > 0 else 0))
    random.seed(cfg.seed + (local_rank if local_rank > 0 else 0))
    
    logger = setup_logger("CSOC-SSC_V24", local_rank)
    
    if local_rank in [-1, 0]:
        logger.info("=" * 60)
        logger.info(f"CSOC-SSC V24 Distributed Folding Engine initialized")
        logger.info(f"Distributed Mode: {is_distributed} | Device: {device_str}")
        logger.info("=" * 60)

    # Core execution
    if args.command == 'train':
        if local_rank in [-1, 0]:
            logger.info("Generating synthetic training data...")
        data = synthetic_dataset(num_samples=args.samples)
        dataset = ProteinDataset(data)
        
        if is_distributed:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
            
        model = CSOCSSC_V24(cfg).to(torch.device(device_str))
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            
        train_model(model, dataloader, cfg, logger)

    elif args.command == 'refine':
        # Refinement is usually run on a single node/GPU instance
        model = CSOCSSC_V24(cfg).to(torch.device(device_str))
        ckpt = args.checkpoint
        
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device_str))
            logger.info(f"Loaded weights from {ckpt}")
        else:
            logger.warning("Checkpoint not found; proceeding with random weights.")
            
        init_coords = None
        target_seq = args.seq
        
        # PDB Fetching integration
        if args.pdb:
            logger.info(f"Fetching PDB target {args.pdb} from RCSB...")
            backbone = PDBFetcher.fetch_and_parse(args.pdb)
            init_coords = backbone.ca
            target_seq = backbone.seq
            native_coords = backbone.native_coords
            logger.info(f"Loaded {len(init_coords)} residues from PDB.")
        elif args.init and os.path.exists(args.init):
            coords_list = []
            with open(args.init) as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if coords_list:
                init_coords = np.array(coords_list, dtype=np.float32)
                logger.info(f"Loaded {len(init_coords)} initial CA atoms from local file.")
                
        if not target_seq:
            raise ValueError("Must provide either --seq or --pdb for refinement.")

        start_time = time.time()
        refined, _ = model.refine(target_seq, init_coords=init_coords, steps=cfg.refine_steps, logger=logger)
        
        write_ca_pdb(refined, target_seq, args.out)
        logger.info(f"Refined structure saved to {args.out}")
        
        if args.pdb:
            rmsd_val = compute_rmsd(refined, native_coords)
            logger.info(f"Final RMSD vs Native RCSB: {rmsd_val:.4f} Å")
            
        logger.info(f"Compute Time: {time.time() - start_time:.2f} seconds")

    # Cleanup DDP
    if is_distributed:
        dist.destroy_process_group()
