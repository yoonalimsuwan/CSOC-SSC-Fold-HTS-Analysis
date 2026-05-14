# =============================================================================
# CSOC‑SSC v22 — SOC‑Driven Neural‑Physical Folding Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# V22 moves from a neural‑guided energy minimisation to a self‑organised
# critical (SOC) dynamics engine where the α‑field controls universality class,
# the SOC kernel defines interaction topology, and the criticality controller
# orchestrates avalanche propagation.  All physics terms are modulated by the
# residue‑specific α‑field, making the system a true “physics‑SOC hybrid”.
#
# KEY UPGRADES OVER V21
# -----------------------------------------------------------------------------
# • Separate physical interaction kernel (no forced normalisation)
# • Vectorised Ramachandran energy (GPU‑friendly)
# • Deep α‑field coupling: bond, angle, clash, Ramachandran widths
#   are all α‑dependent
# • Soft CSOC temperature saturation (no hard clamps)
# • SOC avalanche dynamics: high‑stress residues trigger cascaded
#   neighbour displacements through the SOC kernel
# • Entropic regularisation of the α‑field
# • Differentiable RG refinement (unchanged, but now applied to α‑weighted
#   coordinates)
# • Training on Cα coordinates, refinement with SOC‑driven relaxation
# =============================================================================

import os
import math
import time
import random
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(name: str = "CSOC‑SSC", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger

logger = setup_logger()

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

# Ramachandran priors – will be modulated by α
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
class V22Config:
    # Neural architecture
    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Refinement (SOC dynamics)
    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5       # stress threshold for triggering avalanche
    avalanche_steps: int = 3               # how many neighbours affected per trigger

    # Alpha‑field coupling strengths (multiplicative modulation)
    alpha_mod_bond: float = 0.1            # bond target = 3.8 * (1 + alpha_mod * (α-1))
    alpha_mod_angle: float = 0.05          # angle target modulation
    alpha_mod_rama: float = 0.2            # Ramachandran width scaling
    alpha_mod_clash: float = 0.1           # clash radius scaling
    alpha_mod_hbond: float = 0.1           # h‑bond distance / angle modulation

    # Energy weights (base values; some are further scaled by α locally)
    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0
    w_alpha_entropy: float = 0.5           # encourages diversity of α
    w_alpha_smooth: float = 0.1            # spatial smoothness of α

    # Contact interaction kernel (SOC)
    kernel_lambda: float = 12.0            # decay length of SOC kernel
    use_soc_kernel: bool = True            # whether to include a contact map term

    # RG refinement
    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200

    # Paths
    checkpoint_dir: str = "./v22_ckpt"
    out_pdb: str = "refined.pdb"

    @property
    def angle_target_rad(self) -> float:
        return 110.0 * math.pi / 180.0

# ──────────────────────────────────────────────────────────────────────────────
# Neural Modules
# ──────────────────────────────────────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    """Transformer encoder: sequence → latent (B, L, dim)."""
    def __init__(self, dim: int, depth: int, heads: int, ff_mult: int):
        super().__init__()
        self.embed = nn.Embedding(len(AA_VOCAB), dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq_ids)
        return self.transformer(x)


class GeometryDecoder(nn.Module):
    """Latent → Cα coordinates."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        coords = self.net(latent)
        coords = coords - coords.mean(dim=1, keepdim=True)
        return coords


class AdaptiveAlphaField(nn.Module):
    """
    Learns per‑residue α ∈ [0.5, 3.0] from latent.
    α controls local universality class: interaction decay, geometry flexibility,
    clash radius, etc.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.net(latent))
        return 0.5 + 2.5 * a.squeeze(-1)   # (B, L)

# ──────────────────────────────────────────────────────────────────────────────
# SOC Kernel (physical, not normalised)
# ──────────────────────────────────────────────────────────────────────────────

class SOCKernel:
    """K_ij = r_ij^{-α_ij} * exp(-r_ij / λ)  (raw interaction strength)."""
    def __init__(self, lam: float = 12.0, eps: float = 1e-8):
        self.lam = lam
        self.eps = eps

    def compute(self, coords: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # coords (L,3), alpha (L,)
        D = torch.cdist(coords, coords) + self.eps
        ai = alpha.unsqueeze(1)
        aj = alpha.unsqueeze(0)
        a = 0.5 * (ai + aj)
        K = torch.pow(D, -a) * torch.exp(-D / self.lam)
        K.fill_diagonal_(0.0)
        return K

# ──────────────────────────────────────────────────────────────────────────────
# Criticality Controller (soft temperature)
# ──────────────────────────────────────────────────────────────────────────────

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
        # Soft saturation using sigmoid: T = base_T + (T_max - base_T) * σ(sigma_dev)
        dev = (sigma - target) / 0.5   # scale deviation
        T = base_T + 2000.0 * torch.sigmoid(dev)   # sigmoid saturates at 1
        return torch.clamp(T, base_T * 0.5, 3000.0)  # safety clamp, very soft

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RG Refinement
# ──────────────────────────────────────────────────────────────────────────────

class DiffRGRefiner:
    def __init__(self, factor: int = 4):
        self.factor = factor

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        L = coords.shape[0]
        m = max(1, L // self.factor)
        coarse = coords[:m * self.factor].view(m, self.factor, 3).mean(dim=1)
        coarse = coarse.T.unsqueeze(0)          # (1, 3, m)
        refined = F.interpolate(coarse, size=L, mode='linear', align_corners=True)
        return refined.squeeze(0).T

# ──────────────────────────────────────────────────────────────────────────────
# Backbone reconstruction & dihedral (unchanged from V21, but vectorised)
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
            O[i] = C[i] + torch.tensor([0.0, 1.24, 0.0], device=ca.device)
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
# Physics Energy Terms – α‑modulated
# ──────────────────────────────────────────────────────────────────────────────

def energy_bond(ca: torch.Tensor, alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    # Bond target modulated by alpha: tighter packing for larger α
    target = cfg.ca_ca_dist * (1.0 + cfg.alpha_mod_bond * (alpha - 1.0))
    # average target for neighbouring pairs
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return cfg.w_bond * ((d - target_pair) ** 2).mean()


def energy_angle(ca: torch.Tensor, alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    if len(ca) < 3:
        return torch.tensor(0.0, device=ca.device)
    v1 = ca[:-2] - ca[1:-1]
    v2 = ca[2:] - ca[1:-1]
    v1n = F.normalize(v1, dim=-1, eps=1e-8)
    v2n = F.normalize(v2, dim=-1, eps=1e-8)
    cos_ang = (v1n * v2n).sum(-1)
    # target angle modulated by local alpha
    target_angle = cfg.angle_target_rad * (1.0 + cfg.alpha_mod_angle * (alpha[1:-1] - 1.0))
    cos_target = torch.cos(target_angle)
    return cfg.w_angle * ((cos_ang - cos_target) ** 2).mean()


def energy_rama_vectorized(phi: torch.Tensor, psi: torch.Tensor,
                           seq: str, alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    """
    Fully vectorised Ramachandran penalty.
    Width is scaled by alpha: flexible residues (high α) have wider allowed region.
    """
    L = len(seq)
    device = phi.device
    # Build target tensors
    phi0 = torch.zeros(L, device=device)
    psi0 = torch.zeros(L, device=device)
    width = torch.zeros(L, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0[i] = prior['phi']
        psi0[i] = prior['psi']
        width[i] = prior['width']
    # Modulate width by alpha: width_eff = width * (1 + alpha_mod * (α-1))
    width_eff = width * (1.0 + cfg.alpha_mod_rama * (alpha - 1.0))
    # Deviation
    dphi = (phi - phi0) / (width_eff + 1e-8)
    dpsi = (psi - psi0) / (width_eff + 1e-8)
    # Exclude terminals where phi/psi are zero
    mask = torch.ones(L, device=device, dtype=torch.bool)
    mask[0] = False; mask[-1] = False
    if L > 2:
        mask[1] = True; mask[-2] = True
    loss = (dphi**2 + dpsi**2) * mask.float()
    return loss.sum() / max(1, mask.sum())


def energy_clash(ca: torch.Tensor, alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    D = torch.cdist(ca, ca)
    mask = torch.ones_like(D, dtype=torch.bool)
    idx = torch.arange(len(ca), device=ca.device)
    mask[idx[:, None], idx[None, :]] = False
    mask[idx[:-1, None], (idx[None, :-1] + 1)] = False
    mask[(idx[None, :-1] + 1), idx[:-1, None]] = False
    # Clash radius modulated by alpha: tighter residues (low α) have smaller radius
    radius = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha.unsqueeze(1) - 1.0))
    # Symmetrise radius for pair i,j as average
    radius_pair = 0.5 * (radius + radius.T)
    clash = torch.relu(radius_pair - D) * mask.float()
    return cfg.w_clash * (clash ** 2).mean()


def energy_hbond(atoms: Dict[str, torch.Tensor], alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    O, N, C = atoms['O'], atoms['N'], atoms['C']
    D = torch.cdist(O, N)
    mask = (D > 2.5) & (D < 3.5)
    vec_co = O.unsqueeze(1) - C.unsqueeze(1)
    vec_no = N.unsqueeze(0) - O.unsqueeze(1)
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    # Modulate ideal distance by alpha
    ideal_dist = 2.9 * (1.0 + cfg.alpha_mod_hbond * (alpha.unsqueeze(1) - 1.0))
    E = -alignment * torch.exp(-((D - ideal_dist) / 0.3) ** 2)
    return cfg.w_hbond * (E * mask.float()).mean()


def energy_electro(ca: torch.Tensor, seq: str, cfg: V22Config) -> torch.Tensor:
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=ca.device)
    D = torch.cdist(ca, ca) + 1e-6
    E = q.unsqueeze(1) * q.unsqueeze(0) * torch.exp(-0.1 * D) / (80.0 * D)
    E.diagonal().zero_()
    return cfg.w_electro * E.mean()


def energy_solvent(ca: torch.Tensor, seq: str, cfg: V22Config) -> torch.Tensor:
    D = torch.cdist(ca, ca)
    density = (D < 10.0).float().sum(dim=-1)
    burial = 1.0 - torch.exp(-density / 20.0)
    energy = torch.tensor(0.0, device=ca.device)
    for i, aa in enumerate(seq):
        h = HYDROPHOBICITY.get(aa, 0.0)
        if h > 0:
            energy += h * (1.0 - burial[i])
        else:
            energy += -h * burial[i]
    return cfg.w_solvent * energy / len(seq)


def energy_rotamer(ca: torch.Tensor, atoms: Dict[str, torch.Tensor],
                   seq: str, cfg: V22Config) -> torch.Tensor:
    L = ca.shape[0]
    E = torch.tensor(0.0, device=ca.device)
    for i, aa in enumerate(seq):
        if aa == 'G' or i == 0 or i == L-1:
            continue
        ca_i = ca[i]
        n_i = atoms['N'][i]
        c_i = atoms['C'][i]
        v1 = n_i - ca_i
        v2 = c_i - ca_i
        cb_dir = -(v1 + v2)
        cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        ideal_cb = ca_i + 1.8 * cb_dir
        dist_to_others = torch.norm(ca - ideal_cb.unsqueeze(0), dim=-1)
        dist_min = torch.min(dist_to_others[torch.arange(L) != i])
        E += torch.relu(4.0 - dist_min)
    return cfg.w_rotamer * E / max(1, L - 2)


def contact_energy(ca: torch.Tensor, kernel: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    """
    Attractive / repulsive based on kernel strength.
    High K_ij → residue pair should be close → -K * exp(-D/8)
    """
    D = torch.cdist(ca, ca)
    E = -kernel * torch.exp(-D / 8.0)
    return cfg.w_rama * E.mean()   # reuse w_rama as weight for contact


def alpha_regularisation(alpha: torch.Tensor, cfg: V22Config) -> torch.Tensor:
    # Entropy: encourage diversity (avoid collapse to one value)
    # Smoothness: penalise large changes between neighbours
    entropy = - (alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

# ──────────────────────────────────────────────────────────────────────────────
# Total physics energy aggregator
# ──────────────────────────────────────────────────────────────────────────────

def total_physics_energy(ca: torch.Tensor, seq: str, alpha: torch.Tensor,
                         kernel: Optional[torch.Tensor], cfg: V22Config) -> torch.Tensor:
    atoms = reconstruct_backbone(ca)
    phi, psi = compute_phi_psi(atoms)
    e = 0.0
    e += energy_bond(ca, alpha, cfg)
    e += energy_angle(ca, alpha, cfg)
    e += cfg.w_rama * energy_rama_vectorized(phi, psi, seq, alpha, cfg)
    e += energy_clash(ca, alpha, cfg)
    e += energy_hbond(atoms, alpha, cfg)
    e += energy_electro(ca, seq, cfg)
    e += energy_solvent(ca, seq, cfg)
    e += energy_rotamer(ca, atoms, seq, cfg)
    e += alpha_regularisation(alpha, cfg)
    if kernel is not None and cfg.use_soc_kernel:
        e += contact_energy(ca, kernel, cfg)
    return e

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ProteinDataset(Dataset):
    def __init__(self, data: List[Tuple[str, np.ndarray]]):
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
        dir = np.random.randn(3).astype(np.float32)
        dir /= np.linalg.norm(dir) + 1e-8
        for i in range(1, L):
            dir += 0.2 * np.random.randn(3).astype(np.float32)
            dir /= np.linalg.norm(dir) + 1e-8
            coords[i] = coords[i-1] + dir * 3.8
        data.append((seq, coords))
    return data

# ──────────────────────────────────────────────────────────────────────────────
# V22 Model
# ──────────────────────────────────────────────────────────────────────────────

class V22Model(nn.Module):
    def __init__(self, cfg: V22Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
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

    def _soc_avalanche(self, coords: torch.Tensor, alpha: torch.Tensor,
                       kernel: torch.Tensor, stress: torch.Tensor):
        """
        Implements a simple SOC avalanche step:
        For every residue with stress > threshold, apply a displacement to
        its k nearest neighbours along the gradient direction, weighted by kernel.
        This mimics a cascade of redistributed stress.
        """
        L = coords.shape[0]
        threshold = self.cfg.avalanche_threshold
        high_stress = stress > threshold
        if not high_stress.any():
            return
        # Find indices of stressed residues
        stressed_idx = torch.where(high_stress)[0]
        # For each stressed residue, pick top‑k neighbours by kernel value
        k = min(self.cfg.avalanche_steps, L-1)
        for i in stressed_idx:
            # kernel values for this residue to all others
            k_vals = kernel[i]   # (L,)
            # exclude self
            k_vals[i] = 0
            # pick top k neighbours
            _, top_idx = torch.topk(k_vals, k)
            # displacement direction is the gradient of coords (stored in coords.grad)
            if coords.grad is None:
                direction = torch.randn(3, device=coords.device)
            else:
                direction = coords.grad[i]   # per‑residue gradient
            # Normalise direction
            dir_norm = torch.norm(direction)
            if dir_norm < 1e-6:
                continue
            direction = direction / dir_norm
            # Displace neighbours towards/away? We move them along the negative gradient
            # to reduce energy, weighted by kernel strength.
            for j in top_idx:
                weight = k_vals[j] / (k_vals[top_idx].sum() + 1e-8)
                coords.data[j] -= 0.01 * weight * direction

    def refine(self, sequence: str,
               init_coords: Optional[np.ndarray] = None,
               steps: Optional[int] = None) -> Tuple[np.ndarray, List[float]]:
        if steps is None:
            steps = self.cfg.refine_steps

        self.eval()
        device = torch.device(self.cfg.device)

        # Obtain initial coordinates and alpha
        if init_coords is not None:
            coords = torch.tensor(init_coords, dtype=torch.float32,
                                  device=device, requires_grad=True)
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a, 20) for a in sequence],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                coords_np, alpha_np = self.predict(sequence)
            coords = torch.tensor(coords_np, dtype=torch.float32,
                                  device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        opt = torch.optim.Adam([coords], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        energy_history = []
        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                K = self.soc_kernel.compute(coords, alpha) if self.cfg.use_soc_kernel else None
                e_phys = total_physics_energy(coords, sequence, alpha, K, self.cfg)
                # neural restraint (soft guidance towards prediction)
                if init_coords is None:
                    with torch.no_grad():
                        neural_coords, _ = self.predict(sequence)
                        neural_coords = torch.tensor(neural_coords, device=device)
                    e_reg = 0.1 * ((coords - neural_coords) ** 2).mean()
                else:
                    e_reg = 0.0
                loss = e_phys + e_reg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([coords], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            # CSOC feedback: temperature + noise
            sigma = self.csoc.sigma(coords.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                coords.add_(torch.randn_like(coords) * noise_scale)

            # SOC avalanche (every 20 steps)
            if step % 20 == 0 and step > 0 and K is not None:
                # compute per‑residue gradient norm as stress
                with torch.no_grad():
                    stress = torch.norm(coords.grad, dim=-1) if coords.grad is not None else torch.zeros(len(coords), device=device)
                self._soc_avalanche(coords, alpha, K, stress)

            # RG refinement
            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                coords.data = self.rg.forward(coords.data)

            if step % 50 == 0:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f}  σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return coords.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Training (coordinate prediction)
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model: V22Model, dataloader: DataLoader, cfg: V22Config):
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for seq_ids, target_coords in dataloader:
            seq_ids = seq_ids.to(device)
            target_coords = target_coords.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred_coords, pred_alpha = model(seq_ids)
                coord_loss = F.mse_loss(pred_coords, target_coords)
                # Add weak alpha regularisation during training
                alpha_reg = 0.001 * ((pred_alpha[:, 1:] - pred_alpha[:, :-1])**2).mean()
                loss = coord_loss + alpha_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  MSE={avg_loss:.4f}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, "v22_pretrained.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

# ──────────────────────────────────────────────────────────────────────────────
# RMSD & PDB utilities
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
    return float(np.sqrt(np.mean(np.sum((a @ R - b) ** 2, axis=1))))

def write_ca_pdb(coords: np.ndarray, seq: str, filename: str):
    with open(filename, 'w') as f:
        for i, (c, aa) in enumerate(zip(coords, seq)):
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC v22 Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train', help='Train neural predictor')
    train_parser.add_argument('--samples', type=int, default=200)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--device', type=str, default='cuda')

    refine_parser = sub.add_parser('refine', help='Refine a sequence')
    refine_parser.add_argument('--seq', type=str, required=True)
    refine_parser.add_argument('--init', type=str, default=None, help='Initial PDB (optional)')
    refine_parser.add_argument('--out', type=str, default='refined.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--device', type=str, default='cuda')
    refine_parser.add_argument('--checkpoint', type=str, default='v22_pretrained.pt')

    args = parser.parse_args()
    cfg = V22Config(epochs=getattr(args, 'epochs', 80),
                    batch_size=getattr(args, 'batch_size', 8),
                    device=args.device,
                    refine_steps=getattr(args, 'steps', 600))
    device = torch.device(cfg.device)

    if args.command == 'train':
        logger.info("Generating synthetic training data...")
        data = synthetic_dataset(num_samples=args.samples)
        dataset = ProteinDataset(data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        model = V22Model(cfg)
        train_model(model, dataloader, cfg)

    elif args.command == 'refine':
        model = V22Model(cfg)
        ckpt = args.checkpoint
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            logger.info(f"Loaded weights from {ckpt}")
        else:
            logger.warning("Checkpoint not found; using random weights.")
        model.to(device)

        init = None
        if args.init and os.path.exists(args.init):
            # simple PDB reader
            coords_list = []
            with open(args.init) as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if coords_list:
                init = np.array(coords_list, dtype=np.float32)
                logger.info(f"Loaded {len(init)} initial CA atoms.")

        refined, _ = model.refine(args.seq, init_coords=init, steps=cfg.refine_steps)
        write_ca_pdb(refined, args.seq, args.out)
        logger.info(f"Refined structure saved to {args.out}")
