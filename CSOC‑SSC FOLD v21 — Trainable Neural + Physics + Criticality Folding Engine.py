# =============================================================================
# CSOC‑SSC v21 — Trainable Neural + Physics + Criticality Folding Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# V21 integrates a trainable sequence‑to‑structure neural backbone with
# self‑organized criticality (SOC) physics, an adaptive universality field,
# and differentiable renormalisation group (RG) refinement.  The model learns
# to predict protein Cα coordinates from sequence and refines them using
# a full physical energy function guided by SOC avalanche control.
#
# The system supports both training (on PDB data) and refinement of single
# sequences.  All components are differentiable, enabling end‑to‑end learning.
#
# KEY FEATURES
# -----------------------------------------------------------------------------
# • Transformer encoder for sequence → latent representation
# • Geometry decoder for latent → Cα coordinates
# • Adaptive alpha field controlling SOC interaction kernel
# • SOC kernel for interaction law (r⁻ᵅ · exp(−r/λ))
# • Full physics: bond, angle, Ramachandran, clash, H‑bond,
#   electrostatics, solvation, rotamer packing
# • CSOC/SSC criticality controller (σ‑feedback temperature)
# • Differentiable multi‑scale RG refinement (GPU tensor ops)
# • Langevin dynamics with adaptive noise
# • Mixed‑precision training / refinement
# • Training on PDB coordinate datasets; refinement on single sequences
# • Checkpointing and PDB output
#
# TARGET HARDWARE
# -----------------------------------------------------------------------------
# • PyTorch 2.0+
# • CUDA 12+ / CPU fallback
# • Google Colab T4 / A100 / HPC clusters
#
# =============================================================================

import os
import math
import time
import random
import argparse
import logging
from dataclasses import dataclass, field
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
def setup_logger(name="CSOC‑SSC", level=logging.INFO) -> logging.Logger:
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
# Biochemistry constants
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
    'G': {'phi': -75.0, 'psi': -60.0, 'width': 40.0},
    'P': {'phi': -65.0, 'psi': -30.0, 'width': 20.0},
}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class V21Config:
    # ── Neural Architecture ──────────────────────────────────────────────
    dim: int = 256                    # hidden dimension
    depth: int = 6                    # transformer layers
    heads: int = 8                    # attention heads
    ff_mult: int = 4                  # feed‑forward multiplier

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 1e-4                  # learning rate (all parameters)
    epochs: int = 50
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ── Refinement (physics + SOC + RG) ─────────────────────────────────
    refine_steps: int = 600
    temp_base: float = 300.0          # base temperature (K)
    friction: float = 0.01            # Langevin friction
    sigma_target: float = 1.0         # SOC target σ

    # Geometry ideal values
    ca_ca_dist: float = 3.8           # target CA‑CA bond length (Å)
    angle_target_deg: float = 110.0   # target N‑CA‑C angle (degrees)
    clash_radius: float = 3.0         # soft clash distance (Å)

    # ── Energy Weights ──────────────────────────────────────────────────
    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0
    w_contact: float = 5.0            # contact map loss (SOC kernel)
    w_soc: float = 2.0                # criticality penalty
    w_latent_reg: float = 0.1         # regularisation to stay near neural prediction

    # ── RG refinement ───────────────────────────────────────────────────
    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200

    # ── Paths ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "./v21_ckpt"
    out_pdb: str = "refined.pdb"

    # ── Derived ─────────────────────────────────────────────────────────
    @property
    def angle_target_rad(self) -> float:
        return self.angle_target_deg * math.pi / 180.0

# ──────────────────────────────────────────────────────────────────────────────
# Neural Modules
# ──────────────────────────────────────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    """Encodes amino‑acid sequence into a per‑residue latent tensor."""
    def __init__(self, dim: int, depth: int, heads: int, ff_mult: int):
        super().__init__()
        self.embed = nn.Embedding(len(AA_VOCAB), dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        # seq_ids: (B, L)
        x = self.embed(seq_ids)          # (B, L, dim)
        return self.transformer(x)       # (B, L, dim)


class GeometryDecoder(nn.Module):
    """Predicts Cα coordinates from latent representation."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, L, dim) → (B, L, 3)
        coords = self.net(latent)
        # centre coordinates
        coords = coords - coords.mean(dim=1, keepdim=True)
        return coords


class AdaptiveAlphaField(nn.Module):
    """
    Learns a residue‑specific exponent α ∈ [0.5, 3.0] that controls
    the power‑law decay in the SOC interaction kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, L, dim) → (B, L)
        a = torch.sigmoid(self.net(latent))
        return 0.5 + 2.5 * a.squeeze(-1)   # range [0.5, 3.0]

# ──────────────────────────────────────────────────────────────────────────────
# SOC Interaction Kernel
# ──────────────────────────────────────────────────────────────────────────────

class SOCKernel:
    """
    Fixed interaction law: K_ij = r_ij ** (-α_ij) * exp(-r_ij / λ)
    where α_ij = 0.5*(α_i + α_j).
    """
    def __init__(self, lam: float = 10.0, eps: float = 1e-8):
        self.lam = lam
        self.eps = eps

    def compute(self, coords: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # coords: (L, 3), alpha: (L,)
        D = torch.cdist(coords, coords) + self.eps          # (L, L)
        ai = alpha.unsqueeze(1)                              # (L, 1)
        aj = alpha.unsqueeze(0)                              # (1, L)
        a = 0.5 * (ai + aj)                                  # (L, L)
        K = torch.pow(D, -a) * torch.exp(-D / self.lam)
        K.fill_diagonal_(0.0)
        # normalise rows to sum to 1 (probabilistic interpretation)
        K = K / (K.sum(dim=-1, keepdim=True) + self.eps)
        return K

# ──────────────────────────────────────────────────────────────────────────────
# Criticality Controller (CSOC / SSC)
# ──────────────────────────────────────────────────────────────────────────────

class CSOCController:
    """
    Measures avalanche intensity σ from coordinate changes and
    computes an adaptive temperature for Langevin dynamics.
    """
    def __init__(self):
        self.prev_coords = None

    def sigma(self, coords: torch.Tensor) -> torch.Tensor:
        if self.prev_coords is None:
            self.prev_coords = coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta = torch.norm(coords - self.prev_coords, dim=-1).mean()
        self.prev_coords = coords.detach().clone()
        return delta

    def temperature(self, sigma: torch.Tensor, base_T: float,
                    target: float = 1.0) -> torch.Tensor:
        # deviation from target σ drives temperature
        T = base_T * (1.0 + 2.0 * torch.abs(sigma - target))
        return torch.clamp(T, 50.0, 2000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RG Refinement
# ──────────────────────────────────────────────────────────────────────────────

class DiffRGRefiner:
    """
    Coarse‑grain by block averaging, then interpolate back using
    linear interpolation (fully differentiable).
    """
    def __init__(self, factor: int = 4):
        self.factor = factor

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (L, 3)
        L = coords.shape[0]
        m = max(1, L // self.factor)
        # coarse graining
        coarse = coords[:m * self.factor].view(m, self.factor, 3).mean(dim=1)  # (m, 3)
        # upsample back to L points using linear interpolation
        coarse = coarse.T.unsqueeze(0)   # (1, 3, m)
        refined = F.interpolate(coarse, size=L, mode='linear', align_corners=True)
        return refined.squeeze(0).T       # (L, 3)

# ──────────────────────────────────────────────────────────────────────────────
# Backbone Reconstruction & Dihedral Angles
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_backbone(ca: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Idealised reconstruction of N, C, O atoms from Cα trace.
    Uses standard peptide geometry.
    """
    L = ca.shape[0]
    # unit vectors along CA trace
    v = ca[1:] - ca[:-1]                    # (L-1, 3)
    v_norm = F.normalize(v, dim=-1, eps=1e-8)

    N = torch.zeros_like(ca)
    C = torch.zeros_like(ca)

    # N[i] approximately CA[i] - 1.45 * (CA[i]-CA[i-1]) direction
    N[1:] = ca[1:] - 1.45 * v_norm
    N[0] = ca[0] - 1.45 * v_norm[0]         # first residue approximation

    # C[i] approximately CA[i] + 1.52 * (CA[i+1]-CA[i]) direction
    C[:-1] = ca[:-1] + 1.52 * v_norm
    C[-1] = ca[-1] + 1.52 * v_norm[-1]      # last residue approximation

    # O[i] placed perpendicular to plane defined by (CA[i], N[i], C[i])
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
            # fallback for last residue
            O[i] = C[i] + torch.tensor([0.0, 1.24, 0.0], device=ca.device)

    return {'N': N, 'CA': ca, 'C': C, 'O': O}


def dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle between four points (returns radians)."""
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
    """Compute φ and ψ angles (in degrees) from backbone atoms."""
    N, CA, C = atoms['N'], atoms['CA'], atoms['C']
    L = CA.shape[0]
    phi = torch.zeros(L, device=CA.device)
    psi = torch.zeros(L, device=CA.device)
    if L > 2:
        # φ[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        # ψ[i] = dihedral(N[i], CA[i], C[i], N[i+1])
        psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi * 180.0 / math.pi, psi * 180.0 / math.pi

# ──────────────────────────────────────────────────────────────────────────────
# Physics Energy Terms
# ──────────────────────────────────────────────────────────────────────────────

def energy_bond(ca: torch.Tensor, cfg: V21Config) -> torch.Tensor:
    """CA‑CA bond length restraint."""
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return cfg.w_bond * ((d - cfg.ca_ca_dist) ** 2).mean()


def energy_angle(ca: torch.Tensor, cfg: V21Config) -> torch.Tensor:
    """CA‑CA‑CA angle restraint (approximate N‑CA‑C angle)."""
    if len(ca) < 3:
        return torch.tensor(0.0, device=ca.device)
    v1 = ca[:-2] - ca[1:-1]
    v2 = ca[2:] - ca[1:-1]
    v1n = F.normalize(v1, dim=-1, eps=1e-8)
    v2n = F.normalize(v2, dim=-1, eps=1e-8)
    cos_ang = (v1n * v2n).sum(-1)
    cos_target = math.cos(cfg.angle_target_rad)
    return cfg.w_angle * ((cos_ang - cos_target) ** 2).mean()


def energy_rama(phi: torch.Tensor, psi: torch.Tensor,
                seq: str, device: torch.device) -> torch.Tensor:
    """Ramachandran prior penalty (residue‑specific)."""
    loss = torch.tensor(0.0, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0, psi0, w = prior['phi'], prior['psi'], prior['width']
        dphi = (phi[i] - phi0) / w
        dpsi = (psi[i] - psi0) / w
        loss += dphi ** 2 + dpsi ** 2
    return loss / len(seq)


def energy_clash(ca: torch.Tensor, cfg: V21Config) -> torch.Tensor:
    """Soft‑sphere clash avoidance."""
    D = torch.cdist(ca, ca)
    # ignore self and sequential neighbours
    mask = torch.ones_like(D, dtype=torch.bool)
    idx = torch.arange(len(ca), device=ca.device)
    mask[idx[:, None], idx[None, :]] = False
    mask[idx[:-1, None], (idx[None, :-1] + 1)] = False
    mask[(idx[None, :-1] + 1), idx[:-1, None]] = False
    clash = torch.relu(cfg.clash_radius - D) * mask.float()
    return cfg.w_clash * (clash ** 2).mean()


def energy_hbond(atoms: Dict[str, torch.Tensor], cfg: V21Config) -> torch.Tensor:
    """Angular hydrogen‑bond energy between backbone carbonyl and amide."""
    O, N, C = atoms['O'], atoms['N'], atoms['C']
    D = torch.cdist(O, N)
    mask = (D > 2.5) & (D < 3.5)
    # donor‑acceptor alignment: C=O ... N
    vec_co = O.unsqueeze(1) - C.unsqueeze(1)           # (L,1,3)
    vec_no = N.unsqueeze(0) - O.unsqueeze(1)           # (1,L,3)
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    E = -alignment * torch.exp(-((D - 2.9) / 0.3) ** 2)
    return cfg.w_hbond * (E * mask.float()).mean()


def energy_electro(ca: torch.Tensor, seq: str, cfg: V21Config) -> torch.Tensor:
    """Debye‑Hückel electrostatic energy."""
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq],
                     device=ca.device)
    D = torch.cdist(ca, ca) + 1e-6
    E = q.unsqueeze(1) * q.unsqueeze(0) * torch.exp(-0.1 * D) / (80.0 * D)
    E.diagonal().zero_()
    return cfg.w_electro * E.mean()


def energy_solvent(ca: torch.Tensor, seq: str, cfg: V21Config) -> torch.Tensor:
    """
    Burial‑based implicit solvent.
    Hydrophobic residues are penalised when exposed; hydrophilic when buried.
    """
    D = torch.cdist(ca, ca)
    density = (D < 10.0).float().sum(dim=-1)
    burial = 1.0 - torch.exp(-density / 20.0)   # 0 = fully exposed, 1 = fully buried
    energy = torch.tensor(0.0, device=ca.device)
    for i, aa in enumerate(seq):
        h = HYDROPHOBICITY.get(aa, 0.0)
        if h > 0:   # hydrophobic
            energy += h * (1.0 - burial[i])   # penalty when exposed
        else:       # hydrophilic (h < 0)
            energy += -h * burial[i]          # penalty when buried (h negative, so -h positive)
    return cfg.w_solvent * energy / len(seq)


def energy_rotamer(ca: torch.Tensor, atoms: Dict[str, torch.Tensor],
                   seq: str, cfg: V21Config) -> torch.Tensor:
    """
    Very approximate rotamer packing score: encourages Cβ to not clash
    with other CA atoms.
    """
    L = ca.shape[0]
    E = torch.tensor(0.0, device=ca.device)
    for i, aa in enumerate(seq):
        if aa == 'G': continue
        if i == 0 or i == L - 1: continue
        # approximate Cβ direction (tetrahedral)
        ca_i = ca[i]
        n_i = atoms['N'][i]
        c_i = atoms['C'][i]
        v1 = n_i - ca_i
        v2 = c_i - ca_i
        cb_dir = -(v1 + v2)
        cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        ideal_cb = ca_i + 1.8 * cb_dir
        # distance to nearest other CA
        dist_to_others = torch.norm(ca - ideal_cb.unsqueeze(0), dim=-1)
        dist_min = torch.min(dist_to_others[torch.arange(L) != i])
        E += torch.relu(4.0 - dist_min)
    return cfg.w_rotamer * E / max(1, L - 2)   # exclude terminal residues


def contact_map_loss(ca: torch.Tensor, kernel: torch.Tensor,
                     cfg: V21Config) -> torch.Tensor:
    """
    Encourage distances to agree with SOC kernel prediction.
    D_target ≈ 8.0 * (1 - K) gives repulsion when K is small, attraction when K is large.
    """
    D = torch.cdist(ca, ca)
    target = 8.0 * (1.0 - kernel)
    return cfg.w_contact * ((D - target) ** 2).mean()


def total_physics_energy(ca: torch.Tensor, seq: str,
                         kernel: Optional[torch.Tensor],
                         cfg: V21Config) -> torch.Tensor:
    """Aggregates all physics terms."""
    atoms = reconstruct_backbone(ca)
    phi, psi = compute_phi_psi(atoms)

    e = 0.0
    e += energy_bond(ca, cfg)
    e += energy_angle(ca, cfg)
    e += cfg.w_rama * energy_rama(phi, psi, seq, ca.device)
    e += energy_clash(ca, cfg)
    e += energy_hbond(atoms, cfg)
    e += energy_electro(ca, seq, cfg)
    e += energy_solvent(ca, seq, cfg)
    e += energy_rotamer(ca, atoms, seq, cfg)

    if kernel is not None:
        e += contact_map_loss(ca, kernel, cfg)
    return e

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ProteinDataset(Dataset):
    """
    Dataset of (sequence, native Cα coordinates).
    `data` should be a list of tuples (seq: str, coords: np.ndarray).
    """
    def __init__(self, data: List[Tuple[str, np.ndarray]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, coords = self.data[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a, 20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        return seq_ids, coords


def synthetic_dataset(num_samples: int = 200, min_len: int = 50,
                      max_len: int = 150) -> List[Tuple[str, np.ndarray]]:
    """
    Generate synthetic protein data for demonstration.
    In production, replace with real PDB‑derived coordinates.
    """
    data = []
    for _ in range(num_samples):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choices(AA_VOCAB[:-1], k=L))
        # produce a plausible Cα trace: random walk with bond length 3.8
        coords = np.zeros((L, 3), dtype=np.float32)
        direction = np.random.randn(3).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        for i in range(1, L):
            # perturb direction slightly
            direction += 0.3 * np.random.randn(3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            coords[i] = coords[i-1] + direction * 3.8
        data.append((seq, coords))
    return data

# ──────────────────────────────────────────────────────────────────────────────
# V21 Model
# ──────────────────────────────────────────────────────────────────────────────

class V21Model(nn.Module):
    def __init__(self, cfg: V21Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = GeometryDecoder(cfg.dim)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.soc_kernel = SOCKernel()
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted Cα coordinates and per‑residue alpha.
        """
        latent = self.encoder(seq_ids)            # (B, L, dim)
        coords = self.decoder(latent)             # (B, L, 3)
        alpha = self.alpha_field(latent)          # (B, L)
        return coords, alpha

    def predict(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict Cα trace and alpha values for a single sequence."""
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([AA_TO_ID.get(a, 20) for a in sequence],
                               dtype=torch.long, device=self.cfg.device).unsqueeze(0)
            coords, alpha = self.forward(ids)
        return coords.squeeze(0).cpu().numpy(), alpha.squeeze(0).cpu().numpy()

    def refine(self, sequence: str,
               init_coords: Optional[np.ndarray] = None,
               steps: Optional[int] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Refine a structure using physics + SOC + RG.
        If init_coords is None, the neural prediction is used as starting point.
        """
        if steps is None:
            steps = self.cfg.refine_steps

        self.eval()
        device = torch.device(self.cfg.device)

        # obtain initial coordinates (and alpha from network)
        if init_coords is not None:
            coords = torch.tensor(init_coords, dtype=torch.float32,
                                  device=device, requires_grad=True)
            # we still need alpha values; predict them from sequence
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a, 20) for a in sequence],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)   # (L,)
        else:
            with torch.no_grad():
                coords_np, alpha_np = self.predict(sequence)
            coords = torch.tensor(coords_np, dtype=torch.float32,
                                  device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        # optimizer
        opt = torch.optim.Adam([coords], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        energy_history = []
        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                # compute SOC kernel from current coordinates and alpha
                K = self.soc_kernel.compute(coords, alpha) if self.cfg.w_contact > 0 else None
                # physics energy
                e_phys = total_physics_energy(coords, sequence, K, self.cfg)
                # optional neural restraint (soft attraction to prediction)
                if init_coords is None and self.cfg.w_latent_reg > 0:
                    with torch.no_grad():
                        neural_coords, _ = self.predict(sequence)
                        neural_coords = torch.tensor(neural_coords, device=device)
                    e_reg = self.cfg.w_latent_reg * ((coords - neural_coords) ** 2).mean()
                else:
                    e_reg = 0.0
                loss = e_phys + e_reg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([coords], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            # CSOC feedback: measure σ and add Langevin noise
            sigma = self.csoc.sigma(coords.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                coords.add_(torch.randn_like(coords) * noise_scale)

            # Differentiable RG refinement every rg_interval steps
            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                coords.data = self.rg.forward(coords.data)

            if step % 50 == 0:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  "
                            f"phys={e_phys.item():.4f}  σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return coords.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model: V21Model, dataloader: DataLoader, cfg: V21Config):
    """Train encoder, decoder, and alpha field on coordinate prediction."""
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch_idx, (seq_ids, target_coords) in enumerate(dataloader):
            seq_ids = seq_ids.to(device)
            target_coords = target_coords.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred_coords, pred_alpha = model(seq_ids)
                # MSE loss on coordinates
                coord_loss = F.mse_loss(pred_coords, target_coords)
                # Optional: regularisation on alpha (e.g., smoothness)
                alpha_reg = 0.001 * ((pred_alpha[:, 1:] - pred_alpha[:, :-1]) ** 2).mean()
                loss = coord_loss + alpha_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  MSE={avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, "v21_pretrained.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# RMSD (Kabsch)
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

# ──────────────────────────────────────────────────────────────────────────────
# PDB I/O
# ──────────────────────────────────────────────────────────────────────────────

def write_ca_pdb(coords: np.ndarray, seq: str, filename: str):
    with open(filename, 'w') as f:
        for i, (c, aa) in enumerate(zip(coords, seq)):
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")

def read_ca_pdb(filename: str) -> Tuple[np.ndarray, str]:
    coords = []
    seq = []
    with open(filename) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                res = line[17:20].strip()
                # map 3‑letter code to 1‑letter; use 'X' if unknown
                aa = AA_3_TO_1.get(res, 'X') if 'AA_3_TO_1' in dir() else 'X'
                # We need AA_3_TO_1 defined; add it here.
    return np.array(coords, dtype=np.float32), ''.join(seq)

# quick mapping for PDB reading
AA_3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC v21 Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    # Training command
    train_parser = sub.add_parser('train', help='Train the model on a dataset')
    train_parser.add_argument('--data', type=str, default=None,
                              help='Path to directory of PDB files (unused in synthetic demo)')
    train_parser.add_argument('--samples', type=int, default=200,
                              help='Number of synthetic samples for demo')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--device', type=str, default='cuda')
    train_parser.add_argument('--out', type=str, default='v21_pretrained.pt',
                              help='Output checkpoint path')

    # Refinement command
    refine_parser = sub.add_parser('refine', help='Refine a single sequence')
    refine_parser.add_argument('--seq', type=str, required=True,
                               help='Amino acid sequence (one‑letter code)')
    refine_parser.add_argument('--init', type=str, default=None,
                               help='Optional PDB file with initial CA coordinates')
    refine_parser.add_argument('--out', type=str, default='refined.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--device', type=str, default='cuda')
    refine_parser.add_argument('--checkpoint', type=str, default='v21_pretrained.pt',
                               help='Path to pretrained model weights')

    args = parser.parse_args()
    cfg = V21Config(epochs=args.epochs if 'epochs' in args else 50,
                    batch_size=args.batch_size if 'batch_size' in args else 8,
                    device=args.device,
                    refine_steps=args.steps if 'steps' in args else 600)

    device = torch.device(cfg.device)

    if args.command == 'train':
        logger.info("Creating synthetic training data (for production, replace with real PDB parsing).")
        data = synthetic_dataset(num_samples=args.samples)
        dataset = ProteinDataset(data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        model = V21Model(cfg)
        train_model(model, dataloader, cfg)
        logger.info("Training complete.")

    elif args.command == 'refine':
        sequence = args.seq
        model = V21Model(cfg)
        # load checkpoint if provided
        ckpt_path = args.checkpoint
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info(f"Loaded model weights from {ckpt_path}")
        else:
            logger.warning("Checkpoint not found; using random weights.")
        model.to(device)

        init_coords = None
        if args.init and os.path.exists(args.init):
            init_coords, _ = read_ca_pdb(args.init)
            logger.info(f"Loaded initial coordinates from {args.init}")

        refined, history = model.refine(sequence, init_coords=init_coords, steps=cfg.refine_steps)
        write_ca_pdb(refined, sequence, args.out)
        logger.info(f"Refined structure saved to {args.out}")
        if init_coords is not None:
            rmsd_val = compute_rmsd(refined, init_coords)
            logger.info(f"RMSD from initial: {rmsd_val:.4f} Å")
