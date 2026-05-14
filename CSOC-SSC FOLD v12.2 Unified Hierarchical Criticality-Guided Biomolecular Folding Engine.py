# =============================================================================
# CSOC-SSC v12.2
# Unified Hierarchical Criticality-Guided Biomolecular Folding Engine
# -----------------------------------------------------------------------------
# MIT License — Yoon A Limsuwan 2026
# =============================================================================

import os
import math
import time
import random

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# GLOBALS
# =============================================================================

__version__ = "12.2.0"
__author__ = "Yoon A Limsuwan"
__license__ = "MIT"

# =============================================================================
# BIOCHEMISTRY
# =============================================================================

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"

AA_TO_ID = {
    aa: i for i, aa in enumerate(AA_VOCAB)
}

HYDROPHOBICITY = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3
}

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class V122Config:

    device: str = "cuda"

    seed: int = 42

    embedding_dim: int = 192

    hidden_dim: int = 256

    use_soc: bool = True

    sigma_target: float = 1.0

    temperature_base: float = 300.0

    avalanche_power: float = 1.5

    criticality_strength: float = 5.0

    coarse_factor: int = 4

    n_stages: int = 5

    refinement_steps: int = 300

    universality_strength: float = 0.25

    contact_cutoff: float = 18.0

    max_neighbors: int = 32

    weight_bond: float = 30.0

    weight_angle: float = 8.0

    weight_clash: float = 50.0

    weight_contact: float = 5.0

    weight_hydrophobic: float = 8.0

    weight_solvation: float = 5.0

    weight_criticality: float = 5.0

    learning_rate: float = 1e-3

    gradient_clip: float = 1.0

    use_amp: bool = True

    checkpoint_dir: str = "./v122_checkpoints"

    verbose: int = 1

# =============================================================================
# BACKBONE
# =============================================================================

@dataclass
class BackboneFrame:

    ca: np.ndarray

    seq: str

    n: Optional[np.ndarray] = None

    c: Optional[np.ndarray] = None

    o: Optional[np.ndarray] = None

# =============================================================================
# LOGGING
# =============================================================================

def log(msg, verbose=1):

    if verbose > 0:

        t = time.strftime("%H:%M:%S")

        print(f"[V12.2 {t}] {msg}")

# =============================================================================
# LAYER 1 — BIOLOGICAL PRIORS
# =============================================================================

class SequenceEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.embedding = nn.Embedding(
            len(AA_VOCAB),
            config.embedding_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim
            )
        )

    def forward(self, sequence):

        ids = torch.tensor(
            [
                AA_TO_ID.get(aa, 20)
                for aa in sequence
            ],
            dtype=torch.long,
            device=self.embedding.weight.device
        )

        x = self.embedding(ids)

        x = self.encoder(x)

        return x

# =============================================================================
# LAYER 2 — ADAPTIVE UNIVERSALITY
# =============================================================================

class AdaptiveAlphaPredictor(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent):

        alpha = self.net(latent)

        alpha = 0.5 + 2.5 * torch.sigmoid(alpha)

        return alpha.squeeze(-1)

# =============================================================================
# CSOC KERNEL
# =============================================================================

class CSOCKernel(nn.Module):

    def __init__(self, lam=12.0):

        super().__init__()

        self.lam = lam

    def forward(self,
                coords,
                alpha):

        D = torch.cdist(coords, coords)

        D = D + 1e-4

        alpha_i = alpha.unsqueeze(0)

        alpha_j = alpha.unsqueeze(1)

        alpha_pair = 0.5 * (
            alpha_i + alpha_j
        )

        kernel = (
            (D ** (-alpha_pair))
            *
            torch.exp(-D / self.lam)
        )

        kernel.fill_diagonal_(0.0)

        return kernel

# =============================================================================
# SSC CRITICALITY
# =============================================================================

class SSCCriticalityEngine:

    def __init__(self,
                 sigma_target=1.0):

        self.sigma_target = sigma_target

    def sigma(self,
              displacement):

        eps = 1e-8

        local_sigma = (
            displacement /
            (
                displacement.mean()
                + eps
            )
        )

        global_sigma = local_sigma.mean()

        return local_sigma, global_sigma

    def criticality_loss(self,
                         sigma):

        return (sigma - self.sigma_target) ** 2

    def temperature(self,
                    sigma,
                    base_T=300.0):

        delta = abs(sigma - 1.0)

        return base_T * (
            1.0 + delta
        )

# =============================================================================
# SPARSE GRAPH
# =============================================================================

class SparseContactGraph:

    def __init__(self,
                 coords,
                 cutoff=18.0,
                 k=32):

        self.coords = coords

        self.cutoff = cutoff

        self.k = k

        self.tree = cKDTree(coords)

    def build(self):

        pairs = []

        for i in range(len(self.coords)):

            neigh = self.tree.query_ball_point(
                self.coords[i],
                self.cutoff
            )

            neigh = [
                j for j in neigh
                if j > i and abs(i - j) > 3
            ]

            neigh = neigh[:self.k]

            for j in neigh:

                pairs.append([i, j])

        if len(pairs) == 0:

            return np.zeros((0, 2), dtype=np.int64)

        return np.array(pairs)

# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class PhysicsEngine:

    def __init__(self,
                 config):

        self.cfg = config

    def bond_energy(self,
                    coords):

        dv = coords[1:] - coords[:-1]

        d = torch.norm(dv, dim=1)

        return (
            self.cfg.weight_bond
            *
            torch.mean((d - 3.8) ** 2)
        )

    def clash_energy(self,
                     coords):

        D = torch.cdist(coords, coords)

        mask = (
            (D < 3.0)
            &
            (D > 0)
        )

        clash = (3.0 - D[mask]) ** 2

        if len(clash) == 0:

            return torch.tensor(
                0.0,
                device=coords.device
            )

        return (
            self.cfg.weight_clash
            *
            clash.mean()
        )

    def hydrophobic_energy(self,
                           coords,
                           seq):

        D = torch.cdist(coords, coords)

        density = (
            D < 8.0
        ).float().sum(dim=1)

        burial = 1.0 - torch.exp(
            -density / 15.0
        )

        E = 0.0

        for i, aa in enumerate(seq):

            hydro = HYDROPHOBICITY.get(
                aa,
                0.0
            )

            if hydro > 0:

                E += hydro * burial[i]

            else:

                E += hydro * (
                    1.0 - burial[i]
                )

        return (
            self.cfg.weight_hydrophobic
            *
            E
        )

    def sasa_approximation(self,
                           coords):

        D = torch.cdist(coords, coords)

        neighbors = (
            D < 10.0
        ).float().sum(dim=1)

        sasa = 1.0 / (
            neighbors + 1.0
        )

        return sasa.mean()

# =============================================================================
# RG MULTISCALE
# =============================================================================

class RGRefinement:

    def __init__(self,
                 factor=4):

        self.factor = factor

    def coarse_grain(self,
                     coords):

        n = len(coords)

        nc = (
            n + self.factor - 1
        ) // self.factor

        out = np.zeros(
            (nc, 3),
            dtype=np.float32
        )

        for i in range(nc):

            s = i * self.factor

            e = min(
                (i + 1) * self.factor,
                n
            )

            out[i] = coords[s:e].mean(axis=0)

        return out

    def upsample(self,
                 coarse,
                 n_target):

        x_coarse = np.linspace(
            0,
            n_target - 1,
            len(coarse)
        )

        x_fine = np.arange(n_target)

        out = np.zeros(
            (n_target, 3)
        )

        for d in range(3):

            cs = CubicSpline(
                x_coarse,
                coarse[:, d]
            )

            out[:, d] = cs(x_fine)

        return out.astype(np.float32)

# =============================================================================
# SOC LANGEVIN OPTIMIZER
# =============================================================================

class SOCLangevinOptimizer(torch.optim.AdamW):

    def __init__(self,
                 params,
                 lr=1e-3,
                 temperature=300.0):

        super().__init__(
            params,
            lr=lr
        )

        self.temperature = temperature

    @torch.no_grad()
    def step(self,
             closure=None):

        loss = super().step(closure)

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue

                noise_scale = (
                    math.sqrt(
                        self.temperature / 300.0
                    )
                    *
                    group["lr"]
                )

                p.add_(
                    torch.randn_like(p)
                    * noise_scale
                )

        return loss

# =============================================================================
# CONTACT DIFFUSION
# =============================================================================

class ContactDiffusion(nn.Module):

    def __init__(self,
                 hidden_dim):

        super().__init__()

        self.proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(self,
                latent,
                kernel):

        propagated = torch.matmul(
            kernel,
            latent
        )

        propagated = self.proj(
            propagated
        )

        return propagated

# =============================================================================
# MAIN ENGINE
# =============================================================================

class CSOCSSC_V122:

    def __init__(self,
                 config):

        self.cfg = config

        torch.manual_seed(config.seed)

        np.random.seed(config.seed)

        self.device = torch.device(
            config.device
            if torch.cuda.is_available()
            else "cpu"
        )

        Path(
            config.checkpoint_dir
        ).mkdir(exist_ok=True)

        self.encoder = SequenceEncoder(
            config
        ).to(self.device)

        self.alpha_predictor = (
            AdaptiveAlphaPredictor(
                config.hidden_dim
            ).to(self.device)
        )

        self.kernel = CSOCKernel()

        self.contact_diffusion = (
            ContactDiffusion(
                config.hidden_dim
            ).to(self.device)
        )

        self.criticality = (
            SSCCriticalityEngine(
                config.sigma_target
            )
        )

        self.physics = PhysicsEngine(
            config
        )

        self.rg = RGRefinement(
            config.coarse_factor
        )

    def optimize(self,
                 backbone):

        seq = backbone.seq

        coords = torch.tensor(
            backbone.ca,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        latent = self.encoder(seq)

        alpha = self.alpha_predictor(
            latent
        )

        latent = latent.unsqueeze(0)

        scaler = GradScaler(
            enabled=self.cfg.use_amp
        )

        previous_coords = coords.detach().clone()

        for stage in range(
            self.cfg.n_stages
        ):

            log(
                f"Stage {stage}",
                self.cfg.verbose
            )

            optimizer = SOCLangevinOptimizer(
                [coords],
                lr=self.cfg.learning_rate,
                temperature=self.cfg.temperature_base
            )

            for step in range(
                self.cfg.refinement_steps
            ):

                optimizer.zero_grad()

                with autocast(
                    enabled=self.cfg.use_amp
                ):

                    K = self.kernel(
                        coords,
                        alpha
                    )

                    latent_diffused = (
                        self.contact_diffusion(
                            latent,
                            K.unsqueeze(0)
                        )
                    )

                    E_bond = (
                        self.physics
                        .bond_energy(coords)
                    )

                    E_clash = (
                        self.physics
                        .clash_energy(coords)
                    )

                    E_hydro = (
                        self.physics
                        .hydrophobic_energy(
                            coords,
                            seq
                        )
                    )

                    E_sasa = (
                        self.physics
                        .sasa_approximation(
                            coords
                        )
                    )

                    E_latent = (
                        latent_diffused.norm()
                        *
                        self.cfg.universality_strength
                    )

                    displacement = torch.norm(
                        coords - previous_coords,
                        dim=1
                    )

                    local_sigma, global_sigma = (
                        self.criticality.sigma(
                            displacement
                        )
                    )

                    E_criticality = (
                        self.cfg.weight_criticality
                        *
                        self.criticality
                        .criticality_loss(
                            global_sigma
                        )
                    )

                    E_total = (
                        E_bond
                        + E_clash
                        + E_hydro
                        + E_sasa
                        + E_latent
                        + E_criticality
                    )

                scaler.scale(
                    E_total
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    [coords],
                    self.cfg.gradient_clip
                )

                scaler.step(optimizer)

                scaler.update()

                T_dynamic = (
                    self.criticality
                    .temperature(
                        global_sigma.item(),
                        self.cfg.temperature_base
                    )
                )

                optimizer.temperature = T_dynamic

                if (
                    self.cfg.use_soc
                    and
                    step % 20 == 0
                ):

                    avalanche_noise = (
                        torch.randn_like(coords)
                        *
                        (
                            0.01
                            *
                            math.sqrt(
                                T_dynamic / 300.0
                            )
                        )
                    )

                    coords.data += avalanche_noise

                previous_coords = (
                    coords.detach().clone()
                )

                if step % 50 == 0:

                    log(
                        f"Stage={stage} "
                        f"Step={step} "
                        f"E={E_total.item():.4f} "
                        f"Sigma={global_sigma.item():.4f} "
                        f"T={T_dynamic:.2f}",
                        self.cfg.verbose
                    )

            if stage < self.cfg.n_stages - 1:

                coarse = self.rg.coarse_grain(
                    coords.detach()
                    .cpu()
                    .numpy()
                )

                refined = self.rg.upsample(
                    coarse,
                    len(coords)
                )

                coords.data = torch.tensor(
                    refined,
                    dtype=torch.float32,
                    device=self.device
                )

        final_coords = (
            coords.detach()
            .cpu()
            .numpy()
        )

        return BackboneFrame(
            ca=final_coords,
            seq=seq
        )

# =============================================================================
# RMSD
# =============================================================================

def rmsd(a, b):

    a = a - a.mean(axis=0)

    b = b - b.mean(axis=0)

    H = a.T @ b

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    a_rot = a @ R

    return np.sqrt(
        np.mean(
            np.sum(
                (a_rot - b) ** 2,
                axis=1
            )
        )
    )

# =============================================================================
# PDB EXPORT
# =============================================================================

def save_pdb(coords,
             sequence,
             path="v122_output.pdb"):

    with open(path, "w") as f:

        for i, xyz in enumerate(coords):

            x, y, z = xyz

            aa = (
                sequence[i]
                if i < len(sequence)
                else "A"
            )

            line = (
                f"ATOM  {i+1:5d}  CA  {aa:>3s} A{i+1:4d}    "
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"  1.00  0.00           C\n"
            )

            f.write(line)

        f.write("END\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("CSOC-SSC v12.2")
    print("Unified Criticality-Guided Folding Engine")
    print("=" * 80)

    config = V122Config(
        n_stages=4,
        refinement_steps=250,
        verbose=1
    )

    engine = CSOCSSC_V122(config)

    n_res = 600

    coords = (
        np.random.randn(n_res, 3)
        .astype(np.float32)
        * 25.0
    )

    sequence = "".join(
        random.choice(AA_VOCAB[:-1])
        for _ in range(n_res)
    )

    backbone = BackboneFrame(
        ca=coords,
        seq=sequence
    )

    start = time.time()

    result = engine.optimize(
        backbone
    )

    elapsed = time.time() - start

    final_rmsd = rmsd(
        backbone.ca,
        result.ca
    )

    save_pdb(
        result.ca,
        result.seq,
        "csoc_ssc_v122_output.pdb"
    )

    print("\nOptimization Complete")
    print(f"Residues : {n_res}")
    print(f"RMSD     : {final_rmsd:.4f} Å")
    print(f"Elapsed  : {elapsed:.2f} sec")
    print("Saved    : csoc_ssc_v122_output.pdb")

    print("=" * 80)
