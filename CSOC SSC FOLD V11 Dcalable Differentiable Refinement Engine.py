# =============================================================================
# CSOC-SSC v11.0
# =============================================================================
# MIT License
# Author: Yoon A Limsuwan
# =============================================================================

import os
import json
import time
import math
import gzip
import pickle
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

__version__ = "11.0.0"
__author__ = "Yoon A Limsuwan"
__license__ = "MIT"

EPS = 1e-8

# =============================================================================
# BIOCHEMICAL PRIORS
# =============================================================================

THREE2ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y','SEC':'U','MSE':'M','HSD':'H','HSE':'H',
}

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

RAMACHANDRAN_PRIORS = {
    'DEFAULT': {
        'alpha': {'phi': -60.0, 'psi': -45.0, 'std': 20.0},
        'beta': {'phi': -120.0, 'psi': 120.0, 'std': 25.0},
    },
    'GLY': {
        'alpha': {'phi': -75.0, 'psi': -10.0, 'std': 35.0},
        'beta': {'phi': -120.0, 'psi': 120.0, 'std': 40.0},
    },
    'PRO': {
        'pro': {'phi': -60.0, 'psi': -40.0, 'std': 15.0},
    }
}

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CSOCConfig:

    # optimization
    n_stages: int = 5
    n_iter_per_stage: int = 500
    learning_rate: float = 1e-3

    # sparse network
    sparse_cutoff: float = 20.0
    knn_k: int = 32

    # coarse graining
    coarse_grain_factor: int = 4

    # AMP
    use_amp: bool = True
    amp_dtype: str = "float16"

    # scheduling
    use_criticality_schedule: bool = True
    criticality_power: float = 1.5

    # weights
    weight_bond: float = 20.0
    weight_angle: float = 5.0
    weight_dihedral: float = 4.0
    weight_clash: float = 50.0
    weight_rama: float = 5.0
    weight_solvation: float = 2.0
    weight_hbond: float = 1.0

    # system
    checkpoint_dir: str = "./checkpoints"
    trajectory_dir: str = "./trajectory"

    # runtime
    verbose: int = 1
    gradient_clip_norm: float = 1.0

    # solvation
    solvation_cutoff: float = 8.0
    solvation_block_size: int = 4096

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

# =============================================================================
# BACKBONE STRUCTURE
# =============================================================================

@dataclass
class BackboneFrame:
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    o: np.ndarray
    residue_ids: List[int] = field(default_factory=list)
    seq: str = ""

    def to_torch(self, device="cuda"):
        return {
            'n': torch.tensor(self.n, dtype=torch.float32, device=device),
            'ca': torch.tensor(self.ca, dtype=torch.float32, device=device),
            'c': torch.tensor(self.c, dtype=torch.float32, device=device),
            'o': torch.tensor(self.o, dtype=torch.float32, device=device),
        }

# =============================================================================
# PDB LOADER
# =============================================================================

def load_pdb_backbone(path: str, chain='A') -> BackboneFrame:

    opener = gzip.open if path.endswith('.gz') else open

    n_atoms = []
    ca_atoms = []
    c_atoms = []
    o_atoms = []
    residue_ids = []
    seq = []

    with opener(path, 'rt', errors='ignore') as f:

        current_res = None
        current_res_name = None
        residue_buffer = {}

        for line in f:

            if not line.startswith("ATOM"):
                continue

            if line[21] != chain:
                continue

            atom = line[12:16].strip()
            res = line[17:20].strip()
            res_id = int(line[22:26])

            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ], dtype=np.float32)

            if current_res is None:
                current_res = res_id
                current_res_name = res

            if res_id != current_res:

                if all(k in residue_buffer for k in ['N','CA','C']):
                    n_atoms.append(residue_buffer['N'])
                    ca_atoms.append(residue_buffer['CA'])
                    c_atoms.append(residue_buffer['C'])
                    o_atoms.append(residue_buffer.get('O', residue_buffer['C']))
                    residue_ids.append(current_res)
                    seq.append(THREE2ONE.get(current_res_name, 'X'))

                residue_buffer = {}
                current_res = res_id
                current_res_name = res

            residue_buffer[atom] = xyz

    return BackboneFrame(
        n=np.array(n_atoms),
        ca=np.array(ca_atoms),
        c=np.array(c_atoms),
        o=np.array(o_atoms),
        residue_ids=residue_ids,
        seq=''.join(seq)
    )

# =============================================================================
# DIFFERENTIABLE GEOMETRY
# =============================================================================

def compute_dihedral_torch(p0, p1, p2, p3):

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / (torch.norm(b1) + EPS)

    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1

    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v, dim=0), w)

    return torch.atan2(y, x)

# =============================================================================
# TORSION EXTRACTION
# =============================================================================

def extract_phi_psi_torch(n, ca, c):

    phi = []
    psi = []

    L = ca.shape[0]

    for i in range(1, L):
        phi_i = compute_dihedral_torch(
            c[i-1], n[i], ca[i], c[i]
        )
        phi.append(phi_i)

    for i in range(L - 1):
        psi_i = compute_dihedral_torch(
            n[i], ca[i], c[i], n[i+1]
        )
        psi.append(psi_i)

    return torch.stack(phi), torch.stack(psi)

# =============================================================================
# SPARSE CONTACT GRAPH
# =============================================================================

class SparseContactNetwork:

    def __init__(self, coords, cutoff=20.0, k=32):

        self.coords = coords
        self.cutoff = cutoff
        self.k = k

        self.tree = cKDTree(coords)
        self.pairs = self._build_sparse_pairs()

    def _build_sparse_pairs(self):

        pairs = []

        for i in range(len(self.coords)):

            idx = self.tree.query_ball_point(self.coords[i], self.cutoff)

            for j in idx:
                if j <= i:
                    continue
                if abs(i - j) < 4:
                    continue
                pairs.append((i, j))

        return np.array(pairs, dtype=np.int32)

    def to_torch(self, device='cuda'):

        if len(self.pairs) == 0:
            return torch.empty((0,2), dtype=torch.long, device=device)

        return torch.tensor(self.pairs, dtype=torch.long, device=device)

# =============================================================================
# ENERGY REGISTRY
# =============================================================================

class EnergyRegistry:

    def __init__(self):
        self.terms = {}

    def register(self, name, fn):
        self.terms[name] = fn

    def compute(self, state, weights):

        total = torch.tensor(0.0, device=state['ca'].device)
        components = {}

        for name, fn in self.terms.items():
            e = fn(state) * weights.get(name, 1.0)
            total = total + e
            components[name] = e.detach().item()

        return total, components

# =============================================================================
# ENERGY TERMS
# =============================================================================

def bond_energy(state):

    ca = state['ca']
    dv = ca[1:] - ca[:-1]
    d = torch.norm(dv, dim=1)

    return torch.mean((d - 3.8) ** 2)


def clash_energy(state):

    ca = state['ca']
    pairs = state['pairs']

    if len(pairs) == 0:
        return torch.tensor(0.0, device=ca.device)

    dv = ca[pairs[:,0]] - ca[pairs[:,1]]
    d = torch.norm(dv, dim=1)

    mask = d < 3.2

    if not mask.any():
        return torch.tensor(0.0, device=ca.device)

    return torch.mean((3.2 - d[mask]) ** 2)


def ramachandran_energy(state):

    n = state['n']
    ca = state['ca']
    c = state['c']
    seq = state['seq']

    phi, psi = extract_phi_psi_torch(n, ca, c)

    E = torch.tensor(0.0, device=ca.device)

    for i in range(min(len(phi), len(seq)-1)):

        aa = seq[i]
        priors = RAMACHANDRAN_PRIORS.get(
            aa,
            RAMACHANDRAN_PRIORS['DEFAULT']
        )

        best = None

        for mode in priors.values():

            phi_ref = torch.tensor(
                math.radians(mode['phi']),
                device=ca.device
            )

            psi_ref = torch.tensor(
                math.radians(mode['psi']),
                device=ca.device
            )

            std = math.radians(mode['std'])
            kappa = 1.0 / (std**2 + EPS)

            val = -kappa * torch.cos(phi[i] - phi_ref)
            val += -kappa * torch.cos(psi[i] - psi_ref)

            if best is None:
                best = val
            else:
                best = torch.minimum(best, val)

        E = E + best

    return E / len(phi)


def solvation_energy(state):

    ca = state['ca']
    seq = state['seq']

    D = torch.cdist(ca, ca)
    density = (D < 8.0).float().sum(dim=1)

    burial = 1.0 - torch.exp(-density / 15.0)

    E = torch.tensor(0.0, device=ca.device)

    for i, aa in enumerate(seq):

        hydro = HYDROPHOBICITY.get(aa, 0.0)

        if hydro > 0:
            E = E + hydro * burial[i]
        else:
            E = E + hydro * (1.0 - burial[i])

    return E / len(seq)

# =============================================================================
# HYDROGEN BOND MODULE
# =============================================================================

def hydrogen_bond_energy(state):

    n = state['n']
    o = state['o']

    D = torch.cdist(n, o)

    mask = (D > 2.5) & (D < 3.5)

    if not mask.any():
        return torch.tensor(0.0, device=n.device)

    return -torch.mean(torch.exp(-(D[mask] - 2.9)**2))

# =============================================================================
# HYBRID OPTIMIZER
# =============================================================================

class HybridOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, T=300.0):

        defaults = dict(lr=lr, T=T)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            T = group['T']

            for p in group['params']:

                if p.grad is None:
                    continue

                grad = p.grad
                noise = torch.randn_like(p) * math.sqrt(lr * T / 300.0)

                p.add_(-lr * grad + noise)

# =============================================================================
# SCHEDULER
# =============================================================================

class CriticalityScheduler:

    def __init__(self, n_stages, power=1.5):

        self.n_stages = n_stages
        self.power = power

    def temperature(self, stage):

        t = stage / max(1, self.n_stages - 1)
        return 300.0 * (1.0 - 0.9 * t**self.power)

# =============================================================================
# TRAJECTORY WRITER
# =============================================================================

class TrajectoryWriter:

    def __init__(self, outdir):

        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)

    def save_xyz(self, coords, step):

        path = self.outdir / f"frame_{step:05d}.xyz"

        with open(path, 'w') as f:

            f.write(f"{len(coords)}\n")
            f.write("CSOC-SSC trajectory\n")

            for xyz in coords:
                f.write(f"C {xyz[0]} {xyz[1]} {xyz[2]}\n")

# =============================================================================
# MAIN ENGINE
# =============================================================================

class StructuralOptimizationEngine:

    def __init__(self, config: CSOCConfig):

        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        self.scheduler = CriticalityScheduler(
            config.n_stages,
            config.criticality_power
        )

        self.registry = EnergyRegistry()

        self.registry.register('bond', bond_energy)
        self.registry.register('clash', clash_energy)
        self.registry.register('rama', ramachandran_energy)
        self.registry.register('solvation', solvation_energy)
        self.registry.register('hbond', hydrogen_bond_energy)

        self.writer = TrajectoryWriter(config.trajectory_dir)

    def optimize(self, backbone: BackboneFrame):

        state = backbone.to_torch(self.device)

        state['ca'] = nn.Parameter(state['ca'])

        sparse = SparseContactNetwork(
            backbone.ca,
            cutoff=self.config.sparse_cutoff,
            k=self.config.knn_k
        )

        state['pairs'] = sparse.to_torch(self.device)
        state['seq'] = backbone.seq

        optimizer = HybridOptimizer(
            [state['ca']],
            lr=self.config.learning_rate,
            T=300.0
        )

        scaler = GradScaler(enabled=self.config.use_amp)

        weights = {
            'bond': self.config.weight_bond,
            'clash': self.config.weight_clash,
            'rama': self.config.weight_rama,
            'solvation': self.config.weight_solvation,
            'hbond': self.config.weight_hbond,
        }

        history = []

        for stage in range(self.config.n_stages):

            T = self.scheduler.temperature(stage)
            optimizer.param_groups[0]['T'] = T

            for step in range(self.config.n_iter_per_stage):

                optimizer.zero_grad()

                with autocast(
                    enabled=self.config.use_amp,
                    dtype=torch.float16
                ):

                    E, terms = self.registry.compute(state, weights)

                scaler.scale(E).backward()

                torch.nn.utils.clip_grad_norm_(
                    [state['ca']],
                    self.config.gradient_clip_norm
                )

                scaler.step(optimizer)
                scaler.update()

                history.append(E.item())

                if step % 50 == 0:
                    print(
                        f"[Stage {stage}] "
                        f"Step {step} "
                        f"Energy={E.item():.6f} "
                        f"T={T:.2f}"
                    )

                if step % 100 == 0:
                    self.writer.save_xyz(
                        state['ca'].detach().cpu().numpy(),
                        stage * self.config.n_iter_per_stage + step
                    )

            self.save_checkpoint(stage, state)

        final_ca = state['ca'].detach().cpu().numpy()

        return {
            'coords': final_ca,
            'energy_history': history,
        }

    def save_checkpoint(self, stage, state):

        ckpt = {
            'stage': stage,
            'coords': state['ca'].detach().cpu().numpy(),
        }

        path = os.path.join(
            self.config.checkpoint_dir,
            f'stage_{stage}.pkl'
        )

        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":

    config = CSOCConfig(
        n_stages=3,
        n_iter_per_stage=200,
        verbose=1,
    )

    config.save("config_v11.json")

    print("CSOC-SSC v11.0")
    print("Initializing synthetic backbone...")

    np.random.seed(42)

    n_res = 500

    ca = np.random.randn(n_res, 3).astype(np.float32) * 20

    backbone = BackboneFrame(
        n=ca - np.array([0.5,0,0], dtype=np.float32),
        ca=ca,
        c=ca + np.array([0.5,0,0], dtype=np.float32),
        o=ca + np.array([1.0,1.0,0], dtype=np.float32),
        seq='A' * n_res,
    )

    engine = StructuralOptimizationEngine(config)

    result = engine.optimize(backbone)

    print("Optimization complete")
    print(f"Frames optimized: {len(result['coords'])}")
    print(f"Energy samples: {len(result['energy_history'])}")

# =============================================================================
# END OF FILE
# =============================================================================
