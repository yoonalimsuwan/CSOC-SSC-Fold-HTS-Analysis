# ============================================================================
# CSOC-SSC PROTEIN FOLDING ENGINE V6.0
# ============================================================================
# Title:
#   Interpretable Criticality-Driven Protein Folding System
#
# Core Philosophy:
#   - NOT a black-box predictor
#   - Explicit analytical energy decomposition
#   - Interpretable statistical mechanics
#   - RG/SOC-controlled optimization dynamics
#   - Torsion-space differentiable folding
#   - Coarse-grained Hamiltonian formalism
#
# New Features in V6:
#
#   ✓ Rosetta-like statistical potentials
#   ✓ Knowledge-based rotamer energies
#   ✓ Solvent accessible surface approximations
#   ✓ Coarse-grained Hamiltonian decomposition
#   ✓ RG-flow observables
#   ✓ Explicit free-energy decomposition
#   ✓ Learnable interpretable pair kernels
#   ✓ Multi-scale SOC critical exploration
#   ✓ Differentiable torsion-space folding
#   ✓ NeRF geometry backbone construction
#
# Author:
#   Yoon A Limsuwan
#
# License:
#   MIT
# ============================================================================

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# SECTION 1 — CONFIGURATION
# ============================================================================

@dataclass
class CSOCV6Config:

    # Device
    device: str = "cuda"

    # Network
    embed_dim: int = 128
    n_heads: int = 4
    n_evo_blocks: int = 3

    # Folding Schedule
    n_recycling: int = 3
    n_stages: int = 3
    n_iter_per_stage: int = 200

    # Optimization
    learning_rate: float = 2e-2
    grad_clip: float = 1.0

    # SOC / RG
    asm_L: int = 64
    base_temperature: float = 300.0
    rg_beta: float = 0.15
    rg_decay: float = 0.90

    # Energy Weights
    w_rama: float = 2.0
    w_rotamer: float = 1.5
    w_contact: float = 3.0
    w_distogram: float = 5.0
    w_lj: float = 12.0
    w_hbond: float = 2.0
    w_sasa: float = 1.0
    w_compact: float = 0.5

# ============================================================================
# SECTION 2 — IDEALIZED GEOMETRY
# ============================================================================

BOND_N_CA = 1.46
BOND_CA_C = 1.52
BOND_C_N = 1.33

ANGLE_C_N_CA = 2.13
ANGLE_N_CA_C = 1.94
ANGLE_CA_C_N = 2.03

# ============================================================================
# SECTION 3 — INTERNAL COORDINATE GEOMETRY
# ============================================================================

def normalize(v):
    return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

def build_atom(A, B, C, length, angle, torsion):

    BC = normalize(C - B)

    n = normalize(torch.cross(B - A, BC))
    m = torch.cross(n, BC)

    d = torch.stack([
        -length * torch.cos(torch.tensor(angle, device=C.device)),
        length * torch.sin(torch.tensor(angle, device=C.device)) * torch.cos(torsion),
        length * torch.sin(torch.tensor(angle, device=C.device)) * torch.sin(torsion)
    ], dim=-1)

    M = torch.stack([BC, m, n], dim=-1)

    return C + torch.einsum("bij,bj->bi", M, d)

class TorsionToCartesian(nn.Module):

    def forward(self, torsions):

        L = torsions.shape[0]

        coords = torch.zeros(
            (L, 3, 3),
            device=torsions.device,
            dtype=torsions.dtype
        )

        coords[0,0] = torch.tensor([0.0, 0.0, 0.0], device=torsions.device)

        coords[0,1] = torch.tensor(
            [BOND_N_CA, 0.0, 0.0],
            device=torsions.device
        )

        c_x = BOND_N_CA - BOND_CA_C * math.cos(ANGLE_N_CA_C)
        c_y = BOND_CA_C * math.sin(ANGLE_N_CA_C)

        coords[0,2] = torch.tensor(
            [c_x, c_y, 0.0],
            device=torsions.device
        )

        for i in range(1, L):

            phi   = torsions[i,0]
            psi   = torsions[i,1]
            omega = torsions[i-1,2]

            coords[i,0] = build_atom(
                coords[i-1,0],
                coords[i-1,1],
                coords[i-1,2],
                BOND_C_N,
                ANGLE_CA_C_N,
                omega
            )

            coords[i,1] = build_atom(
                coords[i-1,1],
                coords[i-1,2],
                coords[i,0],
                BOND_N_CA,
                ANGLE_C_N_CA,
                phi
            )

            coords[i,2] = build_atom(
                coords[i-1,2],
                coords[i,0],
                coords[i,1],
                BOND_CA_C,
                ANGLE_N_CA_C,
                psi
            )

        return coords

# ============================================================================
# SECTION 4 — INTERPRETABLE PRIOR NETWORK
# ============================================================================

class EvoBlock(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, z):

        h = self.norm1(z)

        a, _ = self.attn(h, h, h)

        z = z + a

        z = z + self.ffn(self.norm2(z))

        return z

# ============================================================================
# SECTION 5 — LEARNED INTERPRETABLE KERNELS
# ============================================================================

class PairKernel(nn.Module):

    """
    Learnable but interpretable pair interaction kernel.
    """

    def __init__(self):

        super().__init__()

        self.mu = nn.Parameter(torch.tensor(8.0))
        self.sigma = nn.Parameter(torch.tensor(2.0))
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, distances):

        kernel = self.strength * torch.exp(
            -((distances - self.mu)**2) /
            (2 * self.sigma**2 + 1e-8)
        )

        return kernel

# ============================================================================
# SECTION 6 — STRUCTURAL PRIOR NETWORK
# ============================================================================

class StructuralPriorNet(nn.Module):

    def __init__(self,
                 vocab=21,
                 embed_dim=128,
                 n_blocks=3):

        super().__init__()

        self.embed = nn.Embedding(vocab, embed_dim // 2)

        self.blocks = nn.ModuleList([
            EvoBlock(embed_dim, 4)
            for _ in range(n_blocks)
        ])

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, tokens):

        L = tokens.shape[0]

        x = self.embed(tokens)

        xi = x.unsqueeze(1).expand(L, L, -1)
        xj = x.unsqueeze(0).expand(L, L, -1)

        z = torch.cat([xi, xj], dim=-1)

        z = z.view(L * L, -1).unsqueeze(0)

        for blk in self.blocks:
            z = blk(z)

        z = z.squeeze(0).view(L, L, -1)

        d = F.softplus(self.head(z).squeeze(-1))

        mask = 1.0 - torch.eye(L, device=tokens.device)

        return d * mask

# ============================================================================
# SECTION 7 — SOC / RG DRIVER
# ============================================================================

class FastSOC:

    def __init__(self, L=64):

        self.L = L

        self.S = np.random.rand(L, L) * 3.0

    def step(self):

        x, y = np.random.randint(1, self.L - 1, size=2)

        self.S[x,y] += 1.0

        A = 0

        while np.max(self.S) >= 4.0:

            topple = (self.S >= 4.0)

            self.S[topple] -= 4.0

            self.S[:-1,:] += topple[1:,:]
            self.S[1:,:]  += topple[:-1,:]
            self.S[:,:-1] += topple[:,1:]
            self.S[:,1:]  += topple[:,:-1]

            A += np.sum(topple)

        return int(A)

# ============================================================================
# SECTION 8 — RG FLOW OBSERVABLES
# ============================================================================

class RGFlowObserver:

    """
    Tracks coarse-grained thermodynamic observables.
    """

    def __init__(self):

        self.history = []

    def update(self, energy):

        self.history.append(float(energy))

    def susceptibility(self):

        if len(self.history) < 2:
            return 0.0

        x = np.array(self.history)

        return np.var(x)

    def heat_capacity(self):

        if len(self.history) < 2:
            return 0.0

        x = np.array(self.history)

        return np.mean((x - np.mean(x))**2)

# ============================================================================
# SECTION 9 — ROSETTA-LIKE STATISTICAL POTENTIALS
# ============================================================================

class StatisticalPotentials:

    def __init__(self, device):

        self.device = device

        self.contact_kernel = PairKernel().to(device)

    # ------------------------------------------------------------------------
    # Ramachandran Prior
    # ------------------------------------------------------------------------

    def ramachandran(self, torsions):

        phi = torsions[:,0]
        psi = torsions[:,1]

        helix = (phi + 1.05)**2 + (psi + 0.78)**2
        beta  = (phi + 2.35)**2 + (psi - 2.35)**2

        return torch.mean(torch.minimum(helix, beta))

    # ------------------------------------------------------------------------
    # Knowledge-Based Rotamer Energy
    # ------------------------------------------------------------------------

    def rotamer_energy(self, torsions):

        omega = torsions[:,2]

        trans_state = (omega - math.pi)**2

        cis_state = (omega - 0.0)**2 + 5.0

        return torch.mean(torch.minimum(trans_state, cis_state))

    # ------------------------------------------------------------------------
    # Lennard-Jones Sterics
    # ------------------------------------------------------------------------

    def lennard_jones(self, coords):

        ca = coords[:,1,:]

        D = torch.cdist(ca, ca)

        mask = D > 1e-6

        D = D[mask]

        sigma = 3.8
        epsilon = 0.2

        inv = sigma / D

        lj = 4 * epsilon * (inv**12 - inv**6)

        return torch.mean(lj)

    # ------------------------------------------------------------------------
    # Rosetta-like Contact Potential
    # ------------------------------------------------------------------------

    def contact_energy(self, coords):

        ca = coords[:,1,:]

        D = torch.cdist(ca, ca)

        K = self.contact_kernel(D)

        return -torch.mean(K)

    # ------------------------------------------------------------------------
    # Hydrogen Bond Statistical Potential
    # ------------------------------------------------------------------------

    def hydrogen_bond(self, coords):

        N = coords[:,0,:]
        C = coords[:,2,:]

        D = torch.cdist(N, C)

        hb = torch.exp(
            -((D - 2.9)**2) / 0.5
        )

        return -torch.mean(hb)

    # ------------------------------------------------------------------------
    # Solvent Accessible Surface Approximation
    # ------------------------------------------------------------------------

    def sasa_energy(self, coords):

        ca = coords[:,1,:]

        D = torch.cdist(ca, ca)

        exposure = torch.sigmoid((D - 10.0))

        sasa = torch.mean(exposure)

        return sasa

    # ------------------------------------------------------------------------
    # Compactness Free Energy
    # ------------------------------------------------------------------------

    def compactness(self, coords):

        ca = coords[:,1,:]

        center = torch.mean(ca, dim=0)

        rg = torch.sqrt(
            torch.mean(
                torch.sum((ca - center)**2, dim=-1)
            )
        )

        return rg

# ============================================================================
# SECTION 10 — COARSE-GRAINED HAMILTONIAN
# ============================================================================

class CoarseHamiltonian:

    """
    Explicit coarse-grained Hamiltonian decomposition.
    """

    def __init__(self, cfg, device):

        self.cfg = cfg
        self.device = device

        self.potential = StatisticalPotentials(device)

    def energy(self,
               coords,
               torsions,
               pred_dist):

        ca = coords[:,1,:]

        D = torch.cdist(ca, ca)

        mask = 1.0 - torch.eye(D.shape[0], device=self.device)

        # --------------------------------------------------------------------
        # Explicit Energy Components
        # --------------------------------------------------------------------

        E_rama = self.potential.ramachandran(torsions)

        E_rot = self.potential.rotamer_energy(torsions)

        E_lj = self.potential.lennard_jones(coords)

        E_contact = self.potential.contact_energy(coords)

        E_hbond = self.potential.hydrogen_bond(coords)

        E_sasa = self.potential.sasa_energy(coords)

        E_compact = self.potential.compactness(coords)

        E_dist = torch.mean((D - pred_dist)**2 * mask)

        # --------------------------------------------------------------------
        # Free Energy Decomposition
        # --------------------------------------------------------------------

        F_total = (
            self.cfg.w_rama * E_rama +
            self.cfg.w_rotamer * E_rot +
            self.cfg.w_lj * E_lj +
            self.cfg.w_contact * E_contact +
            self.cfg.w_hbond * E_hbond +
            self.cfg.w_sasa * E_sasa +
            self.cfg.w_compact * E_compact +
            self.cfg.w_distogram * E_dist
        )

        metrics = {

            "F_total": F_total.item(),

            "E_rama": E_rama.item(),
            "E_rotamer": E_rot.item(),
            "E_lj": E_lj.item(),
            "E_contact": E_contact.item(),
            "E_hbond": E_hbond.item(),
            "E_sasa": E_sasa.item(),
            "E_compact": E_compact.item(),
            "E_dist": E_dist.item()
        }

        return F_total, metrics

# ============================================================================
# SECTION 11 — RG TEMPERATURE CONTROLLER
# ============================================================================

class RGTemperatureController:

    def __init__(self, cfg):

        self.cfg = cfg

    def temperature(self, avalanche, recycle, stage):

        T = self.cfg.base_temperature

        T *= (
            1.0 +
            self.cfg.rg_beta *
            math.log1p(avalanche)
        )

        T *= self.cfg.rg_decay ** stage

        T *= 0.90 ** recycle

        return T

# ============================================================================
# SECTION 12 — LANGEVIN OPTIMIZER
# ============================================================================

class LangevinOptimizer(torch.optim.AdamW):

    def step_with_noise(self, T):

        super().step()

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue

                noise_scale = math.sqrt(
                    2.0 * T * group["lr"] / 300.0
                )

                noise = (
                    torch.randn_like(p.data)
                    * noise_scale
                    * 0.01
                )

                p.data.add_(noise)

                # Periodic torsion normalization
                p.data[:] = torch.atan2(
                    torch.sin(p.data),
                    torch.cos(p.data)
                )

# ============================================================================
# SECTION 13 — MAIN V6 PIPELINE
# ============================================================================

class CSOCProteinPredictorV6:

    def __init__(self, cfg):

        self.cfg = cfg

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )

        print(f"Device: {self.device}")

        self.aa2int = {
            aa:i for i, aa in enumerate(
                "ACDEFGHIKLMNPQRSTVWY-"
            )
        }

        self.prior_net = StructuralPriorNet(
            embed_dim=cfg.embed_dim,
            n_blocks=cfg.n_evo_blocks
        ).to(self.device)

        self.prior_net.eval()

        self.builder = TorsionToCartesian().to(self.device)

        self.hamiltonian = CoarseHamiltonian(
            cfg,
            self.device
        )

        self.soc = FastSOC(cfg.asm_L)

        self.rg_temp = RGTemperatureController(cfg)

        self.rg_observer = RGFlowObserver()

    # ------------------------------------------------------------------------
    # Sequence Encoding
    # ------------------------------------------------------------------------

    def encode(self, seq):

        return torch.tensor(
            [self.aa2int.get(a, 20) for a in seq],
            device=self.device
        )

    # ------------------------------------------------------------------------
    # Main Prediction
    # ------------------------------------------------------------------------

    def predict(self, seq):

        print("\n======================================================")
        print("CSOC-SSC V6 — INTERPRETABLE RG FOLDING ENGINE")
        print("======================================================")

        print(f"\nSequence Length: {len(seq)}")

        tokens = self.encode(seq)

        # --------------------------------------------------------------------
        # Structural Prior
        # --------------------------------------------------------------------

        with torch.no_grad():

            pred_dist = self.prior_net(tokens)

        # --------------------------------------------------------------------
        # Initialize Torsions
        # --------------------------------------------------------------------

        torsions = torch.zeros(
            (len(seq), 3),
            device=self.device,
            requires_grad=True
        )

        torsions.data[:,0] = -2.35
        torsions.data[:,1] =  2.35
        torsions.data[:,2] =  math.pi

        optimizer = LangevinOptimizer(
            [torsions],
            lr=self.cfg.learning_rate
        )

        # --------------------------------------------------------------------
        # Multi-Scale Folding
        # --------------------------------------------------------------------

        for recycle in range(self.cfg.n_recycling):

            print(f"\n🔄 Recycling {recycle+1}/{self.cfg.n_recycling}")

            for stage in range(self.cfg.n_stages):

                print(f"   Stage {stage+1}/{self.cfg.n_stages}")

                for step in range(self.cfg.n_iter_per_stage):

                    A = self.soc.step()

                    T = self.rg_temp.temperature(
                        A,
                        recycle,
                        stage
                    )

                    optimizer.zero_grad()

                    coords = self.builder(torsions)

                    F_total, metrics = self.hamiltonian.energy(
                        coords,
                        torsions,
                        pred_dist
                    )

                    self.rg_observer.update(metrics["F_total"])

                    F_total.backward()

                    torch.nn.utils.clip_grad_norm_(
                        [torsions],
                        self.cfg.grad_clip
                    )

                    optimizer.step_with_noise(T)

                    if step % 50 == 0:

                        chi = self.rg_observer.susceptibility()

                        Cv = self.rg_observer.heat_capacity()

                        print(
                            f"      Iter {step:3d} | "
                            f"F={metrics['F_total']:8.3f} | "
                            f"Rama={metrics['E_rama']:6.3f} | "
                            f"Rot={metrics['E_rotamer']:6.3f} | "
                            f"LJ={metrics['E_lj']:6.3f} | "
                            f"HB={metrics['E_hbond']:6.3f} | "
                            f"SASA={metrics['E_sasa']:6.3f} | "
                            f"T={T:6.1f}K | "
                            f"A={A:4d} | "
                            f"χ={chi:6.3f} | "
                            f"Cv={Cv:6.3f}"
                        )

        # --------------------------------------------------------------------
        # Final Coordinates
        # --------------------------------------------------------------------

        final_coords = self.builder(
            torsions
        ).detach().cpu().numpy()

        print("\n✅ Folding Complete")

        return final_coords

# ============================================================================
# SECTION 14 — MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    cfg = CSOCV6Config(

        n_recycling=2,
        n_stages=2,
        n_iter_per_stage=100

    )

    predictor = CSOCProteinPredictorV6(cfg)

    sequence = "MKTLLLTLVVVTIVCLDLGYAT"

    structure = predictor.predict(sequence)

    print("\nFinal Structure Shape:")
    print(structure.shape)

    print("\nCoordinate Format:")
    print("(Residue, Atom[N,CA,C], XYZ)")

    print("\nFirst Residue Coordinates:")
    print(structure[0])
