# =============================================================================
# full_denovo_pipeline.py
# CSOC-SSC Full De Novo Folding Pipeline
# MIT License — Yoon A Limsuwan 2026
# =============================================================================
"""
Full De Novo Folding Pipeline for CSOC-SSC

Pipeline:
---------
Step 1: Sequence → DistogramNet
Step 2: Distogram → Distance Matrix
Step 3: Distance Matrix → Initial 3D Coordinates (MDS)
Step 4: CSOC-SSC Refinement

Architecture:
--------------
Sequence
   ↓
DistogramNet
   ↓
36-bin Distance Probabilities
   ↓
Expected Distance Matrix
   ↓
Metric Multidimensional Scaling (MDS)
   ↓
Initial Backbone Coordinates
   ↓
CSOC-SSC v10.3 Physics Refinement
   ↓
Final Folded Structure

Designed for:
--------------
• Experimental de novo folding research
• Physics-informed structural optimization
• Distogram-guided folding
• Large-scale differentiable refinement

NOTE:
-----
This is a research framework.
Not intended to outperform AlphaFold/RoseTTAFold.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy.spatial.distance import squareform
from scipy.linalg import eigh

# =============================================================================
# IMPORT CSOC-SSC ENGINE
# =============================================================================

# Assumes your main framework file is:
# csoc_ssc_v10_3.py

from csoc_ssc_v10_3 import (
    BackboneFrame,
    StructuralOptimizationEngine,
    V103Config,
)

# =============================================================================
# AMINO ACID TOKENIZATION
# =============================================================================

AA_TO_IDX = {
    'A':0,'C':1,'D':2,'E':3,'F':4,
    'G':5,'H':6,'I':7,'K':8,'L':9,
    'M':10,'N':11,'P':12,'Q':13,'R':14,
    'S':15,'T':16,'V':17,'W':18,'Y':19,
    'X':20
}

IDX_TO_AA = {v:k for k,v in AA_TO_IDX.items()}

# =============================================================================
# DISTOGRAM NETWORK
# =============================================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        res = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + res)

class DistogramNet(nn.Module):

    def __init__(self, embed_dim=64):
        super().__init__()

        # =========================================================
        # 1D SEQUENCE EMBEDDING
        # =========================================================

        self.embed = nn.Embedding(21, embed_dim)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(embed_dim) for _ in range(4)]
        )

        # =========================================================
        # 2D PAIRWISE NETWORK
        # =========================================================

        self.conv2d_1 = nn.Conv2d(
            embed_dim * 2,
            128,
            kernel_size=3,
            padding=1
        )

        self.res2d = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # =========================================================
        # OUTPUT HEADS
        # =========================================================

        # 36 bins from 2Å → 20Å
        self.dist_head = nn.Conv2d(128, 36, kernel_size=1)

        # Contact probability
        self.contact_head = nn.Conv2d(128, 1, kernel_size=1)

        # Secondary structure
        self.q3_head = nn.Conv1d(embed_dim, 3, kernel_size=1)

    def forward(self, x):

        # x shape: (B, L)

        e = self.embed(x)

        # (B, L, C) -> (B, C, L)
        e = e.transpose(1, 2)

        e = self.res_blocks(e)

        B, C, L = e.shape

        # =========================================================
        # OUTER CONCATENATION
        # =========================================================

        e1 = e.unsqueeze(3).expand(B, C, L, L)
        e2 = e.unsqueeze(2).expand(B, C, L, L)

        pair_rep = torch.cat([e1, e2], dim=1)

        pair_rep = torch.relu(self.conv2d_1(pair_rep))

        pair_rep = self.res2d(pair_rep) + pair_rep

        # =========================================================
        # OUTPUTS
        # =========================================================

        distogram_logits = self.dist_head(pair_rep)

        contact_logits = self.contact_head(pair_rep)

        q3_logits = self.q3_head(e)

        return (
            distogram_logits,
            contact_logits,
            q3_logits
        )

# =============================================================================
# STEP 1 — SEQUENCE → DISTOGRAM
# =============================================================================

def tokenize_sequence(seq: str) -> torch.Tensor:
    """
    Convert AA sequence to integer tokens.
    """

    tokens = [AA_TO_IDX.get(aa, 20) for aa in seq]

    return torch.tensor(tokens, dtype=torch.long)

@torch.no_grad()
def predict_distogram(
    model: DistogramNet,
    sequence: str,
    device: str = "cuda"
) -> Dict:
    """
    Step 1:
    Sequence → Distogram prediction
    """

    model.eval()

    seq_tokens = tokenize_sequence(sequence).unsqueeze(0).to(device)

    dist_logits, contact_logits, q3_logits = model(seq_tokens)

    # Convert logits -> probabilities
    dist_probs = torch.softmax(dist_logits, dim=1)

    contact_probs = torch.sigmoid(contact_logits)

    q3_probs = torch.softmax(q3_logits, dim=1)

    return {
        "distogram_probs": dist_probs,
        "contact_probs": contact_probs,
        "q3_probs": q3_probs,
    }

# =============================================================================
# STEP 2 — DISTOGRAM → DISTANCE MATRIX
# =============================================================================

def convert_distogram_to_distances(
    distogram_probs: torch.Tensor,
    d_min: float = 2.0,
    d_max: float = 20.0,
    n_bins: int = 36
) -> np.ndarray:
    """
    Step 2:
    Convert distogram probabilities into expected distance matrix.

    Expected Distance:
        E[d] = Σ p_i * d_i
    """

    probs = distogram_probs[0].detach().cpu().numpy()

    _, L, _ = probs.shape

    # Bin centers
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)

    bin_centers = 0.5 * (
        bin_edges[:-1] + bin_edges[1:]
    )

    D = np.zeros((L, L), dtype=np.float32)

    for i in range(L):
        for j in range(L):

            p = probs[:, i, j]

            expected_distance = np.sum(
                p * bin_centers
            )

            D[i, j] = expected_distance

    # Symmetrize
    D = 0.5 * (D + D.T)

    # Zero diagonal
    np.fill_diagonal(D, 0.0)

    return D

# =============================================================================
# STEP 3 — DISTANCE MATRIX → 3D COORDINATES (MDS)
# =============================================================================

def classical_mds(
    D: np.ndarray,
    n_components: int = 3
) -> np.ndarray:
    """
    Classical Metric MDS.

    Convert distance matrix into coordinates.

    Steps:
    -------
    D -> Gram Matrix -> Eigen Decomposition -> Coordinates
    """

    n = D.shape[0]

    # Squared distances
    D2 = D ** 2

    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n

    # Double centered Gram matrix
    B = -0.5 * J @ D2 @ J

    # Eigen decomposition
    eigvals, eigvecs = eigh(B)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep top components
    eigvals = np.maximum(eigvals[:n_components], 1e-8)

    eigvecs = eigvecs[:, :n_components]

    coords = eigvecs * np.sqrt(eigvals)

    return coords.astype(np.float32)

def initialize_backbone_from_ca(
    ca_coords: np.ndarray,
    seq: str
) -> BackboneFrame:
    """
    Build approximate backbone from CA trace.
    """

    n_atoms = ca_coords - np.array([0.5, 0.0, 0.0])

    c_atoms = ca_coords + np.array([0.5, 0.0, 0.0])

    o_atoms = c_atoms + np.array([0.3, 0.8, 0.0])

    residue_ids = list(range(1, len(seq) + 1))

    return BackboneFrame(
        n=n_atoms.astype(np.float32),
        ca=ca_coords.astype(np.float32),
        c=c_atoms.astype(np.float32),
        o=o_atoms.astype(np.float32),
        residue_ids=residue_ids,
        seq=seq
    )

# =============================================================================
# STEP 4 — CSOC-SSC REFINEMENT
# =============================================================================

def refine_structure_csoc_ssc(
    backbone_init: BackboneFrame,
    n_stages: int = 5,
    n_iter_per_stage: int = 500
) -> Dict:
    """
    Step 4:
    Physics-informed CSOC-SSC refinement
    """

    config = V103Config(

        n_stages=n_stages,

        n_iter_per_stage=n_iter_per_stage,

        use_sparse=True,

        use_ramachandran=True,

        use_backbone_atoms=True,

        use_criticality_schedule=True,

        use_amp=True,

        optimizer_type='hybrid',

        verbose=1,
    )

    engine = StructuralOptimizationEngine(config)

    results = engine.optimize_backbone(
        backbone_init,
        use_reference_distogram=False
    )

    return results

# =============================================================================
# FULL PIPELINE
# =============================================================================

class FullDeNovoPipeline:
    """
    Complete De Novo Folding Pipeline.

    Pipeline:
    ---------
    Sequence
      -> DistogramNet
      -> Distance Matrix
      -> MDS Coordinates
      -> CSOC-SSC Refinement
      -> Final Structure
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):

        self.device = device

        self.model = DistogramNet().to(device)

        if checkpoint is not None:

            print(f"[Loading] {checkpoint}")

            self.model.load_state_dict(
                torch.load(checkpoint, map_location=device)
            )

        self.model.eval()

    @torch.no_grad()
    def fold(
        self,
        sequence: str,
        refine: bool = True
    ) -> Dict:

        print("=" * 80)
        print("STEP 1 — DISTOGRAM PREDICTION")
        print("=" * 80)

        pred = predict_distogram(
            self.model,
            sequence,
            device=self.device
        )

        distogram_probs = pred["distogram_probs"]

        print("Distogram shape:", distogram_probs.shape)

        print("\n" + "=" * 80)
        print("STEP 2 — DISTANCE MATRIX")
        print("=" * 80)

        D = convert_distogram_to_distances(
            distogram_probs
        )

        print("Distance matrix shape:", D.shape)

        print("\n" + "=" * 80)
        print("STEP 3 — MDS INITIALIZATION")
        print("=" * 80)

        coords_init = classical_mds(D)

        print("Initial coordinates:", coords_init.shape)

        backbone_init = initialize_backbone_from_ca(
            coords_init,
            sequence
        )

        results = {
            "distance_matrix": D,
            "coords_init": coords_init,
            "backbone_init": backbone_init,
        }

        # =============================================================
        # STEP 4 — PHYSICS REFINEMENT
        # =============================================================

        if refine:

            print("\n" + "=" * 80)
            print("STEP 4 — CSOC-SSC REFINEMENT")
            print("=" * 80)

            refine_results = refine_structure_csoc_ssc(
                backbone_init
            )

            results["refinement"] = refine_results

        return results

# =============================================================================
# SAVE PDB
# =============================================================================

def save_backbone_to_pdb(
    backbone: BackboneFrame,
    out_path: str
):
    """
    Save backbone to PDB file.
    """

    atom_id = 1

    with open(out_path, "w") as f:

        for i in range(len(backbone.ca)):

            res_id = i + 1

            aa = backbone.seq[i]

            atoms = [
                ("N",  backbone.n[i]),
                ("CA", backbone.ca[i]),
                ("C",  backbone.c[i]),
                ("O",  backbone.o[i]),
            ]

            for atom_name, coord in atoms:

                x, y, z = coord

                line = (
                    f"ATOM  {atom_id:5d} "
                    f"{atom_name:^4s} "
                    f"ALA A{res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           C\n"
                )

                f.write(line)

                atom_id += 1

        f.write("END\n")

# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("CSOC-SSC FULL DE NOVO FOLDING PIPELINE")
    print("=" * 80)

    # =============================================================
    # EXAMPLE SEQUENCE
    # =============================================================

    sequence = (
        "MKTFFVAGVILLLAALPATANAD"
        "QISFVKSHFSRQLEERLGLIEVQ"
    )

    print("\nSequence Length:", len(sequence))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using Device:", device)

    # =============================================================
    # INITIALIZE PIPELINE
    # =============================================================

    pipeline = FullDeNovoPipeline(
        checkpoint=None,
        device=device
    )

    # =============================================================
    # RUN FULL FOLDING
    # =============================================================

    results = pipeline.fold(
        sequence,
        refine=True
    )

    # =============================================================
    # SAVE OUTPUT
    # =============================================================

    if "refinement" in results:

        backbone_final = results["refinement"]["backbone_final"]

    else:

        backbone_final = results["backbone_init"]

    out_pdb = "denovo_folded.pdb"

    save_backbone_to_pdb(
        backbone_final,
        out_pdb
    )

    print("\n")
    print("=" * 80)
    print("FOLDING COMPLETE")
    print("=" * 80)

    print(f"Saved structure: {out_pdb}")

    if "refinement" in results:

        print(
            f"Peak VRAM: "
            f"{results['refinement']['memory_peak_gb']:.2f} GB"
        )

        print(
            f"Runtime: "
            f"{results['refinement']['time_total_sec']:.1f} sec"
        )

    print()
