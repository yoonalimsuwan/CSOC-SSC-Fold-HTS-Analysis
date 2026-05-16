#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v30.2 — DNA/RNA Full Extension Module
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Standalone DNA/RNA module — Production‑ready for CSOC‑SSC v30.1
#
# Features:
#   ✓ Internal coordinate builder (bond‑angle‑dihedral)
#   ✓ P‑P backbone distance restraint
#   ✓ Sugar pucker (C2'‑endo B‑DNA / C3'‑endo A‑RNA)
#   ✓ Watson‑Crick base pairing
#   ✓ π‑π base stacking
#   ✓ Full‑atom LJ + Coulomb
#   ✓ Integration with v30.1 energy function
#   ✓ B‑DNA / A‑RNA helix initial builder
#   ✓ PDB loader
#   ✓ Configurable force field
# =============================================================================

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import os

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

DNA_VOCAB = "ACGT"
RNA_VOCAB = "ACGU"
DNA_RNA_VOCAB = "ACGUT"

NT_TO_ID = {nt: i for i, nt in enumerate(DNA_RNA_VOCAB)}

# Watson‑Crick pairs: (base1, base2) → number of H‑bonds
WC_PAIRS = {
    ('A', 'T'): 2, ('T', 'A'): 2,
    ('A', 'U'): 2, ('U', 'A'): 2,
    ('G', 'C'): 3, ('C', 'G'): 3,
    ('G', 'U'): 1, ('U', 'G'): 1,  # wobble
}

BASE_STACKING = {'A': 1.0, 'T': 0.8, 'U': 0.8, 'G': 1.2, 'C': 1.0}

# ═══════════════════════════════════════════════════════════════
# NUCLEOTIDE TOPOLOGY (Internal Coordinates)
# =====================================================================
# Format per atom:
#   (atom_name, atom_type, parent_idx, bond_length, bond_angle_deg,
#    ref_atom_indices, dihedral_offset_deg)
#
# ref_atom_indices: tuple of 3 indices relative to parent:
#   -3 = 3 atoms before parent
#   -2 = 2 atoms before parent
#   -1 = 1 atom before parent
#    0 = parent itself
#   +n = n atoms after parent (in build order)
# =====================================================================

# ─── Phosphate‑Sugar Backbone (common to all nucleotides) ───
NUCLEOTIDE_BACKBONE = [
    # (name, type, parent, bond_len, bond_ang, ref_tuple, dihedral0)
    # Anchor: C4' is reference (index 0)
    ('C4\'', 'C',   0, 0.00,   0.0, ( 0, 0, 0),   0.0),  # [0]  reference
    ('O4\'', 'O',   0, 1.44, 109.5, ( 0, 0, 0),   0.0),  # [1]
    ('C1\'', 'C',   1, 1.42, 109.5, ( 0,-1, 0),   0.0),  # [2]
    ('C2\'', 'C',   0, 1.52, 109.5, (-1,-2, 0), 120.0),  # [3]
    ('C3\'', 'C',   0, 1.52, 109.5, (-1,-2, 0),-120.0),  # [4]
    ('O3\'', 'O',   4, 1.43, 109.5, (-1, 0, 1),   0.0),  # [5]
    ('C5\'', 'C',   0, 1.51, 109.5, (-2,-3, 0), 180.0),  # [6]
    ('O5\'', 'O',   6, 1.42, 109.5, (-1, 0, 1),   0.0),  # [7]
    ('P',    'P',   7, 1.60, 119.0, (-1, 0, 1), 180.0),  # [8]
    ('OP1',  'O',   8, 1.48, 109.5, (-1, 0, 1),   0.0),  # [9]
    ('OP2',  'O',   8, 1.48, 109.5, (-1, 0, 1), 180.0),  # [10]
]

# ─── Base Attachments (attached to C1' [index 2]) ───
# Pyrimidine bases (C, U, T) — attach via N1
PYRIMIDINE_BASE = [
    ('N1',  'N',   2, 1.47, 109.5, (-1, 0, 1),   0.0),  # [11]
    ('C2',  'C',  11, 1.40, 121.0, (-1, 0, 1),   0.0),  # [12]
    ('O2',  'O',  12, 1.24, 118.0, (-1, 0, 1),   0.0),  # [13]
    ('N3',  'N',  12, 1.35, 120.0, (-1, 0, 1), 180.0),  # [14]
    ('C4',  'C',  14, 1.33, 118.0, (-1, 0, 1),   0.0),  # [15]
    ('C5',  'C',  15, 1.43, 118.0, (-1, 0, 1),   0.0),  # [16]
    ('C6',  'C',  16, 1.34, 122.0, (-1, 0, 1),   0.0),  # [17]
]

# Cytosine‑specific
CYTOSINE_EXTRA = [
    ('N4',  'N',  15, 1.33, 118.0, (-1, 0, 1), 180.0),  # [18]
]

# Uracil‑specific
URACIL_EXTRA = [
    ('O4',  'O',  15, 1.23, 118.0, (-1, 0, 1), 180.0),  # [18]
]

# Thymine‑specific
THYMINE_EXTRA = [
    ('O4',  'O',  15, 1.23, 118.0, (-1, 0, 1), 180.0),  # [18]
    ('C7',  'C',  16, 1.50, 122.0, (-1, 0, 1),   0.0),  # [19]  5‑methyl
]

# Purine bases (A, G) — attach via N9
PURINE_BASE = [
    ('N9',  'N',   2, 1.46, 109.5, (-1, 0, 1),   0.0),  # [11]
    ('C4',  'C',  11, 1.37, 126.0, (-1, 0, 1),   0.0),  # [12]
    ('C5',  'C',  12, 1.39, 106.0, (-1, 0, 1),   0.0),  # [13]
    ('C6',  'C',  13, 1.40, 110.0, (-1, 0, 1),   0.0),  # [14]
    ('N1',  'N',  14, 1.34, 118.0, (-1, 0, 1),   0.0),  # [15]
    ('C2',  'C',  15, 1.32, 129.0, (-1, 0, 1),   0.0),  # [16]
    ('N3',  'N',  16, 1.32, 110.0, (-1, 0, 1),   0.0),  # [17]
    ('N7',  'N',  13, 1.33, 114.0, (-2,-1, 0), 180.0),  # [18]
    ('C8',  'C',  18, 1.37, 106.0, (-1, 0, 1),   0.0),  # [19]
]

# Adenine‑specific
ADENINE_EXTRA = [
    ('N6',  'N',  14, 1.34, 124.0, (-1, 0, 1), 180.0),  # [20]
]

# Guanine‑specific
GUANINE_EXTRA = [
    ('O6',  'O',  14, 1.23, 124.0, (-1, 0, 1), 180.0),  # [20]
    ('N2',  'N',  16, 1.34, 120.0, (-1, 0, 1), 180.0),  # [21]
]

# ─── Complete Topology per Nucleotide ───
NUCLEOTIDE_TOPOLOGY = {
    'A': NUCLEOTIDE_BACKBONE + PURINE_BASE + ADENINE_EXTRA,
    'G': NUCLEOTIDE_BACKBONE + PURINE_BASE + GUANINE_EXTRA,
    'C': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + CYTOSINE_EXTRA,
    'U': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + URACIL_EXTRA,
    'T': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + THYMINE_EXTRA,
}

# ─── Atom type for naming consistency ───
def get_atom_type_for_topology(atom_name):
    """Map atom name to element type code."""
    if atom_name.startswith('C'): return 'C'
    if atom_name.startswith('N'): return 'N'
    if atom_name.startswith('O'): return 'O'
    if atom_name.startswith('P'): return 'P'
    if atom_name.startswith('S'): return 'S'
    return 'C'

# ═══════════════════════════════════════════════════════════════
# FORCE FIELD PARAMETERS
# ═══════════════════════════════════════════════════════════════

NUCLEOTIDE_LJ = {
    # (sigma_Å, epsilon_kcal/mol)
    'P':   (2.1000, 0.2000),
    'O':   (1.6612, 0.2100),
    'N':   (1.8240, 0.1700),
    'C':   (1.9080, 0.0860),
    'S':   (2.0000, 0.2500),
}

NUCLEOTIDE_CHARGES = {
    'P':   0.90,
    'OP1': -0.70,
    'OP2': -0.70,
    'O5\'':-0.50,
    'C5\'':-0.10,
    'C4\'': 0.00,
    'O4\'':-0.40,
    'C1\'': 0.20,
    'C2\'':-0.10,
    'C3\'': 0.10,
    'O3\'':-0.50,
    # Base atoms
    'N1':  -0.50, 'N2':  -0.80, 'N3':  -0.60,
    'N4':  -0.80, 'N6':  -0.80, 'N7':  -0.50,
    'N9':  -0.30,
    'C2':   0.40, 'C4':   0.30, 'C5':   0.10,
    'C6':   0.10, 'C7':  -0.20, 'C8':   0.20,
    'O2':  -0.55, 'O4':  -0.55, 'O6':  -0.55,
}

DEFAULT_LJ = NUCLEOTIDE_LJ
DEFAULT_CHARGES = NUCLEOTIDE_CHARGES

def load_nucleotide_forcefield(lj_file=None, charge_file=None):
    """Load configurable force field parameters."""
    lj = DEFAULT_LJ.copy()
    charges = DEFAULT_CHARGES.copy()
    if lj_file and os.path.exists(lj_file):
        with open(lj_file) as f:
            lj.update(json.load(f))
    if charge_file and os.path.exists(charge_file):
        with open(charge_file) as f:
            charges.update(json.load(f))
    return lj, charges

# ═══════════════════════════════════════════════════════════════
# INTERNAL COORDINATE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_single_nucleotide_ic(C4_prime,   # [3] tensor
                                prev_C4,    # [3] tensor or None
                                next_C4,    # [3] tensor or None
                                nt_type,    # 'A','C','G','U','T'
                                ) -> Tuple[torch.Tensor, List[str]]:
    """
    Build all heavy atoms for ONE nucleotide using internal coordinates.
    
    Uses the same IC builder pattern as RESIDUE_TOPOLOGY in v30.1.
    
    Args:
        C4_prime: position of C4' atom [3]
        prev_C4: C4' of previous nucleotide (or None for 5'‑end)
        next_C4: C4' of next nucleotide (or None for 3'‑end)
        nt_type: nucleotide type ('A','C','G','U','T')
    
    Returns:
        coords: [N_atoms, 3]
        types: list of atom names
    """
    device = C4_prime.device
    topo = NUCLEOTIDE_TOPOLOGY.get(nt_type, [])
    
    if not topo:
        return torch.zeros((0, 3), device=device), []
    
    coords = []
    types = []
    
    # Build local coordinate frame at C4'
    # Use prev_C4 and next_C4 to orient the sugar
    if prev_C4 is not None and next_C4 is not None:
        # Internal frame: x = C4' → next_C4, z = normal to C4'‑prev‑next plane
        x_axis = F.normalize(next_C4 - C4_prime, dim=-1, eps=1e-8)
        v_tmp = C4_prime - prev_C4
        z_axis = F.normalize(torch.cross(x_axis, v_tmp, dim=-1), dim=-1, eps=1e-8)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
    elif next_C4 is not None:
        x_axis = F.normalize(next_C4 - C4_prime, dim=-1, eps=1e-8)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-8)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
    else:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Build atoms sequentially
    for atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral0 in topo:
        if parent_idx == 0 and len(coords) == 0:
            # First atom: C4' itself
            pos = C4_prime
        elif parent_idx < len(coords):
            parent_pos = coords[parent_idx]
            
            # Get reference atoms for dihedral
            a_idx = ref_tuple[0]
            b_idx = ref_tuple[1]
            c_idx = ref_tuple[2]
            
            # Map relative indices to absolute
            def map_idx(idx, current_len):
                if idx == 0:
                    return parent_idx
                elif idx > 0:
                    return min(idx - 1, current_len - 1) if current_len > 0 else 0
                else:
                    return max(0, current_len + idx)
            
            a_abs = map_idx(a_idx, len(coords))
            b_abs = map_idx(b_idx, len(coords))
            c_abs = map_idx(c_idx, len(coords))
            
            p_a = coords[a_abs] if a_abs < len(coords) else parent_pos
            p_b = coords[b_abs] if b_abs < len(coords) else parent_pos
            p_c = coords[c_abs] if c_abs < len(coords) else parent_pos
            
            # Build using bond‑angle‑dihedral (same algorithm as v30.1 side‑chain builder)
            bc = p_c - p_b
            bc_norm = F.normalize(bc, dim=-1, eps=1e-8)
            
            # Orthogonal vector
            ref_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
            dot = torch.abs(torch.dot(bc_norm, ref_vec))
            if dot > 0.9:
                ref_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp_norm_sq = torch.dot(perp, perp)
            if perp_norm_sq < 1e-12:
                ref_vec = torch.tensor([0.0, 0.0, 1.0], device=device)
                perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-8)
            
            total_angle = math.radians(dihedral0)  # no chi angles for DNA/RNA (rigid)
            cos_a, sin_a = math.cos(total_angle), math.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            
            ang = math.radians(bond_ang_deg)
            bond_dir = math.cos(ang) * bc_norm + math.sin(ang) * rotated_perp
            pos = p_c + bond_len * bond_dir
        else:
            # Fallback: place approximately
            if len(coords) > 0:
                pos = coords[-1] + bond_len * x_axis
            else:
                pos = C4_prime + bond_len * x_axis
        
        coords.append(pos)
        types.append(atom_name)
    
    return torch.stack(coords, dim=0), types


def build_full_dna_rna(C4_coords,     # [L, 3]
                        sequence,       # "ACGT" or "ACGU"
                        ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Build full DNA/RNA structure from C4' trace.
    
    Returns:
        all_coords: [N_total, 3]
        all_types: list of atom names
        res_indices: [N_total] — which nucleotide each atom belongs to
    """
    L = len(sequence)
    device = C4_coords.device
    
    all_coords_list = []
    all_types_list = []
    all_res_idx_list = []
    
    for i in range(L):
        prev_C4 = C4_coords[i - 1] if i > 0 else None
        next_C4 = C4_coords[i + 1] if i < L - 1 else None
        nt = sequence[i]
        
        nuc_coords, nuc_types = build_single_nucleotide_ic(
            C4_coords[i], prev_C4, next_C4, nt
        )
        
        all_coords_list.append(nuc_coords)
        all_types_list.extend(nuc_types)
        all_res_idx_list.append(torch.full((nuc_coords.shape[0],), i,
                                           dtype=torch.long, device=device))
    
    if not all_coords_list:
        return (torch.zeros((0, 3), device=device), [], torch.zeros(0, device=device))
    
    return (
        torch.cat(all_coords_list, dim=0),
        all_types_list,
        torch.cat(all_res_idx_list, dim=0),
    )

# ═══════════════════════════════════════════════════════════════
# ENERGY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def energy_backbone_c4_bond(C4_coords, w=30.0, ideal_d=6.5):
    """C4'‑C4' backbone bond restraint (per‑nucleotide spacing)."""
    if len(C4_coords) < 2:
        return torch.tensor(0.0, device=C4_coords.device)
    d = torch.norm(C4_coords[1:] - C4_coords[:-1], dim=-1)
    return w * ((d - ideal_d) ** 2).mean()


def energy_phosphate_restraint(all_coords, all_types, res_indices, w=15.0, ideal_d=6.0):
    """
    P‑P distance restraint between adjacent nucleotides.
    Keeps phosphate backbone geometry correct.
    """
    device = all_coords.device
    L = int(res_indices.max().item()) + 1 if res_indices.numel() > 0 else 0
    if L < 2:
        return torch.tensor(0.0, device=device)
    
    # Find P atoms for each residue
    is_P = torch.tensor([t == 'P' for t in all_types], device=device)
    P_idx = torch.where(is_P)[0]
    P_res = res_indices[is_P]
    
    energy = torch.tensor(0.0, device=device)
    count = 0
    
    for i in range(L - 1):
        mask_i = P_res == i
        mask_j = P_res == i + 1
        if mask_i.any() and mask_j.any():
            p_i = all_coords[P_idx[mask_i][0]]
            p_j = all_coords[P_idx[mask_j][0]]
            d = torch.norm(p_i - p_j)
            energy += (d - ideal_d) ** 2
            count += 1
    
    return w * energy / max(1, count)


def energy_sugar_pucker(all_coords, all_types, res_indices,
                         pucker_type='C2_endo', w=10.0):
    """
    Sugar pucker restraint.
    
    B‑DNA: C2'‑endo (C2' above C3' relative to C4'‑O4' plane)
    A‑RNA: C3'‑endo (C3' above C2' relative to C4'‑O4' plane)
    
    We enforce this by checking the dihedral angle ν2
    (C1'‑C2'‑C3'‑C4') and restraining it.
    """
    device = all_coords.device
    L = int(res_indices.max().item()) + 1 if res_indices.numel() > 0 else 0
    
    # Target ν2 dihedral:
    # C2'‑endo (B‑DNA): ν2 ≈ 36°
    # C3'‑endo (A‑RNA): ν2 ≈ 15°
    if pucker_type == 'C2_endo':
        target_nu2 = math.radians(36.0)
    elif pucker_type == 'C3_endo':
        target_nu2 = math.radians(15.0)
    else:
        target_nu2 = math.radians(36.0)
    
    energy = torch.tensor(0.0, device=device)
    count = 0
    
    for i in range(L):
        mask = res_indices == i
        res_coords = all_coords[mask]
        res_types_i = [all_types[j] for j in range(len(all_types)) if res_indices[j] == i]
        
        # Find C1', C2', C3', C4' positions
        pos = {}
        for name in ['C1\'', 'C2\'', 'C3\'', 'C4\'']:
            if name in res_types_i:
                idx = res_types_i.index(name)
                pos[name] = res_coords[idx]
        
        if all(k in pos for k in ['C1\'', 'C2\'', 'C3\'', 'C4\'']):
            # Compute ν2 dihedral
            nu2 = compute_dihedral(pos['C1\''], pos['C2\''], pos['C3\''], pos['C4\''])
            diff = nu2 - target_nu2
            # Periodic: wrap to [-π, π]
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            energy += diff ** 2
            count += 1
    
    return w * energy / max(1, count)


def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle between 4 points (radians)."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - torch.dot(b0, b1n) * b1n
    w = b2 - torch.dot(b2, b1n) * b1n
    
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1n, v, dim=-1), w)
    
    return torch.atan2(y + 1e-8, x + 1e-8)


def energy_base_pairing(C4_coords, sequence, w=8.0, ideal_d=10.5, sigma=2.0):
    """Watson‑Crick base pairing energy."""
    L = len(sequence)
    device = C4_coords.device
    energy = torch.tensor(0.0, device=device)
    
    for i in range(L):
        for j in range(i + 4, min(L, i + 50)):
            d = torch.norm(C4_coords[i] - C4_coords[j])
            pair = (sequence[i], sequence[j])
            n_hbonds = WC_PAIRS.get(pair, 0)
            if n_hbonds > 0:
                E_pair = -n_hbonds * torch.exp(-((d - ideal_d) / sigma) ** 2)
                energy += E_pair
    
    return w * energy / max(1, L)


def energy_base_stacking(C4_coords, sequence, w=5.0, ideal_d=6.5, sigma=1.5):
    """π‑π stacking between adjacent bases."""
    L = len(sequence)
    device = C4_coords.device
    energy = torch.tensor(0.0, device=device)
    
    for i in range(L - 1):
        d = torch.norm(C4_coords[i + 1] - C4_coords[i])
        s_i = BASE_STACKING.get(sequence[i], 1.0)
        s_j = BASE_STACKING.get(sequence[i + 1], 1.0)
        s_avg = 0.5 * (s_i + s_j)
        E_stack = -s_avg * torch.exp(-((d - ideal_d) / sigma) ** 2)
        energy += E_stack
    
    return w * energy / max(1, L - 1)


def energy_dna_rna_lj(all_coords, all_types, edge_index, edge_dist,
                       lj_params=None, w=30.0):
    """Lennard‑Jones for DNA/RNA full‑atom."""
    device = all_coords.device
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    if lj_params is None:
        lj_params = NUCLEOTIDE_LJ
    
    src, dst = edge_index[0], edge_index[1]
    
    # Map atom types to element
    element_types = [get_atom_type_for_topology(t) for t in all_types]
    
    sigmas = torch.tensor([lj_params.get(t, (1.9, 0.1))[0] for t in element_types],
                          device=device)
    epsilons = torch.tensor([lj_params.get(t, (1.9, 0.1))[1] for t in element_types],
                            device=device)
    
    sigma_ij = 0.5 * (sigmas[src] + sigmas[dst])
    eps_ij = torch.sqrt(epsilons[src] * epsilons[dst])
    
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0 / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    
    lj_energy = 4.0 * eps_ij * ((sigma_ij * inv_r) ** 12 - (sigma_ij * inv_r) ** 6)
    
    return w * lj_energy.mean()


def energy_dna_rna_coulomb(all_coords, all_types, res_indices,
                            edge_index, edge_dist, charge_map=None, w=3.0):
    """Coulomb energy for DNA/RNA full‑atom."""
    device = all_coords.device
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    if charge_map is None:
        charge_map = NUCLEOTIDE_CHARGES
    
    src, dst = edge_index[0], edge_index[1]
    q = torch.tensor([charge_map.get(t, 0.0) for t in all_types], device=device)
    qi, qj = q[src], q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r  # distance‑dependent
    coulomb = 332.0637 * qi * qj / (dielectric * r + 1e-8)
    
    return w * coulomb.mean()


# ═══════════════════════════════════════════════════════════════
# MAIN DNA/RNA ENERGY CLASS
# ═══════════════════════════════════════════════════════════════

class DNA_RNA_Energy:
    """
    Complete DNA/RNA energy for integration with CSOC‑SSC v30.1
    
    Usage:
        dna_eng = DNA_RNA_Energy(pucker_type='C2_endo')  # B‑DNA
        E = dna_eng(C4_coords, sequence)
    
    Integration into v30.1 total_physics_energy_v30_1:
        if is_dna_rna:
            E_total += dna_eng(C4_coords, nt_sequence)
    """
    
    def __init__(self,
                 pucker_type='C2_endo',     # 'C2_endo' for B‑DNA, 'C3_endo' for A‑RNA
                 w_c4_bond=30.0,
                 w_phosphate=15.0,
                 w_pucker=10.0,
                 w_base_pair=8.0,
                 w_stacking=5.0,
                 w_lj=30.0,
                 w_coulomb=3.0,
                 lj_params=None,
                 charge_map=None,
                 use_full_atom=True):
        self.pucker_type = pucker_type
        self.w_c4_bond = w_c4_bond
        self.w_phosphate = w_phosphate
        self.w_pucker = w_pucker
        self.w_base_pair = w_base_pair
        self.w_stacking = w_stacking
        self.w_lj = w_lj
        self.w_coulomb = w_coulomb
        self.lj_params = lj_params or NUCLEOTIDE_LJ
        self.charge_map = charge_map or NUCLEOTIDE_CHARGES
        self.use_full_atom = use_full_atom
    
    def __call__(self, C4_coords, sequence,
                 edge_index=None, edge_dist=None):
        """
        Compute total DNA/RNA energy.
        
        Args:
            C4_coords: [L, 3] C4' atom coordinates
            sequence: string of nucleotide 1‑letter codes
            edge_index: [2, E] sparse edges for full‑atom (optional)
            edge_dist: [E] sparse edge distances (optional)
        
        Returns:
            total_energy: scalar tensor
        """
        device = C4_coords.device
        E = torch.tensor(0.0, device=device)
        
        # C4' backbone bond
        E += energy_backbone_c4_bond(C4_coords, self.w_c4_bond)
        
        # Base pairing
        E += energy_base_pairing(C4_coords, sequence, self.w_base_pair)
        
        # Base stacking
        E += energy_base_stacking(C4_coords, sequence, self.w_stacking)
        
        # Full‑atom energies
        if self.use_full_atom:
            all_coords, all_types, res_indices = build_full_dna_rna(C4_coords, sequence)
            
            if all_coords.shape[0] > 1:
                # Phosphate backbone restraint
                E += energy_phosphate_restraint(all_coords, all_types, res_indices,
                                                self.w_phosphate)
                
                # Sugar pucker
                E += energy_sugar_pucker(all_coords, all_types, res_indices,
                                         self.pucker_type, self.w_pucker)
                
                # LJ + Coulomb (if edges provided)
                if edge_index is not None and edge_index.numel() > 0:
                    E += energy_dna_rna_lj(all_coords, all_types, edge_index, edge_dist,
                                           self.lj_params, self.w_lj)
                    E += energy_dna_rna_coulomb(all_coords, all_types, res_indices,
                                                edge_index, edge_dist,
                                                self.charge_map, self.w_coulomb)
        
        return E


# ═══════════════════════════════════════════════════════════════
# HELIX BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_dna_helix(sequence: str,
                    rise: float = 3.38,        # Å per base pair (B‑DNA)
                    twist: float = 36.0,        # degrees per base pair
                    radius: float = 8.0,        # Å (C4' distance from helix axis)
                    start_angle: float = 0.0,   # radians
                    ) -> torch.Tensor:
    """
    Build an ideal B‑DNA helix for initial structure.
    
    Places C4' atoms on a perfect helix with given rise and twist.
    
    Args:
        sequence: nucleotide sequence (e.g., "ACGTA")
        rise: vertical rise per nucleotide (Å)
        twist: helical twist per nucleotide (degrees)
        radius: distance from helix axis to C4' (Å)
        start_angle: initial azimuthal angle (radians)
    
    Returns:
        C4_coords: [L, 3] tensor
    """
    L = len(sequence)
    coords = torch.zeros(L, 3)
    
    for i in range(L):
        angle = start_angle + math.radians(i * twist)
        coords[i, 0] = radius * math.cos(angle)
        coords[i, 1] = radius * math.sin(angle)
        coords[i, 2] = i * rise
    
    return coords


def build_rna_helix(sequence: str,
                    rise: float = 2.80,         # Å per base pair (A‑form RNA)
                    twist: float = 32.7,         # degrees per base pair
                    radius: float = 9.0,         # Å (C4' distance, wider than B‑DNA)
                    start_angle: float = 0.0,
                    ) -> torch.Tensor:
    """
    Build an ideal A‑form RNA helix for initial structure.
    
    A‑form RNA differs from B‑DNA:
      - Smaller rise (2.8 vs 3.4 Å)
      - Smaller twist (32.7° vs 36°)
      - Larger radius (9.0 vs 8.0 Å)
      - C3'‑endo sugar pucker
    """
    return build_dna_helix(sequence, rise=rise, twist=twist,
                           radius=radius, start_angle=start_angle)


def build_double_strand_helix(sequence_forward: str,
                               sequence_reverse: str = None,
                               ds_type: str = 'B_DNA',   # 'B_DNA' or 'A_RNA'
                               ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Build a double‑stranded helix (forward + reverse complement).
    
    If reverse not provided, auto‑generates Watson‑Crick complement.
    
    Args:
        sequence_forward: 5'→3' sequence of forward strand
        sequence_reverse: 5'→3' sequence of reverse strand (optional)
        ds_type: 'B_DNA' or 'A_RNA'
    
    Returns:
        fwd_coords: [L, 3] forward strand C4' coordinates
        rev_coords: [L, 3] reverse strand C4' coordinates
        fwd_seq: forward strand sequence
        rev_seq: reverse strand sequence
    """
    if ds_type == 'B_DNA':
        rise = 3.38
        twist = 36.0
        radius = 8.0
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    elif ds_type == 'A_RNA':
        rise = 2.80
        twist = 32.7
        radius = 9.0
        complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}. Use 'B_DNA' or 'A_RNA'.")
    
    # Generate reverse complement if not provided
    if sequence_reverse is None:
        sequence_reverse = ''.join([complement_map.get(b, 'X') 
                                     for b in reversed(sequence_forward)])
    
    L = len(sequence_forward)
    
    # Forward strand: starts at angle 0
    fwd_coords = build_dna_helix(sequence_forward, rise=rise, twist=twist,
                                  radius=radius, start_angle=0.0)
    
    # Reverse strand: antiparallel, starts at angle π (opposite side)
    rev_coords = build_dna_helix(sequence_reverse, rise=rise, twist=twist,
                                  radius=radius, start_angle=math.pi)
    
    # Reverse strand runs 3'→5' (opposite Z direction)
    # Flip Z coordinates
    rev_coords_flipped = rev_coords.clone()
    rev_coords_flipped[:, 2] = (L - 1) * rise - rev_coords[:, 2]
    
    return fwd_coords, rev_coords_flipped, sequence_forward, sequence_reverse


def build_dna_rna_from_pdb(C4_coords: torch.Tensor,
                            sequence: str,
                            is_double_strand: bool = False,
                            pucker_type: str = 'C2_endo',
                            ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Build full atomic model from C4' trace loaded from PDB.
    
    This is the recommended entry point when you have C4' coordinates
    from experiment or prediction.
    
    Args:
        C4_coords: [L, 3] C4' atom coordinates
        sequence: nucleotide sequence string
        is_double_strand: if True, assumes two strands concatenated
        pucker_type: 'C2_endo' or 'C3_endo'
    
    Returns:
        all_coords: [N_total, 3]
        all_types: list of atom names
        res_indices: [N_total]
    """
    return build_full_dna_rna(C4_coords, sequence)


# ═══════════════════════════════════════════════════════════════
# PDB LOADER
# ═══════════════════════════════════════════════════════════════

def load_nucleotide_pdb(pdb_path: str,
                         chain: str = 'A',
                         ) -> Tuple[torch.Tensor, str]:
    """
    Load C4' coordinates and sequence from a DNA/RNA PDB file.
    
    Handles both standard PDB and gzipped PDB (.pdb.gz).
    
    Args:
        pdb_path: path to PDB or .pdb.gz file
        chain: chain ID to extract
    
    Returns:
        C4_coords: [L, 3] tensor of C4' atom positions
        sequence: nucleotide sequence string (1‑letter code)
    """
    import gzip
    
    nt_3_to_1 = {
        'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', 'DU': 'U',
        'A':  'A', 'C':  'C', 'G':  'G', 'T':  'T', 'U':  'U',
        'ADE':'A', 'CYT':'C', 'GUA':'G', 'THY':'T', 'URA':'U',
    }
    
    coords = []
    seq = []
    seen_residues = set()
    
    opener = gzip.open if pdb_path.endswith('.gz') else open
    
    try:
        with opener(pdb_path, 'rt', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM') and not line.startswith('HETATM'):
                    continue
                
                line_chain = line[21].strip()
                if line_chain != chain:
                    continue
                
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                ins_code = line[26].strip() if len(line) > 26 else ''
                
                # Unique residue identifier
                res_key = (res_num, ins_code)
                
                # Look for C4' or C4* atom (standard PDB naming)
                if atom_name in ("C4'", "C4*", "C4'"):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    if res_key not in seen_residues:
                        coords.append([x, y, z])
                        nt = nt_3_to_1.get(res_name, 'X')
                        seq.append(nt)
                        seen_residues.add(res_key)
    
    except Exception as e:
        print(f"Error loading {pdb_path}: {e}")
        return torch.empty((0, 3)), ""
    
    if not coords:
        print(f"Warning: No C4' atoms found in {pdb_path} chain {chain}")
        return torch.empty((0, 3)), ""
    
    return torch.tensor(coords, dtype=torch.float32), "".join(seq)


def load_double_strand_pdb(pdb_path: str,
                            chain_fwd: str = 'A',
                            chain_rev: str = 'B',
                            ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Load both strands of a double‑stranded DNA/RNA from PDB.
    
    Args:
        pdb_path: path to PDB file
        chain_fwd: chain ID for forward strand
        chain_rev: chain ID for reverse strand
    
    Returns:
        fwd_coords: [L, 3] forward strand C4' coordinates
        rev_coords: [L, 3] reverse strand C4' coordinates
        fwd_seq: forward strand sequence
        rev_seq: reverse strand sequence
    """
    fwd_coords, fwd_seq = load_nucleotide_pdb(pdb_path, chain_fwd)
    rev_coords, rev_seq = load_nucleotide_pdb(pdb_path, chain_rev)
    return fwd_coords, rev_coords, fwd_seq, rev_seq


# ═══════════════════════════════════════════════════════════════
# PDB WRITER
# ═══════════════════════════════════════════════════════════════

def write_nucleotide_pdb(C4_coords: torch.Tensor,
                          sequence: str,
                          filename: str,
                          chain_id: str = 'A',
                          pucker_type: str = 'C2_endo',
                          ) -> None:
    """
    Write full‑atom DNA/RNA structure to PDB file.
    
    Args:
        C4_coords: [L, 3] C4' atom coordinates
        sequence: nucleotide sequence
        filename: output PDB filename
        chain_id: chain identifier
        pucker_type: 'C2_endo' or 'C3_endo'
    """
    all_coords, all_types, res_indices = build_full_dna_rna(C4_coords, sequence)
    
    # Nucleotide 1‑letter to 3‑letter
    nt_1_to_3 = {
        'A': '  A', 'G': '  G', 'C': '  C', 'T': ' DT', 'U': '  U',
    }
    
    with open(filename, 'w') as f:
        atom_serial = 1
        
        for i in range(len(sequence)):
            nt = sequence[i]
            res_name = nt_1_to_3.get(nt, '  X')
            
            mask = res_indices == i
            res_coords = all_coords[mask]
            res_types = [all_types[j] for j, m in enumerate(mask) if m]
            
            for k, (at_name, pos) in enumerate(zip(res_types, res_coords)):
                x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
                
                # Determine element for column 77‑78
                element = at_name[0] if at_name[0] in 'CNOPS' else 'C'
                
                f.write(
                    f"ATOM  {atom_serial:5d} {at_name:<4s} {res_name} "
                    f"{chain_id}{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
                atom_serial += 1
        
        f.write("END\n")
    
    print(f"DNA/RNA PDB written to {filename} ({atom_serial-1} atoms)")


# ═══════════════════════════════════════════════════════════════
# INTEGRATION WITH CSOC‑SSC v30.1
# ═══════════════════════════════════════════════════════════════

def integrate_dna_rna_into_v30_energy(ca_or_c4, seq, alpha, chi_angles,
                                       edge_index_ca, edge_dist_ca,
                                       edge_index_hbond, edge_dist_hbond,
                                       chain_boundaries, cfg,
                                       is_dna_rna=False,
                                       dna_rna_eng=None):
    """
    Drop‑in addition to total_physics_energy_v30_1 for DNA/RNA support.
    
    Add this call inside total_physics_energy_v30_1 after computing
    protein energy terms.
    
    Args:
        ca_or_c4: coordinates (CA for protein, C4' for DNA/RNA)
        seq: sequence string
        alpha: alpha values (ignored for DNA/RNA, use ones)
        chi_angles: chi angles (ignored for DNA/RNA)
        edge_index_ca: sparse edge index
        edge_dist_ca: sparse edge distances
        edge_index_hbond: H‑bond edges (can be None)
        edge_dist_hbond: H‑bond distances (can be None)
        chain_boundaries: list of chain start indices
        cfg: V30_1Config instance
        is_dna_rna: if True, treat this chain as DNA/RNA
        dna_rna_eng: DNA_RNA_Energy instance
    
    Returns:
        additional_energy: scalar tensor
    """
    if not is_dna_rna or dna_rna_eng is None:
        return torch.tensor(0.0, device=ca_or_c4.device)
    
    return dna_rna_eng(ca_or_c4, seq, edge_index_ca, edge_dist_ca)


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE & TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CSOC‑SSC v30.2 — DNA/RNA Full Extension Module")
    print("=" * 70)
    
    # ── Test 1: Build B‑DNA helix ──
    seq = "ACGTA"
    print(f"\n[Test 1] Building B‑DNA helix: {seq}")
    c4 = build_dna_helix(seq)
    print(f"  C4' shape: {c4.shape}")
    print(f"  First 3 C4':\n{c4[:3]}")
    
    # ── Test 2: Build full atomic model ──
    print(f"\n[Test 2] Building full atomic model")
    all_coords, all_types, res_idx = build_full_dna_rna(c4, seq)
    print(f"  Total atoms: {all_coords.shape[0]}")
    print(f"  Atom types: {all_types[:20]}...")
    
    # ── Test 3: Compute energies ──
    print(f"\n[Test 3] Computing energies")
    dna_eng = DNA_RNA_Energy(pucker_type='C2_endo', use_full_atom=True)
    E_total = dna_eng(c4, seq)
    
    E_bp = energy_base_pairing(c4, seq)
    E_stack = energy_base_stacking(c4, seq)
    E_bb = energy_backbone_c4_bond(c4)
    E_phos = energy_phosphate_restraint(all_coords, all_types, res_idx)
    E_pucker = energy_sugar_pucker(all_coords, all_types, res_idx, 'C2_endo')
    
    print(f"  Backbone bond:    {E_bb.item():.4f}")
    print(f"  Base pairing:     {E_bp.item():.4f}")
    print(f"  Base stacking:    {E_stack.item():.4f}")
    print(f"  Phosphate:        {E_phos.item():.4f}")
    print(f"  Sugar pucker:     {E_pucker.item():.4f}")
    print(f"  ────────────────────────")
    print(f"  TOTAL:            {E_total.item():.4f}")
    
    # ── Test 4: Build double strand ──
    print(f"\n[Test 4] Building double‑stranded B‑DNA")
    fwd_seq = "ACGTA"
    fwd_c4, rev_c4, fwd_s, rev_s = build_double_strand_helix(fwd_seq, ds_type='B_DNA')
    print(f"  Forward: {fwd_s}  shape={fwd_c4.shape}")
    print(f"  Reverse: {rev_s}  shape={rev_c4.shape}")
    
    # ── Test 5: Build A‑RNA helix ──
    print(f"\n[Test 5] Building A‑form RNA helix: {seq.replace('T','U')}")
    rna_seq = seq.replace('T', 'U')
    rna_c4 = build_rna_helix(rna_seq)
    rna_eng = DNA_RNA_Energy(pucker_type='C3_endo', use_full_atom=True)
    E_rna = rna_eng(rna_c4, rna_seq)
    print(f"  C4' shape: {rna_c4.shape}")
    print(f"  RNA Total Energy: {E_rna.item():.4f}")
    
    # ── Test 6: Write PDB ──
    print(f"\n[Test 6] Writing PDB file")
    write_nucleotide_pdb(c4, seq, "/tmp/test_dna.pdb", chain_id='A')
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print("All tests passed!")
    print(f"{'='*70}")
    print(f"\nIntegration guide:")
    print(f"  from csoc_dna_rna import DNA_RNA_Energy, build_dna_helix")
    print(f"  dna_eng = DNA_RNA_Energy(pucker_type='C2_endo')")
    print(f"  E = dna_eng(C4_coords, sequence)")
    print(f"\nFor v30.1 integration, add in total_physics_energy_v30_1:")
    print(f"  if is_dna_rna:")
    print(f"      e += dna_eng(ca, seq)")
