#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v30.2.1 — DNA/RNA Full Extension Module (standalone)
# =============================================================================
# Can Use With CSOC‑SSC v30.1.1.1.1 by import 
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

WC_PAIRS = {
    ('A', 'T'): 2, ('T', 'A'): 2,
    ('A', 'U'): 2, ('U', 'A'): 2,
    ('G', 'C'): 3, ('C', 'G'): 3,
    ('G', 'U'): 1, ('U', 'G'): 1,
}

BASE_STACKING = {'A': 1.0, 'T': 0.8, 'U': 0.8, 'G': 1.2, 'C': 1.0}

# ─── Phosphate‑Sugar Backbone ───
NUCLEOTIDE_BACKBONE = [
    ('C4\'', 'C',   0, 0.00,   0.0, ( 0, 0, 0),   0.0),
    ('O4\'', 'O',   0, 1.44, 109.5, ( 0, 0, 0),   0.0),
    ('C1\'', 'C',   1, 1.42, 109.5, ( 0,-1, 0),   0.0),
    ('C2\'', 'C',   0, 1.52, 109.5, (-1,-2, 0), 120.0),
    ('C3\'', 'C',   0, 1.52, 109.5, (-1,-2, 0),-120.0),
    ('O3\'', 'O',   4, 1.43, 109.5, (-1, 0, 1),   0.0),
    ('C5\'', 'C',   0, 1.51, 109.5, (-2,-3, 0), 180.0),
    ('O5\'', 'O',   6, 1.42, 109.5, (-1, 0, 1),   0.0),
    ('P',    'P',   7, 1.60, 119.0, (-1, 0, 1), 180.0),
    ('OP1',  'O',   8, 1.48, 109.5, (-1, 0, 1),   0.0),
    ('OP2',  'O',   8, 1.48, 109.5, (-1, 0, 1), 180.0),
]

PYRIMIDINE_BASE = [
    ('N1',  'N',   2, 1.47, 109.5, (-1, 0, 1),   0.0),
    ('C2',  'C',  11, 1.40, 121.0, (-1, 0, 1),   0.0),
    ('O2',  'O',  12, 1.24, 118.0, (-1, 0, 1),   0.0),
    ('N3',  'N',  12, 1.35, 120.0, (-1, 0, 1), 180.0),
    ('C4',  'C',  14, 1.33, 118.0, (-1, 0, 1),   0.0),
    ('C5',  'C',  15, 1.43, 118.0, (-1, 0, 1),   0.0),
    ('C6',  'C',  16, 1.34, 122.0, (-1, 0, 1),   0.0),
]

CYTOSINE_EXTRA = [('N4', 'N', 15, 1.33, 118.0, (-1, 0, 1), 180.0)]
URACIL_EXTRA = [('O4', 'O', 15, 1.23, 118.0, (-1, 0, 1), 180.0)]
THYMINE_EXTRA = [
    ('O4', 'O', 15, 1.23, 118.0, (-1, 0, 1), 180.0),
    ('C7', 'C', 16, 1.50, 122.0, (-1, 0, 1),   0.0),
]

PURINE_BASE = [
    ('N9',  'N',   2, 1.46, 109.5, (-1, 0, 1),   0.0),
    ('C4',  'C',  11, 1.37, 126.0, (-1, 0, 1),   0.0),
    ('C5',  'C',  12, 1.39, 106.0, (-1, 0, 1),   0.0),
    ('C6',  'C',  13, 1.40, 110.0, (-1, 0, 1),   0.0),
    ('N1',  'N',  14, 1.34, 118.0, (-1, 0, 1),   0.0),
    ('C2',  'C',  15, 1.32, 129.0, (-1, 0, 1),   0.0),
    ('N3',  'N',  16, 1.32, 110.0, (-1, 0, 1),   0.0),
    ('N7',  'N',  13, 1.33, 114.0, (-2,-1, 0), 180.0),
    ('C8',  'C',  18, 1.37, 106.0, (-1, 0, 1),   0.0),
]

ADENINE_EXTRA = [('N6', 'N', 14, 1.34, 124.0, (-1, 0, 1), 180.0)]
GUANINE_EXTRA = [
    ('O6', 'O', 14, 1.23, 124.0, (-1, 0, 1), 180.0),
    ('N2', 'N', 16, 1.34, 120.0, (-1, 0, 1), 180.0),
]

NUCLEOTIDE_TOPOLOGY = {
    'A': NUCLEOTIDE_BACKBONE + PURINE_BASE + ADENINE_EXTRA,
    'G': NUCLEOTIDE_BACKBONE + PURINE_BASE + GUANINE_EXTRA,
    'C': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + CYTOSINE_EXTRA,
    'U': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + URACIL_EXTRA,
    'T': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + THYMINE_EXTRA,
}

def get_atom_type_for_topology(atom_name):
    if atom_name.startswith('C'): return 'C'
    if atom_name.startswith('N'): return 'N'
    if atom_name.startswith('O'): return 'O'
    if atom_name.startswith('P'): return 'P'
    if atom_name.startswith('S'): return 'S'
    return 'C'

# ═══════════════════════════════════════════════════════════════
# FORCE FIELD
# ═══════════════════════════════════════════════════════════════
NUCLEOTIDE_LJ = {
    'P': (2.1000, 0.2000),
    'O': (1.6612, 0.2100),
    'N': (1.8240, 0.1700),
    'C': (1.9080, 0.0860),
    'S': (2.0000, 0.2500),
}

NUCLEOTIDE_CHARGES = {
    'P': 0.90, 'OP1': -0.70, 'OP2': -0.70, 'O5\'': -0.50, 'C5\'': -0.10,
    'C4\'': 0.00, 'O4\'': -0.40, 'C1\'': 0.20, 'C2\'': -0.10, 'C3\'': 0.10, 'O3\'': -0.50,
    'N1': -0.50, 'N2': -0.80, 'N3': -0.60, 'N4': -0.80, 'N6': -0.80, 'N7': -0.50, 'N9': -0.30,
    'C2': 0.40, 'C4': 0.30, 'C5': 0.10, 'C6': 0.10, 'C7': -0.20, 'C8': 0.20,
    'O2': -0.55, 'O4': -0.55, 'O6': -0.55,
}

def load_nucleotide_forcefield(lj_file=None, charge_file=None):
    lj = NUCLEOTIDE_LJ.copy()
    charges = NUCLEOTIDE_CHARGES.copy()
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
def build_single_nucleotide_ic(C4_prime, prev_C4, next_C4, nt_type):
    device = C4_prime.device
    topo = NUCLEOTIDE_TOPOLOGY.get(nt_type, [])
    if not topo:
        return torch.zeros((0, 3), device=device), []

    coords, types = [], []
    # Local frame
    if prev_C4 is not None and next_C4 is not None:
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

    for atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral0 in topo:
        if parent_idx == 0 and len(coords) == 0:
            pos = C4_prime
        elif parent_idx < len(coords):
            parent_pos = coords[parent_idx]
            a_idx, b_idx, c_idx = ref_tuple
            def map_idx(idx, current_len):
                if idx == 0: return parent_idx
                elif idx > 0: return min(idx - 1, current_len - 1) if current_len > 0 else 0
                else: return max(0, current_len + idx)
            a_abs = map_idx(a_idx, len(coords))
            b_abs = map_idx(b_idx, len(coords))
            c_abs = map_idx(c_idx, len(coords))
            p_a = coords[a_abs] if a_abs < len(coords) else parent_pos
            p_b = coords[b_abs] if b_abs < len(coords) else parent_pos
            p_c = coords[c_abs] if c_abs < len(coords) else parent_pos

            bc = p_c - p_b
            bc_norm = F.normalize(bc, dim=-1, eps=1e-8)
            ref_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
            dot = torch.abs(torch.dot(bc_norm, ref_vec))
            if dot > 0.9: ref_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            if torch.dot(perp, perp) < 1e-12:
                ref_vec = torch.tensor([0.0, 0.0, 1.0], device=device)
                perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-8)

            total_angle = math.radians(dihedral0)
            cos_a, sin_a = math.cos(total_angle), math.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            ang = math.radians(bond_ang_deg)
            bond_dir = math.cos(ang) * bc_norm + math.sin(ang) * rotated_perp
            pos = p_c + bond_len * bond_dir
        else:
            pos = coords[-1] + bond_len * x_axis if coords else C4_prime + bond_len * x_axis
        coords.append(pos)
        types.append(atom_name)
    return torch.stack(coords, dim=0), types

def build_full_dna_rna(C4_coords, sequence):
    L = len(sequence)
    device = C4_coords.device
    all_coords_list, all_types_list, all_res_idx_list = [], [], []
    for i in range(L):
        prev = C4_coords[i-1] if i>0 else None
        next_ = C4_coords[i+1] if i<L-1 else None
        nuc_coords, nuc_types = build_single_nucleotide_ic(C4_coords[i], prev, next_, sequence[i])
        all_coords_list.append(nuc_coords)
        all_types_list.extend(nuc_types)
        all_res_idx_list.append(torch.full((nuc_coords.shape[0],), i, dtype=torch.long, device=device))
    if not all_coords_list:
        return torch.zeros((0,3), device=device), [], torch.zeros(0, device=device)
    return torch.cat(all_coords_list, dim=0), all_types_list, torch.cat(all_res_idx_list, dim=0)

# ═══════════════════════════════════════════════════════════════
# ENERGY FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def energy_backbone_c4_bond(C4_coords, w=30.0, ideal_d=6.5):
    if len(C4_coords) < 2: return torch.tensor(0.0, device=C4_coords.device)
    d = torch.norm(C4_coords[1:] - C4_coords[:-1], dim=-1)
    return w * ((d - ideal_d) ** 2).mean()

def energy_phosphate_restraint(all_coords, all_types, res_indices, w=15.0, ideal_d=6.0):
    device = all_coords.device
    L = int(res_indices.max().item()) + 1 if res_indices.numel() > 0 else 0
    if L < 2: return torch.tensor(0.0, device=device)
    is_P = torch.tensor([t == 'P' for t in all_types], device=device)
    P_idx = torch.where(is_P)[0]
    P_res = res_indices[is_P]
    energy = torch.tensor(0.0, device=device)
    count = 0
    for i in range(L-1):
        mask_i = P_res == i; mask_j = P_res == i+1
        if mask_i.any() and mask_j.any():
            d = torch.norm(all_coords[P_idx[mask_i][0]] - all_coords[P_idx[mask_j][0]])
            energy += (d - ideal_d)**2
            count += 1
    return w * energy / max(1, count)

def compute_dihedral(p0, p1, p2, p3):
    b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - torch.dot(b0, b1n) * b1n
    w = b2 - torch.dot(b2, b1n) * b1n
    x = torch.dot(v, w); y = torch.dot(torch.cross(b1n, v, dim=-1), w)
    return torch.atan2(y+1e-8, x+1e-8)

def energy_sugar_pucker(all_coords, all_types, res_indices, pucker_type='C2_endo', w=10.0):
    device = all_coords.device
    L = int(res_indices.max().item())+1 if res_indices.numel()>0 else 0
    target = math.radians(36.0) if pucker_type == 'C2_endo' else math.radians(15.0)
    energy = torch.tensor(0.0, device=device)
    count = 0
    for i in range(L):
        mask = res_indices == i
        res_coords = all_coords[mask]
        res_types_i = [all_types[j] for j in range(len(all_types)) if res_indices[j]==i]
        pos = {}
        for name in ['C1\'', 'C2\'', 'C3\'', 'C4\'']:
            if name in res_types_i:
                pos[name] = res_coords[res_types_i.index(name)]
        if len(pos)==4:
            nu2 = compute_dihedral(pos['C1\''], pos['C2\''], pos['C3\''], pos['C4\''])
            diff = torch.atan2(torch.sin(nu2-target), torch.cos(nu2-target))
            energy += diff**2
            count += 1
    return w * energy / max(1, count)

def energy_base_pairing(C4_coords, sequence, w=8.0, ideal_d=10.5, sigma=2.0):
    L = len(sequence); device = C4_coords.device
    energy = torch.tensor(0.0, device=device)
    for i in range(L):
        for j in range(i+4, min(L, i+50)):
            d = torch.norm(C4_coords[i]-C4_coords[j])
            n = WC_PAIRS.get((sequence[i], sequence[j]), 0)
            if n>0: energy += -n * torch.exp(-((d-ideal_d)/sigma)**2)
    return w * energy / max(1, L)

def energy_base_stacking(C4_coords, sequence, w=5.0, ideal_d=6.5, sigma=1.5):
    L = len(sequence); device = C4_coords.device
    energy = torch.tensor(0.0, device=device)
    for i in range(L-1):
        d = torch.norm(C4_coords[i+1]-C4_coords[i])
        s = 0.5*(BASE_STACKING.get(sequence[i],1.0)+BASE_STACKING.get(sequence[i+1],1.0))
        energy += -s * torch.exp(-((d-ideal_d)/sigma)**2)
    return w * energy / max(1, L-1)

def energy_dna_rna_lj(all_coords, all_types, edge_index, edge_dist, lj_params=None, w=30.0):
    if edge_index is None or edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    if lj_params is None: lj_params = NUCLEOTIDE_LJ
    src, dst = edge_index
    elem = [get_atom_type_for_topology(t) for t in all_types]
    sigmas = torch.tensor([lj_params.get(e,(1.9,0.1))[0] for e in elem], device=all_coords.device)
    epsilons = torch.tensor([lj_params.get(e,(1.9,0.1))[1] for e in elem], device=all_coords.device)
    sigma_ij = 0.5*(sigmas[src]+sigmas[dst])
    eps_ij = torch.sqrt(epsilons[src]*epsilons[dst])
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0/r
    lj_energy = 4.0*eps_ij*((sigma_ij*inv_r)**12 - (sigma_ij*inv_r)**6)
    return w * lj_energy.mean()

def energy_dna_rna_coulomb(all_coords, all_types, res_indices, edge_index, edge_dist, charge_map=None, w=3.0):
    if edge_index is None or edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    if charge_map is None: charge_map = NUCLEOTIDE_CHARGES
    src, dst = edge_index
    q = torch.tensor([charge_map.get(t,0.0) for t in all_types], device=all_coords.device)
    qi, qj = q[src], q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r
    coulomb = 332.0637 * qi * qj / (dielectric * r + 1e-8)
    return w * coulomb.mean()

# ═══════════════════════════════════════════════════════════════
# MAIN ENERGY CLASS
# ═══════════════════════════════════════════════════════════════
class DNA_RNA_Energy:
    def __init__(self, pucker_type='C2_endo', w_c4_bond=30.0, w_phosphate=15.0, w_pucker=10.0,
                 w_base_pair=8.0, w_stacking=5.0, w_lj=30.0, w_coulomb=3.0,
                 lj_params=None, charge_map=None, use_full_atom=True):
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

    def __call__(self, C4_coords, sequence, edge_index=None, edge_dist=None):
        device = C4_coords.device
        E = torch.tensor(0.0, device=device)
        E += energy_backbone_c4_bond(C4_coords, self.w_c4_bond)
        E += energy_base_pairing(C4_coords, sequence, self.w_base_pair)
        E += energy_base_stacking(C4_coords, sequence, self.w_stacking)
        if self.use_full_atom:
            all_coords, all_types, res_indices = build_full_dna_rna(C4_coords, sequence)
            if all_coords.shape[0] > 1:
                E += energy_phosphate_restraint(all_coords, all_types, res_indices, self.w_phosphate)
                E += energy_sugar_pucker(all_coords, all_types, res_indices, self.pucker_type, self.w_pucker)
                if edge_index is not None and edge_index.numel() > 0:
                    E += energy_dna_rna_lj(all_coords, all_types, edge_index, edge_dist, self.lj_params, self.w_lj)
                    E += energy_dna_rna_coulomb(all_coords, all_types, res_indices, edge_index, edge_dist, self.charge_map, self.w_coulomb)
        return E

# ═══════════════════════════════════════════════════════════════
# HELIX BUILDERS (for initial structure)
# ═══════════════════════════════════════════════════════════════
def build_dna_helix(sequence, rise=3.38, twist=36.0, radius=8.0, start_angle=0.0):
    L = len(sequence)
    coords = torch.zeros(L, 3)
    for i in range(L):
        angle = start_angle + math.radians(i * twist)
        coords[i,0] = radius * math.cos(angle)
        coords[i,1] = radius * math.sin(angle)
        coords[i,2] = i * rise
    return coords

def build_rna_helix(sequence, rise=2.80, twist=32.7, radius=9.0, start_angle=0.0):
    return build_dna_helix(sequence, rise=rise, twist=twist, radius=radius, start_angle=start_angle)

def build_double_strand_helix(seq_fwd, seq_rev=None, ds_type='B_DNA'):
    if ds_type == 'B_DNA':
        rise, twist, radius = 3.38, 36.0, 8.0
        comp = {'A':'T','T':'A','G':'C','C':'G'}
    elif ds_type == 'A_RNA':
        rise, twist, radius = 2.80, 32.7, 9.0
        comp = {'A':'U','U':'A','G':'C','C':'G'}
    else:
        raise ValueError(f"Unknown ds_type {ds_type}")
    if seq_rev is None:
        seq_rev = ''.join(comp.get(b,'X') for b in reversed(seq_fwd))
    fwd = build_dna_helix(seq_fwd, rise=rise, twist=twist, radius=radius, start_angle=0.0)
    rev = build_dna_helix(seq_rev, rise=rise, twist=twist, radius=radius, start_angle=math.pi)
    rev[:,2] = (len(seq_fwd)-1)*rise - rev[:,2]
    return fwd, rev, seq_fwd, seq_rev

# ═══════════════════════════════════════════════════════════════
# PDB LOADER
# ═══════════════════════════════════════════════════════════════
def load_nucleotide_pdb(pdb_path, chain='A'):
    import gzip
    nt_3_to_1 = {'DA':'A','DC':'C','DG':'G','DT':'T','DU':'U',
                 'A':'A','C':'C','G':'G','T':'T','U':'U',
                 'ADE':'A','CYT':'C','GUA':'G','THY':'T','URA':'U'}
    coords, seq = [], []
    seen = set()
    opener = gzip.open if pdb_path.endswith('.gz') else open
    with opener(pdb_path, 'rt', errors='ignore') as f:
        for line in f:
            if not line.startswith(('ATOM','HETATM')): continue
            if line[21].strip() != chain: continue
            atom = line[12:16].strip()
            if atom not in ("C4'", "C4*", "C4'"): continue
            res = line[17:20].strip()
            key = (line[22:26].strip(), line[26].strip() if len(line)>26 else '')
            if key in seen: continue
            seen.add(key)
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords.append([x,y,z])
            seq.append(nt_3_to_1.get(res, 'X'))
    if not coords: return torch.empty((0,3)), ""
    return torch.tensor(coords, dtype=torch.float32), "".join(seq)

def load_double_strand_pdb(pdb_path, chain_fwd='A', chain_rev='B'):
    fwd_c, fwd_s = load_nucleotide_pdb(pdb_path, chain_fwd)
    rev_c, rev_s = load_nucleotide_pdb(pdb_path, chain_rev)
    return fwd_c, rev_c, fwd_s, rev_s

# ═══════════════════════════════════════════════════════════════
# PDB WRITER (full atom)
# ═══════════════════════════════════════════════════════════════
def write_nucleotide_pdb(C4_coords, sequence, filename, chain_id='A', pucker_type='C2_endo'):
    all_coords, all_types, res_indices = build_full_dna_rna(C4_coords, sequence)
    nt_1_to_3 = {'A':'  A','G':'  G','C':'  C','T':' DT','U':'  U'}
    with open(filename, 'w') as f:
        serial = 1
        for i, nt in enumerate(sequence):
            mask = res_indices == i
            c = all_coords[mask]
            t = [all_types[j] for j, m in enumerate(mask) if m]
            for k, (at, pos) in enumerate(zip(t, c)):
                elem = at[0] if at[0] in 'CNOPS' else 'C'
                f.write(f"ATOM  {serial:5d} {at:<4s} {nt_1_to_3.get(nt,'  X')} "
                        f"{chain_id}{i+1:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                        f"  1.00  0.00          {elem:>2s}\n")
                serial += 1
        f.write("END\n")
    print(f"DNA/RNA PDB written to {filename} ({serial-1} atoms)")
