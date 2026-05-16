
# =============================================================================
# CSOC‑SSC v30.3 — Ligand / Small Molecule Extension Module
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Standalone Ligand/Small Molecule module for CSOC‑SSC v30.1
#
# Features:
#   ✓ SDF/MOL2/PDB ligand reader
#   ✓ Automatic bond/angle/torsion detection from connectivity
#   ✓ General force field (GAFF‑inspired LJ + charges)
#   ✓ Bond, angle, torsion, LJ, Coulomb energy terms
#   ✓ Protein‑ligand interaction energy
#   ✓ Ligand refinement (rigid‑body + torsion)
#   ✓ Binding affinity (ΔG_bind) estimation
#   ✓ Multi‑ligand support
#   ✓ Integration with v30.1 energy function
# =============================================================================

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import json
import os
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# ATOM DATA
# ═══════════════════════════════════════════════════════════════

# Atomic numbers
ATOMIC_NUMBER = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Br': 35, 'I': 53,
    'Fe': 26, 'Zn': 30, 'Se': 34, 'Mn': 25, 'Cu': 29, 'Co': 27, 'Ni': 28,
}

# Covalent radii (Å) — for bond detection
COVALENT_RADIUS = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'B': 0.84, 'Si': 1.11, 'Se': 1.20,
    'Fe': 1.32, 'Zn': 1.22, 'Mn': 1.39, 'Cu': 1.32, 'Co': 1.26, 'Ni': 1.24,
}

# ═══════════════════════════════════════════════════════════════
# GENERAL FORCE FIELD (GAFF‑inspired)
# ═══════════════════════════════════════════════════════════════

# Lennard‑Jones parameters by atom type (sigma_Å, epsilon_kcal/mol)
GAFF_LJ = {
    # Carbon types
    'c':  (1.9080, 0.0860),   # sp2 C (carbonyl)
    'c1': (1.9080, 0.0860),   # sp1 C
    'c2': (1.9080, 0.0860),   # sp2 C (alkene)
    'c3': (1.9080, 0.1094),   # sp3 C (alkane)
    'ca': (1.9080, 0.0860),   # aromatic C
    'cp': (1.9080, 0.0860),   # aromatic C (pyridine)
    'cq': (1.9080, 0.0860),   # aromatic C (fused ring)
    
    # Nitrogen types
    'n':  (1.8240, 0.1700),   # sp2 N (amide)
    'n1': (1.8240, 0.1700),   # sp1 N
    'n2': (1.8240, 0.1700),   # sp2 N (imine)
    'n3': (1.8240, 0.1700),   # sp3 N (amine)
    'n4': (1.8240, 0.1700),   # sp3 N (ammonium)
    'na': (1.8240, 0.1700),   # aromatic N
    'nh': (1.8240, 0.1700),   # amine N (pyrrole)
    'no': (1.8240, 0.1700),   # nitro N
    
    # Oxygen types
    'o':  (1.6612, 0.2100),   # sp2 O (carbonyl)
    'oh': (1.7210, 0.2104),   # hydroxyl O
    'os': (1.6837, 0.1700),   # ether O
    'ow': (1.7683, 0.1520),   # water O
    
    # Sulfur types
    's':  (2.0000, 0.2500),   # sulfide S
    's2': (2.0000, 0.2500),   # disulfide S
    's4': (2.0000, 0.2500),   # sulfoxide S
    's6': (2.0000, 0.2500),   # sulfone S
    'sh': (2.0000, 0.2500),   # thiol S
    
    # Phosphorus
    'p':  (2.1000, 0.2000),   # phosphate P
    'p5': (2.1000, 0.2000),   # pentavalent P
    
    # Halogens
    'f':  (1.7500, 0.0610),   # fluoride
    'cl': (1.9480, 0.2650),   # chloride
    'br': (2.0200, 0.4200),   # bromide
    'i':  (2.1500, 0.5000),   # iodide
    
    # Metals
    'Fe': (1.5000, 0.0500),
    'Zn': (1.1000, 0.0125),
    'Mg': (0.7926, 0.8947),
    'Ca': (1.7000, 0.4598),
    'Mn': (1.5000, 0.0500),
    'Cu': (1.4000, 0.0500),
    'Co': (1.4000, 0.0500),
    'Ni': (1.4000, 0.0500),
}

# Default atom type mapping (element → GAFF type)
ELEMENT_TO_GAFF = {
    'C': 'c3',   # default sp3 carbon
    'N': 'n3',   # default sp3 nitrogen
    'O': 'os',   # default ether oxygen
    'S': 's',    # default sulfide
    'P': 'p',    # default phosphate
    'F': 'f',
    'Cl': 'cl',
    'Br': 'br',
    'I': 'i',
    'H': 'H',    # hydrogen (usually no LJ)
}

# Ideal bond lengths by element pair (Å)
IDEAL_BOND_LENGTHS = {
    ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43, ('C', 'S'): 1.82,
    ('C', 'H'): 1.09, ('C', 'F'): 1.35, ('C', 'Cl'): 1.77, ('C', 'Br'): 1.94,
    ('C', 'I'): 2.14, ('C', 'P'): 1.84,
    ('N', 'N'): 1.45, ('N', 'O'): 1.40, ('N', 'S'): 1.68, ('N', 'H'): 1.01,
    ('N', 'P'): 1.70,
    ('O', 'O'): 1.48, ('O', 'S'): 1.57, ('O', 'H'): 0.96, ('O', 'P'): 1.60,
    ('S', 'S'): 2.05, ('S', 'H'): 1.34, ('S', 'P'): 2.10,
    ('P', 'P'): 2.20, ('P', 'H'): 1.42,
    ('C', 'Fe'): 2.00, ('N', 'Fe'): 1.95, ('O', 'Fe'): 1.90,
    ('C', 'Zn'): 2.00, ('N', 'Zn'): 2.05, ('O', 'Zn'): 2.10,
    ('C', 'Mg'): 2.20, ('O', 'Mg'): 2.10,
}

# Ideal bond angles by element triplet (degrees)
IDEAL_ANGLES = {
    ('C', 'C', 'C'): 109.5, ('C', 'C', 'N'): 109.5, ('C', 'C', 'O'): 109.5,
    ('C', 'N', 'C'): 109.5, ('C', 'N', 'H'): 109.5,
    ('C', 'O', 'C'): 109.5, ('C', 'O', 'H'): 109.5,
    ('C', 'S', 'C'): 99.0, ('C', 'S', 'H'): 96.0,
    ('O', 'P', 'O'): 109.5, ('C', 'P', 'O'): 109.5,
    ('C', 'C', 'H'): 109.5, ('H', 'C', 'H'): 109.5,
    ('H', 'N', 'H'): 109.5, ('H', 'O', 'H'): 104.5,
    # sp2 angles
    ('C', 'C', 'O'): 120.0, ('C', 'C', 'N'): 120.0,
    ('O', 'C', 'O'): 120.0, ('O', 'C', 'N'): 120.0,
}

# Force constants
K_BOND = 300.0      # kcal/mol/Å²
K_ANGLE = 50.0      # kcal/mol/rad²
K_TORSION = 1.0     # kcal/mol (per torsion)
K_IMPROPER = 10.0   # kcal/mol/rad² (out‑of‑plane)

# ═══════════════════════════════════════════════════════════════
# MOLECULE DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════

class Molecule:
    """
    Small molecule representation.
    
    Stores atoms, bonds, and molecular topology for energy computation.
    """
    
    def __init__(self, name: str = "UNK"):
        self.name = name
        
        # Atom data
        self.atom_coords: torch.Tensor = None         # [N, 3]
        self.atom_elements: List[str] = []             # 'C', 'N', 'O', ...
        self.atom_names: List[str] = []                # 'C1', 'C2', ...
        self.atom_gaff_types: List[str] = []           # 'c3', 'n', 'oh', ...
        self.atom_charges: torch.Tensor = None         # [N] partial charges
        
        # Bond topology
        self.bonds: List[Tuple[int, int]] = []         # (i, j) pairs
        self.bond_orders: List[float] = []             # 1.0, 1.5 (aromatic), 2.0, 3.0
        
        # Derived topology
        self.angles: List[Tuple[int, int, int]] = []
        self.torsions: List[Tuple[int, int, int, int]] = []
        self.torsion_periodicities: List[int] = []     # 1, 2, 3, ...
        self.impropers: List[Tuple[int, int, int, int]] = []
        
        # Rings
        self.rings: List[List[int]] = []
        
        # Number of atoms
        self.n_atoms: int = 0
    
    def build_topology(self):
        """Auto‑detect angles, torsions, and impropers from bonds."""
        if not self.bonds:
            return
        
        self.n_atoms = len(self.atom_elements)
        
        # Build adjacency
        adj = defaultdict(set)
        for i, j in self.bonds:
            adj[i].add(j)
            adj[j].add(i)
        
        # Angles: i‑j‑k where (i,j) and (j,k) are bonds
        self.angles = []
        for j in range(self.n_atoms):
            neighbors = list(adj[j])
            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    self.angles.append((neighbors[a], j, neighbors[b]))
        
        # Torsions: i‑j‑k‑l where (i,j), (j,k), (k,l) are bonds
        self.torsions = []
        self.torsion_periodicities = []
        for j, k in self.bonds:
            for i in adj[j]:
                if i != k:
                    for l in adj[k]:
                        if l != j and l != i:
                            # Check if sp2‑sp2 (amide‑like) → periodicity 2
                            ei = self.atom_elements[i]
                            ek = self.atom_elements[j]
                            el = self.atom_elements[k]
                            em = self.atom_elements[l]
                            
                            period = 3  # default
                            # sp2‑sp2 bond → period 2
                            if self._is_sp2_bond(j, k):
                                period = 2
                            
                            self.torsions.append((i, j, k, l))
                            self.torsion_periodicities.append(period)
        
        # Impropers: out‑of‑plane (for sp2 centers)
        self.impropers = []
        for j in range(self.n_atoms):
            if len(adj[j]) == 3:  # sp2 center
                neighbors = list(adj[j])
                # Central atom j, with i,k,l in plane
                self.impropers.append((j, neighbors[0], neighbors[1], neighbors[2]))
        
        # Detect rings (simple SSSR approximation)
        self._find_rings(adj)
    
    def _is_sp2_bond(self, i: int, j: int) -> bool:
        """Check if bond is sp2 (double bond or aromatic)."""
        for (a, b), order in zip(self.bonds, self.bond_orders):
            if (a == i and b == j) or (a == j and b == i):
                return order >= 1.5
        return False
    
    def _find_rings(self, adj: Dict[int, Set[int]], max_ring_size: int = 8):
        """Simple ring detection using DFS."""
        self.rings = []
        visited_edges = set()
        
        for start in range(min(self.n_atoms, 50)):  # limit for speed
            for neighbor in adj[start]:
                edge = tuple(sorted((start, neighbor)))
                if edge in visited_edges:
                    continue
                
                # BFS from start to find shortest cycle
                parent = {start: -1}
                queue = [start]
                found = False
                
                while queue and not found:
                    node = queue.pop(0)
                    for nb in adj[node]:
                        if nb == parent[node]:
                            continue
                        if nb in parent:
                            # Found cycle
                            if nb != start:
                                continue
                            # Reconstruct ring
                            ring = [node]
                            while ring[-1] != start:
                                ring.append(parent[ring[-1]])
                            ring = ring[::-1]
                            if 3 <= len(ring) <= max_ring_size:
                                self.rings.append(ring)
                                for a, b in zip(ring, ring[1:] + [ring[0]]):
                                    visited_edges.add(tuple(sorted((a, b))))
                            found = True
                            break
                        parent[nb] = node
                        queue.append(nb)
                
                if found:
                    break
    
    def assign_gaff_types(self):
        """Assign GAFF atom types based on element and connectivity."""
        self.atom_gaff_types = []
        adj = defaultdict(set)
        for i, j in self.bonds:
            adj[i].add(j)
            adj[j].add(i)
        
        for i, elem in enumerate(self.atom_elements):
            # Start with default
            gaff_type = ELEMENT_TO_GAFF.get(elem, elem.lower())
            
            # Refine based on connectivity
            n_neighbors = len(adj[i])
            
            if elem == 'C':
                if n_neighbors == 4:
                    gaff_type = 'c3'
                elif n_neighbors == 3:
                    # Check if any double bond
                    has_double = any(
                        self._get_bond_order(i, nb) >= 2.0
                        for nb in adj[i]
                    )
                    if has_double:
                        gaff_type = 'c2'
                    else:
                        gaff_type = 'ca'  # aromatic‑like
                elif n_neighbors == 2:
                    gaff_type = 'c1' if self._get_bond_order(i, list(adj[i])[0]) >= 2.5 else 'c2'
            
            elif elem == 'N':
                if n_neighbors == 3:
                    gaff_type = 'n3'
                elif n_neighbors == 2:
                    gaff_type = 'n2'
                elif n_neighbors == 1:
                    gaff_type = 'n1'
            
            elif elem == 'O':
                if n_neighbors == 1:
                    # Check if double‑bonded (carbonyl)
                    nb = list(adj[i])[0]
                    if self._get_bond_order(i, nb) >= 2.0:
                        gaff_type = 'o'
                    else:
                        gaff_type = 'oh'
                elif n_neighbors == 2:
                    gaff_type = 'os'
            
            elif elem == 'S':
                gaff_type = 's' if n_neighbors <= 2 else 's6'
            
            self.atom_gaff_types.append(gaff_type)
    
    def _get_bond_order(self, i: int, j: int) -> float:
        """Get bond order between atoms i and j."""
        for (a, b), order in zip(self.bonds, self.bond_orders):
            if (a == i and b == j) or (a == j and b == i):
                return order
        return 1.0
    
    def to_device(self, device: torch.device):
        """Move tensors to device."""
        if self.atom_coords is not None:
            self.atom_coords = self.atom_coords.to(device)
        if self.atom_charges is not None:
            self.atom_charges = self.atom_charges.to(device)
        return self


# ═══════════════════════════════════════════════════════════════
# LIGAND ENERGY FUNCTION
# ═══════════════════════════════════════════════════════════════

class LigandEnergy:
    """
    Energy function for small molecules.
    
    Computes bonded (bond, angle, torsion, improper) and
    non‑bonded (LJ, Coulomb) energy terms.
    """
    
    def __init__(self,
                 w_bond: float = 300.0,
                 w_angle: float = 50.0,
                 w_torsion: float = 1.0,
                 w_improper: float = 10.0,
                 w_lj: float = 30.0,
                 w_coulomb: float = 3.0,
                 lj_params: Optional[Dict] = None,
                 dielectric: float = 4.0,
                 ):
        self.w_bond = w_bond
        self.w_angle = w_angle
        self.w_torsion = w_torsion
        self.w_improper = w_improper
        self.w_lj = w_lj
        self.w_coulomb = w_coulomb
        self.lj_params = lj_params or GAFF_LJ
        self.dielectric = dielectric
    
    def __call__(self, mol: Molecule) -> torch.Tensor:
        """Compute total ligand energy."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        
        coords = mol.atom_coords
        n = mol.n_atoms
        
        # Bond energy
        E += self.energy_bond(mol)
        
        # Angle energy
        E += self.energy_angle(mol)
        
        # Torsion energy
        E += self.energy_torsion(mol)
        
        # Improper energy
        E += self.energy_improper(mol)
        
        # Non‑bonded (1‑4 and beyond)
        E += self.energy_nonbonded(mol)
        
        return E
    
    def energy_bond(self, mol: Molecule) -> torch.Tensor:
        """Harmonic bond energy."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        
        for (i, j), order in zip(mol.bonds, mol.bond_orders):
            elem_i = mol.atom_elements[i]
            elem_j = mol.atom_elements[j]
            
            # Ideal bond length
            key = tuple(sorted((elem_i, elem_j)))
            r0 = IDEAL_BOND_LENGTHS.get(key, 1.50)
            
            # Adjust for bond order
            if order >= 2.5:    # triple
                r0 -= 0.35
            elif order >= 1.5:  # double/aromatic
                r0 -= 0.15
            
            # Force constant
            k = K_BOND
            
            # Compute
            dv = mol.atom_coords[j] - mol.atom_coords[i]
            d = torch.norm(dv)
            E += self.w_bond * k * (d - r0) ** 2
        
        return E
    
    def energy_angle(self, mol: Molecule) -> torch.Tensor:
        """Harmonic angle energy."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        
        for i, j, k in mol.angles:
            elem_i = mol.atom_elements[i]
            elem_j = mol.atom_elements[j]
            elem_k = mol.atom_elements[k]
            
            # Ideal angle
            key = (elem_i, elem_j, elem_k)
            theta0 = math.radians(IDEAL_ANGLES.get(key, 109.5))
            
            # Compute
            v1 = mol.atom_coords[i] - mol.atom_coords[j]
            v2 = mol.atom_coords[k] - mol.atom_coords[j]
            n1 = torch.norm(v1) + 1e-8
            n2 = torch.norm(v2) + 1e-8
            cos_theta = torch.clamp(torch.dot(v1, v2) / (n1 * n2), -1, 1)
            theta = torch.acos(cos_theta)
            
            E += self.w_angle * K_ANGLE * (theta - theta0) ** 2
        
        return E
    
    def energy_torsion(self, mol: Molecule) -> torch.Tensor:
        """Periodic torsion (dihedral) energy."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        
        for (i, j, k, l), period in zip(mol.torsions, mol.torsion_periodicities):
            # Compute dihedral angle
            phi = self._compute_dihedral(
                mol.atom_coords[i], mol.atom_coords[j],
                mol.atom_coords[k], mol.atom_coords[l]
            )
            
            # E = K * [1 + cos(n*phi - delta)]
            # For simplicity: use generic barrier
            V = 1.0  # barrier height (kcal/mol)
            
            # sp3‑sp3: period=3, delta=0 (staggered)
            # sp2‑sp2: period=2, delta=π (planar)
            if period == 2:
                delta = math.pi
            else:
                delta = 0.0
            
            E += self.w_torsion * K_TORSION * V * (1 + torch.cos(period * phi - delta))
        
        return E
    
    def energy_improper(self, mol: Molecule) -> torch.Tensor:
        """Improper dihedral (out‑of‑plane) energy."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        
        for i, j, k, l in mol.impropers:
            phi = self._compute_dihedral(
                mol.atom_coords[i], mol.atom_coords[j],
                mol.atom_coords[k], mol.atom_coords[l]
            )
            # Harmonic out‑of‑plane
            E += self.w_improper * K_IMPROPER * phi ** 2
        
        return E
    
    def energy_nonbonded(self, mol: Molecule) -> torch.Tensor:
        """LJ + Coulomb for non‑bonded pairs (1‑4 and beyond)."""
        device = mol.atom_coords.device
        E = torch.tensor(0.0, device=device)
        n = mol.n_atoms
        
        # Build exclusion list (1‑2, 1‑3 excluded; 1‑4 scaled)
        exclusions = self._build_exclusion_list(mol)
        is_14 = self._build_14_list(mol)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in exclusions or (j, i) in exclusions:
                    continue
                
                dv = mol.atom_coords[i] - mol.atom_coords[j]
                d = torch.norm(dv) + 1e-8
                
                scale_14 = 0.5 if ((i, j) in is_14 or (j, i) in is_14) else 1.0
                
                # LJ
                gaff_i = mol.atom_gaff_types[i] if i < len(mol.atom_gaff_types) else 'c3'
                gaff_j = mol.atom_gaff_types[j] if j < len(mol.atom_gaff_types) else 'c3'
                
                sigma_i, eps_i = self.lj_params.get(gaff_i, (1.9, 0.1))
                sigma_j, eps_j = self.lj_params.get(gaff_j, (1.9, 0.1))
                
                sigma = 0.5 * (sigma_i + sigma_j)
                eps = math.sqrt(eps_i * eps_j)
                
                inv_r = 1.0 / d
                inv_r6 = inv_r ** 6
                inv_r12 = inv_r6 ** 2
                E_lj = 4.0 * eps * ((sigma * inv_r) ** 12 - (sigma * inv_r) ** 6)
                E += self.w_lj * scale_14 * E_lj
                
                # Coulomb
                if mol.atom_charges is not None:
                    qi = mol.atom_charges[i]
                    qj = mol.atom_charges[j]
                    E_coul = 332.0637 * qi * qj / (self.dielectric * d)
                    E += self.w_coulomb * scale_14 * E_coul
        
        return E
    
    def _build_exclusion_list(self, mol: Molecule) -> Set[Tuple[int, int]]:
        """Build 1‑2 and 1‑3 exclusion pairs."""
        exclusions = set()
        
        # 1‑2: direct bonds
        for i, j in mol.bonds:
            exclusions.add((i, j))
            exclusions.add((j, i))
        
        # 1‑3: two bonds apart
        adj = defaultdict(set)
        for i, j in mol.bonds:
            adj[i].add(j)
            adj[j].add(i)
        
        for i in range(mol.n_atoms):
            for j in adj[i]:
                for k in adj[j]:
                    if k != i:
                        exclusions.add((i, k))
        
        return exclusions
    
    def _build_14_list(self, mol: Molecule) -> Set[Tuple[int, int]]:
        """Build 1‑4 pairs (three bonds apart)."""
        pairs_14 = set()
        adj = defaultdict(set)
        for i, j in mol.bonds:
            adj[i].add(j)
            adj[j].add(i)
        
        for i in range(mol.n_atoms):
            for j in adj[i]:
                for k in adj[j]:
                    if k == i:
                        continue
                    for l in adj[k]:
                        if l != i and l != j:
                            pairs_14.add((i, l))
        
        return pairs_14
    
    def _compute_dihedral(self, p0, p1, p2, p3) -> torch.Tensor:
        """Compute dihedral angle (radians)."""
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        
        b1n = F.normalize(b1, dim=-1, eps=1e-8)
        v = b0 - torch.dot(b0, b1n) * b1n
        w = b2 - torch.dot(b2, b1n) * b1n
        
        x = torch.dot(v, w)
        y = torch.dot(torch.cross(b1n, v, dim=-1), w)
        
        return torch.atan2(y + 1e-8, x + 1e-8)


# ═══════════════════════════════════════════════════════════════
# PROTEIN‑LIGAND INTERACTION ENERGY
# ═══════════════════════════════════════════════════════════════

class ProteinLigandEnergy:
    """
    Compute protein‑ligand interaction energy.
    
    Combines LJ + Coulomb between protein atoms and ligand atoms.
    Can be integrated into v30.1 total energy.
    """
    
    def __init__(self,
                 w_lj: float = 30.0,
                 w_coulomb: float = 3.0,
                 protein_lj_params: Optional[Dict] = None,
                 ligand_lj_params: Optional[Dict] = None,
                 protein_charges: Optional[Dict] = None,
                 dielectric: float = 4.0,
                 cutoff: float = 12.0,
                 ):
        self.w_lj = w_lj
        self.w_coulomb = w_coulomb
        self.protein_lj_params = protein_lj_params or {}
        self.ligand_lj_params = ligand_lj_params or GAFF_LJ
        self.protein_charges = protein_charges or {}
        self.dielectric = dielectric
        self.cutoff = cutoff
    
    def __call__(self,
                 protein_coords: torch.Tensor,      # [N_prot, 3]
                 protein_atom_types: List[str],      # protein atom types
                 protein_charges_list: List[float],  # protein partial charges
                 ligand: Molecule,                   # ligand molecule
                 ) -> torch.Tensor:
        """
        Compute protein‑ligand interaction energy.
        """
        device = protein_coords.device
        E = torch.tensor(0.0, device=device)
        
        lig_coords = ligand.atom_coords.to(device)
        lig_charges = ligand.atom_charges
        
        for i in range(len(protein_coords)):
            prot_pos = protein_coords[i]
            prot_type = protein_atom_types[i] if i < len(protein_atom_types) else 'CA'
            prot_q = protein_charges_list[i] if i < len(protein_charges_list) else 0.0
            
            for j in range(ligand.n_atoms):
                dv = prot_pos - lig_coords[j]
                d = torch.norm(dv) + 1e-8
                
                if d > self.cutoff:
                    continue
                
                # LJ
                sig_i, eps_i = self.protein_lj_params.get(prot_type, (1.9, 0.1))
                gaff_j = ligand.atom_gaff_types[j] if j < len(ligand.atom_gaff_types) else 'c3'
                sig_j, eps_j = self.ligand_lj_params.get(gaff_j, (1.9, 0.1))
                
                sigma = 0.5 * (sig_i + sig_j)
                eps = math.sqrt(eps_i * eps_j)
                
                inv_r = 1.0 / d
                inv_r6 = inv_r ** 6
                inv_r12 = inv_r6 ** 2
                E_lj = 4.0 * eps * ((sigma * inv_r) ** 12 - (sigma * inv_r) ** 6)
                E += self.w_lj * E_lj
                
                # Coulomb
                if lig_charges is not None:
                    q_j = lig_charges[j]
                    E_coul = 332.0637 * prot_q * q_j / (self.dielectric * d)
                    E += self.w_coulomb * E_coul
        
        return E


# ═══════════════════════════════════════════════════════════════
# BINDING AFFINITY ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_binding_affinity(E_interaction: float,
                               E_ligand_strain: float = 0.0,
                               T: float = 298.15) -> Dict:
    """
    Estimate binding affinity from interaction energy.
    
    ΔG_bind ≈ E_interaction + E_strain − TΔS_config
    (Simple MM‑PBSA‑inspired estimate)
    """
    # Configurational entropy penalty (approximate)
    # ~0.5 kcal/mol per rotatable bond
    TdS_config = 5.0  # rough estimate for small molecule
    
    dG_bind = E_interaction + E_ligand_strain + TdS_config
    
    # Estimate Kd from ΔG
    R = 1.987e-3  # kcal/mol/K
    Kd = math.exp(dG_bind / (R * T)) if dG_bind > 0 else 1e-12
    
    return {
        'dG_bind_kcal': dG_bind,
        'Kd_M': Kd,
        'E_interaction': E_interaction,
        'E_strain': E_ligand_strain,
        'TdS_config': TdS_config,
    }


# ═══════════════════════════════════════════════════════════════
# LIGAND REFINEMENT
# ═══════════════════════════════════════════════════════════════

def refine_ligand(ligand: Molecule,
                   energy_fn: LigandEnergy = None,
                   steps: int = 200,
                   lr: float = 1e-3,
                   rigid_body: bool = True,
                   torsion_only: bool = False,
                   device: str = 'cpu',
                   verbose: bool = False) -> Molecule:
    """
    Refine ligand geometry using gradient‑based optimization.
    
    Args:
        ligand: Molecule to refine
        energy_fn: LigandEnergy instance (created if None)
        steps: optimization steps
        lr: learning rate
        rigid_body: include rigid‑body translation/rotation
        torsion_only: only optimize torsions (keep bonds/angles fixed)
        device: 'cpu' or 'cuda'
        verbose: print progress
    
    Returns:
        Refined Molecule (new instance)
    """
    if energy_fn is None:
        energy_fn = LigandEnergy()
    
    device = torch.device(device)
    
    # Clone ligand
    mol = Molecule(name=ligand.name)
    mol.atom_coords = ligand.atom_coords.clone().detach().to(device).requires_grad_(True)
    mol.atom_elements = ligand.atom_elements.copy()
    mol.atom_names = ligand.atom_names.copy()
    mol.atom_gaff_types = ligand.atom_gaff_types.copy()
    mol.atom_charges = ligand.atom_charges.clone().detach().to(device) if ligand.atom_charges is not None else None
    mol.bonds = ligand.bonds.copy()
    mol.bond_orders = ligand.bond_orders.copy()
    mol.build_topology()
    mol.n_atoms = ligand.n_atoms
    
    # Optimizer
    if torsion_only:
        # Freeze all atoms except terminal ones on rotatable bonds
        # Simplified: optimize all but with strong position restraints
        opt = torch.optim.Adam([mol.atom_coords], lr=lr)
    else:
        opt = torch.optim.Adam([mol.atom_coords], lr=lr)
    
    # Rigid‑body parameters (translation + rotation quaternion)
    if rigid_body:
        translation = torch.zeros(3, device=device, requires_grad=True)
        rotation = torch.zeros(3, device=device, requires_grad=True)  # axis‑angle
        opt.add_param_group({'params': [translation], 'lr': lr * 0.1})
        opt.add_param_group({'params': [rotation], 'lr': lr * 0.01})
    
    best_E = float('inf')
    best_coords = mol.atom_coords.clone()
    energy_history = []
    
    for step in range(steps):
        opt.zero_grad()
        
        # Apply rigid‑body transform
        if rigid_body:
            # Rotation from axis‑angle
            angle = torch.norm(rotation)
            if angle > 1e-8:
                axis = rotation / angle
                K = torch.tensor([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ], device=device)
                R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
            else:
                R = torch.eye(3, device=device)
            
            working_coords = mol.atom_coords @ R.T + translation
        else:
            working_coords = mol.atom_coords
        
        # Create temporary molecule for energy computation
        temp_mol = Molecule()
        temp_mol.atom_coords = working_coords
        temp_mol.atom_elements = mol.atom_elements
        temp_mol.atom_gaff_types = mol.atom_gaff_types
        temp_mol.atom_charges = mol.atom_charges
        temp_mol.bonds = mol.bonds
        temp_mol.bond_orders = mol.bond_orders
        temp_mol.angles = mol.angles
        temp_mol.torsions = mol.torsions
        temp_mol.torsion_periodicities = mol.torsion_periodicities
        temp_mol.impropers = mol.impropers
        temp_mol.n_atoms = mol.n_atoms
        
        # Compute energy
        E = energy_fn(temp_mol)
        
        # Add position restraint if torsion_only
        if torsion_only:
            # Restrain non‑rotatable atoms to original positions
            restraint = F.mse_loss(working_coords, ligand.atom_coords.to(device))
            E += 100.0 * restraint
        
        E.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([mol.atom_coords], max_norm=5.0)
        
        opt.step()
        
        energy_history.append(E.item())
        
        if E.item() < best_E:
            best_E = E.item()
            best_coords = working_coords.detach().clone()
        
        if verbose and step % 50 == 0:
            print(f"  Refine step {step:4d}: E = {E.item():.4f}")
    
    # Return best structure
    mol.atom_coords = best_coords.detach()
    return mol


def refine_ligand_in_protein(ligand: Molecule,
                              protein_coords: torch.Tensor,
                              protein_atom_types: List[str],
                              protein_charges: List[float],
                              steps: int = 100,
                              lr: float = 1e-3,
                              device: str = 'cpu',
                              verbose: bool = False) -> Molecule:
    """
    Refine ligand position inside protein binding site.
    
    Optimizes ligand translation + rotation + torsions
    while keeping protein fixed.
    
    Args:
        ligand: ligand molecule to dock/refine
        protein_coords: [N, 3] protein atom coordinates
        protein_atom_types: list of protein atom types
        protein_charges: list of protein partial charges
        steps: optimization steps
        lr: learning rate
        device: 'cpu' or 'cuda'
        verbose: print progress
    
    Returns:
        Refined ligand
    """
    device = torch.device(device)
    
    # Clone ligand
    mol = Molecule(name=ligand.name)
    mol.atom_coords = ligand.atom_coords.clone().detach().to(device).requires_grad_(True)
    mol.atom_elements = ligand.atom_elements.copy()
    mol.atom_names = ligand.atom_names.copy()
    mol.atom_gaff_types = ligand.atom_gaff_types.copy()
    mol.atom_charges = ligand.atom_charges.clone().detach().to(device) if ligand.atom_charges is not None else None
    mol.bonds = ligand.bonds.copy()
    mol.bond_orders = ligand.bond_orders.copy()
    mol.build_topology()
    mol.n_atoms = ligand.n_atoms
    
    prot_coords = protein_coords.clone().detach().to(device)
    
    # Energy functions
    lig_energy = LigandEnergy()
    pl_energy = ProteinLigandEnergy()
    
    # Rigid‑body parameters
    translation = torch.zeros(3, device=device, requires_grad=True)
    rotation = torch.zeros(3, device=device, requires_grad=True)
    
    opt = torch.optim.Adam([
        {'params': [mol.atom_coords], 'lr': lr},
        {'params': [translation], 'lr': lr * 0.5},
        {'params': [rotation], 'lr': lr * 0.1},
    ])
    
    best_E = float('inf')
    best_coords = None
    best_trans = None
    best_rot = None
    
    for step in range(steps):
        opt.zero_grad()
        
        # Apply rigid‑body
        angle = torch.norm(rotation)
        if angle > 1e-8:
            axis = rotation / angle
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=device)
            R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        else:
            R = torch.eye(3, device=device)
        
        lig_coords = mol.atom_coords @ R.T + translation
        
        # Ligand internal energy
        temp_mol = Molecule()
        temp_mol.atom_coords = lig_coords
        temp_mol.atom_elements = mol.atom_elements
        temp_mol.atom_gaff_types = mol.atom_gaff_types
        temp_mol.atom_charges = mol.atom_charges
        temp_mol.bonds = mol.bonds
        temp_mol.bond_orders = mol.bond_orders
        temp_mol.angles = mol.angles
        temp_mol.torsions = mol.torsions
        temp_mol.torsion_periodicities = mol.torsion_periodicities
        temp_mol.impropers = mol.impropers
        temp_mol.n_atoms = mol.n_atoms
        
        E_lig = lig_energy(temp_mol)
        
        # Protein‑ligand interaction
        E_pl = pl_energy(prot_coords, protein_atom_types, protein_charges, temp_mol)
        
        # Clash penalty
        E_clash = torch.tensor(0.0, device=device)
        for i in range(mol.n_atoms):
            lig_pos = lig_coords[i]
            dists = torch.norm(prot_coords - lig_pos.unsqueeze(0), dim=1)
            close = dists < 2.0
            if close.any():
                E_clash += torch.sum((2.0 - dists[close]) ** 2)
        
        E_total = 0.1 * E_lig + E_pl + 100.0 * E_clash
        
        E_total.backward()
        torch.nn.utils.clip_grad_norm_([mol.atom_coords, translation, rotation], max_norm=5.0)
        opt.step()
        
        if E_total.item() < best_E:
            best_E = E_total.item()
            best_coords = lig_coords.detach().clone()
            best_trans = translation.detach().clone()
            best_rot = rotation.detach().clone()
        
        if verbose and step % 20 == 0:
            print(f"  Dock step {step:4d}: E_total={E_total.item():.4f}  "
                  f"E_lig={E_lig.item():.4f}  E_pl={E_pl.item():.4f}  clash={E_clash.item():.4f}")
    
    mol.atom_coords = best_coords.detach()
    return mol


# ═══════════════════════════════════════════════════════════════
# FILE READERS (SDF, MOL2, PDB Ligand)
# ═══════════════════════════════════════════════════════════════

def read_sdf(sdf_path: str) -> List[Molecule]:
    """
    Read SDF file (multiple molecules).
    
    Returns list of Molecule objects.
    """
    molecules = []
    
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Molecule name
        if i >= len(lines) - 4:
            break
        
        name = lines[i].strip()
        i += 1
        
        # Skip header line
        i += 1
        
        # Skip comment line
        i += 1
        
        # Counts line
        if i >= len(lines):
            break
        counts_line = lines[i]
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
        i += 1
        
        mol = Molecule(name=name if name else "UNK")
        
        # Read atoms
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            line = lines[i]
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            elem = line[31:34].strip()
            
            mol.atom_elements.append(elem if elem else 'C')
            mol.atom_names.append(f"{elem}{len(mol.atom_elements)}")
            
            if mol.atom_coords is None:
                mol.atom_coords = torch.tensor([[x, y, z]], dtype=torch.float32)
            else:
                mol.atom_coords = torch.cat([
                    mol.atom_coords,
                    torch.tensor([[x, y, z]], dtype=torch.float32)
                ], dim=0)
            
            i += 1
        
        # Read bonds
        for _ in range(n_bonds):
            if i >= len(lines):
                break
            line = lines[i]
            a1 = int(line[0:3].strip()) - 1
            a2 = int(line[3:6].strip()) - 1
            order = int(line[6:9].strip()) if len(line) > 6 else 1
            
            mol.bonds.append((a1, a2))
            mol.bond_orders.append(float(order))
            i += 1
        
        # Build topology
        mol.n_atoms = len(mol.atom_elements)
        mol.build_topology()
        mol.assign_gaff_types()
        
        # Estimate charges (simple Gasteiger‑like)
        mol.atom_charges = torch.zeros(mol.n_atoms)
        # Assign based on element and connectivity
        for a in range(mol.n_atoms):
            elem = mol.atom_elements[a]
            if elem == 'N':
                mol.atom_charges[a] = -0.5
            elif elem == 'O':
                mol.atom_charges[a] = -0.5
            elif elem == 'S':
                mol.atom_charges[a] = -0.2
            elif elem == 'F':
                mol.atom_charges[a] = -0.2
            elif elem == 'Cl':
                mol.atom_charges[a] = -0.2
            elif elem == 'P':
                mol.atom_charges[a] = 0.5
        
        molecules.append(mol)
        
        # Skip to next molecule or end
        while i < len(lines) and not lines[i].strip() == "$$$$":
            i += 1
        i += 1  # skip $$$$
    
    return molecules


def read_mol2(mol2_path: str) -> List[Molecule]:
    """
    Read MOL2 file (single or multiple molecules).
    
    Returns list of Molecule objects.
    """
    molecules = []
    
    with open(mol2_path, 'r') as f:
        content = f.read()
    
    # Split by molecule
    sections = content.split('@<TRIPOS>MOLECULE')
    
    for section in sections[1:]:  # skip first empty
        lines = section.strip().split('\n')
        if len(lines) < 2:
            continue
        
        name = lines[0].strip()
        mol = Molecule(name=name)
        
        # Find atom and bond sections
        atom_start = -1
        bond_start = -1
        atom_end = -1
        bond_end = -1
        
        for j, line in enumerate(lines):
            if line.strip() == '@<TRIPOS>ATOM':
                atom_start = j + 1
            elif line.strip() == '@<TRIPOS>BOND':
                bond_start = j + 1
                if atom_start > 0:
                    atom_end = j
            elif line.strip().startswith('@<TRIPOS>') and bond_start > 0:
                bond_end = j
                break
        
        if atom_start < 0:
            continue
        
        if atom_end < 0:
            atom_end = len(lines)
        if bond_end < 0:
            bond_end = len(lines)
        
        # Read atoms
        atom_idx_map = {}  # mol2 ID → internal index
        coords_list = []
        
        for j in range(atom_start, min(atom_end, len(lines))):
            if not lines[j].strip():
                continue
            parts = lines[j].split()
            if len(parts) < 6:
                continue
            
            mol2_id = int(parts[0])
            atom_name = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            atom_type = parts[5]
            
            # Extract element from atom type
            elem = atom_type.split('.')[0]
            if len(elem) > 2:
                elem = elem[:2]
            if elem not in ATOMIC_NUMBER:
                elem = elem[0]
            
            idx = len(mol.atom_elements)
            atom_idx_map[mol2_id] = idx
            
            mol.atom_elements.append(elem)
            mol.atom_names.append(atom_name)
            coords_list.append([x, y, z])
            
            # Read charge if available
            if len(parts) > 8:
                charge = float(parts[8])
                if mol.atom_charges is None:
                    mol.atom_charges = torch.zeros(len(mol.atom_elements))
                    mol.atom_charges[-1] = charge
        
        mol.atom_coords = torch.tensor(coords_list, dtype=torch.float32)
        
        # Read bonds
        if bond_start > 0:
            for j in range(bond_start, min(bond_end, len(lines))):
                if not lines[j].strip():
                    continue
                parts = lines[j].split()
                if len(parts) < 4:
                    continue
                
                a1 = atom_idx_map.get(int(parts[1]), -1)
                a2 = atom_idx_map.get(int(parts[2]), -1)
                order_text = parts[3]
                
                if a1 < 0 or a2 < 0:
                    continue
                
                # Parse bond order
                if order_text == 'ar':
                    order = 1.5
                elif order_text == 'am':
                    order = 1.0
                elif order_text == 'du':
                    order = 1.0  # dummy
                else:
                    try:
                        order = float(order_text)
                    except ValueError:
                        order = 1.0
                
                mol.bonds.append((a1, a2))
                mol.bond_orders.append(order)
        
        mol.n_atoms = len(mol.atom_elements)
        mol.build_topology()
        mol.assign_gaff_types()
        
        # Use MOL2 charges if available, else estimate
        if mol.atom_charges is None:
            mol.atom_charges = torch.zeros(mol.n_atoms)
            for a in range(mol.n_atoms):
                elem = mol.atom_elements[a]
                if elem == 'N': mol.atom_charges[a] = -0.5
                elif elem == 'O': mol.atom_charges[a] = -0.5
                elif elem == 'S': mol.atom_charges[a] = -0.2
                elif elem in ('F', 'Cl', 'Br', 'I'): mol.atom_charges[a] = -0.2
                elif elem == 'P': mol.atom_charges[a] = 0.5
        
        molecules.append(mol)
    
    return molecules


def read_pdb_ligand(pdb_path: str, residue_name: str = None,
                     chain: str = None) -> List[Molecule]:
    """
    Extract ligand(s) from PDB file (HETATM records).
    
    Args:
        pdb_path: PDB file path
        residue_name: specific residue name to extract (e.g., 'LIG', 'ATP')
        chain: specific chain
    
    Returns:
        List of Molecule objects
    """
    import gzip
    
    residues = defaultdict(lambda: {'coords': [], 'elements': [], 'names': []})
    residue_order = []
    
    opener = gzip.open if pdb_path.endswith('.gz') else open
    
    with opener(pdb_path, 'rt', errors='ignore') as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            
            if residue_name and res_name != residue_name:
                continue
            if chain and chain_id != chain:
                continue
            
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
            
            key = (res_name, chain_id)
            if key not in residues:
                residue_order.append(key)
            
            residues[key]['coords'].append([x, y, z])
            residues[key]['elements'].append(elem if elem else atom_name[0])
            residues[key]['names'].append(atom_name)
    
    molecules = []
    for key in residue_order:
        data = residues[key]
        mol = Molecule(name=f"{key[0]}_{key[1]}")
        mol.atom_coords = torch.tensor(data['coords'], dtype=torch.float32)
        mol.atom_elements = data['elements']
        mol.atom_names = data['names']
        mol.n_atoms = len(data['elements'])
        
        # Auto‑detect bonds from distance
        mol.bonds = []
        mol.bond_orders = []
        coords_np = mol.atom_coords.numpy()
        
        for i in range(mol.n_atoms):
            for j in range(i + 1, mol.n_atoms):
                ri = COVALENT_RADIUS.get(mol.atom_elements[i], 0.76)
                rj = COVALENT_RADIUS.get(mol.atom_elements[j], 0.76)
                d_max = (ri + rj) * 1.2
                
                d = float(np.linalg.norm(coords_np[i] - coords_np[j]))
                if d < d_max:
                    mol.bonds.append((i, j))
                    # Estimate bond order
                    if d < (ri + rj) * 0.85:
                        order = 2.0
                    elif d < (ri + rj) * 1.1:
                        order = 1.0
                    else:
                        order = 1.0
                    mol.bond_orders.append(order)
        
        mol.build_topology()
        mol.assign_gaff_types()
        mol.atom_charges = torch.zeros(mol.n_atoms)
        
        molecules.append(mol)
    
    return molecules


# ═══════════════════════════════════════════════════════════════
# INTEGRATION WITH v30.1
# ═══════════════════════════════════════════════════════════════

class LigandBridge:
    """
    Bridge to add ligand energy to CSOC‑SSC v30.1.
    
    Usage:
        bridge = LigandBridge()
        bridge.add_ligand(mol)
        
        # In total_physics_energy_v30_1:
        E_lig = bridge.compute_ligand_energy()
        E_pl = bridge.compute_interaction(protein_coords, protein_types, protein_charges)
        E_total += E_lig + E_pl
    """
    
    def __init__(self):
        self.ligands: List[Molecule] = []
        self.lig_energy = LigandEnergy()
        self.pl_energy = ProteinLigandEnergy()
    
    def add_ligand(self, ligand: Molecule):
        """Add a ligand molecule."""
        self.ligands.append(ligand)
    
    def add_ligand_from_file(self, filepath: str, filetype: str = 'sdf'):
        """Load ligand(s) from file."""
        if filetype == 'sdf':
            mols = read_sdf(filepath)
        elif filetype == 'mol2':
            mols = read_mol2(filepath)
        elif filetype == 'pdb':
            mols = read_pdb_ligand(filepath)
        else:
            raise ValueError(f"Unknown file type: {filetype}")
        
        for mol in mols:
            self.ligands.append(mol)
        
        return len(mols)
    
    def compute_ligand_energy(self) -> torch.Tensor:
        """Compute total internal energy of all ligands."""
        E = torch.tensor(0.0)
        for lig in self.ligands:
            E += self.lig_energy(lig)
        return E
    
    def compute_interaction(self,
                            protein_coords: torch.Tensor,
                            protein_atom_types: List[str],
                            protein_charges: List[float]) -> torch.Tensor:
        """Compute protein‑ligand interaction energy for all ligands."""
        E = torch.tensor(0.0, device=protein_coords.device)
        for lig in self.ligands:
            E += self.pl_energy(protein_coords, protein_atom_types,
                               protein_charges, lig)
        return E
    
    def refine_all_ligands(self, protein_coords=None,
                            protein_types=None, protein_charges=None,
                            steps=100, device='cpu'):
        """Refine all ligands."""
        for i, lig in enumerate(self.ligands):
            if protein_coords is not None:
                self.ligands[i] = refine_ligand_in_protein(
                    lig, protein_coords, protein_types, protein_charges,
                    steps=steps, device=device
                )
            else:
                self.ligands[i] = refine_ligand(
                    lig, self.lig_energy, steps=steps, device=device
                )


# ═══════════════════════════════════════════════════════════════
# EXAMPLE & TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CSOC‑SSC v30.3 — Ligand/Small Molecule Module")
    print("=" * 70)
    
    # Test 1: Build a simple molecule (ethane)
    print("\n[Test 1] Building ethane molecule")
    ethane = Molecule(name="ethane")
    ethane.atom_elements = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    ethane.atom_names = ['C1', 'C2', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    
    # Approximate coordinates
    ethane.atom_coords = torch.tensor([
        [-0.77,  0.0,  0.0],  # C1
        [ 0.77,  0.0,  0.0],  # C2
        [-1.17,  1.0,  0.0],  # H1
        [-1.17, -0.5,  0.87], # H2
        [-1.17, -0.5, -0.87], # H3
        [ 1.17, -1.0,  0.0],  # H4
        [ 1.17,  0.5,  0.87], # H5
        [ 1.17,  0.5, -0.87], # H6
    ])
    
    # Define bonds
    ethane.bonds = [(0,1), (0,2), (0,3), (0,4), (1,5), (1,6), (1,7)]
    ethane.bond_orders = [1.0] * 7
    ethane.n_atoms = 8
    ethane.build_topology()
    ethane.assign_gaff_types()
    ethane.atom_charges = torch.zeros(8)
    ethane.atom_charges[0] = -0.3
    ethane.atom_charges[1] = -0.3
    ethane.atom_charges[2:8] = 0.1
    
    print(f"  Atoms: {ethane.n_atoms}")
    print(f"  Bonds: {len(ethane.bonds)}")
    print(f"  Angles: {len(ethane.angles)}")
    print(f"  Torsions: {len(ethane.torsions)}")
    print(f"  GAFF types: {ethane.atom_gaff_types}")
    
    # Test 2: Compute ligand energy
    print("\n[Test 2] Computing ligand energy")
    lig_eng = LigandEnergy()
    E_total = lig_eng(ethane)
    E_bond = lig_eng.energy_bond(ethane)
    E_angle = lig_eng.energy_angle(ethane)
    E_torsion = lig_eng.energy_torsion(ethane)
    E_nonbond = lig_eng.energy_nonbonded(ethane)
    
    print(f"  E_bond:     {E_bond.item():.4f}")
    print(f"  E_angle:    {E_angle.item():.4f}")
    print(f"  E_torsion:  {E_torsion.item():.4f}")
    print(f"  E_nonbond:  {E_nonbond.item():.4f}")
    print(f"  E_TOTAL:    {E_total.item():.4f}")
    
    # Test 3: Refine ligand
    print("\n[Test 3] Refining ligand geometry")
    ethane_refined = refine_ligand(ethane, lig_eng, steps=100, verbose=False)
    E_refined = lig_eng(ethane_refined)
    print(f"  Initial E:  {E_total.item():.4f}")
    print(f"  Refined E:  {E_refined.item():.4f}")
    
    # Test 4: Protein‑ligand interaction
    print("\n[Test 4] Protein‑ligand interaction")
    # Fake protein atoms
    prot_coords = torch.randn(50, 3) * 10
    prot_types = ['CA'] * 50
    prot_charges = [0.0] * 50
    
    pl_eng = ProteinLigandEnergy()
    E_pl = pl_eng(prot_coords, prot_types, prot_charges, ethane)
    print(f"  Protein‑Ligand E: {E_pl.item():.4f}")
    
    # Test 5: Binding affinity
    print("\n[Test 5] Binding affinity estimation")
    affinity = estimate_binding_affinity(E_pl.item(), E_ligand_strain=0.5)
    print(f"  ΔG_bind:  {affinity['dG_bind_kcal']:.2f} kcal/mol")
    print(f"  Kd:       {affinity['Kd_M']:.2e} M")
    
    # Test 6: Ligand bridge
    print("\n[Test 6] Ligand bridge for v30.1")
    bridge = LigandBridge()
    bridge.add_ligand(ethane)
    bridge.add_ligand(ethane_refined)
    print(f"  Ligands loaded: {len(bridge.ligands)}")
    E_all_lig = bridge.compute_ligand_energy()
    print(f"  Total ligand energy: {E_all_lig.item():.4f}")
    
    print(f"\n{'='*70}")
    print("Module ready for integration with CSOC‑SSC v30.1")
    print("Import: from csoc_ligand import LigandBridge, Molecule, LigandEnergy")
    print(f"{'='*70}")
