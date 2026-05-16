#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v30.1 ↔ v30.2 DNA/RNA Integration Bridge
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Bridge module — drop‑in integration of DNA/RNA energy into v30.1
#
# Usage in csoc_v30_1.py:
#     from csoc_v30_1_dna_rna_bridge import DNA_RNA_Bridge
#     bridge = DNA_RNA_Bridge()
#     
#     # In total_physics_energy_v30_1:
#     e += bridge(ca, seq, edge_index_ca, edge_dist_ca)
# =============================================================================

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math

# Import the full DNA/RNA module
try:
    from csoc_dna_rna import (
        DNA_RNA_Energy,
        build_full_dna_rna,
        energy_backbone_c4_bond,
        energy_phosphate_restraint,
        energy_sugar_pucker,
        energy_base_pairing,
        energy_base_stacking,
        energy_dna_rna_lj,
        energy_dna_rna_coulomb,
        compute_dihedral,
        get_atom_type_for_topology,
        NUCLEOTIDE_LJ,
        NUCLEOTIDE_CHARGES,
        WC_PAIRS,
        BASE_STACKING,
        DNA_VOCAB,
        RNA_VOCAB,
        DNA_RNA_VOCAB,
        NT_TO_ID,
    )
    HAS_DNA_RNA = True
except ImportError:
    HAS_DNA_RNA = False
    print("WARNING: csoc_dna_rna module not found. DNA/RNA features disabled.")


# ═══════════════════════════════════════════════════════════════
# SEQUENCE DETECTION UTILITIES
# ═══════════════════════════════════════════════════════════════

# Protein amino acid alphabet
PROTEIN_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Nucleotide alphabet
DNA_NT = set("ACGT")
RNA_NT = set("ACGU")
ALL_NT = DNA_NT | RNA_NT


def detect_sequence_type(sequence: str) -> str:
    """
    Auto‑detect whether a sequence is Protein, DNA, or RNA.
    
    Heuristic: if >70% of characters are nucleotide‑only and <30% are
    amino‑acid‑only, classify as DNA/RNA.
    
    Args:
        sequence: 1‑letter code sequence string
    
    Returns:
        'protein', 'dna', 'rna', or 'unknown'
    """
    if not sequence:
        return 'unknown'
    
    seq_set = set(sequence.upper())
    
    # Count characters belonging to each alphabet
    n_nt = sum(1 for c in sequence.upper() if c in ALL_NT)
    n_aa = sum(1 for c in sequence.upper() if c in PROTEIN_AA)
    n_total = len(sequence)
    
    if n_total == 0:
        return 'unknown'
    
    frac_nt = n_nt / n_total
    frac_aa = n_aa / n_total
    
    # If almost all characters are nucleotides
    if frac_nt > 0.8 and 'T' in seq_set and 'U' not in seq_set:
        return 'dna'
    elif frac_nt > 0.8 and 'U' in seq_set:
        return 'rna'
    elif frac_nt > 0.8:
        return 'dna'  # default to DNA if no T/U distinction
    
    # If it has more protein characters
    if frac_aa > 0.7:
        return 'protein'
    
    # Ambiguous
    return 'unknown'


def is_dna_rna_sequence(sequence: str) -> bool:
    """Quick check if sequence is DNA/RNA."""
    return detect_sequence_type(sequence) in ('dna', 'rna')


# ═══════════════════════════════════════════════════════════════
# MAIN BRIDGE CLASS
# ═══════════════════════════════════════════════════════════════

class DNA_RNA_Bridge:
    """
    Drop‑in bridge to add DNA/RNA energy to CSOC‑SSC v30.1.
    
    This class handles:
      - Auto‑detection of DNA/RNA vs protein sequences
      - Creation and caching of DNA_RNA_Energy engines
      - Seamless integration with v30.1 energy function signature
      - Zero overhead for protein‑only chains
    
    Usage:
        # One‑time setup (in v30.1 main or config)
        bridge = DNA_RNA_Bridge(
            dna_pucker='C2_endo',   # B‑DNA
            rna_pucker='C3_endo',   # A‑RNA
        )
        
        # In total_physics_energy_v30_1:
        E_total += bridge(
            coords=ca_or_c4,
            sequence=seq,
            edge_index=edge_index_ca,
            edge_dist=edge_dist_ca,
        )
    """
    
    def __init__(self,
                 dna_pucker: str = 'C2_endo',
                 rna_pucker: str = 'C3_endo',
                 w_c4_bond: float = 30.0,
                 w_phosphate: float = 15.0,
                 w_pucker: float = 10.0,
                 w_base_pair: float = 8.0,
                 w_stacking: float = 5.0,
                 w_lj: float = 30.0,
                 w_coulomb: float = 3.0,
                 use_full_atom: bool = True,
                 lj_params: Optional[Dict] = None,
                 charge_map: Optional[Dict] = None,
                 auto_detect: bool = True,
                 verbose: bool = False,
                 ):
        """
        Args:
            dna_pucker: 'C2_endo' for B‑DNA
            rna_pucker: 'C3_endo' for A‑RNA
            w_c4_bond: C4'‑C4' backbone bond weight
            w_phosphate: P‑P distance restraint weight
            w_pucker: sugar pucker restraint weight
            w_base_pair: Watson‑Crick pairing weight
            w_stacking: base stacking weight
            w_lj: Lennard‑Jones weight
            w_coulomb: Coulomb weight
            use_full_atom: build full atomic model for LJ/Coulomb
            lj_params: custom LJ parameters dict
            charge_map: custom charge map dict
            auto_detect: auto‑detect DNA/RNA from sequence
            verbose: print debug info
        """
        self.dna_pucker = dna_pucker
        self.rna_pucker = rna_pucker
        self.w_c4_bond = w_c4_bond
        self.w_phosphate = w_phosphate
        self.w_pucker = w_pucker
        self.w_base_pair = w_base_pair
        self.w_stacking = w_stacking
        self.w_lj = w_lj
        self.w_coulomb = w_coulomb
        self.use_full_atom = use_full_atom
        self.lj_params = lj_params
        self.charge_map = charge_map
        self.auto_detect = auto_detect
        self.verbose = verbose
        
        # Cache for DNA_RNA_Energy instances
        self._dna_eng: Optional[DNA_RNA_Energy] = None
        self._rna_eng: Optional[DNA_RNA_Energy] = None
        
        # Sequence type cache (avoid re‑detecting every step)
        self._seq_type_cache: Dict[str, str] = {}
        
        # Statistics
        self.n_calls = 0
        self.n_dna = 0
        self.n_rna = 0
        self.n_protein = 0
        
        if not HAS_DNA_RNA:
            print("DNA_RNA_Bridge: csoc_dna_rna not available. All DNA/RNA calls will return 0.")
    
    def _get_dna_engine(self) -> 'DNA_RNA_Energy':
        """Lazy‑init B‑DNA energy engine."""
        if self._dna_eng is None and HAS_DNA_RNA:
            self._dna_eng = DNA_RNA_Energy(
                pucker_type=self.dna_pucker,
                w_c4_bond=self.w_c4_bond,
                w_phosphate=self.w_phosphate,
                w_pucker=self.w_pucker,
                w_base_pair=self.w_base_pair,
                w_stacking=self.w_stacking,
                w_lj=self.w_lj,
                w_coulomb=self.w_coulomb,
                lj_params=self.lj_params,
                charge_map=self.charge_map,
                use_full_atom=self.use_full_atom,
            )
        return self._dna_eng
    
    def _get_rna_engine(self) -> 'DNA_RNA_Energy':
        """Lazy‑init A‑RNA energy engine."""
        if self._rna_eng is None and HAS_DNA_RNA:
            self._rna_eng = DNA_RNA_Energy(
                pucker_type=self.rna_pucker,
                w_c4_bond=self.w_c4_bond,
                w_phosphate=self.w_phosphate,
                w_pucker=self.w_pucker,
                w_base_pair=self.w_base_pair,
                w_stacking=self.w_stacking,
                w_lj=self.w_lj,
                w_coulomb=self.w_coulomb,
                lj_params=self.lj_params,
                charge_map=self.charge_map,
                use_full_atom=self.use_full_atom,
            )
        return self._rna_eng
    
    def _detect_type(self, sequence: str) -> str:
        """Detect sequence type with caching."""
        if sequence in self._seq_type_cache:
            return self._seq_type_cache[sequence]
        
        if self.auto_detect:
            seq_type = detect_sequence_type(sequence)
        else:
            seq_type = 'unknown'
        
        self._seq_type_cache[sequence] = seq_type
        return seq_type
    
    def __call__(self,
                 coords: torch.Tensor,        # [L, 3]
                 sequence: str,                # 1‑letter sequence
                 edge_index: Optional[torch.Tensor] = None,  # [2, E]
                 edge_dist: Optional[torch.Tensor] = None,   # [E]
                 chain_type: Optional[str] = None,  # 'protein','dna','rna', or None for auto
                 ) -> torch.Tensor:
        """
        Compute DNA/RNA energy if applicable.
        
        Args:
            coords: C4' coordinates for DNA/RNA [L, 3]
            sequence: nucleotide sequence string
            edge_index: sparse edge index [2, E] (optional)
            edge_dist: sparse edge distances [E] (optional)
            chain_type: force chain type, or None for auto‑detect
        
        Returns:
            Energy scalar tensor (0.0 if protein or detection fails)
        """
        self.n_calls += 1
        
        if not HAS_DNA_RNA:
            return torch.tensor(0.0, device=coords.device)
        
        # Determine chain type
        if chain_type is None:
            chain_type = self._detect_type(sequence)
        
        # Only process DNA/RNA
        if chain_type == 'dna':
            self.n_dna += 1
            engine = self._get_dna_engine()
        elif chain_type == 'rna':
            self.n_rna += 1
            engine = self._get_rna_engine()
        else:
            self.n_protein += 1
            return torch.tensor(0.0, device=coords.device)
        
        if engine is None:
            return torch.tensor(0.0, device=coords.device)
        
        # Compute DNA/RNA energy
        try:
            energy = engine(coords, sequence, edge_index, edge_dist)
        except Exception as e:
            if self.verbose:
                print(f"DNA_RNA_Bridge: energy computation failed: {e}")
            return torch.tensor(0.0, device=coords.device)
        
        return energy
    
    def get_statistics(self) -> Dict:
        """Return usage statistics."""
        return {
            'n_calls': self.n_calls,
            'n_dna': self.n_dna,
            'n_rna': self.n_rna,
            'n_protein': self.n_protein,
            'has_module': HAS_DNA_RNA,
        }
    
    def print_statistics(self):
        """Print usage statistics."""
        stats = self.get_statistics()
        print(f"DNA_RNA_Bridge Statistics:")
        print(f"  Total calls:  {stats['n_calls']}")
        print(f"  DNA chains:   {stats['n_dna']}")
        print(f"  RNA chains:   {stats['n_rna']}")
        print(f"  Protein:      {stats['n_protein']}")
        print(f"  Module OK:    {stats['has_module']}")


# ═══════════════════════════════════════════════════════════════
# MULTI‑CHAIN SUPPORT
# ═══════════════════════════════════════════════════════════════

class MultiChainBridge:
    """
    Bridge for multi‑chain systems with mixed protein/DNA/RNA.
    
    Automatically splits concatenated coordinates by chain boundaries
    and applies the correct energy function to each chain.
    
    Usage:
        mcb = MultiChainBridge()
        
        # In total_physics_energy_v30_1:
        E_total += mcb(
            all_coords=ca_concatenated,      # [total_L, 3]
            all_sequence=all_seq,             # concatenated sequence
            chain_boundaries=boundaries,      # [n_chains-1] start indices
            chain_types=None,                 # auto‑detect, or list
            edge_index=edge_index,
            edge_dist=edge_dist,
        )
    """
    
    def __init__(self,
                 dna_pucker: str = 'C2_endo',
                 rna_pucker: str = 'C3_endo',
                 **kwargs):
        self.bridge = DNA_RNA_Bridge(
            dna_pucker=dna_pucker,
            rna_pucker=rna_pucker,
            **kwargs
        )
    
    def __call__(self,
                 all_coords: torch.Tensor,       # [total_L, 3]
                 all_sequence: str,              # concatenated
                 chain_boundaries: List[int],    # [n_chains-1]
                 chain_types: Optional[List[str]] = None,  # per‑chain
                 edge_index: Optional[torch.Tensor] = None,
                 edge_dist: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        """
        Compute DNA/RNA energy for all chains.
        
        Returns:
            Total energy across all DNA/RNA chains.
        """
        device = all_coords.device
        E_total = torch.tensor(0.0, device=device)
        
        # Split by chain boundaries
        boundaries = [0] + list(chain_boundaries) + [len(all_sequence)]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            chain_seq = all_sequence[start:end]
            chain_coords = all_coords[start:end]
            
            # Determine chain type
            if chain_types and i < len(chain_types):
                ct = chain_types[i]
            else:
                ct = None  # auto‑detect
            
            # Compute (returns 0 for protein)
            E_chain = self.bridge(chain_coords, chain_seq,
                                  edge_index, edge_dist, ct)
            E_total += E_chain
        
        return E_total


# ═══════════════════════════════════════════════════════════════
# INTEGRATION PATCH FOR v30.1
# =====================================================================
# Copy‑paste this snippet into csoc_v30_1.py to enable DNA/RNA support
# =====================================================================

def get_v30_1_integration_snippet():
    """
    Returns the code snippet to paste into csoc_v30_1.py.
    
    Print this and follow the instructions.
    """
    snippet = '''
# ─── DNA/RNA Integration (add near other imports) ───
from csoc_v30_1_dna_rna_bridge import DNA_RNA_Bridge, MultiChainBridge

# ─── Create bridge (add near V30_1Config initialization) ───
# Option A: Single chain
dna_rna_bridge = DNA_RNA_Bridge(
    dna_pucker='C2_endo',    # B‑DNA
    rna_pucker='C3_endo',    # A‑RNA
    verbose=False,
)

# Option B: Multi‑chain
# dna_rna_bridge = MultiChainBridge(
#     dna_pucker='C2_endo',
#     rna_pucker='C3_endo',
# )

# ─── Add in total_physics_energy_v30_1 (before final return) ───
# After computing all protein energy terms:
# 
# E_dna_rna = dna_rna_bridge(
#     all_coords=ca,              # or all concatenated coords
#     all_sequence=all_seq,       # or concatenated sequence
#     chain_boundaries=boundaries, # for MultiChainBridge
#     edge_index=edge_index_ca,
#     edge_dist=edge_dist_ca,
# )
# e += E_dna_rna

# ─── Optional: print statistics at end ───
# dna_rna_bridge.print_statistics()
'''
    return snippet


# ═══════════════════════════════════════════════════════════════
# TEST & DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CSOC‑SSC v30.1 ↔ v30.2 DNA/RNA Bridge — Test")
    print("=" * 70)
    
    # Test 1: Sequence detection
    print("\n[Test 1] Sequence type detection")
    test_seqs = [
        ("ACDEFGHIKLMNPQRSTVWY", "protein"),
        ("ACGTACGTACGT", "dna"),
        ("ACGUAACGU", "rna"),
        ("AAAAATTTTT", "dna"),
        ("GGGGCCCC", "dna"),
    ]
    for seq, expected in test_seqs:
        detected = detect_sequence_type(seq)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} '{seq[:20]}...' → {detected} (expected {expected})")
    
    # Test 2: Bridge with DNA
    if HAS_DNA_RNA:
        print("\n[Test 2] Bridge with B‑DNA")
        from csoc_dna_rna import build_dna_helix
        
        bridge = DNA_RNA_Bridge(verbose=True)
        
        # Build a small DNA helix
        seq = "ACGTACGT"
        coords = build_dna_helix(seq)
        
        E = bridge(coords, seq)
        print(f"  DNA Energy: {E.item():.4f}")
        bridge.print_statistics()
        
        # Test 3: Bridge with protein (should return 0)
        print("\n[Test 3] Bridge with Protein (should return 0)")
        prot_seq = "ACDEFGHI"
        prot_coords = torch.randn(len(prot_seq), 3)
        E_prot = bridge(prot_coords, prot_seq)
        print(f"  Protein Energy: {E_prot.item():.4f} (should be 0)")
        
        # Test 4: Multi‑chain
        print("\n[Test 4] Multi‑Chain Bridge")
        mcb = MultiChainBridge()
        
        # Mix: protein + DNA + protein
        prot_seq1 = "ACDE"
        dna_seq = "ACGT"
        prot_seq2 = "FGHI"
        
        all_seq = prot_seq1 + dna_seq + prot_seq2
        all_coords = torch.randn(len(all_seq), 3)
        boundaries = [len(prot_seq1), len(prot_seq1) + len(dna_seq)]
        
        E_multi = mcb(all_coords, all_seq, boundaries)
        print(f"  Multi‑chain Energy: {E_multi.item():.4f}")
        mcb.bridge.print_statistics()
    
    else:
        print("\n[WARNING] csoc_dna_rna module not available. Skipping integration tests.")
    
    # Print integration snippet
    print("\n" + "=" * 70)
    print("Integration Snippet for csoc_v30_1.py:")
    print("=" * 70)
    print(get_v30_1_integration_snippet())
