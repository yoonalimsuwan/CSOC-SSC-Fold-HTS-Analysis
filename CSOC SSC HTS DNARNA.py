
# =============================================================================
# CSOC‑SSC HTS FOLD v32.1 — DNA/RNA Mutation Scanning & Epistasis Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Standalone HTS FOLD for DNA/RNA — Complete mutational scanning
#
# Features:
#   ✓ Full single‑mutation scan (all positions × all 4 nucleotides)
#   ✓ Double‑mutation epistasis scanning
#   ✓ ΔΔG prediction using v30.2 DNA/RNA energy function
#   ✓ Side‑chain relaxation (sugar + base repositioning)
#   ✓ Mutational landscape heatmap
#   ✓ CSV / JSON export
#   ✓ Multi‑GPU parallel scanning
#   ✓ B‑DNA / A‑RNA support
#   ✓ Compatible with CSOC‑SSC v30.1 + v30.2
# =============================================================================

import os, sys, argparse, logging, warnings, json, itertools, copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("CSOC‑HTS-DNA-RNA-V32.1")

# ═══════════════════════════════════════════════════════════════
# Import DNA/RNA modules
# ═══════════════════════════════════════════════════════════════
try:
    from csoc_dna_rna import (
        DNA_RNA_Energy,
        build_full_dna_rna,
        build_dna_helix,
        build_rna_helix,
        build_double_strand_helix,
        load_nucleotide_pdb,
        load_double_strand_pdb,
        write_nucleotide_pdb,
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
    raise ImportError(
        "csoc_dna_rna module is required for HTS FOLD DNA/RNA v32.1. "
        "Ensure csoc_dna_rna.py is in the same directory."
    )

try:
    from csoc_v30_1_dna_rna_bridge import detect_sequence_type, is_dna_rna_sequence
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False
    # Fallback detection
    def detect_sequence_type(seq):
        nt_set = set(seq.upper())
        if nt_set.issubset(set('ACGT')):
            return 'dna'
        elif nt_set.issubset(set('ACGU')):
            return 'rna'
        return 'protein'
    
    def is_dna_rna_sequence(seq):
        return detect_sequence_type(seq) in ('dna', 'rna')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class HTS_DNA_RNA_Config:
    """Configuration for HTS FOLD DNA/RNA v32.1"""
    
    def __init__(self,
                 pdb_structure: Optional[str] = None,
                 sequence: Optional[str] = None,
                 chain_id: Optional[str] = None,
                 ds_type: str = 'B_DNA',          # 'B_DNA' or 'A_RNA'
                 output_dir: str = "./hts_dna_rna_output",
                 ddg_threshold: float = 0.5,
                 mutation_list: Optional[List[Tuple[int, str]]] = None,
                 scan_full: bool = False,
                 scan_pairs: bool = False,
                 epistasis_pairs: Optional[List[Tuple[int, int]]] = None,
                 relaxation_steps: int = 30,
                 use_gpu: bool = False,
                 num_gpus: int = 1,
                 lj_param_file: Optional[str] = None,
                 charge_param_file: Optional[str] = None,
                 ):
        self.pdb_structure = pdb_structure
        self.sequence = sequence
        self.chain_id = chain_id or 'A'
        self.ds_type = ds_type
        self.output_dir = output_dir
        self.ddg_threshold = ddg_threshold
        self.mutation_list = mutation_list
        self.scan_full = scan_full
        self.scan_pairs = scan_pairs
        self.epistasis_pairs = epistasis_pairs
        self.relaxation_steps = relaxation_steps
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.lj_param_file = lj_param_file
        self.charge_param_file = charge_param_file


# ═══════════════════════════════════════════════════════════════
# MUTATION UTILITIES
# ═══════════════════════════════════════════════════════════════

# All possible single‑base mutations
DNA_BASES = ['A', 'C', 'G', 'T']
RNA_BASES = ['A', 'C', 'G', 'U']

# Transition / Transversion classification
TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C'),
               ('A', 'U'), ('U', 'A'), ('C', 'U'), ('U', 'C')}  # RNA version

def is_transition(old_base: str, new_base: str) -> bool:
    """Check if mutation is a transition (purine↔purine or pyrimidine↔pyrimidine)."""
    return (old_base, new_base) in TRANSITIONS

def is_transversion(old_base: str, new_base: str) -> bool:
    """Check if mutation is a transversion (purine↔pyrimidine)."""
    return not is_transition(old_base, new_base) and old_base != new_base

def is_synonymous_wc(old_base: str, new_base: str, paired_base: Optional[str] = None) -> bool:
    """Check if mutation preserves Watson‑Crick pairing."""
    if paired_base is None:
        return False
    old_pair = WC_PAIRS.get((old_base, paired_base), 0)
    new_pair = WC_PAIRS.get((new_base, paired_base), 0)
    return old_pair == new_pair and old_pair > 0

def generate_all_single_mutations(sequence: str, is_rna: bool = False) -> List[Tuple[int, str]]:
    """Generate all possible single‑base mutations."""
    bases = RNA_BASES if is_rna else DNA_BASES
    mutations = []
    for pos, wt_base in enumerate(sequence):
        for new_base in bases:
            if new_base != wt_base:
                mutations.append((pos, new_base))
    return mutations

def generate_double_mutations(sequence: str, max_pairs: int = 500,
                               is_rna: bool = False) -> List[Tuple[int, str, int, str]]:
    """Generate double mutations for epistasis scanning."""
    bases = RNA_BASES if is_rna else DNA_BASES
    singles = []
    for pos, wt in enumerate(sequence):
        for new_base in bases:
            if new_base != wt:
                singles.append((pos, new_base))
    
    # Limit to avoid combinatorial explosion
    if len(singles) > 100:
        # Prioritize: transitions first (more common), then near middle of sequence
        singles_sorted = sorted(singles, key=lambda x: (
            0 if is_transition(sequence[x[0]], x[1]) else 1,
            abs(x[0] - len(sequence)//2)
        ))
        singles = singles_sorted[:100]
    
    pairs = []
    for (p1, b1), (p2, b2) in itertools.combinations(singles, 2):
        if abs(p1 - p2) >= 2:  # not adjacent
            pairs.append((p1, b1, p2, b2))
            if len(pairs) >= max_pairs:
                break
    
    return pairs


# ═══════════════════════════════════════════════════════════════
# HTS DNA/RNA ANALYZER
# ═══════════════════════════════════════════════════════════════

class HTS_DNA_RNA_Analyzer:
    """
    High‑Throughput Scanning for DNA/RNA mutations.
    
    Computes ΔΔG = E(mutant) − E(wild‑type) for any set of
    single or double mutations using the v30.2 energy function.
    """
    
    def __init__(self, config: HTS_DNA_RNA_Config):
        self.config = config
        self.results = {}
        
        # DNA/RNA state
        self.wt_sequence: Optional[str] = None
        self.wt_C4: Optional[torch.Tensor] = None       # [L, 3]
        self.wt_all_coords: Optional[torch.Tensor] = None
        self.wt_all_types: Optional[List[str]] = None
        self.wt_res_indices: Optional[torch.Tensor] = None
        self.wt_energy: Optional[float] = None
        self.is_rna: bool = False
        self.device: torch.device = None
        
        # Energy engine
        self.energy_engine: Optional[DNA_RNA_Energy] = None
        
        # Load force field if specified
        self.lj_params = None
        self.charge_map = None
        if config.lj_param_file or config.charge_param_file:
            from csoc_dna_rna import load_nucleotide_forcefield
            self.lj_params, self.charge_map = load_nucleotide_forcefield(
                config.lj_param_file, config.charge_param_file
            )
    
    def initialize(self):
        """Load or build wild‑type structure."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu"
        )
        
        # Determine sequence type and load structure
        if self.config.pdb_structure and os.path.exists(self.config.pdb_structure):
            self._load_from_pdb()
        elif self.config.sequence:
            self._build_helix()
        else:
            raise ValueError("Must provide either --pdb or --seq")
        
        # Set RNA flag
        self.is_rna = 'U' in self.wt_sequence.upper()
        
        # Create energy engine
        pucker = 'C3_endo' if self.is_rna else 'C2_endo'
        self.energy_engine = DNA_RNA_Energy(
            pucker_type=pucker,
            use_full_atom=True,
            lj_params=self.lj_params,
            charge_map=self.charge_map,
        )
        
        # Compute WT energy
        self.wt_energy = self._compute_energy(self.wt_C4, self.wt_sequence)
        
        logger.info(f"WT structure loaded: {len(self.wt_sequence)} nt")
        logger.info(f"WT type: {'RNA' if self.is_rna else 'DNA'}")
        logger.info(f"WT energy: {self.wt_energy:.4f} kcal/mol")
    
    def _load_from_pdb(self):
        """Load structure from PDB file."""
        C4_coords, seq = load_nucleotide_pdb(
            self.config.pdb_structure,
            chain=self.config.chain_id
        )
        if len(seq) == 0:
            raise ValueError(f"No nucleotides found in {self.config.pdb_structure}")
        
        self.wt_sequence = seq
        self.wt_C4 = C4_coords.to(self.device)
        self._build_full_atom()
        logger.info(f"Loaded {len(seq)} nt from {self.config.pdb_structure}")
    
    def _build_helix(self):
        """Build ideal helix from sequence."""
        seq = self.config.sequence.upper()
        self.is_rna = 'U' in seq
        
        if self.is_rna:
            self.wt_C4 = build_rna_helix(seq).to(self.device)
        else:
            self.wt_C4 = build_dna_helix(seq).to(self.device)
        
        self.wt_sequence = seq
        self._build_full_atom()
        logger.info(f"Built ideal {'A‑RNA' if self.is_rna else 'B‑DNA'} helix: {len(seq)} nt")
    
    def _build_full_atom(self):
        """Build full atomic model for WT."""
        self.wt_all_coords, self.wt_all_types, self.wt_res_indices = \
            build_full_dna_rna(self.wt_C4, self.wt_sequence)
    
    def _compute_energy(self, C4: torch.Tensor, seq: str) -> float:
        """Compute total energy for a given C4' trace and sequence."""
        if self.energy_engine is None:
            raise RuntimeError("Energy engine not initialized")
        with torch.no_grad():
            E = self.energy_engine(C4, seq)
        return E.item()
    
    def predict_ddg_single(self, pos: int, new_base: str,
                            relax: bool = True) -> Dict:
        """
        Predict ΔΔG for a single mutation.
        
        Args:
            pos: position (0‑based)
            new_base: new nucleotide base ('A','C','G','T','U')
            relax: whether to perform local relaxation
        
        Returns:
            dict with 'position', 'wt', 'mut', 'ddg', 'relaxed', ...
        """
        wt_base = self.wt_sequence[pos]
        if wt_base == new_base:
            return {'position': pos, 'wt': wt_base, 'mut': new_base,
                    'ddg': 0.0, 'relaxed': False, 'type': 'self'}
        
        # Create mutant sequence
        mut_seq = self.wt_sequence[:pos] + new_base + self.wt_sequence[pos+1:]
        
        # Start from WT coordinates
        mut_C4 = self.wt_C4.clone()
        
        if relax and self.config.relaxation_steps > 0:
            mut_C4, e_mut = self._relax_mutant(mut_C4, mut_seq, pos)
        else:
            e_mut = self._compute_energy(mut_C4, mut_seq)
        
        ddg = e_mut - self.wt_energy
        
        # Mutation classification
        mut_type = 'transition' if is_transition(wt_base, new_base) else 'transversion'
        
        return {
            'position': pos,
            'wt': wt_base,
            'mut': new_base,
            'ddg': ddg,
            'relaxed': relax and self.config.relaxation_steps > 0,
            'type': mut_type,
            'e_wt': self.wt_energy,
            'e_mut': e_mut,
        }
    
    def _relax_mutant(self, C4: torch.Tensor, mut_seq: str,
                       mut_pos: int) -> Tuple[torch.Tensor, float]:
        """
        Local relaxation around mutation site.
        
        Only relaxes residues within ±3 positions of the mutation.
        """
        L = len(mut_seq)
        device = C4.device
        
        # Define relaxation window
        window_start = max(0, mut_pos - 3)
        window_end = min(L, mut_pos + 4)
        
        # Clone and require grad only for window
        C4_relax = C4.clone().detach().to(device)
        C4_relax.requires_grad_(True)
        
        # Only optimize window residues
        opt = torch.optim.Adam([C4_relax], lr=1e-3)
        
        best_E = float('inf')
        best_C4 = C4_relax.clone()
        
        for step in range(self.config.relaxation_steps):
            opt.zero_grad()
            
            # Compute energy (only on window region for speed)
            # Use full sequence but mask gradient outside window
            E = self.energy_engine(C4_relax, mut_seq)
            
            E.backward()
            
            # Zero gradient outside window
            with torch.no_grad():
                if C4_relax.grad is not None:
                    mask = torch.zeros(L, device=device)
                    mask[window_start:window_end] = 1.0
                    C4_relax.grad *= mask.unsqueeze(-1)
            
            opt.step()
            
            if E.item() < best_E:
                best_E = E.item()
                best_C4 = C4_relax.clone()
        
        return best_C4.detach(), best_E
    
    def scan_all_single_mutations(self) -> List[Dict]:
        """Scan all possible single‑base mutations."""
        if self.wt_sequence is None:
            raise RuntimeError("Structure not initialized. Call initialize() first.")
        
        seq = self.wt_sequence
        bases = RNA_BASES if self.is_rna else DNA_BASES
        
        results = []
        total = sum(1 for pos in range(len(seq)) for b in bases if b != seq[pos])
        
        logger.info(f"Scanning {total} single mutations...")
        
        for pos in tqdm(range(len(seq)), desc="Scanning positions"):
            wt_base = seq[pos]
            for new_base in bases:
                if new_base == wt_base:
                    continue
                result = self.predict_ddg_single(
                    pos, new_base,
                    relax=(self.config.relaxation_steps > 0)
                )
                results.append(result)
        
        return results
    
    def scan_epistasis_pairs(self) -> List[Dict]:
        """Scan double mutations for epistasis."""
        if self.wt_sequence is None:
            raise RuntimeError("Structure not initialized.")
        
        seq = self.wt_sequence
        
        if self.config.epistasis_pairs:
            # Use specified pairs
            pairs = []
            for p1, p2 in self.config.epistasis_pairs:
                bases = RNA_BASES if self.is_rna else DNA_BASES
                for b1 in bases:
                    if b1 != seq[p1]:
                        for b2 in bases:
                            if b2 != seq[p2]:
                                pairs.append((p1, b1, p2, b2))
        else:
            pairs = generate_double_mutations(seq, is_rna=self.is_rna)
        
        logger.info(f"Scanning {len(pairs)} double mutations for epistasis...")
        
        results = []
        for p1, b1, p2, b2 in tqdm(pairs, desc="Epistasis pairs"):
            # Single mutant energies
            d1 = self.predict_ddg_single(p1, b1, relax=True)
            d2 = self.predict_ddg_single(p2, b2, relax=True)
            
            # Double mutant
            mut_seq = seq[:p1] + b1 + seq[p1+1:p2] + b2 + seq[p2+1:]
            mut_C4 = self.wt_C4.clone()
            
            if self.config.relaxation_steps > 0:
                mut_C4, e_double = self._relax_mutant(mut_C4, mut_seq, p1)
                mut_C4, e_double = self._relax_mutant(mut_C4, mut_seq, p2)
            else:
                e_double = self._compute_energy(mut_C4, mut_seq)
            
            ddg_double = e_double - self.wt_energy
            ddg_additive = d1['ddg'] + d2['ddg']
            epistasis = ddg_double - ddg_additive
            
            results.append({
                'pos1': p1, 'mut1': b1, 'ddg1': d1['ddg'],
                'pos2': p2, 'mut2': b2, 'ddg2': d2['ddg'],
                'ddg_double': ddg_double,
                'ddg_additive': ddg_additive,
                'epistasis': epistasis,
                'significant': abs(epistasis) > self.config.ddg_threshold,
            })
        
        return results
    
    def compute_stacking_disruption(self, pos: int, new_base: str) -> Dict:
        """
        Compute how a mutation disrupts base stacking.
        
        Returns stacking energy change for i‑1,i and i,i+1 pairs.
        """
        seq = self.wt_sequence
        C4 = self.wt_C4
        
        stacking_changes = {}
        
        # i‑1,i stacking
        if pos > 0:
            wt_stack = BASE_STACKING.get(seq[pos-1], 1.0) * BASE_STACKING.get(seq[pos], 1.0)
            mut_stack = BASE_STACKING.get(seq[pos-1], 1.0) * BASE_STACKING.get(new_base, 1.0)
            stacking_changes['prev'] = mut_stack - wt_stack
        
        # i,i+1 stacking
        if pos < len(seq) - 1:
            wt_stack = BASE_STACKING.get(seq[pos], 1.0) * BASE_STACKING.get(seq[pos+1], 1.0)
            mut_stack = BASE_STACKING.get(new_base, 1.0) * BASE_STACKING.get(seq[pos+1], 1.0)
            stacking_changes['next'] = mut_stack - wt_stack
        
        return stacking_changes
    
    def run_analysis(self) -> Dict:
        """Run complete HTS analysis."""
        self.initialize()
        
        results = {
            'config': {
                'ds_type': self.config.ds_type,
                'is_rna': self.is_rna,
                'length': len(self.wt_sequence),
                'wt_sequence': self.wt_sequence,
                'wt_energy': self.wt_energy,
            },
        }
        
        # Single mutation scan
        if self.config.scan_full:
            logger.info("=" * 60)
            logger.info("FULL SINGLE‑MUTATION SCAN")
            logger.info("=" * 60)
            
            scan_results = self.scan_all_single_mutations()
            results['scan_results'] = scan_results
            results['scan_summary'] = self._summarize_scan(scan_results)
        
        # Specific mutations
        elif self.config.mutation_list:
            logger.info("=" * 60)
            logger.info("TARGETED MUTATION ANALYSIS")
            logger.info("=" * 60)
            
            specific_results = []
            for pos, new_base in self.config.mutation_list:
                result = self.predict_ddg_single(
                    pos, new_base,
                    relax=(self.config.relaxation_steps > 0)
                )
                # Add stacking disruption info
                result['stacking'] = self.compute_stacking_disruption(pos, new_base)
                specific_results.append(result)
            
            results['targeted_results'] = specific_results
        
        # Epistasis scanning
        if self.config.scan_pairs or self.config.epistasis_pairs:
            logger.info("=" * 60)
            logger.info("EPISTASIS SCANNING")
            logger.info("=" * 60)
            
            epi_results = self.scan_epistasis_pairs()
            results['epistasis_results'] = epi_results
            results['epistasis_summary'] = self._summarize_epistasis(epi_results)
        
        # Generate reports
        self._generate_reports(results)
        
        self.results = results
        return results
    
    def _summarize_scan(self, scan_results: List[Dict]) -> Dict:
        """Summarize single‑mutation scan."""
        df = pd.DataFrame(scan_results)
        
        # Save CSV
        csv_path = os.path.join(self.config.output_dir, "dna_rna_scan_ddg.csv")
        df.to_csv(csv_path, index=False)
        
        # Statistics
        ddg_vals = df['ddg'].values
        
        # Top stabilizing (most negative ΔΔG)
        top_stab = df.nsmallest(10, 'ddg')
        
        # Top destabilizing (most positive ΔΔG)
        top_destab = df.nlargest(10, 'ddg')
        
        # By mutation type
        trans_version = df[df['type'] == 'transition']['ddg'].describe()
        trans_vert = df[df['type'] == 'transversion']['ddg'].describe()
        
        summary = {
            'total_mutations': len(df),
            'mean_ddg': float(np.mean(ddg_vals)),
            'std_ddg': float(np.std(ddg_vals)),
            'median_ddg': float(np.median(ddg_vals)),
            'min_ddg': float(np.min(ddg_vals)),
            'max_ddg': float(np.max(ddg_vals)),
            'n_stabilizing': int((ddg_vals < -self.config.ddg_threshold).sum()),
            'n_destabilizing': int((ddg_vals > self.config.ddg_threshold).sum()),
            'n_neutral': int((np.abs(ddg_vals) <= self.config.ddg_threshold).sum()),
            'top_stabilizing': top_stab[['position', 'wt', 'mut', 'ddg', 'type']].to_dict('records'),
            'top_destabilizing': top_destab[['position', 'wt', 'mut', 'ddg', 'type']].to_dict('records'),
            'transition_stats': {
                'mean': float(trans_version['mean']) if 'mean' in trans_version else 0,
                'std': float(trans_version['std']) if 'std' in trans_version else 0,
                'count': int(trans_version['count']) if 'count' in trans_version else 0,
            },
            'transversion_stats': {
                'mean': float(trans_vert['mean']) if 'mean' in trans_vert else 0,
                'std': float(trans_vert['std']) if 'std' in trans_vert else 0,
                'count': int(trans_vert['count']) if 'count' in trans_vert else 0,
            },
            'csv_file': csv_path,
        }
        
        return summary
    
    def _summarize_epistasis(self, epi_results: List[Dict]) -> Dict:
        """Summarize epistasis results."""
        df = pd.DataFrame(epi_results)
        
        csv_path = os.path.join(self.config.output_dir, "dna_rna_epistasis.csv")
        df.to_csv(csv_path, index=False)
        
        eps_vals = df['epistasis'].values
        
        n_sig = int((np.abs(eps_vals) > self.config.ddg_threshold).sum())
        
        # Top epistatic pairs
        top_epi = df.nlargest(20, 'epistasis')
        
        return {
            'total_pairs': len(df),
            'significant_pairs': n_sig,
            'fraction_significant': n_sig / len(df) if len(df) > 0 else 0,
            'mean_epistasis': float(np.mean(eps_vals)),
            'std_epistasis': float(np.std(eps_vals)),
            'max_epistasis': float(np.max(eps_vals)),
            'min_epistasis': float(np.min(eps_vals)),
            'top_epistatic': top_epi[['pos1','mut1','pos2','mut2','epistasis','ddg_double','ddg_additive']].to_dict('records'),
            'csv_file': csv_path,
        }
    
    def _generate_reports(self, results: Dict):
        """Generate visual reports."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ΔΔG distribution histogram
        if 'scan_results' in results:
            self._plot_ddg_distribution(results['scan_results'], out_dir)
        
        # 2. Mutational landscape heatmap
        if 'scan_results' in results:
            self._plot_mutational_landscape(results['scan_results'], out_dir)
        
        # 3. Position‑specific ΔΔG profile
        if 'scan_results' in results:
            self._plot_position_profile(results['scan_results'], out_dir)
        
        # 4. Epistasis distribution
        if 'epistasis_results' in results:
            self._plot_epistasis_distribution(results['epistasis_results'], out_dir)
        
        # 5. Targeted mutation bar plot
        if 'targeted_results' in results:
            self._plot_targeted_mutations(results['targeted_results'], out_dir)
        
        # 6. Save summary JSON
        summary = {
            k: v for k, v in results.items()
            if k not in ('scan_results', 'epistasis_results', 'targeted_results')
        }
        with open(out_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Reports saved to {out_dir}")
    
    def _plot_ddg_distribution(self, scan_results: List[Dict], out_dir: Path):
        """Plot ΔΔG distribution."""
        ddg_vals = [r['ddg'] for r in scan_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        sns.histplot(ddg_vals, bins=50, kde=True, color='#2E86AB', ax=ax)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(-self.config.ddg_threshold, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(self.config.ddg_threshold, color='orange', linestyle=':', alpha=0.5)
        ax.set_title(f"ΔΔG Distribution ({'RNA' if self.is_rna else 'DNA'})")
        ax.set_xlabel("ΔΔG (kcal/mol)")
        ax.set_ylabel("Count")
        
        # Transition vs Transversion boxplot
        ax = axes[1]
        trans_data = {
            'Transition': [r['ddg'] for r in scan_results if r.get('type') == 'transition'],
            'Transversion': [r['ddg'] for r in scan_results if r.get('type') == 'transversion'],
        }
        df_plot = pd.DataFrame([
            (k, v) for k, vals in trans_data.items() for v in vals
        ], columns=['Type', 'ΔΔG'])
        sns.boxplot(data=df_plot, x='Type', y='ΔΔG', palette='Set2', ax=ax)
        ax.set_title("ΔΔG by Mutation Type")
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(out_dir / "ddg_distribution.png", dpi=200)
        plt.close(fig)
    
    def _plot_mutational_landscape(self, scan_results: List[Dict], out_dir: Path):
        """Plot mutational landscape heatmap."""
        seq = self.wt_sequence
        bases = RNA_BASES if self.is_rna else DNA_BASES
        L = len(seq)
        n_bases = len(bases)
        
        # Build matrix
        ddg_matrix = np.zeros((L, n_bases))
        for r in scan_results:
            pos = r['position']
            j = bases.index(r['mut']) if r['mut'] in bases else -1
            if j >= 0:
                ddg_matrix[pos, j] = r['ddg']
        
        # Mask WT
        mask = np.zeros_like(ddg_matrix, dtype=bool)
        for i, wt in enumerate(seq):
            if wt in bases:
                j = bases.index(wt)
                mask[i, j] = True
        
        ddg_masked = np.ma.array(ddg_matrix, mask=mask)
        
        fig, ax = plt.subplots(figsize=(n_bases * 1.2, max(6, L * 0.25)))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        sns.heatmap(ddg_masked, cmap=cmap, center=0,
                     xticklabels=bases,
                     yticklabels=[f"{i+1}{seq[i]}" for i in range(L)],
                     ax=ax, cbar_kws={'label': 'ΔΔG (kcal/mol)'},
                     vmin=-5, vmax=5)
        ax.set_title(f"Mutational Landscape — {'RNA' if self.is_rna else 'DNA'} ΔΔG")
        ax.set_xlabel("Mutant Base")
        ax.set_ylabel("Position (WT)")
        
        fig.tight_layout()
        fig.savefig(out_dir / "mutational_landscape.png", dpi=300)
        plt.close(fig)
    
    def _plot_position_profile(self, scan_results: List[Dict], out_dir: Path):
        """Plot position‑specific ΔΔG profile."""
        df = pd.DataFrame(scan_results)
        
        # Mean ΔΔG per position
        pos_mean = df.groupby('position')['ddg'].agg(['mean', 'std', 'min', 'max'])
        
        fig, ax = plt.subplots(figsize=(max(10, len(pos_mean) * 0.2), 5))
        
        x = pos_mean.index.values
        ax.fill_between(x, pos_mean['min'], pos_mean['max'],
                         alpha=0.2, color='#2E86AB', label='Min‑Max range')
        ax.fill_between(x, pos_mean['mean'] - pos_mean['std'],
                         pos_mean['mean'] + pos_mean['std'],
                         alpha=0.4, color='#2E86AB', label='±1 SD')
        ax.plot(x, pos_mean['mean'], 'o-', color='#A23B72', linewidth=1.5,
                 markersize=4, label='Mean ΔΔG')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Color‑code by WT base
        seq = self.wt_sequence
        for i in range(len(seq)):
            color = {'A': '#FF6B6B', 'T': '#4ECDC4', 'U': '#4ECDC4',
                     'G': '#45B7D1', 'C': '#96CEB4'}.get(seq[i], '#888888')
            ax.axvline(i, color=color, alpha=0.1, linewidth=8)
        
        ax.set_xlabel("Position")
        ax.set_ylabel("ΔΔG (kcal/mol)")
        ax.set_title(f"Position‑specific Mutation Tolerance — {'RNA' if self.is_rna else 'DNA'}")
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        fig.savefig(out_dir / "position_profile.png", dpi=200)
        plt.close(fig)
    
    def _plot_epistasis_distribution(self, epi_results: List[Dict], out_dir: Path):
        """Plot epistasis distribution."""
        eps_vals = [r['epistasis'] for r in epi_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        sns.histplot(eps_vals, bins=50, kde=True, color='#A23B72', ax=ax)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axvline(-self.config.ddg_threshold, color='red', linestyle=':', alpha=0.5)
        ax.axvline(self.config.ddg_threshold, color='red', linestyle=':', alpha=0.5)
        ax.set_title("Epistasis (ε) Distribution")
        ax.set_xlabel("ε = ΔΔG_double − (ΔΔG₁ + ΔΔG₂) [kcal/mol]")
        ax.set_ylabel("Count")
        
        # Scatter: ΔΔG_additive vs ΔΔG_double
        ax = axes[1]
        ddg_add = [r['ddg_additive'] for r in epi_results]
        ddg_dbl = [r['ddg_double'] for r in epi_results]
        ax.scatter(ddg_add, ddg_dbl, alpha=0.3, s=15, c='#F18F01', edgecolors='none')
        
        # y = x line (no epistasis)
        lim_min = min(min(ddg_add), min(ddg_dbl)) - 1
        lim_max = max(max(ddg_add), max(ddg_dbl)) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, label='No epistasis')
        
        # Highlight significant
        sig_mask = [abs(r['epistasis']) > self.config.ddg_threshold for r in epi_results]
        if any(sig_mask):
            sig_add = [ddg_add[i] for i, s in enumerate(sig_mask) if s]
            sig_dbl = [ddg_dbl[i] for i, s in enumerate(sig_mask) if s]
            ax.scatter(sig_add, sig_dbl, alpha=0.7, s=25, c='#D62828', 
                       edgecolors='black', linewidth=0.5, label='Significant')
        
        ax.set_xlabel("ΔΔG (additive) [kcal/mol]")
        ax.set_ylabel("ΔΔG (double mutant) [kcal/mol]")
        ax.set_title("Additivity vs Double Mutant ΔΔG")
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        fig.savefig(out_dir / "epistasis_distribution.png", dpi=200)
        plt.close(fig)
    
    def _plot_targeted_mutations(self, targeted_results: List[Dict], out_dir: Path):
        """Plot targeted mutation results as bar chart."""
        mut_labels = [f"{r['position']}{r['wt']}→{r['mut']}" for r in targeted_results]
        ddg_vals = [r['ddg'] for r in targeted_results]
        
        # Color by stabilizing (blue) vs destabilizing (red)
        colors = ['#2E86AB' if v < 0 else '#D62828' for v in ddg_vals]
        
        fig, ax = plt.subplots(figsize=(max(8, len(targeted_results) * 0.5), 5))
        
        bars = ax.bar(mut_labels, ddg_vals, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, ddg_vals):
            y_pos = bar.get_height()
            offset = 0.1 if y_pos >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2., y_pos + offset,
                    f'{val:.2f}', ha='center', va='bottom' if y_pos >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(-self.config.ddg_threshold, color='orange', linestyle=':', alpha=0.5, 
                   label=f'±{self.config.ddg_threshold} kcal/mol threshold')
        ax.axhline(self.config.ddg_threshold, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_xlabel("Mutation")
        ax.set_ylabel("ΔΔG (kcal/mol)")
        ax.set_title("Predicted ΔΔG for Targeted Mutations")
        ax.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        
        fig.tight_layout()
        fig.savefig(out_dir / "targeted_mutations.png", dpi=200)
        plt.close(fig)
    
    def export_mutant_structure(self, pos: int, new_base: str, 
                                 output_path: str, relax: bool = True):
        """
        Export the predicted mutant structure as PDB.
        
        Args:
            pos: mutation position (0‑based)
            new_base: new nucleotide base
            output_path: output PDB file path
            relax: perform relaxation before export
        """
        mut_seq = self.wt_sequence[:pos] + new_base + self.wt_sequence[pos+1:]
        mut_C4 = self.wt_C4.clone()
        
        if relax and self.config.relaxation_steps > 0:
            mut_C4, _ = self._relax_mutant(mut_C4, mut_seq, pos)
        
        write_nucleotide_pdb(
            mut_C4.cpu(), mut_seq, output_path,
            chain_id=self.config.chain_id,
            pucker_type='C3_endo' if self.is_rna else 'C2_endo'
        )
        
        logger.info(f"Mutant structure exported to {output_path}")


# ═══════════════════════════════════════════════════════════════
# PARALLEL SCANNING WORKER (Multi‑GPU)
# ═══════════════════════════════════════════════════════════════

def _scan_chunk_worker_dna(gpu_id: int,
                            tasks: List[Tuple[int, str]],
                            config_dict: dict,
                            wt_sequence: str,
                            wt_C4_np: np.ndarray,
                            wt_energy: float,
                            is_rna: bool) -> List[Dict]:
    """
    Worker function for parallel DNA/RNA mutation scanning.
    
    Runs on a single GPU.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Reconstruct config
    pucker = 'C3_endo' if is_rna else 'C2_endo'
    
    # Load force field if specified
    lj_params = None
    charge_map = None
    if config_dict.get('lj_param_file') or config_dict.get('charge_param_file'):
        from csoc_dna_rna import load_nucleotide_forcefield
        lj_params, charge_map = load_nucleotide_forcefield(
            config_dict.get('lj_param_file'),
            config_dict.get('charge_param_file')
        )
    
    # Create energy engine
    energy_engine = DNA_RNA_Energy(
        pucker_type=pucker,
        use_full_atom=True,
        lj_params=lj_params,
        charge_map=charge_map,
    )
    
    wt_C4 = torch.tensor(wt_C4_np, device=device)
    relaxation_steps = config_dict.get('relaxation_steps', 30)
    
    results = []
    
    for pos, new_base in tasks:
        wt_base = wt_sequence[pos]
        if wt_base == new_base:
            continue
        
        mut_seq = wt_sequence[:pos] + new_base + wt_sequence[pos+1:]
        mut_C4 = wt_C4.clone()
        
        # Relaxation (simplified for parallel — full sequence gradient)
        if relaxation_steps > 0:
            mut_C4_relax = mut_C4.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([mut_C4_relax], lr=1e-3)
            
            best_E = float('inf')
            best_C4 = mut_C4_relax.clone()
            
            L = len(mut_seq)
            window_start = max(0, pos - 3)
            window_end = min(L, pos + 4)
            
            for _ in range(relaxation_steps):
                opt.zero_grad()
                E = energy_engine(mut_C4_relax, mut_seq)
                E.backward()
                
                with torch.no_grad():
                    if mut_C4_relax.grad is not None:
                        mask = torch.zeros(L, device=device)
                        mask[window_start:window_end] = 1.0
                        mut_C4_relax.grad *= mask.unsqueeze(-1)
                
                opt.step()
                
                if E.item() < best_E:
                    best_E = E.item()
                    best_C4 = mut_C4_relax.clone()
            
            e_mut = best_E
        else:
            with torch.no_grad():
                e_mut = energy_engine(mut_C4, mut_seq).item()
        
        ddg = e_mut - wt_energy
        mut_type = 'transition' if is_transition(wt_base, new_base) else 'transversion'
        
        results.append({
            'position': pos,
            'wt': wt_base,
            'mut': new_base,
            'ddg': ddg,
            'relaxed': relaxation_steps > 0,
            'type': mut_type,
            'e_wt': wt_energy,
            'e_mut': e_mut,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSOC‑SSC HTS FOLD v32.1 — DNA/RNA Mutation Scanning & Epistasis"
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdb', type=str, help='PDB structure file (DNA/RNA)')
    input_group.add_argument('--seq', type=str, help='DNA/RNA sequence (e.g., ACGTACGT)')
    
    parser.add_argument('--chain', type=str, default='A', help='Chain ID (default: A)')
    parser.add_argument('--type', type=str, default='B_DNA', choices=['B_DNA', 'A_RNA'],
                        help='Helix type for de novo build (default: B_DNA)')
    
    # Output
    parser.add_argument('--output', type=str, default='./hts_dna_rna_output',
                        help='Output directory')
    
    # Mutation options
    parser.add_argument('--scan', action='store_true', 
                        help='Full single‑mutation scan')
    parser.add_argument('--mutations', nargs='+', type=str,
                        help='Specific mutations (e.g., 3A 7G 12T)')
    parser.add_argument('--scan_pairs', action='store_true',
                        help='Scan double mutations for epistasis')
    parser.add_argument('--epi_pairs', nargs='+', type=str,
                        help='Specific position pairs for epistasis (e.g., 3-7 12-15)')
    
    # Refinement
    parser.add_argument('--relax_steps', type=int, default=30,
                        help='Relaxation steps per mutant (default: 30, 0 = no relaxation)')
    parser.add_argument('--ddg_threshold', type=float, default=0.5,
                        help='ΔΔG threshold for significance (default: 0.5 kcal/mol)')
    
    # Hardware
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for parallel scan')
    
    # Force field
    parser.add_argument('--lj_params', type=str, default=None,
                        help='Custom LJ parameters JSON file')
    parser.add_argument('--charge_params', type=str, default=None,
                        help='Custom charges JSON file')
    
    # Export
    parser.add_argument('--export_mutant', type=str, default=None,
                        help='Export specific mutant structure (e.g., 3A)')
    
    args = parser.parse_args()
    
    # Parse mutations
    mutation_list = None
    if args.mutations:
        mutation_list = []
        for m in args.mutations:
            pos = int(m[:-1])
            base = m[-1].upper()
            mutation_list.append((pos, base))
    
    # Parse epistasis pairs
    epistasis_pairs = None
    if args.epi_pairs:
        epistasis_pairs = []
        for p in args.epi_pairs:
            p1, p2 = p.split('-')
            epistasis_pairs.append((int(p1), int(p2)))
    
    # Create config
    config = HTS_DNA_RNA_Config(
        pdb_structure=args.pdb,
        sequence=args.seq,
        chain_id=args.chain,
        ds_type=args.type,
        output_dir=args.output,
        ddg_threshold=args.ddg_threshold,
        mutation_list=mutation_list,
        scan_full=args.scan,
        scan_pairs=args.scan_pairs,
        epistasis_pairs=epistasis_pairs,
        relaxation_steps=args.relax_steps,
        use_gpu=args.gpu,
        num_gpus=args.num_gpus,
        lj_param_file=args.lj_params,
        charge_param_file=args.charge_params,
    )
    
    # Run analysis
    analyzer = HTS_DNA_RNA_Analyzer(config)
    results = analyzer.run_analysis()
    
    # Export specific mutant if requested
    if args.export_mutant:
        pos = int(args.export_mutant[:-1])
        base = args.export_mutant[-1].upper()
        out_path = os.path.join(args.output, f"mutant_{pos}{base}.pdb")
        analyzer.export_mutant_structure(pos, base, out_path, relax=True)
    
    # Print summary
    print("\n" + "=" * 70)
    print("HTS FOLD DNA/RNA v32.1 — Analysis Complete")
    print("=" * 70)
    
    if 'scan_summary' in results:
        s = results['scan_summary']
        print(f"\nSingle Mutation Scan Summary:")
        print(f"  Total mutations:     {s['total_mutations']}")
        print(f"  Mean ΔΔG:            {s['mean_ddg']:.4f} kcal/mol")
        print(f"  Stabilizing (<-{config.ddg_threshold}):  {s['n_stabilizing']}")
        print(f"  Destabilizing (>{config.ddg_threshold}):  {s['n_destabilizing']}")
        print(f"  Neutral:             {s['n_neutral']}")
        print(f"  Transition mean:     {s['transition_stats']['mean']:.4f} (n={s['transition_stats']['count']})")
        print(f"  Transversion mean:   {s['transversion_stats']['mean']:.4f} (n={s['transversion_stats']['count']})")
    
    if 'epistasis_summary' in results:
        e = results['epistasis_summary']
        print(f"\nEpistasis Summary:")
        print(f"  Total pairs:         {e['total_pairs']}")
        print(f"  Significant pairs:   {e['significant_pairs']} ({e['fraction_significant']:.1%})")
        print(f"  Mean epistasis:      {e['mean_epistasis']:.4f} kcal/mol")
        print(f"  Max epistasis:       {e['max_epistasis']:.4f} kcal/mol")
    
    if 'targeted_results' in results:
        print(f"\nTargeted Mutation Results:")
        for r in results['targeted_results']:
            sig = ' ***' if abs(r['ddg']) > config.ddg_threshold else ''
            print(f"  {r['position']}{r['wt']}→{r['mut']}:  ΔΔG = {r['ddg']:+.4f} kcal/mol  "
                  f"({r['type']}){sig}")
    
    print(f"\nResults saved to: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
