# =============================================================================
# CSOC‑SSC v30.4 — Antibody Design & DNA Origami Extension
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Extends CSOC‑SSC v30.1‑v32.1 with:
#   • Antibody affinity maturation pipeline
#   • Inverse folding for CDR loops (physics‑based + ProteinMPNN wrapper)
#   • Simple DNA origami staple routing (grid‑based)
#   • Crossover energy term for DNA junctions
# =============================================================================

import math, random, os, json, itertools
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

# Import existing modules (make sure they are in the same directory)
try:
    from csoc_v30_1 import CSOCSSC_V30_1, V30_1Config, total_physics_energy_v30_1
    from csoc_hts_v31_1 import HTSAnalyzerV31_1, HTSConfig as HTSProteinConfig
    from csoc_dna_rna import (
        DNA_RNA_Energy,
        build_dna_helix,
        build_double_strand_helix,
        write_nucleotide_pdb,
        energy_base_pairing,
        energy_base_stacking,
        energy_backbone_c4_bond,
        WC_PAIRS,
        BASE_STACKING,
        DNA_VOCAB,
    )
    from csoc_hts_dna_rna_v32_1 import HTS_DNA_RNA_Analyzer, HTS_DNA_RNA_Config
    HAS_DNA_RNA = True
except ImportError:
    HAS_DNA_RNA = False
    print("Warning: Some DNA/RNA modules not found; DNA origami features will be limited.")

# ═══════════════════════════════════════════════════════════════
# 1. ANTIBODY AFFINITY MATURATION PIPELINE
# ═══════════════════════════════════════════════════════════════

class AffinityMaturation:
    """
    Automatically find mutations that improve antibody‑antigen binding.
    
    Uses HTS FOLD v31.1 to scan CDR loops and ranks mutations by ΔΔG.
    """
    
    def __init__(self, antibody_pdb: str, antigen_chain: str = 'A',
                 antibody_chains: List[str] = ['H','L'],
                 cdr_regions: Dict[str, Tuple[int,int]] = None):
        """
        Args:
            antibody_pdb: PDB file of antibody‑antigen complex
            antigen_chain: chain ID of antigen
            antibody_chains: list of antibody chain IDs
            cdr_regions: dict mapping chain -> (start_res, end_res) for CDR loops
                         if None, will auto‑detect from Chothia numbering
        """
        self.pdb = antibody_pdb
        self.antigen_chain = antigen_chain
        self.antibody_chains = antibody_chains
        self.cdr_regions = cdr_regions
        
        # Will be loaded
        self.complex_model = None
        self.wt_energy = None
    
    def auto_detect_cdrs(self, sequences: Dict[str, str]) -> Dict[str, List[Tuple[int,int]]]:
        """
        Simple CDR detection using Chothia‑like rules.
        (In practice, you would use ANARCI or similar)
        """
        # Placeholder: just return whole variable domain as CDR region
        cdrs = {}
        for chain, seq in sequences.items():
            # Assume CDRs are in the variable domain (first ~120 residues)
            cdrs[chain] = [(25, 35), (50, 60), (95, 110)]
        return cdrs
    
    def run(self, n_top_mutations: int = 20, relax_steps: int = 30,
            output_dir: str = "./affinity_maturation"):
        """
        Scan all single mutations in CDR regions and return top candidates.
        
        Returns:
            List of (mutation, ddG) sorted by most stabilizing (negative)
        """
        import pandas as pd
        from csoc_v30_1 import MultimerPDBFetcher, CSOCSSC_V30_1, V30_1Config
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load structure
        logger.info(f"Loading antibody complex from {self.pdb}")
        backbones, chain_ids = MultimerPDBFetcher.fetch(self.pdb)
        sequences = {b.chain_id: b.seq for b in backbones}
        
        # Auto‑detect CDRs if not provided
        if self.cdr_regions is None:
            self.cdr_regions = self.auto_detect_cdrs(sequences)
        
        # Create HTS analyzer config
        # We'll scan only CDR positions
        all_mutations = []
        for chain in self.antibody_chains:
            if chain in sequences and chain in self.cdr_regions:
                seq = sequences[chain]
                for start, end in self.cdr_regions[chain]:
                    for pos in range(start, min(end, len(seq))):
                        wt_aa = seq[pos]
                        for new_aa in "ACDEFGHIKLMNPQRSTVWY":
                            if new_aa != wt_aa:
                                all_mutations.append((chain, pos, new_aa))
        
        logger.info(f"Total CDR mutations to scan: {len(all_mutations)}")
        
        # Use HTS FOLD v31.1 infrastructure
        config = HTSProteinConfig(
            pdb_structure=self.pdb,
            output_dir=output_dir,
            mutation_list=[(p, aa) for (_, p, aa) in all_mutations],
            relaxation_steps=relax_steps,
            use_gpu=torch.cuda.is_available(),
        )
        analyzer = HTSAnalyzerV31_1(config)
        analyzer.load_v30_engine()
        
        results = []
        for (chain, pos, new_aa) in all_mutations:
            # HTS expects single chain; we can refine per chain
            # For full complex energy, we need to refine the whole multimer
            # This is a simplification – a full implementation would use
            # refine_multimer with the mutation and compute binding ΔΔG
            wt_energy = analyzer.wt_energy
            ddg = analyzer.predict_ddg_mutation(pos, new_aa, relax=True)
            results.append({
                'chain': chain,
                'position': pos,
                'mutation': f"{sequences[chain][pos]}{pos+1}{new_aa}",
                'ddg': ddg,
            })
        
        # Sort by ΔΔG (most negative = best)
        results.sort(key=lambda x: x['ddg'])
        
        # Save
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "affinity_maturation_results.csv"), index=False)
        
        # Top N
        top = results[:n_top_mutations]
        logger.info(f"Top {n_top_mutations} mutations:")
        for r in top:
            logger.info(f"  {r['mutation']}: ΔΔG = {r['ddg']:.4f} kcal/mol")
        
        return top


# ═══════════════════════════════════════════════════════════════
# 2. INVERSE FOLDING (CDR Sequence Design)
# ═══════════════════════════════════════════════════════════════

class InverseFolding:
    """
    Design amino‑acid sequences that fold into a given backbone structure.
    
    Two modes:
      - 'physics': Monte Carlo sequence optimization using CSOC energy
      - 'proteinmpnn': use ProteinMPNN (if installed) for one‑shot design
    """
    
    def __init__(self, mode: str = 'physics'):
        self.mode = mode
        if mode == 'proteinmpnn':
            self._init_proteinmpnn()
    
    def _init_proteinmpnn(self):
        """Try to import ProteinMPNN."""
        try:
            import protein_mpnn
            self.proteinmpnn = protein_mpnn
            self.mpnn_available = True
        except ImportError:
            print("ProteinMPNN not found. Install with: pip install proteinmpnn")
            print("Falling back to physics mode.")
            self.mode = 'physics'
            self.mpnn_available = False
    
    def design_sequence(self, backbone_coords: torch.Tensor,
                        target_sequence: str = None,
                        positions_to_design: List[int] = None,
                        num_iterations: int = 500,
                        temperature: float = 1.0) -> str:
        """
        Design an amino‑acid sequence for the given backbone.
        
        Args:
            backbone_coords: [L, 3] CA coordinates (or C4' for DNA/RNA)
            target_sequence: initial sequence (can be all 'X' for unknown)
            positions_to_design: list of positions to redesign (default: all)
            num_iterations: for physics mode
            temperature: Monte Carlo temperature
            
        Returns:
            Designed sequence string
        """
        if self.mode == 'proteinmpnn' and self.mpnn_available:
            return self._design_proteinmpnn(backbone_coords, positions_to_design)
        else:
            return self._design_physics(backbone_coords, target_sequence,
                                       positions_to_design, num_iterations, temperature)
    
    def _design_physics(self, coords, target_seq, design_positions,
                        num_iter, T):
        """Monte Carlo sequence optimization using CSOC energy."""
        L = len(coords)
        if target_seq is None:
            target_seq = 'G' * L  # start with all glycine (flexible)
        if design_positions is None:
            design_positions = list(range(L))
        
        seq_list = list(target_seq)
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Compute initial energy
        best_seq = target_seq
        best_energy = self._compute_energy(coords, best_seq)
        current_seq = best_seq
        current_energy = best_energy
        
        for it in range(num_iter):
            # Random position
            pos = random.choice(design_positions)
            old_aa = current_seq[pos]
            new_aa = random.choice(aa_list)
            if new_aa == old_aa:
                continue
            
            # Mutate
            mut_seq = current_seq[:pos] + new_aa + current_seq[pos+1:]
            mut_energy = self._compute_energy(coords, mut_seq)
            
            delta = mut_energy - current_energy
            
            # Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_seq = mut_seq
                current_energy = mut_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_seq = current_seq
        
        return best_seq
    
    def _compute_energy(self, coords, seq):
        """Use v30.1 total physics energy (backbone only)."""
        from csoc_v30_1 import (
            reconstruct_backbone, compute_phi_psi, energy_rama_vectorized,
            energy_clash_sparse, energy_solvent_sparse, sparse_edges,
            V30_1Config
        )
        cfg = V30_1Config()
        L = len(seq)
        alpha = torch.ones(L)
        atoms = reconstruct_backbone(coords)
        phi, psi = compute_phi_psi(atoms)
        ei, ed = sparse_edges(coords, cfg.sparse_cutoff, cfg.max_neighbors)
        
        E = energy_rama_vectorized(phi, psi, seq, alpha, cfg)
        E += energy_clash_sparse(coords, alpha, ei, ed, cfg)
        E += energy_solvent_sparse(coords, seq, ei, ed, cfg)
        return E.item()
    
    def _design_proteinmpnn(self, coords, design_positions):
        """Use ProteinMPNN for design (requires GPU)."""
        # Convert coords to PDB‑like format
        # This is a wrapper – actual implementation depends on proteinmpnn API
        raise NotImplementedError("ProteinMPNN wrapper requires full atom types; "
                                  "please use the physics mode or adapt manually.")


# ═══════════════════════════════════════════════════════════════
# 3. DNA ORIGAMI STAPLE ROUTING
# ═══════════════════════════════════════════════════════════════

class DNAOrigamiDesigner:
    """
    Simple grid‑based DNA origami design.
    
    Given a set of vertices (in 2D/3D), creates scaffold and staple strands
    using parallel helices connected by crossovers.
    """
    
    def __init__(self, shape: List[Tuple[float,float,float]],
                 helix_axis: str = 'z',
                 n_helices: int = 6,
                 scaffold_seq: str = None):
        """
        Args:
            shape: list of (x,y,z) vertices defining the target shape
            helix_axis: direction of helices ('x','y','z')
            n_helices: number of parallel helices
            scaffold_seq: sequence of scaffold strand (usually M13mp18)
        """
        self.shape = shape
        self.helix_axis = helix_axis
        self.n_helices = n_helices
        self.scaffold_seq = scaffold_seq
        if scaffold_seq is None:
            # M13mp18 partial sequence (first 1000 nt)
            self.scaffold_seq = "AATGCTACTACTATTAGTAGAATTGATGCCACCTTTTCAGCTCGCGCCCCAAATGAAAATATAGCTAAACAGGTTATTGACCATTTGCGAAATGTATCTAATGGTCAAACTAAATCTACTCGTTCGCAGAATTGGGAATCAACTGTTACATGGAATGAAACTTCCAGACACCGTACTTTAGTTGCATATTTAAAACATGTTGAGCTACAGCATTATATTCAGCAATTAAGCTCTAAGCCATCCGCAAAAATGACCTCTTATCAAAAGGAGCAATTAAAGGTACTCTCTAATCCTGACCTGTTGGAGTTTGCTTCCGGTCTGGTTCGCTTTGAAGCTCGAATTAAAACGCGATATTTGAAGTCTTTCGGGCTTCCTCTTAATCTTTTTGATGCAATCCGCTTTGCTTCTGACTATAATAGTCAGGGTAAAGACCTGATTTTTGATTTATGGTCATTCTCGTTTTCTGAACTGTTTAAAGCATTTGAGGGGGATTCAATGAATATTTATGACGATTCCGCAGTATTGGACGCTATCCAGTCTAAACATTTTACTATTACCCCCTCTGGCAAAACTTCTTTTGCAAAAGCCTCTCGCTATTTTGGTTTTTATCGTCGTCTGGTAAACGAGGGTTATGATAGTGTTGCTCTTACTATGCCTCGTAATTCCTTTTGGCGTTATGTATCTGCATTAGTTGAATGTGGTATTCCTAAATCTCAACTGATGAATCTTTCTACCTGTAATAATGTTGTTCCGTTAGTTCGTTTTATTAACGTAGATTTTTCTTCCCAACGTCCTGACTGGTATAATGAGCCAGTTCTTAAAATCGCATAAGGTAATTCACAATGATTAAAGTTGAAATTAAACCATCTCAAGCCCAATTTACTACTCGTTCTGGTGTTTCTCGTCAGGGCAAGCCTTATTCACTGAATGAGCAGCTTTGTTACGTTGATTTGGGTAATGAATATCCGGTTCTTGTCAAGATTACTCTTGATGAAGGTCAGCCAGCCTATGCGCCTGGTCTGTACACCGTTCATCTGTCCTCTTTCAAAGTTGGTCAGTTCGGTTCCCTTATGATTGACCGTCTGCGCCTCGTTCCGGCTAAGTAACATGGAGCAGGTCGCGGATTTCGACACAATTTATCAGGCGATGATACAAATCTCCGTTGTACTTTGTTTCGCGCTTGGTATAATCGCTGGGGGTCAAAGATGAGTGTTTTAGTGTATTCTTTTGCCTCTTTCGTTTTAGGTTGGTGCCTTCGTAGTGGCATTACGTATTTTACCCGTTTAATGGAAACTTCCTCATGAAAAAGTCTTTAGTCCTCAAAGCCTCTGTAGCCGTTGCTACCCTCGTTCCGATGCTGTCTTTCGCTGCTGAGGGTGACGATCCCGCAAAAGCGGCCTTTAACTCCCTGCAAGCCTCAGCGACCGAATATATCGGTTATGCGTGGGCGATGGTTGTTGTCATTGTCGGCGCAACTATCGGTATCAAGCTGTTTAAGAAATTCACCTCGAAAGCAAGCTGATAAACCGATACAATTAAAGGCTCCTTTTGGAGCCTTTTTTTTTGGAGATTTTCAACGTGAAAAAATTATTATTCGCAATTCCTTTAGTTGTTCCTTTCTATTCTCACTCCGCTGAAACTGTTGAAAGTTGTTTAGCAAAATCCCATACAGAAAATTCATTTACTAACGTCTGGAAAGACGACAAAACTTTAGATCGTTACGCTAACTATGAGGGCTGTCTGTGGAATGCTACAGGCGTTGTAGTTTGTACTGGTGACGAAACTCAGTGTTACGGTACATGGGTTCCTATTGGGCTTGCTATCCCTGAAAATGAGGGTGGTGGCTCTGAGGGTGGCGGTTCTGAGGGTGGCGGTTCTGAGGGTGGCGGTACTAAACCTCCTGAGTACGGTGATACACCTATTCCGGGCTATACTTATATCAACCCTCTCGACGGCACTTATCCGCCTGGTACTGAGCAAAACCCCGCTAATCCTAATCCTTCTCTTGAGGAGTCTCAGCCTCTTAATACTTTCATGTTTCAGAATAATAGGTTCCGAAATAGGCAGGGGGCATTAACTGTTTATACGGGCACTGTTACTCAAGGCACTGACCCCGTTAAAACTTATTACCAGTACACTCCTGTATCATCAAAAGCCATGTATGACGCTTACTGGAACGGTAAATTCAGAGACTGCGCTTTCCATTCTGGCTTTAATGAGGATTTATTTGTTTGTGAATATCAAGGCCAATCGTCTGACCTGCCTCAACCTCCTGTCAATGCTGGCGGCGGCTCTGGTGGTGGTTCTGGTGGCGGCTCTGAGGGTGGTGGCTCTGAGGGTGGCGGTTCTGAGGGTGGTGGCTCTGAGGGAGGCGGTTCCGGTGGTGGCTCTGGTTCCGGTGATTTTGATTATGAAAAGATGGCAAACGCTAATAAGGGGGCTATGACCGAAAATGCCGATGAAAACGCGCTACAGTCTGACGCTAAAGGCAAACTTGATTCTGTCGCTACTGATTACGGTGCTGCTATCGATGGTTTCATTGGTGACGTTTCCGGCCTTGCTAATGGTAATGGTGCTACTGGTGATTTTGCTGGCTCTAATTCCCAAATGGCTCAAGTCGGTGACGGTGATAATTCACCTTTAATGAATAATTTCCGTCAATATTTACCTTCCCTCCCTCAATCGGTTGAATGTCGCCCTTTTGTCTTTGGCGCTGGTAAACCATATGAATTTTCTATTGATTGTGACAAAATAAACTTATTCCGTGGTGTCTTTGCGTTTCTTTTATATGTTGCCACCTTTATGTATGTATTTTCTACGTTTGCTAACATACTGCGTAATAAGGAGTCTTAATCATGCCAGTTCTTTTGGGTATTCCGTTATTATTGCGTTTCCTCGGTTTCCTTCTGGTAACTTTGTTCGGCTATCTGCTTACTTTTCTTAAAAAGGGCTTCGGTAAGATAGCTATTGCTATTTCATTGTTTCTTGCTCTTATTATTGGGCTTAACTCAATTCTTGTGGGTTATCTCTCTGATATTAGCGCTCAATTACCCTCTGACTTTGTTCAGGGTGTTCAGTTAATTCTCCCGTCTAATGCGCTTCCCTGTTTTTATGTTATTCTCTCTGTAAAGGCTGCTATTTTCATTTTTGACGTTAAACAAAAAATCGTTTCTTATTTGGATTGGGATAAATAATATGGCTGTTTATTTTGTAACTGGCAAATTAGGCTCTGGAAAGACGCTCGTTAGCGTTGGTAAGATTCAGGATAAAATTGTAGCTGGGTGCAAAATAGCAACTAATCTTGATTTAAGGCTTCAAAACCTCCCGCAAGTCGGGAGGTTCGCTAAAACGCCTCGCGTTCTTAGAATACCGGATAAGCCTTCTATATCTGATTTGCTTGCTATTGGGCGCGGTAATGATTCCTACGATGAAAATAAAAACGGCTTGCTTGTTCTCGATGAGTGCGGTACTTGGTTTAATACCCGTTCTTGGAATGATAAGGAAAGACAGCCGATTATTGATTGGTTTCTACATGCTCGTAAATTAGGATGGGATATTATTTTTCTTGTTCAGGACTTATCTATTGTTGATAAACAGGCGCGTTCTGCATTAGCTGAACATGTTGTTTATTGTCGTCGTCTGGACAGAATTACTTTACCTTTTGTCGGTACTTTATATTCTCTTATTACTGGCTCGAAAATGCCTCTGCCTAAATTACATGTTGGCGTTGTTAAATATGGCGATTCTCAATTAAGCCCTACTGTTGAGCGTTGGCTTTATACTGGTAAGAATTTGTATAACGCATATGATACTAAACAGGCTTTTTCTAGTAATTATGATTCCGGTGTTTATTCTTATTTAACGCCTTATTTATCACACGGTCGGTATTTCAAACCATTAAATTTAGGTCAGAAGATGAAATTAACTAAAATATATTTGAAAAAGTTTTCTCGCGTTCTTTGTCTTGCGATTGGATTTGCATCAGCATTTACATATAGTTATATAACCCAACCTAAGCCGGAGGTTAAAAAGGTAGTCTCTCAGACCTATGATTTTGATAAATTCACTATTGACTCTTCTCAGCGTCTTAATCTAAGCTATCGCTATGTTTTCAAGGATTCTAAGGGAAAATTAATTAATAGCGACGATTTACAGAAGCAAGGTTATTCACTCACATATATTGATTTATGTACTGTTTCCATTAAAAAAGGTAATTCAAATGAAATTGTTAAATGTAATTAATTTTGTTTTCTTGATGTTTGTTTCATCATCTTCTTTTGCTCAGGTAATTGAAATGAATAATTCGCCTCTGCGCGATTTTGTAACTTGGTATTCAAAGCAATCAGGCGAATCCGTTATTGTTTCTCCCGATGTAAAAGGTACTGTTACTGTATATTCATCTGACGTTAAACCTGAAAATCTACGCAATTTCTTTATTTCTGTTTTACGTGCAAATAATTTTGATATGGTAGGTTCTAACCCTTCCATTATTCAGAAGTATAATCCAAACAATCAGGATTATATTGATGAATTGCCATCATCTGATAATCAGGAATATGATGATAATTCCGCTCCTTCTGGTGGTTTCTTTGTTCCGCAAAATGATAATGTTACTCAAACTTTTAAAATTAATAACGTTCGGGCAAAGGATTTAATACGAGTTGTCGAATTGTTTGTAAAGTCTAATACTTCTAAATCCTCAAATGTATTATCTATTGACGGCTCTAATCTATTAGTTGTTAGTGCTCCTAAAGATATTTTAGATAACCTTCCTCAATTCCTTTCAACTGTTGATTTGCCAACTGACCAGATATTGATTGAGGGTTTGATATTTGAGGTTCAGCAAGGTGATGCTTTAGATTTTTCATTTGCTGCTGGCTCTCAGCGTGGCACTGTTGCAGGCGGTGTTAATACTGACCGCCTCACCTCTGTTTTATCTTCTGCTGGTGGTTCGTTCGGTATTTTTAATGGCGATGTTTTAGGGCTATCAGTTCGCGCATTAAAGACTAATAGCCATTCAAAAATATTGTCTGTGCCACGTATTCTTACGCTTTCAGGTCAGAAGGGTTCTATCTCTGTTGGCCAGAATGTCCCTTTTATTACTGGTCGTGTGACTGGTGAATCTGCCAATGTAAATAATCCATTTCAGACGATTGAGCGTCAAAATGTAGGTATTTCCATGAGCGTTTTTCCTGTTGCAATGGCTGGCGGTAATATTGTTCTGGATATTACCAGCAAGGCCGATAGTTTGAGTTCTTCTACTCAGGCAAGTGATGTTATTACTAATCAAAGAAGTATTGCTACAACGGTTAATTTGCGTGATGGACAGACTCTTTTACTCGGTGGCCTCACTGATTATAAAAACACTTCTCAGGATTCTGGCGTACCGTTCCTGTCTAAAATCCCTTTAATCGGCCTCCTGTTTAGCTCCCGCTCTGATTCTAACGAGGAAAGCACGTTATACGTGCTCGTCAAAGCAACCATAGTACGCGCCCTGTAGC"
            self.scaffold_seq = self.scaffold_seq[:300]  # enough for testing
        
        self.helix_radius = 1.0  # nm
        self.crossover_spacing = 21  # bases between crossovers (2 turns)
    
    def design_staples(self) -> Dict[str, str]:
        """
        Generate staple strands for a rectangular origami.
        
        Returns:
            dict: staple_id -> nucleotide sequence
        """
        # This is a simplified rectangular origami design.
        # In practice, you would use caDNAno or similar.
        
        staples = {}
        scaffold = self.scaffold_seq
        
        # Create parallel helices along helix_axis
        # For a rectangle, helices lie in the perpendicular plane
        n_staple_bases = 32  # typical staple length
        
        for h in range(self.n_helices):
            # Each helix has multiple staples
            helix_offset = h * 2  # offset in scaffold
            
            for start in range(0, len(scaffold) - n_staple_bases, self.crossover_spacing):
                if start + helix_offset + n_staple_bases > len(scaffold):
                    break
                # Staple is complementary to scaffold segment
                staple_seq = self._complement(scaffold[start+helix_offset : start+helix_offset+n_staple_bases])
                staple_id = f"staple_h{h}_s{start}"
                staples[staple_id] = staple_seq
        
        return staples
    
    def _complement(self, seq: str) -> str:
        """Watson‑Crick complement."""
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(comp.get(b, 'N') for b in seq)
    
    def build_3d_structure(self, staple_sequences: Dict[str, str],
                            scaffold_seq: str = None) -> torch.Tensor:
        """
        Generate approximate 3D coordinates for the origami.
        
        Returns:
            C4_coords: [total_bases, 3]
            sequence: concatenated sequence
        """
        if scaffold_seq is None:
            scaffold_seq = self.scaffold_seq
        
        # Place helices in a plane (X‑Y) extending along Z
        coords = []
        full_seq = scaffold_seq
        
        # Simplified: single scaffold trace with staples placed later
        helix_positions = []
        for h in range(self.n_helices):
            x = h * 2.5  # spacing between helices (nm)
            y = 0
            for i in range(len(scaffold_seq)):
                z = i * 0.34  # rise per base (nm)
                helix_positions.append((x, y, z))
        
        coords = torch.tensor(helix_positions[:len(scaffold_seq)], dtype=torch.float32)
        return coords, scaffold_seq
    
    def compute_origami_energy(self, coords: torch.Tensor, seq: str) -> float:
        """Use DNA_RNA_Energy to evaluate the origami structure."""
        if not HAS_DNA_RNA:
            return 0.0
        dna_eng = DNA_RNA_Energy(pucker_type='C2_endo')
        with torch.no_grad():
            E = dna_eng(coords, seq)
        return E.item()


# ═══════════════════════════════════════════════════════════════
# 4. CROSSOVER ENERGY TERM FOR DNA JUNCTIONS
# ═══════════════════════════════════════════════════════════════

def crossover_energy(C4_coords_helix1: torch.Tensor,  # [L1, 3]
                      C4_coords_helix2: torch.Tensor,  # [L2, 3]
                      crossover_pos1: int,              # index in helix1
                      crossover_pos2: int,              # index in helix2
                      ideal_distance: float = 2.0,     # ideal C4' distance at crossover
                      angle_weight: float = 5.0) -> torch.Tensor:
    """
    Energy penalty for a crossover (Holliday junction) between two helices.
    
    Ensures proper geometry: the crossing strands should be close and aligned.
    """
    device = C4_coords_helix1.device
    E = torch.tensor(0.0, device=device)
    
    if crossover_pos1 < 0 or crossover_pos1 >= len(C4_coords_helix1):
        return E
    if crossover_pos2 < 0 or crossover_pos2 >= len(C4_coords_helix2):
        return E
    
    p1 = C4_coords_helix1[crossover_pos1]
    p2 = C4_coords_helix2[crossover_pos2]
    
    # Distance restraint
    d = torch.norm(p1 - p2)
    E += 50.0 * (d - ideal_distance) ** 2
    
    # Angle restraint (helices should be approximately antiparallel at crossover)
    # Get local helix directions
    if crossover_pos1 > 0 and crossover_pos1 < len(C4_coords_helix1) - 1:
        dir1 = C4_coords_helix1[crossover_pos1+1] - C4_coords_helix1[crossover_pos1-1]
    else:
        dir1 = torch.tensor([0.,0.,1.], device=device)
    if crossover_pos2 > 0 and crossover_pos2 < len(C4_coords_helix2) - 1:
        dir2 = C4_coords_helix2[crossover_pos2+1] - C4_coords_helix2[crossover_pos2-1]
    else:
        dir2 = torch.tensor([0.,0.,-1.], device=device)
    
    dir1_n = F.normalize(dir1, dim=-1, eps=1e-8)
    dir2_n = F.normalize(dir2, dim=-1, eps=1e-8)
    dot = torch.dot(dir1_n, dir2_n)
    # Antiparallel: dot ≈ -1
    E += angle_weight * (dot + 1) ** 2
    
    return E


# ═══════════════════════════════════════════════════════════════
# DEMO & CLI (optional)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CSOC‑SSC v30.4 — Antibody & DNA Origami Design Tools")
    print("=" * 60)
    
    # Test affinity maturation (if PDB available)
    # mat = AffinityMaturation("6UF2.pdb", antigen_chain='A', antibody_chains=['B','C'])
    # mat.run()
    
    # Test inverse folding
    coords = torch.randn(10, 3) * 10
    designer = InverseFolding(mode='physics')
    seq = designer.design_sequence(coords, target_sequence='GGGGGGGGGG', 
                                   positions_to_design=list(range(10)),
                                   num_iterations=100)
    print(f"Designed sequence: {seq}")
    
    # Test origami design
    shape = [(i*2, 0, 0) for i in range(50)]
    origami = DNAOrigamiDesigner(shape, n_helices=6)
    staples = origami.design_staples()
    print(f"Generated {len(staples)} staples")
    
    print("Module ready for antibody and DNA origami design.")
