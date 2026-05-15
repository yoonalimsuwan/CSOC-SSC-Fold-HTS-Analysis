#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC HTS FOLD v30 — High‑Throughput Mutational Scanning & Epistasis
#                            with Structure‑Based ΔΔG Prediction (V30 Engine)
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# v30 integrates with CSOC‑SSC V30 (Ewald PME, configurable force field,
# corrected SOC).  All analyses (ΔΔG stats, epistasis, GEMME) preserved.
# =============================================================================

import os, sys, argparse, logging, warnings, zipfile, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("CSOC‑HTS-V30")

# ──────────────────────────────────────────────────────────────────────────────
# Try to import CSOC‑SSC V30
# ──────────────────────────────────────────────────────────────────────────────
try:
    from csoc_v30 import (
        CSOCSSC_V30, V30Config,
        get_full_atom_coords_and_types, reconstruct_backbone,
        total_physics_energy_v30, sparse_edges, cross_sparse_edges,
        build_sidechain_atoms, MAX_CHI, RESIDUE_NCHI, RESIDUE_TOPOLOGY,
        AA_TO_ID, AA_VOCAB, LJ_PARAMS, DEFAULT_CHARGE_MAP
    )
    HAS_V30 = True
except ImportError:
    HAS_V30 = False
    logger.warning("csoc_v30 module not found. Structure‑based ΔΔG will be disabled.")
    # dummy placeholders
    sparse_edges = cross_sparse_edges = lambda *a, **k: (None, None)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
class HTSConfig:
    def __init__(self,
                 data_sources: List[str],
                 output_dir: str = "./hts_output_v30",
                 ddg_threshold: float = 0.5,
                 pdb_structure: Optional[str] = None,
                 v30_checkpoint: Optional[str] = None,
                 mutation_list: Optional[List[Tuple[int, str]]] = None,
                 use_gpu: bool = False,
                 lj_param_file: Optional[str] = None,
                 charge_param_file: Optional[str] = None):
        self.data_sources = data_sources
        self.output_dir = output_dir
        self.ddg_threshold = ddg_threshold
        self.pdb_structure = pdb_structure
        self.v30_checkpoint = v30_checkpoint
        self.mutation_list = mutation_list
        self.use_gpu = use_gpu
        self.lj_param_file = lj_param_file
        self.charge_param_file = charge_param_file

# ──────────────────────────────────────────────────────────────────────────────
# Data Loader (unchanged from V29)
# ──────────────────────────────────────────────────────────────────────────────
class DataLoaderV30:
    @staticmethod
    def discover_files(sources: List[str]) -> List[Tuple[str, str]]:
        files = []
        for src in sources:
            p = Path(src)
            if p.is_dir():
                for csv_file in p.glob("*.csv"):
                    files.append((str(csv_file), csv_file.stem))
                for zip_file in p.glob("*.zip"):
                    files.append((str(zip_file), zip_file.stem))
            elif p.is_file():
                files.append((str(p), p.stem))
            else:
                logger.warning(f"Source not found: {src}")
        return files

    @staticmethod
    def load_table(file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    raise ValueError(f"No CSV found in zip: {file_path}")
                target = next((n for n in csv_names if 'dG_site_feature' in n), csv_names[0])
                with z.open(target) as f:
                    return pd.read_csv(f)
        else:
            return pd.read_csv(file_path)

# ──────────────────────────────────────────────────────────────────────────────
# HTS Analyzer V30
# ──────────────────────────────────────────────────────────────────────────────
class HTSAnalyzerV30:
    def __init__(self, config: HTSConfig):
        self.config = config
        self.tables: Dict[str, pd.DataFrame] = {}
        self.results = {}
        self.v30_model = None
        self.v30_cfg = None
        self.wt_sequence = None
        self.wt_ca = None
        self.wt_chi = None

    def load_data(self):
        files = DataLoaderV30.discover_files(self.config.data_sources)
        logger.info(f"Found {len(files)} data source(s).")
        for fpath, tag in files:
            try:
                df = DataLoaderV30.load_table(fpath)
                self.tables[tag] = df
                logger.info(f"Loaded {tag}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")

    def load_v30_engine(self):
        """Load CSOC‑SSC V30 model and compute WT energy if a structure is given."""
        if not HAS_V30:
            logger.warning("V30 engine not available.")
            return
        if not self.config.pdb_structure or not os.path.exists(self.config.pdb_structure):
            logger.warning("No PDB structure provided; structure‑based ΔΔG skipped.")
            return

        device = "cuda" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu"
        # Build V30Config with default energy weights; allow external force field files
        cfg = V30Config(
            device=device,
            use_pme=True,  # enable PME by default for accuracy
            lj_param_file=self.config.lj_param_file,
            charge_param_file=self.config.charge_param_file
        )
        model = CSOCSSC_V30(cfg).to(device)
        if self.config.v30_checkpoint and os.path.exists(self.config.v30_checkpoint):
            model.load_state_dict(torch.load(self.config.v30_checkpoint, map_location=device))
            logger.info(f"Loaded V30 weights from {self.config.v30_checkpoint}")
        else:
            logger.warning("No V30 checkpoint found; using random weights.")
        model.eval()
        self.v30_model = model
        self.v30_cfg = cfg

        # Parse PDB to get sequence and CA coordinates
        ca_coords, seq = self.parse_pdb_to_ca(self.config.pdb_structure)
        if len(seq) == 0:
            raise ValueError("No CA atoms found in PDB.")
        self.wt_sequence = seq

        # Prepare initial coordinates (centered)
        ca_np = ca_coords - ca_coords.mean(axis=0)
        init = [ca_np]

        # Short refinement to obtain chi angles that are physically consistent with the given CA
        logger.info("Running short refinement (50 steps) to obtain side‑chain χ angles...")
        with torch.no_grad():
            sequences = [self.wt_sequence]
            refined_ca, refined_chi, _ = model.refine_multimer(
                sequences,
                init_coords_list=init,
                steps=50,
                logger=None
            )
        self.wt_ca = torch.tensor(refined_ca, device=device, requires_grad=False)
        self.wt_chi = torch.tensor(refined_chi, device=device, requires_grad=False)

        # Compute WT energy
        self.wt_energy = self.compute_energy(self.wt_ca, self.wt_sequence, self.wt_chi)
        logger.info(f"WT total energy: {self.wt_energy:.4f} kcal/mol")

    def parse_pdb_to_ca(self, pdb_path):
        ca_coords = []
        seq = []
        # Use the same AA_3_TO_1 mapping
        from csoc_v30 import AA_3_TO_1
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ca_coords.append([x, y, z])
                    res_name = line[17:20].strip()
                    aa = AA_3_TO_1.get(res_name, 'X')
                    seq.append(aa)
        return np.array(ca_coords, dtype=np.float32), "".join(seq)

    def compute_energy(self, ca, seq, chi):
        """Compute total physics energy using V30 (no grad)."""
        device = ca.device
        cfg = self.v30_cfg
        # Build sparse graphs (V30 uses direct PyG calls, no chunk_size)
        edge_index_ca, edge_dist_ca = sparse_edges(
            ca, cfg.sparse_cutoff, cfg.max_neighbors
        )
        atoms = reconstruct_backbone(ca)
        edge_index_hbond, edge_dist_hbond = cross_sparse_edges(
            atoms['O'], atoms['N'], 3.5, cfg.max_neighbors
        )
        # Alpha field set to 1.0 (neutral modulation)
        alpha = torch.ones(len(seq), device=device)
        # No chain boundaries (single chain)
        chain_boundaries = []
        e = total_physics_energy_v30(
            ca, seq, alpha, chi,
            edge_index_ca, edge_dist_ca,
            edge_index_hbond, edge_dist_hbond,
            chain_boundaries, cfg
        )
        return e.item()

    def predict_ddg_mutation(self, pos: int, new_aa: str):
        """Compute ΔΔG = E(mutant) - E(WT) for a point mutation.
           Uses WT backbone but rebuilds sidechain with new sequence and original χ.
           (Note: a full relaxation would be more accurate, but this quick approximation
            captures the change in force‑field parameters.)
        """
        if self.wt_ca is None:
            raise RuntimeError("Structure not loaded.")
        if pos < 0 or pos >= len(self.wt_sequence):
            raise ValueError("Position out of range.")
        mut_seq = self.wt_sequence[:pos] + new_aa + self.wt_sequence[pos+1:]
        # Use the same CA and χ as WT
        ca = self.wt_ca.clone()
        chi = self.wt_chi.clone()
        # For the mutated residue, the χ angles might be inappropriate, but we keep them.
        # A better method would reset χ to 0 for that residue and run a few steps of
        # sidechain‑only relaxation. For demonstration, we accept this approximation.
        e_mut = self.compute_energy(ca, mut_seq, chi)
        return e_mut - self.wt_energy

    def run_analysis(self):
        self.load_data()
        self.compute_ddg_statistics()
        self.compute_epistasis()
        self.compute_gemme_correlations()
        if self.config.pdb_structure and HAS_V30:
            self.load_v30_engine()
            if self.config.mutation_list:
                self.compute_structure_based_ddg()
        self.generate_report()
        return self.results

    def compute_ddg_statistics(self):
        all_ddg = []
        for tag, df in self.tables.items():
            ddg_cols = [c for c in df.columns if 'ddg' in c.lower() or c.startswith('ddG')]
            if ddg_cols:
                row_mean = df[ddg_cols].mean(axis=1)
                all_ddg.append(row_mean)
        if all_ddg:
            combined = pd.concat(all_ddg, ignore_index=True)
            stats = {
                'mean_ddg': combined.mean(),
                'std_ddg': combined.std(),
                'median_ddg': combined.median(),
                'q1_ddg': combined.quantile(0.25),
                'q3_ddg': combined.quantile(0.75),
                'count': len(combined)
            }
            self.results['ddg_statistics'] = stats
            self.results['ddg_data'] = combined
            logger.info(f"ΔΔG statistics computed on {stats['count']} data points.")

    def compute_epistasis(self):
        epistasis_list = []
        for tag, df in self.tables.items():
            ep_col = next((c for c in df.columns if 'thermo_dynamics' in c.lower() or 'epistasis' in c.lower()), None)
            if ep_col:
                vals = df[ep_col].dropna().astype(float)
                epistasis_list.append(vals)
        if epistasis_list:
            all_eps = pd.concat(epistasis_list, ignore_index=True)
            n_sig = (np.abs(all_eps) > self.config.ddg_threshold).sum()
            stats = {
                'total_pairs': len(all_eps),
                'significant_epistasis': n_sig,
                'fraction_significant': n_sig / len(all_eps) if len(all_eps) > 0 else 0,
                'mean_eps': all_eps.mean(),
                'std_eps': all_eps.std()
            }
            self.results['epistasis'] = stats
            self.results['epistasis_data'] = all_eps
            logger.info(f"Epistasis: {n_sig}/{len(all_eps)} above threshold ±{self.config.ddg_threshold}.")

    def compute_gemme_correlations(self):
        cors = []
        for tag, df in self.tables.items():
            gemme_cols = [c for c in df.columns if 'gemme' in c.lower()]
            ddg_cols = [c for c in df.columns if 'ddg' in c.lower() or c.startswith('ddG')]
            if gemme_cols and ddg_cols:
                gemme_mean = df[gemme_cols].mean(axis=1)
                ddg_mean = df[ddg_cols].mean(axis=1)
                valid = gemme_mean.notna() & ddg_mean.notna()
                if valid.sum() > 5:
                    r, p = pearsonr(gemme_mean[valid], ddg_mean[valid])
                    rho, _ = spearmanr(gemme_mean[valid], ddg_mean[valid])
                    cors.append({'tag': tag, 'n': valid.sum(), 'pearson_r': r, 'pearson_p': p, 'spearman_rho': rho})
        self.results['gemme_correlations'] = cors
        if cors:
            for c in cors:
                logger.info(f"GEMME vs ΔΔG ({c['tag']}): r={c['pearson_r']:.3f}, p={c['pearson_p']:.1e}, n={c['n']}")

    def compute_structure_based_ddg(self):
        """Compute ΔΔG for a list of mutations using the loaded V30 model."""
        if not self.wt_ca:
            return
        mutations = self.config.mutation_list
        ddg_pred = []
        for pos, aa in mutations:
            try:
                ddg = self.predict_ddg_mutation(pos, aa)
                ddg_pred.append({'position': pos, 'mutation': aa, 'predicted_ddg': ddg})
            except Exception as e:
                logger.error(f"Mutation {pos}{aa} failed: {e}")
        self.results['structure_ddg_predictions'] = ddg_pred
        if ddg_pred:
            logger.info(f"Structure‑based ΔΔG computed for {len(ddg_pred)} mutations.")

    def generate_report(self):
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. ΔΔG distribution
        if 'ddg_data' in self.results:
            fig, ax = plt.subplots(figsize=(8,4))
            ddg = self.results['ddg_data']
            sns.histplot(ddg, bins=80, kde=True, color='#2E86AB', ax=ax)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax.set_title("Global ΔΔG Distribution")
            ax.set_xlabel("ΔΔG (kcal/mol)")
            fig.tight_layout()
            fig.savefig(out_dir / "ddg_distribution.png", dpi=200)

        # 2. Epistasis histogram
        if 'epistasis_data' in self.results:
            fig, ax = plt.subplots(figsize=(8,4))
            eps = self.results['epistasis_data']
            sns.histplot(eps, bins=80, color='#A23B72', ax=ax, kde=True)
            ax.axvline(self.config.ddg_threshold, color='black', linestyle='--')
            ax.axvline(-self.config.ddg_threshold, color='black', linestyle='--')
            ax.set_title("Epistatic Coupling (ε) Distribution")
            ax.set_xlabel("ε (kcal/mol)")
            fig.tight_layout()
            fig.savefig(out_dir / "epistasis_distribution.png", dpi=200)

        # 3. GEMME correlation bar chart
        if 'gemme_correlations' in self.results:
            cors = self.results['gemme_correlations']
            if cors:
                tags = [c['tag'] for c in cors]
                r_vals = [c['pearson_r'] for c in cors]
                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(tags, r_vals, color='#F18F01')
                ax.set_title("GEMME–ΔΔG Pearson Correlation")
                ax.set_ylabel("Pearson r")
                ax.axhline(0, color='gray', linestyle='--')
                fig.tight_layout()
                fig.savefig(out_dir / "gemme_correlations.png", dpi=200)

        # 4. Structure‑based ΔΔG predictions
        if 'structure_ddg_predictions' in self.results:
            preds = self.results['structure_ddg_predictions']
            if preds:
                mut_labels = [f"{p['position']}{p['mutation']}" for p in preds]
                ddg_vals = [p['predicted_ddg'] for p in preds]
                fig, ax = plt.subplots(figsize=(max(6, len(preds)*0.3), 4))
                ax.bar(mut_labels, ddg_vals, color='#3A7D44')
                ax.set_title("Predicted ΔΔG from CSOC‑SSC V30")
                ax.set_ylabel("ΔΔG (kcal/mol)")
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / "structure_ddg.png", dpi=200)

        # Save summary JSON
        summary = {
            'ddg_statistics': self.results.get('ddg_statistics', {}),
            'epistasis': self.results.get('epistasis', {}),
            'gemme_correlations': self.results.get('gemme_correlations', []),
            'structure_ddg_predictions': self.results.get('structure_ddg_predictions', []),
            'n_sources': len(self.tables)
        }
        with open(out_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Report generated in {out_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# Command‑line interface
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSOC‑SSC HTS FOLD v30 – Mutation & Stability Analyzer")
    parser.add_argument('--data', nargs='+', required=True, help='CSV or ZIP files or directories')
    parser.add_argument('--output', default='./hts_output_v30', help='Output directory')
    parser.add_argument('--ddg_threshold', type=float, default=0.5, help='Epistasis threshold')
    parser.add_argument('--pdb', default=None, help='Refined PDB for structure‑based ΔΔG')
    parser.add_argument('--checkpoint', default=None, help='CSOC‑SSC V30 checkpoint')
    parser.add_argument('--mutations', nargs='+', help='Mutations in format "posAA" e.g. "30F" "45A"')
    parser.add_argument('--gpu', action='store_true', help='Use GPU (if available)')
    parser.add_argument('--lj_params', default=None, help='JSON file with custom LJ parameters')
    parser.add_argument('--charge_params', default=None, help='JSON file with custom atomic charges')
    args = parser.parse_args()

    mutation_list = None
    if args.mutations:
        mutation_list = []
        for m in args.mutations:
            pos = int(m[:-1])
            aa = m[-1]
            mutation_list.append((pos, aa))

    config = HTSConfig(
        data_sources=args.data,
        output_dir=args.output,
        ddg_threshold=args.ddg_threshold,
        pdb_structure=args.pdb,
        v30_checkpoint=args.checkpoint,
        mutation_list=mutation_list,
        use_gpu=args.gpu,
        lj_param_file=args.lj_params,
        charge_param_file=args.charge_params
    )

    analyzer = HTSAnalyzerV30(config)
    results = analyzer.run_analysis()
    print("Analysis complete. Summary JSON saved to", config.output_dir)

if __name__ == "__main__":
    main()
