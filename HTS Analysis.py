"""
hts_analysis.py — High-Throughput Screening Analysis
Processes cell-based ΔΔG stability landscapes, Epistasis, and GEMME scores.
"""
import zipfile, os, pandas as pd, numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def full_hts_report(zip_path, out_dir='/tmp/csoc_out'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Analyzing HTS Data from {zip_path}...")
    
    with zipfile.ZipFile(zip_path) as z:
        fmap = {
            'fig3': 'Data_tables_for_figs/dG_site_feature_Fig3.csv',
            'fig4': 'Data_tables_for_figs/dG_for_double_mutants_Fig4.csv',
            'fig6': 'Data_tables_for_figs/dG_GEMME_non_redundant_natural_Fig6.csv'
        }
        tables = {k: pd.read_csv(z.open(v)) for k, v in fmap.items()}

    df3 = tables['fig3']
    ddg_cols = [c for c in df3.columns if c.startswith('ddg_')]
    ddg_mean = df3[ddg_cols].mean(axis=1)

    df4 = tables['fig4']
    eps = df4['thermo_dynamics'].dropna().values
    n_ep = (np.abs(eps) > 0.5).sum()
    print(f"Epistasis: {n_ep}/{len(eps)} ({100*n_ep/len(eps):.1f}%) |ε|>0.5")

    df6 = tables['fig6']
    gemme_cols = [c for c in df6.columns if 'gemme' in c.lower()]
    ddg6_cols = [c for c in df6.columns if c.startswith('ddG_')]
    
    valid_gemme = False
    if gemme_cols and ddg6_cols:
        gm = df6[gemme_cols].mean(axis=1).dropna()
        dg = df6[ddg6_cols].mean(axis=1)[:len(gm)]
        valid = ~(np.isnan(gm) | np.isnan(dg))
        r, p = pearsonr(gm[valid], dg[valid])
        valid_gemme = True

    # Generate Dashboard Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("HTS Analysis — Cell-based Stability Landscape\n"
                 "CSOC-SSC: SOC Universality Class ↔ ΔΔG Constraint", 
                 fontsize=11, fontweight='bold')

    # Plot 1
    ax = axes[0]
    aa_col = 'wt_aa' if 'wt_aa' in df3.columns else None
    if aa_col:
        for aa, col in [('A', '#E91E63'), ('G', '#2196F3'), ('P', '#FF9800')]:
            vals = ddg_mean[df3[aa_col] == aa].dropna()
            if len(vals) > 10:
                ax.hist(vals, bins=30, alpha=0.5, color=col, label=f'{aa} (n={len(vals)})', density=True)
    ax.set_xlabel('Mean ΔΔG (kcal/mol)', fontsize=9)
    ax.set_title('ΔΔG distribution by residue type', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Plot 2
    ax = axes[1]
    ax.hist(eps, bins=60, color='#9C27B0', alpha=0.8, edgecolor='white', lw=0.3)
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.axvline(0.5, color='red', ls=':', lw=1.5, label='|ε|=0.5 threshold')
    ax.axvline(-0.5, color='red', ls=':', lw=1.5)
    ax.set_xlabel('Epistatic coupling ε (kcal/mol)', fontsize=9)
    ax.set_title(f'Double mutant epistasis\nEpistatic: {n_ep}/{len(eps)}', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Plot 3
    ax = axes[2]
    if valid_gemme:
        ax.scatter(gm[valid], dg[valid], alpha=0.2, s=4, c='steelblue')
        ax.set_xlabel('GEMME score (evolutionary constraint)', fontsize=9)
        ax.set_ylabel('Experimental ΔΔG (kcal/mol)', fontsize=9)
        ax.set_title(f'GEMME vs ΔΔG\nr={r:.3f} n={valid.sum()}', fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = f"{out_dir}/hts_analysis.png"
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"Report Generated: {save_path}")
    return save_path

if __name__ == "__main__":
    full_hts_report('/mnt/user-data/uploads/Data_tables_for_figs.zip')
