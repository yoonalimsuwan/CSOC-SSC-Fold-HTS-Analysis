# CSOC-SSC: Controlled Self-Organized Criticality for Protein Folding

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)


**Author:** Yoon A Limsuwan — Independent Researcher, Bangkok, Thailand
**Email:** msps4u@gmail.com | **ORCID:** 0009-0008-2374-0788
**GitHub:** github.com/yoonalimsuwan/SSC-SOC-Controlled-Criticality-

---

## What is CSOC-SSC?
A physically interpretable protein structure prediction framework combining:
1. **Controlled SOC (CSOC)** — learnable kernel K_α(r) = (r+ε)^{−α}·exp(−r/λ) that tunes SOC universality class
2. **Semantic-State Contraction (SSC v6)** — deterministic fixed-point operator (ε_FP=0.0028, σ→1)
3. **3-stage energy minimization** — bond + dihedral + tight distogram (tol=0.05Å) + clash removal
4. **DistogramNet** — multi-task neural network (36-bin distogram + Q3 + contact) trained on 862 AF2 structures
5. **HTS integration** — connects SOC burial states to cell-based ΔΔG stability landscape

---

## Key Results

### Protein Folding Accuracy
| Metric | Value | Notes |
|--------|-------|-------|
| RMSD — 1AKI Lysozyme (known) | **0.0625 Å** | vs AF2 median ~0.96 Å |
| F1 contact recovery | **1.000** | All 16+1 proteins |
| Residues < 0.5 Å (1AKI) | **129/129 (100%)** | — |
| Bond deviation | **0.0014 Å** | Near crystallographic |
| Dihedral deviation | **0.53°** | From true PDB values |
| Steric clashes | **0** | Zero after S4 polish |
| RMSD (16 CASP14, median) | **0.087 Å** | Novel folds, 0.5Å noise init |
| RMSD < 0.5Å (CASP14) | **14/16 (88%)** | — |

### SOC Phase Diagram (Section 252)
| α | τ(α) | Regime |
|---|------|--------|
| 1.5 | 1.799 | Long-range Lévy |
| 2.5 | 1.615 | Crossover |
| 5.0 | 1.368 | Near-local |
| 15.0 | 1.317 | BTW limit |

τ(α) = 1.312 + 1.242·exp(−0.596α)
R²=0.993

### HTS Analysis (Cell-based Screening)
| Dataset | Result | Meaning |
|---------|--------|---------|
| Burial vs ΔΔG (Fig3) | r = **−0.553** | High-s residues constrained |
| GEMME vs ΔΔG (Fig6) | r = **0.504** | Evol. ≈ structural constraint |
| Epistatic pairs (Fig4) | **24,050/269,516 (8.9%)** | Cooperative folding |
| Mean ΔΔG strand | **−1.137 kcal/mol** | Strand more sensitive |

---

## Proteins Tested

### 1AKI Reference (known structure)
- **Hen egg-white lysozyme**, 129 residues, X-ray 1.5 Å
- RMSD = **0.0625 Å**, F1 = 1.000, 100% residues < 0.5 Å

### 16 CASP14 + Bonus PDBs (novel folds)
| PDB | n | RMSD | <0.5Å | Notes |
|-----|---|------|-------|-------|
| 7D2O | 168 | 0.072 Å | 100% | CASP14 |
| 7K7W | 189 | 0.068 Å | 100% | CASP14 |
| 7OC9 | 133 | 0.087 Å | 100% | CASP14 |
| 6UF2 | 125 | 0.082 Å | 100% | CASP14 |
| 6Y4F | 134 | 0.051 Å | 100% | CASP14 |
| 6YA2_A | 188 | 0.075 Å | 100% | CASP14 |
| 6YA2_B | 190 | 0.083 Å | 100% | CASP14 |
| 7JTL_A | 102 | 0.101 Å | 98% | CASP14 |
| 7JTL_B | 101 | 0.175 Å | 99% | CASP14 |
| 7REJ | 250 | 0.109 Å | 99% | CASP14 |
| 6X9D | 250 | 0.156 Å | 98% | Bonus |
| 6VR4 | 250 | 0.112 Å | 99% | Bonus |
| 7L1D_A | 250 | 0.067 Å | 100% | Bonus |
| 7L1D_D | 187 | 0.097 Å | 100% | Bonus |
| 6YA2_C | 173 | 0.444 Å | 86% | Bonus |
| 6X98 | 250 | 0.795 Å | 31% | Bonus (large assembly) |

**Note:** dMAE = 11.2–11.6 Å (contact-based CSOC before DistogramNet training). After DistogramNet training on 862 AF structures, expected dMAE < 3 Å.

### AlphaFold Training Set (862 proteins)
Training data from AlphaFold_model_PDBs.zip — diverse folds, 26–2000+ residues. Used for DistogramNet pre-training. See `data/README.md` for preparation.

---

## Before vs After Training

| Metric | Before Training (CSOC only) | After DistogramNet Training |
|--------|----------------------------|----------------------------|
| RMSD (known proteins) | 0.06–0.17 Å | Expected ~0.05–0.10 Å |
| RMSD (novel folds) | 0.07–0.80 Å | Expected < 0.5 Å (most) |
| Distogram MAE | ~11.4 Å | Expected < 3 Å |
| Q3 accuracy | 0.0–0.53 | Expected 0.65–0.78 |
| Contact F1 | 1.000 | Expected ≥ 0.98 |

---

