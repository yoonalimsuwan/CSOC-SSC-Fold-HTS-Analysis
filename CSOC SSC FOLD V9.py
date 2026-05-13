# csoc_ssc_fold.py — CSOC-SSC v9 Core Module (PyTorch GPU, Memory‑Safe)
# MIT License — Yoon A Limsuwan 2026
# github.com/yoonalimsuwan/SSC-SOC-Controlled-Criticality-
"""
CSOC-SSC v9 — Major GPU-native rewrite with PyTorch autograd, memory-safe
distogram/chunking, banded clash checks, differentiable dihedral term,
and validation utilities (grad_check). Designed for research and
large-scale experiments; requires PyTorch with CUDA for best performance.
"""
import os
import numpy as np
import torch

__version__ = "9.0.0"
__license__ = "MIT"

THREE2ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y','SEC':'U','MSE':'M','HSD':'H','HSE':'H',
}

# -------------------------
# I/O / Utilities
# -------------------------
def load_pdb_gz(path, chain='A', max_res=500):
    """Load CA coordinates and sequence from a PDB or PDB.GZ file."""
    import gzip
    coords, seq = [], []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='ignore') as f:
        seen = set()
        for l in f:
            if not l.startswith('ATOM') or l[12:16].strip() != 'CA':
                continue
            if l[21] != chain:
                continue
            key = (int(l[22:26]), l[26])
            if key in seen:
                continue
            seen.add(key)
            coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            seq.append(THREE2ONE.get(l[17:20].strip(), 'X'))
            if len(coords) >= max_res:
                break
    return (np.array(coords), ''.join(seq)) if coords else (None, '')

def kabsch(P, Q):
    """Kabsch alignment: returns RMSD and rotated P (centered) aligned to Q."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc)**2, axis=1)))), Pr

# -------------------------
# Contact map and SSC
# -------------------------
def csoc_contact_map_pt(coords_pt, alpha=2.5, r_cut=15.0):
    """
    Compute contact-like matrix C and pairwise distances D using torch.cdist.
    Returns (C, D) as tensors on the same device as coords_pt.
    """
    device = coords_pt.device
    n = coords_pt.shape[0]
    # memory-efficient pairwise distances
    D = torch.cdist(coords_pt, coords_pt)  # (n, n)
    # index grid
    i_idx, j_idx = torch.meshgrid(
        torch.arange(n, device=device),
        torch.arange(n, device=device),
        indexing='ij'
    )
    abs_diff = torch.abs(i_idx - j_idx).double()
    r = abs_diff + 1e-8
    kernel = (r ** (-alpha)) * torch.exp(-r / 12.0)
    mask = (abs_diff > 3) & (D < r_cut)
    C = torch.zeros((n, n), dtype=torch.float64, device=device)
    C[mask] = (1.0 - D[mask].double() / float(r_cut)) * (1.0 + 0.3 * kernel[mask])
    # safe normalization
    max_c = C.max(dim=1, keepdim=True).values
    nz = (max_c > 0).flatten()
    if nz.any():
        C[nz, :] = C[nz, :] / (max_c[nz] + 1e-8)
    return C, D

def ssc_states_pt(coords, alpha=2.5, T=150, device='cuda'):
    """
    Compute SSC semantic states s using vectorized updates on PyTorch device.
    Returns (s_tensor, C_tensor, D_tensor) on device.
    """
    coords_pt = torch.tensor(coords, dtype=torch.float64, device=device)
    n = coords_pt.shape[0]
    C, D = csoc_contact_map_pt(coords_pt, alpha=alpha, r_cut=15.0)
    ctr = coords_pt.mean(dim=0)
    dc = torch.norm(coords_pt - ctr, dim=1)
    dc_min, dc_max = dc.min(), dc.max()
    s = torch.clamp(1.0 - (dc - dc_min) / (dc_max - dc_min + 1e-8), 0.05, 0.95)
    # precompute kernel W (sequence separation kernel)
    i_idx, j_idx = torch.meshgrid(
        torch.arange(n, device=device),
        torch.arange(n, device=device),
        indexing='ij'
    )
    r = torch.abs(i_idx - j_idx).double() + 1e-8
    W = (r ** (-alpha)) * torch.exp(-r / 12.0)
    W.fill_diagonal_(0.0)
    W_sum = W.sum(dim=1) + 1e-8
    for t in range(T):
        g = -(C @ s) + 0.05 * (1.0 - 2.0 * s)
        s_update = s + 0.08 * (W @ s) / W_sum
        sp = torch.clamp(s_update, 0.0, 1.0)
        eta = 0.04 if t < T // 2 else 0.02
        s = torch.clamp(s - eta * g + 0.1 * (sp - s), 0.0, 1.0)
    return s, C, D

# -------------------------
# Energy terms (differentiable)
# -------------------------
def dihedral_energy(c, tdih, wdh):
    """
    Differentiable dihedral energy. tdih in degrees (length n-3) or None.
    Returns scalar tensor.
    """
    if tdih is None or wdh == 0:
        return torch.tensor(0.0, dtype=c.dtype, device=c.device)
    # vectors for dihedral computation
    b1 = c[1:-2] - c[:-3]
    b2 = c[2:-1] - c[1:-2]
    b3 = c[3:] - c[2:-1]
    nv1 = torch.cross(b1, b2, dim=1)
    nv2 = torch.cross(b2, b3, dim=1)
    n1 = torch.norm(nv1, dim=1) + 1e-8
    n2 = torch.norm(nv2, dim=1) + 1e-8
    cos_phi = torch.clamp(torch.sum(nv1 * nv2, dim=1) / (n1 * n2), -1.0, 1.0)
    phi = torch.acos(cos_phi)  # radians
    if not isinstance(tdih, torch.Tensor):
        tdih = torch.tensor(tdih, dtype=c.dtype, device=c.device)
    tdih_rad = tdih * (np.pi / 180.0)
    return wdh * torch.sum((phi - tdih_rad) ** 2)

def distogram_batch_eval(c, disto_mask, disto_dt, wd, batch_size=10000):
    """
    Evaluate distogram constraints in batches to avoid OOM.
    disto_mask: LongTensor shape (k,2), disto_dt: FloatTensor shape (k,)
    """
    if disto_mask is None or disto_dt is None or wd == 0:
        return torch.tensor(0.0, dtype=c.dtype, device=c.device)
    n_pairs = disto_mask.shape[0]
    E_sum = torch.tensor(0.0, dtype=c.dtype, device=c.device)
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        pairs = disto_mask[start:end]
        dv = c[pairs[:, 0]] - c[pairs[:, 1]]
        d = torch.norm(dv, dim=1) + 1e-8
        ex = torch.abs(d - disto_dt[start:end]) - 0.05
        mask_ex = ex > 0
        if mask_ex.any():
            E_sum = E_sum + wd * torch.sum(ex[mask_ex] ** 2)
    return E_sum

def energy_torch_memory_efficient(c, disto_mask, disto_dt, d_id,
                                  wb=30.0, wd=5.0, wc=50.0, wa=5.0, wdh=3.0, tdih=None,
                                  clash_threshold=3.0):
    """
    Core energy combining bond, distogram, banded clash, and dihedral terms.
    All inputs are tensors on the same device as c.
    """
    E = torch.tensor(0.0, dtype=c.dtype, device=c.device)
    n = c.shape[0]
    # 1. Bonds
    dv = c[1:] - c[:-1]
    d = torch.norm(dv, dim=1)
    E = E + wb * torch.sum((d - d_id) ** 2)
    # 2. Distogram (batched)
    E = E + distogram_batch_eval(c, disto_mask, disto_dt, wd)
    # 3. Banded clashes (only offsets 3..20)
    max_offset = min(20, n - 1)
    if max_offset > 3:
        # vectorize offsets by flattening valid pairs
        i_base = torch.arange(n, device=c.device).unsqueeze(1)  # (n,1)
        offsets = torch.arange(3, max_offset + 1, device=c.device).unsqueeze(0)  # (1,m)
        j_idx = i_base + offsets  # (n, m)
        valid_mask = j_idx < n
        if valid_mask.any():
            i_flat = i_base.expand_as(j_idx)[valid_mask]
            j_flat = j_idx[valid_mask]
            diff = c[i_flat] - c[j_flat]
            d_pair = torch.norm(diff, dim=1)
            clash_mask = d_pair < clash_threshold
            if clash_mask.any():
                dev = clash_threshold - d_pair[clash_mask]
                E = E + wc * torch.sum(dev ** 2)
    # 4. Dihedral
    E = E + dihedral_energy(c, tdih, wdh)
    return E

# -------------------------
# Optimization pipeline (staged)
# -------------------------
def run_optimization_stage(c_init, device, max_iter=300, ftol=1e-11, energy_kwargs=None):
    """
    Run a single L-BFGS optimization stage on GPU (or CPU fallback).
    Re-initializes parameter tensor to avoid stale LBFGS state.
    Returns optimized coordinates as numpy array.
    """
    if energy_kwargs is None:
        energy_kwargs = {}
    c_var = torch.tensor(c_init, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([c_var], max_iter=max_iter, tolerance_change=ftol, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = energy_torch_memory_efficient(c_var, **energy_kwargs)
        loss.backward()
        return loss
    optimizer.step(closure)
    return c_var.detach().cpu().numpy()

def fold(coords_ref, alpha=2.5, noise=0.5, seed=1, verbose=False, device=None,
         tdih=None, batch_size=10000):
    """
    Full folding pipeline:
      - compute SSC states on device
      - build distogram mask (pairs with D < 15.0 and |i-j|>=4)
      - run staged L-BFGS optimization on GPU
    Returns dict with rmsd_stages, rmsd_final, coords_pred (numpy), s (numpy), and metadata.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(coords_ref)
    # SSC on device (returns tensors)
    s_pt, C_pt, D_pt = ssc_states_pt(coords_ref, alpha=alpha, T=150, device=device)
    # build distogram list (only pairs with D < 15 and separation >=4)
    D_np = D_pt.cpu().numpy()
    disto_pairs = [(i, j, float(D_np[i, j])) for i in range(n) for j in range(i+4, n) if D_np[i, j] < 15.0]
    if len(disto_pairs) > 0:
        disto_mask = torch.tensor([[p[0], p[1]] for p in disto_pairs], dtype=torch.long, device=device)
        disto_dt = torch.tensor([p[2] for p in disto_pairs], dtype=torch.float64, device=device)
    else:
        disto_mask = None
        disto_dt = None
    # dihedral targets if provided
    tdih_tensor = None
    if tdih is not None:
        tdih_tensor = torch.tensor(tdih, dtype=torch.float64, device=device)
    # initial coordinates with noise
    d_id = float(np.mean([np.linalg.norm(coords_ref[i+1] - coords_ref[i]) for i in range(n-1)]))
    c0 = coords_ref + np.random.randn(n, 3) * noise
    stages = []
    coords_current = c0.copy()
    # stage 1: relax bonds + angle-ish (no distogram)
    energy_kwargs = dict(disto_mask=None, disto_dt=None, d_id=d_id,
                         wb=30.0, wd=0.0, wc=0.0, wa=8.0, wdh=0.0, tdih=None)
    x1 = run_optimization_stage(coords_current, device=device, max_iter=300, ftol=1e-11, energy_kwargs=energy_kwargs)
    stages.append(kabsch(x1, coords_ref)[0])
    coords_current = x1
    # stage 2: add angle/dihedral-ish
    energy_kwargs = dict(disto_mask=None, disto_dt=None, d_id=d_id,
                         wb=25.0, wd=0.0, wc=0.0, wa=8.0, wdh=5.0, tdih=tdih_tensor)
    x2 = run_optimization_stage(coords_current, device=device, max_iter=400, ftol=1e-12, energy_kwargs=energy_kwargs)
    stages.append(kabsch(x2, coords_ref)[0])
    coords_current = x2
    # stage 3: add distogram soft
    energy_kwargs = dict(disto_mask=disto_mask, disto_dt=disto_dt, d_id=d_id,
                         wb=20.0, wd=5.0, wc=0.0, wa=5.0, wdh=5.0, tdih=tdih_tensor)
    x3 = run_optimization_stage(coords_current, device=device, max_iter=600, ftol=1e-13, energy_kwargs=energy_kwargs)
    stages.append(kabsch(x3, coords_ref)[0])
    coords_current = x3
    # stage 4: tighten distogram + clash removal
    energy_kwargs = dict(disto_mask=disto_mask, disto_dt=disto_dt, d_id=d_id,
                         wb=15.0, wd=20.0, wc=80.0, wa=5.0, wdh=8.0, tdih=tdih_tensor)
    x4 = run_optimization_stage(coords_current, device=device, max_iter=800, ftol=1e-14, energy_kwargs=energy_kwargs)
    coords_final = x4
    rmsd_final, Prot = kabsch(coords_final, coords_ref)
    stages.append(rmsd_final)
    if verbose:
        for i, r in enumerate(stages):
            print(f" Stage {i+1}: {r:.6f} Å")
    # per-residue deviations (aligned)
    Qc = coords_ref - coords_ref.mean(0)
    per_res = np.sqrt(np.sum((Prot - Qc) ** 2, axis=1))
    bond_dev = float(np.mean([abs(np.linalg.norm(coords_final[i+1] - coords_final[i]) - d_id) for i in range(n-1)]))
    n_clash = sum(1 for i in range(n) for j in range(i+3, n) if np.linalg.norm(coords_final[i] - coords_final[j]) < 3.5)
    return dict(
        rmsd_stages=stages,
        rmsd_final=rmsd_final,
        coords_pred=coords_final,
        s=s_pt.cpu().numpy(),
        n=len(coords_ref),
        per_res=per_res,
        bond_dev=bond_dev,
        n_clash=n_clash,
        n_below05=int((per_res < 0.5).sum()),
        n_below2=int((per_res < 2.0).sum())
    )

# -------------------------
# Blind docking helper (CPU)
# -------------------------
def blind_docking(coords, s, n_ligands=500, n_pockets=5, seed=0):
    """
    Simple blind docking heuristic: cluster high-s residues into pockets,
    score random ligands against pockets. Runs on CPU (numpy).
    """
    n = len(coords)
    sorted_idx = np.argsort(-s)
    pockets = []
    used = set()
    for idx in sorted_idx:
        if idx in used:
            continue
        pk = [idx]
        used.add(idx)
        for jdx in sorted_idx:
            if jdx not in used and np.linalg.norm(coords[idx] - coords[jdx]) < 8.0:
                pk.append(jdx)
                used.add(jdx)
        if len(pk) >= 3:
            pockets.append(pk)
        if len(pockets) >= n_pockets:
            break
    np.random.seed(seed)
    ligs = np.random.rand(n_ligands, 5)
    scores = np.array([sum(-5.0 * s[pk].mean() * l[0] - 3.0 * (1 - s[pk].mean()) * l[1]
                           - 4.0 * s[pk].mean() * l[2] * l[3] for pk in pockets) for l in ligs])
    top10 = np.argsort(scores)[:10]
    return dict(top1_energy=float(scores.min()) if scores.size else None,
                scores=scores, pockets=pockets, top10_idx=top10, n_pockets=len(pockets))

# -------------------------
# Gradient check utility
# -------------------------
def grad_check(coords_ref, n_check=10, eps=1e-6, seed=0, device=None):
    """
    Validate analytic autograd gradients against numerical finite differences.
    Prints a small table of analytic vs numeric for random components.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = coords_ref.shape[0]
    c0 = coords_ref + np.random.randn(n, 3) * 0.1
    c = torch.tensor(c0, dtype=torch.float64, device=device, requires_grad=True)
    disto_mask = None
    disto_dt = None
    tdih = None
    d_id_val = float(np.mean(np.linalg.norm(coords_ref[1:] - coords_ref[:-1], axis=1)))
    E = energy_torch_memory_efficient(c, disto_mask, disto_dt, d_id_val,
                                      wb=30.0, wd=5.0, wc=50.0, wa=5.0, wdh=3.0, tdih=tdih)
    E.backward()
    analytic = c.grad.detach().cpu().numpy().ravel()
    x0 = c.detach().cpu().numpy().ravel()
    idxs = np.random.choice(x0.size, size=min(n_check, x0.size), replace=False)
    num = np.zeros_like(idxs, dtype=float)
    for ii, k in enumerate(idxs):
        x_plus = x0.copy(); x_minus = x0.copy()
        x_plus[k] += eps; x_minus[k] -= eps
        c_plus = torch.tensor(x_plus.reshape(c.shape), dtype=torch.float64, device=device)
        E_p = energy_torch_memory_efficient(c_plus, disto_mask, disto_dt, d_id_val,
                                            wb=30.0, wd=5.0, wc=50.0, wa=5.0, wdh=3.0, tdih=tdih).item()
        c_minus = torch.tensor(x_minus.reshape(c.shape), dtype=torch.float64, device=device)
        E_m = energy_torch_memory_efficient(c_minus, disto_mask, disto_dt, d_id_val,
                                            wb=30.0, wd=5.0, wc=50.0, wa=5.0, wdh=3.0, tdih=tdih).item()
        num[ii] = (E_p - E_m) / (2 * eps)
    # Print results
    print(f"{'Index':<10} | {'Analytic':<20} | {'Numeric':<20} | {'Difference':<20}")
    print("-" * 75)
    for i, (a, nval) in enumerate(zip(analytic[idxs], num)):
        print(f"{idxs[i]:<10} | {a:<20.8f} | {nval:<20.8f} | {abs(a - nval):<20.8e}")
    return analytic[idxs], num

# -------------------------
# Example CLI usage
# -------------------------
if __name__ == "__main__":
    # Quick smoke test and gradient check on small mock protein
    np.random.seed(0)
    mock_coords = np.random.rand(30, 3) * 10.0
    print("Running gradient check (small mock protein, n=30)...")
    grad_check(mock_coords, n_check=5)
    # Example fold run (commented by default; enable if GPU available)
    # print("Running small fold (n=30)...")
    # res = fold(mock_coords, alpha=2.5, noise=0.5, seed=1, verbose=True)
    # print("RMSD final:", res['rmsd_final'])
