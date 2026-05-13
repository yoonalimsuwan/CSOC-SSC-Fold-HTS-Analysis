"""
csoc_ssc_fold.py — CSOC-SSC v7 Core Module (PyTorch GPU Version - Memory Optimized)
MIT License — Yoon A Limsuwan 2026
github.com/yoonalimsuwan/SSC-SOC-Controlled-Criticality-
"""
import numpy as np
import torch
import os

__version__ = "7.1.0-pytorch-gpu-optimized"
__license__ = "MIT"

THREE2ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y','SEC':'U','MSE':'M','HSD':'H','HSE':'H',
}

def load_pdb_gz(path, chain='A', max_res=500):
    import gzip
    coords, seq = [], []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path,'rt',errors='ignore') as f:
        seen=set()
        for l in f:
            if not l.startswith('ATOM') or l[12:16].strip()!='CA': continue
            if l[21]!=chain: continue
            key=(int(l[22:26]),l[26])
            if key in seen: continue
            seen.add(key)
            coords.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
            seq.append(THREE2ONE.get(l[17:20].strip(),'X'))
            if len(coords)>=max_res: break
    return (np.array(coords),''.join(seq)) if coords else (None,'')

# 🚀 PyTorch Accelerated Contact Map (Memory Efficient)
def csoc_contact_map_pt(coords_pt, alpha=2.5, r_cut=15.0):
    n = coords_pt.shape[0]
    
    # Use torch.cdist for VRAM-efficient pairwise distances instead of unsqueeze broadcasting
    D = torch.cdist(coords_pt, coords_pt)
    
    # Generate meshgrid indices
    i_idx, j_idx = torch.meshgrid(
        torch.arange(n, device=coords_pt.device), 
        torch.arange(n, device=coords_pt.device), 
        indexing='ij'
    )
    abs_diff = torch.abs(i_idx - j_idx)
    
    r = abs_diff.double() + 1e-8
    kernel = (r**(-alpha)) * torch.exp(-r/12.0)
    
    mask = (abs_diff > 3) & (D < r_cut)
    C = torch.zeros((n, n), dtype=torch.float64, device=coords_pt.device)
    C[mask] = (1.0 - D[mask] / r_cut) * (1.0 + 0.3 * kernel[mask])
    
    # Safe normalization preventing zero division
    max_c = C.max(dim=1, keepdim=True).values
    nz = (max_c > 0).flatten()
    C[nz, :] /= (max_c[nz] + 1e-8)
    return C, D

# 🚀 PyTorch Accelerated SSC States (Vectorized)
def ssc_states_pt(coords, alpha=2.5, T=150, device='cuda'):
    coords_pt = torch.tensor(coords, dtype=torch.float64, device=device)
    n = coords_pt.shape[0]
    C, D = csoc_contact_map_pt(coords_pt, alpha)
    
    ctr = coords_pt.mean(dim=0)
    dc = torch.norm(coords_pt - ctr, dim=1)
    dc_min, dc_max = dc.min(), dc.max()
    s = torch.clamp(1.0 - (dc - dc_min) / (dc_max - dc_min + 1e-8), 0.05, 0.95)
    
    i_idx, j_idx = torch.meshgrid(
        torch.arange(n, device=device), 
        torch.arange(n, device=device), 
        indexing='ij'
    )
    r = torch.abs(i_idx - j_idx).double() + 1e-8
    W = (r**(-alpha)) * torch.exp(-r/12.0)
    W.fill_diagonal_(0.0)
    W_sum = W.sum(dim=1) + 1e-8
    
    for t in range(T):
        g = -(C @ s) + 0.05 * (1.0 - 2.0 * s)
        sp = s.clone()
        
        # Vectorized network update
        s_update = s + 0.08 * (W @ s) / W_sum
        sp = torch.clamp(s_update, 0.0, 1.0)
        
        eta = 0.04 if t < T//2 else 0.02
        s = torch.clamp(s - eta*g + 0.1*(sp - s), 0.0, 1.0)
        
    return s.cpu().numpy(), C.cpu().numpy(), D.cpu().numpy()

def kabsch(P, Q):
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc)**2, axis=1)))), Pr

# 🚀 PyTorch Energy Function (Memory-Efficient & Fully Differentiable)
def energy_torch_memory_efficient(c, disto_mask, disto_dt, d_id, wb, wd, wc, wa, wdh):
    n = c.shape[0]
    E = torch.tensor(0.0, dtype=c.dtype, device=c.device)
    
    # 1. Bonds (O(n) logic, no changes needed)
    dv = c[1:] - c[:-1]
    d = torch.norm(dv, dim=1) + 1e-8
    E = E + wb * torch.sum((d - d_id)**2)
    
    # 2. Banded Clashes (Vectorized without loops for Autograd)
    max_offset = min(20, n)
    if max_offset > 3:
        # Create base indices and offsets
        i_base = torch.arange(n, device=c.device).unsqueeze(1)
        offsets = torch.arange(3, max_offset, device=c.device).unsqueeze(0)
        
        j_idx = i_base + offsets
        
        # Mask out invalid indices (j >= n)
        valid_mask = j_idx < n
        
        # Flatten valid pairs to avoid N x N matrices
        i_flat = i_base.expand_as(j_idx)[valid_mask]
        j_flat = j_idx[valid_mask]
        
        diff = c[i_flat] - c[j_flat]
        d_pair = torch.norm(diff, dim=1) + 1e-8
        mask_clash = d_pair < 3.8
        
        if mask_clash.any():
            dev = 3.8 - d_pair[mask_clash]
            E = E + wc * torch.sum(dev**2)

    # 3. Distogram Sparse Masking
    # Only computes distances for the explicit pairs defined in disto_mask
    if disto_mask is not None and disto_mask.shape[0] > 0:
        dv_disto = c[disto_mask[:, 0]] - c[disto_mask[:, 1]]
        d_disto = torch.norm(dv_disto, dim=1) + 1e-8
        ex = torch.abs(d_disto - disto_dt) - 0.05
        mask_ex = ex > 0
        if mask_ex.any():
            E = E + wd * torch.sum(ex[mask_ex]**2)

    return E

def fold(coords_ref, alpha=2.5, noise=0.5, seed=1, verbose=False):
    n = len(coords_ref)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose: print(f"Executing PyTorch pipeline on device: {device}")
    
    # Run SSC on PyTorch
    s, C, D_true = ssc_states_pt(coords_ref, alpha=alpha, device=device)
    
    disto = [(i, j, D_true[i,j]) for i in range(n) for j in range(i+4, n) if D_true[i,j] < 15.0]
    if len(disto) > 0:
        disto_mask_pt = torch.tensor([[d[0], d[1]] for d in disto], dtype=torch.long, device=device)
        disto_dt_pt = torch.tensor([d[2] for d in disto], dtype=torch.float64, device=device)
    else:
        disto_mask_pt, disto_dt_pt = None, None
    
    d_id = float(np.mean([np.linalg.norm(coords_ref[i+1]-coords_ref[i]) for i in range(n-1)]))
    c0 = coords_ref + np.random.randn(n, 3) * noise
    
    # Initialize optimization tensor on GPU with requires_grad
    c_var = torch.tensor(c0, dtype=torch.float64, device=device, requires_grad=True)
    stages = []
    
    def _run_optimization_stage(use_disto, wb, wd, wc, wa, wdh, max_iter, ftol):
        d_m = disto_mask_pt if use_disto else None
        d_dt = disto_dt_pt if use_disto else None
        
        # PyTorch L-BFGS (Runs entirely on GPU)
        optimizer = torch.optim.LBFGS([c_var], max_iter=max_iter, tolerance_change=ftol, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            E = energy_torch_memory_efficient(c_var, d_m, d_dt, d_id, wb, wd, wc, wa, wdh)
            E.backward()
            return E
            
        optimizer.step(closure)
        return c_var.detach().cpu().numpy()
    
    if verbose: print("Minimizing strictly on GPU via PyTorch Autograd...")
    
    x1 = _run_optimization_stage(False, 30, 0, 0, 8, 0, 300, 1e-11)
    stages.append(kabsch(x1, coords_ref)[0])
    
    x2 = _run_optimization_stage(False, 25, 0, 0, 8, 5, 400, 1e-12)
    stages.append(kabsch(x2, coords_ref)[0])
    
    x3 = _run_optimization_stage(True, 20, 5, 0, 5, 5, 600, 1e-13)
    stages.append(kabsch(x3, coords_ref)[0])
    
    c4 = _run_optimization_stage(True, 15, 20, 80, 5, 8, 800, 1e-14)
    rmsd, Prot = kabsch(c4, coords_ref)
    stages.append(rmsd)
    
    if verbose:
        for i, r in enumerate(stages): print(f" Stage {i+1}: {r:.4f} Å")
        
    return dict(rmsd_final=rmsd, coords_pred=c4, s=s)

def blind_docking(coords, s, n_ligands=500, n_pockets=5, seed=0):
    n = len(coords)
    sorted_idx = np.argsort(-s)
    pockets = []
    used = set()
    
    for idx in sorted_idx:
        if idx in used: continue
        pk = [idx]
        used.add(idx)  # Added to prevent self-overlap bug
        for jdx in sorted_idx:
            if jdx not in used and np.linalg.norm(coords[idx] - coords[jdx]) < 8:
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
    
    return dict(top1_energy=float(scores.min()), scores=scores, 
                pockets=pockets, top10_idx=top10, n_pockets=len(pockets))
