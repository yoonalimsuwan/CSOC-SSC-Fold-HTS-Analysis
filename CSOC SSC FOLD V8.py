import torch
import numpy as np
import time

# ==========================================
# 1. Energy Functions
# ==========================================

def dihedral_energy(c, tdih, wdh):
    """
    Computes the differentiable dihedral angle energy penalty.
    c: (n, 3) coordinate tensor
    tdih: (n-3,) target dihedral angles in degrees
    """
    if tdih is None or wdh == 0:
        return torch.tensor(0.0, dtype=c.dtype, device=c.device)
        
    b1 = c[1:-2] - c[:-3]
    b2 = c[2:-1] - c[1:-2]
    b3 = c[3:]   - c[2:-1]
    
    nv1 = torch.cross(b1, b2, dim=1)
    nv2 = torch.cross(b2, b3, dim=1)
    
    n1 = torch.norm(nv1, dim=1) + 1e-8
    n2 = torch.norm(nv2, dim=1) + 1e-8
    
    # Clamp to prevent NaN in arccos
    cos_phi = torch.clamp(torch.sum(nv1 * nv2, dim=1) / (n1 * n2), -1.0, 1.0)
    phi = torch.acos(cos_phi)  # in radians
    
    if not isinstance(tdih, torch.Tensor):
        tdih = torch.tensor(tdih, dtype=c.dtype, device=c.device)
        
    tdih_rad = tdih * (np.pi / 180.0)
    return wdh * torch.sum((phi - tdih_rad) ** 2)


def distogram_batch_eval(c, disto_mask, disto_dt, wd, batch_size=10000):
    """
    Evaluates distogram pairs in batches to prevent GPU OOM errors on large proteins.
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
        
        # Allow a 0.05 tolerance margin
        ex = torch.abs(d - disto_dt[start:end]) - 0.05
        mask_ex = ex > 0
        
        if mask_ex.any():
            E_sum = E_sum + wd * torch.sum(ex[mask_ex] ** 2)
            
    return E_sum


def energy_torch_memory_efficient(c, disto_mask, disto_dt, d_id, wb=30, wd=5, wc=50, wa=5, wdh=3, tdih=None):
    """
    Core energy computation combining all physical and statistical constraints.
    """
    E = torch.tensor(0.0, dtype=c.dtype, device=c.device)
    n = c.shape[0]
    
    # 1. Bond Length Energy (consecutive points)
    dv = c[1:] - c[:-1]
    d = torch.norm(dv, dim=1)
    E = E + wb * torch.sum((d - d_id) ** 2)
    
    # 2. Distogram Energy (Batched)
    E = E + distogram_batch_eval(c, disto_mask, disto_dt, wd)
    
    # 3. Clash Energy (Banded local search)
    max_offset = min(20, n - 1)
    if max_offset > 3:
        for offset in range(3, max_offset + 1):
            dv_clash = c[:-offset] - c[offset:]
            d_clash = torch.norm(dv_clash, dim=1)
            clash_mask = d_clash < 3.0  # Assumed 3.0 Angstrom threshold
            if clash_mask.any():
                E = E + wc * torch.sum((3.0 - d_clash[clash_mask]) ** 2)

    # 4. Dihedral Energy
    E = E + dihedral_energy(c, tdih, wdh)
    
    return E


# ==========================================
# 2. Optimization Pipeline
# ==========================================

def run_optimization_stage(c_init, max_iter, ftol, device, **energy_kwargs):
    """
    Runs a single optimization stage. Re-initializes the parameter tensor to 
    clear stale L-BFGS states and prevent memory leaks.
    """
    # Create a fresh parameter tensor
    c_var = torch.tensor(c_init, dtype=torch.float64, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS(
        [c_var], 
        max_iter=max_iter, 
        tolerance_change=ftol, 
        line_search_fn='strong_wolfe'
    )
    
    def closure():
        optimizer.zero_grad()
        loss = energy_torch_memory_efficient(c_var, **energy_kwargs)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    # Return as numpy only at the absolute end of the stage if required by CPU tools
    return c_var.detach().cpu().numpy()


# ==========================================
# 3. Validation & Testing
# ==========================================

def grad_check(coords_ref, n_check=10, eps=1e-6, seed=0):
    """
    Validates analytic autograd gradients against numerical finite differences.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n = coords_ref.shape[0]
    c0 = coords_ref + np.random.randn(n, 3) * 0.1
    c = torch.tensor(c0, dtype=torch.float64, device=device, requires_grad=True)

    # Mock variables for testing
    disto_mask = None
    disto_dt = None
    tdih = None
    
    d_id_val = float(np.mean(np.linalg.norm(coords_ref[1:] - coords_ref[:-1], axis=1)))

    # Compute Analytic Gradients
    E = energy_torch_memory_efficient(
        c, disto_mask, disto_dt, d_id=d_id_val,
        wb=30, wd=5, wc=50, wa=5, wdh=3, tdih=tdih
    )
    E.backward()
    analytic = c.grad.detach().cpu().numpy().ravel()

    # Compute Numerical Approximation
    x0 = c.detach().cpu().numpy().ravel()
    idxs = np.random.choice(x0.size, size=min(n_check, x0.size), replace=False)
    num = np.zeros_like(idxs, dtype=float)
    
    for ii, k in enumerate(idxs):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[k] += eps
        x_minus[k] -= eps
        
        c_plus = torch.tensor(x_plus.reshape(c.shape), dtype=torch.float64, device=device)
        E_p = energy_torch_memory_efficient(
            c_plus, disto_mask, disto_dt, d_id=d_id_val,
            wb=30, wd=5, wc=50, wa=5, wdh=3, tdih=tdih
        ).item()
        
        c_minus = torch.tensor(x_minus.reshape(c.shape), dtype=torch.float64, device=device)
        E_m = energy_torch_memory_efficient(
            c_minus, disto_mask, disto_dt, d_id=d_id_val,
            wb=30, wd=5, wc=50, wa=5, wdh=3, tdih=tdih
        ).item()
        
        num[ii] = (E_p - E_m) / (2 * eps)
        
    # Print results
    print(f"{'Index':<10} | {'Analytic':<20} | {'Numeric':<20} | {'Difference':<20}")
    print("-" * 75)
    for i, (a, n) in enumerate(zip(analytic[idxs], num)):
        print(f"{idxs[i]:<10} | {a:<20.8f} | {n:<20.8f} | {abs(a - n):<20.8e}")
        
    return analytic[idxs], num

# Example Usage:
if __name__ == "__main__":
    # Generate random mock coordinates for a small protein (N=30)
    mock_coords = np.random.rand(30, 3) * 10.0
    print("Running Gradient Check...")
    grad_check(mock_coords, n_check=5)
