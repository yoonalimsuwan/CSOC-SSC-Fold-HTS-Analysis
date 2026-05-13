# ============================================================================
# CSOC-SSC UNIFIED FRAMEWORK
# Title: Criticality-Driven Differentiable Protein Folding via 3D ASM
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
# ============================================================================

import os
import time
import math
import json
import gzip
import pickle
import warnings
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List
from enum import Enum

import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial import cKDTree

__version__ = "12.0.0-Unified"

# ============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class CSOCUnifiedConfig:
    """Unified Configuration for ASM and Protein Physics."""
    # System
    device_id: int = 0
    cupy_vram_fraction: float = 0.3  # Leave 70% for PyTorch
    
    # ASM Configuration (Criticality Driver)
    asm_L: int = 64
    asm_alpha: float = 2.5
    asm_cutoff_factor: float = 4.0
    asm_gravity: float = 0.85
    
    # Protein Physics
    n_max: int = 100000
    n_stages: int = 3
    n_iter_per_stage: int = 500
    coarse_grain_factor: int = 4
    
    # Energies (Base Weights)
    weight_bond: float = 10.0
    weight_angle: float = 5.0
    weight_dihedral: float = 3.0
    weight_clash: float = 50.0
    weight_rama: float = 2.0
    weight_solvation: float = 5.0
    
    # Criticality Optimization
    optimizer_type: str = 'hybrid'
    learning_rate: float = 1e-3
    base_langevin_temp: float = 300.0
    use_amp: bool = True
    gradient_clip_norm: float = 1.0

# (Biochemical Constants Omitted for Brevity - Assume VDW_RADII, RAMACHANDRAN_PRIORS, HYDROPHOBICITY exist here as in original)
THREE2ONE = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

# ============================================================================
# SECTION 2: 3D ASM ENGINE (CUPY) - THE CRITICALITY DRIVER
# ============================================================================

class ZeroCopyFFTBuffer:
    def __init__(self, L: int, dtype=cp.float32):
        self.L = L
        target_shape = 2 * L - 1
        self.fft_size = next_fast_len(target_shape)
        self.fshape = (self.fft_size, self.fft_size, self.fft_size)
        self.padded = cp.zeros(self.fshape, dtype=dtype)
        self.view = self.padded[:L, :L, :L]

    def write_to_view(self, data: cp.ndarray):
        self.view[:] = data

class FFTConvolution3D:
    def __init__(self, L: int, kernel: cp.ndarray):
        self.L = L
        self.fft_buffer = ZeroCopyFFTBuffer(L, dtype=cp.float32)
        kernel_padded = cp.zeros(self.fft_buffer.fshape, dtype=cp.float32)
        kernel_padded[:L, :L, :L] = kernel
        self.kernel_fft = cp.fft.rfftn(kernel_padded)
        self.fshape = self.fft_buffer.fshape
        self.fft_size = self.fft_buffer.fft_size

    def convolve(self, signal: cp.ndarray) -> cp.ndarray:
        self.fft_buffer.write_to_view(signal)
        signal_fft = cp.fft.rfftn(self.fft_buffer.padded)
        product_fft = signal_fft * self.kernel_fft
        result_padded = cp.fft.irfftn(product_fft, s=self.fshape)
        start = (self.fft_size - self.L) // 2
        return result_padded[start:start+self.L, start:start+self.L, start:start+self.L]

class SandpileDynamics3D:
    """Provides realtime avalanches to drive the optimizer."""
    def __init__(self, config: CSOCUnifiedConfig):
        self.L = config.asm_L
        self.gravity = config.asm_gravity
        
        # Spatial Kernel setup
        z, y, x = np.meshgrid(np.fft.fftfreq(self.L)*self.L, np.fft.fftfreq(self.L)*self.L, np.fft.fftfreq(self.L)*self.L, indexing='ij')
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-6
        K = (r ** (-config.asm_alpha)) * np.exp(-r / (self.L / config.asm_cutoff_factor))
        K[self.L//2, self.L//2, self.L//2] = 0.0
        K /= K.sum()
        
        self.fft_conv = FFTConvolution3D(self.L, cp.array(K, dtype=cp.float32))
        self.S = cp.random.rand(self.L, self.L, self.L, dtype=cp.float32) * 0.8
        self.tp = cp.zeros((self.L, self.L, self.L), dtype=cp.float32)
        
        self.topple_kernel = cp.ElementwiseKernel(
            'float32 S_in', 'float32 tp, float32 S_out',
            'if (S_in >= 1.0f) { tp = floorf(S_in); S_out = S_in - tp; } else { tp = 0.0f; S_out = S_in; }',
            'topple_kernel'
        )

    def step_avalanche(self) -> int:
        """Add grain and compute avalanche size."""
        xi, yi, zi = cp.random.randint(1, self.L-1, size=3)
        self.S[xi, yi, zi] += self.gravity
        
        A = 0
        while True:
            self.topple_kernel(self.S, self.tp, self.S)
            num_topple = int(self.tp.sum())
            if num_topple == 0: break
            A += num_topple
            self.S += self.fft_conv.convolve(self.tp)
            # Absorbing Boundaries
            self.S[0,:,:] = self.S[-1,:,:] = self.S[:,0,:] = self.S[:,-1,:] = self.S[:,:,0] = self.S[:,:,-1] = 0
            self.tp[:] = 0
        return A

# ============================================================================
# SECTION 3: PROTEIN PHYSICS ENGINE (PYTORCH)
# ============================================================================

@dataclass
class BackboneFrame:
    n: np.ndarray; ca: np.ndarray; c: np.ndarray; o: np.ndarray
    residue_ids: List[int] = field(default_factory=list); seq: str = ""

class SparseContactNetwork:
    """PyTorch KD-Tree Sparse Contact Network"""
    def __init__(self, coords: np.ndarray, r_cut: float = 20.0):
        self.coords = coords
        self.tree = cKDTree(coords)
        pairs = []
        for i in range(len(coords)):
            idx = self.tree.query_ball_point(coords[i], r_cut)
            for j in idx:
                if j > i + 3: pairs.append([i, j])
        self.pairs = np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
        
    def to_torch(self, device) -> torch.Tensor:
        return torch.tensor(self.pairs, dtype=torch.long, device=device)

class HybridOptimizer(torch.optim.Optimizer):
    """AdamW + Langevin Brownian Dynamics."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, langevin_temperature=300.0)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                b1, b2 = group['betas']
                state['step'] += 1
                
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - b2 ** state['step'])).add_(group['eps'])
                step_size = group['lr'] / (1 - b1 ** state['step'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Langevin Thermal Noise injection
                T = group['langevin_temperature']
                if T > 0:
                    noise_scale = math.sqrt(T * group['lr'] / 300.0)
                    p.data.add_(torch.randn_like(p.data) * noise_scale)

# ============================================================================
# SECTION 4: UNIFIED ENGINE (THE BRIDGE)
# ============================================================================

class CriticalityDrivenOptimizationEngine:
    def __init__(self, config: CSOCUnifiedConfig):
        self.config = config
        
        # Init CuPy limits for ASM
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=config.cupy_vram_fraction)
        
        # Init ASM Driver
        print("[Unified Engine] Initializing 3D ASM Criticality Driver...")
        self.asm_driver = SandpileDynamics3D(config)
        
    def _compute_energy(self, ca_pt: torch.Tensor, backbone: BackboneFrame, sparse_pairs: torch.Tensor) -> torch.Tensor:
        """Simplified Differentiable Physics Energy."""
        E = torch.tensor(0.0, device=ca_pt.device)
        
        # Bond Energy
        dv = ca_pt[1:] - ca_pt[:-1]
        E += self.config.weight_bond * torch.sum((torch.norm(dv, dim=1) - 3.8)**2)
        
        # Clash Energy
        if len(sparse_pairs) > 0:
            dv_clash = ca_pt[sparse_pairs[:, 0]] - ca_pt[sparse_pairs[:, 1]]
            d_clash = torch.norm(dv_clash, dim=1)
            clash_mask = d_clash < 3.2
            if clash_mask.any():
                E += self.config.weight_clash * torch.sum((3.2 - d_clash[clash_mask])**2)
                
        return E

    def optimize(self, backbone: BackboneFrame) -> BackboneFrame:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ca_pt = torch.tensor(backbone.ca, dtype=torch.float32, device=device, requires_grad=True)
        
        optimizer = HybridOptimizer([ca_pt], lr=self.config.learning_rate)
        scaler = GradScaler() if self.config.use_amp else None
        
        print(f"Starting Criticality-Driven Folding ({len(backbone.ca)} residues)...")
        
        for stage in range(self.config.n_stages):
            sparse_net = SparseContactNetwork(ca_pt.detach().cpu().numpy())
            sparse_pairs = sparse_net.to_torch(device)
            
            for i in range(self.config.n_iter_per_stage):
                # 1. Step the ASM to get avalanche size (A)
                A = self.asm_driver.step_avalanche()
                
                # 2. Translate Criticality to Langevin Temperature: T(t) = T_base * (1 + log(1 + A))
                current_temp = self.config.base_langevin_temp * (1.0 + math.log1p(A))
                optimizer.param_groups[0]['langevin_temperature'] = current_temp
                
                # 3. Structural Optimization Step
                optimizer.zero_grad()
                with autocast(enabled=self.config.use_amp):
                    loss = self._compute_energy(ca_pt, backbone, sparse_pairs)
                
                if scaler:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_([ca_pt], self.config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                if i % 100 == 0:
                    print(f"  Stage {stage} | Iter {i} | Energy: {loss.item():.2f} | ASM Avalanche: {A} | Temp: {current_temp:.1f}K")

        # Return updated structure
        return BackboneFrame(n=backbone.n, ca=ca_pt.detach().cpu().numpy(), c=backbone.c, o=backbone.o, seq=backbone.seq)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("CSOC-SSC 3D ASM For Criticality-Driven Differentiable Protein Folding")
    print("="*70)
    
    config = CSOCUnifiedConfig()
    engine = CriticalityDrivenOptimizationEngine(config)
    
    # Generate Synthetic Backbone
    n_res = 200
    ca_synth = np.random.randn(n_res, 3).astype(np.float32) * 20.0
    synth_backbone = BackboneFrame(
        n=ca_synth - 0.5, ca=ca_synth, c=ca_synth + 0.5, o=ca_synth + 1.0, seq='A'*n_res
    )
    
    # Run Optimization
    optimized_backbone = engine.optimize(synth_backbone)
    print("\n✅ Optimization Complete!")
