# ============================================================================
# CSOC-SSC DE NOVO PREDICTOR
# Title: Criticality-Driven End-to-End Differentiable Protein Predictor
# ============================================================================

import math
import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================================
# SECTION 1: CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class CSOCDeNovoConfig:
    device_id: int = 0
    cupy_vram_fraction: float = 0.3
    
    # ASM Criticality
    asm_L: int = 64
    asm_alpha: float = 2.5
    asm_cutoff_factor: float = 4.0
    asm_gravity: float = 0.85
    
    # ML & Recycling (De Novo specifics)
    embed_dim: int = 64
    n_recycling: int = 3          # Structural Recycling rounds
    n_stages: int = 3
    n_iter_per_stage: int = 300
    
    # Loss Weights
    w_bond: float = 10.0
    w_angle: float = 5.0
    w_clash: float = 50.0
    w_distogram: float = 20.0     # ML Prior weight
    
    # Optimizer
    learning_rate: float = 2e-3
    base_langevin_temp: float = 300.0

@dataclass
class BackboneFrame:
    ca: np.ndarray
    seq: str
    n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o: np.ndarray = field(default_factory=lambda: np.zeros(0))

# ============================================================================
# SECTION 2: MACHINE LEARNING MODULE (DISTOGRAM NET)
# ============================================================================

class DistogramNet(nn.Module):
    """
    Simulated Sequence-to-Distogram Network.
    In a fully trained model, this contains Evoformer blocks and uses MSA.
    Here, it predicts pairwise distances from sequence embeddings.
    """
    def __init__(self, vocab_size=21, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 1D to 2D pairwise feature extraction
        self.pairwise_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.resnet = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Output: Predicted distance expected value (regression for simplicity)
        self.dist_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        # seq_tokens: (L)
        L = seq_tokens.shape[0]
        x_1d = self.embedding(seq_tokens) # (L, E)
        
        # Create pairwise features: (L, L, 2E)
        x_i = x_1d.unsqueeze(1).expand(L, L, -1)
        x_j = x_1d.unsqueeze(0).expand(L, L, -1)
        x_pair = torch.cat([x_i, x_j], dim=-1)
        
        z = self.pairwise_proj(x_pair) # (L, L, E)
        z = z.permute(2, 0, 1).unsqueeze(0) # (1, E, L, L)
        
        z = z + self.resnet(z)
        dist_pred = F.softplus(self.dist_head(z).squeeze(0).squeeze(0)) # (L, L), > 0
        
        # Mask diagonal
        mask = 1.0 - torch.eye(L, device=seq_tokens.device)
        return dist_pred * mask

# ============================================================================
# SECTION 3: 3D ASM ENGINE (CUPY) - CRITICALITY DRIVER
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
        self.fft_buffer = ZeroCopyFFTBuffer(L)
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
    def __init__(self, config: CSOCDeNovoConfig):
        self.L = config.asm_L
        self.gravity = config.asm_gravity
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
        xi, yi, zi = cp.random.randint(1, self.L-1, size=3)
        self.S[xi, yi, zi] += self.gravity
        A = 0
        while True:
            self.topple_kernel(self.S, self.tp, self.S)
            num_topple = int(self.tp.sum())
            if num_topple == 0: break
            A += num_topple
            self.S += self.fft_conv.convolve(self.tp)
            self.S[0,:,:] = self.S[-1,:,:] = self.S[:,0,:] = self.S[:,-1,:] = self.S[:,:,0] = self.S[:,:,-1] = 0
            self.tp[:] = 0
        return A

# ============================================================================
# SECTION 4: HYBRID OPTIMIZER
# ============================================================================

class HybridOptimizer(torch.optim.Optimizer):
    """AdamW with Langevin Dynamics modulated by ASM Criticality."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps, langevin_temperature=300.0)
        super().__init__(params, defaults)
    
    def step(self):
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
                
                T = group['langevin_temperature']
                if T > 0:
                    noise_scale = math.sqrt(T * group['lr'] / 300.0)
                    p.data.add_(torch.randn_like(p.data) * noise_scale)

# ============================================================================
# SECTION 5: DE NOVO FOLDING PIPELINE
# ============================================================================

class DeNovoPredictor:
    def __init__(self, config: CSOCDeNovoConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Init Submodules
        self.asm_driver = SandpileDynamics3D(config)
        self.distogram_net = DistogramNet(embed_dim=config.embed_dim).to(self.device)
        
        # Create a mapping for Amino Acids to integer tokens
        self.aa2int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    def _seq_to_tokens(self, seq: str) -> torch.Tensor:
        tokens = [self.aa2int.get(aa, 20) for aa in seq]
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def _compute_loss(self, ca_pt: torch.Tensor, pred_distogram: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Calculates Physical + ML Prior Losses."""
        metrics = {}
        L = ca_pt.shape[0]
        
        # 1. Physical Physics Constraints (Realism upgrades)
        # 1.1 Bond Distance (C_alpha to C_alpha ~ 3.8A)
        dv_bond = ca_pt[1:] - ca_pt[:-1]
        bond_dist = torch.norm(dv_bond, dim=1)
        loss_bond = torch.mean((bond_dist - 3.8)**2)
        
        # 1.2 Angle Geometry (Adjacent bond vectors should form reasonable angles)
        v1 = dv_bond[:-1]
        v2 = dv_bond[1:]
        cos_theta = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-6)
        # Encourage angles around 90-120 degrees (cos_theta ~ -0.5 to 0)
        loss_angle = torch.mean((cos_theta + 0.25)**2) 
        
        # 1.3 Steric Clash Removal (Global)
        dist_matrix = torch.cdist(ca_pt, ca_pt)
        mask_clash = (dist_matrix < 3.2) & (torch.eye(L, device=self.device) == 0)
        loss_clash = torch.sum((3.2 - dist_matrix[mask_clash])**2) / (torch.sum(mask_clash) + 1e-6)
        
        # 2. ML Distogram Prior Constraint
        # Pull actual distances toward DistogramNet's predicted distances
        mask_valid = torch.eye(L, device=self.device) == 0
        loss_distogram = torch.mean((dist_matrix[mask_valid] - pred_distogram[mask_valid])**2)

        # 3. Total Energy
        total_loss = (self.config.w_bond * loss_bond) + \
                     (self.config.w_angle * loss_angle) + \
                     (self.config.w_clash * loss_clash) + \
                     (self.config.w_distogram * loss_distogram)
                     
        metrics = {
            'bond': loss_bond.item(), 'angle': loss_angle.item(),
            'clash': loss_clash.item(), 'dist': loss_distogram.item()
        }
        return total_loss, metrics

    def predict(self, seq: str) -> BackboneFrame:
        print(f"🧬 Starting De Novo Prediction for Sequence Length: {len(seq)}")
        
        # 1. ML Encoding: Sequence to Predicted Distogram Prior
        seq_tokens = self._seq_to_tokens(seq)
        with torch.no_grad(): # In inference, we don't train the network
            pred_distogram = self.distogram_net(seq_tokens)
        
        # 2. Initialize Latent Coordinates (Random)
        n_res = len(seq)
        ca_init = np.random.randn(n_res, 3).astype(np.float32) * 10.0
        ca_pt = torch.tensor(ca_init, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 3. Structural Recycling Loop
        for recycle in range(self.config.n_recycling):
            print(f"\n🔄 Recycling Round {recycle + 1}/{self.config.n_recycling}")
            
            optimizer = HybridOptimizer([ca_pt], lr=self.config.learning_rate)
            scaler = GradScaler()
            
            for stage in range(self.config.n_stages):
                for i in range(self.config.n_iter_per_stage):
                    # ASM Criticality drives Langevin Exploration
                    A = self.asm_driver.step_avalanche()
                    T_current = self.config.base_langevin_temp * (1.0 + math.log1p(A))
                    
                    # Reduce temperature slightly over stages (Annealing logic)
                    T_current *= (1.0 - stage/self.config.n_stages)
                    optimizer.param_groups[0]['langevin_temperature'] = T_current
                    
                    optimizer.zero_grad()
                    with autocast():
                        loss, metrics = self._compute_loss(ca_pt, pred_distogram)
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_([ca_pt], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if i % 100 == 0:
                        print(f"  Stage {stage}| Iter {i:3} | Loss: {loss.item():.2f} | "
                              f"DistLoss: {metrics['dist']: .2f} | Temp: {T_current:.1f}K | A: {A}")
            
            # Reparameterize and center coordinates before next recycle
            with torch.no_grad():
                ca_pt -= torch.mean(ca_pt, dim=0)
        
        print("\n✅ Final Prediction Complete!")
        return BackboneFrame(seq=seq, ca=ca_pt.detach().cpu().numpy())

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    config = CSOCDeNovoConfig()
    predictor = DeNovoPredictor(config)
    
    # Example Target Sequence (Small arbitrary domain)
    target_sequence = "MKTLLLTLVVVTIVCLDLGYAT" * 3 
    
    # Run the Predictor
    predicted_structure = predictor.predict(target_sequence)
