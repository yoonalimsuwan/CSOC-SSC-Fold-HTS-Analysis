# ============================================================================
# CSOC-SSC DE NOVO PREDICTOR
# Title: Criticality-Driven End-to-End Differentiable Protein Predictor
# Version: 2.0 - Production Ready
# ============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

# Optional CuPy import with graceful fallback
try:
    import cupy as cp
    from cupyx.scipy.fft import next_fast_len
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. ASM Driver will be skipped on CPU execution.", UserWarning)

# ============================================================================
# SECTION 1: CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class CSOCDeNovoConfig:
    """Configuration for CSOC De Novo Protein Predictor."""
    device_id: int = 0
    cupy_vram_fraction: float = 0.3
    
    # ASM (Avalanche Sandpile Model) Criticality Parameters
    asm_L: int = 64
    asm_alpha: float = 2.5
    asm_cutoff_factor: float = 4.0
    asm_gravity: float = 0.85
    
    # Machine Learning & Recycling Parameters
    embed_dim: int = 64
    n_recycling: int = 3
    n_stages: int = 3
    n_iter_per_stage: int = 300
    
    # Loss Function Weights
    w_bond: float = 10.0
    w_angle: float = 5.0
    w_clash: float = 50.0
    w_distogram: float = 20.0
    
    # Optimizer & Temperature Parameters
    learning_rate: float = 2e-3
    base_langevin_temp: float = 300.0
    grad_clip_norm: float = 1.0


@dataclass
class BackboneFrame:
    """Represents protein backbone coordinates and metadata."""
    seq: str
    ca: np.ndarray
    n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ============================================================================
# SECTION 2: MACHINE LEARNING MODULE (DISTOGRAM NET)
# ============================================================================

class DistogramNet(nn.Module):
    """
    Sequence-to-Distogram Neural Network.
    
    Converts amino acid sequences into predicted pairwise distance distributions.
    Uses embedding layer followed by 2D convolutional refinement.
    """
    def __init__(self, vocab_size: int = 21, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 1D-to-2D pairwise feature extraction
        self.pairwise_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # 2D convolutional blocks for refinement
        self.resnet = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Output head for distance prediction
        self.dist_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sequence tokens -> predicted distance matrix.
        
        Args:
            seq_tokens: Tensor of shape (L,) containing token indices
            
        Returns:
            dist_pred: Tensor of shape (L, L) with predicted distances
        """
        L = seq_tokens.shape[0]
        
        # Embed sequence
        x_1d = self.embedding(seq_tokens)  # (L, embed_dim)
        
        # Create pairwise features
        x_i = x_1d.unsqueeze(1).expand(L, L, -1)
        x_j = x_1d.unsqueeze(0).expand(L, L, -1)
        x_pair = torch.cat([x_i, x_j], dim=-1)  # (L, L, 2*embed_dim)
        
        # Project to embedding dimension
        z = self.pairwise_proj(x_pair)  # (L, L, embed_dim)
        z = z.permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, L, L)
        
        # Refine with ResNet
        z = z + self.resnet(z)
        
        # Predict distances (positive values)
        dist_pred = F.softplus(self.dist_head(z).squeeze(0).squeeze(0))  # (L, L)
        
        # Mask diagonal (self-distances)
        mask = 1.0 - torch.eye(L, device=seq_tokens.device)
        return dist_pred * mask


# ============================================================================
# SECTION 3: 3D ASM ENGINE (CUPY) - CRITICALITY DRIVER
# ============================================================================

class ZeroCopyFFTBuffer:
    """Memory-efficient FFT buffer with zero-copy view."""
    def __init__(self, L: int, dtype=np.float32):
        self.L = L
        target_shape = 2 * L - 1
        self.fft_size = self._next_fast_len(target_shape)
        self.fshape = (self.fft_size, self.fft_size, self.fft_size)
        
        if CUPY_AVAILABLE:
            self.padded = cp.zeros(self.fshape, dtype=dtype)
            self.view = self.padded[:L, :L, :L]
        else:
            self.padded = np.zeros(self.fshape, dtype=dtype)
            self.view = self.padded[:L, :L, :L]
    
    @staticmethod
    def _next_fast_len(n: int) -> int:
        """Find the next power of 2 >= n for FFT efficiency."""
        return 2 ** int(np.ceil(np.log2(n)))

    def write_to_view(self, data):
        """Write data to the view with type conversion if needed."""
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            self.view[:] = data
        else:
            self.view[:] = data if isinstance(self.view, type(data)) else np.array(data)


class FFTConvolution3D:
    """3D convolution using FFT for computational efficiency."""
    def __init__(self, L: int, kernel):
        self.L = L
        self.fft_buffer = ZeroCopyFFTBuffer(L)
        
        if CUPY_AVAILABLE:
            kernel_padded = cp.zeros(self.fft_buffer.fshape, dtype=cp.float32)
            kernel_padded[:L, :L, :L] = cp.array(kernel) if not isinstance(kernel, cp.ndarray) else kernel
            self.kernel_fft = cp.fft.rfftn(kernel_padded)
        else:
            kernel_padded = np.zeros(self.fft_buffer.fshape, dtype=np.float32)
            kernel_padded[:L, :L, :L] = kernel if isinstance(kernel, np.ndarray) else np.array(kernel)
            self.kernel_fft = np.fft.rfftn(kernel_padded)
        
        self.fshape = self.fft_buffer.fshape
        self.fft_size = self.fft_buffer.fft_size

    def convolve(self, signal):
        """Perform 3D convolution in Fourier space."""
        self.fft_buffer.write_to_view(signal)
        
        if CUPY_AVAILABLE:
            signal_fft = cp.fft.rfftn(self.fft_buffer.padded)
            product_fft = signal_fft * self.kernel_fft
            result_padded = cp.fft.irfftn(product_fft, s=self.fshape)
        else:
            signal_fft = np.fft.rfftn(self.fft_buffer.padded)
            product_fft = signal_fft * self.kernel_fft
            result_padded = np.fft.irfftn(product_fft, s=self.fshape)
        
        start = (self.fft_size - self.L) // 2
        return result_padded[start:start+self.L, start:start+self.L, start:start+self.L]


class SandpileDynamics3D:
    """
    3D Sandpile model for avalanche-driven criticality.
    
    Models self-organized criticality to guide exploration temperature
    in the optimization landscape.
    """
    def __init__(self, config: CSOCDeNovoConfig):
        self.L = config.asm_L
        self.gravity = config.asm_gravity
        
        # Construct 3D kernel using power-law decay
        z, y, x = np.meshgrid(
            np.fft.fftfreq(self.L) * self.L,
            np.fft.fftfreq(self.L) * self.L,
            np.fft.fftfreq(self.L) * self.L,
            indexing='ij'
        )
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-6
        K = (r ** (-config.asm_alpha)) * np.exp(-r / (self.L / config.asm_cutoff_factor))
        K[self.L//2, self.L//2, self.L//2] = 0.0
        K /= K.sum()
        
        # Initialize FFT convolution engine
        if CUPY_AVAILABLE:
            self.fft_conv = FFTConvolution3D(self.L, cp.array(K, dtype=cp.float32))
            self.S = cp.random.rand(self.L, self.L, self.L, dtype=cp.float32) * 0.8
            self.tp = cp.zeros((self.L, self.L, self.L), dtype=cp.float32)
        else:
            self.fft_conv = FFTConvolution3D(self.L, K.astype(np.float32))
            self.S = np.random.rand(self.L, self.L, self.L).astype(np.float32) * 0.8
            self.tp = np.zeros((self.L, self.L, self.L), dtype=np.float32)

    def step_avalanche(self) -> int:
        """
        Execute one avalanche step in the sandpile.
        
        Returns:
            Number of toppling events (avalanche size)
        """
        if CUPY_AVAILABLE:
            xi, yi, zi = cp.random.randint(1, self.L-1, size=3)
        else:
            xi, yi, zi = np.random.randint(1, self.L-1, size=3)
        
        self.S[xi, yi, zi] += self.gravity
        
        A = 0
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            # Topple: S >= 1.0 -> tp += floor(S), S -= floor(S)
            if CUPY_AVAILABLE:
                self.tp = cp.floor(cp.maximum(self.S, 0.0))
                self.S = self.S - self.tp
            else:
                self.tp = np.floor(np.maximum(self.S, 0.0))
                self.S = self.S - self.tp
            
            num_topple = int(self.tp.sum()) if CUPY_AVAILABLE else int(np.sum(self.tp))
            if num_topple == 0:
                break
            
            A += num_topple
            
            # Distribute toppled sand via convolution
            self.S = self.S + self.fft_conv.convolve(self.tp)
            
            # Apply boundary conditions (closed boundaries)
            if CUPY_AVAILABLE:
                self.S[0, :, :] = self.S[-1, :, :] = 0.0
                self.S[:, 0, :] = self.S[:, -1, :] = 0.0
                self.S[:, :, 0] = self.S[:, :, -1] = 0.0
            else:
                self.S[0, :, :] = self.S[-1, :, :] = 0.0
                self.S[:, 0, :] = self.S[:, -1, :] = 0.0
                self.S[:, :, 0] = self.S[:, :, -1] = 0.0
            
            self.tp[:] = 0.0
            iteration += 1
        
        return A


# ============================================================================
# SECTION 4: HYBRID OPTIMIZER
# ============================================================================

class HybridOptimizer(torch.optim.Optimizer):
    """
    AdamW optimizer enhanced with Langevin dynamics.
    
    Temperature modulation based on ASM criticality enables adaptive
    exploration-exploitation balance during protein folding optimization.
    """
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 1e-4):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            langevin_temperature=300.0
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step with optional Langevin noise.
        
        Args:
            closure: Optional function that reevaluates the model
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                b1, b2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                
                # Bias correction
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                
                # Compute step size with bias correction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # AdamW update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Optional Langevin noise for exploration
                T = group['langevin_temperature']
                if T > 0:
                    # Scale noise by temperature and learning rate
                    noise_scale = math.sqrt(2 * T * group['lr'] / 300.0)
                    noise = torch.randn_like(p.data) * noise_scale
                    p.data.add_(noise)
        
        return loss


# ============================================================================
# SECTION 5: DE NOVO FOLDING PIPELINE
# ============================================================================

class DeNovoPredictor:
    """
    Main De Novo protein structure prediction pipeline.
    
    Combines machine learning priors (distogram) with physics-based
    constraints and criticality-driven exploration.
    """
    def __init__(self, config: CSOCDeNovoConfig):
        self.config = config
        
        # Determine device
        self.device = torch.device(
            f'cuda:{config.device_id}' if torch.cuda.is_available() else 'cpu'
        )
        print(f"📊 Device: {self.device}")
        
        # Initialize ASM driver (only on CUDA with CuPy)
        self.asm_driver = None
        if self.device.type == 'cuda' and CUPY_AVAILABLE:
            try:
                self.asm_driver = SandpileDynamics3D(config)
                print("✅ ASM Driver initialized with CuPy")
            except Exception as e:
                warnings.warn(f"Failed to initialize ASM Driver: {e}. Proceeding without it.", UserWarning)
        else:
            print("⚠️  ASM Driver disabled (requires CUDA + CuPy)")
        
        # Initialize neural network
        self.distogram_net = DistogramNet(embed_dim=config.embed_dim).to(self.device)
        self.distogram_net.eval()
        
        # Amino acid to token mapping
        self.aa2int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    def _seq_to_tokens(self, seq: str) -> torch.Tensor:
        """Convert amino acid sequence to token indices."""
        tokens = [self.aa2int.get(aa, 20) for aa in seq]
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def _initialize_backbone(self, n_res: int) -> torch.Tensor:
        """
        Initialize backbone coordinates with proper scaling.
        
        Uses radius proportional to sequence length for realistic protein size.
        """
        # Estimate radius based on sequence length
        # Typical C-alpha spacing ~ 3.8 Angstrom
        estimated_radius = max(3.8 * (n_res ** (1/3)), 15.0)
        
        # Random initialization with appropriate scale
        ca_init = np.random.randn(n_res, 3).astype(np.float32)
        ca_init = ca_init / np.linalg.norm(ca_init, axis=1, keepdims=True)
        ca_init = ca_init * (estimated_radius / 3.0)
        
        return torch.tensor(ca_init, dtype=torch.float32, device=self.device, requires_grad=True)

    def _compute_loss(self, ca_pt: torch.Tensor, pred_distogram: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss combining physics and machine learning priors.
        
        Components:
        1. Bond distance constraint (C-alpha spacing)
        2. Angle geometry constraint (realistic backbone angles)
        3. Steric clash removal (repulsive constraint)
        4. Distogram prior (ML guidance)
        """
        metrics = {}
        L = ca_pt.shape[0]
        
        # 1. BOND DISTANCE CONSTRAINT
        # Target: C-alpha to C-alpha distance ~ 3.8 Angstroms
        dv_bond = ca_pt[1:] - ca_pt[:-1]
        bond_dist = torch.norm(dv_bond, dim=1)
        target_bond_dist = 3.8
        loss_bond = torch.mean((bond_dist - target_bond_dist) ** 2)
        
        # 2. ANGLE GEOMETRY CONSTRAINT
        # Encourage reasonable backbone angles (90-120 degrees)
        if L > 2:
            v1 = dv_bond[:-1]
            v2 = dv_bond[1:]
            v1_norm = torch.norm(v1, dim=1, keepdim=True)
            v2_norm = torch.norm(v2, dim=1, keepdim=True)
            
            cos_theta = torch.sum(v1 * v2, dim=1) / (v1_norm.squeeze() * v2_norm.squeeze() + 1e-6)
            # Target: cos(theta) ~ -0.25 (approximately 105 degrees)
            target_cos = -0.25
            loss_angle = torch.mean((cos_theta - target_cos) ** 2)
        else:
            loss_angle = torch.tensor(0.0, device=self.device)
        
        # 3. STERIC CLASH REMOVAL
        # Prevent atoms from overlapping (distance < 3.2 Angstroms)
        dist_matrix = torch.cdist(ca_pt, ca_pt)
        mask_clash = (dist_matrix < 3.2) & (torch.eye(L, device=self.device) == 0)
        
        if mask_clash.any():
            loss_clash = torch.mean((3.2 - dist_matrix[mask_clash]) ** 2)
        else:
            loss_clash = torch.tensor(0.0, device=self.device)
        
        # 4. MACHINE LEARNING DISTOGRAM PRIOR
        # Pull predicted structure toward ML-guided distances
        mask_valid = torch.eye(L, device=self.device) == 0
        loss_distogram = torch.mean((dist_matrix[mask_valid] - pred_distogram[mask_valid]) ** 2)
        
        # Total energy
        total_loss = (
            self.config.w_bond * loss_bond +
            self.config.w_angle * loss_angle +
            self.config.w_clash * loss_clash +
            self.config.w_distogram * loss_distogram
        )
        
        metrics = {
            'bond': loss_bond.item(),
            'angle': loss_angle.item(),
            'clash': loss_clash.item(),
            'distogram': loss_distogram.item(),
            'total': total_loss.item()
        }
        
        return total_loss, metrics

    def predict(self, seq: str) -> BackboneFrame:
        """
        Predict 3D protein structure from amino acid sequence.
        
        Args:
            seq: Amino acid sequence (single letter code)
            
        Returns:
            BackboneFrame containing predicted C-alpha coordinates
        """
        print(f"\n{'='*70}")
        print(f"🧬 CSOC De Novo Predictor - Sequence Length: {len(seq)}")
        print(f"{'='*70}\n")
        
        # 1. GENERATE MACHINE LEARNING PRIOR
        print("📡 Step 1: Computing ML Prior (Distogram Network)...")
        seq_tokens = self._seq_to_tokens(seq)
        
        with torch.no_grad():
            pred_distogram = self.distogram_net(seq_tokens)
        print(f"   ✓ Distogram shape: {pred_distogram.shape}")
        
        # 2. INITIALIZE STRUCTURE
        print("\n🔧 Step 2: Initializing Backbone Coordinates...")
        n_res = len(seq)
        ca_pt = self._initialize_backbone(n_res)
        print(f"   ✓ Initial radius of gyration: {torch.norm(ca_pt - ca_pt.mean(dim=0)).item():.2f} Å")
        
        # 3. STRUCTURAL RECYCLING LOOP
        print("\n🔄 Step 3: Structural Optimization (Recycling)...\n")
        
        for recycle in range(self.config.n_recycling):
            print(f"   {'─'*66}")
            print(f"   🔁 Recycling Round {recycle + 1}/{self.config.n_recycling}")
            print(f"   {'─'*66}")
            
            optimizer = HybridOptimizer(
                [ca_pt],
                lr=self.config.learning_rate,
                weight_decay=1e-4
            )
            
            scaler = GradScaler(enabled=self.device.type == 'cuda')
            
            for stage in range(self.config.n_stages):
                print(f"\n   Stage {stage + 1}/{self.config.n_stages}:")
                
                for i in range(self.config.n_iter_per_stage):
                    # Get ASM-driven temperature
                    T_current = self.config.base_langevin_temp
                    
                    if self.asm_driver is not None:
                        A = self.asm_driver.step_avalanche()
                        # Scale temperature by avalanche size (criticality)
                        T_current = self.config.base_langevin_temp * (1.0 + 0.1 * math.log1p(A))
                    else:
                        A = 0
                    
                    # Temperature annealing over stages
                    annealing_factor = 1.0 - (stage / self.config.n_stages)
                    T_current = T_current * annealing_factor
                    
                    # Update optimizer temperature
                    optimizer.param_groups[0]['langevin_temperature'] = T_current
                    
                    # Optimization step
                    optimizer.zero_grad()
                    
                    with autocast(enabled=self.device.type == 'cuda'):
                        loss, metrics = self._compute_loss(ca_pt, pred_distogram)
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_([ca_pt], self.config.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Print progress
                    if i % 100 == 0:
                        print(f"      Iter {i:3d} | Loss: {metrics['total']:7.3f} | "
                              f"Bond: {metrics['bond']:5.2f} | Clash: {metrics['clash']:6.2f} | "
                              f"T: {T_current:6.1f}K | A: {A:5d}")
            
            # Reparameterization: center coordinates before next recycle
            with torch.no_grad():
                ca_pt.data -= ca_pt.data.mean(dim=0)
        
        # 4. FINALIZATION
        print(f"\n   {'─'*66}")
        print("✅ Optimization Complete!\n")
        
        ca_final = ca_pt.detach().cpu().numpy()
        
        # Compute final structure statistics
        rg = np.sqrt(np.mean(np.sum((ca_final - ca_final.mean(axis=0))**2, axis=1)))
        print(f"📏 Final Structure Statistics:")
        print(f"   • Radius of gyration: {rg:.2f} Å")
        print(f"   • N-terminus position: {ca_final[0]}")
        print(f"   • C-terminus position: {ca_final[-1]}")
        print(f"   • Max distance from center: {np.max(np.linalg.norm(ca_final - ca_final.mean(axis=0), axis=1)):.2f} Å")
        
        print(f"\n{'='*70}\n")
        
        return BackboneFrame(seq=seq, ca=ca_final)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution routine for De Novo prediction."""
    # Initialize configuration
    config = CSOCDeNovoConfig(
        n_recycling=3,
        n_stages=3,
        n_iter_per_stage=100,  # Reduced for demo; increase to 300 for production
        base_langevin_temp=300.0
    )
    
    # Create predictor instance
    predictor = DeNovoPredictor(config)
    
    # Example target sequence
    # Using a repeated mini-domain for demonstration
    target_sequence = "MKTLLLTLVVVTIVCLDLGYAT" * 3
    
    # Run prediction
    predicted_structure = predictor.predict(target_sequence)
    
    # Output results
    print(f"🎯 Final Result:")
    print(f"   Sequence length: {len(predicted_structure.seq)}")
    print(f"   C-alpha coordinates shape: {predicted_structure.ca.shape}")
    print(f"   First 5 coordinates:\n{predicted_structure.ca[:5]}")


if __name__ == "__main__":
    main()
