"""
GRIT-LoRA: Curvature-aware Low-Rank Adaptation implementation.

Integrates Fisher Information Matrix with LoRA for natural gradient updates
and projection onto most informative curvature directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import math
from peft import LoraConfig

from .fisher_info import FisherInformationMatrix, FisherApproximationType


@dataclass
class GRITLoRAConfig:
    """Configuration for GRIT-LoRA layers."""
    
    # Standard LoRA parameters
    r: int = 16                          # Rank of adaptation
    lora_alpha: float = 32               # LoRA scaling factor
    lora_dropout: float = 0.1            # Dropout probability
    
    # GRIT-specific parameters
    fisher_approximation: str = "diagonal"  # "diagonal", "kfac", "block_diagonal"
    fisher_damping: float = 1e-4         # Damping for Fisher matrix inversion
    fisher_ema_decay: float = 0.95       # EMA decay for Fisher updates
    fisher_update_freq: int = 10         # Update Fisher every N steps
    
    # Projection parameters
    projection_budget_start: int = 32    # Initial projection budget k
    projection_budget_end: int = 96      # Final projection budget k  
    projection_schedule: str = "linear"  # "linear", "cosine", "adaptive"
    
    # Natural gradient parameters
    natural_gradient_damping: float = 1e-3
    enable_projection: bool = True
    enable_natural_gradient: bool = True
    
    # Target modules (same as standard LoRA)
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default targets for transformer models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class GRITLoRALayer(nn.Module):
    """
    GRIT-LoRA layer implementing curvature-aware low-rank adaptation.
    
    Combines standard LoRA with:
    1. Fisher Information Matrix estimation
    2. Natural gradient computation  
    3. Projection onto most informative directions
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        config: GRITLoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Get dimensions from base layer
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            self.in_features = base_layer.in_channels
            self.out_features = base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")
        
        # Initialize LoRA matrices with same dtype and device as base layer
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype
        
        self.lora_A = nn.Parameter(
            torch.randn(config.r, self.in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, config.r, device=device, dtype=dtype)
        )
        
        # Scaling factor
        self.scaling = config.lora_alpha / config.r
        
        # Dropout
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        
        # Fisher Information Matrix
        self.fisher_matrix = FisherInformationMatrix(
            approximation_type=FisherApproximationType(config.fisher_approximation),
            damping=config.fisher_damping,
            ema_decay=config.fisher_ema_decay,
            update_freq=config.fisher_update_freq
        )
        
        # Initialize Fisher for this layer
        self._initialize_fisher()
        
        # Activation buffer for Fisher computation
        self.activation_buffer: List[torch.Tensor] = []
        self.step_count = 0
        
        # Projection components
        self.projection_matrix: Optional[torch.Tensor] = None
        self.eigenvalues: Optional[torch.Tensor] = None
        
        # Register hooks for activation and gradient capture
        self._register_hooks()
    
    def _initialize_fisher(self):
        """Initialize Fisher matrix for this layer."""
        # Create a temporary module containing just this layer's parameters
        temp_module = nn.Module()
        temp_module.register_parameter('lora_A', self.lora_A)
        temp_module.register_parameter('lora_B', self.lora_B)
        
        self.fisher_matrix.initialize_for_model(temp_module)
    
    def _register_hooks(self):
        """Register forward and backward hooks for Fisher computation."""
        
        def forward_hook(module, input, output):
            """Capture activations for Fisher computation."""
            if isinstance(input, tuple):
                activation = input[0].detach()
            else:
                activation = input.detach()
            
            # Store activation (limit buffer size)
            if len(self.activation_buffer) < 100:  # Limit memory usage
                self.activation_buffer.append(activation)
        
        def backward_hook(module, grad_input, grad_output):
            """Hook for potential future gradient capture (currently unused)."""
            # Currently not capturing gradients as Fisher computation
            # uses parameter gradients directly from autograd
            pass
        
        # Register hooks
        self.register_forward_hook(forward_hook)
        self.register_full_backward_hook(backward_hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GRIT-LoRA adaptation.
        
        Output = base_layer(x) + scaling * lora_B @ lora_A @ x
        """
        # Base layer output
        base_output = self.base_layer(x)
        
        # LoRA adaptation path
        lora_input = self.lora_dropout(x)
        
        if isinstance(self.base_layer, nn.Linear):
            lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        elif isinstance(self.base_layer, nn.Conv2d):
            # For conv layers, need to handle properly
            # This is simplified - full implementation would handle conv specifics
            lora_output = F.linear(lora_input.view(-1, self.in_features), self.lora_A)
            lora_output = F.linear(lora_output, self.lora_B)
            lora_output = lora_output.view(base_output.shape)
        else:
            lora_output = torch.zeros_like(base_output)
        
        return base_output + self.scaling * lora_output
    
    def update_fisher_and_projection(self):
        """
        Update Fisher matrix and projection components.
        Called periodically during training.
        """
        self.step_count += 1
        
        # Update Fisher matrix if we have gradients
        if hasattr(self.lora_A, 'grad') and hasattr(self.lora_B, 'grad'):
            gradients = {
                'lora_A': self.lora_A.grad,
                'lora_B': self.lora_B.grad
            }
            
            # Create activations dict if available
            activations = {}
            if self.activation_buffer:
                activations[self.adapter_name] = torch.stack(self.activation_buffer[-10:])  # Use recent activations
            
            # Update Fisher approximation
            temp_module = nn.Module()
            temp_module.register_parameter('lora_A', self.lora_A)
            temp_module.register_parameter('lora_B', self.lora_B)
            
            self.fisher_matrix.update_fisher(temp_module, gradients, activations)
        
        # Update projection matrix if enabled
        if self.config.enable_projection:
            self._update_projection_matrix()
        
        # Clear activation buffer to prevent memory leak
        self.activation_buffer = self.activation_buffer[-10:]  # Keep only recent ones
    
    def _update_projection_matrix(self):
        """Update projection matrix based on Fisher eigendecomposition."""
        if self.fisher_matrix.approximation_type == FisherApproximationType.DIAGONAL:
            self._update_diagonal_projection()
        else:
            # For other approximations, could implement eigen-decomposition of blocks
            pass
    
    def _update_diagonal_projection(self):
        """Update projection matrix for diagonal Fisher approximation."""
        if self.fisher_matrix.fisher_diagonal is None:
            return
        
        # Get current projection budget
        k = self._get_current_projection_budget()
        
        # For diagonal Fisher, select top-k parameters by Fisher value
        fisher_values = self.fisher_matrix.fisher_diagonal
        
        if len(fisher_values) >= k:
            # Get top-k indices
            top_k_values, top_k_indices = torch.topk(fisher_values, k)
            
            # Create projection matrix (sparse)
            projection_matrix = torch.zeros_like(fisher_values)
            projection_matrix[top_k_indices] = 1.0
            
            self.projection_matrix = projection_matrix
            self.eigenvalues = top_k_values
    
    def _get_current_projection_budget(self) -> int:
        """Get current projection budget based on schedule."""
        start_k = self.config.projection_budget_start
        end_k = self.config.projection_budget_end
        
        if self.config.projection_schedule == "linear":
            # Linear interpolation (need to define total training steps)
            # For now, use step count as proxy
            progress = min(1.0, self.step_count / 1000)  # Assume 1000 steps for schedule
            k = int(start_k + (end_k - start_k) * progress)
        elif self.config.projection_schedule == "cosine":
            progress = min(1.0, self.step_count / 1000)
            k = int(start_k + 0.5 * (end_k - start_k) * (1 + math.cos(math.pi * progress)))
        else:  # adaptive or fallback
            k = start_k
        
        return max(start_k, min(k, end_k))
    
    def get_natural_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradients using Fisher matrix.
        
        Returns:
            Dictionary of natural gradients for LoRA parameters
        """
        if not self.config.enable_natural_gradient:
            return {
                'lora_A': self.lora_A.grad if self.lora_A.grad is not None else torch.zeros_like(self.lora_A),
                'lora_B': self.lora_B.grad if self.lora_B.grad is not None else torch.zeros_like(self.lora_B)
            }
        
        # Get standard gradients
        gradients = {}
        if self.lora_A.grad is not None:
            gradients['lora_A'] = self.lora_A.grad
        if self.lora_B.grad is not None:
            gradients['lora_B'] = self.lora_B.grad
        
        if not gradients:
            return {}
        
        # Compute natural gradients using Fisher matrix
        natural_gradients = self.fisher_matrix.get_natural_gradient(gradients, self.adapter_name)
        
        # Apply projection if enabled
        if self.config.enable_projection and self.projection_matrix is not None:
            natural_gradients = self._apply_projection(natural_gradients)
        
        return natural_gradients
    
    def _apply_projection(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply projection onto most informative directions."""
        if self.projection_matrix is None:
            return gradients
        
        projected_gradients = {}
        
        # Flatten gradients for projection
        grad_vector = torch.cat([grad.flatten() for grad in gradients.values()])
        
        # Apply projection (element-wise multiplication for diagonal case)
        if len(self.projection_matrix) == len(grad_vector):
            projected_vector = grad_vector * self.projection_matrix
            
            # Reconstruct gradient dictionary
            start_idx = 0
            for name, grad in gradients.items():
                end_idx = start_idx + grad.numel()
                projected_gradients[name] = projected_vector[start_idx:end_idx].view(grad.shape)
                start_idx = end_idx
        else:
            projected_gradients = gradients  # Fallback
        
        return projected_gradients
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.config.r}, alpha={self.config.lora_alpha}, '
                f'fisher_approx={self.config.fisher_approximation}')


class GRITLoRALinear(GRITLoRALayer):
    """GRIT-LoRA adaptation specifically for Linear layers."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: GRITLoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__(base_layer, config, adapter_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass for Linear layers."""
        base_output = self.base_layer(x)
        
        # LoRA path: x -> A^T -> dropout -> B^T  
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        
        return base_output + self.scaling * lora_output


class GRITLoRAConv2d(GRITLoRALayer):
    """GRIT-LoRA adaptation specifically for Conv2d layers."""
    
    def __init__(
        self,
        base_layer: nn.Conv2d,
        config: GRITLoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__(base_layer, config, adapter_name)
        
        # For conv layers, adapt the LoRA matrices to work with convolutions
        self.lora_A = nn.Parameter(torch.randn(config.r, base_layer.in_channels) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_channels, config.r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass for Conv2d layers."""
        base_output = self.base_layer(x)
        
        # Simplified LoRA for conv - can be extended for full conv LoRA
        batch_size, channels, height, width = x.shape
        
        # Flatten spatial dimensions for linear operations
        x_flat = x.view(batch_size, channels, -1).mean(dim=2)  # Global average pooling
        
        # Apply LoRA transformation
        lora_input = self.lora_dropout(x_flat)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        
        # Broadcast back to conv output shape
        lora_output = lora_output.unsqueeze(-1).unsqueeze(-1)
        lora_output = lora_output.expand_as(base_output)
        
        return base_output + self.scaling * lora_output


def create_grit_lora_layer(
    base_layer: nn.Module,
    config: GRITLoRAConfig,
    adapter_name: str = "default"
) -> GRITLoRALayer:
    """
    Factory function to create appropriate GRIT-LoRA layer.
    
    Args:
        base_layer: The base layer to adapt
        config: GRIT-LoRA configuration
        adapter_name: Name of the adapter
        
    Returns:
        Appropriate GRIT-LoRA layer instance
    """
    if isinstance(base_layer, nn.Linear):
        return GRITLoRALinear(base_layer, config, adapter_name)
    elif isinstance(base_layer, nn.Conv2d):
        return GRITLoRAConv2d(base_layer, config, adapter_name)
    else:
        # Fallback to generic implementation
        return GRITLoRALayer(base_layer, config, adapter_name)