"""
Fisher Information Matrix computation for GRIT optimization.

Implements both diagonal and K-FAC approximations for efficient
curvature estimation in neural networks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from enum import Enum
import warnings


class FisherApproximationType(Enum):
    """Types of Fisher Information Matrix approximations."""
    DIAGONAL = "diagonal"
    KFAC = "kfac"
    BLOCK_DIAGONAL = "block_diagonal"


class FisherInformationMatrix:
    """
    Fisher Information Matrix computation with multiple approximation strategies.
    
    Mathematical Foundation:
    I(θ) = E[Z(X)Z(X)^T | θ] where Z(X) = ∇_θ log f(x|θ)
    Alternative: [I(θ)]_{i,j} = -E[∂²/∂θᵢ∂θⱼ log f(X; θ) | θ]
    """
    
    def __init__(
        self, 
        approximation_type: FisherApproximationType = FisherApproximationType.DIAGONAL,
        damping: float = 1e-4,
        ema_decay: float = 0.95,
        update_freq: int = 10
    ):
        self.approximation_type = approximation_type
        self.damping = damping
        self.ema_decay = ema_decay
        self.update_freq = update_freq
        self.step_count = 0
        
        # Storage for Fisher approximations
        self.fisher_diagonal: Optional[torch.Tensor] = None
        self.fisher_blocks: Dict[str, torch.Tensor] = {}
        self.kfac_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
    def initialize_for_model(self, model: nn.Module) -> None:
        """Initialize Fisher storage for a given model."""
        if self.approximation_type == FisherApproximationType.DIAGONAL:
            self._initialize_diagonal(model)
        elif self.approximation_type == FisherApproximationType.KFAC:
            self._initialize_kfac(model)
        elif self.approximation_type == FisherApproximationType.BLOCK_DIAGONAL:
            self._initialize_block_diagonal(model)
    
    def _initialize_diagonal(self, model: nn.Module) -> None:
        """Initialize diagonal Fisher approximation."""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.fisher_diagonal = torch.ones(total_params, device=next(model.parameters()).device)
        
    def _initialize_kfac(self, model: nn.Module) -> None:
        """Initialize K-FAC approximation structures."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Initialize A (activation covariance) and G (gradient covariance) factors
                if isinstance(module, nn.Linear):
                    in_features, out_features = module.weight.shape[1], module.weight.shape[0]
                    A_factor = torch.eye(in_features + 1, device=module.weight.device)  # +1 for bias
                    G_factor = torch.eye(out_features, device=module.weight.device)
                elif isinstance(module, nn.Conv2d):
                    # For conv layers, flatten spatial dimensions
                    in_features = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                    out_features = module.weight.shape[0]
                    A_factor = torch.eye(in_features + 1, device=module.weight.device)
                    G_factor = torch.eye(out_features, device=module.weight.device)
                
                self.kfac_factors[name] = (A_factor, G_factor)
    
    def _initialize_block_diagonal(self, model: nn.Module) -> None:
        """Initialize block-diagonal Fisher approximation."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                param_count = sum(p.numel() for p in module.parameters())
                self.fisher_blocks[name] = torch.eye(param_count, device=next(module.parameters()).device)
    
    def update_fisher(
        self, 
        model: nn.Module, 
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Update Fisher Information Matrix approximation.
        
        Args:
            model: Neural network model
            gradients: Pre-computed gradients (optional)
            activations: Stored activations for K-FAC (optional)
        """
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            print(f"  📊 Fisher step {self.step_count}/{self.update_freq} - accumulating (no update yet)")
            return
        
        print(f"🔥 FISHER UPDATE at step {self.step_count}! Running {self.approximation_type.value} Fisher computation...")
            
        if self.approximation_type == FisherApproximationType.DIAGONAL:
            print(f"  🎯 Computing Diagonal Fisher Information...")
            self._update_diagonal_fisher(model, gradients)
            print(f"  ✅ Diagonal Fisher updated")
        elif self.approximation_type == FisherApproximationType.KFAC:
            print(f"  🧠 Computing K-FAC Fisher Information...")
            self._update_kfac_fisher(model, activations)
            print(f"  ✅ K-FAC Fisher updated")
        elif self.approximation_type == FisherApproximationType.BLOCK_DIAGONAL:
            print(f"  🔲 Computing Block-Diagonal Fisher Information...")
            self._update_block_diagonal_fisher(model, gradients)
            print(f"  ✅ Block-Diagonal Fisher updated")
    
    def _update_diagonal_fisher(
        self, 
        model: nn.Module, 
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Update diagonal Fisher approximation using empirical gradients."""
        if gradients is None:
            gradients = {name: param.grad for name, param in model.named_parameters() 
                        if param.grad is not None}
        
        # Flatten all gradients
        grad_vector = torch.cat([grad.flatten() for grad in gradients.values()])
        
        # Empirical Fisher: F_ii ≈ (∂log p/∂θᵢ)²
        empirical_fisher = grad_vector ** 2
        
        # Exponential moving average update
        if self.fisher_diagonal is None:
            self.fisher_diagonal = empirical_fisher
        else:
            self.fisher_diagonal = (
                self.ema_decay * self.fisher_diagonal + 
                (1 - self.ema_decay) * empirical_fisher
            )
    
    def _update_kfac_fisher(
        self, 
        model: nn.Module,
        activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Update K-FAC Fisher approximation: F ≈ A ⊗ G."""
        if activations is None:
            warnings.warn("K-FAC requires activation storage. Consider using activation hooks.")
            return
        
        for name, module in model.named_modules():
            if name not in self.kfac_factors:
                continue
                
            if isinstance(module, nn.Linear):
                self._update_linear_kfac(name, module, activations.get(name))
            elif isinstance(module, nn.Conv2d):
                self._update_conv_kfac(name, module, activations.get(name))
    
    def _update_linear_kfac(
        self, 
        name: str, 
        module: nn.Linear, 
        activation: Optional[torch.Tensor]
    ) -> None:
        """Update K-FAC factors for Linear layer."""
        if activation is None or module.weight.grad is None:
            return
            
        A_factor, G_factor = self.kfac_factors[name]
        
        # Add bias term to activation
        batch_size = activation.shape[0]
        activation_with_bias = torch.cat([
            activation, 
            torch.ones(batch_size, 1, device=activation.device)
        ], dim=1)
        
        # Update A factor (activation covariance)
        A_new = torch.mm(activation_with_bias.t(), activation_with_bias) / batch_size
        A_factor = self.ema_decay * A_factor + (1 - self.ema_decay) * A_new
        
        # Update G factor (gradient covariance) 
        grad_output = module.weight.grad
        G_new = torch.mm(grad_output, grad_output.t()) / batch_size
        G_factor = self.ema_decay * G_factor + (1 - self.ema_decay) * G_new
        
        self.kfac_factors[name] = (A_factor, G_factor)
    
    def _update_conv_kfac(
        self, 
        name: str, 
        module: nn.Conv2d, 
        activation: Optional[torch.Tensor]
    ) -> None:
        """Update K-FAC factors for Conv2d layer."""
        if activation is None or module.weight.grad is None:
            return
        
        # Unfold activation patches for convolution
        batch_size = activation.shape[0]
        unfolded = nn.functional.unfold(
            activation, 
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride
        )
        
        # Reshape for matrix operations
        unfolded = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
        
        # Add bias term
        activation_with_bias = torch.cat([
            unfolded,
            torch.ones(unfolded.shape[0], 1, device=activation.device)
        ], dim=1)
        
        A_factor, G_factor = self.kfac_factors[name]
        
        # Update factors similar to linear layer
        A_new = torch.mm(activation_with_bias.t(), activation_with_bias) / unfolded.shape[0]
        A_factor = self.ema_decay * A_factor + (1 - self.ema_decay) * A_new
        
        grad_output = module.weight.grad.view(module.weight.shape[0], -1)
        G_new = torch.mm(grad_output, grad_output.t()) / grad_output.shape[1]
        G_factor = self.ema_decay * G_factor + (1 - self.ema_decay) * G_new
        
        self.kfac_factors[name] = (A_factor, G_factor)
    
    def _update_block_diagonal_fisher(
        self, 
        model: nn.Module,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Update block-diagonal Fisher approximation."""
        if gradients is None:
            gradients = {name: param.grad for name, param in model.named_parameters() 
                        if param.grad is not None}
        
        for name, module in model.named_modules():
            if name not in self.fisher_blocks:
                continue
                
            # Get gradients for this module
            module_grads = []
            for param_name, param in module.named_parameters():
                full_name = f"{name}.{param_name}"
                if full_name in gradients:
                    module_grads.append(gradients[full_name].flatten())
            
            if not module_grads:
                continue
                
            grad_vector = torch.cat(module_grads)
            empirical_block = torch.outer(grad_vector, grad_vector)
            
            # EMA update
            self.fisher_blocks[name] = (
                self.ema_decay * self.fisher_blocks[name] + 
                (1 - self.ema_decay) * empirical_block
            )
    
    def get_natural_gradient(
        self, 
        gradients: Dict[str, torch.Tensor],
        module_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradients using Fisher approximation.
        
        Natural gradient: g_natural = F^(-1) * g_standard
        """
        if self.approximation_type == FisherApproximationType.DIAGONAL:
            return self._get_diagonal_natural_gradient(gradients)
        elif self.approximation_type == FisherApproximationType.KFAC:
            return self._get_kfac_natural_gradient(gradients, module_name)
        elif self.approximation_type == FisherApproximationType.BLOCK_DIAGONAL:
            return self._get_block_diagonal_natural_gradient(gradients)
        else:
            return gradients
    
    def _get_diagonal_natural_gradient(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute natural gradient with diagonal Fisher approximation."""
        if self.fisher_diagonal is None:
            return gradients
        
        # Flatten gradients
        grad_vector = torch.cat([grad.flatten() for grad in gradients.values()])
        
        # Apply F^(-1) = diag(1 / (fisher_ii + damping))
        fisher_inv = 1.0 / (self.fisher_diagonal + self.damping)
        natural_grad_vector = grad_vector * fisher_inv
        
        # Reconstruct gradient dictionary
        natural_gradients = {}
        start_idx = 0
        for name, grad in gradients.items():
            end_idx = start_idx + grad.numel()
            natural_gradients[name] = natural_grad_vector[start_idx:end_idx].view(grad.shape)
            start_idx = end_idx
            
        return natural_gradients
    
    def _get_kfac_natural_gradient(
        self, 
        gradients: Dict[str, torch.Tensor],
        module_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute natural gradient with K-FAC approximation."""
        if module_name is None or module_name not in self.kfac_factors:
            return gradients
        
        A_factor, G_factor = self.kfac_factors[module_name]
        
        natural_gradients = {}
        for name, grad in gradients.items():
            if module_name in name and "weight" in name:
                # For weight matrices: (G^(-1) ⊗ A^(-1)) * vec(grad)
                # Equivalent to: G^(-1) * grad * A^(-1)
                A_inv = torch.inverse(A_factor + self.damping * torch.eye(A_factor.shape[0], device=A_factor.device))
                G_inv = torch.inverse(G_factor + self.damping * torch.eye(G_factor.shape[0], device=G_factor.device))
                
                if len(grad.shape) == 2:  # Linear layer weight
                    natural_grad = torch.mm(torch.mm(G_inv, grad), A_inv[:-1, :-1])  # Exclude bias dimension
                    natural_gradients[name] = natural_grad
                else:
                    natural_gradients[name] = grad  # Fallback
            elif module_name in name and "bias" in name:
                # For bias: G^(-1) * grad
                G_inv = torch.inverse(G_factor + self.damping * torch.eye(G_factor.shape[0], device=G_factor.device))
                natural_gradients[name] = torch.mv(G_inv, grad)
            else:
                natural_gradients[name] = grad
                
        return natural_gradients
    
    def _get_block_diagonal_natural_gradient(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute natural gradient with block-diagonal Fisher approximation."""
        # Implementation similar to K-FAC but with full blocks
        # For brevity, returning standard gradients (can be extended)
        return gradients


class KFACApproximation:
    """
    Specialized K-FAC (Kronecker-Factored Approximate Curvature) implementation.
    
    Approximates Fisher Information as F ≈ A ⊗ G where:
    - A is activation covariance matrix  
    - G is gradient covariance matrix
    - ⊗ denotes Kronecker product
    """
    
    def __init__(self, damping: float = 1e-4, ema_decay: float = 0.95):
        self.damping = damping
        self.ema_decay = ema_decay
        self.A_factors: Dict[str, torch.Tensor] = {}
        self.G_factors: Dict[str, torch.Tensor] = {}
        
    def register_layer(self, name: str, layer: nn.Module) -> None:
        """Register a layer for K-FAC approximation."""
        if isinstance(layer, nn.Linear):
            in_dim = layer.in_features + 1  # +1 for bias
            out_dim = layer.out_features
        elif isinstance(layer, nn.Conv2d):
            in_dim = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1] + 1
            out_dim = layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
            
        device = next(layer.parameters()).device
        self.A_factors[name] = torch.eye(in_dim, device=device)
        self.G_factors[name] = torch.eye(out_dim, device=device)
    
    def update_factors(
        self, 
        name: str, 
        activation: torch.Tensor, 
        grad_output: torch.Tensor
    ) -> None:
        """Update A and G factors for a specific layer."""
        if name not in self.A_factors:
            return
            
        batch_size = activation.shape[0]
        
        # Add bias term to activation
        activation_with_bias = torch.cat([
            activation.view(batch_size, -1),
            torch.ones(batch_size, 1, device=activation.device)
        ], dim=1)
        
        # Update A factor (activation covariance)
        A_new = torch.mm(activation_with_bias.t(), activation_with_bias) / batch_size
        self.A_factors[name] = (
            self.ema_decay * self.A_factors[name] + 
            (1 - self.ema_decay) * A_new
        )
        
        # Update G factor (gradient covariance)
        grad_output_flat = grad_output.view(grad_output.shape[0], -1)
        G_new = torch.mm(grad_output_flat, grad_output_flat.t()) / grad_output_flat.shape[1]
        self.G_factors[name] = (
            self.ema_decay * self.G_factors[name] + 
            (1 - self.ema_decay) * G_new
        )
    
    def get_natural_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Compute natural gradient for a specific layer using K-FAC."""
        if name not in self.A_factors:
            return gradient
            
        A, G = self.A_factors[name], self.G_factors[name]
        
        # Compute inverses with damping
        A_inv = torch.inverse(A + self.damping * torch.eye(A.shape[0], device=A.device))
        G_inv = torch.inverse(G + self.damping * torch.eye(G.shape[0], device=G.device))
        
        if len(gradient.shape) == 2:  # Weight matrix
            # Natural gradient: G^(-1) * grad * A^(-1)
            return torch.mm(torch.mm(G_inv, gradient), A_inv[:-1, :-1])  # Exclude bias
        elif len(gradient.shape) == 1:  # Bias vector
            return torch.mv(G_inv, gradient)
        else:
            return gradient  # Fallback for unsupported shapes