"""
Natural Gradient Optimizer for GRIT implementation.

Implements curvature-aware optimization using Fisher Information Matrix
to scale gradients according to parameter sensitivity.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Union, Callable
import warnings
from collections import defaultdict

from ..core.grit_lora import GRITLoRALayer


class GRITOptimizer(optim.Optimizer):
    """
    Natural gradient optimizer for GRIT-LoRA training.
    
    Applies Fisher Information Matrix preconditioning to gradients,
    implementing the natural gradient: g_natural = F^(-1) * g_standard
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        natural_gradient_damping: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        enable_natural_gradient: bool = True,
        fisher_update_freq: int = 10,
        eps: float = 1e-8
    ):
        """
        Initialize GRIT optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            natural_gradient_damping: Damping for Fisher matrix inversion
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            enable_natural_gradient: Whether to use natural gradients
            fisher_update_freq: Update Fisher matrix every N steps
            eps: Small epsilon for numerical stability
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if natural_gradient_damping < 0.0:
            raise ValueError(f"Invalid damping: {natural_gradient_damping}")
            
        defaults = dict(
            lr=lr,
            natural_gradient_damping=natural_gradient_damping,
            momentum=momentum,
            weight_decay=weight_decay,
            enable_natural_gradient=enable_natural_gradient,
            fisher_update_freq=fisher_update_freq,
            eps=eps
        )
        super().__init__(params, defaults)
        
        # Track GRIT layers for Fisher updates
        self.grit_layers: List[GRITLoRALayer] = []
        self.step_count = 0
        
        # State for momentum
        self.state = defaultdict(dict)
    
    def register_grit_layer(self, layer: GRITLoRALayer):
        """Register a GRIT-LoRA layer for Fisher updates."""
        self.grit_layers.append(layer)
    
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step with natural gradients.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        # Update Fisher matrices in GRIT layers
        if self.step_count % self.defaults['fisher_update_freq'] == 0:
            self._update_fisher_matrices()
        
        # Apply parameter updates
        for group in self.param_groups:
            self._update_param_group(group)
        
        return loss
    
    def _update_fisher_matrices(self):
        """Update Fisher Information matrices in all registered GRIT layers."""
        for layer in self.grit_layers:
            try:
                layer.update_fisher_and_projection()
            except Exception as e:
                warnings.warn(f"Failed to update Fisher matrix in layer {layer}: {e}")
    
    def _update_param_group(self, group: Dict):
        """Update parameters in a parameter group."""
        for p in group['params']:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Apply weight decay
            if group['weight_decay'] != 0:
                grad = grad.add(p.data, alpha=group['weight_decay'])
            
            # Get parameter state
            state = self.state[p]
            
            # Initialize momentum buffer
            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['step'] = 0
            
            state['step'] += 1
            
            # Apply natural gradient if this parameter belongs to a GRIT layer
            natural_grad = self._get_natural_gradient_for_param(p, grad)
            
            # Apply momentum
            momentum_buffer = state['momentum_buffer']
            momentum_buffer.mul_(group['momentum']).add_(natural_grad)
            
            # Update parameter
            p.data.add_(momentum_buffer, alpha=-group['lr'])
    
    def _get_natural_gradient_for_param(self, param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
        """
        Get natural gradient for a specific parameter.
        
        Args:
            param: Parameter tensor
            grad: Standard gradient
            
        Returns:
            Natural gradient (or standard gradient if not from GRIT layer)
        """
        # Find which GRIT layer this parameter belongs to
        for layer in self.grit_layers:
            if param is layer.lora_A or param is layer.lora_B:
                if self.defaults['enable_natural_gradient']:
                    natural_grads = layer.get_natural_gradients()
                    
                    # Return appropriate natural gradient
                    if param is layer.lora_A and 'lora_A' in natural_grads:
                        return natural_grads['lora_A']
                    elif param is layer.lora_B and 'lora_B' in natural_grads:
                        return natural_grads['lora_B']
        
        # Default to standard gradient
        return grad
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients of all parameters."""
        super().zero_grad(set_to_none)
        
        # Also clear activation/gradient buffers in GRIT layers
        for layer in self.grit_layers:
            layer.activation_buffer.clear()
            layer.gradient_buffer.clear()
    
    def state_dict(self):
        """Return optimizer state dictionary."""
        state_dict = super().state_dict()
        state_dict['step_count'] = self.step_count
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dictionary."""
        self.step_count = state_dict.pop('step_count', 0)
        super().load_state_dict(state_dict)


class AdaptiveGRITOptimizer(GRITOptimizer):
    """
    Adaptive GRIT optimizer that adjusts learning rates based on curvature.
    
    Implements curvature-aware learning rate tuning:
    η^cdat_t = σ * n_t / d_t
    where n_t = max{-∇f(w_t)^T u_t, 0} and d_t = |u_t^T ∇²f(w_t) u_t| + ε
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        natural_gradient_damping: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        enable_natural_gradient: bool = True,
        fisher_update_freq: int = 10,
        eps: float = 1e-8,
        adaptive_lr_sigma: float = 1.0,
        curvature_window: int = 10
    ):
        """
        Initialize adaptive GRIT optimizer.
        
        Args:
            adaptive_lr_sigma: Scaling factor for adaptive learning rate
            curvature_window: Window size for curvature estimation
        """
        super().__init__(
            params, lr, natural_gradient_damping, momentum, weight_decay,
            enable_natural_gradient, fisher_update_freq, eps
        )
        
        self.adaptive_lr_sigma = adaptive_lr_sigma
        self.curvature_window = curvature_window
        
        # Store gradient history for curvature estimation
        self.gradient_history: List[Dict[int, torch.Tensor]] = []
    
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with adaptive learning rates."""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Store current gradients for curvature estimation
        current_grads = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    key = f"{group_idx}_{param_idx}"
                    current_grads[key] = p.grad.data.clone()
        
        self.gradient_history.append(current_grads)
        
        # Limit history size
        if len(self.gradient_history) > self.curvature_window:
            self.gradient_history.pop(0)
        
        # Update learning rates based on curvature
        self._update_adaptive_learning_rates()
        
        # Perform standard GRIT step
        return super().step(closure)
    
    def _update_adaptive_learning_rates(self):
        """Update learning rates based on estimated curvature."""
        if len(self.gradient_history) < 2:
            return  # Need at least 2 gradient samples
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                key = f"{group_idx}_{param_idx}"
                
                # Estimate curvature using finite differences
                curvature_est = self._estimate_curvature(key)
                
                if curvature_est > self.defaults['eps']:
                    # Compute adaptive learning rate
                    grad_norm = p.grad.data.norm().item()
                    adaptive_lr = self.adaptive_lr_sigma * grad_norm / (curvature_est + self.defaults['eps'])
                    
                    # Update learning rate (with bounds)
                    original_lr = group['lr']
                    group['lr'] = min(max(adaptive_lr, original_lr * 0.1), original_lr * 10)
    
    def _estimate_curvature(self, param_key: str) -> float:
        """Estimate curvature using gradient differences."""
        if len(self.gradient_history) < 2:
            return 1.0
        
        # Get recent gradients
        recent_grads = [hist.get(param_key) for hist in self.gradient_history[-2:]]
        recent_grads = [g for g in recent_grads if g is not None]
        
        if len(recent_grads) < 2:
            return 1.0
        
        # Estimate second derivative using finite differences
        grad_diff = recent_grads[-1] - recent_grads[-2]
        curvature_est = grad_diff.norm().item()
        
        return curvature_est


class GRITAdamOptimizer(optim.Adam):
    """
    Adam optimizer with GRIT natural gradient preconditioning.
    
    Combines Adam's adaptive learning rates with Fisher Information Matrix
    preconditioning for improved convergence in curved loss landscapes.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        natural_gradient_damping: float = 1e-3,
        enable_natural_gradient: bool = True,
        fisher_update_freq: int = 10
    ):
        """Initialize GRIT-Adam optimizer."""
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        
        self.natural_gradient_damping = natural_gradient_damping
        self.enable_natural_gradient = enable_natural_gradient
        self.fisher_update_freq = fisher_update_freq
        
        self.grit_layers: List[GRITLoRALayer] = []
        self.step_count = 0
    
    def register_grit_layer(self, layer: GRITLoRALayer):
        """Register a GRIT-LoRA layer."""
        self.grit_layers.append(layer)
    
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with natural gradient preconditioning."""
        self.step_count += 1
        
        # Update Fisher matrices
        if self.step_count % self.fisher_update_freq == 0:
            for layer in self.grit_layers:
                layer.update_fisher_and_projection()
        
        # Apply natural gradient preconditioning before Adam step
        if self.enable_natural_gradient:
            self._apply_natural_gradient_preconditioning()
        
        return super().step(closure)
    
    def _apply_natural_gradient_preconditioning(self):
        """Apply natural gradient preconditioning to gradients."""
        for layer in self.grit_layers:
            natural_grads = layer.get_natural_gradients()
            
            # Replace gradients with natural gradients
            if 'lora_A' in natural_grads and layer.lora_A.grad is not None:
                layer.lora_A.grad.data = natural_grads['lora_A']
            if 'lora_B' in natural_grads and layer.lora_B.grad is not None:
                layer.lora_B.grad.data = natural_grads['lora_B']


def create_grit_optimizer(
    model_parameters,
    optimizer_type: str = "sgd",
    grit_layers: Optional[List[GRITLoRALayer]] = None,
    **kwargs
) -> Union[GRITOptimizer, AdaptiveGRITOptimizer, GRITAdamOptimizer]:
    """
    Factory function to create GRIT optimizers.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: Type of optimizer ("sgd", "adaptive", "adam")
        grit_layers: List of GRIT layers to register
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured GRIT optimizer
    """
    if grit_layers is None:
        grit_layers = []
    
    if optimizer_type == "sgd":
        optimizer = GRITOptimizer(model_parameters, **kwargs)
    elif optimizer_type == "adaptive":
        optimizer = AdaptiveGRITOptimizer(model_parameters, **kwargs)
    elif optimizer_type == "adam":
        optimizer = GRITAdamOptimizer(model_parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Register GRIT layers
    for layer in grit_layers:
        optimizer.register_grit_layer(layer)
    
    return optimizer