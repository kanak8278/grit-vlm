"""
Projection mechanisms and budget scheduling for GRIT optimization.

Implements projection onto most informative curvature directions and
dynamic scheduling of projection budget k (32 → 96).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Union
from enum import Enum
import math
import numpy as np


class ProjectionScheduleType(Enum):
    """Types of projection budget scheduling."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    STEP = "step"


class ProjectionScheduler:
    """
    Schedules projection budget k according to various strategies.
    
    Implements dynamic scheduling from k_start (32) to k_end (96)
    with empirical rule: k ≈ 1.2 × rank(A_vision)
    """
    
    def __init__(
        self,
        k_start: int = 32,
        k_end: int = 96,
        total_steps: int = 1000,
        schedule_type: ProjectionScheduleType = ProjectionScheduleType.LINEAR,
        warmup_steps: int = 100,
        vision_rank_multiplier: float = 1.2
    ):
        """
        Initialize projection scheduler.
        
        Args:
            k_start: Initial projection budget
            k_end: Final projection budget  
            total_steps: Total training steps
            schedule_type: Type of scheduling
            warmup_steps: Steps before scheduling begins
            vision_rank_multiplier: Multiplier for vision rank empirical rule
        """
        self.k_start = k_start
        self.k_end = k_end
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.vision_rank_multiplier = vision_rank_multiplier
        
        self.current_step = 0
        self.current_k = k_start
        
        # For adaptive scheduling
        self.eigenvalue_history: List[torch.Tensor] = []
        self.gap_threshold = 0.1
    
    def step(self, eigenvalues: Optional[torch.Tensor] = None, vision_rank: Optional[int] = None) -> int:
        """
        Update projection budget for current step.
        
        Args:
            eigenvalues: Current eigenvalues for adaptive scheduling
            vision_rank: Vision component rank for empirical rule
            
        Returns:
            Current projection budget k
        """
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            self.current_k = self.k_start
            return self.current_k
        
        # Calculate progress (0 to 1)
        progress = min(1.0, (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
        if self.schedule_type == ProjectionScheduleType.LINEAR:
            self.current_k = self._linear_schedule(progress)
        elif self.schedule_type == ProjectionScheduleType.COSINE:
            self.current_k = self._cosine_schedule(progress)
        elif self.schedule_type == ProjectionScheduleType.EXPONENTIAL:
            self.current_k = self._exponential_schedule(progress)
        elif self.schedule_type == ProjectionScheduleType.ADAPTIVE:
            self.current_k = self._adaptive_schedule(eigenvalues)
        elif self.schedule_type == ProjectionScheduleType.STEP:
            self.current_k = self._step_schedule(progress)
        
        # Apply empirical rule if vision rank is provided
        if vision_rank is not None:
            empirical_k = int(self.vision_rank_multiplier * vision_rank)
            self.current_k = min(self.current_k, empirical_k)
        
        # Ensure bounds
        self.current_k = max(self.k_start, min(self.current_k, self.k_end))
        
        return self.current_k
    
    def _linear_schedule(self, progress: float) -> int:
        """Linear interpolation from k_start to k_end."""
        return int(self.k_start + (self.k_end - self.k_start) * progress)
    
    def _cosine_schedule(self, progress: float) -> int:
        """Cosine annealing schedule."""
        return int(self.k_start + 0.5 * (self.k_end - self.k_start) * (1 + math.cos(math.pi * progress)))
    
    def _exponential_schedule(self, progress: float) -> int:
        """Exponential growth schedule."""
        return int(self.k_start * ((self.k_end / self.k_start) ** progress))
    
    def _step_schedule(self, progress: float) -> int:
        """Step-wise schedule with discrete jumps."""
        if progress < 0.25:
            return self.k_start
        elif progress < 0.5:
            return int(self.k_start + 0.3 * (self.k_end - self.k_start))
        elif progress < 0.75:
            return int(self.k_start + 0.7 * (self.k_end - self.k_start))
        else:
            return self.k_end
    
    def _adaptive_schedule(self, eigenvalues: Optional[torch.Tensor]) -> int:
        """
        Adaptive schedule based on eigenvalue gaps.
        
        Increases k when eigenvalue gaps are large (informative directions),
        decreases when gaps are small (redundant directions).
        """
        if eigenvalues is None or len(eigenvalues) < self.k_start:
            return self.current_k
        
        # Store eigenvalue history
        self.eigenvalue_history.append(eigenvalues.detach().cpu())
        if len(self.eigenvalue_history) > 10:
            self.eigenvalue_history.pop(0)
        
        # Calculate eigenvalue gaps
        sorted_eigenvals = torch.sort(eigenvalues, descending=True)[0]
        gaps = sorted_eigenvals[:-1] - sorted_eigenvals[1:]
        
        # Find elbow point (largest gap)
        if len(gaps) > self.k_start:
            gap_ratios = gaps[self.k_start:self.k_end] / (gaps[self.k_start-1] + 1e-8)
            significant_gaps = (gap_ratios > self.gap_threshold).sum().item()
            
            # Adjust k based on significant gaps
            target_k = self.k_start + significant_gaps
            
            # Smooth transition
            self.current_k = int(0.9 * self.current_k + 0.1 * target_k)
        
        return self.current_k
    
    def get_current_budget(self) -> int:
        """Get current projection budget."""
        return self.current_k
    
    def reset(self):
        """Reset scheduler state."""
        self.current_step = 0
        self.current_k = self.k_start
        self.eigenvalue_history.clear()


class ProjectionOperator:
    """
    Implements projection onto most informative curvature directions.
    
    Supports various projection strategies including:
    - Top-k eigenvalue projection
    - Adaptive thresholding
    - Block-wise projection
    """
    
    def __init__(
        self,
        projection_type: str = "top_k",
        threshold: float = 1e-6,
        normalize: bool = True
    ):
        """
        Initialize projection operator.
        
        Args:
            projection_type: Type of projection ("top_k", "threshold", "block")
            threshold: Threshold for eigenvalue filtering
            normalize: Whether to normalize projection
        """
        self.projection_type = projection_type
        self.threshold = threshold
        self.normalize = normalize
        
        # Cached projection matrices
        self.projection_cache: Dict[str, torch.Tensor] = {}
        self.eigenvalue_cache: Dict[str, torch.Tensor] = {}
    
    def compute_projection_matrix(
        self,
        fisher_matrix: torch.Tensor,
        k: int,
        layer_name: str = "default"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute projection matrix from Fisher Information Matrix.
        
        Args:
            fisher_matrix: Fisher Information Matrix or diagonal approximation
            k: Projection budget
            layer_name: Name for caching
            
        Returns:
            Tuple of (projection_matrix, eigenvalues)
        """
        if len(fisher_matrix.shape) == 1:
            # Diagonal Fisher approximation
            return self._compute_diagonal_projection(fisher_matrix, k, layer_name)
        elif len(fisher_matrix.shape) == 2:
            # Full or block Fisher matrix
            return self._compute_full_projection(fisher_matrix, k, layer_name)
        else:
            raise ValueError(f"Unsupported Fisher matrix shape: {fisher_matrix.shape}")
    
    def _compute_diagonal_projection(
        self,
        fisher_diagonal: torch.Tensor,
        k: int,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute projection for diagonal Fisher approximation."""
        if self.projection_type == "top_k":
            # Select top-k parameters by Fisher value
            top_k_values, top_k_indices = torch.topk(fisher_diagonal, min(k, len(fisher_diagonal)))
            
            # Create sparse projection matrix
            projection_matrix = torch.zeros_like(fisher_diagonal)
            projection_matrix[top_k_indices] = 1.0
            
            if self.normalize:
                projection_matrix = projection_matrix / (top_k_values.sum() + 1e-8)
            
        elif self.projection_type == "threshold":
            # Threshold-based projection
            projection_matrix = (fisher_diagonal > self.threshold).float()
            top_k_values = fisher_diagonal[projection_matrix > 0]
            
            if self.normalize and len(top_k_values) > 0:
                projection_matrix = projection_matrix / (top_k_values.sum() + 1e-8)
        
        else:
            projection_matrix = torch.ones_like(fisher_diagonal)
            top_k_values = fisher_diagonal
        
        # Cache for reuse
        self.projection_cache[layer_name] = projection_matrix
        self.eigenvalue_cache[layer_name] = top_k_values
        
        return projection_matrix, top_k_values
    
    def _compute_full_projection(
        self,
        fisher_matrix: torch.Tensor,
        k: int,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute projection for full Fisher matrix using eigendecomposition."""
        try:
            # Ensure symmetric matrix
            fisher_symmetric = 0.5 * (fisher_matrix + fisher_matrix.t())
            
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.symeig(fisher_symmetric, eigenvectors=True)
            
            # Sort by eigenvalue magnitude
            sorted_indices = torch.argsort(torch.abs(eigenvalues), descending=True)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            if self.projection_type == "top_k":
                # Select top-k eigenvectors
                k = min(k, len(eigenvalues))
                top_k_eigenvals = eigenvalues[:k]
                top_k_eigenvecs = eigenvectors[:, :k]
                
                # Projection matrix: P = V_k V_k^T
                projection_matrix = torch.mm(top_k_eigenvecs, top_k_eigenvecs.t())
                
            elif self.projection_type == "threshold":
                # Threshold-based selection
                significant_indices = torch.abs(eigenvalues) > self.threshold
                significant_eigenvals = eigenvalues[significant_indices]
                significant_eigenvecs = eigenvectors[:, significant_indices]
                
                if len(significant_eigenvals) > 0:
                    projection_matrix = torch.mm(significant_eigenvecs, significant_eigenvecs.t())
                    top_k_eigenvals = significant_eigenvals
                else:
                    projection_matrix = torch.eye(fisher_matrix.shape[0], device=fisher_matrix.device)
                    top_k_eigenvals = eigenvalues
            
            else:
                projection_matrix = torch.eye(fisher_matrix.shape[0], device=fisher_matrix.device)
                top_k_eigenvals = eigenvalues
            
            # Normalize if requested
            if self.normalize:
                trace = torch.trace(projection_matrix)
                if trace > 1e-8:
                    projection_matrix = projection_matrix / trace
            
            # Cache results
            self.projection_cache[layer_name] = projection_matrix
            self.eigenvalue_cache[layer_name] = top_k_eigenvals
            
            return projection_matrix, top_k_eigenvals
            
        except Exception as e:
            # Fallback to identity projection
            print(f"Eigendecomposition failed for {layer_name}: {e}")
            identity = torch.eye(fisher_matrix.shape[0], device=fisher_matrix.device)
            return identity, torch.ones(fisher_matrix.shape[0], device=fisher_matrix.device)
    
    def apply_projection(
        self,
        gradients: torch.Tensor,
        projection_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply projection to gradients.
        
        Args:
            gradients: Input gradients
            projection_matrix: Projection matrix
            
        Returns:
            Projected gradients
        """
        if len(projection_matrix.shape) == 1:
            # Diagonal projection
            return gradients * projection_matrix
        elif len(projection_matrix.shape) == 2:
            # Full matrix projection
            if gradients.dim() == 1:
                return torch.mv(projection_matrix, gradients)
            else:
                # For multi-dimensional gradients, flatten and project
                original_shape = gradients.shape
                flattened = gradients.view(-1)
                projected = torch.mv(projection_matrix, flattened)
                return projected.view(original_shape)
        else:
            return gradients
    
    def clear_cache(self):
        """Clear projection cache."""
        self.projection_cache.clear()
        self.eigenvalue_cache.clear()


class MultiModalProjector:
    """
    Specialized projector for Vision-Language Models.
    
    Handles mixed-modal Fisher computation and projection with
    separate treatment for vision, text, and cross-modal components.
    """
    
    def __init__(
        self,
        vision_weight: float = 0.4,
        text_weight: float = 0.4, 
        cross_weight: float = 0.2,
        separate_projections: bool = True
    ):
        """
        Initialize multimodal projector.
        
        Args:
            vision_weight: Weight for vision components
            text_weight: Weight for text components
            cross_weight: Weight for cross-modal components
            separate_projections: Whether to compute separate projections per modality
        """
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.cross_weight = cross_weight
        self.separate_projections = separate_projections
        
        # Separate projection operators
        self.vision_projector = ProjectionOperator()
        self.text_projector = ProjectionOperator()
        self.cross_projector = ProjectionOperator()
    
    def compute_mixed_modal_projection(
        self,
        vision_fisher: torch.Tensor,
        text_fisher: torch.Tensor,
        cross_fisher: Optional[torch.Tensor],
        k_vision: int,
        k_text: int,
        k_cross: int = 0
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute projections for mixed-modal Fisher matrices.
        
        Args:
            vision_fisher: Fisher matrix for vision components
            text_fisher: Fisher matrix for text components  
            cross_fisher: Fisher matrix for cross-modal components
            k_vision: Projection budget for vision
            k_text: Projection budget for text
            k_cross: Projection budget for cross-modal
            
        Returns:
            Dictionary of projection matrices and eigenvalues per modality
        """
        projections = {}
        
        if self.separate_projections:
            # Compute separate projections
            projections['vision'] = self.vision_projector.compute_projection_matrix(
                vision_fisher, k_vision, "vision"
            )
            projections['text'] = self.text_projector.compute_projection_matrix(
                text_fisher, k_text, "text"
            )
            
            if cross_fisher is not None and k_cross > 0:
                projections['cross'] = self.cross_projector.compute_projection_matrix(
                    cross_fisher, k_cross, "cross"
                )
        else:
            # Compute joint projection
            joint_fisher = (
                self.vision_weight * vision_fisher +
                self.text_weight * text_fisher
            )
            
            if cross_fisher is not None:
                joint_fisher += self.cross_weight * cross_fisher
            
            joint_projector = ProjectionOperator()
            projections['joint'] = joint_projector.compute_projection_matrix(
                joint_fisher, k_vision + k_text + k_cross, "joint"
            )
        
        return projections
    
    def apply_mixed_modal_projection(
        self,
        gradients: Dict[str, torch.Tensor],
        projections: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Apply mixed-modal projection to gradients."""
        projected_gradients = {}
        
        for modality, grad in gradients.items():
            if modality in projections:
                proj_matrix, _ = projections[modality]
                projected_gradients[modality] = self.vision_projector.apply_projection(grad, proj_matrix)
            elif 'joint' in projections:
                proj_matrix, _ = projections['joint']
                projected_gradients[modality] = self.vision_projector.apply_projection(grad, proj_matrix)
            else:
                projected_gradients[modality] = grad
        
        return projected_gradients
    
    def estimate_vision_rank(self, vision_activations: torch.Tensor) -> int:
        """
        Estimate effective rank of vision component.
        
        Used for empirical rule: k ≈ 1.2 × rank(A_vision)
        """
        try:
            # Compute SVD to estimate rank
            U, S, V = torch.svd(vision_activations.float())
            
            # Estimate rank using threshold on singular values
            threshold = 0.01 * S[0]  # 1% of largest singular value
            effective_rank = (S > threshold).sum().item()
            
            return effective_rank
        except:
            # Fallback estimate
            return min(32, vision_activations.shape[1])


# Factory functions for common configurations
def create_linear_scheduler(k_start: int = 32, k_end: int = 96, total_steps: int = 1000) -> ProjectionScheduler:
    """Create linear projection scheduler."""
    return ProjectionScheduler(k_start, k_end, total_steps, ProjectionScheduleType.LINEAR)

def create_cosine_scheduler(k_start: int = 32, k_end: int = 96, total_steps: int = 1000) -> ProjectionScheduler:
    """Create cosine projection scheduler."""
    return ProjectionScheduler(k_start, k_end, total_steps, ProjectionScheduleType.COSINE)

def create_adaptive_scheduler(k_start: int = 32, k_end: int = 96, total_steps: int = 1000) -> ProjectionScheduler:
    """Create adaptive projection scheduler."""
    return ProjectionScheduler(k_start, k_end, total_steps, ProjectionScheduleType.ADAPTIVE)

def create_top_k_projector() -> ProjectionOperator:
    """Create top-k projection operator."""
    return ProjectionOperator("top_k", normalize=True)

def create_threshold_projector(threshold: float = 1e-6) -> ProjectionOperator:
    """Create threshold-based projection operator."""
    return ProjectionOperator("threshold", threshold=threshold, normalize=True)