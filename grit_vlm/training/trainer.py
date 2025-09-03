"""
GRIT training loop with HuggingFace integration.

Provides custom Trainer class that integrates GRIT natural gradients,
Fisher updates, and projection scheduling with the HuggingFace ecosystem.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Callable
import warnings
from dataclasses import dataclass
from transformers import (
    Trainer, TrainingArguments, DataCollator,
    PreTrainedModel, PreTrainedTokenizerBase
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import Dataset
import numpy as np

from ..core.grit_lora import GRITLoRALayer
from ..models.vlm_adapter import VLMGRITAdapter
from ..optimizers.natural_gradient import create_grit_optimizer


@dataclass
class GRITTrainingArguments(TrainingArguments):
    """Extended training arguments for GRIT training."""
    
    # GRIT-specific arguments
    fisher_update_freq: int = 10
    natural_gradient_damping: float = 1e-3
    enable_natural_gradient: bool = True
    enable_projection: bool = True
    
    # Projection scheduling
    projection_budget_start: int = 32
    projection_budget_end: int = 96
    projection_schedule: str = "linear"
    
    # Fisher approximation
    fisher_approximation: str = "diagonal"
    fisher_ema_decay: float = 0.95
    fisher_damping: float = 1e-4
    
    # Optimizer settings
    grit_optimizer_type: str = "sgd"  # "sgd", "adaptive", "adam"
    
    # Evaluation and logging
    log_fisher_stats: bool = True
    save_fisher_matrices: bool = False


class GRITTrainer(Trainer):
    """
    Custom Trainer for GRIT-LoRA fine-tuning.
    
    Integrates natural gradient optimization, Fisher Information Matrix updates,
    and projection scheduling with standard HuggingFace training loop.
    """
    
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[GRITTrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        grit_adapter: Optional[VLMGRITAdapter] = None
    ):
        """
        Initialize GRIT Trainer.
        
        Args:
            grit_adapter: VLM GRIT adapter instance
            Other args: Standard HuggingFace Trainer arguments
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        
        self.grit_adapter = grit_adapter
        self.grit_args = args if isinstance(args, GRITTrainingArguments) else GRITTrainingArguments()
        
        # Fisher and projection tracking
        self.fisher_update_count = 0
        self.projection_update_count = 0
        
        # Statistics for logging
        self.fisher_stats = {
            'diagonal_norms': [],
            'eigenvalue_ranges': [],
            'projection_ranks': []
        }
        
        # Setup GRIT optimizer if not provided
        if optimizers[0] is None:
            self.optimizer = self._create_grit_optimizer()
    
    def _create_grit_optimizer(self):
        """Create GRIT optimizer with appropriate configuration."""
        if self.grit_adapter is None:
            # Fallback to standard optimizer
            return super().create_optimizer()
        
        # Get GRIT layers for optimizer registration
        grit_layers = self.grit_adapter.get_grit_layers()
        trainable_params = self.grit_adapter.get_trainable_parameters()
        
        optimizer = create_grit_optimizer(
            trainable_params,
            optimizer_type=self.grit_args.grit_optimizer_type,
            grit_layers=grit_layers,
            lr=self.grit_args.learning_rate,
            natural_gradient_damping=self.grit_args.natural_gradient_damping,
            enable_natural_gradient=self.grit_args.enable_natural_gradient,
            fisher_update_freq=self.grit_args.fisher_update_freq
        )
        
        return optimizer
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Custom training step with GRIT enhancements.
        
        Performs standard forward/backward pass plus:
        1. Fisher Information Matrix updates
        2. Natural gradient computation
        3. Projection scheduling
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Forward pass
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Backward pass
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        # GRIT-specific updates
        if self.grit_adapter is not None:
            self._update_grit_components(inputs)
        
        return loss.detach()
    
    def _update_grit_components(self, inputs: Dict[str, Any]):
        """Update GRIT components (Fisher, projections, etc.)."""
        
        # Extract activations for mixed-modal Fisher computation
        vision_activations = self._extract_vision_activations(inputs)
        text_activations = self._extract_text_activations(inputs)
        
        # Update mixed-modal Fisher matrices
        self.grit_adapter.update_mixed_modal_fisher(
            vision_activations=vision_activations,
            text_activations=text_activations
        )
        
        # Update projection matrices if needed
        if self.state.global_step % self.grit_args.fisher_update_freq == 0:
            self._update_projections()
            self._log_fisher_statistics()
    
    def _extract_vision_activations(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract vision activations from model inputs."""
        # This depends on the specific VLM architecture
        # For demonstration, assuming pixel_values or images key
        if 'pixel_values' in inputs:
            return inputs['pixel_values']
        elif 'images' in inputs:
            return inputs['images']
        else:
            return None
    
    def _extract_text_activations(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract text activations from model inputs."""
        if 'input_ids' in inputs:
            return inputs['input_ids']
        else:
            return None
    
    def _update_projections(self):
        """Update projection matrices based on current Fisher estimates."""
        if self.grit_adapter is None:
            return
        
        try:
            projections = self.grit_adapter.get_mixed_modal_projections()
            self.projection_update_count += 1
            
            # Log projection statistics
            if self.grit_args.log_fisher_stats:
                for modality, (proj_matrix, eigenvalues) in projections.items():
                    if len(eigenvalues) > 0:
                        self.fisher_stats['eigenvalue_ranges'].append({
                            'modality': modality,
                            'min': eigenvalues.min().item(),
                            'max': eigenvalues.max().item(),
                            'mean': eigenvalues.mean().item()
                        })
                        
                        # Estimate effective rank
                        threshold = 0.01 * eigenvalues.max()
                        effective_rank = (eigenvalues > threshold).sum().item()
                        self.fisher_stats['projection_ranks'].append({
                            'modality': modality,
                            'rank': effective_rank,
                            'step': self.state.global_step
                        })
        
        except Exception as e:
            warnings.warn(f"Failed to update projections: {e}")
    
    def _log_fisher_statistics(self):
        """Log Fisher Information statistics."""
        if not self.grit_args.log_fisher_stats or self.grit_adapter is None:
            return
        
        try:
            # Collect Fisher diagonal norms from GRIT layers
            for layer_name, grit_layer in self.grit_adapter.grit_layers.items():
                if grit_layer.fisher_matrix.fisher_diagonal is not None:
                    diagonal_norm = grit_layer.fisher_matrix.fisher_diagonal.norm().item()
                    self.fisher_stats['diagonal_norms'].append({
                        'layer': layer_name,
                        'norm': diagonal_norm,
                        'step': self.state.global_step
                    })
            
            # Log to tensorboard/wandb if available
            if self.state.global_step % (self.grit_args.fisher_update_freq * 5) == 0:
                self._log_to_tensorboard()
        
        except Exception as e:
            warnings.warn(f"Failed to log Fisher statistics: {e}")
    
    def _log_to_tensorboard(self):
        """Log statistics to tensorboard."""
        if len(self.fisher_stats['diagonal_norms']) > 0:
            recent_norms = [stat['norm'] for stat in self.fisher_stats['diagonal_norms'][-10:]]
            avg_fisher_norm = np.mean(recent_norms)
            
            self.log({
                'grit/avg_fisher_norm': avg_fisher_norm,
                'grit/fisher_updates': self.fisher_update_count,
                'grit/projection_updates': self.projection_update_count
            })
        
        if len(self.fisher_stats['projection_ranks']) > 0:
            recent_ranks = [stat['rank'] for stat in self.fisher_stats['projection_ranks'][-5:]]
            avg_rank = np.mean(recent_ranks)
            
            self.log({
                'grit/avg_projection_rank': avg_rank
            })
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with GRIT adapter weights."""
        super().save_model(output_dir, _internal_call)
        
        # Save GRIT adapter if available
        if self.grit_adapter is not None and output_dir is not None:
            adapter_path = f"{output_dir}/grit_adapter.pth"
            self.grit_adapter.save_adapter(adapter_path)
            
            # Save Fisher matrices if requested
            if self.grit_args.save_fisher_matrices:
                fisher_path = f"{output_dir}/fisher_matrices.pth"
                self._save_fisher_matrices(fisher_path)
    
    def _save_fisher_matrices(self, path: str):
        """Save Fisher Information matrices for analysis."""
        fisher_data = {}
        
        if self.grit_adapter is not None:
            for layer_name, grit_layer in self.grit_adapter.grit_layers.items():
                if grit_layer.fisher_matrix.fisher_diagonal is not None:
                    fisher_data[layer_name] = {
                        'diagonal': grit_layer.fisher_matrix.fisher_diagonal.cpu(),
                        'step_count': grit_layer.fisher_matrix.step_count
                    }
        
        torch.save(fisher_data, path)
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Evaluation with GRIT-specific metrics."""
        
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add GRIT-specific metrics
        if self.grit_adapter is not None:
            grit_metrics = self._compute_grit_metrics()
            eval_results.update({f"{metric_key_prefix}/{k}": v for k, v in grit_metrics.items()})
        
        return eval_results
    
    def _compute_grit_metrics(self) -> Dict[str, float]:
        """Compute GRIT-specific evaluation metrics."""
        metrics = {}
        
        if self.grit_adapter is None:
            return metrics
        
        # Parameter efficiency metrics
        total_params = sum(p.numel() for p in self.model.parameters())
        grit_params = sum(p.numel() for p in self.grit_adapter.get_trainable_parameters())
        metrics['grit_param_efficiency'] = grit_params / total_params
        
        # Fisher Information metrics
        if len(self.fisher_stats['diagonal_norms']) > 0:
            recent_norms = [stat['norm'] for stat in self.fisher_stats['diagonal_norms'][-20:]]
            metrics['grit_avg_fisher_norm'] = np.mean(recent_norms)
            metrics['grit_fisher_std'] = np.std(recent_norms)
        
        # Projection metrics
        if len(self.fisher_stats['projection_ranks']) > 0:
            recent_ranks = [stat['rank'] for stat in self.fisher_stats['projection_ranks'][-10:]]
            metrics['grit_avg_projection_rank'] = np.mean(recent_ranks)
            metrics['grit_projection_efficiency'] = np.mean(recent_ranks) / self.grit_args.projection_budget_end
        
        # Convergence metrics
        metrics['grit_fisher_updates'] = self.fisher_update_count
        metrics['grit_projection_updates'] = self.projection_update_count
        
        return metrics


class GRITTrainerCallback(TrainerCallback):
    """Callback for GRIT-specific training events."""
    
    def __init__(self, grit_adapter: Optional[VLMGRITAdapter] = None):
        self.grit_adapter = grit_adapter
        self.best_fisher_norm = float('inf')
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize GRIT components at training start."""
        if self.grit_adapter is not None:
            print("Starting GRIT training with curvature-aware optimization...")
            self.grit_adapter.print_adaptation_summary()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor GRIT components during training."""
        # Could add adaptive learning rate adjustments based on Fisher norms
        pass
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Log additional GRIT metrics during evaluation."""
        pass


def create_grit_trainer(
    model: PreTrainedModel,
    grit_adapter: VLMGRITAdapter,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    training_args: Optional[GRITTrainingArguments] = None,
    **kwargs
) -> GRITTrainer:
    """
    Factory function to create a GRIT trainer.
    
    Args:
        model: The model to train
        grit_adapter: GRIT adapter instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: GRIT training arguments
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured GRIT trainer
    """
    if training_args is None:
        training_args = GRITTrainingArguments(
            output_dir="./grit_output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # GRIT-specific settings
            fisher_update_freq=10,
            enable_natural_gradient=True,
            enable_projection=True,
            log_fisher_stats=True
        )
    
    # Add GRIT callback
    callbacks = kwargs.get('callbacks', [])
    callbacks.append(GRITTrainerCallback(grit_adapter))
    kwargs['callbacks'] = callbacks
    
    trainer = GRITTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        grit_adapter=grit_adapter,
        **kwargs
    )
    
    return trainer