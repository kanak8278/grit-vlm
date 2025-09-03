"""
VLM-specific GRIT adaptations for Vision-Language Models.

Provides specialized implementations for applying GRIT-LoRA to multimodal
architectures with separate treatment for vision, text, and cross-modal components.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import warnings

from ..core.grit_lora import GRITLoRALayer, GRITLoRAConfig, create_grit_lora_layer
from ..core.fisher_info import FisherInformationMatrix, FisherApproximationType
from ..utils.projection import MultiModalProjector, ProjectionScheduler, create_linear_scheduler
from ..config.model_configs import (
    ModelGRITConfig, ModalityConfig, get_model_config, filter_layers_by_strategy,
    LayerSelectionStrategy
)


class VLMGRITAdapter:
    """
    GRIT adapter for Vision-Language Models.
    
    Handles mixed-modal Fisher computation with separate treatment for:
    - Vision encoder components
    - Language decoder components  
    - Cross-modal attention layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[GRITLoRAConfig, ModelGRITConfig],
        model_config_name: Optional[str] = None,
        vision_layers: Optional[List[str]] = None,
        text_layers: Optional[List[str]] = None,
        cross_modal_layers: Optional[List[str]] = None
    ):
        """
        Initialize VLM GRIT adapter.
        
        Args:
            model: The VLM model to adapt
            config: GRIT-LoRA configuration or ModelGRITConfig
            model_config_name: Name of predefined model config (e.g., 'smolvlm_fast')
            vision_layers: Names of vision encoder layers to adapt (overrides config)
            text_layers: Names of text decoder layers to adapt (overrides config)
            cross_modal_layers: Names of cross-modal layers to adapt (overrides config)
        """
        self.model = model
        
        # Handle different config types
        if isinstance(config, ModelGRITConfig):
            self.model_config = config
            # Convert ModelGRITConfig to GRITLoRAConfig for backward compatibility
            self.config = self._convert_to_grit_lora_config(config)
        else:
            self.config = config
            # Try to get model config from model name or explicit name
            if model_config_name:
                self.model_config = get_model_config(model_config_name)
            else:
                # Try to infer from model class name
                model_name = getattr(model, '_name_or_path', str(type(model).__name__))
                self.model_config = get_model_config(model_name)
        
        # Get layer configurations
        if self.model_config:
            # Use model-specific configuration with layer selection strategies
            if vision_layers is None and self.model_config.vision:
                vision_layers = self._get_layers_from_config(self.model_config.vision, "vision")
            if text_layers is None and self.model_config.text:
                text_layers = self._get_layers_from_config(self.model_config.text, "text")
            if cross_modal_layers is None and self.model_config.cross_modal:
                cross_modal_layers = self._get_layers_from_config(self.model_config.cross_modal, "cross_modal")
        
        # Fallback to default layer configurations
        if vision_layers is None:
            vision_layers = self._get_default_vision_layers()
        if text_layers is None:
            text_layers = self._get_default_text_layers()
        if cross_modal_layers is None:
            cross_modal_layers = self._get_default_cross_modal_layers()
        
        self.vision_layers = vision_layers
        self.text_layers = text_layers
        self.cross_modal_layers = cross_modal_layers
        
        # GRIT layer storage
        self.grit_layers: Dict[str, GRITLoRALayer] = {}
        
        # Mixed-modal Fisher computation
        self.multimodal_projector = MultiModalProjector()
        
        # Projection schedulers for different modalities
        self.vision_scheduler = create_linear_scheduler(
            config.projection_budget_start, 
            config.projection_budget_end, 
            1000
        )
        self.text_scheduler = create_linear_scheduler(
            config.projection_budget_start, 
            config.projection_budget_end, 
            1000
        )
        
        # Apply GRIT adaptations
        self._apply_grit_adaptations()
        
        # Activation storage for mixed-modal Fisher
        self.vision_activations: List[torch.Tensor] = []
        self.text_activations: List[torch.Tensor] = []
        self.cross_activations: List[torch.Tensor] = []
    
    def _convert_to_grit_lora_config(self, model_config: ModelGRITConfig) -> GRITLoRAConfig:
        """Convert ModelGRITConfig to GRITLoRAConfig for backward compatibility."""
        # Use global settings from model config
        return GRITLoRAConfig(
            r=model_config.global_rank,
            lora_alpha=model_config.global_alpha,
            lora_dropout=model_config.global_dropout,
            fisher_damping=model_config.fisher_damping,
            fisher_ema_decay=model_config.fisher_ema_decay,
            fisher_update_freq=model_config.fisher_update_freq,
            projection_budget_start=model_config.projection_budget_start,
            projection_budget_end=model_config.projection_budget_end,
            projection_schedule=model_config.projection_schedule
        )
    
    def _get_layers_from_config(self, modality_config: ModalityConfig, modality_name: str) -> List[str]:
        """Get filtered layer names from modality configuration."""
        # Get all matching layers for this modality
        all_matching_layers = []
        
        for pattern in modality_config.layer_patterns:
            # Convert pattern to regex and find matching layers
            matching_layers = self._find_layers_by_pattern(pattern)
            all_matching_layers.extend(matching_layers)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_layers = []
        for layer in all_matching_layers:
            if layer not in seen:
                seen.add(layer)
                unique_layers.append(layer)
        
        # Apply selection strategy
        filtered_layers = filter_layers_by_strategy(
            unique_layers,
            modality_config.strategy,
            n_layers=modality_config.n_layers,
            step_size=modality_config.step_size,
            specific_indices=modality_config.specific_indices
        )
        
        # Apply performance limit if set
        if self.model_config and self.model_config.max_layers_per_modality:
            max_layers = self.model_config.max_layers_per_modality
            if len(filtered_layers) > max_layers:
                print(f"⚠️  Limiting {modality_name} layers to {max_layers} (was {len(filtered_layers)}) for performance")
                filtered_layers = filtered_layers[:max_layers]
        
        return filtered_layers
    
    def _find_layers_by_pattern(self, pattern: str) -> List[str]:
        """Find layer names matching a pattern (supports * wildcards)."""
        import re
        
        # Convert glob pattern to regex
        regex_pattern = pattern.replace('*', '[^.]*')  # * matches any characters except dots
        regex_pattern = f"^{regex_pattern}$"
        
        matching_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and re.match(regex_pattern, name):
                matching_layers.append(name)
        
        return matching_layers

    def _get_default_vision_layers(self) -> List[str]:
        """Get default vision layer names based on model architecture."""
        model_name = self.model.__class__.__name__.lower()
        
        if "idefics3" in model_name:
            # SmolVLM and other Idefics3 models
            return [
                "model.vision_model.encoder.layers.*.self_attn.q_proj",
                "model.vision_model.encoder.layers.*.self_attn.k_proj",
                "model.vision_model.encoder.layers.*.self_attn.v_proj", 
                "model.vision_model.encoder.layers.*.self_attn.out_proj"
            ]
        elif "qwen2vl" in model_name:
            return [
                "visual.attn_pool.attn.q_proj",
                "visual.attn_pool.attn.k_proj",
                "visual.attn_pool.attn.v_proj",
                "visual.attn_pool.attn.out_proj"
            ]
        elif "phi" in model_name and "vision" in model_name:
            return [
                "vision_embed_tokens.img_processor.vision_model.encoder.layers.*.self_attn.q_proj",
                "vision_embed_tokens.img_processor.vision_model.encoder.layers.*.self_attn.k_proj",
                "vision_embed_tokens.img_processor.vision_model.encoder.layers.*.self_attn.v_proj"
            ]
        else:
            # Generic vision transformer patterns
            return [
                "*.vision*.attn.q_proj",
                "*.vision*.attn.k_proj", 
                "*.vision*.attn.v_proj",
                "*.vision*.attn.out_proj"
            ]
    
    def _get_default_text_layers(self) -> List[str]:
        """Get default text layer names based on model architecture."""
        model_name = self.model.__class__.__name__.lower()
        
        if "idefics3" in model_name:
            # SmolVLM and other Idefics3 models 
            return [
                "model.text_model.layers.*.self_attn.q_proj",
                "model.text_model.layers.*.self_attn.k_proj",
                "model.text_model.layers.*.self_attn.v_proj",
                "model.text_model.layers.*.self_attn.o_proj"
            ]
        else:
            # Generic text model patterns
            return [
                "language_model.model.layers.*.self_attn.q_proj",
                "language_model.model.layers.*.self_attn.k_proj",
                "language_model.model.layers.*.self_attn.v_proj", 
                "language_model.model.layers.*.self_attn.o_proj",
                "language_model.model.layers.*.mlp.gate_proj",
                "language_model.model.layers.*.mlp.up_proj",
                "language_model.model.layers.*.mlp.down_proj"
            ]
    
    def _get_default_cross_modal_layers(self) -> List[str]:
        """Get default cross-modal layer names based on model architecture."""
        model_name = self.model.__class__.__name__.lower()
        
        if "idefics3" in model_name:
            # SmolVLM and other Idefics3 models
            return [
                "model.connector.modality_projection.proj"
            ]
        else:
            # Generic cross-modal patterns
            return [
                "multi_modal_projector.*",
                "*.cross_attn.*",
                "*.vision_language_connector.*"
            ]
    
    def _apply_grit_adaptations(self):
        """Apply GRIT-LoRA adaptations to target layers."""
        target_modules = (
            self.vision_layers + 
            self.text_layers + 
            self.cross_modal_layers
        )
        
        for name, module in self.model.named_modules():
            if self._should_adapt_layer(name, target_modules) and isinstance(module, (nn.Linear, nn.Conv2d)):
                # Determine modality type
                modality = self._get_layer_modality(name)
                
                # Create GRIT-LoRA layer
                grit_layer = create_grit_lora_layer(module, self.config, f"{modality}_{name}")
                
                # Replace module in model
                self._replace_module(name, grit_layer)
                
                # Store reference
                self.grit_layers[name] = grit_layer
    
    def _should_adapt_layer(self, layer_name: str, target_patterns: List[str]) -> bool:
        """Check if a layer should be adapted based on target patterns."""
        for pattern in target_patterns:
            # Simple pattern matching (can be extended with regex)
            if "*" in pattern:
                pattern_parts = pattern.split("*")
                if all(part in layer_name for part in pattern_parts if part):
                    return True
            elif pattern in layer_name:
                return True
        return False
    
    def _get_layer_modality(self, layer_name: str) -> str:
        """Determine the modality of a layer based on its name."""
        if any(pattern in layer_name for pattern in ["vision", "visual", "img", "image"]):
            return "vision"
        elif any(pattern in layer_name for pattern in ["language", "text", "lm", "llm"]):
            return "text"
        elif any(pattern in layer_name for pattern in ["cross", "modal", "projector", "connector"]):
            return "cross"
        else:
            return "unknown"
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model hierarchy."""
        name_parts = module_name.split(".")
        parent = self.model
        
        # Navigate to parent module
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the target module
        setattr(parent, name_parts[-1], new_module)
    
    def update_mixed_modal_fisher(
        self,
        vision_activations: Optional[torch.Tensor] = None,
        text_activations: Optional[torch.Tensor] = None,
        cross_activations: Optional[torch.Tensor] = None
    ):
        """
        Update mixed-modal Fisher Information matrices.
        
        Args:
            vision_activations: Recent vision activations
            text_activations: Recent text activations  
            cross_activations: Recent cross-modal activations
        """
        # Store activations for Fisher computation
        if vision_activations is not None:
            self.vision_activations.append(vision_activations.detach())
            # Limit buffer size
            if len(self.vision_activations) > 50:
                self.vision_activations.pop(0)
        
        if text_activations is not None:
            self.text_activations.append(text_activations.detach())
            if len(self.text_activations) > 50:
                self.text_activations.pop(0)
        
        if cross_activations is not None:
            self.cross_activations.append(cross_activations.detach())
            if len(self.cross_activations) > 50:
                self.cross_activations.pop(0)
        
        # Update Fisher matrices in individual layers
        print(f"\u2699\ufe0f  GRIT updating {len(self.grit_layers)} layers...")
        for layer_name, grit_layer in self.grit_layers.items():
            modality = self._get_layer_modality(layer_name)
            print(f"    \ud83d\udcca {modality} layer: {layer_name.split('.')[-2:]}")
            grit_layer.update_fisher_and_projection()
    
    def get_mixed_modal_projections(self) -> Dict[str, Any]:
        """
        Compute mixed-modal projections using current Fisher estimates.
        
        Returns:
            Dictionary containing projection matrices for each modality
        """
        # Collect Fisher matrices by modality
        vision_fishers = []
        text_fishers = []
        cross_fishers = []
        
        for layer_name, grit_layer in self.grit_layers.items():
            modality = self._get_layer_modality(layer_name)
            fisher_matrix = grit_layer.fisher_matrix.fisher_diagonal
            
            if fisher_matrix is not None:
                if modality == "vision":
                    vision_fishers.append(fisher_matrix)
                elif modality == "text":
                    text_fishers.append(fisher_matrix)
                elif modality == "cross":
                    cross_fishers.append(fisher_matrix)
        
        # Aggregate Fisher matrices
        vision_fisher = torch.cat(vision_fishers) if vision_fishers else torch.tensor([])
        text_fisher = torch.cat(text_fishers) if text_fishers else torch.tensor([])
        cross_fisher = torch.cat(cross_fishers) if cross_fishers else None
        
        # Get projection budgets
        k_vision = self.vision_scheduler.step(
            vision_rank=self.multimodal_projector.estimate_vision_rank(
                torch.stack(self.vision_activations[-10:]) if self.vision_activations else torch.randn(10, 100)
            )
        )
        k_text = self.text_scheduler.step()
        k_cross = max(8, k_vision // 4) if cross_fisher is not None else 0
        
        # Compute projections
        if len(vision_fisher) > 0 and len(text_fisher) > 0:
            projections = self.multimodal_projector.compute_mixed_modal_projection(
                vision_fisher, text_fisher, cross_fisher, k_vision, k_text, k_cross
            )
            return projections
        else:
            return {}
    
    def get_grit_layers(self) -> List[GRITLoRALayer]:
        """Get all GRIT-LoRA layers for optimizer registration."""
        return list(self.grit_layers.values())
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters from GRIT layers."""
        params = []
        for grit_layer in self.grit_layers.values():
            params.extend([grit_layer.lora_A, grit_layer.lora_B])
        return params
    
    def save_adapter(self, path: str):
        """Save GRIT adapter weights."""
        state_dict = {}
        for name, layer in self.grit_layers.items():
            state_dict[f"{name}.lora_A"] = layer.lora_A.data
            state_dict[f"{name}.lora_B"] = layer.lora_B.data
        
        torch.save({
            'grit_state_dict': state_dict,
            'config': self.config,
            'layer_mappings': {
                'vision': self.vision_layers,
                'text': self.text_layers, 
                'cross': self.cross_modal_layers
            }
        }, path)
    
    def load_adapter(self, path: str):
        """Load GRIT adapter weights."""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['grit_state_dict']
        
        for name, layer in self.grit_layers.items():
            if f"{name}.lora_A" in state_dict:
                layer.lora_A.data.copy_(state_dict[f"{name}.lora_A"])
            if f"{name}.lora_B" in state_dict:
                layer.lora_B.data.copy_(state_dict[f"{name}.lora_B"])
    
    def print_adaptation_summary(self):
        """Print summary of applied adaptations."""
        total_params = sum(p.numel() for p in self.model.parameters())
        grit_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        print(f"GRIT-VLM Adaptation Summary:")
        print(f"- Total model parameters: {total_params:,}")
        print(f"- GRIT trainable parameters: {grit_params:,}")
        print(f"- Adaptation ratio: {grit_params/total_params:.2%}")
        print(f"- Number of adapted layers: {len(self.grit_layers)}")
        print(f"  - Vision layers: {sum(1 for name in self.grit_layers if 'vision' in self._get_layer_modality(name))}")
        print(f"  - Text layers: {sum(1 for name in self.grit_layers if 'text' in self._get_layer_modality(name))}")
        print(f"  - Cross-modal layers: {sum(1 for name in self.grit_layers if 'cross' in self._get_layer_modality(name))}")


def create_vlm_grit_adapter(
    model_name_or_path: str,
    config: Optional[GRITLoRAConfig] = None,
    **model_kwargs
) -> Tuple[nn.Module, VLMGRITAdapter]:
    """
    Create a VLM with GRIT adaptations.
    
    Args:
        model_name_or_path: HuggingFace model identifier
        config: GRIT configuration (uses defaults if None)
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (adapted_model, grit_adapter)
    """
    if config is None:
        config = GRITLoRAConfig()
    
    # Load base model
    try:
        # Generic model loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name_or_path}: {e}")
    
    # Create GRIT adapter
    grit_adapter = VLMGRITAdapter(model, config)
    
    return model, grit_adapter


# Note: Specialized adapters for other VLM architectures can be added here
# when tested and validated with the configuration system