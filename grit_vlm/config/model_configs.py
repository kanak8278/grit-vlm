"""
Model-specific GRIT configuration system.

This module defines which layers to adapt with GRIT for different model architectures.
Allows fine-grained control over which layers get GRIT adaptations.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ModalityType(Enum):
    """Types of modalities in VLM."""

    VISION = "vision"
    TEXT = "text"
    CROSS_MODAL = "cross_modal"
    FUSION = "fusion"


class LayerSelectionStrategy(Enum):
    """Strategies for selecting which layers to adapt."""

    ALL = "all"  # Adapt all matching layers
    FIRST_N = "first_n"  # Adapt first N layers
    LAST_N = "last_n"  # Adapt last N layers
    EVERY_NTH = "every_nth"  # Adapt every Nth layer
    SPECIFIC = "specific"  # Adapt specific layer indices
    SKIP_FIRST_LAST = "skip_first_last"  # Skip first and last layers


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""

    # Layer patterns to match
    layer_patterns: List[str] = field(default_factory=list)

    # Selection strategy
    strategy: LayerSelectionStrategy = LayerSelectionStrategy.ALL

    # Strategy-specific parameters
    n_layers: Optional[int] = None  # For first_n, last_n
    step_size: Optional[int] = None  # For every_nth
    specific_indices: Optional[List[int]] = None  # For specific

    # GRIT parameters for this modality
    rank: Optional[int] = None
    alpha: Optional[float] = None
    dropout: Optional[float] = None

    # Enable/disable GRIT features
    enable_natural_gradient: bool = True
    enable_projection: bool = True

    # Fisher approximation type for this modality
    fisher_approximation: str = "diagonal"


@dataclass
class ModelGRITConfig:
    """GRIT configuration for a specific model architecture."""

    # Model identification
    model_name: str
    model_type: str  # e.g., "idefics3", "llava", "blip2"

    # Modality configurations
    vision: Optional[ModalityConfig] = None
    text: Optional[ModalityConfig] = None
    cross_modal: Optional[ModalityConfig] = None
    fusion: Optional[ModalityConfig] = None

    # Global GRIT settings (can be overridden by modality configs)
    global_rank: int = 16
    global_alpha: float = 32
    global_dropout: float = 0.1

    # Fisher settings
    fisher_damping: float = 1e-4
    fisher_ema_decay: float = 0.95
    fisher_update_freq: int = 10

    # Projection settings
    projection_budget_start: int = 32
    projection_budget_end: int = 96
    projection_schedule: str = "linear"

    # Performance settings
    max_layers_per_modality: Optional[int] = None  # Limit for performance


# Pre-defined model configurations
MODEL_CONFIGS: Dict[str, ModelGRITConfig] = {
    # SmolVLM-256M-Instruct (Idefics3)
    "smolvlm_256m": ModelGRITConfig(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        model_type="idefics3",
        vision=ModalityConfig(
            layer_patterns=[
                "model.vision_model.encoder.layers.*.self_attn.q_proj",
                "model.vision_model.encoder.layers.*.self_attn.k_proj",
                "model.vision_model.encoder.layers.*.self_attn.v_proj",
                "model.vision_model.encoder.layers.*.self_attn.out_proj",
            ],
            strategy=LayerSelectionStrategy.EVERY_NTH,
            step_size=2,  # Adapt every 2nd layer for performance
            rank=8,
            alpha=16,
        ),
        text=ModalityConfig(
            layer_patterns=[
                "model.text_model.layers.*.self_attn.q_proj",
                "model.text_model.layers.*.self_attn.k_proj",
                "model.text_model.layers.*.self_attn.v_proj",
                "model.text_model.layers.*.self_attn.o_proj",
            ],
            strategy=LayerSelectionStrategy.LAST_N,
            n_layers=8,  # Adapt only last 8 text layers
            rank=12,
            alpha=24,
        ),
        cross_modal=ModalityConfig(
            layer_patterns=["model.multi_modal_projector.*"],
            strategy=LayerSelectionStrategy.ALL,
            rank=16,
            alpha=32,
        ),
        max_layers_per_modality=20,  # Performance limit
    ),
    # SmolVLM Fast (minimal adaptation)
    "smolvlm_fast": ModelGRITConfig(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        model_type="idefics3",
        vision=ModalityConfig(
            layer_patterns=[
                "model.vision_model.encoder.layers.*.self_attn.q_proj",
                "model.vision_model.encoder.layers.*.self_attn.v_proj",
            ],
            strategy=LayerSelectionStrategy.FIRST_N,
            n_layers=3,  # Only first 3 layers
            rank=4,
            alpha=8,
        ),
        text=ModalityConfig(
            layer_patterns=[
                "model.text_model.layers.*.self_attn.q_proj",
                "model.text_model.layers.*.self_attn.v_proj",
            ],
            strategy=LayerSelectionStrategy.LAST_N,
            n_layers=4,  # Only last 4 layers
            rank=4,
            alpha=8,
        ),
        cross_modal=ModalityConfig(
            layer_patterns=["model.multi_modal_projector.*"],
            strategy=LayerSelectionStrategy.ALL,
            rank=8,
            alpha=16,
        ),
    ),
}


# Utility functions
def get_model_config(model_name_or_path: str) -> Optional[ModelGRITConfig]:
    """Get GRIT configuration for a model."""

    # Exact match first
    if model_name_or_path in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name_or_path]

    # Pattern matching
    model_lower = model_name_or_path.lower()

    if "smolvlm" in model_lower:
        return MODEL_CONFIGS["smolvlm_256m"]

    return None


def filter_layers_by_strategy(
    all_layers: List[str], strategy: LayerSelectionStrategy, **kwargs
) -> List[str]:
    """Filter layers based on selection strategy."""

    if strategy == LayerSelectionStrategy.ALL:
        return all_layers

    elif strategy == LayerSelectionStrategy.FIRST_N:
        n = kwargs.get("n_layers", len(all_layers))
        return all_layers[:n]

    elif strategy == LayerSelectionStrategy.LAST_N:
        n = kwargs.get("n_layers", len(all_layers))
        return all_layers[-n:]

    elif strategy == LayerSelectionStrategy.EVERY_NTH:
        step = kwargs.get("step_size", 2)
        return all_layers[::step]

    elif strategy == LayerSelectionStrategy.SPECIFIC:
        indices = kwargs.get("specific_indices", [])
        return [all_layers[i] for i in indices if i < len(all_layers)]

    elif strategy == LayerSelectionStrategy.SKIP_FIRST_LAST:
        if len(all_layers) <= 2:
            return all_layers
        return all_layers[1:-1]

    return all_layers


def create_custom_config(
    model_name: str,
    vision_patterns: List[str] = None,
    text_patterns: List[str] = None,
    cross_modal_patterns: List[str] = None,
    **kwargs,
) -> ModelGRITConfig:
    """Create a custom GRIT configuration."""

    config = ModelGRITConfig(model_name=model_name, model_type="custom", **kwargs)

    if vision_patterns:
        config.vision = ModalityConfig(layer_patterns=vision_patterns)
    if text_patterns:
        config.text = ModalityConfig(layer_patterns=text_patterns)
    if cross_modal_patterns:
        config.cross_modal = ModalityConfig(layer_patterns=cross_modal_patterns)

    return config


# Example usage configurations
EXAMPLE_CONFIGS = {
    "minimal_test": {
        "description": "Minimal config for fast testing",
        "config": ModelGRITConfig(
            model_name="test",
            model_type="minimal",
            vision=ModalityConfig(
                layer_patterns=["*.q_proj", "*.v_proj"],
                strategy=LayerSelectionStrategy.FIRST_N,
                n_layers=2,
                rank=4,
                alpha=8,
            ),
        ),
    },
    "balanced_performance": {
        "description": "Balanced config for good performance/quality tradeoff",
        "config": ModelGRITConfig(
            model_name="balanced",
            model_type="balanced",
            vision=ModalityConfig(
                strategy=LayerSelectionStrategy.EVERY_NTH, step_size=2, rank=12
            ),
            text=ModalityConfig(
                strategy=LayerSelectionStrategy.LAST_N, n_layers=8, rank=16
            ),
        ),
    },
}
