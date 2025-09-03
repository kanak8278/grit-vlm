"""GRIT-VLM configuration module."""

from .model_configs import (
    ModalityType,
    LayerSelectionStrategy, 
    ModalityConfig,
    ModelGRITConfig,
    MODEL_CONFIGS,
    get_model_config,
    filter_layers_by_strategy,
    create_custom_config,
    EXAMPLE_CONFIGS
)

from .device_config import (
    DeviceType,
    DeviceStrategy,
    get_available_devices,
    get_device_info,
    select_optimal_device,
    optimize_model_kwargs,
    print_device_summary,
    get_device_config,
    DEVICE_CONFIGS
)

__all__ = [
    # Model configuration
    "ModalityType",
    "LayerSelectionStrategy",
    "ModalityConfig", 
    "ModelGRITConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "filter_layers_by_strategy",
    "create_custom_config",
    "EXAMPLE_CONFIGS",
    
    # Device configuration
    "DeviceType",
    "DeviceStrategy",
    "get_available_devices",
    "get_device_info", 
    "select_optimal_device",
    "optimize_model_kwargs",
    "print_device_summary",
    "get_device_config",
    "DEVICE_CONFIGS"
]