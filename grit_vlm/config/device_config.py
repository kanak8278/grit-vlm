"""
Device configuration utilities for GRIT-VLM.

Handles automatic device selection and configuration for CUDA, MPS, and CPU.
"""

import torch
from typing import Dict, Any, Optional
from enum import Enum

class DeviceType(Enum):
    """Available device types."""
    AUTO = "auto"           # Automatic selection
    CUDA = "cuda"          # NVIDIA GPU
    MPS = "mps"            # Apple Silicon GPU
    CPU = "cpu"            # CPU only

class DeviceStrategy(Enum):
    """Device selection strategies."""
    PERFORMANCE = "performance"     # Fastest available device
    MEMORY = "memory"              # Best memory efficiency
    STABILITY = "stability"        # Most stable for training
    INFERENCE = "inference"        # Optimized for inference only

def get_available_devices() -> Dict[str, bool]:
    """Get information about available devices."""
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "cpu": True  # Always available
    }

def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        "available_devices": get_available_devices(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "cpu_count": torch.get_num_threads()
    }
    return info

def select_optimal_device(
    strategy: DeviceStrategy = DeviceStrategy.PERFORMANCE,
    use_case: str = "training"  # "training" or "inference"
) -> str:
    """
    Select optimal device based on strategy and use case.
    
    Args:
        strategy: Device selection strategy
        use_case: "training" or "inference"
        
    Returns:
        Device map string for transformers
    """
    available = get_available_devices()
    
    if strategy == DeviceStrategy.PERFORMANCE:
        # Priority: CUDA > MPS > CPU
        if available["cuda"]:
            return "auto"  # Let transformers handle CUDA placement
        elif available["mps"] and use_case == "inference":
            return "auto"  # MPS good for inference
        else:
            return "cpu"   # CPU for training or if no GPU
    
    elif strategy == DeviceStrategy.MEMORY:
        # Priority: CUDA (with memory management) > CPU > MPS
        if available["cuda"]:
            return "auto"  # CUDA has better memory management
        else:
            return "cpu"   # CPU more predictable than MPS
    
    elif strategy == DeviceStrategy.STABILITY:
        # Priority: CPU > CUDA > MPS (most stable first)
        if use_case == "training":
            return "cpu"   # Most stable for training
        elif available["cuda"]:
            return "auto"  # CUDA stable for inference
        else:
            return "cpu"
    
    elif strategy == DeviceStrategy.INFERENCE:
        # Priority: CUDA > MPS > CPU (fastest inference)
        if available["cuda"]:
            return "auto"
        elif available["mps"]:
            return "auto"  # MPS good for inference
        else:
            return "cpu"
    
    return "cpu"  # Fallback

def get_torch_device(device_map: str) -> torch.device:
    """Convert device_map to torch.device."""
    if device_map == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_map)

def get_model_dtype(device: torch.device) -> torch.dtype:
    """Get optimal dtype for device."""
    if device.type in ["cuda", "mps"]:
        return torch.float16  # Half precision for GPU
    else:
        return torch.float32  # Full precision for CPU

def optimize_model_kwargs(
    device_strategy: DeviceStrategy = DeviceStrategy.PERFORMANCE,
    use_case: str = "training",
    **base_kwargs
) -> Dict[str, Any]:
    """
    Optimize model loading kwargs based on device strategy.
    
    Args:
        device_strategy: Device selection strategy
        use_case: "training" or "inference"
        **base_kwargs: Base model loading arguments
        
    Returns:
        Optimized kwargs for model loading
    """
    device_map = select_optimal_device(device_strategy, use_case)
    device = get_torch_device(device_map)
    dtype = get_model_dtype(device)
    
    kwargs = base_kwargs.copy()
    kwargs.update({
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    })
    
    # Device-specific optimizations
    if device.type == "cuda":
        kwargs.update({
            "max_memory": {0: "80%"},  # Limit CUDA memory usage
        })
    elif device.type == "cpu":
        kwargs.update({
            "torch_dtype": torch.float32,  # CPU works better with float32
        })
    
    return kwargs

def print_device_summary(device_strategy: DeviceStrategy, use_case: str):
    """Print device configuration summary."""
    info = get_device_info()
    device_map = select_optimal_device(device_strategy, use_case)
    device = get_torch_device(device_map)
    dtype = get_model_dtype(device)
    
    print("🖥️  Device Configuration Summary")
    print("=" * 35)
    print(f"Strategy: {device_strategy.value}")
    print(f"Use case: {use_case}")
    print(f"Selected device: {device}")
    print(f"Device map: {device_map}")
    print(f"Data type: {dtype}")
    
    print(f"\n📋 Available Devices:")
    for device_name, available in info["available_devices"].items():
        status = "✅" if available else "❌"
        print(f"  {status} {device_name.upper()}")
        
        if device_name == "cuda" and available:
            print(f"      └─ {info['cuda_device_count']} device(s)")
            if info["cuda_device_name"]:
                print(f"      └─ {info['cuda_device_name']}")

# Predefined device configurations
DEVICE_CONFIGS = {
    "auto_performance": {
        "strategy": DeviceStrategy.PERFORMANCE,
        "use_case": "training"
    },
    "auto_inference": {
        "strategy": DeviceStrategy.INFERENCE, 
        "use_case": "inference"
    },
    "stable_training": {
        "strategy": DeviceStrategy.STABILITY,
        "use_case": "training"
    },
    "memory_efficient": {
        "strategy": DeviceStrategy.MEMORY,
        "use_case": "training"
    },
    "cpu_only": {
        "strategy": DeviceStrategy.STABILITY,
        "use_case": "training",
        "force_device": "cpu"
    },
    "gpu_only": {
        "strategy": DeviceStrategy.PERFORMANCE,
        "use_case": "inference",
        "force_device": "auto"
    }
}

def get_device_config(config_name: str) -> Dict[str, Any]:
    """Get predefined device configuration."""
    if config_name not in DEVICE_CONFIGS:
        raise ValueError(f"Unknown device config: {config_name}. Available: {list(DEVICE_CONFIGS.keys())}")
    
    config = DEVICE_CONFIGS[config_name].copy()
    
    # Handle forced device
    if "force_device" in config:
        device_map = config.pop("force_device")
    else:
        device_map = select_optimal_device(
            strategy=config["strategy"],
            use_case=config["use_case"]
        )
    
    return optimize_model_kwargs(
        device_strategy=config["strategy"],
        use_case=config["use_case"]
    )