# GRIT-VLM Configuration Guide

## Overview

The GRIT-VLM configuration system allows you to specify exactly which layers to train with GRIT adaptations for different Vision-Language Models. This gives you fine-grained control over performance, memory usage, and training quality.

## Quick Start

### Using Predefined Configurations

```python
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
from grit_vlm import GRITLoRAConfig

# Fast config (7 layers) - for quick testing
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast"
)

# Full config (up to 20 layers/modality) - for best quality  
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(), 
    model_config_name="smolvlm_256m"
)
```

## Available Predefined Configurations

| Config Name | Model | Vision Layers | Text Layers | Cross Layers | Description |
|-------------|-------|---------------|-------------|--------------|-------------|
| `smolvlm_fast` | SmolVLM-256M | 3 (first_n) | 4 (last_n) | 1 (all) | Fast testing (7 total) |
| `smolvlm_256m` | SmolVLM-256M | 12 (every_2nd) | 8 (last_n) | 1 (all) | Full quality (max 20/modality) |

## Layer Selection Strategies

### 1. ALL - Adapt all matching layers
```python
strategy=LayerSelectionStrategy.ALL
```

### 2. FIRST_N - Adapt first N layers
```python
strategy=LayerSelectionStrategy.FIRST_N
n_layers=3  # Adapt first 3 layers
```

### 3. LAST_N - Adapt last N layers  
```python
strategy=LayerSelectionStrategy.LAST_N
n_layers=8  # Adapt last 8 layers
```

### 4. EVERY_NTH - Adapt every Nth layer
```python
strategy=LayerSelectionStrategy.EVERY_NTH
step_size=2  # Adapt every 2nd layer
```

### 5. SPECIFIC - Adapt specific indices
```python
strategy=LayerSelectionStrategy.SPECIFIC
specific_indices=[0, 5, 10]  # Adapt layers at indices 0, 5, 10
```

### 6. SKIP_FIRST_LAST - Skip first and last layers
```python
strategy=LayerSelectionStrategy.SKIP_FIRST_LAST
```

## Creating Custom Configurations

### Method 1: Define ModelGRITConfig

```python
from grit_vlm.config import ModelGRITConfig, ModalityConfig, LayerSelectionStrategy

custom_config = ModelGRITConfig(
    model_name="my-custom-vlm",
    model_type="custom",
    
    # Vision configuration
    vision=ModalityConfig(
        layer_patterns=[
            "model.vision_model.encoder.layers.*.self_attn.q_proj",
            "model.vision_model.encoder.layers.*.self_attn.v_proj"
        ],
        strategy=LayerSelectionStrategy.FIRST_N,
        n_layers=5,
        rank=8,
        alpha=16
    ),
    
    # Text configuration
    text=ModalityConfig(
        layer_patterns=[
            "model.text_model.layers.*.self_attn.q_proj"
        ],
        strategy=LayerSelectionStrategy.LAST_N,
        n_layers=6,
        rank=12,
        alpha=24
    ),
    
    # Global settings
    global_rank=16,
    global_alpha=32,
    max_layers_per_modality=10  # Performance limit
)

# Use the custom config
adapter = VLMGRITAdapter(model=model, config=custom_config)
```

### Method 2: Override Specific Layers

```python
# Use predefined config but override specific modalities
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast",
    vision_layers=[
        "model.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.vision_model.encoder.layers.1.self_attn.v_proj"
    ]
)
```

## Configuration Parameters

### ModelGRITConfig Parameters

```python
@dataclass
class ModelGRITConfig:
    # Model identification
    model_name: str                          # Model identifier
    model_type: str                          # Architecture type
    
    # Modality configurations
    vision: Optional[ModalityConfig] = None
    text: Optional[ModalityConfig] = None
    cross_modal: Optional[ModalityConfig] = None
    fusion: Optional[ModalityConfig] = None
    
    # Global GRIT settings
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
    max_layers_per_modality: Optional[int] = None
```

### ModalityConfig Parameters

```python
@dataclass
class ModalityConfig:
    # Layer patterns to match
    layer_patterns: List[str] = field(default_factory=list)
    
    # Selection strategy
    strategy: LayerSelectionStrategy = LayerSelectionStrategy.ALL
    
    # Strategy-specific parameters
    n_layers: Optional[int] = None          # For first_n, last_n
    step_size: Optional[int] = None         # For every_nth
    specific_indices: Optional[List[int]] = None  # For specific
    
    # GRIT parameters for this modality
    rank: Optional[int] = None
    alpha: Optional[float] = None
    dropout: Optional[float] = None
    
    # Enable/disable features
    enable_natural_gradient: bool = True
    enable_projection: bool = True
    
    # Fisher approximation
    fisher_approximation: str = "diagonal"
```

## Layer Pattern Matching

Layer patterns use glob-style wildcards:

```python
# Examples for SmolVLM
patterns = [
    "model.vision_model.encoder.layers.*.self_attn.q_proj",     # All vision q_proj layers
    "model.text_model.layers.*.self_attn.*_proj",              # All text attention projections
    "model.multi_modal_projector.*"                            # All cross-modal layers
]
```

The `*` wildcard matches any characters except dots (`.`), making it perfect for layer indices.

## Performance Optimization

### Memory Usage

- **Fast configs**: Use fewer layers (3-10 total)
- **Performance limits**: Set `max_layers_per_modality` to cap adaptation size
- **Rank reduction**: Use smaller ranks (4-8) for less memory

### Training Speed

- **Layer selection**: Use `FIRST_N` or `LAST_N` instead of `ALL`
- **Skip strategies**: Use `EVERY_NTH` to adapt every 2nd or 3rd layer
- **Modality focus**: Only adapt the most important modality

### Quality vs Performance

| Priority | Config | Layers | Memory | Speed |
|----------|--------|--------|---------|-------|
| Speed | `smolvlm_fast` | 7 | Low | Fast |
| Balanced | Custom first_n=5 | 15 | Medium | Medium |
| Quality | `smolvlm_256m` | 20+ | High | Slow |

## Device Configuration

The configuration system works with all device setups:

```python
# CPU only (current setup)
model = load_model(device_map="cpu")

# MPS (Apple Silicon) 
model = load_model(device_map="auto")  # Auto-selects MPS

# GPU
model = load_model(device_map="auto", torch_dtype=torch.float16)
```

## Troubleshooting

### No Layers Adapted
```
✓ GRIT applied to 0 layers
```
**Solution**: Check layer patterns match your model architecture. Use the pattern matching test:

```python
# Debug layer matching
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_fast")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"Available layer: {name}")
```

### Too Many Layers (Timeout)
```
⚠️ Limiting vision layers to 20 (was 48) for performance
```
**Solution**: Use `max_layers_per_modality` or selection strategies:

```python
# Limit total layers
config.max_layers_per_modality = 10

# Or use selection strategy
config.vision.strategy = LayerSelectionStrategy.EVERY_NTH
config.vision.step_size = 3  # Every 3rd layer
```

### Memory Issues
**Solution**: Reduce ranks and use fast configs:

```python
# Minimal memory config
config = ModelGRITConfig(
    vision=ModalityConfig(rank=4, alpha=8),
    text=ModalityConfig(rank=4, alpha=8),
    max_layers_per_modality=5
)
```

## Examples by Use Case

### Quick Testing (Fast)
```python
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast"
)
# Result: 7 layers, ~123K parameters, fast training
```

### Research/Development (Balanced)
```python
custom_config = ModelGRITConfig(
    model_name="research",
    vision=ModalityConfig(strategy=LayerSelectionStrategy.EVERY_NTH, step_size=2, rank=12),
    text=ModalityConfig(strategy=LayerSelectionStrategy.LAST_N, n_layers=10, rank=16),
    max_layers_per_modality=15
)
# Result: ~15 layers, balanced performance/quality
```

### Production (Quality)
```python
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_256m"
)
# Result: 20+ layers, maximum quality, slower training
```

## Device Configuration

The GRIT configuration system includes intelligent device selection for CUDA, MPS, and CPU:

### Automatic Device Selection

```python
from grit_vlm.config import get_device_config

# Optimized for different scenarios
training_config = get_device_config("stable_training")    # CPU for stable training
inference_config = get_device_config("auto_inference")    # GPU for fast inference
memory_config = get_device_config("memory_efficient")     # Memory optimized

# Use with model loading
model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    **training_config  # Automatically selects best device + settings
)
```

### Device Strategies

| Strategy | Use Case | Device Priority | Description |
|----------|----------|-----------------|-------------|
| `stable_training` | Training/Fine-tuning | CPU → CUDA → MPS | Most stable gradients |
| `auto_inference` | Inference only | CUDA → MPS → CPU | Fastest inference |
| `auto_performance` | General use | CUDA → MPS → CPU | Best overall performance |
| `memory_efficient` | Limited memory | CUDA (limited) → CPU | Memory optimized |

### Device-Specific Optimizations

**CUDA (NVIDIA GPU):**
- ✅ Best for training (fast backward passes)
- ✅ Best for inference (fastest overall)
- ✅ Memory management with limits
- ✅ Float16 support

**MPS (Apple Silicon):**
- ✅ Good for inference (1.7x faster than CPU)
- ⚠️ Slower for training (backward passes)
- ✅ Float16 support
- ⚠️ Less predictable memory

**CPU:**
- ✅ Most stable for training
- ✅ Predictable memory usage
- ✅ Always available
- ❌ Slowest option

### Integrated Usage

```python
# Combine model config + device config
from grit_vlm.config import get_model_config, get_device_config

# Get optimal device settings
device_kwargs = get_device_config("stable_training")

# Load model with device optimization
model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    **device_kwargs
)

# Apply GRIT with model-specific config
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast"  # Layer selection strategy
)
```

### Performance Recommendations

Based on your hardware:

**🚀 CUDA Available:**
```python
# Training
device_kwargs = get_device_config("stable_training")  # Usually selects CUDA
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_256m")

# Inference  
device_kwargs = get_device_config("auto_inference")   # Selects CUDA
```

**🍎 MPS Available (Mac):**
```python
# Training (CPU more stable)
device_kwargs = get_device_config("stable_training")  # Selects CPU
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_fast")

# Inference (MPS faster)
device_kwargs = get_device_config("auto_inference")   # Selects MPS
```

**💻 CPU Only:**
```python
# All scenarios
device_kwargs = get_device_config("memory_efficient") # Optimized CPU settings
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_fast")
```

This configuration system gives you complete control over GRIT adaptations while providing sensible defaults for common use cases.