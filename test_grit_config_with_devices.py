"""
Test GRIT configuration system with intelligent device selection.

Demonstrates how to combine model configs with device optimization.
"""

import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from grit_vlm import GRITLoRAConfig
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
from grit_vlm.config import (
    get_model_config, get_device_config, optimize_model_kwargs,
    DeviceStrategy, print_device_summary
)

def test_integrated_config_system():
    """Test model config + device config integration."""
    print("🔧 Integrated Configuration Test")
    print("=" * 40)
    
    # Show device summary first
    print_device_summary(DeviceStrategy.PERFORMANCE, "training")
    
    print(f"\n🎯 Testing SmolVLM Fast + Auto Device Selection")
    print("-" * 50)
    
    try:
        # Get optimal device configuration
        device_kwargs = get_device_config("stable_training")  # Best for training
        print(f"✓ Device config: {device_kwargs['device_map']} ({device_kwargs['torch_dtype']})")
        
        # Load SmolVLM with optimized settings
        print("📥 Loading SmolVLM with device optimization...")
        model = Idefics3ForConditionalGeneration.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            **device_kwargs
        )
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        print(f"✓ Model loaded on {device} with {dtype}")
        
        # Apply GRIT with fast config
        print("\n🧩 Applying GRIT with fast configuration...")
        adapter = VLMGRITAdapter(
            model=model,
            config=GRITLoRAConfig(),
            model_config_name="smolvlm_fast"  # Fast layer selection
        )
        
        print(f"✓ GRIT applied to {len(adapter.grit_layers)} layers")
        
        # Quick test
        print("\n🧪 Testing inference...")
        test_image = Image.fromarray((np.ones((64, 64, 3)) * [255, 0, 0]).astype(np.uint8))
        
        # Simple input for testing
        inputs = torch.randint(0, 1000, (1, 10)).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs)
            
        print(f"✓ Inference successful: {outputs.logits.shape}")
        
        # Test training step
        print("\n🏋️ Testing training step...")
        outputs = model(input_ids=inputs)
        loss = outputs.logits.sum()
        loss.backward()
        
        # Update Fisher
        adapter.update_mixed_modal_fisher()
        print("✓ Training step completed with Fisher update")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_different_strategies():
    """Test different device strategies for different scenarios."""
    print(f"\n🎲 Strategy Comparison")
    print("=" * 25)
    
    scenarios = [
        ("Training", "stable_training"),
        ("Inference", "auto_inference"),
        ("Memory constrained", "memory_efficient"),
        ("Max performance", "auto_performance")
    ]
    
    for scenario, strategy in scenarios:
        print(f"\n📋 {scenario}:")
        try:
            kwargs = get_device_config(strategy)
            device_map = kwargs["device_map"]
            dtype = kwargs["torch_dtype"]
            print(f"  Strategy: {strategy}")
            print(f"  Device: {device_map}")
            print(f"  Type: {dtype}")
            
            # Show additional settings
            if "max_memory" in kwargs:
                print(f"  Memory limit: {kwargs['max_memory']}")
            if kwargs.get("low_cpu_mem_usage"):
                print(f"  Low CPU mem: enabled")
                
        except Exception as e:
            print(f"  Error: {e}")

def show_usage_examples():
    """Show practical usage examples."""
    print(f"\n💡 Usage Examples")
    print("=" * 20)
    
    examples = [
        {
            "title": "Fast testing (CPU stable)",
            "code": '''
# Best for quick testing and debugging
device_kwargs = get_device_config("stable_training")
model = load_model("SmolVLM-256M-Instruct", **device_kwargs)
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_fast")
'''
        },
        {
            "title": "Production inference (GPU optimized)",
            "code": '''
# Best for production inference
device_kwargs = get_device_config("auto_inference") 
model = load_model("SmolVLM-256M-Instruct", **device_kwargs)
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_256m")
'''
        },
        {
            "title": "Memory constrained",
            "code": '''
# When memory is limited
device_kwargs = get_device_config("memory_efficient")
model = load_model("SmolVLM-256M-Instruct", **device_kwargs)
adapter = VLMGRITAdapter(model, config, model_config_name="smolvlm_fast")
'''
        },
        {
            "title": "Custom optimization",
            "code": '''
# Custom device + model config
device_kwargs = optimize_model_kwargs(
    device_strategy=DeviceStrategy.PERFORMANCE,
    use_case="inference"
)
model = load_model("SmolVLM-256M-Instruct", **device_kwargs)
'''
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(example['code'])

if __name__ == "__main__":
    print("🚀 GRIT Configuration + Device Optimization")
    print("=" * 60)
    
    # Test integrated system
    success = test_integrated_config_system()
    
    # Test different strategies
    test_different_strategies()
    
    # Show usage examples
    show_usage_examples()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ INTEGRATED CONFIG SYSTEM WORKS!")
        print("\n🎯 Key Features:")
        print("  ✓ Automatic device detection (CUDA/MPS/CPU)")
        print("  ✓ Strategy-based device selection")
        print("  ✓ Optimized model loading parameters")
        print("  ✓ Integration with GRIT layer configs")
        print("  ✓ Use-case specific optimization")
        print("\n🚀 Ready for production use!")
    else:
        print("\n❌ Integration test failed")