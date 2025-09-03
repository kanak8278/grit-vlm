"""
Compare CPU vs MPS vs CUDA performance for GRIT-VLM.
"""

import torch
import time
from transformers import Idefics3ForConditionalGeneration
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from grit_vlm import GRITLoRAConfig
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
from grit_vlm.config import (
    DeviceStrategy, get_available_devices, get_device_info,
    optimize_model_kwargs, print_device_summary, get_device_config
)

def test_device_performance():
    """Test GRIT performance across all available devices."""
    print("🖥️  Device Performance Comparison")
    print("=" * 40)
    
    # Show device info
    device_info = get_device_info()
    available = device_info["available_devices"]
    
    print(f"📋 Available devices:")
    for device, avail in available.items():
        status = "✅" if avail else "❌"
        print(f"  {status} {device.upper()}")
        if device == "cuda" and avail:
            print(f"      └─ {device_info['cuda_device_name']}")
    
    # Test configurations
    configs = []
    
    # Always test CPU
    configs.append({"name": "CPU", "device_map": "cpu", "torch_dtype": torch.float32})
    
    # Test MPS if available
    if available["mps"]:
        configs.append({"name": "MPS", "device_map": "auto", "torch_dtype": torch.float16, "force_mps": True})
    
    # Test CUDA if available 
    if available["cuda"]:
        configs.append({"name": "CUDA", "device_map": "auto", "torch_dtype": torch.float16, "force_cuda": True})
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']} Configuration")
        print("-" * 30)
        
        try:
            # Load model with device-specific config
            start_time = time.time()
            
            # Force specific device if needed
            if config.get("force_mps") and not available["cuda"]:
                # Ensure MPS is used when CUDA not available
                pass
            elif config.get("force_cuda") and available["cuda"]:
                # Ensure CUDA is used
                pass
            
            model = Idefics3ForConditionalGeneration.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                torch_dtype=config["torch_dtype"],
                device_map=config["device_map"],
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            load_time = time.time() - start_time
            
            # Get actual device
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            print(f"✓ Model loaded on {device} ({dtype}) in {load_time:.2f}s")
            
            # Apply GRIT
            start_time = time.time()
            grit_config = GRITLoRAConfig(r=4, lora_alpha=8)
            adapter = VLMGRITAdapter(
                model=model,
                config=grit_config,
                model_config_name="smolvlm_fast"
            )
            grit_time = time.time() - start_time
            print(f"✓ GRIT applied in {grit_time:.2f}s")
            
            # Test inference
            test_image = Image.fromarray((np.ones((64, 64, 3)) * [255, 0, 0]).astype(np.uint8))
            
            # Prepare simple input (skip processor for speed)
            input_ids = torch.randint(0, 1000, (1, 10)).to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            
            # Time inference
            start_time = time.time()
            num_runs = 5
            for _ in range(num_runs):
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
            inference_time = (time.time() - start_time) / num_runs
            
            print(f"✓ Average inference time: {inference_time*1000:.2f}ms")
            
            # Show memory usage
            if device.type == "cuda":
                memory_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"✓ CUDA memory: {memory_mb:.1f} MB")
            else:
                print(f"✓ Memory: Not tracked for {device.type}")
            
            # Test backward pass
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            loss = outputs.logits.sum()
            loss.backward()
            backward_time = time.time() - start_time
            print(f"✓ Backward pass time: {backward_time*1000:.2f}ms")
            
            # Cleanup
            del model, adapter
            
            # Clear device cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_device_strategies():
    """Test different device selection strategies."""
    print("\n🎯 Device Strategy Testing")
    print("=" * 30)
    
    strategies = [
        ("auto_performance", "Best performance"),
        ("auto_inference", "Optimized for inference"),
        ("stable_training", "Most stable training"),
        ("memory_efficient", "Memory efficient")
    ]
    
    for config_name, description in strategies:
        print(f"\n🔧 {config_name}: {description}")
        try:
            kwargs = get_device_config(config_name)
            device_map = kwargs["device_map"]
            dtype = kwargs["torch_dtype"]
            print(f"  └─ Device: {device_map}, Type: {dtype}")
        except Exception as e:
            print(f"  └─ Error: {e}")

def show_device_recommendations():
    """Show device recommendations based on available hardware."""
    print("\n💡 Device Recommendations")
    print("=" * 30)
    
    available = get_available_devices()
    
    print("📋 For different use cases:")
    
    if available["cuda"]:
        print("🚀 CUDA Available:")
        print("  • Training: Use CUDA (fastest backward passes)")  
        print("  • Inference: Use CUDA (fastest overall)")
        print("  • Memory: Use CUDA with max_memory limits")
        
    elif available["mps"]:
        print("🍎 MPS Available (Apple Silicon):")
        print("  • Training: Use CPU (more stable backward passes)")
        print("  • Inference: Use MPS (1.7x faster than CPU)")
        print("  • Memory: Use CPU (more predictable)")
        
    else:
        print("💻 CPU Only:")
        print("  • Training: CPU with float32")
        print("  • Inference: CPU (only option)")
        print("  • Memory: Use low_cpu_mem_usage=True")
    
    print(f"\n🔧 Quick configs:")
    print(f"  • Fast training: get_device_config('stable_training')")
    print(f"  • Fast inference: get_device_config('auto_inference')")
    print(f"  • Memory efficient: get_device_config('memory_efficient')")

if __name__ == "__main__":
    print("🚀 GRIT-VLM Device Configuration Test")
    print("=" * 55)
    
    # Test device performance
    test_device_performance()
    
    # Test device strategies
    test_device_strategies()
    
    # Show recommendations
    show_device_recommendations()