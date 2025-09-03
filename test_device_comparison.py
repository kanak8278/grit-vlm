"""
Compare CPU vs MPS performance for GRIT-VLM.
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

def test_device_performance():
    """Test GRIT performance on CPU vs MPS."""
    print("🖥️  Device Performance Comparison")
    print("=" * 40)
    
    # Check available devices
    print(f"✓ MPS available: {torch.backends.mps.is_available()}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    configs = [
        {"name": "CPU", "device_map": "cpu"},
        {"name": "MPS (Auto)", "device_map": "auto"}
    ]
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']} Configuration")
        print("-" * 30)
        
        try:
            # Load model
            start_time = time.time()
            model = Idefics3ForConditionalGeneration.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                torch_dtype=torch.float16,
                device_map=config['device_map'],
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            load_time = time.time() - start_time
            
            # Get actual device
            device = next(model.parameters()).device
            print(f"✓ Model loaded on {device} in {load_time:.2f}s")
            
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
            print(f"✓ Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A'}")
            
            # Test backward pass
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            loss = outputs.logits.sum()
            loss.backward()
            backward_time = time.time() - start_time
            print(f"✓ Backward pass time: {backward_time*1000:.2f}ms")
            
            # Cleanup
            del model, adapter
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 GRIT-VLM Device Performance Test")
    print("=" * 50)
    test_device_performance()
    
    print(f"\n💡 Recommendations:")
    print(f"  • Use device_map='auto' for MPS acceleration")
    print(f"  • MPS typically 2-3x faster than CPU")
    print(f"  • MPS supports float16 for memory efficiency")
    print(f"  • CPU is more stable for debugging")