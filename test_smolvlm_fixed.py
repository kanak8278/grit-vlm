"""
Test GRIT with SmolVLM-256M-Instruct with fixed layer matching.
"""

import torch
import torch.nn as nn
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from grit_vlm import GRITLoRAConfig
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
import warnings
warnings.filterwarnings("ignore")

def create_test_images():
    """Create minimal test images."""
    print("🖼️ Creating test images...")
    
    images = []
    # Red square
    red_img = Image.fromarray((np.ones((64, 64, 3)) * [255, 0, 0]).astype(np.uint8))
    images.append(red_img)
    
    # Blue square
    blue_img = Image.fromarray((np.ones((64, 64, 3)) * [0, 0, 255]).astype(np.uint8))
    images.append(blue_img)
    
    print(f"✓ Created {len(images)} test images (64x64)")
    return images

def test_smolvlm_layer_matching():
    """Test that GRIT properly matches SmolVLM layers."""
    print("🔍 Testing SmolVLM Layer Matching")
    print("=" * 40)
    
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    print(f"📥 Loading {model_name}...")
    
    try:
        # Load SmolVLM model
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ SmolVLM loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Model class: {type(model).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to load SmolVLM: {e}")
        return False
    
    # Test GRIT layer matching
    print("\n🔧 Testing GRIT layer matching...")
    
    # Create minimal GRIT config
    grit_config = GRITLoRAConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        fisher_approximation="diagonal",
        enable_natural_gradient=True,
        enable_projection=True
    )
    
    print(f"✓ GRIT config: rank={grit_config.r}")
    
    # Debug: Check what layers the adapter will target
    temp_adapter = VLMGRITAdapter.__new__(VLMGRITAdapter)
    temp_adapter.model = model
    temp_adapter.config = grit_config
    
    vision_patterns = temp_adapter._get_default_vision_layers()
    text_patterns = temp_adapter._get_default_text_layers()
    cross_patterns = temp_adapter._get_default_cross_modal_layers()
    
    print(f"\n🎯 Target patterns:")
    print(f"   Vision: {vision_patterns}")
    print(f"   Text: {text_patterns}")
    print(f"   Cross-modal: {cross_patterns}")
    
    # Check pattern matching
    all_patterns = vision_patterns + text_patterns + cross_patterns
    matching_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pattern in all_patterns:
                if temp_adapter._should_adapt_layer(name, [pattern]):
                    matching_layers.append((name, pattern, f"{module.in_features}→{module.out_features}"))
                    break
    
    print(f"\n📋 Matching layers ({len(matching_layers)} found):")
    for name, pattern, dims in matching_layers:
        print(f"   {name:<45} {dims:<12} [matched: {pattern}]")
    
    if len(matching_layers) == 0:
        print("⚠️  No matching layers found - pattern matching failed!")
        return False
    
    # Now apply GRIT adapter
    print(f"\n🧩 Applying GRIT adapter to {len(matching_layers)} layers...")
    
    try:
        grit_adapter = VLMGRITAdapter(model, grit_config)
        adapted_layers = len(grit_adapter.grit_layers)
        print(f"✅ GRIT adapter applied!")
        print(f"   Adapted layers: {adapted_layers}")
        
        if adapted_layers > 0:
            grit_adapter.print_adaptation_summary()
            
            # Quick test with forward pass
            print("\n🧪 Testing forward pass...")
            
            images = create_test_images()
            text = "What color is this?"
            
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": text}
                ]
            }]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images[0], input_text, return_tensors="pt")
            
            print(f"   Input tokens: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"   Pixel values: {inputs['pixel_values'].shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                
            print(f"   ✅ Forward pass successful!")
            print(f"   Logits shape: {outputs.logits.shape}")
            
            # Test trainable parameters
            trainable_params = grit_adapter.get_trainable_parameters()
            print(f"   Trainable GRIT parameters: {len(trainable_params)}")
            
            # Test backward pass
            print("\n🏋️ Testing backward pass...")
            
            model.train()
            
            # Simple training step
            outputs = model(**inputs)
            labels = inputs["input_ids"].clone()
            loss = nn.CrossEntropyLoss()(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1)
            )
            
            print(f"   Loss: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            print("   ✅ Backward pass successful!")
            
            # Check gradients
            grad_found = False
            for param in trainable_params:
                if param.grad is not None:
                    grad_found = True
                    print(f"   ✓ Gradient found: {param.grad.shape}")
                    break
            
            if grad_found:
                # Update Fisher matrices
                grit_adapter.update_mixed_modal_fisher()
                print("   ✓ Fisher matrices updated")
                
                return True
            else:
                print("   ⚠️  No gradients found")
                return False
        else:
            print("   ❌ No layers were adapted!")
            return False
            
    except Exception as e:
        print(f"❌ GRIT adapter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 SmolVLM + GRIT Fixed Layer Matching Test")
    print("=" * 50)
    
    success = test_smolvlm_layer_matching()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SMOLVLM + GRIT FIXED TEST PASSED!")
        print("\n📋 Results:")
        print("  ✅ SmolVLM-256M loaded successfully")
        print("  ✅ Layer patterns correctly matched")
        print("  ✅ GRIT adaptations applied")
        print("  ✅ Forward passes working")
        print("  ✅ Backward passes working")
        print("  ✅ Fisher matrices updating")
        print("\n🚀 GRIT-SmolVLM integration is working!")
    else:
        print("\n❌ Test failed - check logs above")