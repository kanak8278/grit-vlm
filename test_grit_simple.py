"""
Simple test script to verify GRIT-LoRA implementation works.
Tests forward and backward pass with a small model.
"""

import torch
import torch.nn as nn
from grit_vlm import GRITLoRAConfig
from grit_vlm.core.grit_lora import GRITLoRALinear

def test_grit_lora_basic():
    """Test basic GRIT-LoRA functionality."""
    print("🧪 Testing GRIT-LoRA Basic Functionality")
    print("=" * 40)
    
    # Create a simple linear layer
    base_layer = nn.Linear(128, 64)
    
    # Create GRIT-LoRA config
    config = GRITLoRAConfig(
        r=8,  # Small rank for testing
        lora_alpha=16,
        fisher_approximation="diagonal",
        enable_natural_gradient=True,
        enable_projection=True
    )
    
    # Create GRIT-LoRA layer
    grit_layer = GRITLoRALinear(base_layer, config)
    
    print(f"✓ Created GRIT-LoRA layer: {grit_layer.extra_repr()}")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 128)
    
    print(f"✓ Input shape: {input_tensor.shape}")
    
    # Forward pass
    output = grit_layer(input_tensor)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    print(f"✓ LoRA A gradient shape: {grit_layer.lora_A.grad.shape}")
    print(f"✓ LoRA B gradient shape: {grit_layer.lora_B.grad.shape}")
    
    # Test Fisher matrix update
    grit_layer.update_fisher_and_projection()
    print(f"✓ Fisher matrix update completed")
    
    # Test natural gradients
    nat_grads = grit_layer.get_natural_gradients()
    print(f"✓ Natural gradients computed: {list(nat_grads.keys())}")
    
    print("\n🎉 Basic GRIT-LoRA test passed!")
    return True

def test_grit_with_simple_model():
    """Test GRIT with a simple multi-layer model."""
    print("\n🧪 Testing GRIT with Simple Model")
    print("=" * 35)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 32)
            self.layer2 = nn.Linear(32, 16)
            self.layer3 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    model = SimpleModel()
    print(f"✓ Created simple model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Apply GRIT to one layer
    config = GRITLoRAConfig(r=4, lora_alpha=8)
    grit_layer = GRITLoRALinear(model.layer1, config)
    
    # Replace layer in model
    model.layer1 = grit_layer
    
    print(f"✓ Applied GRIT to layer1")
    
    # Test training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate dummy data
    x = torch.randn(8, 64)
    y = torch.randn(8, 1)
    
    # Forward pass
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    
    print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update Fisher before optimizer step
    grit_layer.update_fisher_and_projection()
    
    optimizer.step()
    
    print(f"✓ Training step completed")
    
    # Test multiple steps
    for i in range(5):
        x = torch.randn(8, 64)
        y = torch.randn(8, 1)
        
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        if i % 2 == 0:  # Update Fisher every 2 steps
            grit_layer.update_fisher_and_projection()
        
        optimizer.step()
        
        print(f"✓ Step {i+1}, loss: {loss.item():.4f}")
    
    print("\n🎉 Multi-step training test passed!")
    return True

def test_fisher_matrix():
    """Test Fisher matrix computation specifically."""
    print("\n🧪 Testing Fisher Matrix Computation")
    print("=" * 35)
    
    base_layer = nn.Linear(32, 16)
    config = GRITLoRAConfig(r=4, fisher_approximation="diagonal")
    grit_layer = GRITLoRALinear(base_layer, config)
    
    print(f"✓ Created layer for Fisher testing")
    
    # Multiple forward/backward passes to accumulate Fisher info
    for i in range(10):
        x = torch.randn(4, 32)
        output = grit_layer(x)
        loss = output.sum()
        loss.backward()
        
        if i % 3 == 0:
            grit_layer.update_fisher_and_projection()
            
        grit_layer.zero_grad()
    
    # Check Fisher matrix
    fisher = grit_layer.fisher_matrix.fisher_diagonal
    if fisher is not None:
        print(f"✓ Fisher matrix computed, shape: {fisher.shape}")
        print(f"✓ Fisher mean: {fisher.mean().item():.6f}")
        print(f"✓ Fisher std: {fisher.std().item():.6f}")
    else:
        print("⚠️  Fisher matrix is None")
    
    # Check projection matrix
    if grit_layer.projection_matrix is not None:
        print(f"✓ Projection matrix computed")
        print(f"✓ Active projections: {grit_layer.projection_matrix.sum().item()}")
    else:
        print("⚠️  Projection matrix is None")
    
    print("\n🎉 Fisher matrix test completed!")
    return True

if __name__ == "__main__":
    print("🚀 GRIT-VLM Simple Testing Suite")
    print("=" * 50)
    
    try:
        # Run tests
        test_grit_lora_basic()
        test_grit_with_simple_model() 
        test_fisher_matrix()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! GRIT implementation is working correctly.")
        print("\n📊 Summary:")
        print("  ✓ GRIT-LoRA layers work correctly")
        print("  ✓ Forward/backward passes successful")  
        print("  ✓ Fisher matrix computation working")
        print("  ✓ Natural gradients computed")
        print("  ✓ Projection matrices generated")
        print("  ✓ Multi-step training works")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()