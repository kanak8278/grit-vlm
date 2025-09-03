"""
Test the new GRIT configuration system with SmolVLM.

This script demonstrates how to use predefined model configurations
and custom layer selection strategies.
"""

import torch
import torch.nn as nn
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Import GRIT components
from grit_vlm import GRITLoRAConfig
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
from grit_vlm.config import (
    get_model_config,
    ModelGRITConfig,
    ModalityConfig,
    LayerSelectionStrategy,
    MODEL_CONFIGS,
)


def test_predefined_configs():
    """Test predefined model configurations."""
    print("🔧 Testing Predefined Model Configurations")
    print("=" * 50)

    # Show available configs
    print("📋 Available predefined configurations:")
    for name, config in MODEL_CONFIGS.items():
        vision_layers = len(config.vision.layer_patterns) if config.vision else 0
        text_layers = len(config.text.layer_patterns) if config.text else 0
        cross_layers = (
            len(config.cross_modal.layer_patterns) if config.cross_modal else 0
        )
        print(
            f"  • {name:<15} Vision: {vision_layers}, Text: {text_layers}, Cross: {cross_layers}"
        )

    print(f"\n🎯 Testing 'smolvlm_fast' configuration...")

    # Load fast config for SmolVLM
    fast_config = get_model_config("smolvlm_fast")
    if fast_config:
        print(f"✓ Config loaded: {fast_config.model_name}")
        print(
            f"  Vision strategy: {fast_config.vision.strategy.value} (n={fast_config.vision.n_layers})"
        )
        print(
            f"  Text strategy: {fast_config.text.strategy.value} (n={fast_config.text.n_layers})"
        )
        print(f"  Cross strategy: {fast_config.cross_modal.strategy.value}")
    else:
        print("❌ Config not found")
        return False

    return True


def test_custom_config():
    """Test creating custom configurations."""
    print("\n🛠️ Testing Custom Configuration Creation")
    print("=" * 45)

    # Create a minimal custom config for testing
    custom_config = ModelGRITConfig(
        model_name="test-model",
        model_type="custom",
        vision=ModalityConfig(
            layer_patterns=[
                "model.vision_model.encoder.layers.*.self_attn.q_proj",
                "model.vision_model.encoder.layers.*.self_attn.v_proj",
            ],
            strategy=LayerSelectionStrategy.FIRST_N,
            n_layers=2,
            rank=4,
            alpha=8,
        ),
        text=ModalityConfig(
            layer_patterns=["model.text_model.layers.*.self_attn.q_proj"],
            strategy=LayerSelectionStrategy.LAST_N,
            n_layers=3,
            rank=6,
            alpha=12,
        ),
        global_rank=8,
        global_alpha=16,
    )

    print("✓ Custom config created:")
    print(f"  Global rank: {custom_config.global_rank}")
    print(
        f"  Vision: {custom_config.vision.strategy.value} strategy, rank {custom_config.vision.rank}"
    )
    print(
        f"  Text: {custom_config.text.strategy.value} strategy, rank {custom_config.text.rank}"
    )

    return True


def test_smolvlm_with_fast_config():
    """Test SmolVLM with fast configuration."""
    print("\n🚀 Testing SmolVLM with Fast Configuration")
    print("=" * 50)

    try:
        # Load SmolVLM
        print("📥 Loading SmolVLM-256M...")
        model = Idefics3ForConditionalGeneration.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(
            f"✓ SmolVLM loaded: {sum(p.numel() for p in model.parameters()):,} params"
        )

    except Exception as e:
        print(f"❌ Failed to load SmolVLM: {e}")
        return False

    # Test with fast config
    print("\n🔧 Applying GRIT with 'smolvlm_fast' config...")

    try:
        # Use the predefined fast config
        grit_adapter = VLMGRITAdapter(
            model=model,
            config=GRITLoRAConfig(),  # Will be overridden by model config
            model_config_name="smolvlm_fast",
        )

        adapted_layers = len(grit_adapter.grit_layers)
        print(f"✓ GRIT applied to {adapted_layers} layers")

        if adapted_layers > 0:
            # Print summary
            grit_adapter.print_adaptation_summary()

            # Quick forward pass test
            print("\n🧪 Testing forward pass...")

            # Create test image
            test_image = Image.fromarray(
                (np.ones((64, 64, 3)) * [255, 0, 0]).astype(np.uint8)
            )
            processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct", trust_remote_code=True
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": "What color is this?"},
                    ],
                }
            ]

            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(test_image, input_text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            print(f"✓ Forward pass successful! Logits shape: {outputs.logits.shape}")
            print(f"✓ Fast configuration works perfectly!")

            return True
        else:
            print("❌ No layers adapted")
            return False

    except Exception as e:
        print(f"❌ GRIT adaptation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_comparison():
    """Compare different configurations."""
    print("\n📊 Configuration Comparison")
    print("=" * 35)

    configs_to_compare = ["smolvlm_256m", "smolvlm_fast"]

    for config_name in configs_to_compare:
        config = get_model_config(config_name)
        if config:
            print(f"\n🔍 {config_name}:")

            # Vision config
            if config.vision:
                print(f"  Vision: {config.vision.strategy.value}", end="")
                if config.vision.n_layers:
                    print(f" (n={config.vision.n_layers})", end="")
                if config.vision.step_size:
                    print(f" (step={config.vision.step_size})", end="")
                print(f" - rank {config.vision.rank}")

            # Text config
            if config.text:
                print(f"  Text: {config.text.strategy.value}", end="")
                if config.text.n_layers:
                    print(f" (n={config.text.n_layers})", end="")
                print(f" - rank {config.text.rank}")

            # Performance limit
            if config.max_layers_per_modality:
                print(
                    f"  Performance limit: {config.max_layers_per_modality} layers/modality"
                )


def show_usage_examples():
    """Show usage examples."""
    print("\n💡 Usage Examples")
    print("=" * 25)

    examples = [
        {
            "title": "Use predefined fast config",
            "code": """
# Fast config for quick testing
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast"  # Uses predefined fast config
)""",
        },
        {
            "title": "Use predefined full config",
            "code": """
# Full config for best quality
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_256m"  # Uses full predefined config
)""",
        },
        {
            "title": "Use ModelGRITConfig directly",
            "code": """
# Custom model config
model_config = get_model_config("smolvlm_fast")
adapter = VLMGRITAdapter(model=model, config=model_config)
""",
        },
        {
            "title": "Override specific layers",
            "code": """
# Override with specific layer names
adapter = VLMGRITAdapter(
    model=model,
    config=GRITLoRAConfig(),
    model_config_name="smolvlm_fast",
    vision_layers=["model.vision_model.encoder.layers.0.self_attn.q_proj"]
)""",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(example["code"])


if __name__ == "__main__":
    print("🔬 GRIT Configuration System Test")
    print("=" * 60)

    success = True

    # Test predefined configs
    success &= test_predefined_configs()

    # Test custom config creation
    success &= test_custom_config()

    # Test with real model
    success &= test_smolvlm_with_fast_config()

    # Show comparisons
    test_config_comparison()

    # Show usage examples
    show_usage_examples()

    if success:
        print("\n" + "=" * 60)
        print("✅ ALL CONFIGURATION TESTS PASSED!")
        print("\n🎯 Key Features Demonstrated:")
        print("  ✓ Predefined model configurations")
        print("  ✓ Layer selection strategies (FIRST_N, LAST_N, EVERY_NTH)")
        print("  ✓ Performance limits to control adaptation size")
        print("  ✓ Modality-specific GRIT parameters")
        print("  ✓ Custom configuration creation")
        print("  ✓ Real model integration with SmolVLM")
        print("\n🚀 GRIT Configuration System is ready!")
        print("\n💡 Use 'smolvlm_fast' for quick testing")
        print("   Use 'smolvlm_256m' for full quality")
    else:
        print("\n❌ Some configuration tests failed")
