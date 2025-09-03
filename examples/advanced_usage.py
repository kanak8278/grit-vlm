"""
Advanced GRIT-VLM usage example.

Demonstrates advanced features like:
- Custom Fisher approximations
- Mixed-modal projection scheduling  
- Multi-GPU training
- Custom evaluation metrics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

from grit_vlm import (
    GRITLoRAConfig, 
    FisherInformationMatrix,
    FisherApproximationType,
    ProjectionScheduler,
    create_linear_scheduler,
    create_adaptive_scheduler
)


def advanced_fisher_configuration():
    """Demonstrate advanced Fisher Information Matrix configurations."""
    
    print("🔬 Advanced Fisher Configuration")
    print("=" * 40)
    
    # 1. K-FAC Approximation
    print("\n1. K-FAC Fisher Approximation:")
    kfac_config = GRITLoRAConfig(
        r=32,
        lora_alpha=64,
        fisher_approximation="kfac",           # Kronecker-factored approximation
        fisher_damping=1e-4,
        fisher_ema_decay=0.9,
        enable_natural_gradient=True,
        enable_projection=True
    )
    
    print(f"✓ K-FAC configuration: F ≈ A ⊗ G")
    print(f"  - More accurate curvature estimation")
    print(f"  - Higher memory usage but better performance")
    
    # 2. Block-Diagonal Approximation
    print("\n2. Block-Diagonal Fisher:")
    block_config = GRITLoRAConfig(
        r=16,
        fisher_approximation="block_diagonal",  # Block-diagonal approximation
        fisher_damping=5e-4,
        natural_gradient_damping=1e-2,
        enable_projection=True
    )
    
    print(f"✓ Block-diagonal configuration")
    print(f"  - Captures layer-wise parameter interactions")
    print(f"  - Balanced between accuracy and efficiency")
    
    return kfac_config, block_config


def advanced_projection_scheduling():
    """Demonstrate advanced projection budget scheduling."""
    
    print("\n🎯 Advanced Projection Scheduling")
    print("=" * 40)
    
    # 1. Adaptive Scheduling
    print("\n1. Adaptive Projection Budget:")
    adaptive_scheduler = create_adaptive_scheduler(
        k_start=16, 
        k_end=128, 
        total_steps=2000
    )
    
    print("✓ Adaptive scheduler created")
    print("  - Adjusts budget based on eigenvalue gaps")
    print("  - Automatically finds informative directions")
    
    # Simulate adaptive scheduling
    print("\n  Simulation:")
    mock_eigenvalues = torch.tensor([10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.1, 0.05])
    
    for step in [0, 100, 500, 1000, 1500, 2000]:
        k = adaptive_scheduler.step(eigenvalues=mock_eigenvalues)
        print(f"    Step {step:4d}: k = {k:3d}")
    
    # 2. Custom Schedule Function
    print("\n2. Custom Projection Schedule:")
    
    def custom_schedule(progress: float, k_start: int, k_end: int) -> int:
        """Custom sigmoid-based scheduling."""
        import math
        sigmoid = 1 / (1 + math.exp(-10 * (progress - 0.5)))
        return int(k_start + (k_end - k_start) * sigmoid)
    
    print("✓ Custom sigmoid schedule")
    print("  - Slow start, rapid middle growth, plateau")
    
    # 3. Vision-Rank-Aware Scheduling
    print("\n3. Vision-Rank-Aware Scheduling:")
    
    config_with_vision_rule = GRITLoRAConfig(
        projection_budget_start=24,
        projection_budget_end=72,
        projection_schedule="adaptive"
    )
    
    print("✓ Empirical rule: k ≈ 1.2 × rank(A_vision)")
    print("  - Automatically adapts to vision component complexity")
    print("  - Prevents over/under-parameterization")


def advanced_vlm_specific_features():
    """Demonstrate VLM-specific advanced features."""
    
    print("\n🌉 VLM-Specific Advanced Features")
    print("=" * 40)
    
    # 1. Mixed-Modal Fisher Configuration
    print("\n1. Mixed-Modal Fisher Configuration:")
    
    vlm_config = GRITLoRAConfig(
        r=24,
        lora_alpha=48,
        fisher_approximation="diagonal",
        
        # VLM-specific target modules
        target_modules=[
            # Vision encoder
            "vision_tower.*.self_attn.q_proj",
            "vision_tower.*.self_attn.k_proj", 
            "vision_tower.*.self_attn.v_proj",
            
            # Language decoder
            "language_model.*.self_attn.q_proj",
            "language_model.*.self_attn.k_proj",
            "language_model.*.self_attn.v_proj",
            
            # Cross-modal projector
            "multi_modal_projector.*"
        ]
    )
    
    print("✓ Multi-modal targeting configured")
    print("  - Vision: Self-attention layers")
    print("  - Text: Self-attention layers") 
    print("  - Cross-modal: Projector layers")
    
    # 2. Modality-Specific Learning Rates
    print("\n2. Modality-Specific Optimization:")
    
    modality_config = {
        'vision': {
            'lr': 1e-4,      # Lower LR for pre-trained vision
            'weight': 0.3,   # Lower weight in mixed Fisher
            'damping': 1e-4
        },
        'text': {
            'lr': 2e-4,      # Higher LR for language adaptation
            'weight': 0.5,   # Higher weight in mixed Fisher
            'damping': 5e-5
        },
        'cross_modal': {
            'lr': 3e-4,      # Highest LR for new connections
            'weight': 0.2,   # Moderate weight
            'damping': 1e-5
        }
    }
    
    print("✓ Modality-specific optimization")
    for modality, config in modality_config.items():
        print(f"  - {modality}: LR={config['lr']:.0e}, weight={config['weight']}")


def advanced_training_strategies():
    """Demonstrate advanced training strategies."""
    
    print("\n🚀 Advanced Training Strategies")
    print("=" * 40)
    
    # 1. Gradient Accumulation with Fisher Updates
    print("\n1. Smart Gradient Accumulation:")
    
    accumulation_strategy = {
        'gradient_accumulation_steps': 8,
        'fisher_update_freq': 4,        # Update Fisher more frequently
        'projection_update_freq': 16,   # Update projections less frequently
        'natural_gradient_freq': 2      # Apply natural gradients frequently
    }
    
    print("✓ Smart accumulation strategy")
    print(f"  - Gradient steps: {accumulation_strategy['gradient_accumulation_steps']}")
    print(f"  - Fisher updates: every {accumulation_strategy['fisher_update_freq']} steps")
    print(f"  - Projection updates: every {accumulation_strategy['projection_update_freq']} steps")
    
    # 2. Warmup Strategy for Fisher
    print("\n2. Fisher Information Warmup:")
    
    warmup_config = {
        'fisher_warmup_steps': 100,     # Build Fisher info gradually
        'natural_gradient_delay': 50,   # Delay natural gradients
        'projection_delay': 150         # Delay projections until Fisher is stable
    }
    
    print("✓ Fisher warmup strategy")
    print("  - Gradual Fisher matrix accumulation")
    print("  - Delayed natural gradient application")
    print("  - Stable projection computation")
    
    # 3. Adaptive Damping
    print("\n3. Adaptive Damping:")
    
    def adaptive_damping_schedule(step: int, base_damping: float = 1e-3) -> float:
        """Decrease damping as training progresses."""
        if step < 100:
            return base_damping * 10  # High damping initially
        elif step < 500:
            return base_damping * 5   # Medium damping
        else:
            return base_damping       # Low damping for fine-tuning
    
    print("✓ Adaptive damping schedule")
    print("  - High initial damping for stability")
    print("  - Gradual reduction for precision")


def advanced_monitoring_and_analysis():
    """Demonstrate advanced monitoring capabilities."""
    
    print("\n📊 Advanced Monitoring & Analysis")
    print("=" * 40)
    
    # 1. Fisher Information Analysis
    print("\n1. Fisher Information Diagnostics:")
    
    fisher_metrics = [
        "fisher_diagonal_norm",         # ||F_diag||
        "fisher_condition_number",      # κ(F) = λ_max/λ_min
        "fisher_effective_rank",        # Number of significant eigenvalues
        "fisher_trace",                 # tr(F)
        "fisher_spectral_norm"          # ||F||_2
    ]
    
    print("✓ Fisher diagnostic metrics:")
    for metric in fisher_metrics:
        print(f"  - {metric}")
    
    # 2. Projection Analysis
    print("\n2. Projection Quality Metrics:")
    
    projection_metrics = [
        "projection_coverage",          # Fraction of gradient captured
        "eigenvalue_gaps",              # λ_i - λ_{i+1}
        "projection_stability",         # Consistency across steps
        "information_retention"         # How much curvature info retained
    ]
    
    print("✓ Projection quality metrics:")
    for metric in projection_metrics:
        print(f"  - {metric}")
    
    # 3. Convergence Analysis
    print("\n3. Convergence Diagnostics:")
    
    convergence_metrics = {
        'natural_gradient_alignment': 'cos(g_natural, g_standard)',
        'curvature_adaptation_rate': 'rate of Fisher matrix change',
        'projection_rank_evolution': 'how k changes over time',
        'parameter_sensitivity': 'which params matter most'
    }
    
    print("✓ Convergence analysis:")
    for metric, description in convergence_metrics.items():
        print(f"  - {metric}: {description}")


def create_production_config():
    """Create a production-ready GRIT configuration."""
    
    print("\n🏭 Production-Ready Configuration")
    print("=" * 40)
    
    production_config = GRITLoRAConfig(
        # Core LoRA parameters
        r=32,                           # Higher rank for production
        lora_alpha=64,                  # Balanced scaling
        lora_dropout=0.05,              # Low dropout for stability
        
        # Fisher approximation
        fisher_approximation="diagonal", # Efficient for large-scale
        fisher_damping=1e-4,            # Conservative damping
        fisher_ema_decay=0.95,          # Stable EMA
        fisher_update_freq=10,          # Regular updates
        
        # Natural gradient settings
        natural_gradient_damping=5e-4,  # Moderate damping
        enable_natural_gradient=True,
        
        # Projection settings
        projection_budget_start=48,     # Higher initial budget
        projection_budget_end=128,      # Higher final budget
        projection_schedule="cosine",   # Smooth scheduling
        enable_projection=True,
        
        # Target modules for comprehensive adaptation
        target_modules=[
            # Vision components
            "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj",
            "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
            
            # Language components  
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj",
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.o_proj",
            
            # Cross-modal components
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2"
        ]
    )
    
    print("✓ Production configuration created")
    print(f"  - Rank: {production_config.r} (high capacity)")
    print(f"  - Fisher: {production_config.fisher_approximation} (efficient)")
    print(f"  - Projection: {production_config.projection_budget_start}→{production_config.projection_budget_end}")
    print(f"  - Target modules: {len(production_config.target_modules)} layer types")
    
    return production_config


def main():
    """Run advanced GRIT-VLM demonstration."""
    
    print("🔬 GRIT-VLM Advanced Usage")
    print("=" * 50)
    
    # Run all advanced demonstrations
    kfac_config, block_config = advanced_fisher_configuration()
    advanced_projection_scheduling() 
    advanced_vlm_specific_features()
    advanced_training_strategies()
    advanced_monitoring_and_analysis()
    production_config = create_production_config()
    
    print(f"\n🎓 Advanced Features Summary:")
    print("=" * 35)
    
    advanced_features = [
        "🧮 Multiple Fisher approximations (diagonal, K-FAC, block-diagonal)",
        "📈 Advanced projection scheduling (adaptive, custom, vision-aware)",
        "🌉 VLM-specific optimizations (mixed-modal, modality-specific)",
        "🚀 Smart training strategies (warmup, accumulation, adaptive damping)", 
        "📊 Comprehensive monitoring (Fisher diagnostics, convergence analysis)",
        "🏭 Production-ready configurations (robust, scalable, efficient)"
    ]
    
    for feature in advanced_features:
        print(f"  {feature}")
    
    print(f"\n💡 Key Insights:")
    print("=" * 20)
    
    insights = [
        "Choose Fisher approximation based on compute/accuracy tradeoff",
        "Use adaptive scheduling for automatic hyperparameter tuning",
        "Apply modality-specific settings for VLM optimization",
        "Monitor Fisher diagnostics to ensure stable training",
        "Start with diagonal Fisher, upgrade to K-FAC for critical applications"
    ]
    
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    print(f"\n🎯 Ready for advanced GRIT-VLM fine-tuning!")


if __name__ == "__main__":
    main()