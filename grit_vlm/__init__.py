"""
GRIT: Parameter-Efficient Finetuning Method for LLMs and VLMs

Core implementation of curvature-aware LoRA using Fisher Information Matrix
for natural gradient optimization.
"""

from .core.fisher_info import FisherInformationMatrix
from .core.grit_lora import GRITLoRALayer, GRITLoRAConfig
from .models.vlm_adapter import VLMGRITAdapter

__version__ = "0.1.0"
__author__ = "GRIT-VLM Implementation"

__all__ = [
    "FisherInformationMatrix",
    "GRITLoRALayer", 
    "GRITLoRAConfig",
    "VLMGRITAdapter"
]