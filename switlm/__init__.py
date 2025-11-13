"""
SwitLM - SimpleWitty Language Model Training Library
Create and train custom LLMs with minimal code
"""

from .trainer import SwitLMTrainer
from .config import ModelConfig
from .generator import TextGenerator

__version__ = "0.1.0"
__all__ = ["SwitLMTrainer", "ModelConfig", "TextGenerator"]