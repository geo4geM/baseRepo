"""Model package containing Mini-GPT2 and alternative baseline architectures."""

# Model module
from .base_model import BaseModel
from .mini_gpt2 import MiniGPT2

__all__ = [
    "BaseModel",
    "MiniGPT2",
]