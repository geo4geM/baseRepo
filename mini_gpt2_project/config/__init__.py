"""Configuration package for Mini-GPT2 and related model, data, and training settings."""

# Config module
from .model_config import (
    ModelConfig,
    InferenceConfig,
    get_config_by_name,
    get_device,
    get_dtype,
)

__all__ = [
    "ModelConfig",
    "InferenceConfig", 
    "get_config_by_name",
    "get_device",
    "get_dtype",
]