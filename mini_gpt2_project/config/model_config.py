"""Configuration definitions for Recurrent BDH architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping

import torch


@dataclass
class ModelConfig:
    """Configuration for Recurrent BDH and Mini-GPT2 models (~30M parameters).

    This dataclass encapsulates the architectural hyperparameters for the
    models. With the default settings, the resulting model has approximately
    30 million trainable parameters, suitable for lightweight experimentation
    and deployment.

    Attributes:
        model_type: Architecture to use ("bdh", "minigpt2", "gemini", or "gemma").
        vocab_size: Size of the token vocabulary.
        n_embd: Dimensionality of token embeddings and hidden states.
        n_layer: Number of layers (blocks).
        n_head: Number of attention heads per block.
        max_seq_len: Maximum supported sequence length.
        num_classes: Number of output classes for classification.
        mlp_internal_dim_multiplier: Multiplier for the internal dimension (N)
            relative to the embedding dimension.
        dropout: Dropout probability.
        tokenizer_type: Type of tokenizer to use ("byte" or "bpe").
    """

    # --- MODEL SWITCH ---
    model_type: str = "gemini"  # Options: "bdh", "minigpt2", "gemini", "gemma"
    # --------------------

    vocab_size: int = 5000
    n_embd: int = 704
    n_layer: int = 4
    n_head: int = 11
    max_seq_len: int = 1024
    num_classes: int = 2
    
    # BDH specific parameters
    mlp_internal_dim_multiplier: int = 4
    dropout: float = 0.1

    tokenizer_type: str = "bpe"  # "byte" or "bpe"

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "ModelConfig":
        """Create a `ModelConfig` instance from a configuration mapping.

        Unknown keys in the input mapping are ignored to allow forward
        compatibility with extended configuration dictionaries.

        Args:
            config_dict: Mapping of configuration field names to values.

        Returns:
            An initialized `ModelConfig` instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {
            key: value for key, value in config_dict.items() if key in valid_fields
        }
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this `ModelConfig` instance to a plain dictionary.

        Returns:
            A dictionary representation of the configuration.
        """
        return asdict(self)


@dataclass
class InferenceConfig:
    """Configuration for inference-time behavior.

    Attributes:
        chunk_size: Maximum sequence chunk size processed at once.
        damping: Stabilization factor for iterative or streaming decoding.
    """

    chunk_size: int = 1024
    damping: float = 0.1


def get_device() -> torch.device:
    """Get the default torch device for model execution.

    Returns:
        A CUDA device when available, otherwise the CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype() -> torch.dtype:
    """Get the default floating point data type for model parameters.

    Returns:
        The torch floating point dtype used for model weights and activations.
    """
    return torch.float32


def get_config_by_name(name: str) -> ModelConfig:
    """Retrieve a predefined `ModelConfig` instance by name.

    The name argument is currently ignored and a default `ModelConfig` is
    returned. This function exists to match the senior API and can be
    extended later to support multiple named configurations.

    Args:
        name: Identifier for a particular configuration preset.

    Returns:
        A default `ModelConfig` instance.
    """
    _ = name
    return ModelConfig()