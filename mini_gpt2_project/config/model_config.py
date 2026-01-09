"""Configuration definitions for Mini-GPT2 and BDH model architectures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping


@dataclass
class ModelConfig:
    """Configuration for Mini-GPT2-style and BDH models (~30M parameters).

    This dataclass encapsulates the architectural hyperparameters for the
    Mini-GPT2-style Transformer and compatible baseline BDH models. With the
    default settings, the resulting Transformer has approximately 30 million
    trainable parameters, suitable for lightweight experimentation and
    deployment.

    Attributes:
        vocab_size: Size of the token vocabulary.
        n_embd: Dimensionality of token embeddings and hidden states.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads per block.
        max_seq_len: Maximum supported sequence length.
        num_classes: Number of output classes for classification.
    """

    vocab_size: int = 50257
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    max_seq_len: int = 512
    num_classes: int = 2

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

