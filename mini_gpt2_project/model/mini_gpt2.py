"""Mini-GPT2 Transformer architecture for backstory consistency classification."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from ..config.model_config import ModelConfig
from .base_model import BaseModel


class MiniGPT2(BaseModel):
    """Mini-GPT2 model for backstory consistency classification.

    This model wraps a ``GPT2Model`` backbone followed by a lightweight
    classification head. Architectural hyperparameters are provided via
    a ``ModelConfig`` instance so that the Mini-GPT2 backbone can be
    swapped with alternative architectures (e.g., BDH) without changing
    the training or inference pipelines.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Mini-GPT2 model.

        Args:
            config: Model configuration specifying architecture details such
                as embedding size, number of layers, and number of classes.
        """
        super().__init__()

        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            resid_pdrop=0.1,
        )

        self.gpt = GPT2Model(gpt_config)
        self.classifier = nn.Linear(config.n_embd, config.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run a forward pass through the Mini-GPT2 classifier.

        This method computes:

        - Logits for backstory consistency classification.
        - Optional cross-entropy loss when labels are provided.
        - Velocity: mean L2 norm of differences between adjacent token states.
        - Sparsity: fraction of elements in the sequence output that are zero.

        Args:
            input_ids: Token ids of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
            labels: Optional target labels of shape ``(batch_size,)``.

        Returns:
            A dictionary with:

            - ``"loss"``: Cross-entropy loss tensor or ``None``.
            - ``"logits"``: Class logits of shape ``(batch_size, num_classes)``.
            - ``"velocity"``: Scalar tensor with mean L2 velocity.
            - ``"sparsity"``: Scalar tensor with fraction of zero elements.
        """
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        sequence_output: torch.Tensor = outputs.last_hidden_state  # (B, T, H)

        # Velocity: L2 norm of differences between adjacent token states.
        if sequence_output.size(1) > 1:
            diffs = sequence_output[:, 1:, :] - sequence_output[:, :-1, :]
            l2_norms = torch.norm(diffs, dim=-1)  # (B, T-1)
            velocity = l2_norms.mean()
        else:
            velocity = torch.zeros((), dtype=sequence_output.dtype, device=sequence_output.device)

        # Sparsity: fraction of elements that are exactly zero.
        sparsity = (sequence_output == 0).float().mean()

        batch_size, seq_len, _ = sequence_output.shape
        if attention_mask is not None:
            # Use the last non-masked token as the representation.
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            last_indices = lengths - 1  # (B,)
            last_hidden = sequence_output[
                torch.arange(batch_size, device=sequence_output.device), last_indices
            ]
        else:
            last_hidden = sequence_output[:, -1, :]

        logits = self.classifier(last_hidden)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "velocity": velocity,
            "sparsity": sparsity,
        }

