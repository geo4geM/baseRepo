"""Abstract base model definitions for modular Mini-GPT2 and BDH architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for Mini-GPT2-style and BDH models.

    This class defines the minimal interface required by the training and
    inference pipelines. All concrete architectures (e.g., Mini-GPT2,
    BDH baseline) must implement the ``forward`` method with a consistent
    signature and return format so that models can be swapped via config
    without changing downstream code.

    Subclasses should implement the forward pass for backstory classification
    tasks, returning both the loss (when labels are provided) and raw logits.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run a forward pass through the model.

        Args:
            input_ids: Token ids of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``,
                where non-zero values indicate tokens to attend to.
            labels: Optional target labels of shape ``(batch_size,)`` or
                ``(batch_size, num_classes)`` depending on the loss design.

        Returns:
            A dictionary containing:

            - ``"loss"``: A scalar tensor with the training loss (or ``None``
              when ``labels`` is not provided or the model is in pure
              inference mode).
            - ``"logits"``: A tensor of shape
              ``(batch_size, num_classes)`` with unnormalized class scores.
        """
        raise NotImplementedError

