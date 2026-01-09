"""High-level prediction interface for running Mini-GPT2 and baseline models on new data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..config.model_config import InferenceConfig, ModelConfig
from ..model.mini_gpt2 import MiniGPT2
from ..utils.data_loader import ByteTokenizer


class MiniGPT2Wrapper:
    """Wrapper around `MiniGPT2` for novel-level reasoning and analysis.

    This class mimics the senior `BDHReasoningWrapper` API while using the
    Mini-GPT2 backbone. It exposes utilities to compute aggregate hidden
    states for entire novels and backstories, and to compare them via a
    simple velocity-like distance metric.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        device: torch.device,
    ) -> None:
        """Initialize the MiniGPT2Wrapper.

        Args:
            model_config: Configuration for the Mini-GPT2 architecture.
            inference_config: Configuration for inference-time behavior such
                as chunk size and damping.
            device: Torch device on which the model will be placed.
        """
        self.model = MiniGPT2(model_config).to(device)
        self.model.eval()

        self.inference_config = inference_config
        self.device = device

        self.tokenizer = ByteTokenizer()

    def _compute_mean_hidden_state_from_tokens(
        self,
        token_chunks: list[list[int]],
    ) -> Tensor:
        """Compute the mean hidden state from a list of token chunks.

        Args:
            token_chunks: List of token id sequences, each representing a text
                chunk.

        Returns:
            A tensor representing the mean hidden state over all tokens in all
            chunks.
        """
        hidden_means: list[Tensor] = []

        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk:
                    continue
                input_ids = torch.tensor(chunk, dtype=torch.long, device=self.device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, device=self.device)

                outputs = self.model.gpt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                sequence_output: Tensor = outputs.last_hidden_state  # (1, T, H)

                # Mean over sequence dimension (tokens), keep batch dimension.
                chunk_mean = sequence_output.mean(dim=1).squeeze(0)  # (H,)
                hidden_means.append(chunk_mean)

        if not hidden_means:
            # Return a zero vector if no tokens were processed.
            # The dimension will be inferred from the model's hidden size.
            hidden_size = self.model.gpt.config.n_embd
            return torch.zeros(hidden_size, device=self.device)

        stacked = torch.stack(hidden_means, dim=0)  # (num_chunks, H)
        return stacked.mean(dim=0)  # (H,)

    def compute_novel_state(
        self,
        book_path: Union[str, Path],
        verbose: bool = False,
    ) -> Tensor:
        """Compute the mean hidden state representation for an entire novel.

        Args:
            book_path: Path to the book text file.
            verbose: Whether to print progress information.

        Returns:
            A tensor on CPU representing the mean hidden state of the novel.
        """
        path = Path(book_path)
        if verbose:
            print(f"Loading book from: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        chunk_size = self.inference_config.chunk_size

        token_chunks = self.tokenizer.chunk_text(text, chunk_size)
        if verbose:
            print(f"Tokenized into {len(token_chunks)} chunks (chunk_size={chunk_size}).")

        mean_state = self._compute_mean_hidden_state_from_tokens(token_chunks)
        return mean_state.detach().cpu()

    def prime_with_backstory(
        self,
        text: str,
        verbose: bool = False,
    ) -> Tuple[Tensor, Optional[None]]:
        """Compute a mean hidden state representation for a backstory.

        Args:
            text: Backstory text to encode.
            verbose: Whether to print progress information.

        Returns:
            A tuple of ``(mean_hidden_state, None)`` to mirror the senior API.
        """
        if verbose:
            print("Tokenizing backstory text.")

        chunk_size = self.inference_config.chunk_size
        token_chunks = self.tokenizer.chunk_text(text, chunk_size)

        mean_state = self._compute_mean_hidden_state_from_tokens(token_chunks)
        return mean_state.detach().cpu(), None

    def compute_velocity_from_states(
        self,
        state_backstory: Tensor,
        state_novel: Tensor,
    ) -> float:
        """Compute a velocity-like distance between backstory and novel states.

        This is defined as the L2 norm between the two hidden-state vectors.

        Args:
            state_backstory: Mean hidden state tensor for the backstory.
            state_novel: Mean hidden state tensor for the novel.

        Returns:
            The L2 distance as a Python float.
        """
        # Ensure both states are on the same device and dtype.
        sb = state_backstory.to(self.device)
        sn = state_novel.to(self.device)

        distance = torch.norm(sb - sn)
        return float(distance.item())

