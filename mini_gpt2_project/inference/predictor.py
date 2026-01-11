"""High-level prediction interface for running Narrative Consistency models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..config.model_config import InferenceConfig, ModelConfig
from ..model.mini_gpt2 import MiniGPT2
from ..model.bdh_recurrent import RecurrentBDH, RecurrentState
from ..model.gemini_pro import GeminiPro
from ..model.gemma_model import GemmaModel
from ..utils.data_loader import get_tokenizer


class NarrativePredictor:
    """Unified wrapper for Narrative Consistency models (GPT-2, BDH, Gemini, or Gemma).

    This class abstracts the underlying model differences:
    - GPT-2: Uses mean hidden states for representation.
    - BDH: Uses accumulated ρ-matrix (associative memory) for representation.
    - Gemini: Uses API-based embeddings for representation.
    - Gemma: Uses Hugging Face API-based embeddings for representation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        device: torch.device,
        model: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Predictor.

        Args:
            model_config: Configuration for the architecture.
            inference_config: Configuration for inference-time behavior.
            device: Torch device.
            model: Optional pre-trained model instance.
            lm_head: Optional external language model head (for GPT-2).
            api_key: Optional API key for Gemini model.
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = device
        
        # For Gemini and Gemma, we don't need a tokenizer
        if self.model_config.model_type not in ["gemini", "gemma"]:
            self.tokenizer = get_tokenizer(self.model_config)
        else:
            self.tokenizer = None

        # 1. Initialize Model if not provided
        if model is not None:
            if self.model_config.model_type in ["gemini", "gemma"]:
                self.model = model  # API models don't use .to(device)
            else:
                self.model = model.to(device)
        else:
            if self.model_config.model_type == "bdh":
                print("Initializing RecurrentBDH for inference...")
                self.model = RecurrentBDH(model_config).to(device)
            elif self.model_config.model_type == "gemini":
                print("Initializing Gemini Pro for inference...")
                self.model = GeminiPro(model_config, api_key=api_key)
            elif self.model_config.model_type == "gemma":
                print("Initializing Gemma Model for inference...")
                self.model = GemmaModel(model_config, api_key=api_key)
            else:
                print("Initializing MiniGPT2 for inference...")
                self.model = MiniGPT2(model_config).to(device)
        
        if self.model_config.model_type not in ["gemini", "gemma"]:
            self.model.eval()

        # 2. Handle LM Head (not needed for Gemini/Gemma)
        if self.model_config.model_type == "bdh":
            # BDH has internal head, exposed as property or attribute
            self.lm_head = getattr(self.model, 'lm_head', None)
        elif self.model_config.model_type in ["gemini", "gemma"]:
            # API models don't use LM head
            self.lm_head = None
        else:
            # GPT-2 needs external or internal head management
            if lm_head is not None:
                self.lm_head = lm_head.to(device)
                self.lm_head.eval()
            else:
                # Fallback: try to find classifier or create dummy if needed for pure representation
                # For classification tasks, MiniGPT2 has self.classifier.
                # For LM tasks, we might need a Linear layer.
                self.lm_head = getattr(self.model, 'classifier', None)

    def compute_novel_state(
        self,
        book_path: Union[str, Path],
        verbose: bool = False,
    ) -> Tensor:
        """Compute the state representation for an entire novel.
        
        Dispatches to specific logic based on model architecture.
        """
        path = Path(book_path)
        if verbose:
            print(f"Loading book from: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        
        # Gemini and Gemma don't need tokenization - handle them first
        if self.model_config.model_type == "gemini":
            return self._compute_gemini_state(text)
        elif self.model_config.model_type == "gemma":
            return self._compute_gemma_state(text)
        
        # For other models, tokenize the text
        chunk_size = self.inference_config.chunk_size
        token_chunks = self.tokenizer.chunk_text(text, chunk_size)

        if verbose:
            print(f"Tokenized into {len(token_chunks)} chunks.")

        if self.model_config.model_type == "bdh":
            return self._compute_bdh_state(token_chunks)
        else:
            return self._compute_gpt2_state(token_chunks)

    def prime_with_backstory(
        self,
        text: str,
        verbose: bool = False,
    ) -> Tuple[Tensor, None]:
        """Compute state representation for a backstory.
        
        Args:
            text: Backstory text.
            verbose: Print debug info.
            
        Returns:
            Tuple of (Representation Tensor, None).
        """
        if self.model_config.model_type == "gemini":
            return self._compute_gemini_state(text), None
        elif self.model_config.model_type == "gemma":
            return self._compute_gemma_state(text), None
        
        chunk_size = self.inference_config.chunk_size
        token_chunks = self.tokenizer.chunk_text(text, chunk_size)

        if self.model_config.model_type == "bdh":
            return self._compute_bdh_state(token_chunks), None
        else:
            return self._compute_gpt2_state(token_chunks), None

    def compute_velocity_from_states(
        self, 
        state_backstory: Tensor, 
        state_novel: Tensor
    ) -> float:
        """Compute distance between backstory and novel states.
        
        Calculates L2 Euclidean distance between the representation vectors
        (whether they are ρ-matrices or averaged hidden states).
        """
        # Ensure both states are on the same device
        sb = state_backstory.to(self.device)
        sn = state_novel.to(self.device)
        
        # Calculate L2 Norm of difference
        distance = torch.norm(sb - sn)
        return float(distance.item())

    # --- Internal Logic for BDH (Recurrent ρ-Matrix) ---
    def _compute_bdh_state(self, token_chunks: list[list[int]]) -> Tensor:
        """Accumulate ρ-matrix across chunks sequentially."""
        state = self.model.reset_state()
        
        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                
                # Forward pass updating state
                # Returns: logits, state, rho_update
                _, state, _ = self.model(
                    idx=input_ids, 
                    state=state, 
                    return_state=True
                )
                
                # Detach to prevent graph buildup
                if state: 
                    state = state.detach()
        
        # Return flattened ρ-matrix or zero vector
        if state and state.rho_matrix is not None:
            return state.rho_matrix.squeeze(0).cpu()
        else:
            # Calculate dimension: N * nh * D
            # We access internal config to calculate correct zero vector size
            multiplier = getattr(self.model_config, 'mlp_internal_dim_multiplier', 4)
            N = multiplier * self.model_config.n_embd // self.model_config.n_head
            dim = self.model_config.n_head * N * self.model_config.n_embd
            return torch.zeros(dim, device="cpu")

    # --- Internal Logic for GPT-2 (Hidden State Averaging) ---
    def _compute_gpt2_state(self, token_chunks: list[list[int]]) -> Tensor:
        """Compute average of last hidden states across all chunks."""
        hidden_means = []
        
        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                input_ids = torch.tensor(chunk, dtype=torch.long, device=self.device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                
                # Forward pass
                outputs = self.model.gpt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get last hidden state: [1, seq_len, n_embd]
                last_hidden = outputs.last_hidden_state
                
                # Average over sequence length: [n_embd]
                chunk_mean = last_hidden.mean(dim=1).squeeze(0)
                hidden_means.append(chunk_mean)
        
        if not hidden_means:
            return torch.zeros(self.model_config.n_embd, device="cpu")
        
        # Average over all chunks
        final_state = torch.stack(hidden_means).mean(dim=0).cpu()
        return final_state

    # --- Internal Logic for Gemini (API-based Embeddings) ---
    def _compute_gemini_state(self, text: str) -> Tensor:
        """Compute state representation using Gemini API embeddings."""
        chunk_size = self.inference_config.chunk_size
        state = self.model.compute_text_state(text, chunk_size=chunk_size)
        return state.cpu()
    
    # --- Internal Logic for Gemma (API-based Embeddings) ---
    def _compute_gemma_state(self, text: str) -> Tensor:
        """Compute state representation using Gemma API embeddings."""
        chunk_size = self.inference_config.chunk_size
        state = self.model.compute_text_state(text, chunk_size=chunk_size)
        return state.cpu()

    def compute_loss(self, text: str) -> float:
        """Compute language modeling loss (surprisal) for given text.
        
        Adapts to the input requirements of the specific architecture.
        """
        # Gemini and Gemma don't support loss computation (API-based)
        if self.model_config.model_type in ["gemini", "gemma"]:
            return 0.0
        
        self.model.eval()
        tokens = self.tokenizer.encode(text)
        chunk_size = self.inference_config.chunk_size
        
        # Split into chunks
        if len(tokens) > chunk_size:
            token_chunks = self.tokenizer.chunk_text(text, chunk_size)
        else:
            token_chunks = [tokens] if tokens else []
        
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # BDH needs state tracking across chunks
        state = self.model.reset_state() if self.model_config.model_type == "bdh" else None

        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                # Prepare inputs
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                labels = input_ids.clone()
                
                logits = None

                if self.model_config.model_type == "bdh":
                    # BDH Forward
                    logits, state, _ = self.model(
                        idx=input_ids, 
                        state=state, 
                        return_state=True
                    )
                else:
                    # GPT-2 Forward
                    # We need explicit attention mask for GPT-2
                    attention_mask = (input_ids != 0).long()
                    outputs = self.model.gpt(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    
                    # Project hidden states to vocab if we have a head
                    if self.lm_head is not None:
                        logits = self.lm_head(outputs.last_hidden_state)
                    else:
                        # Cannot compute LM loss without a head
                        return 0.0

                # Shift logits and labels for Causal LM loss
                # logits[..., :-1, :] predicts labels[..., 1:]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                # Accumulate weighted by token count
                num_tokens = shift_labels.ne(0).sum().item()
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

        return total_loss / total_tokens if total_tokens > 0 else 0.0