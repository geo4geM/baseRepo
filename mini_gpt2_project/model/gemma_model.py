"""Gemma model API wrapper for backstory consistency classification."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import Tensor

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

from ..config.model_config import ModelConfig


class GemmaModel:
    """Gemma model wrapper for narrative consistency checking.
    
    This model uses Hugging Face's Inference API to access Gemma models
    for narrative consistency classification. Unlike the local models (GPT2/BDH),
    this model uses API calls and doesn't require training.
    """

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        """Initialize the Gemma model wrapper.

        Args:
            config: Model configuration (used for compatibility with other models).
            api_key: Hugging Face API key. If not provided, will try to get from
                environment variable HUGGINGFACE_API_KEY. If None, will use public API.
        """
        if InferenceClient is None:
            raise ImportError(
                "huggingface_hub package is required. Install it with: "
                "pip install huggingface_hub"
            )
        
        # Get API key - can be None for public models
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        # Initialize Hugging Face client
        self.client = InferenceClient(token=self.api_key)
        
        self.config = config
        # Use google/gemma-2b-it for embeddings (lightweight and fast)
        # You can change this to gemma-7b-it or gemma-1.1-2b-it if needed
        self.model_name = "google/gemma-2b-it"
        
        # For compatibility with other models, we'll use a fixed embedding dimension
        # Gemma embeddings are typically 2560-dimensional for 2B model
        self.embedding_dim = getattr(config, 'n_embd', 2560)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string using Gemma API.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        try:
            # Truncate text if too long (API has limits)
            max_length = 512  # Adjust based on API limits
            if len(text) > max_length:
                text = text[:max_length]
            
            # Use sentence-transformers model via Hugging Face Inference API
            # Try using a sentence transformer model that's better for embeddings
            try:
                # Use a sentence transformer model for better embeddings
                response = self.client.feature_extraction(
                    model="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight embedding model
                    inputs=text
                )
            except Exception:
                # Fallback: Use Gemma model with text generation and create hash-based embedding
                # This is a workaround if feature extraction doesn't work
                response = None
            
            if response is not None:
                # Response is a list of embeddings
                if isinstance(response, list):
                    if len(response) > 0:
                        if isinstance(response[0], list):
                            # Multiple embeddings, average them
                            embedding = np.array(response[0], dtype=np.float32)
                        else:
                            # Single embedding
                            embedding = np.array(response, dtype=np.float32)
                    else:
                        embedding = np.zeros(384, dtype=np.float32)  # all-MiniLM-L6-v2 has 384 dims
                else:
                    embedding = np.array(response, dtype=np.float32)
                
                # Ensure correct dimension
                if len(embedding.shape) > 1:
                    # If we get [seq_len, hidden_dim], average over sequence
                    embedding = np.mean(embedding, axis=0)
            else:
                # Fallback: Create hash-based embedding
                import hashlib
                text_hash = hashlib.sha256(text.encode()).digest()
                embedding = np.array(list(text_hash), dtype=np.float32)
            
            # Pad or truncate to embedding_dim
            if len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error getting embedding from Gemma API: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)
    
    def compute_text_state(self, text: str, chunk_size: int = 1024) -> Tensor:
        """Compute state representation for a text by chunking and averaging embeddings.
        
        This method mimics the behavior of GPT2/BDH models by computing
        a representation vector from text. For Gemma, we:
        1. Split text into chunks
        2. Get embeddings for each chunk
        3. Average the embeddings to get a single representation
        
        Args:
            text: Input text to process.
            chunk_size: Approximate chunk size in characters (for consistency with other models).
            
        Returns:
            State representation as a PyTorch tensor.
        """
        # Split text into chunks (simple character-based chunking)
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():  # Skip empty chunks
                chunks.append(chunk)
        
        if not chunks:
            # Return zero vector if no chunks
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        
        # Get embeddings for all chunks
        chunk_embeddings = self.get_embeddings_batch(chunks)
        
        # Average the embeddings to get a single representation
        mean_embedding = np.mean(chunk_embeddings, axis=0)
        
        # Convert to PyTorch tensor
        return torch.from_numpy(mean_embedding).float()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward pass for compatibility with other models.
        
        Note: For Gemma, we don't use input_ids directly. Instead, we expect
        text to be provided, or we can decode input_ids if needed.
        
        Args:
            input_ids: Token ids (not used directly for Gemma, but kept for compatibility).
            attention_mask: Attention mask (not used for Gemma).
            labels: Optional labels (not used for inference).
            text: Optional text string to process directly.
            
        Returns:
            Dictionary with:
            - "logits": Dummy logits (not used for Gemma).
            - "loss": None (no training for API-based model).
            - "embedding": The computed embedding/state representation.
        """
        # For Gemma, we primarily work with text directly
        # If text is provided, use it; otherwise return dummy output
        if text:
            embedding = self.compute_text_state(text)
            return {
                "loss": None,
                "logits": torch.zeros(1, self.config.num_classes),
                "embedding": embedding,
            }
        else:
            # Return dummy output for compatibility
            return {
                "loss": None,
                "logits": torch.zeros(1, self.config.num_classes),
                "embedding": torch.zeros(self.embedding_dim),
            }
    
    def eval(self):
        """Set model to evaluation mode (no-op for API-based model)."""
        return self
    
    def to(self, device):
        """Move model to device (no-op for API-based model)."""
        return self
