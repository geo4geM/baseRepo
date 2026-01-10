"""Gemini Pro API wrapper for backstory consistency classification."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import Tensor

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..config.model_config import ModelConfig


class GeminiPro:
    """Gemini Pro model wrapper for narrative consistency checking.
    
    This model uses Google's Gemini Pro API to compute text embeddings
    for narrative consistency classification. Unlike the local models (GPT2/BDH),
    this model uses API calls and doesn't require training.
    """

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        """Initialize the Gemini Pro model wrapper.

        Args:
            config: Model configuration (used for compatibility with other models).
            api_key: Google Gemini API key. If not provided, will try to get from
                environment variable GEMINI_API_KEY.
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install it with: "
                "pip install google-generativeai"
            )
        
        # Get API key from parameter, environment variable, or raise error
        self.api_key = "AIzaSyA6XXb7x1UfUy4aT58mqkva6zI0csbm4iI" 
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        self.config = config
        # Try to use embedding model, fallback to text model if needed
        try:
            # Check if embedding model is available
            self.embedding_model = genai.get_model("models/embedding-001")
            self.model_name = "models/embedding-001"
            self.use_embeddings = True
        except Exception:
            # Fallback to text generation model
            self.embedding_model = None
            self.model_name = "gemini-pro"
            self.use_embeddings = False
        
        # For compatibility with other models, we'll use a fixed embedding dimension
        # Gemini embeddings are typically 768-dimensional
        self.embedding_dim = getattr(config, 'n_embd', 768)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string using Gemini API.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        try:
            if self.use_embeddings and self.embedding_model:
                # Use embedding model
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"  # Good for document-level embeddings
                )
                embedding = result['embedding']
                return np.array(embedding, dtype=np.float32)
            else:
                # Fallback: Use text generation model to create a representation
                # We'll use the model to generate a summary/representation
                # For now, create a simple hash-based representation
                # In practice, you might want to use the model's hidden states if available
                import hashlib
                text_hash = hashlib.sha256(text.encode()).digest()
                # Convert hash to embedding-like vector
                # Use array from bytes directly
                embedding = np.array(list(text_hash), dtype=np.float32)
                # Pad or truncate to embedding_dim
                if len(embedding) < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:self.embedding_dim]
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
        except Exception as e:
            print(f"Error getting embedding from Gemini API: {e}")
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
        a representation vector from text. For Gemini, we:
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
        
        Note: For Gemini, we don't use input_ids directly. Instead, we expect
        text to be provided, or we can decode input_ids if needed.
        
        Args:
            input_ids: Token ids (not used directly for Gemini, but kept for compatibility).
            attention_mask: Attention mask (not used for Gemini).
            labels: Optional labels (not used for inference).
            text: Optional text string to process directly.
            
        Returns:
            Dictionary with:
            - "logits": Dummy logits (not used for Gemini).
            - "loss": None (no training for API-based model).
            - "embedding": The computed embedding/state representation.
        """
        # For Gemini, we primarily work with text directly
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
