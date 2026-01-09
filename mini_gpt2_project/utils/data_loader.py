"""Data loading utilities for Mini-GPT2 backstory datasets and dataloaders."""

"""
BDH Track B: Data Loading Utilities

Handles CSV loading, book text loading, and byte-level tokenization.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

class BaseTokenizer:
    """
    Abstract tokenizer interface for modular tokenization.
    All tokenizers should inherit from this.
    """
    def encode(self, text: str):
        """Encode text to a list of tokens (integers)."""
        raise NotImplementedError("encode() must be implemented by subclass.")

    def decode(self, tokens):
        """Decode a list of tokens to a string."""
        raise NotImplementedError("decode() must be implemented by subclass.")

    def chunk_text(self, text: str, chunk_size: int):
        """
        Optional: Chunk text into multiple segments of length <= chunk_size.
        Returns a list of token lists.
        """
        raise NotImplementedError("chunk_text() must be implemented by subclass.")


class DataLoader:
    """Load and manage dataset for BDH Track B."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.book_texts: Dict[str, str] = {}
        
        # Book name to file mapping
        self.book_mapping = {
            "In Search of the Castaways": "In search of the castaways.txt",
            "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
        }
    
    def load_train(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load training CSV."""
        if path is None:
            path = self.base_path / "Dataset" / "train.csv"
        else:
            path = Path(path)
        
        self.train_df = pd.read_csv(path)
        # Handle any missing values
        self.train_df = self.train_df.fillna('')
        
        # Convert labels to binary
        if 'label' in self.train_df.columns:
            self.train_df['label_binary'] = self.train_df['label'].apply(
                lambda x: 1 if x == 'consistent' else 0
            )
        
        return self.train_df
    
    def load_test(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load test CSV."""
        if path is None:
            path = self.base_path / "Dataset" / "test.csv"
        else:
            path = Path(path)
        
        self.test_df = pd.read_csv(path)
        self.test_df = self.test_df.fillna('')
        return self.test_df
    
    def load_book(self, book_name: str) -> str:
        """Load book text by name."""
        if book_name in self.book_texts:
            return self.book_texts[book_name]
        
        if book_name not in self.book_mapping:
            raise ValueError(f"Unknown book: {book_name}. "
                           f"Available: {list(self.book_mapping.keys())}")
        
        filename = self.book_mapping[book_name]
        path = self.base_path / "Dataset" / "Books" / filename
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        self.book_texts[book_name] = text
        return text
    
    def get_book_path(self, book_name: str) -> Path:
        """Get path to book file."""
        if book_name not in self.book_mapping:
            raise ValueError(f"Unknown book: {book_name}")
        
        filename = self.book_mapping[book_name]
        return self.base_path / "Dataset" / "Books" / filename
    
    def preload_all_books(self):
        """Preload all books into memory."""
        for book_name in self.book_mapping:
            self.load_book(book_name)
    
    def get_train_examples(self) -> List[Dict]:
        """Get list of training examples."""
        if self.train_df is None:
            self.load_train()
        
        examples = []
        for _, row in self.train_df.iterrows():
            examples.append({
                'id': int(row['id']),
                'book_name': row['book_name'],
                'char': row['char'],
                'caption': row.get('caption', ''),
                'content': row['content'],
                'label': row.get('label', ''),
                'label_binary': row.get('label_binary', -1),
            })
        return examples
    
    def get_test_examples(self) -> List[Dict]:
        """Get list of test examples."""
        if self.test_df is None:
            self.load_test()
        
        examples = []
        for _, row in self.test_df.iterrows():
            examples.append({
                'id': int(row['id']),
                'book_name': row['book_name'],
                'char': row['char'],
                'caption': row.get('caption', ''),
                'content': row['content'],
            })
        return examples


class ByteTokenizer(BaseTokenizer):
    """
    Simple byte-level tokenizer.
    Each byte in UTF-8 is a token in [0, 255].
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode('utf-8'))

    def decode(self, tokens):
        return bytes(tokens).decode('utf-8', errors='replace')

    def chunk_text(self, text: str, chunk_size: int):
        tokens = self.encode(text)
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers



class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding tokenizer using HuggingFace tokenizers.
    You can train it on your corpus or load pretrained vocab/merges.
    """
    def __init__(self, vocab_path=None, merges_path=None, training_files=None, vocab_size=8000):
        if vocab_path and merges_path:
            # Load pre-existing BPE model
            self.tokenizer = Tokenizer(models.BPE.from_file(vocab_path, merges_path))
        elif training_files is not None:
            # Train new BPE model
            self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=True, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
            self.tokenizer.train(training_files, trainer)
        else:
            raise ValueError("Must provide either vocab/merges files or training files to BPETokenizer.")

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def chunk_text(self, text: str, chunk_size: int):
        tokens = self.encode(text)
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]



def get_tokenizer(config):
    """
    Factory function to select between ByteTokenizer and BPETokenizer
    based on config.tokenizer_type.
    """
    if getattr(config, "tokenizer_type", "byte") == "byte":
        return ByteTokenizer()
    elif getattr(config, "tokenizer_type", "byte") == "bpe":
        # You can also enhance this to read vocab/merges path or training_files from config as needed!
        return BPETokenizer(
            vocab_path=getattr(config, "bpe_vocab_path", None),
            merges_path=getattr(config, "bpe_merges_path", None),
            training_files=getattr(config, "bpe_training_files", None),
            vocab_size=getattr(config, "bpe_vocab_size", 8000),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {getattr(config, 'tokenizer_type', None)}")




def stream_book_chunks(book_path: Path, chunk_size: int = 2048) -> List[List[int]]:
    """Stream book as chunked byte tokens."""
    tokenizer = ByteTokenizer()
    
    with open(book_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    return tokenizer.chunk_text(text, chunk_size)


def get_dataset_stats(loader: DataLoader) -> Dict:
    """Get statistics about the dataset."""
    train_examples = loader.get_train_examples()
    test_examples = loader.get_test_examples()
    
    # Count by book
    train_by_book = {}
    for ex in train_examples:
        book = ex['book_name']
        train_by_book[book] = train_by_book.get(book, 0) + 1
    
    test_by_book = {}
    for ex in test_examples:
        book = ex['book_name']
        test_by_book[book] = test_by_book.get(book, 0) + 1
    
    # Label distribution
    labels = [ex['label_binary'] for ex in train_examples if ex['label_binary'] >= 0]
    consistent_count = sum(1 for l in labels if l == 1)
    contradict_count = sum(1 for l in labels if l == 0)
    
    return {
        'train_total': len(train_examples),
        'test_total': len(test_examples),
        'train_by_book': train_by_book,
        'test_by_book': test_by_book,
        'consistent_count': consistent_count,
        'contradict_count': contradict_count,
    }
