"""Data loading utilities for Mini-GPT2 backstory datasets and dataloaders."""

"""
BDH Track B: Data Loading Utilities

Handles CSV loading, book text loading, and byte-level tokenization.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


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


class ByteTokenizer:
    """Simple byte-level tokenizer for BDH (vocab_size=256)."""
    
    def __init__(self):
        self.vocab_size = 256
    
    def encode(self, text: str) -> List[int]:
        """Convert text to byte tokens."""
        return list(text.encode('utf-8'))
    
    def decode(self, tokens: List[int]) -> str:
        """Convert byte tokens back to text."""
        return bytes(tokens).decode('utf-8', errors='replace')
    
    def chunk_text(self, text: str, chunk_size: int) -> List[List[int]]:
        """Split text into chunks of byte tokens."""
        tokens = self.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
        
        return chunks


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
