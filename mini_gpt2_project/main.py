"""Local entry point for running training and evaluation of Mini-GPT2 models."""

"""
Mini-GPT2 Track B: Main Pipeline
Narrative consistency classification using Mini-GPT2 architecture.
"""

import argparse
import json
import os
import sys
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mini_gpt2_project.utils.data_loader import get_tokenizer

# Add project to path (for local execution)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- MODIFIED IMPORTS FOR MINI-GPT2 ---
from .config.model_config import (
    get_config_by_name,
    InferenceConfig,
    ModelConfig,
    get_device,
    get_dtype,
)
from .metrics.analysis_metrics import ConsistencyMetrics, CalibrationResult
from .utils.data_loader import DataLoader, ByteTokenizer, get_dataset_stats
from .inference.predictor import MiniGPT2Wrapper
from .model.mini_gpt2 import MiniGPT2
# ---------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mini-GPT2 Narrative Consistency")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Run calibration/training only")
    mode_group.add_argument("--inference", action="store_true", help="Run test inference only")
    
    # Model size (kept for compatibility, though we only have one config now)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--default", action="store_true", help="Use default model")
    size_group.add_argument("--small", action="store_true", help="Use small model")
    
    # Pipeline options
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser.parse_args()

def setup_directories(output_dir: str) -> Dict[str, Path]:
    paths = {
        "output": Path(output_dir),
        "checkpoints": Path(output_dir) / "checkpoints",
        "model_checkpoints": Path(output_dir) / "checkpoints" / "models",
        "plots": Path(output_dir) / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def save_checkpoint(calibration: CalibrationResult, path: Path, model_config_name: str):
    data = {
        "timestamp": datetime.now().isoformat(),
        "model_config": model_config_name,
        "calibration": calibration.to_dict(),
        "example_ids": calibration.example_ids,
        "max_velocities": calibration.max_velocities,
        "labels": calibration.labels,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved checkpoint: {path}")

def load_checkpoint(path: Path) -> CalibrationResult:
    with open(path, "r") as f:
        data = json.load(f)
    
    calibration = CalibrationResult(
        optimal_threshold=data["calibration"]["optimal_threshold"],
        train_accuracy=data["calibration"]["train_accuracy"],
        example_ids=data["example_ids"],
        max_velocities=data["max_velocities"],
        labels=data["labels"],
    )
    calibration.consistent_mean = data["calibration"]["consistent_mean"]
    calibration.consistent_std = data["calibration"]["consistent_std"]
    calibration.contradict_mean = data["calibration"]["contradict_mean"]
    calibration.contradict_std = data["calibration"]["contradict_std"]
    return calibration

class BookTextDataset(Dataset):
    """Dataset for fine-tuning GPT2 on book texts."""
    
    def __init__(self, token_chunks: List[List[int]], max_seq_len: int = 1024):
        self.token_chunks = token_chunks
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        return len(self.token_chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.token_chunks[idx]
        # Truncate or pad to max_seq_len
        if len(chunk) > self.max_seq_len:
            chunk = chunk[:self.max_seq_len]
        else:
            chunk = chunk + [0] * (self.max_seq_len - len(chunk))
        
        input_ids = torch.tensor(chunk, dtype=torch.long)
        # For language modeling: input_ids and labels are the same (shifted in model)
        labels = input_ids.clone()
        attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def train_gpt2_on_books(
    model: MiniGPT2,
    loader: DataLoader,
    model_config: ModelConfig,
    device: torch.device,
    paths: Dict[str, Path],
    max_steps: int = 1000,
    batch_size: int = 24,
) -> Tuple[Path, nn.Module]:
    """
    Fine-tune GPT2 on book texts using language modeling objective.
    Now supports training the tokenizer if vocab files are missing.
    """
    print("\n" + "="*60 + "\nPHASE 0: FINE-TUNING GPT2 ON BOOK TEXTS\n" + "="*60)
    
    # 1. Gather all book paths for tokenizer training
    book_paths = [str(loader.get_book_path(name)) for name in loader.book_mapping.keys()]
    
    # 2. Initialize (and potentially train) the Tokenizer
    print("Initializing Tokenizer...")
    # This will trigger training if 'vocab.json' is missing
    tokenizer = get_tokenizer(model_config, training_files=book_paths)
    
    # 3. Load and Tokenize text
    all_chunks: List[List[int]] = []
    print("Loading and tokenizing book texts...")
    
    for book_name in loader.book_mapping.keys():
        book_path = loader.get_book_path(book_name)
        print(f"Processing: {book_name}")
        
        with open(book_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            
        # chunk_text handles the encoding internally
        chunks = tokenizer.chunk_text(text, chunk_size=model_config.max_seq_len)
        all_chunks.extend(chunks)
        print(f"  Added {len(chunks)} chunks from {book_name}")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # 4. Prepare Dataset and Loader
    train_dataset = BookTextDataset(all_chunks, max_seq_len=model_config.max_seq_len)
    
    # Determine num_workers based on CPU count (safe default is 2 or 4)
    num_workers = min(os.cpu_count() or 1, 4)
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 5. Setup Optimizer and Loss
    # We use a lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    # Create a fresh LM head (output layer)
    lm_head = nn.Linear(model_config.n_embd, model_config.vocab_size).to(device)
    lm_head.train()
    model.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is padding in our dataset

    # 6. Training Loop
    step = 0
    total_loss = 0.0
    pbar = tqdm(total=max_steps, desc="Training", unit="step")
    
    # Infinite iterator over the data loader until max_steps is reached
    train_iter = iter(train_loader)
    
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Compute logits and loss
        logits = lm_head(hidden_states)
        
        # Flatten for loss calculation: (batch * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        total_loss += loss.item()
        
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss/step:.4f}",
        })

    pbar.close()

    # 7. Save Final Model
    final_checkpoint_path = paths["model_checkpoints"] / "model_final.pt"
    save_model_checkpoint(model, lm_head, optimizer, step, final_checkpoint_path)
    
    print(f"\n✓ Training complete! Final checkpoint: {final_checkpoint_path}")
    print(f"  Total steps: {step}")
    print(f"  Average loss: {total_loss/step:.4f}")

    return final_checkpoint_path, lm_head

def save_model_checkpoint(
    model: MiniGPT2,
    lm_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "lm_head_state_dict": lm_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }, checkpoint_path)

def load_model_checkpoint(
    model: MiniGPT2,
    lm_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """Load model checkpoint and return step number."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    lm_head.load_state_dict(checkpoint["lm_head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    print(f"✓ Loaded checkpoint from step {step}")
    return step

def evaluate_on_train_csv(
    model: MiniGPT2,
    loader: DataLoader,
    model_config: ModelConfig,
    device: torch.device,
    paths: Dict[str, Path],
) -> float:
    """Evaluate trained model on train.csv and compute accuracy.
    Returns:
        Accuracy score on train.csv.
    """
    print("\n" + "="*60 + "\nEVALUATING ON TRAIN.CSV\n" + "="*60)
    model.eval()
    tokenizer = get_tokenizer(model_config)
    train_examples = loader.get_train_examples()
    predictions = []
    true_labels = []
    print(f"Evaluating on {len(train_examples)} examples...")

    with torch.no_grad():
        for example in tqdm(train_examples, desc="Evaluating"):
            try:
                tokens = tokenizer.encode(example['content'])
                if len(tokens) > model_config.max_seq_len:
                    tokens = tokens[:model_config.max_seq_len]
                else:
                    tokens = tokens + [0] * (model_config.max_seq_len - len(tokens))
                input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
                attention_mask = (input_ids != 0).long().to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                pred = logits.argmax(dim=-1).item()
                predictions.append(pred)
                true_labels.append(example['label_binary'])
            except Exception as e:
                print(f"Error on example {example['id']}: {e}")
                predictions.append(1)  # Default to consistent
                true_labels.append(example['label_binary'])
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n✓ Accuracy on train.csv: {accuracy:.4f} ({accuracy*100:.2f}%)")
    results_df = pd.DataFrame({
        "id": [ex['id'] for ex in train_examples],
        "true_label": true_labels,
        "predicted_label": predictions,
    })
    results_df.to_csv(paths["output"] / "train_evaluation.csv", index=False)
    print(f"✓ Saved evaluation results to {paths['output']}/train_evaluation.csv")
    return accuracy

def precompute_novel_states(wrapper: MiniGPT2Wrapper, loader: DataLoader, paths: Dict[str, Path]) -> Dict[str, any]:
    cache_path = paths["checkpoints"] / "novel_states.pkl"
    if cache_path.exists():
        print(f"\n✓ Loading cached novel states from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    print("\n" + "="*60 + "\nPHASE 0: PRE-COMPUTING NOVEL STATES\n" + "="*60)
    novel_states = {}
    for book_name in loader.book_mapping.keys():
        print(f"\nProcessing: {book_name}")
        novel_path = loader.get_book_path(book_name)
        novel_state = wrapper.compute_novel_state(novel_path, verbose=True)
        novel_states[book_name] = novel_state
        print(f"✓ Cached state for {book_name}")
    with open(cache_path, 'wb') as f:
        pickle.dump(novel_states, f)
    return novel_states

def run_calibration(wrapper: MiniGPT2Wrapper, loader: DataLoader, novel_states: Dict, paths: Dict, args, config_name: str, is_validation: bool = False) -> CalibrationResult:
    phase_name = "VALIDATION" if is_validation else "CALIBRATION"
    print(f"\n{'='*60}\nPHASE {'2' if is_validation else '1'}: {phase_name}\n{'='*60}")
    train_examples = loader.get_train_examples()
    if not args.dry_run and not args.limit:
        train_split, val_split = train_test_split(
            train_examples, train_size=60, test_size=20, 
            random_state=42, stratify=[ex['label_binary'] for ex in train_examples]
        )
        examples = val_split if is_validation else train_split
    else:
        examples = train_examples[:args.limit] if args.limit else train_examples
    calibration = CalibrationResult()
    pbar = tqdm(examples, desc=phase_name.title())
    for i, example in enumerate(pbar):
        try:
            book_name = example['book_name']
            if book_name not in novel_states: continue
            backstory_state, _ = wrapper.prime_with_backstory(example['content'])
            velocity = wrapper.compute_velocity_from_states(backstory_state, novel_states[book_name])
            calibration.add_example(example['id'], velocity, example['label_binary'])
            pbar.set_postfix({"vel": f"{velocity:.4f}", "label": example['label_binary']})
            if not is_validation and (i + 1) % 10 == 0:
                calibration.compute_optimal_threshold()
                save_checkpoint(calibration, paths["checkpoints"] / f"calibration_partial_{i+1}.json", config_name)
        except Exception as e:
            print(f"Error: {e}")
            continue
    calibration.compute_optimal_threshold()
    print(f"Optimal Threshold: {calibration.optimal_threshold:.6f}")
    if not is_validation:
        save_checkpoint(calibration, paths["checkpoints"] / "calibration_final.json", config_name)
    return calibration

def run_inference(wrapper: MiniGPT2Wrapper, loader: DataLoader, novel_states: Dict, calibration: CalibrationResult, paths: Dict, args) -> pd.DataFrame:
    print("\n" + "="*60 + "\nPHASE 3: TEST INFERENCE\n" + "="*60)
    test_examples = loader.get_test_examples()
    if args.limit: test_examples = test_examples[:args.limit]
    results = []
    for example in tqdm(test_examples, desc="Predicting"):
        try:
            book_name = example['book_name']
            novel_state = novel_states.get(book_name)
            if novel_state is None:
                prediction, velocity = 1, 0.0
            else:
                backstory_state, _ = wrapper.prime_with_backstory(example['content'])
                velocity = wrapper.compute_velocity_from_states(backstory_state, novel_state)
                prediction = calibration.predict(velocity)
            results.append({"id": example['id'], "prediction": prediction, "velocity": velocity})
        except Exception as e:
            results.append({"id": example['id'], "prediction": 1, "velocity": 0.0})
    results_df = pd.DataFrame(results)
    results_df[["id", "prediction"]].to_csv(paths["output"] / "results.csv", index=False)
    print(f"✓ Saved results to {paths['output']}/results.csv")
    return results_df

def main():
    args = parse_args()
    print("="*60 + "\nMINI-GPT2 TRACK B PIPELINE\n" + "="*60)
    
    model_config = get_config_by_name("default")
    inference_config = InferenceConfig()
    device = get_device()
    
    paths = setup_directories(args.output_dir)
    loader = DataLoader(base_path=PROJECT_ROOT.parent)
    
    print("Initializing Model...")
    model = MiniGPT2(model_config).to(device)
    
    # Check if we should load a pretrained model
    model_checkpoint = args.checkpoint
    if model_checkpoint and Path(model_checkpoint).exists() and model_checkpoint.endswith('.pt'):
        print(f"Loading pretrained model from {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Loaded pretrained model")
        vocab_size = model_config.vocab_size
        n_embd = model_config.n_embd
        lm_head = nn.Linear(n_embd, vocab_size).to(device)
        if "lm_head_state_dict" in checkpoint:
            lm_head.load_state_dict(checkpoint["lm_head_state_dict"])
            print("✓ Loaded LM head from checkpoint")
    else:
        print("\n" + "="*60)
        print("TRAINING PHASE: Fine-tuning GPT2 on book texts")
        print("="*60)
        best_checkpoint_path, lm_head = train_gpt2_on_books(
            model=model,
            loader=loader,
            model_config=model_config,
            device=device,
            paths=paths,
            max_steps=1000,         # <--- number of gradient steps to train for (ex: 2000, adjust as needed)
            batch_size=24,           # <--- large, since you have plenty of memory
        )

    # Create wrapper with the trained/loaded model and LM head
    wrapper = MiniGPT2Wrapper(model_config, inference_config, device, model=model, lm_head=lm_head)
    
    # Evaluate on train.csv
    accuracy = evaluate_on_train_csv(
        model=model,
        loader=loader,
        model_config=model_config,
        device=device,
        paths=paths,
    )
    
    run_train = not args.inference
    run_infer = not args.train
    
    novel_states = precompute_novel_states(wrapper, loader, paths)
    calibration = None
    
    if run_train:
        calibration = run_calibration(wrapper, loader, novel_states, paths, args, "default", is_validation=False)
        if not args.dry_run and not args.limit:
            run_calibration(wrapper, loader, novel_states, paths, args, "default", is_validation=True)
            
    if run_infer:
        if calibration is None:
            ckpt = args.checkpoint or (paths["checkpoints"] / "calibration_final.json")
            calibration = load_checkpoint(Path(ckpt))
        run_inference(wrapper, loader, novel_states, calibration, paths, args)

if __name__ == "__main__":
    main()