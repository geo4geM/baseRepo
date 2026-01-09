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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project to path (for local execution)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- MODIFIED IMPORTS FOR MINI-GPT2 ---
# Use relative imports so this works both locally and when imported as a package
from .config.model_config import (
    get_config_by_name,
    InferenceConfig,
    get_device,
    get_dtype,
)
from .metrics.analysis_metrics import ConsistencyMetrics, CalibrationResult
from .utils.data_loader import DataLoader, get_dataset_stats
from .inference.predictor import MiniGPT2Wrapper
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
    wrapper = MiniGPT2Wrapper(model_config, inference_config, device)
    
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