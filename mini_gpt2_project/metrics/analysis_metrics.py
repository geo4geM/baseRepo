"""Analysis and evaluation metrics for backstory consistency classification."""

"""
BDH Track B: Consistency Metrics

Dataclasses and utilities for tracking velocity, sparsity, and surprisal
during novel scanning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ConsistencyMetrics:
    """Metrics collected during novel scanning."""
    
    # Core velocity tracking
    velocities: List[float] = field(default_factory=list)
    chunk_positions: List[int] = field(default_factory=list)
    
    # Derived metrics
    max_velocity: float = 0.0
    max_velocity_position: int = -1
    mean_velocity: float = 0.0
    std_velocity: float = 0.0
    
    # Sparsity metrics
    sparsity_values: List[float] = field(default_factory=list)
    mean_sparsity: float = 0.0
    
    # Surprisal (optional, if computing losses)
    surprisal_values: List[float] = field(default_factory=list)
    max_surprisal: float = 0.0
    
    # State tracking
    total_tokens_processed: int = 0
    total_chunks: int = 0
    
    def update(self, velocity: float, chunk_idx: int, 
               sparsity: Optional[float] = None,
               surprisal: Optional[float] = None,
               tokens_in_chunk: int = 0):
        """Update metrics with new chunk data."""
        self.velocities.append(velocity)
        self.chunk_positions.append(chunk_idx)
        self.total_chunks += 1
        self.total_tokens_processed += tokens_in_chunk
        
        # Update max velocity
        if velocity > self.max_velocity:
            self.max_velocity = velocity
            self.max_velocity_position = chunk_idx
        
        # Update sparsity
        if sparsity is not None:
            self.sparsity_values.append(sparsity)
        
        # Update surprisal
        if surprisal is not None:
            self.surprisal_values.append(surprisal)
            if surprisal > self.max_surprisal:
                self.max_surprisal = surprisal
    
    def finalize(self):
        """Compute final aggregate statistics."""
        if self.velocities:
            self.mean_velocity = float(np.mean(self.velocities))
            self.std_velocity = float(np.std(self.velocities))
        
        if self.sparsity_values:
            self.mean_sparsity = float(np.mean(self.sparsity_values))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_velocity": self.max_velocity,
            "max_velocity_position": self.max_velocity_position,
            "mean_velocity": self.mean_velocity,
            "std_velocity": self.std_velocity,
            "mean_sparsity": self.mean_sparsity,
            "max_surprisal": self.max_surprisal,
            "total_tokens": self.total_tokens_processed,
            "total_chunks": self.total_chunks,
        }


@dataclass 
class CalibrationResult:
    """Result of threshold calibration on training set."""
    
    optimal_threshold: float = 0.0
    train_accuracy: float = 0.0
    
    # Per-example data
    example_ids: List[int] = field(default_factory=list)
    max_velocities: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)  # 1=consistent, 0=contradict
    
    # Distribution stats
    consistent_mean: float = 0.0
    consistent_std: float = 0.0
    contradict_mean: float = 0.0
    contradict_std: float = 0.0
    
    def add_example(self, example_id: int, max_velocity: float, label: int):
        """Add a training example result."""
        self.example_ids.append(example_id)
        self.max_velocities.append(max_velocity)
        self.labels.append(label)
    
    def compute_optimal_threshold(self) -> float:
        """Find threshold that maximizes accuracy."""
        if not self.max_velocities:
            return 0.0
        
        velocities = np.array(self.max_velocities)
        labels = np.array(self.labels)
        
        # Compute distribution stats
        consistent_mask = labels == 1
        contradict_mask = labels == 0
        
        if consistent_mask.sum() > 0:
            self.consistent_mean = float(velocities[consistent_mask].mean())
            self.consistent_std = float(velocities[consistent_mask].std())
        
        if contradict_mask.sum() > 0:
            self.contradict_mean = float(velocities[contradict_mask].mean())
            self.contradict_std = float(velocities[contradict_mask].std())
        
        # Grid search for optimal threshold
        # Hypothesis: consistent examples have LOWER max velocity
        thresholds = np.linspace(velocities.min(), velocities.max(), 100)
        best_acc = 0.0
        best_thresh = thresholds[len(thresholds) // 2]
        
        for thresh in thresholds:
            # Predict: max_vel < thresh → consistent (1), else contradict (0)
            predictions = (velocities < thresh).astype(int)
            accuracy = (predictions == labels).mean()
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_thresh = thresh
        
        self.optimal_threshold = float(best_thresh)
        self.train_accuracy = float(best_acc)
        
        return self.optimal_threshold
    
    def predict(self, max_velocity: float) -> int:
        """Predict label given max velocity."""
        # max_vel < threshold → consistent (1)
        return 1 if max_velocity < self.optimal_threshold else 0
    
    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "optimal_threshold": self.optimal_threshold,
            "train_accuracy": self.train_accuracy,
            "consistent_mean": self.consistent_mean,
            "consistent_std": self.consistent_std,
            "contradict_mean": self.contradict_mean,
            "contradict_std": self.contradict_std,
            "n_examples": len(self.example_ids),
        }


def compute_sparsity(tensor) -> float:
    """Compute sparsity (fraction of zeros) in a tensor."""
    if tensor is None:
        return 0.0
    total = tensor.numel()
    if total == 0:
        return 0.0
    zeros = (tensor == 0).sum().item()
    return zeros / total


def compute_velocity(state_old, state_new) -> float:
    """Compute L2 norm of state change (velocity)."""
    if state_old is None or state_new is None:
        return 0.0
    diff = state_new - state_old
    return float(diff.norm(p=2).item())
