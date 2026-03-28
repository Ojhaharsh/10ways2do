"""
Configuration settings for ML Philosophy Benchmark
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import os


@dataclass
class DataConfig:
    """Data configuration"""
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    synthetic_dir: Path = Path("data/synthetic")
    
    # Synthetic data sizes
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 2000
    
    # Noise levels for robustness testing
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    
    # Random seed
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration"""
    # Classical ML
    classical_max_iter: int = 1000
    
    # Tree-based
    n_estimators: int = 100
    max_depth: int = 10
    
    # Deep Learning
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5
    
    # Transformer
    num_heads: int = 8
    transformer_layers: int = 4
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000


@dataclass 
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics to compute
    compute_latency: bool = True
    compute_memory: bool = True
    compute_robustness: bool = True
    compute_data_efficiency: bool = True
    
    # Data efficiency sample sizes
    efficiency_samples: List[int] = field(
        default_factory=lambda: [100, 500, 1000, 2000, 5000, 10000]
    )
    
    # Number of runs for statistical significance
    n_runs: int = 5
    
    # Latency measurement
    latency_warmup: int = 10
    latency_iterations: int = 100


@dataclass
class SystemsConfig:
    """Systems evaluation configuration"""
    # Scalability test sizes
    scalability_sizes: List[int] = field(
        default_factory=lambda: [1000, 10000, 100000, 1000000]
    )
    
    # Concurrent requests for throughput
    concurrent_requests: List[int] = field(
        default_factory=lambda: [1, 10, 50, 100]
    )
    
    # Memory limits (MB)
    memory_limit: int = 4096


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    systems: SystemsConfig = field(default_factory=SystemsConfig)
    
    # Results directory
    results_dir: Path = Path("results")
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)


# Global configuration instance
CONFIG = Config()


def get_config() -> Config:
    """Get global configuration"""
    return CONFIG


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)