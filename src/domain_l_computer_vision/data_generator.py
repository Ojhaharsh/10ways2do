"""Synthetic image dataset generator for computer vision benchmarking."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


def generate_image_dataset(
    n_train: int = 1000,
    n_test: int = 200,
    image_size: Tuple[int, int] = (32, 32),
    n_classes: int = 10,
    seed: int = 42,
    output_dir: str = "data/domain_l",
) -> Dict[str, Any]:
    """Generate synthetic image dataset for classification."""
    rng = np.random.RandomState(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate synthetic images
    X_train = rng.randn(n_train, 3, image_size[0], image_size[1]).astype(np.float32)
    y_train = rng.randint(0, n_classes, n_train)

    X_test = rng.randn(n_test, 3, image_size[0], image_size[1]).astype(np.float32)
    y_test = rng.randint(0, n_classes, n_test)

    # Add class-dependent patterns
    for i in range(n_train):
        X_train[i] += y_train[i] * 0.5

    for i in range(n_test):
        X_test[i] += y_test[i] * 0.5

    # Normalize
    X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
    X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_classes": n_classes,
        "image_size": image_size,
    }
