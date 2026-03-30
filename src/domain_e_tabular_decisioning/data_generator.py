"""Synthetic tabular decisioning dataset generator."""

from typing import Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_tabular_decision_dataset(
    n_samples: int = 6000,
    n_features: int = 24,
    class_weight_positive: float = 0.18,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create an imbalanced binary classification dataset for risk decisioning."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(6, n_features // 3),
        n_redundant=max(2, n_features // 6),
        n_repeated=0,
        n_classes=2,
        weights=[1.0 - class_weight_positive, class_weight_positive],
        class_sep=1.0,
        flip_y=0.02,
        random_state=42,
    )

    # Add a mild non-linearity to make tree/boosting advantages observable.
    x[:, 0] = np.tanh(x[:, 0])
    x[:, 1] = x[:, 1] * x[:, 2]

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.4, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    return {
        "train": {"X": x_train, "y": y_train},
        "val": {"X": x_val, "y": y_val},
        "test": {"X": x_test, "y": y_test},
    }
