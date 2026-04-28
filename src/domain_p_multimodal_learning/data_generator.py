"""Multimodal dataset generator."""
import numpy as np
from typing import Dict, Any

def generate_multimodal_dataset(n_samples: int = 1000, n_image_features: int = 256, n_text_features: int = 128, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    images = rng.randn(n_samples, n_image_features).astype(np.float32)
    texts = rng.randn(n_samples, n_text_features).astype(np.float32)
    labels = rng.randint(0, 10, n_samples)
    return {"images": images, "texts": texts, "labels": labels}
