"""Tests for Domain F: Cyber Threat Hunting."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_f_cyber_threat_hunting.data_generator import create_threat_hunting_dataset
from src.domain_f_cyber_threat_hunting.run_all import run_all_approaches


def test_threat_hunting_dataset_shapes():
    dataset = create_threat_hunting_dataset(n_samples=300, n_features=12, seed=42)

    assert set(dataset.keys()) == {"train", "val", "test"}
    assert dataset["train"]["X"].shape[1] == 12
    assert dataset["test"]["X"].shape[1] == 12
    assert dataset["train"]["y"].ndim == 1
    assert dataset["test"]["y"].ndim == 1
    assert np.unique(dataset["train"]["y"]).size >= 1


def test_threat_hunting_smoke_runner(tmp_path):
    output_dir = tmp_path / "domain_f"

    result = run_all_approaches(
        n_samples=300,
        n_features=12,
        n_runs=1,
        seed=42,
        smoke_test=True,
        save_results=True,
        output_dir=str(output_dir),
    )

    assert "aggregated_results" in result
    assert len(result["aggregated_results"]) > 0
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "results_aggregated.json").exists()
    assert (output_dir / "comparison_canonical.csv").exists()
    assert (output_dir / "comparison_variants.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
