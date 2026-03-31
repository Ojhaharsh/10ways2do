"""Tests for snapshot restoration functionality."""
import json
import tempfile
from pathlib import Path

import pytest

from src.core.snapshot_restore import restore_snapshot


def test_restore_snapshot_successful_restore(tmp_path):
    """Test successful restoration of a snapshot."""
    # Setup snapshot structure
    snapshots_root = tmp_path / "releases"
    snapshots_root.mkdir()
    snapshot_dir = snapshots_root / "v1.0.0"
    snapshot_dir.mkdir()
    
    # Create domain subdirectories with artifacts
    domain_a_dir = snapshot_dir / "domain_a"
    domain_a_dir.mkdir()
    domain_b_dir = snapshot_dir / "domain_b"
    domain_b_dir.mkdir()
    
    # Create fake artifact files
    (domain_a_dir / "manifest.json").write_text('{"domain": "domain_a"}')
    (domain_a_dir / "runs.json").write_text('{"runs": []}')
    (domain_b_dir / "manifest.json").write_text('{"domain": "domain_b"}')
    (domain_b_dir / "runs.json").write_text('{"runs": []}')
    
    # Create snapshot.json
    snapshot_manifest = {
        "protocol_version": "1.0.0",
        "tag": "v1.0.0",
        "domains": {
            "domain_a": {
                "artifacts": {
                    "manifest": "manifest.json",
                    "runs": "runs.json",
                }
            },
            "domain_b": {
                "artifacts": {
                    "manifest": "manifest.json",
                    "runs": "runs.json",
                }
            },
        },
    }
    (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot_manifest))
    
    # Restore snapshot
    output_dir = tmp_path / "results"
    result = restore_snapshot(
        snapshot_tag="v1.0.0",
        snapshots_root=snapshots_root,
        output_dir=output_dir,
    )
    
    # Verify restoration
    assert result["status"] == "PASS"
    assert result["tag"] == "v1.0.0"
    assert result["protocol_version"] == "1.0.0"
    assert sorted(result["domains_restored"]) == ["domain_a", "domain_b"]
    assert not result["error"]
    
    # Verify files were restored
    assert (output_dir / "domain_a" / "manifest.json").exists()
    assert (output_dir / "domain_a" / "runs.json").exists()
    assert (output_dir / "domain_b" / "manifest.json").exists()
    assert (output_dir / "domain_b" / "runs.json").exists()


def test_restore_snapshot_missing_snapshot_directory(tmp_path):
    """Test restoration fails when snapshot directory doesn't exist."""
    snapshots_root = tmp_path / "releases"
    output_dir = tmp_path / "results"
    
    result = restore_snapshot(
        snapshot_tag="v1.0.0",
        snapshots_root=snapshots_root,
        output_dir=output_dir,
    )
    
    assert result["status"] == "FAIL"
    assert "not found" in result["error"].lower()
    assert result["domains_restored"] == []


def test_restore_snapshot_protocol_version_mismatch(tmp_path):
    """Test restoration fails on protocol version mismatch."""
    snapshots_root = tmp_path / "releases"
    snapshots_root.mkdir()
    snapshot_dir = snapshots_root / "v1.0.0"
    snapshot_dir.mkdir()
    
    # Create snapshot with mismatched protocol version
    snapshot_manifest = {
        "protocol_version": "2.0.0",  # Different from current 1.0.0
        "tag": "v1.0.0",
        "domains": {},
    }
    (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot_manifest))
    
    output_dir = tmp_path / "results"
    result = restore_snapshot(
        snapshot_tag="v1.0.0",
        snapshots_root=snapshots_root,
        output_dir=output_dir,
    )
    
    assert result["status"] == "FAIL"
    assert "protocol version mismatch" in result["error"].lower()
    assert result["domains_restored"] == []


def test_restore_snapshot_partial_artifacts(tmp_path):
    """Test restoration succeeds even if some artifacts are missing."""
    snapshots_root = tmp_path / "releases"
    snapshots_root.mkdir()
    snapshot_dir = snapshots_root / "v1.0.0"
    snapshot_dir.mkdir()
    
    domain_a_dir = snapshot_dir / "domain_a"
    domain_a_dir.mkdir()
    
    # Create only one of the expected artifacts
    (domain_a_dir / "manifest.json").write_text('{"domain": "domain_a"}')
    # Missing: runs.json
    
    snapshot_manifest = {
        "protocol_version": "1.0.0",
        "tag": "v1.0.0",
        "domains": {
            "domain_a": {
                "artifacts": {
                    "manifest": "manifest.json",
                    "runs": "runs.json",  # This file doesn't exist
                }
            },
        },
    }
    (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot_manifest))
    
    output_dir = tmp_path / "results"
    result = restore_snapshot(
        snapshot_tag="v1.0.0",
        snapshots_root=snapshots_root,
        output_dir=output_dir,
    )
    
    # Should succeed with warning status
    assert result["status"] in ["PASS", "PASS_WITH_WARNINGS"]
    assert "domain_a" in result["domains_restored"]
    # The one existing artifact should be restored
    assert (output_dir / "domain_a" / "manifest.json").exists()
