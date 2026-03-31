"""Restore benchmark results from versioned snapshots."""
import json
import shutil
from pathlib import Path
from typing import Dict, Any


def restore_snapshot(
    snapshot_tag: str,
    snapshots_root: Path = Path("releases"),
    output_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Restore benchmark results from a snapshot.
    
    Loads snapshot.json from releases/<tag>, validates protocol version,
    and copies artifacts back to output_dir for reproducible reruns.
    
    Args:
        snapshot_tag: Tag identifying the snapshot (e.g., "v1.0.0", "nightly-20260401")
        snapshots_root: Root directory containing snapshot bundles (default: releases/)
        output_dir: Target directory to restore artifacts to (default: results/)
    
    Returns:
        Dictionary with:
        - "status": "PASS" or "FAIL"
        - "tag": The snapshot tag restored
        - "protocol_version": Version from restored snapshot.json
        - "domains_restored": List of domains restored
        - "details": String description of restoration
        - "error": (if FAIL) Error message
    
    Raises:
        FileNotFoundError: If snapshot directory or snapshot.json not found
        ValueError: If protocol version mismatch or corrupted snapshot
    """
    from src.core.benchmark_utils import BENCHMARK_PROTOCOL_VERSION
    
    snapshots_root = Path(snapshots_root)
    output_dir = Path(output_dir)
    snapshot_dir = snapshots_root / snapshot_tag
    snapshot_file = snapshot_dir / "snapshot.json"
    
    result = {
        "status": "FAIL",
        "tag": snapshot_tag,
        "protocol_version": None,
        "domains_restored": [],
        "details": "",
        "error": None,
    }
    
    # Validate snapshot directory exists
    if not snapshot_dir.exists():
        result["error"] = f"Snapshot directory not found: {snapshot_dir}"
        result["details"] = f"Available snapshots: {list(snapshots_root.glob('*'))}"
        return result
    
    # Validate snapshot.json exists
    if not snapshot_file.exists():
        result["error"] = f"snapshot.json not found in {snapshot_dir}"
        return result
    
    # Load and validate snapshot manifest
    try:
        with open(snapshot_file) as f:
            snapshot_manifest = json.load(f)
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON in snapshot.json: {e}"
        return result
    except Exception as e:
        result["error"] = f"Failed to read snapshot.json: {e}"
        return result
    
    # Validate protocol version
    manifest_protocol = snapshot_manifest.get("protocol_version")
    result["protocol_version"] = manifest_protocol
    
    if manifest_protocol != BENCHMARK_PROTOCOL_VERSION:
        result["error"] = (
            f"Protocol version mismatch: snapshot={manifest_protocol}, "
            f"current={BENCHMARK_PROTOCOL_VERSION}"
        )
        return result
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Restore artifacts for each domain
    domains_restored = []
    errors = []
    
    for domain_name, domain_data in snapshot_manifest.get("domains", {}).items():
        domain_artifacts = domain_data.get("artifacts", {})
        
        if not domain_artifacts:
            continue
        
        # Create domain output subdirectory
        domain_output = output_dir / domain_name
        domain_output.mkdir(parents=True, exist_ok=True)
        
        # Copy each artifact file
        copied_count = 0
        for artifact_name, artifact_filename in domain_artifacts.items():
            source_file = snapshot_dir / domain_name / artifact_filename
            target_file = domain_output / artifact_filename
            
            if not source_file.exists():
                errors.append(
                    f"Artifact not found: {domain_name}/{artifact_filename}"
                )
                continue
            
            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
            except Exception as e:
                errors.append(
                    f"Failed to copy {domain_name}/{artifact_filename}: {e}"
                )
        
        if copied_count > 0:
            domains_restored.append(domain_name)
    
    result["status"] = "PASS" if not errors else "PASS_WITH_WARNINGS"
    result["domains_restored"] = domains_restored
    
    if errors:
        result["error"] = "; ".join(errors)
        result["details"] = f"Restored {len(domains_restored)} domains with {len(errors)} errors"
    else:
        result["details"] = f"Restored {len(domains_restored)} domains successfully"
    
    return result
