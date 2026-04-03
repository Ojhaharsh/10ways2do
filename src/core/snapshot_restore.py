"""Restore benchmark results from versioned snapshots."""
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List


def list_available_snapshots(snapshots_root: Path = Path("releases")) -> List[Dict[str, Any]]:
    """List available snapshots with lightweight metadata."""
    snapshots_root = Path(snapshots_root)
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        return []

    rows: List[Dict[str, Any]] = []
    for entry in snapshots_root.iterdir():
        if not entry.is_dir():
            continue

        snapshot_file = entry / "snapshot.json"
        row: Dict[str, Any] = {
            "tag": entry.name,
            "generated_at_utc": None,
            "protocol_version": None,
            "domain_count": 0,
            "has_report": (entry / "REPORT.md").exists(),
            "restorable": False,
            "valid": False,
        }

        if snapshot_file.exists():
            try:
                with open(snapshot_file, encoding="utf-8") as f:
                    payload = json.load(f)
                row["generated_at_utc"] = payload.get("generated_at_utc")
                row["protocol_version"] = payload.get("protocol_version") or payload.get(
                    "benchmark_protocol_version"
                )
                domains = payload.get("domains", {})
                if isinstance(domains, dict):
                    row["domain_count"] = len(domains)
                    row["restorable"] = any(
                        isinstance(domain_data, dict) and domain_data.get("artifacts")
                        for domain_data in domains.values()
                    )
                else:
                    # Older schemas may store domain summaries as a list.
                    domain_summaries = payload.get("domain_summaries", domains if isinstance(domains, list) else [])
                    row["domain_count"] = len(domain_summaries) if isinstance(domain_summaries, list) else 0
                    row["restorable"] = False
                row["valid"] = True
            except Exception:
                row["valid"] = False

        rows.append(row)

    # Sort newest first by generated timestamp when present, then tag name.
    rows.sort(
        key=lambda r: (str(r.get("generated_at_utc") or ""), str(r.get("tag") or "")),
        reverse=True,
    )
    return rows


def get_snapshot_info(snapshot_tag: str, snapshots_root: Path = Path("releases")) -> Dict[str, Any]:
    """Load full metadata for a single snapshot tag."""
    snapshot_dir = Path(snapshots_root) / snapshot_tag
    snapshot_file = snapshot_dir / "snapshot.json"
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
    if not snapshot_file.exists():
        raise FileNotFoundError(f"snapshot.json not found in {snapshot_dir}")

    with open(snapshot_file, encoding="utf-8") as f:
        payload = json.load(f)

    domains = payload.get("domains", {})
    report_artifacts = payload.get("report_artifacts", {})
    if isinstance(domains, dict):
        domain_names = sorted(list(domains.keys()))
        restorable = any(
            isinstance(domain_data, dict) and domain_data.get("artifacts")
            for domain_data in domains.values()
        )
        domain_count = len(domain_names)
    else:
        # Older snapshots may only have summaries and no artifact map.
        domain_summaries = payload.get("domain_summaries", domains if isinstance(domains, list) else [])
        domain_names = []
        for row in domain_summaries if isinstance(domain_summaries, list) else []:
            if isinstance(row, dict) and row.get("domain"):
                domain_names.append(str(row["domain"]))
        domain_names = sorted(domain_names)
        restorable = False
        domain_count = len(domain_names)

    return {
        "tag": snapshot_tag,
        "path": str(snapshot_dir),
        "generated_at_utc": payload.get("generated_at_utc"),
        "protocol_version": payload.get("protocol_version")
        or payload.get("benchmark_protocol_version"),
        "domain_count": domain_count,
        "domains": domain_names,
        "has_report": bool(report_artifacts.get("report")) and (snapshot_dir / "REPORT.md").exists(),
        "restorable": restorable,
    }


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
    if manifest_protocol is None:
        manifest_protocol = snapshot_manifest.get("benchmark_protocol_version")
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
    
    domains_payload = snapshot_manifest.get("domains", {})
    if not isinstance(domains_payload, dict):
        result["error"] = (
            "Snapshot does not contain restorable domain artifacts (legacy schema detected). "
            "Recreate snapshot with current --snapshot-tag flow."
        )
        return result

    for domain_name, domain_data in domains_payload.items():
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

    # Restore top-level artifacts (e.g., REPORT.md) when present.
    report_artifacts = snapshot_manifest.get("report_artifacts", {})
    for _artifact_key, artifact_filename in report_artifacts.items():
        source_file = snapshot_dir / artifact_filename
        target_file = output_dir / artifact_filename

        if not source_file.exists():
            errors.append(f"Artifact not found: {artifact_filename}")
            continue

        try:
            shutil.copy2(source_file, target_file)
        except Exception as e:
            errors.append(f"Failed to copy {artifact_filename}: {e}")
    
    if not domains_restored:
        result["error"] = (
            "No domain artifacts were restored from this snapshot. "
            "Snapshot may be incomplete or non-restorable."
        )
        result["details"] = "0 domains restored"
        return result

    result["status"] = "PASS" if not errors else "PASS_WITH_WARNINGS"
    result["domains_restored"] = domains_restored
    
    if errors:
        result["error"] = "; ".join(errors)
        result["details"] = f"Restored {len(domains_restored)} domains with {len(errors)} errors"
    else:
        result["details"] = f"Restored {len(domains_restored)} domains successfully"
    
    return result
