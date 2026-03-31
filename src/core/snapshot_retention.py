"""Retention policy utilities for release snapshot directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


NIGHTLY_TAG_RE = re.compile(r"^nightly-(\d{8})(?:$|[-_].*)")
SEMVER_TAG_RE = re.compile(r"^v\d+(?:\.\d+){0,2}(?:[-+].*)?$")


class SnapshotRetentionError(ValueError):
    """Raised when retention policy operations cannot be completed."""



def _is_protected(tag: str, protect_prefixes: Iterable[str]) -> bool:
    if SEMVER_TAG_RE.match(tag):
        return True

    prefixes = [p.strip() for p in protect_prefixes if str(p).strip()]
    return any(tag.startswith(prefix) for prefix in prefixes)



def prune_snapshot_directories(
    snapshots_root: str | Path = "releases",
    keep_nightly: int = 30,
    protect_prefixes: Iterable[str] = ("v", "stable", "release"),
) -> Dict[str, List[str]]:
    """Prune old nightly snapshots while protecting stable/release tags.

    Returns a summary dict with kept/deleted/protected tags.
    """
    if keep_nightly < 0:
        raise SnapshotRetentionError("keep_nightly must be >= 0")

    root = Path(snapshots_root)
    if not root.exists():
        return {"kept": [], "deleted": [], "protected": []}
    if not root.is_dir():
        raise SnapshotRetentionError(f"Snapshots root is not a directory: {root}")

    protected: List[str] = []
    nightly: List[tuple[int, str, Path]] = []
    kept: List[str] = []
    deleted: List[str] = []

    for child in root.iterdir():
        if not child.is_dir():
            continue

        tag = child.name
        if _is_protected(tag, protect_prefixes):
            protected.append(tag)
            kept.append(tag)
            continue

        match = NIGHTLY_TAG_RE.match(tag)
        if match:
            nightly_date = int(match.group(1))
            nightly.append((nightly_date, tag, child))
            continue

        # Non-nightly and non-protected directories are retained by default.
        kept.append(tag)

    nightly.sort(key=lambda row: (row[0], row[1]), reverse=True)

    for idx, (_date, tag, path) in enumerate(nightly):
        if idx < keep_nightly:
            kept.append(tag)
            continue

        shutil.rmtree(path, ignore_errors=False)
        deleted.append(tag)

    return {
        "kept": sorted(kept),
        "deleted": sorted(deleted),
        "protected": sorted(protected),
    }
