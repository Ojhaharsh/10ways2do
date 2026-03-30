"""Publish-ready summary artifact helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class PublishReadyError(ValueError):
    """Raised when publish-ready summary generation fails."""



def _render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Publish Ready Summary: {payload['publish_tag']}")
    lines.append("")
    lines.append(f"Generated UTC: {payload['generated_at_utc']}")
    lines.append(f"Results Directory: {payload['results_dir']}")
    lines.append(f"Snapshot Directory: {payload['snapshot_dir']}")
    lines.append("")
    lines.append("## Stage Results")
    lines.append("")
    lines.append("| Stage | Status | Duration (s) | Details |")
    lines.append("|-------|--------|--------------|---------|")

    for stage in payload.get("stages", []):
        lines.append(
            f"| {stage.get('name', 'unknown')} | {stage.get('status', 'unknown')} | "
            f"{stage.get('duration_seconds', 0.0):.2f} | {stage.get('details', '')} |"
        )

    lines.append("")
    lines.append(f"Overall Status: **{payload.get('overall_status', 'unknown')}**")
    lines.append("")
    return "\n".join(lines)



def save_publish_ready_summary(
    publish_tag: str,
    results_dir: str | Path,
    snapshot_dir: str | Path,
    stages: List[Dict[str, Any]],
) -> Path:
    """Write publish-ready summary artifacts and return output directory path."""
    clean_tag = str(publish_tag).strip()
    if not clean_tag:
        raise PublishReadyError("publish_tag must be non-empty")

    output_dir = Path(snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_status = "PASS" if all(s.get("status") == "PASS" for s in stages) else "FAIL"
    payload = {
        "publish_tag": clean_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "snapshot_dir": str(output_dir),
        "overall_status": overall_status,
        "stages": stages,
    }

    with (output_dir / "publish_ready_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    markdown = _render_markdown(payload)
    (output_dir / "PUBLISH_READY_SUMMARY.md").write_text(markdown, encoding="utf-8")

    return output_dir
