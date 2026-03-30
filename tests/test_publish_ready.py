import json

from src.core.publish_ready import save_publish_ready_summary


def test_save_publish_ready_summary_writes_expected_files(tmp_path):
    snapshot_dir = tmp_path / "releases" / "v1.2-test"
    stages = [
        {"name": "report_generation", "status": "PASS", "duration_seconds": 1.2, "details": "ok"},
        {"name": "release_gate", "status": "PASS", "duration_seconds": 0.4, "details": "ok"},
    ]

    out = save_publish_ready_summary(
        publish_tag="v1.2-test",
        results_dir="results",
        snapshot_dir=snapshot_dir,
        stages=stages,
    )

    assert out == snapshot_dir
    assert (snapshot_dir / "publish_ready_summary.json").exists()
    assert (snapshot_dir / "PUBLISH_READY_SUMMARY.md").exists()

    payload = json.loads((snapshot_dir / "publish_ready_summary.json").read_text(encoding="utf-8"))
    assert payload["publish_tag"] == "v1.2-test"
    assert payload["overall_status"] == "PASS"
    assert len(payload["stages"]) == 2
