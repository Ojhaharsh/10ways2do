from src.core.snapshot_retention import prune_snapshot_directories


def _mkdir(base, name):
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_prune_snapshot_directories_keeps_recent_nightly_and_stable(tmp_path):
    root = tmp_path / "releases"
    _mkdir(root, "nightly-20260325")
    _mkdir(root, "nightly-20260326")
    _mkdir(root, "nightly-20260327")
    _mkdir(root, "nightly-20260328")
    _mkdir(root, "v1.1")
    _mkdir(root, "stable-baseline")

    summary = prune_snapshot_directories(root, keep_nightly=2)

    assert "nightly-20260328" in summary["kept"]
    assert "nightly-20260327" in summary["kept"]
    assert "nightly-20260325" in summary["deleted"]
    assert "nightly-20260326" in summary["deleted"]
    assert "v1.1" in summary["protected"]
    assert "stable-baseline" in summary["protected"]


def test_prune_snapshot_directories_keep_zero_removes_all_nightly(tmp_path):
    root = tmp_path / "releases"
    _mkdir(root, "nightly-20260329")
    _mkdir(root, "nightly-20260330")

    summary = prune_snapshot_directories(root, keep_nightly=0)

    assert summary["deleted"] == ["nightly-20260329", "nightly-20260330"]
