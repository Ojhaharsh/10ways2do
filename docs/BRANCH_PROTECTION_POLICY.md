# Branch Protection Policy

This policy defines the minimum merge gate requirements for `main` / `master`.

## Required Status Checks

Require these checks to pass before merging:

1. `benchmark-smoke / smoke`
2. `release-quality-gate / quality-gate`

These checks ensure:

- smoke benchmark execution is healthy,
- artifact contract validation passes,
- release-gate report/protocol checks pass,
- reliability tests for validator/release-gate continue to pass.

## GitHub Settings (Repository Admin)

Apply these in GitHub repository settings:

1. Open `Settings` -> `Branches`.
2. Create or edit a branch protection rule for `main` (and `master` if used).
3. Enable `Require a pull request before merging`.
4. Enable `Require status checks to pass before merging`.
5. Add required checks:
   - `benchmark-smoke / smoke`
   - `release-quality-gate / quality-gate`
6. Enable `Require branches to be up to date before merging`.
7. Optional but recommended:
   - `Require conversation resolution before merging`
   - `Require approvals` (at least 1)
   - `Do not allow bypassing the above settings`

## Local Pre-PR Quality Gate

Run locally before opening or updating a PR:

```bash
python main.py --all --smoke-test --n-runs 1 --seed 42
python main.py --validate
python main.py --release-gate
pytest tests/test_artifact_validator.py tests/test_release_gate.py -v
```

## Change Control

Any change to CI workflow names/jobs should update this file so required check names stay accurate.

## Nightly Snapshot Automation

The nightly publish-ready workflow is informational and archival (not a required PR status check):

- `.github/workflows/nightly-publish-ready.yml`

It runs the publish-ready pipeline on a schedule and uploads `releases/<tag>` as a workflow artifact.

The workflow also enforces snapshot retention policy for nightly tags:

- keeps latest 30 `nightly-*` snapshots
- protects stable tags by prefix (`v`, `stable`, `release`)
