# Release Checklist

Use this checklist before creating any benchmark release tag.

## A. Environment and Inputs

- [ ] Python version is recorded and supported.
- [ ] Dependencies are installed from documented setup path.
- [ ] Protocol version is set and documented.
- [ ] Seed list for official run is fixed and documented.

## B. Execution

- [ ] Smoke benchmark passes:
  - [ ] python main.py --all --smoke-test --n-runs 1 --seed 42
- [ ] Full benchmark run completes for required domains.
- [ ] No silent skips of methods.

## C. Artifacts

For each domain output:

- [ ] run_manifest.json exists
- [ ] results_raw_by_run.json exists
- [ ] results_aggregated.json exists
- [ ] comparison_variants.csv exists
- [ ] comparison_canonical.csv exists

## D. Data Quality

- [ ] Schema validation passes for raw and aggregated outputs.
- [ ] Canonical table contains required columns.
- [ ] run_success_rate is present and non-empty.
- [ ] Primary metric mean/std are present for official rows.

## E. Reporting

- [ ] results/REPORT.md generated from latest run artifacts.
- [ ] Cross-domain statistical summary present in results/REPORT.md.
- [ ] Per-domain statistical significance sections present or explicitly marked unavailable.
- [ ] Limitations section updated.
- [ ] Known failures documented.
- [ ] Any protocol deviations documented.

## F. Repository Hygiene

- [ ] README commands are current.
- [ ] CONTRIBUTING guidance matches current workflow.
- [ ] Required CI checks are green: `benchmark-smoke / smoke`, `release-quality-gate / quality-gate`.
- [ ] CHANGELOG (or release notes) includes what changed.
- [ ] Tag name follows release convention.

## G. External Reproducibility (Recommended)

- [ ] One clean-machine reproduction completed.
- [ ] Reproduction notes captured.
- [ ] Runtime and hardware context recorded.
