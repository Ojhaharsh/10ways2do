# Project Status

Date: March 2026

## Current state

Core benchmark infrastructure is implemented:

- Multi-seed execution support
- Output schema validation
- Domain runners for A, B, C, D
- Smoke-test execution mode
- Report generation pipeline

## What is complete

- Documentation baseline updated
- Reproducibility utilities are integrated
- Artifact schema checks are integrated
- Smoke tests exist for benchmark artifact generation

## What is pending

- Complete full benchmark run with all required dependencies installed
- Verify final artifact quality across all domains
- Final pass on cross-domain report content

## Run commands

Full run:

```bash
python main.py --all --n-runs 5 --seed 42
```

Fast validation run:

```bash
python main.py --all --smoke-test --n-runs 1 --seed 42
```

Generate report from existing outputs:

```bash
python main.py --report
```

## Expected artifacts per domain

- run_manifest.json
- results_raw_by_run.json
- results_aggregated.json
- comparison_variants.csv
- comparison_canonical.csv

## Notes

If any optional package is missing, some approaches will fail to initialize. Install dependencies from requirements.txt before full runs.
