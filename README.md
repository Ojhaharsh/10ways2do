# 10ways2do

10ways2do is an ML comparison framework that evaluates different approach families on five domains:

- Domain A: Information Extraction
- Domain B: Anomaly Detection
- Domain C: Recommendation
- Domain D: Time Series Forecasting
- Domain E: Tabular Decisioning

The goal is not to claim a universal winner. The goal is to make trade-offs explicit: accuracy, latency, data needs, and implementation complexity.

## What this repo is

- A reproducible benchmark harness
- A codebase with multiple approach implementations per domain
- A practical base for experimentation and method comparison

## What this repo is not

- A production service
- A final academic benchmark paper
- A guaranteed fair comparison for all methods and all hyperparameter budgets

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows cmd:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## CLI usage

Run all domains:

```bash
python main.py --all
```

Run one domain:

```bash
python main.py --domain a
python main.py --domain b
python main.py --domain c
python main.py --domain d
python main.py --domain e
```

Run with reproducibility settings:

```bash
python main.py --all --n-runs 5 --seed 42
python main.py --domain c --seed-list 11 22 33
```

Run fast smoke mode:

```bash
python main.py --all --smoke-test --n-runs 1 --seed 42
```

Generate report from existing outputs:

```bash
python main.py --report
```

Validate artifact completeness and structure:

```bash
python main.py --validate
```

Run full release-gate checks:

```bash
python main.py --release-gate
```

Run the same quality-gate sequence used by CI:

```bash
python main.py --all --smoke-test --n-runs 1 --seed 42
python main.py --validate
python main.py --release-gate
```

## Output artifacts

Each domain writes outputs under results/domain_x:

- run_manifest.json
- results_raw_by_run.json
- results_aggregated.json
- comparison_variants.csv
- comparison_canonical.csv

Cross-domain report:

- results/REPORT.md

## Documentation

- Running guide: RUNNING.md
- Methodology: docs/METHODOLOGY.md
- Extended methodology: docs/COMPREHENSIVE_METHODOLOGY.md
- Architecture: docs/ARCHITECTURE.md
- Publication track: docs/PUBLICATION_TRACK.md
- Insights: docs/INSIGHTS.md
- API notes: docs/API.md
- Contributing: CONTRIBUTING.md
- Branch protection policy: docs/BRANCH_PROTECTION_POLICY.md
- Roadmap: ROADMAP.md

## Tests

```bash
pytest tests -v
```

## Known limitations

- Some domains contain multiple variants inside one conceptual category.
- Dependency footprints are large for deep-learning and graph-heavy methods.
- Results quality depends on your hardware and installed package versions.

## License

MIT (see LICENSE)
