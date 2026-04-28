# 10ways2do

10ways2do is an ML comparison framework that evaluates different approach families on **sixteen domains**:

## Original Domains (A-K)

- Domain A: Information Extraction
- Domain B: Anomaly Detection
- Domain C: Recommendation
- Domain D: Time Series Forecasting
- Domain E: Tabular Decisioning
- Domain F: Cyber Threat Hunting
- Domain G: Operations Optimization
- Domain H: Fraud Risk Assessment
- Domain I: Capacity Planning
- Domain J: Model Risk Monitoring
- Domain K: Infrastructure Cost Forecasting

## New Domains (L-P)

- Domain L: Computer Vision (10 CNN architectures)
- Domain M: Graph Neural Networks (GCN, GAT, GraphSAGE, etc.)
- Domain N: Few-Shot Learning (Prototypical Networks, MAML, etc.)
- Domain O: Reinforcement Learning (Bandit algorithms, UCB, Thompson, etc.)
- Domain P: Multimodal Learning (CLIP, VisualBERT, LLaVA, etc.)

## Key Features

- **160 Method Comparisons**: 10 approaches × 16 domains
- **Advanced Metrics**: Cost tracking (latency, memory, FLOPS), fairness assessment, robustness testing
- **Leaderboards**: Per-domain and cross-domain rankings with statistical significance
- **Statistical Testing**: Welch's t-test, Mann-Whitney U, bootstrap confidence intervals
- **Distributed Running**: Auto-scaling parallel execution with job priority queues
- **Release Governance**: Mandatory benchmark cards, artifact validation, CI/CD gates
- **Reproducibility**: Deterministic seeding, multi-run validation (n_runs configurable)

The goal is not to claim a universal winner. The goal is to make trade-offs explicit: accuracy, latency, data needs, implementation complexity, fairness, and robustness.

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
python main.py --domain f
python main.py --domain g
python main.py --domain h
python main.py --domain i
python main.py --domain j
python main.py --domain k
python main.py --domain l    # Computer Vision
python main.py --domain m    # Graph Neural Networks
python main.py --domain n    # Few-Shot Learning
python main.py --domain o    # Reinforcement Learning
python main.py --domain p    # Multimodal Learning
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

Generate leaderboard and dashboard:

```bash
python main.py --leaderboard
python main.py --dashboard
```

Generate statistical significance report:

```bash
python main.py --significance-report
```

Generate advanced metrics (cost, fairness, robustness):

```bash
python main.py --advanced-metrics
```

```bash
python main.py --report
```

Generate benchmark card from existing outputs:

```bash
python main.py --benchmark-card
python main.py --benchmark-card --benchmark-card-output-dir releases/v1.2
```

Generate strategy playbook from cross-domain frontier:

```bash
python main.py --strategy-playbook
```

Run what-if policy simulation with custom priorities and constraints:

```bash
python main.py --simulate-policy --policy-name latency-ops --w-speed 0.60 --w-quality 0.20 --w-resilience 0.15 --w-consistency 0.05 --min-resilience 0.70
```

Run policy weight optimization to auto-discover best policy under constraints:

```bash
python main.py --optimize-policy --policy-name resilient-prod --opt-objective max_coverage --weight-step 0.25 --min-resilience 0.80
```

Run multi-objective policy frontier optimization (non-dominated policies + stability bands):

```bash
python main.py --optimize-policy-frontier --policy-name policy-frontier --weight-step 0.25 --min-resilience 0.80
```

Run diversity-constrained frontier optimization (keeps multiple archetypes in the frontier):

```bash
python main.py --optimize-policy-frontier --policy-name policy-frontier --weight-step 0.25 --frontier-min-archetypes 3 --frontier-balance-threshold 0.10
```

Validate artifact completeness and structure:

```bash
python main.py --validate
```

Run full release-gate checks:

```bash
python main.py --release-gate
```

Create a versioned release snapshot:

```bash
python main.py --snapshot-tag v1.1
```

Restore artifacts from a versioned snapshot:

```bash
python main.py --restore-snapshot v1.1
```

Restore-fidelity check (snapshot -> restore -> gate):

```bash
python main.py --snapshot-tag restore-check
python main.py --restore-snapshot restore-check --output-dir restored_results
python main.py --release-gate --output-dir restored_results
```

Run full publish-ready pipeline (smoke + report + gates + snapshot + summary):

```bash
python main.py --publish-ready-tag v1.1
```

Run local preflight before pushing (smoke + validate + release-gate + critical tests):

```bash
python main.py --preflight
python main.py --preflight --preflight-full-tests
```

Publish-ready with retention policy (keep latest 30 nightly snapshots, protect stable tags):

```bash
python main.py --publish-ready-tag nightly-20260331 --prune-nightly-keep 30 --protect-tag-prefixes v stable release
```

Run the same quality-gate sequence used by CI:

```bash
python main.py --all --smoke-test --n-runs 1 --seed 42
python main.py --validate
python main.py --release-gate
python main.py --snapshot-tag v1.1
python main.py --publish-ready-tag v1.1
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
- results/CROSS_DOMAIN_FRONTIER.json
- results/STRATEGY_PLAYBOOK.json
- results/STRATEGY_PLAYBOOK.md
- results/POLICY_SIMULATION_<policy-name>.json
- results/POLICY_SIMULATION_<policy-name>.md
- results/POLICY_OPTIMIZATION_<policy-name>.json
- results/POLICY_OPTIMIZATION_<policy-name>.md
- results/POLICY_FRONTIER_<policy-name>.json
- results/POLICY_FRONTIER_<policy-name>.md
- results/BENCHMARK_CARD.json
- results/BENCHMARK_CARD.md

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

## CI automation

- PR/merge quality gates:
	- `.github/workflows/benchmark-smoke.yml`
	- `.github/workflows/release-quality-gate.yml`
- Nightly publish-ready snapshot:
	- `.github/workflows/nightly-publish-ready.yml`

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
