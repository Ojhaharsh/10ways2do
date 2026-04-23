# Methodology

## Objective

This benchmark compares 10 ML philosophy categories under one contract, then reports both quality and operational behavior. The goal is not to declare a universal winner, but to show when each modeling philosophy is strong, weak, and practical.

## Domain Coverage

The benchmark currently evaluates 11 domains:

1. Domain A: Information Extraction
2. Domain B: Anomaly Detection
3. Domain C: Recommendation
4. Domain D: Time Series Forecasting
5. Domain E: Tabular Decisioning
6. Domain F: Cyber Threat Hunting
7. Domain G: Operations Optimization
8. Domain H: Fraud Risk Assessment
9. Domain I: Capacity Planning
10. Domain J: Model Risk Monitoring
11. Domain K: Infrastructure Cost Forecasting

Each domain maps methods into 10 canonical categories for headline comparability while retaining full variant-level detail.

## Benchmark Philosophy Categories

The 10 categories are:

1. Rule-based / heuristic
2. Classical ML
3. Tree-based
4. Sequence / temporal neural
5. CNN or local pattern neural
6. Transformer or attention-first
7. Transfer / pretrained
8. Prompt or LLM-driven
9. Hybrid / ensemble
10. Systems / production wrapper

Category names are domain-adapted but normalized into canonical output rows.

## Experimental Protocol

### Seed and run policy

- Multi-run execution uses a deterministic seed list derived from base seed and run count.
- All approaches in the same run share the same split and seed context.
- Aggregation reports mean, standard deviation, and CI95 fields where available.

### Artifact contract

Each domain output must include:

- run_manifest.json
- results_raw_by_run.json
- results_aggregated.json
- comparison_variants.csv
- comparison_canonical.csv

The release gate rejects incomplete or malformed artifact bundles.

### Budget and reliability fields

Per run and per approach, the framework tracks:

- train_time_cap_seconds
- memory_cap_mb
- tuning_trials_cap
- out_of_budget
- success/failure with error payloads

Failures are retained in raw artifacts; they are not silently dropped.

## Metric Selection

Metrics are domain-specific but follow one rule: each domain has a clear primary ranking signal and supporting operational signals.

Examples:

- Classification-style domains: F1, Accuracy, ROC-AUC
- Forecasting/regression-style domains: RMSE, MAE, R2
- Recommendation domain: ranking quality metrics
- Systems view across domains: training time, inference behavior, run success rate

Lower-is-better and higher-is-better logic is domain-aware in report and snapshot generation.

## Statistical Reporting

For approaches with successful runs:

- Mean and standard deviation are computed on repeated runs.
- CI95 bounds are emitted in comparison tables.
- Significance payloads compare each approach to the best in-domain approach.

The significance payload includes p-value and effect-size-related fields used by report and release checks.

## Reproducibility and Release Process

### Local preflight

Use preflight before push:

```bash
python main.py --preflight --n-runs 1 --seed 42 --output-dir results_prepush
```

Preflight runs smoke benchmark, artifact validation, release gate, and critical test slices.

### Publish-ready flow

For release packaging:

```bash
python main.py --publish-ready-tag <tag> --n-runs 1 --seed 42
```

This produces report artifacts, validates release contract, and generates a versioned snapshot.

## Known Limitations

1. Most benchmark datasets are synthetic and may not capture all production shift patterns.
2. Some deep approaches are intentionally lightweight for broad reproducibility on standard hardware.
3. Default tuning budgets prioritize comparability and runtime control over domain-maximal performance.

## Extension Rules

### Adding a new approach

1. Keep canonical category mapping intact.
2. Emit required run-row fields and metric summary keys.
3. Ensure approach appears in variants table and can compete for canonical representative row.

### Adding a new domain

1. Implement data generator and run_all orchestration.
2. Integrate domain across CLI, loaders, reports, validators, release gates, and snapshots.
3. Add domain-specific tests and update domain-count assumptions in release tests.

## Interpretation Guidance

Use results as decision support, not a leaderboard absolute:

- Canonical table answers: best representative per philosophy category.
- Variant table answers: which concrete approach variant performs best and at what cost.
- Cross-domain outputs answer: which approaches are robust generalists versus domain specialists.