# Complete Running Instructions

This document contains **ALL** instructions for setting up, running, testing, and using the ML Philosophy Benchmark project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure Overview](#project-structure-overview)
4. [Running Benchmarks](#running-benchmarks)
5. [Running Individual Domains](#running-individual-domains)
6. [Running Individual Approaches](#running-individual-approaches)
7. [Running Tests](#running-tests)
8. [Generating Reports](#generating-reports)
9. [Creating Visualizations](#creating-visualizations)
10. [Using the Python API](#using-the-python-api)
11. [Command Line Reference](#command-line-reference)
12. [Configuration Options](#configuration-options)
13. [Troubleshooting](#troubleshooting)
14. [Examples](#examples)

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU optional (for deep learning approaches)

### Check Python Version

```bash
python --version
# or
python3 --version
```

### Check pip Version

```bash
pip --version
# or
pip3 --version
```

---

## Installation

### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/username/ml-philosophy-benchmark.git
cd ml-philosophy-benchmark

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install base dependencies
pip install -e .
```

### Option 2: Full Installation (All Features)

```bash
# Clone and setup virtual environment (same as above)
git clone https://github.com/username/ml-philosophy-benchmark.git
cd ml-philosophy-benchmark
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with all optional dependencies
pip install -e ".[full]"
```

### Option 3: Development Installation

```bash
# For contributors and developers
pip install -e ".[full,dev]"
```

### Option 4: Install from requirements.txt

```bash
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
# Check if installation worked
python -c "from src.core.base_model import BaseApproach; print('Installation successful!')"
```

---

## Running Benchmarks

### Run All Domains (Complete Benchmark)

```bash
# Run everything
python main.py --all

# This will:
# 1. Run all 10 approaches on Domain A (Information Extraction)
# 2. Run all 10 approaches on Domain B (Anomaly Detection)
# 3. Run all 10 approaches on Domain C (Recommendation)
# 4. Run all 10 approaches on Domain D (Time Series)
# 5. Run all 10 approaches on Domain E (Tabular Decisioning)
# 6. Generate comparison reports
# 7. Save results to results/ directory
```

### Run All Domains with Custom Parameters

```bash
# With custom training size
python main.py --all --n-train 5000 --n-test 1000

# With custom output directory
python main.py --all --output-dir my_results
```

### Run All Domains (Python API)

```python
from main import run_all, generate_report

# Run all benchmarks
results = run_all()

# Generate report
generate_report()
```

---

## Running Individual Domains

### Domain A: Information Extraction

```bash
# Command line
python main.py --domain a

# or
python main.py --domain ie

# With custom parameters
python main.py --domain a --n-train 2000 --n-test 500
```

```python
# Python API
from src.domain_a_information_extraction.run_all import run_all_approaches

results = run_all_approaches(
    n_train=2000,
    n_val=500,
    n_test=500,
    save_results=True,
    output_dir="results/domain_a"
)

print(results['comparison'])
```

### Domain B: Anomaly Detection

```bash
# Command line
python main.py --domain b

# or
python main.py --domain anomaly

# With custom parameters
python main.py --domain b --n-train 5000 --n-test 1000
```

```python
# Python API
from src.domain_b_anomaly_detection.run_all import run_all_approaches

results = run_all_approaches(
    n_train=5000,
    n_val=1000,
    n_test=1000,
    save_results=True,
    output_dir="results/domain_b"
)

print(results['comparison'])
```

### Domain C: Recommendation

```bash
# Command line
python main.py --domain c

# or
python main.py --domain rec

# With custom parameters
python main.py --domain c --n-train 1000 --n-test 200
```

```python
# Python API
from src.domain_c_recommendation.run_all import run_all_approaches

results = run_all_approaches(
    n_users=500,
    n_items=200,
    save_results=True,
    output_dir="results/domain_c"
)

print(results['comparison'])
```

### Domain D: Time Series

```bash
# Command line
python main.py --domain d

# or
python main.py --domain ts

# With custom parameters
python main.py --domain d --n-train 2000 --n-test 500
```

```python
# Python API
from src.domain_d_time_series.run_all import run_all_approaches

results = run_all_approaches(
    n_samples=2000,
    forecast_horizon=24,
    lookback=168,
    save_results=True,
    output_dir="results/domain_d"
)

print(results['comparison'])
```

### Domain E: Tabular Decisioning

```bash
# Command line
python main.py --domain e

# or
python main.py --domain tabular

# With custom parameters
python main.py --domain e --n-train 6000 --n-test 1200
```

```python
# Python API
from src.domain_e_tabular_decisioning.run_all import run_all_approaches

results = run_all_approaches(
    n_samples=6000,
    n_features=24,
    save_results=True,
    output_dir="results/domain_e"
)

print(results['canonical_comparison'])
```

---

## Running Individual Approaches

### Domain A: Information Extraction Approaches

```python
from src.domain_a_information_extraction.data_generator import create_ie_dataset, get_ie_fields
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.domain_a_information_extraction.approach_02_classical_ml import ClassicalMLIE
from src.domain_a_information_extraction.approach_03_tree_based import TreeBasedIE
from src.domain_a_information_extraction.approach_04_rnn_lstm import RNNLSTMIE
from src.domain_a_information_extraction.approach_05_cnn_sequence import CNNIE
from src.domain_a_information_extraction.approach_06_transformer_scratch import TransformerScratchIE
from src.domain_a_information_extraction.approach_07_pretrained_transformer import PretrainedTransformerIE
from src.domain_a_information_extraction.approach_08_prompt_llm import PromptLLMIE
from src.domain_a_information_extraction.approach_09_hybrid import HybridIE
from src.core.metrics import compute_ie_metrics

# Generate dataset
dataset = create_ie_dataset(n_train=1000, n_val=200, n_test=200)

# Example: Run Rule-Based approach
model = RuleBasedIE()
model.train(dataset['train']['X'], dataset['train']['y'])
predictions = model.predict(dataset['test']['X'])

# Compute metrics
metrics = compute_ie_metrics(
    dataset['test']['y'], 
    predictions, 
    get_ie_fields()
)
print(metrics)
```

### Domain B: Anomaly Detection Approaches

```python
from src.domain_b_anomaly_detection.data_generator import create_anomaly_dataset
from src.domain_b_anomaly_detection.approach_01_statistical import StatisticalAnomalyDetector
from src.domain_b_anomaly_detection.approach_02_distance_based import KNNAnomalyDetector, LOFDetector
from src.domain_b_anomaly_detection.approach_03_tree_based import IsolationForestDetector
from src.domain_b_anomaly_detection.approach_04_autoencoder import AutoencoderDetector
from src.domain_b_anomaly_detection.approach_05_rnn_lstm import LSTMAnomalyDetector
from src.domain_b_anomaly_detection.approach_06_transformer import TransformerAnomalyDetector
from src.domain_b_anomaly_detection.approach_07_graph_based import GraphAnomalyDetector
from src.domain_b_anomaly_detection.approach_08_ensemble import EnsembleAnomalyDetector
from src.domain_b_anomaly_detection.approach_09_hybrid import HybridAnomalyDetector
from src.core.metrics import compute_anomaly_metrics

# Generate dataset
dataset = create_anomaly_dataset(n_train=5000, n_val=1000, n_test=1000)

# Example: Run Isolation Forest
model = IsolationForestDetector({'n_estimators': 100})
model.train(dataset['train']['X'], dataset['train']['y'])
predictions = model.predict(dataset['test']['X'])
scores = model.score(dataset['test']['X'])

# Compute metrics
metrics = compute_anomaly_metrics(
    dataset['test']['y'], 
    predictions, 
    scores
)
print(metrics)
```

### Domain C: Recommendation Approaches

```python
from src.domain_c_recommendation.data_generator import create_recommendation_dataset
from src.domain_c_recommendation.approach_01_popularity import PopularityRecommender
from src.domain_c_recommendation.approach_02_collaborative_filtering import UserBasedCF, ItemBasedCF
from src.domain_c_recommendation.approach_03_content_based import ContentBasedRecommender
from src.domain_c_recommendation.approach_04_matrix_factorization import SVDRecommender, ALSRecommender
from src.domain_c_recommendation.approach_05_deep_learning import NCFRecommender
from src.domain_c_recommendation.approach_06_sequence_based import SequentialRecommender
from src.domain_c_recommendation.approach_07_graph_neural import GraphNeuralRecommender
from src.domain_c_recommendation.approach_08_transformer import TransformerRecommender
from src.domain_c_recommendation.approach_09_hybrid import HybridRecommender
from src.core.metrics import compute_ranking_metrics

# Generate dataset
dataset = create_recommendation_dataset(n_users=500, n_items=200)

# Example: Run SVD
model = SVDRecommender({'n_factors': 50})
model.train(dataset['train_matrix'], dataset['train_interactions'])

# Get recommendations for test users
test_users = list(dataset['test_items'].keys())
predictions = model.predict(test_users, k=10)
ground_truth = [dataset['test_items'][u] for u in test_users]

# Compute metrics
metrics = compute_ranking_metrics(ground_truth, predictions, k_values=[5, 10])
print(metrics)
```

### Domain D: Time Series Approaches

```python
from src.domain_d_time_series.data_generator import create_timeseries_dataset
from src.domain_d_time_series.approach_01_statistical import ARIMAForecaster
from src.domain_d_time_series.approach_02_exponential_smoothing import ExpSmoothingForecaster
from src.domain_d_time_series.approach_03_tree_based import TreeBasedForecaster
from src.domain_d_time_series.approach_04_rnn_lstm import LSTMForecaster
from src.domain_d_time_series.approach_05_cnn_temporal import TCNForecaster
from src.domain_d_time_series.approach_06_transformer import TransformerForecaster
from src.domain_d_time_series.approach_07_neural_prophet import ProphetStyleForecaster
from src.domain_d_time_series.approach_08_ensemble import EnsembleForecaster
from src.domain_d_time_series.approach_09_hybrid import HybridForecaster
from src.core.metrics import compute_timeseries_metrics

# Generate dataset
dataset = create_timeseries_dataset(
    n_samples=2000, 
    lookback=48, 
    forecast_horizon=12
)

# Example: Run LSTM
model = LSTMForecaster({'epochs': 20, 'hidden_dim': 64})
model.train(
    dataset['train']['X'], 
    dataset['train']['y'],
    forecast_horizon=12
)
predictions = model.predict(dataset['test']['X'])

# Compute metrics
y_test_flat = dataset['test']['y'].reshape(-1)
y_pred_flat = predictions.reshape(-1)

metrics = compute_timeseries_metrics(y_test_flat, y_pred_flat)
print(metrics)
```

---

## Running Tests

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_domain_a.py -v
pytest tests/test_domain_b.py -v
pytest tests/test_domain_c.py -v
pytest tests/test_domain_d.py -v
```

### Run Specific Test Classes

```bash
# Run specific test class
pytest tests/test_domain_a.py::TestRuleBasedIE -v

# Run specific test method
pytest tests/test_domain_a.py::TestRuleBasedIE::test_email_extraction -v
```

### Run Tests with Markers

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only integration tests
pytest tests/ -v -m "integration"
```

### Run Tests in Parallel

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -v -n auto
```

### Generate Coverage Report

```bash
# HTML report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Terminal report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Generating Reports

### Generate Full Report

```bash
# Command line
python main.py --report
```

```python
# Python API
from src.analysis.report_generator import ReportGenerator

generator = ReportGenerator(results_dir="results")

# Generate and save full report
generator.save_report("results/REPORT.md")

# Generate domain-specific report
domain_a_report = generator.generate_domain_report('domain_a')
print(domain_a_report)

# Generate full cross-domain report
full_report = generator.generate_full_report()
print(full_report)
```

### Generate Cross-Domain Analysis

```python
from src.analysis.cross_domain_analysis import CrossDomainAnalyzer

analyzer = CrossDomainAnalyzer(results_dir="results")
analyzer.load_results()

# Compare philosophies across domains
philosophy_df = analyzer.compare_philosophies()
print(philosophy_df)

# Analyze trade-offs
trade_offs = analyzer.analyze_trade_offs()
print(trade_offs)

# Generate insights
insights = analyzer.generate_insights()
print(insights)

# Create complete summary
summary = analyzer.create_summary_report()
print(summary)
```

---

## Creating Visualizations

### Basic Plots

```python
from src.analysis.visualization import Visualizer
import json

# Load results
with open('results/domain_a/results.json') as f:
    results = json.load(f)

viz = Visualizer()

# Bar chart comparison
viz.plot_comparison(
    results=results,
    metric='overall_exact_match',
    title='Information Extraction: Exact Match Comparison',
    save_path='plots/ie_comparison.png'
)

# Trade-off scatter plot
viz.plot_trade_off(
    results=results,
    x_metric='training_time',
    y_metric='overall_exact_match',
    title='Accuracy vs Training Time',
    save_path='plots/ie_tradeoff.png'
)
```

### Advanced Plots

```python
from src.analysis.visualization import Visualizer
import json

with open('results/domain_b/results.json') as f:
    results = json.load(f)

viz = Visualizer()

# Radar chart
viz.plot_radar(
    results=results,
    metrics=['precision', 'recall', 'f1', 'roc_auc'],
    title='Anomaly Detection: Multi-Metric Comparison',
    save_path='plots/anomaly_radar.png'
)

# Data efficiency curves
viz.plot_data_efficiency(
    results=results,
    save_path='plots/anomaly_efficiency.png'
)

# Robustness curves
viz.plot_robustness(
    results=results,
    save_path='plots/anomaly_robustness.png'
)
```

### Create All Plots for a Domain

```python
from src.analysis.visualization import Visualizer
import json

with open('results/domain_a/results.json') as f:
    results = json.load(f)

viz = Visualizer()

# Create all standard plots
viz.create_all_plots(
    results=results,
    output_dir='plots/domain_a',
    domain='domain_a'
)
```

---

## Using the Python API

### Complete Workflow Example

```python
# Step 1: Import required modules
from src.domain_a_information_extraction.data_generator import create_ie_dataset, get_ie_fields
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.domain_a_information_extraction.approach_02_classical_ml import ClassicalMLIE
from src.domain_a_information_extraction.approach_09_hybrid import HybridIE
from src.core.metrics import compute_ie_metrics
from src.core.evaluation import Evaluator
import pandas as pd

# Step 2: Generate dataset
print("Generating dataset...")
dataset = create_ie_dataset(n_train=1000, n_val=200, n_test=200)

# Step 3: Initialize approaches
approaches = [
    RuleBasedIE(),
    ClassicalMLIE(),
    HybridIE()
]

# Step 4: Evaluate each approach
results = []
for approach in approaches:
    print(f"Evaluating {approach.name}...")
    
    # Train
    approach.train(
        dataset['train']['X'], 
        dataset['train']['y'],
        dataset['val']['X'],
        dataset['val']['y']
    )
    
    # Predict
    predictions = approach.predict(dataset['test']['X'])
    
    # Compute metrics
    metrics = compute_ie_metrics(
        dataset['test']['y'],
        predictions,
        get_ie_fields()
    )
    
    # Store results
    results.append({
        'Approach': approach.name,
        'Exact Match': metrics['overall_exact_match'],
        'Partial Match': metrics['overall_partial_match'],
        'Philosophy': approach.get_philosophy()['mental_model']
    })

# Step 5: Compare results
df = pd.DataFrame(results)
print("\nResults:")
print(df.to_string(index=False))

# Step 6: Get philosophy insights
print("\nPhilosophy Insights:")
for approach in approaches:
    philosophy = approach.get_philosophy()
    print(f"\n{approach.name}:")
    print(f"  Mental Model: {philosophy['mental_model']}")
    print(f"  Strengths: {philosophy['strengths']}")
    print(f"  Weaknesses: {philosophy['weaknesses']}")
```

### Using the Evaluator Framework

```python
from src.core.evaluation import Evaluator
from src.domain_a_information_extraction.data_generator import create_ie_dataset
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.domain_a_information_extraction.approach_02_classical_ml import ClassicalMLIE

# Create dataset
dataset = create_ie_dataset(n_train=1000, n_val=200, n_test=200)

# Initialize evaluator
evaluator = Evaluator(domain='ie')

# Prepare approaches
approaches = [RuleBasedIE(), ClassicalMLIE()]

# Evaluate all
results = evaluator.evaluate_all(
    approaches=approaches,
    X_train=dataset['train']['X'],
    y_train=dataset['train']['y'],
    X_val=dataset['val']['X'],
    y_val=dataset['val']['y'],
    X_test=dataset['test']['X'],
    y_test=dataset['test']['y'],
    fields=dataset['fields']
)

# Get comparison table
comparison_df = evaluator.compare()
print(comparison_df)

# Save results
evaluator.save_results('results/my_evaluation.json')

# Generate markdown report
report = evaluator.generate_report()
print(report)
```

---

## Command Line Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `python main.py --all` | Run all domains |
| `python main.py --domain a` | Run Domain A (Information Extraction) |
| `python main.py --domain b` | Run Domain B (Anomaly Detection) |
| `python main.py --domain c` | Run Domain C (Recommendation) |
| `python main.py --domain d` | Run Domain D (Time Series) |
| `python main.py --report` | Generate report from existing results |
| `python main.py --help` | Show help message |

### Domain Aliases

| Alias | Domain |
|-------|--------|
| `a`, `ie` | Information Extraction |
| `b`, `anomaly` | Anomaly Detection |
| `c`, `rec` | Recommendation |
| `d`, `ts` | Time Series |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-train` | 1000 | Training set size |
| `--n-test` | 200 | Test set size |
| `--output-dir` | `results` | Output directory |

### Examples

```bash
# Run Domain A with 5000 training samples
python main.py --domain a --n-train 5000

# Run Domain B with custom output directory
python main.py --domain b --output-dir my_results/anomaly

# Run all domains with more data
python main.py --all --n-train 10000 --n-test 2000

# Generate report only (after running benchmarks)
python main.py --report
```

---

## Configuration Options

### Approach-Specific Configuration

Each approach accepts a configuration dictionary:

```python
# Rule-Based (no config needed)
model = RuleBasedIE()

# Classical ML
model = ClassicalMLIE({'max_iter': 1000})

# Tree-Based
model = TreeBasedIE({'n_estimators': 100, 'max_depth': 10})

# LSTM
model = LSTMForecaster({
    'hidden_dim': 128,
    'num_layers': 2,
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.001
})

# Transformer
model = TransformerForecaster({
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'epochs': 50
})

# Autoencoder
model = AutoencoderDetector({
    'hidden_dims': [64, 32],
    'latent_dim': 16,
    'epochs': 100
})

# Pretrained Transformer
model = PretrainedTransformerIE({
    'model_name': 'bert-base-uncased',
    'epochs': 3,
    'batch_size': 16,
    'lr': 2e-5
})

# LLM/Prompt
model = PromptLLMIE({
    'model': 'gpt-3.5-turbo',
    'temperature': 0.0,
    'simulate': True  # Use simulation instead of API
})
```

### Global Configuration

Edit `config/settings.py` for global settings:

```python
from config.settings import Config, get_config

config = get_config()

# Data settings
config.data.train_size = 10000
config.data.test_size = 2000
config.data.seed = 42

# Model settings
config.model.batch_size = 32
config.model.max_epochs = 50
config.model.learning_rate = 0.001

# Evaluation settings
config.evaluation.compute_latency = True
config.evaluation.compute_memory = True
config.evaluation.n_runs = 5
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'

# Solution: Install the package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. CUDA/GPU Issues

```bash
# Error: CUDA out of memory

# Solution: Reduce batch size or use CPU
model = LSTMForecaster({'batch_size': 16})

# Or force CPU
import torch
torch.cuda.is_available = lambda: False
```

#### 3. Missing Dependencies

```bash
# Error: ModuleNotFoundError: No module named 'transformers'

# Solution: Install full dependencies
pip install -e ".[full]"

# Or install specific package
pip install transformers
```

#### 4. Memory Issues

```python
# Solution: Use smaller dataset sizes
dataset = create_ie_dataset(n_train=500, n_val=100, n_test=100)

# Or process in batches
for batch in batched(data, batch_size=100):
    process(batch)
```

#### 5. Slow Training

```python
# Solution: Reduce epochs or model complexity
model = LSTMForecaster({
    'epochs': 10,      # Reduce from 50
    'hidden_dim': 32   # Reduce from 128
})
```

#### 6. statsmodels Not Found

```bash
# Error: ModuleNotFoundError: No module named 'statsmodels'

# Solution:
pip install statsmodels
```

#### 7. XGBoost Not Found

```bash
# Error: ModuleNotFoundError: No module named 'xgboost'

# Solution:
pip install xgboost
```

### Getting Help

```bash
# Show CLI help
python main.py --help

# Check installation
python -c "import src; print('OK')"

# Run tests to verify everything works
pytest tests/ -v --tb=short
```

---

## Examples

### Example 1: Quick Start

```bash
# Run a quick benchmark on Domain A
python main.py --domain a --n-train 500 --n-test 100
```

### Example 2: Full Benchmark

```bash
# Run complete benchmark
python main.py --all --n-train 5000 --n-test 1000
python main.py --report
```

### Example 3: Compare Two Approaches

```python
from src.domain_a_information_extraction.data_generator import create_ie_dataset, get_ie_fields
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.domain_a_information_extraction.approach_07_pretrained_transformer import PretrainedTransformerIE
from src.core.metrics import compute_ie_metrics

# Dataset
dataset = create_ie_dataset(n_train=500, n_val=100, n_test=100)

# Rule-Based
rule_model = RuleBasedIE()
rule_model.train(dataset['train']['X'], dataset['train']['y'])
rule_pred = rule_model.predict(dataset['test']['X'])
rule_metrics = compute_ie_metrics(dataset['test']['y'], rule_pred, get_ie_fields())

# Pretrained Transformer
bert_model = PretrainedTransformerIE({'epochs': 2})
bert_model.train(dataset['train']['X'], dataset['train']['y'])
bert_pred = bert_model.predict(dataset['test']['X'])
bert_metrics = compute_ie_metrics(dataset['test']['y'], bert_pred, get_ie_fields())

# Compare
print(f"Rule-Based: {rule_metrics['overall_exact_match']:.4f}")
print(f"BERT: {bert_metrics['overall_exact_match']:.4f}")
```

### Example 4: Data Efficiency Analysis

```python
from src.domain_a_information_extraction.data_generator import create_ie_dataset, get_ie_fields
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.core.metrics import compute_ie_metrics

# Test with different training sizes
sizes = [100, 500, 1000, 2000, 5000]
results = []

for size in sizes:
    dataset = create_ie_dataset(n_train=size, n_val=100, n_test=200)
    
    model = RuleBasedIE()
    model.train(dataset['train']['X'], dataset['train']['y'])
    predictions = model.predict(dataset['test']['X'])
    
    metrics = compute_ie_metrics(dataset['test']['y'], predictions, get_ie_fields())
    results.append({
        'train_size': size,
        'exact_match': metrics['overall_exact_match']
    })
    print(f"Train size {size}: {metrics['overall_exact_match']:.4f}")
```

### Example 5: Robustness Analysis

```python
from src.domain_b_anomaly_detection.data_generator import create_anomaly_dataset, add_noise_to_data
from src.domain_b_anomaly_detection.approach_03_tree_based import IsolationForestDetector
from src.core.metrics import compute_anomaly_metrics

# Create dataset
dataset = create_anomaly_dataset(n_train=2000, n_test=500)

# Train model
model = IsolationForestDetector()
model.train(dataset['train']['X'], dataset['train']['y'])

# Test with different noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

for noise in noise_levels:
    X_noisy = add_noise_to_data(dataset['test']['X'], noise)
    predictions = model.predict(X_noisy)
    metrics = compute_anomaly_metrics(dataset['test']['y'], predictions)
    print(f"Noise {noise}: F1={metrics['f1']:.4f}")
```

### Example 6: Latency Measurement

```python
from src.domain_a_information_extraction.data_generator import create_ie_dataset
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
import time

dataset = create_ie_dataset(n_train=1000, n_test=100)

model = RuleBasedIE()
model.train(dataset['train']['X'], dataset['train']['y'])

# Measure latency
latencies = []
for text in dataset['test']['X']:
    start = time.perf_counter()
    model.predict([text])
    latencies.append((time.perf_counter() - start) * 1000)

import numpy as np
print(f"Latency p50: {np.percentile(latencies, 50):.2f}ms")
print(f"Latency p95: {np.percentile(latencies, 95):.2f}ms")
print(f"Latency p99: {np.percentile(latencies, 99):.2f}ms")
```

---

## Quick Reference Card

```bash
# Installation
pip install -e ".[full]"

# Run all benchmarks
python main.py --all

# Run single domain
python main.py --domain a|b|c|d|e

# Generate report
python main.py --report

# Validate artifacts
python main.py --validate

# Run full release gate
python main.py --release-gate

# Local quality-gate sequence (matches CI intent)
python main.py --all --smoke-test --n-runs 1 --seed 42
python main.py --validate
python main.py --release-gate

# Run tests
pytest tests/ -v

# Get help
python main.py --help
```

---

Happy benchmarking.
```
