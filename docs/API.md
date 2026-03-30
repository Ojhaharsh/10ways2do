# API Reference

## Core Classes

### BaseApproach

Base class for all modeling approaches

```python
from src.core.base_model import BaseApproach

class MyApproach(BaseApproach):
    def __init__(self, config=None):
        super().__init__("My Approach Name", config)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        # Training logic here
        self.is_trained = True
    
    def predict(self, X):
        """Make predictions."""
        # Inference logic here
        return predictions
    
    def get_philosophy(self):
        """Return approach philosophy."""
        return {
            'mental_model': 'Description of the approach',
            'inductive_bias': 'What assumptions it makes',
            'strengths': 'Where it excels',
            'weaknesses': 'Where it struggles',
            'best_for': 'Ideal use cases'
        }
```

### ModelMetrics

Container for all model metrics

```python
from src.core.base_model import ModelMetrics

metrics = ModelMetrics()
metrics.primary_metrics = {'accuracy': 0.95, 'f1': 0.92}
metrics.training_time = 10.5
metrics.inference_latency_p95 = 5.2
metrics.memory_inference_mb = 256.0
```

### MetricsComputer

Unified metrics computation across domains

```python
from src.core.metrics import MetricsComputer

# Information Extraction
metrics = MetricsComputer.compute(
    domain='ie',
    y_true=y_true,
    y_pred=y_pred,
    fields=['name', 'email', 'phone']
)

# Anomaly Detection
metrics = MetricsComputer.compute(
    domain='anomaly',
    y_true=y_true,
    y_pred=y_pred,
    y_scores=anomaly_scores
)

# Recommendation
metrics = MetricsComputer.compute(
    domain='recommendation',
    y_true=y_true,
    y_pred=y_pred,
    k_values=[5, 10, 20]
)

# Time Series
metrics = MetricsComputer.compute(
    domain='timeseries',
    y_true=y_true,
    y_pred=y_pred,
    naive_pred=naive_baseline
)
```

### Evaluator

Comprehensive evaluation framework

```python
from src.core.evaluation import Evaluator

# Initialize evaluator for a domain
evaluator = Evaluator(domain='ie')

# Evaluate single approach
result = evaluator.evaluate_approach(
    approach=my_approach,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    add_noise_fn=noise_function
)

# Evaluate all approaches
results = evaluator.evaluate_all(
    approaches=list_of_approaches,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

# Compare results
comparison_df = evaluator.compare()

# Save results
evaluator.save_results('results/evaluation.json')

# Generate report
report = evaluator.generate_report()
```


## Data Generators

### Information Extraction

```python
from src.domain_a_information_extraction.data_generator import (
    create_ie_dataset,
    get_ie_fields,
    ResumeGenerator
)

# Create full dataset
dataset = create_ie_dataset(
    n_train=5000,
    n_val=1000,
    n_test=1000,
    seed=42
)

# Access data
X_train = dataset['train']['X']  # List of text documents
y_train = dataset['train']['y']  # List of label dictionaries
fields = dataset['fields']       # List of field names

# Generate single resume
generator = ResumeGenerator(seed=42)
text, structured_data = generator.generate_resume()
```

### Anomaly Detection

```python
from src.domain_b_anomaly_detection.data_generator import (
    create_anomaly_dataset,
    add_noise_to_data,
    TimeSeriesGenerator
)

# Create dataset
dataset = create_anomaly_dataset(
    n_train=8000,
    n_val=1000,
    n_test=1000,
    n_features=5,
    anomaly_ratio=0.05,
    data_type='timeseries',
    seed=42
)

# Access data
X_train = dataset['train']['X']  # numpy array (n_samples, n_features)
y_train = dataset['train']['y']  # numpy array (n_samples,) binary labels
```

### Recommendation

```python
from src.domain_c_recommendation.data_generator import (
    create_recommendation_dataset,
    RecommendationGenerator
)

# Create dataset
dataset = create_recommendation_dataset(
    n_users=1000,
    n_items=500,
    sparsity=0.95,
    test_ratio=0.2,
    seed=42
)

# Access data
train_matrix = dataset['train_matrix']             # (n_users, n_items) rating matrix
train_interactions = dataset['train_interactions'] # List of (user, item, rating)
test_items = dataset['test_items']                 # Dict: user_id -> list of relevant items
```

### Time Series

```python
from src.domain_d_time_series.data_generator import (
    create_timeseries_dataset,
    add_noise_to_series,
    TimeSeriesGenerator
)

# Create dataset
dataset = create_timeseries_dataset(
    n_samples=2000,
    n_features=1,
    forecast_horizon=24,
    lookback=168,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
)

# Access data
X_train = dataset['train']['X']    # (n_samples, lookback, n_features)
y_train = dataset['train']['y']    # (n_samples, horizon, n_features)
raw_series = dataset['raw_series'] # Original time series
```

### Tabular Decisioning

```python
from src.domain_e_tabular_decisioning.data_generator import create_tabular_decision_dataset

# Create dataset
dataset = create_tabular_decision_dataset(
    n_samples=6000,
    n_features=24,
    class_weight_positive=0.18,
)

# Access data
X_train = dataset['train']['X']  # (n_samples, n_features)
y_train = dataset['train']['y']  # binary labels
```

## Running Benchmarks

### Run Single Domain

```python
from src.domain_a_information_extraction.run_all import run_all_approaches

results = run_all_approaches(
    n_train=2000,
    n_val=500,
    n_test=500,
    save_results=True,
    output_dir="results/domain_a"
)

# Access results
comparison_df = results['comparison']
detailed_results = results['results']
```

### Run All Domains

```python
from main import run_all, generate_report

# Run all benchmarks
all_results = run_all()
# Generate comprehensive report
generate_report()
```

### Command Line Interface

```bash
# Run all domains
python main.py --all

# Run specific domain
python main.py --domain a
python main.py --domain b
python main.py --domain c
python main.py --domain d
python main.py --domain e

# Custom parameters
python main.py --domain a --n-train 5000 --n-test 1000

# Generate report only
python main.py --report
```

## Analysis Tools

### Cross-Domain Analysis

```python
from src.analysis.cross_domain_analysis import CrossDomainAnalyzer

analyzer = CrossDomainAnalyzer(results_dir="results")
analyzer.load_results()

# Compare philosophies
philosophy_df = analyzer.compare_philosophies()

# Analyze trade-offs
trade_offs = analyzer.analyze_trade_offs()

# Generate insights
insights = analyzer.generate_insights()

# Create summary report
report = analyzer.create_summary_report()
```

### Visualization

```python
from src.analysis.visualization import Visualizer

viz = Visualizer()

# Bar chart comparison
viz.plot_comparison(
    results=results,
    metric='f1',
    title='F1 Score Comparison',
    save_path='plots/comparison.png'
)

# Trade-off scatter plot
viz.plot_trade_off(
    results=results,
    x_metric='training_time',
    y_metric='accuracy',
    save_path='plots/tradeoff.png'
)

# Radar chart
viz.plot_radar(
    results=results,
    metrics=['accuracy', 'speed', 'interpretability'],
    save_path='plots/radar.png'
)

# Data efficiency curves
viz.plot_data_efficiency(results, save_path='plots/efficiency.png')

# Robustness curves
viz.plot_robustness(results, save_path='plots/robustness.png')
```

### Report Generation

```python
from src.analysis.report_generator import ReportGenerator

generator = ReportGenerator(results_dir="results")

# Generate domain-specific report
domain_report = generator.generate_domain_report('domain_a')

# Generate full cross-domain report
full_report = generator.generate_full_report()

# Save report to file
generator.save_report('results/REPORT.md')
```