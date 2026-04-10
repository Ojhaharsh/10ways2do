# Architecture & System Design

10 Ways to Do ML benchmark framework

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (CLI)                        │
│  python main.py --all --n-runs 5 --seed 42                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────────┐      ┌──────────────────────┐
│  Domain Dispatcher  │      │  CLI Argument Parser │
│  ├─ domain_a        │      │  ├─ --all            │
│  ├─ domain_b        │      │  ├─ --domain X       │
│  ├─ domain_c        │      │  ├─ --n-runs         │
│  ├─ domain_d        │      │  ├─ --seed           │
│  ├─ domain_e        │      │  └─ --smoke-test     │
│  ├─ domain_f        │
│  └─ domain_g        │
└──────────┬──────────┘      │  └─ --smoke-test     │
           │                 └──────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  Domain Runner: run_all_approaches(...)            │
    │  ┌──────────────────────────────────────────────┴──┐
    │  │  1. Resolve seed list (n_runs → [seed_0...]) │
    │  │  2. For each seed:                            │
    │  │     a. Generate data (reproducible)          │
    │  │     b. For each approach:                     │
    │  │        - Train (with timeout)                │
    │  │        - Predict                             │
    │  │        - Compute metrics                      │
    │  │        - Validate output (schema check)       │
    │  │     c. Aggregate per-approach results         │
    │  │  3. Generate outputs:                          │
    │  │     - run_manifest.json                       │
    │  │     - results_raw_by_run.json                 │
    │  │     - results_aggregated.json                 │
    │  │     - comparison_canonical.csv                │
    │  │     - comparison_variants.csv                 │
    └────────────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │  Shared Infrastructure           │
    ├──────────────────────────────────┤
    │  • benchmark_utils.py            │
    │    - set_global_seed()           │
    │    - resolve_seed_list()         │
    │    - create_run_manifest()       │
    │    - aggregate_numeric_dicts()   │
    │                                  │
    │  • benchmark_schema.py           │
    │    - validate_run_row()          │
    │    - validate_aggregated()       │
    │    - validate_comparison_df()    │
    │                                  │
    │  • base_model.py                 │
    │    - BaseApproach (abstract)     │
    │                                  │
    │  • evaluation.py                 │
    │    - Metric computation          │
    └──────┬───────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │  Approach Classes (10 per domain)       │
    │  ├─ rule_based.py                      │
    │  ├─ classical_ml.py                    │
    │  ├─ tree_based.py                      │
    │  ├─ rnn_lstm.py                        │
    │  ├─ cnn_sequence.py                    │
    │  ├─ transformer_scratch.py             │
    │  ├─ pretrained_transformer.py          │
    │  ├─ prompt_llm.py                      │
    │  ├─ hybrid.py                          │
    │  └─ integration.py                     │
    └──────┬───────────────────────────────────┘
           │
    ┌──────▼────────────────────────┐
    │  Cross-Domain Analysis        │
    │  ├─ report_generator.py       │
    │  ├─ cross_domain_analysis.py  │
    │  └─ visualization.py          │
    └───────────────────────────────┘
```

---

## 2. Core Principles

### 2.1 Reproducibility Architecture

**Problem**: ML results are non-deterministic due to randomness in data shuffling, weight initialization, etc.

**Solution**: Multi-level seed control with manifest metadata

```python
# Level 1: Seed Resolution
--n-runs 5 --seed 42
    ↓
resolve_seed_list(5, 42, None)
    ↓
[42, 43, 44, 45, 46]  # Explicit seed list

# Level 2: Global Seed Setting
for seed in seed_list:
    set_global_seed(seed)  # Set numpy, random, torch, cuda
    generate_data(seed)     # Uses seed
    train_all_approaches()  # Uses seed-controlled randomness

# Level 3: Metadata Capture
create_run_manifest(domain, config, seed_list)
    → Captures Python version, OS, package versions
    → Allows exact reproduction later
```

**Result**: Given the same seed and manifest, any machine reproduces exact same results.

### 2.2 Fairness Architecture

**Problem**: Different approaches have different computational costs (RBF SVM vs LSTM).

**Solution**: Budget-equivalent implementation
- All approaches get 5-minute timeout per training
- Hyperparameters use sensible defaults (not grid-tuned)
- No approach gets GPU-accelerated acceleration unless others do too

### 2.3 Schema Validation Architecture

**Problem**: Approaches may fail to produce valid outputs (wrong metric shapes, NaN values, missing keys).

**Solution**: Fail-fast validation at write time

```
Per-Run Row Validation:
├─ Required keys: [seed, approach, success, metrics]
├─ No NaNs or Infinities in metrics
└─ Metrics match domain spec (e.g., IE has exact_match)

Aggregated Validation:
├─ Each approach appears N times (N seeds)
├─ Summary stats present: [mean, std, n]
└─ n_success ≤ n_runs (tracks partial failures)

Comparison DataFrame Validation:
├─ Canonical table has 10 rows (one per category)
├─ Column names match standard format
└─ Rankings by primary metric are deterministic
```

### 2.4 Extensibility Architecture

**Problem**: Adding new approaches or domains requires modifying multiple places.

**Solution**: Plugin architecture with standard interfaces

```python
# All approaches inherit from BaseApproach
class MyApproach(BaseApproach):
    def train(self, X_train, y_train):
        """Must implement"""
    
    def predict(self, X_test):
        """Must implement"""
    
    def get_metrics(self, y_pred, y_true, X_test=None):
        """Must implement"""
```

**Result**: New approaches can be added by:
1. Creating approach_XX_name.py
2. Registering in build_approaches()
3. (That's it—everything else is automatic)

---

## 3. Data Flow

### 3.1 Single-Run Flow (1 seed, 1 approach)

```
Input: (X_train, y_train), (X_test, y_test), seed=42
  │
  ├─ set_global_seed(42)  ← All randomness now deterministic
  │
  ├─ approach = MyApproach(config)
  │
  ├─ approach.train(X_train, y_train)
  │   ├─ Model initialization (seed-controlled)
  │   ├─ Optimizer setup
  │   └─ Training loop (all RNG seeded)
  │
  ├─ y_pred = approach.predict(X_test)
  │
  ├─ metrics = approach.get_metrics(y_pred, y_test)
  │   └─ Returns: {"EM": 0.85, "FM": 0.91, ...}
  │
  ├─ Validation: validate_run_row(
  │     {seed: 42, approach: "MyApproach", success: True, metrics: {...}}
  │   )
  │
  └─ Output: {"seed": 42, "approach": "MyApproach", "success": True, "metrics": {...}}
```

### 3.2 Multi-Run Aggregation (5 seeds, 1 approach)

```
Input: [run_seed_42, run_seed_43, run_seed_44, run_seed_45, run_seed_46]
       with metrics EM: [0.85, 0.86, 0.84, 0.87, 0.85]
  │
  ├─ Filter successful runs
  │   → [0.85, 0.86, 0.84, 0.87, 0.85]
  │
  ├─ Compute statistics:
  │   mean = (0.85 + 0.86 + 0.84 + 0.87 + 0.85) / 5 = 0.854
  │   std  = sqrt(sum((x - mean)^2) / (n-1)) = 0.0104
  │   n    = 5
  │
  ├─ 95% CI = [mean ± 1.96*(std/√n)]
  │        = [0.854 ± 0.0091] = [0.845, 0.863]
  │
  ├─ Format aggregated result:
  │   {
  │     "EM": {"mean": 0.854, "std": 0.0104, "n": 5},
  │     "FM": {"mean": 0.910, "std": 0.008, "n": 5},
  │     ...
  │   }
  │
  ├─ Validation: validate_aggregated_results(result)
  │   ✓ Required keys present
  │   ✓ Finite values
  │   ✓ Stats structure correct
  │
  └─ Output: Aggregated dict with mean ± std for all metrics
```

### 3.3 Full Domain Flow (5 seeds × 10 approaches)

```
For each seed in [42, 43, 44, 45, 46]:
  For each approach in [Rule, Classical, Tree, ..., Integration]:
    │
    ├─ Generate data deterministically (seed-controlled)
    ├─ Train approach with timeout
    ├─ Record metrics or failure
    └─ Validate per-run row
    
Aggregate results across seeds:
  For each approach:
    ├─ Collect all successful runs
    ├─ Compute mean, std, n
    └─ Validate aggregated result

Output generation:
  ├─ run_manifest.json (metadata)
  ├─ results_raw_by_run.json (all individual runs)
  ├─ results_aggregated.json (mean ± std)
  ├─ comparison_canonical.csv (10 categories)
  └─ comparison_variants.csv (all variants)
```

---

## 4. Module Dependency Graph

```
main.py (CLI entry)
├─ run_all() or run_domain()
│  ├─ src.domain_X.run_all.run_all_approaches()
│  │  ├─ src.core.benchmark_utils
│  │  │  ├─ set_global_seed()
│  │  │  ├─ resolve_seed_list()
│  │  │  ├─ create_run_manifest()
│  │  │  └─ aggregate_numeric_dicts()
│  │  │
│  │  ├─ src.core.benchmark_schema
│  │  │  ├─ validate_run_row()
│  │  │  ├─ validate_aggregated_results()
│  │  │  └─ validate_comparison_dataframe()
│  │  │
│  │  ├─ src.domain_X.data_generator
│  │  │  └─ generate_data()
│  │  │
│  │  ├─ src.domain_X.approach_01_...
│  │  ├─ src.domain_X.approach_02_...
│  │  ├─ ... (approach_03 through approach_10)
│  │  │  └─ All inherit from src.core.base_model.BaseApproach
│  │  │
│  │  └─ src.core.evaluation
│  │     └─ Metric computation functions
│  │
│  └─ src.analysis.report_generator
│     ├─ Load all domain results
│     ├─ Generate cross-domain insights
│     └─ Save REPORT.md
│
└─ configure, install, test (via pytest)
```

---

## 5. Class Hierarchy

### 5.1 BaseApproach (Abstract)

```python
class BaseApproach:
    """Abstract base class for all ML approaches.
    
    All concrete approaches (rule-based, neural, etc.) must inherit
    and implement train(), predict(), and get_metrics().
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize approach with config dict."""
        self.config = config or {}
        self.model = None
    
    def train(self, X_train, y_train):
        """Train model on labeled data.
        
        Raises:
            NotImplementedError: Subclasses must implement
        """
        raise NotImplementedError
    
    def predict(self, X_test):
        """Generate predictions on test data.
        
        Raises:
            NotImplementedError: Subclasses must implement
        """
        raise NotImplementedError
    
    def get_metrics(self, y_pred, y_true, X_test=None):
        """Compute evaluation metrics.
        
        Returns:
            dict: Domain-specific metrics (exact keys matter for validation)
        
        Raises:
            NotImplementedError: Subclasses must implement
        """
        raise NotImplementedError
```

### 5.2 Concrete Approach Examples

```python
# Domain A: Information Extraction
class RuleBasedIE(BaseApproach):
    """Rule-based pattern matching."""
    def train(self, X_train, y_train):
        self.patterns = self._extract_patterns(X_train, y_train)
    
    def predict(self, X_test):
        return self._match_patterns(X_test)
    
    def get_metrics(self, y_pred, y_true, X_test=None):
        return {'overall_exact_match': ..., 'overall_partial_match': ...}


class TransformerIE(BaseApproach):
    """Fine-tune pre-trained transformer."""
    def train(self, X_train, y_train):
        self.model = AutoModelForTokenClassification.from_pretrained('bert-base')
        # Training loop with fine-tuning
    
    def predict(self, X_test):
        return self.model.forward(X_test).argmax(dim=-1)
    
    def get_metrics(self, y_pred, y_true, X_test=None):
        return {'overall_exact_match': ..., 'overall_partial_match': ...}


# Domain B: Anomaly Detection
class IsolationForestAD(BaseApproach):
    """Tree-based anomaly detection."""
    def train(self, X_train, y_train):
        self.model = IsolationForest(n_estimators=100)
        self.model.fit(X_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)  # -1 for anomaly, 1 for normal
    
    def get_metrics(self, y_pred, y_true, X_test=None):
        return {'precision': ..., 'recall': ..., 'f1': ..., 'roc_auc': ...}
```

---

## 6. Configuration Management

### 6.1 Hierarchical Config (CLI → Domain → Approach)

```
┌─────────────────────────────────────┐
│  CLI Arguments (main.py)            │
│  python main.py --all --n-runs 5    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Domain Config (run_all.py)         │
│  {                                  │
│    'n_train': 1000,                 │
│    'n_test': 200,                   │
│    'n_runs': 5,                     │
│    'seed': 42,                      │
│  }                                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Approach Config (build_approaches) │
│  {                                  │
│    'epochs': 10,                    │
│    'batch_size': 32,                │
│    'learning_rate': 0.001,          │
│  }                                  │
└─────────────────────────────────────┘
```

### 6.2 Config Merging

```python
# Domain level
domain_config = {
    'n_train': 1000,
    'n_test': 200,
    'n_runs': 5,
    'seed': 42,
}

# Approach level (overrides domain defaults)
approach_config = {
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
}

# Per-approach instantiation
approach = RuleBasedIE(config=approach_config)
# Approach can access: approach.config['epochs'], etc.
```

---

## 7. Error Handling & Resilience

### 7.1 Approach-Level Isolation

```python
# In run_all_approaches:
for approach in approaches:
    try:
        model = approach_class(config)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = model.get_metrics(y_pred, y_test)
        
        # Validate
        validate_run_row({'seed': seed, 'approach': name, ...})
        all_results.append({'success': True, ...})
        
    except MemoryError as e:
        logger.warning(f"{name}: Out of memory")
        all_results.append({'success': False, 'error': str(e)})
    
    except TimeoutError as e:
        logger.warning(f"{name}: Timeout after 300s")
        all_results.append({'success': False, 'error': 'timeout'})
    
    except Exception as e:
        logger.error(f"{name}: {type(e).__name__}: {e}")
        all_results.append({'success': False, 'error': str(e)})
        # Continue to next approach
```

**Result**: One failed approach doesn't crash entire benchmark.

### 7.2 Validation Error Handling

```python
# Validation catches issues before commit
try:
    validate_run_row(row)
except ValueError as e:
    logger.error(f"Invalid run row: {e}")
    raise  # Stop: this is a schema violation (bug in approach)

try:
    validate_aggregated_results(agg_dict)
except ValueError as e:
    logger.error(f"Invalid aggregated results: {e}")
    # Could be due to all approaches failing; warn but continue
```

---

## 8. Smoke Testing Architecture

### 8.1 Smoke Mode: Reduced Approach Subset

**Purpose**: Fast validation (< 5 min) that benchmark runs end-to-end without errors.

```python
# build_approaches(smoke_test=False)
if smoke_test:
    # Only 3 fastest approaches per domain
    approaches = [
        RuleBasedIE(),              # ~100ms
        ClassicalMLIE(),            # ~1s  
        TreeBasedIE(n_estimators=20),  # ~2s (reduced from 100)
    ]
else:
    # All 10 canonical approaches
    approaches = [
        RuleBasedIE(),
        ClassicalMLIE(),
        TreeBasedIE(),
        RNNIE(),
        CNNIE(),
        TransformerScratchIE(),
        TransformerPretrainedIE(),
        PromptLLMIE(),
        HybridIE(),
        IntegrationIE(),
    ]
```

### 8.2 Smoke Data: Tiny Size

```python
# Smoke mode data sizes
domain_a: n_train=80, n_test=20    # vs. 1000, 200 full
domain_b: n_train=80, n_test=20    # vs. 1000, 200 full
domain_c: n_users=60, n_items=20   # vs. 500, 200 full
domain_d: n_samples=300            # vs. 1000 full
domain_e: n_samples=600            # tabular decisioning smoke
domain_f: n_samples=800            # cyber threat hunting smoke
domain_g: n_samples=800            # operations optimization smoke
```

### 8.3 Smoke Test: Continuous Validation

```bash
# CI/CD runs this on every commit
pytest tests/test_benchmark_smoke_artifacts.py -v

# Validates:
# ✓ All domains run without error
# ✓ Output files exist (manifest, raw, aggregated, CSVs)
# ✓ Schema validation passes (no NaNs, required keys)
# ✓ Canonical table has expected rows/columns
# ✓ Results are deterministic (same seed = same output)
```

---

## 9. Performance Characteristics

### 9.1 Time Complexity

```
Full Benchmark = O(# domains × # seeds × # approaches × time_per_approach)

Typical execution:
├─ Domains: 5
├─ Seeds: 5
├─ Approaches per domain: 10
└─ Time per approach (training + inference):
   ├─ Rule-based: ~0.1s
   ├─ Classical ML: ~1s
   ├─ Tree-based: ~5s
   ├─ LSTM: ~10s
   ├─ Transformer: ~15s
   ├─ Pre-trained: ~10s
   ├─ LLM: ~100s (API call)
   ├─ Hybrid: ~15s
   ├─ Integration: ~20s
   └─ Total per domain: ~186s × 5 seeds = ~15 min

Total for 7 domains: ~105 min + contingency = ~3.5 hours
```

### 9.2 Memory Requirements

```
Peak memory during full benchmark:
├─ Dataset X_train @ 1000 × 1000 features = ~4 MB
├─ Dataset y_train = ~4 KB
├─ LSTM model @ 256 hidden = ~10 MB
├─ Transformer model @ 12 layers = ~100 MB
├─ Pre-trained BERT = ~400 MB
└─ Total: ~500 MB (well within typical GPU/CPU memory)
```

---

## 10. Security & Reproducibility Guarantees

### 10.1 Seed Control (No Nondeterminism)

✓ Seed applied to:
- Python's `random` module
- NumPy's random state
- PyTorch's CPU and CUDA RNG
- TensorFlow's global seed
- Data shuffling in batch generators

### 10.2 Environment Capture

Run manifest includes:
- Python version (e.g., 3.10.8)
- OS and architecture (e.g., Windows-10-x86_64)
- All installed package versions (numpy 1.24.3, torch 2.0.1, etc.)
- Timestamp (UTC)

**Goal**: Later, someone with manifest can recreate exact environment.

### 10.3 No Hard-Coded Paths

✓ All paths use `Path` objects and `output_dir` parameter
✗ Never: `/home/user/data/...` (user-specific)
✗ Never: `C:\Users\KIIT\...` (machine-specific)

---

## 11. Extension Points

### 11.1 Adding New Approach

```python
# 1. Create file
src/domain_X/approach_YY_name.py

# 2. Implement BaseApproach
class NewApproach(BaseApproach):
    def train(self, X_train, y_train): ...
    def predict(self, X_test): ...
    def get_metrics(self, y_pred, y_true, X_test=None): ...

# 3. Register in domain
in src/domain_X/run_all.py, find build_approaches():
    approaches = [..., NewApproach(config), ...]

# 4. Run it
python main.py --domain X
# Automatic inclusion in results!
```

### 11.2 Adding New Domain

```python
# 1. Create directory
src/domain_E_name/
src/domain_F_name/
src/domain_G_name/

# 2. Implement data_generator.py
def generate_data(n_train, n_test, seed): ...

# 3. Implement 10 approaches
src/domain_E/approach_0{1..10}_*.py
src/domain_F/approach_0{1..10}_*.py
src/domain_G/approach_0{1..10}_*.py

# 4. Implement run_all.py
def run_all_approaches(...): ...

# 5. Register in main.py
elif domain == 'e':
    from src.domain_e.run_all import run_all_approaches
    from src.domain_f.run_all import run_all_approaches
    from src.domain_g.run_all import run_all_approaches
    return run_all_approaches(...)

# 6. Add tests
tests/test_domain_e.py
tests/test_domain_f.py
tests/test_domain_g.py

# 7. Run it
python main.py --domain e
```

---

**Architecture Version**: 1.0  
**Last Updated**: March 2026

