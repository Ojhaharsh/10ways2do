# Methodology

## Philosophy-First Approach

This benchmark is unique in that it focuses on **mental models** rather than just metrics.

### Core Principle

> "The goal is not to find the best algorithm, but to understand **when and why** different approaches work."

We evaluate each approach not just on accuracy, but on:

1. **Inductive Bias** - What assumptions does the approach make?
2. **Data Efficiency** - How much data does it need?
3. **Robustness** - How does it handle noise and edge cases?
4. **Interpretability** - Can we understand its decisions?
5. **Computational Cost** - Training and inference requirements
6. **Maintenance Burden** - How hard is it to update and debug?

---

## The 10 Paradigms

### 1. Rule-Based / Heuristics

**Mental Model:** Encode human knowledge directly as explicit rules.

**Characteristics:**
- No training data needed
- Highest interpretability
- Fastest inference
- Limited by human knowledge
- Requires manual maintenance

**When to Use:**
- Patterns are well-defined and stable
- Interpretability is mandatory
- Zero tolerance for latency
- No training data available

**Example Approaches:**
- Regular expressions for pattern matching
- Decision trees crafted by domain experts
- Lookup tables and dictionaries

---

### 2. Classical ML

**Mental Model:** Learn patterns from engineered features.

**Characteristics:**
- Requires feature engineering
- Moderate data requirements
- Interpretable (especially linear models)
- Fast training and inference
- Well-understood theory

**When to Use:**
- Moderate data available (1K-100K samples)
- Features can be meaningfully engineered
- Interpretability matters
- Fast iteration needed

**Example Approaches:**
- Logistic Regression
- SVM
- Naive Bayes
- CRF (for sequences)

---

### 3. Tree-Based

**Mental Model:** Learn decision boundaries through recursive splitting.

**Characteristics:**
- Handles non-linear relationships
- Built-in feature importance
- Robust to outliers
- Can overfit without regularization
- Works well with tabular data

**When to Use:**
- Mixed feature types (categorical + numerical)
- Non-linear relationships exist
- Feature importance needed
- Tabular or structured data

**Example Approaches:**
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

### 4. RNN / LSTM

**Mental Model:** Process sequences through recurrent hidden states.

**Characteristics:**
- Captures temporal dependencies
- Variable-length inputs
- Sequential processing (slow)
- Vanishing gradient issues
- Memory through hidden state

**When to Use:**
- Clear sequential patterns
- Moderate sequence lengths
- Temporal order matters
- Limited compute resources

**Example Approaches:**
- Vanilla RNN
- LSTM
- GRU
- Bidirectional variants

---

### 5. CNN

**Mental Model:** Extract local patterns through convolutions.

**Characteristics:**
- Fast parallel computation
- Translation invariance
- Fixed receptive field
- Good for local patterns
- Efficient on GPUs

**When to Use:**
- Local patterns are important
- Speed is critical
- Position-invariant features
- Image-like data structures

**Example Approaches:**
- 1D CNN for sequences
- Temporal Convolutional Networks (TCN)
- Dilated convolutions

---

### 6. Transformer (from scratch)

**Mental Model:** All positions attend to all other positions.

**Characteristics:**
- Global context modeling
- Parallel training
- O(n²) complexity
- Data hungry
- Flexible architecture

**When to Use:**
- Long-range dependencies matter
- Sufficient training data available
- Compute resources available
- Complex patterns exist

**Example Approaches:**
- Encoder-only (BERT-like)
- Decoder-only (GPT-like)
- Encoder-Decoder

---

### 7. Pretrained Transformer

**Mental Model:** Transfer knowledge from massive pretraining.

**Characteristics:**
- Strong out-of-the-box performance
- Reduced data requirements
- Large model size
- Fine-tuning required
- Domain mismatch possible

**When to Use:**
- Limited labeled data
- Domain is close to pretraining data
- State-of-the-art needed
- Fine-tuning infrastructure available

**Example Approaches:**
- BERT, RoBERTa
- GPT variants
- Domain-specific models (BioBERT, FinBERT)

---

### 8. LLM / Prompt-Based

**Mental Model:** Use language understanding for zero/few-shot learning.

**Characteristics:**
- No task-specific training
- Flexible through prompting
- High latency
- API costs
- Potential hallucination

**When to Use:**
- Rapid prototyping
- Low volume predictions
- Complex reasoning required
- Schema flexibility needed

**Example Approaches:**
- Zero-shot prompting
- Few-shot in-context learning
- Chain-of-thought reasoning

---

### 9. Hybrid

**Mental Model:** Combine multiple approaches strategically.

**Characteristics:**
- Best of multiple worlds
- More complex maintenance
- Graceful degradation
- Flexible architecture
- Production-ready

**When to Use:**
- Production reliability critical
- Diverse input types
- Fallback mechanisms needed
- Best accuracy required

**Example Approaches:**
- Rules + ML
- Statistical + Deep Learning
- Ensemble methods

---

### 10. Systems Perspective

**Mental Model:** Consider production constraints alongside accuracy.

**Characteristics:**
- Latency awareness
- Cost optimization
- Monitoring and observability
- Scalability considerations
- Reliability engineering

**When to Use:**
- Always in production
- Real-world deployment
- SLA requirements exist
- Cost constraints exist

**Key Considerations:**
- p50, p95, p99 latency
- Memory footprint
- Throughput (QPS)
- Cost per prediction
- Error handling

---

## Evaluation Dimensions

### Beyond Accuracy

Traditional benchmarks focus on accuracy metrics. We go further:

| Dimension | What We Measure | Why It Matters |
|-----------|-----------------|----------------|
| **Accuracy** | Task-specific metrics | Primary goal |
| **Data Efficiency** | Performance vs training size | Data is expensive |
| **Robustness** | Degradation under noise | Real data is messy |
| **Latency** | p50, p95, p99 inference time | User experience |
| **Memory** | Training and inference footprint | Infrastructure costs |
| **Interpretability** | Explainability of decisions | Trust and debugging |
| **Maintenance** | Complexity of updates | Long-term costs |

### Trade-off Analysis

We explicitly analyze trade-offs:

```
Accuracy vs Speed
├── Rules: Low accuracy, High speed
├── Classical ML: Medium accuracy, High speed
├── Deep Learning: High accuracy, Low speed
└── LLM: High accuracy, Very low speed

Interpretability vs Performance
├── Rules: High interpretability, Low performance
├── Linear Models: High interpretability, Medium performance
├── Trees: Medium interpretability, High performance
└── Deep Networks: Low interpretability, High performance

Data Needs vs Robustness
├── Rules: No data needs, Medium robustness
├── Classical ML: Low data needs, Medium robustness
├── Deep Learning: High data needs, Variable robustness
└── Pretrained: Low data needs, High robustness
```

---

## Experimental Design

### Dataset Generation

We use synthetic data to ensure:

1. **Reproducibility** - Same data across runs
2. **Controlled Complexity** - Known ground truth
3. **Scalability** - Any size dataset
4. **Privacy** - No real user data

### Evaluation Protocol

1. **Train/Val/Test Split** - Standard 70/15/15
2. **Multiple Runs** - 5 runs for statistical significance
3. **Controlled Environment** - Same hardware/software
4. **Fair Comparison** - Same data, same evaluation

### Metrics by Domain

| Domain | Primary Metrics | Secondary Metrics |
|--------|-----------------|-------------------|
| Information Extraction | Exact Match, Partial Match | Per-field F1 |
| Anomaly Detection | F1, Precision, Recall | ROC-AUC, PR-AUC |
| Recommendation | NDCG@K, Recall@K | MRR, Hit Rate |
| Time Series | MAE, RMSE | MAPE, MASE |

---

## Reproducibility

### Environment

```bash
Python 3.8+
PyTorch 1.9+
scikit-learn 1.0+
```

### Random Seeds

All experiments use fixed seeds:

- Data generation: 42
- Model initialization: 42
- Train/test splits: 42

### Hardware

Results reported on:

- CPU: Standard multi-core
- GPU: Optional (for deep learning)
- RAM: 16GB minimum

---

## Limitations

### What This Benchmark Does NOT Cover

1. **Real-world data** - We use synthetic data
2. **All possible approaches** - We select representative methods
3. **Hyperparameter optimization** - We use reasonable defaults
4. **Distributed training** - Single-machine focus
5. **Online learning** - Batch learning only

### Known Biases

1. **Synthetic data** may not capture all real-world patterns
2. **Default hyperparameters** may not be optimal for each approach
3. **Limited compute** may disadvantage large models

---

## Contributing

### Adding New Approaches

1. Implement `BaseApproach` interface
2. Define `get_philosophy()` method
3. Add to domain's `run_all.py`
4. Add tests

### Adding New Domains

1. Create `data_generator.py`
2. Implement 10 approaches
3. Define domain-specific metrics
4. Add to cross-domain analysis

---

## References

### Papers

- Benchmark design principles
- Individual approach papers
- Evaluation methodology

### Books

- Pattern Recognition and Machine Learning (Bishop)
- Deep Learning (Goodfellow et al.)
- The Elements of Statistical Learning (Hastie et al.)

---

## Citation

If you use this benchmark, please cite:

```bibtex
@software{ml_philosophy_benchmark,
  title = {ML Philosophy Benchmark},
  author = {ML Benchmark Team},
  year = {2024},
  url = {https://github.com/username/ml-philosophy-benchmark}
}
```
```

---

## Summary of Fixes

| Issue | Location | Fix |
|-------|----------|-----|
| Missing code block closure | Trade-off Analysis | Added ``` before and after the tree diagram |
| Content outside code block | Reproducibility section | Properly formatted with headers and lists |
| Extra content | End of file | Removed `.gitkeep` content (belongs in separate files) |
| Broken formatting | Random Seeds, Hardware, Limitations | Fixed list formatting |

---

## Reminder: `.gitkeep` Files

The `.gitkeep` content should be in **separate files**, not in `METHODOLOGY.md`:

- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/synthetic/.gitkeep`
- `results/domain_a/.gitkeep`
- `results/domain_b/.gitkeep`
- `results/domain_c/.gitkeep`
- `results/domain_d/.gitkeep`

Each `.gitkeep` file can simply contain:

```
# This file keeps the empty directory in git
```

Or just be an empty file (0 bytes).