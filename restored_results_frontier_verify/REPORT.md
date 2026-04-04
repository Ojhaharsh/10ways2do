# ML Philosophy Benchmark: Complete Report

Generated: 2026-04-04 03:29:33

## Overview


This benchmark evaluates 10 fundamentally different ML approaches across 5 real-world domains:

1. **Information Extraction** - Extracting structured data from text
2. **Anomaly Detection** - Identifying unusual patterns in data
3. **Recommendation** - Suggesting relevant items to users
4. **Time Series Forecasting** - Predicting future values
5. **Tabular Decisioning** - Risk scoring and binary decision support on structured features

Each approach represents a different **mental model** for solving ML problems, with distinct 
trade-offs in terms of accuracy, speed, interpretability, data efficiency, and robustness.


---

# Information Extraction: Benchmark Report

Generated: 2026-04-04 03:29:33

## Summary

                     Approach Training Time (s) name_exact_match name_partial_match email_exact_match
                Rule-Based IE              0.00           0.7200             0.7367            1.0000
     Classical ML (CRF-style)              0.00           0.1000             0.3121            1.0000
         Tree-Based (XGBoost)              0.00           0.0150             0.0750            1.0000
                     RNN/LSTM              0.00           0.0000             0.7502            0.0000
                 CNN Sequence              0.00           0.0000             0.1103            0.0000
        Transformer (Scratch)              0.00           0.0000             0.3437            0.0000
             Prompt-Based LLM              0.00           0.7200             0.7367            1.0000
Pretrained Transformer (BERT)              0.00           0.5275             0.5694            0.8550
          Hybrid (Rules + ML)              0.00           0.3500             0.5687            1.0000

## Detailed Results

### Rule-Based IE


**Metrics:**



### Classical ML (CRF-style)


**Metrics:**



### Tree-Based (XGBoost)


**Metrics:**



### RNN/LSTM


**Metrics:**



### CNN Sequence


**Metrics:**



### Transformer (Scratch)


**Metrics:**



### Prompt-Based LLM


**Metrics:**



### Pretrained Transformer (BERT)


**Metrics:**



### Hybrid (Rules + ML)


**Metrics:**



## Statistical Significance

Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

- Rule-Based IE: compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.1700, Cohen's d=N/A, effect=N/A

- Classical ML (CRF-style): compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.6178, Cohen's d=N/A, effect=N/A

- Tree-Based (XGBoost): compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.6533, Cohen's d=N/A, effect=N/A

- RNN/LSTM: compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.8094, Cohen's d=N/A, effect=N/A

- CNN Sequence: compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.7261, Cohen's d=N/A, effect=N/A

- Transformer (Scratch): compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.8089, Cohen's d=N/A, effect=N/A

- Prompt-Based LLM: compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.1261, Cohen's d=N/A, effect=N/A

- Pretrained Transformer (BERT): reference best approach (p-value=1.0000, Cohen's d=N/A, effect=N/A)

- Hybrid (Rules + ML): compared vs Pretrained Transformer (BERT), p-value=0.0000, mean difference=-0.5044, Cohen's d=N/A, effect=N/A

## Key Insights

1. **Best Performance**: Rule-Based IE achieved the best overall_exact_match of 0.0000
2. **Fastest Training**: Rule-Based IE trained in 0.00 seconds

### Approach Categories:
- **Rule/Statistical**: Rule-Based IE, Hybrid (Rules + ML) - High interpretability, low latency
- **Classical ML**: Classical ML (CRF-style), Tree-Based (XGBoost), Hybrid (Rules + ML) - Good balance of performance and speed
- **Deep Learning**: RNN/LSTM, CNN Sequence, Transformer (Scratch), Pretrained Transformer (BERT) - Best accuracy, higher compute
- **Hybrid**: Hybrid (Rules + ML) - Production-ready combinations


---

# Anomaly Detection: Benchmark Report

Generated: 2026-04-04 03:29:33

## Summary

                     Approach Training Time (s) accuracy precision recall
  Statistical (Z-score + IQR)              0.00   0.5850    0.2857 0.7895
            Statistical (MAD)              0.00   0.8650    0.6897 0.5263
Tree-Based (Isolation Forest)              0.07   0.6750    0.3373 0.7368

## Detailed Results

### Statistical (Z-score + IQR)


**Metrics:**



### Statistical (MAD)


**Metrics:**



### Tree-Based (Isolation Forest)


**Metrics:**



## Statistical Significance

Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

- Statistical (Z-score + IQR): compared vs Statistical (MAD), p-value=N/A, mean difference=-0.1774, Cohen's d=N/A, effect=N/A

- Statistical (MAD): reference best approach (p-value=1.0000, Cohen's d=0.0000, effect=negligible)

- Tree-Based (Isolation Forest): compared vs Statistical (MAD), p-value=N/A, mean difference=-0.1342, Cohen's d=N/A, effect=N/A

## Key Insights

1. **Best Performance**: Statistical (MAD) achieved the best f1 of 0.5970
2. **Fastest Training**: Statistical (MAD) trained in 0.00 seconds

### Approach Categories:
- **Rule/Statistical**: Statistical (Z-score + IQR), Statistical (MAD) - High interpretability, low latency
- **Classical ML**: Tree-Based (Isolation Forest) - Good balance of performance and speed


---

# Recommendation: Benchmark Report

Generated: 2026-04-04 03:29:33

## Summary

                  Approach Training Time (s) precision@5 recall@5   f1@5
          Popularity-Based              0.00      0.0032   0.4181 0.0064
             User-Based CF              0.02      0.0037   0.4173 0.0073
Matrix Factorization (SVD)              0.01      0.0064   0.4250 0.0126

## Detailed Results

### Popularity-Based


**Metrics:**



### User-Based CF


**Metrics:**



### Matrix Factorization (SVD)


**Metrics:**



## Statistical Significance

Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

- Popularity-Based: compared vs Matrix Factorization (SVD), p-value=N/A, mean difference=-0.0064, Cohen's d=N/A, effect=N/A

- User-Based CF: compared vs Matrix Factorization (SVD), p-value=N/A, mean difference=-0.0104, Cohen's d=N/A, effect=N/A

- Matrix Factorization (SVD): reference best approach (p-value=1.0000, Cohen's d=0.0000, effect=negligible)

## Key Insights

1. **Best Performance**: Matrix Factorization (SVD) achieved the best ndcg@10 of 0.0252
2. **Fastest Training**: Popularity-Based trained in 0.00 seconds

### Approach Categories:


---

# Time Series Forecasting: Benchmark Report

Generated: 2026-04-04 03:29:33

## Summary

             Approach Training Time (s)     mae       mse    rmse
Exponential Smoothing              0.45 58.1770 4533.5230 67.3314
 Tree-Based (XGBoost)             79.81  0.3517    0.1945  0.4410

## Detailed Results

### Exponential Smoothing


**Metrics:**



### Tree-Based (XGBoost)


**Metrics:**



## Statistical Significance

Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

- Exponential Smoothing: compared vs Tree-Based (XGBoost), p-value=N/A, mean difference=-66.8904, Cohen's d=N/A, effect=N/A

- Tree-Based (XGBoost): reference best approach (p-value=1.0000, Cohen's d=0.0000, effect=negligible)

## Key Insights

1. **Best Performance**: Tree-Based (XGBoost) achieved the best rmse of 0.4410
2. **Fastest Training**: Exponential Smoothing trained in 0.45 seconds

### Approach Categories:
- **Classical ML**: Tree-Based (XGBoost) - Good balance of performance and speed


---

# Tabular Decisioning: Benchmark Report

Generated: 2026-04-04 03:29:33

## Summary

                    Approach Training Time (s) accuracy precision recall
     Rule-Based Thresholding              0.00   0.7500    0.3860 0.5946
Linear (Logistic Regression)              0.22   0.8000    0.4651 0.5405
  Tree-Based (Random Forest)              0.12   0.8150    0.0000 0.0000

## Detailed Results

### Rule-Based Thresholding


**Metrics:**



### Linear (Logistic Regression)


**Metrics:**



### Tree-Based (Random Forest)


**Metrics:**



## Statistical Significance

Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

- Rule-Based Thresholding: compared vs Linear (Logistic Regression), p-value=N/A, mean difference=-0.0319, Cohen's d=N/A, effect=N/A

- Linear (Logistic Regression): reference best approach (p-value=1.0000, Cohen's d=0.0000, effect=negligible)

- Tree-Based (Random Forest): compared vs Linear (Logistic Regression), p-value=N/A, mean difference=-0.5000, Cohen's d=N/A, effect=N/A

## Key Insights

1. **Best Performance**: Linear (Logistic Regression) achieved the best f1 of 0.5000
2. **Fastest Training**: Rule-Based Thresholding trained in 0.00 seconds

### Approach Categories:
- **Rule/Statistical**: Rule-Based Thresholding - High interpretability, low latency
- **Classical ML**: Tree-Based (Random Forest) - Good balance of performance and speed


---

# Cross-Domain Insights

## Cross-Domain Statistical Summary
- Information Extraction: best=Pretrained Transformer (BERT), best_mean=0.8128, criterion=higher-is-better, significant_vs_best=8/8.
- Anomaly Detection: best=Statistical (MAD), best_mean=0.5970, criterion=higher-is-better, significant_vs_best=0/0.
- Recommendation: best=Matrix Factorization (SVD), best_mean=0.0252, criterion=higher-is-better, significant_vs_best=0/0.
- Time Series Forecasting: best=Tree-Based (XGBoost), best_mean=0.4410, criterion=lower-is-better, significant_vs_best=0/0.
- Tabular Decisioning: best=Linear (Logistic Regression), best_mean=0.5000, criterion=higher-is-better, significant_vs_best=0/0.
- Most frequent domain winner: Pretrained Transformer (BERT) (1 domain(s)).

## Cross-Domain Pareto Frontier
Extraordinary Index combines normalized quality, speed efficiency, budget resilience, and execution consistency to identify practical champions.
- Information Extraction: champion=Prompt-Based LLM (index=0.930), Pareto top=Prompt-Based LLM, Pretrained Transformer (BERT).
- Anomaly Detection: champion=Statistical (MAD) (index=0.875), Pareto top=Statistical (MAD).
- Recommendation: champion=Matrix Factorization (SVD) (index=0.875), Pareto top=Matrix Factorization (SVD).
- Time Series Forecasting: champion=Exponential Smoothing (index=0.875), Pareto top=Exponential Smoothing.
- Tabular Decisioning: champion=Linear (Logistic Regression) (index=0.875), Pareto top=Linear (Logistic Regression).
- Cross-domain generalists (avg Extraordinary Index):
  - Prompt-Based LLM: 0.930 across 1 domain(s)
  - Rule-Based IE: 0.905 across 1 domain(s)
  - Statistical (MAD): 0.875 across 1 domain(s)
  - Matrix Factorization (SVD): 0.875 across 1 domain(s)
  - Exponential Smoothing: 0.875 across 1 domain(s)


## Universal Patterns

### 1. Rule-Based Methods
- **Across all domains**: Highest interpretability, lowest latency
- **Trade-off**: Limited coverage, requires manual maintenance
- **Best when**: Patterns are well-defined and stable

### 2. Classical ML (Trees, Linear Models)
- **Across all domains**: Good performance with moderate data
- **Trade-off**: Requires feature engineering
- **Best when**: Interpretability matters, moderate scale

### 3. Deep Learning (RNN, CNN, Transformers)
- **Across all domains**: Best raw performance with sufficient data
- **Trade-off**: Higher compute, less interpretable
- **Best when**: Accuracy is paramount, data is abundant

### 4. Pretrained Models / Transfer Learning
- **Across all domains**: Reduces data requirements significantly
- **Trade-off**: Model size, domain mismatch possible
- **Best when**: Limited labeled data, similar pretrain domain

### 5. Hybrid Approaches
- **Across all domains**: Often best production choice
- **Trade-off**: More complex to maintain
- **Best when**: Reliability is critical, diverse inputs

### 6. Systems Considerations
- **Across all domains**: Often the deciding factor
- **Key metrics**: Latency, throughput, memory, cost
- **Best practice**: Always evaluate with production constraints


## Recommendations by Use Case


### By Priority:

**Maximum Accuracy:**
- Use pretrained transformers (BERT, etc.)
- Ensemble multiple approaches
- Accept higher latency and cost

**Low Latency (<10ms):**
- Rule-based or simple ML models
- Heavy caching for repeated queries
- Model distillation from larger models

**Limited Training Data (<1000 samples):**
- Transfer learning / pretrained models
- Rule-based augmentation
- Few-shot learning with LLMs

**High Interpretability:**
- Rule-based systems
- Linear models with feature importance
- Decision trees / SHAP explanations

**Tabular Risk / Decisioning Workloads:**
- Start with tree boosting and calibrated linear baselines
- Use ensembles when operating point stability is critical
- Tune threshold policies against precision-recall business targets

**Production Reliability:**
- Hybrid approaches with fallbacks
- Ensemble for robustness
- Comprehensive monitoring

**Minimal Infrastructure:**
- Rule-based or classical ML
- Avoid GPU requirements
- Consider LLM APIs for complex tasks
