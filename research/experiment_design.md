# Experiment Design

## Overview

This document describes the experimental framework, dataset, and procedures for evaluating the DataHawk platform against research questions and baselines.

---

## 1. Dataset

### Dataset Characteristics

**Source:** Synthetic social media posts generated for controlled experimentation
- **Small dataset:** 100 posts (pilot/demo)
- **Medium dataset:** 500 posts (evaluation)
- **Large dataset:** 1000 posts (stress testing)

**Post generation process:**
1. Create diverse topics across categories: AI, technology, entertainment, social issues
2. Generate posts with natural text variation, hashtags, mentions, URLs
3. Add temporal patterns (some topics trending up, others declining)
4. Include duplicates and near-duplicates to test deduplication
5. Add spam-like posts to test spam filtering

**Field structure:**
```json
{
  "post_id": "unique identifier",
  "platform": "source platform",
  "timestamp": "ISO8601 datetime",
  "author": "anonymized author ID",
  "text": "post content",
  "language": "detected language",
  "url": "optional external link",
  "engagement": {
    "likes": "integer",
    "comments": "integer",
    "shares": "integer",
    "views": "integer"
  }
}
```

### Ground Truth Annotations

For evaluation, a subset (~100 posts) was manually annotated with:
- **Topic:** Correct semantic topic label
- **Sentiment:** Positive/Neutral/Negative
- **Emotion:** Joy/Anger/Sadness/Fear/Surprise/Disgust/Neutral
- **Trend category:** Emerging/Viral/Rising/Stable/Declining

**Inter-rater agreement:** Cohen's Kappa > 0.65 (substantial agreement)

---

## 2. Experimental Protocol

### Experiment Flow

```
1. Load Dataset
   ↓
2. Preprocessing (all methods, same preprocessing)
   - Clean text
   - Deduplicate
   - Filter spam
   - Detect language
   ↓
3. Run All Methods in Parallel
   ├─ Baseline 1: Frequency
   ├─ Baseline 2: TF-IDF + K-Means
   ├─ Baseline 3: LDA
   ├─ Baseline 4: Naive/MA/Linear forecasting
   └─ Proposed: Semantic clustering + Composite scoring + Emerging detection + XGBoost
   ↓
4. Evaluate Each Method
   - Topic metrics (coherence, silhouette, purity)
   - Sentiment metrics (accuracy, F1)
   - Trend metrics (precision, recall, early detection time)
   - Forecast metrics (MAE, RMSE, MAPE, directional accuracy)
   ↓
5. Ablation Study
   - Disable volume weight (α=0)
   - Disable growth weight (β=0)
   - Disable engagement weight (γ=0)
   - Disable novelty weight (δ=0)
   ↓
6. Report Results
   - Summary table with all metrics
   - Statistical significance (if multiple runs)
   - Visualization of key findings
```

### Reproducibility

**Random seed:** 42 (fixed for reproducibility)
- All clustering (K-Means, HDBSCAN)
- All train/test splits
- All model initialization

**Number of runs:** 3 (for statistical confidence intervals)

**Platform:** Python 3.10+, CPU-based (no GPU dependency for reproducibility)

---

## 3. Baseline Methods

### Baseline 1: Frequency-Based Trending

**Algorithm:**
1. Count keyword/hashtag frequency
2. Rank by count
3. Label top 10 as "trending"

**Implementation:** `evaluation/baselines.py::frequency_baseline()`

**Metrics:**
- Precision (detected trends are real?)
- Recall (real trends detected?)
- F1 score

### Baseline 2: TF-IDF + K-Means Clustering

**Algorithm:**
1. Vectorize text with TF-IDF
2. Run K-Means clustering (K=5 fixed)
3. Extract top terms per cluster as topic labels
4. Rank clusters by size

**Implementation:** `topics/baselines.py::TfidfKMeansBaseline`

**Metrics:**
- Silhouette score
- Topic coherence
- Purity (vs. ground truth)

### Baseline 3: LDA (Latent Dirichlet Allocation)

**Algorithm:**
1. Vectorize text with bag-of-words
2. Run LDA (n_topics=5, 100 iterations)
3. Extract top terms per topic
4. Rank by topic prominence

**Implementation:** `topics/baselines.py::LDABaseline`

**Metrics:**
- Topic coherence
- Perplexity (if available)
- Purity vs. ground truth

### Baseline 4: Simple Forecasting

**Naive baseline:**
- Predict tomorrow = today (no change)

**Moving average:**
- Predict as average of last 3 days

**Linear regression:**
- Fit line to historical data, extrapolate

**Implementation:** `forecasting/baseline.py`

**Metrics:**
- MAE (mean absolute error)
- RMSE (root mean squared error)
- Directional accuracy

### Proposed Method

**Algorithm:**
1. Load and preprocess data (same as all baselines)
2. Generate semantic embeddings (sentence-transformers)
3. Cluster with HDBSCAN (adaptive density clustering)
4. Label clusters with LLM
5. Calculate composite trend scores (α·V + β·G + γ·E + δ·N)
6. Classify trends (EMERGING/VIRAL/RISING/STABLE/DECLINING)
7. Forecast with XGBoost ensemble

**Implementation:** `research/run_experiment.py`

**Metrics:** All of the above, plus:
- Early detection time
- Explanation quality

---

## 4. Evaluation Metrics

### Topic Quality

**Silhouette Score:** -1 to 1, higher is better
- Measures cluster cohesion and separation
- Formula: (b - a) / max(a, b) where a=intra-cluster distance, b=inter-cluster distance
- **Target:** > 0.4

**Topic Coherence (C_v):** 0 to 1, higher is better
- Measures semantic coherence of top-K terms per topic
- Uses word embeddings to measure term similarity
- **Target:** > 0.55

**Purity:** 0 to 1, higher is better
- For each cluster, assign the most frequent true label
- Purity = (correct assignments) / (total)
- **Target:** > 0.70

### Sentiment Analysis

**Accuracy:** Correct classifications / total
- **Target:** > 85%

**Precision:** TP / (TP + FP)
- **Target:** > 80% per class

**Recall:** TP / (TP + FN)
- **Target:** > 80% per class

**F1 Score:** 2 · (Precision · Recall) / (Precision + Recall)
- **Target:** > 0.80

### Trend Detection

**Precision@K:** Fraction of top-K detected trends that are true trends
- **Target:** > 80%

**Recall@K:** Fraction of true trends that appear in top-K detected
- **Target:** > 75%

**F1 Score:** 2 · (Precision · Recall) / (Precision + Recall)
- **Target:** > 0.77

**Early Detection Time:** Days from trend start to detection
- **Target:** < 2 days

**NDCG (Normalized DCG):** Ranking quality metric
- Compares algorithm ranking vs. ground truth ranking
- **Target:** > 0.70

### Forecasting

**MAE (Mean Absolute Error):** Average absolute difference
- Metric: | predicted - actual |
- **Target:** < 10

**RMSE (Root Mean Squared Error):** Penalizes larger errors more
- **Target:** < 15

**MAPE (Mean Absolute Percentage Error):** Relative error percentage
- Metric: | (predicted - actual) / actual | · 100%
- **Target:** < 15%

**Directional Accuracy:** Correct prediction of trend direction (up/down)
- **Target:** > 70%

### Performance Metrics

**Latency:** Time from input to results
- **Target:** < 5 seconds per 100 posts

**Memory:** Peak RAM usage
- **Target:** < 2 GB for 1000 posts

**Token Cost:** Estimated Google Gemini API cost
- **Target:** < $0.01 per post

---

## 5. Statistical Analysis

### Multiple Runs

Each method is run 3 times on the same dataset (same random seed = 42 each time to reproduce, or different seeds for variance).

**Report:**
- Mean performance
- Standard deviation
- 95% confidence interval
- Improvement over baseline ± CI

### Statistical Significance

If comparing two methods:
- Use paired t-test (same dataset, multiple runs)
- Report p-value
- Claim significance only if p < 0.05

### Ablation Study

For each component (volume, growth, engagement, novelty):
1. Run with component disabled (weight = 0)
2. Measure performance degradation
3. Calculate importance as (F1_full - F1_ablated) / F1_full

---

## 6. Experiment Execution

### Step 1: Prepare Dataset
```bash
python scripts/prepare_dataset.py --input dataset/test_data_medium.csv --output dataset/prepared.csv
```

### Step 2: Run Baselines
```bash
python scripts/run_baselines.py --data dataset/prepared.csv --output results/baselines.json
```

### Step 3: Run Proposed Method
```bash
python scripts/run_experiments.py --data dataset/prepared.csv --output results/proposed.json
```

### Step 4: Run Ablation Study
```bash
python scripts/run_ablation.py --data dataset/prepared.csv --output results/ablation.json
```

### Step 5: Evaluate All
```bash
python scripts/evaluate.py --baseline results/baselines.json --proposed results/proposed.json --ablation results/ablation.json --output results/summary.json
```

### Step 6: Generate Report
```bash
python scripts/evaluate.py --generate-report results/summary.json > results/RESULTS.md
```

---

## 7. Success Criteria

The experiment is considered **successful** if:

1. ✅ All methods run to completion without errors
2. ✅ Results are reproducible (≥80% consistency across runs)
3. ✅ Proposed method outperforms all baselines on majority of metrics
4. ✅ Improvements are ≥10% on key metrics (coherence, precision, early detection)
5. ✅ Statistical significance achieved (p < 0.05) on primary metrics
6. ✅ Explanations are meaningful and interpretable
7. ✅ Performance trade-offs are documented

---

## 8. Failure Modes and Contingencies

**If topic clustering fails:**
- Check embedding model is loaded
- Verify HDBSCAN dependencies
- Fall back to K-Means with fixed K=5

**If LLM labeling fails:**
- Use fallback: top-3 keywords as label
- Document in limitations

**If forecasting accuracy is poor:**
- Check sufficient historical data (min 10 time points)
- May need longer time series than available in synthetic data

**If baselines underperform:**
- May indicate dataset is too easy/hard
- Adjust dataset complexity and retry

---

*This experiment design provides a rigorous, reproducible framework for evaluating DataHawk against research questions and baselines.*
