# Evaluation Metrics

## Overview

This document defines all metrics used to evaluate topic discovery, sentiment analysis, trend detection, and forecasting in the DataHawk platform.

---

## 1. Topic Quality Metrics

### 1.1 Silhouette Score

**Definition:** Measure of how well-separated and cohesive clusters are.

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

where:
- $a_i$ = average distance from point i to others in its cluster (cohesion)
- $b_i$ = average distance from point i to points in nearest other cluster (separation)

**Range:** [-1, 1]
- **+1:** Point is well-clustered (far from other clusters, close to its own)
- **0:** Point is on cluster boundary
- **-1:** Point is in wrong cluster

**Target:** > 0.4 (indicates reasonable separation)

**Implementation:** `sklearn.metrics.silhouette_score()`

**Interpretation:**
- High silhouette → topics are semantically distinct
- Low silhouette → topics are overlapping or poorly separated

---

### 1.2 Topic Coherence (C_v Score)

**Definition:** Measures semantic coherence of top-K terms in each topic.

**Algorithm:**
1. For each topic, take top-5 terms
2. Compute pairwise semantic similarity (using word embeddings or PMI)
3. Average similarity = coherence of this topic
4. Average across all topics = corpus coherence

**Range:** [0, 1]
- **1.0:** Top terms are highly semantically related
- **0.0:** Top terms are unrelated

**Target:** > 0.55 (indicates interpretable topics)

**Implementation:** `topics/coherence.py::CoherenceEvaluator`

**Interpretation:**
- High coherence → topics are interpretable ("AI agents, coding, automation")
- Low coherence → topic contains noise ("AI agents the of a to")

---

### 1.3 Topic Purity

**Definition:** When ground-truth topic labels exist, purity measures accuracy of discovered clusters.

**Algorithm:**
1. For each discovered cluster, find the most frequent true label
2. Count correct assignments to that label
3. Purity = (total correct) / (total posts)

$$\text{Purity} = \frac{1}{N} \sum_{i=1}^{K} \max_j |C_i \cap T_j|$$

where:
- $C_i$ = discovered cluster i
- $T_j$ = true topic j
- $N$ = total posts

**Range:** [0, 1]
- **1.0:** Every cluster corresponds to a single true topic
- **0.0:** Random clustering

**Target:** > 0.70 (indicates good alignment with ground truth)

**Implementation:** `evaluation/metrics.py::compute_purity()`

**Interpretation:**
- High purity → algorithm discovers real semantic groups
- Low purity → algorithm is fragmented or mixing unrelated posts

---

## 2. Sentiment Analysis Metrics

### 2.1 Accuracy

**Definition:** Fraction of correctly classified sentiments.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Range:** [0, 1]
- **1.0:** All sentiments correct
- **0.5:** Coin-flip performance
- **0.0:** All sentiments incorrect

**Target:** > 0.85 (indicates strong model)

---

### 2.2 Precision

**Definition:** Of predicted positive instances, how many are actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Range:** [0, 1]
- **1.0:** Zero false positives
- **0.0:** All predictions are false positives

**Target:** > 0.80 per sentiment class

**Interpretation:**
- High precision → when we say "positive," it's usually right
- Low precision → many false alarms

---

### 2.3 Recall

**Definition:** Of actual positive instances, how many did we find?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Range:** [0, 1]
- **1.0:** Zero false negatives (found all positives)
- **0.0:** Missed all positives

**Target:** > 0.80 per sentiment class

**Interpretation:**
- High recall → we find most of the positive posts
- Low recall → we miss many positives

---

### 2.4 F1 Score

**Definition:** Harmonic mean of precision and recall.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Range:** [0, 1]
- **1.0:** Perfect precision and recall
- **0.0:** Either precision or recall is 0

**Target:** > 0.80 (balanced performance)

**Interpretation:**
- F1 balances precision and recall
- Use when you care about both false positives AND false negatives

---

### 2.5 Confusion Matrix

**Definition:** Tabular breakdown of predictions vs. actual.

```
                Predicted Positive    Predicted Negative
Actually Pos        TP                    FN (miss)
Actually Neg        FP (false alarm)      TN
```

**Interpretation:** Shows which classes are confused (e.g., Negative often predicted as Neutral)

---

## 3. Trend Detection Metrics

### 3.1 Precision@K

**Definition:** Of top-K predicted trends, how many are true trends?

$$\text{Precision@K} = \frac{\text{# true trends in top-K}}{\text{K}}$$

**Range:** [0, 1]
- **1.0:** All top-K are real trends
- **0.0:** None of top-K are real trends

**Target:** > 0.80

**Use case:** "How many of our top 10 trending topics should users trust?"

---

### 3.2 Recall@K

**Definition:** Of total true trends, how many appear in top-K?

$$\text{Recall@K} = \frac{\text{# true trends in top-K}}{\text{total true trends}}$$

**Range:** [0, 1]
- **1.0:** All true trends appear in top-K
- **0.0:** No true trends in top-K

**Target:** > 0.75

**Use case:** "What fraction of real trends do we catch?"

---

### 3.3 F1@K

**Definition:** Harmonic mean of Precision@K and Recall@K.

$$F_1@K = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

**Target:** > 0.77

---

### 3.4 NDCG (Normalized Discounted Cumulative Gain)

**Definition:** Ranking quality metric (penalizes misranked items, with more penalty at higher positions).

$$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

where:
$$\text{DCG} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

- $\text{rel}_i$ = relevance score of item at position i (1 if true trend, 0 otherwise)
- Dividing by $\log_2(i+1)$ discounts lower positions

**Range:** [0, 1]
- **1.0:** Perfect ranking (all true trends at top)
- **0.0:** Reverse ranking

**Target:** > 0.70

**Interpretation:**
- High NDCG → real trends are ranked higher
- Better than Precision/Recall for partially-correct rankings

---

### 3.5 Early Detection Time

**Definition:** How many days after trend start before we detect it?

$$\text{Detection Time} = t_{\text{detected}} - t_{\text{start}}$$

**Metric:** Days (or hours, or "time steps")

**Target:** < 2 days

**Interpretation:**
- Emerging trends by definition are detected EARLY
- Early detection time is core research contribution

---

## 4. Forecasting Metrics

### 4.1 MAE (Mean Absolute Error)

**Definition:** Average absolute difference between predicted and actual values.

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Range:** [0, ∞]
- **0:** Perfect predictions
- Larger = worse

**Target:** < 10 (depends on scale of y)

**Interpretation:**
- On average, we're off by this amount
- Interpretable in original units (posts, likes, etc.)

---

### 4.2 RMSE (Root Mean Squared Error)

**Definition:** Square root of average squared error.

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

**Range:** [0, ∞]
- **0:** Perfect predictions
- Penalizes large errors more than MAE

**Target:** < 15 (depends on scale)

**Interpretation:**
- Larger errors are weighted heavily
- Use when outlier misses are costly

---

### 4.3 MAPE (Mean Absolute Percentage Error)

**Definition:** Average error as percentage of actual value.

$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

**Range:** [0, ∞]
- **0%:** Perfect predictions
- **10%:** On average, off by 10% of actual value

**Target:** < 15%

**Interpretation:**
- Scale-independent (useful across different trend sizes)
- "For most predictions, we're within 15% of reality"

---

### 4.4 Directional Accuracy

**Definition:** Fraction of time we predict trend direction (up/down) correctly.

$$\text{DirectionalAcc} = \frac{\# \text{ correct directions}}{N}$$

**Range:** [0, 1]
- **1.0:** Perfect direction prediction
- **0.5:** Coin-flip (random)

**Target:** > 0.70

**Interpretation:**
- "Do we know if a trend is growing or declining?"
- Easier than predicting exact values
- Useful for actionable predictions

---

## 5. System Performance Metrics

### 5.1 Latency

**Definition:** Time from input to results.

$$\text{Latency} = t_{\text{end}} - t_{\text{start}}$$

**Units:** Seconds (or ms for micro-benchmarks)

**Target:** < 5 seconds per 100 posts

**Interpretation:**
- Real-time systems need <1 second
- Batch systems can tolerate minutes
- Trade-off: accuracy vs. speed

---

### 5.2 Memory Usage

**Definition:** Peak RAM used during processing.

**Units:** GB (or MB)

**Target:** < 2 GB for 1000 posts

**Interpretation:**
- Resource-constrained environments have strict limits
- Embeddings and clustering are memory-heavy

---

### 5.3 API Cost

**Definition:** Estimated cost to run per-post through LLM APIs.

**Units:** $ (dollars)

**Calculation:**
```
Cost = (input_tokens * $0.000075 + output_tokens * $0.0003) / posts
```

(Assuming Google Gemini Pro pricing)

**Target:** < $0.01 per post

**Interpretation:**
- 1000 posts should cost < $10
- Trade-off: LLM quality vs. cost

---

## 6. Metrics Used Per Research Question

| RQ | Primary Metrics | Secondary Metrics |
|----|-----------------|-------------------|
| **RQ1** | Coherence, Silhouette, Purity | — |
| **RQ2** | Precision (after cleaning) | Recall, F1 |
| **RQ3** | Precision@K, Recall@K, NDCG | F1@K |
| **RQ4** | Early Detection Time | Precision@K, False Positive Rate |
| **RQ5** | Human rating (interpretability) | Semantic similarity (consistency) |
| **RQ6** | Latency, Memory, API Cost | — |
| **RQ7** | MAPE, DirectionalAcc, MAE/RMSE | — |

---

## 7. Statistical Reporting

For each metric, report:

```
Mean: X.XX
Std Dev: ±Y.YY
95% CI: [L, U]
Min/Max: Z.Z / W.W
```

Example:
```
Coherence: 0.623 ± 0.045 [0.578, 0.668] (min=0.54, max=0.71)
```

Enables:
- Confidence in the estimate (narrow CI = high confidence)
- Consistency across runs (low std dev = reproducible)

---

*All metrics are implemented in `evaluation/metrics.py` with proper handling of edge cases (division by zero, empty clusters, etc.).*
