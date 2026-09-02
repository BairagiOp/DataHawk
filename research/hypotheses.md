# Research Hypotheses

## Overview

Each research question (RQ) is formulated with formal hypotheses that can be tested empirically.

---

## RQ1: Semantic Clustering vs. Frequency-Based Topics

**Research Question:**
Does semantic clustering (embeddings + HDBSCAN) identify more coherent and interpretable topics than frequency-based methods (TF-IDF + K-Means, LDA)?

### Hypotheses

**H1.1 (Coherence):** Semantic clustering achieves higher topic coherence scores than frequency-based baselines.
- **Null:** μ(Coherence_Semantic) ≤ μ(Coherence_TF-IDF) ≤ μ(Coherence_LDA)
- **Alt:** μ(Coherence_Semantic) > max(μ(Coherence_TF-IDF), μ(Coherence_LDA))
- **Metric:** Topic coherence (C_v score, 0-1 scale)
- **Success:** Coherence improvement ≥ 10%

**H1.2 (Interpretability):** Human annotators rate semantic cluster labels as more interpretable than frequency-based labels.
- **Null:** P(Semantic is better) ≤ 50%
- **Alt:** P(Semantic is better) > 60% (or Cohen's Kappa > 0.5 inter-rater agreement)
- **Metric:** Manual rating (Likert 1-5) by 3 independent annotators
- **Success:** Majority preference for semantic approach

**H1.3 (Silhouette Score):** Semantic embeddings produce more separable clusters.
- **Null:** μ(Silhouette_Semantic) ≤ 0.3
- **Alt:** μ(Silhouette_Semantic) > 0.4
- **Metric:** Silhouette coefficient (-1 to 1)
- **Success:** Score > 0.4

---

## RQ2: Preprocessing Impact on Trend Reliability

**Research Question:**
Does preprocessing (deduplication, spam filtering) improve the reliability and precision of trend detection?

### Hypotheses

**H2.1 (Precision):** Removing duplicates and spam reduces false positive trends.
- **Null:** Precision(Raw) ≥ Precision(Cleaned)
- **Alt:** Precision(Cleaned) > Precision(Raw) + 5%
- **Metric:** Precision of detected trends vs. ground truth
- **Success:** Precision improvement ≥ 5%

**H2.2 (Spam Detection):** Heuristic spam filter identifies low-quality content with >70% accuracy.
- **Null:** Accuracy(SpamFilter) ≤ 60%
- **Alt:** Accuracy(SpamFilter) > 70%
- **Metric:** Classification accuracy on annotated spam/non-spam subset (50 samples)
- **Success:** >70% accuracy on validation set

**H2.3 (Duplicate Impact):** Exact + near-duplicate removal prevents artificial trend inflation.
- **Null:** TrendScore(Raw) ≈ TrendScore(Cleaned)
- **Alt:** TrendScore(Cleaned) significantly uncorrelated with raw duplicate count
- **Metric:** Correlation between duplicate count and trend score change
- **Success:** Correlation < 0.3 after deduplication

---

## RQ3: Composite Trend Scoring vs. Simple Baseline

**Research Question:**
Does composite trend scoring (α·Volume + β·Growth + γ·Engagement + δ·Novelty) outperform simple frequency-based baseline?

### Hypotheses

**H3.1 (Ranking Quality):** Composite scoring produces better-ranked trends.
- **Null:** NDCG(Composite) ≤ NDCG(Frequency)
- **Alt:** NDCG(Composite) > NDCG(Frequency) + 10%
- **Metric:** Normalized Discounted Cumulative Gain (NDCG@10) vs. ground truth ranking
- **Success:** NDCG improvement ≥ 10%

**H3.2 (Weight Importance):** Growth (β) is the most important weight factor.
- **Null:** All weights contribute equally (β ≤ 0.35)
- **Alt:** β = 0.4 is optimal, and removing it degrades performance
- **Metric:** Ablation study: performance with β = 0 vs. β = 0.4
- **Success:** Performance drop ≥ 15% when β = 0

**H3.3 (Interpretability):** Human judges agree that composite factors explain trend classification.
- **Null:** P(Agreement) ≤ 50%
- **Alt:** P(Agreement) > 65% with Cohen's Kappa > 0.6
- **Metric:** Humans rate whether explanations justify trend scores
- **Success:** >65% agreement on explanation quality

---

## RQ4: Early Emerging Trend Detection

**Research Question:**
Can we detect emerging trends earlier (with fewer data points) than frequency-based systems?

### Hypotheses

**H4.1 (Detection Time):** Composite scoring detects emerging trends at least 1 day earlier than frequency baseline.
- **Null:** TimeToDetect(Composite) ≥ TimeToDetect(Frequency)
- **Alt:** TimeToDetect(Composite) < TimeToDetect(Frequency) - 24 hours
- **Metric:** Days until trend reaches confidence threshold
- **Success:** Early detection by ≥1 day

**H4.2 (Classification Accuracy):** Emerging trend classification (EMERGING/VIRAL/RISING/STABLE/DECLINING) achieves >75% accuracy.
- **Null:** Accuracy ≤ 60%
- **Alt:** Accuracy > 75%
- **Metric:** Classification accuracy on labeled test set
- **Success:** >75% accuracy

**H4.3 (False Positive Rate):** Emerging trend detector has <20% false positive rate.
- **Null:** FPR ≥ 25%
- **Alt:** FPR < 20%
- **Metric:** % of labeled non-emerging trends classified as emerging
- **Success:** FPR < 20%

---

## RQ5: LLM-Assisted Labeling and Interpretability

**Research Question:**
Does LLM-assisted semantic labeling of topics improve human interpretability compared to automatic keyword extraction?

### Hypotheses

**H5.1 (Label Quality):** LLM-generated labels are rated as more interpretable than top-k keywords.
- **Null:** P(LLM is better) ≤ 50%
- **Alt:** P(LLM is better) > 65%
- **Metric:** Human rating (Likert 1-5) comparing "AI Agents" (LLM) vs. "agents code automate" (keywords)
- **Success:** >65% prefer LLM labels

**H5.2 (Label Consistency):** LLM labels are semantically consistent across multiple runs.
- **Null:** Semantic similarity < 0.7 between runs
- **Alt:** Semantic similarity > 0.85
- **Metric:** Embedding cosine similarity of labels across 3 runs
- **Success:** Similarity > 0.85

**H5.3 (Reproducibility):** LLM labels with low temperature (0.3) reproduce reliably.
- **Null:** Exact match across runs < 70%
- **Alt:** Exact match > 80%
- **Metric:** String exact match of labels across 3 runs
- **Success:** >80% exact match rate

---

## RQ6: Performance vs. Accuracy Trade-offs

**Research Question:**
What computational and financial trade-offs exist between the proposed approach and baselines?

### Hypotheses

**H6.1 (Latency):** End-to-end pipeline latency is <5 seconds per 100 posts.
- **Null:** Latency > 5 sec/100 posts
- **Alt:** Latency < 5 sec/100 posts
- **Metric:** Time from input to predictions
- **Success:** <5 sec/100 posts

**H6.2 (API Cost):** LLM-assisted features add <$0.01 per post to API costs.
- **Null:** Cost > $0.01/post
- **Alt:** Cost < $0.01/post
- **Metric:** Estimated Google Gemini API cost per post
- **Success:** <$0.01/post

**H6.3 (Memory):** Peak memory usage is <2GB for 1000 posts.
- **Null:** Memory > 2GB
- **Alt:** Memory < 2GB
- **Metric:** Measured peak RSS during processing
- **Success:** <2GB

---

## RQ7: Trend Forecasting Accuracy

**Research Question:**
Can we predict short-term (1-3 day) trend growth using historical data?

### Hypotheses

**H7.1 (MAE):** Ensemble forecasting (XGBoost) achieves <15% MAPE on test set.
- **Null:** MAPE ≥ 25%
- **Alt:** MAPE < 15%
- **Metric:** Mean Absolute Percentage Error on holdout test set
- **Success:** MAPE < 15%

**H7.2 (Directional Accuracy):** Forecaster correctly predicts trend direction (up/down) >70% of the time.
- **Null:** Accuracy ≤ 55%
- **Alt:** Accuracy > 70%
- **Metric:** % correct direction predictions
- **Success:** >70% directional accuracy

**H7.3 (Feature Importance):** Historical volume and engagement are more predictive than random features.
- **Null:** Feature importance is random (not significantly above 1/K for K features)
- **Alt:** Volume and engagement importance > 20% each (vs. 10% random baseline)
- **Metric:** Permutation feature importance score
- **Success:** Volume and engagement >20% importance each

---

## Summary Table

| RQ | Hypothesis | Null | Alternative | Success Metric |
|----|-----------|------|-------------|----------------|
| **RQ1** | Semantic coherence | ≤ TF-IDF | > TF-IDF +10% | Coherence ≥ 0.55 |
| **RQ2** | Preprocessing precision | ≥ Raw | > Raw +5% | Precision ≥ 75% |
| **RQ3** | Composite scoring | ≤ Frequency | > Frequency +10% | NDCG ≥ 0.70 |
| **RQ4** | Early detection | ≥ Frequency | < Frequency -1d | Detection ≤ 1d |
| **RQ5** | LLM interpretability | ≤ 50% | > 65% | Human rating >3.5/5 |
| **RQ6** | Latency | > 5 sec/100 | < 5 sec/100 | ≤ 4.5 sec/100 |
| **RQ7** | Forecast MAE | ≥ 25% | < 15% | MAPE ≤ 14% |

---

*These hypotheses are testable and will be verified through experiments on benchmark datasets with multiple runs and statistical validation.*
