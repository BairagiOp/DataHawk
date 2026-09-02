# Research Questions and Hypotheses

**Project:** An Adaptive LLM-Assisted Framework for Social Media Data Parsing, Topic Discovery, Trend Detection, and Emerging Trend Prediction

**Institution:** B.Tech Computer Science Final Year Project  
**Date:** 2026-08-27

---

## 1. PROBLEM STATEMENT

Social media platforms generate vast amounts of unstructured textual data containing valuable signals about emerging topics, trends, and public discourse. Traditional frequency-based approaches to trend detection often identify topics only after they have already become popular, limiting their utility for early-warning systems. Moreover, simple keyword counting fails to capture semantic similarity between posts discussing the same topic with different terminology.

Existing approaches have the following limitations:

1. **Frequency Bias:** Most-mentioned topics are not necessarily emerging or novel
2. **Semantic Blindness:** Keyword matching misses semantically similar content
3. **Lack of Explainability:** Trend detection systems rarely explain why a topic is trending
4. **Late Detection:** Trends are identified only after reaching peak popularity
5. **Noise Sensitivity:** Duplicate content and spam inflate trend scores
6. **Single-Signal Reliance:** Most systems use only volume, ignoring engagement, velocity, and novelty

**Research Gap:** There is a need for an integrated framework that combines:
- Semantic understanding (LLM-assisted parsing + embeddings)
- Temporal dynamics (growth rate, acceleration, velocity)
- Engagement signals (likes, shares, comments)
- Novelty detection (emergence from low baseline)
- Noise filtering (duplicate/spam removal)
- Explainability (show why a topic is trending)

---

## 2. RESEARCH QUESTIONS

### RQ1: Topic Discovery Quality

**Question:** Does semantic clustering using sentence embeddings identify more coherent social-media topics than frequency-based approaches (TF-IDF + clustering, LDA)?

**Motivation:** Posts about the same topic often use different terminology. For example:
- "AI agents are transforming software development"
- "Autonomous coding assistants are becoming mainstream"
- "LLM-powered development tools are changing programming"

These should cluster together semantically, even though they share few exact keywords.

**Hypothesis (H1):** Semantic clustering using sentence embeddings (e.g., sentence-transformers) will produce higher topic coherence scores and better cluster separation (silhouette score) than frequency-based methods (TF-IDF + K-Means, LDA).

**Evaluation Metrics:**
- Silhouette Score (cluster separation)
- Topic Coherence (semantic consistency within clusters)
- Cluster Purity (against ground truth where available)
- Human evaluation of topic interpretability

**Baseline Methods:**
- Baseline 1: TF-IDF + K-Means clustering
- Baseline 2: Latent Dirichlet Allocation (LDA)
- Proposed: Sentence embeddings + HDBSCAN/K-Means

**Experimental Design:**
- Dataset: Social media posts (minimum 5,000 posts across 10+ topics)
- Compare clustering quality across methods
- Use multiple k values for K-Means
- Measure coherence using c_v coherence metric where applicable
- Conduct human evaluation on sample clusters

---

### RQ2: Noise Filtering Impact

**Question:** Does duplicate/near-duplicate detection and spam filtering improve the reliability and accuracy of trend detection?

**Motivation:** Social media contains substantial noise:
- Exact duplicates (reposts, retweets)
- Near-duplicates (minor variations)
- Spam and promotional content
- Bot-generated content

This noise can artificially inflate trend scores.

**Hypothesis (H2):** Removing duplicates, near-duplicates, and spam-like content will:
1. Reduce false positive trend detections by at least 20%
2. Improve precision of trend detection by at least 15%
3. Improve topic coherence scores by at least 10%

**Evaluation Metrics:**
- Trend detection precision/recall/F1 (with and without filtering)
- False positive rate reduction
- Topic coherence improvement
- Number of duplicate clusters eliminated

**Baseline Methods:**
- No filtering (raw data)
- Exact duplicate removal only
- Exact + near-duplicate removal
- Full pipeline (exact + near-duplicate + spam filtering)

**Experimental Design:**
- Create dataset with known duplicates and spam
- Measure trend detection quality before and after filtering
- Calculate reduction in false positives
- Statistical significance test (paired t-test or Wilcoxon)

---

### RQ3: Composite Trend Scoring

**Question:** Does combining volume, growth rate, engagement, and semantic novelty outperform simple frequency-based trend detection?

**Motivation:** A topic that suddenly increases from 10 to 100 posts (10x growth) is more "trending" than a topic that increases from 10,000 to 10,100 posts (1% growth), even though the latter has more absolute volume.

Traditional frequency-based approaches rank by volume only, missing emerging signals.

**Proposed Trend Score Formula:**

```
TrendScore(topic, t) = α·V(t) + β·G(t) + γ·E(t) + δ·N(t)

Where:
V(t) = Normalized volume (post count)
G(t) = Growth rate (% change from previous period)
E(t) = Normalized engagement (likes, shares, comments)
N(t) = Novelty score (topic emergence from low baseline)

α, β, γ, δ = configurable weights (sum to 1.0)
```

**Hypothesis (H3):** The composite trend score will achieve:
1. Higher precision in identifying emerging topics (vs. just popular topics)
2. Earlier detection of trends before peak popularity
3. Better correlation with human judgments of "trending topics"

**Evaluation Metrics:**
- Precision@k (top k detected trends match ground truth)
- Recall@k
- Early detection time (hours/days before frequency-based detection)
- Normalized Discounted Cumulative Gain (NDCG)
- Kendall's Tau correlation with human rankings

**Baseline Methods:**
- Baseline 1: Pure frequency ranking (most mentioned keywords/topics)
- Baseline 2: TF-IDF ranking
- Baseline 3: Volume + growth only
- Proposed: Full composite score (volume + growth + engagement + novelty)

**Experimental Design:**
- Collect time-series social media data
- Identify known trends (manual labeling or historical data)
- Compare detection precision and timing across methods
- Vary weight parameters (ablation study)
- Conduct human evaluation study

---

### RQ4: Early Trend Detection

**Question:** Can the proposed system identify emerging trends earlier than frequency-based baselines?

**Motivation:** The primary value of a trend detection system is early warning—identifying topics before they reach mainstream popularity. A system that only detects trends after they peak has limited practical value.

**Hypothesis (H4):** The proposed system (composite score with novelty detection) will detect emerging trends an average of 12-48 hours earlier than frequency-based methods.

**Evaluation Metrics:**
- Early detection advantage (time difference in hours/days)
- Percentage of trends detected early (before baseline)
- False alarm rate (topics flagged as emerging but failed to trend)
- Area Under Curve (AUC) for temporal detection

**Experimental Design:**
- Identify historical trends with known start dates
- Run detection algorithms on historical data in sliding windows
- Measure first detection time for each method
- Calculate time advantage
- Statistical test (paired samples)

**Ground Truth Construction:**
- Use external trend databases (e.g., Google Trends, Twitter Trends archives)
- Manual annotation of trend start dates
- Multiple annotators for reliability

---

### RQ5: LLM-Assisted Topic Labeling

**Question:** Does LLM-assisted semantic labeling improve human interpretability and accuracy of discovered topics compared to keyword-based topic representation?

**Motivation:** Traditional topic models (LDA, TF-IDF clustering) represent topics as lists of keywords:
- Topic 1: ["AI", "agent", "coding", "software", "automation"]

This requires human interpretation. LLM-assisted labeling can generate semantic descriptions:
- Topic 1: "AI-powered coding agents and software automation"

**Hypothesis (H5):** LLM-generated topic labels will:
1. Achieve higher human agreement scores (inter-rater reliability)
2. Be rated as more interpretable in user studies
3. Achieve higher accuracy when matched to ground truth topic names

**Evaluation Metrics:**
- Human interpretability rating (1-5 Likert scale)
- Agreement with ground truth labels (where available)
- Inter-annotator agreement (Cohen's Kappa)
- Label accuracy score

**Baseline Methods:**
- Top-k keywords only
- Top-k keywords + example post
- LLM-generated semantic label
- LLM-generated label + keywords

**Experimental Design:**
- Generate topics using clustering
- Create labels using each method
- Conduct user study with human raters
- Measure interpretability and accuracy
- Calculate inter-rater reliability

---

### RQ6: Efficiency Trade-offs

**Question:** What is the trade-off between trend detection accuracy, computational cost, latency, and LLM API usage?

**Motivation:** Research systems must balance accuracy with practical constraints:
- LLM API calls have monetary cost
- Embedding generation requires computation
- Real-time trend detection requires low latency

**Hypothesis (H6):** The proposed system will achieve:
1. Higher accuracy than frequency baselines despite higher cost
2. Acceptable latency (<10 seconds per 1000 posts for analysis)
3. Cost-effectiveness ratio superior to naive LLM-based approaches

**Evaluation Metrics:**
- Accuracy (F1, precision, recall)
- Total LLM tokens consumed
- Estimated API cost (USD)
- End-to-end latency (seconds)
- Cost per correctly detected trend (USD)
- Throughput (posts processed per second)

**Experimental Design:**
- Measure all methods on same dataset
- Track token usage for each LLM call
- Measure execution time
- Calculate cost-effectiveness ratio
- Create Pareto frontier (accuracy vs. cost)

---

### RQ7: Trend Forecasting Accuracy

**Question:** Can historical topic activity be used to predict short-term future trend growth with acceptable accuracy?

**Motivation:** Predicting whether an emerging topic will continue to grow or plateau is valuable for decision-making. However, social media trends are notoriously unpredictable.

**Hypothesis (H7):** Machine learning models trained on historical features (past volume, growth rate, engagement, velocity) can predict next-period activity with:
1. MAE (Mean Absolute Error) < 30% of actual value
2. Directional accuracy > 65% (predict up/down correctly)
3. Better performance than naive baseline (carry-forward)

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy (% of correct up/down predictions)
- R² score

**Baseline Methods:**
- Naive forecast (last value carried forward)
- Moving average (window = 3, 7 periods)
- Linear regression on time
- Exponential smoothing

**Proposed Methods:**
- Random Forest Regressor
- XGBoost
- LSTM (if dataset size permits)

**Experimental Design:**
- Split data into train/validation/test (70/15/15)
- Use time-series cross-validation
- Train on historical windows
- Predict 1-day ahead, 3-day ahead, 7-day ahead
- Measure forecast accuracy
- Compare across methods

**Feature Engineering:**
- Past volume (t-1, t-2, ..., t-7)
- Growth rate trends
- Engagement metrics
- Day of week, hour effects
- Topic novelty decay

---

## 3. SUMMARY OF RESEARCH CONTRIBUTIONS

If hypotheses are supported by experimental evidence, this research will contribute:

### Primary Contributions:

1. **An integrated framework** combining semantic understanding, temporal dynamics, engagement signals, and novelty detection for social media trend analysis

2. **Experimental validation** showing that composite trend scoring outperforms frequency-based methods in precision and early detection time

3. **A benchmark and evaluation protocol** for comparing social media trend detection approaches

### Secondary Contributions:

4. An explainable trend detection mechanism that shows measurable factors contributing to trend classification

5. Empirical analysis of the impact of noise filtering on trend detection quality

6. Comparative evaluation of topic modeling approaches for social media data

7. Evaluation of LLM-assisted semantic labeling for topic interpretability

---

## 4. LIMITATIONS AND CONSTRAINTS

### Dataset Limitations:
- Social media data collection is constrained by platform Terms of Service
- May rely on publicly available datasets or CSV uploads
- Ground truth annotations are time-consuming and subjective

### Methodological Limitations:
- Trend prediction is inherently uncertain
- Semantic embeddings may not capture all cultural context
- LLM-based methods depend on model quality and cost

### Evaluation Limitations:
- Human evaluation is expensive and has limited sample size
- Ground truth for "emerging trends" is partly subjective
- Temporal evaluation requires historical data with known outcomes

### Ethical Limitations:
- Cannot scrape private accounts or bypass authentication
- Must respect rate limits and platform policies
- Author privacy must be protected (anonymization/hashing)

---

## 5. SUCCESS CRITERIA

This research will be considered successful if:

1. ✅ **RQ1-RQ4** show statistically significant improvements over baselines
2. ✅ At least 3 of the 7 hypotheses are supported by experimental evidence
3. ✅ The system achieves reproducible results with documented methodology
4. ✅ The evaluation framework enables fair comparison with baselines
5. ✅ The work is suitable for submission to an IEEE conference or workshop

**Minimum Viable Contribution:**
- Demonstrate that composite scoring outperforms frequency ranking (RQ3)
- Show measurable early detection advantage (RQ4)
- Provide reproducible benchmark and evaluation framework

---

## 6. RESEARCH QUESTIONS MAPPED TO IMPLEMENTATION

| Research Question | Module | Baseline | Evaluation Metric | Priority |
|-------------------|--------|----------|-------------------|----------|
| RQ1: Topic Quality | `topics/clustering.py` | TF-IDF, LDA | Silhouette, Coherence | HIGH |
| RQ2: Noise Filtering | `preprocessing/deduplication.py` | No filtering | Precision improvement | HIGH |
| RQ3: Composite Scoring | `trends/trend_score.py` | Frequency | Precision@k, NDCG | HIGH |
| RQ4: Early Detection | `trends/emerging.py` | Frequency | Time advantage | HIGH |
| RQ5: LLM Labeling | `topics/labeling.py` | Keywords | Human rating | MEDIUM |
| RQ6: Efficiency | `evaluation/metrics.py` | All | Cost/accuracy ratio | MEDIUM |
| RQ7: Forecasting | `trends/forecasting.py` | Naive, MA | MAE, RMSE | LOW |

**Implementation Priority:**
1. Core pipeline (RQ1, RQ2, RQ3, RQ4) - Essential for paper
2. Explainability and efficiency (RQ5, RQ6) - Strong contribution
3. Forecasting (RQ7) - Optional extension

---

## 7. TIMELINE AND MILESTONES

**Week 1-2:** Data collection and preprocessing pipeline  
**Week 3-4:** NLP analysis and topic discovery implementation  
**Week 5-6:** Trend detection and emerging trend classification  
**Week 7:** Baseline implementations and evaluation framework  
**Week 8-9:** Experiments and data collection  
**Week 10-11:** Analysis, ablation studies, and statistical tests  
**Week 12:** Paper writing and final dashboard polish

---

*Document Version: 1.0*  
*Last Updated: 2026-08-27*  
*Status: Draft - Awaiting Advisor Review*
