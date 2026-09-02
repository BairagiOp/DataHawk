# Research Methodology

**Project:** An Adaptive LLM-Assisted Framework for Social Media Data Parsing, Topic Discovery, Trend Detection, and Emerging Trend Prediction

**Date:** 2026-08-27

---

## 1. RESEARCH DESIGN

### 1.1 Research Type

**Type:** Experimental Research with Quantitative Evaluation

**Approach:** Design Science Research Methodology (DSRM)

**Phases:**
1. Problem identification and motivation
2. Objectives of solution
3. Design and development
4. Demonstration
5. Evaluation
6. Communication

### 1.2 Research Philosophy

This research follows an **empirical, evidence-based approach**:
- Measurable metrics over subjective claims
- Baseline comparisons for fairness
- Statistical validation where applicable
- Reproducible experiments
- Open acknowledgment of limitations

---

## 2. SYSTEM ARCHITECTURE METHODOLOGY

### 2.1 Design Principles

1. **Modularity:** Each component is independently testable and replaceable
2. **Extensibility:** New data sources, NLP models, and trend algorithms can be added
3. **Reproducibility:** All experiments are logged with parameters and random seeds
4. **Efficiency:** Balance accuracy with computational cost
5. **Explainability:** Every prediction must be traceable to source data
6. **Ethical Compliance:** Respect platform ToS and privacy

### 2.2 Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│ Layer 1: Data Ingestion (ingestion/)                  │
│ - Adapter pattern for multiple data sources            │
│ - Unified schema transformation                        │
│ - Validation and error handling                        │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Layer 2: Preprocessing (preprocessing/)               │
│ - Text cleaning (social media specific)                │
│ - Language detection                                   │
│ - Duplicate/near-duplicate detection                   │
│ - Spam/noise filtering                                 │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Layer 3: NLP Analysis (nlp/)                          │
│ - LLM-assisted semantic parsing                        │
│ - Sentiment analysis (baseline + LLM)                  │
│ - Emotion classification                               │
│ - Named Entity Recognition                             │
│ - Keyword/hashtag extraction                           │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Layer 4: Topic Discovery (topics/)                    │
│ - Semantic embedding generation                        │
│ - Clustering (K-Means, HDBSCAN)                        │
│ - LLM-based topic labeling                             │
│ - Coherence evaluation                                 │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Layer 5: Trend Detection (trends/)                    │
│ - Temporal aggregation                                 │
│ - Composite trend scoring                              │
│ - Emerging trend classification                        │
│ - Explainable predictions                              │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Layer 6: Evaluation (evaluation/)                     │
│ - Metrics calculation                                  │
│ - Baseline comparison                                  │
│ - Ablation studies                                     │
│ - Statistical tests                                    │
└────────────────────────────────────────────────────────┘
```

---

## 3. DATA COLLECTION METHODOLOGY

### 3.1 Data Sources (Ethical Constraints)

Due to platform Terms of Service and ethical considerations, data will be collected from:

1. **User-provided datasets:** CSV/JSON files uploaded by researchers
2. **Public benchmark datasets:** Existing research datasets (with proper citation)
3. **Permitted APIs:** Only APIs that explicitly allow research use
4. **Public web pages:** News articles, blog posts (respecting robots.txt)
5. **Synthetic/augmented data:** Generated for testing purposes (clearly labeled)

**Explicitly NOT included:**
- Scraped data from platforms that prohibit scraping
- Data obtained by bypassing authentication
- Private messages or accounts
- Data violating rate limits

### 3.2 Data Schema

Every record will conform to the **Social Media Post Schema**:

```json
{
  "post_id": "unique_identifier",
  "platform": "twitter|reddit|news|csv|synthetic",
  "timestamp": "2026-08-27T12:00:00Z",
  "author_id_hash": "anonymized_hash_or_null",
  "text": "original post content",
  "language": "en|hi|unknown",
  "url": "source_url_if_available",
  "hashtags": ["AI", "Technology"],
  "mentions": ["@user"],
  "media_type": "text|image|video|link",
  "engagement": {
    "likes": 0,
    "comments": 0,
    "shares": 0,
    "views": 0
  },
  "metadata": {
    "collection_date": "2026-08-27",
    "source": "csv_upload",
    "batch_id": "batch_001"
  }
}
```

### 3.3 Dataset Requirements

**Minimum Dataset Size:**
- Training: 5,000 posts minimum
- Evaluation: 1,000 posts minimum
- At least 5 distinct topics
- Temporal span: At least 7 days (for trend analysis)

**Desired Dataset Characteristics:**
- Multiple topics with varying popularity
- Mix of emerging and established topics
- Temporal dynamics (growth/decline patterns)
- Ground truth annotations for subset (100-500 posts)

### 3.4 Ground Truth Annotation

**Annotation Protocol:**

1. **Sampling:** Stratified random sample from each time period
2. **Annotators:** Minimum 2 annotators per sample
3. **Annotation Fields:**
   - Topic category (from predefined list)
   - Sentiment (positive/negative/neutral/mixed)
   - Emotion (joy/anger/sadness/fear/surprise/disgust/neutral)
   - Relevance (relevant/irrelevant/spam)
   - Trend status (emerging/stable/declining) [for subset]

4. **Inter-Annotator Agreement:**
   - Calculate Cohen's Kappa
   - Resolve disagreements through discussion
   - Target: Kappa > 0.6 (substantial agreement)

5. **Annotation Tool:**
   - Simple CSV-based annotation template
   - Or web-based annotation interface (if time permits)

---

## 4. PREPROCESSING METHODOLOGY

### 4.1 Text Cleaning Pipeline

**Social Media Text Cleaning:**

```python
# Pseudocode for cleaning pipeline
def clean_social_media_text(text):
    original_text = text
    
    # 1. Normalize unicode
    text = normalize_unicode(text)
    
    # 2. Extract and preserve hashtags
    hashtags = extract_hashtags(text)
    
    # 3. Extract and preserve mentions
    mentions = extract_mentions(text)
    
    # 4. Handle URLs
    urls = extract_urls(text)
    text = replace_urls_with_token(text, "[URL]")
    
    # 5. Handle emojis (preserve meaning, remove repetition)
    text = normalize_emojis(text)
    
    # 6. Normalize repeated characters
    text = normalize_repeated_chars(text)  # "sooo good" → "so good"
    
    # 7. Remove excessive whitespace
    text = normalize_whitespace(text)
    
    # 8. Remove HTML entities
    text = decode_html_entities(text)
    
    return {
        "original_text": original_text,
        "clean_text": text,
        "hashtags": hashtags,
        "mentions": mentions,
        "urls": urls
    }
```

**Important:** Both `original_text` and `clean_text` are preserved for traceability.

### 4.2 Language Detection

**Method:** Use `langdetect` library (Google's language detection)

**Supported Languages:**
- English (en)
- Hindi (hi)
- Hinglish detection (heuristic: script mixing)
- Other (detected by library)
- Unknown (detection failure)

**Evaluation:** Measure accuracy on manually labeled subset.

### 4.3 Duplicate Detection

**Exact Duplicate Detection:**
```python
def detect_exact_duplicates(posts):
    seen_hashes = set()
    duplicates = []
    for post in posts:
        normalized = normalize_for_dedup(post.clean_text)
        hash_val = hash(normalized)
        if hash_val in seen_hashes:
            duplicates.append(post)
        else:
            seen_hashes.add(hash_val)
    return duplicates
```

**Near-Duplicate Detection:**
- Compute embedding similarity (cosine distance)
- Threshold: similarity > 0.95 = near-duplicate
- Use MinHash LSH for efficiency if dataset is large

**Evaluation:** Measure impact on trend detection precision.

### 4.4 Spam/Noise Detection

**Heuristic Rules:**
- Excessive repetition (character n-grams)
- URL-to-text ratio > 0.5
- Promotional keywords (>3 per post)
- Identical posts from same author (>5)
- Malformed text (invalid UTF-8, excessive special chars)

**Scoring:**
```python
spam_score = (
    repetition_score * 0.3 +
    url_ratio * 0.2 +
    promotional_score * 0.3 +
    frequency_abuse * 0.2
)

is_spam = spam_score > 0.6
```

**Evaluation:** Measure impact on topic coherence and trend precision.

---

## 5. NLP ANALYSIS METHODOLOGY

### 5.1 LLM-Assisted Semantic Parsing

**Method:** Extend existing DataHawk `core/extractor.py`

**Schema for Social Media:**
```json
{
  "fields": {
    "main_topic": {"type": "string", "description": "Primary topic discussed"},
    "entities": {"type": "array", "description": "Named entities mentioned"},
    "keywords": {"type": "array", "description": "Key terms"},
    "stance": {"type": "string", "description": "positive|negative|neutral|mixed"},
    "intent": {"type": "string", "description": "opinion|news|question|promotion"},
    "summary": {"type": "string", "description": "One-sentence summary"}
  }
}
```

**LLM Prompt:**
```
Extract structured information from this social media post.

Post: "{text}"

Return JSON with: main_topic, entities, keywords, stance, intent, summary.

Rules:
- Only extract information present in the text
- If uncertain, return "unknown"
- Do not invent or guess information
```

**Evaluation:** Compare extraction accuracy with ground truth annotations.

### 5.2 Sentiment Analysis

**Baseline Method:** VADER sentiment analyzer (rule-based, optimized for social media)

**Proposed Method:** LLM-based sentiment classification

**Prompt:**
```
Classify the sentiment of this text: "{text}"

Return one of: positive, negative, neutral, mixed

Sentiment:
```

**Evaluation Metrics:**
- Accuracy
- Precision, Recall, F1 (per class)
- Confusion matrix
- Macro F1 score

**Comparison:** Baseline vs. LLM on annotated test set.

### 5.3 Emotion Analysis

**Method:** LLM-based classification (no robust baseline for 7 emotions)

**Classes:** Joy, Anger, Sadness, Fear, Surprise, Disgust, Neutral

**Prompt:**
```
Classify the emotion expressed in this text: "{text}"

Choose one: joy, anger, sadness, fear, surprise, disgust, neutral

Emotion:
```

**Evaluation:** Accuracy on annotated subset (if annotations available).

### 5.4 Named Entity Recognition

**Method:** spaCy NER (en_core_web_sm or en_core_web_lg)

**Entity Types:**
- PERSON
- ORGANIZATION (ORG)
- LOCATION (GPE, LOC)
- PRODUCT
- EVENT
- Technology/Brand (custom extension if needed)

**Storage:**
```json
{
  "entity_text": "OpenAI",
  "entity_type": "ORG",
  "confidence": 0.95,
  "start_char": 10,
  "end_char": 16
}
```

**Evaluation:** Entity extraction recall on annotated sample.

### 5.5 Keyword & Hashtag Extraction

**Hashtag Extraction:** Regex pattern `#\w+`

**Keyword Extraction Methods:**
1. TF-IDF (top k terms per post)
2. KeyBERT (embedding-based keyword extraction)
3. YAKE (statistical keyword extraction)

**Co-occurrence Analysis:**
- Build hashtag co-occurrence matrix
- Identify frequently co-occurring pairs
- Visualize as network graph

**Evaluation:** Correlation with manual keyword annotations.

---

## 6. TOPIC DISCOVERY METHODOLOGY

### 6.1 Embedding Generation

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Fast inference (suitable for large datasets)
- Good semantic similarity performance

**Alternative (if needed):** `sentence-transformers/all-mpnet-base-v2`
- 768-dimensional embeddings
- Higher quality, slower inference

**Process:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(posts['clean_text'].tolist())
```

### 6.2 Dimensionality Reduction (Optional)

**Method:** UMAP (Uniform Manifold Approximation and Projection)
- Reduce to 5-50 dimensions for clustering efficiency
- Preserve local and global structure

**Parameters:**
- n_components = 10 (default)
- n_neighbors = 15
- min_dist = 0.0
- metric = 'cosine'

### 6.3 Clustering Algorithms

**Method 1: K-Means**
- Simple, fast, interpretable
- Requires specifying k (number of clusters)
- Evaluation: silhouette score, elbow method

**Method 2: HDBSCAN** (Hierarchical Density-Based Clustering)
- Automatically determines number of clusters
- Handles noise and varying density
- Better for real-world data

**Parameters (HDBSCAN):**
- min_cluster_size = 50 (adjust based on dataset)
- min_samples = 10
- metric = 'euclidean' (on reduced embeddings)

**Cluster Selection:**
- Remove noise cluster (label = -1)
- Require minimum cluster size (e.g., 30 posts)
- Maximum number of clusters = 50 (for interpretability)

### 6.4 Topic Labeling

**Baseline:** Top-5 TF-IDF terms from cluster

**Proposed:** LLM-based semantic labeling

**Prompt:**
```
These are posts from a social media topic cluster:

Post 1: "{post1}"
Post 2: "{post2}"
Post 3: "{post3}"
...
Post 10: "{post10}"

Generate a concise topic label (max 5 words) that describes what these posts are about.

Topic Label:
```

**Evaluation:** Human raters compare interpretability (baseline vs. LLM labels).

### 6.5 Topic Coherence Evaluation

**Metric:** C_V coherence (if applicable)

**Calculation:**
```python
from gensim.models.coherencemodel import CoherenceModel

# For each cluster, get top terms
cluster_terms = get_top_terms_per_cluster(clusters)

# Calculate coherence
coherence_model = CoherenceModel(
    topics=cluster_terms,
    texts=tokenized_posts,
    dictionary=dictionary,
    coherence='c_v'
)
coherence_score = coherence_model.get_coherence()
```

**Baseline Comparison:**
- TF-IDF + K-Means coherence
- LDA coherence
- Embedding + clustering coherence

---

## 7. TREND DETECTION METHODOLOGY

### 7.1 Temporal Aggregation

**Time Windows:**
- Hourly (if sufficient data)
- Daily (default)
- Weekly (for longer-term trends)

**Aggregation:**
```python
def aggregate_by_time_window(posts, topic_assignments, window='1D'):
    df = pd.DataFrame(posts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['topic'] = topic_assignments
    
    aggregated = df.groupby([pd.Grouper(key='timestamp', freq=window), 'topic']).agg({
        'post_id': 'count',  # volume
        'likes': 'sum',
        'comments': 'sum',
        'shares': 'sum',
        'views': 'sum'
    }).rename(columns={'post_id': 'volume'})
    
    return aggregated
```

### 7.2 Feature Engineering for Trend Detection

**For each topic at time t:**

1. **Volume Features:**
   - V(t) = post count at time t
   - V_norm(t) = (V(t) - V_min) / (V_max - V_min)

2. **Growth Features:**
   - G(t) = (V(t) - V(t-1)) / V(t-1)  [percentage growth]
   - G_abs(t) = V(t) - V(t-1)  [absolute growth]
   - Velocity = average growth over last 3 periods

3. **Engagement Features:**
   - E(t) = likes + comments + shares + views
   - E_norm(t) = normalized engagement
   - E_growth(t) = (E(t) - E(t-1)) / E(t-1)

4. **Novelty Features:**
   - First appearance time
   - Days since first appearance
   - Novelty score = exp(-days_since_first / decay_constant)

5. **Acceleration:**
   - A(t) = G(t) - G(t-1)  [second derivative]

### 7.3 Composite Trend Score Formula

**Proposed Formula:**

```
TrendScore(topic, t) = α·V_norm(t) + β·G(t) + γ·E_norm(t) + δ·N(t)

Where:
- α = 0.2  (volume weight)
- β = 0.4  (growth weight)
- γ = 0.3  (engagement weight)
- δ = 0.1  (novelty weight)

All features normalized to [0, 1]
```

**Normalization:**
- Min-max normalization per feature
- Applied across all topics in the current time window

**Ablation Study:**
Test different weight configurations:
- Equal weights (0.25, 0.25, 0.25, 0.25)
- Volume-only (1.0, 0, 0, 0)
- Volume + growth (0.5, 0.5, 0, 0)
- Proposed (0.2, 0.4, 0.3, 0.1)

### 7.4 Emerging Trend Classification

**Trend Categories:**

1. **EMERGING:** Low baseline → rapid growth
   - Criteria: V(t-3) < threshold_low AND G(t) > growth_threshold
   
2. **RISING:** Consistent growth
   - Criteria: G(t) > 0 for last 3 periods
   
3. **VIRAL:** Explosive growth
   - Criteria: G(t) > viral_threshold (e.g., 200%)
   
4. **STABLE:** Consistent volume
   - Criteria: |G(t)| < stability_threshold (e.g., 10%)
   
5. **DECLINING:** Decreasing volume
   - Criteria: G(t) < 0 for last 2 periods

**Thresholds (configurable):**
- threshold_low = 10 posts
- growth_threshold = 50%
- viral_threshold = 200%
- stability_threshold = 10%

### 7.5 Explainable Trend Detection

**For each trend, generate explanation:**

```python
def explain_trend(topic, t, trend_score, features):
    explanation = f"Topic '{topic}' classified as {features['category']} because:\n\n"
    
    if features['volume_growth'] > 50:
        explanation += f"• Post volume increased by {features['volume_growth']:.0f}%\n"
    
    if features['engagement_growth'] > 30:
        explanation += f"• Engagement increased by {features['engagement_growth']:.0f}%\n"
    
    if features['novelty'] > 0.7:
        explanation += f"• Topic is relatively new (novelty score: {features['novelty']:.2f})\n"
    
    if features['velocity'] > 20:
        explanation += f"• Growth velocity is high ({features['velocity']:.1f} posts/day)\n"
    
    explanation += f"\nTrend Score: {trend_score:.3f}"
    
    return explanation
```

**Output Example:**
```
Topic 'AI Coding Agents' classified as EMERGING because:

• Post volume increased by 173%
• Engagement increased by 128%
• Topic appeared across 4 sources
• Growth velocity is high (45.2 posts/day)

Trend Score: 0.847
```

---

## 8. BASELINE IMPLEMENTATION METHODOLOGY

### 8.1 Baseline 1: Frequency-Based Ranking

**Method:** Count keyword/hashtag mentions, rank by frequency

```python
def frequency_baseline(posts):
    keywords = extract_all_keywords(posts)
    frequency = Counter(keywords)
    return frequency.most_common(50)
```

**No temporal dynamics, no semantic understanding.**

### 8.2 Baseline 2: TF-IDF Ranking

**Method:** TF-IDF scores, rank by importance

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_baseline(posts):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(posts['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return ranked[:50]
```

### 8.3 Baseline 3: Volume + Growth Only

**Method:** Composite score without engagement or novelty

```python
def volume_growth_baseline(aggregated_data):
    scores = 0.5 * V_norm + 0.5 * G_norm
    return scores
```

### 8.4 Baseline 4: LLM Topic Extraction (Raw)

**Method:** Send raw posts to LLM, ask for top topics

```python
def llm_raw_baseline(posts_sample):
    prompt = f"""
    These are social media posts. What are the top 10 topics discussed?
    
    Posts:
    {posts_sample}
    
    Return JSON: {{"topics": ["topic1", "topic2", ...]}}
    """
    response = llm.generate_json(prompt)
    return response['topics']
```

**No clustering, no structured analysis.**

---

## 9. EVALUATION METHODOLOGY

### 9.1 Metrics for Topic Discovery (RQ1)

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(embeddings, cluster_labels, metric='cosine')
# Range: [-1, 1], higher is better
```

**Topic Coherence:** C_V coherence using Gensim

**Cluster Purity (if ground truth):**
```python
def cluster_purity(true_labels, predicted_labels):
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels, predicted_labels)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
```

### 9.2 Metrics for Trend Detection (RQ3, RQ4)

**Precision@k:**
```python
def precision_at_k(predicted_trends, ground_truth_trends, k=10):
    predicted_top_k = set(predicted_trends[:k])
    true_positives = len(predicted_top_k & ground_truth_trends)
    return true_positives / k
```

**Recall@k, F1@k:** Similarly defined

**Early Detection Time:**
```python
def early_detection_advantage(baseline_detection_time, proposed_detection_time):
    return baseline_detection_time - proposed_detection_time  # in hours or days
```

**NDCG (Normalized Discounted Cumulative Gain):**
```python
from sklearn.metrics import ndcg_score

ndcg = ndcg_score([ground_truth_relevance], [predicted_scores])
```

### 9.3 Statistical Tests

**Paired T-Test (for comparing methods on same dataset):**
```python
from scipy.stats import ttest_rel

statistic, p_value = ttest_rel(baseline_scores, proposed_scores)
# If p_value < 0.05, difference is statistically significant
```

**Effect Size (Cohen's d):**
```python
def cohens_d(group1, group2):
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return diff / pooled_std
# |d| > 0.5 = medium effect, |d| > 0.8 = large effect
```

### 9.4 Ablation Study Design

**Test configurations:**

| Config | Volume | Growth | Engagement | Novelty | Deduplication | Spam Filter |
|--------|--------|--------|------------|---------|---------------|-------------|
| A      | ✓      |        |            |         |               |             |
| B      | ✓      | ✓      |            |         |               |             |
| C      | ✓      | ✓      | ✓          |         |               |             |
| D      | ✓      | ✓      | ✓          | ✓       |               |             |
| E      | ✓      | ✓      | ✓          | ✓       | ✓             |             |
| F (Full)| ✓     | ✓      | ✓          | ✓       | ✓             | ✓           |

**Measure trend detection F1 for each configuration.**

---

## 10. EXPERIMENT EXECUTION PROTOCOL

### 10.1 Experiment Logging

**Every experiment run logs:**

```python
experiment_log = {
    "experiment_id": "exp_001",
    "timestamp": "2026-08-27T12:00:00Z",
    "dataset": "synthetic_tech_trends_v1",
    "method": "proposed_full_pipeline",
    "parameters": {
        "alpha": 0.2,
        "beta": 0.4,
        "gamma": 0.3,
        "delta": 0.1,
        "clustering_algorithm": "hdbscan",
        "min_cluster_size": 50
    },
    "metrics": {
        "precision_at_10": 0.85,
        "recall_at_10": 0.72,
        "f1_at_10": 0.78,
        "silhouette_score": 0.42,
        "avg_latency_ms": 3450.2,
        "total_llm_tokens": 125000,
        "estimated_cost_usd": 0.045
    },
    "git_commit": "abc123def",
    "random_seed": 42
}
```

**Storage:** SQLite database + CSV export

### 10.2 Reproducibility Checklist

✅ Random seed fixed for all experiments  
✅ All parameters logged  
✅ Dataset version tracked  
✅ Code version tracked (git commit hash)  
✅ Dependencies version locked (requirements.txt)  
✅ Experiment scripts provided  

### 10.3 Experiment Scripts

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py --input data/raw/posts.csv --output dataset/processed/

# 2. Run baselines
python scripts/run_baselines.py --dataset dataset/processed/ --output results/baselines/

# 3. Run proposed method
python scripts/run_experiments.py --dataset dataset/processed/ --config configs/proposed.yaml

# 4. Run ablation study
python scripts/run_ablation.py --dataset dataset/processed/ --output results/ablation/

# 5. Evaluate and compare
python scripts/evaluate.py --results results/ --output results/final_metrics.csv

# 6. Generate visualizations
python scripts/visualize_results.py --results results/ --output results/figures/
```

---

## 11. LIMITATIONS AND MITIGATION

### 11.1 Methodological Limitations

**Limitation 1:** Ground truth for "emerging trends" is subjective
- **Mitigation:** Use multiple sources (human annotations + historical data + external trend databases)

**Limitation 2:** Limited dataset size due to ethical constraints
- **Mitigation:** Use data augmentation techniques, synthetic data for testing

**Limitation 3:** LLM-based methods are non-deterministic
- **Mitigation:** Use low temperature (0.1), run multiple trials, report variance

**Limitation 4:** Temporal evaluation requires historical data
- **Mitigation:** Use existing trend datasets, simulate temporal analysis

### 11.2 Evaluation Limitations

**Limitation 1:** Human evaluation is expensive and small-scale
- **Mitigation:** Focus human evaluation on critical comparisons (e.g., topic labels)

**Limitation 2:** Cannot test on real-time trending data ethically
- **Mitigation:** Use historical datasets with known trends

### 11.3 System Limitations

**Limitation 1:** Trend prediction is inherently uncertain
- **Mitigation:** Report confidence intervals, evaluate directional accuracy

**Limitation 2:** Computational cost of embeddings
- **Mitigation:** Use efficient models (MiniLM), batch processing

---

## 12. ETHICAL CONSIDERATIONS

### 12.1 Data Collection Ethics

✅ Only use publicly accessible data or provided datasets  
✅ Respect platform Terms of Service  
✅ No authentication bypass or rate limit evasion  
✅ Anonymize author identifiers (hash or remove)  

### 12.2 Privacy Protection

✅ Do not store personally identifiable information unnecessarily  
✅ Use hashed identifiers for authors where needed  
✅ Do not re-publish collected data without consent  

### 12.3 Bias and Fairness

⚠️ Acknowledge that trend detection may reflect platform biases  
⚠️ Sentiment/emotion models may have cultural biases  
⚠️ Topic labels generated by LLMs may reflect training data biases  

**Mitigation:** Acknowledge limitations in paper, test on diverse data where possible

### 12.4 Dual Use Concerns

⚠️ Trend detection could be misused for manipulation  
⚠️ Sentiment analysis could be misused for surveillance  

**Mitigation:** Focus on research applications, emphasize ethical use in documentation

---

## 13. VALIDATION STRATEGY

### 13.1 Internal Validation

- ✅ Unit tests for all modules
- ✅ Integration tests for end-to-end pipeline
- ✅ Sanity checks (e.g., trend scores in [0, 1])
- ✅ Manual inspection of sample outputs

### 13.2 External Validation

- ✅ Compare with external trend databases (Google Trends, if available)
- ✅ User study with domain experts (if feasible)
- ✅ Cross-validation on multiple datasets

---

## 14. TIMELINE

**Week 1-2:** Data collection, schema definition, ingestion layer  
**Week 3-4:** Preprocessing pipeline, NLP modules  
**Week 5-6:** Topic discovery, clustering, labeling  
**Week 7:** Trend detection, explainability  
**Week 8:** Baseline implementations  
**Week 9-10:** Experiments, ablation studies  
**Week 11:** Statistical analysis, visualization  
**Week 12:** Paper writing, dashboard polish  

---

*Methodology Version: 1.0*  
*Last Updated: 2026-08-27*  
*Status: Draft - Ready for Implementation*
