# Baseline Methods

## Overview

This document describes the baseline methods used to evaluate the proposed DataHawk approach. Baselines provide a reference point for claims about improvement.

---

## 1. Frequency-Based Trending (Baseline 1)

### Algorithm

The simplest possible trend detection: count occurrences of keywords or hashtags, rank by frequency.

**Steps:**
1. Extract all hashtags and keywords from posts
2. Count occurrences of each term
3. Rank by count (descending)
4. Label top N as "trending"

### Implementation

```python
def frequency_baseline(posts, top_k=10):
    """
    Simple frequency-based trending.
    
    Args:
        posts: List of dictionaries with 'text' field
        top_k: Number of top trends to return
        
    Returns:
        List of (keyword, count) tuples sorted by count
    """
    keyword_counts = {}
    for post in posts:
        keywords = extract_keywords(post['text'])
        for kw in keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
    
    sorted_keywords = sorted(
        keyword_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    return sorted_keywords[:top_k]
```

### Strengths

- ✅ Simple and fast
- ✅ Interpretable (count is direct)
- ✅ No tuning required

### Weaknesses

- ❌ Cannot distinguish popularity from emergence
  - A consistently popular topic (steady 50 posts/day) ranks the same as a rapidly emerging topic (1 → 50 posts)
- ❌ Vulnerable to duplicates and spam
  - 100 identical posts rank as 100× more "trending" than 100 unique posts on the same topic
- ❌ No temporal awareness
  - Cannot detect trends that peaked yesterday vs. rising today
- ❌ No engagement weighting
  - 1000 low-engagement posts rank above 100 high-engagement posts

### File

`evaluation/baselines.py::frequency_baseline()`

---

## 2. TF-IDF + K-Means Clustering (Baseline 2)

### Algorithm

Use classical text vectorization + unsupervised clustering to discover topics.

**Steps:**
1. Vectorize post texts using TF-IDF (Term Frequency-Inverse Document Frequency)
2. Cluster vectors with K-Means (K=5 fixed)
3. Extract top terms from each cluster
4. Rank clusters by size

### TF-IDF Rationale

TF-IDF measures how "important" a word is in a document relative to the corpus:
- High TF: word appears frequently in this post (relevant to post topic)
- High IDF: word appears rarely across corpus (not a stopword)
- High TF-IDF: word is specific and important

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def tfidf_kmeans_baseline(texts, n_topics=5):
    """
    TF-IDF vectorization + K-Means clustering.
    
    Args:
        texts: List of text documents
        n_topics: Number of clusters (K)
        
    Returns:
        {
            'labels': cluster assignments,
            'terms': top terms per cluster,
            'silhouette': cluster quality score
        }
    """
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    
    # Extract top terms per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for i in range(n_topics):
        top_terms[i] = [terms[ind] for ind in order_centroids[i][:5]]
    
    return {'labels': labels, 'terms': top_terms}
```

### Strengths

- ✅ Unsupervised (no manual labeling required)
- ✅ Well-established method (decades of research)
- ✅ Interpretable (top terms = topic label)
- ✅ Fast computation

### Weaknesses

- ❌ Fixed K=5 may not match true topic count
- ❌ TF-IDF ignores semantic meaning
  - "AI agents" and "autonomous coding agents" are treated as completely different topics
- ❌ Sensitive to preprocessing (stopword list affects results)
- ❌ Can produce incoherent clusters (unrelated terms grouped)
- ❌ No temporal or engagement signals
- ❌ Requires pre-specifying cluster count

### File

`topics/baselines.py::TfidfKMeansBaseline`

---

## 3. Latent Dirichlet Allocation (LDA) (Baseline 3)

### Algorithm

Probabilistic topic model: each document is a mixture of topics, each topic is a distribution over words.

**Steps:**
1. Vectorize texts (bag-of-words)
2. Fit LDA model with n_topics=5, max_iter=100
3. Extract most probable terms per topic
4. Rank topics by document prevalence

### LDA Rationale

LDA models document generation as:
1. Each document has a topic distribution (e.g., 60% AI, 40% Policy)
2. For each word, sample topic from distribution
3. From selected topic, sample word from topic's word distribution

Inference (fitting) reverses this: given documents and words, infer the most likely topic distributions.

### Implementation

```python
from sklearn.decomposition import LatentDirichletAllocation

def lda_baseline(texts, n_topics=5):
    """
    Latent Dirichlet Allocation topic modeling.
    
    Args:
        texts: List of text documents
        n_topics: Number of topics
        
    Returns:
        {
            'labels': most likely topic per document,
            'terms': top terms per topic,
            'coherence': topic coherence score
        }
    """
    # Bag-of-words vectorization
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    bow_matrix = vectorizer.fit_transform(texts)
    
    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=100,
        learning_method='online',
        random_state=42
    )
    doc_topic_dist = lda.fit_transform(bow_matrix)
    labels = doc_topic_dist.argmax(axis=1)
    
    # Extract top terms per topic
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_terms[topic_idx] = [
            terms[i] for i in topic.argsort()[-5:][::-1]
        ]
    
    return {'labels': labels, 'terms': top_terms}
```

### Strengths

- ✅ Probabilistic (interpretable topic distributions)
- ✅ Long history in academic literature
- ✅ Well-understood failure modes
- ✅ Produces interpretable topic terms

### Weaknesses

- ❌ Slow convergence (100+ iterations needed)
- ❌ Still doesn't capture semantic meaning (just statistics)
- ❌ Fixed K=5 assumption
- ❌ No temporal signals
- ❌ Can produce generic topics (e.g., "word1 word2 the a")
- ❌ Not robust to short texts (Twitter posts, social media)

### File

`topics/baselines.py::LDABaseline`

---

## 4. Simple Forecasting Baselines

### 4a. Naive Forecast

**Algorithm:** Predict tomorrow = today

$$\hat{y}_{t+1} = y_t$$

**Pros:** Simple, 0 computation
**Cons:** No trend awareness, predicts constant values

### 4b. Moving Average

**Algorithm:** Predict as average of last K days

$$\hat{y}_{t+1} = \frac{1}{K} \sum_{i=0}^{K-1} y_{t-i}$$

**Pros:** Captures trend direction
**Cons:** Lags real trends, no acceleration awareness

### 4c. Linear Regression

**Algorithm:** Fit line to historical data, extrapolate

$$y = mx + b$$
$$\hat{y}_{t+1} = m(t+1) + b$$

**Pros:** Handles consistent growth/decay
**Cons:** Breaks under curvature, no error bounds

### Implementation

```python
def naive_forecast(timeseries, steps=1):
    """Last value repeated."""
    return [timeseries[-1]] * steps

def ma_forecast(timeseries, window=3, steps=1):
    """Moving average."""
    return [np.mean(timeseries[-window:])] * steps

def linear_forecast(timeseries, steps=1):
    """Linear regression forecast."""
    X = np.arange(len(timeseries)).reshape(-1, 1)
    y = np.array(timeseries)
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(timeseries), len(timeseries) + steps).reshape(-1, 1)
    return model.predict(X_future).tolist()
```

### File

`forecasting/baseline.py`

---

## 5. Comparison Summary

| Baseline | Type | Pros | Cons |
|----------|------|------|------|
| **Frequency** | Simple | Fast, interpretable | No growth awareness, vulnerable to spam |
| **TF-IDF+KMeans** | Clustering | Established, unsupervised | Fixed K, no semantics, static |
| **LDA** | Probabilistic | Interpretable, probabilistic | Slow, short-text fragile, static |
| **Naive** | Forecast | Zero computation | No trend awareness |
| **MA** | Forecast | Captures trend | Lags, no acceleration |
| **Linear** | Forecast | Handles steady growth | Breaks on curves |

---

## 6. Why Baselines Matter

These baselines are NOT straw men:

1. **Frequency-based trending** is industry standard (Twitter trending, Reddit /r/all)
2. **TF-IDF+KMeans** is in academic textbooks and production systems
3. **LDA** is the reference topic model (decades of papers)
4. **Simple forecasting** is how many systems make predictions today

If the proposed DataHawk method cannot outperform these on fair evaluation, the research claim is invalid.

---

*All baselines are implemented in production code with proper error handling and reproducible seeding.*
