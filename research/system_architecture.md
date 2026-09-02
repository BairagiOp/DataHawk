# System Architecture

**Project:** An Adaptive LLM-Assisted Framework for Social Media Data Parsing, Topic Discovery, Trend Detection, and Emerging Trend Prediction

**Version:** 1.0  
**Date:** 2026-08-27

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                        │
│  Streamlit Research Dashboard (app/dashboard.py)                │
│  - Data Collection Status                                        │
│  - NLP Analysis Results                                          │
│  - Topic Explorer                                                │
│  - Trend Detection & Ranking                                     │
│  - Emerging Trends Visualization                                 │
│  - Baseline Comparison                                           │
│  - Ablation Study Results                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  Pipeline Orchestration & Workflow Management                    │
│  - Data loading and validation                                   │
│  - Experiment execution                                          │
│  - Result aggregation                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Ingestion   │  │  Processing  │  │   Analysis   │          │
│  │   Service    │→ │   Service    │→ │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                  ↓                  ↓                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Topic     │  │    Trend     │  │  Evaluation  │          │
│  │   Service    │→ │   Service    │→ │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Raw Data    │  │  Processed   │  │   Results    │          │
│  │   Storage    │  │    Data      │  │   Storage    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  dataset/raw/       dataset/processed/ results/                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Gemini    │  │  Embeddings  │  │   Optional   │          │
│  │     API      │  │    Models    │  │   Services   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Modularity:** Each component is independently testable and replaceable
2. **Extensibility:** Easy to add new data sources, NLP models, or trend algorithms
3. **Reproducibility:** All experiments are logged with full parameters
4. **Efficiency:** Balance accuracy with computational cost
5. **Traceability:** Every result traces back to source data
6. **Configuration-Driven:** Externalized configuration via `.env` and YAML files

---

## 2. DETAILED COMPONENT ARCHITECTURE

### 2.1 Data Ingestion Layer (`ingestion/`)

**Purpose:** Load data from multiple sources into unified schema

```
ingestion/
├── __init__.py
├── schema.py              # Social media data schema definition
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py   # Abstract base class
│   ├── csv_adapter.py    # CSV file ingestion
│   ├── json_adapter.py   # JSON file ingestion
│   ├── web_adapter.py    # Web scraping (reuses scrapers/)
│   └── api_adapter.py    # API connectors
├── loader.py             # Unified data loader
└── validator.py          # Data validation
```

**Key Classes:**

```python
@dataclass
class SocialMediaPost:
    """Unified social media post schema"""
    post_id: str
    platform: str
    timestamp: datetime
    text: str
    author_id_hash: Optional[str] = None
    language: Optional[str] = None
    url: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    media_type: str = "text"
    engagement: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Derived fields (populated by pipeline)
    clean_text: Optional[str] = None
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    entities: List[Dict] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topic_id: Optional[int] = None
    trend_score: Optional[float] = None
```

**Data Flow:**

```
CSV/JSON/Web → Adapter → Validation → SocialMediaPost → Storage
```

**Validation Rules:**
- Required: post_id, platform, timestamp, text
- text length > 3 characters
- timestamp is valid datetime
- engagement values are non-negative integers

---

### 2.2 Preprocessing Layer (`preprocessing/`)

**Purpose:** Clean, normalize, and filter social media text

```
preprocessing/
├── __init__.py
├── cleaner.py           # Social media text cleaning
├── language.py          # Language detection
├── deduplication.py     # Duplicate detection
├── spam_filter.py       # Noise/spam filtering
└── normalizer.py        # Text normalization utilities
```

**Pipeline:**

```python
class PreprocessingPipeline:
    def __init__(self):
        self.cleaner = SocialMediaCleaner()
        self.language_detector = LanguageDetector()
        self.dedup = DuplicateDetector()
        self.spam_filter = SpamFilter()
    
    def process(self, posts: List[SocialMediaPost]) -> List[SocialMediaPost]:
        # 1. Clean text
        posts = [self.cleaner.clean(p) for p in posts]
        
        # 2. Detect language
        posts = [self.language_detector.detect(p) for p in posts]
        
        # 3. Remove duplicates
        posts = self.dedup.filter_duplicates(posts)
        
        # 4. Filter spam
        posts = self.spam_filter.filter(posts)
        
        return posts
```

**Key Operations:**

1. **Text Cleaning:**
   - Extract hashtags, mentions, URLs
   - Normalize Unicode
   - Handle emojis (convert to text or remove)
   - Normalize repeated characters ("sooo" → "so")
   - Remove HTML entities

2. **Language Detection:**
   - Use `langdetect` library
   - Support English, Hindi, Hinglish detection
   - Handle detection failures gracefully

3. **Duplicate Detection:**
   - Exact: Hash normalized text
   - Near-duplicate: Embedding cosine similarity > 0.95
   - Efficient: MinHash LSH for large datasets

4. **Spam Filtering:**
   - Heuristic rules (excessive URLs, repetition)
   - Configurable threshold

---

### 2.3 NLP Analysis Layer (`nlp/`)

**Purpose:** Extract semantic information from text

```
nlp/
├── __init__.py
├── sentiment.py         # Sentiment analysis (baseline + LLM)
├── emotion.py           # Emotion classification
├── ner.py               # Named Entity Recognition
├── embeddings.py        # Semantic embeddings
├── keywords.py          # Hashtag/keyword extraction
└── llm_parser.py        # LLM-based semantic parsing
```

**Architecture:**

```python
class NLPPipeline:
    def __init__(self, llm_client: GeminiClient):
        self.sentiment_analyzer = SentimentAnalyzer(llm_client)
        self.emotion_classifier = EmotionClassifier(llm_client)
        self.ner = NamedEntityRecognizer()
        self.embedding_model = EmbeddingModel()
        self.keyword_extractor = KeywordExtractor()
        self.llm_parser = LLMParser(llm_client)
    
    def analyze(self, post: SocialMediaPost) -> SocialMediaPost:
        # Parallel execution where possible
        post.sentiment = self.sentiment_analyzer.analyze(post.clean_text)
        post.emotion = self.emotion_classifier.classify(post.clean_text)
        post.entities = self.ner.extract(post.clean_text)
        post.keywords = self.keyword_extractor.extract(post.clean_text)
        
        # Optional: LLM semantic parsing
        if config.use_llm_parsing:
            semantic_info = self.llm_parser.parse(post)
            post.metadata['semantic_info'] = semantic_info
        
        return post
```

**Components:**

1. **Sentiment Analysis:**
   - Baseline: VADER (rule-based, social media optimized)
   - Proposed: LLM-based classification
   - Output: positive | negative | neutral | mixed

2. **Emotion Analysis:**
   - LLM-based classification
   - Output: joy | anger | sadness | fear | surprise | disgust | neutral

3. **Named Entity Recognition:**
   - spaCy NER (en_core_web_sm or en_core_web_lg)
   - Entity types: PERSON, ORG, GPE, PRODUCT, EVENT

4. **Keyword Extraction:**
   - Methods: TF-IDF, KeyBERT, YAKE
   - Co-occurrence analysis

5. **Embeddings:**
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - 384-dimensional semantic embeddings
   - Cached for efficiency

---

### 2.4 Topic Discovery Layer (`topics/`)

**Purpose:** Discover and label semantic topics

```
topics/
├── __init__.py
├── clustering.py        # Topic clustering (K-Means, HDBSCAN)
├── labeling.py          # LLM-based topic labeling
├── coherence.py         # Topic coherence evaluation
└── baselines.py         # Baseline topic models (TF-IDF, LDA)
```

**Architecture:**

```python
class TopicDiscoveryPipeline:
    def __init__(self, method='hdbscan'):
        self.embedding_model = EmbeddingModel()
        self.clusterer = self._get_clusterer(method)
        self.labeler = TopicLabeler(llm_client)
        self.coherence_evaluator = CoherenceEvaluator()
    
    def discover_topics(self, posts: List[SocialMediaPost]) -> TopicResult:
        # 1. Generate embeddings
        embeddings = self.embedding_model.encode([p.clean_text for p in posts])
        
        # 2. Cluster
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # 3. Label topics
        topics = self.labeler.label_clusters(posts, cluster_labels)
        
        # 4. Evaluate coherence
        coherence = self.coherence_evaluator.evaluate(topics)
        
        return TopicResult(topics, coherence, cluster_labels)
```

**Clustering Methods:**

1. **K-Means:**
   - Fast, requires k specification
   - Good for uniform cluster sizes

2. **HDBSCAN:**
   - Hierarchical density-based
   - Automatic cluster count
   - Handles noise

**Topic Labeling:**

```python
class TopicLabeler:
    def label_cluster(self, posts: List[str], max_examples=10):
        sample = posts[:max_examples]
        prompt = f"""
        These posts belong to one topic cluster:
        
        {sample}
        
        Generate a concise topic label (max 5 words):
        """
        label = self.llm.generate(prompt)
        return label
```

---

### 2.5 Trend Detection Layer (`trends/`)

**Purpose:** Detect and explain emerging trends

```
trends/
├── __init__.py
├── trend_score.py       # Composite trend scoring
├── emerging.py          # Emerging trend classification
├── temporal.py          # Temporal aggregation & analysis
├── explanation.py       # Explainable trend detection
└── forecasting.py       # Time-series forecasting
```

**Architecture:**

```python
class TrendDetectionPipeline:
    def __init__(self, config: TrendConfig):
        self.temporal_analyzer = TemporalAnalyzer()
        self.trend_scorer = TrendScorer(config.weights)
        self.classifier = EmergingTrendClassifier(config.thresholds)
        self.explainer = TrendExplainer()
    
    def detect_trends(
        self, 
        posts: List[SocialMediaPost],
        topics: Dict[int, Topic],
        time_window: str = '1D'
    ) -> List[Trend]:
        # 1. Temporal aggregation
        aggregated = self.temporal_analyzer.aggregate(posts, topics, time_window)
        
        # 2. Feature engineering
        features = self.temporal_analyzer.compute_features(aggregated)
        
        # 3. Trend scoring
        scores = self.trend_scorer.score(features)
        
        # 4. Classification
        trends = self.classifier.classify(scores, features)
        
        # 5. Explanation generation
        for trend in trends:
            trend.explanation = self.explainer.explain(trend)
        
        return sorted(trends, key=lambda t: t.score, reverse=True)
```

**Trend Score Formula:**

```python
class TrendScorer:
    def __init__(self, weights: Dict[str, float]):
        self.alpha = weights.get('volume', 0.2)
        self.beta = weights.get('growth', 0.4)
        self.gamma = weights.get('engagement', 0.3)
        self.delta = weights.get('novelty', 0.1)
    
    def score(self, features: TrendFeatures) -> float:
        V = self.normalize(features.volume)
        G = self.normalize(features.growth_rate)
        E = self.normalize(features.engagement)
        N = features.novelty_score  # Already normalized
        
        score = self.alpha * V + self.beta * G + self.gamma * E + self.delta * N
        return score
    
    def normalize(self, value: float) -> float:
        """Min-max normalization to [0, 1]"""
        return (value - self.min_val) / (self.max_val - self.min_val)
```

**Emerging Trend Classification:**

```python
class EmergingTrendClassifier:
    def classify(self, score: float, features: TrendFeatures) -> TrendCategory:
        if (features.baseline_volume < self.low_threshold and 
            features.growth_rate > self.growth_threshold):
            return TrendCategory.EMERGING
        
        elif features.growth_rate > self.viral_threshold:
            return TrendCategory.VIRAL
        
        elif features.growth_rate > 0:
            return TrendCategory.RISING
        
        elif abs(features.growth_rate) < self.stability_threshold:
            return TrendCategory.STABLE
        
        else:
            return TrendCategory.DECLINING
```

---

### 2.6 Evaluation Framework (`evaluation/`)

**Purpose:** Measure system performance and compare baselines

```
evaluation/
├── __init__.py
├── metrics.py           # ✅ Already excellent (reuse)
├── baselines.py         # ✅ Already implemented (extend)
├── benchmark.py         # Benchmarking infrastructure
├── experiments.py       # Experiment logging
├── ablation.py          # NEW: Ablation study framework
└── statistical_tests.py # NEW: Statistical comparison
```

**Extension:**

```python
# Add to metrics.py

@dataclass
class TopicMetrics:
    """Topic discovery quality metrics"""
    silhouette_score: float = 0.0
    coherence_score: float = 0.0
    cluster_purity: float = 0.0
    num_clusters: int = 0
    noise_ratio: float = 0.0

@dataclass
class TrendMetrics:
    """Trend detection quality metrics"""
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # k=10,20,50
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    early_detection_time_hours: float = 0.0
    false_positive_rate: float = 0.0
```

**Ablation Framework:**

```python
class AblationStudy:
    def __init__(self, dataset, ground_truth):
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.results = []
    
    def run_configuration(self, config: AblationConfig) -> Dict:
        """Run one ablation configuration"""
        pipeline = TrendDetectionPipeline(config)
        predictions = pipeline.detect_trends(self.dataset)
        metrics = evaluate_predictions(predictions, self.ground_truth)
        
        result = {
            'config': config.to_dict(),
            'metrics': metrics.to_dict()
        }
        self.results.append(result)
        return result
    
    def run_all_configurations(self):
        """Run full ablation study"""
        configs = self.generate_ablation_configs()
        for config in configs:
            self.run_configuration(config)
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze which components contribute most"""
        df = pd.DataFrame(self.results)
        # Statistical analysis
        # Component importance ranking
        # Visualization
```

---

### 2.7 Configuration Layer (`config/`)

**Purpose:** Centralized configuration management

**Extend existing `config/settings.py`:**

```python
@dataclass
class Settings:
    # ... existing settings ...
    
    # ── NLP Settings ────────────────────────────────────────────
    sentiment_method: str = "llm"  # "vader" or "llm"
    emotion_enabled: bool = True
    ner_model: str = "en_core_web_sm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    
    # ── Topic Discovery ─────────────────────────────────────────
    clustering_method: str = "hdbscan"  # "kmeans" or "hdbscan"
    min_cluster_size: int = 50
    max_clusters: int = 50
    use_dimensionality_reduction: bool = True
    umap_n_components: int = 10
    
    # ── Trend Detection ─────────────────────────────────────────
    trend_time_window: str = "1D"  # pandas frequency string
    trend_weight_volume: float = 0.2
    trend_weight_growth: float = 0.4
    trend_weight_engagement: float = 0.3
    trend_weight_novelty: float = 0.1
    emerging_threshold_low: int = 10
    emerging_threshold_growth: float = 0.5
    viral_threshold: float = 2.0
    
    # ── Preprocessing ───────────────────────────────────────────
    enable_deduplication: bool = True
    enable_spam_filter: bool = True
    duplicate_similarity_threshold: float = 0.95
    spam_score_threshold: float = 0.6
    
    # ── Experiment Settings ─────────────────────────────────────
    random_seed: int = 42
    enable_experiment_logging: bool = True
    experiment_output_dir: str = "results/experiments"
```

---

### 2.8 Research Dashboard (`app/dashboard.py`)

**Purpose:** Interactive research visualization and experiment control

**Dashboard Pages:**

1. **Home / Overview**
   - Dataset statistics
   - Pipeline status
   - Recent experiment results

2. **Data Collection**
   - Upload CSV/JSON
   - View raw data
   - Data quality report

3. **NLP Analysis**
   - Sentiment distribution
   - Emotion distribution
   - Entity frequency
   - Keyword analysis

4. **Topic Explorer**
   - Topic clusters visualization (2D/3D scatter)
   - Topic labels and representative posts
   - Topic coherence scores
   - Cluster size distribution

5. **Trend Detection**
   - Trend ranking table
   - Trend scores over time
   - Growth curves
   - Explainable trend cards

6. **Emerging Trends**
   - Emerging trend ranking
   - Early warning dashboard
   - Trend category distribution

7. **Forecasting**
   - Predicted vs. actual charts
   - Forecast accuracy metrics
   - Model comparison

8. **Baseline Comparison**
   - Side-by-side metric comparison
   - Precision/recall curves
   - Statistical test results

9. **Ablation Study**
   - Component contribution analysis
   - Configuration comparison table
   - Interactive ablation visualization

10. **Experiment Control**
    - Configure and run experiments
    - View experiment logs
    - Export results

**Implementation Pattern:**

```python
import streamlit as st

def render_trend_detection_page():
    st.title("🔥 Trend Detection")
    
    # Load data
    trends = load_trend_results()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.multiselect(
            "Category",
            ["EMERGING", "VIRAL", "RISING", "STABLE", "DECLINING"]
        )
    with col2:
        min_score = st.slider("Min Trend Score", 0.0, 1.0, 0.5)
    
    # Filter trends
    filtered = filter_trends(trends, category_filter, min_score)
    
    # Visualization
    st.subheader("📊 Trend Scores Over Time")
    fig = plot_trend_scores_timeline(filtered)
    st.plotly_chart(fig)
    
    # Trend table
    st.subheader("📋 Detected Trends")
    display_trend_table(filtered)
    
    # Explainable cards
    st.subheader("🔍 Trend Explanations")
    for trend in filtered[:10]:
        with st.expander(f"🔥 {trend.topic_label} (Score: {trend.score:.3f})"):
            st.markdown(trend.explanation)
            st.markdown("**Representative Posts:**")
            for post in trend.representative_posts[:3]:
                st.markdown(f"> {post.text}")
```

---

## 3. DATA FLOW ARCHITECTURE

### 3.1 End-to-End Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: CSV/JSON/Web                                          │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ INGESTION: Adapter → Validation → SocialMediaPost           │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ PREPROCESSING:                                               │
│  • Text Cleaning                                             │
│  • Language Detection                                        │
│  • Duplicate Removal                                         │
│  • Spam Filtering                                            │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ NLP ANALYSIS:                                                │
│  • Sentiment (VADER or LLM)                                  │
│  • Emotion (LLM)                                             │
│  • NER (spaCy)                                               │
│  • Keywords (TF-IDF/KeyBERT)                                 │
│  • Embeddings (sentence-transformers)                        │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ TOPIC DISCOVERY:                                             │
│  • Clustering (HDBSCAN/K-Means)                              │
│  • Topic Labeling (LLM)                                      │
│  • Coherence Evaluation                                      │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ TREND DETECTION:                                             │
│  • Temporal Aggregation                                      │
│  • Feature Engineering                                       │
│  • Trend Scoring (composite formula)                         │
│  • Emerging Trend Classification                             │
│  • Explanation Generation                                    │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ EVALUATION:                                                  │
│  • Metrics Calculation                                       │
│  • Baseline Comparison                                       │
│  • Statistical Tests                                         │
│  • Ablation Analysis                                         │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: Dashboard, Reports, Research Results                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. DEPLOYMENT ARCHITECTURE

### 4.1 Local Development

```
┌─────────────────────────────────────────────────────────────┐
│ Developer Machine                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Streamlit App (localhost:8501)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Python Backend (local modules)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Local File Storage (dataset/, results/)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                        ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│ External APIs                                                │
│  • Google Gemini API                                         │
│  • Sentence Transformers (HuggingFace)                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 File System Layout

```
DataHawk/
├── dataset/
│   ├── raw/                    # Original uploaded data
│   ├── processed/              # After preprocessing
│   ├── embeddings/             # Cached embeddings
│   └── templates/              # Dataset templates
│
├── results/
│   ├── experiments/            # Experiment logs
│   ├── baselines/              # Baseline results
│   ├── ablation/               # Ablation study results
│   ├── figures/                # Generated visualizations
│   └── final_metrics.csv       # Summary metrics
│
├── experiments/
│   └── experiment_log.db       # SQLite database
│
└── .cache/
    ├── embeddings_cache/       # Embedding cache
    └── llm_cache/              # LLM response cache (optional)
```

---

## 5. SCALABILITY CONSIDERATIONS

### 5.1 Current Scale (B.Tech Project)

- **Dataset Size:** 5,000 - 50,000 posts
- **Topics:** 10-50 topics
- **Time Period:** 7-30 days
- **Processing Time:** Minutes to hours (acceptable)

### 5.2 Optimization Strategies

1. **Embedding Caching:**
   ```python
   @lru_cache(maxsize=10000)
   def get_embedding(text_hash):
       return model.encode(text)
   ```

2. **Batch Processing:**
   ```python
   # Process in batches of 32-64
   for batch in chunk_list(posts, batch_size=32):
       embeddings = model.encode(batch)
   ```

3. **Parallel Processing:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_post, posts)
   ```

4. **Incremental Processing:**
   - Process only new posts
   - Update trend scores incrementally
   - Cache intermediate results

---

## 6. ERROR HANDLING & RESILIENCE

### 6.1 Error Handling Strategy

```python
class DataHawkException(Exception):
    """Base exception for DataHawk"""
    pass

class IngestionError(DataHawkException):
    """Data ingestion failed"""
    pass

class PreprocessingError(DataHawkException):
    """Preprocessing failed"""
    pass

class AnalysisError(DataHawkException):
    """NLP analysis failed"""
    pass

# Usage
try:
    posts = load_data(file_path)
except IngestionError as e:
    logger.error(f"Ingestion failed: {e}")
    # Graceful fallback
```

### 6.2 Resilience Patterns

1. **Retry Logic:** Already implemented in `core/llm_client.py`
2. **Graceful Degradation:** Continue pipeline even if one component fails
3. **Partial Results:** Return partial results rather than failing completely
4. **Logging:** Comprehensive logging for debugging

---

## 7. TESTING ARCHITECTURE

### 7.1 Test Structure

```
tests/
├── unit/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_nlp.py
│   ├── test_topics.py
│   ├── test_trends.py
│   └── test_evaluation.py
│
├── integration/
│   ├── test_pipeline.py
│   └── test_end_to_end.py
│
├── fixtures/
│   ├── sample_posts.json
│   ├── sample_embeddings.npy
│   └── ground_truth.json
│
└── conftest.py
```

### 7.2 Test Coverage Goals

- Unit tests: >80% coverage
- Integration tests: Critical paths
- End-to-end test: Full pipeline with synthetic data

---

## 8. SECURITY & PRIVACY

### 8.1 Data Privacy

- **Anonymization:** Hash author IDs if needed
- **No PII Storage:** Avoid storing unnecessary personal information
- **Data Retention:** Clear policy on how long data is kept

### 8.2 API Security

- **API Keys:** Stored in `.env`, never committed
- **Rate Limiting:** Respect API rate limits
- **Error Handling:** Don't expose sensitive info in error messages

---

## 9. DOCUMENTATION ARCHITECTURE

### 9.1 Code Documentation

- **Docstrings:** All public functions and classes
- **Type Hints:** Full type annotations
- **Inline Comments:** For complex logic

### 9.2 Research Documentation

```
research/
├── problem_statement.md
├── research_questions.md
├── hypotheses.md
├── methodology.md
├── system_architecture.md          # This document
├── experiment_design.md
├── baseline_methods.md
├── evaluation_metrics.md
├── ablation_study.md
├── limitations.md
├── ethical_considerations.md
└── paper_outline.md
```

---

## 10. MIGRATION FROM EXISTING DATAHAWK

### 10.1 Reusable Components (60%)

✅ **core/**: LLM client, extractor, schema, validator, confidence  
✅ **evaluation/**: Metrics, baselines, benchmark, experiments  
✅ **config/**: Settings management  
✅ **scrapers/**: Base classes and implementations  
✅ **processing/**: Chunker (extend others)  

### 10.2 New Components (40%)

🆕 **ingestion/**: Social media data ingestion  
🆕 **preprocessing/**: Social media text cleaning  
🆕 **nlp/**: Sentiment, emotion, NER, embeddings  
🆕 **topics/**: Clustering, labeling, coherence  
🆕 **trends/**: Trend scoring, detection, explanation  
🆕 **research/**: Documentation  
🆕 **scripts/**: Reproducibility scripts  
🆕 **app/dashboard.py**: Research-oriented UI  

### 10.3 Refactor Needed

⚠️ **main.py**: Rewrite to use core/ modules  
⚠️ **scrape.py**: Integrate into scrapers/ architecture  
⚠️ **parse.py**: Use core/extractor.py instead  

---

*Architecture Version: 1.0*  
*Last Updated: 2026-08-27*  
*Status: Ready for Implementation*
