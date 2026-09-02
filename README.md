# DataHawk: Social Media Intelligence Research Platform

> **AI-Powered Social Media Analysis for Academic Research**

A comprehensive research platform for social media intelligence, designed for academic publication and B.Tech final year projects. Features advanced NLP, semantic topic discovery, composite trend detection, and explainable forecasting.

## 🎯 Research Contributions

This platform addresses **7 research questions** with rigorous methodology:

1. **RQ1**: Does semantic clustering identify more coherent topics than frequency-based approaches?
2. **RQ2**: Do LLM-based sentiment models outperform lexicon-based baselines?
3. **RQ3**: Does composite scoring (volume + growth + engagement + novelty) detect trends more accurately?
4. **RQ4**: Can we classify emerging trends into categories (EMERGING/VIRAL/RISING/STABLE/DECLINING)?
5. **RQ5**: Are composite trend predictions explainable to human analysts?
6. **RQ6**: Can ensemble methods predict trend trajectories more accurately than statistical baselines?
7. **RQ7**: Which features contribute most to trend detection accuracy?

## 🏗️ Architecture

```
DataHawk/
├── ingestion/          # Data loading (CSV, JSON, multi-source)
├── preprocessing/      # Cleaning, deduplication, spam filtering
├── nlp/               # Sentiment, emotion, NER, embeddings
├── topics/            # Semantic clustering, topic labeling
├── trends/            # Composite scoring, classification, explanation
├── forecasting/       # Time-series prediction (baselines + ensemble)
├── evaluation/        # Metrics, ablation studies
├── dataset/           # Sample data generation
├── research/          # Experiment scripts
└── app/              # Streamlit research dashboard
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DataHawk

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced forecasting (XGBoost)
pip install xgboost scikit-learn

# Set up API keys (optional, for LLM features)
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Generate Sample Data

```bash
# Create synthetic datasets for testing
python dataset/sample_data.py
```

This generates:
- `dataset/test_data_small.csv` (100 posts)
- `dataset/test_data_medium.csv` (500 posts)
- `dataset/test_data_large.csv` (1000 posts)

### Run Research Experiment

```bash
# Run complete experiment pipeline
python research/run_experiment.py --data dataset/test_data_small.csv
```

This executes all 7 research questions and generates results.

### Launch Research Dashboard

```bash
# Start interactive web interface
streamlit run app/main.py
```

Navigate to http://localhost:8501

## 📊 Core Modules

### 1. Data Ingestion

```python
from ingestion.loader import DataLoader

loader = DataLoader()
posts = loader.load('data.csv')  # Auto-detects format
```

Supports: CSV, JSON, JSON Lines, batch loading, directory scanning

### 2. Preprocessing

```python
from preprocessing import SocialMediaCleaner, DuplicateDetector, SpamFilter

cleaner = SocialMediaCleaner()
cleaned = cleaner.clean(text)

dedup = DuplicateDetector()
duplicates = dedup.find_near_duplicates(texts)

spam_filter = SpamFilter()
spam_score = spam_filter.calculate_spam_score(text)
```

### 3. NLP Analysis

```python
from nlp import SentimentAnalyzer, EmotionClassifier, KeywordExtractor

# Sentiment (VADER baseline or LLM)
analyzer = SentimentAnalyzer(method='vader')
sentiment = analyzer.analyze(text)

# Emotion classification
emotion_clf = EmotionClassifier()
emotion = emotion_clf.classify(text)

# Keyword extraction
keywords = KeywordExtractor().extract_keywords(text)
```

### 4. Topic Discovery

```python
from nlp import EmbeddingModel
from topics import TopicClusterer, TopicLabeler

# Generate embeddings
embeddings = EmbeddingModel().encode(texts)

# Semantic clustering (proposed)
clusterer = TopicClusterer(method='hdbscan')
result = clusterer.fit_predict(embeddings)

# Topic labeling with LLM
labeler = TopicLabeler()
labels = labeler.label_cluster(cluster_texts, cluster_id)
```

### 5. Trend Detection

```python
from trends import TrendScorer, EmergingTrendClassifier, TrendExplainer

# Composite trend scoring
scorer = TrendScorer(alpha=0.2, beta=0.4, gamma=0.3, delta=0.1)
score = scorer.compute_trend_score(volume, timestamps, engagement, is_novel)

# Classify emerging trends
classifier = EmergingTrendClassifier()
category = classifier.classify_trend(counts, timestamps)

# Explain predictions
explainer = TrendExplainer()
explanation = explainer.explain_trend(trend_result)
```

### 6. Forecasting

```python
from forecasting import NaiveForecaster, MovingAverageForecaster, TrendForecaster

# Baseline: Naive
naive = NaiveForecaster().fit(time_series)
forecast = naive.predict(steps=7)

# Proposed: XGBoost
forecaster = TrendForecaster(method='xgboost')
forecaster.fit(time_series)
result = forecaster.predict(steps=7)
```

### 7. Evaluation

```python
from evaluation import MetricsCalculator, AblationStudy

# Compute metrics
calculator = MetricsCalculator()
metrics = calculator.compute_trend_detection_metrics(y_true, y_pred)

# Run ablation study
study = AblationStudy(components, evaluation_fn, metric_name='F1')
results = study.run_full_ablation()
study.print_summary()
```

## 🔬 Research Methodology

### Baseline Comparisons

Every proposed method has a fair baseline:

| Component | Baseline | Proposed |
|-----------|----------|----------|
| Sentiment | VADER (lexicon) | Gemini LLM |
| Topics | TF-IDF + K-Means | Embeddings + HDBSCAN |
| Trends | Frequency counting | Composite scoring (α·V + β·G + γ·E + δ·N) |
| Forecasting | Naive, Moving Average | XGBoost ensemble |

### Evaluation Metrics

- **Topics**: Coherence (NPMI), Diversity, Silhouette Score
- **Sentiment**: Accuracy, Precision, Recall, F1
- **Trends**: Precision, Recall, F1, Early Detection Rate
- **Forecasting**: MAE, RMSE, MAPE, Direction Accuracy
- **Ablation**: Feature importance via leave-one-out

### Reproducibility

All experiments use:
- Fixed random seeds
- Documented hyperparameters
- Version-controlled code
- Synthetic data generators for testing

## 📈 Dataset Guidelines

### Ethical Data Collection

**IMPORTANT**: This platform is designed for ethical research only.

✅ **Allowed**:
- Publicly accessible content
- User-provided CSV/JSON datasets
- Research benchmark datasets
- Content with explicit consent

❌ **Not Allowed**:
- Bypassing login/authentication
- Violating Terms of Service
- Scraping private/protected content
- Collecting personal identifiable information

### Data Format

CSV example:
```csv
post_id,text,timestamp,likes,comments,shares
post1,"AI is transforming healthcare",2026-08-01,50,10,5
```

JSON example:
```json
[
  {
    "post_id": "post1",
    "text": "AI is transforming healthcare",
    "timestamp": "2026-08-01T10:00:00Z",
    "likes": 50,
    "comments": 10,
    "shares": 5
  }
]
```

## 🎓 For Researchers

### Using for IEEE Paper

1. Run experiments with your dataset:
   ```bash
   python research/run_experiment.py --data your_data.csv
   ```

2. Review generated results in `research/experiment_results.json`

3. Use the provided research paper template: `RESEARCH_PAPER.md`

4. Cite this work appropriately

### Using for B.Tech Project

1. **Phase 1 (Week 1-2)**: Set up environment, generate sample data
2. **Phase 2 (Week 3-4)**: Run experiments, understand baselines
3. **Phase 3 (Week 5-6)**: Collect real data (ethically), run full pipeline
4. **Phase 4 (Week 7-8)**: Analyze results, write report

### Key Constraints

- ⚠️ **NEVER fabricate results or accuracy scores**
- ⚠️ **Always use ethically collected data**
- ⚠️ **Document all limitations honestly**
- ⚠️ **Compare against fair baselines**

## 🛠️ Configuration

Edit `config/settings.py` to customize:

```python
# Trend scoring weights (for ablation)
ALPHA = 0.2  # Volume weight
BETA = 0.4   # Growth weight
GAMMA = 0.3  # Engagement weight
DELTA = 0.1  # Novelty weight

# Clustering parameters
MIN_CLUSTER_SIZE = 5
UMAP_DIMENSIONS = 5

# Forecasting horizon
FORECAST_STEPS = 7
```

## 📚 Documentation

- `RESEARCH_PAPER.md` - Complete IEEE-style paper
- `IMPLEMENTATION_STATUS.md` - Development progress
- `ARCHITECTURE_ANALYSIS.md` - System design
- `docs/` - Detailed module documentation

## 🧪 Testing

```bash
# Test individual modules (built-in __main__ blocks)
python preprocessing/cleaner.py
python nlp/sentiment.py
python topics/clustering.py
python trends/trend_score.py
python forecasting/baseline.py

# Full integration test
python research/run_experiment.py --data dataset/test_data_small.csv
```

## 🤝 Contributing

This is a research platform. Contributions should:
1. Maintain scientific rigor
2. Include baseline comparisons
3. Document methodology
4. Follow ethical guidelines

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built as a B.Tech final year project and IEEE paper submission.

Special thanks to:
- Research advisors
- Open-source NLP community
- Claude AI for development assistance

## 📧 Contact

For research collaboration or questions:
- Project Lead: [Your Name]
- Email: [Your Email]
- Institution: [Your University]

---

**Note**: This is a research platform. All results should be validated on real-world data before publication. Never fabricate experimental results.
