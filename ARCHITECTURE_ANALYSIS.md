# DataHawk Architecture Analysis
## Existing System Analysis (AI Web Scraper → Social Media Intelligence Platform)

**Analysis Date:** 2026-08-27  
**Analyst Role:** Senior AI/ML Researcher, NLP Researcher, Software Architect, Project Mentor

---

## 1. EXISTING DATAHAWK ARCHITECTURE

### 1.1 Current System Overview

**Project Type:** AI-Powered Web Scraper  
**Primary Purpose:** Extract structured data from arbitrary webpages using natural language descriptions

**Technology Stack:**
- **Frontend:** Streamlit (modern dark-themed UI)
- **Web Scraping:** Selenium WebDriver + BeautifulSoup4
- **AI/LLM:** Google Gemini 1.5 Flash / 2.0 Flash
- **Proxy:** BrightData SuperProxy (optional)
- **Language:** Python 3.8+

### 1.2 Current File Structure

```
DataHawk/
├── main.py                    # Streamlit web interface (1093 lines)
├── scrape.py                  # Web scraping logic (105 lines)
├── parse.py                   # AI parsing with Gemini (44 lines)
├── setup.py                   # Installation script
├── requirements.txt           # Basic dependencies (7 packages)
├── .env.example              # Configuration template
│
├── app/
│   ├── dashboard.py          # Research dashboard (already started)
│   └── components/           # UI components
│
├── core/                     # ⭐ SOPHISTICATED EXTRACTION ENGINE (EXCELLENT BASE)
│   ├── llm_client.py        # Unified Gemini client with rate limiting
│   ├── extractor.py         # Schema-guided LLM extraction
│   ├── schema_generator.py  # Extraction schema definition
│   ├── validator.py         # Data validation
│   ├── classifier.py        # Field classification
│   ├── correction.py        # Self-correction mechanism
│   ├── confidence.py        # Confidence scoring
│   └── strategy_selector.py # Adaptive strategy selection
│
├── scrapers/                # ⭐ PLUGGABLE SCRAPER ARCHITECTURE (REUSABLE)
│   ├── base_scraper.py     # Abstract scraper interface
│   ├── static_scraper.py   # requests-based scraper
│   ├── dynamic_scraper.py  # Selenium-based scraper
│   └── api_scraper.py      # API connector base
│
├── processing/             # Content processing utilities
│   ├── chunker.py         # Content chunking
│   ├── dom_cleaner.py     # HTML cleaning
│   └── relevance.py       # Relevance filtering
│
├── evaluation/            # ⭐ RESEARCH EVALUATION FRAMEWORK (EXCELLENT!)
│   ├── metrics.py        # Precision, recall, F1, system metrics
│   ├── baselines.py      # Baseline implementations
│   ├── benchmark.py      # Benchmarking infrastructure
│   └── experiments.py    # Experiment logging
│
├── config/
│   └── settings.py       # ⭐ CENTRALIZED CONFIG (EXCELLENT PATTERN)
│
├── dataset/              # Dataset storage (empty)
├── experiments/          # Experiment logs (empty)
├── results/              # Results storage (empty)
└── tests/                # Test suite (empty)
```

### 1.3 Current Capabilities Analysis

#### ✅ **STRENGTHS - What Can Be Reused**

1. **Sophisticated LLM Integration (core/)**
   - **llm_client.py**: Unified Gemini client with:
     - Rate limiting (15 RPM for free tier)
     - Automatic retry logic
     - Token counting and cost estimation
     - JSON parsing with markdown fence cleaning
     - Request counters and usage tracking
   - **extractor.py**: Schema-guided extraction with:
     - Field-level extraction status (FOUND/NOT_FOUND/UNCERTAIN)
     - Source snippet traceability
     - Multi-chunk extraction support
     - Structured JSON output enforcement
   - **schema_generator.py**: Dynamic schema generation
   - **validator.py**: Field validation (12,781 lines - comprehensive)
   - **correction.py**: Self-correction mechanism (14,817 lines)
   - **confidence.py**: Confidence scoring
   
   **VERDICT:** ⭐⭐⭐⭐⭐ Excellent foundation. This is research-grade already.

2. **Modular Scraper Architecture (scrapers/)**
   - Abstract base class pattern
   - Static (requests) vs Dynamic (Selenium) scrapers
   - API scraper base
   - Result dataclasses with metadata
   - Retry logic built-in
   
   **VERDICT:** ⭐⭐⭐⭐⭐ Perfect for social media data ingestion adapters.

3. **Evaluation Framework (evaluation/)**
   - **metrics.py**: Already implements:
     - Extraction metrics (precision, recall, F1, exact match)
     - System metrics (latency, tokens, cost)
     - Efficiency comparison metrics
     - Value matching with normalization
   - **baselines.py**: Three baseline implementations:
     - BeautifulSoup + rule-based
     - Selenium + rule-based
     - Raw LLM (no schema/validation)
   - **benchmark.py**: Benchmarking infrastructure
   - **experiments.py**: Experiment logging
   
   **VERDICT:** ⭐⭐⭐⭐⭐ This is exactly what we need for research evaluation!

4. **Centralized Configuration (config/settings.py)**
   - Environment-based configuration
   - Singleton pattern
   - All parameters externalized
   - BrightData proxy support
   - Rate limiting configuration
   
   **VERDICT:** ⭐⭐⭐⭐ Excellent pattern. Easy to extend.

5. **Content Processing (processing/)**
   - Chunking with token limits
   - DOM cleaning
   - Relevance filtering
   
   **VERDICT:** ⭐⭐⭐ Good base, needs extension for social media text.

#### ⚠️ **LIMITATIONS - What Must Be Added**

1. **No Social Media Data Model**
   - Current focus: arbitrary webpage extraction
   - No unified social media schema (post_id, platform, timestamp, engagement)
   - No concept of temporal data
   - No author/user handling

2. **No NLP Pipeline**
   - No sentiment analysis
   - No emotion analysis
   - No Named Entity Recognition
   - No language detection
   - No hashtag/mention extraction
   - No text normalization for social media

3. **No Topic Modeling**
   - No semantic embeddings
   - No clustering
   - No topic discovery
   - No topic coherence evaluation

4. **No Trend Detection**
   - No temporal analysis
   - No trend scoring
   - No emerging trend detection
   - No forecasting

5. **No Duplicate Detection**
   - No exact duplicate filtering
   - No near-duplicate detection
   - No spam/noise filtering

6. **No Research Documentation**
   - No research questions
   - No hypotheses
   - No methodology documentation
   - No paper outline

7. **UI Not Research-Oriented**
   - Current UI: data extraction tool
   - Not designed for trend analysis
   - No visualizations for temporal data
   - No baseline comparisons shown

#### 🔧 **TECHNICAL DEBT**

1. **Original Files (main.py, scrape.py, parse.py)**
   - Simple implementation, not using core/ modules
   - main.py is monolithic (1093 lines)
   - Hardcoded BrightData credentials in scrape.py
   - Should be refactored to use core/ architecture

2. **Empty Directories**
   - dataset/ has no structure
   - experiments/ has no logging system
   - tests/ has no test suite
   - results/ has no organization

3. **No Database**
   - Everything is in-memory
   - No persistent storage for experiments
   - No dataset management

### 1.4 Current Workflow

```
User enters URL
        ↓
URL validation & auto-fixing
        ↓
Selenium scraping (with BrightData proxy)
        ↓
CAPTCHA solving (BrightData)
        ↓
HTML extraction
        ↓
BeautifulSoup cleaning
        ↓
Content chunking (6000 chars)
        ↓
Gemini extraction (per chunk)
        ↓
Result merging
        ↓
Display (text/table/CSV/JSON)
        ↓
Download
```

**Analysis:** This is a **synchronous, user-driven, single-page extraction** workflow. It works well for one-off data extraction but is not designed for:
- Batch processing
- Dataset management
- Time-series analysis
- Research experiments

---

## 2. REQUIRED TRANSFORMATION

### 2.1 New System Objective

**From:** AI Web Scraper  
**To:** Social Media Intelligence Research Platform

**New Purpose:**
1. Collect social media data from multiple sources
2. Apply NLP/ML analysis (sentiment, emotion, entities, topics)
3. Detect emerging trends with explainability
4. Forecast future trend activity
5. Provide reproducible research experiments
6. Generate publishable research results

### 2.2 New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  CSV/JSON Upload  │  Web Scraping  │  API Connectors        │
│  (adapter pattern - reuse scrapers/)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED SOCIAL MEDIA DATA SCHEMA                │
│  post_id, platform, timestamp, text, engagement, etc.        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                     │
│  • Content Cleaning (URLs, mentions, emojis, HTML)          │
│  • Language Detection (English, Hindi, Hinglish, Other)      │
│  • Duplicate Detection (exact + near-duplicate)              │
│  • Spam/Noise Filtering (repetition, malformed content)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    NLP ANALYSIS LAYER                        │
│  • LLM-Assisted Parsing (reuse core/extractor.py)           │
│  • Sentiment Analysis (baseline + LLM)                       │
│  • Emotion Analysis (7 emotions)                             │
│  • Named Entity Recognition (PERSON, ORG, etc.)              │
│  • Hashtag/Keyword Extraction & Analysis                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     TOPIC DISCOVERY LAYER                    │
│  • Semantic Embeddings (sentence-transformers)               │
│  • Topic Clustering (K-Means, HDBSCAN)                       │
│  • LLM Topic Labeling (reuse core/llm_client.py)             │
│  • Topic Coherence Evaluation                                │
│  • Baseline: TF-IDF + clustering, LDA                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    TREND DETECTION LAYER                     │
│  • Temporal Analysis (volume, growth, acceleration)          │
│  • Composite Trend Score (volume + engagement + novelty)     │
│  • Emerging Trend Classification (EMERGING/RISING/etc.)      │
│  • Early Warning Score                                       │
│  • Explainable Trend Detection (show why)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     FORECASTING LAYER                        │
│  • Baseline: naive, moving average, linear regression        │
│  • Proposed: XGBoost / Random Forest                         │
│  • Evaluation: MAE, RMSE, directional accuracy               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                      │
│  • Metrics Calculation (reuse evaluation/metrics.py)         │
│  • Baseline Comparison (reuse evaluation/baselines.py)       │
│  • Ablation Studies                                          │
│  • Statistical Tests                                         │
│  • Experiment Logging (reuse evaluation/experiments.py)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH DASHBOARD                        │
│  • Data Collection Status                                    │
│  • NLP Analysis Results                                      │
│  • Topic Explorer                                            │
│  • Trend Detection                                           │
│  • Emerging Trends Ranking                                   │
│  • Forecasting Visualizations                                │
│  • Baseline Comparison Charts                                │
│  • Ablation Study Results                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Component Reuse Matrix

| Existing Component | Reuse? | Modification Required | New Purpose |
|-------------------|--------|----------------------|-------------|
| `core/llm_client.py` | ✅ YES | Minimal | LLM calls for sentiment, topic labeling, semantic parsing |
| `core/extractor.py` | ✅ YES | Adapt schema | Extract structured social media fields |
| `core/schema_generator.py` | ✅ YES | Extend | Generate social media extraction schemas |
| `core/validator.py` | ✅ YES | Extend | Validate social media data |
| `core/confidence.py` | ✅ YES | Reuse as-is | Confidence scoring for extractions |
| `scrapers/base_scraper.py` | ✅ YES | Extend | Base for data source adapters |
| `scrapers/static_scraper.py` | ✅ YES | Reuse | Scrape public web pages |
| `scrapers/api_scraper.py` | ✅ YES | Extend | API connectors for permitted sources |
| `evaluation/metrics.py` | ✅ YES | Extend | Add topic/trend metrics |
| `evaluation/baselines.py` | ✅ YES | Extend | Add frequency/TF-IDF baselines |
| `evaluation/benchmark.py` | ✅ YES | Adapt | Social media benchmarks |
| `evaluation/experiments.py` | ✅ YES | Extend | Log trend detection experiments |
| `config/settings.py` | ✅ YES | Extend | Add NLP/trend detection configs |
| `processing/chunker.py` | ✅ YES | Reuse | Text chunking |
| `processing/dom_cleaner.py` | ⚠️ PARTIAL | Heavy modification | Social media text cleaning |
| `main.py` | ❌ REFACTOR | Complete rewrite | Use core/ modules, research UI |
| `scrape.py` | ❌ REFACTOR | Merge into scrapers/ | Use modular architecture |
| `parse.py` | ❌ REFACTOR | Use core/extractor.py | Deprecated by core/ |

### 2.4 New Components Required

#### Must Implement (Core Research Components):

1. **ingestion/** - Data source adapters
   - `csv_adapter.py` - CSV file ingestion
   - `json_adapter.py` - JSON file ingestion
   - `web_adapter.py` - Web scraping adapter
   - `schema.py` - Social media data schema
   - `loader.py` - Unified data loader

2. **preprocessing/** - Social media preprocessing
   - `cleaner.py` - Social media text cleaning
   - `language.py` - Language detection
   - `deduplication.py` - Duplicate detection
   - `spam_filter.py` - Noise/spam detection

3. **nlp/** - NLP analysis pipeline
   - `sentiment.py` - Sentiment analysis (baseline + LLM)
   - `emotion.py` - Emotion classification
   - `ner.py` - Named Entity Recognition
   - `embeddings.py` - Semantic embeddings
   - `keywords.py` - Hashtag/keyword extraction
   - `llm_parser.py` - LLM-based semantic parsing

4. **topics/** - Topic discovery
   - `clustering.py` - Topic clustering (K-Means, HDBSCAN)
   - `labeling.py` - LLM-based topic labeling
   - `coherence.py` - Topic coherence evaluation
   - `baselines.py` - TF-IDF, LDA baselines

5. **trends/** - Trend detection & forecasting
   - `trend_score.py` - Composite trend scoring
   - `emerging.py` - Emerging trend classification
   - `temporal.py` - Temporal analysis
   - `explanation.py` - Explainable trend detection
   - `forecasting.py` - Time-series forecasting

6. **research/** - Research documentation
   - `research_questions.md`
   - `hypotheses.md`
   - `methodology.md`
   - `system_architecture.md`
   - `experiment_design.md`
   - `baseline_methods.md`
   - `evaluation_metrics.md`
   - `ablation_study.md`
   - `limitations.md`
   - `ethical_considerations.md`
   - `paper_outline.md`

7. **scripts/** - Reproducibility scripts
   - `prepare_dataset.py`
   - `run_baselines.py`
   - `run_experiments.py`
   - `run_ablation.py`
   - `evaluate.py`

8. **tests/** - Comprehensive test suite
   - `test_ingestion.py`
   - `test_preprocessing.py`
   - `test_nlp.py`
   - `test_topics.py`
   - `test_trends.py`
   - `test_evaluation.py`

---

## 3. MIGRATION STRATEGY

### Phase 1: Foundation (Keep Existing Core)
✅ Preserve core/, evaluation/, config/, scrapers/  
✅ Create new directory structure  
✅ Define social media data schema  
✅ Refactor main.py to use core/ modules  

### Phase 2: Data Ingestion
✅ Implement ingestion/ adapters  
✅ Create CSV/JSON loaders  
✅ Create unified data loader  

### Phase 3: Preprocessing
✅ Implement preprocessing/ pipeline  
✅ Text cleaning for social media  
✅ Language detection  
✅ Duplicate detection  

### Phase 4: NLP Analysis
✅ Implement nlp/ modules  
✅ Sentiment analysis (baseline + LLM)  
✅ Emotion analysis  
✅ Named Entity Recognition  
✅ Keyword/hashtag extraction  

### Phase 5: Topic Discovery
✅ Implement topics/ modules  
✅ Semantic embeddings  
✅ Topic clustering  
✅ Topic labeling  
✅ Baseline topic models  

### Phase 6: Trend Detection
✅ Implement trends/ modules  
✅ Temporal analysis  
✅ Trend scoring  
✅ Emerging trend detection  
✅ Explainability  

### Phase 7: Forecasting
✅ Implement forecasting baselines  
✅ Implement proposed forecasting model  
✅ Evaluation metrics  

### Phase 8: Research Infrastructure
✅ Extend evaluation/ framework  
✅ Create research/ documentation  
✅ Create reproducibility scripts  
✅ Implement ablation framework  

### Phase 9: Research Dashboard
✅ Transform app/dashboard.py  
✅ Add visualizations  
✅ Add baseline comparisons  
✅ Add ablation study results  

### Phase 10: Testing & Documentation
✅ Comprehensive test suite  
✅ Research paper outline  
✅ README updates  
✅ Dataset templates  

---

## 4. FINAL ASSESSMENT

### What Makes This Transformation Successful:

1. **Existing DataHawk has excellent research-grade infrastructure**
   - core/ modules are sophisticated
   - evaluation/ framework is exactly what we need
   - Modular architecture is already in place

2. **We are NOT starting from scratch**
   - Reuse ~60% of existing codebase
   - Extend rather than replace
   - Build on proven patterns

3. **Clear separation of concerns**
   - Data ingestion (new)
   - NLP analysis (new)
   - Topic discovery (new)
   - Trend detection (new)
   - Evaluation (extend existing)

4. **Research-first mindset**
   - Every component has baselines
   - Every metric is measurable
   - Every claim is verifiable

### Risk Assessment:

**LOW RISK:**
- LLM integration (already proven)
- Evaluation framework (already excellent)
- Configuration management (already solid)

**MEDIUM RISK:**
- Topic clustering quality depends on embeddings
- Trend detection formula requires tuning
- Forecasting accuracy depends on dataset size

**HIGH RISK:**
- Dataset collection (ethical constraints)
- Ground truth annotation (time-consuming)
- Statistical significance (need sufficient data)

### Recommendation:

✅ **PROCEED WITH TRANSFORMATION**

This is an excellent base for a research project. The existing DataHawk architecture is sophisticated enough to support serious research work. The transformation is feasible, well-scoped for a B.Tech final year project, and has clear research value.

**Key Success Factor:** Do not overengineer. Focus on:
1. Strong baseline comparisons
2. Measurable improvements
3. Reproducible experiments
4. Honest evaluation

This can become a legitimate IEEE-publishable research contribution.

---

**Next Steps:**
1. Create research/ documentation directory
2. Define formal research questions and hypotheses
3. Design social media data schema
4. Implement data ingestion layer
5. Build preprocessing pipeline
6. Add NLP analysis modules
7. Implement topic discovery
8. Build trend detection engine
9. Add forecasting
10. Extend evaluation framework
11. Transform dashboard
12. Write research paper outline

**Estimated Timeline:** 8-12 weeks for full implementation + evaluation + paper writing.

---

*Analysis completed by: AI/ML Research Architect*  
*Date: 2026-08-27*
