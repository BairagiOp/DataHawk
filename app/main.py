"""
DataHawk Research Dashboard

Interactive research interface for social media intelligence analysis.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="DataHawk Research Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .research-question {
        background: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""

    # Header
    st.markdown('<div class="main-header">🦅 DataHawk Research Platform</div>',
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Social Media Intelligence for Academic Research")

    # Sidebar navigation
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Home",
            "📊 Data Collection",
            "🔍 NLP Analysis",
            "🎯 Topic Discovery",
            "📈 Trend Detection",
            "🚀 Emerging Trends",
            "🔮 Forecasting",
            "⚖️ Baseline Comparison",
            "🧪 Ablation Study",
            "📋 Experiments",
            "📄 Documentation"
        ]
    )

    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Collection":
        show_data_collection()
    elif page == "🔍 NLP Analysis":
        show_nlp_analysis()
    elif page == "🎯 Topic Discovery":
        show_topic_discovery()
    elif page == "📈 Trend Detection":
        show_trend_detection()
    elif page == "🚀 Emerging Trends":
        show_emerging_trends()
    elif page == "🔮 Forecasting":
        show_forecasting()
    elif page == "⚖️ Baseline Comparison":
        show_baseline_comparison()
    elif page == "🧪 Ablation Study":
        show_ablation_study()
    elif page == "📋 Experiments":
        show_experiments()
    elif page == "📄 Documentation":
        show_documentation()


def show_home():
    """Home page with research overview"""

    st.markdown("---")

    # Research Questions
    st.header("Research Questions")

    questions = [
        ("RQ1", "Does semantic clustering identify more coherent topics than frequency-based approaches?"),
        ("RQ2", "Do LLM-based sentiment models outperform lexicon-based baselines?"),
        ("RQ3", "Does composite scoring detect trends more accurately than frequency counting?"),
        ("RQ4", "Can we classify emerging trends into meaningful categories?"),
        ("RQ5", "Are composite trend predictions explainable to human analysts?"),
        ("RQ6", "Can ensemble methods predict trend trajectories more accurately?"),
        ("RQ7", "Which features contribute most to trend detection accuracy?")
    ]

    for rq_id, question in questions:
        st.markdown(f"""
        <div class="research-question">
            <strong>{rq_id}:</strong> {question}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System Architecture
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Core Modules")
        st.markdown("""
        - **Data Ingestion**: CSV, JSON, multi-source loading
        - **Preprocessing**: Cleaning, deduplication, spam filtering
        - **NLP Analysis**: Sentiment, emotion, NER, embeddings
        - **Topic Discovery**: Semantic clustering, LLM labeling
        - **Trend Detection**: Composite scoring, classification
        - **Forecasting**: Time-series prediction models
        - **Evaluation**: Metrics, baselines, ablation studies
        """)

    with col2:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Fair Baselines**: Every method compared against simple baseline
        - **Explainable AI**: Human-readable trend explanations
        - **Ethical Design**: No ToS violations, privacy-preserving
        - **Reproducible**: Fixed seeds, documented parameters
        - **Modular**: Independent component testing
        - **Publication-Ready**: IEEE paper template included
        """)

    st.markdown("---")

    # Quick Start
    st.header("Quick Start")

    st.code("""
# 1. Generate sample data
python dataset/sample_data.py

# 2. Run complete experiment
python research/run_experiment.py --data dataset/test_data_small.csv

# 3. Launch dashboard
streamlit run app/main.py
    """, language="bash")

    st.info("💡 **Tip**: Start with the Data Collection page to load your dataset.")


def show_data_collection():
    """Data collection and loading interface"""

    st.header("📊 Data Collection")

    st.warning("⚠️ **Ethical Guidelines**: Only use publicly accessible content. Never bypass authentication or violate Terms of Service.")

    # File upload
    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=['csv', 'json', 'jsonl'],
        help="Upload your social media dataset"
    )

    if uploaded_file:
        st.success(f"[OK] File uploaded: {uploaded_file.name}")

        # Load preview
        try:
            from ingestion.loader import DataLoader

            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            loader = DataLoader()
            posts = loader.load(temp_path)

            st.info(f"Loaded {len(posts)} posts")

            # Show preview
            if posts:
                st.subheader("Data Preview")
                st.write(f"**First post:**")
                st.json({
                    'post_id': posts[0].post_id,
                    'text': posts[0].text[:100] + "...",
                    'timestamp': str(posts[0].timestamp),
                    'engagement': posts[0].engagement.total_engagement
                })

            # Clean up
            os.remove(temp_path)

        except Exception as e:
            st.error(f"Error loading data: {e}")

    st.markdown("---")

    # Sample data
    st.subheader("Or Use Sample Data")

    if st.button("Generate Sample Dataset"):
        with st.spinner("Generating synthetic data..."):
            try:
                os.system("python dataset/sample_data.py")
                st.success("[OK] Sample datasets generated in `dataset/` directory")
            except Exception as e:
                st.error(f"Error: {e}")

    # Data format guide
    with st.expander("📋 Data Format Guide"):
        st.markdown("""
        ### Required Fields

        **CSV Format:**
        ```csv
        post_id,text,timestamp,likes,comments,shares
        post1,"Sample text",2026-08-01,50,10,5
        ```

        **JSON Format:**
        ```json
        [
          {
            "post_id": "post1",
            "text": "Sample text",
            "timestamp": "2026-08-01T10:00:00Z",
            "likes": 50,
            "comments": 10,
            "shares": 5
          }
        ]
        ```
        """)


def show_nlp_analysis():
    """NLP analysis interface"""

    st.header("🔍 NLP Analysis")
    st.markdown("**RQ2**: Do LLM-based sentiment models outperform lexicon-based baselines?")

    # Text input
    st.subheader("Analyze Text")

    sample_texts = [
        "AI agents are transforming software development!",
        "Disappointed with the latest update. Many bugs.",
        "Interesting developments in machine learning."
    ]

    text = st.text_area(
        "Enter text to analyze",
        value=sample_texts[0],
        height=100
    )

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            try:
                from nlp import SentimentAnalyzer, EmotionClassifier, KeywordExtractor

                # Sentiment
                sentiment_analyzer = SentimentAnalyzer(method='vader')
                sentiment = sentiment_analyzer.analyze(text)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sentiment", sentiment.label.upper())

                with col2:
                    st.metric("Confidence", f"{sentiment.confidence:.2%}")

                with col3:
                    st.metric("Score", f"{sentiment.scores[sentiment.label]:.3f}")

                # Emotion
                emotion_clf = EmotionClassifier()
                emotion = emotion_clf.classify(text)

                st.subheader("Emotion Distribution")
                st.bar_chart(emotion.scores)

                # Keywords
                keyword_extractor = KeywordExtractor()
                keywords = keyword_extractor.extract_keywords(text, top_n=5)

                st.subheader("Keywords")
                st.write(", ".join([f"`{kw}`" for kw in keywords]))

            except Exception as e:
                st.error(f"Error: {e}")

    # Batch analysis
    st.markdown("---")
    st.subheader("Batch Analysis")
    st.info("Upload a dataset in the Data Collection page to run batch NLP analysis.")


def show_topic_discovery():
    """Topic discovery interface"""

    st.header("🎯 Topic Discovery")
    st.markdown("**RQ1**: Does semantic clustering identify more coherent topics?")

    st.info("Load a dataset to discover topics using semantic clustering (HDBSCAN) vs frequency-based methods (TF-IDF + K-Means)")

    # Configuration
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Clustering Method",
            ["HDBSCAN (Proposed)", "K-Means (Baseline)"]
        )

    with col2:
        min_cluster_size = st.slider("Min Cluster Size", 3, 20, 5)

    if st.button("Discover Topics"):
        st.info("Please load data in Data Collection page first")


def show_trend_detection():
    """Trend detection interface"""

    st.header("📈 Trend Detection")
    st.markdown("**RQ3**: Does composite scoring detect trends more accurately?")

    st.info("Composite scoring: **TrendScore = α·Volume + β·Growth + γ·Engagement + δ·Novelty**")

    # Weight configuration
    st.subheader("Scoring Weights (for Ablation)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        alpha = st.slider("α (Volume)", 0.0, 1.0, 0.2, 0.05)

    with col2:
        beta = st.slider("β (Growth)", 0.0, 1.0, 0.4, 0.05)

    with col3:
        gamma = st.slider("γ (Engagement)", 0.0, 1.0, 0.3, 0.05)

    with col4:
        delta = st.slider("δ (Novelty)", 0.0, 1.0, 0.1, 0.05)

    total = alpha + beta + gamma + delta
    st.caption(f"Total weight: {total:.2f} (should be ~1.0)")

    if st.button("Detect Trends"):
        st.info("Please load data in Data Collection page first")


def show_emerging_trends():
    """Emerging trends interface"""

    st.header("🚀 Emerging Trends")
    st.markdown("**RQ4**: Can we classify emerging trends into categories?")

    st.markdown("""
    ### Trend Categories

    - **🔥 VIRAL**: Explosive growth (>200% increase)
    - **📈 RISING**: Consistent upward trend
    - **🌱 EMERGING**: Low baseline + high growth
    - **➡️ STABLE**: Low variance over time
    - **📉 DECLINING**: Negative growth trend
    """)

    if st.button("Classify Trends"):
        st.info("Please load data in Data Collection page first")


def show_forecasting():
    """Forecasting interface"""

    st.header("🔮 Trend Forecasting")
    st.markdown("**RQ6**: Can ensemble methods predict trajectories more accurately?")

    st.subheader("Methods")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baselines:**")
        st.markdown("- Naive (repeat last value)")
        st.markdown("- Moving Average")
        st.markdown("- Linear Regression")

    with col2:
        st.markdown("**Proposed:**")
        st.markdown("- XGBoost ensemble")
        st.markdown("- Random Forest")
        st.markdown("- Feature engineering")

    forecast_steps = st.slider("Forecast Horizon (days)", 1, 14, 7)

    if st.button("Generate Forecast"):
        st.info("Please load data in Data Collection page first")


def show_baseline_comparison():
    """Baseline comparison interface"""

    st.header("⚖️ Baseline Comparison")

    st.markdown("""
    ### Fair Comparisons

    Every proposed method is compared against a simple, interpretable baseline:

    | Component | Baseline | Proposed |
    |-----------|----------|----------|
    | **Sentiment** | VADER (lexicon) | Gemini LLM |
    | **Topics** | TF-IDF + K-Means | Embeddings + HDBSCAN |
    | **Trends** | Frequency counting | Composite scoring |
    | **Forecasting** | Naive/Moving Average | XGBoost ensemble |
    """)

    if st.button("Run Comparison"):
        st.info("Please load data in Data Collection page first")


def show_ablation_study():
    """Ablation study interface"""

    st.header("🧪 Ablation Study")
    st.markdown("**RQ7**: Which features contribute most to trend detection?")

    st.markdown("""
    ### Tested Configurations

    1. **Full System**: All features (volume + growth + engagement + novelty)
    2. **Leave-One-Out**: Remove each feature individually
    3. **Minimal**: Only one feature at a time
    """)

    if st.button("Run Ablation Study"):
        st.info("This will test all feature combinations and rank their importance")


def show_experiments():
    """Experiments interface"""

    st.header("📋 Reproducible Experiments")

    st.markdown("""
    ### Run Complete Experiment Pipeline

    This executes all 7 research questions in sequence:
    1. Load and preprocess data
    2. Run NLP analysis (RQ2)
    3. Discover topics (RQ1)
    4. Detect trends (RQ3-RQ5)
    5. Forecast trajectories (RQ6)
    6. Run ablation study (RQ7)
    7. Generate results report
    """)

    data_source = st.selectbox(
        "Select Dataset",
        [
            "dataset/test_data_small.csv",
            "dataset/test_data_medium.csv",
            "dataset/test_data_large.csv",
            "Custom (upload first)"
        ]
    )

    if st.button("Run Full Experiment"):
        with st.spinner("Running experiment... This may take several minutes."):
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "research/run_experiment.py", "--data", data_source],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    st.success("[OK] Experiment completed!")
                    st.text(result.stdout)

                    # Load results
                    try:
                        import json
                        with open('research/experiment_results.json') as f:
                            results = json.load(f)

                        st.subheader("Results Summary")
                        st.json(results)
                    except:
                        pass
                else:
                    st.error("Experiment failed")
                    st.text(result.stderr)

            except Exception as e:
                st.error(f"Error: {e}")


def show_documentation():
    """Documentation page"""

    st.header("📄 Documentation")

    tabs = st.tabs([
        "Research Paper",
        "Implementation",
        "API Reference",
        "Ethical Guidelines"
    ])

    with tabs[0]:
        st.markdown("""
        ### Research Paper

        A complete IEEE-style research paper is included in `RESEARCH_PAPER.md`.

        **Sections:**
        - Abstract
        - Introduction
        - Related Work
        - Methodology
        - Experimental Design
        - Results (7 RQs)
        - Discussion
        - Conclusion
        - References

        📄 [View Full Paper](RESEARCH_PAPER.md)
        """)

    with tabs[1]:
        st.markdown("""
        ### Implementation Status

        Current progress and remaining tasks are documented in `IMPLEMENTATION_STATUS.md`.

        **Completed:**
        - [OK] Preprocessing module
        - [OK] NLP analysis layer
        - [OK] Topic discovery system
        - [OK] Trend detection engine
        - [OK] Forecasting module
        - [OK] Evaluation framework

        📊 [View Status](IMPLEMENTATION_STATUS.md)
        """)

    with tabs[2]:
        st.markdown("""
        ### API Reference

        Quick reference for core modules:

        ```python
        # Data Loading
        from ingestion.loader import DataLoader
        posts = DataLoader().load('data.csv')

        # NLP Analysis
        from nlp import SentimentAnalyzer
        sentiment = SentimentAnalyzer(method='vader').analyze(text)

        # Topic Discovery
        from topics import TopicClusterer
        clusters = TopicClusterer(method='hdbscan').fit_predict(embeddings)

        # Trend Detection
        from trends import TrendScorer
        score = TrendScorer().compute_trend_score(volume, timestamps, engagement, novelty)

        # Forecasting
        from forecasting import TrendForecaster
        forecast = TrendForecaster(method='xgboost').fit(series).predict(steps=7)
        ```
        """)

    with tabs[3]:
        st.markdown("""
        ### Ethical Guidelines

        ⚠️ **IMPORTANT**: This platform is designed for ethical research only.

        **✅ Allowed:**
        - Publicly accessible content
        - User-provided CSV/JSON datasets
        - Research benchmark datasets
        - Content with explicit consent

        **❌ Not Allowed:**
        - Bypassing login/authentication
        - Violating Terms of Service
        - Scraping private/protected content
        - Collecting personal identifiable information

        **Privacy:**
        - Anonymize user identifiers
        - Use aggregated data where possible
        - Never expose sensitive personal information
        - Respect platform robots.txt

        **Publication:**
        - Never fabricate results or accuracy scores
        - Document all limitations honestly
        - Use fair baseline comparisons
        - Ensure reproducibility
        """)


# Run app
if __name__ == "__main__":
    main()
