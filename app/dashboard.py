"""
DataHawk — Research Dashboard

Multi-tab Streamlit dashboard that serves as both a practical web
scraping tool and a research experiment interface.

Tabs:
  🔍 Scraping    — URL input, classification, adaptive strategy
  📊 Extraction  — NL requirement, schema, structured output
  ✅ Validation  — Field-level validation, confidence, source tracing
  🔄 Correction  — Self-correction history and improvement
  📈 Analytics   — Metrics, charts, experiment logging
  🏆 Benchmark   — Method comparison and ablation results
"""

import json
import logging
import os
import sys
import time
import base64
from typing import Optional

import streamlit as st
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from core.llm_client import GeminiClient
from core.schema_generator import SchemaGenerator, ExtractionSchema
from core.classifier import PageClassifier, PageType
from core.strategy_selector import StrategySelector
from core.extractor import LLMExtractor, ExtractionResult, FieldStatus
from core.validator import ValidationEngine, ValidationResult
from core.confidence import ConfidenceScorer, ConfidenceResult
from core.correction import SelfCorrectionLoop, CorrectionResult
from processing.dom_cleaner import DOMCleaner, CleaningResult
from processing.relevance import RelevanceScorer
from processing.chunker import SemanticChunker
from scrapers.base_scraper import ScrapeResult
from evaluation.benchmark import ExperimentLogger

logger = logging.getLogger(__name__)


def get_logo_base64() -> Optional[str]:
    """Load logo as base64 for HTML embedding."""
    try:
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logo.png",
        )
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def apply_custom_css():
    """Apply the dark-theme CSS."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;
            text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .main-header h1 { color: #fff; font-size: 2rem; margin: 0; }
        .main-header p { color: rgba(255,255,255,0.7); margin: 0.3rem 0 0; font-size: 0.95rem; }
        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            padding: 1rem; border-radius: 12px; text-align: center;
            border: 1px solid #4b5563; margin: 0.3rem 0;
        }
        .status-badge {
            display: inline-block; padding: 0.25rem 0.75rem;
            border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        }
        .badge-success { background: #065f46; color: #6ee7b7; }
        .badge-warning { background: #78350f; color: #fcd34d; }
        .badge-error { background: #7f1d1d; color: #fca5a5; }
        .badge-info { background: #1e3a5f; color: #93c5fd; }
        .info-card {
            background: #1f2937; padding: 1rem; border-radius: 10px;
            border-left: 4px solid #3b82f6; margin: 0.5rem 0;
            border: 1px solid #374151;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""
        <div class="main-header">
            <img src="data:image/png;base64,{logo_b64}" style="height:100px; margin-bottom:0.5rem;" alt="DataHawk"/>
            <p>Adaptive LLM-Assisted Web Data Extraction Framework</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-header">
            <h1>🦅 DataHawk</h1>
            <p>Adaptive LLM-Assisted Web Data Extraction Framework</p>
        </div>
        """, unsafe_allow_html=True)


# ── Tab: Scraping ─────────────────────────────────────────────────

def render_scraping_tab():
    """Render the scraping tab with classification and adaptive strategy."""
    st.header("🔍 Scraping & Page Analysis")

    url = st.text_input(
        "Target URL",
        placeholder="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        key="scrape_url",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        scrape_btn = st.button("🚀 Scrape & Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if scrape_btn and url:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        with st.spinner("🔍 Analyzing page and selecting strategy..."):
            settings = get_settings()
            selector = StrategySelector()
            scrape_result, decision = selector.scrape(url)

            st.session_state["scrape_result"] = scrape_result
            st.session_state["strategy_decision"] = decision
            st.session_state["url"] = url

        if scrape_result.success:
            st.success(f"✅ Scraped successfully using **{scrape_result.strategy.value}** strategy")

            # Classification display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Page Type", decision.page_type.value)
            with col2:
                st.metric("⚙️ Strategy", scrape_result.strategy.value)
            with col3:
                st.metric("📦 HTML Size", f"{scrape_result.html_size_bytes:,} B")
            with col4:
                st.metric("⏱️ Latency", f"{scrape_result.execution_time_ms:.0f} ms")

            # Strategy decision explanation
            with st.expander("🧠 Strategy Decision Details"):
                st.markdown(f"**Reason:** {decision.reason}")
                st.markdown(f"**Fallback chain:** {[f.value for f in decision.fallback_chain]}")
                if decision.features:
                    feat = decision.features
                    st.json({
                        "script_count": feat.script_count,
                        "script_ratio": round(feat.script_ratio, 3),
                        "json_ld_count": feat.json_ld_count,
                        "table_count": feat.table_count,
                        "text_to_html_ratio": round(feat.text_to_html_ratio, 3),
                        "has_spa": feat.has_spa_indicator,
                    })

            # DOM Cleaning
            with st.spinner("🧹 Cleaning DOM..."):
                cleaner = DOMCleaner()
                cleaning_result = cleaner.clean(scrape_result.html)
                st.session_state["cleaning_result"] = cleaning_result

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Original Size", f"{cleaning_result.original_size:,} B")
            with col2:
                st.metric("✨ Cleaned Size", f"{cleaning_result.cleaned_size:,} B")
            with col3:
                st.metric("📉 Reduction", f"{cleaning_result.reduction_ratio:.1%}")

            with st.expander("📄 View Cleaned Content Preview"):
                preview = cleaning_result.cleaned_text[:3000]
                st.text_area("Cleaned content", preview, height=250, disabled=True)

        else:
            st.error(f"❌ Scraping failed: {scrape_result.error}")


# ── Tab: Extraction ───────────────────────────────────────────────

def render_extraction_tab():
    """Render extraction tab with schema generation and LLM extraction."""
    st.header("📊 Schema-Guided Extraction")

    if "scrape_result" not in st.session_state:
        st.info("👆 First scrape a page in the **Scraping** tab.")
        return

    # NL Requirement
    requirement = st.text_area(
        "Describe what to extract",
        placeholder="Extract product name, price, rating, and availability",
        height=80,
        key="extraction_requirement",
    )

    if st.button("🧠 Generate Schema & Extract", type="primary", use_container_width=True):
        if not requirement.strip():
            st.warning("Please describe what you want to extract.")
            return

        llm = GeminiClient()

        # Step 1: Generate schema
        with st.spinner("📋 Generating extraction schema..."):
            gen = SchemaGenerator(llm)
            schema, schema_response = gen.generate(requirement)
            st.session_state["schema"] = schema

        st.subheader("📋 Generated Schema")
        st.json(schema.to_dict())

        if schema_response:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔤 Schema Tokens", schema_response.total_tokens)
            with col2:
                st.metric("⏱️ Schema Gen Time", f"{schema_response.latency_ms:.0f} ms")

        # Step 2: Chunk content
        cleaning = st.session_state.get("cleaning_result")
        if not cleaning:
            st.error("No cleaned content. Re-scrape the page.")
            return

        with st.spinner("✂️ Chunking content..."):
            # Relevance filtering
            scorer = RelevanceScorer()
            filtered_text, scored_sections = scorer.filter_relevant(
                cleaning.cleaned_html, keywords=schema.field_names
            )

            chunker = SemanticChunker()
            chunks = chunker.chunk(filtered_text)
            st.session_state["chunks"] = chunks

        st.info(f"📦 Content split into **{len(chunks)} chunk(s)** after relevance filtering")

        # Step 3: LLM Extraction
        with st.spinner("🤖 Extracting structured data..."):
            extractor = LLMExtractor(llm)
            chunk_texts = [c.full_content for c in chunks]
            results = extractor.extract_multi_chunk(chunk_texts, schema)
            extraction = LLMExtractor.merge_results(results)
            st.session_state["extraction_result"] = extraction

        if extraction.success and extraction.records:
            st.success(f"✅ Extracted **{len(extraction.records)} record(s)**")

            # Display as table
            flat = extraction.flat_records
            if flat:
                df = pd.DataFrame(flat)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Source traceability
                with st.expander("🔗 Source Traceability"):
                    for i, record in enumerate(extraction.records):
                        st.markdown(f"**Record {i+1}:**")
                        for name, field in record.items():
                            status_icon = {"FOUND": "✅", "NOT_FOUND": "❌", "UNCERTAIN": "⚠️"}.get(field.status.value, "❓")
                            st.markdown(
                                f"- `{name}`: {status_icon} **{field.value}** "
                                f"— *{field.source_snippet[:80]}*" if field.source_snippet else
                                f"- `{name}`: {status_icon} **{field.value}**"
                            )

            # Download options
            st.subheader("💾 Download")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📄 JSON", json.dumps(flat, indent=2, default=str),
                                   f"datahawk_{int(time.time())}.json", "application/json",
                                   use_container_width=True)
            with col2:
                if flat:
                    csv_data = pd.DataFrame(flat).to_csv(index=False)
                    st.download_button("📊 CSV", csv_data,
                                       f"datahawk_{int(time.time())}.csv", "text/csv",
                                       use_container_width=True)
            with col3:
                st.download_button("📝 TXT", str(flat),
                                   f"datahawk_{int(time.time())}.txt", "text/plain",
                                   use_container_width=True)

            # Metrics
            if extraction.llm_response:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Records", len(extraction.records))
                with col2:
                    st.metric("🔤 Tokens", extraction.llm_response.total_tokens)
                with col3:
                    st.metric("⏱️ Time", f"{extraction.extraction_time_ms:.0f}ms")
                with col4:
                    st.metric("💰 Cost", f"${extraction.llm_response.estimated_cost_usd:.5f}")

        else:
            st.warning("No data extracted. Try a different extraction description.")


# ── Tab: Validation ───────────────────────────────────────────────

def render_validation_tab():
    """Render validation results with confidence scores."""
    st.header("✅ Validation & Confidence")

    extraction = st.session_state.get("extraction_result")
    schema = st.session_state.get("schema")

    if not extraction or not schema:
        st.info("👆 First extract data in the **Extraction** tab.")
        return

    # Run validation
    validator = ValidationEngine()
    validation = validator.validate(extraction, schema)
    st.session_state["validation_result"] = validation

    # Run confidence scoring
    scorer = ConfidenceScorer()
    confidence = scorer.score(extraction, schema, validation)
    st.session_state["confidence_result"] = confidence

    # Overall status
    if validation.valid:
        st.markdown('<span class="status-badge badge-success">✅ VALID</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-error">❌ INVALID</span>', unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Completeness", f"{validation.completeness_score:.1%}")
    with col2:
        st.metric("🎯 Validity", f"{validation.validity_score:.1%}")
    with col3:
        st.metric("🔒 Confidence", f"{confidence.overall_confidence:.1%}")
    with col4:
        st.metric("⚠️ Errors", validation.total_errors)

    # Missing fields
    if validation.missing_fields:
        st.warning(f"**Missing required fields:** {', '.join(validation.missing_fields)}")

    # Invalid fields
    if validation.invalid_fields:
        st.error(f"**Invalid fields:** {json.dumps(validation.invalid_fields, indent=2)}")

    # Per-field confidence
    st.subheader("📊 Field-Level Confidence")
    conf_data = []
    for name, fc in confidence.field_scores.items():
        conf_data.append({
            "Field": name,
            "Confidence": f"{fc.confidence:.1%}",
            "Status Signal": f"{fc.components.get('status', 0):.2f}",
            "Validation": f"{fc.components.get('validation', 0):.2f}",
            "Source Quality": f"{fc.components.get('snippet', 0):.2f}",
        })

    if conf_data:
        st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

    # Per-field validation details
    with st.expander("🔍 Detailed Validation Results"):
        for name, fv in validation.field_results.items():
            icon = "✅" if fv.valid else "❌"
            st.markdown(f"**{icon} {name}**")
            if fv.errors:
                for err in fv.errors:
                    st.markdown(f"  - 🔴 {err}")
            if fv.warnings:
                for warn in fv.warnings:
                    st.markdown(f"  - 🟡 {warn}")


# ── Tab: Self-Correction ─────────────────────────────────────────

def render_correction_tab():
    """Render self-correction interface."""
    st.header("🔄 Self-Correction")

    extraction = st.session_state.get("extraction_result")
    schema = st.session_state.get("schema")
    cleaning = st.session_state.get("cleaning_result")
    validation = st.session_state.get("validation_result")

    if not extraction or not schema or not cleaning:
        st.info("👆 First extract and validate data in previous tabs.")
        return

    needs_correction = validation and not validation.valid

    if needs_correction:
        st.warning("⚠️ Extraction has validation errors. Self-correction can attempt to fix them.")
    else:
        st.success("✅ Extraction passed validation. Correction is optional.")

    if st.button("🔄 Run Self-Correction", type="primary", use_container_width=True):
        with st.spinner("🔄 Running self-correction loop..."):
            llm = GeminiClient()
            correction_loop = SelfCorrectionLoop(llm_client=llm)

            content = cleaning.cleaned_text[:15000]
            correction_result = correction_loop.extract_with_correction(
                content, schema, initial_extraction=extraction
            )
            st.session_state["correction_result"] = correction_result

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Before Correction")
            if correction_result.initial_confidence:
                st.metric("Confidence", f"{correction_result.initial_confidence.overall_confidence:.1%}")
            if correction_result.initial_validation:
                st.metric("Completeness", f"{correction_result.initial_validation.completeness_score:.1%}")

        with col2:
            st.subheader("✨ After Correction")
            if correction_result.final_confidence:
                st.metric("Confidence", f"{correction_result.final_confidence.overall_confidence:.1%}",
                          delta=f"{correction_result.improvement_delta:+.1%}")
            if correction_result.final_validation:
                st.metric("Completeness", f"{correction_result.final_validation.completeness_score:.1%}")

        # Correction metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔄 Attempts", correction_result.total_attempts)
        with col2:
            st.metric("🔤 Extra Tokens", correction_result.total_additional_tokens)
        with col3:
            st.metric("⏱️ Extra Time", f"{correction_result.total_additional_latency_ms:.0f}ms")
        with col4:
            st.metric("✅ Success", "Yes" if correction_result.correction_successful else "No")

        # Corrected data
        if correction_result.final_extraction and correction_result.final_extraction.records:
            st.subheader("📊 Corrected Data")
            corrected_flat = correction_result.final_extraction.flat_records
            st.dataframe(pd.DataFrame(corrected_flat), use_container_width=True, hide_index=True)

            # Update session state
            st.session_state["extraction_result"] = correction_result.final_extraction

        # Correction history
        if correction_result.attempts:
            with st.expander("📜 Correction History"):
                for attempt in correction_result.attempts:
                    st.markdown(
                        f"**Attempt {attempt.attempt_number}**: "
                        f"Type: {attempt.failure_type}, "
                        f"Failed fields: {attempt.failed_fields}, "
                        f"Improved: {'✅' if attempt.improved else '❌'}, "
                        f"Tokens: {attempt.input_tokens + attempt.output_tokens}"
                    )


# ── Tab: Analytics ────────────────────────────────────────────────

def render_analytics_tab():
    """Render analytics with charts and experiment logging."""
    st.header("📈 Analytics & Experiment Log")

    # Current session metrics
    extraction = st.session_state.get("extraction_result")
    validation = st.session_state.get("validation_result")
    confidence = st.session_state.get("confidence_result")
    scrape_result = st.session_state.get("scrape_result")
    cleaning = st.session_state.get("cleaning_result")

    if extraction and extraction.llm_response:
        st.subheader("📊 Current Session Metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Records Extracted", extraction.record_count)
            if validation:
                st.metric("✅ Completeness", f"{validation.completeness_score:.1%}")
        with col2:
            st.metric("🔤 Total Tokens", extraction.llm_response.total_tokens)
            if confidence:
                st.metric("🔒 Confidence", f"{confidence.overall_confidence:.1%}")
        with col3:
            st.metric("⏱️ Total Latency", f"{extraction.extraction_time_ms:.0f}ms")
            st.metric("💰 Est. Cost", f"${extraction.llm_response.estimated_cost_usd:.6f}")

        # Token breakdown
        if cleaning:
            st.subheader("📦 Content Processing")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw HTML", f"{cleaning.original_size:,} B")
            with col2:
                st.metric("Cleaned", f"{cleaning.cleaned_size:,} B")
            with col3:
                st.metric("Reduction", f"{cleaning.reduction_ratio:.1%}")

    # Experiment log browser
    st.subheader("📜 Experiment History")
    try:
        exp_logger = ExperimentLogger()
        logs = exp_logger.get_all_logs()
        if logs:
            df = pd.DataFrame(logs)
            display_cols = [
                "timestamp", "method", "category", "f1", "precision", "recall",
                "latency_ms", "input_tokens", "estimated_cost",
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

            # Summary by method
            if "method" in df.columns and "f1" in df.columns:
                st.subheader("📊 Average F1 by Method")
                summary = df.groupby("method").agg({
                    "f1": "mean", "precision": "mean", "recall": "mean",
                    "latency_ms": "mean", "estimated_cost": "sum",
                }).round(4)
                st.dataframe(summary, use_container_width=True)
        else:
            st.info("No experiments logged yet. Run benchmark tests to populate.")
    except Exception as e:
        st.info(f"Experiment log not available: {e}")


# ── Tab: Benchmark ────────────────────────────────────────────────

def render_benchmark_tab():
    """Render benchmark comparison interface."""
    st.header("🏆 Benchmark & Ablation Studies")

    st.markdown("""
    Compare DataHawk against baseline methods on the benchmark dataset.

    **Methods compared:**
    1. **BS4 + Rules**: BeautifulSoup + regex extraction
    2. **Selenium + Rules**: Browser rendering + regex extraction
    3. **Raw LLM**: Full page → LLM (original DataHawk approach)
    4. **DataHawk**: Complete adaptive pipeline
    """)

    # Load experiment results
    try:
        exp_logger = ExperimentLogger()
        logs = exp_logger.get_all_logs()

        if logs:
            df = pd.DataFrame(logs)

            if "method" in df.columns:
                # Method comparison table
                st.subheader("📊 Method Comparison")
                methods = df["method"].unique()

                comparison_data = []
                for method in methods:
                    method_df = df[df["method"] == method]
                    comparison_data.append({
                        "Method": method,
                        "Avg F1": f"{method_df['f1'].mean():.4f}",
                        "Avg Precision": f"{method_df['precision'].mean():.4f}",
                        "Avg Recall": f"{method_df['recall'].mean():.4f}",
                        "Avg Latency (ms)": f"{method_df['latency_ms'].mean():.0f}",
                        "Total Tokens": f"{method_df['input_tokens'].sum():,}",
                        "Total Cost ($)": f"{method_df['estimated_cost'].sum():.6f}",
                        "Success Rate": f"{method_df['success'].mean():.1%}",
                    })

                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

                # Charts
                st.subheader("📈 Visual Comparison")

                chart_metric = st.selectbox(
                    "Select metric to compare:",
                    ["f1", "precision", "recall", "latency_ms", "input_tokens", "estimated_cost"],
                )

                chart_data = df.groupby("method")[chart_metric].mean()
                st.bar_chart(chart_data)

        else:
            st.info("No benchmark results yet.")

    except Exception as e:
        st.info(f"Benchmark data not available: {e}")

    # Ablation study section
    st.subheader("🔬 Ablation Study")
    st.markdown("""
    Run experiments A–F to isolate each component's contribution:
    - **A**: LLM only
    - **B**: LLM + DOM cleaning
    - **C**: LLM + schema
    - **D**: LLM + schema + validation
    - **E**: LLM + schema + validation + self-correction
    - **F**: Complete DataHawk
    """)

    if "scrape_result" in st.session_state and st.session_state["scrape_result"].success:
        requirement = st.text_input(
            "Extraction task for ablation:",
            value=st.session_state.get("extraction_requirement", ""),
            key="ablation_requirement",
        )

        if st.button("🔬 Run Ablation Study", type="primary"):
            if requirement:
                with st.spinner("Running 6 ablation experiments... This may take a few minutes."):
                    from evaluation.experiments import AblationStudy as AblStudy
                    llm = GeminiClient()
                    study = AblStudy(llm_client=llm)
                    html = st.session_state["scrape_result"].html
                    ablation_results = study.run_all(html, requirement)

                st.subheader("🔬 Ablation Results")
                ablation_data = []
                for name, res in ablation_results.items():
                    ablation_data.append({
                        "Experiment": name,
                        "Description": res.description,
                        "Success": "✅" if res.success else "❌",
                        "Records": len(res.records),
                        "Latency (ms)": f"{res.latency_ms:.0f}",
                        "Tokens": res.input_tokens + res.output_tokens,
                        "Confidence": f"{res.confidence:.2f}" if res.confidence else "N/A",
                    })

                st.dataframe(pd.DataFrame(ablation_data), use_container_width=True, hide_index=True)
    else:
        st.info("Scrape a page first to run ablation experiments.")


# ── Sidebar ───────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar with system status and info."""
    with st.sidebar:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem;">
                <img src="data:image/png;base64,{logo_b64}" style="height:80px;" alt="DataHawk"/>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔧 System Status")
        settings = get_settings()

        if settings.is_gemini_configured:
            st.success(f"🟢 Gemini: Connected ({settings.gemini_model})")
        else:
            st.error("🔴 Gemini: Not configured")

        if settings.is_brightdata_configured and settings.use_brightdata:
            st.success("🟢 BrightData: Enabled")
        else:
            st.info("⚪ BrightData: Disabled (using local scrapers)")

        st.markdown("---")
        st.markdown("### 📚 Quick Reference")
        st.markdown("""
        1. **Scraping** → Enter URL & scrape
        2. **Extraction** → Describe what to extract
        3. **Validation** → Review quality & confidence
        4. **Correction** → Fix extraction errors
        5. **Analytics** → View metrics & history
        6. **Benchmark** → Compare methods
        """)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#6b7280; font-size:0.75rem;">
            <p>DataHawk Research Framework</p>
            <p>© 2025 Janith Prabash</p>
        </div>
        """, unsafe_allow_html=True)


# ── Main Dashboard ────────────────────────────────────────────────

def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="DataHawk — Research Dashboard",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()
    render_header()
    render_sidebar()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Scraping",
        "📊 Extraction",
        "✅ Validation",
        "🔄 Correction",
        "📈 Analytics",
        "🏆 Benchmark",
    ])

    with tab1:
        render_scraping_tab()
    with tab2:
        render_extraction_tab()
    with tab3:
        render_validation_tab()
    with tab4:
        render_correction_tab()
    with tab5:
        render_analytics_tab()
    with tab6:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
