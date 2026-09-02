"""
DataHawk — Ablation Study Framework

Runs experiments that isolate the contribution of each component.

Experiments:
  A: LLM only (raw content, no schema, no cleaning)
  B: LLM + DOM cleaning
  C: LLM + schema generation
  D: LLM + schema + validation
  E: LLM + schema + validation + self-correction
  F: Complete DataHawk pipeline (adaptive scraping + all components)

Each experiment is recorded with full metrics for research analysis.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from config.settings import get_settings
from core.llm_client import GeminiClient
from core.schema_generator import SchemaGenerator, ExtractionSchema
from core.extractor import LLMExtractor, ExtractionResult
from core.validator import ValidationEngine
from core.confidence import ConfidenceScorer
from core.correction import SelfCorrectionLoop
from processing.dom_cleaner import DOMCleaner
from processing.relevance import RelevanceScorer
from processing.chunker import SemanticChunker
from evaluation.benchmark import ExperimentLogger, ExperimentLog

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    experiment_name: str
    description: str
    records: List[Dict[str, Any]]
    success: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    html_size: int = 0
    cleaned_size: int = 0
    validation_errors: int = 0
    confidence: float = 0.0
    retry_count: int = 0


class AblationStudy:
    """
    Framework for running ablation experiments.

    Each experiment activates/deactivates specific components to
    measure their individual contribution to overall performance.
    """

    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
    ):
        self.llm = llm_client or GeminiClient()
        self.exp_logger = experiment_logger or ExperimentLogger()

    def run_all(
        self,
        html: str,
        extraction_task: str,
        schema: Optional[ExtractionSchema] = None,
    ) -> Dict[str, AblationResult]:
        """
        Run all ablation experiments on the same content.

        Args:
            html: Raw HTML content.
            extraction_task: NL extraction requirement.
            schema: Pre-computed schema (if None, generates one).

        Returns:
            Dict mapping experiment name to AblationResult.
        """
        results = {}

        # Pre-compute schema (used by experiments C-F)
        if schema is None:
            generator = SchemaGenerator(self.llm)
            schema, _ = generator.generate(extraction_task)

        results["A_llm_only"] = self.experiment_a_llm_only(html, extraction_task)
        results["B_llm_dom_clean"] = self.experiment_b_llm_dom_cleaning(html, extraction_task)
        results["C_llm_schema"] = self.experiment_c_llm_schema(html, schema)
        results["D_llm_schema_validation"] = self.experiment_d_llm_schema_validation(html, schema)
        results["E_llm_schema_validation_correction"] = self.experiment_e_with_correction(html, schema)
        results["F_complete_datahawk"] = self.experiment_f_complete(html, schema)

        logger.info(
            f"Ablation study complete: {len(results)} experiments. "
            f"Results: {', '.join(f'{k}={v.success}' for k, v in results.items())}"
        )

        return results

    def experiment_a_llm_only(
        self, html: str, task: str,
    ) -> AblationResult:
        """Experiment A: Raw content → LLM, no schema, no cleaning."""
        result = AblationResult(
            experiment_name="A_llm_only",
            description="LLM only — raw content, no schema, no cleaning",
            records=[],
        )

        try:
            # Basic text extraction
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)[:15000]
            result.html_size = len(html)
            result.cleaned_size = len(text)

            start = time.time()
            prompt = (
                f"Extract the following from this content: {task}\n\n"
                f"Content:\n{text}\n\n"
                f"Return JSON array of objects:"
            )
            parsed, llm_resp = self.llm.generate_json(prompt=prompt)
            result.latency_ms = (time.time() - start) * 1000
            result.input_tokens = llm_resp.input_tokens
            result.output_tokens = llm_resp.output_tokens

            if parsed:
                if isinstance(parsed, list):
                    result.records = parsed
                elif isinstance(parsed, dict):
                    result.records = parsed.get("records", [parsed])
                result.success = True

        except Exception as e:
            logger.error(f"Experiment A failed: {e}")

        return result

    def experiment_b_llm_dom_cleaning(
        self, html: str, task: str,
    ) -> AblationResult:
        """Experiment B: DOM cleaning + LLM (no schema)."""
        result = AblationResult(
            experiment_name="B_llm_dom_clean",
            description="LLM + DOM cleaning — cleaned content, no schema",
            records=[],
        )

        try:
            result.html_size = len(html)
            cleaner = DOMCleaner()
            cleaning = cleaner.clean(html)
            text = cleaning.cleaned_text[:15000]
            result.cleaned_size = len(text)

            start = time.time()
            prompt = (
                f"Extract the following from this cleaned webpage content: {task}\n\n"
                f"Content:\n{text}\n\n"
                f"Return JSON array of objects:"
            )
            parsed, llm_resp = self.llm.generate_json(prompt=prompt)
            result.latency_ms = (time.time() - start) * 1000
            result.input_tokens = llm_resp.input_tokens
            result.output_tokens = llm_resp.output_tokens

            if parsed:
                if isinstance(parsed, list):
                    result.records = parsed
                elif isinstance(parsed, dict):
                    result.records = parsed.get("records", [parsed])
                result.success = True

        except Exception as e:
            logger.error(f"Experiment B failed: {e}")

        return result

    def experiment_c_llm_schema(
        self, html: str, schema: ExtractionSchema,
    ) -> AblationResult:
        """Experiment C: Schema-guided LLM extraction (no cleaning, no validation)."""
        result = AblationResult(
            experiment_name="C_llm_schema",
            description="LLM + schema — schema-guided extraction, no cleaning",
            records=[],
        )

        try:
            result.html_size = len(html)
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            result.cleaned_size = len(text)

            start = time.time()
            extractor = LLMExtractor(self.llm)
            extraction = extractor.extract(text[:15000], schema)
            result.latency_ms = (time.time() - start) * 1000

            if extraction.llm_response:
                result.input_tokens = extraction.llm_response.input_tokens
                result.output_tokens = extraction.llm_response.output_tokens

            if extraction.success:
                result.records = extraction.flat_records
                result.success = True

        except Exception as e:
            logger.error(f"Experiment C failed: {e}")

        return result

    def experiment_d_llm_schema_validation(
        self, html: str, schema: ExtractionSchema,
    ) -> AblationResult:
        """Experiment D: Schema + validation (no cleaning, no correction)."""
        result = AblationResult(
            experiment_name="D_llm_schema_validation",
            description="LLM + schema + validation — with type/semantic checks",
            records=[],
        )

        try:
            result.html_size = len(html)

            # Clean content
            cleaner = DOMCleaner()
            cleaning = cleaner.clean(html)
            text = cleaning.cleaned_text
            result.cleaned_size = len(text)

            # Schema-guided extraction
            start = time.time()
            extractor = LLMExtractor(self.llm)
            chunker = SemanticChunker()
            chunks = chunker.chunk(text)
            chunk_texts = [c.full_content for c in chunks]

            extraction_results = extractor.extract_multi_chunk(chunk_texts, schema)
            extraction = LLMExtractor.merge_results(extraction_results)

            # Validate
            validator = ValidationEngine()
            validation = validator.validate(extraction, schema)
            result.validation_errors = validation.total_errors

            # Score confidence
            scorer = ConfidenceScorer()
            confidence = scorer.score(extraction, schema, validation)
            result.confidence = confidence.overall_confidence

            result.latency_ms = (time.time() - start) * 1000

            if extraction.llm_response:
                result.input_tokens = extraction.llm_response.input_tokens
                result.output_tokens = extraction.llm_response.output_tokens

            if extraction.success:
                result.records = extraction.flat_records
                result.success = True

        except Exception as e:
            logger.error(f"Experiment D failed: {e}")

        return result

    def experiment_e_with_correction(
        self, html: str, schema: ExtractionSchema,
    ) -> AblationResult:
        """Experiment E: Schema + validation + self-correction."""
        result = AblationResult(
            experiment_name="E_llm_schema_validation_correction",
            description="LLM + schema + validation + self-correction",
            records=[],
        )

        try:
            result.html_size = len(html)

            cleaner = DOMCleaner()
            cleaning = cleaner.clean(html)
            text = cleaning.cleaned_text
            result.cleaned_size = len(text)

            start = time.time()

            # Use chunked content
            chunker = SemanticChunker()
            chunks = chunker.chunk(text)
            combined_text = "\n\n".join(c.full_content for c in chunks)[:15000]

            # Extraction with self-correction
            correction_loop = SelfCorrectionLoop(llm_client=self.llm)
            correction_result = correction_loop.extract_with_correction(
                combined_text, schema
            )

            result.latency_ms = (time.time() - start) * 1000
            result.retry_count = correction_result.total_attempts

            if correction_result.final_extraction:
                result.records = correction_result.final_extraction.flat_records
                result.success = True

                if correction_result.final_extraction.llm_response:
                    result.input_tokens = correction_result.final_extraction.llm_response.input_tokens
                    result.output_tokens = correction_result.final_extraction.llm_response.output_tokens

                result.input_tokens += correction_result.total_additional_tokens

            if correction_result.final_validation:
                result.validation_errors = correction_result.final_validation.total_errors

            if correction_result.final_confidence:
                result.confidence = correction_result.final_confidence.overall_confidence

        except Exception as e:
            logger.error(f"Experiment E failed: {e}")

        return result

    def experiment_f_complete(
        self, html: str, schema: ExtractionSchema,
    ) -> AblationResult:
        """Experiment F: Complete DataHawk pipeline."""
        # Same as E but with relevance filtering
        result = AblationResult(
            experiment_name="F_complete_datahawk",
            description="Complete DataHawk — adaptive scraping + all components",
            records=[],
        )

        try:
            result.html_size = len(html)

            # DOM cleaning + relevance filtering
            cleaner = DOMCleaner()
            cleaning = cleaner.clean(html)

            relevance_scorer = RelevanceScorer()
            keywords = schema.field_names
            filtered_text, scored_sections = relevance_scorer.filter_relevant(
                cleaning.cleaned_html, keywords=keywords
            )
            result.cleaned_size = len(filtered_text)

            start = time.time()

            # Smart chunking
            chunker = SemanticChunker()
            chunks = chunker.chunk(filtered_text)
            combined = "\n\n".join(c.full_content for c in chunks)[:15000]

            # Extraction with self-correction
            correction_loop = SelfCorrectionLoop(llm_client=self.llm)
            correction_result = correction_loop.extract_with_correction(
                combined, schema
            )

            result.latency_ms = (time.time() - start) * 1000
            result.retry_count = correction_result.total_attempts

            if correction_result.final_extraction:
                result.records = correction_result.final_extraction.flat_records
                result.success = True

                if correction_result.final_extraction.llm_response:
                    result.input_tokens = correction_result.final_extraction.llm_response.input_tokens
                    result.output_tokens = correction_result.final_extraction.llm_response.output_tokens

                result.input_tokens += correction_result.total_additional_tokens

            if correction_result.final_validation:
                result.validation_errors = correction_result.final_validation.total_errors

            if correction_result.final_confidence:
                result.confidence = correction_result.final_confidence.overall_confidence

        except Exception as e:
            logger.error(f"Experiment F failed: {e}")

        return result
