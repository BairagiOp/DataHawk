"""
DataHawk — Unified Gemini LLM Client

Provides a single, reusable interface to Google Gemini for all LLM
operations (schema generation, extraction, self-correction).
Handles rate limiting, retries, token counting, and cost estimation.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    success: bool = True
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate for Gemini 2.0 Flash (may change with pricing)."""
        # Gemini 2.0 Flash pricing (approximate, as of mid-2025):
        # Input: $0.10 per 1M tokens, Output: $0.40 per 1M tokens
        input_cost = (self.input_tokens / 1_000_000) * 0.10
        output_cost = (self.output_tokens / 1_000_000) * 0.40
        return input_cost + output_cost


class GeminiClient:
    """
    Unified Gemini client with retry logic, rate limiting,
    and automatic token/cost tracking.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._configure()
        self._last_request_time = 0.0
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _configure(self) -> None:
        """Configure the Gemini API."""
        if not self.settings.is_gemini_configured:
            logger.warning("Gemini API key not configured.")
            return
        genai.configure(api_key=self.settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            self.settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=self.settings.gemini_temperature,
                max_output_tokens=self.settings.gemini_max_output_tokens,
            ),
        )

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 60.0 / self.settings.gemini_rpm_limit  # ~4s for 15 RPM
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        expect_json: bool = False,
        max_retries: int = 3,
    ) -> LLMResponse:
        """
        Send a prompt to Gemini and return a structured response.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction prefix.
            expect_json: If True, attempt to parse JSON from response.
            max_retries: Number of retries on failure.

        Returns:
            LLMResponse with text, token counts, and timing.
        """
        if not self.settings.is_gemini_configured:
            return LLMResponse(
                text="",
                success=False,
                error="Gemini API key not configured. Set GEMINI_API_KEY in .env",
            )

        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                start = time.time()

                response = self._model.generate_content(full_prompt)
                latency = (time.time() - start) * 1000

                self._last_request_time = time.time()
                self._request_count += 1

                # Extract token counts from usage metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                text = response.text.strip() if response.text else ""

                # If expecting JSON, try to clean markdown fences
                if expect_json and text:
                    text = self._clean_json_response(text)

                return LLMResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency,
                    model=self.settings.gemini_model,
                    success=True,
                )

            except Exception as e:
                logger.warning(
                    f"Gemini request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(self.settings.retry_delay * (attempt + 1))
                else:
                    return LLMResponse(
                        text="",
                        success=False,
                        error=f"All {max_retries} attempts failed: {str(e)}",
                    )

        # Should not reach here, but just in case
        return LLMResponse(text="", success=False, error="Unknown error")

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        max_retries: int = 3,
    ) -> tuple[Optional[Any], LLMResponse]:
        """
        Generate and parse JSON from Gemini.

        Returns:
            Tuple of (parsed_json_or_None, LLMResponse).
        """
        response = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            expect_json=True,
            max_retries=max_retries,
        )

        if not response.success:
            return None, response

        try:
            parsed = json.loads(response.text)
            return parsed, response
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw text: {response.text[:200]}")
            response.success = False
            response.error = f"Invalid JSON: {str(e)}"
            return None, response

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Remove markdown code fences from JSON response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @property
    def total_tokens_used(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self._total_input_tokens / 1_000_000) * 0.10
        output_cost = (self._total_output_tokens / 1_000_000) * 0.40
        return input_cost + output_cost

    @property
    def request_count(self) -> int:
        return self._request_count

    def reset_counters(self) -> None:
        """Reset usage counters (useful between experiments)."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0
