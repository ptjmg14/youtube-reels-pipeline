from __future__ import annotations

import json
import logging
from typing import Any

from ..models import RenderedScript

logger = logging.getLogger(__name__)


class EvalError(RuntimeError):
    pass


class Evaluator:
    """Evaluates generated scripts to ensure they meet quality and copyright standards."""

    def __init__(self, api_key: str, model: str, language: str) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language

    def evaluate(self, script: RenderedScript, source_name: str) -> bool:
        """Evaluates a script. Returns True if passed, raises EvalError if failed."""
        from ..llm_utils import generate_with_fallback

        prompt = self._eval_prompt(script, source_name)

        try:
            response_text = generate_with_fallback(
                self.api_key, prompt, model=self.model
            )
        except RuntimeError as error:
            raise EvalError(str(error)) from error

        result = self._parse_evaluation(response_text)
        status = str(result.get("status", "")).strip().upper()
        if status == "PASS":
            return True

        reason = result.get("reason") or "Quality evaluation failed"
        raise EvalError(f"Evaluation failed: {reason}")

    def _eval_prompt(self, script: RenderedScript, source_name: str) -> str:
        chart_str = json.dumps(script.chart.to_dict()) if script.chart else "None"
        return f"""You are a content quality and legal compliance evaluator for a media company.
Target Language: {self.language}

Evaluate the following script based on these rules:
1. PARAPHRASE: The narration must NOT copy the original source text verbatim. It must explain the information with original sentence structure and phrasing. Specific facts, proper names, entities, numbers, percentages, and dates SHOULD be accurately preserved.
2. CITATION: The narration MUST begin with a clear attribution/credit to the Official Source Name provided below.
3. CHART DATA: If the script includes a chart, ensure its categories and numbers are factually grounded in the Original Source Text (not invented). The narration itself is brief and is NOT required to recite every single data point from the chart.

Official Source Name:
"{source_name}"

Original Source Text:
"{script.source_text}"

Script to evaluate:
Narration: "{script.narration}"
Citation: "{script.citation}"
Chart: {chart_str}

Reply STRICTLY in JSON:
{{
  "status": "PASS" or "FAIL",
  "reason": "Clear explanation of what failed, or 'None' if passed"
}}
"""

    def _parse_evaluation(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise EvalError(f"Evaluation response is not valid JSON: {error} (raw: {raw[:200]})") from error
