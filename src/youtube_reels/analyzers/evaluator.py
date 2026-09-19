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

    def evaluate(self, script: RenderedScript) -> bool:
        """Evaluates a script. Returns True if passed, raises EvalError if failed."""
        from ..llm_utils import generate_with_fallback
        
        prompt = self._eval_prompt(script)
        
        try:
            response_text = generate_with_fallback(
                self.api_key, prompt, model=self.model
            )
        except RuntimeError as error:
            raise EvalError(str(error)) from error
        
        result = self._parse_evaluation(response_text)
        if result.get("status") == "PASS":
            return True
        else:
            raise EvalError(f"Evaluation failed: {result.get('reason', 'Unknown reason')}")

    def _eval_prompt(self, script: RenderedScript) -> str:
        return f"""You are a content quality and legal compliance evaluator for a media company.
Evaluate the following script based on these rules:
1. PARAPHRASE: The narration must NOT copy the original source text word-for-word. It must paraphrase the facts and ideas using completely different sentence structures and vocabulary.
2. CITATION: The narration MUST begin with a source citation.
3. CHART DATA: If the script includes a chart, ensure it is derived from the narration and not fake.

Original Source Text:
"{script.source_text}"

Script to evaluate:
Narration: "{script.narration}"
Citation: "{script.citation}"
Chart: {json.dumps(script.chart.to_dict()) if script.chart else "None"}

Reply STRICTLY in JSON:
{{
  "status": "PASS" or "FAIL",
  "reason": "Why it failed, or 'None' if passed"
}}
"""

    def _parse_evaluation(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise EvalError(f"Evaluation response is not valid JSON: {error}")
