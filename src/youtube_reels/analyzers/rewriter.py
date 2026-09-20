from __future__ import annotations

import json
import logging
from typing import Any

from ..localization import citation_text, script_length_guideline
from ..models import ChartSpec, RenderedScript, SegmentWindow

logger = logging.getLogger(__name__)

SUPPORTED_CHARTS = {"bar", "line"}


class RewriteError(RuntimeError):
    pass


class Rewriter:
    """Mandatory rewriting step: turns filtered transcript windows into original
    short-video scripts via an LLM.

    Copyright rules are enforced in the prompt: the narration must paraphrase
    facts in a different sentence structure, begin with a source citation, and
    only ever hand back structured data (never media) for chart regeneration.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        source_name: str,
        language: str,
        groq_api_key: str | None = None,
        groq_models: list[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.source_name = source_name
        self.language = language
        self.groq_api_key = groq_api_key
        self.groq_models = groq_models

    def rewrite(
        self, 
        windows: list[SegmentWindow], 
        source_title: str, 
        max_clips: int,
        feedback: str | None = None,
        previous_script: str | None = None,
    ) -> list[RenderedScript]:
        if not windows:
            return []

        from ..llm_utils import generate_with_fallback

        prompt = _rewrite_prompt(
            windows,
            source_title,
            self.source_name,
            self.language,
            max_clips,
            feedback,
            previous_script,
        )
        try:
            response_text = generate_with_fallback(
                self.api_key,
                prompt,
                model=self.model,
                groq_api_key=self.groq_api_key,
                groq_models=self.groq_models,
            )
        except RuntimeError as error:
            raise RewriteError(str(error)) from error
        scripts = _parse_scripts(response_text, max_clips, windows)
        if not scripts:
            raise RewriteError("The rewriter selected no scripts.")
        return scripts


def _rewrite_prompt(
    windows: list[SegmentWindow],
    source_title: str,
    source_name: str,
    language: str,
    max_clips: int,
    feedback: str | None = None,
    previous_script: str | None = None,
) -> str:
    excerpts = "\n".join(
        f"{index}. [{window.start:.1f}s–{window.end:.1f}s] {window.text}"
        for index, window in enumerate(windows, start=1)
    )
    
    feedback_section = ""
    if feedback:
        prev_chunk = f'\nPrevious rejected narration:\n"{previous_script}"\n' if previous_script else ""
        feedback_section = f"""
IMPORTANT: Your previous attempt was REJECTED by the quality guardrail with this feedback:
"{feedback}"{prev_chunk}
Please correct these issues in your new response.
"""

    example_cite = citation_text(language, source_name)
    length_rule = script_length_guideline(language)

    return f"""You are a short-form video scriptwriter for social media.
Source: "{source_title}", aired by {source_name}. The material is copyright-protected.
{feedback_section}
Below is the full transcript as numbered candidate blocks — they are only
reference material so you can see every moment in order. Pick the best {max_clips} moments
for engaging vertical shorts and rewrite ONLY those. You are free to choose any contiguous
timeline moment (20–60 seconds), including across block boundaries; do NOT feel limited to
a single numbered block.

MANDATORY RULES (the user may break these — you must not):
1. ORIGINAL REWRITE: for each picked moment, write ONE narration script in {language} that conveys the SAME facts and data in your own words.
2. DIFFERENT STRUCTURE: the sentence structure and the order of ideas must be clearly different from the source moment. Do NOT copy verbatim sentences or word order. Facts, numbers, names, and statistics should be preserved accurately.
3. CITATION: every narration must open with a clear source attribution in {language}, e.g. "{example_cite}".
4. CHART DATA: if the chosen moment contains comparable quantitative data (numbers, shares, values over time), extract it into the "chart" field — {{
  "title": "short chart title in {language}",
  "kind": "bar" or "line",
  "x_labels": ["labels", "of", "categories"],
  "values": [numbers, one per label],
  "unit": "unit when applicable, otherwise empty string"
}}. Only include numbers that appear literally (or are certainly derivable) from the source; NEVER invent values. If there is no chart data, chart is null.
5. SEPARATE CITATION: besides opening the narration with the citation, return the same citation in the "citation" field.
6. LENGTH: each narration must be {length_rule}.
7. TIMING: each clip must include "start" and "end" (seconds in the source video) of the moment you rewrote. Prefer round numbers. No two clips may be the same moment.

Reply STRICTLY in JSON, no Markdown:
[
  {{"start": 12, "end": 45, "hook": "one-line hook", "title": "clip title", "narration": "full script including the citation", "citation": "exact citation used at the start of the narration", "reason": "why this moment", "chart": <object or null>}}
]

Transcript blocks (reference material, in chronological order):
{excerpts}"""


def _parse_scripts(
    raw: str, max_clips: int, windows: list[SegmentWindow]
) -> list[RenderedScript]:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload: list[dict[str, Any]] = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RewriteError(f"Rewriter response is not valid JSON: {error}") from error
    if not payload:
        raise RewriteError("The rewriter returned no scripts.")
    if len(payload) > max_clips:
        raise RewriteError(
            f"O rewriter devolveu {len(payload)} scripts para um máximo de {max_clips}."
        )
    scripts: list[RenderedScript] = []
    for number, item in enumerate(payload, start=1):
        narration = str(item.get("narration", "")).strip()
        if not narration:
            raise RewriteError("The rewriter returned a script without narration.")
        title = str(item.get("title", narration[:24])).strip() or "Original clip"
        hook = str(item.get("hook", title)).strip() or title
        reason = str(item.get("reason", "")).strip()
        citation = str(item.get("citation", "")).strip()
        chart = _parse_chart(item.get("chart"))
        start, end = _clip_timing(item, windows)
        source_text = _source_text(start, end, windows)
        scripts.append(
            RenderedScript(
                number=number, hook=hook, title=title, narration=narration,
                citation=citation, reason=reason, chart=chart,
                start=start, end=end, source_text=source_text,
            )
        )
    return scripts


def _clip_timing(item: dict[str, Any], windows: list[SegmentWindow]) -> tuple[float, float]:
    try:
        start = float(item.get("start", windows[0].start if windows else 0) or 0)
        end = float(item.get("end", windows[-1].end if windows else 0) or 0)
    except (TypeError, ValueError):
        start = windows[0].start if windows else 0
        end = windows[-1].end if windows else 0
    start = max(start, 0)
    if end <= start:
        start = max(window.start for window in windows) if windows else 0
        end = max(window.end for window in windows) if windows else 1
    return start, end


def _source_text(start: float, end: float, windows: list[SegmentWindow]) -> str:
    """Reconstruct the source wording for a clip time range.

    Used as retry material if the guardrail rejects the script; the narration
    never copies it verbatim.
    """
    matching = [
        window
        for window in windows
        if window.start < end and window.end > start
    ]
    if not matching:
        return ""
    return " ".join(window.text for window in matching)


def _parse_chart(raw: Any) -> ChartSpec | None:
    if not raw or not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", "")).strip().lower()
    if kind not in SUPPORTED_CHARTS:
        raise RewriteError(f"Unknown chart type: {kind}")
    labels = [str(label) for label in raw.get("x_labels", []) or []]
    try:
        values = [float(value) for value in raw.get("values", []) or []]
    except (TypeError, ValueError) as error:
        raise RewriteError("Chart values are not numeric.") from error
    if not labels or len(labels) != len(values):
        raise RewriteError("Chart needs one label per value.")
    if min(values) < 0:
        raise RewriteError("Negative values are not supported in this demo.")
    return ChartSpec(
        title=str(raw.get("title", "Source data")).strip() or "Source data",
        kind=kind,
        x_labels=labels,
        values=values,
        unit=str(raw.get("unit", "")).strip(),
    )