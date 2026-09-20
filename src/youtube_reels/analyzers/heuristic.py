"""Cost-free heuristic pre-filter for transcript windowing.

Strategy (all local, zero API cost):
1. **Sliding-window with 50 % step overlap** so no good content falls between
   two non-overlapping greedy windows (the old algorithm's main flaw).
2. **Information-density scoring** — regex signals for numbers/percentages,
   named entities (capitalised words, common nouns), contrast/surprise markers,
   and explicit question hooks.  Weights are language-agnostic.
3. **Natural sentence-boundary alignment** — each candidate window is trimmed
   so it starts and ends at a sentence boundary (period / question / exclamation
   / CJK full-stop variants).
4. **Top-N selection** — the scored windows are ranked and only the top
   ``top_n`` are forwarded to the LLM, preventing the rewrite prompt from
   ballooning with low-signal material.

No external libraries, no ML models — purely stdlib ``re``.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..models import SegmentWindow, TranscriptSegment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_WINDOW_SEC: float = 15.0
_MAX_WINDOW_SEC: float = 55.0
_STEP_RATIO: float = 0.5          # overlap: next window starts at 50 % of span
_DEFAULT_TOP_N: int = 30          # cap before sending to LLM

# ---------------------------------------------------------------------------
# Scoring patterns (language-agnostic, work for CJK and Latin scripts)
# ---------------------------------------------------------------------------

_RE_NUMBERS = re.compile(
    r"""
    (?:
        \d[\d,.\uff0c\uff0e]*\s*%        # percentages: 30%, 30.5%
      | \d[\d,.\uff0c\uff0e]*            # plain numbers: 1,200 / 1.5萬
      | [\u516c\u5143\u767e\u5343\u842c\u5104\u5146]  # CJK numerics 元百千萬億兆
    )
    """,
    re.VERBOSE,
)
_RE_CONTRAST = re.compile(
    r"""
    (?:
        but\b|however\b|yet\b|despite\b|although\b|nevertheless\b
      | 但(?:是)?|然而|卻|不過|雖然|儘管|反而|可是
      | mas\b|porém\b|contudo\b|entretanto\b|embora\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
_RE_QUESTION = re.compile(
    r"""
    (?:
        \?                        # literal question mark
      | \uff1f                   # full-width ？
      | what\b|why\b|how\b|when\b|which\b|who\b|where\b
      | 什麼|為什麼|怎麼|幾時|哪|誰|如何
      | como\b|por\s*que\b|onde\b|quando\b|quem\b|qual\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
_RE_SURPRISE = re.compile(
    r"""
    (?:
        \!                        # exclamation
      | \uff01                   # full-width ！
      | actually\b|shocking\b|surprisingly\b|incredible\b|unbelievable\b
      | 竟(?:然)?|居然|沒想到|驚|暴增|暴跌|大幅|翻倍
      | surpreend|incrív|chocant
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
# Capitalised words as a rough named-entity proxy (only meaningful in Latin)
_RE_CAPS = re.compile(r"\b[A-Z][a-z]{1,}\b")

# Sentence boundary markers (end of sentence)
_RE_SENT_END = re.compile(r"[.!?\u3002\uff01\uff1f]")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class HeuristicPrefilter:
    """Cost-free structural pass: scores and ranks transcript windows as
    reference material for the LLM rewrite step.

    Improvements over the original greedy non-overlapping packer:
    - Sliding window with 50 % step keeps boundary content.
    - Information-density scoring surfaces the most quotable moments.
    - Top-N cap prevents the rewrite prompt from inflating with dull windows.
    """

    def suggest(
        self,
        transcript: list[TranscriptSegment],
        max_clips: int | None = None,
        top_n: int = _DEFAULT_TOP_N,
    ) -> list[SegmentWindow]:
        if not transcript:
            return []

        windows = self._sliding_windows(transcript)
        scored = [replace(w, score=self._score(w.text)) for w in windows]
        scored.sort(key=lambda w: w.score, reverse=True)

        # Honour max_clips as a hard cap when provided (allow some headroom for
        # the LLM to have choices), but never exceed top_n.
        limit = top_n
        if max_clips is not None:
            limit = min(top_n, max(max_clips * 3, max_clips + 5))

        selected = scored[:limit]
        # Restore chronological order so the rewrite prompt reads naturally
        selected.sort(key=lambda w: w.start)
        return selected

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def _sliding_windows(
        self, transcript: list[TranscriptSegment]
    ) -> list[SegmentWindow]:
        """Generate overlapping windows with a 50 % step, aligned to sentence
        boundaries where possible."""
        if not transcript:
            return []

        total_end = transcript[-1].end
        windows: list[SegmentWindow] = []
        seen: set[tuple[float, float]] = set()

        i = 0
        while i < len(transcript):
            seg_start = transcript[i]
            win_start = seg_start.start

            # Collect segments until window exceeds max duration
            j = i
            texts: list[str] = []
            while j < len(transcript):
                seg = transcript[j]
                if seg.end - win_start > _MAX_WINDOW_SEC:
                    break
                texts.append(seg.text)
                j += 1

            if not texts:
                i += 1
                continue

            win_end = transcript[j - 1].end

            # Skip too-short windows (unless we are at the very end of the video)
            duration = win_end - win_start
            if duration < _MIN_WINDOW_SEC and win_end < total_end - 1:
                i += 1
                continue

            # Trim to sentence boundary (optional, best-effort)
            win_text, win_start, win_end = self._trim_to_sentence(
                transcript, i, j - 1, texts
            )

            key = (round(win_start, 1), round(win_end, 1))
            if key not in seen:
                seen.add(key)
                windows.append(SegmentWindow(start=win_start, end=win_end, text=win_text))

            # Advance by ~50 % of the window span
            step_target = win_start + max(duration * _STEP_RATIO, _MIN_WINDOW_SEC)
            while i < len(transcript) and transcript[i].start < step_target:
                i += 1

        return windows

    @staticmethod
    def _trim_to_sentence(
        transcript: list[TranscriptSegment],
        i_start: int,
        i_end: int,
        texts: list[str],
    ) -> tuple[str, float, float]:
        """Try to start/end at a sentence boundary; falls back to raw window."""
        joined = " ".join(texts)

        # Find the first sentence-ending punctuation near the start
        # (skip the very first segment so we don't collapse the window)
        first_break = _RE_SENT_END.search(joined, 1)
        if first_break and first_break.start() < len(joined) // 3:
            joined = joined[first_break.start() + 1:].lstrip()

        # Find the last sentence-ending punctuation
        last_break = None
        for m in _RE_SENT_END.finditer(joined):
            last_break = m
        if last_break and last_break.start() > len(joined) * 2 // 3:
            joined = joined[: last_break.start() + 1]

        win_start = transcript[i_start].start
        win_end = transcript[i_end].end
        return joined.strip() or " ".join(texts), win_start, win_end

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score(text: str) -> float:
        """Return a float information-density score (higher = more quotable).

        Signals (all weighted):
        - numbers / percentages / CJK numerics  → high factual density
        - contrast markers                        → narrative tension
        - question words / marks                  → hook potential
        - surprise/exclamation markers            → engagement signal
        - capitalised words (Latin NE proxy)      → named entities
        - word/character count                    → length normalisation
        """
        if not text:
            return 0.0

        n_numbers = len(_RE_NUMBERS.findall(text))
        n_contrast = len(_RE_CONTRAST.findall(text))
        n_question = len(_RE_QUESTION.findall(text))
        n_surprise = len(_RE_SURPRISE.findall(text))
        n_caps = len(_RE_CAPS.findall(text))

        # Normalise by text length so short windows don't dominate
        length = max(len(text), 1)
        norm = 100.0 / length

        score = (
            n_numbers  * 3.0 * norm
            + n_contrast * 2.5 * norm
            + n_question * 2.0 * norm
            + n_surprise * 2.0 * norm
            + n_caps     * 1.0 * norm
        )
        return round(score, 4)