from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    title: str
    source_url: str
    audio_path: Path
    video_path: Path | None = None
    subtitle_path: Path | None = None


@dataclass(frozen=True)
class SimilarVideo:
    """An already-indexed video whose transcript overlaps the new one.

    ``score`` is the Jaccard overlap of the normalized character n-grams of
    both transcripts, in ``[0, 1]``. Used to stop duplicate content before the
    paid rewrite and the expensive render.
    """

    video_id: str
    title: str
    score: float


@dataclass(frozen=True)
class SegmentWindow:
    """A coherent transcript slice selected by the cost-free pre-filter.

    It is reference material only — never used verbatim in the output.
    """

    start: float
    end: float
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ChartSpec:
    """Structured data extracted from the source so the chart can be rebuilt.

    Charts are regenerated programmatically (matplotlib) — never screenshotted.
    """

    title: str
    kind: str
    x_labels: list[str]
    values: list[float]
    unit: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RenderedScript:
    """An original, paraphrased short-video script. The narration is spoken
    by TTS; no source wording or sentence structure is reproduced verbatim."""

    number: int
    hook: str
    title: str
    narration: str
    citation: str
    reason: str = ""
    chart: ChartSpec | None = None
    start: float | None = None
    end: float | None = None
    source_text: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["chart"] = self.chart.to_dict() if self.chart else None
        return data


@dataclass(frozen=True)
class AnalysisResult:
    """Persisted analysis: reference transcript plus rewritten scripts."""

    transcript: list[TranscriptSegment]
    scripts: list[RenderedScript]

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            "transcript": [item.to_dict() for item in self.transcript],
            "scripts": [item.to_dict() for item in self.scripts],
        }