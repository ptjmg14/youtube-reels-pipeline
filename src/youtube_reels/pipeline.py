from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .analyzers.evaluator import EvalError, Evaluator
from .analyzers.heuristic import HeuristicPrefilter
from .analyzers.rewriter import RewriteError, Rewriter
from .assembler import assemble_short
from .config import Settings
from .downloader import download, probe_video_id

if TYPE_CHECKING:
    from .pinecone_store import PineconeStore

from .models import AnalysisResult, RenderedScript, SegmentWindow, TranscriptSegment, VideoAsset
from .subtitles import parse_vtt
from .transcriber import transcribe
from .tts import synthesize
from .vectordb import VectorStore, open_vector_store

logger = logging.getLogger(__name__)


class ProcessError(RuntimeError):
    pass


class AlreadyProcessed(RuntimeError):
    pass


class DuplicateContent(RuntimeError):
    pass


@contextmanager
def _timed(label: str) -> None:
    start = time.perf_counter()
    yield
    print(f"  ⏱ {label}: {time.perf_counter() - start:.1f}s")


def process(
    url: str,
    settings: Settings,
    max_clips: int,
    force_transcribe: bool = False,
    force: bool = False,
) -> list[Path]:
    with _timed("index (ChromaDB/embeddings startup)"):
        store = _try_store(settings)
    video_id = probe_video_id(url)
    if store and store.is_processed(video_id) and not force:
        raise AlreadyProcessed(
            f"Video {video_id} is already indexed. Use --force to process it again."
        )
    with _timed("download + audio extraction"):
        asset = download(url, settings)
    with _timed("transcription"):
        transcript = _get_transcript(asset, settings, force_transcribe)
    if not transcript:
        raise ProcessError("Could not obtain a transcript for this video.")

    if store and settings.dedupe_threshold > 0 and not force:
        with _timed("content dedupe (semantic recall + n-gram check)"):
            match = store.find_similar(transcript, settings.dedupe_threshold)
        if match:
            raise DuplicateContent(
                f"Duplicate content: {match.video_id} ({match.title}) is "
                f"{match.score:.0%} textually similar. Use --force to process anyway."
            )

    with _timed("pre-filter (window packing)"):
        windows = HeuristicPrefilter().suggest(transcript)
    if not windows:
        raise ProcessError("The pre-filter found no windows to rewrite.")

    with _timed(f"Gemini rewrite (select + rewrite up to {max_clips} scripts)"):
        scripts = _rewrite(asset, windows, settings, max_clips)
    with _timed("Guardrail evaluation"):
        try:
            scripts = _evaluate(scripts, windows, asset, settings)
        except (RewriteError, EvalError) as error:
            raise ProcessError(str(error)) from error
    analysis = AnalysisResult(transcript=transcript, scripts=scripts)
    _save_analysis(analysis, settings.video_dir(asset.video_id) / "analysis.json")

    clips_dir = settings.video_dir(asset.video_id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale in clips_dir.glob("*.mp4"):
        stale.unlink()
    output_paths: list[Path] = []
    for script in scripts:
        with _timed(f"TTS + assembling short {script.number}"):
            narration = synthesize(
                script.narration,
                clips_dir / f"narration-{script.number:02d}.mp3",
                settings.tts_voice,
                settings.tts_rate,
            )
            output_paths.append(assemble_short(clips_dir, script, narration))

    with _timed("ChromaDB indexing (transcript + scripts)"):
        if store:
            store.index(asset, transcript, scripts)
    return output_paths


def _rewrite(
    asset: VideoAsset, windows: list, settings: Settings, max_clips: int
) -> list[RenderedScript]:
    if not settings.gemini_api_key:
        raise ProcessError(
            "The mandatory rewrite requires GEMINI_API_KEY "
            "(free tier). Configure the .env file."
        )
    rewritter = Rewriter(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        source_name=settings.source_name,
        language=settings.output_language,
    )
    try:
        scripts = rewritter.rewrite(windows, asset.title, max_clips)
    except RewriteError as error:
        raise ProcessError(str(error)) from error
    default_citation = _citation_text(settings.output_language, settings.source_name)
    return [
        replace(script, citation=script.citation or default_citation) for script in scripts
    ]


def _evaluate(
    scripts: list[RenderedScript], 
    windows: list, 
    asset: VideoAsset, 
    settings: Settings
) -> list[RenderedScript]:
    if not settings.gemini_api_key:
        raise ProcessError("Evaluation requires GEMINI_API_KEY.")
    
    evaluator = Evaluator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        language=settings.output_language,
    )
    rewriter = Rewriter(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        source_name=settings.source_name,
        language=settings.output_language,
    )
    
    final_scripts = []
    
    for i, script in enumerate(scripts):
        success = False
        attempts = 0
        current_script = script
        
        while not success and attempts < 2:
            try:
                evaluator.evaluate(current_script)
                success = True
            except EvalError as error:
                attempts += 1
                logger.warning(f"Script {i+1} failed evaluation (attempt {attempts}/2): {error}")
                if script.source_text:
                    source = SegmentWindow(
                        start=script.start or 0.0,
                        end=script.end or 0.0,
                        text=script.source_text,
                    )
                else:
                    source = windows[i]
                new_scripts = rewriter.rewrite([source], asset.title, 1)
                current_script = replace(
                    new_scripts[0],
                    citation=script.citation,
                    start=script.start,
                    end=script.end,
                    source_text=script.source_text,
                )
        
        if success:
            final_scripts.append(current_script)
        else:
            raise ProcessError(f"Script {i+1} failed evaluation after 2 retries.")
            
    return final_scripts


def _citation_text(language: str, source_name: str) -> str:
    language = (language or "").lower()
    if any(marker in language for marker in ("english", "en", "英语", "英文")):
        return f"According to {source_name}'s reporting"
    return f"根據 {source_name} 報導"


def _get_transcript(
    asset: VideoAsset, settings: Settings, force: bool
) -> list[TranscriptSegment]:
    if asset.subtitle_path and not force:
        segments = parse_vtt(asset.subtitle_path)
        if segments:
            return segments
    source = str(asset.audio_path or asset.video_path)
    return transcribe(
        source,
        settings.whisper_model,
        transcriber=settings.transcriber,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
    )


def _save_analysis(analysis: AnalysisResult, path: Path) -> None:
    path.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _try_store(settings: Settings) -> VectorStore | PineconeStore | None:
    """Return the configured index (hosted Pinecone or local Chroma).

    The factory in :mod:`youtube_reels.vectordb` decides the backend from
    ``REELS_VECTORDB`` and prints its own reason when the index is skipped.
    """
    return open_vector_store(settings)
