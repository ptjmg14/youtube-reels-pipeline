from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Settings
from .media import ffmpeg_executable
from .models import VideoAsset


class DownloadError(RuntimeError):
    pass


_SOURCE_SUFFIXES = {".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".opus", ".mka", ".ogg"}


def probe_video_id(url: str) -> str:
    """Resolve an ID without downloading; used to stop duplicate work early."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as error:  # pragma: no cover
        raise DownloadError("Install base dependencies: pip install -e .") from error
    with YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
        return str(ydl.extract_info(url, download=False)["id"])


def download(url: str, settings: Settings) -> VideoAsset:
    """Download a source video and any manually supplied or auto-generated subtitles."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as error:  # pragma: no cover - dependency guard
        raise DownloadError("Install the base dependency: pip install -e .") from error

    probe_options = {"quiet": True, "skip_download": True, "noplaylist": True}
    with YoutubeDL(probe_options) as ydl:
        info = ydl.extract_info(url, download=False)
    video_id = str(info["id"])
    output_dir = settings.video_dir(video_id)
    subtitle = next(iter(sorted(output_dir.glob("original*.vtt"))), None)
    existing = sorted(
        file
        for file in output_dir.glob("original.*")
        if file.suffix.lower() in _SOURCE_SUFFIXES
    )
    audio = output_dir / "audio.mp3"
    source = existing[0] if existing else None
    if source is None:
        with YoutubeDL(_subtitle_options(output_dir)) as ydl:
            ydl.extract_info(url, download=True)
        sources = sorted(output_dir.glob("original.*"))
        source = next(
            (file for file in sources if file.suffix.lower() in _SOURCE_SUFFIXES), None
        )
        if source is None:
            raise DownloadError("yt-dlp finished without producing an audio file.")
        subtitle = next(iter(sorted(output_dir.glob("original*.vtt"))), None)
    elif subtitle is None:
        subtitle = _fetch_subtitles_only(output_dir, info)
    source_name = _channel_name(info)
    if not audio.exists():
        _extract_audio(source, audio)
    return VideoAsset(
        video_id,
        str(info.get("title", video_id)),
        url,
        audio,
        None,
        subtitle,
        source_name,
    )


def _channel_name(info: dict) -> str | None:
    """Best-effort display name of the source channel, used for citation."""
    for key in ("channel", "uploader", "creator", "channel_id"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _subtitle_options(output_dir: Path) -> dict:
    return {
        "format": "ba/b",
        "outtmpl": str(output_dir / "original.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["pt", "pt-PT", "en", "en-US", "zh", "zh-CN", "zh-TW", "zh-Hant"],
        "subtitlesformat": "vtt/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": str(ffmpeg_executable()),
    }


def _fetch_subtitles_only(output_dir: Path, info: dict) -> Path | None:
    """Video already cached: fetch only subtitles (fast) to avoid local whisper."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:  # pragma: no cover - dependency guard
        return None
    options = _subtitle_options(output_dir) | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.extract_info(info["webpage_url"], download=True)
    except Exception:  # noqa: BLE001 - fallback para whisper já está garantido
        return None
    return next(iter(sorted(output_dir.glob("original*.vtt"))), None)


def _extract_audio(source: Path, audio: Path) -> None:
    command = [ffmpeg_executable(), "-y", "-i", str(source), "-vn", "-ac", "1", "-b:a", "64k", str(audio)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise DownloadError("FFmpeg was not found. Install it and add it to the PATH.") from error
    except subprocess.CalledProcessError as error:
        raise DownloadError(f"Failed to extract audio: {error.stderr[-800:]}") from error
