from __future__ import annotations

import shutil


def ffmpeg_executable() -> str:
    """Use a system FFmpeg when available, otherwise the portable package binary."""
    if installed := shutil.which("ffmpeg"):
        return installed
    try:
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("Instale o FFmpeg portátil: python -m pip install -r requirements.txt") from error
    return imageio_ffmpeg.get_ffmpeg_exe()
