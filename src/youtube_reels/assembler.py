from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .localization import CardStrings, get_card_strings
from .media import ffmpeg_executable
from .models import RenderedScript
from .paths import FONT_BOLD, FONT_REGULAR

WIDTH, HEIGHT = 1080, 1920
MARGIN_X = 96
SAFE_TEXT_WIDTH = 860
_BG_TOP = "#101736"
_BG_BOTTOM = "#1B2A5E"
_CARD = "#0B1026"
_CHIP = "#2A3557"
_TEXT = "#FFFFFF"
_SUBTEXT = "#9FB0E0"
_ACCENT = "#38E1E1"


class AssembleError(RuntimeError):
    pass


def assemble_short(
    video_dir: Path,
    script: RenderedScript,
    narration_path: Path,
    language: str | None = None,
) -> Path:
    """Compose an original 9:16 short: text cards + regenerated chart + TTS audio."""
    video_dir.mkdir(parents=True, exist_ok=True)
    duration = _audio_duration(narration_path)
    durations = _split_durations(duration, script.chart is not None)
    chart_path = _render_script_chart(video_dir, script)
    cards = get_card_strings(language)
    frames = [
        _render_title_card(video_dir / "frame1.png", script, cards),
        _render_content_card(video_dir / "frame2.png", script, chart_path, cards),
        _render_cta_card(video_dir / "frame3.png", script, cards),
    ]
    video_path = video_dir / f"{script.number:02d}-{_slug(script.title)}.mp4"
    _compose(video_path, frames, narration_path, durations)
    return video_path


def _split_durations(duration: float, has_chart: bool) -> list[float]:
    if has_chart:
        title_share, content_share = 0.28, 0.52
    else:
        title_share, content_share = 0.35, 0.45
    return [
        max(2.5, duration * title_share),
        max(3.0, duration * content_share),
        max(2.0, duration * (1.0 - title_share - content_share)),
    ]


def _render_script_chart(video_dir: Path, script: RenderedScript) -> Path:
    if not script.chart:
        return Path()
    from .charts import render_chart

    chart_path = video_dir / f"chart-{script.number:02d}.png"
    return render_chart(script.chart, chart_path)


def _render_title_card(path: Path, script: RenderedScript, cards: CardStrings) -> Path:
    image = _background()
    draw = ImageDraw.Draw(image)
    bold = ImageFont.truetype(str(FONT_BOLD), 64)
    regular = ImageFont.truetype(str(FONT_REGULAR), 40)

    _pill(draw, cards.top_pill, _CHIP, MARGIN_X, 180, regular)
    title = _wrap(script.title, bold, SAFE_TEXT_WIDTH)
    y = _draw_block(draw, title, bold, 480, _TEXT, x=MARGIN_X)

    short = _wrap(script.hook, regular, SAFE_TEXT_WIDTH)
    _draw_block(draw, short, regular, y + 80, _SUBTEXT, x=MARGIN_X)
    cite = ImageFont.truetype(str(FONT_REGULAR), 38)
    # Safe zone: keep citation above bottom media player controls and scrubber
    draw.text((MARGIN_X, HEIGHT - 380), cards.source_label, font=cite, fill=_SUBTEXT)
    draw.text((MARGIN_X, HEIGHT - 320), script.citation, font=cite, fill=_ACCENT)
    image.save(path)
    return path


def _render_content_card(
    path: Path, script: RenderedScript, chart_path: Path, cards: CardStrings
) -> Path:
    image = _background()
    draw = ImageDraw.Draw(image)
    bold = ImageFont.truetype(str(FONT_BOLD), 74)
    regular = ImageFont.truetype(str(FONT_REGULAR), 44)

    if script.chart and chart_path.exists():
        chart = Image.open(chart_path).convert("RGBA")
        chart_width = SAFE_TEXT_WIDTH
        chart_height = round(chart.height * chart_width / chart.width)
        chart = chart.resize((chart_width, chart_height), Image.LANCZOS)
        chart_x = (WIDTH - chart_width) // 2
        image.paste(chart, (chart_x, 520), chart)
        label = f"{cards.chart_label_prefix}{script.chart.title}"
        draw.text((MARGIN_X, 420), label, font=regular, fill=_SUBTEXT)
        y = HEIGHT - 340
    else:
        hook = _wrap(script.hook, bold, SAFE_TEXT_WIDTH)
        _draw_block(draw, hook, bold, 420, _TEXT, x=MARGIN_X)
        y = HEIGHT - 340

    citations = _wrap(script.citation, regular, SAFE_TEXT_WIDTH)
    _draw_block(draw, citations, regular, y, _SUBTEXT, x=MARGIN_X)
    image.save(path)
    return path


def _render_cta_card(path: Path, script: RenderedScript, cards: CardStrings) -> Path:
    image = _background()
    draw = ImageDraw.Draw(image)
    bold = ImageFont.truetype(str(FONT_BOLD), 96)
    sub = ImageFont.truetype(str(FONT_BOLD), 64)
    regular = ImageFont.truetype(str(FONT_REGULAR), 40)

    _draw_block(draw, _wrap(cards.cta_question, bold, SAFE_TEXT_WIDTH), bold, 560, _TEXT, x=MARGIN_X)
    y = _draw_block(draw, _wrap(cards.cta_sub, sub, SAFE_TEXT_WIDTH), sub, 800, _ACCENT, x=MARGIN_X)
    citation = _wrap(script.citation, regular, SAFE_TEXT_WIDTH)
    y = _draw_block(draw, citation, regular, y + 140, _SUBTEXT, x=MARGIN_X)
    draw.text((MARGIN_X, HEIGHT - 340), cards.ai_footer, font=regular, fill=_SUBTEXT)
    image.save(path)
    return path


def _background() -> Image.Image:
    top = _hex(_BG_TOP)
    bottom = _hex(_BG_BOTTOM)
    image = Image.new("RGB", (WIDTH, HEIGHT), top)
    draw = ImageDraw.Draw(image)
    steps = 48
    for index in range(steps):
        ratio = index / (steps - 1)
        color = tuple(
            round(top[channel] * (1 - ratio) + bottom[channel] * ratio)
            for channel in range(3)
        )
        y0 = round(HEIGHT * index / steps)
        y1 = round(HEIGHT * (index + 1) / steps)
        draw.line([(0, y0), (WIDTH, y0)], fill=color, width=y1 - y0 + 1)
    return image


def _hex(value: str) -> tuple[int, int, int]:
    clean = value.lstrip("#")
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def _pill(
    draw: ImageDraw.ImageDraw,
    text: str,
    fill: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    padding = 24
    width = draw.textlength(text, font=font)
    draw.rounded_rectangle(
        (x - padding, y - 14, x + width + padding, y + font.size + 14),
        radius=int(font.size / 2),
        fill=fill,
    )
    draw.text((x, y), text, font=font, fill=_TEXT)


def _draw_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    y: int,
    color: str,
    x: int = MARGIN_X,
) -> int:
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += int(font.size * 1.4)
    return y


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wraps text intelligently respecting margins, natural punctuation boundaries,
    spaces in European languages, and Kinsoku Shori rules in CJK scripts."""
    lines: list[str] = []
    no_start = set("，、。！？；：!?,;:)）]】}'\"”’»")

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Split into tokens ending in punctuation or whitespace
        tokens = re.findall(r"[^，、。！？；：!?,;:\s]+[，、。！？；：!?,;:]*|\s+", paragraph)
        if not tokens:
            tokens = [paragraph]

        units: list[str] = []
        for token in tokens:
            if not token.strip():
                continue
            if font.getbbox(token)[2] > max_width:
                units.extend(list(token))
            else:
                units.append(token)

        current = ""
        for unit in units:
            sep = (
                " "
                if (
                    " " in paragraph
                    and current
                    and not current.endswith(
                        (" ", "，", "、", "。", "！", "？", "；", "：", "-", "—")
                    )
                )
                else ""
            )
            candidate = f"{current}{sep}{unit}" if current else unit
            if font.getbbox(candidate)[2] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = unit
        if current:
            lines.append(current)

    # Kinsoku Shori post-processing: avoid starting any line with closing punctuation
    refined: list[str] = []
    for line in lines:
        if refined and line and line[0] in no_start:
            punct = ""
            while line and line[0] in no_start:
                punct += line[0]
                line = line[1:].lstrip()
            refined[-1] += punct
            if line:
                refined.append(line)
        else:
            refined.append(line)

    return refined or [text]


def _compose(
    video_path: Path,
    frames: list[Path],
    narration_path: Path,
    durations: list[float],
) -> None:
    total = max(2.0, sum(durations))
    inputs: list[str] = []
    for path, seconds in zip(frames, durations):
        inputs.extend(
            ["-loop", "1", "-framerate", "30", "-t", f"{seconds:.3f}", "-i", str(path)]
        )
    inputs.extend(["-i", str(narration_path)])
    filter_v = (
        "".join(f"[{index}:v]format=yuv420p[v{index}];" for index in range(3))
        + "[v0][v1][v2]concat=n=3:v=1:a=0,"
        + f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0.0, total - 0.6):.2f}:d=0.6[v]"
    )
    command = [
        ffmpeg_executable(),
        "-y",
        *inputs,
        "-filter_complex",
        filter_v,
        "-map",
        "[v]",
        "-map",
        "3:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError as error:
        raise AssembleError("FFmpeg was not found. Install it and add it to the PATH.") from error
    except subprocess.CalledProcessError as error:
        raise AssembleError(error.stderr[-1200:]) from error


def _audio_duration(path: Path) -> float:
    try:
        import av
    except ImportError:  # pragma: no cover - PyAV is available with Chroma.
        return 30.0
    with av.open(str(path)) as container:
        duration = container.duration
    if duration and duration > 0:
        return float(duration) / av.time_base
    return 30.0


def _slug(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return (clean or "clip")[:45]