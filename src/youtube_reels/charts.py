from __future__ import annotations

from pathlib import Path

from .models import ChartSpec
from .paths import FONT_BOLD, FONT_REGULAR

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
except ImportError as error:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Install local chart generation: pip install -e '.[local]'"
    ) from error


_BG = "#0B1026"
_ACCENT = "#4C8DFF"
_CYAN = "#38E1E1"


def _register_fonts() -> None:
    for path in (FONT_BOLD, FONT_REGULAR):
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except RuntimeError:
                pass


_register_fonts()


def render_chart(
    spec: ChartSpec,
    output_path: Path,
    width: int = 1080,
    height: int = 800,
    dpi: int = 144,
) -> Path:
    """Render a ChartSpec into a PNG. The chart is built from the extracted
    data points only — never a screenshot of the source material."""
    _register_fonts()
    title_font = FontProperties(fname=str(FONT_BOLD))
    label_font = FontProperties(fname=str(FONT_REGULAR))
    figure = matplotlib.figure.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.patch.set_facecolor("none")
    axis = figure.add_subplot(111)
    axis.set_facecolor(_BG)

    colors = [_CYAN if (value == max(spec.values)) else _ACCENT for value in spec.values]
    if spec.kind == "line":
        axis.plot(spec.x_labels, spec.values, marker="o", linewidth=3, color=_ACCENT)
        axis.fill_between(
            range(len(spec.values)), spec.values, color=_ACCENT, alpha=0.25
        )
    else:
        axis.bar(
            spec.x_labels,
            spec.values,
            color=colors,
            width=0.58,
            zorder=3,
        )

    for index, value in enumerate(spec.values):
        unit = f" {spec.unit}" if spec.unit else ""
        axis.annotate(
            f"{value:,.0f}{unit}",
            (index, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontproperties=label_font,
            fontsize=22,
            color="#FFFFFF",
        )

    axis.set_xticks(range(len(spec.x_labels)))
    axis.set_xticklabels(spec.x_labels, fontproperties=label_font, color="#B9C2E0", fontsize=20)
    axis.tick_params(axis="x", colors="#B9C2E0", length=0, pad=10)
    axis.tick_params(axis="y", colors="#B9C2E0", labelsize=20, length=0)
    axis.grid(axis="y", color="#2A3557", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color("#2A3557")
    axis.spines["bottom"].set_color("#2A3557")

    if spec.unit:
        axis.set_ylabel(spec.unit, fontproperties=label_font, color="#B9C2E0", fontsize=18)
    axis.set_title(
        spec.title,
        fontproperties=title_font,
        color="#FFFFFF",
        fontsize=30,
        pad=24,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, transparent=True, bbox_inches="tight")
    plt.close(figure)
    return output_path