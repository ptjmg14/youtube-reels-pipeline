from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
FONT_BOLD = FONTS_DIR / "NotoSansCJKtc-Bold.otf"
FONT_REGULAR = FONTS_DIR / "NotoSansCJKtc-Regular.otf"