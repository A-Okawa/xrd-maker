"""Colour palettes and named colours shared with the graph app (figapp).

The palette lists, their names and the way shades are sampled are the same as in
figapp's ``figcore.py``: with the same palette and the same number of series, both
apps give exactly the same colours. ``colors.csv`` uses figapp's format (one
``名前,#RRGGBB`` per line), so a colors.csv copied from the graph app's folder gives
the same named colours here.
"""

from __future__ import annotations

import re

import matplotlib
from matplotlib.colors import to_hex


PALETTES = {
    "paper": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"],
    "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "blues": "Blues", "oranges": "Oranges", "greens": "Greens", "greys": "Greys", "viridis": "viridis",
    "tab10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
}
PALETTE_NAMES = {
    "ja": {"paper": "論文用（青・朱・緑…）", "okabe_ito": "Okabe–Ito", "blues": "青の濃淡", "oranges": "朱の濃淡",
           "greens": "緑の濃淡", "greys": "灰の濃淡", "viridis": "viridis", "tab10": "matplotlib 標準"},
    "en": {"paper": "Publication (blue, vermilion, green…)", "okabe_ito": "Okabe–Ito", "blues": "Blue shades",
           "oranges": "Orange shades", "greens": "Green shades", "greys": "Grey shades", "viridis": "viridis",
           "tab10": "Matplotlib default"},
}
DEFAULT_PALETTE = "paper"
MAX_SHADE_STEPS = 12

# References start in black, then take distinct Okabe–Ito hues (used unless the palette is applied to them).
REFERENCE_COLORS = ["#000000", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2", "#4D4D4D"]

# figapp's named colours (Okabe–Ito and greys); its default colors.csv also lists 750/950/1050 °C first.
NAMED_COLORS = {"青": "#0072B2", "朱": "#D55E00", "緑": "#009E73", "赤紫": "#CC79A7", "橙": "#E69F00", "空色": "#56B4E9",
                "黄": "#F0E442", "黒": "#000000", "濃い灰": "#4D4D4D", "灰": "#999999", "赤": "#D62728", "紫": "#9467BD",
                "茶": "#8C564B"}
_EMOJI_RGB = {"🟥": (214, 40, 40), "🟧": (240, 140, 20), "🟨": (245, 215, 50), "🟩": (40, 160, 80), "🟦": (30, 110, 210),
              "🟪": (150, 80, 190), "🟫": (130, 80, 50), "⬛": (25, 25, 25), "⬜": (235, 235, 235)}


def is_shade_palette(name: str) -> bool:
    """True for the palettes sampled from a colour map (the shades and viridis)."""
    return isinstance(PALETTES.get(name), str)


def palette_colors(name: str, n: int, reverse: bool = False) -> list[str]:
    """Colours for n series, exactly as figapp gives them.

    Shades run from light to dark over 0.35–0.95 of the colour map (one series: 0.75);
    reverse=True starts with the dark one. Other palettes repeat their list, and
    reverse=True reverses the order of the colours.
    """
    n = int(n)
    if n <= 0:
        return []
    p = PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
    if isinstance(p, str):
        cmap = matplotlib.colormaps[p]
        if n == 1:
            return [to_hex(cmap(0.75)).upper()]
        cols = [to_hex(cmap(0.35 + 0.6 * i / (n - 1))).upper() for i in range(n)]
    else:
        cols = [p[i % len(p)].upper() for i in range(n)]
    return cols[::-1] if reverse else cols


def palette_sequence(name: str, n: int, reverse: bool = False, steps: int = 0) -> list[str]:
    """Colours for n items in display order.

    ``steps`` (shade palettes only) samples the shades figapp gives for that many
    series, so a figure from the graph app with a different number of series can be
    matched; with more items than steps the shades repeat.
    """
    n, steps = int(n), int(steps or 0)
    if n <= 0:
        return []
    if steps > 0 and is_shade_palette(name):
        base = palette_colors(name, min(steps, MAX_SHADE_STEPS), reverse)
        return [base[i % len(base)] for i in range(n)]
    return palette_colors(name, n, reverse)


def swatch_colors(name: str, n_items: int, reverse: bool = False, steps: int = 0) -> list[str]:
    """Colours offered for one pattern: the shades for the current number of items (or the set steps), or the whole list."""
    if is_shade_palette(name):
        k = int(steps or 0) or max(1, int(n_items))
        return palette_colors(name, min(k, MAX_SHADE_STEPS), reverse)
    p = PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
    return palette_colors(name, len(p))


def parse_colors(text: str) -> dict[str, str]:
    """'名前,#RRGGBB' (or 名前 = #RRGGBB) per line → {name: '#RRGGBB'}; lines that do not parse are skipped."""
    out = {}
    for line in str(text or "").splitlines():
        m = re.match(r"\s*(.+?)\s*[,=\t]\s*(#?[0-9A-Fa-f]{6})\s*$", line)
        if m and m.group(1).lower() not in ("name", "名前"):
            out[m.group(1)] = "#" + m.group(2).lstrip("#").upper()
    return out


def load_colors(path) -> dict[str, str]:
    """The named colours: colors.csv if it has any, otherwise NAMED_COLORS."""
    try:
        with open(path, encoding="utf-8-sig") as stream:
            named = parse_colors(stream.read())
    except (OSError, UnicodeError):
        named = {}
    return named or dict(NAMED_COLORS)


def color_emoji(hexcol: str) -> str:
    """The coloured square closest to a colour (a hint in a list; the real colour is the code)."""
    try:
        r, g, b = (int(str(hexcol).lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, AttributeError):
        return "▫️"
    if max(r, g, b) - min(r, g, b) < 40:                      # greys as a black or white square
        return "⬛" if (r + g + b) / 3 < 110 else "⬜"
    return min(_EMOJI_RGB, key=lambda e: sum((a - c) ** 2 for a, c in zip(_EMOJI_RGB[e], (r, g, b))))
