"""Portable figure appearance, separate from measurements and sample settings."""
from __future__ import annotations

import json
from figure_style import STYLE_KEYS
from project_io import clean_settings, ProjectError

TEMPLATE_KEYS = frozenset(STYLE_KEYS) | {
    "fig_width", "fig_height", "dpi_export", "font_size", "show_legend",
    "show_cif_legend", "show_side_labels", "major_tick", "show_minor", "minor_tick",
    "cif_label_side", "cif_label_fontsize", "cif_label_offset_x", "cif_label_offset_y",
    "label_side", "label_fontsize", "label_offset_x", "label_offset_y",
    "annotation_font_size", "tick_font_size", "legend_font_size", "x_label", "y_label",
}
MAX_TEMPLATE_BYTES = 64 * 1024


def encode_style_template(settings: dict, name: str = "Figure style") -> bytes:
    if not isinstance(name, str) or len(name) > 200:
        raise ProjectError("Invalid style name.")
    cleaned = clean_settings({k: v for k, v in settings.items() if k in TEMPLATE_KEYS})
    return json.dumps({"format": "xrd-maker-style", "version": 1,
                       "name": name, "settings": cleaned},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode_style_template(raw: bytes) -> dict:
    if len(raw) > MAX_TEMPLATE_BYTES:
        raise ProjectError("Style template exceeds the 64 KB limit.")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a valid figure style JSON file.") from exc
    if (not isinstance(document, dict) or document.get("format") != "xrd-maker-style"
            or type(document.get("version")) is not int or document["version"] != 1
            or not isinstance(document.get("settings"), dict)):
        raise ProjectError("Unsupported figure style template.")
    if any(k not in TEMPLATE_KEYS for k in document["settings"]):
        raise ProjectError("Style template contains settings unrelated to figure appearance.")
    return clean_settings(document["settings"])
