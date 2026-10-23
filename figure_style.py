"""Publication figure styling and exports with an unchanged physical canvas.

``fig_width`` and ``fig_height`` in existing projects remain inches. Unit
conversion is only for the controls: store converted values in those legacy
keys, and convert them back when displaying millimetres.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text


MM_PER_INCH = 25.4
FONT_FAMILIES = ("Arial", "DejaVu Sans", "Times New Roman")
# These are convenient starting sizes, not journal-specific specifications.
FIGURE_PRESETS_MM = {
    "custom": None,
    "single": (85.0, 70.0),
    "double": (180.0, 120.0),
}
STYLE_DEFAULTS = {
    "size_unit": "mm",
    "figure_preset": "custom",
    "font_family": "Arial",
    "line_width": 1.2,
    "reference_line_width": 1.0,
    "axes_line_width": 0.8,
    "margin_left": 0.14,
    "margin_right": 0.97,
    "margin_bottom": 0.14,
    "margin_top": 0.96,
    "transparent_background": False,
    "svg_text_editable": True,
}
STYLE_KEYS = frozenset(STYLE_DEFAULTS)
EXPORT_MIME_TYPES = {
    "png": "image/png",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def mm_to_inches(value: float) -> float:
    return _finite_number(value, "Millimetres") / MM_PER_INCH


def inches_to_mm(value: float) -> float:
    return _finite_number(value, "Inches") * MM_PER_INCH


def _unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise ValueError("Figure size unit must be 'mm' or 'inch'.")
    aliases = {"mm": "mm", "inch": "inch", "in": "inch", "inches": "inch"}
    try:
        return aliases[unit.lower().strip()]
    except KeyError as exc:
        raise ValueError("Figure size unit must be 'mm' or 'inch'.") from exc


def size_to_inches(width: float, height: float, unit: str = "mm") -> tuple[float, float]:
    """Convert control values into the legacy, inch-valued project settings."""
    width, height = _finite_number(width, "Width"), _finite_number(height, "Height")
    if width <= 0 or height <= 0:
        raise ValueError("Figure width and height must be positive.")
    if _unit(unit) == "mm":
        width, height = mm_to_inches(width), mm_to_inches(height)
    if width <= 0 or height <= 0:
        raise ValueError("Figure width and height are too small.")
    return width, height


def size_from_inches(width: float, height: float, unit: str = "mm") -> tuple[float, float]:
    width, height = size_to_inches(width, height, "inch")
    if _unit(unit) == "mm":
        return inches_to_mm(width), inches_to_mm(height)
    return width, height


def preset_size_inches(preset: str) -> tuple[float, float] | None:
    try:
        size = FIGURE_PRESETS_MM[preset]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown figure size preset.") from exc
    return None if size is None else size_to_inches(*size, "mm")


def _font_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Unknown font family.")
    normalized = "".join(name.casefold().split())
    choices = {"".join(font.casefold().split()): font for font in FONT_FAMILIES}
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError("Unknown font family.") from exc


def validate_style_settings(settings: Mapping | None, *, fill_defaults: bool = False) -> dict:
    """Validate only this module's settings, ignoring other project keys.

    Partial legacy projects need no added fields. ``fill_defaults=True`` is
    useful for rendering. Invalid fields raise ValueError before state changes.
    Margins are canvas fractions and are checked together with missing defaults.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("Figure style settings must be an object.")
    cleaned = {}
    for key in STYLE_KEYS.intersection(settings):
        value = settings[key]
        if key == "size_unit":
            value = _unit(value)
        elif key == "figure_preset":
            if not isinstance(value, str) or value not in FIGURE_PRESETS_MM:
                raise ValueError("Unknown figure size preset.")
        elif key == "font_family":
            value = _font_name(value)
        elif key in ("transparent_background", "svg_text_editable"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false.")
        else:
            value = _finite_number(value, key)
            lower, upper = (0.0, 1.0) if key.startswith("margin_") else (0.1, 5.0)
            if not lower <= value <= upper:
                raise ValueError(f"{key} must be between {lower:g} and {upper:g}.")
        cleaned[key] = value
    merged = {**STYLE_DEFAULTS, **cleaned}
    if merged["margin_left"] >= merged["margin_right"]:
        raise ValueError("The left margin must be smaller than the right margin.")
    if merged["margin_bottom"] >= merged["margin_top"]:
        raise ValueError("The bottom margin must be smaller than the top margin.")
    return merged if fill_defaults else cleaned


def resolve_font_family(requested: str = "Arial") -> str:
    """Resolve an installed font, keeping serif/sans-serif character on fallback."""
    requested = _font_name(requested)
    candidates = {
        "Arial": ("Arial", "Liberation Sans", "DejaVu Sans"),
        "Times New Roman": ("Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"),
        "DejaVu Sans": ("DejaVu Sans",),
    }[requested]
    for family in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False,
            )
            return font_manager.get_font(path).family_name
        except (ValueError, OSError, RuntimeError):
            continue
    # DejaVu is shipped with Matplotlib, so a normal installation always reaches
    # an installed fallback. A damaged installation should report its real error.
    raise ValueError("No usable installed font was found.")


def resolve_font(family: str = "Arial") -> str:
    """Installed family name shared by the Matplotlib and Plotly renderers."""
    return resolve_font_family(family)


def font_rcparams(settings: Mapping | None = None) -> dict:
    style = validate_style_settings(settings, fill_defaults=True)
    family = resolve_font_family(style["font_family"])
    return {
        "font.family": [family],
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none" if style["svg_text_editable"] else "path",
        "axes.linewidth": style["axes_line_width"],
        "xtick.major.width": style["axes_line_width"],
        "xtick.minor.width": style["axes_line_width"],
        "ytick.major.width": style["axes_line_width"],
        "ytick.minor.width": style["axes_line_width"],
    }


def apply_mpl_style(fig, settings: Mapping | None = None):
    """Apply margins, axes and font to an existing figure, then return it.

    Curve/reference widths are intentionally explicit renderer inputs because
    artists such as peak markers and reference sticks have different roles.
    Call this after constructing axes; do not call tight_layout afterwards.
    """
    style = validate_style_settings(settings, fill_defaults=True)
    family = resolve_font_family(style["font_family"])
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)
        fig.set_tight_layout(False)
    fig.subplots_adjust(
        left=style["margin_left"], right=style["margin_right"],
        bottom=style["margin_bottom"], top=style["margin_top"],
    )
    background = "none" if style["transparent_background"] else "white"
    fig.patch.set_facecolor(background)
    for ax in fig.axes:
        ax.set_facecolor(background)
        for spine in ax.spines.values():
            spine.set_linewidth(style["axes_line_width"])
        ax.tick_params(which="both", width=style["axes_line_width"])
    for text in fig.findobj(match=Text):
        text.set_fontfamily(family)
    return fig


def export_figure(fig, format: str, dpi: float = 300, settings: Mapping | None = None, *, style: Mapping | None = None) -> bytes:
    """Export PNG/TIFF/PDF/SVG without cropping or resizing the physical canvas.

    Raster dimensions are quantized to integer pixels by the backend. Editable
    SVG text references the resolved font; it does not embed font files. PDF
    uses TrueType embedding. This function does not close the caller's figure.
    ``style`` is an alias for ``settings`` for callers that name the style.
    """
    if settings is not None and style is not None:
        raise ValueError("Specify either settings or style, not both.")
    style_values = validate_style_settings(settings if style is None else style, fill_defaults=True)
    if not isinstance(format, str):
        raise ValueError("Unknown export format.")
    format = format.lower().lstrip(".")
    if format == "tif":
        format = "tiff"
    if format not in EXPORT_MIME_TYPES:
        raise ValueError("Export format must be PNG, TIFF, PDF or SVG.")
    dpi = _finite_number(dpi, "DPI")
    if not 0 < dpi <= 2400:
        raise ValueError("DPI must be greater than zero and at most 2400.")
    size_to_inches(*fig.get_size_inches(), "inch")
    apply_mpl_style(fig, style_values)
    background = "none" if style_values["transparent_background"] else "white"
    rc = {**font_rcparams(style_values), "savefig.bbox": None, "savefig.pad_inches": 0.0}
    result = io.BytesIO()
    save_options = {}
    if format == "tiff":
        from PIL import features
        # Standard Pillow wheels include libtiff. Keep custom minimal Pillow
        # builds usable too: their uncompressed TIFF remains pixel-identical.
        if features.check("libtiff"):
            save_options["pil_kwargs"] = {"compression": "tiff_adobe_deflate"}
    with mpl.rc_context(rc):
        fig.savefig(
            result, format=format, dpi=dpi, bbox_inches=None, pad_inches=0,
            facecolor=background, edgecolor=background,
            transparent=style_values["transparent_background"],
            **save_options,
        )
    return result.getvalue()
