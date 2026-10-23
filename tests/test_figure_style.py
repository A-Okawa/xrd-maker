"""Physical canvas sizes, portable fonts, and lossless project-unit conversions."""

import io
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, features

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import (
    FIGURE_PRESETS_MM, FONT_FAMILIES, STYLE_DEFAULTS, apply_mpl_style, export_figure,
    inches_to_mm, mm_to_inches, preset_size_inches, resolve_font_family,
    size_from_inches, size_to_inches, validate_style_settings,
)


class FigureStyleTests(unittest.TestCase):
    def figure(self):
        fig, ax = plt.subplots(figsize=size_to_inches(85, 70))
        ax.plot([10, 20, 30], [1, 5, 2], label="Sample A")
        ax.set_xlabel(r"2$\theta$ (deg.)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(frameon=False)
        self.addCleanup(plt.close, fig)
        return fig

    def test_units_presets_and_legacy_inch_values(self):
        self.assertEqual(size_to_inches(10, 8, "inch"), (10., 8.))
        self.assertEqual(size_from_inches(10, 8, "inch"), (10., 8.))
        self.assertEqual(size_from_inches(10, 8, "mm"), (254., 203.2))
        for value in (0., 0.1, 85., 180., 254.):
            self.assertAlmostEqual(inches_to_mm(mm_to_inches(value)), value)
        self.assertEqual(preset_size_inches("single"), size_to_inches(*FIGURE_PRESETS_MM["single"]))
        self.assertIsNone(preset_size_inches("custom"))
        # Old projects need no style migration or figure-size reinterpretation.
        self.assertEqual(validate_style_settings({"fig_width": 10., "fig_height": 8.}), {})
        for arguments in ((0, 10, "mm"), (-1, 10, "inch"), (10, 10, "pixels")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                size_to_inches(*arguments)

    def test_style_validation_rejects_corrupt_and_crossed_margins(self):
        valid = validate_style_settings({"font_family": "TimesNewRoman", "size_unit": "in"}, fill_defaults=True)
        self.assertEqual(valid["font_family"], "Times New Roman")
        self.assertEqual(valid["size_unit"], "inch")
        self.assertEqual(valid["line_width"], STYLE_DEFAULTS["line_width"])
        for settings in (
            {"line_width": float("nan")}, {"axes_line_width": True},
            {"reference_line_width": 0}, {"margin_left": .8, "margin_right": .7},
            {"margin_bottom": .96, "margin_top": .96}, {"margin_top": 1.1},
            {"font_family": "missing font"}, {"figure_preset": "unknown"},
            {"svg_text_editable": "yes"}, {"transparent_background": 1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_style_settings(settings)

    def test_installed_font_fallback_preserves_serif_and_sans(self):
        original_findfont = font_manager.findfont
        paths = {name: original_findfont(font_manager.FontProperties(family=name))
                 for name in ("DejaVu Sans", "DejaVu Serif")}

        def only_bundled(prop, **_kwargs):
            name = prop.get_family()[0]
            if name not in paths:
                raise ValueError("not installed")
            return paths[name]

        with patch("figure_style.font_manager.findfont", side_effect=only_bundled):
            self.assertEqual(resolve_font_family("Arial"), "DejaVu Sans")
            self.assertEqual(resolve_font_family("Times New Roman"), "DejaVu Serif")

    def test_margins_fonts_and_axes_apply_without_changing_data_widths(self):
        fig = self.figure()
        ax = fig.axes[0]
        original_width = ax.lines[0].get_linewidth()
        fig.set_layout_engine("constrained")
        apply_mpl_style(fig, {"font_family": "DejaVu Sans", "axes_line_width": 1.5,
                              "margin_left": .2, "margin_right": .9,
                              "margin_bottom": .18, "margin_top": .88})
        self.assertIsNone(fig.get_layout_engine())
        self.assertAlmostEqual(ax.get_position().x0, .2)
        self.assertAlmostEqual(ax.get_position().y0, .18)
        self.assertAlmostEqual(ax.get_position().x1, .9)
        self.assertAlmostEqual(ax.get_position().y1, .88)
        self.assertEqual(ax.spines["left"].get_linewidth(), 1.5)
        self.assertEqual(ax.xaxis.label.get_fontfamily(), ["DejaVu Sans"])
        self.assertEqual(ax.lines[0].get_linewidth(), original_width)

    def test_raster_exports_keep_canvas_even_with_global_tight_bbox(self):
        fig = self.figure()
        fig.axes[0].text(-1, 2, "Outside canvas", transform=fig.axes[0].transAxes)
        requested = tuple(fig.get_size_inches())
        with matplotlib.rc_context({"savefig.bbox": "tight", "pdf.fonttype": 3, "svg.fonttype": "path"}):
            for format in ("png", "tiff"):
                with self.subTest(format=format):
                    data = export_figure(fig, format, dpi=300, settings={"font_family": "DejaVu Sans"})
                    with Image.open(io.BytesIO(data)) as image:
                        for actual, inches in zip(image.size, requested):
                            self.assertLess(abs(actual - inches * 300), 1)
                        self.assertAlmostEqual(image.info["dpi"][0], 300, delta=.1)
            self.assertEqual(matplotlib.rcParams["savefig.bbox"], "tight")
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 3)
            self.assertEqual(matplotlib.rcParams["svg.fonttype"], "path")
        self.assertEqual(tuple(fig.get_size_inches()), requested)

    def test_pdf_and_svg_keep_exact_physical_size_and_vector_text(self):
        fig = self.figure()
        width, height = fig.get_size_inches() * 72
        pdf = export_figure(fig, "pdf", settings={"font_family": "DejaVu Sans"})
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(float(box[1]), width, places=7)
        self.assertAlmostEqual(float(box[2]), height, places=7)
        self.assertIn(b"/CIDFontType2", pdf)
        self.assertNotIn(b"/Subtype /Type3", pdf)
        svg = export_figure(fig, "svg", settings={"font_family": "DejaVu Sans"})
        root = ET.fromstring(svg)
        self.assertAlmostEqual(float(root.get("width").removesuffix("pt")), width, places=5)
        self.assertAlmostEqual(float(root.get("height").removesuffix("pt")), height, places=5)
        self.assertTrue(root.findall(".//{http://www.w3.org/2000/svg}text"))
        self.assertIn(b"Sample A", svg)
        self.assertIn(b"DejaVu Sans", svg)
        paths = export_figure(fig, "svg", settings={"svg_text_editable": False})
        self.assertFalse(ET.fromstring(paths).findall(".//{http://www.w3.org/2000/svg}text"))

    def test_export_rejects_invalid_formats_and_supports_transparency(self):
        fig = self.figure()
        for kwargs in ({"format": "jpg"}, {"format": "pdf", "dpi": 0}, {"format": "png", "dpi": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                export_figure(fig, **kwargs)
        data = export_figure(fig, "png", settings={"transparent_background": True})
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.convert("RGBA").getpixel((0, 0))[3], 0)

    def test_all_font_choices_export_greek_and_formatted_labels(self):
        fig = self.figure()
        fig.axes[0].set_title(r"$\it{Sample}$ $_{2}$ $^{3}$")
        for family in FONT_FAMILIES:
            with self.subTest(family=family):
                pdf = export_figure(fig, "pdf", settings={"font_family": family})
                self.assertIn(b"/CIDFontType2", pdf)
                self.assertNotIn(b"/Subtype /Type3", pdf)
                svg = export_figure(fig, "svg", settings={"font_family": family})
                self.assertTrue(ET.fromstring(svg).findall(".//{http://www.w3.org/2000/svg}text"))

    def test_tiff_deflate_is_lossless_with_raw_fallback(self):
        fig = self.figure()
        if not features.check("libtiff"):
            self.skipTest("This Pillow build does not include libtiff.")
        compressed = export_figure(fig, "tiff", dpi=150)
        with patch("PIL.features.check", return_value=False):
            raw = export_figure(fig, "tiff", dpi=150)
        self.assertLess(len(compressed), len(raw))
        with Image.open(io.BytesIO(compressed)) as small, Image.open(io.BytesIO(raw)) as large:
            self.assertEqual(small.tag_v2[259], 8)  # Adobe Deflate
            self.assertEqual(large.tag_v2[259], 1)  # Uncompressed fallback
            self.assertEqual(small.size, large.size)
            self.assertEqual(small.tobytes(), large.tobytes())


if __name__ == "__main__":
    unittest.main()
