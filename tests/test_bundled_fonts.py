"""Arial- and Times-compatible fonts ship with the app, so figures never fall back to DejaVu.

On a server where matplotlib knows no Arial, Helvetica, Liberation Sans or Arimo (Streamlit
Cloud: no such font is installed there), "Arial" resolved to DejaVu Sans and every text was
drawn in it. The bundled Liberation Sans and Liberation Serif (same character widths as Arial
and Times New Roman, SIL OFL 1.1) are registered at import, so every text uses them.
"""

import contextlib
import io
from pathlib import Path
import re
import sys
import unittest
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from build_bundle import create_bundle_bytes
from figure_style import (BUNDLED_FONT_FILES, FONT_DIR, STYLE_DEFAULTS, export_figure, font_rcparams,
                          register_bundled_fonts, resolve_font_family)
import test_xrd_regressions as regression

SUBSTITUTES = {"arial", "helvetica", "liberation sans", "arimo", "times new roman", "times", "liberation serif", "tinos"}
BUNDLED = {str(FONT_DIR / name) for name in BUNDLED_FONT_FILES}


@contextlib.contextmanager
def server_fonts(keep_bundled=True):
    """matplotlib as on a server with no Arial or Times font installed (optionally without the bundled ones too)."""
    manager = font_manager.fontManager
    saved = list(manager.ttflist)
    manager.ttflist = [f for f in saved if f.name.lower() not in SUBSTITUTES or (keep_bundled and f.fname in BUNDLED)]
    manager._findfont_cached.cache_clear()
    try:
        yield
    finally:
        manager.ttflist = saved
        manager._findfont_cached.cache_clear()


def figure_with_references(style):
    ns = regression.renderer_namespace(False)
    ns.update(x_min=20.0, x_max=50.0, show_side_labels=False, plot_style=style,
              active_cif=[ns["RestoredFile"]("a.cif", b"VALID"), ns["RestoredFile"]("b.cif", b"VALID")],
              cif_sort_idx=[0, 1], cif_visibles=[True, True], cif_labels=["Yb_{2}Si_{2}O_{7}#16048", "SiC#24217"],
              cif_colors=["#000000", "#000000"], cif_orders=[1, 2], cif_ref_lines=[False, False],
              calc_cif_pattern=lambda *a, **k: (np.array([28.0, 29.7, 35.7]), np.array([100.0, 60.0, 80.0])))
    return ns["build_figure"]()


def pdf_fonts(pdf):
    """Base names of the fonts embedded in a PDF (subset prefixes removed)."""
    return {name.decode() for name in re.findall(rb"/BaseFont /(?:[A-Z]{6}\+)?([A-Za-z\-]+)", pdf)}


class BundledFontTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_fonts_ship_unmodified_with_their_licence(self):
        self.assertEqual(len(BUNDLED_FONT_FILES), 8)
        for name in BUNDLED_FONT_FILES:
            path = FONT_DIR / name
            self.assertTrue(path.is_file(), name)
            family = "Liberation Serif" if name.startswith("LiberationSerif") else "Liberation Sans"
            font = matplotlib.ft2font.FT2Font(str(path))
            self.assertEqual(font.family_name, family)
            self.assertEqual(font.get_sfnt_table("OS/2")["fsType"], 0)   # installable embedding (PDF)
        licence = (FONT_DIR / "OFL.txt").read_text(encoding="utf-8")
        self.assertIn("SIL OPEN FONT LICENSE Version 1.1", licence)
        self.assertIn("Reserved Font Name Liberation", licence)
        with zipfile.ZipFile(io.BytesIO(create_bundle_bytes())) as bundle:
            names = set(bundle.namelist())
        for name in (*BUNDLED_FONT_FILES, "OFL.txt"):
            self.assertIn(f"XRD-Maker/fonts/{name}", names)

    def test_registration_is_done_once(self):
        self.assertEqual(register_bundled_fonts(), [])         # already registered at import
        entries = [f for f in font_manager.fontManager.ttflist if f.fname in BUNDLED]
        self.assertEqual(len(entries), len(BUNDLED_FONT_FILES))

    def test_without_an_arial_compatible_font_matplotlib_falls_back_to_dejavu(self):
        # The cause of tick labels and axis titles drawn in DejaVu Sans on the web version.
        with server_fonts(keep_bundled=False):
            self.assertEqual(resolve_font_family("Arial"), "DejaVu Sans")
            self.assertEqual(resolve_font_family("Times New Roman"), "DejaVu Serif")

    def test_times_new_roman_gets_the_bundled_liberation_serif(self):
        with server_fonts(keep_bundled=True):
            self.assertEqual(resolve_font_family("Times New Roman"), "Liberation Serif")
            style = {**STYLE_DEFAULTS, "font_family": "Times New Roman"}
            fonts = pdf_fonts(export_figure(figure_with_references(style), "pdf", 300, style=style))
            self.assertTrue(fonts and all(f.startswith("LiberationSerif") for f in fonts), fonts)

    def test_bundled_liberation_sans_draws_every_text_on_such_a_server(self):
        with server_fonts(keep_bundled=True):
            self.assertEqual(resolve_font_family("Arial"), "Liberation Sans")
            path = font_manager.findfont(font_manager.FontProperties(family=["Liberation Sans"]), fallback_to_default=False)
            self.assertIn(str(path), BUNDLED)              # FontPath (mpl 3.11) is not equal to its str
            style = {**STYLE_DEFAULTS, "font_family": "Arial"}
            fig = figure_with_references(style)
            pdf = export_figure(fig, "pdf", 300, style=style)
            fonts = pdf_fonts(pdf)
            self.assertTrue(fonts and all(f.startswith("LiberationSans") for f in fonts), fonts)
            self.assertIn("LiberationSans-Italic", fonts)                 # 2θ in the axis title
            svg = export_figure(fig, "svg", 300, style=style).decode()
            self.assertNotIn("Liberation Sans", svg)                       # the SVG names Arial
            self.assertIn("font-family: 'Arial'", svg)

    def test_generic_sans_serif_also_gets_the_chosen_font(self):
        rc = font_rcparams({"font_family": "Arial"})
        self.assertEqual(rc["font.sans-serif"][0], rc["font.family"][0])
        with matplotlib.rc_context(rc):
            # A bare string would be read as a fontconfig pattern ("sans-serif" does not parse).
            path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
        self.assertEqual(matplotlib.ft2font.FT2Font(path).family_name, rc["font.family"][0])


if __name__ == "__main__":
    unittest.main()
