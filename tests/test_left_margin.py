"""The left margin fits the y-axis title: 0.1 inch (2.5 mm) between the title and the figure's edge.

The intensity axis has no tick labels, so a margin set as a share of the width (14 %) stood mostly
empty in a wide figure: 1.4 inch at the default 10 inch width, of which over 1 inch was blank left
of the title. With the fit on (the default) the figure keeps its size and the plot takes that room;
switched off, the left margin (%) applies as before and starts from the fitted value.
"""

import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import test_xrd_regressions as regression
from test_panel_columns import three_columns
from project_io import clean_settings
from style_templates import decode_style_template, encode_style_template

PAD_IN = 0.1
# The strip measured in a saved image: text extents differ a little between matplotlib versions
# and resolutions (3.10 rounds glyphs to pixels), so about 2.5 mm means within ±0.035 inch (±0.9 mm).
STRIP_DELTA = 0.035


def left_blank_inches(png, dpi):
    """Width of the white strip at the left edge of a saved image, in inches."""
    grey = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    return np.nonzero((grey < 250).any(axis=0))[0][0] / dpi


def saved_png(ns, fig, dpi=150):
    style = dict(ns["plot_style"], margin_left=fig.subplotpars.left)
    return ns["export_figure"](fig, "png", dpi, style=style)


def wide_namespace(**changes):
    ns = regression.renderer_namespace()
    ns.update(fig_width=10.0, fig_height=8.0, font_size=14, tick_font_size=14, legend_font_size=14,
              y_title="Intensity (a.u.)")
    ns.update(changes)
    return ns


class FittedMarginTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_a_wide_figure_leaves_only_a_small_strip_left_of_the_title(self):
        ns = wide_namespace()
        before = ns["build_figure"]()
        self.assertGreater(left_blank_inches(saved_png(ns, before), 150), 1.0)      # the reported figure
        fig = ns["build_figure"](fit_left=True)
        self.assertLess(fig.subplotpars.left, ns["plot_style"]["margin_left"])
        self.assertAlmostEqual(left_blank_inches(saved_png(ns, fig), 150), PAD_IN, delta=STRIP_DELTA)
        self.assertEqual(tuple(fig.get_size_inches()), (10.0, 8.0))                  # same figure size
        self.assertAlmostEqual(fig.subplotpars.right, ns["plot_style"]["margin_right"])

    def test_the_strip_stays_the_same_for_any_title_size_or_figure_width(self):
        margins = {}
        for width, size in ((10.0, 12), (3.35, 12), (10.0, 24), (3.35, 7)):
            with self.subTest(width=width, size=size):
                ns = wide_namespace(fig_width=width, font_size=size)
                fig = ns["build_figure"](fit_left=True)
                self.assertAlmostEqual(left_blank_inches(saved_png(ns, fig, 300), 300), PAD_IN, delta=STRIP_DELTA)
                margins[width, size] = fig.subplotpars.left * width      # inches
                plt.close(fig)
        self.assertAlmostEqual(margins[10.0, 12], margins[3.35, 12], delta=0.01)   # not a share of the width
        self.assertLess(margins[10.0, 12], margins[10.0, 24])                       # a larger title takes more room

    def test_without_a_title_the_axis_sits_at_the_strip(self):
        ns = wide_namespace(y_title="")
        fig = ns["build_figure"](fit_left=True)
        self.assertAlmostEqual(left_blank_inches(saved_png(ns, fig), 150), PAD_IN, delta=STRIP_DELTA)

    def test_columns_keep_their_gaps(self):
        ns = three_columns(wide_namespace(), refs=True)
        fig = ns["build_figure"](fit_left=True)
        mains = fig.axes[0::2]
        gaps = [(right.get_position().x0 - left.get_position().x1) * 10.0 for left, right in zip(mains, mains[1:])]
        for gap in gaps:
            self.assertAlmostEqual(gap, ns["panel_gap_inches"](14), places=6)
        self.assertAlmostEqual(mains[-1].get_position().x1, ns["plot_style"]["margin_right"])
        self.assertAlmostEqual(left_blank_inches(saved_png(ns, fig), 150), PAD_IN, delta=STRIP_DELTA)


def run_app(extra=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
    state = {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": dict(regression.XRD_BYTES),
             "_cif_bytes": {}, "_pdf_bytes": {}, "show_panel": False}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


class AppLeftMarginTests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_on_by_default_in_the_saved_figure_and_the_preview(self):
        at = run_app()
        self.assertNoException(at)
        self.assertTrue(at.checkbox(key="auto_margin_left").value)
        self.assertNotIn("_percent_margin_left", [n.key for n in at.number_input])   # no box to type it in
        report = at.session_state["_regression_report"]
        self.assertLess(report["margin_left"], 0.14)
        self.assertAlmostEqual(left_blank_inches(report["png"], 150), PAD_IN, delta=STRIP_DELTA)
        self.assertEqual(report["preview_margin_l"], int(10.0 * 96 * report["margin_left"]))
        shown = [c.value for c in at.caption if c.value.startswith("左余白（自動）")]
        self.assertEqual(shown, [f"左余白（自動）：{100 * report['margin_left']:.1f} %"])
        self.assertIs(report["session"]["settings"]["auto_margin_left"], True)

    def test_switching_it_off_starts_from_the_fitted_margin(self):
        at = run_app()
        fitted = at.session_state["_regression_report"]["margin_left"]
        at.checkbox(key="auto_margin_left").uncheck().run()
        self.assertNoException(at)
        self.assertAlmostEqual(at.session_state["margin_left"], fitted)
        self.assertAlmostEqual(at.number_input(key="_percent_margin_left").value, 100 * fitted, places=4)
        report = at.session_state["_regression_report"]
        self.assertAlmostEqual(left_blank_inches(report["png"], 150), PAD_IN, delta=STRIP_DELTA)   # no jump
        self.assertIs(report["session"]["settings"]["auto_margin_left"], False)
        at.number_input(key="_percent_margin_left").set_value(14.0).run()                  # by hand, as before
        self.assertNoException(at)
        self.assertGreater(left_blank_inches(at.session_state["_regression_report"]["png"], 150), 1.0)

    def test_projects_and_styles_keep_the_choice(self):
        self.assertEqual(clean_settings({"auto_margin_left": False}), {"auto_margin_left": False})
        self.assertEqual(decode_style_template(encode_style_template({"auto_margin_left": False, "margin_left": .2})),
                         {"auto_margin_left": False, "margin_left": .2})
        older = json.dumps({"format": "xrd-maker-style", "version": 1, "name": "older",
                            "settings": {"margin_left": .2}}).encode("utf-8")
        self.assertEqual(decode_style_template(older), {"margin_left": .2})
        at = run_app({"auto_margin_left": False, "margin_left": .2})
        self.assertNoException(at)
        report = at.session_state["_regression_report"]
        self.assertAlmostEqual(report["margin_left"], .2)
        self.assertGreater(left_blank_inches(report["png"], 150), 1.5)          # the margin set by hand


if __name__ == "__main__":
    unittest.main()
