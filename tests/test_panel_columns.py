"""Columns of panels: each column is its own stack of patterns, with the references under it.

For example firing times in columns (2 h, 24 h, 48 h) and three particle sizes stacked in each.
The k-th pattern of every column takes the same palette colour; the columns share the intensity
scale; the axis title is under every column and the intensity title on the first one.
"""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from palettes import palette_colors
from project_io import ProjectError, clean_settings, decode_project, encode_project
import test_xrd_regressions as regression


def three_columns(ns, refs=False):
    ns["st"].session_state.update({"panel_columns": 3, "col_0": 1, "col_1": 2, "col_2": 3,
                                   "panel_title_1": "2 h", "panel_title_2": "24 h", "panel_title_3": "48 h",
                                   "extra_offset_0": 0.0, "extra_offset_1": 0.0, "extra_offset_2": 0.0})
    ns.update(sort_idx=[0, 1, 2], visibles=[True, True, True], labels=["A", "B", "C"])
    if refs:
        ns.update(active_cif=[ns["RestoredFile"]("ref.cif", b"VALID")], cif_sort_idx=[0], cif_visibles=[True],
                  cif_labels=["Yb_{2}Si_{2}O_{7}"], cif_colors=["#000000"], cif_orders=[1], cif_ref_lines=[False],
                  calc_cif_pattern=lambda *a, **k: (np.array([20.0, 30.0]), np.array([100.0, 40.0])))
    return ns


class MatplotlibColumnTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_one_column_is_the_usual_figure(self):
        ns = regression.renderer_namespace(True)
        fig = ns["build_figure"]()
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_title(), "")

    def test_each_column_stacks_its_own_patterns_with_titles_and_shared_scale(self):
        ns = three_columns(regression.renderer_namespace(True))
        fig = ns["build_figure"]()
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual([[line.get_label() for line in ax.get_lines()] for ax in fig.axes], [["A"], ["B"], ["C"]])
        self.assertEqual([ax.get_title() for ax in fig.axes], ["2 h", "24 h", "48 h"])
        self.assertEqual([ax.get_ylabel() for ax in fig.axes], ["Intensity (a.u.)", "", ""])
        self.assertTrue(all(ax.get_xlabel() for ax in fig.axes))
        self.assertEqual({ax.get_ylim() for ax in fig.axes}.__len__(), 1)      # one intensity scale
        lefts = [ax.get_position().x0 for ax in fig.axes]
        self.assertEqual(lefts, sorted(lefts))                                   # left to right
        # Each column starts its stack at its own bottom (no offset carried over from the previous column).
        self.assertTrue(all(np.min(ax.get_lines()[0].get_ydata()) < 1.0 for ax in fig.axes))

    def test_references_sit_under_every_column(self):
        ns = three_columns(regression.renderer_namespace(True), refs=True)
        fig = ns["build_figure"]()
        self.assertEqual(len(fig.axes), 6)
        mains, refs = fig.axes[0::2], fig.axes[1::2]
        for main, ref in zip(mains, refs):
            self.assertAlmostEqual(main.get_position().x0, ref.get_position().x0)
            self.assertGreater(main.get_position().y0, ref.get_position().y0)
            self.assertTrue(ref.collections, "reference sticks missing")
            self.assertTrue(ref.get_xlabel())
            self.assertFalse(main.get_xlabel())

    def test_columns_and_legends_in_the_exported_file(self):
        ns = three_columns(regression.renderer_namespace(True), refs=True)
        fig = ns["build_figure"]()
        legends = [ax.get_legend() for ax in fig.axes[0::2]]
        self.assertTrue(all(legend is not None for legend in legends))
        svg = ns["export_figure"](fig, "svg", 300, style=ns["plot_style"]).decode()
        for title in ("2 h", "24 h", "48 h"):
            self.assertIn(f">{title}<", svg)

    def test_titles_fit_and_edge_tick_labels_do_not_collide(self):
        ns = three_columns(regression.renderer_namespace(True), refs=True)
        style = dict(ns["plot_style"], margin_top=0.99)               # titles still need their room
        ns.update(fig_width=180 / 25.4, fig_height=120 / 25.4, tick_font_size=10, font_size=10)
        top = min(style["margin_top"], 1 - (ns["font_size"] * 1.3 + 4) / 72 / ns["fig_height"])
        ns["plot_style"] = dict(style, margin_top=top)
        fig = ns["build_figure"]()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        figure_box = fig.bbox
        for ax in fig.axes[0::2]:
            box = ax.title.get_window_extent(renderer)
            self.assertLessEqual(box.y1, figure_box.y1)
        def shown_labels(ax):                      # tick labels inside the axis limits (others are not drawn)
            lo, hi = ax.get_xlim()
            return [label for label, loc in zip(ax.get_xticklabels(), ax.get_xticks())
                    if lo - 1e-9 <= loc <= hi + 1e-9 and label.get_text()]
        bottoms = fig.axes[1::2]
        for left, right in zip(bottoms, bottoms[1:]):
            last = shown_labels(left)[-1].get_window_extent(renderer)
            first = shown_labels(right)[0].get_window_extent(renderer)
            self.assertEqual((shown_labels(left)[-1].get_text(), shown_labels(right)[0].get_text()), ("40", "10"))
            self.assertLess(last.x1, first.x0)

    def test_an_empty_column_is_drawn_empty(self):
        ns = three_columns(regression.renderer_namespace(True))
        ns["st"].session_state.update({"col_1": 1})              # column 2 has nothing
        fig = ns["build_figure"]()
        self.assertEqual([len(ax.get_lines()) for ax in fig.axes], [2, 0, 1])
        self.assertIsNone(fig.axes[1].get_legend())


class PlotlyColumnTests(unittest.TestCase):
    def test_preview_has_the_same_columns_and_one_legend(self):
        ns = three_columns(regression.renderer_namespace(True), refs=True)
        ns["labels"] = ["1 µm", "1 µm", "5 µm"]
        ns["colors_sel"] = ["#0072B2", "#0072B2", "#D55E00"]
        pfig = ns["build_plotly_figure"]()
        lines = [t for t in pfig.data if t.mode == "lines" and t.name in ("1 µm", "5 µm")]
        self.assertEqual([t.xaxis for t in lines], ["x", "x2", "x3"])
        # The same name and colour in two columns is listed once and toggles together.
        self.assertEqual([t.showlegend for t in lines], [True, False, True])
        self.assertEqual(lines[0].legendgroup, lines[1].legendgroup)
        titles = [a.text for a in pfig.layout.annotations if a.text]
        self.assertEqual(titles[:3], ["2 h", "24 h", "48 h"])
        sticks = [t for t in pfig.data if t.mode == "lines" and t.name == "Yb<sub>2</sub>Si<sub>2</sub>O<sub>7</sub>"]
        self.assertEqual([t.xaxis for t in sticks], ["x4", "x5", "x6"])
        self.assertEqual(pfig.layout.yaxis2.matches, "y")
        self.assertEqual(pfig.layout.yaxis3.matches, "y")
        self.assertTrue(pfig.layout.yaxis.title.text)
        self.assertFalse(pfig.layout.yaxis2.title.text)


class PaletteByPositionTests(unittest.TestCase):
    def plan(self, extra):
        ns = regression.renderer_namespace()
        state = ns["st"].session_state
        state.clear()
        names = [f"p{k}" for k in range(9)]
        state.update({"_xrd_pattern_names": names, "palette": "blues", **extra})
        return ns["palette_plan"]()

    def test_same_position_same_colour_in_every_column(self):
        columns = {f"col_{k}": k // 3 + 1 for k in range(9)}
        plan, count = self.plan({"panel_columns": 3, **columns})
        shades = palette_colors("blues", 3)
        self.assertEqual(count, 3)
        for column in range(3):
            self.assertEqual([plan[f"xrd_color_{3 * column + k}"] for k in range(3)], shades)

    def test_the_range_counts_positions_within_each_column(self):
        columns = {f"col_{k}": k // 3 + 1 for k in range(9)}
        plan, count = self.plan({"panel_columns": 3, "palette_from": 2, "palette_to": 0, **columns})
        shades = palette_colors("blues", 2)
        for column in range(3):
            self.assertEqual(plan[f"xrd_color_{3 * column}"], "#000000")           # the base pattern of each column
            self.assertEqual([plan[f"xrd_color_{3 * column + k}"] for k in (1, 2)], shades)

    def test_one_column_keeps_the_display_order_palette(self):
        plan, count = self.plan({})
        self.assertEqual(count, 9)
        self.assertEqual([plan[f"xrd_color_{k}"] for k in range(9)], palette_colors("blues", 9))


class ColumnSettingsTests(unittest.TestCase):
    def test_settings_validate_and_round_trip(self):
        settings = {"panel_columns": 3, "panel_title_1": "2 h", "panel_title_3": "<i>t</i> = 48 h", "col_0": 3}
        self.assertEqual(clean_settings(settings), settings)
        raw = encode_project({"a.xy": b"10 1\n20 2\n"}, {}, {}, settings)
        self.assertEqual(decode_project(raw)["settings"], settings)
        for bad in ({"panel_columns": 0}, {"panel_columns": 5}, {"panel_columns": 2.0}, {"col_0": 5},
                    {"col_0": "2"}, {"panel_title_1": 3}, {"panel_title_2": "x" * 501}):
            with self.subTest(bad=bad), self.assertRaises(ProjectError):
                clean_settings(bad)


def run_app(extra=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
    state = {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": dict(regression.XRD_BYTES),
             "_cif_bytes": {"cif1.cif": b"VALID"}, "_pdf_bytes": {}, "show_panel": True}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


class ColumnUITests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_choose_columns_spread_titles_and_save(self):
        at = run_app()
        self.assertNoException(at)
        self.assertFalse([s for s in at.selectbox if s.key == "col_0"])          # one column: no column choice
        at.selectbox(key="panel_columns").set_value(3).run()
        self.assertNoException(at)
        self.assertEqual([at.selectbox(key=f"col_{i}").value for i in range(3)], [1, 1, 1])
        spread = next(b for b in at.button if b.label == "表示順で列に振り分ける")
        spread.click().run()
        self.assertNoException(at)
        self.assertEqual([at.session_state[f"col_{i}"] for i in range(3)], [1, 2, 3])
        at.text_input(key="panel_title_2").input("24 h").run()
        self.assertNoException(at)
        report = at.session_state["_regression_report"]
        settings = report["session"]["settings"]
        self.assertEqual((settings["panel_columns"], settings["panel_title_2"]), (3, "24 h"))
        self.assertEqual([settings[f"col_{i}"] for i in range(3)], [1, 2, 3])
        # The same colour for the first pattern of each column.
        colors = [at.session_state[f"xrd_color_{i}"] for i in range(3)]
        self.assertEqual(len(set(colors)), 1)

    def test_back_to_one_column_keeps_the_choices(self):
        at = run_app({"panel_columns": 3, "col_0": 1, "col_1": 2, "col_2": 3})
        self.assertNoException(at)
        at.selectbox(key="panel_columns").set_value(1).run()
        self.assertNoException(at)
        at.selectbox(key="panel_columns").set_value(3).run()
        self.assertNoException(at)
        self.assertEqual([at.selectbox(key=f"col_{i}").value for i in range(3)], [1, 2, 3])

    def test_fewer_columns_fold_patterns_into_the_last(self):
        at = run_app({"panel_columns": 3, "col_0": 1, "col_1": 2, "col_2": 3})
        at.selectbox(key="panel_columns").set_value(2).run()
        self.assertNoException(at)
        self.assertEqual([at.selectbox(key=f"col_{i}").value for i in range(3)], [1, 2, 2])


if __name__ == "__main__":
    unittest.main()
