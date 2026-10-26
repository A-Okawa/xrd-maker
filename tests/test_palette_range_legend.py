"""Legend order that follows the stacked patterns, and the palette applied to a range of patterns.

The legend lists the patterns from top to bottom as they appear in the figure (both the
exported figure and the Plotly preview). The palette can be limited to a range of
patterns in display order; patterns outside it (a base measurement, for example) take
the outside colour, black by default.
"""

import copy
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from palettes import palette_colors
from project_io import ProjectError, clean_settings, decode_project, encode_project
from rich_text import rich
import test_xrd_regressions as regression


def legend_texts(fig):
    legend = fig.axes[0].get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def preview_legend(pfig):
    """Names of the pattern traces in the order the Plotly legend shows them (rank, then trace order)."""
    traces = [t for t in pfig.data if t.mode == "lines" and t.xaxis == "x" and t.showlegend is not False]
    order = sorted(range(len(traces)), key=lambda k: (traces[k].legendrank if traces[k].legendrank is not None else 1000, k))
    return [traces[k].name for k in order]


def stacked_names(fig, labels):
    """Pattern names from top to bottom by where their curves are drawn (median height)."""
    import numpy as np
    curves = [(float(np.median(line.get_ydata())), line.get_label()) for line in fig.axes[0].get_lines()
              if line.get_label() in labels]
    return [name for _, name in sorted(curves, key=lambda c: -c[0])]


class LegendOrderTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_normalized_stack_legend_reads_top_to_bottom(self):
        ns = regression.renderer_namespace(True)
        ns["st"].session_state.update({"extra_offset_0": 0.0, "extra_offset_1": 0.0, "extra_offset_2": 0.0})
        ns.update(sort_idx=[0, 1, 2], visibles=[True, True, True], labels=["Base", "750 °C", "1050 °C"])
        fig = ns["build_figure"]()
        drawn = [rich(n) for n in ns["labels"]]
        # Display order 1 is drawn at the bottom; the legend now starts with the top curve.
        self.assertEqual(stacked_names(fig, drawn), ["1050 °C", "750 °C", "Base"])
        self.assertEqual(legend_texts(fig), stacked_names(fig, drawn))
        # Plot order (and so the drawing and the exported curves) is unchanged: display order.
        self.assertEqual([l.get_label() for l in fig.axes[0].get_lines()], drawn)
        pfig = ns["build_plotly_figure"]()
        self.assertEqual(preview_legend(pfig), ["1050 °C", "750 °C", "Base"])
        self.assertEqual([t.name for t in pfig.data if t.mode == "lines" and t.xaxis == "x"], ["Base", "750 °C", "1050 °C"])

    def test_extra_offsets_and_hidden_patterns_follow_the_drawing(self):
        ns = regression.renderer_namespace(True)
        # sort_idx [1, 2, 0], pattern 1 hidden: 2 is drawn at 0 + 1.1, 0 at 1.25 + 0.35 → 0 is on top.
        fig = ns["build_figure"]()
        self.assertEqual(legend_texts(fig), ["Custom A", "Custom C"])
        self.assertEqual(stacked_names(fig, ["Custom A", "Custom C"]), ["Custom A", "Custom C"])
        self.assertEqual(preview_legend(ns["build_plotly_figure"]()), ["Custom A", "Custom C"])
        # A large extra offset moves pattern 2 above pattern 0, and the legend follows.
        ns["st"].session_state["extra_offset_2"] = 3.0
        fig = ns["build_figure"]()
        self.assertEqual(legend_texts(fig), ["Custom C", "Custom A"])
        self.assertEqual(preview_legend(ns["build_plotly_figure"]()), ["Custom C", "Custom A"])

    def test_raw_intensities_follow_the_y_positions_and_ties_keep_display_order(self):
        ns = regression.renderer_namespace(False)
        ns.update(sort_idx=[0, 1, 2], visibles=[True, True, True], labels=["A", "B", "C"], abs_offsets=[0.0, 200.0, 100.0])
        fig = ns["build_figure"]()
        self.assertEqual(legend_texts(fig), ["B", "C", "A"])
        self.assertEqual(preview_legend(ns["build_plotly_figure"]()), ["B", "C", "A"])
        ns["abs_offsets"] = [0.0, 0.0, 0.0]          # overlapping: the display order
        fig = ns["build_figure"]()
        self.assertEqual(legend_texts(fig), ["A", "B", "C"])
        self.assertEqual(preview_legend(ns["build_plotly_figure"]()), ["A", "B", "C"])

    def test_empty_labels_stay_out_and_phase_markers_come_after_the_patterns(self):
        import numpy as np
        ns = regression.renderer_namespace(True)
        ns["st"].session_state.update({"extra_offset_0": 0.0, "extra_offset_1": 0.0, "extra_offset_2": 0.0})
        ns.update(sort_idx=[0, 1, 2], visibles=[True, True, True], labels=["A", "", "C"])
        self.assertEqual(legend_texts(ns["build_figure"]()), ["C", "A"])
        ns.update(labels=["A", "B", "C"], show_phase_markers=True, phase_tolerance=0.5,
                  active_cif=[ns["RestoredFile"]("phase.cif", b"VALID")], cif_sort_idx=[0], cif_visibles=[True],
                  cif_labels=["Phase"], cif_colors=["#8e24aa"], cif_orders=[1], cif_ref_lines=[False],
                  calc_cif_pattern=lambda *a, **k: (np.array([20.0]), np.array([100.0])))
        fig = ns["build_figure"]()
        self.assertEqual(legend_texts(fig), ["C", "B", "A", "Phase"])
        pfig = ns["build_plotly_figure"]()
        self.assertEqual(preview_legend(pfig), ["C", "B", "A"])
        phase = [t for t in pfig.data if (t.legendgroup or "").startswith("phase:") and t.showlegend]
        self.assertTrue(phase)
        self.assertTrue(all(t.legendrank is None or t.legendrank > 3 for t in phase))


def run_app(extra=None, files=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
    state = {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": dict(files or regression.XRD_BYTES),
             "_cif_bytes": {"cif1.cif": b"VALID"}, "_pdf_bytes": {}, "show_panel": True}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def colors(at, n=3):
    return [at.session_state[f"xrd_color_{i}"] for i in range(n)]


class PaletteRangeUITests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_base_pattern_black_and_shades_spread_over_the_range(self):
        at = run_app()
        self.assertNoException(at)
        self.assertEqual(at.slider(key="_palette_range").value, (1, 3))
        at.selectbox(key="palette").set_value("blues").run()
        self.assertEqual(colors(at), palette_colors("blues", 3))
        at.slider(key="_palette_range").set_range(2, 3).run()
        self.assertNoException(at)
        # Pattern 1 (the base) is black; the two others take the blue shades for two series.
        self.assertEqual(colors(at), ["#000000"] + palette_colors("blues", 2))
        self.assertEqual((at.session_state["palette_from"], at.session_state["palette_to"]), (2, 0))
        self.assertEqual(at.session_state["cif_color_0"], "#000000")
        at.checkbox(key="palette_reverse").check().run()
        self.assertEqual(colors(at), ["#000000"] + palette_colors("blues", 2, reverse=True))
        at.color_picker(key="palette_base_color").pick("#4D4D4D").run()
        self.assertNoException(at)
        self.assertEqual(colors(at)[0], "#4D4D4D")

    def test_range_follows_display_order_and_both_ends(self):
        at = run_app({"ord_0": 3, "ord_1": 1, "ord_2": 2})       # display order: 1, 2, 0
        at.selectbox(key="palette").set_value("greens").run()
        at.slider(key="_palette_range").set_range(1, 2).run()
        self.assertNoException(at)
        shades = palette_colors("greens", 2)
        self.assertEqual(at.session_state["xrd_color_1"], shades[0])
        self.assertEqual(at.session_state["xrd_color_2"], shades[1])
        self.assertEqual(at.session_state["xrd_color_0"], "#000000")
        self.assertEqual((at.session_state["palette_from"], at.session_state["palette_to"]), (1, 2))
        report = at.session_state["_regression_report"]
        self.assertEqual(report["session"]["settings"]["palette_to"], 2)
        caption = " ".join(c.value for c in at.caption)
        self.assertIn("範囲外：a", caption)

    def test_hidden_patterns_in_range_do_not_take_a_shade(self):
        at = run_app({"vis_2": False})
        at.selectbox(key="palette").set_value("blues").run()
        at.slider(key="_palette_range").set_range(2, 3).run()
        self.assertNoException(at)
        self.assertEqual(at.session_state["xrd_color_0"], "#000000")
        self.assertEqual(at.session_state["xrd_color_1"], palette_colors("blues", 1)[0])

    def test_until_the_end_includes_patterns_added_later(self):
        files = dict(regression.XRD_BYTES)
        at = run_app({"palette": "blues", "palette_from": 2, "palette_to": 0}, files)
        self.assertNoException(at)
        self.assertEqual(at.slider(key="_palette_range").value, (2, 3))
        files["d.xy"] = b"10 4\n20 9\n30 5\n40 4\n"
        at.session_state["_xrd_bytes"] = files
        at.run()
        self.assertNoException(at)
        self.assertEqual(at.slider(key="_palette_range").value, (2, 4))
        at.button(key=next(b.key for b in at.button if b.label == "色を当てはめ直す")).click().run()
        self.assertEqual(colors(at, 4), ["#000000"] + palette_colors("blues", 3))

    def test_single_pattern_has_no_range_slider(self):
        at = run_app(files={"a.xy": regression.XRD_BYTES["a.xy"]})
        self.assertNoException(at)
        self.assertFalse([s for s in at.slider if s.key == "_palette_range"])


class RangeSettingsTests(unittest.TestCase):
    def test_range_settings_validate_and_round_trip(self):
        settings = {"palette": "blues", "palette_from": 2, "palette_to": 0, "palette_base_color": "#000000"}
        self.assertEqual(clean_settings(settings), settings)
        raw = encode_project({"a.xy": b"10 1\n20 2\n"}, {}, {}, settings)
        self.assertEqual(decode_project(raw)["settings"], settings)
        for bad in ({"palette_from": 0}, {"palette_from": 1.5}, {"palette_to": -1}, {"palette_to": True},
                    {"palette_base_color": "black"}, {"palette_base_color": "#12345"}):
            with self.subTest(bad=bad), self.assertRaises(ProjectError):
                clean_settings(bad)

    def test_projects_saved_before_the_range_colour_every_pattern(self):
        old = json.loads(encode_project({"a.xy": b"10 1\n"}, {}, {}, {"palette": "blues"}))
        restored = decode_project(json.dumps(old).encode())
        self.assertNotIn("palette_from", restored["settings"])
        ns = regression.renderer_namespace()
        state = ns["st"].session_state
        state.clear()
        state.update(copy.deepcopy({"_xrd_pattern_names": ["a", "b", "c"], "palette": "blues"}))
        plan, count = ns["palette_plan"]()
        self.assertEqual(count, 3)
        self.assertEqual([plan[f"xrd_color_{i}"] for i in range(3)], palette_colors("blues", 3))


if __name__ == "__main__":
    unittest.main()
