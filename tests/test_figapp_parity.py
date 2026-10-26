"""Colours, text markup and fonts shared with the graph app (figapp).

The palette and markup code must give the same results as the graph app, so the
graph app's palette function and palette lists are reproduced here verbatim and
compared colour by colour. UI tests run the real app with synthetic data.
"""

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from figure_style import (STYLE_DEFAULTS, css_font_stack, export_figure, font_family_list, font_rcparams,
                          resolve_font_family, svg_font_names, validate_style_settings)
from palettes import (NAMED_COLORS, PALETTES, PALETTE_NAMES, color_emoji, load_colors, palette_colors,
                      palette_sequence, parse_colors, swatch_colors)
from project_io import ProjectError, clean_settings, decode_project, encode_project
from rich_text import plain, rich, to_html
from style_templates import TEMPLATE_KEYS, decode_style_template, encode_style_template
import test_xrd_regressions as regression


# ---- the graph app's figcore.py, verbatim (v1.10)
FIGAPP_PALETTES = {
    'paper': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442'],
    'okabe_ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'],
    'blues': 'Blues', 'oranges': 'Oranges', 'greens': 'Greens', 'greys': 'Greys', 'viridis': 'viridis',
    'tab10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
}
FIGAPP_PALETTE_NAMES = {'paper': '論文用（青・朱・緑…）', 'okabe_ito': 'Okabe–Ito', 'blues': '青の濃淡', 'oranges': '朱の濃淡',
                        'greens': '緑の濃淡', 'greys': '灰の濃淡', 'viridis': 'viridis', 'tab10': 'matplotlib 標準'}


def figapp_palette_colors(name, n, reverse=False):
    p = FIGAPP_PALETTES.get(name, FIGAPP_PALETTES['paper'])
    if isinstance(p, str):
        cmap = plt.get_cmap(p)
        if n == 1:
            return [matplotlib.colors.to_hex(cmap(0.75))]
        cols = [matplotlib.colors.to_hex(cmap(0.35 + 0.6 * i / (n - 1))) for i in range(n)]
    else:
        cols = [p[i % len(p)] for i in range(n)]
    return cols[::-1] if reverse else cols


class PaletteParityTests(unittest.TestCase):
    def test_same_palettes_names_and_colours_as_the_graph_app(self):
        self.assertEqual(PALETTES, FIGAPP_PALETTES)
        self.assertEqual(PALETTE_NAMES["ja"], FIGAPP_PALETTE_NAMES)
        self.assertEqual(set(PALETTE_NAMES["en"]), set(PALETTES))
        for name in PALETTES:
            for n in range(1, 13):
                for reverse in (False, True):
                    with self.subTest(name=name, n=n, reverse=reverse):
                        self.assertEqual([c.upper() for c in palette_colors(name, n, reverse)],
                                         [c.upper() for c in figapp_palette_colors(name, n, reverse)])

    def test_known_shades_are_pinned(self):
        # Blues, 3 series: what the graph app draws for 750 / 950 / 1050 °C.
        self.assertEqual(palette_colors("blues", 3), ["#A6CEE4", "#3B8BC2", "#083C7D"])
        self.assertEqual(palette_colors("blues", 3, reverse=True), ["#083C7D", "#3B8BC2", "#A6CEE4"])
        self.assertEqual(palette_colors("greys", 1), [matplotlib.colors.to_hex(plt.get_cmap("Greys")(0.75)).upper()])
        self.assertEqual(palette_colors("paper", 0), [])

    def test_steps_reproduce_a_figure_with_another_series_count(self):
        self.assertEqual(palette_sequence("blues", 4, steps=3), palette_colors("blues", 3) + palette_colors("blues", 3)[:1])
        self.assertEqual(palette_sequence("blues", 2, steps=3), palette_colors("blues", 3)[:2])
        # Steps only matter for shades; lists keep their order.
        self.assertEqual(palette_sequence("paper", 3, steps=5), palette_colors("paper", 3))
        self.assertEqual(swatch_colors("blues", 4), palette_colors("blues", 4))
        self.assertEqual(swatch_colors("blues", 4, steps=6), palette_colors("blues", 6))
        self.assertEqual(swatch_colors("tab10", 2), palette_colors("tab10", 10))

    def test_named_colours_read_the_graph_apps_colors_csv(self):
        text = "名前,色\n750 °C,#0072B2\n950 °C, d55e00\nbroken line\n1050 °C = #009E73\n"
        self.assertEqual(parse_colors(text), {"750 °C": "#0072B2", "950 °C": "#D55E00", "1050 °C": "#009E73"})
        shipped = load_colors(APP_DIR / "colors.csv")
        self.assertEqual(list(shipped)[:3], ["750 °C", "950 °C", "1050 °C"])
        for name, value in NAMED_COLORS.items():
            self.assertEqual(shipped[name], value)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_colors(Path(folder) / "missing.csv"), NAMED_COLORS)
        self.assertEqual(color_emoji("#0072B2"), "🟦")
        self.assertEqual(color_emoji("#999999"), "⬜")
        self.assertEqual(color_emoji("#000000"), "⬛")


class MarkupTests(unittest.TestCase):
    def test_graph_app_markup_for_matplotlib(self):
        cases = {
            "K_{IC} (MPa m^{1/2})": r"K$_{\mathrm{IC}}$ (MPa m$^{\mathrm{1/2}}$)",
            "Yb_{2}Si_{2}O_{7}": r"Yb$_{\mathrm{2}}$Si$_{\mathrm{2}}$O$_{\mathrm{7}}$",
            "Eu^{3+}": r"Eu$^{\mathrm{3{+}}}$",
            "cm^-1": r"cm$^{\mathrm{{-}1}}$",
            "Diffraction angle, 2<i>θ</i> (deg.)": r"Diffraction angle, 2$\mathit{θ}$ (deg.)",
            "Δ<i>l</i>": r"Δ$\mathit{l}$",
            "<b>a</b>": r"$\mathbf{a}$",
            "<i>Sample A</i>": r"$\mathit{Sample\ A}$",
            "Yb<sub>2</sub>O<sub>3</sub>": r"Yb$_{\mathrm{2}}$O$_{\mathrm{3}}$",
            "5SiC_Yb2Si2O7_750C": "5SiC_Yb2Si2O7_750C",   # file names stay as written
            r"a\_{b}": "a_{b}",
            "Intensity (a.u.)": "Intensity (a.u.)",
            r"Sample$_{2}$": r"Sample$_{2}$",               # earlier XRD Maker labels are mathtext
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(rich(source), expected)

    def test_preview_html_and_plain_text(self):
        cases = {
            "Yb_{2}Si_{2}O_{7}": ("Yb<sub>2</sub>Si<sub>2</sub>O<sub>7</sub>", "Yb2Si2O7"),
            "Diffraction angle, 2<i>θ</i> (deg.)": ("Diffraction angle, 2<i>θ</i> (deg.)", "Diffraction angle, 2θ (deg.)"),
            r"Diffraction angle, 2$\it{\theta}$ (deg.)": ("Diffraction angle, 2<i>θ</i> (deg.)", "Diffraction angle, 2θ (deg.)"),
            r"Sample$_{2}$": ("Sample<sub>2</sub>", "Sample2"),
            r"Sample$^{N}$": ("Sample<sup><i>N</i></sup>", "SampleN"),
            r"$\it{TEXT}$": ("<i>TEXT</i>", "TEXT"),
            r"$H_\mathrm{V}$ (GPa)": ("<i>H</i><sub>V</sub> (GPa)", "HV (GPa)"),
            "Eu^{3+} & <i>x</i> < 1": ("Eu<sup>3+</sup> &amp; <i>x</i> &lt; 1", "Eu3+ & x < 1"),
            "cm^-1": ("cm<sup>-1</sup>", "cm-1"),
        }
        for source, (expected_html, expected_plain) in cases.items():
            with self.subTest(source=source):
                self.assertEqual(to_html(source), expected_html)
                self.assertEqual(plain(source), expected_plain)

    def test_marked_up_labels_render_in_every_format(self):
        fig, ax = plt.subplots(figsize=(3, 2))
        self.addCleanup(plt.close, fig)
        rc = font_rcparams({"font_family": "Arial"})
        with matplotlib.rc_context(rc):
            ax.set_xlabel(rich("Diffraction angle, 2<i>θ</i> (deg.)"))
            ax.set_ylabel(rich("Yb_{2}Si_{2}O_{7}, Eu^{3+}, cm^-1, <b>bold</b>"))
            ax.text(.1, .5, rich("○ △ □ ◇ β-Yb_{2}Si_{2}O_{7}"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for fmt in ("png", "pdf", "svg"):
                self.assertTrue(export_figure(fig, fmt, 150, settings={"font_family": "Arial"}))
        self.assertFalse([w for w in caught if "missing" in str(w.message).lower()],
                         "△ and ◇ must come from the glyph fallback, not a missing-glyph box")


class FontTests(unittest.TestCase):
    def test_fallback_list_and_math_fonts_follow_the_resolved_font(self):
        family = resolve_font_family("Arial")
        rc = font_rcparams({"font_family": "Arial"})
        self.assertEqual(rc["font.family"], font_family_list(family))
        self.assertEqual(rc["font.family"][-1], "DejaVu Sans")
        for key, suffix in (("mathtext.rm", ""), ("mathtext.it", ":italic"), ("mathtext.bf", ":bold"),
                            ("mathtext.sf", ""), ("mathtext.cal", "")):
            self.assertEqual(rc[key], family + suffix)
        self.assertEqual(font_family_list("DejaVu Sans"), ["DejaVu Sans"])
        self.assertIn("Arial", css_font_stack("Arial").split(",")[0])
        self.assertTrue(css_font_stack("Times New Roman").endswith("serif"))

    def test_svg_names_arial_when_a_same_width_font_stood_in(self):
        svg = b"<text style=\"font-size: 7px; font-family: 'Liberation Sans', 'DejaVu Sans'\">a</text>"
        from unittest.mock import patch
        with patch("figure_style.resolve_font_family", return_value="Liberation Sans"):
            self.assertIn(b"font-family: 'Arial', 'DejaVu Sans'", svg_font_names(svg, "Arial"))
        with patch("figure_style.resolve_font_family", return_value="DejaVu Sans"):
            self.assertEqual(svg_font_names(svg, "Arial"), svg)     # other widths: not renamed
        with patch("figure_style.resolve_font_family", return_value="Arial"):
            self.assertEqual(svg_font_names(svg, "Arial"), svg)

    def test_real_export_uses_one_family_for_text_and_italics(self):
        ns = regression.renderer_namespace(True)
        ns["labels"] = ["5 vol% SiC", "Hidden", "Yb_{2}Si_{2}O_{7}/SiC"]
        style = {**STYLE_DEFAULTS, "font_family": "Arial"}
        ns["plot_style"] = style
        fig = ns["build_figure"]()
        self.addCleanup(plt.close, fig)
        family = resolve_font_family("Arial").replace(" ", "")
        pdf = export_figure(fig, "pdf", 300, style=style)
        fonts = set(re.findall(rb"/BaseFont /[A-Z]+\+([A-Za-z\-]+)", pdf))
        self.assertTrue(fonts, pdf[:200])
        self.assertTrue(all(f.decode().startswith(family) for f in fonts), fonts)
        self.assertTrue(any(f.decode().endswith(("Italic", "Oblique")) for f in fonts), fonts)
        svg = export_figure(fig, "svg", 300, style=style).decode()
        root = ET.fromstring(svg)
        self.assertTrue(root.findall(".//{http://www.w3.org/2000/svg}text"))
        if resolve_font_family("Arial") in ("Liberation Sans", "Arimo", "Helvetica"):
            self.assertNotIn("Liberation Sans", svg)
            self.assertIn("font-family: 'Arial'", svg)

    def test_style_validation_for_new_keys(self):
        self.assertEqual(validate_style_settings({"tick_length": 2.5, "axes_box": False, "tick_direction": "out"}),
                         {"tick_length": 2.5, "axes_box": False, "tick_direction": "out"})
        for bad in ({"tick_length": 0}, {"tick_length": 11}, {"axes_box": "no"}, {"font_family": "Comic Sans"},
                    {"tick_direction": "up"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_style_settings(bad)
        for family in ("Helvetica", "Liberation Sans"):
            self.assertEqual(validate_style_settings({"font_family": family}), {"font_family": family})


class RendererTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_axis_labels_sizes_ticks_and_frame_reach_both_renderers(self):
        ns = regression.renderer_namespace(False)
        ns.update(x_title="2<i>θ</i> (°)", y_title="Intensity, <i>I</i> (counts)",
                  font_size=7.0, tick_font_size=6.5, legend_font_size=6.0,
                  labels=["Yb_{2}Si_{2}O_{7}", "Hidden", "β-SiC"],
                  plot_style={**STYLE_DEFAULTS, "tick_length": 2.5, "axes_box": False, "tick_direction": "out"})
        fig = ns["build_figure"]()
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), rich("2<i>θ</i> (°)"))
        self.assertEqual(ax.get_ylabel(), rich("Intensity, <i>I</i> (counts)"))
        self.assertEqual(ax.xaxis.label.get_fontsize(), 7.0)
        self.assertEqual(ax.xaxis.get_major_ticks()[0].label1.get_fontsize(), 6.5)
        self.assertEqual(ax.xaxis.get_major_ticks()[0].tick1line.get_markersize(), 2.5)
        self.assertEqual(ax.xaxis.get_major_ticks()[0].get_tickdir(), "out")
        self.assertFalse(ax.spines["top"].get_visible())
        self.assertFalse(ax.spines["right"].get_visible())
        legend = ax.get_legend()
        self.assertEqual([t.get_text() for t in legend.get_texts()], [rich("β-SiC"), rich("Yb_{2}Si_{2}O_{7}")])
        self.assertEqual(legend.get_texts()[0].get_fontsize(), 6.0)
        pfig = ns["build_plotly_figure"]()
        self.assertEqual(pfig.layout.xaxis.title.text, "2<i>θ</i> (°)")
        self.assertEqual(pfig.layout.yaxis.title.text, "Intensity, <i>I</i> (counts)")
        self.assertEqual(pfig.layout.xaxis.tickfont.size, 6.5)
        self.assertEqual(pfig.layout.legend.font.size, 6.0)
        self.assertFalse(pfig.layout.xaxis.mirror)
        self.assertEqual(pfig.layout.xaxis.ticks, "outside")
        self.assertIn("Arial", pfig.layout.font.family)
        names = [t.name for t in pfig.data if t.mode == "lines" and t.xaxis == "x"]
        self.assertEqual(names, ["β-SiC", "Yb<sub>2</sub>Si<sub>2</sub>O<sub>7</sub>"])

    def test_defaults_draw_the_same_axis_titles_as_before(self):
        ns = regression.renderer_namespace(False)
        fig = ns["build_figure"]()
        self.assertEqual(fig.axes[0].get_ylabel(), "Intensity (a.u.)")
        # 2θ is italic in the same font as the rest of the title, as with the earlier fixed title.
        self.assertEqual(fig.axes[0].get_xlabel(), r"Diffraction angle, 2$\mathit{θ}$ (deg.)")
        self.assertTrue(fig.axes[0].spines["top"].get_visible())
        self.assertEqual(fig.axes[0].xaxis.get_major_ticks()[0].tick1line.get_markersize(), 5.0)
        self.assertEqual(fig.axes[0].xaxis.get_major_ticks()[0].get_tickdir(), "in")     # as before

    def test_reference_panel_keeps_inward_boundary_ticks(self):
        ns = regression.renderer_namespace(False)
        ns.update(active_cif=[ns["RestoredFile"]("reference.cif", b"VALID")], cif_sort_idx=[0], cif_visibles=[True],
                  cif_labels=["β-Yb_{2}Si_{2}O_{7}"], cif_colors=["#000000"], cif_orders=[1], cif_ref_lines=[False])
        for direction, expected in (("auto", "out"), ("in", "in"), ("out", "out")):
            with self.subTest(direction=direction):
                ns["plot_style"] = {**STYLE_DEFAULTS, "tick_direction": direction}
                fig = ns["build_figure"]()
                main, ref = fig.axes[0], fig.axes[1]
                self.assertEqual(main.xaxis.get_major_ticks()[0].get_tickdir(), "in")
                self.assertEqual(ref.xaxis.get_major_ticks()[0].get_tickdir(), expected)
                self.assertEqual([t.get_text() for t in ref.texts], [rich("β-Yb_{2}Si_{2}O_{7}")])
                pfig = ns["build_plotly_figure"]()
                self.assertEqual(pfig.layout.xaxis.ticks, "inside")
                self.assertEqual(pfig.layout.xaxis2.ticks, {"in": "inside", "out": "outside"}[expected])
                plt.close(fig)


class ProjectCompatibilityTests(unittest.TestCase):
    def test_new_settings_validate_and_old_projects_still_open(self):
        settings = {"palette": "blues", "palette_reverse": True, "palette_refs": False, "palette_steps": 3,
                    "x_label": "2<i>θ</i> (°)", "y_label": "", "font_size": 7.0, "tick_font_size": 6.5,
                    "legend_font_size": 6.5, "tick_length": 2.5, "axes_box": False}
        self.assertEqual(clean_settings(settings), settings)
        self.assertEqual(clean_settings({"font_size": 14})["font_size"], 14)      # earlier integer sizes
        for bad in ({"palette": "rainbow"}, {"palette_steps": 13}, {"palette_steps": 1.5}, {"palette_reverse": "yes"},
                    {"x_label": "x" * 501}, {"y_label": 3}, {"tick_font_size": 3.5}, {"legend_font_size": 29}):
            with self.subTest(bad=bad), self.assertRaises(ProjectError):
                clean_settings(bad)
        old = json.loads(encode_project({"a.xy": b"10 1\n20 2\n"}, {}, {}, {"font_size": 12, "xrd_color_0": "#7b0000"}))
        restored = decode_project(json.dumps(old).encode())
        self.assertEqual(restored["settings"], {"font_size": 12, "xrd_color_0": "#7b0000"})

    def test_style_templates_carry_text_settings_but_not_colours(self):
        source = {"x_label": "2<i>θ</i> (°)", "y_label": "Intensity (a.u.)", "tick_font_size": 6.5,
                  "legend_font_size": 6.5, "tick_length": 2.5, "axes_box": False,
                  "palette": "blues", "palette_reverse": True, "xrd_color_0": "#0072B2"}
        restored = decode_style_template(encode_style_template(source))
        for key in ("x_label", "y_label", "tick_font_size", "legend_font_size", "tick_length", "axes_box"):
            self.assertIn(key, TEMPLATE_KEYS)
            self.assertEqual(restored[key], source[key])
        for key in ("palette", "palette_reverse", "xrd_color_0"):
            self.assertNotIn(key, restored)


def run_app(extra=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
    state = {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": dict(regression.XRD_BYTES),
             "_cif_bytes": {"cif1.cif": b"VALID"}, "_pdf_bytes": {"pdf1.pdf": b"VALID"}, "show_panel": True}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def colors(at, n=3):
    return [at.session_state[f"xrd_color_{i}"] for i in range(n)]


class PaletteUITests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_new_project_starts_with_the_graph_apps_paper_palette(self):
        at = run_app()
        self.assertNoException(at)
        self.assertEqual(colors(at), palette_colors("paper", 3))
        self.assertEqual(at.session_state["cif_color_0"], "#000000")    # references start in black
        self.assertEqual(at.session_state["pdf_color_0"], "#D55E00")

    def test_palette_follows_display_order_and_skips_hidden_patterns(self):
        at = run_app({"ord_0": 3, "ord_1": 1, "ord_2": 2, "vis_2": False,
                      "xrd_color_2": "#123456"})
        self.assertNoException(at)
        at.selectbox(key="palette").set_value("blues").run()
        self.assertNoException(at)
        shades = palette_colors("blues", 2)
        # Display order: pattern 1, then pattern 0 (pattern 2 is hidden and keeps its colour).
        self.assertEqual(at.session_state["xrd_color_1"], shades[0])
        self.assertEqual(at.session_state["xrd_color_0"], shades[1])
        self.assertEqual(at.session_state["xrd_color_2"], "#123456")
        at.checkbox(key="palette_reverse").check().run()
        self.assertEqual(at.session_state["xrd_color_1"], shades[1])
        self.assertEqual(at.session_state["xrd_color_0"], shades[0])

    def test_steps_references_and_reapply(self):
        at = run_app()
        at.selectbox(key="palette").set_value("greens").run()
        at.number_input(key="palette_steps").set_value(4).run()
        self.assertNoException(at)
        self.assertEqual(colors(at), palette_colors("greens", 4)[:3])
        at.checkbox(key="palette_refs").check().run()
        sequence = palette_colors("greens", 4)
        self.assertEqual(colors(at) + [at.session_state["cif_color_0"], at.session_state["pdf_color_0"]],
                         sequence[:3] + sequence[3:] + sequence[:1])
        at.button(key="xrd_color_0_sw_0").click().run()           # a single colour from the swatches…
        self.assertEqual(at.session_state["xrd_color_0"], sequence[0])
        at.selectbox(key="_cname_xrd_color_1").set_value("朱").run()   # …or by name (colors.csv)
        self.assertEqual(at.session_state["xrd_color_1"], "#D55E00")
        reapply = next(b for b in at.button if b.label == "色を当てはめ直す")
        reapply.click().run()
        self.assertNoException(at)
        self.assertEqual(colors(at), sequence[:3])

    def test_saved_project_keeps_palette_and_colours(self):
        at = run_app()
        at.selectbox(key="palette").set_value("oranges").run()
        at.text_input(key="x_label").input("2<i>θ</i> (°)").run()
        self.assertNoException(at)
        settings = at.session_state["_regression_report"]["session"]["settings"]
        self.assertEqual(settings["palette"], "oranges")
        self.assertEqual(settings["x_label"], "2<i>θ</i> (°)")
        self.assertEqual([settings[f"xrd_color_{i}"] for i in range(3)], palette_colors("oranges", 3))
        cleaned = clean_settings(copy.deepcopy(settings))            # the saved file opens again
        for key in ("palette", "x_label", "xrd_color_0", "font_size", "tick_font_size"):
            self.assertEqual(cleaned[key], settings[key])


if __name__ == "__main__":
    unittest.main()
