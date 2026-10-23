#!/usr/bin/env python3
"""Local, credential-free XRD Maker regressions.

Run with the task venv:
  .venv/bin/python test_xrd_regressions.py --app /absolute/path/to/app.py -v

AppTest executes the real UI with authenticated session state and synthetic files.
Only CIF/PDF parsers are substituted with deterministic in-memory peak data.
Renderer tests execute the app's actual function ASTs, without starting its UI.
No Streamlit secrets or external files other than the selected app are read.
"""

import argparse
import ast
import copy
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


DEFAULT_APP = Path(__file__).resolve().parents[1] / "app.py"
PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--app", type=Path, default=DEFAULT_APP)
OPTIONS, UNITTEST_ARGS = PARSER.parse_known_args()
APP = OPTIONS.app.resolve()
sys.path.insert(0, str(APP.parent))
from project_io import GLOBAL_KEYS, PATTERN_KEY, encode_project, decode_project, ProjectError
from xrd_io import read_patterns, XRDParseError
from figure_style import (STYLE_DEFAULTS, STYLE_KEYS, FIGURE_PRESETS_MM, mm_to_inches,
    inches_to_mm, font_rcparams, apply_mpl_style, export_figure, resolve_font, validate_style_settings)
from style_templates import encode_style_template, decode_style_template
from project_io import clean_settings
from plot_annotations import (PreparedCurve, ReferencePeaks, match_reference_peaks,
    add_mpl_phase_annotations, add_plotly_phase_annotations, add_mpl_reference_hkl_labels,
    add_plotly_reference_hkl_labels, add_mpl_zoom_inset, add_plotly_zoom_inset, format_hkl_groups)
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(APP))


XRD_BYTES = {
    "a.xy": b"10 1\n20 4\n30 2\n40 1\n",
    "b.xy": b"10 2\n20 6\n30 3\n40 2\n",
    "c.xy": b"10 3\n20 8\n30 4\n40 3\n",
}


def peak_data(raw, two_theta_range=(5, 90)):
    if b"EMPTY" in raw:
        return np.array([]), np.array([])
    positions = np.array([15.0, 25.0, 35.0])
    intensities = np.array([100.0, 50.0, 25.0])
    selected = (positions >= two_theta_range[0]) & (positions <= two_theta_range[1])
    return positions[selected], intensities[selected]


def renderer_namespace(normalize=False):
    """Extract real renderers and their helper definitions; configure known inputs."""
    namespace = dict(
        np=np, pd=pd, plt=plt, gridspec=gridspec, go=go,
        make_subplots=make_subplots, find_peaks=find_peaks,
        io=io, os=os, re=re, json=json,
        GLOBAL_KEYS=GLOBAL_KEYS, PATTERN_KEY=PATTERN_KEY,
        encode_project=encode_project, decode_project=decode_project,
        ProjectError=ProjectError, read_patterns=read_patterns, XRDParseError=XRDParseError,
        STYLE_DEFAULTS=STYLE_DEFAULTS, STYLE_KEYS=STYLE_KEYS, FIGURE_PRESETS_MM=FIGURE_PRESETS_MM,
        mm_to_inches=mm_to_inches, inches_to_mm=inches_to_mm, font_rcparams=font_rcparams,
        apply_mpl_style=apply_mpl_style, export_figure=export_figure, resolve_font=resolve_font,
        validate_style_settings=validate_style_settings, clean_settings=clean_settings,
        encode_style_template=encode_style_template, decode_style_template=decode_style_template,
        PreparedCurve=PreparedCurve, ReferencePeaks=ReferencePeaks, match_reference_peaks=match_reference_peaks,
        add_mpl_phase_annotations=add_mpl_phase_annotations, add_plotly_phase_annotations=add_plotly_phase_annotations,
        add_mpl_reference_hkl_labels=add_mpl_reference_hkl_labels, add_plotly_reference_hkl_labels=add_plotly_reference_hkl_labels,
        add_mpl_zoom_inset=add_mpl_zoom_inset, add_plotly_zoom_inset=add_plotly_zoom_inset,
        format_hkl_groups=format_hkl_groups,
        plot_style=dict(STYLE_DEFAULTS), _lang_code="ja", show_hkl=False, show_phase_markers=False,
        show_zoom=False, zoom_range=(15.,25.), zoom_location="upper left", phase_tolerance=.1, annotation_font_size=8,
        st=SimpleNamespace(session_state={
            "extra_offset_0": 0.35, "extra_offset_1": -0.2, "extra_offset_2": 1.1,
        }, cache_data=lambda **kwargs: lambda fn: fn),
        show_legend=True, show_cif_legend=True, normalize=normalize,
        fig_width=6.0, fig_height=4.0, font_size=10,
        major_tick=10.0, show_minor=True, minor_tick=2.0,
        x_min=10.0, x_max=40.0, show_side_labels=True,
        label_side="右", label_fontsize=10, label_offset_x=0.5, label_offset_y=0.05,
        show_peaks=False, peak_prom=0.1, global_offset=1.25,
        cif_label_side="左", cif_label_offset_x=0.5, cif_label_offset_y=0.5,
        cif_label_fontsize=9, show_ref_lines=True,
        sort_idx=[1, 2, 0], visibles=[True, False, True],
        labels=["Custom A", "Hidden B", "Custom C"],
        colors_sel=["#b71c1c", "#1976d2", "#00897b"],
        abs_offsets=[40.0, 80.0, 120.0],
        cif_sort_idx=[], cif_visibles=[], cif_labels=[], cif_colors=[],
        cif_orders=[], cif_ref_lines=[],
        pdf_sort_idx=[], pdf_visibles=[], pdf_labels=[], pdf_colors=[],
        pdf_orders=[], pdf_ref_lines=[], active_cif=[], active_pdf=[],
    )
    definitions = []
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions.append(copy.deepcopy(node))
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "active_xrd":
            definitions.extend(copy.deepcopy(child) for child in node.body
                               if isinstance(child, ast.FunctionDef))
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(APP), "exec"), namespace)
    namespace["active_xrd"] = [namespace["RestoredFile"](n, b) for n, b in XRD_BYTES.items()]
    namespace["calc_cif_pattern"] = peak_data
    namespace["parse_pdf_card"] = peak_data
    return namespace


def ui_test_source():
    """Keep real UI/state handling; mock only scientific external-file parsers."""
    tree = copy.deepcopy(TREE)
    replacements = ast.parse('''
def calc_cif_pattern(cif_bytes, two_theta_range=(5, 90)):
    if b"EMPTY" in cif_bytes:
        return np.array([]), np.array([])
    return np.array([15., 25., 35.]), np.array([100., 50., 25.])
def parse_pdf_card(pdf_bytes, two_theta_range=(5, 90)):
    return np.array([15., 25., 35.]), np.array([100., 50., 25.])
def extract_pdf_card_label(pdf_bytes):
    return None
''').body
    by_name = {node.name: node for node in replacements}
    tree.body = [copy.deepcopy(by_name.get(node.name, node))
                 if isinstance(node, ast.FunctionDef) else node for node in tree.body]
    # Streamlit 1.32 AppTest serializes formatted selectbox values incorrectly.
    # Keep the application's real options/callbacks; remove only presentation
    # formatting from the test source (browser verification covers labels).
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "selectbox":
            node.keywords = [kw for kw in node.keywords if kw.arg != "format_func"]
    tree = ast.fix_missing_locations(tree)
    capture = '''
if active_xrd:
    st.session_state["_regression_report"] = {
        "sort_idx": list(sort_idx), "orders": list(orders),
        "visibles": list(visibles), "labels": list(labels),
        "colors": list(colors_sel), "abs_offsets": list(abs_offsets),
        "refs": [(r["label"], r["color"], r["order"], r["show_line"]) for r in build_ref_list()],
        "main_lines": [(l.get_label(), l.get_color(), np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist())
                       for l in fig.axes[0].get_lines() if l.get_label() in labels],
        "preview_lines": [(t.name, t.line.color, list(t.x), list(t.y))
                          for t in pfig.data if t.mode == "lines" and t.xaxis == "x"],
        "session": json.loads(build_session_json(len(active_xrd), len(active_cif), len(active_pdf))),
        "png_header": png_bytes[:8], "tiff_header": tiff_bytes[:4],
    }
'''
    return ast.unparse(tree) + "\n" + capture


class RendererRegressionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_preview_export_numeric_parity_raw_and_normalized(self):
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                ns = renderer_namespace(normalize)
                fig = ns["build_figure"]()
                pfig = ns["build_plotly_figure"]()
                lines = fig.axes[0].get_lines()
                traces = [t for t in pfig.data if t.mode == "lines" and t.xaxis == "x"]
                self.assertEqual([l.get_label() for l in lines], ["Custom C", "Custom A"])
                self.assertEqual([t.name for t in traces], ["Custom C", "Custom A"])
                for line, trace in zip(lines, traces):
                    self.assertEqual(line.get_color(), trace.line.color)
                    np.testing.assert_array_equal(line.get_xdata(), trace.x)
                    np.testing.assert_allclose(line.get_ydata(), trace.y)
                expected_c = np.array([3, 8, 4, 3])
                expected_a = np.array([1, 4, 2, 1])
                if normalize:
                    expected_c = expected_c / 8 + 1.1
                    expected_a = expected_a / 4 + 1.25 + 0.35
                else:
                    expected_c = expected_c + 120
                    expected_a = expected_a + 40
                np.testing.assert_allclose(lines[0].get_ydata(), expected_c)
                np.testing.assert_allclose(lines[1].get_ydata(), expected_a)
                plt.close(fig)

    def test_reference_guides_keep_main_y_limits_and_span_plotly_domain(self):
        ns = renderer_namespace()
        ns.update(
            active_cif=[ns["RestoredFile"]("reference.cif", b"VALID")],
            cif_sort_idx=[0], cif_visibles=[True], cif_labels=["Reference"],
            cif_colors=["#8e24aa"], cif_orders=[1], cif_ref_lines=[True],
            show_ref_lines=False,
        )
        without_guides = ns["build_figure"]()
        expected_limits = without_guides.axes[0].get_ylim()
        ns["show_ref_lines"] = True
        with_guides = ns["build_figure"]()
        np.testing.assert_allclose(with_guides.axes[0].get_ylim(), expected_limits)
        guide_lines = [line for line in with_guides.axes[0].get_lines()
                       if line.get_label() not in ns["labels"]]
        self.assertEqual(len(guide_lines), 3)
        for guide in guide_lines:
            np.testing.assert_allclose(guide.get_ydata(), [0, 1])
        preview = ns["build_plotly_figure"]()
        self.assertEqual(len(preview.layout.shapes), 3)
        for guide in preview.layout.shapes:
            self.assertEqual(guide.yref, "y domain")
            self.assertEqual((guide.y0, guide.y1), (0, 1))
        plt.close(without_guides)
        plt.close(with_guides)

    def test_empty_cif_does_not_crash_either_renderer_or_exports(self):
        for kind in ("matplotlib", "plotly"):
            for empty_only in (False, True):
                with self.subTest(renderer=kind, empty_only=empty_only):
                    ns = renderer_namespace()
                    refs = [("Empty CIF", b"EMPTY")]
                    if not empty_only:
                        refs.append(("Valid CIF", b"VALID"))
                    ns.update(
                        active_cif=[ns["RestoredFile"](label + ".cif", data) for label, data in refs],
                        cif_sort_idx=list(range(len(refs))), cif_visibles=[True] * len(refs),
                        cif_labels=[label for label, _ in refs],
                        cif_colors=["#8e24aa", "#1976d2"][:len(refs)],
                        cif_orders=list(range(1, len(refs) + 1)),
                        cif_ref_lines=[True] * len(refs),
                    )
                    if kind == "matplotlib":
                        fig = ns["build_figure"]()
                        collections = fig.axes[1].collections
                        self.assertEqual(len(collections), 0 if empty_only else 1)
                        if not empty_only:
                            self.assertEqual(collections[0].get_label(), "Valid CIF")
                            np.testing.assert_allclose(collections[0].get_segments()[0], [[15, 110], [15, 190]])
                        for fmt, signature in (("png", b"\x89PNG\r\n\x1a\n"), ("tiff", b"II*\x00")):
                            output = io.BytesIO()
                            fig.savefig(output, format=fmt, dpi=40)
                            self.assertTrue(output.getvalue().startswith(signature))
                        plt.close(fig)
                    else:
                        pfig = ns["build_plotly_figure"]()
                        refs_out = [t for t in pfig.data if t.mode == "lines" and t.xaxis == "x2"]
                        self.assertEqual(len(refs_out), 0 if empty_only else 1)
                        if not empty_only:
                            self.assertEqual(refs_out[0].name, "Valid CIF")
                            self.assertEqual(list(refs_out[0].y[:2]), [110.0, 190.0])


class ParserRegressionTests(unittest.TestCase):
    def test_invalid_single_column_data_returns_none(self):
        ns = renderer_namespace()
        self.assertIsNone(ns["read_xrd_data"](b"10\n20\n30\n", "single-column.csv"))

    def test_nonfinite_rows_removed_and_angles_sorted_stably(self):
        ns = renderer_namespace()
        for filename, data in (
            ("unsorted.xy", b"30 4\n10 1\n20 3\n20 5\nnan 2\n40 nan\n"),
            ("unsorted.csv", b"angle,intensity\n30,4\n10,1\n20,3\n20,5\nnan,2\n40,nan\n"),
        ):
            with self.subTest(filename=filename):
                actual = ns["read_xrd_data"](data, filename)
                np.testing.assert_array_equal(actual, [[10, 1], [20, 3], [20, 5], [30, 4]])

    def test_duplicate_angles_do_not_crash_peak_detection(self):
        import warnings
        ns = renderer_namespace()
        for angles in (np.array([10., 10., 10., 10.]), np.array([10., 10., 20., 30.])):
            with self.subTest(angles=angles.tolist()), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                peaks = ns["detect_peaks"](angles, np.array([0., 1., 0., 0.]))
                self.assertTrue(np.issubdtype(peaks.dtype, np.integer))


class UIRegressionTests(unittest.TestCase):
    def test_hidden_panel_preserves_order_settings_session_and_rendering(self):
        from streamlit.testing.v1 import AppTest
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                at = AppTest.from_string(ui_test_source(), default_timeout=60)
                initial = {
                    "authenticated": True, "lang_radio": "日本語",
                    "_xrd_bytes": dict(XRD_BYTES),
                    "_cif_bytes": {"cif1.cif": b"VALID", "cif2.cif": b"VALID"},
                    "_pdf_bytes": {"pdf1.pdf": b"VALID", "pdf2.pdf": b"VALID"},
                    "normalize": normalize, "show_ref_lines": True,
                    "fig_width": 6.0, "fig_height": 4.0, "font_size": 10,
                    "show_panel": True,
                    "ord_0": 3, "ord_1": 1, "ord_2": 2,
                    "vis_0": True, "vis_1": False, "vis_2": True,
                    "_val_lbl_0": "Custom A", "_val_lbl_1": "Hidden B", "_val_lbl_2": "Custom C",
                    "xrd_color_0": "#b71c1c", "xrd_color_1": "#1976d2", "xrd_color_2": "#00897b",
                    "extra_offset_0": 0.35, "extra_offset_1": -0.2, "extra_offset_2": 1.1,
                    "yoff_0": 40.0, "yoff_1": 80.0, "yoff_2": 120.0,
                    "cord_0": 6, "cord_1": 2, "cvis_0": True, "cvis_1": True,
                    "_val_clbl_0": "CIF 1", "_val_clbl_1": "CIF 2",
                    "cif_color_0": "#8e24aa", "cif_color_1": "#1976d2",
                    "cref_line_0": False, "cref_line_1": True,
                    "pord_0": 4, "pord_1": 1, "pvis_0": True, "pvis_1": False,
                    "_val_plbl_0": "PDF 1", "_val_plbl_1": "Hidden PDF 2",
                    "pdf_color_0": "#43a047", "pdf_color_1": "#d32f2f",
                    "pref_line_0": False, "pref_line_1": True,
                }
                for key, value in initial.items():
                    at.session_state[key] = value
                at.run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                baseline = copy.deepcopy(at.session_state["_regression_report"])
                self.assertEqual(baseline["sort_idx"], [1, 2, 0])
                self.assertEqual([r[0] for r in baseline["refs"]], ["CIF 2", "PDF 1", "CIF 1"])
                self.assertEqual([r[0] for r in baseline["main_lines"]], ["Custom C", "Custom A"])
                for phase in ("hide", "hidden rerun 1", "hidden rerun 2", "show"):
                    if phase == "hide":
                        at.toggle(key="show_panel").set_value(False)
                    elif phase == "show":
                        at.toggle(key="show_panel").set_value(True)
                    at.run()
                    self.assertFalse(at.exception, [e.message for e in at.exception])
                    actual = at.session_state["_regression_report"]
                    for field in ("sort_idx", "orders", "visibles", "labels", "colors", "abs_offsets", "refs", "main_lines", "preview_lines"):
                        self.assertEqual(actual[field], baseline[field], f"{phase}: {field}")
                    for key, value in baseline["session"]["settings"].items():
                        if key.startswith(("ord_", "vis_", "_val_", "xrd_color_", "yoff_", "extra_offset_", "cord_", "cvis_", "cif_color_", "cref_line_", "pord_", "pvis_", "pdf_color_", "pref_line_")):
                            self.assertIn(key, actual["session"]["settings"], f"{phase}: session lost {key}")
                            self.assertEqual(actual["session"]["settings"][key], value, f"{phase}: {key}")
                    self.assertEqual(actual["main_lines"], actual["preview_lines"])
                    self.assertEqual(actual["png_header"], b"\x89PNG\r\n\x1a\n")
                    self.assertIn(actual["tiff_header"], (b"II*\x00", b"MM\x00*"))


if __name__ == "__main__":
    print(f"Testing {APP}", flush=True)
    unittest.main(argv=[sys.argv[0]] + UNITTEST_ARGS)
