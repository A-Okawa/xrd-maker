"""Publication workflows across styles, project resume and the real renderers.

Synthetic data only. The UI harness bypasses authentication using session state
and replaces CIF/PDF parsers with deterministic, in-memory reference peaks.
"""

import ast
import copy
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from figure_style import STYLE_DEFAULTS, export_figure, mm_to_inches, resolve_font
from project_io import ProjectError, clean_settings, decode_project, encode_project
from style_templates import TEMPLATE_KEYS, decode_style_template, encode_style_template
from plot_annotations import format_hkl_groups
import test_xrd_regressions as regression


PROTECTED = {
    "_xrd_bytes": dict(regression.XRD_BYTES),
    "_cif_bytes": {"phase.cif": b"VALID"},
    "_pdf_bytes": {"card.pdf": b"VALID"},
    "normalize": True, "xrange": (10.0, 40.0), "global_offset": 1.25,
    "ord_0": 3, "ord_1": 1, "ord_2": 2,
    "vis_0": True, "vis_1": False, "vis_2": True,
    "_val_lbl_0": "Sample A", "_val_lbl_1": "Hidden B", "_val_lbl_2": "Sample C",
    "xrd_color_0": "#b71c1c", "xrd_color_1": "#1976d2", "xrd_color_2": "#00897b",
    "extra_offset_0": 0.35, "extra_offset_1": -0.2, "extra_offset_2": 1.1,
    "cord_0": 2, "cvis_0": True, "_val_clbl_0": "CIF phase",
    "cif_color_0": "#8e24aa", "cref_line_0": True,
    "pord_0": 1, "pvis_0": True, "_val_plbl_0": "PDF card",
    "pdf_color_0": "#43a047", "pref_line_0": False,
}
PUBLICATION = {
    **STYLE_DEFAULTS,
    "size_unit": "mm", "figure_preset": "custom",
    "fig_width": mm_to_inches(85), "fig_height": mm_to_inches(65),
    "font_family": "DejaVu Sans", "font_size": 11, "dpi_export": 300,
    "line_width": 0.9, "reference_line_width": 0.7, "axes_line_width": 0.6,
    "margin_left": 0.17, "margin_right": 0.96,
    "margin_bottom": 0.16, "margin_top": 0.94,
    "show_phase_markers": True, "show_hkl": True, "phase_tolerance": 0.2,
    "annotation_font_size": 8, "show_zoom": True,
    "zoom_range": (14.0, 28.0), "zoom_location": "upper right",
}


def publication_namespace(normalize=True):
    ns = regression.renderer_namespace(normalize)
    ns.update(PUBLICATION)
    ns["normalize"] = normalize
    ns["plot_style"] = dict(PUBLICATION)
    ns["resolved_font"] = resolve_font(PUBLICATION["font_family"])
    ns["st"].session_state.update(copy.deepcopy({**PROTECTED, **PUBLICATION}))
    ns["calc_cif_details"] = lambda *args, **kwargs: (
        np.array([20., 30.]), np.array([100., 50.]), ["(1 0 0)", "(1 1 0)"],
    )
    ns["calc_cif_pattern"] = lambda *args, **kwargs: (
        np.array([20., 30.]), np.array([100., 50.]),
    )
    ns["parse_pdf_card"] = lambda *args, **kwargs: (
        np.array([20.1, 35.]), np.array([100., 25.]),
    )
    ns.update(
        active_cif=[ns["RestoredFile"]("phase.cif", b"VALID")],
        cif_sort_idx=[0], cif_visibles=[True], cif_labels=["CIF phase"],
        cif_colors=["#8e24aa"], cif_orders=[2], cif_ref_lines=[True],
        active_pdf=[ns["RestoredFile"]("card.pdf", b"VALID")],
        pdf_sort_idx=[0], pdf_visibles=[True], pdf_labels=["PDF card"],
        pdf_colors=["#43a047"], pdf_orders=[1], pdf_ref_lines=[False],
    )
    return ns


def assert_physical_exports(test, exports, width_inches, height_inches, dpi, *, png_dpi=None):
    for format in ("png", "tiff"):
        with Image.open(io.BytesIO(exports[format])) as image:
            expected_dpi = png_dpi if format == "png" and png_dpi is not None else dpi
            test.assertEqual(image.size, (int(width_inches * expected_dpi), int(height_inches * expected_dpi)))
            test.assertAlmostEqual(image.info["dpi"][0], expected_dpi, delta=.1)
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", exports["pdf"])
    test.assertIsNotNone(box)
    test.assertAlmostEqual(float(box[1]), width_inches * 72, places=7)
    test.assertAlmostEqual(float(box[2]), height_inches * 72, places=7)
    svg = ET.fromstring(exports["svg"])
    test.assertAlmostEqual(float(svg.get("width").removesuffix("pt")), width_inches * 72, places=5)
    test.assertAlmostEqual(float(svg.get("height").removesuffix("pt")), height_inches * 72, places=5)
    test.assertTrue(svg.findall(".//{http://www.w3.org/2000/svg}text"))


def publication_ui_source():
    tree = ast.parse(regression.ui_test_source())
    mock_functions = {node.name: node for node in ast.parse('''
def calc_cif_details(cif_bytes, two_theta_range=(5, 90)):
    return np.array([20., 30.]), np.array([100., 50.]), ["(1 0 0)", "(1 1 0)"]
def calc_cif_pattern(cif_bytes, two_theta_range=(5, 90)):
    return np.array([20., 30.]), np.array([100., 50.])
def parse_pdf_card(pdf_bytes, two_theta_range=(5, 90)):
    return np.array([20.1, 35.]), np.array([100., 25.])
''').body}
    tree.body = [copy.deepcopy(mock_functions.get(node.name, node))
                 if isinstance(node, ast.FunctionDef) else node for node in tree.body]

    class InstrumentDownloads(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "download_button":
                    node.func = ast.Name(id="_publication_download", ctx=ast.Load())
                elif node.func.attr in ("radio", "selectbox"):
                    # Streamlit 1.32 AppTest serializes formatted selections
                    # incorrectly. Only presentation formatting is omitted.
                    node.keywords = [kw for kw in node.keywords if kw.arg != "format_func"]
            return node

    tree = ast.fix_missing_locations(InstrumentDownloads().visit(tree))
    preamble = '''
_publication_downloads = {}
def _publication_download(*args, **kwargs):
    name = kwargs.get("file_name", "")
    if name.startswith("xrd_result."):
        _publication_downloads[name.rsplit(".", 1)[1]] = kwargs.get("data", args[1] if len(args) > 1 else None)
    return st.sidebar.download_button(*args, **kwargs)
'''
    capture = '''
if active_xrd:
    st.session_state["_publication_report"] = {
        "downloads": dict(_publication_downloads),
        "canvas": tuple(fig.get_size_inches()),
    }
'''
    return preamble + ast.unparse(tree) + "\n" + capture


class StyleTemplateWorkflowTests(unittest.TestCase):
    def test_template_roundtrip_changes_appearance_without_measurement_settings(self):
        raw = encode_style_template({**PROTECTED, **PUBLICATION}, "Publication 85 mm")
        restored = decode_style_template(raw)
        self.assertEqual(json.loads(raw)["name"], "Publication 85 mm")
        self.assertEqual(set(restored), set(PUBLICATION).intersection(TEMPLATE_KEYS))
        for key in set(PROTECTED) - set(TEMPLATE_KEYS):
            self.assertNotIn(key, restored)
        target = copy.deepcopy(PROTECTED)
        target.update(restored)
        for key, value in PROTECTED.items():
            self.assertEqual(target[key], value, key)
        self.assertEqual(restored["fig_width"], mm_to_inches(85))
        self.assertEqual(restored["line_width"], 0.9)

    def test_style_aliases_are_canonical_in_templates_and_projects(self):
        raw_style = encode_style_template({"size_unit": "in", "font_family": "TimesNewRoman"})
        expected = {"size_unit": "inch", "font_family": "Times New Roman"}
        self.assertEqual(decode_style_template(raw_style), expected)
        encoded = encode_project(PROTECTED["_xrd_bytes"], {}, {}, {
            **expected, "size_unit": "in", "font_family": "TimesNewRoman",
            "fig_width": 10.0, "fig_height": 8.0,
        })
        restored = decode_project(encoded)
        for key, value in expected.items():
            self.assertEqual(restored["settings"][key], value)
        self.assertEqual(restored["settings"]["fig_width"], 10.0)

    def test_template_rejects_data_keys_invalid_margins_and_out_of_bounds_sizes(self):
        base = {"format": "xrd-maker-style", "version": 1, "settings": {}}
        for settings in (
            {"ord_0": 2}, {"_val_lbl_0": "injected label"}, {"normalize": False},
            {"zoom_range": [15, 20]}, {"margin_left": .9, "margin_right": .8},
            {"fig_width": .5}, {"fig_height": 21}, {"font_size": 29},
        ):
            with self.subTest(settings=settings), self.assertRaises(ProjectError):
                decode_style_template(json.dumps({**base, "settings": settings}).encode())

    def test_project_roundtrip_includes_all_publication_controls_and_native_bytes(self):
        settings = {k: v for k, v in {**PROTECTED, **PUBLICATION}.items() if not k.endswith("_bytes")}
        raw = encode_project(PROTECTED["_xrd_bytes"], PROTECTED["_cif_bytes"], PROTECTED["_pdf_bytes"], settings)
        restored = decode_project(raw)
        self.assertEqual(restored["xrd_files"], PROTECTED["_xrd_bytes"])
        self.assertEqual(restored["cif_files"], PROTECTED["_cif_bytes"])
        self.assertEqual(restored["pdf_files"], PROTECTED["_pdf_bytes"])
        self.assertEqual(restored["settings"], clean_settings(settings))


class PublicationCallbackTests(unittest.TestCase):
    def test_style_template_and_dimension_callbacks_preserve_patterns(self):
        ns = regression.renderer_namespace()
        state = ns["st"].session_state
        state.clear()
        state.update(copy.deepcopy(PROTECTED))
        state.update(figure_preset="custom", size_unit="mm", fig_width=10., fig_height=8.)
        payload = encode_style_template(PUBLICATION)
        ns.update(decode_style_template=decode_style_template, clean_settings=clean_settings)
        state["_style_upload"] = SimpleNamespace(getvalue=lambda: payload)
        ns["_open_style_template"]()
        self.assertNotIn("_style_error", state)
        self.assertTrue(state["_style_loaded"])
        for key, value in decode_style_template(payload).items():
            self.assertEqual(state[key], value, key)
        state["figure_preset"] = "single"
        ns["_apply_figure_preset"]()
        self.assertAlmostEqual(state["fig_width"], mm_to_inches(85))
        self.assertAlmostEqual(state["fig_height"], mm_to_inches(70))
        state["_figure_width"] = 92.5
        ns["_update_figure_dimension"]("width")
        self.assertAlmostEqual(state["fig_width"], mm_to_inches(92.5))
        self.assertEqual(state["figure_preset"], "custom")
        for key, value in PROTECTED.items():
            self.assertEqual(state[key], value, key)

    def test_partial_template_conflict_is_atomic_and_keeps_patterns(self):
        ns = regression.renderer_namespace()
        ns.update(decode_style_template=decode_style_template, clean_settings=clean_settings)
        state = ns["st"].session_state
        state.update(copy.deepcopy(PROTECTED))
        state.update(margin_left=.2, margin_right=.8, font_size=11)
        before = copy.deepcopy(dict(state))
        payload = encode_style_template({"margin_left": .9, "font_size": 13})
        state["_style_upload"] = SimpleNamespace(getvalue=lambda: payload)
        ns["_open_style_template"]()
        self.assertIn("_style_error", state)
        self.assertNotIn("_style_loaded", state)
        for key, value in before.items():
            self.assertEqual(state[key], value, key)


class PublicationRendererWorkflowTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_real_app_figure_exports_exact_canvas_in_all_four_formats(self):
        ns = publication_namespace()
        fig = ns["build_figure"]()
        self.assertEqual(tuple(fig.get_size_inches()), (PUBLICATION["fig_width"], PUBLICATION["fig_height"]))
        # Even a global tight-save preference must not crop this app figure.
        with matplotlib.rc_context({"savefig.bbox": "tight"}):
            exports = {kind: export_figure(fig, kind, dpi=300, settings=PUBLICATION)
                       for kind in ("png", "tiff", "pdf", "svg")}
        assert_physical_exports(self, exports, PUBLICATION["fig_width"], PUBLICATION["fig_height"], 300)

    def test_cif_metadata_retains_all_reflection_families_after_range_filter(self):
        ns = regression.renderer_namespace()
        paths = []

        class SyntheticParser:
            def __init__(self, path):
                paths.append(Path(path))

            def parse_structures(self, primitive=False):
                return [object()]

        class SyntheticCalculator:
            def __init__(self, **kwargs):
                pass

            def get_pattern(self, structure, two_theta_range):
                return SimpleNamespace(
                    x=np.array([5., 20., 45.]), y=np.array([15., 100., 30.]),
                    hkls=[[{"hkl": (0, 0, 1)}],
                          [{"hkl": (1, 0, 0)}, {"hkl": (0, 1, 0)}, {"hkl": (-1, 0, 1)}],
                          [{"hkl": (1, 1, 0)}]],
                )

        ns.update(PYMATGEN_AVAILABLE=True, CifParser=SyntheticParser,
                  XRDCalculator=SyntheticCalculator, tempfile=tempfile,
                  format_hkl_groups=format_hkl_groups)
        angles, intensities, labels = ns["calc_cif_details"](b"data_synthetic\n", (10., 40.))
        np.testing.assert_array_equal(angles, [20.])
        np.testing.assert_array_equal(intensities, [100.])
        self.assertEqual(labels, ["(1 0 0) / (0 1 0) / (-1 0 1)"])
        self.assertTrue(paths)
        self.assertFalse(paths[0].exists(), "The CIF temporary file should be removed.")

    def test_phase_markers_and_cif_hkl_labels_match_export_and_preview_coordinates(self):
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                ns = publication_namespace(normalize)
                fig, pfig = ns["build_figure"](), ns["build_plotly_figure"]()
                marker_lines = [line for line in fig.axes[0].lines
                                if line.get_marker() in ("o", "^", "s", "D")]
                marker_traces = [trace for trace in pfig.data
                                 if trace.mode == "markers" and (trace.legendgroup or "").startswith("phase:")]
                mpl_points = sorted((float(x), float(y), line.get_color())
                                    for line in marker_lines
                                    for x, y in zip(line.get_xdata(), line.get_ydata()))
                plotly_points = sorted((float(x), float(y), trace.marker.color)
                                       for trace in marker_traces for x, y in zip(trace.x, trace.y))
                self.assertEqual(len(mpl_points), 4)
                self.assertEqual(mpl_points, plotly_points)
                expected_y = [2.1, 2.6] if normalize else [44., 128.]
                np.testing.assert_allclose(sorted(y for _, y, _ in mpl_points), sorted(expected_y * 2))
                self.assertTrue(all(x == 20. for x, _, _ in mpl_points))
                mpl_hkl = sorted((text.get_text(), tuple(text.xy)) for text in fig.axes[1].texts
                                 if text.get_text().startswith("("))
                plotly_hkl = sorted((annotation.text, (annotation.x, annotation.y))
                                    for annotation in pfig.layout.annotations
                                    if annotation.xref == "x2" and annotation.text.startswith("("))
                sticks = next(collection for collection in fig.axes[1].collections
                              if collection.get_label() == "CIF phase").get_segments()
                expected_hkl = [("(1 0 0)", tuple(sticks[0][1])), ("(1 1 0)", tuple(sticks[1][1]))]
                self.assertEqual([xy[0] for _, xy in expected_hkl], [20., 30.])
                self.assertEqual(mpl_hkl, expected_hkl)
                self.assertEqual(mpl_hkl, plotly_hkl)

    def test_inset_keeps_prepared_values_and_reference_subplot_in_both_renderers(self):
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                ns = publication_namespace(normalize)
                ns["show_zoom"] = False
                ns["st"].session_state["show_zoom"] = False
                before_mpl = ns["build_figure"]()
                before_plotly = ns["build_plotly_figure"]()
                ns["show_zoom"] = True
                ns["st"].session_state["show_zoom"] = True
                after_mpl = ns["build_figure"]()
                after_plotly = ns["build_plotly_figure"]()
                np.testing.assert_allclose(after_mpl.axes[0].get_ylim(), before_mpl.axes[0].get_ylim())
                self.assertEqual(len(after_mpl.axes[0].child_axes), 1)
                inset = after_mpl.axes[0].child_axes[0]
                np.testing.assert_allclose(inset.get_xlim(), PUBLICATION["zoom_range"])
                inset_traces = [t for t in after_plotly.data
                                if t.xaxis == "x3" and t.name != "_zoom_background"]
                self.assertEqual(len(inset.lines), 2)
                self.assertEqual([line.get_label() for line in inset.lines], [t.name for t in inset_traces])
                for line, trace in zip(inset.lines, inset_traces):
                    np.testing.assert_array_equal(line.get_xdata(), trace.x)
                    np.testing.assert_allclose(line.get_ydata(), trace.y)
                np.testing.assert_allclose(after_plotly.layout.xaxis3.range, inset.get_xlim())
                np.testing.assert_allclose(after_plotly.layout.yaxis3.range, inset.get_ylim())
                for suffix in ("", "2"):
                    old_axis = getattr(before_plotly.layout, "xaxis" + suffix)
                    new_axis = getattr(after_plotly.layout, "xaxis" + suffix)
                    self.assertEqual(new_axis.range, old_axis.range)
                    self.assertEqual(new_axis.domain, old_axis.domain)
                old_refs = [t.to_plotly_json() for t in before_plotly.data if t.xaxis == "x2"]
                new_refs = [t.to_plotly_json() for t in after_plotly.data if t.xaxis == "x2"]
                self.assertEqual(len(old_refs), len(new_refs))
                for old, new in zip(old_refs, new_refs):
                    self.assertEqual(old.get("name"), new.get("name"))
                    np.testing.assert_array_equal(old["x"], new["x"])
                    np.testing.assert_array_equal(old["y"], new["y"])
                old_sticks, new_sticks = before_mpl.axes[1].collections, after_mpl.axes[1].collections
                self.assertEqual(len(new_sticks), 2)
                for old, new in zip(old_sticks, new_sticks):
                    self.assertEqual(old.get_label(), new.get_label())
                    np.testing.assert_array_equal(old.get_segments(), new.get_segments())


class PublicationResumeWorkflowTests(unittest.TestCase):
    def test_edit_mm_canvas_autosave_and_resume_preserve_publication_controls(self):
        from streamlit.testing.v1 import AppTest
        source = publication_ui_source()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
            "XRD_MAKER_LOCAL": "1", "XRD_MAKER_DATA_DIR": directory,
        }):
            at = AppTest.from_string(source, default_timeout=60)
            for key, value in {**copy.deepcopy(PROTECTED), **PUBLICATION,
                               "authenticated": True, "lang_radio": "日本語",
                               "show_panel": False, "show_ref_lines": True}.items():
                at.session_state[key] = value
            at.run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            before = copy.deepcopy(at.session_state["_regression_report"])
            self.assertEqual(before["sort_idx"], [1, 2, 0])
            self.assertEqual([r[0] for r in before["refs"]], ["PDF card", "CIF phase"])
            at.number_input(key="_figure_width").set_value(92.5).run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            self.assertAlmostEqual(at.session_state["fig_width"], mm_to_inches(92.5))
            self.assertEqual(at.session_state["figure_preset"], "custom")
            edited = copy.deepcopy(at.session_state["_regression_report"])
            for field in ("sort_idx", "orders", "visibles", "labels", "colors", "refs", "main_lines", "preview_lines"):
                self.assertEqual(edited[field], before[field], field)
            path = Path(directory) / "autosave.json"
            restored_project = decode_project(path.read_bytes())
            for key, value in PUBLICATION.items():
                expected = mm_to_inches(92.5) if key == "fig_width" else value
                self.assertEqual(restored_project["settings"][key], expected, key)
            expected_width, expected_height = mm_to_inches(92.5), PUBLICATION["fig_height"]
            downloads = at.session_state["_publication_report"]["downloads"]
            self.assertEqual(set(downloads), {"png", "tiff", "pdf", "svg"})
            assert_physical_exports(self, downloads, expected_width, expected_height, 300)
            reopened = AppTest.from_string(source, default_timeout=60)
            reopened.session_state["authenticated"] = True
            reopened.run()
            self.assertFalse(reopened.exception, [e.message for e in reopened.exception])
            resume = next(button for button in reopened.button if "自動保存から再開" in button.label)
            resume.click().run()
            self.assertFalse(reopened.exception, [e.message for e in reopened.exception])
            self.assertEqual(reopened.session_state["_xrd_bytes"], PROTECTED["_xrd_bytes"])
            self.assertEqual(reopened.session_state["_cif_bytes"], PROTECTED["_cif_bytes"])
            self.assertEqual(reopened.session_state["_pdf_bytes"], PROTECTED["_pdf_bytes"])
            for key, value in PUBLICATION.items():
                expected = mm_to_inches(92.5) if key == "fig_width" else value
                self.assertEqual(reopened.session_state[key], expected, key)
            resumed = reopened.session_state["_regression_report"]
            for field in ("sort_idx", "orders", "visibles", "labels", "colors", "refs", "main_lines", "preview_lines"):
                self.assertEqual(resumed[field], edited[field], field)
            resumed_downloads = reopened.session_state["_publication_report"]["downloads"]
            assert_physical_exports(self, resumed_downloads, expected_width, expected_height, 300)


if __name__ == "__main__":
    unittest.main()
