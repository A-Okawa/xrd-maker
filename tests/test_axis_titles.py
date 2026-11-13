"""Default axis titles: 2θ (°) with an italic θ, and Intensity (a.u.), which turns into
Normalized intensity (a.u.) when the intensities are normalised (and back).

Only the default titles are swapped: a title the user wrote, or an empty one (no title), stays.
"""

from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import test_xrd_regressions as regression


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


def normalize_box(at):
    return next(c for c in at.checkbox if c.key == "normalize")


class AxisTitleTests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_default_x_title_is_two_theta_in_degrees_with_an_italic_theta(self):
        at = run_app()
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="x_label").value, "2<i>θ</i> (°)")
        report = at.session_state["_regression_report"]
        self.assertEqual(set(report["xlabels"]), {r"2$\mathit{θ}$ (°)"})       # the saved figure: θ in italics
        self.assertEqual(set(report["preview_xtitles"]), {"2<i>θ</i> (°)"})    # the interactive preview
        self.assertEqual(report["session"]["settings"]["x_label"], "2<i>θ</i> (°)")

    def test_default_y_title_follows_the_normalisation(self):
        at = run_app()
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")
        normalize_box(at).check().run()
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Normalized intensity (a.u.)")
        report = at.session_state["_regression_report"]
        self.assertEqual(report["ylabel"], "Normalized intensity (a.u.)")             # the saved figure
        self.assertEqual(report["preview_ytitle"], "Normalized intensity (a.u.)")     # the interactive preview
        self.assertEqual(report["session"]["settings"]["y_label"], "Normalized intensity (a.u.)")
        normalize_box(at).uncheck().run()
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")

    def test_the_earlier_normalised_title_goes_back_too(self):
        # A project saved before "(a.u.)" was added holds "Normalized intensity".
        at = run_app({"normalize": True, "y_label": "Normalized intensity"})
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Normalized intensity")
        normalize_box(at).uncheck().run()
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")

    def test_titles_written_by_the_user_stay(self):
        for title in ("Counts (cps)", ""):
            with self.subTest(title=title):
                at = run_app({"y_label": title})
                normalize_box(at).check().run()
                self.assertNoException(at)
                self.assertEqual(at.text_input(key="y_label").value, title)
                normalize_box(at).uncheck().run()
                self.assertEqual(at.text_input(key="y_label").value, title)

    def test_saved_projects_keep_their_titles(self):
        # Opening a project does not rewrite its titles: only switching the normalisation changes the y title.
        at = run_app({"normalize": True, "y_label": "Intensity (a.u.)",
                      "x_label": "Diffraction angle, 2<i>θ</i> (deg.)"})
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")
        self.assertEqual(at.text_input(key="x_label").value, "Diffraction angle, 2<i>θ</i> (deg.)")


if __name__ == "__main__":
    unittest.main()
