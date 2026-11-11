"""Normalising the intensities switches the default y-axis title to "Normalized intensity" (and back).

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


class NormalizedTitleTests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_default_title_follows_the_normalisation(self):
        at = run_app()
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")
        normalize_box(at).check().run()
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Normalized intensity")
        report = at.session_state["_regression_report"]
        self.assertEqual(report["ylabel"], "Normalized intensity")             # the saved figure
        self.assertEqual(report["preview_ytitle"], "Normalized intensity")     # the interactive preview
        self.assertEqual(report["session"]["settings"]["y_label"], "Normalized intensity")
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

    def test_a_saved_normalised_project_keeps_its_title(self):
        # Opening a project does not rewrite its title: only switching the normalisation does.
        at = run_app({"normalize": True, "y_label": "Intensity (a.u.)"})
        self.assertNoException(at)
        self.assertEqual(at.text_input(key="y_label").value, "Intensity (a.u.)")


if __name__ == "__main__":
    unittest.main()
