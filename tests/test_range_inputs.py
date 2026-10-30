"""The 2θ ranges can be typed as well as dragged.

Number boxes under the main 2θ range slider and the inset range slider show the limits and apply
typed ones to the slider's value, which projects save and the figure uses. A range whose start is
not below its end is not applied: the boxes return to the slider's limits and a warning shows.
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


def boxes(at, key):
    return at.number_input(key=f"_{key}_from").value, at.number_input(key=f"_{key}_to").value


class RangeInputTests(unittest.TestCase):
    def assertNoException(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_typed_limits_set_the_range_and_reach_the_figure_and_the_project(self):
        at = run_app()
        self.assertNoException(at)
        self.assertEqual(boxes(at, "xrange"), (10.0, 80.0))
        at.number_input(key="_xrange_from").set_value(25.3).run()
        self.assertNoException(at)
        at.number_input(key="_xrange_to").set_value(61.75).run()
        self.assertNoException(at)
        self.assertEqual(tuple(at.session_state["xrange"]), (25.3, 61.75))
        self.assertEqual(tuple(at.slider(key="xrange").value), (25.3, 61.75))
        self.assertEqual(boxes(at, "xrange"), (25.3, 61.75))
        report = at.session_state["_regression_report"]
        self.assertEqual(tuple(report["session"]["settings"]["xrange"]), (25.3, 61.75))
        main_x = [x for _, _, xs, _ in report["main_lines"] for x in xs]
        self.assertTrue(main_x and all(25.3 <= x <= 61.75 for x in main_x))

    def test_dragging_and_the_full_range_button_update_the_boxes(self):
        at = run_app()
        at.slider(key="xrange").set_range(20.0, 60.0).run()
        self.assertNoException(at)
        self.assertEqual(boxes(at, "xrange"), (20.0, 60.0))
        full = next(b for b in at.button if b.label == "測定データの全範囲を表示")
        full.click().run()
        self.assertNoException(at)
        self.assertEqual(boxes(at, "xrange"), tuple(at.session_state["xrange"]))
        self.assertEqual(tuple(at.session_state["xrange"]), (10.0, 40.0))

    def test_a_start_not_below_the_end_is_not_applied(self):
        at = run_app()
        at.number_input(key="_xrange_from").set_value(85.0).run()
        self.assertNoException(at)
        self.assertEqual(tuple(at.session_state["xrange"]), (10.0, 80.0))
        self.assertEqual(boxes(at, "xrange"), (10.0, 80.0))
        self.assertTrue(any("下限は上限より小さく" in w.value for w in at.warning))
        at.run()                                   # the warning is for that change only
        self.assertFalse(any("下限は上限より小さく" in w.value for w in at.warning))

    def test_inset_range_has_the_same_boxes(self):
        at = run_app({"show_zoom": True})
        self.assertNoException(at)
        self.assertEqual(boxes(at, "zoom_range"), (25.0, 35.0))
        at.number_input(key="_zoom_range_to").set_value(31.25).run()
        self.assertNoException(at)
        self.assertEqual(tuple(at.session_state["zoom_range"]), (25.0, 31.25))
        self.assertEqual(tuple(at.slider(key="zoom_range").value), (25.0, 31.25))

    def test_boxes_are_labelled_in_each_language(self):
        expected = {"English": ("From (°)", "To (°)"), "Tiếng Việt": ("Từ (°)", "Đến (°)"), "ไทย": ("จาก (°)", "ถึง (°)")}
        for language, labels in expected.items():
            with self.subTest(language=language):
                at = run_app({"lang_radio": language})
                self.assertNoException(at)
                self.assertEqual((at.number_input(key="_xrange_from").label, at.number_input(key="_xrange_to").label), labels)


if __name__ == "__main__":
    unittest.main()
