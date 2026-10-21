"""Portable project round-trips, validation, and real desktop resume behavior."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_io import (ProjectError, atomic_save, decode_project, encode_project,
                        project_fingerprint)


class ProjectTests(unittest.TestCase):
    def project(self):
        return encode_project(
            {"sample.raw": b"RAW1.01\x00\xff\x00"}, {"phase.cif": b"data_test"},
            {"phase.pdf": b"%PDF-test"},
            {"normalize": True, "xrange": (5., 120.), "_val_lbl_0": "Sample A",
             "xrd_color_0": "#1976d2", "ord_0": 2, "lang_radio": "日本語"},
        )

    def test_original_binary_data_and_settings_roundtrip(self):
        restored = decode_project(self.project())
        self.assertEqual(restored["xrd_files"]["sample.raw"], b"RAW1.01\x00\xff\x00")
        self.assertEqual(restored["settings"]["xrange"], (5., 120.))
        self.assertEqual(restored["settings"]["_val_lbl_0"], "Sample A")
        self.assertEqual(restored["cif_files"], {"phase.cif": b"data_test"})
        self.assertEqual(restored["pdf_files"], {"phase.pdf": b"%PDF-test"})

    def test_old_version_two_and_unrelated_keys(self):
        raw = json.dumps({"version": 2, "xrd_files": {"x.xy": base64.b64encode(b"10 2\n").decode()},
                          "settings": {"normalize": False, "authenticated": True, "_file_epoch": 99}}).encode()
        project = decode_project(raw)
        self.assertEqual(project["xrd_files"]["x.xy"], b"10 2\n")
        self.assertEqual(project["settings"], {"normalize": False})

    def test_corruption_and_bad_settings_are_rejected(self):
        for mutate in (
            lambda p: p["xrd_files"].update({"sample.raw": "not base64!"}),
            lambda p: p["xrd_files"].update({"sample.raw": base64.b64encode(b"different").decode()}),
            lambda p: p["settings"].update(fig_width=100.),
            lambda p: p["settings"].update(xrange=[80., 10.]),
            lambda p: p["settings"].update(normalize="true"),
            lambda p: p.update(version=999),
        ):
            project = json.loads(self.project())
            mutate(project)
            with self.subTest(project=project), self.assertRaises(ProjectError):
                decode_project(json.dumps(project).encode())
        for raw in (b"not-json", b"[]", b"null", b"{\"version\":true}"):
            with self.subTest(raw=raw), self.assertRaises(ProjectError):
                decode_project(raw)

    def test_fingerprint_ignores_only_timestamp(self):
        first = self.project()
        second = json.loads(first)
        second["saved_at"] = "a different time"
        self.assertEqual(project_fingerprint(first), project_fingerprint(json.dumps(second).encode()))
        second["settings"]["ord_0"] = 3
        self.assertNotEqual(project_fingerprint(first), project_fingerprint(json.dumps(second).encode()))

    def test_fingerprint_keeps_source_file_order(self):
        first = encode_project({"a.xy": b"1 2", "b.xy": b"1 3"}, {}, {}, {})
        second = encode_project({"b.xy": b"1 3", "a.xy": b"1 2"}, {}, {}, {})
        self.assertNotEqual(project_fingerprint(first), project_fingerprint(second))

    def test_atomic_save_keeps_last_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "projects" / "autosave.json"
            atomic_save(target, b"previous")
            with patch("project_io.os.replace", side_effect=OSError("disk problem")):
                with self.assertRaises(OSError):
                    atomic_save(target, b"incomplete")
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(target.parent.iterdir()), [target])
            atomic_save(target, b"complete")
            self.assertEqual(target.read_bytes(), b"complete")


class DesktopResumeTests(unittest.TestCase):
    def test_resume_edit_and_reopen_without_reloading_old_settings(self):
        from streamlit.testing.v1 import AppTest
        app = Path(__file__).resolve().parents[1] / "app.py"
        original = {"native.ras": b'*RAS_DATA_START\n*RAS_HEADER_START\n*MEAS_SCAN_AXIS_X "TwoTheta"\n*RAS_HEADER_END\n*RAS_INT_START\n10 1 1\n20 4 1\n30 2 1\n*RAS_INT_END\n*RAS_DATA_END\n'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autosave.json"
            atomic_save(path, encode_project(original, {}, {}, {
                "normalize": True, "show_panel": False, "fig_width": 6., "fig_height": 4.,
                "_val_lbl_0": "Restored sample", "xrd_color_0": "#1976d2",
            }))
            with patch.dict("os.environ", {"XRD_MAKER_LOCAL": "1", "XRD_MAKER_DATA_DIR": directory}):
                at = AppTest.from_file(str(app), default_timeout=60).run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                resume = next(button for button in at.button if "自動保存から再開" in button.label)
                resume.click().run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                self.assertEqual(at.session_state["_xrd_bytes"], original)
                self.assertEqual(at.session_state["_val_lbl_0"], "Restored sample")
                at.checkbox(key="normalize").uncheck().run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                at.run()
                self.assertFalse(at.checkbox(key="normalize").value)
                self.assertFalse(decode_project(path.read_bytes())["settings"]["normalize"])
                reopened = AppTest.from_file(str(app), default_timeout=60).run()
                next(button for button in reopened.button if "自動保存から再開" in button.label).click().run()
                self.assertFalse(reopened.exception, [e.message for e in reopened.exception])
                self.assertFalse(reopened.checkbox(key="normalize").value)
                self.assertEqual(reopened.session_state["_val_lbl_0"], "Restored sample")


if __name__ == "__main__":
    unittest.main()
