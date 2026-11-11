"""After an update pulled into a running server, the app reloads its own modules whose files changed.

On Streamlit Cloud the new app.py ran against a module imported before the update:
"ImportError: cannot import name 'COLUMN_PALETTES' from 'palettes'" right after the column palettes came.
"""

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import test_xrd_regressions as regression


def refresh_functions():
    ns = regression.renderer_namespace()
    ns.update(sys=sys, importlib=importlib, os=os)
    return ns["refresh_local_modules"], ns["record_local_modules"]


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.refresh, self.record = refresh_functions()
        self.folder = Path(tempfile.mkdtemp())
        for name, text in (("xrd_probe_base", "A = 1\n"), ("xrd_probe_user", "from xrd_probe_base import A\n")):
            (self.folder / f"{name}.py").write_text(text, encoding="utf-8")
            self.addCleanup(sys.modules.pop, name, None)
        sys.path.insert(0, str(self.folder))
        self.addCleanup(sys.path.remove, str(self.folder))
        self.names = ("xrd_probe_base", "xrd_probe_user")

    def change(self, name, text):
        path = self.folder / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        later = path.stat().st_mtime + 10
        os.utime(path, (later, later))

    def test_first_check_reloads_once_then_only_after_a_change(self):
        importlib.import_module("xrd_probe_user")           # imports xrd_probe_base
        old_base = sys.modules["xrd_probe_base"]
        stamps = {}
        self.assertEqual(self.refresh(self.names, stamps), list(self.names))    # first time in this server
        self.record(self.names, stamps)
        self.assertEqual(self.refresh(self.names, stamps), [])                  # nothing changed
        self.record(self.names, stamps)
        self.change("xrd_probe_base", "A = 1\nB = 2\n")
        self.assertEqual(self.refresh(self.names, stamps), list(self.names))    # both, in import order
        self.assertEqual(sys.modules["xrd_probe_base"].B, 2)
        self.assertFalse(hasattr(old_base, "B"))                                # earlier references keep working

    def test_modules_that_take_names_from_a_changed_one_are_reloaded_too(self):
        importlib.import_module("xrd_probe_user")
        stamps = {}
        self.record(self.names, stamps)
        self.change("xrd_probe_base", "A = 'new'\n")
        self.refresh(self.names, stamps)
        self.assertEqual(sys.modules["xrd_probe_user"].A, "new")

    def test_modules_not_loaded_yet_are_left_to_the_import(self):
        stamps = {"__recorded__": True}
        self.assertEqual(self.refresh(self.names, stamps), [])
        self.assertNotIn("xrd_probe_base", sys.modules)

    def test_a_palettes_module_from_before_the_update_gets_the_new_names(self):
        original = sys.modules["palettes"]
        stale = type(original)("palettes")             # as loaded before the column palettes existed
        stale.__file__ = original.__file__
        stale.PALETTES = original.PALETTES
        sys.modules["palettes"] = stale
        stamps = {"palettes": (0, 0), "__recorded__": True}   # and its file has changed since
        try:
            self.assertEqual(self.refresh(("palettes",), stamps), ["palettes"])
            from palettes import COLUMN_PALETTES       # the line that failed on Streamlit Cloud
            self.assertEqual(COLUMN_PALETTES, original.COLUMN_PALETTES)
        finally:
            sys.modules["palettes"] = original


class AppStartsAfterAnUpdateTests(unittest.TestCase):
    def run_app(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
        for key, value in {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": dict(regression.XRD_BYTES),
                           "_cif_bytes": {}, "_pdf_bytes": {}}.items():
            at.session_state[key] = value
        at.run()
        return at

    def test_the_app_runs_when_an_old_module_is_still_loaded(self):
        self.run_app()                                   # the server has loaded and recorded its modules
        current = sys.modules["palettes"]
        stale = type(current)("palettes")              # an update came; the module in memory is the old one
        stale.__file__ = current.__file__
        stale.PALETTES = current.PALETTES
        sys.modules["palettes"] = stale
        sys.__dict__["_xrd_maker_module_stamps"]["palettes"] = (0, 0)
        at = self.run_app()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertTrue(hasattr(sys.modules["palettes"], "COLUMN_PALETTES"))


if __name__ == "__main__":
    unittest.main()
