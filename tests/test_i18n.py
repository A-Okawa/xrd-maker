"""Vietnamese and Thai screens, and the width keyword that suits the installed Streamlit.

Every screen text has a Vietnamese and a Thai version (keyed texts and inline ui(ja, en) texts alike),
with the same {placeholders}. The language switch lists 日本語, English, Tiếng Việt, ไทย. Opening a project
keeps the screen language; projects record only Japanese or English, which earlier versions accept.
"""

import ast
import inspect
from pathlib import Path
import re
import string
import sys
import unittest

import streamlit

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from i18n import COLOR_NAME_TEXT, INLINE_TEXT, KEYED_TEXT, LANGUAGES, color_name
from palettes import NAMED_COLORS, PALETTE_NAMES, PALETTES, parse_colors
from project_io import ProjectError, clean_settings, decode_project, encode_project
import test_xrd_regressions as regression

TREE = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
THAI = re.compile(r"[฀-๿]")
VIETNAMESE = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.IGNORECASE)
SYMBOLS_ONLY = {"+x₂", "+x²", "Okabe–Ito", "viridis"}
WIDGETS = ("button", "download_button", "popover", "plotly_chart")
DISPLAY = {"title", "header", "subheader", "markdown", "caption", "success", "info", "warning", "error"}


def app_translations():
    for node in ast.walk(TREE):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRANSLATIONS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("TRANSLATIONS not found")


def ui_calls():
    return [node for node in ast.walk(TREE) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ui"]


def fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}


def in_language(code, text):
    if text in SYMBOLS_ONLY:
        return True
    return bool(THAI.search(text)) if code == "th" else bool(VIETNAMESE.search(text))


class TranslationTableTests(unittest.TestCase):
    def test_switch_lists_japanese_english_vietnamese_thai(self):
        self.assertEqual(list(LANGUAGES), ["日本語", "English", "Tiếng Việt", "ไทย"])
        self.assertEqual(list(LANGUAGES.values()), ["ja", "en", "vi", "th"])

    def test_keyed_texts_are_complete_with_the_same_placeholders(self):
        tables = app_translations()
        self.assertEqual(set(tables["ja"]), set(tables["en"]))
        for code in ("vi", "th"):
            with self.subTest(code=code):
                self.assertEqual(set(KEYED_TEXT[code]), set(tables["en"]))
                for key, text in KEYED_TEXT[code].items():
                    self.assertEqual(fields(text), fields(tables["en"][key]), key)
                    self.assertTrue(text.strip() and not JAPANESE.search(text), key)
                    self.assertTrue(in_language(code, text), f"{key}: {text}")
                    # Markdown headings and bold stay markdown; trailing ': ' stays for the appended detail.
                    self.assertEqual(text.startswith("####"), tables["en"][key].startswith("####"), key)
                    self.assertEqual(text.endswith(": "), tables["en"][key].endswith(": "), key)

    def test_every_inline_text_is_translated_and_none_is_left_over(self):
        english = set()
        for call in ui_calls():
            # Literal texts only: an f-string would be looked up with its values filled in and never be found.
            self.assertEqual(len(call.args), 2, ast.unparse(call))
            self.assertTrue(all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in call.args),
                            f"line {call.lineno}: {ast.unparse(call)[:80]}")
            ja, en = (a.value for a in call.args)
            self.assertEqual(fields(ja), fields(en), call.lineno)
            english.add(en)
        for code in ("vi", "th"):
            with self.subTest(code=code):
                self.assertEqual(set(INLINE_TEXT[code]), english)
                for en, text in INLINE_TEXT[code].items():
                    self.assertEqual(fields(text), fields(en), en)
                    self.assertTrue(text.strip() and not JAPANESE.search(text), en)
                    self.assertTrue(in_language(code, text), f"{en}: {text}")

    def test_palette_names_in_every_language(self):
        for code in LANGUAGES.values():
            self.assertEqual(set(PALETTE_NAMES[code]), set(PALETTES), code)

    def test_colour_names_of_the_graph_app_are_shown_translated(self):
        shipped = parse_colors((APP_DIR / "colors.csv").read_text(encoding="utf-8-sig"))
        japanese_names = {name for name in [*shipped, *NAMED_COLORS] if JAPANESE.search(name)}
        for code in ("en", "vi", "th"):
            with self.subTest(code=code):
                self.assertEqual(set(COLOR_NAME_TEXT[code]), japanese_names)
                for name in japanese_names:
                    shown = color_name(name, code)
                    self.assertFalse(JAPANESE.search(shown), shown)
                    if code != "en":
                        self.assertTrue(in_language(code, shown) or shown.isascii(), shown)
        self.assertEqual(color_name("青", "ja"), "青")
        self.assertEqual(color_name("750 °C", "th"), "750 °C")          # the user's own names stay as written


class ProjectLanguageTests(unittest.TestCase):
    def test_projects_accept_the_four_languages(self):
        for name in LANGUAGES:
            self.assertEqual(clean_settings({"lang_radio": name}), {"lang_radio": name})
        with self.assertRaises(ProjectError):
            clean_settings({"lang_radio": "Français"})

    def test_saving_records_only_japanese_or_english_and_opening_keeps_the_screen_language(self):
        ns = regression.renderer_namespace()
        state = ns["st"].session_state
        for name, recorded in (("日本語", True), ("English", True), ("Tiếng Việt", False), ("ไทย", False)):
            with self.subTest(language=name):
                state.clear()
                state.update({"lang_radio": name, "palette": "blues", "_xrd_bytes": {"a.xy": b"10 1\n20 2\n"}})
                settings = decode_project(ns["build_session_json"](1, 0, 0))["settings"]
                self.assertEqual("lang_radio" in settings, recorded)
                self.assertEqual(settings["palette"], "blues")
        state.clear()
        state.update({"lang_radio": "ไทย", "palette": "blues"})
        ns["restore_session"](encode_project({"a.xy": b"10 1\n20 2\n"}, {}, {},
                                             {"lang_radio": "English", "palette": "greens"}))
        self.assertEqual(state["lang_radio"], "ไทย")
        self.assertEqual(state["palette"], "greens")


class FullWidthTests(unittest.TestCase):
    def test_width_keyword_suits_the_installed_streamlit(self):
        ns = regression.renderer_namespace()
        ns["inspect"] = inspect
        for name in WIDGETS:
            parameter = inspect.signature(getattr(streamlit, name)).parameters.get("width")
            new_api = parameter is not None and isinstance(parameter.default, str)
            expected = {"width": "stretch"} if new_api else {"use_container_width": True}
            self.assertEqual(ns["_full_width"](getattr(streamlit, name)), expected, name)
        if tuple(int(v) for v in re.findall(r"\d+", streamlit.__version__)[:2]) >= (1, 50):
            # Where use_container_width is deprecated, no widget passes it (no warnings in the log).
            for name in WIDGETS:
                self.assertEqual(ns["_full_width"](getattr(streamlit, name)), {"width": "stretch"}, name)

    def test_no_widget_passes_use_container_width_directly(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Call):
                self.assertNotIn("use_container_width", [kw.arg for kw in node.keywords], ast.unparse(node)[:80])


def run_app(language, extra=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=90)
    state = {"authenticated": True, "lang_radio": language, "_xrd_bytes": dict(regression.XRD_BYTES),
             "_cif_bytes": {"cif1.cif": b"VALID"}, "_pdf_bytes": {}, "show_panel": True,
             "show_cif_legend": True, "show_side_labels": True}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def screen_texts(at):
    """Labels, help, captions, markdown and messages on the screen. The language switch is left out: it names
    every language in its own script (日本語, ไทย …) on every screen."""
    texts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "button", "checkbox", "radio", "selectbox",
                 "slider", "number_input", "text_input", "color_picker", "tabs", "expander", "success", "info",
                 "warning", "error", "toggle", "multiselect"):
        try:
            elements = list(getattr(at, kind))
        except Exception:                                   # element kind unknown to this AppTest version
            continue
        for element in elements:
            if getattr(element, "key", None) == "lang_radio":
                continue
            # A widget's value is its stored state (an option, the user's text), not a text of the screen.
            attrs = ("value", "body") if kind in DISPLAY else ("label", "help")
            for attr in attrs:
                value = getattr(element, attr, None)
                if isinstance(value, str):
                    texts.append(value)
            if kind == "radio":
                texts.extend(str(o) for o in getattr(element, "options", []))
    return texts


class ScreenTests(unittest.TestCase):
    def test_each_language_runs_and_shows_its_own_texts(self):
        tables = app_translations()
        for name, code in LANGUAGES.items():
            with self.subTest(language=name):
                at = run_app(name)
                self.assertFalse(at.exception, [e.message for e in at.exception])
                switch = at.radio(key="lang_radio")
                self.assertEqual(list(switch.options), list(LANGUAGES))
                self.assertEqual(switch.value, name)
                expected = KEYED_TEXT[code] if code in KEYED_TEXT else tables[code]
                headers = [h.value for h in at.sidebar.header]
                self.assertIn(expected["graph_settings"], headers)
                self.assertIn(expected["fig_size"], headers)
                texts = screen_texts(at)
                self.assertGreater(len(texts), 60)
                if code != "ja":
                    leaked = [t for t in texts if JAPANESE.search(t)]
                    self.assertEqual(leaked, [], name)
                if code == "vi":
                    self.assertIn(INLINE_TEXT["vi"]["Axis labels"], [s.value for s in at.sidebar.subheader])
                    self.assertIn(INLINE_TEXT["vi"]["Output DPI"], [s.label for s in at.selectbox])
                if code == "th":
                    self.assertIn(INLINE_TEXT["th"]["Axis labels"], [s.value for s in at.sidebar.subheader])
                    self.assertIn(INLINE_TEXT["th"]["Output DPI"], [s.label for s in at.selectbox])
                session = at.session_state["_regression_report"]["session"]["settings"]
                self.assertEqual("lang_radio" in session, code in ("ja", "en"))

    def test_switching_language_keeps_the_work(self):
        # Without the label-side radios: Streamlit 1.32's AppTest cannot re-send a radio that formats its options.
        at = run_app("日本語", {"palette": "blues", "show_side_labels": False, "show_cif_legend": False})
        at.selectbox(key="palette").set_value("greens").run()
        colors = [at.session_state[f"xrd_color_{i}"] for i in range(3)]
        for name in ("ไทย", "Tiếng Việt", "English", "日本語"):
            at.radio(key="lang_radio").set_value(name).run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            self.assertEqual([at.session_state[f"xrd_color_{i}"] for i in range(3)], colors)
            self.assertEqual(at.session_state["palette"], "greens")
            self.assertEqual(at.radio(key="lang_radio").value, name)


if __name__ == "__main__":
    unittest.main()
