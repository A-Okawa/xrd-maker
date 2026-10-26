import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import inspect
import io
import os
import re
import tempfile
import json
import base64
from datetime import datetime
from pathlib import Path

from project_io import (
    GLOBAL_KEYS, PATTERN_KEY, ProjectError, encode_project, decode_project,
    atomic_save, project_fingerprint, clean_settings,
)
from xrd_io import read_patterns, XRDParseError, SUPPORTED_EXTENSIONS
from figure_style import (
    STYLE_DEFAULTS, STYLE_KEYS, FIGURE_PRESETS_MM, FONT_FAMILIES, METRIC_COMPATIBLE, mm_to_inches, inches_to_mm,
    font_rcparams, apply_mpl_style, export_figure, resolve_font, css_font_stack,
    validate_style_settings,
)
from palettes import (
    PALETTES, PALETTE_NAMES, DEFAULT_PALETTE, REFERENCE_COLORS, MAX_SHADE_STEPS,
    palette_sequence, swatch_colors, load_colors, color_emoji,
)
from rich_text import rich, to_html, plain
from style_templates import encode_style_template, decode_style_template
from plot_annotations import (
    PreparedCurve, ReferencePeaks, match_reference_peaks,
    add_mpl_phase_annotations, add_plotly_phase_annotations,
    add_mpl_reference_hkl_labels, add_plotly_reference_hkl_labels,
    add_mpl_zoom_inset, add_plotly_zoom_inset,
    format_hkl_groups,
)

LOCAL_MODE = os.environ.get("XRD_MAKER_LOCAL") == "1"
# Named colours in the graph app's format; copy that app's colors.csv here to share its names.
COLORS_FILE = Path(__file__).resolve().parent / "colors.csv"
# Axis titles in the graph app's markup (<i>…</i>, _{…}, ^{…}); the defaults draw as before.
DEFAULT_X_LABEL = "Diffraction angle, 2<i>θ</i> (deg.)"
DEFAULT_Y_LABEL = "Intensity (a.u.)"
LOCAL_DATA_DIR = (Path(os.environ["XRD_MAKER_DATA_DIR"]).expanduser()
                  if LOCAL_MODE and os.environ.get("XRD_MAKER_DATA_DIR") else None)

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# ===== 翻訳辞書 / Translation dictionary =====
TRANSLATIONS = {
    "ja": {
        "password_prompt": "パスワードを入力してください",
        "login": "ログイン",
        "wrong_password": "パスワードが違います",
        "app_caption": "複数パターン重ね合わせ・CIF/PDFカード リファレンス・論文用 PDF / SVG / TIFF 出力",
        "color_red": "赤系",
        "color_orange": "橙・黄系",
        "color_green": "緑系",
        "color_blue": "青系",
        "color_purple": "紫・ピンク系",
        "color_brown": "茶・ベージュ系",
        "color_gray": "黒・グレー系",
        "label": "ラベル",
        "add_italic": "＋斜体",
        "italic_help": "末尾に `<i>TEXT</i>` を追加。TEXT を書き換えて使用。文字を選んで Ctrl+I（Mac は ⌘+I）でも斜体",
        "add_subscript": "＋x₂",
        "subscript_help": "下付き：末尾に `_{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+,（カンマ）でも下付き",
        "add_superscript": "＋x²",
        "superscript_help": "上付き：末尾に `^{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+.（ピリオド）でも上付き",
        "change_color": "🎨 色を変更",
        "session_header": "💾 セッション",
        "load_session": "保存した作業を開く (.json)",
        "project_help": "測定データ、リファレンス、表示順、色、ラベル、図の設定をまとめて保存・再開できます。",
        "project_loaded": "保存した作業を読み込みました。",
        "project_error": "作業ファイルを開けませんでした: ",
        "new_project": "新しい作業を始める",
        "resume_local": "前回の自動保存から再開",
        "autosaved": "PCに自動保存済み · {time}",
        "autosave_error": "自動保存できませんでした。作業ファイルをダウンロードして保存してください: ",
        "local_mode": "PC版 · データと自動保存はこのPC内に保存されます。",
        "native_help": "Rigaku・Brukerの対応形式を読み込めます。複数スキャンは別パターンとして表示します。",
        "read_error": "測定データを読み込めませんでした: ",
        "duplicate_names": "同名のファイルがあります。ファイル名を変更して選び直してください（現在の作業は保持しています）: ",
        "project_save_error": "作業ファイルを保存できません。データ量やラベルを確認してください: ",
        "scan_count": "{files} ファイル · {scans} パターン",
        "data_range": "測定データの全範囲を表示",
        "download_desktop": "PC用をダウンロード (.zip)",
        "desktop_help": "Mac・Windows対応。初回セットアップ後はオフラインで使用できます。",
        "xrd_data_header": "📂 XRDデータ",
        "xrd_upload": "XRDデータ（装置の生データ・XY・CSV）",
        "ref_header": "📂 リファレンス",
        "cif_upload": "CIFファイル (.cif)",
        "pdf_upload": "PDFカード (.pdf)",
        "restored_xrd": "復元済み XRD: ",
        "restored_cif": "復元済み CIF: ",
        "restored_pdf": "復元済み PDF: ",
        "graph_settings": "⚙️ グラフ設定",
        "xrange": "2θ 範囲 (°)",
        "normalize_cb": "強度を正規化（最大=1）",
        "show_legend_cb": "メイン凡例を表示",
        "show_cif_legend_cb": "リファレンスラベルをグラフ内に表示",
        "show_ref_lines_cb": "リファレンスピーク線をメインに表示（破線）",
        "show_ref_line_each": "この破線を表示",
        "show_peaks_cb": "ピーク位置を表示",
        "peak_sensitivity": "ピーク感度",
        "offset_multiplier": "オフセット（倍率）",
        "tick_settings": "目盛り設定",
        "major_tick": "主目盛り間隔 (°)",
        "show_minor_cb": "副目盛りを表示",
        "minor_tick": "副目盛り間隔 (°)",
        "icdd_label_settings": "リファレンス ラベル設定",
        "icdd_label_pos": "リファレンスラベル位置",
        "left": "左",
        "right": "右",
        "icdd_label_fontsize": "リファレンスラベル文字サイズ",
        "icdd_offset_x": "リファレンス横オフセット (°)",
        "icdd_offset_x_help": "端からの距離（°）",
        "icdd_offset_y": "リファレンス縦オフセット（行高さ比）",
        "icdd_offset_y_help": "0=行の下端、1.0=行の上端",
        "sample_labels": "サンプルラベル（グラフ内）",
        "show_side_labels_cb": "グラフ内にラベルを表示",
        "label_pos": "ラベル位置",
        "label_fontsize": "ラベル文字サイズ",
        "label_offset_x": "横オフセット（°）",
        "label_offset_x_help": "正=グラフ内側、負=グラフ外側",
        "label_offset_y": "縦オフセット（パターン高さ比）",
        "label_offset_y_help": "0=ベースライン、1.0=ピーク付近",
        "fig_size": "📐 図サイズ・出力",
        "fig_width": "図の幅 (inch)",
        "fig_height": "図の高さ (inch)",
        "font_size": "フォントサイズ",
        "save_session": "💾 作業を保存・ダウンロード (.json)",
        "show_panel": "⚙️ パターン設定パネルを表示",
        "xrd_patterns": "#### XRD パターン",
        "display_order": "表示順",
        "show_cb": "表示する",
        "offset_adj": "オフセット調整",
        "y_position": "Y位置（絶対値）",
        "reference_header": "#### リファレンス",
        "pdf_card_md": "**PDFカード**",
        "drag_zoom": "ドラッグで範囲ズーム ／ ダブルクリックでリセット",
        "interactive_preview": "インタラクティブ",
        "export_preview": "保存画像のプレビュー",
        "export_preview_caption": "図サイズ: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI（プレビューは150 DPI）",
        "save_tiff": "📥 TIFF として保存 ({dpi} DPI)",
        "save_png": "📥 PNG として保存 ({dpi} DPI)",
        "upload_prompt": "XRDデータを選択するか、保存した作業ファイル（.json）を開いてください。",
        "no_pymatgen": "pymatgen が未インストールのため CIF リファレンス機能は無効です。",
        "no_pdfplumber": "pdfplumber が未インストールのため PDFカード リファレンス機能は無効です。",
        "cif_error": "CIF エラー: ",
        "pdf_error": "PDFカード解析エラー: ",
    },
    "en": {
        "password_prompt": "Enter password",
        "login": "Login",
        "wrong_password": "Incorrect password",
        "app_caption": "Multi-pattern overlay · CIF/PDF card reference · Publication-quality PDF / SVG / TIFF export",
        "color_red": "Reds",
        "color_orange": "Orange/Yellow",
        "color_green": "Greens",
        "color_blue": "Blues",
        "color_purple": "Purple/Pink",
        "color_brown": "Brown/Beige",
        "color_gray": "Black/Gray",
        "label": "Label",
        "add_italic": "+Italic",
        "italic_help": "Appends `<i>TEXT</i>`; replace TEXT. Or select text and press Ctrl+I (⌘+I on a Mac).",
        "add_subscript": "+x₂",
        "subscript_help": "Subscript: appends `_{N}`; replace N. Or select text and press Ctrl+, (comma).",
        "add_superscript": "+x²",
        "superscript_help": "Superscript: appends `^{N}`; replace N. Or select text and press Ctrl+. (period).",
        "change_color": "🎨 Change color",
        "session_header": "💾 Session",
        "load_session": "Open saved project (.json)",
        "project_help": "Save and reopen measurements, references, order, colors, labels and figure settings together.",
        "project_loaded": "Project restored.",
        "project_error": "Could not open the project: ",
        "new_project": "Start a new project",
        "resume_local": "Resume the previous autosave",
        "autosaved": "Autosaved on this PC · {time}",
        "autosave_error": "Could not autosave. Download the project to keep your work: ",
        "local_mode": "PC edition · Data and autosaves stay on this computer.",
        "native_help": "Read supported Rigaku and Bruker files. Multiple scans are shown as separate patterns.",
        "read_error": "Could not read measurement data: ",
        "duplicate_names": "Files have duplicate names. Rename them and select again; your current work is kept: ",
        "project_save_error": "Could not save the project. Check the data size and labels: ",
        "scan_count": "{files} files · {scans} patterns",
        "data_range": "Show the full measured range",
        "download_desktop": "Download PC edition (.zip)",
        "desktop_help": "For Mac and Windows. Works offline after the first setup.",
        "xrd_data_header": "📂 XRD Data",
        "xrd_upload": "XRD measurements (instrument files, XY, CSV)",
        "ref_header": "📂 Reference",
        "cif_upload": "CIF File (.cif)",
        "pdf_upload": "PDF Card (.pdf)",
        "restored_xrd": "Restored XRD: ",
        "restored_cif": "Restored CIF: ",
        "restored_pdf": "Restored PDF: ",
        "graph_settings": "⚙️ Graph Settings",
        "xrange": "2θ Range (°)",
        "normalize_cb": "Normalize intensity (max=1)",
        "show_legend_cb": "Show main legend",
        "show_cif_legend_cb": "Show reference labels in plot",
        "show_ref_lines_cb": "Show reference peak lines in main panel (dashed)",
        "show_ref_line_each": "Show this line",
        "show_peaks_cb": "Show peak positions",
        "peak_sensitivity": "Peak sensitivity",
        "offset_multiplier": "Offset (multiplier)",
        "tick_settings": "Tick settings",
        "major_tick": "Major tick interval (°)",
        "show_minor_cb": "Show minor ticks",
        "minor_tick": "Minor tick interval (°)",
        "icdd_label_settings": "Reference label settings",
        "icdd_label_pos": "Reference label position",
        "left": "Left",
        "right": "Right",
        "icdd_label_fontsize": "Reference label font size",
        "icdd_offset_x": "Reference horizontal offset (°)",
        "icdd_offset_x_help": "Distance from edge (°)",
        "icdd_offset_y": "Reference vertical offset (row height ratio)",
        "icdd_offset_y_help": "0=bottom of row, 1.0=top of row",
        "sample_labels": "Sample labels (in plot)",
        "show_side_labels_cb": "Show labels in plot",
        "label_pos": "Label position",
        "label_fontsize": "Label font size",
        "label_offset_x": "Horizontal offset (°)",
        "label_offset_x_help": "Positive=inside, Negative=outside",
        "label_offset_y": "Vertical offset (pattern height ratio)",
        "label_offset_y_help": "0=baseline, 1.0=near peak",
        "fig_size": "📐 Figure Size & Export",
        "fig_width": "Figure width (inch)",
        "fig_height": "Figure height (inch)",
        "font_size": "Font size",
        "save_session": "💾 Save and download project (.json)",
        "show_panel": "⚙️ Show pattern settings panel",
        "xrd_patterns": "#### XRD Patterns",
        "display_order": "Display order",
        "show_cb": "Show",
        "offset_adj": "Offset adjustment",
        "y_position": "Y position (absolute)",
        "reference_header": "#### Reference",
        "pdf_card_md": "**PDF Card**",
        "drag_zoom": "Drag to zoom / Double-click to reset",
        "interactive_preview": "Interactive",
        "export_preview": "Export preview",
        "export_preview_caption": "Figure size: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI (preview: 150 DPI)",
        "save_tiff": "📥 Save as TIFF ({dpi} DPI)",
        "save_png": "📥 Save as PNG ({dpi} DPI)",
        "upload_prompt": "Choose XRD measurements or open a saved project (.json) to continue.",
        "no_pymatgen": "CIF reference is disabled (pymatgen not installed).",
        "no_pdfplumber": "PDF card reference is disabled (pdfplumber not installed).",
        "cif_error": "CIF error: ",
        "pdf_error": "PDF card parse error: ",
    },
}

# ===== 色データ（言語非依存） =====
_COLOR_FAMILIES_DATA = OrderedDict([
    ("red", [
        ("#7b0000", "Deep Maroon"), ("#b71c1c", "Dark Red"),
        ("#d32f2f", "Red"),         ("#e53935", "Bright Red"),
        ("#ef9a9a", "Light Red"),   ("#f48fb1", "Light Pink"),
        ("#f8bbd0", "Pale Pink"),
    ]),
    ("orange", [
        ("#bf360c", "Deep Orange"), ("#e64a19", "Dark Orange"),
        ("#ff7043", "Orange"),      ("#ffa726", "Amber"),
        ("#ffca28", "Yellow"),      ("#fff176", "Light Yellow"),
    ]),
    ("green", [
        ("#1b5e20", "Deep Green"),  ("#2e7d32", "Dark Green"),
        ("#43a047", "Green"),       ("#66bb6a", "Medium Green"),
        ("#00897b", "Teal"),        ("#26c6da", "Cyan"),
        ("#b2dfdb", "Pale Teal"),
    ]),
    ("blue", [
        ("#0d47a1", "Deep Blue"),   ("#1565c0", "Dark Blue"),
        ("#1976d2", "Blue"),        ("#1e88e5", "Medium Blue"),
        ("#42a5f5", "Sky Blue"),    ("#90caf9", "Light Blue"),
        ("#bbdefb", "Pale Blue"),
    ]),
    ("purple", [
        ("#4a148c", "Deep Purple"), ("#6a1b9a", "Dark Purple"),
        ("#8e24aa", "Purple"),      ("#ab47bc", "Medium Purple"),
        ("#ba68c8", "Violet"),      ("#ce93d8", "Lavender"),
        ("#f06292", "Pink"),
    ]),
    ("brown", [
        ("#3e2723", "Deep Brown"),  ("#5d4037", "Brown"),
        ("#8d6e63", "Medium Brown"),("#a1887f", "Warm Beige"),
        ("#d7ccc8", "Light Beige"),
    ]),
    ("gray", [
        ("#000000", "Black"),       ("#212121", "Near Black"),
        ("#424242", "Very Dark Gray"),("#616161", "Dark Gray"),
        ("#9e9e9e", "Medium Gray"), ("#bdbdbd", "Gray"),
        ("#e0e0e0", "Light Gray"),
    ]),
])

_COLOR_FAMILY_NAME_KEYS = {
    "red": "color_red",
    "orange": "color_orange",
    "green": "color_green",
    "blue": "color_blue",
    "purple": "color_purple",
    "brown": "color_brown",
    "gray": "color_gray",
}

st.set_page_config(page_title="XRD Maker", page_icon="🔬", layout="wide")
# Full-width preview image on old and new Streamlit (use_column_width now shows a warning box).
_IMAGE_PARAMETERS = inspect.signature(st.image).parameters
FULL_WIDTH_IMAGE = ({"width": "stretch"} if "int" not in str(_IMAGE_PARAMETERS["width"].annotation)
                    else {"use_container_width": True} if "use_container_width" in _IMAGE_PARAMETERS
                    else {"use_column_width": True})

# ===== 言語選択 / Language selection =====
_lang_option = st.sidebar.radio(
    "🌐 Language / 言語", ["日本語", "English"],
    horizontal=True, key="lang_radio",
)
_lang_code = "en" if _lang_option == "English" else "ja"
T = TRANSLATIONS[_lang_code]


def ui(ja, en):
    return en if _lang_code == "en" else ja


def _widget_default(key, default, checkbox=False, number=False):
    # A restored value has one owner: session state. Passing a second widget
    # default causes Streamlit's duplicate-default warning after project loads.
    if st.session_state.get(key) is None:
        st.session_state[key] = default
    return False if checkbox else "min" if number else None

# 言語付き色ファミリー名
COLOR_FAMILIES = OrderedDict(
    (T[_COLOR_FAMILY_NAME_KEYS[k]], v)
    for k, v in _COLOR_FAMILIES_DATA.items()
)

# ===== パスワード認証 =====
def check_password():
    if LOCAL_MODE:
        return True
    if st.session_state.get("authenticated"):
        return True
    pwd = st.secrets.get("password", "")
    st.title("🔬 XRD Maker")
    entered = st.text_input(T["password_prompt"], type="password")
    if st.button(T["login"]):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error(T["wrong_password"])
    return False

if not check_password():
    st.stop()

st.title("XRD Maker")
st.caption(T["app_caption"])
if LOCAL_MODE:
    st.caption(T["local_mode"])


# ===== RestoredFile =====
class RestoredFile:
    """セッション復元時のファイル代替オブジェクト"""
    def __init__(self, name: str, data: bytes, pattern=None):
        self.name = name
        self._data = data
        self.pattern = pattern

    def read(self) -> bytes:
        return self._data

    def seek(self, pos: int):
        pass


def mpl_to_plotly(text: str) -> str:
    r"""Label markup (<i>…</i>, _{…}, ^{…}, and older $\it{x}$, $_{x}$, $^{x}$) → Plotly HTML."""
    return to_html(text)


# ===== ラベル入力（書式ボタン付き） =====
def label_input(key: str, default: str = "") -> str:
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    if inp_key not in st.session_state:
        st.session_state[inp_key] = st.session_state[val_key]
    inp = st.text_input(T["label"], key=inp_key)
    st.session_state[val_key] = inp

    c1, c2, c3 = st.columns(3)
    if c1.button(T["add_italic"], key=f"_bi_{key}", use_container_width=True,
                 help=T["italic_help"]):
        st.session_state[val_key] = inp + "<i>TEXT</i>"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button(T["add_subscript"], key=f"_bs_{key}", use_container_width=True,
                 help=T["subscript_help"]):
        st.session_state[val_key] = inp + "_{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button(T["add_superscript"], key=f"_bp_{key}", use_container_width=True,
                 help=T["superscript_help"]):
        st.session_state[val_key] = inp + "^{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    return st.session_state[val_key]


# ===== データ読み込み =====
def read_xrd_data(file_bytes: bytes, filename: str = ""):
    """Compatibility helper for a single text/measurement pattern."""
    try:
        patterns = read_patterns(file_bytes, filename)
    except XRDParseError:
        return None
    if not patterns:
        return None
    return np.column_stack((patterns[0].x, patterns[0].y))


@st.cache_data(show_spinner=False)
def calc_cif_details(cif_bytes: bytes, two_theta_range=(5, 90)):
    if not PYMATGEN_AVAILABLE:
        return None, None, []
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_bytes)
        tmp_path = tmp.name
    try:
        parser = CifParser(tmp_path)
        structs = parser.parse_structures(primitive=False)
        if not structs:
            return None, None, []
        calc = XRDCalculator(wavelength="CuKa", symprec=0.01)
        pat = calc.get_pattern(structs[0], two_theta_range=two_theta_range)
        mask = (pat.x >= two_theta_range[0]) & (pat.x <= two_theta_range[1])
        hkls = []
        for selected, families in zip(mask, pat.hkls):
            if selected:
                hkls.append(format_hkl_groups(families))
        return pat.x[mask], pat.y[mask], hkls
    except Exception as e:
        st.warning(T["cif_error"] + str(e))
        return None, None, []
    finally:
        os.unlink(tmp_path)


def calc_cif_pattern(cif_bytes: bytes, two_theta_range=(5, 90)):
    x, y, _ = calc_cif_details(cif_bytes, two_theta_range)
    return x, y


def parse_pdf_card(pdf_bytes: bytes, two_theta_range=(5, 90)):
    """ICDDのPDFカードから 2θ と規格化強度を抽出する。
    テーブル形式: No. | 2θ(°) | d値 | 規格化強度 | hkl (左右2列構成)"""
    if not PDFPLUMBER_AVAILABLE:
        return None, None
    two_thetas = []
    intensities = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row is None:
                            continue
                        for offset in [0, 5]:
                            if len(row) < offset + 4:
                                continue
                            try:
                                int(str(row[offset]).strip())
                                two_theta = float(str(row[offset + 1]).strip())
                                intensity = float(str(row[offset + 3]).strip())
                                if two_theta_range[0] <= two_theta <= two_theta_range[1]:
                                    two_thetas.append(two_theta)
                                    intensities.append(intensity)
                            except (ValueError, TypeError, AttributeError):
                                continue
    except Exception as e:
        st.warning(T["pdf_error"] + str(e))
        return None, None

    if not two_thetas:
        return None, None

    pairs = sorted(zip(two_thetas, intensities))
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return x, y


def extract_cif_label(cif_bytes: bytes) -> str | None:
    """CIFからICSD番号・組成式・鉱物名を抽出してラベル文字列を返す。"""
    text = cif_bytes.decode("utf-8", errors="ignore")
    icsd_no = None
    formula = None
    name = None
    for line in text.splitlines():
        s = line.strip()
        if re.match(r'_database_code_ICSD\b', s, re.I):
            m = re.search(r'(\d+)', s)
            if m:
                icsd_no = m.group(1)
        elif re.match(r'_chemical_formula_sum\b', s, re.I) and formula is None:
            m = re.search(r"['\"](.+?)['\"]", s)
            if m:
                formula = m.group(1).replace(" ", "")
        elif re.match(r'_chemical_name_(?:mineral|common|systematic)\b', s, re.I) and name is None:
            m = re.search(r"['\"](.+?)['\"]", s)
            if m:
                val = m.group(1).strip()
                if val and val != "?" and val.lower() != "unknown":
                    name = val
    parts = []
    if icsd_no:
        parts.append(f"ICSD No. {icsd_no}")
    if name:
        parts.append(name)
    elif formula:
        parts.append(formula)
    return ", ".join(parts) if parts else None


def extract_pdf_card_label(pdf_bytes: bytes) -> str | None:
    """PDFカードからカード番号と物質名を抽出してラベル文字列を返す。"""
    if not PDFPLUMBER_AVAILABLE:
        return None
    card_no = None
    mat_name = None
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return None
            text = pdf.pages[0].extract_text() or ""
            m = re.search(r'PDFカード番号[:\s：]+([0-9\-]+)', text)
            if m:
                card_no = m.group(1).strip()
            m = re.search(r'名称[:\s：]*(.+?)(?:\s+I/Ic|\s+RIR|\n|$)', text)
            if m:
                mat_name = m.group(1).strip()
    except Exception:
        return None
    parts = []
    if card_no:
        parts.append(f"PDF No. {card_no}")
    if mat_name:
        parts.append(mat_name)
    return ", ".join(parts) if parts else None


def detect_peaks(x, y, prominence=0.1, min_dist_deg=0.5):
    if len(y) < 3:
        return np.array([], dtype=int)
    spacings = np.diff(x)
    positive_spacings = spacings[spacings > 0]
    dx = np.mean(positive_spacings) if len(positive_spacings) else 1.0
    dist = max(1, int(min_dist_deg / dx))
    pks, _ = find_peaks(y, prominence=prominence * np.max(y), distance=dist)
    return pks


# ===== 色（グラフ描画appと同じ色の組み合わせ） =====
OTHER_COLOR = "—"
# The palettes of the graph app, then this app's earlier colour families (kept for earlier figures).
SWATCH_SETS = OrderedDict([(name, None) for name in PALETTES] +
                          [(f"xrd:{family}", colors) for family, colors in COLOR_FAMILIES.items()])


def swatch_set_label(key):
    if key in PALETTES:
        return PALETTE_NAMES[_lang_code][key]
    return "XRD Maker · " + key.split(":", 1)[1]


def named_colors():
    return load_colors(COLORS_FILE)


def _int_or(value, default):
    return value if type(value) is int else default


def pattern_display_order():
    """Pattern indices in display order (the 表示順 numbers; equal numbers keep the file order)."""
    state = st.session_state
    n_xrd = len(state.get("_xrd_pattern_names") or [])
    return sorted(range(n_xrd), key=lambda i: (_int_or(state.get(f"ord_{i}"), i + 1), i))


def palette_range(n):
    """First and last display positions (1-based, inclusive) that take the palette; the rest are the base colour.

    palette_to = 0 means "to the last pattern", so patterns added later are included.
    """
    state = st.session_state
    lo = min(max(_int_or(state.get("palette_from"), 1), 1), max(n, 1))
    to = _int_or(state.get("palette_to"), 0)
    hi = n if to <= 0 else min(max(to, lo), n)
    return lo, hi


def palette_base_color():
    value = st.session_state.get("palette_base_color")
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else "#000000"


def palette_items():
    """Colour keys in display order: [(key, visible)] for measured patterns and for references.

    Uses only session state, so the palette callbacks see the same order as the panel.
    """
    state = st.session_state
    patterns = pattern_display_order()
    n_cif = len(state.get("_cif_bytes") or {})
    refs = ([(_int_or(state.get(f"cord_{i}"), i + 1), f"cif_color_{i}", state.get(f"cvis_{i}", True)) for i in range(n_cif)]
            + [(_int_or(state.get(f"pord_{i}"), n_cif + i + 1), f"pdf_color_{i}", state.get(f"pvis_{i}", True))
               for i in range(len(state.get("_pdf_bytes") or {}))])
    refs.sort(key=lambda item: item[0])
    return ([(f"xrd_color_{i}", state.get(f"vis_{i}", True) is not False) for i in patterns],
            [(key, visible is not False) for _, key, visible in refs])


def palette_plan():
    """The palette colour for every pattern and reference key, and the number of items it colours.

    Visible patterns within the chosen range take the palette in display order (shades spread
    over them, as the graph app spreads them over its series); hidden ones continue after them.
    Patterns outside the range (a base measurement, for example) take the base colour, black by
    default. References keep black-then-Okabe–Ito colours unless the palette is applied to them too.
    """
    state = st.session_state
    name = state.get("palette", DEFAULT_PALETTE)
    name = name if name in PALETTES else DEFAULT_PALETTE
    reverse, steps = bool(state.get("palette_reverse", False)), _int_or(state.get("palette_steps"), 0)
    patterns, refs = palette_items()
    lo, hi = palette_range(len(patterns))
    shaded = patterns[lo - 1:hi]
    targets = shaded + refs if state.get("palette_refs", False) else shaded
    visible = [key for key, shown in targets if shown]
    hidden = [key for key, shown in targets if not shown]
    plan = {key: palette_base_color() for key, _ in patterns[:lo - 1] + patterns[hi:]}
    if not state.get("palette_refs", False):
        plan.update({key: REFERENCE_COLORS[rank % len(REFERENCE_COLORS)] for rank, (key, _) in enumerate(refs)})
    plan.update(zip(visible, palette_sequence(name, len(visible), reverse, steps)))
    plan.update(zip(hidden, palette_sequence(name, len(targets), reverse, steps)[len(visible):]))
    return plan, len(visible)


def _apply_palette():
    """Colour the visible patterns (and references, if chosen): the palette within the range, the base colour outside."""
    plan, _ = palette_plan()
    patterns, refs = palette_items()
    targets = patterns + refs if st.session_state.get("palette_refs", False) else patterns
    for key, shown in targets:
        if shown:
            st.session_state[key] = plan[key]
    for key in list(st.session_state):
        if str(key).startswith(("_palview_", "_cname_")):
            del st.session_state[key]


def _set_palette_range(n):
    lo, hi = st.session_state.get("_palette_range", (1, n))
    st.session_state["palette_from"] = int(lo)
    st.session_state["palette_to"] = 0 if int(hi) >= n else int(hi)
    _apply_palette()


def top_to_bottom(levels):
    """Positions ordered as the curves appear from top to bottom, for the legend.

    ``levels`` are the curves' vertical positions (stacking offsets) in display order;
    equal levels (overlapping curves) keep the display order.
    """
    return sorted(range(len(levels)), key=lambda k: -levels[k])


def _set_color(key, value):
    st.session_state[key] = value


def _named_color_chosen(key, name_key):
    value = named_colors().get(st.session_state.get(name_key))
    if value:
        st.session_state[key] = value


def swatch_set_colors(set_key, n_items):
    if set_key in PALETTES:
        return swatch_colors(set_key, n_items, bool(st.session_state.get("palette_reverse", False)),
                             _int_or(st.session_state.get("palette_steps"), 0))
    return [hex_c for hex_c, _ in SWATCH_SETS[set_key]]


# ===== カラーポップアップ =====
def color_picker_popover(key: str, default_hex: str, n_items: int = 1):
    """Colour of one pattern or reference: swatches of the graph app's palettes, colours by name, or any colour."""
    _widget_default(key, default_hex)
    current = str(st.session_state[key])

    st.markdown(
        f'<div style="background:{current};height:20px;border-radius:5px;'
        f'border:1px solid #ccc;margin-bottom:4px"></div>',
        unsafe_allow_html=True,
    )
    with st.popover(T["change_color"], use_container_width=True):
        view_key = f"_palview_{key}"
        if st.session_state.get(view_key) not in SWATCH_SETS:
            st.session_state[view_key] = st.session_state.get("palette", DEFAULT_PALETTE)
        st.selectbox(ui("色の組み合わせ", "Palette"), list(SWATCH_SETS), key=view_key,
                     format_func=swatch_set_label,
                     help=ui("上の 8 つはグラフ描画appと同じ色です。濃淡は、色を付けるパターンの数（または「濃淡の段数」）で分けた色です。",
                             "The first entries are the graph app's colours. Shades are split by the number of coloured patterns (or the shade steps)."))
        colors = swatch_set_colors(st.session_state[view_key], n_items)
        cols = st.columns(len(colors))
        for j, hex_c in enumerate(colors):
            with cols[j]:
                selected = hex_c.lower() == current.lower()
                st.markdown(
                    f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                    f'border:{"3px solid #333" if selected else "1px solid #ccc"};'
                    f'margin-bottom:2px"></div>',
                    unsafe_allow_html=True,
                )
                st.button("✓" if selected else " ", key=f"{key}_sw_{j}", help=hex_c.upper(),
                          use_container_width=True, on_click=_set_color, args=(key, hex_c))
        names = named_colors()
        name_key = f"_cname_{key}"
        st.session_state[name_key] = next((n for n, h in names.items() if h.lower() == current.lower()), OTHER_COLOR)
        st.selectbox(ui("名前で選ぶ", "By name"), [OTHER_COLOR] + list(names), key=name_key,
                     on_change=_named_color_chosen, args=(key, name_key),
                     format_func=lambda n: n if n == OTHER_COLOR else f"{color_emoji(names[n])} {n}  {names[n]}",
                     help=ui("colors.csv の色（グラフ描画appと同じ書式「名前,#RRGGBB」）。グラフ描画appの colors.csv をこのアプリのフォルダに置くと、同じ名前の色を選べます。",
                             "Colours from colors.csv (the graph app's format, name,#RRGGBB). Put the graph app's colors.csv in this app's folder to share its names."))
        st.color_picker(ui("自由に選ぶ", "Any colour"), key=key)
    return st.session_state[key]


# ===== セッション保存/復元 =====
# Keep conditional widget values when their controls are temporarily hidden.
# Language is already instantiated above; its value is retained by that widget.
for _key in list(st.session_state):
    if _key != "lang_radio" and (_key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(_key)):
        st.session_state[_key] = st.session_state[_key]


def build_session_json(n_xrd: int, n_cif: int, n_pdf: int) -> bytes:
    settings = {key: value for key, value in st.session_state.items()
                if key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key)}
    return encode_project(
        st.session_state.get("_xrd_bytes", {}),
        st.session_state.get("_cif_bytes", {}),
        st.session_state.get("_pdf_bytes", {}), settings,
    )


def restore_session(json_bytes: bytes):
    # This function is called only from callbacks, before widgets are created.
    project = decode_project(json_bytes)
    for key in list(st.session_state):
        if (key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key)
                or re.fullmatch(r"_(?:inp|ver)_(?:lbl|clbl|plbl)_.*|_(?:palview|cname)_.*", key)):
            del st.session_state[key]
    for kind in ("xrd", "cif", "pdf"):
        st.session_state[f"_{kind}_bytes"] = project[f"{kind}_files"]
        st.session_state.pop(f"_upload_error_{kind}", None)
    st.session_state.update(project["settings"])
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_xrd_pattern_names"] = None
    st.session_state["_project_loaded"] = True
    st.session_state.pop("_project_error", None)
    st.session_state.pop("_autosave_fingerprint", None)


def _open_uploaded_project():
    upload = st.session_state.get("_session_upload")
    if upload is not None:
        try:
            restore_session(upload.getvalue())
        except (ProjectError, ValueError) as exc:
            st.session_state["_project_error"] = str(exc)


def _open_local_autosave():
    try:
        restore_session((LOCAL_DATA_DIR / "autosave.json").read_bytes())
    except (OSError, ProjectError) as exc:
        st.session_state["_project_error"] = str(exc)


def _open_style_template():
    upload = st.session_state.get("_style_upload")
    if upload is None:
        return
    try:
        settings = decode_style_template(upload.getvalue())
        clean_settings({**{key: st.session_state[key] for key in STYLE_KEYS if key in st.session_state}, **settings})
        st.session_state.update(settings)
        st.session_state.pop("_size_signature", None)
        st.session_state["_style_loaded"] = True
        st.session_state.pop("_style_error", None)
    except (ProjectError, ValueError) as exc:
        st.session_state["_style_error"] = str(exc)


def _apply_figure_preset():
    name = st.session_state["figure_preset"]
    preset = FIGURE_PRESETS_MM.get(name)
    if preset:
        st.session_state["fig_width"], st.session_state["fig_height"] = map(mm_to_inches, preset)
        if name == "figapp":
            # The graph app's defaults: Arial 7 pt labels, 6.5 pt ticks and legend, 0.5 pt axes, 0.9 pt lines.
            st.session_state.update(font_family="Arial", font_size=7.0, tick_font_size=6.5, legend_font_size=6.5,
                label_fontsize=7, cif_label_fontsize=6, annotation_font_size=6,
                line_width=0.9, reference_line_width=0.6, axes_line_width=0.5, tick_length=2.5, tick_direction="out",
                axes_box=False,
                margin_left=.06, margin_right=.98, margin_bottom=.14, margin_top=.98)
        else:
            compact = name == "single"
            size = 9.0 if compact else 11.0
            st.session_state.update(font_size=size, tick_font_size=size, legend_font_size=size,
                label_fontsize=8 if compact else 10, cif_label_fontsize=7 if compact else 8,
                annotation_font_size=6 if compact else 7,
                margin_left=.16 if compact else .12, margin_bottom=.22 if compact else .15)
    st.session_state.pop("_size_signature", None)


def _update_figure_dimension(axis):
    value = st.session_state[f"_figure_{axis}"]
    st.session_state[f"fig_{axis}"] = mm_to_inches(value) if st.session_state["size_unit"] == "mm" else value
    st.session_state["figure_preset"] = "custom"


def _update_margin(key):
    value = st.session_state[f"_percent_{key}"] / 100
    st.session_state[key] = 1 - value if key in ("margin_right", "margin_top") else value


def _new_project():
    for key in list(st.session_state):
        if ((key in GLOBAL_KEYS and key != "lang_radio") or PATTERN_KEY.fullmatch(key)
                or re.fullmatch(r"_(?:inp|ver)_(?:lbl|clbl|plbl)_.*|_(?:palview|cname)_.*", key)):
            del st.session_state[key]
    for kind in ("xrd", "cif", "pdf"):
        st.session_state[f"_{kind}_bytes"] = {}
        st.session_state.pop(f"_upload_error_{kind}", None)
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_xrd_pattern_names"] = None
    st.session_state.pop("_project_loaded", None)
    st.session_state.pop("_project_error", None)


def _uploaded_files_changed(kind, widget_key):
    files = st.session_state.get(widget_key) or []
    names = [file.name for file in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        st.session_state[f"_upload_error_{kind}"] = ", ".join(duplicates)
        return
    st.session_state.pop(f"_upload_error_{kind}", None)
    # Comparing the reference identities happens later, before their widgets.
    if kind != "xrd":
        old = list(st.session_state.get(f"_{kind}_bytes", {}))
        new = [file.name for file in files]
        if old != new:
            templates = (["cvis_{}", "cord_{}", "_val_clbl_{}", "_ver_clbl_{}", "cif_color_{}", "cref_line_{}"]
                         if kind == "cif" else
                         ["pvis_{}", "pord_{}", "_val_plbl_{}", "_ver_plbl_{}", "pdf_color_{}", "pref_line_{}"])
            _remap_file_session_state(old, new, templates)
    st.session_state[f"_{kind}_bytes"] = {file.name: file.getvalue() for file in files}


def _remap_file_session_state(old_names: list, new_names: list, key_tmpls: list):
    """ファイルセットが変更された時、位置インデックスベースのキーをファイル名基準で再マップする。"""
    # 旧インデックス→値 をファイル名でまとめて保存
    old_vals: dict[str, dict] = {}
    for old_i, name in enumerate(old_names):
        old_vals[name] = {
            tmpl: st.session_state.get(tmpl.format(old_i))
            for tmpl in key_tmpls
        }
    # 旧キーをすべて削除
    for old_i in range(len(old_names)):
        for tmpl in key_tmpls:
            st.session_state.pop(tmpl.format(old_i), None)
    # 新インデックスに対応するファイル名の旧値を書き戻す
    for new_i, name in enumerate(new_names):
        if name in old_vals:
            for tmpl, val in old_vals[name].items():
                if val is None:
                    continue
                new_k = tmpl.format(new_i)
                st.session_state[new_k] = val + 1 if tmpl.startswith("_ver_") else val


# ===== サイドバー =====

st.sidebar.caption("Last updated: 2026-10-26")
st.sidebar.header(T["session_header"])
st.sidebar.caption(T["project_help"])
session_upload = st.sidebar.file_uploader(
    T["load_session"], type=["json"], key="_session_upload",
    on_change=_open_uploaded_project,
)
if st.session_state.pop("_project_loaded", False):
    st.sidebar.success(T["project_loaded"])
if st.session_state.get("_project_error"):
    st.sidebar.error(T["project_error"] + st.session_state["_project_error"])
if LOCAL_DATA_DIR and (LOCAL_DATA_DIR / "autosave.json").is_file() and not any(
    st.session_state.get(f"_{kind}_bytes") for kind in ("xrd", "cif", "pdf")
):
    st.sidebar.button(T["resume_local"], on_click=_open_local_autosave,
                      use_container_width=True)
st.sidebar.button(T["new_project"], on_click=_new_project, use_container_width=True)
autosave_status = st.sidebar.empty()

_file_epoch = st.session_state.get("_file_epoch", 0)
st.sidebar.header(T["xrd_data_header"])
st.sidebar.caption(T["native_help"])
_xrd_upload_key = f"xrd_upload_{_file_epoch}"
xrd_files = st.sidebar.file_uploader(
    T["xrd_upload"], type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
    key=_xrd_upload_key, on_change=_uploaded_files_changed,
    args=("xrd", _xrd_upload_key),
)
st.sidebar.header(T["ref_header"])
_cif_upload_key = f"cif_upload_{_file_epoch}"
cif_files = st.sidebar.file_uploader(
    T["cif_upload"], type=["cif"], accept_multiple_files=True,
    key=_cif_upload_key, on_change=_uploaded_files_changed,
    args=("cif", _cif_upload_key),
)
_pdf_upload_key = f"pdf_upload_{_file_epoch}"
pdf_ref_files = st.sidebar.file_uploader(
    T["pdf_upload"], type=["pdf"], accept_multiple_files=True,
    key=_pdf_upload_key, on_change=_uploaded_files_changed,
    args=("pdf", _pdf_upload_key),
)
for _kind in ("xrd", "cif", "pdf"):
    if st.session_state.get(f"_upload_error_{_kind}"):
        st.sidebar.error(T["duplicate_names"] + st.session_state[f"_upload_error_{_kind}"])

active_xrd = []
for _source_name, _source_bytes in st.session_state.get("_xrd_bytes", {}).items():
    try:
        _patterns = read_patterns(_source_bytes, _source_name)
        for _scan_index, _pattern in enumerate(_patterns):
            _name = (_source_name if len(_patterns) == 1 else
                     f"{_source_name} [scan {_scan_index + 1}].xy")
            active_xrd.append(RestoredFile(
                _name, _source_bytes, np.column_stack((_pattern.x, _pattern.y)),
            ))
    except XRDParseError as _error:
        st.error(T["read_error"] + f"{_source_name}: {_error}")

_new_pattern_names = [file.name for file in active_xrd]
_old_pattern_names = st.session_state.get("_xrd_pattern_names")
if _old_pattern_names is not None and _old_pattern_names != _new_pattern_names:
    _remap_file_session_state(
        _old_pattern_names, _new_pattern_names,
        ["vis_{}", "ord_{}", "_val_lbl_{}", "_ver_lbl_{}", "xrd_color_{}", "extra_offset_{}", "yoff_{}"],
    )
st.session_state["_xrd_pattern_names"] = _new_pattern_names
active_cif = [RestoredFile(n, d) for n, d in st.session_state.get("_cif_bytes", {}).items()]
active_pdf = [RestoredFile(n, d) for n, d in st.session_state.get("_pdf_bytes", {}).items()]
if active_xrd:
    st.sidebar.caption(T["scan_count"].format(
        files=len(st.session_state.get("_xrd_bytes", {})), scans=len(active_xrd),
    ))

if active_xrd and not xrd_files:
    st.sidebar.caption(T["restored_xrd"] + ", ".join(f.name for f in active_xrd))
if active_cif and not cif_files:
    st.sidebar.caption(T["restored_cif"] + ", ".join(f.name for f in active_cif))
if active_pdf and not pdf_ref_files:
    st.sidebar.caption(T["restored_pdf"] + ", ".join(f.name for f in active_pdf))

st.sidebar.header(T["graph_settings"])
if active_xrd and st.sidebar.button(T["data_range"], use_container_width=True):
    _xmin = min(float(file.pattern[:, 0].min()) for file in active_xrd)
    _xmax = max(float(file.pattern[:, 0].max()) for file in active_xrd)
    _xmin = max(0., np.floor(_xmin * 2) / 2)
    _xmax = min(180., np.ceil(_xmax * 2) / 2)
    if _xmin < _xmax:
        st.session_state["xrange"] = (float(_xmin), float(_xmax))
xrange         = st.sidebar.slider(T["xrange"], 0.0, 180.0, _widget_default('xrange', (10.0, 80.0)), step=0.5, key="xrange")
x_min, x_max   = xrange
normalize      = st.sidebar.checkbox(T["normalize_cb"], value=_widget_default('normalize', False, checkbox=True), key="normalize")
if "show_legend" not in st.session_state:
    st.session_state["show_legend"] = True
show_legend    = st.sidebar.checkbox(T["show_legend_cb"], key="show_legend")
if "show_cif_legend" not in st.session_state:
    st.session_state["show_cif_legend"] = True
show_cif_legend= st.sidebar.checkbox(T["show_cif_legend_cb"], key="show_cif_legend")
show_ref_lines = st.sidebar.checkbox(T["show_ref_lines_cb"], value=_widget_default('show_ref_lines', False, checkbox=True), key="show_ref_lines")
show_peaks     = st.sidebar.checkbox(T["show_peaks_cb"], value=_widget_default('show_peaks', False, checkbox=True), key="show_peaks")
peak_prom      = st.sidebar.slider(T["peak_sensitivity"], 0.01, 0.5, _widget_default('peak_prom', 0.1), step=0.01, key="peak_prom") if show_peaks else 0.1
global_offset  = st.sidebar.slider(T["offset_multiplier"], 0.0, 3.0, _widget_default('global_offset', 1.0), step=0.05, key="global_offset") if normalize else None

st.sidebar.subheader(T["tick_settings"])
major_tick  = st.sidebar.number_input(T["major_tick"], min_value=1.0, max_value=30.0,
                                       value=_widget_default('major_tick', 10.0, number=True), step=1.0, key="major_tick")
show_minor  = st.sidebar.checkbox(T["show_minor_cb"], value=_widget_default('show_minor', True, checkbox=True), key="show_minor")
minor_tick  = st.sidebar.number_input(T["minor_tick"], min_value=0.5, max_value=10.0,
                                       value=_widget_default('minor_tick', 2.0, number=True), step=0.5, key="minor_tick") if show_minor else None

st.sidebar.subheader(ui("軸のラベル", "Axis labels"))
_axis_label_help = ui(
    "グラフ描画appと同じ書き方です。文字を選んで Ctrl+I で斜体、Ctrl+B で太字、Ctrl+,（カンマ）で下付き、"
    "Ctrl+.（ピリオド）で上付き（Mac は ⌘ でも可）。入力欄では `<i>θ</i>`、`_{2}`、`^{3+}` のような記号で出て、"
    "Enter で図に反映されます。直接書いても同じです。空欄にするとラベルを出しません。",
    "Same markup as the graph app. Select text and press Ctrl+I for italic, Ctrl+B bold, Ctrl+, (comma) subscript, "
    "Ctrl+. (period) superscript (⌘ on a Mac). The box shows codes such as `<i>θ</i>`, `_{2}`, `^{3+}`; press Enter "
    "to update the figure. Typing them works too. Leave empty for no label.")
_widget_default("x_label", DEFAULT_X_LABEL)
_widget_default("y_label", DEFAULT_Y_LABEL)
x_title = st.sidebar.text_input(ui("横軸", "x axis"), key="x_label", help=_axis_label_help)
y_title = st.sidebar.text_input(ui("縦軸", "y axis"), key="y_label", help=_axis_label_help)
st.sidebar.caption(ui("例：`2<i>θ</i> (°)`、`Yb_{2}Si_{2}O_{7}`、`cm^-1`。試料・リファレンスのラベルも同じ書き方です。",
                      "e.g. `2<i>θ</i> (°)`, `Yb_{2}Si_{2}O_{7}`, `cm^-1`. Sample and reference labels use the same markup."))

# 左/右 の内部値は言語に関わらず固定し、format_func で表示を翻訳する
_LR_OPTIONS = ["左", "右"]
_lr_fmt = lambda x: T["left"] if x == "左" else T["right"]

if show_cif_legend:
    st.sidebar.subheader(T["icdd_label_settings"])
    cif_label_side     = st.sidebar.radio(T["icdd_label_pos"], _LR_OPTIONS,
                                           format_func=_lr_fmt, horizontal=True,
                                           key="cif_label_side")
    cif_label_fontsize = st.sidebar.slider(T["icdd_label_fontsize"], 5, 20, _widget_default('cif_label_fontsize', 9), key="cif_label_fontsize")
    cif_label_offset_x = st.sidebar.slider(T["icdd_offset_x"], 0.0, 10.0, _widget_default('cif_label_offset_x', 0.5), step=0.1,
                                            help=T["icdd_offset_x_help"], key="cif_label_offset_x")
    cif_label_offset_y = st.sidebar.slider(T["icdd_offset_y"], 0.0, 1.0, _widget_default('cif_label_offset_y', 0.5), step=0.05,
                                            help=T["icdd_offset_y_help"], key="cif_label_offset_y")
else:
    cif_label_side, cif_label_fontsize = "左", 9
    cif_label_offset_x, cif_label_offset_y = 0.5, 0.5

st.sidebar.subheader(T["sample_labels"])
show_side_labels = st.sidebar.checkbox(T["show_side_labels_cb"], value=_widget_default('show_side_labels', False, checkbox=True), key="show_side_labels")
if show_side_labels:
    label_side      = st.sidebar.radio(T["label_pos"], _LR_OPTIONS,
                                        format_func=_lr_fmt, horizontal=True,
                                        key="label_side")
    label_fontsize  = st.sidebar.slider(T["label_fontsize"], 5, 24, _widget_default('label_fontsize', 11), key="label_fontsize")
    label_offset_x  = st.sidebar.slider(T["label_offset_x"], -5.0, 5.0, _widget_default('label_offset_x', 0.5), step=0.1,
                                         help=T["label_offset_x_help"], key="label_offset_x")
    label_offset_y  = st.sidebar.slider(T["label_offset_y"], -0.3, 1.0, _widget_default('label_offset_y', 0.05), step=0.01,
                                         help=T["label_offset_y_help"], key="label_offset_y")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 11, 0.5, 0.05

st.sidebar.header(T["fig_size"])
for _key, _value in STYLE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
st.session_state.setdefault("fig_width", 10.0)
st.session_state.setdefault("fig_height", 8.0)
st.sidebar.selectbox(ui("論文図のひな形", "Publication figure preset"), list(FIGURE_PRESETS_MM),
    format_func=lambda key: {"custom": ui("自由指定", "Custom"),
        "single": ui("1段幅の例 · 85 × 70 mm", "Single column example · 85 × 70 mm"),
        "double": ui("2段幅の例 · 180 × 120 mm", "Double column example · 180 × 120 mm"),
        "figapp": ui("グラフ描画appと同じ体裁 · 88 × 70 mm", "Graph app look · 88 × 70 mm")}[key],
    key="figure_preset", on_change=_apply_figure_preset)
st.sidebar.caption(ui("ひな形は寸法・文字サイズ・余白を設定します。「グラフ描画appと同じ体裁」は Arial 7 pt・目盛り 6.5 pt・細い線で、グラフ描画appの図と並べられます。投稿先の指定に合わせて調整してください。",
                      "Presets set dimensions, font sizes and margins. 'Graph app look' uses Arial 7 pt, 6.5 pt ticks and thin lines to sit next to the graph app's figures. Adjust them to your journal's requirements."))
size_unit = st.sidebar.radio(ui("寸法の単位", "Size unit"), ["mm", "inch"], horizontal=True, key="size_unit")
_signature = (size_unit, st.session_state["fig_width"], st.session_state["fig_height"])
if st.session_state.get("_size_signature") != _signature:
    for _axis in ("width", "height"):
        _value = st.session_state[f"fig_{_axis}"]
        st.session_state[f"_figure_{_axis}"] = inches_to_mm(_value) if size_unit == "mm" else _value
    st.session_state["_size_signature"] = _signature
for _axis, _label in (("width", ui("図の幅", "Figure width")), ("height", ui("図の高さ", "Figure height"))):
    st.sidebar.number_input(f"{_label} ({size_unit})", min_value=25.0 if size_unit == "mm" else 25 / 25.4,
        max_value=508.0 if size_unit == "mm" else 20.0, step=1.0 if size_unit == "mm" else 0.1,
        format="%.2f", key=f"_figure_{_axis}", on_change=_update_figure_dimension, args=(_axis,))
fig_width, fig_height = st.session_state["fig_width"], st.session_state["fig_height"]
dpi_export = st.sidebar.selectbox("Output DPI", [300, 600], index=0, key="dpi_export")
# Older projects store an integer size for everything; ticks and legend start from it.
st.session_state["font_size"] = float(st.session_state.get("font_size") or 14.0)
for _key in ("tick_font_size", "legend_font_size"):
    st.session_state[_key] = float(st.session_state.get(_key) or st.session_state["font_size"])
st.sidebar.caption(ui("文字の大きさ (pt)", "Text sizes (pt)"))
_size_columns = st.sidebar.columns(3)
for _column, _key, _label in ((_size_columns[0], "font_size", ui("軸ラベル", "Axis labels")),
                              (_size_columns[1], "tick_font_size", ui("目盛り", "Ticks")),
                              (_size_columns[2], "legend_font_size", ui("凡例", "Legend"))):
    _column.number_input(_label, min_value=4.0, max_value=28.0, step=0.5, format="%.1f", key=_key)
font_size = st.session_state["font_size"]
tick_font_size = st.session_state["tick_font_size"]
legend_font_size = st.session_state["legend_font_size"]
with st.sidebar.expander(ui("線・フォント・余白", "Lines, fonts and margins")):
    st.selectbox(ui("フォント", "Font family"), list(FONT_FAMILIES), key="font_family",
                 help=ui("グラフ描画appと同じ選択肢です。斜体・下付き・上付きも同じ書体で描きます。",
                         "The graph app's choices. Italics, subscripts and superscripts use the same font."))
    _actual_font = resolve_font(st.session_state["font_family"])
    if _actual_font != st.session_state["font_family"]:
        if _actual_font in METRIC_COMPATIBLE.get(st.session_state["font_family"], ()):
            st.caption(ui(f"この環境には {st.session_state['font_family']} がないため、字幅が同じ {_actual_font} で描きます。"
                          f"SVG には {st.session_state['font_family']} と書き込むので、手元で開けば {st.session_state['font_family']} になります。",
                          f"{st.session_state['font_family']} is not installed here, so {_actual_font} (same character widths) is used. "
                          f"SVG files name {st.session_state['font_family']}, so they open in it on your computer."))
        else:
            st.caption(ui(f"この環境では {_actual_font} を使用します。", f"This environment uses {_actual_font}."))
    for _key, _label in (("line_width", ui("測定曲線の線幅 (pt)", "Pattern line width (pt)")),
                        ("reference_line_width", ui("リファレンスの線幅 (pt)", "Reference line width (pt)")),
                        ("axes_line_width", ui("軸・目盛りの線幅 (pt)", "Axes and tick width (pt)"))):
        st.number_input(_label, .1, 5., step=.1, key=_key)
    st.number_input(ui("目盛りの長さ (pt)", "Tick length (pt)"), .5, 10., step=.5, key="tick_length",
                    help=ui("主目盛りの長さ。副目盛りはその半分です。", "Major tick length; minor ticks are half as long."))
    st.selectbox(ui("目盛りの向き", "Tick direction"), ["auto", "in", "out"], key="tick_direction",
                 format_func=lambda value: {"auto": ui("これまでどおり（リファレンスの軸は外向き）", "As before (outward on the reference axis)"),
                                            "in": ui("内向き", "Inward"), "out": ui("外向き", "Outward")}[value],
                 help=ui("下の軸の目盛りの向き。メインとリファレンスの境目の目盛りは、棒と重ならないように内向きのままです。",
                         "Direction of the bottom axis ticks. Ticks between the main and reference panels stay inward to avoid the sticks."))
    st.checkbox(ui("上と右の枠線を描く", "Draw the top and right frame"), key="axes_box")
    for _key, _label in (("margin_left", ui("左余白 (%)", "Left margin (%)")),
                        ("margin_right", ui("右余白 (%)", "Right margin (%)")),
                        ("margin_bottom", ui("下余白 (%)", "Bottom margin (%)")),
                        ("margin_top", ui("上余白 (%)", "Top margin (%)"))):
        _fraction = st.session_state[_key]
        st.session_state[f"_percent_{_key}"] = round(100 * (1 - _fraction if _key in ("margin_right", "margin_top") else _fraction), 8)
        st.number_input(_label, 0., 100., step=1., key=f"_percent_{_key}", on_change=_update_margin, args=(_key,))
    st.checkbox(ui("背景を透明にする", "Transparent background"), key="transparent_background")
    st.checkbox(ui("SVGの文字を編集可能にする", "Keep SVG text editable"), key="svg_text_editable")
    st.caption(ui("編集可能なSVGの表示には、開くPCにも使用フォントが必要です。", "Editable SVG text uses fonts installed on the computer opening it."))
plot_style = {key: st.session_state[key] for key in STYLE_KEYS}
try:
    plot_style = validate_style_settings(plot_style, fill_defaults=True)
except ValueError as _error:
    st.sidebar.error(ui("余白の合計が100%未満になるよう調整してください。", "Adjust opposing margins so their sum is below 100%."))
    st.stop()

with st.sidebar.expander(ui("作図スタイルの保存・読み込み", "Save and load figure styles")):
    st.caption(ui("寸法・線・フォント・ラベル位置を再利用できます。測定データや試料名は含みません。", "Reuse dimensions, lines, fonts and label positions. Styles do not include measurements or sample names."))
    st.file_uploader(ui("スタイルを開く (.json)", "Open figure style (.json)"), type=["json"], key="_style_upload", on_change=_open_style_template)
    if st.session_state.pop("_style_loaded", False):
        st.success(ui("作図スタイルを読み込みました。", "Figure style loaded."))
    if st.session_state.get("_style_error"):
        st.error(st.session_state["_style_error"])
    st.download_button(ui("スタイルを保存 (.json)", "Save figure style (.json)"),
        encode_style_template(dict(st.session_state)), file_name="xrd_figure_style.json", mime="application/json", use_container_width=True)

with st.sidebar.expander(ui("ピーク注記・拡大図", "Peak annotations and zoom inset")):
    show_phase_markers = st.checkbox(ui("リファレンスと近い実測ピークに記号", "Mark measured peaks near references"), key="show_phase_markers")
    phase_tolerance = st.number_input(ui("照合の許容差 (°)", "Matching tolerance (°)"), .01, 1., value=_widget_default('phase_tolerance', .1, number=True), step=.01, key="phase_tolerance")
    st.caption(ui("表示中のリファレンスと実測の極大位置を照合します。記号の一致は相の同定を意味しません。", "Matches visible references to measured local maxima. Matching markers do not identify phases."))
    show_hkl = st.checkbox(ui("CIFリファレンスにhklを表示", "Show hkl on CIF references"), key="show_hkl")
    st.caption(ui("CIFはCu Kαで計算します。hklは強いピークを優先して最大12個表示します。", "CIF patterns use Cu Kα. Up to 12 strongest hkl labels are shown."))
    annotation_font_size = st.slider(ui("注記の文字サイズ", "Annotation font size"), 5, 16, _widget_default('annotation_font_size', 8), key="annotation_font_size")
    show_zoom = st.checkbox(ui("拡大図を表示", "Show zoom inset"), key="show_zoom")
    zoom_range = st.slider(ui("拡大する2θ範囲 (°)", "Inset 2θ range (°)"), 0., 180., _widget_default('zoom_range', (25., 35.)), step=.1, key="zoom_range")
    zoom_location = st.selectbox(ui("拡大図の位置", "Inset position"), ["upper left", "upper right", "lower left", "lower right"],
        format_func=lambda value: {"upper left": ui("左上", "Upper left"), "upper right": ui("右上", "Upper right"), "lower left": ui("左下", "Lower left"), "lower right": ui("右下", "Lower right")}[value], key="zoom_location")
    if show_zoom and not x_min <= zoom_range[0] < zoom_range[1] <= x_max:
        st.info(ui("拡大範囲をメイン図の2θ範囲内に設定してください。", "Set the inset range within the main 2θ range."))

st.sidebar.divider()


# ===== メインエリア =====

show_panel = st.toggle(T["show_panel"], value=_widget_default('show_panel', True, checkbox=True), key="show_panel")

if active_xrd:
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    orders, visibles, labels, colors_sel, abs_offsets = [], [], [], [], []
    sort_idx = []
    cif_orders, cif_visibles, cif_labels, cif_colors, cif_ref_lines = [], [], [], [], []
    cif_sort_idx = []
    pdf_orders, pdf_visibles, pdf_labels, pdf_colors, pdf_ref_lines = [], [], [], [], []
    pdf_sort_idx = []

    for _key, _value in (("palette", DEFAULT_PALETTE), ("palette_reverse", False),
                         ("palette_refs", False), ("palette_steps", 0),
                         ("palette_from", 1), ("palette_to", 0), ("palette_base_color", "#000000")):
        st.session_state.setdefault(_key, _value)
    if st.session_state["palette"] not in PALETTES:
        st.session_state["palette"] = DEFAULT_PALETTE
    palette_defaults, palette_count = palette_plan()

    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):

                st.markdown(ui("#### 色", "#### Colours"))
                st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key="palette",
                    format_func=lambda key: PALETTE_NAMES[_lang_code][key], on_change=_apply_palette,
                    help=ui("グラフ描画appと同じ色の組み合わせです。選ぶと、表示中のパターンに表示順で色が付きます。",
                            "The graph app's palettes. Choosing one colours the visible patterns in display order."))
                if len(active_xrd) >= 2:
                    _n = len(active_xrd)
                    _lo, _hi = palette_range(_n)
                    st.session_state["_palette_range"] = (_lo, _hi)
                    st.slider(ui("色を付ける範囲（表示順）", "Coloured patterns (display order)"), min_value=1, max_value=_n,
                        step=1, key="_palette_range", on_change=_set_palette_range, args=(_n,),
                        help=ui("この範囲のパターンだけに色の組み合わせ（濃淡）を付けます。範囲外のパターン（ベースの測定など）は「範囲外の色」（初期値は黒）になります。",
                                "Only these patterns take the palette (shades). Patterns outside the range, such as a base measurement, take the outside colour (black by default)."))
                    _names = [plain(st.session_state.get(f"_val_lbl_{i}") or os.path.splitext(active_xrd[i].name)[0])
                              for i in pattern_display_order()]
                    _outside = _names[:_lo - 1] + _names[_hi:]
                    _shaded = _names[_lo - 1:_hi]
                    st.caption(ui("色の組み合わせ：", "Palette: ") + (f"{_shaded[0]} 〜 {_shaded[-1]}" if len(_shaded) > 1 else _shaded[0])
                               + (ui("　／　範囲外：", "  /  outside: ") + "、".join(_outside) if _outside else ""))
                    if _outside:
                        st.color_picker(ui("範囲外の色", "Colour outside the range"), key="palette_base_color",
                                        on_change=_apply_palette,
                                        help=ui("範囲外のパターン（ベースの測定など）の色です。", "Colour of the patterns outside the range."))
                st.checkbox(ui("濃淡（色の順）を逆に", "Reverse shades"), key="palette_reverse", on_change=_apply_palette,
                    help=ui("「…の濃淡」では 1 つ目のパターンが濃い色になります。ほかの組み合わせでは色の順が逆になります。",
                            "With shades, the first pattern gets the dark one; other palettes reverse the order."))
                st.checkbox(ui("リファレンスにも当てはめる", "Apply to references too"), key="palette_refs", on_change=_apply_palette,
                    help=ui("CIF・PDFカードにも、パターンの続きで色を付けます。付けないときは黒から始まる色です。",
                            "Colour the CIF and PDF-card references too, after the patterns. Otherwise they start in black."))
                _pc1, _pc2 = st.columns([0.45, 0.55])
                _pc1.number_input(ui("濃淡の段数", "Shade steps"), min_value=0, max_value=MAX_SHADE_STEPS, step=1,
                    key="palette_steps", on_change=_apply_palette,
                    help=ui("0 はパターンの数で分けます。グラフ描画appの図と系列の数が違うときは、その系列の数を入れると同じ濃さになります（濃淡と viridis）。",
                            "0 splits the shades by the number of patterns. Enter the number of series of a graph-app figure to get its exact shades (shades and viridis)."))
                _pc2.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                _pc2.button(ui("色を当てはめ直す", "Re-apply"), on_click=_apply_palette, use_container_width=True,
                    help=ui("パターンごとに変えた色を戻すときや、表示順・表示を変えたあとに押します。",
                            "Restores the palette after changing single colours, the display order or visibility."))

                st.markdown(T["xrd_patterns"])
                for i, f in enumerate(active_xrd):
                    default_name = os.path.splitext(f.name)[0]
                    default_hex  = palette_defaults.get(f"xrd_color_{i}", REFERENCE_COLORS[0])

                    with st.expander(f"**{i+1}. {default_name}**", expanded=True):
                        order = st.number_input(
                            T["display_order"], value=_widget_default(f'ord_{i}', i + 1, number=True), min_value=1, max_value=10000,
                            key=f"ord_{i}",
                        )
                        visible = st.checkbox(T["show_cb"], value=_widget_default(f'vis_{i}', True, checkbox=True), key=f"vis_{i}")
                        label   = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"xrd_color_{i}", default_hex, palette_count)

                        if normalize:
                            eoff_key = f"extra_offset_{i}"
                            if eoff_key not in st.session_state:
                                st.session_state[eoff_key] = 0.0
                            st.slider(
                                T["offset_adj"],
                                min_value=-5.0, max_value=15.0,
                                step=0.05,
                                key=eoff_key,
                            )
                        else:
                            yoff = st.number_input(
                                T["y_position"], value=_widget_default(f'yoff_{i}', 0.0, number=True), step=100.0,
                                format="%.1f", key=f"yoff_{i}",
                            )
                            abs_offsets.append(yoff)

                    orders.append(order)
                    visibles.append(visible)
                    labels.append(label)
                    colors_sel.append(chosen_color)

                sort_idx = sorted(range(len(active_xrd)), key=lambda i: orders[i])

                n_xrd = len(active_xrd)
                has_any_ref = bool(active_cif) or bool(active_pdf)
                if has_any_ref:
                    st.markdown(T["reference_header"])

                if active_cif:
                    st.markdown("**CIF**")
                    for i, f in enumerate(active_cif):
                        file_stem    = os.path.splitext(f.name)[0]
                        smart_label  = extract_cif_label(f.read()) or file_stem
                        default_hex  = palette_defaults.get(f"cif_color_{i}", REFERENCE_COLORS[0])

                        with st.expander(f"**CIF {i+1}. {file_stem}**", expanded=True):
                            order   = st.number_input(
                                T["display_order"], value=_widget_default(f'cord_{i}', i + 1, number=True), min_value=1, max_value=10000,
                                key=f"cord_{i}",
                            )
                            visible = st.checkbox(T["show_cb"], value=_widget_default(f'cvis_{i}', True, checkbox=True), key=f"cvis_{i}")
                            label   = label_input(key=f"clbl_{i}", default=smart_label)
                            chosen_color = color_picker_popover(f"cif_color_{i}", default_hex, palette_count)
                            if show_ref_lines:
                                ref_line = st.checkbox(T["show_ref_line_each"], value=_widget_default(f'cref_line_{i}', True, checkbox=True), key=f"cref_line_{i}")
                            else:
                                ref_line = st.session_state.get(f"cref_line_{i}", True)

                        cif_orders.append(order)
                        cif_visibles.append(visible)
                        cif_labels.append(label)
                        cif_colors.append(chosen_color)
                        cif_ref_lines.append(ref_line)

                    cif_sort_idx = sorted(range(len(active_cif)), key=lambda i: cif_orders[i])

                if active_pdf:
                    st.markdown(T["pdf_card_md"])
                    n_offset = n_xrd + len(active_cif)
                    for i, f in enumerate(active_pdf):
                        file_stem   = os.path.splitext(f.name)[0]
                        smart_label = extract_pdf_card_label(f.read()) or file_stem
                        default_hex = palette_defaults.get(f"pdf_color_{i}", REFERENCE_COLORS[0])

                        with st.expander(f"**PDF {i+1}. {file_stem}**", expanded=True):
                            order   = st.number_input(
                                T["display_order"], value=_widget_default(f'pord_{i}', len(active_cif) + i + 1, number=True),
                                min_value=1, max_value=10000, key=f"pord_{i}",
                            )
                            visible = st.checkbox(T["show_cb"], value=_widget_default(f'pvis_{i}', True, checkbox=True), key=f"pvis_{i}")
                            label   = label_input(key=f"plbl_{i}", default=smart_label)
                            chosen_color = color_picker_popover(f"pdf_color_{i}", default_hex, palette_count)
                            if show_ref_lines:
                                ref_line = st.checkbox(T["show_ref_line_each"], value=_widget_default(f'pref_line_{i}', True, checkbox=True), key=f"pref_line_{i}")
                            else:
                                ref_line = st.session_state.get(f"pref_line_{i}", True)

                        pdf_orders.append(order)
                        pdf_visibles.append(visible)
                        pdf_labels.append(label)
                        pdf_colors.append(chosen_color)
                        pdf_ref_lines.append(ref_line)

                    pdf_sort_idx = sorted(range(len(active_pdf)), key=lambda i: pdf_orders[i])

    else:
        for i, f in enumerate(active_xrd):
            key = f"xrd_color_{i}"
            orders.append(st.session_state.get(f"ord_{i}", i + 1))
            visibles.append(st.session_state.get(f"vis_{i}", True))
            labels.append(st.session_state.get(f"_val_lbl_{i}", os.path.splitext(f.name)[0]))
            colors_sel.append(st.session_state.get(key, palette_defaults.get(key, REFERENCE_COLORS[0])))
            if normalize:
                eoff_key = f"extra_offset_{i}"
                if eoff_key not in st.session_state:
                    st.session_state[eoff_key] = 0.0
            else:
                abs_offsets.append(st.session_state.get(f"yoff_{i}", 0.0))
        sort_idx = sorted(range(len(active_xrd)), key=lambda i: orders[i])

        n_xrd = len(active_xrd)
        if active_cif:
            for i, f in enumerate(active_cif):
                key = f"cif_color_{i}"
                fallback = extract_cif_label(f.read()) or os.path.splitext(f.name)[0]
                cif_orders.append(st.session_state.get(f"cord_{i}", i + 1))
                cif_visibles.append(st.session_state.get(f"cvis_{i}", True))
                cif_labels.append(st.session_state.get(f"_val_clbl_{i}", fallback))
                cif_colors.append(st.session_state.get(key, palette_defaults.get(key, REFERENCE_COLORS[0])))
                cif_ref_lines.append(st.session_state.get(f"cref_line_{i}", True))
            cif_sort_idx = sorted(range(len(active_cif)), key=lambda i: cif_orders[i])

        if active_pdf:
            n_offset = n_xrd + len(active_cif)
            for i, f in enumerate(active_pdf):
                key = f"pdf_color_{i}"
                fallback = extract_pdf_card_label(f.read()) or os.path.splitext(f.name)[0]
                pdf_orders.append(st.session_state.get(f"pord_{i}", len(active_cif) + i + 1))
                pdf_visibles.append(st.session_state.get(f"pvis_{i}", True))
                pdf_labels.append(st.session_state.get(f"_val_plbl_{i}", fallback))
                pdf_colors.append(st.session_state.get(key, palette_defaults.get(key, REFERENCE_COLORS[0])))
                pdf_ref_lines.append(st.session_state.get(f"pref_line_{i}", True))
            pdf_sort_idx = sorted(range(len(active_pdf)), key=lambda i: pdf_orders[i])

    # ===== 統合リファレンスリストの構築（CIF + PDF、表示順でソート） =====
    def build_ref_list():
        refs = []
        for i in (cif_sort_idx or range(len(active_cif))):
            if i < len(cif_visibles) and not cif_visibles[i]:
                continue
            refs.append({
                "type": "cif",
                "file": active_cif[i],
                "label": cif_labels[i] if i < len(cif_labels) else active_cif[i].name,
                "color": cif_colors[i] if i < len(cif_colors) else "#000000",
                "order": cif_orders[i] if i < len(cif_orders) else i,
                "show_line": cif_ref_lines[i] if i < len(cif_ref_lines) else True,
            })
        for i in (pdf_sort_idx or range(len(active_pdf))):
            if i < len(pdf_visibles) and not pdf_visibles[i]:
                continue
            refs.append({
                "type": "pdf",
                "file": active_pdf[i],
                "label": pdf_labels[i] if i < len(pdf_labels) else active_pdf[i].name,
                "color": pdf_colors[i] if i < len(pdf_colors) else "#000000",
                "order": pdf_orders[i] if i < len(pdf_orders) else i,
                "show_line": pdf_ref_lines[i] if i < len(pdf_ref_lines) else True,
            })
        refs.sort(key=lambda r: r["order"])
        return refs

    # ===== 図の生成 =====
    def reference_data(ref):
        raw = ref["file"].read()
        if ref["type"] == "cif":
            x_ref, y_ref = calc_cif_pattern(raw, two_theta_range=(x_min, x_max))
            hkls = None
            if show_hkl:
                detail_x, _, detail_hkl = calc_cif_details(raw, two_theta_range=(x_min, x_max))
                if detail_x is not None and x_ref is not None and np.array_equal(detail_x, x_ref):
                    hkls = detail_hkl
        else:
            x_ref, y_ref = parse_pdf_card(raw, two_theta_range=(x_min, x_max))
            hkls = None
        return x_ref, y_ref, hkls

    @plt.rc_context()
    def build_figure(show_legend=show_legend, show_cif_legend=show_cif_legend):
        from matplotlib.ticker import MultipleLocator

        refs = build_ref_list()
        has_ref = bool(refs)
        has_xrd = bool(active_xrd) and any(visibles)

        plt.rcParams.update({**font_rcparams(plot_style), "font.size": font_size, "axes.labelsize": font_size,
                             "xtick.labelsize": tick_font_size, "ytick.labelsize": tick_font_size,
                             "legend.fontsize": legend_font_size, "legend.title_fontsize": legend_font_size})
        major_len, minor_len = plot_style["tick_length"], plot_style["tick_length"] / 2
        # Ticks of the bottom axis; between the main and reference panels they stay inward,
        # so they do not run into the reference sticks.
        bottom_dir = plot_style["tick_direction"]
        if bottom_dir == "auto":
            bottom_dir = "out" if has_ref else "in"
        main_dir = "in" if has_ref else bottom_dir

        gs = None
        if has_ref:
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[2 if show_hkl else 3, 1], hspace=0.0, figure=fig)
            ax_main = fig.add_subplot(gs[0])
            ax_ref  = fig.add_subplot(gs[1], sharex=ax_main)
            ax_main.spines["bottom"].set_color("black")
            ax_ref.spines["top"].set_visible(False)
            ax_main.tick_params(axis="x", which="both", length=0)
        else:
            fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))
            ax_ref = None

        ax_main.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax_main.tick_params(which="major", axis="x", length=major_len, direction=main_dir)
        if show_minor and minor_tick:
            ax_main.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax_main.tick_params(which="minor", axis="x", length=minor_len, direction=main_dir)
        if not plot_style["axes_box"]:
            ax_main.spines["top"].set_visible(False)
            ax_main.spines["right"].set_visible(False)
            if ax_ref is not None:
                ax_ref.spines["right"].set_visible(False)

        cumulative_y = 0.0
        side_labels  = []
        prepared_curves, phase_refs = [], []
        legend_levels, legend_lines = [], []

        if has_xrd:
            for i in sort_idx:
                if not visibles[i]:
                    continue
                data = active_xrd[i].pattern
                if data is None:
                    data = read_xrd_data(active_xrd[i].read(), active_xrd[i].name)
                if data is None:
                    continue
                x, y = data[:, 0], data[:, 1]
                mask = (x >= x_min) & (x <= x_max)
                x, y = x[mask], y[mask]
                if len(y) == 0:
                    continue

                y_min = np.min(y)
                y_max = max(np.max(y), 1e-9)

                if normalize:
                    y = y / y_max
                    y_min_n = np.min(y)
                    y_max = 1.0
                    extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))
                    baseline  = cumulative_y + extra_off
                    y_plot    = y + baseline
                    side_labels.append((baseline + y_min_n + label_offset_y * (y_max - y_min_n), colors_sel[i], labels[i]))
                    cumulative_y += global_offset * y_max
                    legend_levels.append(baseline)
                else:
                    y_plot = y + abs_offsets[i]
                    side_labels.append((abs_offsets[i] + y_min + label_offset_y * (y_max - y_min), colors_sel[i], labels[i]))
                    legend_levels.append(abs_offsets[i])

                curve, = ax_main.plot(x, y_plot, color=colors_sel[i], linewidth=plot_style["line_width"], label=rich(labels[i]))
                legend_lines.append(curve)
                prepared_curves.append(PreparedCurve(rich(labels[i]), x, y_plot, colors_sel[i], plot_style["line_width"]))

                if show_peaks:
                    pks = detect_peaks(x, y, prominence=peak_prom)
                    ax_main.plot(x[pks], y_plot[pks], "v", color=colors_sel[i], markersize=6)
                    for pk in pks:
                        ax_main.annotate(
                            f"{x[pk]:.2f}°",
                            xy=(x[pk], y_plot[pk]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=max(font_size - 4, 7),
                            color=colors_sel[i],
                        )

        if show_side_labels:
            for y_c, col, txt in side_labels:
                if label_side == "左":
                    x_pos = x_min + label_offset_x
                    ha    = "left"
                else:
                    x_pos = x_max - label_offset_x
                    ha    = "right"
                ax_main.text(
                    x_pos, y_c, rich(txt),
                    color=col, fontsize=label_fontsize,
                    ha=ha, va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
                )

        def stacked_legend():
            # The patterns from top to bottom as they are stacked, then any other entries (phase markers).
            ordered = [legend_lines[k] for k in top_to_bottom(legend_levels)]
            ordered = [h for h in ordered if h.get_label() and not h.get_label().startswith("_")]
            handles, _ = ax_main.get_legend_handles_labels()
            extras = [h for h in handles if all(h is not o for o in ordered)]
            ax_main.legend(handles=ordered + extras, loc="upper right", frameon=False)

        ax_main.set_ylabel(rich(y_title))
        ax_main.set_yticks([])
        if show_legend:
            stacked_legend()

        xlabel = rich(x_title)
        if ax_ref is None:
            ax_main.set_xlabel(xlabel)
        else:
            plt.setp(ax_main.get_xticklabels(), visible=False)

        if ax_ref is not None and refs:
            bar_height = 70.0 if show_hkl else 80.0
            row_height = 160.0 if show_hkl else 110.0
            row = 0
            for ref in refs:
                x_ref, y_ref, hkls = reference_data(ref)
                if x_ref is None or y_ref is None or len(x_ref) == 0 or len(y_ref) == 0:
                    row += 1
                    continue
                baseline = row * row_height
                y_norm = y_ref / np.max(y_ref) * bar_height if np.max(y_ref) > 0 else y_ref
                reference = ReferencePeaks(rich(ref["label"]), x_ref, y_ref, hkls, ref["color"],
                                           ["circle", "triangle", "square", "diamond"][row % 4])
                phase_refs.append(reference)
                if row > 0:
                    ax_ref.axhline(y=baseline, color="gray", linewidth=0.6,
                                   linestyle="-", alpha=0.5, zorder=0)
                ax_ref.vlines(x_ref, baseline, baseline + y_norm,
                              color=ref["color"], linewidth=plot_style["reference_line_width"], label=rich(ref["label"]))
                if show_hkl and hkls:
                    add_mpl_reference_hkl_labels(ax_ref, reference, baseline=baseline,
                        scale=bar_height / np.max(y_ref) if np.max(y_ref) > 0 else 1., font_size=annotation_font_size)
                if show_ref_lines and ref.get("show_line", True):
                    for xv in x_ref:
                        ax_main.axvline(xv, ymin=0, ymax=1,
                                       color=ref["color"], linewidth=0.7,
                                       linestyle="--", alpha=0.4, zorder=0)
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    lha = "left" if cif_label_side == "左" else "right"
                    ax_ref.text(
                        lx, baseline + cif_label_offset_y * row_height,
                        (["○ ", "△ ", "□ ", "◇ "][row % 4] if show_phase_markers else "") + rich(ref["label"]), fontsize=cif_label_fontsize,
                        color=ref["color"], va="center", ha=lha,
                    )
                row += 1
            ax_ref.axhline(y=row * row_height, color="gray", linewidth=0.6,
                           linestyle="-", alpha=0.5)
            ax_ref.set_xlabel(xlabel)
            ax_ref.set_yticks([])
            ax_ref.set_ylim(0, row * row_height)
            ax_ref.xaxis.set_major_locator(MultipleLocator(major_tick))
            ax_ref.tick_params(which="major", axis="x", length=major_len, direction=bottom_dir)
            if show_minor and minor_tick:
                ax_ref.xaxis.set_minor_locator(MultipleLocator(minor_tick))
                ax_ref.tick_params(which="minor", axis="x", length=minor_len, direction=bottom_dir)

        ax_main.set_xlim(x_min, x_max)
        apply_mpl_style(fig, plot_style)
        if show_phase_markers:
            matches = match_reference_peaks(prepared_curves, phase_refs, phase_tolerance)
            add_mpl_phase_annotations(ax_main, matches, show_hkl=False, font_size=annotation_font_size, show_legend=show_legend)
            if show_legend:
                stacked_legend()
        if show_zoom and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_mpl_zoom_inset(ax_main, prepared_curves, zoom_range, location=zoom_location, font_size=annotation_font_size)
        return fig

    # ===== インタラクティブ Plotly プレビュー =====
    def build_plotly_figure(show_legend=show_legend, show_cif_legend=show_cif_legend):
        refs = build_ref_list()
        has_ref = bool(refs)

        if has_ref:
            pfig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[2 / 3, 1 / 3] if show_hkl else [0.75, 0.25],
                vertical_spacing=0,
            )
        else:
            pfig = make_subplots(rows=1, cols=1)

        cumulative_y = 0.0
        prepared_curves, phase_refs = [], []
        legend_levels, legend_traces = [], []

        for i in sort_idx:
            if not visibles[i]:
                continue
            data = active_xrd[i].pattern
            if data is None:
                data = read_xrd_data(active_xrd[i].read(), active_xrd[i].name)
            if data is None:
                continue
            x, y = data[:, 0], data[:, 1]
            mask = (x >= x_min) & (x <= x_max)
            x, y = x[mask], y[mask]
            if len(y) == 0:
                continue
            y_min = np.min(y)
            y_max = max(np.max(y), 1e-9)

            if normalize:
                y = y / y_max
                y_min_n = np.min(y)
                y_max = 1.0
                extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))
                y_plot  = y + cumulative_y + extra_off
                y_label = cumulative_y + extra_off + y_min_n + label_offset_y * (y_max - y_min_n)
                legend_levels.append(cumulative_y + extra_off)
                cumulative_y += global_offset * y_max
            else:
                y_plot  = y + abs_offsets[i]
                y_label = abs_offsets[i] + y_min + label_offset_y * (y_max - y_min)
                legend_levels.append(abs_offsets[i])

            pfig.add_trace(go.Scatter(
                x=x, y=y_plot, name=mpl_to_plotly(labels[i]),
                line=dict(color=colors_sel[i], width=plot_style["line_width"] * 4 / 3),
                mode="lines", showlegend=show_legend,
            ), row=1, col=1)
            legend_traces.append(len(pfig.data) - 1)
            prepared_curves.append(PreparedCurve(mpl_to_plotly(labels[i]), x, y_plot, colors_sel[i], plot_style["line_width"]))

            if show_peaks:
                pks = detect_peaks(x, y, prominence=peak_prom)
                if len(pks) > 0:
                    pfig.add_trace(go.Scatter(
                        x=x[pks], y=y_plot[pks],
                        mode="markers+text",
                        marker=dict(color=colors_sel[i], symbol="triangle-down", size=8),
                        text=[f"{x[pk]:.2f}°" for pk in pks],
                        textposition="top center",
                        textfont=dict(size=max(font_size - 4, 7), color=colors_sel[i]),
                        showlegend=False,
                    ), row=1, col=1)

            if show_side_labels:
                x_pos = (x_min + label_offset_x) if label_side == "左" else (x_max - label_offset_x)
                textpos = "middle right" if label_side == "左" else "middle left"
                pfig.add_trace(go.Scatter(
                    x=[x_pos], y=[y_label],
                    mode="text",
                    text=[mpl_to_plotly(labels[i])],
                    textposition=textpos,
                    textfont=dict(color=colors_sel[i], size=label_fontsize, family=css_font_stack(plot_style["font_family"])),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=1, col=1)

        # Legend from top to bottom as the patterns are stacked (phase markers keep the default rank, after them).
        for rank, k in enumerate(top_to_bottom(legend_levels), start=1):
            pfig.data[legend_traces[k]].legendrank = rank

        if has_ref and refs:
            bar_height = 70.0 if show_hkl else 80.0
            row_height = 160.0 if show_hkl else 110.0
            row = 0
            for ref in refs:
                x_ref, y_ref, hkls = reference_data(ref)
                if x_ref is None or y_ref is None or len(x_ref) == 0 or len(y_ref) == 0:
                    row += 1
                    continue
                baseline = row * row_height
                y_norm = y_ref / np.max(y_ref) * bar_height if np.max(y_ref) > 0 else y_ref
                reference = ReferencePeaks(mpl_to_plotly(ref["label"]), x_ref, y_ref, hkls, ref["color"],
                                           ["circle", "triangle", "square", "diamond"][row % 4])
                phase_refs.append(reference)
                vx, vy = [], []
                for xi, yi in zip(x_ref, y_norm):
                    vx += [float(xi), float(xi), None]
                    vy += [float(baseline), float(baseline + yi), None]
                pfig.add_trace(go.Scatter(
                    x=vx, y=vy, name=mpl_to_plotly(ref["label"]),
                    line=dict(color=ref["color"], width=plot_style["reference_line_width"] * 4 / 3),
                    mode="lines", showlegend=False,
                ), row=2, col=1)
                if show_hkl and hkls:
                    add_plotly_reference_hkl_labels(pfig, reference, baseline=baseline,
                        scale=bar_height / np.max(y_ref) if np.max(y_ref) > 0 else 1.,
                        font_size=annotation_font_size, xref="x2", yref="y2")
                if show_ref_lines and ref.get("show_line", True):
                    for xv in x_ref:
                        pfig.add_shape(
                            type="line",
                            x0=float(xv), x1=float(xv),
                            y0=0, y1=1,
                            xref="x", yref="y domain",
                            line=dict(color=ref["color"], width=0.7, dash="dash"),
                            opacity=0.4,
                            row=1, col=1,
                        )
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    textpos_cif = "middle right" if cif_label_side == "左" else "middle left"
                    pfig.add_trace(go.Scatter(
                        x=[lx],
                        y=[baseline + cif_label_offset_y * row_height],
                        mode="text",
                        text=[(["○ ", "△ ", "□ ", "◇ "][row % 4] if show_phase_markers else "") + mpl_to_plotly(ref["label"])],
                        textposition=textpos_cif,
                        textfont=dict(color=ref["color"], size=cif_label_fontsize, family=css_font_stack(plot_style["font_family"])),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=2, col=1)
                row += 1
            pfig.update_layout(yaxis2=dict(
                range=[0, row * row_height], autorange=False,
                showticklabels=False, showline=True, linecolor="black",
                showgrid=False, zeroline=False,
            ))

        xlabel = mpl_to_plotly(x_title)
        tick_px = plot_style["tick_length"] * 4 / 3
        bottom_ticks = {"in": "inside", "out": "outside"}.get(plot_style["tick_direction"],
                                                             "outside" if has_ref else "inside")
        main_ticks = "inside" if has_ref else bottom_ticks
        minor_cfg = (dict(ticks=main_ticks, dtick=minor_tick, ticklen=tick_px / 2, showgrid=False)
                     if (show_minor and minor_tick) else {})
        pfig.update_xaxes(
            range=[x_min, x_max], dtick=major_tick, ticks=main_ticks, ticklen=tick_px,
            showline=True, linecolor="black", mirror=plot_style["axes_box"],
            showgrid=False, zeroline=False,
            tickfont=dict(color="black", size=tick_font_size),
            title_font=dict(color="black", size=font_size),
            minor=minor_cfg,
        )
        if has_ref:
            pfig.update_xaxes(ticks=bottom_ticks, row=2, col=1)
            if show_minor and minor_tick:
                pfig.update_xaxes(minor=dict(ticks=bottom_ticks, dtick=minor_tick, ticklen=tick_px / 2), row=2, col=1)
        pfig.update_yaxes(
            showticklabels=False, showline=True, linecolor="black", mirror=plot_style["axes_box"],
            showgrid=False, zeroline=False,
            title_font=dict(color="black", size=font_size),
        )

        if has_ref:
            pfig.update_xaxes(title_text=xlabel, row=2, col=1)
            pfig.update_xaxes(showticklabels=False, row=1, col=1)
            pfig.update_yaxes(title_text=mpl_to_plotly(y_title), row=1, col=1)
        else:
            pfig.update_xaxes(title_text=xlabel, row=1, col=1)
            pfig.update_yaxes(title_text=mpl_to_plotly(y_title), row=1, col=1)

        pfig.update_layout(
            dragmode="zoom",
            font=dict(family=css_font_stack(plot_style["font_family"]), size=font_size, color="black"),
            height=max(320, min(1800, int(fig_height * 96))),
            plot_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            paper_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            showlegend=show_legend or show_cif_legend,
            legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top",
                        font=dict(color="black", size=legend_font_size)),
            margin=dict(l=int(fig_width * 96 * plot_style["margin_left"]),
                        r=int(fig_width * 96 * (1 - plot_style["margin_right"])),
                        t=int(fig_height * 96 * (1 - plot_style["margin_top"])),
                        b=int(fig_height * 96 * plot_style["margin_bottom"])),
        )
        pfig.update_xaxes(linewidth=plot_style["axes_line_width"] * 4 / 3)
        pfig.update_yaxes(linewidth=plot_style["axes_line_width"] * 4 / 3)
        if show_phase_markers:
            matches = match_reference_peaks(prepared_curves, phase_refs, phase_tolerance)
            add_plotly_phase_annotations(pfig, matches, show_hkl=False, font_size=annotation_font_size, show_legend=show_legend)
        if show_zoom and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_plotly_zoom_inset(pfig, prepared_curves, zoom_range, location=zoom_location, font_size=annotation_font_size)
        return pfig

    # ===== 描画 & ダウンロード =====
    with col_graph:
        interactive_tab, export_tab = st.tabs([T["interactive_preview"], T["export_preview"]])
        pfig = build_plotly_figure()
        with interactive_tab:
            st.caption(T["drag_zoom"])
            st.plotly_chart(pfig, use_container_width=True, config={
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["drawrect"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })

        _sl  = bool(show_legend)
        _scl = bool(show_cif_legend)
        fig = build_figure(show_legend=_sl, show_cif_legend=_scl)
        png_bytes = export_figure(fig, "png", 150, style=plot_style)
        pdf_bytes = export_figure(fig, "pdf", dpi_export, style=plot_style)
        svg_bytes = export_figure(fig, "svg", dpi_export, style=plot_style)
        # Bound the in-memory raster canvas for the shared web and PC editions.
        _raster_ok = fig_width * fig_height * dpi_export ** 2 <= 36_000_000
        png_export_bytes = export_figure(fig, "png", dpi_export, style=plot_style) if _raster_ok else None
        tiff_bytes = export_figure(fig, "tiff", dpi_export, style=plot_style) if _raster_ok else None
        plt.close(fig)

        with export_tab:
            st.image(png_bytes, **FULL_WIDTH_IMAGE)
            st.caption(T["export_preview_caption"].format(
                width=inches_to_mm(fig_width), height=inches_to_mm(fig_height), dpi=dpi_export,
            ))
            st.caption(ui("軸や文字が端で切れる場合は、余白または文字サイズを調整してください。", "If text or axes are clipped, adjust the margins or font sizes."))

        st.sidebar.download_button(
            ui("📥 PDF として保存", "📥 Save as PDF"),
            data=pdf_bytes, file_name="xrd_result.pdf", mime="application/pdf", use_container_width=True,
        )
        st.sidebar.download_button(
            ui("📥 SVG として保存", "📥 Save as SVG"),
            data=svg_bytes, file_name="xrd_result.svg", mime="image/svg+xml", use_container_width=True,
        )
        if not _raster_ok:
            st.sidebar.warning(ui("PNG / TIFFは3,600万画素までです。図の寸法またはDPIを下げてください。PDF / SVGは保存できます。", "For PNG / TIFF, reduce the dimensions or DPI to stay within 36 megapixels. PDF / SVG are available."))
        if _raster_ok:
            st.sidebar.download_button(
                T["save_tiff"].format(dpi=dpi_export),
                data=tiff_bytes, file_name="xrd_result.tiff", mime="image/tiff",
                key=f"dl_tiff_{_sl}_{_scl}_{dpi_export}",
                use_container_width=True,
            )
            st.sidebar.download_button(
                T["save_png"].format(dpi=dpi_export),
                data=png_export_bytes, file_name="xrd_result.png", mime="image/png",
                key=f"dl_png_{_sl}_{_scl}",
                use_container_width=True,
            )
        st.sidebar.divider()

else:
    st.info(T["upload_prompt"])
    if not PYMATGEN_AVAILABLE:
        st.warning(T["no_pymatgen"])
    if not PDFPLUMBER_AVAILABLE:
        st.warning(T["no_pdfplumber"])

if any(st.session_state.get(f"_{kind}_bytes") for kind in ("xrd", "cif", "pdf")):
    try:
        _project_bytes = build_session_json(len(active_xrd), len(active_cif), len(active_pdf))
        st.sidebar.download_button(
            T["save_session"],
            data=_project_bytes,
            file_name="xrd_project.json",
            mime="application/json",
            use_container_width=True,
        )

        if LOCAL_DATA_DIR:
            try:
                _fingerprint = project_fingerprint(_project_bytes)
                if _fingerprint != st.session_state.get("_autosave_fingerprint"):
                    atomic_save(LOCAL_DATA_DIR / "autosave.json", _project_bytes)
                    st.session_state["_autosave_fingerprint"] = _fingerprint
                    st.session_state["_autosave_time"] = datetime.now().astimezone().strftime("%H:%M:%S")
                autosave_status.caption(T["autosaved"].format(time=st.session_state.get("_autosave_time", "")))
            except (OSError, ProjectError) as _error:
                autosave_status.warning(T["autosave_error"] + str(_error))
    except ProjectError as _error:
        autosave_status.warning(T["project_save_error"] + str(_error))

st.sidebar.divider()
try:
    from build_bundle import create_bundle_bytes
    st.sidebar.download_button(
        T["download_desktop"], data=create_bundle_bytes(),
        file_name="XRD-Maker-PC.zip", mime="application/zip",
        use_container_width=True,
    )
    st.sidebar.caption(T["desktop_help"])
except (ImportError, FileNotFoundError):
    pass

# ===== 書式のショートカット（グラフ描画appと同じ） =====
# 入力欄の文字を選んで Ctrl+I（斜体）・Ctrl+B（太字）・Ctrl+,（下付き）・Ctrl+.（上付き）。Word の Ctrl+= / Ctrl+Shift+= も。
# Mac は ⌘ でも可。<i>…</i>、<b>…</b>、_{…}、^{…} を書き足すだけなので、保存される作業ファイルはどの版でも同じ。
_SHORTCUTS_HTML = """<script>
(function () {
  const P = window.parent;
  if (!P || !P.document) return;
  const handler = function (e) {
    const el = e.target;
    if (!el || !(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || !el.type)))) return;
    if (el.getAttribute('role') === 'combobox' || (el.closest && el.closest('[data-baseweb="select"]'))) return;
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = e.key;
    let open = null, close = null;
    if (k === 'i' || k === 'I') { open = '<i>'; close = '</i>'; }
    else if (k === 'b' || k === 'B') { open = '<b>'; close = '</b>'; }
    else if (k === ',' || k === '=') { open = '_{'; close = '}'; }
    else if (k === '.' || k === '+') { open = '^{'; close = '}'; }
    if (!open) return;
    e.preventDefault();
    e.stopPropagation();
    const v = el.value || '';
    const a = (el.selectionStart == null) ? v.length : el.selectionStart;
    const b = (el.selectionEnd == null) ? a : el.selectionEnd;
    const nv = v.slice(0, a) + open + v.slice(a, b) + close + v.slice(b);
    const proto = el.tagName === 'TEXTAREA' ? P.HTMLTextAreaElement.prototype : P.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, nv);
    el.dispatchEvent(new P.Event('input', { bubbles: true }));
    el.setSelectionRange(a + open.length, b + open.length);
  };
  if (P.__xrdMakerKeys) P.document.removeEventListener('keydown', P.__xrdMakerKeys, true);
  P.__xrdMakerKeys = handler;
  P.document.addEventListener('keydown', handler, true);
})();
</script>"""
if hasattr(st, "iframe"):     # newer Streamlit, where components.html is being removed
    st.iframe(_SHORTCUTS_HTML, height="content")
else:                         # Streamlit 1.32 and other versions before st.iframe
    import streamlit.components.v1 as components
    components.html(_SHORTCUTS_HTML, height=0)
