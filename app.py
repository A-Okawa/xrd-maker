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
import io
import os
import re
import tempfile
import json
import base64

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
        "app_caption": "複数パターン重ね合わせ・CIF/PDFカード リファレンス・論文用TIFF出力",
        "color_red": "赤系",
        "color_orange": "橙・黄系",
        "color_green": "緑系",
        "color_blue": "青系",
        "color_purple": "紫・ピンク系",
        "color_brown": "茶・ベージュ系",
        "color_gray": "黒・グレー系",
        "label": "ラベル",
        "add_italic": "＋italic",
        "italic_help": r"末尾に $\it{TEXT}$ を追加。TEXTを書き換えて使用",
        "add_subscript": "＋下付き",
        "subscript_help": r"末尾に $_{N}$ を追加。Nを書き換えて使用",
        "add_superscript": "＋上付き",
        "superscript_help": r"末尾に $^{N}$ を追加。Nを書き換えて使用",
        "change_color": "🎨 色を変更",
        "session_header": "💾 セッション",
        "load_session": "セッションを読み込む (.json)",
        "xrd_data_header": "📂 XRDデータ",
        "xrd_upload": "XRDデータ (.xy / .txt / .csv)",
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
        "save_session": "💾 セッションを保存 (.json)",
        "show_panel": "⚙️ パターン設定パネルを表示",
        "xrd_patterns": "#### XRD パターン",
        "display_order": "表示順",
        "show_cb": "表示する",
        "offset_adj": "オフセット調整",
        "y_position": "Y位置（絶対値）",
        "reference_header": "#### リファレンス",
        "pdf_card_md": "**PDFカード**",
        "drag_zoom": "ドラッグで範囲ズーム ／ ダブルクリックでリセット",
        "save_tiff": "📥 TIFF として保存 ({dpi} DPI)",
        "save_png": "📥 PNG として保存（確認用）",
        "upload_prompt": "サイドバーから XRD データファイル（.xy / .txt / .csv）をアップロードしてください。",
        "no_pymatgen": "pymatgen が未インストールのため CIF リファレンス機能は無効です。",
        "no_pdfplumber": "pdfplumber が未インストールのため PDFカード リファレンス機能は無効です。",
        "cif_error": "CIF エラー: ",
        "pdf_error": "PDFカード解析エラー: ",
    },
    "en": {
        "password_prompt": "Enter password",
        "login": "Login",
        "wrong_password": "Incorrect password",
        "app_caption": "Multi-pattern overlay · CIF/PDF card reference · Publication-quality TIFF export",
        "color_red": "Reds",
        "color_orange": "Orange/Yellow",
        "color_green": "Greens",
        "color_blue": "Blues",
        "color_purple": "Purple/Pink",
        "color_brown": "Brown/Beige",
        "color_gray": "Black/Gray",
        "label": "Label",
        "add_italic": "+italic",
        "italic_help": r"Appends $\it{TEXT}$ at the end. Replace TEXT.",
        "add_subscript": "+Subscript",
        "subscript_help": r"Appends $_{N}$ at the end. Replace N.",
        "add_superscript": "+Superscript",
        "superscript_help": r"Appends $^{N}$ at the end. Replace N.",
        "change_color": "🎨 Change color",
        "session_header": "💾 Session",
        "load_session": "Load session (.json)",
        "xrd_data_header": "📂 XRD Data",
        "xrd_upload": "XRD Data (.xy / .txt / .csv)",
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
        "save_session": "💾 Save session (.json)",
        "show_panel": "⚙️ Show pattern settings panel",
        "xrd_patterns": "#### XRD Patterns",
        "display_order": "Display order",
        "show_cb": "Show",
        "offset_adj": "Offset adjustment",
        "y_position": "Y position (absolute)",
        "reference_header": "#### Reference",
        "pdf_card_md": "**PDF Card**",
        "drag_zoom": "Drag to zoom / Double-click to reset",
        "save_tiff": "📥 Save as TIFF ({dpi} DPI)",
        "save_png": "📥 Save as PNG (for preview)",
        "upload_prompt": "Upload XRD data files (.xy / .txt / .csv) from the sidebar.",
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

ALL_COLORS = [h for fam in _COLOR_FAMILIES_DATA.values() for h, _ in fam]

st.set_page_config(page_title="XRD Maker", page_icon="🔬", layout="wide")

# ===== 言語選択 / Language selection =====
_lang_option = st.sidebar.radio(
    "🌐 Language / 言語", ["日本語", "English"],
    horizontal=True, key="lang_radio",
)
_lang_code = "en" if _lang_option == "English" else "ja"
T = TRANSLATIONS[_lang_code]

# 言語付き色ファミリー名
COLOR_FAMILIES = OrderedDict(
    (T[_COLOR_FAMILY_NAME_KEYS[k]], v)
    for k, v in _COLOR_FAMILIES_DATA.items()
)

# ===== パスワード認証 =====
def check_password():
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


# ===== RestoredFile =====
class RestoredFile:
    """セッション復元時のファイル代替オブジェクト"""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def seek(self, pos: int):
        pass


def mpl_to_plotly(text: str) -> str:
    """matplotlib math notation ($_{x}$, $^{x}$, $\it{x}$) → Plotly HTML."""
    import re
    text = re.sub(r'\$_\{([^}]+)\}\$', r'<sub>\1</sub>', text)
    text = re.sub(r'\$\^\{([^}]+)\}\$', r'<sup>\1</sup>', text)
    text = re.sub(r'\$\\it\{([^}]+)\}\$', r'<i>\1</i>', text)
    return text


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
        st.session_state[val_key] = inp + r"$\it{TEXT}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button(T["add_subscript"], key=f"_bs_{key}", use_container_width=True,
                 help=T["subscript_help"]):
        st.session_state[val_key] = inp + r"$_{N}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button(T["add_superscript"], key=f"_bp_{key}", use_container_width=True,
                 help=T["superscript_help"]):
        st.session_state[val_key] = inp + r"$^{N}$"
        st.session_state[ver_key] += 1
        st.rerun()
    return st.session_state[val_key]


# ===== データ読み込み =====
def read_xrd_data(file_bytes: bytes, filename: str = ""):
    ext = os.path.splitext(filename)[1].lower()
    text = file_bytes.decode("utf-8", errors="ignore")
    if ext == ".csv":
        try:
            df = pd.read_csv(io.StringIO(text), header=None)
            try:
                pd.to_numeric(df.iloc[0, 0])
            except (ValueError, TypeError):
                df = pd.read_csv(io.StringIO(text))
            arr = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().values
            return arr if len(arr) > 0 else None
        except Exception:
            pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = line.replace(",", " ").split()
        if len(vals) < 2:
            continue
        try:
            x = float(vals[0])
            y_str = "".join(c for c in vals[1] if c.isdigit() or c in ".+-eE")
            rows.append([x, float(y_str)])
        except ValueError:
            continue
    return np.array(rows) if rows else None


def calc_cif_pattern(cif_bytes: bytes, two_theta_range=(5, 90)):
    if not PYMATGEN_AVAILABLE:
        return None, None
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_bytes)
        tmp_path = tmp.name
    try:
        parser = CifParser(tmp_path)
        structs = parser.parse_structures(primitive=False)
        if not structs:
            return None, None
        calc = XRDCalculator(wavelength="CuKa", symprec=0.01)
        pat = calc.get_pattern(structs[0], two_theta_range=(5, 90))
        mask = (pat.x >= two_theta_range[0]) & (pat.x <= two_theta_range[1])
        return pat.x[mask], pat.y[mask]
    except Exception as e:
        st.warning(T["cif_error"] + str(e))
        return None, None
    finally:
        os.unlink(tmp_path)


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
    if len(y) == 0:
        return np.array([])
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    dist = max(1, int(min_dist_deg / dx))
    pks, _ = find_peaks(y, prominence=prominence * np.max(y), distance=dist)
    return pks


# ===== カラーポップアップ =====
def color_picker_popover(key: str, default_hex: str):
    if key not in st.session_state:
        st.session_state[key] = default_hex
    current = st.session_state[key]

    st.markdown(
        f'<div style="background:{current};height:20px;border-radius:5px;'
        f'border:1px solid #ccc;margin-bottom:4px"></div>',
        unsafe_allow_html=True,
    )
    with st.popover(T["change_color"], use_container_width=True):
        for family, colors in COLOR_FAMILIES.items():
            st.caption(family)
            cols = st.columns(len(colors))
            for j, (hex_c, name) in enumerate(colors):
                with cols[j]:
                    selected = (hex_c == current)
                    st.markdown(
                        f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                        f'border:{"3px solid #333" if selected else "1px solid #ccc"};'
                        f'margin-bottom:2px"></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("✓" if selected else " ",
                                 key=f"{key}_{family}_{j}", help=name,
                                 use_container_width=True):
                        st.session_state[key] = hex_c
                        st.rerun()
    return st.session_state[key]


# ===== セッション保存/復元 =====
GLOBAL_KEYS = [
    "normalize", "show_legend", "show_cif_legend", "show_ref_lines", "show_peaks",
    "peak_prom", "global_offset", "xrange", "major_tick", "show_minor",
    "minor_tick", "cif_label_side", "cif_label_fontsize",
    "cif_label_offset_x", "cif_label_offset_y", "show_side_labels",
    "label_side", "label_fontsize", "label_offset_x", "label_offset_y",
    "fig_width", "fig_height", "dpi_export", "font_size", "show_panel",
]


def build_session_json(n_xrd: int, n_cif: int, n_pdf: int) -> bytes:
    settings = {}
    for k in GLOBAL_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            settings[k] = list(v) if isinstance(v, tuple) else v
    for i in range(n_xrd):
        for pat in ["vis_{}", "ord_{}", "_val_lbl_{}", "xrd_color_{}",
                    "extra_offset_{}", "yoff_{}"]:
            k = pat.format(i)
            if k in st.session_state:
                settings[k] = st.session_state[k]
    for i in range(n_cif):
        for pat in ["cvis_{}", "cord_{}", "_val_clbl_{}", "cif_color_{}", "cref_line_{}"]:
            k = pat.format(i)
            if k in st.session_state:
                settings[k] = st.session_state[k]
    for i in range(n_pdf):
        for pat in ["pvis_{}", "pord_{}", "_val_plbl_{}", "pdf_color_{}", "pref_line_{}"]:
            k = pat.format(i)
            if k in st.session_state:
                settings[k] = st.session_state[k]
    xrd_encoded = {
        name: base64.b64encode(data).decode()
        for name, data in st.session_state.get("_xrd_bytes", {}).items()
    }
    cif_encoded = {
        name: base64.b64encode(data).decode()
        for name, data in st.session_state.get("_cif_bytes", {}).items()
    }
    pdf_encoded = {
        name: base64.b64encode(data).decode()
        for name, data in st.session_state.get("_pdf_bytes", {}).items()
    }
    return json.dumps({
        "version": 2,
        "xrd_files": xrd_encoded,
        "cif_files": cif_encoded,
        "pdf_files": pdf_encoded,
        "settings": settings,
    }, ensure_ascii=False).encode("utf-8")


def restore_session(json_bytes: bytes):
    data = json.loads(json_bytes.decode("utf-8"))
    st.session_state["_xrd_bytes"] = {
        name: base64.b64decode(b64)
        for name, b64 in data.get("xrd_files", {}).items()
    }
    st.session_state["_cif_bytes"] = {
        name: base64.b64decode(b64)
        for name, b64 in data.get("cif_files", {}).items()
    }
    st.session_state["_pdf_bytes"] = {
        name: base64.b64decode(b64)
        for name, b64 in data.get("pdf_files", {}).items()
    }
    for k, v in data.get("settings", {}).items():
        if k == "xrange" and isinstance(v, list):
            v = tuple(v)
        st.session_state[k] = v
    n_xrd = len(st.session_state["_xrd_bytes"])
    n_cif = len(st.session_state["_cif_bytes"])
    n_pdf = len(st.session_state["_pdf_bytes"])
    for i in range(n_xrd):
        vk = f"_ver_lbl_{i}"
        st.session_state[vk] = st.session_state.get(vk, 0) + 1
    for i in range(n_cif):
        vk = f"_ver_clbl_{i}"
        st.session_state[vk] = st.session_state.get(vk, 0) + 1
    for i in range(n_pdf):
        vk = f"_ver_plbl_{i}"
        st.session_state[vk] = st.session_state.get(vk, 0) + 1
    st.rerun()


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
                st.session_state[new_k] = val


# ===== サイドバー =====

st.sidebar.caption("Last updated: 2026-06-20")

st.sidebar.header(T["xrd_data_header"])
xrd_files = st.sidebar.file_uploader(
    T["xrd_upload"],
    type=["xy", "txt", "csv"], accept_multiple_files=True,
)
st.sidebar.header(T["ref_header"])
cif_files = st.sidebar.file_uploader(
    T["cif_upload"], type=["cif"], accept_multiple_files=True,
)
pdf_ref_files = st.sidebar.file_uploader(
    T["pdf_upload"], type=["pdf"], accept_multiple_files=True,
)

if xrd_files:
    _xrd_new = [f.name for f in xrd_files]
    _xrd_old = list(st.session_state.get("_xrd_bytes", {}).keys())
    if _xrd_new != _xrd_old:
        _remap_file_session_state(
            old_names=_xrd_old, new_names=_xrd_new,
            key_tmpls=["vis_{}", "ord_{}", "_val_lbl_{}", "_ver_lbl_{}", "xrd_color_{}", "extra_offset_{}", "yoff_{}"],
        )
    st.session_state["_xrd_bytes"] = {f.name: f.read() for f in xrd_files}
    for f in xrd_files:
        f.seek(0)
if cif_files:
    _cif_new = [f.name for f in cif_files]
    _cif_old = list(st.session_state.get("_cif_bytes", {}).keys())
    if _cif_new != _cif_old:
        _remap_file_session_state(
            old_names=_cif_old, new_names=_cif_new,
            key_tmpls=["cvis_{}", "cord_{}", "_val_clbl_{}", "_ver_clbl_{}", "cif_color_{}", "cref_line_{}"],
        )
    st.session_state["_cif_bytes"] = {f.name: f.read() for f in cif_files}
    for f in cif_files:
        f.seek(0)
if pdf_ref_files:
    _pdf_new = [f.name for f in pdf_ref_files]
    _pdf_old = list(st.session_state.get("_pdf_bytes", {}).keys())
    if _pdf_new != _pdf_old:
        _remap_file_session_state(
            old_names=_pdf_old, new_names=_pdf_new,
            key_tmpls=["pvis_{}", "pord_{}", "_val_plbl_{}", "_ver_plbl_{}", "pdf_color_{}", "pref_line_{}"],
        )
    st.session_state["_pdf_bytes"] = {f.name: f.read() for f in pdf_ref_files}
    for f in pdf_ref_files:
        f.seek(0)

active_xrd = [RestoredFile(n, d) for n, d in st.session_state.get("_xrd_bytes", {}).items()]
active_cif = [RestoredFile(n, d) for n, d in st.session_state.get("_cif_bytes", {}).items()]
active_pdf = [RestoredFile(n, d) for n, d in st.session_state.get("_pdf_bytes", {}).items()]

if active_xrd and not xrd_files:
    st.sidebar.caption(T["restored_xrd"] + ", ".join(f.name for f in active_xrd))
if active_cif and not cif_files:
    st.sidebar.caption(T["restored_cif"] + ", ".join(f.name for f in active_cif))
if active_pdf and not pdf_ref_files:
    st.sidebar.caption(T["restored_pdf"] + ", ".join(f.name for f in active_pdf))

st.sidebar.header(T["graph_settings"])
xrange         = st.sidebar.slider(T["xrange"], 5.0, 90.0, (10.0, 80.0), step=0.5, key="xrange")
x_min, x_max   = xrange
normalize      = st.sidebar.checkbox(T["normalize_cb"], value=False, key="normalize")
if "show_legend" not in st.session_state:
    st.session_state["show_legend"] = True
show_legend    = st.sidebar.checkbox(T["show_legend_cb"], key="show_legend")
if "show_cif_legend" not in st.session_state:
    st.session_state["show_cif_legend"] = True
show_cif_legend= st.sidebar.checkbox(T["show_cif_legend_cb"], key="show_cif_legend")
show_ref_lines = st.sidebar.checkbox(T["show_ref_lines_cb"], value=False, key="show_ref_lines")
show_peaks     = st.sidebar.checkbox(T["show_peaks_cb"], value=False, key="show_peaks")
peak_prom      = st.sidebar.slider(T["peak_sensitivity"], 0.01, 0.5, 0.1, step=0.01, key="peak_prom") if show_peaks else 0.1
global_offset  = st.sidebar.slider(T["offset_multiplier"], 0.0, 3.0, 1.0, step=0.05, key="global_offset") if normalize else None

st.sidebar.subheader(T["tick_settings"])
major_tick  = st.sidebar.number_input(T["major_tick"], min_value=1.0, max_value=30.0,
                                       value=10.0, step=1.0, key="major_tick")
show_minor  = st.sidebar.checkbox(T["show_minor_cb"], value=True, key="show_minor")
minor_tick  = st.sidebar.number_input(T["minor_tick"], min_value=0.5, max_value=10.0,
                                       value=2.0, step=0.5, key="minor_tick") if show_minor else None

# 左/右 の内部値は言語に関わらず固定し、format_func で表示を翻訳する
_LR_OPTIONS = ["左", "右"]
_lr_fmt = lambda x: T["left"] if x == "左" else T["right"]

if show_cif_legend:
    st.sidebar.subheader(T["icdd_label_settings"])
    cif_label_side     = st.sidebar.radio(T["icdd_label_pos"], _LR_OPTIONS,
                                           format_func=_lr_fmt, horizontal=True,
                                           key="cif_label_side")
    cif_label_fontsize = st.sidebar.slider(T["icdd_label_fontsize"], 5, 20, 9, key="cif_label_fontsize")
    cif_label_offset_x = st.sidebar.slider(T["icdd_offset_x"], 0.0, 10.0, 0.5, step=0.1,
                                            help=T["icdd_offset_x_help"], key="cif_label_offset_x")
    cif_label_offset_y = st.sidebar.slider(T["icdd_offset_y"], 0.0, 1.0, 0.5, step=0.05,
                                            help=T["icdd_offset_y_help"], key="cif_label_offset_y")
else:
    cif_label_side, cif_label_fontsize = "左", 9
    cif_label_offset_x, cif_label_offset_y = 0.5, 0.5

st.sidebar.subheader(T["sample_labels"])
show_side_labels = st.sidebar.checkbox(T["show_side_labels_cb"], value=False, key="show_side_labels")
if show_side_labels:
    label_side      = st.sidebar.radio(T["label_pos"], _LR_OPTIONS,
                                        format_func=_lr_fmt, horizontal=True,
                                        key="label_side")
    label_fontsize  = st.sidebar.slider(T["label_fontsize"], 5, 24, 11, key="label_fontsize")
    label_offset_x  = st.sidebar.slider(T["label_offset_x"], -5.0, 5.0, 0.5, step=0.1,
                                         help=T["label_offset_x_help"], key="label_offset_x")
    label_offset_y  = st.sidebar.slider(T["label_offset_y"], -0.3, 1.0, 0.05, step=0.01,
                                         help=T["label_offset_y_help"], key="label_offset_y")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 11, 0.5, 0.05

st.sidebar.header(T["fig_size"])
fig_width  = st.sidebar.slider(T["fig_width"], 6.0, 20.0, 10.0, step=0.5, key="fig_width")
fig_height = st.sidebar.slider(T["fig_height"], 4.0, 20.0, 8.0, step=0.5, key="fig_height")
dpi_export = st.sidebar.selectbox("Output DPI", [300, 600], index=0, key="dpi_export")
font_size  = st.sidebar.slider(T["font_size"], 8, 20, 14, key="font_size")

st.sidebar.divider()


# ===== メインエリア =====

show_panel = st.toggle(T["show_panel"], value=True, key="show_panel")

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

    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):

                st.markdown(T["xrd_patterns"])
                for i, f in enumerate(active_xrd):
                    default_name = os.path.splitext(f.name)[0]
                    default_hex  = ALL_COLORS[i % len(ALL_COLORS)]

                    with st.expander(f"**{i+1}. {default_name}**", expanded=True):
                        order = st.number_input(
                            T["display_order"], value=i + 1, min_value=1, max_value=50,
                            key=f"ord_{i}",
                        )
                        visible = st.checkbox(T["show_cb"], value=True, key=f"vis_{i}")
                        label   = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"xrd_color_{i}", default_hex)

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
                                T["y_position"], value=0.0, step=100.0,
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
                        default_hex  = ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]

                        with st.expander(f"**CIF {i+1}. {file_stem}**", expanded=True):
                            order   = st.number_input(
                                T["display_order"], value=i + 1, min_value=1, max_value=50,
                                key=f"cord_{i}",
                            )
                            visible = st.checkbox(T["show_cb"], value=True, key=f"cvis_{i}")
                            label   = label_input(key=f"clbl_{i}", default=smart_label)
                            chosen_color = color_picker_popover(f"cif_color_{i}", default_hex)
                            if show_ref_lines:
                                ref_line = st.checkbox(T["show_ref_line_each"], value=True, key=f"cref_line_{i}")
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
                        default_hex = ALL_COLORS[(n_offset + i) % len(ALL_COLORS)]

                        with st.expander(f"**PDF {i+1}. {file_stem}**", expanded=True):
                            order   = st.number_input(
                                T["display_order"], value=len(active_cif) + i + 1,
                                min_value=1, max_value=50, key=f"pord_{i}",
                            )
                            visible = st.checkbox(T["show_cb"], value=True, key=f"pvis_{i}")
                            label   = label_input(key=f"plbl_{i}", default=smart_label)
                            chosen_color = color_picker_popover(f"pdf_color_{i}", default_hex)
                            if show_ref_lines:
                                ref_line = st.checkbox(T["show_ref_line_each"], value=True, key=f"pref_line_{i}")
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
            orders.append(i + 1)
            visibles.append(st.session_state.get(f"vis_{i}", True))
            labels.append(st.session_state.get(f"_val_lbl_{i}", os.path.splitext(f.name)[0]))
            colors_sel.append(st.session_state.get(key, ALL_COLORS[i % len(ALL_COLORS)]))
            if normalize:
                eoff_key = f"extra_offset_{i}"
                if eoff_key not in st.session_state:
                    st.session_state[eoff_key] = 0.0
            else:
                abs_offsets.append(st.session_state.get(f"yoff_{i}", 0.0))
        sort_idx = list(range(len(active_xrd)))

        n_xrd = len(active_xrd)
        if active_cif:
            for i, f in enumerate(active_cif):
                key = f"cif_color_{i}"
                fallback = extract_cif_label(f.read()) or os.path.splitext(f.name)[0]
                cif_orders.append(i + 1)
                cif_visibles.append(st.session_state.get(f"cvis_{i}", True))
                cif_labels.append(st.session_state.get(f"_val_clbl_{i}", fallback))
                cif_colors.append(st.session_state.get(key, ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]))
                cif_ref_lines.append(st.session_state.get(f"cref_line_{i}", True))
            cif_sort_idx = list(range(len(active_cif)))

        if active_pdf:
            n_offset = n_xrd + len(active_cif)
            for i, f in enumerate(active_pdf):
                key = f"pdf_color_{i}"
                fallback = extract_pdf_card_label(f.read()) or os.path.splitext(f.name)[0]
                pdf_orders.append(len(active_cif) + i + 1)
                pdf_visibles.append(st.session_state.get(f"pvis_{i}", True))
                pdf_labels.append(st.session_state.get(f"_val_plbl_{i}", fallback))
                pdf_colors.append(st.session_state.get(key, ALL_COLORS[(n_offset + i) % len(ALL_COLORS)]))
                pdf_ref_lines.append(st.session_state.get(f"pref_line_{i}", True))
            pdf_sort_idx = list(range(len(active_pdf)))

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
    def build_figure(show_legend=show_legend, show_cif_legend=show_cif_legend):
        from matplotlib.ticker import MultipleLocator

        refs = build_ref_list()
        has_ref = bool(refs)
        has_xrd = bool(active_xrd) and any(visibles)

        plt.rcParams.update({
            "font.family":      "Arial",
            "font.size":        font_size,
            "mathtext.fontset": "custom",
            "mathtext.it":      "Arial:italic",
            "mathtext.rm":      "Arial",
        })

        gs = None
        if has_ref:
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0, figure=fig)
            ax_main = fig.add_subplot(gs[0])
            ax_ref  = fig.add_subplot(gs[1], sharex=ax_main)
            ax_main.spines["bottom"].set_color("black")
            ax_ref.spines["top"].set_visible(False)
            ax_main.tick_params(axis="x", which="both", length=0)
        else:
            fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))
            ax_ref = None

        ax_main.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax_main.tick_params(which="major", axis="x", length=5, direction="in")
        if show_minor and minor_tick:
            ax_main.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax_main.tick_params(which="minor", axis="x", length=2.5, direction="in")

        cumulative_y = 0.0
        side_labels  = []

        if has_xrd:
            for i in sort_idx:
                if not visibles[i]:
                    continue
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
                else:
                    y_plot = y + abs_offsets[i]
                    side_labels.append((abs_offsets[i] + y_min + label_offset_y * (y_max - y_min), colors_sel[i], labels[i]))

                ax_main.plot(x, y_plot, color=colors_sel[i], linewidth=1.2, label=labels[i])

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
                    x_pos, y_c, txt,
                    color=col, fontsize=label_fontsize,
                    ha=ha, va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
                )

        ax_main.set_ylabel("Intensity (a.u.)")
        ax_main.set_yticks([])
        if show_legend:
            ax_main.legend(loc="upper right", frameon=False)

        xlabel = r"Diffraction angle, 2$\it{\theta}$ (deg.)"
        if ax_ref is None:
            ax_main.set_xlabel(xlabel)
        else:
            plt.setp(ax_main.get_xticklabels(), visible=False)

        if ax_ref is not None and refs:
            bar_height = 80.0
            row_height = 110.0
            row = 0
            for ref in refs:
                if ref["type"] == "cif":
                    x_ref, y_ref = calc_cif_pattern(
                        ref["file"].read(), two_theta_range=(x_min, x_max)
                    )
                else:
                    x_ref, y_ref = parse_pdf_card(
                        ref["file"].read(), two_theta_range=(x_min, x_max)
                    )
                if x_ref is None:
                    row += 1
                    continue
                baseline = row * row_height
                y_norm = y_ref / np.max(y_ref) * bar_height if np.max(y_ref) > 0 else y_ref
                if row > 0:
                    ax_ref.axhline(y=baseline, color="gray", linewidth=0.6,
                                   linestyle="-", alpha=0.5, zorder=0)
                ax_ref.vlines(x_ref, baseline, baseline + y_norm,
                              color=ref["color"], linewidth=1.0, label=ref["label"])
                if show_ref_lines and ref.get("show_line", True):
                    ax_main.vlines(x_ref, *ax_main.get_ylim(),
                                   color=ref["color"], linewidth=0.7,
                                   linestyle="--", alpha=0.4, zorder=0)
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    lha = "left" if cif_label_side == "左" else "right"
                    ax_ref.text(
                        lx, baseline + cif_label_offset_y * row_height,
                        ref["label"], fontsize=cif_label_fontsize,
                        color=ref["color"], va="center", ha=lha,
                    )
                row += 1
            ax_ref.axhline(y=row * row_height, color="gray", linewidth=0.6,
                           linestyle="-", alpha=0.5)
            ax_ref.set_xlabel(xlabel)
            ax_ref.set_yticks([])
            ax_ref.set_ylim(0, row * row_height)
            ax_ref.xaxis.set_major_locator(MultipleLocator(major_tick))
            ax_ref.tick_params(which="major", axis="x", length=5, direction="out")
            if show_minor and minor_tick:
                ax_ref.xaxis.set_minor_locator(MultipleLocator(minor_tick))
                ax_ref.tick_params(which="minor", axis="x", length=2.5, direction="out")

        ax_main.set_xlim(x_min, x_max)
        if gs is not None:
            fig.tight_layout(h_pad=0)
        else:
            fig.tight_layout()
        return fig

    # ===== インタラクティブ Plotly プレビュー =====
    def build_plotly_figure(show_legend=show_legend, show_cif_legend=show_cif_legend):
        refs = build_ref_list()
        has_ref = bool(refs)

        if has_ref:
            pfig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0,
            )
        else:
            pfig = make_subplots(rows=1, cols=1)

        cumulative_y = 0.0

        for i in sort_idx:
            if not visibles[i]:
                continue
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
                cumulative_y += global_offset * y_max
            else:
                y_plot  = y + abs_offsets[i]
                y_label = abs_offsets[i] + y_min + label_offset_y * (y_max - y_min)

            pfig.add_trace(go.Scatter(
                x=x, y=y_plot, name=mpl_to_plotly(labels[i]),
                line=dict(color=colors_sel[i], width=1.5),
                mode="lines", showlegend=show_legend,
            ), row=1, col=1)

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
                    textfont=dict(color=colors_sel[i], size=label_fontsize, family="Arial"),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=1, col=1)

        if has_ref and refs:
            bar_height = 80.0
            row_height = 110.0
            row = 0
            for ref in refs:
                if ref["type"] == "cif":
                    x_ref, y_ref = calc_cif_pattern(
                        ref["file"].read(), two_theta_range=(x_min, x_max)
                    )
                else:
                    x_ref, y_ref = parse_pdf_card(
                        ref["file"].read(), two_theta_range=(x_min, x_max)
                    )
                if x_ref is None:
                    row += 1
                    continue
                baseline = row * row_height
                y_norm = y_ref / np.max(y_ref) * bar_height if np.max(y_ref) > 0 else y_ref
                vx, vy = [], []
                for xi, yi in zip(x_ref, y_norm):
                    vx += [float(xi), float(xi), None]
                    vy += [float(baseline), float(baseline + yi), None]
                pfig.add_trace(go.Scatter(
                    x=vx, y=vy, name=mpl_to_plotly(ref["label"]),
                    line=dict(color=ref["color"], width=1.0),
                    mode="lines", showlegend=False,
                ), row=2, col=1)
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
                        text=[mpl_to_plotly(ref["label"])],
                        textposition=textpos_cif,
                        textfont=dict(color=ref["color"], size=cif_label_fontsize, family="Arial"),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=2, col=1)
                row += 1
            pfig.update_layout(yaxis2=dict(
                range=[0, row * row_height], autorange=False,
                showticklabels=False, showline=True, linecolor="black",
                showgrid=False, zeroline=False,
            ))

        xlabel = "Diffraction angle, 2<i>θ</i> (deg.)"
        minor_cfg = dict(ticks="inside", dtick=minor_tick, showgrid=False) if (show_minor and minor_tick) else {}
        pfig.update_xaxes(
            range=[x_min, x_max], dtick=major_tick, ticks="inside",
            showline=True, linecolor="black", mirror=False,
            showgrid=False, zeroline=False,
            tickfont=dict(color="black", size=font_size),
            title_font=dict(color="black", size=font_size),
            minor=minor_cfg,
        )
        if has_ref:
            pfig.update_xaxes(ticks="outside", row=2, col=1)
            if show_minor and minor_tick:
                pfig.update_xaxes(minor=dict(ticks="outside", dtick=minor_tick), row=2, col=1)
        pfig.update_yaxes(
            showticklabels=False, showline=True, linecolor="black",
            showgrid=False, zeroline=False,
            title_font=dict(color="black", size=font_size),
        )

        if has_ref:
            pfig.update_xaxes(title_text=xlabel, row=2, col=1)
            pfig.update_xaxes(showticklabels=False, row=1, col=1)
            pfig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
        else:
            pfig.update_xaxes(title_text=xlabel, row=1, col=1)
            pfig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)

        pfig.update_layout(
            dragmode="zoom",
            font=dict(family="Arial", size=font_size, color="black"),
            height=int(fig_height * 85),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=show_legend or show_cif_legend,
            legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top",
                        font=dict(color="black")),
            margin=dict(l=60, r=60, t=30, b=60),
        )
        return pfig

    # ===== 描画 & ダウンロード =====
    with col_graph:
        pfig = build_plotly_figure()
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
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
        buf_png.seek(0)
        png_bytes = buf_png.getvalue()
        buf_tiff = io.BytesIO()
        fig.savefig(buf_tiff, format="tiff", dpi=dpi_export, bbox_inches="tight")
        buf_tiff.seek(0)
        tiff_bytes = buf_tiff.getvalue()
        plt.close(fig)

        st.sidebar.download_button(
            T["save_tiff"].format(dpi=dpi_export),
            data=tiff_bytes, file_name="xrd_result.tiff", mime="image/tiff",
            key=f"dl_tiff_{_sl}_{_scl}_{dpi_export}",
            use_container_width=True,
        )
        st.sidebar.download_button(
            T["save_png"],
            data=png_bytes, file_name="xrd_result.png", mime="image/png",
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

session_upload = st.sidebar.file_uploader(
    T["load_session"], type=["json"], key="_session_upload"
)
if session_upload is not None:
    restore_session(session_upload.read())
if active_xrd:
    st.sidebar.download_button(
        T["save_session"],
        data=build_session_json(len(active_xrd), len(active_cif), len(active_pdf)),
        file_name="xrd_session.json",
        mime="application/json",
        use_container_width=True,
    )
