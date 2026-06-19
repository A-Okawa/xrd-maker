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

st.set_page_config(page_title="XRD Maker", page_icon="🔬", layout="wide")

# ===== パスワード認証 =====
def check_password():
    if st.session_state.get("authenticated"):
        return True
    pwd = st.secrets.get("password", "")
    st.title("🔬 XRD Maker")
    entered = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False

if not check_password():
    st.stop()

st.title("XRD Maker")
st.caption("複数パターン重ね合わせ・CIF/PDFカード リファレンス・論文用TIFF出力")

# ===== 色ファミリー =====
COLOR_FAMILIES = OrderedDict([
    ("赤系", [
        ("#7b0000", "Deep Maroon"), ("#b71c1c", "Dark Red"),
        ("#d32f2f", "Red"),         ("#e53935", "Bright Red"),
        ("#ef9a9a", "Light Red"),   ("#f48fb1", "Light Pink"),
        ("#f8bbd0", "Pale Pink"),
    ]),
    ("橙・黄系", [
        ("#bf360c", "Deep Orange"), ("#e64a19", "Dark Orange"),
        ("#ff7043", "Orange"),      ("#ffa726", "Amber"),
        ("#ffca28", "Yellow"),      ("#fff176", "Light Yellow"),
    ]),
    ("緑系", [
        ("#1b5e20", "Deep Green"),  ("#2e7d32", "Dark Green"),
        ("#43a047", "Green"),       ("#66bb6a", "Medium Green"),
        ("#00897b", "Teal"),        ("#26c6da", "Cyan"),
        ("#b2dfdb", "Pale Teal"),
    ]),
    ("青系", [
        ("#0d47a1", "Deep Blue"),   ("#1565c0", "Dark Blue"),
        ("#1976d2", "Blue"),        ("#1e88e5", "Medium Blue"),
        ("#42a5f5", "Sky Blue"),    ("#90caf9", "Light Blue"),
        ("#bbdefb", "Pale Blue"),
    ]),
    ("紫・ピンク系", [
        ("#4a148c", "Deep Purple"), ("#6a1b9a", "Dark Purple"),
        ("#8e24aa", "Purple"),      ("#ab47bc", "Medium Purple"),
        ("#ba68c8", "Violet"),      ("#ce93d8", "Lavender"),
        ("#f06292", "Pink"),
    ]),
    ("茶・ベージュ系", [
        ("#3e2723", "Deep Brown"),  ("#5d4037", "Brown"),
        ("#8d6e63", "Medium Brown"),("#a1887f", "Warm Beige"),
        ("#d7ccc8", "Light Beige"),
    ]),
    ("黒・グレー系", [
        ("#000000", "Black"),       ("#212121", "Near Black"),
        ("#424242", "Very Dark Gray"),("#616161", "Dark Gray"),
        ("#9e9e9e", "Medium Gray"), ("#bdbdbd", "Gray"),
        ("#e0e0e0", "Light Gray"),
    ]),
])
ALL_COLORS = [h for fam in COLOR_FAMILIES.values() for h, _ in fam]


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


# ===== ラベル入力（書式ボタン付き） =====
def label_input(key: str, default: str = "") -> str:
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    inp = st.text_input("ラベル", value=st.session_state[val_key], key=inp_key)
    st.session_state[val_key] = inp

    c1, c2, c3 = st.columns(3)
    if c1.button("＋italic", key=f"_bi_{key}", use_container_width=True,
                 help="末尾に $\\it{TEXT}$ を追加。TEXTを書き換えて使用"):
        st.session_state[val_key] = inp + r"$\it{TEXT}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button("＋下付き", key=f"_bs_{key}", use_container_width=True,
                 help="末尾に $_{N}$ を追加。Nを書き換えて使用"):
        st.session_state[val_key] = inp + r"$_{N}$"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button("＋上付き", key=f"_bp_{key}", use_container_width=True,
                 help="末尾に $^{N}$ を追加。Nを書き換えて使用"):
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
        st.warning(f"CIF エラー: {e}")
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
                        # 左側: cols 0-4 (No, 2θ, d, intensity, hkl)
                        for offset in [0, 5]:
                            if len(row) < offset + 4:
                                continue
                            try:
                                int(str(row[offset]).strip())  # No.が整数かチェック
                                two_theta = float(str(row[offset + 1]).strip())
                                intensity = float(str(row[offset + 3]).strip())
                                if two_theta_range[0] <= two_theta <= two_theta_range[1]:
                                    two_thetas.append(two_theta)
                                    intensities.append(intensity)
                            except (ValueError, TypeError, AttributeError):
                                continue
    except Exception as e:
        st.warning(f"PDFカード解析エラー: {e}")
        return None, None

    if not two_thetas:
        return None, None

    pairs = sorted(zip(two_thetas, intensities))
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return x, y


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

    c_swatch, c_btn = st.columns([1, 3])
    c_swatch.markdown(
        f'<div style="background:{current};height:30px;border-radius:5px;'
        f'border:1px solid #ccc;margin-top:4px"></div>',
        unsafe_allow_html=True,
    )
    with c_btn.popover("🎨 色を変更", use_container_width=True):
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
    "normalize", "show_legend", "show_cif_legend", "show_peaks",
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
        for pat in ["cvis_{}", "cord_{}", "_val_clbl_{}", "cif_color_{}"]:
            k = pat.format(i)
            if k in st.session_state:
                settings[k] = st.session_state[k]
    for i in range(n_pdf):
        for pat in ["pvis_{}", "pord_{}", "_val_plbl_{}", "pdf_color_{}"]:
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


# ===== サイドバー =====

st.sidebar.header("💾 セッション")
session_upload = st.sidebar.file_uploader(
    "セッションを読み込む (.json)", type=["json"], key="_session_upload"
)
if session_upload is not None:
    restore_session(session_upload.read())

st.sidebar.header("📂 XRDデータ")
xrd_files = st.sidebar.file_uploader(
    "XRDデータ (.xy / .txt / .csv)",
    type=["xy", "txt", "csv"], accept_multiple_files=True,
)
st.sidebar.header("📂 リファレンス")
cif_files = st.sidebar.file_uploader(
    "CIFファイル (.cif)", type=["cif"], accept_multiple_files=True,
)
pdf_ref_files = st.sidebar.file_uploader(
    "PDFカード (.pdf)", type=["pdf"], accept_multiple_files=True,
)

# ファイルの確定（新規アップロード優先 → セッション復元）
if xrd_files:
    st.session_state["_xrd_bytes"] = {f.name: f.read() for f in xrd_files}
    for f in xrd_files:
        f.seek(0)
if cif_files:
    st.session_state["_cif_bytes"] = {f.name: f.read() for f in cif_files}
    for f in cif_files:
        f.seek(0)
if pdf_ref_files:
    st.session_state["_pdf_bytes"] = {f.name: f.read() for f in pdf_ref_files}
    for f in pdf_ref_files:
        f.seek(0)

active_xrd = [RestoredFile(n, d) for n, d in st.session_state.get("_xrd_bytes", {}).items()]
active_cif = [RestoredFile(n, d) for n, d in st.session_state.get("_cif_bytes", {}).items()]
active_pdf = [RestoredFile(n, d) for n, d in st.session_state.get("_pdf_bytes", {}).items()]

if active_xrd and not xrd_files:
    st.sidebar.caption("復元済み XRD: " + ", ".join(f.name for f in active_xrd))
if active_cif and not cif_files:
    st.sidebar.caption("復元済み CIF: " + ", ".join(f.name for f in active_cif))
if active_pdf and not pdf_ref_files:
    st.sidebar.caption("復元済み PDF: " + ", ".join(f.name for f in active_pdf))

st.sidebar.header("⚙️ グラフ設定")
xrange         = st.sidebar.slider("2θ 範囲 (°)", 5.0, 90.0, (10.0, 80.0), step=0.5, key="xrange")
x_min, x_max   = xrange
normalize      = st.sidebar.checkbox("強度を正規化（最大=1）", value=False, key="normalize")
show_legend    = st.sidebar.checkbox("メイン凡例を表示", value=True, key="show_legend")
show_cif_legend= st.sidebar.checkbox("ICDD ラベルをグラフ内に表示", value=True, key="show_cif_legend")
show_peaks     = st.sidebar.checkbox("ピーク位置を表示", value=False, key="show_peaks")
peak_prom      = st.sidebar.slider("ピーク感度", 0.01, 0.5, 0.1, step=0.01, key="peak_prom") if show_peaks else 0.1
global_offset  = st.sidebar.slider("オフセット（倍率）", 0.0, 3.0, 1.0, step=0.05, key="global_offset") if normalize else None

st.sidebar.subheader("目盛り設定")
major_tick  = st.sidebar.number_input("主目盛り間隔 (°)", min_value=1.0, max_value=30.0,
                                       value=10.0, step=1.0, key="major_tick")
show_minor  = st.sidebar.checkbox("副目盛りを表示", value=True, key="show_minor")
minor_tick  = st.sidebar.number_input("副目盛り間隔 (°)", min_value=0.5, max_value=10.0,
                                       value=2.0, step=0.5, key="minor_tick") if show_minor else None

st.sidebar.subheader("ICDD ラベル設定")
if show_cif_legend:
    cif_label_side     = st.sidebar.radio("ICDDラベル位置", ["左", "右"], horizontal=True, key="cif_label_side")
    cif_label_fontsize = st.sidebar.slider("ICDDラベル文字サイズ", 5, 20, 9, key="cif_label_fontsize")
    cif_label_offset_x = st.sidebar.slider("ICDD横オフセット (°)", 0.0, 10.0, 0.5, step=0.1,
                                            help="端からの距離（°）", key="cif_label_offset_x")
    cif_label_offset_y = st.sidebar.slider("ICDD縦オフセット（行高さ比）", 0.0, 1.0, 0.5, step=0.05,
                                            help="0=行の下端、1.0=行の上端", key="cif_label_offset_y")
else:
    cif_label_side, cif_label_fontsize = "左", 9
    cif_label_offset_x, cif_label_offset_y = 0.5, 0.5

st.sidebar.subheader("サンプルラベル（グラフ内）")
show_side_labels = st.sidebar.checkbox("グラフ内にラベルを表示", value=False, key="show_side_labels")
if show_side_labels:
    label_side      = st.sidebar.radio("ラベル位置", ["左", "右"], horizontal=True, key="label_side")
    label_fontsize  = st.sidebar.slider("ラベル文字サイズ", 5, 24, 11, key="label_fontsize")
    label_offset_x  = st.sidebar.slider("横オフセット（°）", -5.0, 5.0, 0.5, step=0.1,
                                         help="正=グラフ内側、負=グラフ外側", key="label_offset_x")
    label_offset_y  = st.sidebar.slider("縦オフセット（パターン高さ比）", -0.3, 1.0, 0.05, step=0.01,
                                         help="0=ベースライン、1.0=ピーク付近", key="label_offset_y")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 11, 0.5, 0.05

st.sidebar.header("📐 図サイズ・出力")
fig_width  = st.sidebar.slider("図の幅 (inch)", 6.0, 20.0, 10.0, step=0.5, key="fig_width")
fig_height = st.sidebar.slider("図の高さ (inch)", 4.0, 20.0, 8.0, step=0.5, key="fig_height")
dpi_export = st.sidebar.selectbox("出力 DPI", [300, 600], index=0, key="dpi_export")
font_size  = st.sidebar.slider("フォントサイズ", 8, 20, 14, key="font_size")

st.sidebar.divider()
if active_xrd:
    st.sidebar.download_button(
        "💾 セッションを保存 (.json)",
        data=build_session_json(len(active_xrd), len(active_cif), len(active_pdf)),
        file_name="xrd_session.json",
        mime="application/json",
        use_container_width=True,
    )


# ===== メインエリア =====

show_panel = st.toggle("⚙️ パターン設定パネルを表示", value=True, key="show_panel")

if active_xrd:
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    orders, visibles, labels, colors_sel, abs_offsets = [], [], [], [], []
    sort_idx = []
    cif_orders, cif_visibles, cif_labels, cif_colors = [], [], [], []
    cif_sort_idx = []
    pdf_orders, pdf_visibles, pdf_labels, pdf_colors = [], [], [], []
    pdf_sort_idx = []

    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):

                st.markdown("#### XRD パターン")
                for i, f in enumerate(active_xrd):
                    default_name = os.path.splitext(f.name)[0]
                    default_hex  = ALL_COLORS[i % len(ALL_COLORS)]

                    with st.expander(f"**{i+1}. {default_name}**", expanded=True):
                        order = st.number_input(
                            "表示順", value=i + 1, min_value=1, max_value=50,
                            key=f"ord_{i}",
                        )
                        visible = st.checkbox("表示する", value=True, key=f"vis_{i}")
                        label   = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"xrd_color_{i}", default_hex)

                        if normalize:
                            eoff_key = f"extra_offset_{i}"
                            if eoff_key not in st.session_state:
                                st.session_state[eoff_key] = 0.0
                            st.slider(
                                "オフセット調整",
                                min_value=-5.0, max_value=15.0,
                                step=0.05,
                                key=eoff_key,
                            )
                        else:
                            yoff = st.number_input(
                                "Y位置（絶対値）", value=0.0, step=100.0,
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
                    st.markdown("#### リファレンス")

                if active_cif:
                    st.markdown("**CIF**")
                    for i, f in enumerate(active_cif):
                        default_name = os.path.splitext(f.name)[0]
                        default_hex  = ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]

                        with st.expander(f"**CIF {i+1}. {default_name}**", expanded=True):
                            order   = st.number_input(
                                "表示順", value=i + 1, min_value=1, max_value=50,
                                key=f"cord_{i}",
                            )
                            visible = st.checkbox("表示する", value=True, key=f"cvis_{i}")
                            label   = label_input(key=f"clbl_{i}", default=default_name)
                            chosen_color = color_picker_popover(f"cif_color_{i}", default_hex)

                        cif_orders.append(order)
                        cif_visibles.append(visible)
                        cif_labels.append(label)
                        cif_colors.append(chosen_color)

                    cif_sort_idx = sorted(range(len(active_cif)), key=lambda i: cif_orders[i])

                if active_pdf:
                    st.markdown("**PDFカード**")
                    n_offset = n_xrd + len(active_cif)
                    for i, f in enumerate(active_pdf):
                        default_name = os.path.splitext(f.name)[0]
                        default_hex  = ALL_COLORS[(n_offset + i) % len(ALL_COLORS)]

                        with st.expander(f"**PDF {i+1}. {default_name}**", expanded=True):
                            order   = st.number_input(
                                "表示順", value=len(active_cif) + i + 1,
                                min_value=1, max_value=50, key=f"pord_{i}",
                            )
                            visible = st.checkbox("表示する", value=True, key=f"pvis_{i}")
                            label   = label_input(key=f"plbl_{i}", default=default_name)
                            chosen_color = color_picker_popover(f"pdf_color_{i}", default_hex)

                        pdf_orders.append(order)
                        pdf_visibles.append(visible)
                        pdf_labels.append(label)
                        pdf_colors.append(chosen_color)

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
                cif_orders.append(i + 1)
                cif_visibles.append(st.session_state.get(f"cvis_{i}", True))
                cif_labels.append(st.session_state.get(f"_val_clbl_{i}", os.path.splitext(f.name)[0]))
                cif_colors.append(st.session_state.get(key, ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]))
            cif_sort_idx = list(range(len(active_cif)))

        if active_pdf:
            n_offset = n_xrd + len(active_cif)
            for i, f in enumerate(active_pdf):
                key = f"pdf_color_{i}"
                pdf_orders.append(len(active_cif) + i + 1)
                pdf_visibles.append(st.session_state.get(f"pvis_{i}", True))
                pdf_labels.append(st.session_state.get(f"_val_plbl_{i}", os.path.splitext(f.name)[0]))
                pdf_colors.append(st.session_state.get(key, ALL_COLORS[(n_offset + i) % len(ALL_COLORS)]))
            pdf_sort_idx = list(range(len(active_pdf)))

    # ===== 統合リファレンスリストの構築（CIF + PDF、表示順でソート） =====
    def build_ref_list():
        """CIFとPDFを統合し、表示順でソートしたリファレンスリストを返す。"""
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
            })
        refs.sort(key=lambda r: r["order"])
        return refs

    # ===== 図の生成 =====
    def build_figure():
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

                y_max = max(np.max(y), 1e-9)

                if normalize:
                    y = y / y_max
                    y_max = 1.0
                    extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))
                    baseline  = cumulative_y + extra_off
                    y_plot    = y + baseline
                    side_labels.append((baseline + label_offset_y * y_max, colors_sel[i], labels[i]))
                    cumulative_y += global_offset * y_max
                else:
                    y_plot = y + abs_offsets[i]
                    side_labels.append((abs_offsets[i] + label_offset_y * y_max, colors_sel[i], labels[i]))

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
    def build_plotly_figure():
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
            y_max = max(np.max(y), 1e-9)

            if normalize:
                y = y / y_max
                y_max = 1.0
                extra_off = float(st.session_state.get(f"extra_offset_{i}", 0.0))
                y_plot  = y + cumulative_y + extra_off
                y_label = cumulative_y + extra_off + label_offset_y * y_max
                cumulative_y += global_offset * y_max
            else:
                y_plot  = y + abs_offsets[i]
                y_label = abs_offsets[i] + label_offset_y * y_max

            pfig.add_trace(go.Scatter(
                x=x, y=y_plot, name=labels[i],
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
                    text=[labels[i]],
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
                    x=vx, y=vy, name=ref["label"],
                    line=dict(color=ref["color"], width=1.0),
                    mode="lines", showlegend=False,
                ), row=2, col=1)
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    textpos_cif = "middle right" if cif_label_side == "左" else "middle left"
                    pfig.add_trace(go.Scatter(
                        x=[lx],
                        y=[baseline + cif_label_offset_y * row_height],
                        mode="text",
                        text=[ref["label"]],
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
        st.caption("ドラッグで範囲ズーム ／ ダブルクリックでリセット")
        st.plotly_chart(pfig, use_container_width=True, config={
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawrect"],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

        fig = build_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="tiff", dpi=dpi_export, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            f"📥 TIFF として保存 ({dpi_export} DPI)",
            data=buf, file_name="xrd_result.tiff", mime="image/tiff",
        )
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
        buf_png.seek(0)
        st.download_button(
            "📥 PNG として保存（確認用）",
            data=buf_png, file_name="xrd_result.png", mime="image/png",
        )
        plt.close(fig)

else:
    st.info("サイドバーから XRD データファイル（.xy / .txt / .csv）をアップロードしてください。")
    if not PYMATGEN_AVAILABLE:
        st.warning("pymatgen が未インストールのため CIF リファレンス機能は無効です。")
    if not PDFPLUMBER_AVAILABLE:
        st.warning("pdfplumber が未インストールのため PDFカード リファレンス機能は無効です。")
