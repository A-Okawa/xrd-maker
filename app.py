import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter, find_peaks
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import tempfile
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

st.set_page_config(page_title="XRD Analyzer", page_icon="🔬", layout="wide")
st.title("XRD Analyzer")
st.caption("複数パターン重ね合わせ・CIFリファレンス・論文用TIFF出力")

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

# ===== ラベル入力（書式ボタン付き） =====

def label_input(key: str, default: str = "") -> str:
    """テキスト入力＋書式テンプレート挿入ボタン付きラベル入力"""
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    # バージョン付きキー → ボタン押下ごとに新しいキーで再初期化され value= が確実に反映される
    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    inp = st.text_input("ラベル", value=st.session_state[val_key], key=inp_key)
    st.session_state[val_key] = inp  # ユーザーの入力を随時同期

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


def apply_smooth(y, window):
    if window is None or len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=3)


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


# ===== サイドバー =====
st.sidebar.header("📂 XRDデータ")
xrd_files = st.sidebar.file_uploader(
    "XRDデータ (.xy / .txt / .csv)",
    type=["xy", "txt", "csv"], accept_multiple_files=True,
)
st.sidebar.header("📂 CIFリファレンス")
cif_files = st.sidebar.file_uploader(
    "CIFファイル (.cif)", type=["cif"], accept_multiple_files=True,
)

st.sidebar.header("⚙️ グラフ設定")
x_min, x_max   = st.sidebar.slider("2θ 範囲 (°)", 5.0, 90.0, (10.0, 80.0), step=0.5)
do_smooth      = st.sidebar.checkbox("スムージングを適用", value=True)
smoothing      = st.sidebar.slider("窓幅（奇数）", 3, 51, 11, step=2) if do_smooth else None
normalize      = st.sidebar.checkbox("強度を正規化（最大=1）", value=True)
show_legend    = st.sidebar.checkbox("メイン凡例を表示", value=True)
show_cif_legend= st.sidebar.checkbox("ICDD ラベルをグラフ内に表示", value=True)
show_peaks     = st.sidebar.checkbox("ピーク位置を表示", value=False)
peak_prom      = st.sidebar.slider("ピーク感度", 0.01, 0.5, 0.1, step=0.01) if show_peaks else 0.1
global_offset  = st.sidebar.slider("オフセット（倍率）", 0.0, 3.0, 1.0, step=0.05) if normalize else None

st.sidebar.subheader("目盛り設定")
major_tick  = st.sidebar.number_input("主目盛り間隔 (°)", min_value=1.0, max_value=30.0,
                                       value=10.0, step=1.0)
show_minor  = st.sidebar.checkbox("副目盛りを表示", value=True)
minor_tick  = st.sidebar.number_input("副目盛り間隔 (°)", min_value=0.5, max_value=10.0,
                                       value=2.0, step=0.5) if show_minor else None

st.sidebar.subheader("ICDD ラベル設定")
if show_cif_legend:
    cif_label_side     = st.sidebar.radio("ICDDラベル位置", ["左", "右"], horizontal=True)
    cif_label_fontsize = st.sidebar.slider("ICDDラベル文字サイズ", 5, 20, 9)
    cif_label_offset_x = st.sidebar.slider("ICDD横オフセット (°)", 0.0, 10.0, 0.5, step=0.1,
                                            help="端からの距離（°）")
    cif_label_offset_y = st.sidebar.slider("ICDD縦オフセット（行高さ比）", 0.0, 1.0, 0.5, step=0.05,
                                            help="0=行の下端、1.0=行の上端")
else:
    cif_label_side, cif_label_fontsize = "左", 9
    cif_label_offset_x, cif_label_offset_y = 0.5, 0.5

st.sidebar.subheader("サンプルラベル（グラフ内）")
show_side_labels = st.sidebar.checkbox("グラフ内にラベルを表示", value=False)
if show_side_labels:
    label_side      = st.sidebar.radio("ラベル位置", ["左", "右"], horizontal=True)
    label_fontsize  = st.sidebar.slider("ラベル文字サイズ", 5, 24, 11)
    label_offset_x  = st.sidebar.slider("横オフセット（°）", -5.0, 5.0, 0.5, step=0.1,
                                         help="正=グラフ内側、負=グラフ外側")
    label_offset_y  = st.sidebar.slider("縦オフセット（パターン高さ比）", -0.3, 1.0, 0.05, step=0.01,
                                         help="0=ベースライン、1.0=ピーク付近")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 11, 0.5, 0.05

st.sidebar.header("📐 図サイズ・出力")
fig_width  = st.sidebar.slider("図の幅 (inch)", 6.0, 20.0, 10.0, step=0.5)
fig_height = st.sidebar.slider("図の高さ (inch)", 4.0, 20.0, 8.0, step=0.5)
dpi_export = st.sidebar.selectbox("出力 DPI", [300, 600], index=0)
font_size  = st.sidebar.slider("フォントサイズ", 8, 20, 14)


# ===== メインエリア: グラフ ＋ 右パネル =====

# 右パネルの表示/非表示トグル
show_panel = st.toggle("⚙️ パターン設定パネルを表示", value=True)

if xrd_files:
    # カラム比率：パネルあり = 7:3、なし = 全幅
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    # ===== 右パネル：パターン設定 =====
    orders, visibles, labels, colors_sel, abs_offsets = [], [], [], [], []
    sort_idx = []
    cif_orders, cif_visibles, cif_labels, cif_colors = [], [], [], []
    cif_sort_idx = []

    if show_panel and col_settings is not None:
        with col_settings:

            with st.container(height=700):

                # --- XRDパターン設定 ---
                st.markdown("#### XRD パターン")
                for i, f in enumerate(xrd_files):
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

                sort_idx = sorted(range(len(xrd_files)), key=lambda i: orders[i])

                # --- CIFリファレンス設定 ---
                if cif_files:
                    st.markdown("#### CIF リファレンス")
                    n_xrd = len(xrd_files)
                    for i, f in enumerate(cif_files):
                        default_name = os.path.splitext(f.name)[0]
                        default_hex  = ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]

                        with st.expander(f"**Ref {i+1}. {default_name}**", expanded=True):
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

                    cif_sort_idx = sorted(range(len(cif_files)), key=lambda i: cif_orders[i])

    else:
        # パネル非表示時：デフォルト値で設定を構築
        for i, f in enumerate(xrd_files):
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
        sort_idx = list(range(len(xrd_files)))

        if cif_files:
            n_xrd = len(xrd_files)
            for i, f in enumerate(cif_files):
                key = f"cif_color_{i}"
                cif_orders.append(i + 1)
                cif_visibles.append(st.session_state.get(f"cvis_{i}", True))
                cif_labels.append(st.session_state.get(f"_val_clbl_{i}", os.path.splitext(f.name)[0]))
                cif_colors.append(st.session_state.get(key, ALL_COLORS[(n_xrd + i) % len(ALL_COLORS)]))
            cif_sort_idx = list(range(len(cif_files)))

    # ===== 図の生成 =====
    def build_figure():
        from matplotlib.ticker import MultipleLocator

        has_cif = bool(cif_files) and bool(cif_visibles) and any(cif_visibles)
        has_xrd = bool(xrd_files) and any(visibles)

        plt.rcParams.update({
            "font.family":      "Arial",
            "font.size":        font_size,
            "mathtext.fontset": "custom",
            "mathtext.it":      "Arial:italic",
            "mathtext.rm":      "Arial",
        })

        gs = None
        if has_cif:
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0, figure=fig)
            ax_main = fig.add_subplot(gs[0])
            ax_ref  = fig.add_subplot(gs[1], sharex=ax_main)
            ax_main.spines["bottom"].set_color("black")  # 黒いx軸線を表示
            ax_ref.spines["top"].set_visible(False)
            ax_main.tick_params(axis="x", which="both", length=0)
        else:
            fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))
            ax_ref = None

        # X軸 目盛り設定（sharex のため ax_main に設定すれば ax_ref にも反映）
        ax_main.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax_main.tick_params(which="major", axis="x", length=5, direction="in")
        if show_minor and minor_tick:
            ax_main.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax_main.tick_params(which="minor", axis="x", length=2.5, direction="in")

        cumulative_y = 0.0
        side_labels  = []   # (y_center, color, text) for side label rendering

        if has_xrd:
            for i in sort_idx:
                if not visibles[i]:
                    continue
                raw = xrd_files[i].read()
                xrd_files[i].seek(0)
                data = read_xrd_data(raw, xrd_files[i].name)
                if data is None:
                    continue
                x, y = data[:, 0], data[:, 1]
                mask = (x >= x_min) & (x <= x_max)
                x, y = x[mask], y[mask]
                if len(y) == 0:
                    continue
                y = apply_smooth(y, smoothing)
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
                    # ベースライン = abs_offsets[i]、縦オフセット = label_offset_y * y_max
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

        # サンプルラベルをグラフ枠付近に描画
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

        if ax_ref is not None and cif_files:
            bar_height = 80.0
            row_height = 110.0
            row = 0
            for i in cif_sort_idx:
                if not cif_visibles[i]:
                    continue
                x_ref, y_ref = calc_cif_pattern(
                    cif_files[i].read(), two_theta_range=(x_min, x_max)
                )
                cif_files[i].seek(0)
                if x_ref is None:
                    row += 1
                    continue
                baseline = row * row_height
                y_norm = y_ref / np.max(y_ref) * bar_height if np.max(y_ref) > 0 else y_ref
                if row > 0:  # row=0 の下線はフレーム枠線と一致させるため省略
                    ax_ref.axhline(y=baseline, color="gray", linewidth=0.6,
                                   linestyle="-", alpha=0.5, zorder=0)
                ax_ref.vlines(x_ref, baseline, baseline + y_norm,
                              color=cif_colors[i], linewidth=1.0, label=cif_labels[i])
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    lha = "left" if cif_label_side == "左" else "right"
                    ax_ref.text(
                        lx, baseline + cif_label_offset_y * row_height,
                        cif_labels[i], fontsize=cif_label_fontsize,
                        color=cif_colors[i], va="center", ha=lha,
                    )
                row += 1
            ax_ref.axhline(y=row * row_height, color="gray", linewidth=0.6,
                           linestyle="-", alpha=0.5)
            ax_ref.set_xlabel(xlabel)
            ax_ref.set_yticks([])
            ax_ref.set_ylim(0, row * row_height)
            # ax_ref にも目盛り設定（外向き）
            ax_ref.xaxis.set_major_locator(MultipleLocator(major_tick))
            ax_ref.tick_params(which="major", axis="x", length=5, direction="out")
            if show_minor and minor_tick:
                ax_ref.xaxis.set_minor_locator(MultipleLocator(minor_tick))
                ax_ref.tick_params(which="minor", axis="x", length=2.5, direction="out")

        ax_main.set_xlim(x_min, x_max)
        if gs is not None:
            fig.tight_layout(h_pad=0)  # h_pad=0 でサブプロット間余白をゼロに保つ
        else:
            fig.tight_layout()
        return fig

    # ===== インタラクティブ Plotly プレビュー =====
    def build_plotly_figure():
        has_cif = bool(cif_files) and bool(cif_visibles) and any(cif_visibles)

        if has_cif:
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
            raw = xrd_files[i].read()
            xrd_files[i].seek(0)
            data = read_xrd_data(raw, xrd_files[i].name)
            if data is None:
                continue
            x, y = data[:, 0], data[:, 1]
            mask = (x >= x_min) & (x <= x_max)
            x, y = x[mask], y[mask]
            if len(y) == 0:
                continue
            y = apply_smooth(y, smoothing)
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

        if has_cif and cif_files:
            bar_height = 80.0
            row_height = 110.0
            row = 0
            for i in cif_sort_idx:
                if not cif_visibles[i]:
                    continue
                x_ref, y_ref = calc_cif_pattern(
                    cif_files[i].read(), two_theta_range=(x_min, x_max)
                )
                cif_files[i].seek(0)
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
                    x=vx, y=vy, name=cif_labels[i],
                    line=dict(color=cif_colors[i], width=1.0),
                    mode="lines", showlegend=False,
                ), row=2, col=1)
                # ICDD ラベル：テキストトレースとして描画（annotation を使わない）
                if show_cif_legend:
                    lx = (x_min + cif_label_offset_x) if cif_label_side == "左" \
                         else (x_max - cif_label_offset_x)
                    textpos_cif = "middle right" if cif_label_side == "左" else "middle left"
                    pfig.add_trace(go.Scatter(
                        x=[lx],
                        y=[baseline + cif_label_offset_y * row_height],
                        mode="text",
                        text=[cif_labels[i]],
                        textposition=textpos_cif,
                        textfont=dict(color=cif_colors[i], size=cif_label_fontsize, family="Arial"),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=2, col=1)
                row += 1
            # ICDD パネルの y 範囲を固定（autorange=False で Plotly の自動余白を無効化）
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
        # ICDD パネルの目盛りは外向き
        if has_cif:
            pfig.update_xaxes(ticks="outside", row=2, col=1)
            if show_minor and minor_tick:
                pfig.update_xaxes(minor=dict(ticks="outside", dtick=minor_tick), row=2, col=1)
        pfig.update_yaxes(
            showticklabels=False, showline=True, linecolor="black",
            showgrid=False, zeroline=False,
            title_font=dict(color="black", size=font_size),
        )

        if has_cif:
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
