"""Optional XRD figure annotations using already prepared plotted curves.

These helpers never normalize intensities or add pattern offsets. Matching is a
positional comparison with measured local maxima, not a phase-identification
algorithm. Different reference phases may match the same unresolved maximum.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import html
import math
import re
from typing import Mapping, Sequence

import numpy as np


MARKER_SHAPES = ("circle", "triangle", "square", "diamond")
_MPL_MARKERS = {"circle": "o", "triangle": "^", "square": "s", "diamond": "D"}
_TAG = re.compile(r"</?(?:i|b|sub|sup|br)\s*/?>", re.I)
_PLOTLY_MARKERS = {"circle": "circle", "triangle": "triangle-up", "square": "square", "diamond": "diamond"}


@dataclass(frozen=True)
class PreparedCurve:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    color: str = "#000000"
    linewidth: float = 1.2


@dataclass(frozen=True)
class ReferencePeaks:
    name: str
    x: Sequence[float]
    intensity: Sequence[float] | None = None
    hkl: Sequence[str | None] | None = None
    color: str = "#666666"
    marker: str = "circle"


@dataclass(frozen=True)
class PeakMatch:
    curve_index: int
    curve_name: str
    phase_index: int
    phase_name: str
    experimental_index: int
    reference_peak_index: int
    x: float
    y: float
    reference_x: float
    delta_deg: float
    hkl: str | None
    color: str
    marker: str
    source: str = "detected"


def _hover_text(name) -> str:
    """Names may carry Plotly's <i>/<sub> markup for the legend; hover text shows them plainly."""
    return html.escape(html.unescape(_TAG.sub("", str(name))))


def _curve_arrays(curve: PreparedCurve):
    x, y = np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Prepared curves need equal-length one-dimensional x/y arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Prepared curves must contain finite x/y values.")
    order = np.argsort(x, kind="stable")
    return x[order], y[order], order


def _reference_arrays(reference: ReferencePeaks):
    x = np.asarray(reference.x, dtype=float)
    intensity = np.ones(len(x), dtype=float) if reference.intensity is None else np.asarray(reference.intensity, dtype=float)
    if x.ndim != 1 or intensity.ndim != 1 or len(x) != len(intensity):
        raise ValueError("Reference positions and intensities must have equal lengths.")
    if not np.isfinite(x).all() or not np.isfinite(intensity).all():
        raise ValueError("Reference peaks must contain finite values.")
    if reference.hkl is not None and len(reference.hkl) != len(x):
        raise ValueError("Reference hkl labels must align with reference positions.")
    if reference.marker not in MARKER_SHAPES:
        raise ValueError("Unknown phase-marker shape.")
    hkl = [None] * len(x) if reference.hkl is None else [str(s).strip() if s is not None else None for s in reference.hkl]
    return x, intensity, hkl


def _selected_indices(length: int, selected: Sequence[int] | None):
    if selected is None:
        return list(range(length))
    result = []
    for index in selected:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < length:
            raise ValueError("A selected reference index is outside the peak list.")
        if int(index) not in result:
            result.append(int(index))
    return result


def format_hkl_groups(groups) -> str:
    """Format a pymatgen-style hkl group without importing pymatgen.

    Reflection families at the same reference position remain joined rather than
    silently discarding all but the first hkl. Negative indices stay explicit.
    """
    if groups is None:
        return ""
    if isinstance(groups, str):
        return groups.strip()
    if isinstance(groups, Mapping):
        groups = [groups]
    groups = list(groups)
    if groups and all(isinstance(value, (int, np.integer)) for value in groups):
        groups = [groups]
    labels = []
    for group in groups:
        indices = group.get("hkl") if isinstance(group, Mapping) else group
        if indices is None:
            continue
        label = "(" + " ".join(str(int(v)) for v in indices) + ")"
        if label not in labels:
            labels.append(label)
    return " / ".join(labels)


def _experimental_peaks(curve: PreparedCurve, prominence_fraction: float,
                        min_distance_deg: float, manual_positions):
    x, y, original_indices = _curve_arrays(curve)
    if not len(x):
        return []
    if manual_positions is not None:
        positions = np.asarray(manual_positions, dtype=float)
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise ValueError("Manual peak positions must be a finite one-dimensional list.")
        selected = set()
        for position in positions:
            if x[0] <= position <= x[-1]:
                selected.add(int(np.argmin(np.abs(x - position))))
        return [(int(original_indices[i]), float(x[i]), float(y[i])) for i in sorted(selected)]
    if len(x) < 3 or np.ptp(y) == 0:
        return []
    # Repeated coordinates are one measured angle, not extra peak neighbours.
    # Pick their strongest stored ordinate only for peak detection; plotted
    # curves and caller-owned arrays remain unchanged.
    starts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1]
    stops = np.r_[starts[1:], len(x)]
    unique_indices = np.asarray([a + int(np.argmax(y[a:b])) for a, b in zip(starts, stops)], dtype=int)
    ux, uy = x[unique_indices], y[unique_indices]
    if len(ux) < 3 or np.ptp(uy) == 0:
        return []
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(uy, prominence=prominence_fraction * float(np.ptp(uy)))
    # Enforce separation in degrees, including for irregular measurement grids.
    ranked = sorted(range(len(peaks)), key=lambda j: (-properties["prominences"][j], -uy[peaks[j]], ux[peaks[j]]))
    kept = []
    for j in ranked:
        candidate = int(peaks[j])
        if all(abs(ux[candidate] - ux[other]) >= min_distance_deg for other in kept):
            kept.append(candidate)
    result = []
    for candidate in sorted(kept):
        i = int(unique_indices[candidate])
        result.append((int(original_indices[i]), float(x[i]), float(y[i])))
    return result


def match_reference_peaks(
    curves: Sequence[PreparedCurve], references: Sequence[ReferencePeaks], tolerance_deg: float,
    *, prominence_fraction: float = 0.03, min_distance_deg: float = 0.15,
    selected_reference_indices: Mapping[int, Sequence[int]] | None = None,
    manual_peak_positions: Mapping[int, Sequence[float]] | None = None,
) -> list[PeakMatch]:
    """Return positional matches to actual measured local maxima.

    Prominence uses each curve's intensity span, so a vertical plotting offset
    does not change matches. Manual positions, if provided for a curve index,
    replace automatic detection for that curve and snap to its nearest sampled
    coordinate. An empty manual list intentionally gives no peaks for that curve.
    Each experimental maximum has at most one match per reference phase: closest
    reference first, then stronger reference intensity as a tie-breaker. A match
    may belong to multiple phases; it does not establish phase identification.
    """
    if not math.isfinite(tolerance_deg) or tolerance_deg < 0:
        raise ValueError("Reference-matching tolerance must be a finite nonnegative number of degrees.")
    if not math.isfinite(prominence_fraction) or not 0 <= prominence_fraction <= 1:
        raise ValueError("Prominence fraction must lie between zero and one.")
    if not math.isfinite(min_distance_deg) or min_distance_deg < 0:
        raise ValueError("Minimum peak separation must be finite and nonnegative.")
    if selected_reference_indices is not None and any(i not in range(len(references)) for i in selected_reference_indices):
        raise ValueError("Selected-reference mapping contains an unknown phase index.")
    if manual_peak_positions is not None and any(i not in range(len(curves)) for i in manual_peak_positions):
        raise ValueError("Manual-peak mapping contains an unknown curve index.")
    prepared_refs = []
    for phase_index, reference in enumerate(references):
        rx, ri, rhkl = _reference_arrays(reference)
        selected = None if selected_reference_indices is None else selected_reference_indices.get(phase_index)
        indices = [j for j in _selected_indices(len(rx), selected) if ri[j] > 0]
        prepared_refs.append((rx, ri, rhkl, indices))
    matches = []
    for curve_index, curve in enumerate(curves):
        manual = None if manual_peak_positions is None else manual_peak_positions.get(curve_index)
        peaks = _experimental_peaks(curve, prominence_fraction, min_distance_deg, manual)
        for phase_index, reference in enumerate(references):
            rx, ri, rhkl, indices = prepared_refs[phase_index]
            if not indices:
                continue
            for original_index, x, y in peaks:
                j = min(indices, key=lambda k: (abs(rx[k] - x), -ri[k], k))
                delta = x - float(rx[j])
                epsilon = 1e-12 * max(1.0, abs(x), abs(float(rx[j])))
                if abs(delta) <= tolerance_deg + epsilon:
                    matches.append(PeakMatch(curve_index, curve.name, phase_index, reference.name,
                        original_index, j, x, y, float(rx[j]), delta, rhkl[j], reference.color,
                        reference.marker, "manual" if manual is not None else "detected"))
    return matches


@contextmanager
def _preserve_mpl_limits(ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    data_lim = ax.dataLim.frozen()
    auto_x, auto_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()
    try:
        yield
    finally:
        ax.dataLim.set(data_lim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_autoscalex_on(auto_x)
        ax.set_autoscaley_on(auto_y)


def add_mpl_phase_annotations(ax, matches: Sequence[PeakMatch], *, show_hkl: bool = False,
                              marker_size: float = 6, font_size: float = 8,
                              show_legend: bool = False):
    """Place open phase symbols on matched experimental ordinates only."""
    artists, seen = [], set()
    with _preserve_mpl_limits(ax):
        for match in matches:
            label = match.phase_name if show_legend and match.phase_index not in seen else "_nolegend_"
            line, = ax.plot([match.x], [match.y], linestyle="none", marker=_MPL_MARKERS[match.marker],
                markersize=marker_size, markerfacecolor="none", markeredgecolor=match.color,
                markeredgewidth=1.0, color=match.color, label=label, zorder=5, clip_on=True)
            artists.append(line)
            seen.add(match.phase_index)
            if show_hkl and match.hkl:
                artists.append(ax.annotate(match.hkl, xy=(match.x, match.y), xytext=(0, marker_size + 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=font_size,
                    color=match.color, clip_on=True, annotation_clip=True, zorder=6))
    return artists


def add_plotly_phase_annotations(fig, matches: Sequence[PeakMatch], *, show_hkl: bool = False,
                                 marker_size: float = 6, font_size: float = 8,
                                 show_legend: bool = False, xref: str = "x", yref: str = "y"):
    import plotly.graph_objects as go
    grouped, seen, trace_indices = defaultdict(list), set(), []
    for match in matches:
        grouped[(match.phase_index, match.curve_index)].append(match)
    for group in grouped.values():
        first = group[0]
        trace_indices.append(len(fig.data))
        fig.add_trace(go.Scatter(x=[m.x for m in group], y=[m.y for m in group], mode="markers",
            xaxis=xref, yaxis=yref, name=first.phase_name, legendgroup=f"phase:{first.phase_index}",
            showlegend=show_legend and first.phase_index not in seen, cliponaxis=True,
            marker=dict(symbol=_PLOTLY_MARKERS[first.marker] + "-open", size=marker_size * 1.4,
                        color=first.color, line=dict(color=first.color, width=1.0)),
            customdata=[[m.reference_x, m.delta_deg] for m in group],
            hovertemplate=_hover_text(first.phase_name) + "<br>2θ=%{x:.4f}°<br>Reference=%{customdata[0]:.4f}°<br>Δ=%{customdata[1]:.4f}°<extra></extra>"))
        seen.add(first.phase_index)
        if show_hkl:
            for match in group:
                if match.hkl:
                    fig.add_annotation(x=match.x, y=match.y, xref=xref, yref=yref,
                        text=html.escape(match.hkl), showarrow=False, yshift=marker_size * 1.4 + 5,
                        yanchor="bottom", font=dict(size=font_size, color=match.color))
    return trace_indices


def _reference_labels(reference, selected_indices, max_labels):
    x, intensity, hkl = _reference_arrays(reference)
    indices = [j for j in _selected_indices(len(x), selected_indices) if hkl[j] and intensity[j] > 0]
    if max_labels is not None:
        if not isinstance(max_labels, int) or max_labels < 0:
            raise ValueError("Maximum hkl-label count must be nonnegative or None.")
        indices = sorted(indices, key=lambda j: (-intensity[j], x[j], j))[:max_labels]
    return x, intensity, hkl, sorted(indices, key=lambda j: (x[j], j))


def add_mpl_reference_hkl_labels(ax, reference: ReferencePeaks, *, baseline: float = 0,
                                 scale: float = 1, selected_indices=None, max_labels: int | None = 12,
                                 font_size: float = 8, rotation: float = 90, offset_points: float = 3):
    """Label selected reference sticks at baseline + scale * stored intensity."""
    x, intensity, hkl, indices = _reference_labels(reference, selected_indices, max_labels)
    annotations = []
    with _preserve_mpl_limits(ax):
        for j in indices:
            display_hkl = hkl[j].replace(" / ", "\n")
            annotations.append(ax.annotate(display_hkl, xy=(x[j], baseline + scale * intensity[j]),
                xytext=(0, offset_points), textcoords="offset points", ha="center", va="bottom",
                fontsize=font_size, rotation=rotation, color=reference.color, clip_on=True,
                annotation_clip=True, zorder=6))
    return annotations


def add_plotly_reference_hkl_labels(fig, reference: ReferencePeaks, *, baseline: float = 0,
                                    scale: float = 1, selected_indices=None, max_labels: int | None = 12,
                                    font_size: float = 8, rotation: float = 90, offset_points: float = 3,
                                    xref: str = "x2", yref: str = "y2"):
    x, intensity, hkl, indices = _reference_labels(reference, selected_indices, max_labels)
    annotation_indices = []
    for j in indices:
        annotation_indices.append(len(fig.layout.annotations))
        display_hkl = html.escape(hkl[j]).replace(" / ", "<br>").replace("\n", "<br>")
        fig.add_annotation(x=float(x[j]), y=float(baseline + scale * intensity[j]), xref=xref, yref=yref,
            text=display_hkl, showarrow=False, yshift=offset_points, yanchor="bottom",
            textangle=-rotation, font=dict(size=font_size, color=reference.color))
    return annotation_indices


def _zoom_geometry(x_range, location, width, height, pad):
    if len(x_range) != 2 or not np.isfinite(x_range).all() or x_range[0] >= x_range[1]:
        raise ValueError("Zoom interval must contain two finite, increasing coordinates.")
    if location not in ("upper left", "upper right", "lower left", "lower right"):
        raise ValueError("Inset location must select an upper/lower left/right corner.")
    if not all(math.isfinite(v) for v in (width, height, pad)) or width <= 0 or height <= 0 or pad < 0 or max(width, height) + 2 * pad > 1:
        raise ValueError("Inset width/height and padding must fit inside the main axes.")
    left = pad if location.endswith("left") else 1 - pad - width
    bottom = 1 - pad - height if location.startswith("upper") else pad
    return (float(x_range[0]), float(x_range[1])), (left, bottom, width, height)


def _zoom_curves(curves, interval, y_padding):
    if not math.isfinite(y_padding) or y_padding < 0:
        raise ValueError("Inset intensity padding must be finite and nonnegative.")
    low, high = interval
    clipped, limit_values = [], []
    for curve in curves:
        x, y, _ = _curve_arrays(curve)
        if not len(x) or high < x[0] or low > x[-1]:
            continue
        start = max(0, int(np.searchsorted(x, low, side="left")) - 1)
        stop = min(len(x), int(np.searchsorted(x, high, side="right")) + 1)
        clipped.append((curve, x[start:stop], y[start:stop]))
        inside = (x >= low) & (x <= high)
        limit_values.extend(y[inside].tolist())
        # The inset draws the original line segments; interpolated boundary
        # values only determine its ordinate limits, not its measurement data.
        for edge in (low, high):
            if x[0] <= edge <= x[-1]:
                limit_values.append(float(np.interp(edge, x, y)))
    if not clipped or not limit_values:
        return [], None
    minimum, maximum = min(limit_values), max(limit_values)
    span = maximum - minimum
    margin = y_padding * (span if span > 0 else max(abs(maximum), 1.0))
    if margin == 0 and span == 0:
        margin = max(abs(maximum), 1.0) * 0.05
    return clipped, (minimum - margin, maximum + margin)


def add_mpl_zoom_inset(ax, curves: Sequence[PreparedCurve], x_range, *, location="upper left",
                       width: float = 0.42, height: float = 0.35, pad: float = 0.05,
                       font_size: float | None = None, show_indicator: bool = True,
                       show_yticklabels: bool = False, y_padding: float = 0.08):
    """Add a fixed-position zoom of the same normalized/offset plotted curves.

    Returns the inset Axes, or None if the chosen interval contains no curve.
    Both positions and intensities come from supplied curves without rescaling.
    """
    from matplotlib.patches import Rectangle
    interval, bounds = _zoom_geometry(x_range, location, width, height, pad)
    clipped, y_range = _zoom_curves(curves, interval, y_padding)
    if not clipped:
        return None
    size = float(font_size) if font_size is not None else max(7, ax.xaxis.label.get_fontsize() * 0.72)
    with _preserve_mpl_limits(ax):
        inset = ax.inset_axes(bounds, transform=ax.transAxes, zorder=10)
        inset.set_in_layout(False)
        inset.set_facecolor("white")
        for curve, x, y in clipped:
            inset.plot(x, y, color=curve.color, linewidth=curve.linewidth, label=curve.name)
        inset.set_xlim(interval)
        inset.set_ylim(y_range)
        parent_ticks = ax.xaxis.get_major_ticks()
        tick_width = parent_ticks[0].tick1line.get_markeredgewidth() if parent_ticks else 0.8
        inset.tick_params(axis="both", direction="in", labelsize=size, length=3, width=tick_width)
        inset.tick_params(axis="y", labelleft=show_yticklabels)
        inset.text(.97, .96, f"{interval[0]:g}–{interval[1]:g}°", transform=inset.transAxes,
                   ha="right", va="top", fontsize=size,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1))
        for spine in inset.spines.values():
            spine.set_color(ax.spines["left"].get_edgecolor())
            spine.set_linewidth(ax.spines["left"].get_linewidth())
        if show_indicator:
            indicator = Rectangle((interval[0], 0), interval[1] - interval[0], 1,
                transform=ax.get_xaxis_transform(), facecolor="none", edgecolor="#777777",
                linewidth=0.8, linestyle="--", zorder=3, clip_on=True)
            ax.add_patch(indicator)
    return inset


def _plotly_axis_key(reference: str, letter: str):
    if not re.fullmatch(letter + r"(?:[2-9]|[1-9]\d+)?", reference):
        raise ValueError("Plotly axis references must use x/y, x2/y2, and so on.")
    return letter + "axis" + reference[1:]


def add_plotly_zoom_inset(fig, curves: Sequence[PreparedCurve], x_range, *, location="upper left",
                          width: float = 0.42, height: float = 0.35, pad: float = 0.05,
                          font_size: float | None = None, show_indicator: bool = True,
                          show_yticklabels: bool = False, y_padding: float = 0.08,
                          parent_xaxis: str = "x", parent_yaxis: str = "y"):
    """Add independent inset axes without changing existing axes or trace data.

    Returns (x-axis reference, y-axis reference), e.g. ('x3', 'y3') when a
    reference subplot already occupies x2/y2, or None for an empty interval.
    """
    import plotly.graph_objects as go
    interval, bounds = _zoom_geometry(x_range, location, width, height, pad)
    clipped, y_range = _zoom_curves(curves, interval, y_padding)
    if not clipped:
        return None
    xkey, ykey = _plotly_axis_key(parent_xaxis, "x"), _plotly_axis_key(parent_yaxis, "y")
    parent_x, parent_y = getattr(fig.layout, xkey), getattr(fig.layout, ykey)
    x_domain, y_domain = list(parent_x.domain or (0, 1)), list(parent_y.domain or (0, 1))
    left, bottom, iw, ih = bounds
    domain_x = [x_domain[0] + value * (x_domain[1] - x_domain[0]) for value in (left, left + iw)]
    domain_y = [y_domain[0] + value * (y_domain[1] - y_domain[0]) for value in (bottom, bottom + ih)]
    used = {1}
    for key in fig.layout:
        match = re.fullmatch(r"[xy]axis(\d*)", key)
        if match:
            used.add(int(match[1] or "1"))
    for trace in fig.data:
        for letter in ("x", "y"):
            reference = getattr(trace, letter + "axis", None)
            if reference:
                used.add(int(reference[1:] or "1"))
    number = max(used) + 1
    xref, yref = f"x{number}", f"y{number}"
    size = float(font_size) if font_size is not None else max(7, float(fig.layout.font.size or 14) * 0.72)
    font = dict(family=fig.layout.font.family or "Arial", color=fig.layout.font.color or "black", size=size)
    linecolor = parent_x.linecolor or "black"
    frame_width = parent_x.linewidth or 0.8
    fig.update_layout({
        f"xaxis{number}": dict(domain=domain_x, anchor=yref, range=list(interval), autorange=False,
            showline=True, mirror=True, linecolor=linecolor, linewidth=frame_width, ticks="inside",
            tickwidth=parent_x.tickwidth or frame_width,
            tickfont=font, nticks=4, showgrid=False, zeroline=False, automargin=False),
        f"yaxis{number}": dict(domain=domain_y, anchor=xref, range=list(y_range), autorange=False,
            showline=True, mirror=True, linecolor=parent_y.linecolor or linecolor, linewidth=parent_y.linewidth or frame_width,
            tickwidth=parent_y.tickwidth or parent_y.linewidth or frame_width,
            ticks="inside", tickfont=font, showticklabels=show_yticklabels, nticks=3,
            showgrid=False, zeroline=False, automargin=False),
    })
    # Opaque fill belongs to the new subplot, after the main subplot's traces.
    # A paper-level white shape would cover the inset's own lines as well.
    fig.add_trace(go.Scatter(x=[interval[0], interval[1], interval[1], interval[0], interval[0]],
        y=[y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]], xaxis=xref, yaxis=yref,
        mode="lines", line=dict(width=0, color="white"), fill="toself", fillcolor="white",
        showlegend=False, hoverinfo="skip", cliponaxis=True, name="_zoom_background"))
    for curve, x, y in clipped:
        fig.add_trace(go.Scatter(x=x, y=y, xaxis=xref, yaxis=yref, mode="lines", name=curve.name,
            line=dict(color=curve.color, width=curve.linewidth * 4 / 3), showlegend=False, cliponaxis=True,
            hovertemplate=_hover_text(curve.name) + "<br>2θ=%{x:.4f}°<br>I=%{y:.4g}<extra></extra>"))
    fig.add_annotation(x=domain_x[1] - .03 * (domain_x[1] - domain_x[0]),
        y=domain_y[1] - .04 * (domain_y[1] - domain_y[0]), xref="paper", yref="paper",
        text=f"{interval[0]:g}–{interval[1]:g}°", showarrow=False, xanchor="right", yanchor="top",
        bgcolor="white", borderpad=1, font=font)
    if show_indicator:
        all_min = min(float(np.min(x)) for _, x, _ in clipped)
        all_max = max(float(np.max(x)) for _, x, _ in clipped)
        fig.add_shape(type="rect", x0=max(interval[0], all_min), x1=min(interval[1], all_max),
            y0=0, y1=1, xref=parent_xaxis, yref=parent_yaxis + " domain", fillcolor="rgba(0,0,0,0)",
            line=dict(color="#777777", width=0.8, dash="dash"), layer="below")
    return xref, yref
