"""Known-result tests for optional matched-peak annotations and linked insets."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_annotations import (
    PreparedCurve, ReferencePeaks, match_reference_peaks, format_hkl_groups,
    add_mpl_phase_annotations, add_plotly_phase_annotations,
    add_mpl_reference_hkl_labels, add_plotly_reference_hkl_labels,
    add_mpl_zoom_inset, add_plotly_zoom_inset,
)


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.curve = PreparedCurve("sample", [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
                                   [0, 5, 0, 0, 9, 0, 0], "#111111")
        self.ref = ReferencePeaks("phase A", [10.11, 10.38, 10.55], [100, 50, 20],
                                  ["(1 0 0)", "(1 1 0)", "(2 0 0)"], "#b71c1c", "triangle")

    def test_only_actual_measured_peaks_within_tolerance_match(self):
        matches = match_reference_peaks([self.curve], [self.ref], .03, min_distance_deg=0)
        self.assertEqual([m.x for m in matches], [10.1, 10.4])
        self.assertEqual([m.y for m in matches], [5, 9])
        self.assertEqual([m.reference_peak_index for m in matches], [0, 1])
        self.assertAlmostEqual(matches[0].delta_deg, -.01)
        self.assertEqual(matches[1].hkl, "(1 1 0)")

    def test_vertical_offset_and_positive_scale_do_not_change_matches(self):
        shifted = PreparedCurve("shifted", self.curve.x, np.asarray(self.curve.y) / 9 + 17)
        original = match_reference_peaks([self.curve], [self.ref], .03)
        matches = match_reference_peaks([shifted], [self.ref], .03)
        self.assertEqual([m.x for m in matches], [m.x for m in original])
        self.assertAlmostEqual(matches[0].y, 17 + 5 / 9)

    def test_each_unresolved_peak_may_match_multiple_reference_phases(self):
        other = ReferencePeaks("phase B", [10.1], [100], marker="square")
        matches = match_reference_peaks([self.curve], [self.ref, other], .03)
        self.assertEqual([(m.phase_name, m.x) for m in matches],
                         [("phase A", 10.1), ("phase A", 10.4), ("phase B", 10.1)])

    def test_duplicate_nearby_reference_sticks_do_not_duplicate_phase_marker(self):
        reference = ReferencePeaks("phase", [10.08, 10.12], [20, 100], ["a", "b"])
        matches = match_reference_peaks([self.curve], [reference], .03)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].reference_peak_index, 1)

    def test_selected_reference_indices_and_empty_selection(self):
        matches = match_reference_peaks([self.curve], [self.ref], .03, selected_reference_indices={0: [1]})
        self.assertEqual([m.x for m in matches], [10.4])
        self.assertEqual(match_reference_peaks([self.curve], [self.ref], .03, selected_reference_indices={0: []}), [])

    def test_manual_peak_selection_is_explicit_and_snaps_to_sampled_coordinate(self):
        matches = match_reference_peaks([self.curve], [self.ref], .03,
            manual_peak_positions={0: [10.095, 99]})
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source, "manual")
        self.assertEqual(matches[0].x, 10.1)
        self.assertEqual(match_reference_peaks([self.curve], [self.ref], .03,
            manual_peak_positions={0: []}), [])

    def test_degree_separation_uses_irregular_grid_not_sample_count(self):
        curve = PreparedCurve("irregular", [0, 1, 1.01, 1.02, 5, 8, 10], [0, 7, 0, 6, 0, 8, 0])
        reference = ReferencePeaks("phase", [1, 1.02, 8])
        matches = match_reference_peaks([curve], [reference], .001, min_distance_deg=.1)
        self.assertEqual([m.x for m in matches], [1, 8])

    def test_empty_constant_and_short_curves_give_no_matches(self):
        for curve in (PreparedCurve("empty", [], []), PreparedCurve("flat", [1, 2, 3], [4, 4, 4]),
                      PreparedCurve("short", [1, 2], [0, 5])):
            self.assertEqual(match_reference_peaks([curve], [self.ref], .1), [])
        self.assertEqual(match_reference_peaks([self.curve], [], .1), [])

    def test_bad_parameters_and_nonfinite_inputs_fail_clearly(self):
        for tolerance in (-1, float("nan")):
            with self.assertRaises(ValueError):
                match_reference_peaks([self.curve], [self.ref], tolerance)
        with self.assertRaises(ValueError):
            match_reference_peaks([PreparedCurve("bad", [1, 2, 3], [0, float("nan"), 1])], [self.ref], .1)

    def test_miller_groups_preserve_families_and_negative_indices(self):
        self.assertEqual(format_hkl_groups([{"hkl": (1, 0, -1)}, {"hkl": (0, 1, 1)}, {"hkl": (1, 0, -1)}]),
                         "(1 0 -1) / (0 1 1)")
        self.assertEqual(format_hkl_groups((1, 1, 0)), "(1 1 0)")
        self.assertEqual(format_hkl_groups(None), "")


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.curve = PreparedCurve("sample", np.arange(7), [0, 5, 0, 0, 9, 0, 0], "#111111")
        self.ref = ReferencePeaks("phase", [1.01, 4.02, 5.5], [100, 50, 20],
                                  ["(1 0 0)", "(1 1 0)", "(2 0 0)"], "#b71c1c", "diamond")
        self.matches = match_reference_peaks([self.curve], [self.ref], .03)

    def tearDown(self):
        plt.close("all")

    def test_mpl_markers_and_labels_keep_limits_and_original_data(self):
        fig, ax = plt.subplots()
        line, = ax.plot(self.curve.x, self.curve.y)
        limits = ax.get_xlim(), ax.get_ylim()
        data_lim = ax.dataLim.get_points().copy()
        artists = add_mpl_phase_annotations(ax, self.matches, show_hkl=True)
        self.assertEqual(len(artists), 4)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        np.testing.assert_array_equal(ax.dataLim.get_points(), data_lim)
        np.testing.assert_array_equal(line.get_ydata(), self.curve.y)
        self.assertEqual([text.get_text() for text in ax.texts], ["(1 0 0)", "(1 1 0)"])

    def test_reference_hkl_labels_use_selected_stick_heads_only(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 7)
        ax.set_ylim(100, 220)
        annotations = add_mpl_reference_hkl_labels(ax, self.ref, baseline=110, scale=.8,
                                                  selected_indices=[1, 2], max_labels=1)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].get_text(), "(1 1 0)")
        self.assertEqual(annotations[0].xy, (4.02, 150))
        self.assertEqual(ax.get_ylim(), (100, 220))

    def test_missing_hkl_and_empty_matches_are_noops(self):
        fig, ax = plt.subplots()
        self.assertEqual(add_mpl_phase_annotations(ax, []), [])
        self.assertEqual(add_mpl_reference_hkl_labels(ax, ReferencePeaks("no hkl", [1], [100])), [])
        pfig = go.Figure()
        self.assertEqual(add_plotly_phase_annotations(pfig, []), [])
        self.assertEqual(len(pfig.data), 0)

    def test_multi_family_hkl_wraps_only_in_display_and_fits_reference_row(self):
        original = "(1 0 0) / (0 1 0)"
        reference = ReferencePeaks("phase", [4], [100], [original])
        fig, (main, ax) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 320)
        annotation = add_mpl_reference_hkl_labels(ax, reference, baseline=160, scale=.7)[0]
        self.assertEqual(annotation.get_text(), "(1 0 0)\n(0 1 0)")
        self.assertEqual(reference.hkl, [original])
        fig.canvas.draw()
        self.assertLessEqual(annotation.get_window_extent(fig.canvas.get_renderer()).y1, ax.bbox.y1)
        pfig = make_subplots(rows=2, cols=1)
        add_plotly_reference_hkl_labels(pfig, reference, baseline=160, scale=.7)
        self.assertEqual(pfig.layout.annotations[-1].text, "(1 0 0)<br>(0 1 0)")

    def test_plotly_annotations_match_positions_without_changing_existing_axes_or_traces(self):
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=self.curve.x, y=self.curve.y), row=1, col=1)
        fig.update_xaxes(range=[0, 7], row=1, col=1)
        fig.update_yaxes(range=[-1, 11], row=1, col=1)
        before = fig.layout.xaxis.to_plotly_json(), fig.layout.yaxis.to_plotly_json()
        indices = add_plotly_phase_annotations(fig, self.matches, show_hkl=True)
        self.assertEqual(indices, [1])
        self.assertEqual(list(fig.data[1].x), [1, 4])
        self.assertEqual(fig.data[1].marker.symbol, "diamond-open")
        self.assertEqual((fig.layout.xaxis.to_plotly_json(), fig.layout.yaxis.to_plotly_json()), before)
        np.testing.assert_array_equal(fig.data[0].y, self.curve.y)
        self.assertEqual(len(fig.layout.annotations), 2)
        add_plotly_reference_hkl_labels(fig, self.ref, baseline=110, scale=.8, selected_indices=[1], font_size=9)
        last = fig.layout.annotations[-1]
        self.assertEqual((last.xref, last.yref, last.x, last.y), ("x2", "y2", 4.02, 150))


class ZoomTests(unittest.TestCase):
    def setUp(self):
        self.curves = [PreparedCurve("a", [0, 1, 2, 3, 4], [4, 4.5, 5, 4.5, 4], "#b71c1c"),
                       PreparedCurve("b", [0, 1, 2, 3, 4], [7, 7.2, 7.8, 7.2, 7], "#1565c0")]

    def tearDown(self):
        plt.close("all")

    def test_mpl_inset_copies_prepared_ordinates_and_preserves_parent(self):
        fig, ax = plt.subplots()
        for curve in self.curves:
            ax.plot(curve.x, curve.y)
        ax.set_xlim(0, 4)
        limits = ax.get_xlim(), ax.get_ylim()
        data_lim = ax.dataLim.get_points().copy()
        inset = add_mpl_zoom_inset(ax, self.curves, (1, 3))
        self.assertIsNotNone(inset)
        self.assertEqual(inset.get_xlim(), (1, 3))
        self.assertEqual(len(inset.lines), 2)
        np.testing.assert_array_equal(inset.lines[0].get_ydata(), self.curves[0].y)
        np.testing.assert_array_equal(inset.lines[1].get_ydata(), self.curves[1].y)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        np.testing.assert_array_equal(ax.dataLim.get_points(), data_lim)

    def test_zoom_between_sample_points_uses_same_line_and_interpolated_limits(self):
        curve = PreparedCurve("line", [0, 10], [5, 15])
        fig, ax = plt.subplots()
        ax.plot(curve.x, curve.y)
        inset = add_mpl_zoom_inset(ax, [curve], (4, 6), y_padding=0)
        self.assertEqual(inset.get_ylim(), (9, 11))
        np.testing.assert_array_equal(inset.lines[0].get_xdata(), [0, 10])
        np.testing.assert_array_equal(inset.lines[0].get_ydata(), [5, 15])

    def test_empty_interval_and_no_samples_do_not_create_axes(self):
        fig, ax = plt.subplots()
        self.assertIsNone(add_mpl_zoom_inset(ax, self.curves, (8, 9)))
        self.assertEqual(len(ax.child_axes), 0)
        pfig = make_subplots(rows=1, cols=1)
        before = pfig.to_plotly_json()
        self.assertIsNone(add_plotly_zoom_inset(pfig, [], (1, 2)))
        self.assertEqual(pfig.to_plotly_json(), before)

    def test_flat_curves_need_no_detected_peaks_or_references_for_zoom(self):
        fig, ax = plt.subplots()
        inset = add_mpl_zoom_inset(ax, [PreparedCurve("flat", [1, 2, 3], [4, 4, 4])], (1, 3))
        self.assertLess(inset.get_ylim()[0], 4)
        self.assertGreater(inset.get_ylim()[1], 4)

    def test_plotly_inset_adds_x3_y3_beside_existing_reference_axes(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.75, .25])
        for curve in self.curves:
            fig.add_trace(go.Scatter(x=curve.x, y=curve.y), row=1, col=1)
        fig.add_trace(go.Scatter(x=[1, 1, None], y=[0, 80, None]), row=2, col=1)
        fig.update_xaxes(range=[0, 4])
        fig.update_yaxes(range=[3, 9], row=1, col=1)
        original_axes = {name: getattr(fig.layout, name).to_plotly_json() for name in ("xaxis", "yaxis", "xaxis2", "yaxis2")}
        original_data = [trace.to_plotly_json() for trace in fig.data]
        result = add_plotly_zoom_inset(fig, self.curves, (1, 3), location="upper right")
        self.assertEqual(result, ("x3", "y3"))
        self.assertEqual(list(fig.layout.xaxis3.range), [1, 3])
        self.assertGreater(fig.layout.yaxis3.domain[0], fig.layout.yaxis.domain[0])
        for name, value in original_axes.items():
            self.assertEqual(getattr(fig.layout, name).to_plotly_json(), value)
        for before, after in zip(original_data, fig.data):
            self.assertEqual(after.to_plotly_json(), before)
        self.assertEqual(fig.data[3].name, "_zoom_background")
        self.assertEqual(fig.data[3].fillcolor, "white")
        np.testing.assert_array_equal(fig.data[4].y, self.curves[0].y)
        np.testing.assert_array_equal(fig.data[5].y, self.curves[1].y)
        self.assertEqual(fig.layout.shapes[-1].yref, "y domain")

    def test_plotly_multiple_insets_do_not_reuse_axis_ids(self):
        fig = make_subplots(rows=1, cols=1)
        self.assertEqual(add_plotly_zoom_inset(fig, self.curves, (1, 2)), ("x2", "y2"))
        self.assertEqual(add_plotly_zoom_inset(fig, self.curves, (2, 3)), ("x3", "y3"))

    def test_inset_preserves_parent_frame_and_tick_style(self):
        fig, ax = plt.subplots()
        ax.spines["left"].set_linewidth(1.7)
        ax.tick_params(width=1.1)
        inset = add_mpl_zoom_inset(ax, self.curves, (1, 3))
        self.assertEqual(inset.spines["left"].get_linewidth(), 1.7)
        self.assertEqual(inset.xaxis.get_major_ticks()[0].tick1line.get_markeredgewidth(), 1.1)
        pfig = make_subplots(rows=1, cols=1)
        pfig.update_xaxes(linewidth=1.7, tickwidth=1.1, linecolor="#123456")
        add_plotly_zoom_inset(pfig, self.curves, (1, 3))
        self.assertEqual(pfig.layout.xaxis2.linewidth, 1.7)
        self.assertEqual(pfig.layout.xaxis2.tickwidth, 1.1)
        self.assertEqual(pfig.layout.xaxis2.linecolor, "#123456")

    def test_all_four_manual_locations_stay_inside_main_axes(self):
        for location in ("upper left", "upper right", "lower left", "lower right"):
            with self.subTest(location=location):
                pfig = make_subplots(rows=2, cols=1, row_heights=[.75, .25])
                add_plotly_zoom_inset(pfig, self.curves, (1, 3), location=location)
                self.assertGreaterEqual(pfig.layout.yaxis3.domain[0], pfig.layout.yaxis.domain[0])
                self.assertLessEqual(pfig.layout.yaxis3.domain[1], pfig.layout.yaxis.domain[1])
                expected_x0 = .05 if location.endswith("left") else .53
                self.assertAlmostEqual(pfig.layout.xaxis3.domain[0], expected_x0)
                fig, ax = plt.subplots()
                inset = add_mpl_zoom_inset(ax, self.curves, (1, 3), location=location)
                fig.canvas.draw()
                parent_bounds = ax.get_position()
                inset_bounds = inset.get_position()
                self.assertGreaterEqual(inset_bounds.y0, parent_bounds.y0)
                self.assertLessEqual(inset_bounds.y1, parent_bounds.y1)

    def test_invalid_zoom_interval_and_bounds_fail(self):
        fig, ax = plt.subplots()
        for interval in ((2, 1), (1, 1), (1, float("nan"))):
            with self.assertRaises(ValueError):
                add_mpl_zoom_inset(ax, self.curves, interval)
        with self.assertRaises(ValueError):
            add_mpl_zoom_inset(ax, self.curves, (1, 3), width=.95, pad=.1)


if __name__ == "__main__":
    unittest.main()
