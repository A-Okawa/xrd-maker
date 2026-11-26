"""Figures stay with the run that draws them when Streamlit runs scripts in parallel threads.

On Streamlit Cloud, quick clicks (choosing Okabe-Ito black for every reference) gave
"AttributeError: 'FigureCanvasBase' object has no attribute 'get_renderer'" in the left-margin fit:
each session's script, and a run cut short by a newer one, runs in its own thread; the figures came
from pyplot's list of open figures, which all threads share, so two runs could get the same figure and
one closed it (which leaves a bare canvas) while the other still drew on it. The figures are now made
without pyplot, and drawing and saving hold one lock, since matplotlib's settings are shared as well.
"""

import io
from pathlib import Path
import sys
import threading
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import test_xrd_regressions as regression


def left_blank_inches(png, dpi):
    grey = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    return np.nonzero((grey < 250).any(axis=0))[0][0] / dpi


class FigureOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.ns = regression.renderer_namespace()

    def test_figures_are_not_registered_with_pyplot(self):
        before = plt.get_fignums()
        fig = self.ns["build_figure"](fit_left=True)
        self.assertEqual(plt.get_fignums(), before)
        self.assertIsInstance(fig.canvas, FigureCanvasAgg)

    def test_closing_pyplot_figures_leaves_ours_alone(self):
        fig = self.ns["build_figure"]()
        plt.figure()
        plt.close("all")                                   # what another run's cleanup could do
        self.assertIsInstance(fig.canvas, FigureCanvasAgg)
        self.ns["fitted_left_margin"](fig, self.ns["plot_style"])

    def test_the_fit_needs_no_particular_canvas(self):
        # The line that failed: it measures through the figure, whatever canvas it has.
        fig = self.ns["build_figure"]()
        FigureCanvasBase(fig)
        left = self.ns["fitted_left_margin"](fig, self.ns["plot_style"])
        self.assertGreater(left, 0)
        self.assertLess(left, self.ns["plot_style"]["margin_left"])

    def test_the_lock_is_one_for_all_runs(self):
        self.assertIs(self.ns["matplotlib_lock"](), self.ns["matplotlib_lock"]())


class ParallelRunTests(unittest.TestCase):
    def test_overlapping_runs_each_draw_and_save_their_own_figure(self):
        ns = regression.renderer_namespace()
        lock = ns["matplotlib_lock"]()
        results, errors = [], []

        def run():
            try:
                for _ in range(6):
                    with lock:                                  # as the app does
                        fig = ns["build_figure"](fit_left=True)
                    style = dict(ns["plot_style"], margin_left=fig.subplotpars.left)
                    with lock:
                        png = ns["export_figure"](fig, "png", 150, style=style)
                        ns["export_figure"](fig, "pdf", 300, style=style)
                    results.append((len(fig.axes), round(fig.subplotpars.left, 6), round(left_blank_inches(png, 150), 3)))
            except Exception as error:                          # noqa: BLE001 - reported below
                errors.append(repr(error))

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 24)
        self.assertEqual(len(set(results)), 1, set(results))     # every run drew the same figure, alone


if __name__ == "__main__":
    unittest.main()
