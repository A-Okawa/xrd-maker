"""Data Reliability: the checks of the data a figure is drawn from, and the figure record.

All data here are synthetic, made in this file (no measured research data). The checks only report:
every test that reads a file also checks that the angles and intensities are the ones read before.
"""

import ast
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import data_quality as qc
from xrd_io import read_patterns
import test_xrd_regressions as regression


def statuses(checks):
    return {check.code: check.status for check in checks}


def synthetic_scan(start=10.0, stop=80.0, step=0.02, peaks=((28.4, 1000.0), (47.3, 600.0)), noise=0.0, seed=1):
    """A powder pattern with known Gaussian peaks on a flat background of 100 counts."""
    x = np.round(np.arange(start, stop + step / 2, step), 6)
    y = np.full_like(x, 100.0)
    for centre, height in peaks:
        y += height * np.exp(-0.5 * ((x - centre) / 0.05) ** 2)
    if noise:
        y += np.random.default_rng(seed).normal(0, noise, len(x))
    return x, y


def xy_bytes(x, y, header=None):
    lines = ([header] if header else []) + [f"{a:.6f} {b:.6f}" for a, b in zip(x, y)]
    return ("\n".join(lines) + "\n").encode()


def ras_bytes(x, y, factors=None, wavelength="1.540593", target="Cu"):
    factors = np.ones(len(x)) if factors is None else factors
    header = ['*RAS_DATA_START', '*RAS_HEADER_START', '*MEAS_SCAN_AXIS_X "TwoThetaTheta"',
              '*MEAS_SCAN_UNIT_X "deg"', '*MEAS_SCAN_UNIT_Y "cps"', f'*MEAS_DATA_COUNT "{len(x)}"']
    if wavelength is not None:
        header.append(f'*HW_XG_WAVE_LENGTH_ALPHA1 "{wavelength}"')
    if target is not None:
        header.append(f'*HW_XG_TARGET_NAME "{target}"')
    rows = [f"{a:.4f} {b:.4f} {f:.4f}" for a, b, f in zip(x, y, factors)]
    return "\n".join(header + ["*RAS_HEADER_END", "*RAS_INT_START", *rows, "*RAS_INT_END", "*RAS_DATA_END"]).encode()


def checked(data, name="synthetic.xy", **kwargs):
    pattern = read_patterns(data, name)[0]
    return pattern, qc.check_pattern(pattern.x, pattern.y, pattern.metadata, **kwargs)


class MeasurementCheckTests(unittest.TestCase):
    def test_a_clean_scan_passes_and_is_read_unchanged(self):
        x, y = synthetic_scan()
        pattern, checks = checked(ras_bytes(x, y), "clean.ras")
        np.testing.assert_allclose(pattern.x, x, atol=1e-4)
        np.testing.assert_allclose(pattern.y, np.round(y, 4))
        self.assertEqual(qc.overall(checks), qc.PASS, statuses(checks))
        # the known peak is where it was put: nothing shifts or rescales the data
        self.assertAlmostEqual(pattern.x[np.argmax(pattern.y)], 28.4, places=6)

    def test_noise_alone_raises_nothing(self):
        x, y = synthetic_scan(noise=15.0)
        pattern, checks = checked(xy_bytes(x, y, "2theta Intensity"))
        np.testing.assert_allclose(pattern.y, np.round(y, 6))
        self.assertEqual(qc.overall(checks), qc.PASS, statuses(checks))

    def test_an_outlier_is_kept_and_not_judged(self):
        x, y = synthetic_scan()
        y[500] = 1e7                                     # a single spike stays in the data as stored
        pattern, checks = checked(xy_bytes(x, y, "2theta Intensity"))
        self.assertEqual(pattern.y.max(), 1e7)
        self.assertEqual(qc.overall(checks), qc.PASS)

    def test_rows_that_are_not_numbers_are_reported_by_line(self):
        data = b"2theta Intensity\n10.00 100\n10.02 nan\n10.04 120\n10.06 inf\n10.08 90\n" + b"".join(
            f"{10.1 + 0.02 * i:.2f} 100\n".encode() for i in range(20))
        pattern, checks = checked(data)
        self.assertEqual(len(pattern.x), 23)               # as before: the two rows are not drawn
        check = next(c for c in checks if c.code == "finite")
        self.assertEqual((check.status, check.values["count"], check.values["rows"]), (qc.WARNING, 2, "3, 5"))
        self.assertEqual(qc.overall(checks), qc.WARNING)

    def test_attenuator_factors_other_than_one_warn_with_their_angles(self):
        x, y = synthetic_scan()
        factors = np.ones(len(x))
        factors[(x > 28.3) & (x < 28.5)] = 8.7
        pattern, checks = checked(ras_bytes(x, y, factors), "att.ras")
        np.testing.assert_allclose(pattern.y, np.round(y, 4))          # still as stored, not multiplied
        check = next(c for c in checks if c.code == "attenuator")
        self.assertEqual(check.status, qc.WARNING)
        self.assertEqual(check.values["count"], int(np.count_nonzero(factors != 1)))
        self.assertIn("28.3", check.values["runs"])
        self.assertEqual((check.values["low"], check.values["high"]), (1.0, 8.7))
        self.assertFalse(pattern.metadata["attenuator_column_applied"])

    def test_wavelength_other_than_cu_warns_only_with_references(self):
        x, y = synthetic_scan()
        data = ras_bytes(x, y, wavelength="0.709319", target="Mo")
        _, alone = checked(data, "mo.ras")
        _, with_refs = checked(data, "mo.ras", references_shown=True)
        self.assertEqual(statuses(alone)["wavelength"], qc.PASS)
        self.assertEqual(statuses(with_refs)["wavelength"], qc.WARNING)
        _, cu = checked(ras_bytes(x, y), "cu.ras", references_shown=True)
        self.assertEqual(statuses(cu)["wavelength"], qc.PASS)
        _, unstated = checked(ras_bytes(x, y, wavelength=None, target=None), "none.ras", references_shown=True)
        self.assertEqual(statuses(unstated)["wavelength"], qc.NOT_EVALUATED)

    def test_uxd_states_its_wavelength(self):
        data = (b"_FILEVERSION=1\n_WL1=0.70930\n_ANODE=\"Mo\"\n_DRIVE=\"COUPLED\"\n_START=10\n_STEPSIZE=0.02\n_COUNTS\n"
                + b" ".join(b"100" for _ in range(30)) + b"\n")
        pattern = read_patterns(data, "mo.uxd")[0]
        self.assertEqual(pattern.metadata["wavelength_alpha1_A"], 0.7093)
        self.assertFalse(qc.is_cu(pattern.metadata))

    def test_angles_outside_0_to_180_warn(self):
        data = b"".join(f"{a} 1\n".encode() for a in range(0, 400, 20))
        _, checks = checked(data)
        check = next(c for c in checks if c.code == "angle_range")
        self.assertEqual((check.status, check.values["count"]), (qc.WARNING, 10))        # 200–380°

    def test_unsorted_rows_and_repeated_angles_are_noted_and_drawn_as_before(self):
        data = b"10.0 1\n10.4 5\n10.2 3\n10.2 4\n" + b"".join(f"{10.6 + 0.2 * i:.1f} 2\n".encode() for i in range(10))
        pattern, checks = checked(data)
        self.assertEqual(pattern.x[:4].tolist(), [10.0, 10.2, 10.2, 10.4])
        check = next(c for c in checks if c.code == "order")
        self.assertEqual((check.status, check.values["reordered"], check.values["duplicates"]), (qc.CAUTION, True, 1))

    def test_too_few_points_and_too_few_to_judge_the_step(self):
        _, checks = checked(b"10 1\n20 2\n")
        self.assertEqual(statuses(checks)["points"], qc.CAUTION)
        self.assertEqual(statuses(checks)["step"], qc.NOT_EVALUATED)
        _, more = checked(b"10 1\n20 2\n", min_points=2)       # the threshold can be changed
        self.assertEqual(statuses(more)["points"], qc.PASS)

    def test_joined_scans_with_another_step_are_noted(self):
        x = np.concatenate([np.arange(10, 30, 0.02), np.arange(30, 40, 0.021)])     # 5 % wider steps after 30°
        _, checks = checked(xy_bytes(x, np.full(len(x), 5.0), "2theta Intensity"))
        check = next(c for c in checks if c.code == "step")
        self.assertEqual(check.status, qc.CAUTION)
        self.assertAlmostEqual(check.values["first"], 30.0, places=6)
        _, lenient = checked(xy_bytes(x, np.full(len(x), 5.0), "2theta Intensity"), step_tolerance_percent=10)
        self.assertEqual(statuses(lenient)["step"], qc.PASS)

    def test_negative_intensities_are_a_caution(self):
        x, y = synthetic_scan()
        _, checks = checked(xy_bytes(x, y - 150, "2theta Intensity"))
        self.assertEqual(statuses(checks)["negative"], qc.CAUTION)

    def test_columns_and_axis_in_text_files(self):
        _, three = checked(b"".join(f"{10 + 0.02 * i:.2f} 5 0.3\n".encode() for i in range(20)))
        self.assertEqual(statuses(three)["columns"], qc.CAUTION)
        self.assertEqual(statuses(three)["axis"], qc.NOT_EVALUATED)
        self.assertEqual(qc.overall(three), qc.CAUTION)
        _, named = checked(b"twotheta yobs ycal\n" + b"".join(f"{10 + 0.02 * i:.2f} 5 4\n".encode() for i in range(20)))
        self.assertEqual(statuses(named)["axis"], qc.PASS)
        self.assertEqual(statuses(named)["columns"], qc.PASS)

    def test_normalising_by_a_maximum_of_zero_or_below_warns(self):
        self.assertEqual(qc.check_display(np.array([-5.0, -1.0]), normalize=True).status, qc.WARNING)
        self.assertEqual(qc.check_display(np.array([-5.0, -1.0]), normalize=False).status, qc.PASS)
        self.assertEqual(qc.check_display(np.array([]), normalize=True).status, qc.CAUTION)

    def test_huge_and_tiny_values_raise_no_numerical_error(self):
        x, _ = synthetic_scan()
        with np.errstate(all="raise"):
            for y in (np.full(len(x), 1e300), np.full(len(x), 1e-300), np.zeros(len(x))):
                checks = qc.check_pattern(x, y, {"format": "Rigaku RAS"})
                self.assertIn(qc.overall(checks), (qc.PASS, qc.CAUTION, qc.NOT_EVALUATED))


class OverallTests(unittest.TestCase):
    def test_one_problem_is_not_outweighed(self):
        checks = [qc.Check("a", qc.PASS)] * 9 + [qc.Check("b", qc.WARNING)]
        self.assertEqual(qc.overall(checks), qc.WARNING)
        self.assertEqual(qc.overall(checks + [qc.Check("c", qc.FAIL)]), qc.FAIL)

    def test_a_required_check_not_evaluated_is_never_a_pass(self):
        self.assertEqual(qc.overall([qc.Check("a", qc.PASS), qc.Check("b", qc.NOT_EVALUATED, required=True)]),
                         qc.NOT_EVALUATED)
        self.assertEqual(qc.overall([qc.Check("a", qc.PASS), qc.Check("b", qc.NOT_EVALUATED)]), qc.PASS)
        self.assertEqual(qc.overall([]), qc.NOT_EVALUATED)


class ReferenceCheckTests(unittest.TestCase):
    def setUp(self):
        self.ns = regression.renderer_namespace()
        self.ns.update(qc=qc, hashlib=hashlib, platform=platform, warnings=warnings, tempfile=tempfile,
                       timezone=timezone, T={"pdf_error": "", "cif_error": ""})

    def fake_pdf(self, rows):
        page = SimpleNamespace(extract_tables=lambda: [rows])
        handle = SimpleNamespace(pages=[page], __enter__=None)

        class Opened:
            def __enter__(self):
                return handle

            def __exit__(self, *args):
                return False
        return SimpleNamespace(open=lambda *args, **kwargs: Opened())

    def test_pdf_card_rows_that_cannot_be_read_are_counted(self):
        rows = [["No.", "2θ", "d", "I", "hkl"], ["1", "28.44", "3.13", "100", "111"], ["2", "47.30", "1.92", "x", "220"],
                ["3", "56.12", "1.64", "30", "311", "4", "69.13", "1.36", "6", "400"]]
        self.ns.update(pdfplumber=self.fake_pdf(rows), PDFPLUMBER_AVAILABLE=True)
        report = self.ns["pdf_card_report"](b"card")
        self.assertEqual((report["angles"], report["skipped"]), ([28.44, 56.12, 69.13], 1))
        real = [node for node in ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "parse_pdf_card"]
        exec(compile(ast.Module(real, []), "app.py", "exec"), self.ns)     # the reader the sticks come from
        x, _ = self.ns["parse_pdf_card"](b"card", (5, 90))
        self.assertEqual(sorted(report["angles"]), x.tolist())
        checks = qc.check_pdf_card(3, 1, 2)
        self.assertEqual(statuses(checks), {"pdf_read": qc.WARNING, "in_range": qc.PASS})
        self.assertEqual(qc.overall(qc.check_pdf_card(0, 0, 0)), qc.FAIL)

    @unittest.skipUnless(__import__("importlib").util.find_spec("pymatgen"), "pymatgen is not installed")
    def test_cif_structures_and_failures(self):
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.cif import CifParser, CifWriter
        self.ns.update(CifParser=CifParser, PYMATGEN_AVAILABLE=True)
        si = str(CifWriter(Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])))
        one = self.ns["cif_report"](si.encode())
        self.assertEqual((one["error"], one["structures"]), ("", 1))
        two = self.ns["cif_report"]((si + "\n" + si.replace("data_Si", "data_Si_2")).encode())
        self.assertEqual(two["structures"], 2)
        self.assertEqual(statuses(qc.check_cif("", 2, [], 5))["cif_structures"], qc.CAUTION)
        broken = self.ns["cif_report"](b"data_x\n_cell_length_a 5\n")
        self.assertTrue(broken["error"])
        self.assertEqual(qc.overall(qc.check_cif(broken["error"], 0, [], 0)), qc.FAIL)


def run_app(xrd_bytes, extra=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(regression.ui_test_source(), default_timeout=120)
    state = {"authenticated": True, "lang_radio": "日本語", "_xrd_bytes": xrd_bytes, "_cif_bytes": {}, "_pdf_bytes": {}}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


class PanelTests(unittest.TestCase):
    def test_panel_warning_and_figure_record(self):
        x, y = synthetic_scan()
        bad = b"2theta Intensity\n" + xy_bytes(x[:50], y[:50]) + b"11.1 nan\n"
        files = {"clean.xy": xy_bytes(x, y, "2theta Intensity"), "bad.xy": bad, "broken.ras": b"*RAS_DATA_START\n"}
        at = run_app(files)
        self.assertFalse(at.exception, [e.message for e in at.exception])
        labels = [e.label for e in at.expander if "Data Reliability" in e.label]
        self.assertEqual(labels, ["⚠️ Data Reliability — FAIL 1 · WARNING 1 · PASS 1"])
        self.assertTrue(any("WARNING／FAIL" in w.value for w in at.warning))
        self.assertTrue(any("WARNING／FAIL" in w.value for w in at.sidebar.warning))
        tables = [m.value for m in at.markdown if m.value.startswith("| 項目 |")]
        self.assertEqual(len(tables), 3)
        self.assertIn("bad.xy", " ".join(m.value.replace("\\", "") for m in at.markdown))
        self.assertTrue(any("| 数値でない値（NaN・無限大） | WARNING |" in table for table in tables))
        # once: the notice is not repeated on the next run with the same results
        at.run()
        self.assertFalse(any("件のデータに" in w.value for w in at.warning))
        entries = at.session_state["_regression_report"]
        self.assertIn("png", entries)

    def test_figure_record_lists_inputs_checks_and_versions(self):
        x, y = synthetic_scan()
        files = {"clean.xy": xy_bytes(x, y, "2theta Intensity")}
        at = run_app(files, {"normalize": True})
        record = json.loads(at.session_state["_regression_report"]["figure_record"])
        self.assertEqual(record["format"], "xrd-maker-figure-record")
        measurement = record["measurements"][0]
        self.assertEqual(measurement["sha256"], hashlib.sha256(files["clean.xy"]).hexdigest())
        self.assertEqual(measurement["points_read"], len(x))
        self.assertAlmostEqual(measurement["normalized_by"], float(y[(x >= at.session_state["xrange"][0]) &
                                                                    (x <= at.session_state["xrange"][1])].max()), places=4)
        self.assertEqual(measurement["quality"]["overall"], qc.PASS)
        self.assertTrue(record["figure"]["normalized"])
        self.assertIn("matplotlib", record["environment"])
        self.assertTrue(record["app"]["version"])
        self.assertTrue(any("normalized" in step for step in record["processing"]))

    def test_thresholds_can_be_changed_in_the_panel(self):
        x = np.concatenate([np.arange(10, 30, 0.02), np.arange(30, 40, 0.021)])
        at = run_app({"joined.xy": xy_bytes(x, np.full(len(x), 5.0), "2theta Intensity")})
        self.assertIn("CAUTION", [e.label for e in at.expander if "Data Reliability" in e.label][0])
        at.number_input(key="qc_step_tolerance").set_value(10.0).run()
        self.assertIn("PASS 1", [e.label for e in at.expander if "Data Reliability" in e.label][0])
        self.assertEqual(at.session_state["_regression_report"]["session"]["settings"]["qc_step_tolerance"], 10.0)


if __name__ == "__main__":
    unittest.main()
