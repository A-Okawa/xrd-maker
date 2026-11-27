"""Checks of the data a figure is drawn from (the Data Reliability panel and the figure record).

Nothing here changes the data: the checks read what xrd_io recorded while reading a file and the
arrays as drawn, and report. Each check has a status:

PASS           meets the criterion
CAUTION        worth knowing; the figure is drawn as usual
WARNING        check the data before using the figure
FAIL           could not be read or used
NOT EVALUATED  the file does not say enough to check this

A file's overall status is the most serious status among its checks, so one problem is never
outweighed by checks that pass. A required check that could not be evaluated keeps the overall
status from being PASS. Codes and numbers are returned; the app words them in the screen language.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PASS, CAUTION, WARNING, FAIL, NOT_EVALUATED = "PASS", "CAUTION", "WARNING", "FAIL", "NOT EVALUATED"
STATUSES = (FAIL, WARNING, CAUTION, PASS, NOT_EVALUATED)
_SEVERITY = {PASS: 0, CAUTION: 1, WARNING: 2, FAIL: 3}

# Provisional thresholds (the user can change both in the panel). Not established criteria:
# step uniformity matters here because peak detection converts its minimum spacing with the mean step.
DEFAULT_STEP_TOLERANCE_PERCENT = 1.0
DEFAULT_MIN_POINTS = 10

# 2θ is an angle between the incident and the diffracted beam.
ANGLE_LIMITS = (0.0, 180.0)
# CIF patterns are calculated with pymatgen's "CuKa" (weighted Kα1/Kα2, 1.54184 Å); a measurement whose
# Kα1 differs from Cu Kα1 (1.54056 Å) by more than this puts the reference peaks at other angles.
CU_KALPHA1_A = 1.54056
CIF_WAVELENGTH_A = 1.54184
WAVELENGTH_TOLERANCE_A = 0.002
MAX_LISTED = 10            # line numbers or angle runs listed in a message


@dataclass
class Check:
    code: str
    status: str
    values: dict = field(default_factory=dict)
    required: bool = False


def overall(checks) -> str:
    """The most serious status; NOT EVALUATED when nothing, or a required check, could be evaluated."""
    evaluated = [check.status for check in checks if check.status != NOT_EVALUATED]
    if not evaluated:
        return NOT_EVALUATED
    worst = max(evaluated, key=_SEVERITY.__getitem__)
    if worst == PASS and any(check.required and check.status == NOT_EVALUATED for check in checks):
        return NOT_EVALUATED
    return worst


def _runs(values):
    return ", ".join(f"{a:.3f}–{b:.3f}" if a != b else f"{a:.3f}" for a, b in values[:MAX_LISTED]) + (
        " …" if len(values) > MAX_LISTED else "")


def is_cu(metadata) -> bool | None:
    """Whether the measurement used Cu Kα, from the wavelength or anode the file states (None: not stated)."""
    wavelength = metadata.get("wavelength_alpha1_A")
    if wavelength is not None:
        return abs(wavelength - CU_KALPHA1_A) <= WAVELENGTH_TOLERANCE_A
    anode = str(metadata.get("anode", "")).strip().lower()
    if anode:
        return anode in ("cu", "copper") or anode.startswith("cu ")
    return None


def check_pattern(x, y, metadata, *, references_shown=False,
                  step_tolerance_percent=DEFAULT_STEP_TOLERANCE_PERCENT, min_points=DEFAULT_MIN_POINTS):
    """The checks of one measured scan as read (x sorted by angle, y as stored)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    checks = []
    text = metadata.get("format") == "XY/CSV text"

    dropped = list(metadata.get("rows_dropped_nonfinite", []))
    checks.append(Check("finite", WARNING if dropped else PASS,
                        {"count": len(dropped), "rows": ", ".join(map(str, dropped[:MAX_LISTED])) + (" …" if len(dropped) > MAX_LISTED else "")},
                        required=True))

    if text:
        ignored = int(metadata.get("columns_not_read", 0))
        checks.append(Check("columns", CAUTION if ignored else PASS, {"count": ignored}))
        checks.append(Check("axis", PASS if metadata.get("axis_named_in_file") else NOT_EVALUATED))
    else:
        checks.append(Check("axis", PASS))

    if metadata.get("attenuator_column_present"):
        count = int(metadata.get("attenuator_points_not_one", 0))
        low, high = metadata.get("attenuator_factor_range", [1, 1])
        checks.append(Check("attenuator", WARNING if count else PASS,
                            {"count": count, "runs": _runs(metadata.get("attenuator_angle_runs", [])),
                             "low": low, "high": high}))

    outside = (x < ANGLE_LIMITS[0]) | (x > ANGLE_LIMITS[1])
    checks.append(Check("angle_range", WARNING if outside.any() else PASS,
                        {"count": int(outside.sum()), "low": float(x.min()), "high": float(x.max())}, required=True))

    duplicates = int(metadata.get("duplicate_angles", 0))
    reordered = bool(metadata.get("points_reordered", False))
    checks.append(Check("order", CAUTION if duplicates or reordered else PASS,
                        {"duplicates": duplicates, "reordered": reordered}))

    negative = int(np.count_nonzero(y < 0))
    checks.append(Check("negative", CAUTION if negative else PASS, {"count": negative, "low": float(y.min())}))

    checks.append(Check("points", CAUTION if len(x) < min_points else PASS,
                        {"count": int(len(x)), "minimum": int(min_points)}, required=True))

    steps = np.diff(x)
    steps = steps[steps > 0]
    if len(steps) < 2:
        checks.append(Check("step", NOT_EVALUATED, {"tolerance": step_tolerance_percent}))
    else:
        median = float(np.median(steps))
        deviation = np.abs(steps - median) / median * 100
        uneven = np.nonzero(deviation > step_tolerance_percent)[0]
        positive = np.nonzero(np.diff(x) > 0)[0]
        checks.append(Check("step", CAUTION if len(uneven) else PASS,
                            {"median": median, "count": int(len(uneven)), "largest": float(deviation.max()),
                             "first": float(x[positive[uneven[0]]]) if len(uneven) else None,
                             "tolerance": step_tolerance_percent}))

    cu = is_cu(metadata)
    stated = metadata.get("wavelength_alpha1_A")
    values = {"wavelength": f"{stated:.5f} Å" if stated is not None else "", "anode": metadata.get("anode", ""),
              "cif": CIF_WAVELENGTH_A}
    if cu is None:
        checks.append(Check("wavelength", NOT_EVALUATED, values))
    else:
        checks.append(Check("wavelength", WARNING if (not cu and references_shown) else PASS, values))
    return checks


def check_display(y_in_range, normalize):
    """The scan as shown: points in the 2θ range, and a maximum to normalise by.

    The app divides by max(maximum, 1e-9) when normalising; a maximum of 0 or below would give values
    that mean nothing, so that is reported instead of being drawn silently.
    """
    y = np.asarray(y_in_range, dtype=float)
    if not len(y):
        return Check("display", CAUTION, {"count": 0, "high": None})
    high = float(y.max())
    status = WARNING if (normalize and high <= 0) else PASS
    return Check("display", status, {"count": int(len(y)), "high": high})


def check_pdf_card(total, skipped, in_range):
    """A PDF card: how many peaks its table gave, how many table rows could not be read."""
    if total == 0:
        return [Check("pdf_read", FAIL, {"total": 0, "skipped": skipped})]
    checks = [Check("pdf_read", WARNING if skipped else PASS, {"total": total, "skipped": skipped})]
    checks.append(Check("in_range", CAUTION if in_range == 0 else PASS, {"count": in_range, "total": total}))
    return checks


def check_cif(error, structures, warnings, in_range):
    """A CIF: whether pymatgen read it, which structure is used, its warnings, peaks in the 2θ range."""
    if error:
        return [Check("cif_read", FAIL, {"error": error})]
    checks = [Check("cif_read", PASS, {"cif": CIF_WAVELENGTH_A})]
    checks.append(Check("cif_structures", CAUTION if structures > 1 else PASS, {"count": structures}))
    checks.append(Check("cif_warnings", CAUTION if warnings else PASS,
                        {"count": len(warnings), "messages": " / ".join(warnings[:3])[:300]}))
    checks.append(Check("in_range", CAUTION if in_range == 0 else PASS, {"count": in_range, "total": in_range}))
    return checks


def summary(statuses) -> dict:
    """How many files have each overall status, most serious first (statuses that occur only)."""
    counts = {status: list(statuses).count(status) for status in STATUSES}
    return {status: count for status, count in counts.items() if count}
