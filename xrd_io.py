"""Offline, scan-preserving readers for 1D powder diffraction measurements.

Native-format layouts were checked against the public xylib format descriptions,
GSAS-II importers and FAIRmat/IKZ readers. This is an independent implementation;
no third-party parser code is bundled. Format references:
https://github.com/wojdyr/xylib/tree/master/xylib
https://github.com/AdvancedPhotonSource/GSAS-II/tree/main/GSASII/imports
https://github.com/FAIRmat-NFDI/readers-xrd
https://github.com/scattering/scattio/blob/master/python/scattio/bruker.py
Bruker GADDS reference M86-E01008 4/04, appendices B and C (esrf.fr).

Intensities are returned as stored: no normalization, clipping, absorption
correction or counts/time conversion is applied. Unsupported native variants
raise XRDParseError instead of guessing a column or a binary data offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np


SUPPORTED_EXTENSIONS = ("xy", "txt", "csv", "dat", "ras", "rasx", "raw", "uxd", "brml")
_MAX_POINTS = 10_000_000
_MAX_SCANS = 10_000
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_EXPORT_HELP = "Export the measurement as Rigaku RAS/RASX, Bruker UXD/BRML, or two-column XY/CSV."


class XRDParseError(ValueError):
    """The measurement is damaged, ambiguous, or uses an unsupported format."""


@dataclass
class XRDPattern:
    name: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        raise XRDParseError("Binary data was found where a text measurement was expected. " + _EXPORT_HELP)
    for encoding in ("utf-8-sig", "cp932", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise XRDParseError("The text encoding could not be read. Save a UTF-8 XY/CSV export.")


def _number(value: str) -> float:
    return float(value.strip().replace("D", "E").replace("d", "e"))


def _values(line: str) -> list[float]:
    try:
        return [_number(v) for v in re.split(r"[\s,;]+", line.strip()) if v]
    except ValueError as exc:
        raise XRDParseError("A measurement row contains a nonnumeric value.") from exc


def _pattern(filename: str, x, y, metadata: dict, index: int = 1) -> XRDPattern:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise XRDParseError("A scan has missing or mismatched angle/intensity values.")
    if len(x) > _MAX_POINTS:
        raise XRDParseError("The scan contains too many points.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise XRDParseError("A scan contains NaN or infinite angle/intensity values.")
    order = np.argsort(x, kind="stable")
    meta = {"source_file": Path(filename).name, "scan_index": index,
            "x_axis": "2theta", "x_unit": "deg", "intensity_transform": "none", **metadata}
    return XRDPattern(f"{Path(filename).stem} [scan {index}]", x[order], y[order], meta)


def _axis_is_2theta(value: str) -> bool:
    value = re.sub(r"[^a-z0-9]", "", value.lower().replace("θ", "theta"))
    return value in {"2theta", "twotheta", "2thetatheta", "twothetatheta",
                     "2thetaomega", "twothetaomega", "coupled", "lockedcoupled",
                     "unlockedcoupled", "detectorscan"}


def _check_axis(value: str | None):
    if not value:
        raise XRDParseError("The native measurement does not identify its scan axis. " + _EXPORT_HELP)
    if not _axis_is_2theta(value):
        raise XRDParseError(f"The scan axis '{value}' is not a supported 2theta powder scan. " + _EXPORT_HELP)


def _check_degree_unit(value: str | None):
    if value and value.strip().lower() not in {"deg", "degree", "degrees", "°"}:
        raise XRDParseError(f"The angle unit '{value}' is unsupported; export angles in degrees.")


def _natural(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


_COMMENT_MARKERS = ("#", ";", "'", "//")
# Columns that are not the measurement: calculated profile, background, difference, errors, weights.
_NOT_MEASURED = re.compile(r"calc|^[yi]?cal$|bkg|back|diff|esd|sigma|err|weight|^w$|resid|^fit|model")
_NOT_MEASURED_JA = ("計算", "バックグラウンド", "差", "誤差")


def _split_fields(line: str) -> list[str]:
    if "," in line or ";" in line:
        return next(csv.reader([line], delimiter=";" if ";" in line else ","))
    return line.split()


def _column_role(name: str, position: int) -> str | None:
    """'angle' or 'intensity' for a column name such as 2theta, twotheta, Angle / yobs, Iobs, Intensity, Counts.

    Rietveld and profile programs write the measurement next to calculated, background and
    difference profiles (twotheta, yobs, ycal, bkg, diff); those are never taken as the measurement.
    """
    text = name.strip().lower()
    key = re.sub(r"[^a-z0-9]", "", text.replace("θ", "theta"))
    if _NOT_MEASURED.search(key) or any(word in text for word in _NOT_MEASURED_JA):
        return None
    if ("2theta" in key or "twotheta" in key or key.startswith(("2th", "tth", "angle", "pos2th"))
            or "角度" in text or (key == "x" and position == 0)):
        return "angle"
    if (key in {"y", "i", "obs", "int"} or key.startswith(("yobs", "iobs", "observed", "intensit", "count", "cps"))
            or "強度" in text or "カウント" in text):
        return "intensity"
    return None


def _named_columns(header: list[str] | None, width: int) -> tuple[int, int] | None:
    """Positions of the angle and the measured intensity when a header names exactly one of each."""
    if not header or len(header) != width:
        return None
    roles = [_column_role(name, position) for position, name in enumerate(header)]
    if roles.count("angle") == 1 and roles.count("intensity") == 1:
        return roles.index("angle"), roles.index("intensity")
    return None


def _read_text(data: bytes, filename: str) -> list[XRDPattern]:
    rows = []
    started = False
    ignored_header = False
    header = None              # the last line of names before the data (a '#' comment line counts)
    columns = None             # (angle, intensity) chosen by the header names
    width = 0
    for lineno, original in enumerate(_text(data).splitlines(), 1):
        line = original.strip()
        if not line:
            continue
        if line.startswith(_COMMENT_MARKERS):
            if not started:
                header = [name.strip() for name in _split_fields(line.lstrip("#;'/").strip())]
            continue
        # Comment tails are permitted, but arbitrary suffixes are not stripped
        # from numeric tokens ("12counts" must never silently become 12).
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        fields = _split_fields(line)
        try:
            vals = [_number(v) for v in fields]
        except ValueError:
            axis_header = bool(fields) and any(word in fields[0].lower().replace("θ", "theta") for word in ("theta", "angle"))
            if started or (not axis_header and any(re.match(r"^[+\-]?(?:\d|\.\d)", v.strip()) for v in fields)):
                raise XRDParseError(f"Invalid numeric row at line {lineno}.")
            ignored_header = True
            header = [name.strip() for name in fields]
            continue
        if len(vals) < 2:
            raise XRDParseError(f"Line {lineno} has only one column; two-column angle/intensity data is required.")
        if not started:
            width = len(vals)
            columns = _named_columns(header, width)
            if columns is None and width > 3:
                names = f" ({', '.join(header)})" if header and len(header) == width else ""
                raise XRDParseError(
                    f"Line {lineno} has {width} numeric columns{names}. Name the angle column (2theta) and the "
                    "measured intensity column (yobs, Intensity or Counts) in a header, or export only those two columns.")
        elif columns is not None and len(vals) != width:
            raise XRDParseError(f"Line {lineno} has {len(vals)} columns; the header names {width}.")
        if columns is None and len(vals) > 3:
            raise XRDParseError(f"Line {lineno} has more than three numeric columns; export explicit angle/intensity columns.")
        started = True
        angle, intensity = columns or (0, 1)
        if np.isfinite([vals[angle], vals[intensity]]).all():
            rows.append([vals[angle], vals[intensity]])
    if not rows:
        raise XRDParseError("No finite two-column angle/intensity measurements were found.")
    arr = np.asarray(rows)
    metadata = {"format": "XY/CSV text", "text_header_present": ignored_header or columns is not None}
    if columns is not None:
        metadata["columns"] = {"angle": header[columns[0]], "intensity": header[columns[1]]}
    return [_pattern(filename, arr[:, 0], arr[:, 1], metadata)]


def _header_value(line: str) -> tuple[str, str]:
    match = re.match(r"^[*_]([^\s=]+)\s*(?:=\s*)?(.*)$", line)
    if not match:
        raise XRDParseError("Malformed native text header.")
    return match[1].upper(), match[2].strip().strip("\"'")


def _read_ras(data: bytes, filename: str) -> list[XRDPattern]:
    patterns, header, rows = [], {}, []
    in_data = False
    for lineno, original in enumerate(_text(data).splitlines(), 1):
        line = original.strip()
        if not line:
            continue
        if line == "*RAS_HEADER_START":
            if in_data:
                raise XRDParseError("RAS scan data ended without *RAS_INT_END.")
            header = {}
        elif line == "*RAS_INT_START":
            if in_data:
                raise XRDParseError("Nested RAS intensity blocks are invalid.")
            rows, in_data = [], True
        elif line == "*RAS_INT_END":
            if not in_data:
                raise XRDParseError("RAS intensity block has no start marker.")
            axis = header.get("MEAS_SCAN_AXIS_X_INTERNAL", header.get("MEAS_SCAN_AXIS_X"))
            _check_axis(axis)
            _check_degree_unit(header.get("MEAS_SCAN_UNIT_X"))
            expected = header.get("MEAS_DATA_COUNT")
            if expected and int(expected) != len(rows):
                raise XRDParseError("RAS point count does not match the recorded data (possibly truncated).")
            arr = np.asarray(rows)
            if not len(arr):
                raise XRDParseError("The RAS scan contains no measurement rows.")
            patterns.append(_pattern(filename, arr[:, 0], arr[:, 1],
                {"format": "Rigaku RAS", "scan_axis": axis, "vendor_header": dict(header),
                 "intensity_unit": header.get("MEAS_SCAN_UNIT_Y", "as stored"),
                 "attenuator_column_applied": False}, len(patterns) + 1))
            in_data = False
        elif line.startswith("*"):
            if in_data:
                raise XRDParseError(f"Unexpected RAS header inside intensity data at line {lineno}.")
            if not line.startswith(("*RAS_DATA_", "*RAS_HEADER_")):
                key, value = _header_value(line)
                header[key] = value
        elif in_data:
            vals = _values(line)
            if len(vals) not in (2, 3):
                raise XRDParseError(f"RAS line {lineno} must contain angle, intensity and optional attenuator.")
            rows.append(vals[:2])
        else:
            raise XRDParseError(f"Unexpected text outside a RAS scan at line {lineno}.")
    if in_data:
        raise XRDParseError("RAS scan is truncated: missing *RAS_INT_END.")
    if not patterns:
        raise XRDParseError("No *RAS_INT_START measurement blocks were found.")
    return patterns


def _read_rigaku_dat(data: bytes, filename: str) -> list[XRDPattern]:
    patterns, global_header, header, counts = [], {}, None, []
    reading = False
    for original in _text(data).splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line == "*BEGIN":
            if header is not None:
                raise XRDParseError("Nested Rigaku DAT groups are invalid.")
            header, counts, reading = {}, [], False
        elif line == "*END":
            if header is None:
                raise XRDParseError("Rigaku DAT *END has no matching *BEGIN.")
            try:
                n, start, step = int(header["COUNT"]), _number(header["START"]), _number(header["STEP"])
            except (KeyError, ValueError) as exc:
                raise XRDParseError("Rigaku DAT requires COUNT, START and STEP in each group.") from exc
            if n != len(counts):
                raise XRDParseError("Rigaku DAT intensity count does not match COUNT.")
            if n > 1 and step == 0:
                raise XRDParseError("Rigaku DAT STEP must be nonzero.")
            patterns.append(_pattern(filename, start + step * np.arange(n), counts,
                {"format": "Rigaku DAT", "vendor_header": {**global_header, **header}}, len(patterns) + 1))
            header, reading = None, False
        elif line == "*EOF":
            if header is not None:
                raise XRDParseError("Rigaku DAT group is missing *END.")
            break
        elif line.startswith("*"):
            key, value = _header_value(line)
            (global_header if header is None else header)[key] = value
            if key == "COUNT" and header is not None:
                reading = True
        else:
            if header is None or not reading:
                raise XRDParseError("Rigaku DAT intensities must follow *COUNT inside a group.")
            counts.extend(_values(line))
    if header is not None:
        raise XRDParseError("Rigaku DAT group is truncated (missing *END).")
    try:
        expected = int(global_header["GROUP_COUNT"])
    except (KeyError, ValueError) as exc:
        raise XRDParseError("Rigaku DAT requires GROUP_COUNT.") from exc
    if expected != len(patterns) or not patterns:
        raise XRDParseError("Rigaku DAT GROUP_COUNT does not match its measurement groups.")
    return patterns


def _read_uxd(data: bytes, filename: str) -> list[XRDPattern]:
    patterns, global_header, header, values = [], {}, None, []
    mode = None

    def finish():
        nonlocal values, mode
        if mode is None:
            return
        if header is None or not values:
            raise XRDParseError("UXD contains an empty measurement block.")
        _check_axis(header.get("DRIVE"))
        if mode in ("COUNTS", "COUNT", "CPS"):
            try:
                start, step = _number(header["START"]), _number(header["STEPSIZE"])
            except (KeyError, ValueError) as exc:
                raise XRDParseError("UXD counts-only data requires START and STEPSIZE.") from exc
            if len(values) > 1 and step == 0:
                raise XRDParseError("UXD STEPSIZE must be nonzero.")
            x, y = start + step * np.arange(len(values)), values
        else:
            width = 3 if mode == "2THETACOUNTSTIME" else 2
            if len(values) % width:
                raise XRDParseError("UXD angle/intensity rows contain an incomplete record.")
            arr = np.asarray(values).reshape(-1, width)
            x, y = arr[:, 0], arr[:, 1]
        patterns.append(_pattern(filename, x, y, {"format": "Bruker UXD",
            "intensity_unit": "cps" if "CPS" in mode else "counts",
            "vendor_header": {**global_header, **header}}, len(patterns) + 1))
        values, mode = [], None

    for original in _text(data).splitlines():
        line = original.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("_"):
            key, value = _header_value(line)
            if key == "DRIVE":
                finish()
                header = {"DRIVE": value}
            elif key in {"COUNTS", "COUNT", "CPS", "2THETACOUNTS", "2THETACPS", "2THETACOUNTSTIME"}:
                finish()
                if header is None:
                    raise XRDParseError("UXD measurement block is missing _DRIVE.")
                mode = key
            elif key.startswith(("D-I", "2THETA-I")):
                raise XRDParseError("UXD peak lists are not continuous measurement scans. " + _EXPORT_HELP)
            else:
                if mode is not None:
                    finish()
                (global_header if header is None else header)[key] = value
        else:
            if mode is None:
                raise XRDParseError("UXD numeric data is missing a recognized intensity keyword.")
            values.extend(_values(line.split(";", 1)[0]))
    finish()
    if not patterns:
        raise XRDParseError("No UXD measurement scans were found.")
    return patterns


class _Binary:
    def __init__(self, data: bytes):
        self.data = data

    def need(self, offset: int, length: int):
        if offset < 0 or length < 0 or offset + length > len(self.data):
            raise XRDParseError("The native RAW file is truncated or declares an invalid block size.")

    def get(self, fmt: str, offset: int):
        self.need(offset, struct.calcsize("<" + fmt))
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def string(self, offset: int, length: int):
        self.need(offset, length)
        return self.data[offset:offset + length].split(b"\0", 1)[0].decode("latin-1").strip()

    def counts(self, offset: int, n: int):
        if not 0 < n <= _MAX_POINTS:
            raise XRDParseError("RAW has an invalid or excessive measurement point count.")
        self.need(offset, n * 4)
        return np.frombuffer(self.data, dtype="<f4", count=n, offset=offset).astype(float)


def _stepped_pattern(filename, start, step, y, metadata, index):
    if not np.isfinite([start, step]).all() or (len(y) > 1 and step == 0):
        raise XRDParseError("RAW has an invalid angle start or step.")
    return _pattern(filename, start + step * np.arange(len(y)), y, metadata, index)


def _read_raw(data: bytes, filename: str) -> list[XRDPattern]:
    b, patterns = _Binary(data), []
    if data.startswith(b"RAW "):
        pos, following = 4, 1
        while following:
            if len(patterns) >= _MAX_SCANS:
                raise XRDParseError("RAW has an excessive number of scans.")
            if data[pos:pos + 4] == b"RAW ":
                pos += 4
            n = b.get("I", pos)
            flags = b.get("I", pos + 12)
            if flags & (8192 | 16384):
                raise XRDParseError("RAW v1 contains a rocking/chi/phi scan rather than a 2theta scan. " + _EXPORT_HELP)
            start, step = b.get("f", pos + 20), b.get("f", pos + 8)
            following = b.get("I", pos + 148)
            y = b.counts(pos + 152, n)
            patterns.append(_stepped_pattern(filename, start, step, y,
                {"format": "Bruker RAW v1", "sample_name": b.string(pos + 36, 32),
                 "scan_mode": flags, "time_per_step": b.get("f", pos + 4)}, len(patterns) + 1))
            pos += 152 + n * 4
    elif data.startswith(b"RAW2"):
        # Its range header identifies configurable motor numbers, not physical
        # axes. Without the instrument's motor mapping, even an otherwise valid
        # RAW2 range cannot safely be labeled as 2theta.
        raise XRDParseError("Bruker RAW v2 is recognized, but its configurable motor IDs do not establish a safe 2theta axis. Export this scan as Bruker UXD/BRML or two-column XY/CSV.")
    elif data.startswith(b"RAW1.01"):
        n_scans, pos = b.get("I", 12), 712
        if not 0 < n_scans <= _MAX_SCANS:
            raise XRDParseError("RAW v3 has an invalid scan count.")
        for i in range(n_scans):
            header, n = b.get("I", pos), b.get("I", pos + 4)
            if header != 304:
                raise XRDParseError("Unsupported RAW v3 range header size. " + _EXPORT_HELP)
            scan_type = b.get("I", pos + 196)
            if scan_type not in (0, 1, 2):
                raise XRDParseError("RAW v3 contains a non-powder or unsupported scan type. " + _EXPORT_HELP)
            varying, record_length = b.get("I", pos + 248), b.get("I", pos + 252)
            if varying != 0 or record_length != 4:
                raise XRDParseError("RAW v3 has additional per-point parameters; this layout is unsupported. " + _EXPORT_HELP)
            extra = b.get("I", pos + 256)
            y = b.counts(pos + header + extra, n)
            patterns.append(_stepped_pattern(filename, b.get("d", pos + 16), b.get("d", pos + 176), y,
                {"format": "Bruker RAW v3", "sample_name": b.string(326, 60),
                 "scan_type": scan_type, "time_per_step": b.get("f", pos + 192)}, i + 1))
            pos += header + extra + n * 4
    elif data.startswith(b"RAW4.00"):
        pos = 61
        while pos < len(data):
            kind = b.get("I", pos)
            if kind not in (0, 160):
                length = b.get("I", pos + 4)
                if length < 8:
                    raise XRDParseError("RAW v4 has an invalid metadata segment length.")
                b.need(pos, length)
                pos += length
                continue
            b.need(pos, 160)
            scan_type = b.string(pos + 32, 24)
            _check_axis(scan_type)
            if scan_type not in {"Locked Coupled", "Unlocked Coupled", "Detector Scan"}:
                raise XRDParseError("Unsupported RAW v4 scan type. " + _EXPORT_HELP)
            n = b.get("I", pos + 88)
            datum_size, extra = b.get("I", pos + 136), b.get("I", pos + 140)
            if datum_size != 4:
                raise XRDParseError(f"RAW v4 uses {datum_size}-byte records with additional per-point fields; this variant is unsupported. " + _EXPORT_HELP)
            b.need(pos + 160, extra)
            # Extended metadata uses a sequence of length-delimited segments.
            # HRXRD (type 300) can change the meaning of per-point axes.
            ext, end = pos + 160, pos + 160 + extra
            while ext < end:
                typ, size = b.get("I", ext), b.get("I", ext + 4)
                if size < 8 or ext + size > end:
                    raise XRDParseError("RAW v4 extended header is malformed.")
                if typ == 300:
                    raise XRDParseError("RAW v4 HRXRD extended scans are unsupported. " + _EXPORT_HELP)
                ext += size
            y = b.counts(end, n)
            patterns.append(_stepped_pattern(filename, b.get("d", pos + 72), b.get("d", pos + 80), y,
                {"format": "Bruker RAW v4", "scan_type": scan_type,
                 "time_per_step": b.get("f", pos + 92)}, len(patterns) + 1))
            pos = end + n * 4
            if len(patterns) > _MAX_SCANS:
                raise XRDParseError("RAW has an excessive number of scans.")
    else:
        raise XRDParseError("Unknown RAW signature. Bruker RAW v1/v2/v3/v4 and Rigaku text/archive formats are recognized; arbitrary Rigaku binary RAW is not. " + _EXPORT_HELP)
    if not patterns:
        raise XRDParseError("No supported RAW measurement scans were found.")
    return patterns


def _xml(data: bytes):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise XRDParseError("XML declarations with external entities are unsupported.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise XRDParseError("The native measurement XML is invalid or truncated.") from exc
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    return root


def _read_archive(data: bytes, filename: str, kind: str) -> list[XRDPattern]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise XRDParseError(f"{kind.upper()} must be a valid ZIP measurement archive.") from exc
    with archive as z:
        files = z.infolist()
        if sum(f.file_size for f in files) > _MAX_ARCHIVE_BYTES:
            raise XRDParseError("The measurement archive is too large after decompression.")
        if any(f.flag_bits & 1 for f in files):
            raise XRDParseError("Encrypted measurement archives are unsupported.")
        if kind == "rasx":
            return _read_rasx(z, filename)
        return _read_brml(z, filename)


def _read_rasx(z: zipfile.ZipFile, filename: str) -> list[XRDPattern]:
    names = z.namelist()
    profiles = []
    if "root.xml" in names:
        root = _xml(z.read("root.xml"))
        for group in root:
            if group.get("Type", "").lower() != "profile":
                continue
            for child in group:
                name = child.get("Name", "")
                if re.fullmatch(r"Profile\d+\.txt", name, re.I):
                    profiles.append(f"{group.tag}/{name}")
    if not profiles:
        profiles = sorted([n for n in names if re.search(r"(?:^|/)Profile\d+\.txt$", n, re.I)], key=_natural)
    if not profiles or len(profiles) > _MAX_SCANS:
        raise XRDParseError("RASX contains no supported 1D profiles, or too many scans.")
    patterns = []
    for path in profiles:
        match = re.search(r"Profile(\d+)\.txt$", path, re.I)
        folder = str(PurePosixPath(path).parent)
        prefix = "" if folder == "." else folder + "/"
        companions = [prefix + key + match[1] + ".xml" for key in ("MesurementConditions", "MeasurementConditions")]
        companion = next((n for n in companions if n in names), None)
        if path not in names or companion is None:
            raise XRDParseError("RASX is missing a profile or its measurement conditions XML.")
        root = _xml(z.read(companion))
        scan = root.find("ScanInformation")
        if scan is None:
            raise XRDParseError("RASX is missing ScanInformation.")
        meta = {child.tag: (child.text or "").strip() for child in scan}
        _check_axis(meta.get("AxisName"))
        _check_degree_unit(meta.get("PositionUnit"))
        rows = [_values(line) for line in _text(z.read(path)).splitlines() if line.strip()]
        if not rows or any(len(row) not in (2, 3) for row in rows):
            raise XRDParseError("RASX profiles must contain angle/intensity rows with optional attenuator.")
        expected = meta.get("DataCount")
        if expected and int(expected) != len(rows):
            raise XRDParseError("RASX profile length does not match DataCount.")
        arr = np.asarray([row[:2] for row in rows])
        patterns.append(_pattern(filename, arr[:, 0], arr[:, 1],
            {"format": "Rigaku RASX", "archive_member": path, "scan_axis": meta.get("AxisName"),
             "sample_name": root.findtext("GeneralInformation/SampleName", ""),
             "intensity_unit": meta.get("IntensityUnit", "as stored"), "vendor_header": meta,
             "attenuator_column_applied": False}, len(patterns) + 1))
    return patterns


def _read_brml(z: zipfile.ZipFile, filename: str) -> list[XRDPattern]:
    members = sorted([n for n in z.namelist() if re.search(r"(?:^|/)RawData\d+\.xml$", n, re.I)], key=_natural)
    if not members or len(members) > _MAX_SCANS:
        raise XRDParseError("BRML contains no RawData measurement XML, or too many scans.")
    patterns = []
    for member in members:
        root = _xml(z.read(member))
        for route in root.findall("./DataRoutes/DataRoute"):
            datums = route.findall("Datum")
            if not datums:
                continue
            rows = []
            for datum in datums:
                tokens = (datum.text or "").split(",")
                if any(not token.strip() for token in tokens):
                    raise XRDParseError("BRML Datum contains an empty numeric field.")
                rows.append([_number(token) for token in tokens])
            if not rows or len({len(row) for row in rows}) != 1:
                raise XRDParseError("BRML Datum records have inconsistent column lengths.")
            arr = np.asarray(rows)
            scan = route.find("ScanInformation")
            if scan is None:
                raise XRDParseError("BRML is missing ScanInformation.")
            count_text = scan.findtext("MeasurementPoints")
            if count_text and int(count_text) != len(arr):
                raise XRDParseError("BRML MeasurementPoints does not match the Datum records.")
            theta_axes = [a for a in scan.findall("./ScanAxes/ScanAxisInfo")
                          if _axis_is_2theta(a.get("AxisId", a.get("AxisName", "")))]
            for axis in theta_axes:
                unit = axis.find("Unit")
                _check_degree_unit(unit.get("Base") if unit is not None else axis.get("Unit"))
            counters, x_column = [], None
            for view in route.findall("./DataViews/RawDataView"):
                start, length = int(view.get("Start", "-1")), int(view.get("Length", "0"))
                if start < 0 or length < 1 or start + length > arr.shape[1]:
                    raise XRDParseError("BRML DataView points outside its Datum record.")
                recording = view.find("Recording")
                label = view.get("LogicName", "") if recording is None else recording.get("LogicName", "")
                category = "" if recording is None else recording.get("Category", "")
                if "counter" in label.lower() or category.lower() == "count" or label.lower() in ("intensity", "counts"):
                    counters.append((start, length, label))
                if _axis_is_2theta(label):
                    if length != 1:
                        raise XRDParseError("BRML has an unsupported vector angle channel.")
                    x_column = start
                    _check_degree_unit(view.get("Unit"))
                # VaryingRawDataView describes each field of a scan-axis vector.
                for varying in view.findall("Varying"):
                    fields = varying.findall("FieldDefinitions")
                    for offset, field in enumerate(fields):
                        axis = field.get("AxisId", field.get("FieldName", ""))
                        if _axis_is_2theta(axis):
                            if offset >= length:
                                raise XRDParseError("BRML scan-axis field exceeds its DataView.")
                            x_column = start + offset
                            restriction = field.find("Restriction")
                            if restriction is not None:
                                _check_degree_unit(restriction.get("Unit"))
            if not counters:
                raise XRDParseError("BRML does not identify an intensity counter in DataViews; guessing numeric columns is unsafe. " + _EXPORT_HELP)
            for start, width, label in counters:
                if width != 1:
                    raise XRDParseError("BRML detector-array/vector records are unsupported; export an integrated 1D XY scan.")
                if x_column is not None:
                    x = arr[:, x_column]
                else:
                    if len(theta_axes) != 1:
                        raise XRDParseError("BRML has no unambiguous 2theta scan axis. " + _EXPORT_HELP)
                    axis = theta_axes[0]
                    begin = _number(axis.findtext("Start", "nan")) + _number(axis.findtext("Reference", "0"))
                    step = _number(axis.findtext("Increment", "nan"))
                    if len(arr) > 1 and step == 0:
                        raise XRDParseError("BRML 2theta increment is zero.")
                    x = begin + step * np.arange(len(arr))
                if len(x) > 1 and np.ptp(x) == 0:
                    raise XRDParseError("BRML 2theta remains fixed; a rocking curve cannot be plotted as a 2theta scan. " + _EXPORT_HELP)
                patterns.append(_pattern(filename, x, arr[:, start],
                    {"format": "Bruker BRML", "archive_member": member, "counter": label,
                     "scan_type": scan.get("ScanName", ""), "intensity_unit": "as stored",
                     "time_per_step": scan.findtext("TimePerStep")}, len(patterns) + 1))
    if not patterns:
        raise XRDParseError("BRML contains no supported 1D measurement scans.")
    return patterns


def read_patterns(data: bytes, filename: str = "measurement.xy") -> list[XRDPattern]:
    """Read each measured scan separately, keeping stored intensities unchanged.

    Filename extensions route text/archive readers; binary Bruker RAW magic is
    recognized regardless of extension. Unknown .raw binaries are never treated
    as generic text. Native non-2theta, image and ambiguous detector layouts are
    rejected with an export suggestion.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise XRDParseError("The measurement file is empty or is not byte data.")
    data = bytes(data)
    suffix = Path(filename).suffix.lower()
    try:
        if data.startswith((b"RAW ", b"RAW2", b"RAW1.01", b"RAW4.00")):
            return _read_raw(data, filename)
        if suffix in (".rasx", ".brml"):
            return _read_archive(data, filename, suffix[1:])
        # Known vendor text signatures take precedence over a loose .txt/.dat extension.
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            prefix = _text(data)[:8192]
        else:
            prefix = _text(data[: min(len(data), 8192)]) if b"\0" not in data[:8192] else ""
        if "*RAS_DATA_START" in prefix or "*RAS_HEADER_START" in prefix or suffix == ".ras":
            return _read_ras(data, filename)
        if re.search(r"(?m)^\s*\*TYPE\s*(?:=|\s)", prefix):
            return _read_rigaku_dat(data, filename)
        if re.search(r"(?m)^\s*_FILEVERSION\s*=", prefix) or suffix == ".uxd":
            return _read_uxd(data, filename)
        if suffix == ".raw":
            return _read_raw(data, filename)
        if suffix not in {".xy", ".txt", ".csv", ".dat"}:
            raise XRDParseError("Unrecognized measurement format. " + _EXPORT_HELP)
        return _read_text(data, filename)
    except XRDParseError:
        raise
    except (ValueError, TypeError, OverflowError, KeyError, struct.error, zipfile.BadZipFile,
            RuntimeError, EOFError, zlib.error, csv.Error) as exc:
        raise XRDParseError("The native measurement header is invalid or uses an unsupported layout. " + _EXPORT_HELP) from exc
