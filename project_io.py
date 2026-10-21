"""Portable XRD Maker projects and atomic, local-only autosave files."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

PROJECT_VERSION = 3
MAX_PROJECT_BYTES = 190 * 1024 * 1024
MAX_DATA_BYTES = 140 * 1024 * 1024
GLOBAL_KEYS = (
    "normalize", "show_legend", "show_cif_legend", "show_ref_lines", "show_peaks",
    "peak_prom", "global_offset", "xrange", "major_tick", "show_minor",
    "minor_tick", "cif_label_side", "cif_label_fontsize", "cif_label_offset_x",
    "cif_label_offset_y", "show_side_labels", "label_side", "label_fontsize",
    "label_offset_x", "label_offset_y", "fig_width", "fig_height", "dpi_export",
    "font_size", "show_panel", "lang_radio",
)
PATTERN_KEY = re.compile(
    r"(?:vis|ord|_val_lbl|xrd_color|extra_offset|yoff|cvis|cord|_val_clbl|"
    r"cif_color|cref_line|pvis|pord|_val_plbl|pdf_color|pref_line)_\d+$"
)
BOOL_KEYS = {"normalize", "show_legend", "show_cif_legend", "show_ref_lines",
             "show_peaks", "show_minor", "show_side_labels", "show_panel"}
BOUNDS = {
    "peak_prom": (.01, .5), "global_offset": (0., 3.), "major_tick": (1., 30.),
    "minor_tick": (.5, 10.), "cif_label_fontsize": (5, 20),
    "cif_label_offset_x": (0., 10.), "cif_label_offset_y": (0., 1.),
    "label_fontsize": (5, 24), "label_offset_x": (-5., 5.),
    "label_offset_y": (-.3, 1.), "fig_width": (6., 20.), "fig_height": (4., 20.),
    "font_size": (8, 20),
}
INT_KEYS = {"cif_label_fontsize", "label_fontsize", "font_size"}


class ProjectError(ValueError):
    """A project is damaged, unsupported, or contains invalid display settings."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ProjectError("Project settings must be an object.")
    cleaned = {}
    for key, value in settings.items():
        if key not in GLOBAL_KEYS and not PATTERN_KEY.fullmatch(key):
            continue
        valid = True
        if key in BOOL_KEYS or re.fullmatch(r"(?:vis|cvis|pvis|cref_line|pref_line)_\d+", key):
            valid = isinstance(value, bool)
        elif key in BOUNDS:
            lower, upper = BOUNDS[key]
            valid = _number(value) and lower <= value <= upper
            if key in INT_KEYS:
                valid = valid and isinstance(value, int)
        elif key == "xrange":
            valid = (isinstance(value, (list, tuple)) and len(value) == 2
                     and all(_number(v) for v in value) and 0 <= value[0] < value[1] <= 180)
            if valid:
                value = tuple(float(v) for v in value)
        elif key in ("cif_label_side", "label_side"):
            valid = value in ("左", "右")
        elif key == "lang_radio":
            valid = value in ("日本語", "English")
        elif key == "dpi_export":
            valid = type(value) is int and value in (300, 600)
        elif re.fullmatch(r"(?:ord|cord|pord)_\d+", key):
            valid = type(value) is int and 1 <= value <= 10000
        elif key.startswith("_val_"):
            valid = isinstance(value, str) and len(value) <= 10000
        elif "color_" in key:
            valid = isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None
        elif key.startswith("extra_offset_"):
            valid = _number(value) and -5 <= value <= 15
        elif key.startswith("yoff_"):
            valid = _number(value)
        if not valid:
            raise ProjectError(f"Invalid display setting: {key}")
        cleaned[key] = value
    return cleaned


def encode_project(xrd_files: dict, cif_files: dict, pdf_files: dict, settings: dict) -> bytes:
    result = {
        "format": "xrd-maker-project", "version": PROJECT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": clean_settings(settings), "checksums": {},
    }
    total = 0
    for kind, files in (("xrd_files", xrd_files), ("cif_files", cif_files), ("pdf_files", pdf_files)):
        result[kind] = {}
        result["checksums"][kind] = {}
        for name, raw in files.items():
            if not isinstance(name, str) or not name or len(name) > 1024 or "\0" in name:
                raise ProjectError("Invalid embedded filename.")
            if not isinstance(raw, bytes):
                raise ProjectError(f"Invalid file contents: {name}")
            total += len(raw)
            if total > MAX_DATA_BYTES:
                raise ProjectError("Project data exceeds the 140 MB limit.")
            result[kind][name] = base64.b64encode(raw).decode("ascii")
            result["checksums"][kind][name] = hashlib.sha256(raw).hexdigest()
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_PROJECT_BYTES:
        raise ProjectError("Project file exceeds the 190 MB limit.")
    return payload


def decode_project(raw: bytes) -> dict:
    """Validate the complete project before the caller updates any session state."""
    if len(raw) > MAX_PROJECT_BYTES:
        raise ProjectError("Project file exceeds the 190 MB limit.")
    try:
        project = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a valid XRD Maker JSON project.") from exc
    if not isinstance(project, dict) or type(project.get("version")) is not int or project["version"] not in (2, 3):
        raise ProjectError("Unsupported XRD Maker project version (expected version 2 or 3).")
    if project["version"] == 3 and project.get("format") != "xrd-maker-project":
        raise ProjectError("Unrecognized project format.")
    result = {"settings": clean_settings(project.get("settings", {})),
              "saved_at": project.get("saved_at", ""), "version": project["version"]}
    total = 0
    for kind in ("xrd_files", "cif_files", "pdf_files"):
        files = project.get(kind, {})
        if not isinstance(files, dict):
            raise ProjectError(f"Invalid file list: {kind}")
        result[kind] = {}
        for name, encoded in files.items():
            if not isinstance(name, str) or not name or len(name) > 1024 or "\0" in name or not isinstance(encoded, str):
                raise ProjectError("Invalid embedded file.")
            try:
                contents = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProjectError(f"Corrupted embedded file: {name}") from exc
            total += len(contents)
            if total > MAX_DATA_BYTES:
                raise ProjectError("Project data exceeds the 140 MB limit.")
            if project["version"] == 3:
                try:
                    expected = project["checksums"][kind][name]
                except (KeyError, TypeError) as exc:
                    raise ProjectError(f"Missing file checksum: {name}") from exc
                if expected != hashlib.sha256(contents).hexdigest():
                    raise ProjectError(f"File checksum mismatch: {name}")
            result[kind][name] = contents
    return result


def project_fingerprint(raw: bytes) -> str:
    project = json.loads(raw)
    project.pop("saved_at", None)
    # Source-file order determines scan identities and default plotting order.
    # Sort only settings: sorting all nested objects would hide a file reorder.
    project["settings"] = dict(sorted(project.get("settings", {}).items()))
    return hashlib.sha256(json.dumps(project, ensure_ascii=False).encode()).hexdigest()


def atomic_save(path: str | Path, contents: bytes) -> None:
    """Replace a local checkpoint only after a complete, flushed file is written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".xrd-checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
