"""Offline known-layout tests for native measurement import (no private files)."""

import io
from pathlib import Path
import struct
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xrd_io import SUPPORTED_EXTENSIONS, XRDParseError, read_patterns


def zipped(members):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, value in members.items():
            z.writestr(name, value)
    return out.getvalue()


def raw_v1(scans, repeated_magic=True):
    """DIFFRAC-AT's fixed 156-byte range, including its optional RAW magic."""
    out = bytearray(b"RAW ")
    for i, (start, step, y) in enumerate(scans):
        if i and repeated_magic:
            out.extend(b"RAW ")
        header = bytearray(152)
        struct.pack_into("<IffI", header, 0, len(y), 2.0, step, 0)
        struct.pack_into("<f", header, 20, start)
        header[36:40] = b"test"
        struct.pack_into("<I", header, 148, int(i < len(scans) - 1))
        out.extend(header)
        out.extend(struct.pack("<" + "f" * len(y), *y))
    return bytes(out)


def raw_v2(scans):
    out = bytearray(256)
    out[:4] = b"RAW2"
    struct.pack_into("<H", out, 4, len(scans))
    for start, step, y in scans:
        header = bytearray(64)
        struct.pack_into("<HH", header, 0, 64, len(y))
        struct.pack_into("<H", header, 4, 10)  # drives 1 and 2, coupled
        struct.pack_into("<fff", header, 8, 3.0, step, start)
        out.extend(header)
        out.extend(struct.pack("<" + "f" * len(y), *y))
    return bytes(out)


def raw_v3(scans):
    out = bytearray(712)
    out[:8] = b"RAW1.01\0"
    struct.pack_into("<II", out, 8, 1, len(scans))
    for i, (start, step, y) in enumerate(scans):
        header = bytearray(304)
        struct.pack_into("<II", header, 0, 304, len(y))
        struct.pack_into("<d", header, 16, start)
        struct.pack_into("<d", header, 176, step)
        struct.pack_into("<f", header, 192, 4)
        struct.pack_into("<I", header, 252, 4)  # float32 intensity, no extra fields
        extra = 40 if i == 0 else 0
        struct.pack_into("<I", header, 256, extra)
        out.extend(header)
        out.extend(b"\0" * extra)
        out.extend(struct.pack("<" + "f" * len(y), *y))
    return bytes(out)


def raw_v4(scans, datum_size=4, scan_type="Locked Coupled"):
    out = bytearray(61)
    out[:8] = b"RAW4.00\0"
    # A metadata segment with an unknown tag must be skipped by its own size.
    out.extend(struct.pack("<II8s", 99, 16, b"metadata"))
    for start, step, y in scans:
        header = bytearray(160)
        struct.pack_into("<I", header, 0, 160)
        header[32:32 + len(scan_type)] = scan_type.encode()
        struct.pack_into("<ddIf", header, 72, start, step, len(y), 5.0)
        struct.pack_into("<II", header, 136, datum_size, 64)
        ext = bytearray(64)
        struct.pack_into("<II", ext, 0, 50, 64)
        ext[12:18] = b"2Theta"
        struct.pack_into("<d", ext, 56, start)
        out.extend(header)
        out.extend(ext)
        out.extend(struct.pack("<" + "f" * len(y), *y))
    return bytes(out)


def ras_scan(start=10, y=(3, 7, 5), axis="TwoThetaTheta"):
    rows = "\n".join(f"{start + i * .5} {v} 0" for i, v in enumerate(y))
    return f'''*RAS_HEADER_START
*MEAS_SCAN_AXIS_X_INTERNAL "{axis}"
*MEAS_DATA_COUNT "{len(y)}"
*MEAS_SCAN_UNIT_Y "counts"
*RAS_HEADER_END
*RAS_INT_START
{rows}
*RAS_INT_END
'''


def rasx_meta(count=3, axis="TwoTheta", unit="deg"):
    return f'''<MeasurementConditions><GeneralInformation><SampleName>Sample</SampleName></GeneralInformation>
<ScanInformation><AxisName>{axis}</AxisName><PositionUnit>{unit}</PositionUnit>
<IntensityUnit>counts</IntensityUnit><DataCount>{count}</DataCount></ScanInformation></MeasurementConditions>'''


def brml_route(values=(3, 7, 5), start=10, step=.5, counter_column=4):
    # The actual counter is explicitly described; the last numeric column is
    # temperature and must not be silently selected as intensity.
    datums = "".join(f"<Datum>2,1,{start + i * step},99,{y},300</Datum>" for i, y in enumerate(values))
    return f'''<DataRoute><ScanInformation ScanName="TwoThetaOmegaScan">
<MeasurementPoints>{len(values)}</MeasurementPoints><TimePerStep>2</TimePerStep>
<ScanAxes><ScanAxisInfo AxisName="TwoTheta"><Unit Base="Degree"/><Reference>0</Reference>
<Start>{start}</Start><Stop>{start + (len(values)-1) * step}</Stop><Increment>{step}</Increment>
</ScanAxisInfo></ScanAxes></ScanInformation><DataViews>
<RawDataView Start="0" Length="1" LogicName="MeasuredTime"/>
<RawDataView Start="2" Length="2"><Varying><FieldDefinitions FieldName="TwoTheta" AxisId="TwoTheta"/>
<FieldDefinitions FieldName="Theta" AxisId="Theta"/></Varying></RawDataView>
<RawDataView Start="{counter_column}" Length="1"><Recording LogicName="ScanCounter1" Category="Count"/></RawDataView>
</DataViews>{datums}</DataRoute>'''


class TextReaderTests(unittest.TestCase):
    def test_csv_headers_delimiters_and_stable_sort(self):
        for sep in (",", ";", "\t"):
            with self.subTest(sep=sep):
                raw = f"2theta{sep}Intensity\n20{sep}4\n10{sep}0\n10{sep}-2\n".encode()
                p = read_patterns(raw, "test.csv")[0]
                np.testing.assert_array_equal(p.x, [10, 10, 20])
                np.testing.assert_array_equal(p.y, [0, -2, 4])

    def test_text_header_with_numeric_metadata_is_rejected(self):
        with self.assertRaises(XRDParseError):
            read_patterns(b"10 3\n2026-10-22 sample\n11 4\n", "data.xy")

    def test_text_never_strips_numeric_suffix(self):
        with self.assertRaises(XRDParseError):
            read_patterns(b"10 12counts\n11 4\n", "data.xy")

    def test_single_column_is_rejected(self):
        with self.assertRaises(XRDParseError):
            read_patterns(b"10\n11\n12\n", "data.csv")

    def test_nonfinite_generic_rows_are_filtered(self):
        p = read_patterns(b"10,3\n11,inf\n12,nan\n13,7\n", "data.csv")[0]
        np.testing.assert_array_equal(p.x, [10, 13])

    def test_utf16_and_cp932_headers(self):
        for raw in ("角度,Intensity\n10,3\n11,4\n".encode("utf-16"),
                    "角度,Intensity\n10,3\n11,4\n".encode("cp932")):
            np.testing.assert_array_equal(read_patterns(raw, "data.csv")[0].y, [3, 4])

    def test_unknown_binary_raw_has_vendor_export_guidance(self):
        with self.assertRaisesRegex(XRDParseError, "Unknown RAW signature.*Rigaku.*RAS/RASX"):
            read_patterns(b"\x00\xff\x01\x80unknown", "Rigaku.raw")

    def test_supported_extension_list_contains_actual_readers(self):
        self.assertEqual(set(SUPPORTED_EXTENSIONS), {"xy", "txt", "csv", "dat", "ras", "rasx", "raw", "uxd", "brml"})


class RigakuReaderTests(unittest.TestCase):
    def test_ras_multiple_scans_and_attenuator_unchanged(self):
        raw = "*RAS_DATA_START\n" + ras_scan() + ras_scan(20, (0, -2, 12)) + "*RAS_DATA_END\n"
        patterns = read_patterns(raw.encode(), "native.ras")
        self.assertEqual(len(patterns), 2)
        np.testing.assert_array_equal(patterns[0].x, [10, 10.5, 11])
        np.testing.assert_array_equal(patterns[1].y, [0, -2, 12])
        self.assertFalse(patterns[0].metadata["attenuator_column_applied"])

    def test_ras_axis_and_count_are_validated(self):
        for raw in (ras_scan(axis="Omega"), ras_scan().replace('*MEAS_DATA_COUNT "3"', '*MEAS_DATA_COUNT "4"'),
                    ras_scan().replace('*MEAS_SCAN_AXIS_X_INTERNAL "TwoThetaTheta"', ''),
                    ras_scan().replace("*RAS_INT_END", "")):
            with self.subTest(raw=raw[:70]), self.assertRaises(XRDParseError):
                read_patterns(raw.encode(), "native.ras")

    def test_vendor_text_signature_survives_utf16_encoding(self):
        raw = ("*RAS_DATA_START\n" + ras_scan() + "*RAS_DATA_END\n").encode("utf-16")
        p = read_patterns(raw, "renamed.txt")[0]
        self.assertEqual(p.metadata["format"], "Rigaku RAS")

    def test_rasx_manifest_preserves_order_and_separate_scans(self):
        members = {
            "root.xml": '<Root><Data10 Type="Profile"><ContentHashList Name="Profile10.txt"/></Data10>'
                        '<Data2 Type="Profile"><ContentHashList Name="Profile2.txt"/></Data2></Root>',
            "Data2/Profile2.txt": "\ufeff20 4 0\n20.5 5 0\n21 6 0\n",
            "Data2/MesurementConditions2.xml": rasx_meta(),
            "Data10/Profile10.txt": "\ufeff10 1 0\n10.5 2 0\n11 3 0\n",
            "Data10/MesurementConditions10.xml": rasx_meta(axis="TwoThetaOmega"),
        }
        ps = read_patterns(zipped(members), "data.rasx")
        self.assertEqual([p.x[0] for p in ps], [10, 20])
        self.assertEqual([p.metadata["archive_member"] for p in ps], ["Data10/Profile10.txt", "Data2/Profile2.txt"])

    def test_rasx_missing_conditions_is_rejected(self):
        with self.assertRaises(XRDParseError):
            read_patterns(zipped({"Data0/Profile0.txt": "10 1 0\n11 2 0"}), "data.rasx")

    def test_rasx_count_and_axis_and_unit_validation(self):
        for meta in (rasx_meta(count=4), rasx_meta(axis="Omega"), rasx_meta(unit="rad")):
            with self.subTest(meta=meta), self.assertRaises(XRDParseError):
                read_patterns(zipped({"Data0/Profile0.txt": "10 1 0\n10.5 2 0\n11 3 0",
                                      "Data0/MesurementConditions0.xml": meta}), "data.rasx")

    def test_rigaku_dat_intensity_lists_are_not_xy_columns(self):
        raw = b'''*TYPE = Raw
*GROUP_COUNT = 2
*BEGIN
*GROUP = 0
*START = 10
*STEP = 0.5
*COUNT = 4
1, 5, 0, -2
*END
*BEGIN
*GROUP = 1
*START = 20
*STEP = 1
*COUNT = 2
3, 9
*END
*EOF
'''
        ps = read_patterns(raw, "native.dat")
        self.assertEqual(len(ps), 2)
        np.testing.assert_array_equal(ps[0].x, [10, 10.5, 11, 11.5])
        np.testing.assert_array_equal(ps[0].y, [1, 5, 0, -2])
        with self.assertRaises(XRDParseError):
            read_patterns(raw.replace(b"*COUNT = 4", b"*COUNT = 5"), "native.dat")
        with self.assertRaises(XRDParseError):
            read_patterns(raw.replace(b"*GROUP_COUNT = 2", b"*GROUP_COUNT = 3"), "native.dat")


class BrukerReaderTests(unittest.TestCase):
    scans = [(10, .5, [0, -2, 7]), (30, 1, [3, 9])]

    def test_supported_raw_versions_preserve_scan_boundaries_and_signed_counts(self):
        for builder in (raw_v1, raw_v3, raw_v4):
            with self.subTest(version=builder.__name__):
                ps = read_patterns(builder(self.scans), "native.raw")
                self.assertEqual(len(ps), 2)
                np.testing.assert_array_equal(ps[0].x, [10, 10.5, 11])
                np.testing.assert_array_equal(ps[0].y, [0, -2, 7])
                np.testing.assert_array_equal(ps[1].x, [30, 31])
                np.testing.assert_array_equal(ps[1].y, [3, 9])

    def test_raw_v1_old_ranges_without_repeated_magic(self):
        ps = read_patterns(raw_v1(self.scans, repeated_magic=False), "native.raw")
        self.assertEqual(len(ps), 2)
        np.testing.assert_array_equal(ps[1].y, [3, 9])

    def test_raw_truncation_is_never_silently_partially_imported(self):
        for builder in (raw_v1, raw_v3, raw_v4):
            with self.subTest(version=builder.__name__), self.assertRaises(XRDParseError):
                read_patterns(builder(self.scans)[:-1], "native.raw")

    def test_raw_v2_requires_an_explicit_instrument_motor_mapping(self):
        with self.assertRaisesRegex(XRDParseError, "RAW v2.*motor.*UXD/BRML"):
            read_patterns(raw_v2(self.scans), "native.raw")

    def test_raw_signature_overrides_extension(self):
        self.assertEqual(read_patterns(raw_v3(self.scans), "renamed.dat")[0].metadata["format"], "Bruker RAW v3")

    def test_raw_v4_rejects_wide_records_and_rocking_curves(self):
        for raw in (raw_v4(self.scans, datum_size=8), raw_v4(self.scans, scan_type="Rocking Curve")):
            with self.subTest(raw=raw[:8]), self.assertRaises(XRDParseError):
                read_patterns(raw, "native.raw")

    def test_raw_invalid_header_lengths_do_not_seek_guessed_offsets(self):
        raw = bytearray(raw_v3(self.scans))
        struct.pack_into("<I", raw, 712, 200)
        with self.assertRaises(XRDParseError):
            read_patterns(raw, "native.raw")

    def test_legacy_raw_rocking_and_nonangular_scans_are_rejected(self):
        v1 = bytearray(raw_v1(self.scans))
        struct.pack_into("<I", v1, 16, 8192)  # 2theta fixed: rocking curve
        v2 = bytearray(raw_v2(self.scans))
        struct.pack_into("<H", v2, 256 + 4, 2)  # theta drive only
        v3 = bytearray(raw_v3(self.scans))
        struct.pack_into("<I", v3, 712 + 196, 3)  # rocking curve
        for raw in (v1, v2, v3):
            with self.subTest(signature=raw[:8]), self.assertRaises(XRDParseError):
                read_patterns(raw, "native.raw")

    def test_raw_v3_extra_point_fields_are_not_interpreted_as_counts(self):
        raw = bytearray(raw_v3(self.scans))
        struct.pack_into("<II", raw, 712 + 248, 1, 12)
        with self.assertRaises(XRDParseError):
            read_patterns(raw, "native.raw")

    def test_uxd_counts_and_explicit_angle_scans(self):
        raw = b'''_FILEVERSION=1
_SAMPLE='sample'
_DRIVE='COUPLED'
_START=10
_STEPSIZE=0.5
_COUNTS
0 -2 7
_DRIVE='2THETA'
_2THETACPS
30,3
31,9
'''
        ps = read_patterns(raw, "native.uxd")
        self.assertEqual(len(ps), 2)
        np.testing.assert_array_equal(ps[0].x, [10, 10.5, 11])
        np.testing.assert_array_equal(ps[0].y, [0, -2, 7])
        self.assertEqual(ps[1].metadata["intensity_unit"], "cps")

    def test_uxd_counts_need_start_and_step_and_drive(self):
        for raw in (b"_FILEVERSION=1\n_DRIVE='COUPLED'\n_COUNTS\n1 2 3\n",
                    b"_FILEVERSION=1\n_COUNTS\n1 2 3\n"):
            with self.subTest(raw=raw), self.assertRaises(XRDParseError):
                read_patterns(raw, "native.uxd")

    def test_brml_explicit_views_choose_actual_counts_and_angles(self):
        raw = zipped({"Experiment0/RawData0.xml": "<RawData><DataRoutes>" + brml_route() + "</DataRoutes></RawData>"})
        p = read_patterns(raw, "native.brml")[0]
        np.testing.assert_array_equal(p.x, [10, 10.5, 11])
        np.testing.assert_array_equal(p.y, [3, 7, 5])
        self.assertEqual(p.metadata["intensity_transform"], "none")

    def test_brml_scan_files_and_routes_are_distinct(self):
        raw = zipped({
            "Experiment0/RawData10.xml": "<RawData><DataRoutes>" + brml_route(start=40) + "</DataRoutes></RawData>",
            "Experiment0/RawData2.xml": "<RawData><DataRoutes>" + brml_route(start=20) + brml_route(start=30) + "</DataRoutes></RawData>",
        })
        self.assertEqual([p.x[0] for p in read_patterns(raw, "native.brml")], [20, 30, 40])

    def test_brml_ambiguous_layout_is_rejected_not_guessed(self):
        xml = "<RawData><DataRoutes>" + brml_route().replace('<Recording LogicName="ScanCounter1" Category="Count"/>', '') + "</DataRoutes></RawData>"
        with self.assertRaisesRegex(XRDParseError, "guessing numeric columns"):
            read_patterns(zipped({"Experiment0/RawData0.xml": xml}), "native.brml")

    def test_brml_explicit_column_path_checks_angle_unit(self):
        xml = "<RawData><DataRoutes>" + brml_route().replace('Base="Degree"', 'Base="Radian"') + "</DataRoutes></RawData>"
        with self.assertRaisesRegex(XRDParseError, "unit"):
            read_patterns(zipped({"Experiment0/RawData0.xml": xml}), "native.brml")

    def test_brml_fixed_twotheta_rocking_curve_is_rejected(self):
        xml = "<RawData><DataRoutes>" + brml_route(step=0) + "</DataRoutes></RawData>"
        with self.assertRaisesRegex(XRDParseError, "fixed"):
            read_patterns(zipped({"Experiment0/RawData0.xml": xml}), "native.brml")

    def test_brml_empty_field_cannot_shift_counter_column(self):
        xml = "<RawData><DataRoutes>" + brml_route().replace("2,1,10.0,99,3,300", "2,,10.0,99,3,300") + "</DataRoutes></RawData>"
        with self.assertRaisesRegex(XRDParseError, "empty numeric field"):
            read_patterns(zipped({"Experiment0/RawData0.xml": xml}), "native.brml")

    def test_archive_and_xml_damage_is_reported(self):
        for raw, name in [(b"not zip", "data.brml"), (zipped({"root.xml": "<Root>"}), "data.rasx"),
                          (zipped({"Experiment0/RawData0.xml": '<!DOCTYPE x [<!ENTITY z "x">]><RawData/>'}), "data.brml")]:
            with self.subTest(name=name), self.assertRaises(XRDParseError):
                read_patterns(raw, name)


if __name__ == "__main__":
    unittest.main()
