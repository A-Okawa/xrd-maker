import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_bundle
import desktop
import setup_local


class DesktopPackageTests(unittest.TestCase):
    def source_tree(self, root):
        for name in build_bundle.BUNDLE_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source {name}", encoding="utf-8")

    def test_package_allowlist_excludes_credentials_and_user_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            for name in (".streamlit/secrets.toml", ".streamlit/secrets.toml.save", ".venv/secret", "autosave.json", "private.raw", "ui_draft.html"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private", encoding="utf-8")
            with zipfile.ZipFile(io.BytesIO(build_bundle.create_bundle_bytes(root))) as bundle:
                expected = {f"XRD-Maker/{name}" for name in build_bundle.BUNDLE_FILES}
                self.assertEqual(set(bundle.namelist()), expected | {"XRD-Maker/BUNDLE_MANIFEST.json"})
                manifest = json.loads(bundle.read("XRD-Maker/BUNDLE_MANIFEST.json"))
                self.assertEqual(set(manifest["files_sha256"]), set(build_bundle.BUNDLE_FILES))
                self.assertTrue(bundle.getinfo("XRD-Maker/launch_mac.command").external_attr >> 16 & 0o111)

    def test_bundle_refuses_symlink_and_incomplete_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            path = root / "app.py"
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                build_bundle.create_bundle_bytes(root)
            path.symlink_to(root / "desktop.py")
            with self.assertRaises(FileNotFoundError):
                build_bundle.create_bundle_bytes(root)

    def test_wheelhouse_only_includes_regular_wheels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            wheels = root / "wheelhouse"
            wheels.mkdir()
            (wheels / "example.whl").write_bytes(b"wheel")
            (wheels / "secret.txt").write_bytes(b"private")
            with zipfile.ZipFile(io.BytesIO(build_bundle.create_bundle_bytes(root, wheels))) as bundle:
                self.assertIn("XRD-Maker/wheelhouse/example.whl", bundle.namelist())
                self.assertNotIn("XRD-Maker/wheelhouse/secret.txt", bundle.namelist())
            (wheels / "example.whl").unlink()
            (wheels / "example.whl").symlink_to(wheels / "secret.txt")
            with self.assertRaises(ValueError):
                build_bundle.create_bundle_bytes(root, wheels)

    def test_local_launch_is_loopback_and_disables_telemetry(self):
        command = desktop.server_command(9001)
        self.assertIn("--server.address=127.0.0.1", command)
        self.assertIn("--server.port=9001", command)
        self.assertIn("--browser.gatherUsageStats=false", command)
        self.assertNotIn("pip", command)
        env = desktop.local_environment(Path("/tmp/xrd-testing"))
        self.assertEqual(env["XRD_MAKER_LOCAL"], "1")
        self.assertEqual(env["XRD_MAKER_DATA_DIR"], str(Path("/tmp/xrd-testing").resolve()))
        with patch.object(desktop.sys, "platform", "linux"), patch.dict(desktop.os.environ, {"XDG_DATA_HOME": ""}):
            self.assertEqual(desktop.default_data_dir(), Path.home() / ".local" / "share" / "xrd-maker")

    def test_failed_server_is_reported_before_browser_open(self):
        from unittest.mock import Mock
        process = Mock()
        process.poll.return_value = 1
        process.returncode = 1
        with self.assertRaises(RuntimeError):
            desktop.wait_until_ready(process, 9001)

    def test_check_does_not_start_server(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(desktop, "validate_installation") as validate, patch.object(desktop.subprocess, "Popen") as popen:
                self.assertEqual(desktop.main(["--check", "--data-dir", folder]), 0)
                validate.assert_called_once()
                popen.assert_not_called()

    def test_offline_setup_uses_only_local_wheels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wheels = root / "wheelhouse"
            wheels.mkdir()
            (wheels / "example.whl").touch()
            python = root / ".venv" / ("Scripts/python.exe" if setup_local.os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(setup_local, "APP_DIR", root), patch.object(setup_local.subprocess, "run") as run:
                self.assertEqual(setup_local.main(["--wheelhouse", str(wheels)]), 0)
                command = run.call_args_list[0].args[0]
                self.assertIn("--no-index", command)
                self.assertIn("--only-binary=:all:", command)
                self.assertEqual(command[command.index("--find-links") + 1], str(wheels.resolve()))

    def test_offline_setup_without_wheels_fails_without_running_pip(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(setup_local.subprocess, "run") as run:
                self.assertEqual(setup_local.main(["--offline", "--wheelhouse", folder]), 1)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
