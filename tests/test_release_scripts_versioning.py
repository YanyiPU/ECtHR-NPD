"""Offline version-selection, output and fail-closed release-script tests."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_challenging_view_v2 as view
import build_hf_upload_bundle as upload
import smoke_test_release as smoke
import verify_dataset_version as versions


class SmokeVersionTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'dataset_release/data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_explicit_legacy_reference_passes_own_fifty_feature_schema(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.run("audit", dataset_release=ROOT / "dataset_release", dataset_version="paper_reference"), 0)

    @unittest.skipUnless((ROOT / 'dataset_release/data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_legacy_never_falls_back_for_corrected(self):
        with self.assertRaisesRegex(smoke.ValidationError, "paper_reference only"):
            smoke.run("audit", dataset_release=ROOT / "dataset_release", dataset_version="corrected")

    def test_new_manifest_delegates_exact_selected_version(self):
        with tempfile.TemporaryDirectory() as folder:
            clean = Path(folder)
            (clean / "data").mkdir()
            (clean / "data/ecthr_npd_cases.csv").write_text("itemid\na\n")
            (clean / "VERSION_MANIFEST.json").write_text("{}")
            with patch.object(versions, "verify", return_value={"status": "PASS"}) as checked, contextlib.redirect_stdout(io.StringIO()):
                smoke.run("audit", dataset_release=clean, dataset_version="corrected")
            checked.assert_called_once_with(clean.resolve(), "corrected", mode="audit")

    def test_external_path_failures_are_validation_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.csv"
            for fn in (smoke.read_csv, smoke.load_json, smoke.require):
                with self.subTest(function=fn.__name__), self.assertRaises(smoke.ValidationError):
                    fn(path)

    def test_manifest_traversal_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            release = Path(folder)
            outside = Path(external) / "target"
            outside.write_text("x")
            (release / "alias").symlink_to(outside)
            with patch.object(smoke, "RELEASE", release):
                for relative in ("../target", str(outside), "alias", ".", ""):
                    with self.subTest(path=relative), self.assertRaises(smoke.ValidationError):
                        smoke.release_file(relative)

    def test_all_cli_dataset_arguments_are_required(self):
        for script in ("smoke_test_release.py", "build_challenging_view_v2.py", "build_hf_upload_bundle.py"):
            proc = subprocess.run([sys.executable, "-B", str(SCRIPTS / script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("--dataset-release", proc.stderr)


class ChallengingVersionTests(unittest.TestCase):
    @staticmethod
    def rows():
        canonical = {
            "a": {"split": "test", "test_view": "ID", "test_challenging_view": "False", "violated_articles": "3;6", "is_grand_chamber": "False"},
            "b": {"split": "test", "test_view": "OOD", "test_challenging_view": "True", "violated_articles": "3;6", "is_grand_chamber": "False"},
        }
        source = {key: {"appno": appno, "violated_articles": "3;6;3", "decision_body_category": "Chamber", "doctypebranch": "CHAMBER"}
                  for key, appno in (("a", "123/01;123/01"), ("b", "123/01;234/02"))}
        return canonical, source

    def test_small_version_uses_contract_and_distinct_appnos(self):
        ledger, members, summary = view.build_records(*self.rows(), expected_counts={"splits": {"test": 2}, "challenging": 1})
        self.assertEqual(ledger[0]["hudoc_application_number_count"], "1")
        self.assertEqual([r["itemid"] for r in members], ["b"])
        self.assertEqual(summary["reason_counts"], {view.REASON_MULTI: 1})

    def test_wrong_version_count_or_flag_is_rejected(self):
        for counts in ({"splits": {"test": 3}, "challenging": 1}, {"splits": {"test": 2}, "challenging": 2}):
            with self.assertRaises(view.ViewBuildError):
                view.build_records(*self.rows(), expected_counts=counts)
        canonical, source = self.rows()
        canonical["a"]["test_challenging_view"] = "True"
        with self.assertRaisesRegex(view.ViewBuildError, "canonical Challenging flags"):
            view.build_records(canonical, source, expected_counts={"splits": {"test": 2}, "challenging": 1})

    def test_output_cannot_modify_either_version_or_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clean = root / "paper_reference/clean"
            clean.mkdir(parents=True)
            for output in (clean / "metadata", root / "clean/metadata", root):
                with self.assertRaises(view.ViewBuildError):
                    view.write_release_records(clean, root / "source.csv", output, [], [], {})
        with tempfile.TemporaryDirectory() as release, tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(view.ViewBuildError, "fresh"):
                view.write_release_records(Path(release), Path(release) / "source.csv", Path(output), [], [], {})

    def test_fresh_output_records_canonical_hash_and_never_raw_numbers(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as destination:
            root, output = Path(folder), Path(destination) / "new"
            (root / "data").mkdir()
            (root / "data/ecthr_npd_cases.csv").write_text("itemid\na\n")
            source = root / "source.csv"
            source.write_text("itemid,appno\na,123/01\n")
            ledger, members, summary = view.build_records(*self.rows(), expected_counts={"splits": {"test": 2}, "challenging": 1})
            summary.update(dataset_version="corrected", release_id="synthetic")
            view.write_release_records(root, source, output, ledger, members, summary)
            record = json.loads((output / view.RECORD_NAME).read_text())
            self.assertEqual(record["dataset_version"], "corrected")
            self.assertEqual(record["canonical_table"]["sha256"], view.sha256(root / "data/ecthr_npd_cases.csv"))
            for path in output.iterdir():
                self.assertNotIn("123/01", path.read_text())


class UploadBoundaryTests(unittest.TestCase):
    def test_internal_versions_refused_even_with_override(self):
        for marker in ("VERSION_MANIFEST.json", "release_contract.json", "clean/VERSION_MANIFEST.json", "paper_reference/clean/VERSION_MANIFEST.json"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                file = root / marker
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}")
                with self.assertRaisesRegex(upload.UploadBundleError, "Internal dataset"):
                    upload.build_bundle(root, upload.DEFAULT_MANIFEST_RELATIVE, root / "out", allow_blocked=True)

    def test_function_level_relative_output_and_manifest_paths_guarded(self):
        with self.assertRaises(upload.UploadBundleError):
            upload.build_bundle(ROOT / "dataset_release", Path("../outside.json"), Path("/tmp/new-upload"), allow_blocked=True)
        with self.assertRaisesRegex(upload.UploadBundleError, "inside"):
            upload.build_bundle(ROOT / "dataset_release", upload.DEFAULT_MANIFEST_RELATIVE,
                                ROOT / "dataset_release/data/../new", allow_blocked=True)
        with self.assertRaises(upload.UploadBundleError):
            upload.ensure_inside_release(ROOT / "dataset_release", Path("../README.md"), label="test")

    @unittest.skipUnless((ROOT / 'dataset_release/data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_blocked_legacy_copy_has_nonupload_warning_and_excludes_v1(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "inspection"
            selected = upload.build_bundle(ROOT / "dataset_release", upload.DEFAULT_MANIFEST_RELATIVE, output, allow_blocked=True)
            warning = output / "DO_NOT_UPLOAD_INTERNAL_INSPECTION_ONLY.txt"
            self.assertTrue(warning.is_file())
            self.assertIn("DO NOT UPLOAD", warning.read_text())
            self.assertFalse(any("v1" in str(path) for path in selected))
            self.assertFalse((output / "metadata/test_challenging_view.v1.csv").exists())

    @unittest.skipUnless((ROOT / 'dataset_release/data/ecthr_npd_cases.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_blocked_legacy_normal_export_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new"
            with self.assertRaisesRegex(upload.UploadBundleError, "release gate"):
                upload.build_bundle(ROOT / "dataset_release", upload.DEFAULT_MANIFEST_RELATIVE, output, allow_blocked=False)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
