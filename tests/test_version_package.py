import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("version_package_verifier", ROOT / "scripts/verify_dataset_version.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def put_csv(root, name, rows, columns=None):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pin(root, version):
    import hashlib
    files = {str(p.relative_to(root)): v.sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "VERSION_MANIFEST.json"}
    tables = []
    for name in files:
        if name.endswith(".csv"):
            cols, rows = v.read_csv(root / name)
            tables.append({"path": name, "columns": cols, "rows": len(rows)})
    contract = json.loads((root / "release_contract.json").read_text())
    manifest = {"schema_version": "ecthr-npd-version-1", "dataset_version": version,
                "release_id": contract["release_id"], "counts": contract["counts"], "files": files, "tables": tables,
                "structured_predictor_columns": v.read_csv(root / "model_inputs/structured_tree/features/train.csv")[0][1:],
                "revision_id": "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (root / "VERSION_MANIFEST.json").write_text(json.dumps(manifest))


def fixture(base, version="corrected"):
    root = base / ("clean" if version == "corrected" else "paper_reference/clean")
    root.mkdir(parents=True)
    rows = [{"itemid": f"case-{i}", "split": split, "test_view": "ID" if split == "test" else "",
             "test_challenging_view": "True" if split == "test" else "False", "is_grand_chamber": "False",
             "violated_articles_count": "2", "y_amount_eur": "100", "y_binary": "1"}
            for i, split in enumerate(["train", "validation", "test"])]
    put_csv(root, "data/ecthr_npd_cases.csv", rows)
    put_csv(root, "splits/case_index.csv", rows)
    for row in rows:
        split = row["split"]
        stem = "val" if split == "validation" else split
        put_csv(root, f"data/{split}.csv", [row])
        put_csv(root, f"model_inputs/structured_tree/features/{stem}.csv", [{"itemid": row["itemid"], **{f"f{i}": "1" for i in range(48 if version == "corrected" else 50)}}])
        put_csv(root, f"model_inputs/structured_tree/targets/{stem}.csv", [{k: row[k] for k in ["itemid", "y_amount_eur", "y_binary"]}])
    put_csv(root, "metadata/test_challenging_view_selector_inputs.v2.csv", [{"itemid": "case-2", "hudoc_application_number_count": "2", "distinct_violated_article_code_count": "2", "source_is_grand_chamber": "False"}])
    put_csv(root, "metadata/test_challenging_view.v2.csv", [{"itemid": "case-2", "split": "test", "test_view": "ID", "test_challenging_view": "True"}])
    if version == "corrected":
        put_csv(root, "data/applicants.csv", [{"itemid": "case-0", "applicant_id": "person-1"}])
        put_csv(root, "data/allocations.csv", [{"itemid": "case-0", "applicant_id": "person-1", "allocation_id": "alloc-1"}])
    (root / "release_contract.json").write_text(json.dumps({"release_id": "test-only", "dataset_version": version,
        "counts": {"total": 3, "splits": {"train": 1, "validation": 1, "test": 1}, "test_views": {"ID": 1}, "challenging": 1}}))
    pin(root, version)
    return root


class VersionPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = fixture(self.base)

    def test_corrected_and_reference_are_separately_verified(self):
        fixture(self.base, "paper_reference")
        for version in ["corrected", "paper_reference"]:
            self.assertEqual(v.verify(self.base, version)["status"], "PASS")

    def test_wrong_version_never_falls_back(self):
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "paper_reference")
        with self.assertRaises(v.VersionError):
            v.verify(self.base, "paper_reference")

    def test_publication_is_always_blocked_for_internal_versions(self):
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected", mode="publish")

    def test_unpinned_extra_file_rejected(self):
        (self.root / "secret.txt").write_text("synthetic")
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected")

    def test_csv_hash_tampering_is_rejected(self):
        path = self.root / "data/train.csv"
        path.write_text(path.read_text().replace("100", "101"))
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected")

    def test_repinned_target_mismatch_still_fails(self):
        path = self.root / "model_inputs/structured_tree/targets/train.csv"
        path.write_text(path.read_text().replace("100", "101"))
        pin(self.root, "corrected")
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected")

    def test_repinned_cross_case_person_link_fails(self):
        path = self.root / "data/allocations.csv"
        path.write_text(path.read_text().replace("case-0", "case-1"))
        pin(self.root, "corrected")
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected")

    def test_metadata_selector_is_recomputed(self):
        path = self.root / "metadata/test_challenging_view_selector_inputs.v2.csv"
        path.write_text(path.read_text().replace("case-2,2,2", "case-2,1,2"))
        pin(self.root, "corrected")
        with self.assertRaises(v.VersionError):
            v.verify(self.root, "corrected")

    def test_missing_and_nonfinite_are_not_zero(self):
        for value in ["", "NaN", "Infinity", "-1"]:
            with self.assertRaises(v.VersionError):
                v.amount(value)

    def test_manifest_paths_cannot_escape(self):
        for relative in ["../outside.csv", "/tmp/outside.csv", ""]:
            with self.assertRaises(v.VersionError):
                v.safe_file(self.root, relative)
        outside = self.base / "outside.csv"
        outside.write_text("synthetic")
        (self.root / "alias.csv").symlink_to(outside)
        with self.assertRaises(v.VersionError):
            v.safe_file(self.root, "alias.csv")


if __name__ == "__main__":
    unittest.main()
