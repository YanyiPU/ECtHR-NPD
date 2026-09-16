#!/usr/bin/env python3
"""ECtHR-NPD dataset loader.

These helpers read the fixed chronological splits from the public dataset
release, keep targets separate from model inputs, and reject strict feature
columns whose names indicate Article 41, operative-award, claim,
target-derived, or applicant-identifying leakage.
"""

from __future__ import annotations

import os
import json
import math
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from data import public_adapter


EXPECTED_SPLIT_COUNTS = {
    "train": 10217,
    "validation": 1461,
    "val": 1461,
    "test": 2897,
}

SPLIT_TO_FILE_STEM = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}

EXPECTED_TEST_VIEWS = {"ID": 1000, "OOD": 1897}
EXPECTED_CHALLENGING_COUNT = 699


def load_release_contract(release: Path, dataset_version: str | None = None) -> dict:
    """Require a labelled version contract; missing never means paper/current."""
    if public_adapter.is_public_root(release):
        return public_adapter.contract(release)
    path = release / "release_contract.json"
    if not path.exists():
        raise DatasetReleaseError("Missing release_contract.json; unversioned/frozen directories are not selected implicitly")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("dataset_version") not in {"corrected", "paper_reference"}:
        raise DatasetReleaseError("release_contract requires dataset_version corrected or paper_reference")
    if dataset_version is not None and contract["dataset_version"] != dataset_version:
        raise DatasetReleaseError(f"Selected dataset_version={dataset_version} but contract is {contract['dataset_version']}")
    counts = contract.get("counts", {})
    splits, views = counts.get("splits", {}), counts.get("test_views", {})
    if not contract.get("release_id") or set(splits) != {"train", "validation", "test"} or set(views) != {"ID", "OOD"}:
        raise DatasetReleaseError("invalid release_contract identity/split/view keys")
    numbers = [counts.get("total"), counts.get("challenging"), *splits.values(), *views.values()]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in numbers):
        raise DatasetReleaseError("release_contract counts must be nonnegative integers")
    if sum(splits.values()) != counts["total"] or sum(views.values()) != splits["test"] or counts["challenging"] > splits["test"]:
        raise DatasetReleaseError("release_contract counts do not reconcile")
    return contract

FORBIDDEN_STRICT_INPUT_TERMS = (
    "claim",
    "award_",
    "article_41",
    "article41",
    "operative",
    "raw_extractor",
    "safe_non_pec",
    "safe_pec",
    "safe_total",
    "repair",
    "target",
    "dismissed",
    "satisfaction",
    "costs_eur",
    "pec_eur",
    "amount",
    "case_name",
    "party_applicant",
    "docname",
    "appno",
    "ecli",
    "beneficiary_label",
)
UNVERIFIED_ALLOCATION_FEATURES = {
    "perapp_unique_beneficiary_category_count",
    "perapp_has_joint_beneficiary_category_flag",
}

SAFE_METADATA_COLUMNS = (
    "country_alpha2",
    "hudoc_decision_body",
    "case_importance",
    "has_separate_opinion",
    "represented",
    "num_violations_found",
    "violation_type",
    "num_applicants",
    "judgment_year",
    "judgment_month",
)

VIOLATED_ARTICLE_COLUMNS = (
    "violated_articles_count",
    "violated_article_6",
    "violated_article_3",
    "violated_article_5",
    "violated_article_13",
    "violated_article_8",
    "violated_article_P1_1",
    "violated_article_2",
    "violated_article_10",
    "violated_article_11",
    "violated_article_14",
    "violated_article_34",
    "violated_article_9",
    "violated_article_P1_3",
    "violated_article_38",
    "violated_article_P4_2",
    "violated_article_7",
    "violated_article_P7_4",
    "violated_article_P1_2",
    "violated_article_18",
    "violated_article_P7_2",
)

SERIALIZED_CASE_FACT_COLUMNS = (
    "app_sex_male_ratio",
    "app_sex_female_ratio",
    "app_sex_unknown_ratio",
    "app_age_child_ratio",
    "app_age_adolescent_ratio",
    "app_age_adult_ratio",
    "app_age_elderly_ratio",
    "app_age_unknown_ratio",
    "app_birth_year_median",
    "app_unique_nationality_count",
    "app_multi_nationality_flag",
    "rl_violation_type_procedural_ratio",
    "rl_violation_type_both_ratio",
    "rl_violation_duration_months",
    "rl_violation_duration_months_missing_ratio",
)

STRUCTURED_EXTERNAL_FACTOR_COLUMNS = (
    "gdp_per_capita_log1p",
    "gdp_constant_2015_log1p",
)

# The source economic-covariate table also contains join/audit metadata and
# raw values.  A model must never receive those columns merely because the
# table is merged into an input frame.  Keep this allow-list deliberately
# small and identical to the two external features in the 50-column tree
# matrix (48 corrected predictors; 50 in the audit-only historical matrix).
EXTERNAL_FACTOR_INPUT_COLUMNS = STRUCTURED_EXTERNAL_FACTOR_COLUMNS
EXTERNAL_FACTOR_REQUIRED_SOURCE_COLUMNS = ("itemid", *EXTERNAL_FACTOR_INPUT_COLUMNS)


class DatasetReleaseError(ValueError):
    """Raised when a dataset release does not match the expected protocol."""


@dataclass(frozen=True)
class DatasetSplit:
    """A fixed split with strict features and labels kept separate."""

    name: str
    features: pd.DataFrame
    targets: pd.DataFrame
    cases: pd.DataFrame
    dataset_release: Path

    @property
    def itemids(self) -> pd.Series:
        return self.features["itemid"].copy()

    @property
    def X(self) -> pd.DataFrame:
        assert_no_forbidden_strict_columns(self.features.columns)
        return self.features.drop(columns=["itemid"]).copy()

    @property
    def y_amount_eur(self) -> pd.Series:
        return pd.to_numeric(self.targets["y_amount_eur"], errors="raise")

    @property
    def y_binary(self) -> pd.Series:
        return pd.to_numeric(self.targets["y_binary"], errors="raise").astype(int)

    def merged_for_audit(self) -> pd.DataFrame:
        """Return features plus labels for metrics/audits, not model input."""
        return self.features.merge(self.targets, on="itemid", how="inner", validate="one_to_one")

    @property
    def safe_metadata(self) -> pd.DataFrame:
        return self._feature_group(SAFE_METADATA_COLUMNS)

    @property
    def violated_article_features(self) -> pd.DataFrame:
        return self._feature_group(VIOLATED_ARTICLE_COLUMNS)

    @property
    def serialized_case_features(self) -> pd.DataFrame:
        return self._feature_group(SERIALIZED_CASE_FACT_COLUMNS)

    @property
    def structured_external_factors(self) -> pd.DataFrame:
        return self._feature_group(STRUCTURED_EXTERNAL_FACTOR_COLUMNS)

    def _feature_group(self, columns: Iterable[str]) -> pd.DataFrame:
        present = [column for column in columns if column in self.features.columns]
        return self.features[["itemid", *present]].copy()


def resolve_dataset_release(dataset_release: str | Path | None = None, *, dataset_version: str = "corrected") -> Path:
    """Select only an explicit root/configuration; never scan legacy bundles."""
    if dataset_version not in {"corrected", "paper_reference"}:
        raise DatasetReleaseError("dataset_version must be corrected or paper_reference")
    requested = dataset_release or os.environ.get("ECTHR_NPD_DATASET_RELEASE")
    if not requested:
        raise DatasetReleaseError("Supply --dataset-release (dataset root or clean root), or ECTHR_NPD_DATASET_RELEASE; no legacy auto-discovery")
    explicit = Path(requested).expanduser().resolve()
    if public_adapter.is_public_root(explicit):
        if dataset_version == "paper_reference":
            raise DatasetReleaseError("The public package contains one unified dataset; no paper_reference selection exists")
        public_adapter.tables(explicit)
        return explicit
    if (explicit / "data" / "ecthr_npd_cases.csv").is_file():
        selected = explicit
    else:
        selected = explicit / ("clean" if dataset_version == "corrected" else "paper_reference/clean")
    if not (selected / "data" / "ecthr_npd_cases.csv").is_file():
        raise DatasetReleaseError(f"Explicit {dataset_version} dataset release is invalid: {selected}")
    load_release_contract(selected, dataset_version)
    return selected


def require_corrected_version(dataset_version: str) -> None:
    if dataset_version != "corrected":
        raise DatasetReleaseError("paper_reference retains 50 historical columns and is legacy-audit only, not strict training input; use load_legacy_audit_split")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(dataset_release=None, *, dataset_version="corrected", input_files=None) -> dict:
    """Content-address actual canonical inputs, independently of directory name."""
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    if public_adapter.is_public_root(release):
        return public_adapter.provenance(release, input_files)
    contract = load_release_contract(release, dataset_version)
    manifest_path = release / "VERSION_MANIFEST.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("files", {})
        if manifest.get("schema_version") != "ecthr-npd-version-1" or manifest.get("dataset_version") != dataset_version or manifest.get("release_id") != contract["release_id"] or manifest.get("counts") != contract["counts"] or not files:
            raise DatasetReleaseError("Version manifest identity/contract mismatch")
        for name, expected_hash in files.items():
            relative = Path(name)
            path = release / relative
            if relative.is_absolute() or ".." in relative.parts or release not in path.resolve().parents or path.is_symlink() or any(p.is_symlink() for p in path.parents if p != release.parent) or not path.is_file():
                raise DatasetReleaseError("Unsafe/missing version manifest path")
            if file_sha256(path) != expected_hash:
                raise DatasetReleaseError(f"Version manifest hash mismatch: {name}")
        actual = {str(p.relative_to(release)) for p in release.rglob("*") if p.is_file() and not p.name.startswith(".") and p.name != "VERSION_MANIFEST.json"}
        if set(files) != actual:
            raise DatasetReleaseError("Version manifest file inventory mismatch")
        revision = "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if manifest.get("revision_id") != revision:
            raise DatasetReleaseError("Version manifest revision mismatch")
    paths = ["data/ecthr_npd_cases.csv", "model_inputs/external_factors/economic_covariates.csv"]
    paths += [f"model_inputs/structured_tree/{kind}/{split}.csv" for kind in ("features", "targets") for split in ("train", "val", "test")]
    hashes = {name: file_sha256(release / name) for name in paths if (release / name).is_file()}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"dataset_version": dataset_version, "release_id": contract["release_id"],
            "dataset_fingerprint_sha256": fingerprint, "canonical_input_sha256": hashes,
            "canonical_inputs_missing": [name for name in paths if name not in hashes],
            "integrity_status": "version_manifest_verified" if manifest else "contract_and_actual_input_hashes_only_no_version_manifest",
            "dataset_revision_id": manifest.get("revision_id") if manifest else None,
            "release_contract_sha256": file_sha256(release / "release_contract.json"),
            "version_manifest_sha256": file_sha256(release / "VERSION_MANIFEST.json") if (release / "VERSION_MANIFEST.json").is_file() else None,
            "run_input_sha256": {name: file_sha256(path) for name, path in (input_files or {}).items() if path is not None},
            "training_status": "legacy_audit_only_not_strict_trainable" if dataset_version == "paper_reference" else "corrected_48_features_not_historical_reproduction"}


def validate_selected_targets(itemids, amounts, dataset_release, *, dataset_version="corrected", splits=None, require_all=False) -> dict:
    """Bind supplied labels and optional splits to actual selected case rows."""
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    if public_adapter.is_public_root(release):
        return public_adapter.validate_targets(release, itemids, amounts, splits, require_all)
    cases = _read_csv(release / "data/ecthr_npd_cases.csv")
    required = {"itemid", "split", "y_amount_eur", "y_binary"}
    if not required.issubset(cases) or cases.itemid.isna().any() or cases.itemid.duplicated().any():
        raise DatasetReleaseError("Invalid canonical case target index")
    ids = [str(value) for value in itemids]
    if len(ids) != len(set(ids)) or len(ids) != len(amounts):
        raise DatasetReleaseError("Duplicate IDs or mismatched target lengths")
    indexed = cases.assign(itemid=cases.itemid.astype(str)).set_index("itemid")
    counts = load_release_contract(release)["counts"]
    if len(cases) != counts["total"] or cases["split"].value_counts().to_dict() != {k: v for k, v in counts["splits"].items() if v}:
        raise DatasetReleaseError("Canonical case counts differ from selected contract")
    for split, stem in [("train", "train"), ("validation", "val"), ("test", "test")]:
        target_path = release / f"model_inputs/structured_tree/targets/{stem}.csv"
        if target_path.is_file():
            table = _read_csv(target_path)
            _validate_split_alignment(split, release, table[["itemid"]], table, cases)
    if set(ids) - set(indexed.index) or (require_all and set(ids) != set(indexed.index)):
        raise DatasetReleaseError("Supplied target IDs differ from selected dataset")
    actual = pd.to_numeric(pd.Series(amounts), errors="raise").to_numpy(dtype=float)
    expected = pd.to_numeric(indexed.loc[ids, "y_amount_eur"], errors="raise").to_numpy(dtype=float)
    if not all(math.isfinite(v) and v >= 0 for v in actual) or not (actual == expected).all():
        raise DatasetReleaseError("Supplied y_amount_eur differs from selected dataset version")
    if not (pd.to_numeric(indexed.loc[ids, "y_binary"], errors="raise").to_numpy() == (expected > 0).astype(int)).all():
        raise DatasetReleaseError("Selected case binary targets are inconsistent")
    if splits is not None:
        supplied = ["validation" if str(s) == "val" else str(s) for s in splits]
        if supplied != indexed.loc[ids, "split"].tolist():
            raise DatasetReleaseError("Supplied splits differ from selected dataset version")
    return dataset_provenance(release, dataset_version=dataset_version)


@dataclass(frozen=True)
class LegacyAuditSplit:
    """Historical frames preserved intact; deliberately not a DatasetSplit/.X."""
    name: str
    features: pd.DataFrame
    targets: pd.DataFrame
    cases: pd.DataFrame
    dataset_release: Path
    provenance: dict

    @property
    def X(self):
        raise DatasetReleaseError("Legacy audit frames are not strict model inputs; historical 50-column matrix contains two unverified allocation-side predictors")


def load_legacy_audit_split(split, dataset_release, *, dataset_version="paper_reference") -> LegacyAuditSplit:
    if dataset_version != "paper_reference":
        raise DatasetReleaseError("Legacy audit requires explicit paper_reference version")
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    stem = SPLIT_TO_FILE_STEM[_normalise_split_name(split)]
    features = _read_csv(release / "model_inputs/structured_tree/features" / f"{stem}.csv")
    targets = _read_csv(release / "model_inputs/structured_tree/targets" / f"{stem}.csv")
    cases = _read_csv(release / "data/ecthr_npd_cases.csv")
    if len(features.columns) != 51 or not UNVERIFIED_ALLOCATION_FEATURES.issubset(features.columns):
        raise DatasetReleaseError("paper_reference audit requires the intact historical 50 predictors plus itemid")
    _validate_split_alignment(split, release, features, targets, cases)
    warnings.warn("paper_reference: 50 historical predictors preserved for legacy audit; NOT strict-trainable or independently revalidated", UserWarning, stacklevel=2)
    return LegacyAuditSplit("validation" if stem == "val" else stem, features, targets, cases, release,
                            dataset_provenance(release, dataset_version=dataset_version))


def assert_no_forbidden_strict_columns(columns: Iterable[str]) -> None:
    bad: list[str] = []
    for column in columns:
        lowered = column.lower()
        if column == "itemid":
            continue
        if lowered in UNVERIFIED_ALLOCATION_FEATURES or any(term in lowered for term in FORBIDDEN_STRICT_INPUT_TERMS):
            bad.append(column)
    if bad:
        raise DatasetReleaseError(
            "Strict feature matrix contains leakage-sensitive columns: "
            + ", ".join(sorted(bad))
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalise_split_name(split: str) -> str:
    key = split.strip().lower()
    if key not in SPLIT_TO_FILE_STEM:
        raise DatasetReleaseError(f"Unknown split {split!r}; expected train, validation/val, or test.")
    return key


def _validate_split_alignment(split, release, features, targets, cases):
    canonical = "validation" if _normalise_split_name(split) == "val" else _normalise_split_name(split)
    expected = load_release_contract(release)["counts"]["splits"][canonical]
    if "itemid" not in features or not {"itemid", "y_amount_eur", "y_binary"}.issubset(targets) or not {"itemid", "split", "y_amount_eur", "y_binary"}.issubset(cases):
        raise DatasetReleaseError("Feature/target/case tables are missing required identity, split or target columns")
    if any(frame["itemid"].isna().any() or frame["itemid"].duplicated().any() for frame in (features, targets, cases)):
        raise DatasetReleaseError("Blank/duplicate canonical feature/target/case IDs")
    subset = cases[cases["split"] == canonical]
    if len(features) != expected or len(targets) != expected or len(subset) != expected:
        raise DatasetReleaseError(f"{canonical} feature/target/case counts differ from contract {expected}")
    if list(features["itemid"]) != list(targets["itemid"]) or set(features["itemid"]) != set(subset["itemid"]):
        raise DatasetReleaseError(f"{canonical} feature/target/case identities or order differ")
    amounts = pd.to_numeric(targets["y_amount_eur"], errors="raise")
    binary = pd.to_numeric(targets["y_binary"], errors="raise")
    if not amounts.map(math.isfinite).all() or (amounts < 0).any() or not binary.isin([0, 1]).all() or not (binary.to_numpy() == (amounts.to_numpy() > 0).astype(int)).all():
        raise DatasetReleaseError(f"{canonical} target EUR/binary values are invalid")
    for column in ("y_amount_eur", "y_binary"):
        truth = pd.to_numeric(subset.set_index("itemid")[column].reindex(targets["itemid"]), errors="raise")
        if not (truth.to_numpy() == pd.to_numeric(targets[column], errors="raise").to_numpy()).all():
            raise DatasetReleaseError(f"{canonical} {column} differs between targets and canonical cases")


def load_structured_tree_split(
    split: str,
    dataset_release: str | Path | None = None,
    *,
    validate: bool = True,
    dataset_version: str = "corrected",
) -> DatasetSplit:
    """Load one fixed split from model_inputs/structured_tree.

    The returned ``DatasetSplit.X`` frame is strict model input. Targets are
    available only through ``DatasetSplit.y_amount_eur`` and
    ``DatasetSplit.y_binary``.
    """
    require_corrected_version(dataset_version)
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    split_key = _normalise_split_name(split)
    file_stem = SPLIT_TO_FILE_STEM[split_key]

    if public_adapter.is_public_root(release):
        cases = public_adapter.canonical_cases(release)
        canonical = "validation" if split_key == "val" else split_key
        subset = cases.loc[cases["split"] == canonical].copy()
        features = public_adapter.feature_frame(subset)
        assert_no_forbidden_strict_columns(features.columns)
        targets = subset[["itemid", "y_amount_eur", "y_binary"]].reset_index(drop=True)
        return DatasetSplit(canonical, features, targets, cases, release)

    features = _read_csv(release / "model_inputs" / "structured_tree" / "features" / f"{file_stem}.csv")
    targets = _read_csv(release / "model_inputs" / "structured_tree" / "targets" / f"{file_stem}.csv")
    cases = _read_csv(release / "data" / "ecthr_npd_cases.csv")

    if not validate:
        warnings.warn("validate=False no longer bypasses strict target/leakage gates; use load_legacy_audit_split for paper_reference", UserWarning, stacklevel=2)
    validate = True
    _validate_split_alignment(split, release, features, targets, cases)
    dataset_provenance(release, dataset_version=dataset_version)
    expected_columns = ["itemid", *SAFE_METADATA_COLUMNS, *VIOLATED_ARTICLE_COLUMNS,
                        *SERIALIZED_CASE_FACT_COLUMNS, *STRUCTURED_EXTERNAL_FACTOR_COLUMNS]
    if list(features.columns) != expected_columns:
        raise DatasetReleaseError("Corrected strict feature matrix must have the exact ordered 48 predictors plus itemid")

    if validate:
        canonical_split = "validation" if split_key == "val" else split_key
        expected_count = load_release_contract(release)["counts"]["splits"][canonical_split]
        if len(features) != expected_count:
            raise DatasetReleaseError(f"{split_key} feature count {len(features)} != {expected_count}")
        if len(targets) != expected_count:
            raise DatasetReleaseError(f"{split_key} target count {len(targets)} != {expected_count}")
        if list(features["itemid"]) != list(targets["itemid"]):
            raise DatasetReleaseError(f"{split_key} feature/target itemid order mismatch")
        if features["itemid"].duplicated().any():
            raise DatasetReleaseError(f"{split_key} features contain duplicate itemid values")
        amounts = pd.to_numeric(targets["y_amount_eur"], errors="raise")
        binary = pd.to_numeric(targets["y_binary"], errors="raise")
        if not amounts.map(math.isfinite).all() or (amounts < 0).any() or not binary.isin([0, 1]).all():
            raise DatasetReleaseError(f"{split_key} targets must be finite nonnegative EUR and binary 0/1")
        if not (binary.to_numpy() == (amounts.to_numpy() > 0).astype(int)).all():
            raise DatasetReleaseError(f"{split_key} amount/binary target mismatch")
        assert_no_forbidden_strict_columns(features.columns)

        canonical_split = "validation" if split_key == "val" else split_key
        case_subset = cases[cases["split"] == canonical_split]
        if len(case_subset) != expected_count:
            raise DatasetReleaseError(f"{split_key} case index count {len(case_subset)} != {expected_count}")
        if case_subset["itemid"].duplicated().any() or set(features["itemid"]) != set(case_subset["itemid"]):
            raise DatasetReleaseError(f"{split_key} feature/target IDs do not match the canonical split")
        for column in ("y_amount_eur", "y_binary"):
            if column in case_subset:
                expected = pd.to_numeric(case_subset.set_index("itemid")[column].reindex(targets["itemid"]), errors="raise")
                actual = pd.to_numeric(targets[column], errors="raise")
                if expected.isna().any() or not (expected.to_numpy() == actual.to_numpy()).all():
                    raise DatasetReleaseError(f"{split_key} {column} differs between targets and canonical cases")

    return DatasetSplit(
        name="validation" if split_key == "val" else split_key,
        features=features,
        targets=targets,
        cases=cases,
        dataset_release=release,
    )


def load_structured_tree_splits(
    dataset_release: str | Path | None = None,
    *,
    validate: bool = True,
    dataset_version: str = "corrected",
) -> dict[str, DatasetSplit]:
    """Load train, validation, and test splits from the dataset release."""
    return {
        "train": load_structured_tree_split("train", dataset_release, validate=validate, dataset_version=dataset_version),
        "validation": load_structured_tree_split("validation", dataset_release, validate=validate, dataset_version=dataset_version),
        "test": load_structured_tree_split("test", dataset_release, validate=validate, dataset_version=dataset_version),
    }


def load_external_factors(dataset_release: str | Path | None = None, *, dataset_version="corrected") -> pd.DataFrame:
    """Load only the allow-listed economic covariates for model input.

    The on-disk source may carry state/year join metadata or future audit
    fields.  This function projects the table before returning it, so callers
    cannot accidentally serialize split/view identifiers or unapproved raw
    covariates into a model input.
    """
    require_corrected_version(dataset_version)
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    if public_adapter.is_public_root(release):
        return public_adapter.feature_frame(public_adapter.canonical_cases(release))[["itemid", *public_adapter.EXTERNAL]]
    external = _read_csv(release / "model_inputs" / "external_factors" / "economic_covariates.csv")
    missing = [column for column in EXTERNAL_FACTOR_REQUIRED_SOURCE_COLUMNS if column not in external.columns]
    if missing:
        raise DatasetReleaseError("External factor source missing required columns: " + ", ".join(missing))
    expected_count = load_release_contract(release)["counts"]["total"]
    if len(external) != expected_count:
        raise DatasetReleaseError(f"external factor row_count {len(external)} != {expected_count}")
    projected = external.loc[:, EXTERNAL_FACTOR_REQUIRED_SOURCE_COLUMNS].copy()
    case_ids = _read_csv(release / "data/ecthr_npd_cases.csv")["itemid"]
    if projected["itemid"].isna().any() or projected["itemid"].duplicated().any() or set(projected.itemid) != set(case_ids):
        raise DatasetReleaseError("External factor source contains blank/duplicate/orphan itemid values")
    assert_no_forbidden_strict_columns(projected.columns)
    return projected


def input_contract_summary() -> dict[str, object]:
    """Return the shared strict input groups used by all model families."""
    return {
        "allowed_input_groups": {
            "safe_metadata": list(SAFE_METADATA_COLUMNS),
            "violated_articles": list(VIOLATED_ARTICLE_COLUMNS),
            "case_facts_or_serialized_inputs": list(SERIALIZED_CASE_FACT_COLUMNS),
            "external_factors": list(EXTERNAL_FACTOR_INPUT_COLUMNS),
        },
        "structured_tree_external_factor_columns": list(STRUCTURED_EXTERNAL_FACTOR_COLUMNS),
        "excluded_strict_input_terms": list(FORBIDDEN_STRICT_INPUT_TERMS),
        "target_columns": ["y_amount_eur", "y_binary"],
        "notes": (
            "Targets, split/view labels, Article 41/50 text, operative clauses, "
            "claims, award-side fields, and target-derived fields are not model features."
        ),
    }


def validate_dataset_release(dataset_release: str | Path | None = None, *, dataset_version="corrected") -> dict[str, object]:
    """Validate row counts and diagnostic test views."""
    require_corrected_version(dataset_version)
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    if public_adapter.is_public_root(release):
        contract = public_adapter.contract(release)
        return {"status": "PASS", "errors": [], "release_id": contract["release_id"],
                "row_count": contract["counts"]["total"], "split_counts": contract["counts"]["splits"],
                "test_view_counts": contract["counts"]["test_views"],
                "test_challenging_count": contract["counts"]["challenging"],
                "dataset_provenance": public_adapter.provenance(release),
                "input_contract": public_adapter.feature_schema()}
    contract = load_release_contract(release)
    cases = _read_csv(release / "data" / "ecthr_npd_cases.csv")
    external = load_external_factors(release)
    split_counts = cases["split"].value_counts().to_dict()
    test = cases[cases["split"] == "test"]
    test_views = test["test_view"].value_counts().to_dict()
    challenging = int(pd.to_numeric(test["test_challenging_view"], errors="coerce").fillna(0).sum())

    errors: list[str] = []
    expected_total = contract["counts"]["total"]
    if len(cases) != expected_total:
        errors.append(f"row_count {len(cases)} != {expected_total}")
    expected_splits = contract["counts"]["splits"]
    if {k: int(v) for k, v in split_counts.items()} != expected_splits:
        errors.append(f"split_counts {split_counts} != {expected_splits}")
    if {k: int(v) for k, v in test_views.items()} != contract["counts"]["test_views"]:
        errors.append(f"test_view_counts {test_views} != {contract['counts']['test_views']}")
    if challenging != contract["counts"]["challenging"]:
        errors.append(f"challenging_count {challenging} != {contract['counts']['challenging']}")

    for split in ("train", "validation", "test"):
        load_structured_tree_split(split, release, validate=True)

    if set(external["itemid"]) != set(cases["itemid"]):
        errors.append("external factor itemids do not match canonical cases")

    return {
        "status": "PASS" if not errors else "FAIL",
        "release_id": contract["release_id"],
        "dataset_provenance": dataset_provenance(release, dataset_version=dataset_version),
        "errors": errors,
        "row_count": int(len(cases)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "test_view_counts": {k: int(v) for k, v in test_views.items()},
        "test_challenging_count": challenging,
        "external_factor_rows": int(len(external)),
        "external_factor_columns": list(external.columns),
        "input_contract": input_contract_summary(),
    }


def encode_categorical_columns(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    categorical_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """One-hot encode categoricals with a train-only vocabulary.

    This is the XGBoost/LightGBM branch of the submitted tree protocol.
    CatBoost uses native categorical pools in ``tree_models.train`` instead.
    Every categorical column gets a dedicated unknown-category dummy, so an
    unseen validation/test value is not silently assigned an ordinal code or
    conflated with a known category. Numeric columns are coerced to numbers
    and missing values are filled with the *training-split median*. A column
    that is entirely missing in training is rejected instead of silently
    acquiring a target-independent constant.
    """
    raw_frames = [train.copy(), *(frame.copy() for frame in others)]
    inferred_categorical_cols = [
        col
        for col in train.columns
        if str(train[col].dtype) in {"object", "category", "string"} or train[col].dtype == bool
    ]
    requested_categorical_cols = set(categorical_columns or ())
    unknown_requested = requested_categorical_cols - set(train.columns)
    if unknown_requested:
        raise DatasetReleaseError(
            "Configured categorical columns are absent from training input: " + ", ".join(sorted(unknown_requested))
        )
    categorical_cols = set(inferred_categorical_cols) | requested_categorical_cols

    encoded_columns_by_frame: list[list[pd.Series]] = [[] for _ in raw_frames]
    for col in train.columns:
        if col in categorical_cols:
            observed_train = train[col].dropna().astype(str).tolist()
            missing_token = "<MISSING>"
            while missing_token in observed_train:
                missing_token = f"{missing_token}_"
            unknown_token = "<UNKNOWN>"
            while unknown_token in observed_train or unknown_token == missing_token:
                unknown_token = f"{unknown_token}_"
            known_categories = list(dict.fromkeys([*observed_train, missing_token]))
            all_categories = [*known_categories, unknown_token]
            for index, frame in enumerate(raw_frames):
                values = frame[col].fillna(missing_token).astype(str)
                values = values.where(values.isin(known_categories), unknown_token)
                categorical = pd.Categorical(values, categories=all_categories)
                dummies = pd.get_dummies(categorical, prefix=col, prefix_sep="__", dtype=int)
                # A Series/Index conversion gives stable alignment when the
                # caller supplies a non-default row index.
                dummies.index = frame.index
                encoded_columns_by_frame[index].extend(
                    [dummies[dummy_column] for dummy_column in dummies.columns]
                )
        else:
            train_values = pd.to_numeric(train[col], errors="coerce")
            median_value = train_values.median(skipna=True)
            if pd.isna(median_value):
                raise DatasetReleaseError(
                    f"Cannot median-impute numeric column {col!r}: it is entirely missing in training."
                )
            for index, frame in enumerate(raw_frames):
                encoded_columns_by_frame[index].append(
                    pd.to_numeric(frame[col], errors="coerce").fillna(float(median_value)).rename(col)
                )

    encoded_frames: list[pd.DataFrame] = []
    for frame, columns in zip(raw_frames, encoded_columns_by_frame):
        encoded = pd.concat(columns, axis=1)
        if encoded.columns.duplicated().any():
            duplicates = encoded.columns[encoded.columns.duplicated()].tolist()
            raise DatasetReleaseError(f"One-hot encoding produced duplicate feature columns: {duplicates}")
        if len(encoded) != len(frame):
            raise DatasetReleaseError("One-hot encoding changed the input row count")
        encoded_frames.append(encoded)
    return tuple(encoded_frames)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Explicit corrected validation or intact paper_reference legacy audit (no training)")
    parser.add_argument("--dataset-release", required=True)
    parser.add_argument("--dataset-version", choices=["corrected", "paper_reference"], required=True)
    parser.add_argument("--legacy-audit", action="store_true", help="Required for intact 50-predictor paper_reference audit")
    parser.add_argument("--split", choices=["train", "validation", "val", "test"], default="test")
    args = parser.parse_args()
    if args.dataset_version == "paper_reference":
        if not args.legacy_audit:
            parser.error("paper_reference requires --legacy-audit; it is not strict training input")
        audit = load_legacy_audit_split(args.split, args.dataset_release)
        result = {"status": "legacy_audit_loaded_not_strict_trainable", "rows": len(audit.features),
                  "predictors": len(audit.features.columns) - 1, "dataset_provenance": audit.provenance}
    else:
        if args.legacy_audit:
            parser.error("--legacy-audit selects paper_reference only")
        result = validate_dataset_release(args.dataset_release)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
