#!/usr/bin/env python3
"""ECtHR-NPD dataset loader.

These helpers read the fixed chronological splits from the public dataset
release, keep targets separate from model inputs, and reject strict feature
columns whose names indicate Article 41, operative-award, claim,
target-derived, or applicant-identifying leakage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


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
    "perapp_unique_beneficiary_category_count",
    "perapp_has_joint_beneficiary_category_flag",
    "rl_violation_type_procedural_ratio",
    "rl_violation_type_both_ratio",
    "rl_violation_duration_months",
    "rl_violation_duration_months_missing_ratio",
)

EXTERNAL_FACTOR_COLUMNS = (
    "itemid",
    "hudoc_url",
    "split",
    "test_view",
    "test_challenging_view",
    "respondent_state",
    "country_alpha2",
    "judgment_year",
    "gdp_per_capita_current_usd",
    "gdp_constant_2015_usd",
    "gdp_per_capita_log1p",
    "gdp_constant_2015_log1p",
)

STRUCTURED_EXTERNAL_FACTOR_COLUMNS = (
    "gdp_per_capita_log1p",
    "gdp_constant_2015_log1p",
)


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


def resolve_dataset_release(dataset_release: str | Path | None = None) -> Path:
    """Resolve the public release directory in the repo or inside a bundle."""
    candidates: list[Path] = []
    if dataset_release:
        candidates.append(Path(dataset_release).expanduser())
    if os.environ.get("ECTHR_NPD_DATASET_RELEASE"):
        candidates.append(Path(os.environ["ECTHR_NPD_DATASET_RELEASE"]).expanduser())

    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        candidates.extend(
            [
                parent / "dataset_release",
                parent / "baselines" / "data_release_paper_20260526",
                parent / "data_release_paper_20260526",
            ]
        )

    for candidate in candidates:
        if (candidate / "data" / "ecthr_npd_cases.csv").exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate ECtHR-NPD dataset_release. Searched:\n{searched}")


def assert_no_forbidden_strict_columns(columns: Iterable[str]) -> None:
    bad: list[str] = []
    for column in columns:
        lowered = column.lower()
        if column == "itemid":
            continue
        if any(term in lowered for term in FORBIDDEN_STRICT_INPUT_TERMS):
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


def load_structured_tree_split(
    split: str,
    dataset_release: str | Path | None = None,
    *,
    validate: bool = True,
) -> DatasetSplit:
    """Load one fixed split from model_inputs/structured_tree.

    The returned ``DatasetSplit.X`` frame is strict model input. Targets are
    available only through ``DatasetSplit.y_amount_eur`` and
    ``DatasetSplit.y_binary``.
    """
    release = resolve_dataset_release(dataset_release)
    split_key = _normalise_split_name(split)
    file_stem = SPLIT_TO_FILE_STEM[split_key]

    features = _read_csv(release / "model_inputs" / "structured_tree" / "features" / f"{file_stem}.csv")
    targets = _read_csv(release / "model_inputs" / "structured_tree" / "targets" / f"{file_stem}.csv")
    cases = _read_csv(release / "data" / "ecthr_npd_cases.csv")

    if validate:
        expected_count = EXPECTED_SPLIT_COUNTS[split_key]
        if len(features) != expected_count:
            raise DatasetReleaseError(f"{split_key} feature count {len(features)} != {expected_count}")
        if len(targets) != expected_count:
            raise DatasetReleaseError(f"{split_key} target count {len(targets)} != {expected_count}")
        if list(features["itemid"]) != list(targets["itemid"]):
            raise DatasetReleaseError(f"{split_key} feature/target itemid order mismatch")
        if features["itemid"].duplicated().any():
            raise DatasetReleaseError(f"{split_key} features contain duplicate itemid values")
        assert_no_forbidden_strict_columns(features.columns)

        canonical_split = "validation" if split_key == "val" else split_key
        case_subset = cases[cases["split"] == canonical_split]
        if len(case_subset) != expected_count:
            raise DatasetReleaseError(f"{split_key} case index count {len(case_subset)} != {expected_count}")

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
) -> dict[str, DatasetSplit]:
    """Load train, validation, and test splits from the dataset release."""
    return {
        "train": load_structured_tree_split("train", dataset_release, validate=validate),
        "validation": load_structured_tree_split("validation", dataset_release, validate=validate),
        "test": load_structured_tree_split("test", dataset_release, validate=validate),
    }


def load_external_factors(dataset_release: str | Path | None = None) -> pd.DataFrame:
    """Load respondent-state/year economic covariates for all cases."""
    release = resolve_dataset_release(dataset_release)
    external = _read_csv(release / "model_inputs" / "external_factors" / "economic_covariates.csv")
    missing = [column for column in EXTERNAL_FACTOR_COLUMNS if column not in external.columns]
    if missing:
        raise DatasetReleaseError("External factor table missing columns: " + ", ".join(missing))
    if len(external) != 14575:
        raise DatasetReleaseError(f"external factor row_count {len(external)} != 14575")
    assert_no_forbidden_strict_columns(
        column for column in external.columns if column not in {"hudoc_url", "split", "test_view", "test_challenging_view"}
    )
    return external.copy()


def input_contract_summary() -> dict[str, object]:
    """Return the shared strict input groups used by all model families."""
    return {
        "allowed_input_groups": {
            "safe_metadata": list(SAFE_METADATA_COLUMNS),
            "violated_articles": list(VIOLATED_ARTICLE_COLUMNS),
            "case_facts_or_serialized_inputs": list(SERIALIZED_CASE_FACT_COLUMNS),
            "external_factors": list(EXTERNAL_FACTOR_COLUMNS),
        },
        "structured_tree_external_factor_columns": list(STRUCTURED_EXTERNAL_FACTOR_COLUMNS),
        "excluded_strict_input_terms": list(FORBIDDEN_STRICT_INPUT_TERMS),
        "target_columns": ["y_amount_eur", "y_binary"],
        "notes": (
            "Targets, split/view labels, Article 41/50 text, operative clauses, "
            "claims, award-side fields, and target-derived fields are not model features."
        ),
    }


def validate_dataset_release(dataset_release: str | Path | None = None) -> dict[str, object]:
    """Validate row counts and diagnostic test views."""
    release = resolve_dataset_release(dataset_release)
    cases = _read_csv(release / "data" / "ecthr_npd_cases.csv")
    external = load_external_factors(release)
    split_counts = cases["split"].value_counts().to_dict()
    test = cases[cases["split"] == "test"]
    test_views = test["test_view"].value_counts().to_dict()
    challenging = int(pd.to_numeric(test["test_challenging_view"], errors="coerce").fillna(0).sum())

    errors: list[str] = []
    if len(cases) != 14575:
        errors.append(f"row_count {len(cases)} != 14575")
    expected_splits = {"train": 10217, "validation": 1461, "test": 2897}
    if {k: int(v) for k, v in split_counts.items()} != expected_splits:
        errors.append(f"split_counts {split_counts} != {expected_splits}")
    if {k: int(v) for k, v in test_views.items()} != EXPECTED_TEST_VIEWS:
        errors.append(f"test_view_counts {test_views} != {EXPECTED_TEST_VIEWS}")
    if challenging != EXPECTED_CHALLENGING_COUNT:
        errors.append(f"challenging_count {challenging} != {EXPECTED_CHALLENGING_COUNT}")

    for split in ("train", "validation", "test"):
        load_structured_tree_split(split, release, validate=True)

    if set(external["itemid"]) != set(cases["itemid"]):
        errors.append("external factor itemids do not match canonical cases")

    return {
        "status": "PASS" if not errors else "FAIL",
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
) -> tuple[pd.DataFrame, ...]:
    """Encode object/category columns with train-only vocabularies.

    Unknown categories in validation/test are encoded as -1. Numeric
    columns are coerced to numbers and missing values are filled with 0.
    """
    encoded_frames = [train.copy(), *(frame.copy() for frame in others)]
    categorical_cols = [
        col
        for col in train.columns
        if str(train[col].dtype) in {"object", "category", "string"} or train[col].dtype == bool
    ]

    for col in train.columns:
        if col in categorical_cols:
            categories = {
                value: idx
                for idx, value in enumerate(
                    pd.Series(train[col]).fillna("<MISSING>").astype(str).drop_duplicates().tolist()
                )
            }
            for frame in encoded_frames:
                frame[col] = frame[col].fillna("<MISSING>").astype(str).map(categories).fillna(-1).astype(int)
        else:
            for frame in encoded_frames:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    return tuple(encoded_frames)
