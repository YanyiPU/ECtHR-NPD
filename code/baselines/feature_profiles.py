"""Explicit, evidence-bearing feature contracts for NEW experiments.

Table 28 names X0--X3 but does not give an exact column-to-profile mapping.
This module validates a supplied mapping; it never invents the historical one.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from data.data_loader import assert_no_forbidden_strict_columns

PROFILE_DESCRIPTIONS = {
    "X0": "case metadata", "X1": "applicant-enhanced",
    "X2": "reasoning-enhanced", "X3": "full feature set",
    "metadata_knn": "explicit metadata-only retrieval features",
    "structured_knn": "explicit permissible structured retrieval features",
    "public_structured": "documented new-run projection of the two readable public tables",
}
BLOCKED_PROVENANCE_COLUMNS = {
    "perapp_unique_beneficiary_category_count",
    "perapp_has_joint_beneficiary_category_flag",
}
FORBIDDEN_AUXILIARY = {"itemid", "case_id", "split", "test_view", "test_challenging_view", "hudoc_url"}
ALLOWED_SOURCES = {"metadata", "facts", "merits", "external_covariate"}


def validate_profile(profile: dict[str, Any]) -> list[str]:
    name = profile.get("profile")
    if name not in PROFILE_DESCRIPTIONS:
        raise ValueError("An explicit recognized feature profile is required")
    columns = profile.get("columns")
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
        raise ValueError("profile.columns must be a nonempty ordered string list")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate feature columns")
    assert_no_forbidden_strict_columns(columns)
    if set(columns) & (BLOCKED_PROVENANCE_COLUMNS | FORBIDDEN_AUXILIARY):
        raise ValueError("identifier, split/view or unverified allocation-derived feature")
    categorical = profile.get("categorical_columns", [])
    if not isinstance(categorical, list) or not set(categorical) <= set(columns):
        raise ValueError("categorical_columns must be a subset of columns")
    lineage = profile.get("lineage", {})
    for column in columns:
        record = lineage.get(column, {})
        if record.get("source_group") not in ALLOWED_SOURCES or not record.get("source_reference"):
            raise ValueError(f"missing permitted source lineage for {column}")
        if name == "metadata_knn" and record["source_group"] != "metadata":
            raise ValueError("metadata_knn cannot include non-metadata features")
    # Historical equivalence requires original feature/run evidence elsewhere;
    # a user-authored JSON flag is not a certification.
    if profile.get("historical_reproduction"):
        raise ValueError("this validator cannot certify historical reproduction")
    return columns


class TrainOnlyPreprocessor:
    """Train-median/std numeric scaling and train-only categorical vocabulary.

    Scaling and missing-value handling choices are explicit NEW-run choices,
    not an assertion that the unavailable historical kNN pipeline used these.
    """
    def fit(self, rows: list[dict[str, Any]], profile: dict[str, Any]):
        if not rows:
            raise ValueError("preprocessing requires training rows")
        self.columns = validate_profile(profile)
        self.categorical = set(profile.get("categorical_columns", []))
        self.statistics, self.categories = {}, {}
        for column in self.columns:
            if any(column not in row for row in rows):
                raise ValueError(f"missing feature column {column}")
            if column in self.categorical:
                self.categories[column] = sorted({str(row[column]) if row[column] not in (None, "") else "__MISSING__" for row in rows})
            else:
                values = np.array([self._number(row[column]) for row in rows])
                observed = values[np.isfinite(values)]
                median = float(np.median(observed)) if len(observed) else 0.0
                filled = np.where(np.isfinite(values), values, median)
                self.statistics[column] = {"median": median, "mean": float(filled.mean()), "scale": float(filled.std()) or 1.0,
                                           "all_train_missing": not bool(len(observed))}
        return self

    @staticmethod
    def _number(value: Any) -> float:
        if value is None or value == "":
            return float("nan")
        number = float(value)
        if np.isinf(number):
            raise ValueError("infinite feature value")
        return number

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        output = []
        for row in rows:
            result = []
            for column in self.columns:
                if column not in row:
                    raise ValueError(f"missing feature column {column}")
                if column in self.categorical:
                    value = str(row[column]) if row[column] not in (None, "") else "__MISSING__"
                    result.extend(float(value == level) for level in self.categories[column])
                else:
                    value = self._number(row[column])
                    stat = self.statistics[column]
                    value = value if np.isfinite(value) else stat["median"]
                    result.append((value - stat["mean"]) / stat["scale"])
            output.append(result)
        return np.asarray(output, dtype=float)

    def manifest(self) -> dict[str, Any]:
        return {"fit_split": "train", "columns": self.columns, "categorical_columns": sorted(self.categorical),
                "numeric_statistics": self.statistics, "categories": self.categories,
                "unknown_category": "all_zero", "all_missing_numeric": "zero_train_fallback"}
