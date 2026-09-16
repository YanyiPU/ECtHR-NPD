"""Read the single public two-table dataset without private encoded matrices.

This is a documented NEW feature representation, not a recovered X0--X3 map.
The public CSVs stay human-readable; transformations happen only in memory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ecthr_public_tables", CODE_ROOT / "public_tables.py")
PUBLIC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PUBLIC)

METADATA = (
    "country_alpha2", "hudoc_decision_body", "case_importance",
    "has_separate_opinion", "represented", "num_applicants",
    "hudoc_application_count", "judgment_year", "judgment_month",
)
MERITS = ("num_violations_found", "violation_type", "violated_articles",
          "violated_articles_count", "violation_duration_months")
FACTS = ("applicant_sex", "applicant_age_group", "applicant_birth_years",
         "applicant_nationality_scope")
EXTERNAL = ("gdp_per_capita_log1p", "gdp_constant_2015_log1p")
FEATURES = (*METADATA, *MERITS, *FACTS, *EXTERNAL)
CATEGORICAL = (
    "country_alpha2", "hudoc_decision_body", "case_importance",
    "has_separate_opinion", "represented", "violation_type", "violated_articles", *FACTS,
)


def is_public_root(root):
    return root is not None and (Path(root) / "data/case_level.csv").is_file()


def tables(root):
    """The public validator checks schema, identities, values, counts and hashes."""
    return PUBLIC.load_tables(Path(root))


def canonical_cases(root):
    rows, _ = tables(root)
    result = []
    for row in rows:
        # itemid is solely the legacy runners' JOIN KEY alias for public case_id.
        current = {**row, "itemid": row["case_id"], "judgementdate": row["judgment_date"],
                   "test_challenging_view": int(row["test_challenging_view"] == "yes")}
        date.fromisoformat(current["judgementdate"])
        result.append(current)
    return pd.DataFrame(result)


def _number(value):
    if value in (None, "", "unknown"):
        return float("nan")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Public numeric predictors must be finite nonnegative values or unknown")
    return result


def feature_frame(cases):
    rows = []
    for row in cases.to_dict("records"):
        features = {"itemid": row["itemid"]}
        for column in (*METADATA, *MERITS, *FACTS):
            value = date.fromisoformat(row["judgment_date"]).month if column == "judgment_month" else row[column]
            features[column] = str(value) if column in CATEGORICAL else _number(value)
        for column, source in zip(EXTERNAL, ("gdp_per_capita_current_usd", "gdp_constant_2015_usd")):
            features[column] = math.log1p(_number(row[source]))
        rows.append(features)
    return pd.DataFrame(rows, columns=["itemid", *FEATURES])


def feature_schema():
    sources = {**dict.fromkeys(METADATA, "metadata"), **dict.fromkeys(MERITS, "merits"),
               **dict.fromkeys(FACTS, "facts"), **dict.fromkeys(EXTERNAL, "external_covariate")}
    return {"profile": "public_structured", "profile_status": "new_public_table_representation",
            "historical_reproduction": False, "columns": list(FEATURES),
            "feature_columns": list(FEATURES), "feature_count": len(FEATURES),
            "categorical_columns": list(CATEGORICAL),
            "lineage": {column: {"source_group": group,
                "source_reference": "data/case_level.csv; Table 11 permitted information groups"}
                for column, group in sources.items()},
            "transformations": {"judgment_month": "month of judgment_date",
                "gdp_per_capita_log1p": "log1p(gdp_per_capita_current_usd)",
                "gdp_constant_2015_log1p": "log1p(gdp_constant_2015_usd)"},
            "omitted": ["case_id", "split", "test_view", "test_challenging_view", "beneficiary_type",
                        "has_joint_beneficiary", "y_amount_eur", "y_binary", "target_status",
                        "all applicant-level allocations and allocation status fields"],
            "notes": "Readable demographic summaries are categorical; missing numeric values are imputed from train only. No reconstruction of historical 48/50 predictors or X0--X3 is asserted."}


def contract(root):
    cases, _ = tables(root)
    return {"dataset_version": "unified", "release_id": "ECtHR-NPD",
            "counts": {"total": len(cases), "splits": {name: sum(r["split"] == name for r in cases)
                         for name in ("train", "validation", "test")},
                       "test_views": {name: sum(r["split"] == "test" and r["test_view"] == name for r in cases)
                                      for name in ("ID", "OOD")},
                       "challenging": sum(r["test_challenging_view"] == "yes" for r in cases)}}


def provenance(root, input_files=None):
    root = Path(root)
    cases, _ = tables(root)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("data/case_level.csv", "data/applicant_level.csv")}
    return {"dataset_version": "unified", "release_id": "ECtHR-NPD",
            "dataset_fingerprint_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "canonical_input_sha256": hashes, "integrity_status": "public_tables_validated",
            "training_status": "new_public_table_experiment_not_historical_reproduction",
            "historical_reproduction": False,
            "label_policy": "retain the selected 14575-case cohort with two corrected case totals; five source-ambiguous reference labels remain",
            "target_status_counts": dict(Counter(row["target_status"] for row in cases)),
            "retained_ambiguous_case_ids": sorted(row["case_id"] for row in cases if row["target_status"] in {"mixed_head_reference_label", "unresolved_reference_label"}),
            "run_input_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name, path in (input_files or {}).items() if path is not None}}


def validate_targets(root, itemids, amounts, splits=None, require_all=False):
    cases = canonical_cases(root).set_index("itemid")
    ids, amounts = list(map(str, itemids)), list(amounts)
    if len(ids) != len(set(ids)) or len(ids) != len(amounts) or set(ids) - set(cases.index):
        raise ValueError("Supplied target IDs differ from the public dataset")
    if require_all and set(ids) != set(cases.index):
        raise ValueError("Supplied targets do not cover the public dataset")
    if any(not math.isfinite(float(value)) or float(value) != float(cases.loc[itemid, "y_amount_eur"])
           for itemid, value in zip(ids, amounts)):
        raise ValueError("Supplied y_amount_eur differs from the public dataset")
    if splits is not None and ["validation" if value == "val" else value for value in splits] != cases.loc[ids, "split"].tolist():
        raise ValueError("Supplied splits differ from the public dataset")
    return provenance(root)
