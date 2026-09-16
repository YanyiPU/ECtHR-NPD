"""Hash-bound empirical priors. No API or provider dependency."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

TABLE_NAMES = (
    "article_award_distribution_train.csv", "country_award_distribution_train.csv",
    "article_country_award_distribution_train.csv", "article_single_violation_stats_train.csv",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def amount_text(value):
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError("invalid prior target amount") from exc
    if not number.is_finite() or number < 0:
        raise ValueError("prior targets must be finite nonnegative EUR")
    return format(number.normalize(), "f")


def articles(value):
    if isinstance(value, str):
        text = value.strip()
        value = json.loads(text) if text.startswith("[") else text.split(";")
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("prior Articles must be a list or semicolon-separated string")
    return sorted({str(article).strip().upper() for article in value if str(article).strip()})


def read_training_inputs(case_path, target_path):
    """Strict train membership and exact target coverage; no row-label fallback."""
    cases = {}
    seen = set()
    with Path(case_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            itemid = str(row.get("itemid") or "").strip()
            if not itemid or itemid in seen:
                raise ValueError("blank/duplicate case ID in prior source")
            seen.add(itemid)
            split = str(row.get("split") or row.get("split_role") or "").strip()
            if split not in {"train", "validation", "val", "test"}:
                raise ValueError("explicit recognized split is mandatory for prior source rows")
            if row.get("split") and row.get("split_role") and row["split"] != row["split_role"]:
                raise ValueError("conflicting prior source split fields")
            if split == "train":
                cases[itemid] = row
    targets = {}
    with Path(target_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            itemid = str(row.get("itemid") or "").strip()
            if not itemid or itemid in targets:
                raise ValueError("blank/duplicate train target ID")
            if row.get("split") not in (None, "", "train"):
                raise ValueError("non-training label in prior target file")
            amount_text(row.get("y_amount_eur"))
            if row.get("y_binary") not in (None, "") and str(row["y_binary"]) not in {str(int(Decimal(amount_text(row["y_amount_eur"])) > 0)), str(float(Decimal(amount_text(row["y_amount_eur"])) > 0))}:
                raise ValueError("prior target binary label contradicts EUR amount")
            targets[itemid] = row
    if not cases or set(cases) != set(targets):
        raise ValueError("prior train-case and target ID coverage differs")
    for itemid, row in cases.items():
        if not articles(row.get("violated_articles")) or not str(row.get("country_alpha2") or "").strip():
            raise ValueError("training case missing Articles/country for complete empirical priors")
        if row.get("y_amount_eur") not in (None, "") and amount_text(row["y_amount_eur"]) != amount_text(targets[itemid]["y_amount_eur"]):
            raise ValueError("case source and supplied training target disagree")
    return cases, targets


def cohort_pin(case_path, target_path):
    cases, targets = read_training_inputs(case_path, target_path)
    return cohort_pin_records(cases, targets)


def cohort_pin_records(cases, targets):
    ids = sorted(cases)
    metadata = [{"itemid": itemid, "country": str(cases[itemid]["country_alpha2"]).strip().upper(),
                 "articles": articles(cases[itemid]["violated_articles"]),
                 "judgementdate": str(cases[itemid].get("judgementdate") or "").strip()} for itemid in ids]
    amounts = [{"itemid": itemid, "y_amount_eur": amount_text(targets[itemid]["y_amount_eur"])} for itemid in ids]
    provenance = [{"itemid": itemid, "y_source": str(targets[itemid].get("y_source") or "").strip()} for itemid in ids]
    return {"train_count": len(ids), "train_itemids_sha256": canonical_sha256(ids),
            "train_metadata_sha256": canonical_sha256(metadata), "train_targets_sha256": canonical_sha256(amounts),
            "train_label_provenance_sha256": canonical_sha256(provenance)}


def bind_selected_training_inputs(dataset_release, case_path=None, target_path=None, *, dataset_version="corrected"):
    """Canonical selection plus semantic cohort binding; preserve all prior hash gates."""
    baselines = Path(__file__).resolve().parents[1] / "code/baselines"
    if str(baselines) not in sys.path:
        sys.path.insert(0, str(baselines))
    from data.data_loader import resolve_dataset_release, require_corrected_version, dataset_provenance
    require_corrected_version(dataset_version)
    release = resolve_dataset_release(dataset_release, dataset_version=dataset_version)
    from data.public_adapter import is_public_root, canonical_cases
    if is_public_root(release):
        if not case_path or not target_path:
            raise ValueError("For the public dataset, first run code/prepare_experiments.py prepare, then supply train_cases.csv and train_targets.csv")
        case_path, target_path = Path(case_path), Path(target_path)
        rows = canonical_cases(release)
        cases = {row["itemid"]: row for row in rows.loc[rows["split"] == "train"].to_dict("records")}
        targets = {itemid: {"itemid": itemid, "y_amount_eur": row["y_amount_eur"]} for itemid, row in cases.items()}
        canonical = cohort_pin_records(cases, targets)
        supplied = cohort_pin(case_path, target_path)
        if any(canonical[key] != supplied[key] for key in set(canonical) - {"train_label_provenance_sha256"}):
            raise ValueError("Supplied prior training cohort/metadata/targets differ from the public dataset")
        return release, case_path, target_path, dataset_provenance(release,
            input_files={"train_cases": case_path, "train_targets": target_path})
    canonical_cases = release / "data/ecthr_npd_cases.csv"
    canonical_targets = release / "model_inputs/structured_tree/targets/train.csv"
    case_path, target_path = Path(case_path or canonical_cases), Path(target_path or canonical_targets)
    canonical = cohort_pin(canonical_cases, canonical_targets)
    supplied = cohort_pin(case_path, target_path)
    # Optional richer historical y_source evidence is not a target/metadata override.
    fields = set(canonical) - {"train_label_provenance_sha256"}
    if any(canonical[key] != supplied[key] for key in fields):
        raise ValueError("supplied prior training cohort/metadata/targets differ from selected dataset version")
    provenance = dataset_provenance(release, dataset_version=dataset_version,
                                   input_files={"train_cases": case_path, "train_targets": target_path})
    return release, case_path, target_path, provenance


def verify_prior_bundle(prior_dir, case_path, target_path, *, dataset_version=None):
    prior_dir = Path(prior_dir)
    path = prior_dir / "PRIOR_MANIFEST.v1.json"
    if not path.is_file():
        raise ValueError("prior manifest is missing; rebuild a pinned candidate bundle")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if dataset_version is not None and manifest.get("dataset_version") not in (None, dataset_version):
        raise ValueError("empirical prior dataset version mismatch")
    if manifest.get("manifest_version") != "2.0.0" or not manifest.get("cohort_pin"):
        raise ValueError("legacy/unpinned priors cannot be used for a current-cohort ReAct run")
    current = cohort_pin(case_path, target_path)
    if current != manifest["cohort_pin"]:
        raise ValueError("empirical prior training metadata/target hash mismatch; rebuild for this dataset")
    expected = manifest.get("artifact_sha256", {})
    if set(expected) != set(TABLE_NAMES):
        raise ValueError("prior bundle must pin all four empirical tables")
    for name in TABLE_NAMES:
        if not (prior_dir / name).is_file() or file_sha256(prior_dir / name) != expected[name]:
            raise ValueError(f"prior artifact missing or hash mismatch: {name}")
    return {"status": "verified", "cohort_pin": current, "manifest_sha256": file_sha256(path),
            "artifact_sha256": expected, "scope": "article_country_article_country_and_legacy_article_statistics_only"}
