"""Standard-library reader and validator for the two-table ECtHR-NPD releases.

No network, model calls, preprocessing or private-source access. Run:
python code/public_tables.py /path/to/release
"""
import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

CASE_COLUMNS = [
    'case_id', 'split', 'test_view', 'test_challenging_view', 'judgment_date', 'judgment_year',
    'respondent_state', 'country_alpha2', 'hudoc_decision_body', 'court_formation',
    'is_grand_chamber', 'case_importance', 'has_separate_opinion', 'represented',
    'hudoc_application_count', 'num_applicants', 'num_violations_found', 'has_mixed_outcome', 'violated_articles',
    'violated_articles_count', 'violation_type', 'violation_duration_months',
    'applicant_sex', 'applicant_age_group', 'applicant_birth_years',
    'applicant_nationality_scope', 'beneficiary_type', 'has_joint_beneficiary',
    'gdp_per_capita_current_usd', 'gdp_constant_2015_usd',
    'y_amount_eur', 'y_binary', 'target_status',
]
APPLICANT_COLUMNS = [
    'applicant_id', 'case_id', 'split', 'applicant_name', 'applicant_type', 'sex', 'age_group',
    'birth_year', 'nationality', 'nationality_scope', 'beneficiary_type', 'npd_award_eur', 'npd_award_scope', 'npd_award_status',
]
SPLITS = {'train', 'validation', 'test'}
AMOUNT_STATUSES = {'retained_source_link', 'verified_source_link', 'verified_source_allocation'}
MISSING_AMOUNT_STATUSES = {'no_personal_allocation_found_in_retained_data',
                          'allocation_link_or_scope_unresolved', 'group_or_organisation_record',
                          'reference_case_target_unresolved'}
FORBIDDEN_VALUE = re.compile(r'[^\s@]+@[^\s@]+\.[^\s@]+')


def _read(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV columns')
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError('Malformed CSV row')
        return reader.fieldnames, rows


def _number(value, allow_unknown=False):
    if allow_unknown and value == 'unknown':
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('Invalid numeric value') from exc
    if not result.is_finite() or result < 0:
        raise ValueError('Non-finite or negative numeric value')
    return result


def validate(root, verify_manifest=True):
    root = Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlinks are not allowed')
    csvs = {str(p.relative_to(root)) for p in root.rglob('*.csv')}
    if csvs != {'data/case_level.csv', 'data/applicant_level.csv'}:
        raise ValueError('A release must contain exactly the two declared CSVs')
    ch, cases = _read(root / 'data/case_level.csv')
    ah, applicants = _read(root / 'data/applicant_level.csv')
    if ch != CASE_COLUMNS or ah != APPLICANT_COLUMNS:
        raise ValueError('Unexpected two-table schema')
    by_case = {r['case_id']: r for r in cases}
    if len(by_case) != len(cases) or len({r['applicant_id'] for r in applicants}) != len(applicants):
        raise ValueError('Duplicate primary key')
    if any(not re.fullmatch(r'\d{3}-\d{4,9}', r['case_id']) for r in cases):
        raise ValueError('Invalid HUDOC case ID')
    from datetime import date
    for r in cases:
        if r['split'] not in SPLITS or r['violation_type'] not in {'substantive', 'procedural', 'both', 'unknown', 'other'}:
            raise ValueError('Invalid case category')
        if str(date.fromisoformat(r['judgment_date']).year) != r['judgment_year']:
            raise ValueError('Judgment year/date disagreement')
        codes = {s.strip() for s in r['violated_articles'].split(';') if s.strip()}
        if len(codes) != int(r['violated_articles_count']):
            raise ValueError('Distinct violated-article count disagreement')
        application_count = int(r['hudoc_application_count'])
        if application_count < 1:
            raise ValueError('Missing HUDOC application count')
        challenging = r['split'] == 'test' and (r['is_grand_chamber'] == 'yes' or (application_count > 1 and len(codes) > 1))
        if (r['test_challenging_view'] == 'yes') != challenging:
            raise ValueError('Challenging selector disagreement')
        if r['target_status'] not in {'retained_label', 'corrected_amount', 'mixed_head_reference_label', 'unresolved_reference_label'}:
            raise ValueError('Unrecognized target status')
        amount = _number(r['y_amount_eur'])
        if r['y_binary'] != str(int(amount > 0)):
            raise ValueError('Case target/binary disagreement')
        _number(r['violation_duration_months'], True)
        if r['test_challenging_view'] not in {'yes', 'no'}:
            raise ValueError('Invalid Challenging indicator')
        if r['split'] != 'test' and (r['test_challenging_view'] != 'no' or r['test_view'] != 'not_applicable'):
            raise ValueError('Diagnostic view outside test split')
    linked_sums = defaultdict(Decimal)
    for r in applicants:
        if not re.fullmatch(r'applicant_\d{6,}', r['applicant_id']):
            raise ValueError('Invalid public applicant ID')
        if r['case_id'] not in by_case or r['split'] != by_case[r['case_id']]['split']:
            raise ValueError('Applicant case foreign key or split mismatch')
        if r['applicant_name'] != '[MASKED]':
            raise ValueError('Applicant name is not masked')
        if r['applicant_type'] not in {'applicant_source_record','applicant_estate_record','organisation_record','group_or_ambiguous_record'}:
            raise ValueError('Invalid applicant source-unit type')
        if r['npd_award_scope'] not in {'individual','applicant_estate','group_or_organisation','unknown'}:
            raise ValueError('Invalid award recipient scope')
        if r['npd_award_scope'] == 'applicant_estate' and r['applicant_type'] != 'applicant_estate_record':
            raise ValueError('Estate award must retain estate unit type')
        if r['sex'] not in {'male', 'female', 'unknown', 'mixed'}:
            raise ValueError('Invalid sex category')
        if r['sex'] == 'mixed' and r['applicant_type'] != 'group_or_ambiguous_record':
            raise ValueError('Mixed sex must not be presented as one individual')
        if r['age_group'] not in {'child', 'adolescent', 'adult', 'elderly', 'unknown'}:
            raise ValueError('Invalid age group')
        if r['nationality_scope'] not in {'single', 'multiple', 'stateless', 'unknown'}:
            raise ValueError('Invalid nationality scope')
        if r['birth_year'] != 'unknown' and not re.fullmatch(r'\d{4}', r['birth_year']):
            raise ValueError('Invalid birth year')
        if r['npd_award_status'] in MISSING_AMOUNT_STATUSES:
            if r['npd_award_eur'] != 'unknown' or r['npd_award_scope'] != 'unknown':
                raise ValueError('Unresolved allocation must not have a personal amount')
        elif r['npd_award_status'] in AMOUNT_STATUSES:
            if r['npd_award_scope'] == 'unknown':
                raise ValueError('Numeric allocation must have a recipient scope')
            amount = _number(r['npd_award_eur'])
            linked_sums[r['case_id']] += amount
            if amount > _number(by_case[r['case_id']]['y_amount_eur']):
                raise ValueError('Individual amount exceeds case amount')
        else:
            raise ValueError('Invalid allocation status')
    if any(total > _number(by_case[case_id]['y_amount_eur']) for case_id, total in linked_sums.items()):
        raise ValueError('Retained allocation sum exceeds case target; review source and unit scope')
    for r in cases + applicants:
        if any(FORBIDDEN_VALUE.search(v) for v in r.values()):
            raise ValueError('Unexpected contact-like value in CSV')
    counts = {'case_rows': len(cases), 'applicant_rows': len(applicants),
              'splits': dict(Counter(r['split'] for r in cases)),
              'challenging': sum(r['test_challenging_view'] == 'yes' for r in cases)}
    if verify_manifest:
        manifest = json.loads((root / 'manifest.json').read_text())
        tracked = manifest['file_sha256']
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if actual != set(tracked) | {'manifest.json'}:
            raise ValueError('Unlisted or missing release file')
        for n, h in tracked.items():
            path = Path(n)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('Unsafe manifest path')
            if hashlib.sha256((root / n).read_bytes()).hexdigest() != h:
                raise ValueError('Release hash mismatch')
        if counts != manifest['counts']:
            raise ValueError('Manifest count mismatch')
    return {'status': 'passed', **counts}


def load_tables(root, split=None):
    validate(root)
    cases = _read(Path(root) / 'data/case_level.csv')[1]
    applicants = _read(Path(root) / 'data/applicant_level.csv')[1]
    if split is not None:
        if split not in SPLITS:
            raise ValueError('split must be train, validation or test')
        cases = [r for r in cases if r['split'] == split]
        applicants = [r for r in applicants if r['split'] == split]
    return cases, applicants


def validate_zip(path, release_root):
    """Verify archived file inventory and bytes against the already validated release."""
    result = validate(release_root)
    root = Path(release_root)
    with zipfile.ZipFile(path) as archive:
        expected = {str(p.relative_to(root)): p for p in root.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if set(archive.namelist()) != set(expected) or len(archive.namelist()) != len(expected):
            raise ValueError('ZIP file inventory mismatch')
        for name, p in expected.items():
            if archive.read(name) != p.read_bytes():
                raise ValueError('ZIP payload differs from validated release')
    return {**result, 'zip_bytes_match': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release_root', type=Path)
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_zip(args.zip, args.release_root) if args.zip else validate(args.release_root), indent=2))
