"""Actual packaged data invariants (no APIs, training or source-name access)."""
import importlib.util
import unittest
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unified_tables_test', ROOT/'code/public_tables.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

class PublicDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = p._read(ROOT/'data/case_level.csv')[1]
        cls.apps = p._read(ROOT/'data/applicant_level.csv')[1]

    def test_integrity_and_challenging(self):
        counts = p.validate(ROOT, verify_manifest=False)
        self.assertEqual((counts['case_rows'],counts['applicant_rows'],counts['challenging']), (14575,44581,699))

    def test_supported_awards(self):
        totals = {r['case_id']:Decimal(r['y_amount_eur']) for r in self.cases}
        amounts = defaultdict(list)
        for a in self.apps:
            if a['npd_award_eur'] != 'unknown':
                amounts[a['case_id']].append(Decimal(a['npd_award_eur']))
        for cid, count, each in [('001-210465',3,800),('001-142400',8,12000),
                                 ('001-196605',2,3000),('001-152888',3,7500),
                                 ('001-89058',2,3000),('001-72236',5,4000)]:
            self.assertEqual(amounts[cid], [Decimal(each)]*count)
            self.assertEqual(sum(amounts[cid]), totals[cid])

    def test_scopes_and_names(self):
        self.assertTrue(all(r['applicant_name']=='[MASKED]' for r in self.apps))
        estates = [r for r in self.apps if r['npd_award_scope']=='applicant_estate']
        self.assertEqual(len(estates),2)
        self.assertTrue(all(r['case_id']=='001-72236' and r['npd_award_eur']=='4000' for r in estates))

    def test_five_unresolved_retained(self):
        flagged = {r['case_id'] for r in self.cases if r['target_status'] in {'mixed_head_reference_label','unresolved_reference_label'}}
        self.assertEqual(flagged, {'001-95089','001-195534','001-71339','001-71742','001-228159'})
        self.assertEqual(len([r for r in self.apps if r['case_id'] in flagged]),24)
        self.assertTrue(all(r['npd_award_eur']=='unknown' for r in self.apps if r['case_id'] in flagged))

    def test_readable_columns_only(self):
        self.assertFalse(any('_ratio' in c or 'log1p' in c or '__' in c for c in p.CASE_COLUMNS+p.APPLICANT_COLUMNS))
        self.assertEqual([len(p.CASE_COLUMNS),len(p.APPLICANT_COLUMNS)],[33,14])

if __name__=='__main__':
    unittest.main()
