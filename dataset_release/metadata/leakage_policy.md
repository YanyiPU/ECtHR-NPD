# Leakage Policy

Strict prediction inputs exclude compensation-bearing material:

- Article 41 / Article 50 compensation text
- operative clauses and appendix award tables
- claimed amounts and claim-state fields
- award-side fields and zero-award rationales
- raw extractor outputs and repair/debug fields
- per-applicant award allocations
- target values and split/view labels

The canonical table includes targets for supervised training and
evaluation. Target columns must not be used as features.
