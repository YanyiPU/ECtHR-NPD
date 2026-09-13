# Release status — 2026-09-14

## Updated now

- v1.1 documentation records each of the five exclusions/quarantines, its
  rationale, the treatment and the evidence needed for reconsideration.
- It explains the 14,575→14,570 case and 699→698 Challenging changes without
  changing surviving splits or confusing missing amounts with zero.
- Case references use project-assigned IDs only; source crosswalks, names,
  judgment quotations, source anchors and internal audit ledgers are excluded.

## Dataset availability

The prepared latest v1.1 two-table dataset is in the existing private HF review
repository. The public GitHub and public HF legacy dataset payloads have not
been replaced by this documentation commit. The files under dataset_release
remain historical and must not be cited as the corrected v1.1 two-table release.

The new applicant-level release removes direct identifiers but retains exact
birth years, sex categories, nationality scope and award amounts. These can be
linkable to public decisions. Publication is held pending explicit confirmation
that the prior pause is lifted for these precise fields. This document does not
grant a data/code licence or assert institutional/legal approval or anonymity.

## Version policy

HF's designated latest repository should expose only the latest accepted version
on main, with historical commits/tags preserved, not multiple copies under data/.
The private review repository currently follows this policy. Do not assume the
legacy public HF dataset has already been updated. Cite an exact commit and data
version; documentation-only revisions do not silently change data labels.

Original historical experiment evidence stays separate; known missing review
ledgers and model artifacts have not been fabricated or certified as recovered.
