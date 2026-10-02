# Paper alignment and reproduction boundary

This guide maps the paper’s methods to the public code and lists the artifacts needed for numerical reproduction.

| Paper statement | Public implementation | Notes |
|---|---|---|
| 14,575 cases; Table 1 splits | Full original cohort, unchanged membership | Checked on both released tables |
| 1,000 ID; 1,897 OOD; 699 Challenging | Preserve ID/OOD, recompute Challenging consistency | Actual selector uses Grand Chamber OR (HUDOC application count > 1 AND distinct violated-article-code count > 1), not applicant-table row count |
| Applicant-level awards; Appendix A.5 | Restore only source-stated each-applicant amounts with supported row linkage | Estate/group/joint units are not fabricated individual shares; missing extraction is not proof no award exists |
| Table 11 separates input from target evidence | Source/extraction stages separate facts, merits and Article 41; new model adapters exclude awards, beneficiary information and audit/status fields | Field lineage and judgment segmentation still need source review; names being masked does not make a field leakage-safe |
| Unencoded structured release | Two human-readable tables | Model encodings are created only inside experiment preparation |
| Tree settings / Table 20 and X0–X3 / Table 28 | Original configurations retained; new readable-table adapter explicitly labelled as new projection | Complete original feature-to-run mapping is unavailable; two old beneficiary-derived features lack facts-side provenance and must not be silently restored |
| Raw-text FACTS experiments | Source download/segmentation and runners included | Historical exact FACTS hashes, complete checkpoints and predictions unavailable; reconstruction can differ from original preprocessing |
| Extraction validation B.3 | Candidate workflow, schema checks, cross-source checks, explicit reviewed export | New code/tests do not retroactively certify all historical extractions; full original eight-check evidence ledger missing |
| Table 10 filtering; Table 12 zero rationales; 230-case manual review | Historical claims preserved in the paper | Complete historical evidence chains not recovered; not reconstructed by inventing counts |
| Table 15 / Algorithm 1 ID matching | Existing locked membership retained | Original tie-order/run artifacts unavailable; don't relabel a new rerun as the recovered historical matching process |
| Prompted/agent results | Templates, configurations, retrieval constraints and agent code supplied | Original provider snapshots, output traces and selection artifacts unavailable; no paid calls made for this preview |

Offline tests verify code behaviour. For numerical reproduction, record the dataset content hash, feature profile, complete run configuration and output artifacts. Historical results cannot be guaranteed from recovered source alone; report newly executed experiments with their own run records.
