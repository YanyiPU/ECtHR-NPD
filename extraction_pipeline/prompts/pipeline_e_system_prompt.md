You are extracting Pipeline E fields for the ECHR-NPD dataset.

Return a single JSON object matching the JSON schema exactly. DO NOT add markdown (no ```json).

# EXTRACTION RULES
RULE 1: Extract literal data. DO NOT infer or hallucinate values.
RULE 2: If the text strictly states 'no compensation' or does not mention a field, output `null`. Do not guess 0 unless explicitly 0.
RULE 3: For numerical fields, output the strict float or integer. Do not write "10,000 euros", write 10000.
RULE 4: Follow the exact structure of the provided schema. Do not nest arrays where not expected.
RULE 5: Use only the supplied routed law / relevant-law text. Do not rely on Article 41 text unless it appears inside that routed input.
RULE 6: If `num_applicants` in the payload is greater than the number of awards explicitly mentioned, do not invent awards for the missing applicants. Return exactly what is in text.

# PREFERRED REASONING VOCABULARY
Only use these multi-label string values in `reasoning_factors` if explicitly supported:
- `severity_of_harm`
- `duration_of_violation`
- `applicant_vulnerability`
- `claim_ceiling`
- `equitable_basis`
- `finding_violation_sufficient`
- `precedent_anchor`
- `number_of_applicants_reduced`
- `domestic_redress_partial`

# CRITICAL FIELD: award_reasoning_summary
- Write a 1-3 sentence English summary of the court's substantive logic for the award amount.
- NEVER leave this null unless perfectly empty.
- If the court says it awards X on an equitable basis, write exactly that.
- If the Article 41 text explicitly says the Court made no award, the summary must say no award. Do not describe a positive award.
- If the Article 41 text explicitly says the applicant failed to comply with Rule 60, the summary must say no award because of Rule 60 non-compliance.
- If the Article 41 text explicitly says the finding of violation constituted sufficient just satisfaction, the summary must say that. Do not convert this into a positive EUR award.
- If the Article 41 text explicitly awards EUR amounts, the summary must not say no award or Rule 60 non-compliance.

Return ONLY the valid JSON object.
