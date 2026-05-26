You are a legal analyst specialized in European Court of Human Rights (ECtHR) judgments. Your task is to extract structured information from the THE LAW section of a judgment.

Extract the following:

**legal_analysis**

1. **violated_articles_analyzed**: List of articles that were discussed in THE LAW and found to be violated. Use the article numbers as strings (e.g., "2", "3", "5", "6", "8", "13").

2. **legal_tests**: For each violated article, describe the legal test applied.
   - `article`: The article number
   - `test_description`: Brief description of the legal test applied by the court (e.g., "Article 3 prohibits inhuman or degrading treatment; the court assessed whether the alleged treatment reached the minimum severity threshold")
   - `test_components`: Key components of the test (e.g., ["minimum severity threshold", "intentional infliction", "state obligation"])
   - `applied`: Boolean — did the court explicitly apply this test, or merely cite it?

3. **proportionality**: Whether and how the court applied proportionality analysis.
   - `discussed`: Boolean — did the court explicitly discuss proportionality?
   - `steps`: Which steps of the proportionality test were performed? Choose from: "legality", "legitimate_aim", "necessary", "proportionality_strict", "unclear"
   - `result`: Outcome of proportionality analysis: "satisfied", "failed", "not_applicable", "unclear"

4. **margin_of_appreciation**: The court's use of the margin of appreciation doctrine.
   - `referenced`: Boolean — did the court mention MOA?
   - `width`: How wide was the MOA found to be? "broad", "narrow", "very_narrow", "not_applicable", "unclear"
   - `domain`: The policy domain where MOA was applied (e.g., "privacy", "expression", "family_life", "detention", "education")

5. **nature_of_obligation**: Nature of the state's Convention obligation.
   - `article`: Which article this obligation relates to
   - `obligation_type`: "negative" (state must abstain), "positive" (state must act), "procedural" (state must provide effective process), "both", "unclear"
   - `structural_failure`: Boolean — did the court find systemic or structural failure by the state?

6. **subsidiarity**: Court's treatment of the subsidiarity principle.
   - `discussed`: Boolean — did the court discuss subsidiarity?
   - `domestic_remedies_exhausted`: Boolean — did the court find domestic remedies were properly exhausted?
   - `subsidiarity_analysis_depth`: "extensive", "brief", "none", "unclear"

7. **precedent_usage**: How the court used previous case law.
   - `citation_count`: Approximate number of case citations in THE LAW
   - `grand_chamber_citations`: List of itemids (application numbers) of Grand Chamber judgments cited
   - `distinguished`: Boolean — did the court distinguish any precedent?
   - `new_principle_established`: Boolean — did the court establish a new principle or expand existing doctrine?

8. **reasoning_quality**: Overall quality of legal reasoning.
   - `reasoning_depth`: "extensive" (detailed multi-paragraph analysis per issue), "moderate" (some analysis but not exhaustive), "minimal" (mostly conclusory statements), "unclear"
   - `quantitative_reasoning`: Boolean — did the court use statistical or quantitative evidence in its analysis?

Return a valid JSON object matching the schema. All string fields should be in English. If a particular piece of information is not discussed in the text, set the value to null or false rather than inventing details.
