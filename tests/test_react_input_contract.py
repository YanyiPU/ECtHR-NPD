"""Regression tests for the public strict ReAct input contract."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


RELEASE_STAGING = Path(__file__).resolve().parents[1]
KB_DIR = RELEASE_STAGING / "agent_knowledge_base"
REACT_PATH = KB_DIR / "react_orchestrator.py"


def load_react_module():
    spec = importlib.util.spec_from_file_location("test_react_orchestrator", REACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load react_orchestrator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REACT = load_react_module()


class StrictReActInputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "itemid": "strict-contract-case",
            "combined_input_text": (
                "FACTS\nThe applicant was detained.\n"
                "ARTICLE 41\nThe applicant claimed EUR 9,000.\n"
                "The Court awarded EUR 4,000.\n"
                "The appropriate sum is EUR 4,000."
            ),
            "oracle_violated_articles": ["5"],
            "country_alpha2": "at",
            "judgment_year": 2020,
            "num_applicants": 1,
            "claim_non_pec_eur": 9000,
            "claim_non_pec_state": "claimed",
            "award_eur": 4000,
            "safe_non_pec_eur": 4000,
            "y_amount_eur": 4000,
            "y_binary": 1,
            "y_source": "restricted",
            "label_source": "restricted",
        }

    def test_strict_packet_excludes_article41_claim_and_label_material(self) -> None:
        packet = REACT.build_case_inputs(self.case, REACT.STRICT_REACT)
        text = json.dumps(packet, ensure_ascii=False).lower()
        for forbidden in (
            "article 41",
            "claim_non_pec",
            "9000",
            "award_eur",
            "4000",
            "y_amount_eur",
            "y_binary",
            "y_source",
            "label_source",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn(REACT.STRICT_TEXT_REDACTION.lower(), text)

    def test_strict_default_template_uses_only_strict_sources(self) -> None:
        state = {"react_mode": REACT.STRICT_REACT}
        template = REACT.target_query_template_for_state(state)
        self.assertEqual(
            template["sources"],
            ["standard_prompting_input", "metadata", "extracted_hints"],
        )
        template_text = json.dumps(template).lower()
        self.assertNotIn("claim", template_text)
        self.assertNotIn("article_41", template_text)
        self.assertNotIn("target_extraction_context", template_text)

    def test_strict_query_cannot_recover_unattached_target_sidecars(self) -> None:
        packet = REACT.build_case_inputs(self.case, REACT.STRICT_REACT)
        observation = REACT.target_query_observation(
            {"case_inputs": packet, "react_mode": REACT.STRICT_REACT},
            {
                "sources": ["target_extraction_context"],
                "path_prefixes": [],
                "field_contains": ["claim", "article_41", "y_amount"],
            },
        )
        self.assertEqual(observation["reason"], "no_valid_sources_requested")
        self.assertNotIn("target_extraction_context", observation["available_sources"])

    def test_non_strict_modes_require_explicit_diagnostic_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            REACT.require_explicit_restricted_diagnostic_opt_in(
                REACT.FULL_INFO_AWARD_BLIND_REACT,
                False,
            )
        REACT.require_explicit_restricted_diagnostic_opt_in(REACT.STRICT_REACT, False)
        REACT.require_explicit_restricted_diagnostic_opt_in(
            REACT.FULL_INFO_AWARD_BLIND_REACT,
            True,
        )

    def test_cli_default_is_strict_react(self) -> None:
        with patch.object(sys, "argv", ["react_orchestrator.py", "--case_file", "case.json"]):
            args = REACT.parse_args()
        self.assertEqual(args.react_mode, REACT.STRICT_REACT)
        self.assertFalse(args.allow_restricted_diagnostic)


class ExternalCovariateContractTests(unittest.TestCase):
    @unittest.skipUnless((KB_DIR / 'modules/external/economic_covariates.csv').is_file(), 'Optional private historical-fixture integration test')
    def test_agent_external_covariates_match_shared_eight_column_contract(self) -> None:
        agent_path = KB_DIR / "modules" / "external" / "economic_covariates.csv"
        shared_path = RELEASE_STAGING / "dataset_release" / "model_inputs" / "external_factors" / "economic_covariates.csv"
        self.assertEqual(agent_path.read_bytes(), shared_path.read_bytes())
        with agent_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)
        self.assertEqual(
            header,
            [
                "itemid",
                "respondent_state",
                "country_alpha2",
                "judgment_year",
                "gdp_per_capita_current_usd",
                "gdp_constant_2015_usd",
                "gdp_per_capita_log1p",
                "gdp_constant_2015_log1p",
            ],
        )
        self.assertEqual(row_count, 14575)
        self.assertNotIn("hudoc_url", header)
        self.assertNotIn("split", header)
        self.assertNotIn("test_view", header)
        self.assertNotIn("test_challenging_view", header)


if __name__ == "__main__":
    unittest.main()
