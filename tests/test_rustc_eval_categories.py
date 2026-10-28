"""SFT evaluation runs before run_grpo.sh exports the rustc category table."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rustc_eval', ROOT / 'rustc_perf/entry/start_spod_eval.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class RustcEvaluationCategoryTests(unittest.TestCase):
    def test_fresh_evaluation_uses_six_rustc_categories(self):
        with patch.dict(os.environ, {}, clear=True):
            parser = evaluation.load_category_parser()
            for category in ('B1_backend_codegen', 'B2_type_trait_solver', 'B3_query_incremental',
                             'B4_frontend_parse', 'B5_library_std', 'B6_infra_tooling'):
                self.assertEqual(parser(f'[{category}] idea'), (category, True))

    def test_wrong_scenario_is_rejected(self):
        with patch.dict(os.environ, {'SPOD_PHI_SRC': 'spod2clean'}):
            with self.assertRaises(ValueError):
                evaluation.load_category_parser()
