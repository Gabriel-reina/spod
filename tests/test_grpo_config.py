"""Check the actual launcher config assembly against Hydra's structured config."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from hydra import initialize_config_dir, compose
from omegaconf import open_dict, OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class GrpoConfigTests(unittest.TestCase):
    def test_launcher_can_add_reward_environment_to_structured_config(self):
        for root in (ROOT, ROOT/'rustc_perf'):
            tree = ast.parse((root/'entry/start_rlvr_pipeline.py').read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
            # Execute the real launcher's environment injection block, not a copy of it.
            first = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Import) and n.names[0].name=='os')
            last = next(i for i,n in enumerate(main.body[first:],start=first) if isinstance(n,ast.With))
            code = compile(ast.Module(body=main.body[first:last+1],type_ignores=[]),str(root/'entry/start_rlvr_pipeline.py'),'exec')
            with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
                cfg = compose(config_name='spod_grpo')
                self.assertTrue(OmegaConf.is_struct(cfg))
                with patch.dict(os.environ,{'SPOD_ENABLE':'1','SPOD_BETA':'1.0'}):
                    exec(code, {'cfg':cfg,'open_dict':open_dict})
                self.assertEqual(cfg.rewards.spod.system_envs.SPOD_ENABLE,'1')
                self.assertEqual(cfg.seed,42)
                self.assertEqual(cfg.max_steps,50)

    def test_small_job_storage_cpu_budget(self):
        for root in (ROOT, ROOT/'rustc_perf'):
            cfg = OmegaConf.load(root/'configs/spod_grpo.yaml')
            self.assertEqual(cfg.transfer_backend.backend_name, 'TransferQueue')
            self.assertEqual(cfg.transfer_backend.backend_config.backend.SimpleStorage.num_data_storage_units, 2)
            self.assertEqual(cfg.seed, 42)
            self.assertEqual(cfg.max_steps, 50)


if __name__=='__main__':
    unittest.main()
