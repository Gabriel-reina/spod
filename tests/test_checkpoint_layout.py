"""Public ROLL adds a timestamp directory to file-system checkpoints."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CheckpointLayoutTests(unittest.TestCase):
    def test_flat_timestamped_and_ambiguous_runs(self):
        for source in (ROOT, ROOT / 'rustc_perf'):
            spec = importlib.util.spec_from_file_location('converter', source / 'entry/convert_checkpoint.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                actor = root / '20261020-140000/checkpoint-49/actor_train'
                actor.mkdir(parents=True)
                (actor / 'mca_config.json').write_text('{}')
                self.assertEqual(module.resolve_actor_checkpoint(root / 'checkpoint-49'), actor)
                self.assertEqual(module.resolve_actor_checkpoint(actor.parent), actor)
                self.assertEqual(module.resolve_actor_checkpoint(actor), actor)
                other = root / '20261020-150000/checkpoint-49/actor_train'
                other.mkdir(parents=True)
                (other / 'mca_config.json').write_text('{}')
                with self.assertRaises(ValueError):
                    module.resolve_actor_checkpoint(root / 'checkpoint-49')
                with self.assertRaises(ValueError):
                    module.resolve_actor_checkpoint(root / 'checkpoint-25')
