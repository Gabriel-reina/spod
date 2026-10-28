"""SSH launchers must describe both GPUs as belonging to the local host."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LocalGrpoTopologyTests(unittest.TestCase):
    def test_local_gpu_count_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            python = directory / 'python'
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SPOD_CAPTURE"\n')
            python.chmod(0o755)
            capture = directory / 'args.txt'
            for root in (ROOT, ROOT / 'rustc_perf'):
                for override, expected in ((None, '2'), ('1', '1')):
                    env = dict(os.environ, PATH=str(directory)+os.pathsep+os.environ['PATH'],
                               SPOD_CAPTURE=str(capture), NGPU='2', SPOD_SEED='42')
                    env.pop('SPOD_NUM_GPUS_PER_NODE', None)
                    if override:
                        env['SPOD_NUM_GPUS_PER_NODE'] = override
                    subprocess.run(['bash', str(root/'scripts/run_grpo.sh'), 'bhap'],
                                   env=env, check=True, capture_output=True)
                    args = capture.read_text().splitlines()
                    self.assertEqual(args[args.index('--num_gpus_per_node')+1], expected)
                    self.assertEqual(args[args.index('--seed')+1], '42')
