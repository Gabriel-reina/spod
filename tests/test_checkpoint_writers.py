"""Single-host torchrun must not have multiple writers to one HF checkpoint."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class CheckpointWriterTests(unittest.TestCase):
    def test_shared_filesystem_defaults_to_rank_zero_writer(self):
        for root in (ROOT,ROOT/'rustc_perf'):
            for name in ('start_spod_sft_train.py','start_spod_rm_train.py'):
                tree=ast.parse((root/'entry'/name).read_text())
                guards=[n.test for n in ast.walk(tree) if isinstance(n,ast.If)
                        and 'SPOD_SAVE_LOCAL_REPLICA' in ast.unparse(n.test)]
                self.assertEqual(len(guards),1)
                code=compile(ast.Expression(guards[0]),str(root/'entry'/name),'eval')
                for rank,flag,expected in ((0,'0',True),(1,'0',False),(1,'1',True)):
                    with patch.dict(os.environ,{'SPOD_SAVE_LOCAL_REPLICA':flag}):
                        self.assertEqual(eval(code,{'rank':rank,'os':os}),expected)

if __name__=='__main__':unittest.main()
