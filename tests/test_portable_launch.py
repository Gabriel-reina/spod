"""CPU checks for the portable launcher and release boundary."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('overlay', ROOT/'scripts/install_roll_overlay.py')
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


class PortabilityTests(unittest.TestCase):
    def test_overlay_is_idempotent_and_preserves_raw_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            checkout = Path(d)
            target = checkout/'roll/datasets/chat_template.py'
            target.parent.mkdir(parents=True)
            target.write_text('''registry = {}
def register_chat_template(name):
    def deco(fn):
        registry[name] = fn
        return fn
    return deco
''')
            overlay.install(checkout, ROOT)
            first = target.read_text()
            overlay.install(checkout, ROOT)
            self.assertEqual(first, target.read_text())
            namespace = {}
            exec(compile(target.read_text(), str(target), 'exec'), namespace)
            self.assertEqual(namespace['registry']['spod_raw'](None, [{'role':'user','content':'example'}]), 'example\n')
            for rel in ('utils/spod_reward.py','pipeline/rlvr/rewards/spod_rm_reward_worker.py'):
                self.assertEqual((checkout/'roll'/rel).read_bytes(), (ROOT/'roll_spod'/rel).read_bytes())

    def test_public_archive_contains_only_release_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'release.zip'
            subprocess.run(['python3',str(ROOT/'scripts/export_public_release.py'),'--output',str(out)],check=True)
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
                self.assertTrue('scripts/run_reproduction.sh' in names)
                allowed = {'README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt', 'spod_analyze.py', 'spod_metrics.py', '.gitignore', 'RELEASE_MANIFEST.json', 'entry', 'configs', 'scripts', 'roll_spod', 'data', 'data_pipeline', 'rustc_perf', 'tests'}
                self.assertTrue(all(n.split('/')[0] in allowed for n in names))
                for name in names:
                    if name.endswith('.py'):
                        tree = ast.parse(z.read(name).decode())
                        imports = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
                            elif isinstance(node, ast.ImportFrom): imports.append(node.module or '')

    def test_all_launch_scripts_parse(self):
        for root in (ROOT, ROOT/'rustc_perf'):
            for p in (root/'scripts').glob('*.sh'):
                subprocess.run(['bash','-n',str(p)],check=True)


if __name__ == '__main__':
    unittest.main()
