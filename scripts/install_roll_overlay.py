"""Install this release's SPOD worker, operator and raw template into a ROLL checkout."""
import argparse
import ast
from pathlib import Path
import shutil

TEMPLATE = '''

# SPOD raw text must match SFT / RM: prompt + "\\n" + response.
@register_chat_template("spod_raw")
def spod_raw_chat_template(tokenizer, conversation, **kwargs):
    if len(conversation) != 1 or conversation[0]["role"] != "user":
        raise ValueError("spod_raw expects exactly one user message")
    return conversation[0]["content"] + "\\n"
'''


def install(roll_dir, source):
    roll = Path(roll_dir).resolve() / 'roll'
    template = roll / 'datasets/chat_template.py'
    if not template.is_file():
        raise ValueError(f'Not a ROLL source checkout: {roll_dir}')
    text = template.read_text()
    tree = ast.parse(text)
    registered = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'register_chat_template' and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'spod_raw'
        for node in ast.walk(tree)
    )
    if not registered:
        ast.parse(text + TEMPLATE)
        template.write_text(text + TEMPLATE)
    for rel in ('utils/spod_reward.py', 'pipeline/rlvr/rewards/spod_rm_reward_worker.py'):
        target = roll / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'roll_spod' / rel, target)
    print(f'SPOD overlay installed in {roll}; spod_raw template available')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roll_dir', required=True, type=Path)
    p.add_argument('--scenario', choices=('nanogpt', 'rustc'), default='nanogpt')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    install(args.roll_dir, root if args.scenario == 'nanogpt' else root / 'rustc_perf')


if __name__ == '__main__':
    main()
