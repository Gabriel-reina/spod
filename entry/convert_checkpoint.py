"""Convert a ROLL actor checkpoint using the installed public mcore_adapter package."""
import argparse
from pathlib import Path


def resolve_actor_checkpoint(path):
    """Resolve both flat and timestamped public ROLL checkpoint layouts."""
    path = Path(path)
    if path.exists():
        candidates = list(path.rglob('mca_config.json'))
    elif path.name.startswith('checkpoint-') and path.parent.exists():
        candidates = [p for p in path.parent.rglob('mca_config.json')
                      if path.name in p.relative_to(path.parent).parts]
    else:
        candidates = []
    if len(candidates) != 1:
        raise ValueError(f'Expected one actor mca_config.json for {path}, found {len(candidates)}; '
                         'specify an exact timestamped checkpoint when several runs exist')
    return candidates[0].parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint_path', type=Path, required=True)
    p.add_argument('--output_path', type=Path, required=True)
    p.add_argument('--bf16', action='store_true')
    a = p.parse_args()
    actor_path = resolve_actor_checkpoint(a.checkpoint_path)
    import torch
    from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf
    convert_checkpoint_to_hf(str(actor_path), str(a.output_path),
                             torch_dtype=torch.bfloat16 if a.bf16 else None)


if __name__ == '__main__':
    main()
