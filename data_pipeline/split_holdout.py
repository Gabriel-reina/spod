"""Split the GRPO prompt pool into train / held-out test.

Why this is needed: the early round's eval reused the same 107 training prompts,
so every conclusion was in-distribution and could not support the claim that
"the debiasing effect generalizes to unseen states".

Split rule (deterministic, no randomness; changing it requires updating the paper text too):
    train  track 1 with state_idx <= 79                          79 items
    test   all of track 2 (18) + track 1 with state_idx >= 80 (10)  28 items

Why this split rather than a random one: the two test parts answer two different
generalization questions and can be reported separately ——
    subset=xtrack  the whole of track 2, never touched during training -> cross-track generalization
    subset=late    the later states of track 1, i.e. "train on early optimization states, ask what to do later"
                   -> in-track temporal extrapolation
A random split only yields one mixed in-distribution held-out set, which cannot make either statement above.

⚠ Only RL-stage held-out is achievable: SFT and RM are both trained on the full 109
records, so they have seen the gold idea for the test prompts. But all four arms share
the same SFT and the same RM, so this leakage is perfectly symmetric across arms and
does not affect inter-arm comparison —— this split answers "on prompts unseen during
GRPO, does SPOD's debiasing still hold". Full-pipeline held-out requires retraining SFT/RM too.

⚠ Does not overwrite grpo_prompts.jsonl: the early round's results must stay reproducible.

Usage (idempotent):
    python3 data_pipeline/split_holdout.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
SRC = os.path.join(DATA_DIR, "grpo_prompts.jsonl")
OUT_TRAIN = os.path.join(DATA_DIR, "grpo_prompts_train79.jsonl")
OUT_TEST = os.path.join(DATA_DIR, "grpo_prompts_test28.jsonl")

TRAIN_TRACK = 1
TRAIN_MAX_IDX = 79          # track 1's 1..79 go to train, 80..89 go to test


def parse_pid(pid):
    """t1_s23 -> (1, 23). Raise if it cannot be parsed instead of silently skipping ——
    a missing prompt is easier to debug than a dataset that quietly drops samples."""
    m = re.fullmatch(r"t(\d+)_s(\d+)", pid)
    if not m:
        raise ValueError(f"pid does not match t<track>_s<idx> format: {pid!r}")
    return int(m.group(1)), int(m.group(2))


def main():
    rows = [json.loads(line) for line in open(SRC) if line.strip()]
    train, test = [], []
    for r in rows:
        track, idx = parse_pid(r["pid"])
        if track == TRAIN_TRACK and idx <= TRAIN_MAX_IDX:
            train.append(r)
        else:
            # test carries an extra subset so eval output can be split into two sub-views; train keeps its fields unchanged
            out = dict(r)
            out["subset"] = "late" if track == TRAIN_TRACK else "xtrack"
            test.append(out)

    # the split must have no omissions and no overlap, otherwise the word "held-out" no longer holds
    assert len(train) + len(test) == len(rows), f"{len(train)}+{len(test)} != {len(rows)}"
    pid_tr = {r["pid"] for r in train}
    pid_te = {r["pid"] for r in test}
    assert not (pid_tr & pid_te), f"train/test overlap: {sorted(pid_tr & pid_te)}"
    assert len(pid_tr) == len(train) and len(pid_te) == len(test), "duplicate pid within one split"

    for path, data in ((OUT_TRAIN, train), (OUT_TEST, test)):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {os.path.basename(path)}: {len(data)} items")

    n_x = sum(1 for r in test if r["subset"] == "xtrack")
    n_l = sum(1 for r in test if r["subset"] == "late")
    print(f"  train = track{TRAIN_TRACK} idx 1..{TRAIN_MAX_IDX}")
    print(f"  test  = xtrack {n_x} items (all of track 2) + late {n_l} items"
          f" (track{TRAIN_TRACK} idx >= {TRAIN_MAX_IDX + 1})")


if __name__ == "__main__":
    main()
