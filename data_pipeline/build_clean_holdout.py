#!/usr/bin/env python3
"""Build the "fully held-out" version of the SFT / RM data: drop every item that hangs off a test28 state.

Why this is needed: split_holdout.py lines 16-19 already admit that that round only achieved RL-stage held-out
—— SFT and RM were trained on the full set of items, so they have seen the gold ideas corresponding to the 28 test28 states.
Measured leakage:
    SFT  32 of 139 lines come from test28 states; SFT has seen the answers to 19 of the 28 questions
    RM   42 of 190 items come from test28 states; RM has seen the answers to 27 of the 28 questions
To claim "unseen states" on a public benchmark, SFT and RM must be retrained together.

Drop rule: drop an item if its pid (the baseline state it started from) falls in test28.
    ⚠ Cannot filter by state_idx < 80: test28 is not a pure time split, it is
      t1_s80..s89 (10 items, time extrapolation) + t2_s1..s18 (18 items, cross-track),
      and t2_s5 with idx=5 would be wrongly kept by idx<80. Must judge by the pid set.

Every other convention follows the frozen scripts verbatim, so that the only difference from the earlier round is "leakage removed":
    the three tiers and rank         same as build_pairs.py
    SAME_TIER_GAP / VAL_FRAC         same as build_pairs.py
    the train/val split assignment   reuses the frozen fields in rm_items.jsonl, not re-randomized
    text truncation 1200             same as both places
    SFT uses only strong+null + category balancing cap3   same as build_train_data.py

Outputs (all with the _clean suffix, never overwriting old files —— the earlier round's results must stay reproducible), all written to data/:
    data/rm_items_prompted_clean.jsonl
    data/rm_pairs_slim_train_clean.jsonl / _val_clean.jsonl
    data/sft_train_clean.jsonl / sft_val_clean.jsonl

How to run (idempotent):
    python3 data_pipeline/build_clean_holdout.py
"""
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TEST_PROMPTS = os.path.join(DATA, "grpo_prompts_test28.jsonl")

CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]
TIER_RANK = {"strong": 2, "null": 1, "reject": 0}
SAME_TIER_GAP = 0.5      # same as build_pairs.py
SFT_REPEAT_CAP = 3       # same as build_train_data.py


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(name, rows):
    p = os.path.join(DATA, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {name}  {len(rows)} lines  "
          f"{os.path.getsize(os.path.join(DATA, name))/1024:.0f}KB")


def make_pairs(pool, seed):
    """An exact copy of build_pairs.py's make_pairs, except gold_detrend/tier are taken from the prompted items."""
    pairs = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            ra, rb = TIER_RANK[a["tier"]], TIER_RANK[b["tier"]]
            if ra != rb:
                hi, lo = (a, b) if ra > rb else (b, a)
                kind = f"{hi['tier']}>{lo['tier']}"
            elif (a.get("gold_detrend") is not None and b.get("gold_detrend") is not None
                  and abs(a["gold_detrend"] - b["gold_detrend"]) >= SAME_TIER_GAP):
                hi, lo = (a, b) if a["gold_detrend"] > b["gold_detrend"] else (b, a)
                kind = f"within_{hi['tier']}"
            else:
                continue
            pairs.append({"c": hi["uid"], "r": lo["uid"], "kind": kind})
    random.Random(seed).shuffle(pairs)
    return pairs


def main():
    test_pids = {r["pid"] for r in load(TEST_PROMPTS)}
    subset = {r["pid"]: r.get("subset") for r in load(TEST_PROMPTS)}
    n_late = sum(1 for v in subset.values() if v == "late")
    n_x = sum(1 for v in subset.values() if v == "xtrack")
    print(f"held-out states {len(test_pids)}: late {n_late} + xtrack {n_x}")

    allitems = load(os.path.join(DATA, "rm_items_prompted.jsonl"))
    print(f"\n== before / after removal ==")
    print(f"all items           {len(allitems)}  {dict(Counter(x['tier'] for x in allitems))}")

    clean = [x for x in allitems if x["pid"] not in test_pids]
    drop = [x for x in allitems if x["pid"] in test_pids]
    print(f"dropped (on test28) {len(drop)}  {dict(Counter(x['tier'] for x in drop))}")
    print(f"kept                {len(clean)}  {dict(Counter(x['tier'] for x in clean))}")
    assert len(clean) + len(drop) == len(allitems)
    assert not ({x["pid"] for x in clean} & test_pids), "missed a leak"

    dump("rm_items_prompted_clean.jsonl", clean)

    # ---- RM pairs: split assignment reuses the frozen fields, not re-randomized ----
    print("\n== RM pairs (split reuses the frozen result, only removing, not re-splitting) ==")
    for split in ("train", "val"):
        pool = [x for x in clean if x["split"] == split]
        pairs = make_pairs(pool, seed=0)
        print(f"   {split}: {len(pool)} items -> {len(pairs)} pairs  "
              f"{dict(Counter(p['kind'] for p in pairs))}")
        dump(f"rm_pairs_slim_{split}_clean.jsonl", pairs)

    # ---- SFT: positives only + category balancing, logic copied from build_train_data.py ----
    print("\n== SFT (strong+null only, category balancing cap3) ==")
    for split in ("train", "val"):
        pos = [p for p in clean if p["tier"] in ("strong", "null") and p["split"] == split]
        by_cat = defaultdict(list)
        for p in pos:
            by_cat[p["cat"]].append(p)
        target = max(len(v) for v in by_cat.values()) if by_cat else 0
        out = []
        for c, rows in by_cat.items():
            reps = min(SFT_REPEAT_CAP, max(1, round(target / len(rows))))
            for k in range(reps):
                for r in rows:
                    out.append(dict(r, sft_rep=k))
        random.Random(0).shuffle(out)
        raw = Counter(p["cat"] for p in pos)
        bal = Counter(p["cat"] for p in out)
        print(f"   {split}: unique positives {len(pos)} -> after upsampling {len(out)} lines")
        print(f"   {'category':18s}{'unique':>6s}{'written':>8s}")
        for c in CATS:
            print(f"   {c:18s}{raw.get(c,0):6d}{bal.get(c,0):8d}")
        dump(f"sft_{split}_clean.jsonl", out)

    # ---- leakage check: no test28 state may appear in the clean data ----
    print("\n== leakage check (should all be 0) ==")
    for name in ("rm_items_prompted_clean", "sft_train_clean", "sft_val_clean"):
        rows = load(os.path.join(DATA, name + ".jsonl"))
        bad = sum(1 for r in rows if r["pid"] in test_pids)
        print(f"   {name:28s} {len(rows):5d} lines, {bad} lines falling in test28")
        assert bad == 0

    # also check the pair side: both uids in a pair must be in the clean pool
    ok_uid = {x["uid"] for x in clean}
    for split in ("train", "val"):
        pairs = load(os.path.join(DATA, f"rm_pairs_slim_{split}_clean.jsonl"))
        bad = sum(1 for p in pairs if p["c"] not in ok_uid or p["r"] not in ok_uid)
        print(f"   rm_pairs_slim_{split}_clean  {len(pairs):5d} pairs, {bad} pairs containing a leaked uid")
        assert bad == 0


if __name__ == "__main__":
    main()
