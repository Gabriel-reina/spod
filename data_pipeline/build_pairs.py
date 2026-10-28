#!/usr/bin/env python3
"""
Build the pairwise data for RM training.

Three-tier ranking: strong > null > reject
  strong  on the leaderboard and saves time noticeably
  null    on the leaderboard but the time saved is within the noise
  reject  a PR was proposed but closed

Why pairwise instead of regressing absolute scores:
  reward is only meaningful relatively; pairwise is the standard RM practice; and 105+85 items
  combine into tens of thousands of pairs, so sample size is not the bottleneck.

Conventions (fixed, no further changes):
  - use all items, no category rebalancing — category-frequency bias is exactly what SPOD removes, so it must stay in the RM
  - noise / n_change are only written as fields for appendix robustness checks; the main view does not drop them
  - within the same tier, if the detrended gold gap >= SAME_TIER_GAP, also make a pair (providing within-tier gradient)
  - the split is stratified-random by category, not by time (categories themselves are drifting)
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SAME_TIER_GAP = 0.5      # detrended gold is a 0-1 contemporaneous quantile; a half-quantile gap is needed to be distinguishable
VAL_FRAC = 0.15
TRUNC = 1200             # uniform text truncation, cutting off the "char count" shortcut
random.seed(0)

TIER_RANK = {"strong": 2, "null": 1, "reject": 0}


def main():
    lib = [json.loads(l) for l in open(os.path.join(DATA, "library_reviewed.jsonl"), encoding="utf-8")]

    items = []
    for r in lib:
        tier = r["gold_tier"] if r["source"] == "record" else "reject"
        if tier not in TIER_RANK:
            continue
        items.append({
            "uid": r["id"],
            "text": r["text"][:TRUNC],
            "tier": tier,
            "rank": TIER_RANK[tier],
            "gold_detrend": r.get("gold_detrend"),
            # category: records use the manual primary label, reject uses the keyword label (reject has no manual category label)
            "cat": r.get("cat_manual") or r.get("category") or "UNK",
            "reject_kind": r.get("reject_kind"),
            "n_change": r.get("n_change"),
            "track": r.get("track"),
            "idx": r.get("idx"),
        })

    print(f"pooled {len(items)} items  {dict(Counter(x['tier'] for x in items))}")

    # ---- stratified train/val split by category (an item never crosses splits, to avoid pair leakage) ----
    by_cat = {}
    for x in items:
        by_cat.setdefault(x["cat"], []).append(x)
    train, val = [], []
    for c, group in by_cat.items():
        random.shuffle(group)
        k = max(1, int(round(len(group) * VAL_FRAC)))
        val += group[:k]
        train += group[k:]
    print(f"stratified split: train {len(train)}  val {len(val)}")

    def make_pairs(pool):
        pairs = []
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if a["rank"] != b["rank"]:
                    hi, lo = (a, b) if a["rank"] > b["rank"] else (b, a)
                    kind = f"{hi['tier']}>{lo['tier']}"
                elif (a["gold_detrend"] is not None and b["gold_detrend"] is not None
                      and abs(a["gold_detrend"] - b["gold_detrend"]) >= SAME_TIER_GAP):
                    hi, lo = (a, b) if a["gold_detrend"] > b["gold_detrend"] else (b, a)
                    kind = f"within_{hi['tier']}"
                else:
                    continue
                pairs.append({
                    "chosen": hi["text"], "rejected": lo["text"],
                    "chosen_uid": hi["uid"], "rejected_uid": lo["uid"],
                    "chosen_cat": hi["cat"], "rejected_cat": lo["cat"],
                    "kind": kind,
                    "chosen_len": len(hi["text"]), "rejected_len": len(lo["text"]),
                    "chosen_nchg": hi["n_change"], "rejected_nchg": lo["n_change"],
                })
        random.shuffle(pairs)
        return pairs

    tr_pairs, va_pairs = make_pairs(train), make_pairs(val)
    print(f"\npair count: train {len(tr_pairs)}  val {len(va_pairs)}")
    print(f"  train composition {dict(Counter(p['kind'] for p in tr_pairs))}")
    print(f"  val   composition {dict(Counter(p['kind'] for p in va_pairs))}")

    # ---- two trivial baselines: the RM must beat both simultaneously to count as having learned something ----
    print("\ntwo lower bounds that must be beaten (accuracy computed directly on val pairs):")
    for name, key in (("text length only", "len"), ("packed-change count only", "nchg")):
        ok = tie = 0
        for p in va_pairs:
            a, b = p[f"chosen_{key}"], p[f"rejected_{key}"]
            if a is None or b is None:
                continue
            if a > b:
                ok += 1
            elif a == b:
                tie += 1
        n = sum(1 for p in va_pairs if p[f"chosen_{key}"] is not None and p[f"rejected_{key}"] is not None)
        acc = (ok + 0.5 * tie) / n if n else 0
        print(f"  {name:24s} decidable {n:5d} pairs   accuracy {acc:.3f}")

    # ---- the category bias that will remain in the RM: the category distribution of the positives is the bias source ----
    print("\nbias source (category distribution on the chosen side, this is what SPOD removes):")
    cc = Counter(p["chosen_cat"] for p in tr_pairs)
    tot = sum(cc.values())
    for c, n in cc.most_common():
        print(f"  {c:18s} {n:6d}  {100*n/tot:5.1f}%")

    for name, pairs in (("rm_pairs_train", tr_pairs), ("rm_pairs_val", va_pairs)):
        out = os.path.join(DATA, name + ".jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote {out}  {len(pairs)} lines")

    # keep an item-level file too, needed to compute the category-frequency phi during training
    out = os.path.join(DATA, "rm_items.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for x in items:
            x["split"] = "val" if x in val else "train"
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {out}  {len(items)} lines")


if __name__ == "__main__":
    main()
