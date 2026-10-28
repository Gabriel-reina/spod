#!/usr/bin/env python3
"""
Diagnose and fix the temporal-trend contamination in gold.

The confound check in analyze_bias.py gives a counter-intuitive result:
  strong has a median text length of 332 chars, null has a median of 2538 chars — null is 7.6x longer.
The reason is not "detailed changes are useless", but time:
  early records (large delta, strong) have only a one-sentence description;
  late records (small delta, null) all carry a PR link and a long body.
So "text length" becomes a proxy for "time", and "time" in turn determines the magnitude of delta
(45 min -> 1.23 min; later on a 20% improvement is impossible).
The RM only needs to learn char-counting to predict gold, with no need to understand the technical content at all.

This script does three things:
  1. Quantify this contamination chain: idx -> baseline_time -> delta_pct, and text_len -> gold
  2. Produce a detrended gold: quantile rank within a contemporaneous window (answers "relative to changes of its era, how good is this one")
  3. Run the length-only trivial baseline — a diagnostic that must be written into the paper;
     if the real RM's ranking ability cannot beat it, the RM has learned nothing
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
WINDOW = 10  # take W items before and after within the same track as the "contemporaneous" reference


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    lib = [json.loads(l) for l in open(os.path.join(DATA, "library.jsonl"), encoding="utf-8")]
    rec = [r for r in lib if r["source"] == "record"]
    recs = {}
    for r in rec:
        recs.setdefault(r["track"], []).append(r)
    for v in recs.values():
        v.sort(key=lambda r: r["idx"])

    print("=" * 74)
    print("1) contamination-chain quantification (Spearman, avoids long-tail dominance)")
    idx = [r["idx"] for r in rec]
    gold = [r["gold"] for r in rec]
    tlen = [r["text_len"] for r in rec]
    print(f"   record idx      -> gold(delta_pct)   rho = {spearman(idx, gold):+.3f}")
    print(f"   record idx      -> text_len          rho = {spearman(idx, tlen):+.3f}")
    print(f"   text_len        -> gold(delta_pct)   rho = {spearman(tlen, gold):+.3f}")
    print("   If the first two are both strong, the third is their product rather than a genuine semantic signal.")

    print("\n" + "=" * 74)
    print("2) detrend: quantile rank within a contemporaneous window")
    for tr, v in recs.items():
        for i, r in enumerate(v):
            lo, hi = max(0, i - WINDOW), min(len(v), i + WINDOW + 1)
            peers = [q["gold"] for q in v[lo:hi]]
            # quantile rank: 0=worst of the period, 1=best of the period. Ties within the window are averaged
            less = sum(1 for g in peers if g < r["gold"])
            eq = sum(1 for g in peers if g == r["gold"])
            r["gold_detrend"] = (less + 0.5 * eq) / len(peers)
            r["peer_n"] = len(peers)
    gd = [r["gold_detrend"] for r in rec]
    print(f"   idx      -> gold_detrend   rho = {spearman(idx, gd):+.3f}   (target: close to 0)")
    print(f"   text_len -> gold_detrend   rho = {spearman(tlen, gd):+.3f}   (target: close to 0)")
    print(f"   gold     -> gold_detrend   rho = {spearman(gold, gd):+.3f}   (should stay positive, i.e. ranking info preserved)")

    print("\n" + "=" * 74)
    print("3) length-only trivial baseline (a diagnostic that must be written into the paper)")

    def auc(scores, labels):
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        if not pos or not neg:
            return 0.5
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    # each row must use its own tier: after detrending the positive is defined as "above the contemporaneous median";
    # reusing the original tier would leave AUC unchanged and hide whether detrending had any effect.
    y_raw = [1 if r["gold_tier"] == "strong" else 0 for r in rec]
    y_det = [1 if r["gold_detrend"] >= 0.5 else 0 for r in rec]
    for name, tgt, y in (("gold raw", gold, y_raw), ("gold detrended", gd, y_det)):
        print(f"   text_len only predicting {name:14s}: rho={spearman(tlen, tgt):+.3f}   "
              f"AUC(pos vs rest)={auc(tlen, y):.3f}   n_pos={sum(y)}")
    print("   AUC deviating from 0.5 => length itself is a proxy for the label; after detrending it should return near 0.5.")
    print("   Either way, training still needs to truncate text to a fixed length and write this diagnostic into the paper.")

    print("\n" + "=" * 74)
    print("4) whether categories also drift over time (decides whether train/test can be split by time)")
    for tr, v in recs.items():
        half = len(v) // 2
        early = Counter(r["category"] for r in v[:half])
        late = Counter(r["category"] for r in v[half:])
        print(f"   track {tr} first half {dict(early)}")
        print(f"   track {tr} second half {dict(late)}")

    out = os.path.join(DATA, "library_detrended.jsonl")
    by_id = {r["id"]: r for r in rec}
    with open(out, "w", encoding="utf-8") as f:
        for r in lib:
            if r["id"] in by_id:
                r["gold_detrend"] = by_id[r["id"]]["gold_detrend"]
                r["peer_n"] = by_id[r["id"]]["peer_n"]
            else:
                r["gold_detrend"] = None
                r["peer_n"] = None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
