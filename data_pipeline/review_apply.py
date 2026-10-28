#!/usr/bin/env python3
"""
Apply the manual review labels back to the library and rerun the three checks, to see whether the conclusions are driven by label noise.

Answers four questions:
  1. What is the agreement rate between auto labels and manual labels? Is conf=high really more accurate?
  2. After switching to manual labels, what does the category-bias ratio become? Is the "bias exists" conclusion robust?
  3. Does the motivational conclusion "popularity does not predict effectiveness" still hold under manual labels?
  4. How severe is multi-change packing? Does the single-label setting still stand?

Also cleans the reject pool: drop noise/dup, flag contested.
"""
import json
import os
import random
from collections import Counter

from manual_labels import RECORDS, REJECT_CONTESTED, REJECT_DUP, REJECT_NOISE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PERM = 20000
CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]
random.seed(0)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def perm_test(xs, ys, n_perm=N_PERM):
    obs = pearson(xs, ys)
    ys2 = list(ys)
    cnt = 0
    for _ in range(n_perm):
        random.shuffle(ys2)
        if abs(pearson(xs, ys2)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = s / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def main():
    lib = [json.loads(l) for l in open(os.path.join(DATA, "library_detrended.jsonl"), encoding="utf-8")]

    # ---- apply manual labels ----
    for r in lib:
        if r["source"] == "record":
            m = RECORDS.get(r["id"])
            if m is None:
                r["cat_manual"], r["cat_all"], r["n_change"] = "MISSING", [], None
            else:
                r["cat_manual"], r["cat_all"], r["n_change"] = m[0], m[1], m[2]
            r["reject_kind"] = None
        else:
            n = r["pr_numbers"][0] if r.get("pr_numbers") else None
            if n in REJECT_NOISE:
                kind = "noise"
            elif n in REJECT_DUP:
                kind = "dup"
            elif n in REJECT_CONTESTED:
                kind = "contested"
            else:
                kind = "tech"
            r["reject_kind"] = kind
            r["cat_manual"], r["cat_all"], r["n_change"] = None, [], None

    rec = [r for r in lib if r["source"] == "record"]
    miss = [r["id"] for r in rec if r["cat_manual"] == "MISSING"]
    if miss:
        print(f"⚠️ unreviewed records (missing from the manual table): {miss}")

    # idea pool = drop PORT / COMBINE
    ideas = [r for r in rec if r["cat_manual"] in CATS]
    print("=" * 78)
    print(f"0) record {len(rec)} items -> idea pool after dropping PORT/COMBINE: {len(ideas)} items")
    print(f"   removal detail {dict(Counter(r['cat_manual'] for r in rec if r['cat_manual'] not in CATS))}")

    # ---- 1) auto label vs manual label agreement ----
    print("\n" + "=" * 78)
    print("1) keyword auto label vs manual label: agreement rate (a direct measure of label noise)")
    for conf in ("high", "mid", "low", "ALL"):
        sub = ideas if conf == "ALL" else [r for r in ideas if r["category_conf"] == conf]
        if not sub:
            continue
        agree = sum(1 for r in sub if r["category"] == r["cat_manual"])
        # loose criterion: an auto label counts as correct as long as it falls within the manual multi-label set
        loose = sum(1 for r in sub if r["category"] in r["cat_all"])
        print(f"   conf={conf:4s} n={len(sub):3d}   strict agreement={agree:3d} ({100*agree/len(sub):.0f}%)"
              f"   within multi-label set={loose:3d} ({100*loose/len(sub):.0f}%)")
    print("   If conf=high's agreement rate is not clearly higher, margin is not a trustworthy confidence proxy.")

    # ---- 2) category bias: three versions side by side ----
    print("\n" + "=" * 78)
    print("2) category distribution: auto vs manual primary vs manual multi-label (counting all touched categories)")
    auto = Counter(r["category"] for r in ideas)
    man = Counter(r["cat_manual"] for r in ideas)
    multi = Counter(c for r in ideas for c in r["cat_all"])
    print(f"   {'category':18s}{'auto':>6s}{'manual primary':>9s}{'manual multi':>11s}")
    for c in CATS + ["UNK"]:
        print(f"   {c:18s}{auto.get(c,0):6d}{man.get(c,0):9d}{multi.get(c,0):11d}")

    def ratio(cnt):
        v = [cnt.get(c, 0) for c in CATS]
        v = [x for x in v if x > 0]
        return max(v) / min(v) if v else 0.0
    print(f"\n   most-skewed ratio (C1–C5 only)  auto {ratio(auto):.1f}:1   "
          f"manual primary {ratio(man):.1f}:1   manual multi-label {ratio(multi):.1f}:1")

    # ---- 3) whether the motivational conclusion still holds under manual labels ----
    print("\n" + "=" * 78)
    print("3) motivation-check rerun: does popularity phi predict effectiveness (manual labels)")
    tot = len(ideas)
    for name, cnt, key in (("manual primary", man, "cat_manual"), ("auto label", auto, "category")):
        phi = {c: n / tot for c, n in cnt.items()}
        xs = [phi[r[key]] for r in ideas]
        for tgt, lab in ((lambda r: 1.0 if r["gold_tier"] == "strong" else 0.0, "is_strong"),
                         (lambda r: r["gold"], "continuous gold"),
                         (lambda r: r["gold_detrend"], "detrended gold")):
            ys = [tgt(r) for r in ideas]
            rr, pp = perm_test(xs, ys)
            print(f"   {name:6s} phi -> {lab:10s}  n={len(ideas)}  r={rr:+.3f}  p={pp:.4f}")

    # ---- 4) per-category strong rate (manual labels + detrended gold) ----
    print("\n" + "=" * 78)
    print("4) per-category effectiveness rate (manual primary): one column each for raw tier and detrended, with Wilson CI")
    print(f"   {'category':18s}{'n':>4s}   strong rate (95% CI)     detrend>median (95% CI)")
    for c, n in man.most_common():
        sub = [r for r in ideas if r["cat_manual"] == c]
        s1 = sum(1 for r in sub if r["gold_tier"] == "strong")
        s2 = sum(1 for r in sub if r["gold_detrend"] >= 0.5)
        l1, h1 = wilson(s1, n)
        l2, h2 = wilson(s2, n)
        print(f"   {c:18s}{n:4d}   {100*s1/n:3.0f}% [{100*l1:.0f},{100*h1:.0f}]"
              f"            {100*s2/n:3.0f}% [{100*l2:.0f},{100*h2:.0f}]")

    # ---- 5) severity of multi-change packing ----
    print("\n" + "=" * 78)
    print("5) multi-change packing: does the single-label setting still stand")
    nc = Counter(r["n_change"] for r in ideas)
    ncat = Counter(len(r["cat_all"]) for r in ideas)
    print(f"   distribution of change count within one record {dict(sorted(nc.items()))}")
    print(f"   distribution of categories touched within one record {dict(sorted(ncat.items()))}")
    multi_n = sum(v for k, v in nc.items() if k and k >= 2)
    multi_c = sum(v for k, v in ncat.items() if k >= 2)
    print(f"   >=2 changes: {multi_n}/{len(ideas)} = {100*multi_n/len(ideas):.0f}%")
    print(f"   >=2 categories: {multi_c}/{len(ideas)} = {100*multi_c/len(ideas):.0f}%  <- single-label is undefined by construction on these items")
    # whether the packing count correlates with gold: if positively correlated, gold is really a proxy for "how many changes were packed"
    xs = [r["n_change"] for r in ideas]
    for tgt, lab in ((lambda r: r["gold"], "continuous gold"), (lambda r: r["gold_detrend"], "detrended gold")):
        rr, pp = perm_test(xs, [tgt(r) for r in ideas])
        print(f"   n_change -> {lab:10s}  r={rr:+.3f}  p={pp:.4f}"
              f"   {'⚠️ gold is partly a proxy for “how many changes were packed”' if pp < 0.05 else ''}")

    # ---- 6) reject pool cleaning ----
    print("\n" + "=" * 78)
    print("6) reject pool cleaning (85 closed-unmerged PRs)")
    rj = [r for r in lib if r["source"] != "record"]
    kc = Counter(r["reject_kind"] for r in rj)
    print(f"   {dict(kc)}")
    print(f"   only tech {kc.get('tech',0)} items are usable as hard negatives"
          f" (+ contested {kc.get('contested',0)} items need separate handling)")
    for tier_kind in ("tech", "noise"):
        sub = sorted(r["text_len"] for r in rj if r["reject_kind"] == tier_kind)
        if sub:
            print(f"   {tier_kind:9s} length median={sub[len(sub)//2]:6d}  <80 chars={sum(1 for x in sub if x<80):3d}")
    print("   ⚠️ contested = the same idea was both merged and had a PR closed, self-contradictory labels:")
    for n, (rid, nm) in sorted(REJECT_CONTESTED.items()):
        tgt = next((r for r in rec if r["id"] == rid), None)
        print(f"      pr{n:<4d} <-> {rid:7s} ({tgt['gold_tier'] if tgt else '?'})  {nm}")

    # ---- 7) the final sample pool after cleaning ----
    print("\n" + "=" * 78)
    print("7) the sample pool usable for training the RM after cleaning")
    pos = [r for r in ideas if r["gold_tier"] == "strong"]
    nul = [r for r in ideas if r["gold_tier"] == "null"]
    neg = [r for r in rj if r["reject_kind"] == "tech"]
    con = [r for r in rj if r["reject_kind"] == "contested"]
    print(f"   strong {len(pos)}  null {len(nul)}  reject_tech {len(neg)}  contested {len(con)}")
    print(f"   main-view total {len(pos)+len(nul)+len(neg)} items (contested excluded from the main view, included in the secondary view)")

    out = os.path.join(DATA, "library_reviewed.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in lib:
            # clean up CRLF while at it: 112/190 bodies carry \r, which pollutes text_len and tokenization
            r["text"] = r["text"].replace("\r\n", "\n").replace("\r", "\n")
            r["text_len"] = len(r["text"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
