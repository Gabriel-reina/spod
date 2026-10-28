#!/usr/bin/env python3
"""
Zero-compute motivation check: is there any relationship between popularity phi_freq and "whether a change is truly beneficial"?

Why this must be done first:
  SPOD's premise is that "the model piles probability mass on popular categories" is a bias,
  not a correct reflection of quality. The minimal requirement for this premise is: popularity
  must not positively predict effectiveness.
  If the data showed "more popular really is more effective", then the penalty would be subtracting
  the correct signal, and the method would be wrong at its root.

Simultaneously check two confounds that would ruin the whole dataset:
  A) Label-rule sensitivity — whether the category distribution is robust to changes in the keyword table
  B) Text-length shortcut — whether reject samples are systematically shorter, letting the RM classify by char count alone

Statistics per README §9-⑧: permutation test, 20000 iterations, two-tailed.
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PERM = 20000
random.seed(0)


def load():
    return [json.loads(l) for l in open(os.path.join(DATA, "library.jsonl"), encoding="utf-8")]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def perm_test(xs, ys, n_perm=N_PERM):
    """Paired permutation: shuffle the correspondence between y and x, and see how extreme the observed correlation is"""
    obs = pearson(xs, ys)
    ys2 = list(ys)
    cnt = 0
    for _ in range(n_perm):
        random.shuffle(ys2)
        if abs(pearson(xs, ys2)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    lib = load()
    rec = [r for r in lib if r["source"] == "record"]

    # ---- 1) relationship between phi_freq and being strong (item-level, n=105, has statistical power) ----
    cc = Counter(r["category"] for r in rec)
    tot = len(rec)
    phi = {c: n / tot for c, n in cc.items()}
    xs = [phi[r["category"]] for r in rec]
    ys = [1.0 if r["gold_tier"] == "strong" else 0.0 for r in rec]
    r_all, p_all = perm_test(xs, ys)
    print("=" * 74)
    print("1) popularity phi_freq  vs  being strong (item-level point-biserial, permutation test)")
    print(f"   all records  n={len(rec)}   r={r_all:+.3f}   p={p_all:.4f}")

    # redo using only high-confidence labels: if the conclusion flips, it was driven by labeling noise
    hi = [r for r in rec if r["category_conf"] == "high"]
    if len(hi) > 20:
        cch = Counter(r["category"] for r in hi)
        phih = {c: n / len(hi) for c, n in cch.items()}
        xh = [phih[r["category"]] for r in hi]
        yh = [1.0 if r["gold_tier"] == "strong" else 0.0 for r in hi]
        r_h, p_h = perm_test(xh, yh)
        print(f"   conf=high only n={len(hi):3d}   r={r_h:+.3f}   p={p_h:.4f}")

    # also use continuous gold (delta_pct) instead of binary, to avoid threshold choice affecting the conclusion
    xs2 = [phi[r["category"]] for r in rec]
    ys2 = [r["gold"] for r in rec]
    r_c, p_c = perm_test(xs2, ys2)
    print(f"   continuous gold=delta_pct  n={len(rec)}   r={r_c:+.3f}   p={p_c:.4f}")
    print("   Reading: r significantly positive => popularity really predicts effectiveness, SPOD's penalty would hurt the correct signal.")
    print("            r not significant     => popularity carries no quality information, piling quality onto popular categories is pure loss.")

    # ---- 2) per-category hit rate, with Wilson interval (do not read the point estimate for small samples) ----
    print("\n" + "=" * 74)
    print("2) per-category strong rate + 95% Wilson interval (point estimate unreliable for small-n categories)")
    print(f"   {'category':18s}{'n':>4s}{'phi':>7s}{'strong%':>9s}   95% CI")
    z = 1.96
    for c, n in cc.most_common():
        s = sum(1 for r in rec if r["category"] == c and r["gold_tier"] == "strong")
        p = s / n
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        print(f"   {c:18s}{n:4d}{phi[c]:7.3f}{100*p:8.0f}%   [{100*max(0,ctr-half):.0f}%, {100*min(100,ctr+half):.0f}%]")

    # ---- 3) text-length shortcut check ----
    print("\n" + "=" * 74)
    print("3) confound check: does text length leak the label (the RM might classify by char count alone)")
    for tier in ("strong", "null", "reject"):
        sub = [r["text_len"] for r in lib if r["gold_tier"] == tier]
        sub.sort()
        nb = sum(1 for r in lib if r["gold_tier"] == tier and not r["has_pr_body"])
        print(f"   {tier:7s} n={len(sub):3d}  median len={sub[len(sub)//2]:6d}  "
              f"<80 chars={sum(1 for x in sub if x < 80):3d}  no body={nb:3d}")
    short = [r for r in lib if r["text_len"] < 80]
    ct = Counter(r["gold_tier"] for r in short)
    print(f"   among all {len(short)} items <80 chars, tier distribution = {dict(ct)}")
    if ct.get("reject", 0) > 0.7 * len(short):
        print("   ⚠️ short texts are almost all reject. The RM will learn a 'short=bad' shortcut; must handle:")
        print("      -> main view drops items with no body; secondary view keeps them and reports both (README §9-③ dual view)")

    # ---- 4) main-view preview: how much remains after dropping items with no body ----
    print("\n" + "=" * 74)
    print("4) main view (keep only items with body / sufficient description) remaining size")
    keep = [r for r in lib if r["text_len"] >= 80]
    ck = Counter(r["gold_tier"] for r in keep)
    print(f"   remaining {len(keep)}/{len(lib)} items   {dict(ck)}")
    ckc = Counter(r["category"] for r in keep if r["source"] == "record")
    if ckc:
        mx, mn = ckc.most_common()[0], ckc.most_common()[-1]
        print(f"   record category distribution {dict(ckc)}")
        print(f"   most skewed ratio {mx[1]}:{mn[1]} = {mx[1]/mn[1]:.1f} : 1")


if __name__ == "__main__":
    main()
