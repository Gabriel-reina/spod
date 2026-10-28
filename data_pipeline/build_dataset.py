#!/usr/bin/env python3
"""
Join records.jsonl and prs.jsonl into SPOD's change library, and do an initial semantic category labeling.

Outputs data/library.jsonl, fields:
  id, source('record'|'unmerged_pr'), text(the RM input), gold, gold_tier,
  category, category_hits[], category_conf('high'|'mid'|'low'), ...original fields

gold_tier three tiers (not binary, more informative):
  'strong'  a record with delta_pct >= NOISE_PCT     —— a change with a real benefit
  'null'    a record with delta_pct <  NOISE_PCT     —— merged, quality-passing, but the benefit is indistinguishable
                                                       this is the most valuable hard negative
  'reject'  a closed-unmerged PR                     —— not accepted

The initial category labeling is a keyword rule and is necessarily noisy. The rule's role is to give a reproducible starting point;
items with category_conf mid/low are exported to needs_review.tsv for a manual pass;
the "label-noise ablation" promised in README §13 runs once each on the pre- and post-manual-correction label versions.
"""
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NOISE_PCT = 1.0

# Keyword table. Word order is irrelevant; scored by hit count; ties are broken by PRIORITY below.
# Overly generic words are deliberately excluded (efficient / memory / layer / block / norm / init / target ...):
# such words appear in any thousand-word PR body and would suck every item into the category with the longest word list,
# manufacturing a spurious category bias. The first version fell into this trap (C3 was inflated to 39%, 9 points above the manual coarse split).
KEYWORDS = {
    "C1_sys_kernel": [
        "kernel", "fp8", "bf16", "bfloat16", "torch.compile", "compil", "triton", "cuda",
        "all_reduce", "allreduce", "nccl", "comms", "communication", "distributed",
        "overlap", "precision", "fused", "fusion", "dataloader", "data loading",
        "async", "pipelin", "quantiz", "cuda graph", "shard", "int8",
        "faster implementation", "throughput", "wall-clock", "host-to-device",
    ],
    "C2_optimizer": [
        "muon", "adam", "optimizer", "learning rate", " lr ", "momentum",
        "weight decay", "warmup", "lr schedul", "newton-schulz", "nesterov",
        " wd ", " ema ", "grad clip", "gradient clip", "hyperparameter",
        "tuned", "orthogonal", "spectral norm", "polar express", "coefficients",
    ],
    "C3_architecture": [
        "embedding", "mlp", "skip connection", "gating", " gate", "relu",
        "activation", "u-net", "unet", "residual", "lm head", "tokenizer",
        "vocab", "bigram", "value embed", "untie", "weight tying", "backbone",
        "smear", "shortcut", "depth", "zero-init", "rmsnorm", "layernorm",
    ],
    "C4_attention": [
        "attention", "attn", "flexattention", "flex attention", "rope",
        "rotary", "qk-norm", "qk norm", "sliding window", "blockmask",
        "block mask", "causal mask", "mha", "xsa", "softcap", "soft cap",
        " kv ", "head dim", "num_heads",
    ],
    "C5_loss": [
        "loss", "cross entropy", "cross-entropy", " mtp", "multi-token",
        "multi token", "auxiliary", "label smooth", "objective", "distill",
        "token prediction", "logit soft",
    ],
}
# Tie-break: earlier is higher priority. C1 is early because "a faster implementation of X" is essentially systems work,
# even when X is loss (e.g. "Cross Entropy Kernel Optimizations").
PRIORITY = ["C1_sys_kernel", "C4_attention", "C2_optimizer", "C5_loss", "C3_architecture"]


def classify(text):
    t = " " + text.lower() + " "
    hits = defaultdict(list)
    for cat, words in KEYWORDS.items():
        for w in words:
            if w in t:
                hits[cat].append(w)
    if not hits:
        return "UNK", [], "low", 0.0, ""
    # Long text necessarily hits multiple categories, so "hit only one category" cannot serve as confidence.
    # Use the score margin = (top1 - top2) / top1: only a large margin counts as accurately labeled.
    scores = {c: len(v) for c, v in hits.items()}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], PRIORITY.index(kv[0])))
    best = ranked[0][1]
    tied = [c for c in PRIORITY if scores.get(c) == best]
    cat = tied[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    margin = (best - second) / best
    runner_up = ranked[1][0] if len(ranked) > 1 else ""
    conf = "high" if margin >= 0.5 else ("mid" if margin > 0 else "low")
    flat = sorted({f"{c}:{w}" for c, ws in hits.items() for w in ws})
    return cat, flat, conf, round(margin, 3), runner_up


def main():
    records = [json.loads(l) for l in open(os.path.join(DATA, "records.jsonl"), encoding="utf-8")]
    prs = [json.loads(l) for l in open(os.path.join(DATA, "prs.jsonl"), encoding="utf-8")]
    pr_by_num = {p["number"]: p for p in prs}

    lib = []
    joined = 0
    for r in records:
        if r["flag"] in ("baseline", "retiming"):
            continue
        # the associated PR body is appended after the one-sentence description; it makes a big difference to the info fed to the RM
        bodies = []
        for n in r["pr_numbers"]:
            p = pr_by_num.get(n)
            if p and p["body"].strip():
                bodies.append(p["body"].strip())
        if bodies:
            joined += 1
        text = r["desc"] + ("\n\n" + "\n\n".join(bodies) if bodies else "")
        cat, hits, conf, margin, runner = classify(text)
        lib.append({
            "id": f"t{r['track']}_r{r['idx']}",
            "source": "record",
            "text": text,
            "text_len": len(text),
            "has_pr_body": bool(bodies),
            "gold": r["delta_pct"],
            "gold_tier": "strong" if r["delta_pct"] >= NOISE_PCT else "null",
            "category": cat, "category_hits": hits, "category_conf": conf,
            "category_margin": margin, "category_runner_up": runner,
            "track": r["track"], "idx": r["idx"], "baseline_idx": r["baseline_idx"],
            "date": r["date"], "pr_numbers": r["pr_numbers"],
        })

    for p in prs:
        if p["label_gold"] != "closed_unmerged":
            continue
        text = p["title"] + ("\n\n" + p["body"].strip() if p["body"].strip() else "")
        cat, hits, conf, margin, runner = classify(text)
        lib.append({
            "id": f"pr{p['number']}",
            "source": "unmerged_pr",
            "text": text,
            "text_len": len(text),
            "has_pr_body": bool(p["body"].strip()),
            "gold": None, "gold_tier": "reject",
            "category": cat, "category_hits": hits, "category_conf": conf,
            "category_margin": margin, "category_runner_up": runner,
            "track": None, "idx": None, "baseline_idx": None,
            "date": p["created_at"][:10], "pr_numbers": [p["number"]],
        })

    out = os.path.join(DATA, "library.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in lib:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ----
    print(f"library total {len(lib)} items -> {out}")
    print(f"  records linked to a PR body: {joined}")
    tier = Counter(r["gold_tier"] for r in lib)
    print("\ngold three tiers:")
    for k in ("strong", "null", "reject"):
        print(f"  {k:8s} {tier[k]}")

    print("\ncategory distribution (with popularity phi_freq, counted by record):")
    rec = [r for r in lib if r["source"] == "record"]
    cc = Counter(r["category"] for r in rec)
    tot = len(rec)
    for c, n in cc.most_common():
        strong = sum(1 for r in rec if r["category"] == c and r["gold_tier"] == "strong")
        hit = 100.0 * strong / n if n else 0
        print(f"  {c:18s} {n:3d} items  phi_freq={n/tot:.3f}   of which strong {strong:3d} ({hit:.0f}%)")
    if cc:
        mx, mn = cc.most_common()[0][1], cc.most_common()[-1][1]
        print(f"  most-skewed ratio {mx}:{mn} = {mx/mn:.1f} : 1")

    print("\ncategory x gold_tier cross-tab (this decides whether phi and gold are truly coupled):")
    print(f"  {'category':18s} {'strong':>7s}{'null':>7s}{'reject':>7s}")
    for c in [x[0] for x in Counter(r['category'] for r in lib).most_common()]:
        sub = [r for r in lib if r["category"] == c]
        t = Counter(r["gold_tier"] for r in sub)
        print(f"  {c:18s} {t['strong']:7d}{t['null']:7d}{t['reject']:7d}")

    conf = Counter(r["category_conf"] for r in lib)
    print(f"\nlabeling confidence: {dict(conf)}")
    need = [r for r in lib if r["category_conf"] != "high"]
    rv = os.path.join(DATA, "needs_review.tsv")
    with open(rv, "w", encoding="utf-8") as f:
        f.write("id\tsource\tgold_tier\tauto_category\trunner_up\tmargin\thits\ttext_head\tMANUAL_CATEGORY\n")
        for r in need:
            head = re.sub(r"\s+", " ", r["text"])[:140]
            f.write(f"{r['id']}\t{r['source']}\t{r['gold_tier']}\t{r['category']}\t"
                    f"{r['category_runner_up']}\t{r['category_margin']}\t"
                    f"{';'.join(r['category_hits'][:6])}\t{head}\t\n")
    print(f"needs manual review {len(need)} items -> {rv}")

    nl = [r["text_len"] for r in lib]
    nl.sort()
    print(f"\ntext length: median {nl[len(nl)//2]} chars, shortest {nl[0]}, longest {nl[-1]}, <80 chars {sum(1 for x in nl if x<80)} items")


if __name__ == "__main__":
    main()
