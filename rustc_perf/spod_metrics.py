"""SPOD headline metrics for the rustc-perf scenario: reproduces the paper's main
table (quality + diversity) for the per-change (1591 changes / 177 held-out
states) setup.

Same eight headline metrics as the nanoGPT scenario, only the scenario-specific
parts differ (six B-categories, and gold/prior read from the rustc-perf data):

  quality (truth-based, needs gold)
    Hit@1        fraction of states whose top-RM item hits a gold category
    Hit@8        fraction of states with any hit in the k items (recall ceiling)
    Coverage     mean fraction of a state's gold categories that are covered
    TV_to_prior  total-variation distance from the output category distribution
                 to the library's category prior (lower = closer to the real mix)
  diversity (category-based)
    Ideas        distinct categories produced per state
    MaxClass     largest single category's share of all outputs (lower = better)
    MaxSameClass mean, per state, of the largest same-category pile (lower)
    Collapse     fraction of states whose k outputs all fall in one category

A few extended, non-headline metrics are also printed (Hit@3 / MRR / NDCG@8 and
the text-diversity Distinct-n / Self-BLEU / Jaccard) for completeness.

Where gold comes from (no separate answer file needed):
  Each held-out state id `t{track}_s{baseline_idx}` is looked up in the reward
  item pool (rm_items_prompted.jsonl). A state's gold set is the union of the
  `cat_all` multi-labels of its non-rejected (tier != reject) test-split items,
  i.e. "which categories of change were historically accepted from this state".
  States with no accepted item carry an empty gold set and are excluded from the
  quality metrics (but still count for diversity).

The reference prior is derived in-script from the same pool (train split,
tier != reject, primary category), so there are no hard-coded magic numbers.

Input: eval_samples_<prefix><arm>.jsonl produced by entry/start_spod_eval.py,
       one generated sample per line with pass/pid/idx/cat/rm_score/text.

Usage
  python3 spod_metrics.py --dir <dir with eval_samples_*.jsonl> \
      --prefix "" --arms sft,v,bhap --base v
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

CATS = ["B1_backend_codegen", "B2_type_trait_solver", "B3_query_incremental",
        "B4_frontend_parse", "B5_library_std", "B6_infra_tooling"]
HERE = os.path.dirname(os.path.abspath(__file__))

HEADLINE_QUALITY = ["Hit@1", "Hit@8", "Coverage", "TV_to_prior"]
HEADLINE_DIVERSITY = ["Ideas", "MaxClass", "MaxSameClass", "Collapse"]
EXTRA_QUALITY = ["Hit@3", "MRR", "NDCG@8", "TV_to_truth"]
EXTRA_DIVERSITY = ["CatEntropy", "Distinct-1", "Distinct-2", "Distinct-3",
                   "Self-BLEU", "Jaccard"]
LOWER_BETTER = {"TV_to_prior", "TV_to_truth", "MaxClass", "MaxSameClass",
                "Collapse", "Self-BLEU", "Jaccard"}


def lib_freq(items):
    """Reference prior: train-split accepted items by primary category."""
    c = Counter()
    for it in items:
        if it.get("split") != "train" or it.get("tier") == "reject":
            continue
        if it.get("cat") in CATS:
            c[it["cat"]] += 1
    tot = sum(c.values()) or 1
    return {k: c.get(k, 0) / tot for k in CATS}


def load_gold(test_file, item_file):
    """Return (gold, meta, LIB_FREQ).

    gold[i] = set of gold categories for test row i. Keyed by row index because
    start_spod_eval.py writes each sample's `pid` as the row index of the prompt
    file, so the two must be enumerated in the same order (same shipped file).
    """
    rows = [json.loads(l) for l in open(test_file) if l.strip()]
    items = [json.loads(l) for l in open(item_file) if l.strip()]
    by_state = defaultdict(set)
    for it in items:
        if it.get("split") != "test" or it.get("tier") == "reject":
            continue
        for c in (it.get("cat_all") or [it.get("cat")]):
            if c in CATS:
                by_state[it["pid"]].add(c)
    gold, meta = {}, {}
    for i, r in enumerate(rows):
        gold[i] = by_state.get(r["pid"], set())
        meta[i] = {"subset": r.get("subset"), "track": r.get("track"), "pid_str": r["pid"]}
    return gold, meta, lib_freq(items)


def tv(p, q, cats=CATS):
    return 0.5 * sum(abs(p.get(c, 0.) - q.get(c, 0.)) for c in cats)


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def self_bleu(texts):
    """Each item vs the rest: mean 4-gram precision (geometric mean n=1..4)."""
    if len(texts) < 2:
        return 0.
    tok = [t.split() for t in texts]
    out = []
    for i, h in enumerate(tok):
        refs = [g for j, g in enumerate(tok) if j != i]
        ps = []
        for n in (1, 2, 3, 4):
            hg = ngrams(h, n)
            if not hg:
                ps.append(0.); continue
            rset = Counter()
            for g in refs:
                rset |= Counter(ngrams(g, n))
            hit = sum(min(c, rset[g]) for g, c in Counter(hg).items())
            ps.append(hit / len(hg))
        out.append(math.exp(sum(math.log(p) if p > 0 else -30 for p in ps) / 4))
    return sum(out) / len(out)


def jaccard(texts):
    ss = [set(t.split()) for t in texts]
    vals = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            u = ss[i] | ss[j]
            vals.append(len(ss[i] & ss[j]) / len(u) if u else 0.)
    return sum(vals) / len(vals) if vals else 0.


def metrics(recs, gold, meta, keep_pids, prior):
    """All metrics for one arm over the states in keep_pids."""
    smp = [r for r in recs if r["pass"] == "sample" and r["pid"] in keep_pids]
    by_pid = defaultdict(list)
    for r in smp:
        by_pid[r["pid"]].append(r)
    for v in by_pid.values():
        v.sort(key=lambda x: x["idx"])

    gpids = [p for p in by_pid if gold[p]]     # states that carry a gold label
    m = {}

    # ---------- truth-based quality ----------
    hits = {1: [], 3: [], 8: []}
    mrr, ndcg, cover = [], [], []
    for p in gpids:
        rank = sorted(by_pid[p], key=lambda x: -x["rm_score"])
        cs = [r["cat"] for r in rank]
        g = gold[p]
        for k in (1, 3, 8):
            hits[k].append(1. if any(c in g for c in cs[:k]) else 0.)
        rr = 0.
        for i, c in enumerate(cs):
            if c in g:
                rr = 1. / (i + 1); break
        mrr.append(rr)
        idcg = sum(1. / math.log2(i + 2) for i in range(len(g)))
        seen, dcg = set(), 0.
        for i, c in enumerate(cs):
            if c in g and c not in seen:
                seen.add(c); dcg += 1. / math.log2(i + 2)
        ndcg.append(dcg / idcg)
        cover.append(len(set(cs) & g) / len(g))
    n = len(gpids) or 1
    m["Hit@1"] = sum(hits[1]) / n
    m["Hit@3"] = sum(hits[3]) / n
    m["Hit@8"] = sum(hits[8]) / n
    m["MRR"] = sum(mrr) / n
    m["NDCG@8"] = sum(ndcg) / n
    m["Coverage"] = sum(cover) / n

    out_dist = Counter(r["cat"] for r in smp if r["cat"] in CATS)
    tot = sum(out_dist.values()) or 1
    outp = {c: out_dist[c] / tot for c in CATS}
    gold_cnt = Counter()
    for p in gpids:
        for c in gold[p]:
            gold_cnt[c] += 1.0 / len(gold[p])
    gtot = sum(gold_cnt.values()) or 1
    goldp = {c: gold_cnt[c] / gtot for c in CATS}
    m["TV_to_truth"] = tv(outp, goldp)
    m["TV_to_prior"] = tv(outp, prior)
    m["_out_dist"] = {c: 100 * outp[c] for c in CATS}
    m["_gold_dist"] = {c: 100 * goldp[c] for c in CATS}
    m["_n_gold"] = len(gpids)
    m["_n_prompt"] = len(by_pid)

    # ---------- category diversity (over all states that produced samples) ----------
    nuniq, maxsame, allsame = [], [], []
    for p in by_pid:
        cs = [r["cat"] for r in by_pid[p]]
        valid = [c for c in cs if c in CATS]
        nuniq.append(len(set(valid)))
        cc = Counter(valid)
        top = max(cc.values()) if cc else 0
        maxsame.append(top)
        allsame.append(1. if valid and top == len(valid) and len(valid) > 1 else 0.)
    m["Ideas"] = sum(nuniq) / len(nuniq) if nuniq else 0.
    m["MaxClass"] = max(outp.values()) if outp else 0.
    ent = -sum(v * math.log(v) for v in outp.values() if v > 0)
    m["CatEntropy"] = ent / math.log(len(CATS))
    m["MaxSameClass"] = sum(maxsame) / len(maxsame) if maxsame else 0.
    m["Collapse"] = sum(allsame) / len(allsame) if allsame else 0.

    # ---------- text diversity (supplementary) ----------
    for nn in (1, 2, 3):
        tot_g = uniq_g = 0
        for p in by_pid:
            gs = []
            for r in by_pid[p]:
                gs += ngrams(r["text"].split(), nn)
            tot_g += len(gs); uniq_g += len(set(gs))
        m[f"Distinct-{nn}"] = uniq_g / tot_g if tot_g else 0.
    m["Self-BLEU"] = (sum(self_bleu([r["text"] for r in by_pid[p]]) for p in by_pid)
                      / len(by_pid)) if by_pid else 0.
    m["Jaccard"] = (sum(jaccard([r["text"] for r in by_pid[p]]) for p in by_pid)
                    / len(by_pid)) if by_pid else 0.
    return m


def report(directory, prefix, arms, base):
    test = os.path.join(HERE, "data/grpo_prompts_test.jsonl")
    items = os.path.join(HERE, "data/rm_items_prompted.jsonl")
    gold, meta, prior = load_gold(test, items)
    data = {}
    for arm in arms:
        f = os.path.join(directory, f"eval_samples_{prefix}{arm}.jsonl")
        if not os.path.exists(f):
            print(f"[MISS] {f}"); continue
        data[arm] = [json.loads(l) for l in open(f) if l.strip()]
    arms = [a for a in arms if a in data]
    if not arms:
        print("[WARN] no eval_samples matched"); return

    slices = [("total", set(gold)),
              ("late", {p for p in gold if meta[p]["subset"] == "late"}),
              ("xtrack", {p for p in gold if meta[p]["subset"] == "xtrack"})]

    for sname, pids in slices:
        if not pids:
            continue
        M = {arm: metrics(data[arm], gold, meta, pids, prior) for arm in arms}
        n_gold = M[arms[0]]["_n_gold"]
        n_prompt = M[arms[0]]["_n_prompt"]
        print("\n" + "=" * 92)
        print(f"slice {sname}: {len(pids)} states ({n_prompt} produced samples, {n_gold} have gold)")
        print("=" * 92)
        hdr = f"{'metric':<16}{'dir':>4}" + "".join(f"{a:>11}" for a in arms) + f"{'best':>12}"

        def block(title, keys):
            print(f"\n-- {title} --")
            print(hdr)
            for k in keys:
                d = "v" if k in LOWER_BETTER else "^"
                vals = {a: M[a][k] for a in arms if k in M[a]}
                best = (min if k in LOWER_BETTER else max)(vals, key=vals.get) if vals else "-"
                print(f"{k:<16}{d:>4}"
                      + "".join(f"{M[a][k]:>11.3f}" if k in M[a] else f"{'-':>11}" for a in arms)
                      + f"{best:>12}")

        block("HEADLINE quality", HEADLINE_QUALITY)
        block("HEADLINE diversity", HEADLINE_DIVERSITY)
        block("extended quality", EXTRA_QUALITY)
        block("extended diversity", EXTRA_DIVERSITY)

        print("\n-- category distribution (%) --")
        print(f"{'':<16}" + "".join(f"{c.split('_')[0]:>11}" for c in CATS))
        print(f"{'gold':<16}" + "".join(f"{M[arms[0]]['_gold_dist'][c]:>11.1f}" for c in CATS))
        for a in arms:
            print(f"{a:<16}" + "".join(f"{M[a]['_out_dist'][c]:>11.1f}" for c in CATS))
        if sname == "total" and base in M:
            print(f"\n(base = {base}; ^ = higher better, v = lower better)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding eval_samples_*.jsonl")
    ap.add_argument("--prefix", default="", help="run-tag prefix in the file name")
    ap.add_argument("--arms", required=True, help="comma-separated arm tags, e.g. sft,v,bhap")
    ap.add_argument("--base", default="v", help="reference arm tag")
    a = ap.parse_args()
    report(a.dir, a.prefix, [x for x in a.arms.split(",") if x], a.base)


if __name__ == "__main__":
    main()
