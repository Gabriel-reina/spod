#!/usr/bin/env python3
"""Build the SPOD six-piece set from rustc-perf triage weekly reports, with schema strictly aligned to SPOD/data/.

Correspondence with the nanoGPT version (SPOD/build_library/):
    records.jsonl        -> this script's input usable2.json (already contains gold + crate category)
    library.jsonl        -> library.jsonl        the item library, source of Hit@k ground truth
    prompts.jsonl        -> prompts.jsonl        the state pool for online GRPO
    sft_train/val        -> sft_train/val        strong only, class-balanced
    rm_items_prompted    -> rm_items_prompted    each idea attached to the state it started from
    rm_pairs_slim_*      -> rm_pairs_slim_*      stores only uid pairs, slimmed

Three things differ from nanoGPT, all decided empirically:
  1. No detrend. nanoGPT's gold is contaminated by time/length (rho 0.55~0.72); for rustc all
     |rho| < 0.10 (see diag.py), gold is already a clean semantic signal, and same-period quantilization would only lose information.
  2. tier is given directly by the triage section: Improvements=strong / Mixed=null / Regressions=reject.
     rustc's reject is a real measurement of "merged but measured slower", harder than nanoGPT's "PR was closed".
  3. Full-pipeline hold-out: SFT / RM / phi table all use train only, and test's gold ideas enter no
     training step (the last 32 weeks by date are the hold-out; every item carries a "split" field).

pid follows nanoGPT's t<track>_s<idx> format (track is always 1, a single timeline),
so split_holdout.py's parsing and all downstream code work with zero changes.

State granularity = per-change (per-sample): each landed change occupies one position k on the timeline,
starting from the state "the moment before it", and each prompt is distinct (1591 state points total,
train 1414 / test 177). This is the version actually used for training/eval in the paper; the hold-out convention is unchanged
(date >= the last 32 weeks judged as test, and test's gold ideas do not enter SFT / RM / phi).
"""
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "usable2.json")   # the parsed rustc-perf triage records (public data, shipped with the package)
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

CATS = ["B1_backend_codegen", "B2_type_trait_solver", "B3_query_incremental",
        "B4_frontend_parse", "B5_library_std", "B6_infra_tooling"]
HOLDOUT_WEEKS = 32       # frozen: the last 32 weeks are the late hold-out (see FREEZE.md)
MAX_HISTORY = 8          # at most how many merged changes to list in the prompt, to control length
MAX_RESP_CHARS = 1200    # consistent with nanoGPT's unified truncation convention
SFT_REPEAT_CAP = 3       # at most how many times a minority class may be repeated
TRUNC = 1200
SAME_TIER_GAP = 0.5      # how many percentage points gold must differ within the same tier to be distinguishable
VAL_FRAC = 0.15
PAIR_PARTNERS = 50       # how many random opponents to pair each idea with (the full combination of 1M pairs is too many)
TIER_RANK = {"strong": 2, "null": 1, "reject": 0}
SEC2TIER = {"imp": "strong", "mixed": "null", "reg": "reject"}

INSTRUCTION = (
    "Propose exactly ONE next change that reduces rustc compile time on the "
    "rustc-perf benchmark suite. Answer in a single line of the form:\n"
    "[CATEGORY] <what to change>\n"
    "CATEGORY must be one of: " + ", ".join(CATS) + "."
)


def dump(name, rows):
    p = os.path.join(DATA, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {name}  {len(rows)} rows  {os.path.getsize(p)/1024:.0f}KB")


def main():
    random.seed(0)
    recs = json.load(open(SRC, encoding="utf-8"))

    # ---------- week index + hold-out split ----------
    weeks = sorted({r["date"] for r in recs})
    widx = {w: i for i, w in enumerate(weeks)}
    cut_week = weeks[-HOLDOUT_WEEKS]
    for r in recs:
        r["idx"] = widx[r["date"]]
        r["track"] = 1
        r["split"] = "test" if r["date"] >= cut_week else "train"
        r["gold"] = -r["gold_mean"]          # negate: larger is better, same direction as nanoGPT
        r["tier"] = SEC2TIER[r["section"]]

    print(f"== hold-out split (frozen: last {HOLDOUT_WEEKS} weeks) ==")
    print(f"   week range {weeks[0]} ~ {weeks[-1]}  {len(weeks)} weeks total")
    print(f"   cut point {cut_week} (>= is test)")
    for sp in ("train", "test"):
        sub = [r for r in recs if r["split"] == sp]
        c = Counter(r["tier"] for r in sub)
        print(f"   {sp:<6} {len(sub):>5} items  strong={c['strong']:<4} "
              f"null={c['null']:<4} reject={c['reject']:<4} "
              f"weeks={len({r['date'] for r in sub})}")

    # ---------- library.jsonl ----------
    print("\n== library ==")
    lib = []
    for r in recs:
        cat_all = r.get("bcats") or [r["bcat"]]
        lib.append({
            "id": f"pr{r['pr']}",
            "source": "record",
            "text": r["title"][:TRUNC],
            "text_len": len(r["title"]),
            "gold": round(r["gold"], 4),
            "gold_tier": r["tier"],
            "category": r["bcat"],
            "cat_all": [c for c in cat_all if c in CATS],   # Hit@k ground truth (multi-label)
            "n_change": len(cat_all),                        # how many modules it spans ~ degree of bundling
            "track": 1,
            "idx": r["idx"],
            "baseline_idx": r["idx"],
            "date": r["date"],
            "section": r["section"],
            "n_bench": r.get("n_bench"),
            "pr_numbers": [r["pr"]],
            "split": r["split"],
        })
    # the same PR may appear in multiple weeks (triage re-review); dedup by id keeping the first
    seen, lib_u = set(), []
    for x in lib:
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        lib_u.append(x)
    print(f"   dedup {len(lib)} -> {len(lib_u)} items (same PR re-reviewed across weeks)")
    dump("library.jsonl", lib_u)
    print(f"   category distribution {dict(Counter(x['category'] for x in lib_u).most_common())}")
    print(f"   multi-label item share {sum(1 for x in lib_u if len(x['cat_all'])>1)/len(lib_u):.1%}")

    # ---------- per-change states: one state point per landed change (each prompt is distinct) ----------
    # The only difference from the weekly version (one state per week that has a strong, 200 total) is state granularity: here each change
    # occupies one position k on the timeline, starting from the state "the moment before it". All other conventions are unchanged.
    print("\n== prompts (per-change states) ==")
    lib = sorted(lib_u, key=lambda x: (x["idx"], x["id"]))
    n = len(lib)
    prompts = {}
    pid_of = {}
    for k, x in enumerate(lib):
        pid = f"t1_s{k}"
        pid_of[x["id"]] = pid
        prior = lib[:k]                                   # landed changes strictly earlier than it on the timeline
        hist = prior[-MAX_HISTORY:]
        strong_prior = [y for y in prior if y["gold_tier"] == "strong"]
        recent52 = [y for y in prior if y["gold_tier"] == "strong"
                    and x["idx"] - 52 <= y["idx"] < x["idx"]]
        head = ("You are reducing compile times of the Rust compiler "
                "(rust-lang/rust, measured by the rustc-perf benchmark suite). "
                f"It is week {x['date']}.")
        if recent52:
            med = sorted(v["gold"] for v in recent52)[len(recent52) // 2]
            head += (f" Over the preceding 52 weeks, {len(recent52)} optimizations were "
                     f"merged, each cutting benchmark compile time by {med:.1f}% at the median.")
        if hist:
            body = "Recent changes that landed in the compiler (oldest first):\n" + "".join(
                f"  - [{y['category']}] {y['text'][:200]} ({y['gold']:+.1f}%)\n" for y in hist)
        else:
            body = "No changes have landed yet; this is the initial state.\n"
        full = f"{head}\n\n{body}\n{INSTRUCTION}"
        prompts[pid] = {
            "pid": pid, "track": 1, "state_idx": k, "date": x["date"],
            "n_merged_52w": len(recent52),
            "n_history": len(hist), "n_strong_prior": len(strong_prior),
            "split": x["split"],
            "prompt": full,
            # start_spod_eval.py reads messages[0].content, must be included or eval raises KeyError
            "messages": [{"role": "user", "content": full}],
        }
    allp = [prompts[f"t1_s{k}"] for k in range(n)]
    dump("prompts.jsonl", allp)
    n_tr = sum(1 for p in allp if p["split"] == "train")
    print(f"   per-change states {len(allp)}: train {n_tr} / test {len(allp)-n_tr}"
          f"  (distinct prompt strings {len({p['prompt'] for p in allp})})")
    # the RLVR entry update_dataset_domain uses row["tag"] to look up domain, so train rows must carry tag="spod"
    dump("grpo_prompts_train.jsonl", [dict(p, tag="spod") for p in allp if p["split"] == "train"])
    dump("grpo_prompts_test.jsonl",
         [dict(p, subset="late") for p in allp if p["split"] == "test"])

    # ---------- rm_items_prompted: each idea attached to its own per-change state (1:1) ----------
    print("\n== rm items (with prompt) ==")
    prompted = []
    for x in lib:
        pid = pid_of[x["id"]]
        prompted.append({
            "uid": x["id"], "pid": pid, "prompt": prompts[pid]["prompt"],
            "response": f"[{x['category']}] {x['text'][:MAX_RESP_CHARS]}",
            "cat": x["category"], "cat_all": x["cat_all"], "tier": x["gold_tier"],
            "gold": x["gold"], "n_change": x["n_change"],
            "split": x["split"], "date": x["date"],
        })
    dump("rm_items_prompted.jsonl", prompted)
    tr_ct = sum(1 for p in prompted if p["split"] == "train")
    print(f"   train {tr_ct}  distinct prompts {len({p['prompt'] for p in prompted if p['split']=='train'})}")

    # ---------- rm pairs: pair only within train, stratified val split by category ----------
    print("\n== rm pairs (paired within train, test not involved at all) ==")
    pool = [p for p in prompted if p["split"] == "train"]
    by_cat = defaultdict(list)
    for p in pool:
        by_cat[p["cat"]].append(p)
    tr, va = [], []
    for c, g in by_cat.items():
        random.shuffle(g)
        k = max(1, int(round(len(g) * VAL_FRAC)))
        va += g[:k]
        tr += g[k:]
    print(f"   stratified item split: train {len(tr)}  val {len(va)}")

    def make_pairs(items):
        """pair each item with PAIR_PARTNERS random opponents; the full combination O(n^2) on 1400 items is 1M pairs, excessive."""
        out, seen_p = [], set()
        for a in items:
            for b in random.sample(items, min(PAIR_PARTNERS, len(items))):
                if a["uid"] == b["uid"]:
                    continue
                ra, rb = TIER_RANK[a["tier"]], TIER_RANK[b["tier"]]
                if ra != rb:
                    hi, lo = (a, b) if ra > rb else (b, a)
                    kind = f"{hi['tier']}>{lo['tier']}"
                elif abs(a["gold"] - b["gold"]) >= SAME_TIER_GAP:
                    hi, lo = (a, b) if a["gold"] > b["gold"] else (b, a)
                    kind = f"within_{hi['tier']}"
                else:
                    continue
                key = (hi["uid"], lo["uid"])
                if key in seen_p:
                    continue
                seen_p.add(key)
                out.append({"c": hi["uid"], "r": lo["uid"], "kind": kind,
                            "c_cat": hi["cat"], "r_cat": lo["cat"],
                            "c_len": len(hi["response"]), "r_len": len(lo["response"]),
                            "c_nchg": hi["n_change"], "r_nchg": lo["n_change"]})
        random.shuffle(out)
        return out

    trp, vap = make_pairs(tr), make_pairs(va)
    print(f"   pair count: train {len(trp)}  val {len(vap)}")
    print(f"   train composition {dict(Counter(p['kind'] for p in trp).most_common())}")
    dump("rm_pairs_slim_train.jsonl", [{k: p[k] for k in ("c", "r", "kind")} for p in trp])
    dump("rm_pairs_slim_val.jsonl", [{k: p[k] for k in ("c", "r", "kind")} for p in vap])

    print("\n   two lower bounds it must beat (compute accuracy directly on the val pairs):")
    for nm, key in (("text length only", "len"), ("cross-module count only", "nchg")):
        ok = tie = 0
        for p in vap:
            a, b = p[f"c_{key}"], p[f"r_{key}"]
            ok += a > b
            tie += a == b
        acc = (ok + 0.5 * tie) / len(vap) if vap else 0
        print(f"     {nm:<24} accuracy {acc:.3f}   (0.5 = no shortcut learned, closer is better)")

    print("\n   bias source (category distribution of the chosen side, this is what SPOD removes):")
    cc = Counter(p["c_cat"] for p in trp)
    tot = sum(cc.values())
    for c, n in cc.most_common():
        print(f"     {c:<24}{n:>7}  {100*n/tot:>5.1f}%")

    # ---------- SFT: strong from train only, class-balanced ----------
    print("\n== sft (strong from train only, class-balanced) ==")
    uid_split = {p["uid"]: ("val" if p in va else "train") for p in pool}
    for split in ("train", "val"):
        pos = [p for p in pool if p["tier"] == "strong" and uid_split[p["uid"]] == split]
        bc = defaultdict(list)
        for p in pos:
            bc[p["cat"]].append(p)
        target = max((len(v) for v in bc.values()), default=0)
        out = []
        for c, rows in bc.items():
            reps = min(SFT_REPEAT_CAP, max(1, round(target / len(rows))))
            for k in range(reps):
                out += [dict(r, sft_rep=k) for r in rows]
        random.shuffle(out)
        dump(f"sft_{split}.jsonl", out)
        raw, bal = Counter(p["cat"] for p in pos), Counter(p["cat"] for p in out)
        print(f"   {split}: raw {len(pos)} -> after balancing {len(out)}")
        for c in CATS:
            print(f"     {c:<24}{raw.get(c,0):>5}{bal.get(c,0):>7}")

    # ---------- phi table: train only, SPOD's category prior ----------
    print("\n== phi table (strong from train only, SPOD category prior) ==")
    g = defaultdict(list)
    for p in pool:
        if p["tier"] == "strong":
            g[p["cat"]].append(p["gold"])
    allv = [v for vs in g.values() for v in vs]
    mu = sum(allv) / len(allv)
    sd = (sum((v - mu) ** 2 for v in allv) / len(allv)) ** 0.5
    phi = {c: round((sum(vs) / len(vs) - mu) / sd, 4) for c, vs in g.items()}
    json.dump(phi, open(os.path.join(DATA, "phi_table.json"), "w"), indent=2)
    print(f"   global mean={mu:.3f}%  std={sd:.3f}%")
    for c, v in sorted(phi.items(), key=lambda kv: -kv[1]):
        print(f"     {c:<24}n={len(g[c]):>4}  phi={v:+.4f}")
    print(f"   -> phi span {max(phi.values())-min(phi.values()):.3f}")

    # ---------- leakage check ----------
    print("\n== leakage check (must all be 0) ==")
    test_uid = {x["id"] for x in lib_u if x["split"] == "test"}
    test_pid = {p["pid"] for p in allp if p["split"] == "test"}
    sft_all = set()
    for split in ("train", "val"):
        for l in open(os.path.join(DATA, f"sft_{split}.jsonl"), encoding="utf-8"):
            sft_all.add(json.loads(l)["uid"])
    pair_uid = {p["c"] for p in trp} | {p["r"] for p in trp} | \
               {p["c"] for p in vap} | {p["r"] for p in vap}
    checks = [
        ("test items appear in SFT", len(test_uid & sft_all)),
        ("test items appear in RM pairs", len(test_uid & pair_uid)),
        ("test prompt appears in train prompt pool", len(test_pid & {p["pid"] for p in allp if p["split"] == "train"})),
        ("phi table uses test items", len([1 for p in pool if p["uid"] in test_uid])),
    ]
    for nm, n in checks:
        print(f"   {'OK ' if n == 0 else 'FAIL'} {nm}: {n}")
    # positive control: deliberately check a quantity that must be non-zero, to guard against false negatives from field-type mismatch above
    print(f"   [positive control] train items appear in RM pairs: "
          f"{len({x['id'] for x in lib_u if x['split']=='train'} & pair_uid)} (should be far greater than 0)")


if __name__ == "__main__":
    main()
