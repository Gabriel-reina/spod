"""SPOD evaluation aggregator: pure python, no GPU / no dependencies. Reads two kinds of input, produces cross-arm comparison tables.

Input A (offline evaluation, output of start_spod_eval.py)
    eval_samples_<tag>.jsonl   one generated sample per line: pass/pid/idx/cat/rm_score
Input B (online training logs, output of spod_reward.py)
    step_metrics_<tag>_r*.jsonl / step_detail_<tag>_r*.jsonl

Metric conventions
  quality
    greedy_mean   RM mean of the single temperature=0 item
    mean          RM mean of the k items
    best_of_k     take the top-scored k items per prompt, then average over prompts
    win_rate      per-prompt paired comparison of "this prompt's mean" against the --base arm, wins/(wins+losses)
    d_mean +/- CI paired difference vs --base + bootstrap 95% interval; a CI straddling 0 = the difference is noise
  diversity
    uniq@k        number of categories covered by the k items per prompt / min(k, number of categories in the subset)
    entropy       normalized entropy of the category empirical distribution H/log C (whole-batch convention)
    top_share     the largest category's share (degree of collapse, lower is better)
  WARNING quality always uses rm_score = the raw RM score, never the SPOD-modified r_out: r_out is the optimization
    target, and comparing arms with it amounts to "whoever got deducted less is higher quality", making arms simply not comparable.

Three conventions (sliced by category)
    global    all categories
    drop_C1   remove C1_sys_kernel (the highest-training-share class, 34.3%)
    drop_C4   remove C4_attention (the highest RM offline mean 25.201, i.e. the head)
  drop_C4 is the "tail" convention: what is left after removing the RM's most-favored class -- the part that pure
  GRPO cannot brute-force by piling on popular items is all here. The head's own share is read from the global
  convention's top_share and the category distribution table.

Usage
  python3 spod_analyze.py --dir <directory holding eval_samples_*.jsonl> --base v
  python3 spod_analyze.py --train-dir <directory holding step_metrics_*.jsonl>
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]
# convention = which categories to remove. drop_C4 removes the highest-RM-mean (25.201) class, i.e. the tail convention;
# drop_C1 removes the highest-training-share (34.3%) class.
SUBSETS = [
    ("global", set()),
    ("drop_C1", {"C1_sys_kernel"}),
    ("drop_C4", {"C4_attention"}),
]


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # multi-process append occasionally leaves a half line; skip it rather than discarding the whole file
                continue
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def entropy_norm(counts, ncat):
    tot = sum(counts.values())
    if tot <= 0 or ncat <= 1:
        return float("nan")
    h = -sum((v / tot) * math.log(v / tot) for v in counts.values() if v > 0)
    return h / math.log(ncat)


def arm_stats(recs, drop, mu=None):
    """All metrics for a single arm after removing the categories in drop. Returns (summary dict, {pid: mean}, {pid: z_mean}).

    mu = {category: the all-arms-merged per-category mean}. z = score - mu[cat] is "in-class standardized quality":
    the per-category RM means differ by nearly 9 points (C1 16.4 vs C4 25.2), so as soon as debiasing changes the
    category distribution, the raw mean moves with it -- that is a distribution effect, not a capability change. To
    answer "did it answer better within the same class" you must use z. Report both: mean answers "global score",
    z answers "true skill".
    """
    keep = set(CATS) - set(drop)
    smp = [r for r in recs if r.get("pass") == "sample" and r["cat"] in keep]
    grd = [r for r in recs if r.get("pass") == "greedy" and r["cat"] in keep]
    mu = mu or {}
    by_pid = defaultdict(list)
    by_pid_z = defaultdict(list)
    for r in smp:
        by_pid[r["pid"]].append(r["rm_score"])
        by_pid_z[r["pid"]].append(r["rm_score"] - mu.get(r["cat"], 0.0))
    per_pid_mean = {p: mean(v) for p, v in by_pid.items()}
    per_pid_z = {p: mean(v) for p, v in by_pid_z.items()}
    cats_by_pid = defaultdict(set)
    counts = defaultdict(int)
    for r in smp:
        cats_by_pid[r["pid"]].add(r["cat"])
        counts[r["cat"]] += 1
    k = max((len(v) for v in by_pid.values()), default=0)
    ncat = len(keep)
    if ncat <= 1:
        # when only one category remains, category-granularity diversity is undefined -- give nan rather than fabricating 0/1
        uniq = ent = top_share = float("nan")
    else:
        denom = min(k, ncat) or 1
        uniq = mean([len(v) / denom for v in cats_by_pid.values()])
        ent = entropy_norm(counts, ncat)
        top_share = (max(counts.values()) / len(smp)) if smp else float("nan")
    return {
        "n_sample": len(smp),
        "n_prompt": len(by_pid),
        "fmt_ok": mean([1.0 if r.get("cat_ok") else 0.0 for r in smp]),
        "greedy_mean": mean([r["rm_score"] for r in grd]),
        "mean": mean([r["rm_score"] for r in smp]),
        "z_mean": mean([r["rm_score"] - mu.get(r["cat"], 0.0) for r in smp]),
        "best_of_k": mean([max(v) for v in by_pid.values()]),
        "uniq@k": uniq,
        "entropy": ent,
        "top_share": top_share,
        "counts": dict(counts),
    }, per_pid_mean, per_pid_z


def win_rate(cur, base):
    """Per-prompt pairing: only counted when both arms have that prompt; ties are not counted in the denominator."""
    win = loss = tie = 0
    for p, v in cur.items():
        if p not in base:
            continue
        b = base[p]
        if abs(v - b) < 1e-9:
            tie += 1
        elif v > b:
            win += 1
        else:
            loss += 1
    denom = win + loss
    return (win / denom if denom else float("nan")), win, loss, tie


def boot_ci(cur, base, iters=2000, seed=42):
    """Paired bootstrap: the per-prompt (cur - base) difference, returns (point estimate, lo95, hi95).
    When the quality difference is only +/-2%, use this to judge whether it is a real difference or noise -- a CI straddling 0 means no difference."""
    d = [cur[p] - base[p] for p in cur if p in base]
    if len(d) < 2:
        return (float("nan"),) * 3
    rnd = random.Random(seed)
    n = len(d)
    ms = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return mean(d), ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def report_eval(directory, base_tag):
    files = sorted(glob.glob(os.path.join(directory, "eval_samples_*.jsonl")))
    if not files:
        print(f"[WARN] no eval_samples_*.jsonl under {directory}")
        return
    arms = {}
    for fp in files:
        tag = os.path.basename(fp)[len("eval_samples_"):-len(".jsonl")]
        arms[tag] = load_jsonl(fp)
        print(f"loaded {tag}: {len(arms[tag])} rows")
    order = sorted(arms, key=lambda t: (t != base_tag, t))
    # baseline for in-class standardization: the all-arms-merged per-category mean (endogenous convention, not relying on an offline table)
    allsmp = [r for t in order for r in arms[t] if r.get("pass") == "sample"]
    bycat = defaultdict(list)
    for r in allsmp:
        bycat[r["cat"]].append(r["rm_score"])
    MU = {c: mean(v) for c, v in bycat.items()}
    print("\nin-class standardization baseline mu_c (all arms merged): "
          + "  ".join(f"{c.split('_')[0]}={MU.get(c, float('nan')):.2f}" for c in CATS))
    # use global's sample count as the denominator, so drop_* conventions' shr is meaningful (= remaining sample share after removing that class)
    gtot = {t: arm_stats(arms[t], set(), MU)[0]["n_sample"] for t in order}
    for name, drop in SUBSETS:
        stats, pidmeans, pidzs = {}, {}, {}
        for tag in order:
            stats[tag], pidmeans[tag], pidzs[tag] = arm_stats(arms[tag], drop, MU)
        note = {"global": "all 5 classes",
                "drop_C1": "remove C1_sys_kernel (highest training share)",
                "drop_C4": "remove C4_attention (RM's favorite) = tail convention"}.get(name, "")
        print(f"\n===== {name}  ({note}) =====")
        print(f"{'arm':<14}{'n':>6}{'nP':>5}{'shr':>7}{'greedy':>9}{'mean':>9}{'bestk':>9}"
              f"{'zMean':>8}{'dZ':>8}{'95%CI(dZ)':>18}{'winR':>7}{'uniq@k':>8}{'entH':>7}"
              f"{'topSh':>7}{'fmt':>7}")
        for tag in order:
            s = stats[tag]
            shr = s["n_sample"] / gtot[tag] if gtot.get(tag) else float("nan")
            if tag == base_tag or base_tag not in pidzs:
                wr, dz, lo, hi = (float("nan"),) * 4
            else:
                # winR / CI always run on z: comparing within the same class, not polluted by category distribution changes
                wr = win_rate(pidzs[tag], pidzs[base_tag])[0]
                dz, lo, hi = boot_ci(pidzs[tag], pidzs[base_tag])
            ci = "-" if dz != dz else f"[{lo:+.3f},{hi:+.3f}]"
            print(f"{tag:<14}{s['n_sample']:>6}{s['n_prompt']:>5}{shr:>7.3f}"
                  f"{s['greedy_mean']:>9.3f}{s['mean']:>9.3f}{s['best_of_k']:>9.3f}"
                  f"{s['z_mean']:>8.3f}{dz:>8.3f}{ci:>18}{wr:>7.3f}{s['uniq@k']:>8.3f}"
                  f"{s['entropy']:>7.3f}{s['top_share']:>7.3f}{s['fmt_ok']:>7.3f}")
        if name == "global":
            print("\n-- category distribution (sample pass, share) --")
            print(f"{'arm':<14}" + "".join(f"{c.split('_')[0]:>9}" for c in CATS))
            for tag in order:
                cnt = stats[tag]["counts"]
                tot = sum(cnt.values()) or 1
                print(f"{tag:<14}" + "".join(f"{cnt.get(c, 0) / tot:>9.3f}" for c in CATS))
    print(f"\nHow to read (base = {base_tag}, all methods are compared against it):"
          "\n  mean  = raw RM mean, includes distribution effect; it moves whenever the category distribution changes, do not read it alone"
          "\n  zMean = in-class standardized mean (distribution effect removed) -- read this for 'did it answer better within the same class'"
          "\n  quality not dropping = dZ's 95%CI straddles 0 or is positive, winR not significantly < 0.5"
          "\n  debiasing effective  = global's topSh drops / entH rises / uniq@k rises"
          "\n  true long-tail gain  = drop_C4 (tail) bestk rises, zMean does not drop (still strong after the head is removed)"
          "\n  false-gain warning   = global's topSh drops but drop_C4's zMean also drops => only pushed the head down, did not lift the tail"
          "\n  watch nP  = if nP < total prompt count under some convention, some prompts produced nothing under it, shrinking the pairing base")


def report_train(directory):
    files = sorted(glob.glob(os.path.join(directory, "step_metrics_*.jsonl")))
    if not files:
        print(f"[WARN] no step_metrics_*.jsonl under {directory}")
        return
    by_tag = defaultdict(list)
    for fp in files:
        # step_metrics_<tag>_r<rank>_<pid>.jsonl
        stem = os.path.basename(fp)[len("step_metrics_"):-len(".jsonl")]
        tag = stem.split("_r")[0]
        by_tag[tag].extend(load_jsonl(fp))
    keys = ["spod/cat_entropy", "spod/top1_preserved", "spod/rank_change_frac",
            "spod/min_clip_frac", "spod/phi_pos_frac", "spod/delta_mean",
            "spod/r_raw_mean", "spod/pop_entropy"]
    print(f"\n===== online training metrics (each arm: full-run mean / last-20% mean) =====")
    print(f"{'arm':<14}{'calls':>7}" + "".join(f"{k.split('/')[-1][:11]:>13}" for k in keys))
    for tag in sorted(by_tag):
        rows = by_tag[tag]
        tail = rows[int(len(rows) * 0.8):] or rows
        line_all = "".join(
            f"{mean([r[k] for r in rows if k in r]):>13.4f}" for k in keys)
        line_tail = "".join(
            f"{mean([r[k] for r in tail if k in r]):>13.4f}" for k in keys)
        print(f"{tag:<14}{len(rows):>7}{line_all}")
        print(f"{'  last20%':<14}{len(tail):>7}{line_tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="directory holding eval_samples_*.jsonl")
    ap.add_argument("--train-dir", default="", help="directory holding step_metrics_*.jsonl")
    ap.add_argument("--base", default="v",
                    help="reference arm tag for win rate / dMean (pure GRPO baseline, methods are compared against it)")
    a = ap.parse_args()
    if not a.dir and not a.train_dir:
        ap.error("give at least --dir or --train-dir")
    if a.dir:
        report_eval(a.dir, a.base)
    if a.train_dir:
        report_train(a.train_dir)


if __name__ == "__main__":
    main()
