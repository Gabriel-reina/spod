#!/usr/bin/env python3
"""
Parse the record tables of the two tracks from the README snapshot, computing the wall-time delta of each change.

Core facts (D1 verified):
  1. records are cumulative — #47 = #46's code + one new change.
     So "the benefit of a single change" = the time difference between adjacent records, a first-order approximation of the idea's benefit.
  2. val loss is not a variable but a fixed constraint (track1 ≤3.28 / track2 ≤2.92); gold = wall-time.
  3. use relative percent instead of absolute seconds: 89 items drop from 45min to 1.23min, absolute delta is not comparable.

Outputs data/records.jsonl, per line:
  track, idx, baseline_idx, time_min, baseline_time_min,
  delta_min, delta_pct, gold(=delta_pct), desc, desc_url, date,
  pr_numbers[], contributors[], flag
  flag: 'ok' | 'indistinguishable' | 'regression' | 'retiming' | 'baseline'
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SNAP = os.path.join(DATA, "README_snapshot.md")
OUT = os.path.join(DATA, "records.jsonl")

# Noise threshold: the speedrun rules require a submission to reach p<0.01 to count, and records deliberately keep a 0.001-0.002 loss buffer.
# Changes whose relative improvement falls below this value do not enter the positives; they are labeled separately to avoid treating measurement noise as an "effective idea".
NOISE_PCT = 1.0

LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
PRNUM = re.compile(r"modded-nanogpt/pull/(\d+)")


def to_minutes(s):
    s = s.strip().lower()
    m = re.match(r"([\d.]+)\s*(minute|hour|second)", s)
    if not m:
        return None
    v = float(m.group(1))
    return {"minute": v, "hour": v * 60.0, "second": v / 60.0}[m.group(2)]


def parse_table(lines, start):
    """start points to the header row; returns the consecutive data rows from start+2 onward"""
    rows = []
    i = start + 2
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith("---") or ln.startswith("#"):
            break
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) >= 5 and re.match(r"^\d+$", parts[0]):
            rows.append(parts)
        i += 1
    return rows


def main():
    lines = open(SNAP, encoding="utf-8").read().split("\n")
    heads = [i for i, l in enumerate(lines) if l.strip().startswith("| # | Record time |")]
    assert len(heads) >= 2, f"expected at least 2 record tables, got {len(heads)}"

    out = []
    for track, h in enumerate(heads, start=1):
        raw = parse_table(lines, h)
        prev_t, prev_i, prev_idx = None, None, None
        for parts in raw:
            idx = int(parts[0])
            t = to_minutes(parts[1])
            desc_cell = parts[2]
            links = LINK.findall(desc_cell)
            desc = links[0][0] if links else desc_cell
            desc_url = links[0][1] if links else ""
            log_cell = parts[4] if len(parts) > 4 else ""
            prs = sorted({int(x) for x in PRNUM.findall(log_cell + " " + desc_url)})
            contribs = [c.strip() for c in (parts[5] if len(parts) > 5 else "").split(",") if c.strip()]

            # the reliable criterion for re-timing is "idx repeats the previous row" rather than the wording of the description:
            # track1's #21 appears 3 times (original / with new timing / with latest torch),
            # which is re-timing the same code after a hardware and torch-version change, not a new idea.
            is_retiming = (idx == prev_idx)
            if prev_t is None or t is None:
                flag, dmin, dpct = "baseline", None, None
            else:
                dmin = prev_t - t
                dpct = 100.0 * dmin / prev_t
                if is_retiming:
                    flag = "retiming"
                elif dpct < 0:
                    flag = "regression"
                elif dpct < NOISE_PCT:
                    flag = "indistinguishable"
                else:
                    flag = "ok"

            out.append({
                "track": track, "idx": idx, "baseline_idx": prev_i,
                "time_min": t, "baseline_time_min": prev_t,
                "delta_min": None if dmin is None else round(dmin, 4),
                "delta_pct": None if dpct is None else round(dpct, 3),
                # a retiming row is not an idea and must not produce gold, otherwise two false-negative samples get injected into the library
                "gold": None if (dpct is None or flag == "retiming") else round(dpct, 3),
                "desc": desc, "desc_url": desc_url, "date": parts[3],
                "pr_numbers": prs, "contributors": contribs, "flag": flag,
            })
            # although retiming produces no gold, it does change the reference time of subsequent records, so still advance prev
            prev_t, prev_i, prev_idx = t, idx, idx

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"total {len(out)} items -> {OUT}")
    for tr in (1, 2):
        sub = [r for r in out if r["track"] == tr]
        c = Counter(r["flag"] for r in sub)
        print(f"\ntrack {tr}: {len(sub)} items  {dict(c)}")
        ok = [r["delta_pct"] for r in sub if r["flag"] == "ok"]
        if ok:
            ok_s = sorted(ok)
            print(f"  usable delta_pct: min={ok_s[0]:.2f}%  median={ok_s[len(ok_s)//2]:.2f}%  max={ok_s[-1]:.2f}%")
    linked = sum(1 for r in out if r["pr_numbers"])
    print(f"\nrecords linkable to a PR number: {linked}/{len(out)}")

    # noise-threshold sensitivity: reviewers will always ask "how was 1% chosen", so report the neighborhood too
    print("\nnoise-threshold sensitivity (number of usable positives):")
    cand = [r["delta_pct"] for r in out if r["flag"] in ("ok", "indistinguishable")]
    for th in (0.5, 0.8, 1.0, 1.5, 2.0):
        print(f"  threshold {th}%  ->  positives {sum(1 for d in cand if d >= th)} items")
    below = max([d for d in cand if d < NOISE_PCT], default=None)
    above = min([d for d in cand if d >= NOISE_PCT], default=None)
    print(f"  nearest points on both sides of threshold {NOISE_PCT}%: {below}% | {above}%")


if __name__ == "__main__":
    main()
