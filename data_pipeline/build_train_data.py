#!/usr/bin/env python3
"""Build the training data: one prompt definition, consumed simultaneously by SFT / RM / online GRPO.

Why all three must use the same prompt: the input the RM scores inside GRPO is "prompt + model generation".
If the RM only sees the idea text during training and not the prompt, online scoring is out-of-distribution and the RM score is untrustworthy.
So the prompt is generated once here and written to file; downstream only reads it, never re-assembles it, preventing template drift in two places.

prompt = the description of a baseline state (current time + adopted change chain) + "propose the next change"
response = the change description of that record, with a category label prefix (zero-cost categorization in the online stage)

Outputs (all in data/):
  prompts.jsonl        one per baseline state, the prompt pool for online GRPO
  sft_train.jsonl      SFT training set: positives only (strong + null), category-balanced
  sft_val.jsonl        SFT validation set
  rm_items_prompted.jsonl  190 items, each carrying the prompt it belongs to
  rm_pairs_slim_train.jsonl / rm_pairs_slim_val.jsonl
      store only (chosen_uid, rejected_uid, kind), slimming 8229 pairs from 14MB to ~0.5MB.
      the split reuses build_pairs.py's frozen result, not recomputed, to avoid split drift.

Category balancing is done only for SFT, never for RM (RM's category bias is exactly the disease we want to keep).
"""
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]
MAX_HISTORY = 8      # max number of adopted changes listed in the prompt, to control length
MAX_RESP_CHARS = 1200  # consistent with the unified truncation convention in §4.1b
SFT_REPEAT_CAP = 3   # max repeats for a minority class (balancing via upsampling, not by cutting the majority class)

INSTRUCTION = (
    "Propose exactly ONE next change that makes this training run faster while still "
    "reaching the same target validation loss. Answer in a single line of the form:\n"
    "[CATEGORY] <what to change>\n"
    "CATEGORY must be one of: " + ", ".join(CATS) + "."
)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(name, rows):
    p = os.path.join(DATA, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {name}  {len(rows)} lines  {os.path.getsize(p)/1024:.0f}KB")
    return p


def build_prompts(records):
    """One prompt per record state: current time + the chain of changes already adopted before that state."""
    by_track = defaultdict(dict)
    for r in records:
        if r.get("idx") is not None:
            by_track[r["track"]][int(r["idx"])] = r

    prompts = {}
    for track, recs in by_track.items():
        for idx in sorted(recs):
            cur = recs[idx]
            # adopted chain: the desc of this state and all records before it (the baseline row is not a change)
            hist = [recs[i]["desc"] for i in sorted(recs)
                    if i <= idx and recs[i].get("baseline_idx") is not None]
            hist = hist[-MAX_HISTORY:]
            t = cur.get("time_min")
            head = (f"You are speeding up a GPT-2 training script (nanoGPT speedrun, track {track}). "
                    f"The current script reaches the target validation loss in "
                    f"{float(t):.1f} minutes." if t is not None else
                    f"You are speeding up a GPT-2 training script (nanoGPT speedrun, track {track}).")
            if hist:
                body = "Changes already adopted (oldest first):\n" + \
                    "".join(f"  - {h}\n" for h in hist)
            else:
                body = "No changes have been adopted yet; this is the initial baseline.\n"
            pid = f"t{track}_s{idx}"
            prompts[pid] = {
                "pid": pid, "track": track, "state_idx": idx,
                "time_min": t, "n_history": len(hist),
                "prompt": f"{head}\n\n{body}\n{INSTRUCTION}",
            }
    return prompts


def pid_for_item(item, prompts, records):
    """Attach a library idea to the baseline state it started from.

    record items: use baseline_idx directly (it is exactly "which state it started from").
    unmerged PRs: have no track/idx, only a date —— attach to the nearest track1 state whose date is not later than it,
    i.e. "how far the mainline had progressed when this PR was proposed". If it can't attach (earlier than the first), attach to the initial baseline.
    """
    if item.get("baseline_idx") is not None:
        pid = f"t{item['track']}_s{item['baseline_idx']}"
        if pid in prompts:
            return pid
    d = str(item.get("date") or "")
    if len(d) == 10 and d[4] == "-":                       # a PR's 2024-10-05
        key = d
        cand = []
        for r in records:
            rd = str(r.get("date") or "")                   # a record's 06/06/24
            # track is an integer in records.jsonl; normalize with str() before comparing
            if len(rd) == 8 and str(r.get("track")) == "1" and r.get("idx") is not None:
                mm, dd, yy = rd.split("/")
                cand.append((f"20{yy}-{mm}-{dd}", int(r["idx"])))
        # take the state with the nearest date (on the same day take the later idx)
        prior = sorted(x for x in cand if x[0] <= key)
        if prior:
            pid = f"t1_s{prior[-1][1]}"
            if pid in prompts:
                return pid
    return "t1_s1" if "t1_s1" in prompts else sorted(prompts)[0]


def main():
    random.seed(0)
    records = load("records.jsonl")
    lib = {r["id"]: r for r in load("library_reviewed.jsonl")}
    items = load("rm_items.jsonl")

    print("== prompts ==")
    prompts = build_prompts(records)
    dump("prompts.jsonl", [prompts[k] for k in sorted(prompts)])

    # ---- attach a prompt to each idea ----
    print("== rm items (with prompt) ==")
    prompted = []
    for it in items:
        src = lib[it["uid"]]
        pid = pid_for_item(src, prompts, records)
        prompted.append({
            "uid": it["uid"], "pid": pid, "prompt": prompts[pid]["prompt"],
            "response": f"[{it['cat']}] {it['text'][:MAX_RESP_CHARS]}",
            "cat": it["cat"], "tier": it["tier"], "split": it["split"],
            "gold_detrend": it.get("gold_detrend"), "n_change": it.get("n_change"),
            "reject_kind": it.get("reject_kind"),
        })
    dump("rm_items_prompted.jsonl", prompted)
    print("   attachment distribution top5:", Counter(p["pid"] for p in prompted).most_common(5))

    # ---- pair slimming (split reuses the frozen result) ----
    print("== rm pairs (slimmed, split unchanged) ==")
    for split in ("train", "val"):
        pairs = load(f"rm_pairs_{split}.jsonl")
        dump(f"rm_pairs_slim_{split}.jsonl",
             [{"c": p["chosen_uid"], "r": p["rejected_uid"], "kind": p["kind"]} for p in pairs])

    # ---- SFT: positives only + category balancing (upsample minority classes) ----
    print("== sft (positives only, category-balanced) ==")
    for split in ("train", "val"):
        pos = [p for p in prompted if p["tier"] in ("strong", "null") and p["split"] == split]
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
        random.shuffle(out)
        dump(f"sft_{split}.jsonl", out)
        raw = Counter(p["cat"] for p in pos)
        bal = Counter(p["cat"] for p in out)
        print(f"   {split}: original positives {len(pos)} -> after balancing {len(out)}")
        print(f"   {'category':18s}{'original':>6s}{'balanced':>8s}")
        for c in CATS:
            print(f"   {c:18s}{raw.get(c,0):6d}{bal.get(c,0):8d}")

    print("\n== data volume check ==")
    print(f"   online prompt pool  {len(prompts)} state points  <- the maximum gap mentioned in §4.2c is exactly this number")
    print(f"   RM training pairs   {sum(1 for _ in open(os.path.join(DATA,'rm_pairs_slim_train.jsonl')))}")


if __name__ == "__main__":
    main()
