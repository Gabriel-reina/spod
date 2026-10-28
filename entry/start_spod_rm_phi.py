"""Score the library items with a given RM, producing the per-category mean table in the SPOD rmscore
convention (the raw material for phi).

Why a separate task: the phi table (_RM_MEAN in roll/utils/spod_reward.py) must be re-measured with the
  same RM (spod2clean convention) used for that hold-out training/evaluation batch -- measuring the bias
  with a different scorer and using it to correct this RM is like "using the bias measured by ruler A to
  correct ruler B". The open-source version has only one convention, so we re-score the clean library with
  the current RM.

The convention is byte-for-byte identical to score_all_items in start_spod_rm_train.py, do not change it:
  - text = prompt + "\n" + response
  - tokenizer padding_side="right", truncation max_len=768 (the value used at RM training time)
  - take the logit of the last non-pad token as the score (located via attention_mask.sum()-1)
  - fp32 weights + autocast(bf16) forward
  - only count items with tier != "reject" ("listed items"), and only the five formal categories (UNK excluded)
If any one of these differs, the measured means cannot be viewed side-by-side with the historical table.

How to run (1 GPU):
  python entry/start_spod_rm_phi.py --rm <local RM_checkpoint dir> --data_tag _clean
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]


def resolve_rm_path(uri):
    """RM checkpoint path. The open-source version uses a local HF-format TokenClassification checkpoint dir directly."""
    assert os.path.isdir(uri), f"RM checkpoint dir not found: {uri}"
    return uri


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, help="local dir of the RM checkpoint used for scoring")
    parser.add_argument("--tag", default="phi", help="batch identifier in the output file name")
    # "" = full 190 items; _clean = the 148 items with test28 leakage removed (spod2clean convention, used by open-source)
    parser.add_argument("--data_tag", default="_clean")
    parser.add_argument("--max_len", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out_dir", default="")  # extra output dir (local), leave empty to only write workdir
    parser.add_argument("--tables", default=None)  # for compatibility with external launcher injection, received but unused
    args = parser.parse_args()

    rm_uri = args.rm
    workdir = os.environ.get("WORK_DIR", "./output")
    os.makedirs(workdir, exist_ok=True)
    print(f"[PHI] rm       = {rm_uri}", flush=True)
    print(f"[PHI] data_tag = {args.data_tag!r}", flush=True)

    rm_path = resolve_rm_path(rm_uri)

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    items = load_jsonl(os.path.join(DATA_DIR, f"rm_items_prompted{args.data_tag}.jsonl"))
    print(f"[PHI] items = {len(items)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(rm_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"      # locate the last valid token via attention_mask.sum()-1

    # fp32 weights + autocast(bf16): forward precision identical to score_all_items at RM training time
    net = AutoModelForTokenClassification.from_pretrained(
        rm_path, num_labels=1, torch_dtype=torch.float32).to(device)
    net.config.pad_token_id = tok.pad_token_id
    net.eval()

    texts = [it["prompt"] + "\n" + it["response"] for it in items]
    scores = []
    for s in range(0, len(texts), args.batch):
        enc = tok(texts[s:s + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(**enc).logits.squeeze(-1)
            else:
                logits = net(**enc).logits.squeeze(-1)
        logits = logits.float()
        last = enc["attention_mask"].sum(dim=1) - 1
        scores.extend(logits[torch.arange(logits.size(0), device=device), last].tolist())

    for it, sc in zip(items, scores):
        it["rm_score"] = sc

    # ---- bias diagnostics: same convention as start_spod_rm_train.py (listed items only, five classes only) ----
    keep = [it for it in items if it["tier"] != "reject"]
    tot = len(keep)
    by_cat = defaultdict(list)
    for it in keep:
        by_cat[it["cat"]].append(it["rm_score"])

    print(f"\n=== bias diagnostics: per-category RM mean (listed items only, n={tot}) ===", flush=True)
    print(f"   {'category':20s}{'n':>4s}{'share':>8s}{'RM_mean':>10s}", flush=True)
    rows = []
    for c in CATS:
        sub = by_cat.get(c, [])
        if sub:
            mu = sum(sub) / len(sub)
            rows.append((c, len(sub), 100 * len(sub) / tot, mu))
            print(f"   {c:20s}{len(sub):4d}{100 * len(sub) / tot:7.1f}%{mu:10.3f}", flush=True)
    if len(rows) >= 2:
        hi, lo = max(rows, key=lambda r: r[3]), min(rows, key=lambda r: r[3])
        print(f"   highest {hi[0]}({hi[3]:.3f})  lowest {lo[0]}({lo[3]:.3f})  diff {hi[3] - lo[3]:.3f}",
              flush=True)
        xs = [r[2] for r in rows]
        ys = [r[3] for r in rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        print(f"   share vs mean correlation r={(num / den if den else 0.0):+.3f}", flush=True)

    # ---- also print the phi that spod_reward.spod_phi_table would compute, saving a manual calculation ----
    # rmscore convention: z-score the per-category means, sigma defaults to per-category equal weight (SPOD_ZW=0)
    mu = {c: sum(by_cat[c]) / len(by_cat[c]) for c in CATS if by_cat.get(c)}
    m = sum(mu.values()) / len(mu)
    sd = math.sqrt(sum((v - m) ** 2 for v in mu.values()) / len(mu)) or 1.0
    phi = {c: (v - m) / sd for c, v in mu.items()}
    print("\n=== per-category equal-weight z-score (i.e. the phi in the rmscore convention) ===", flush=True)
    for c in CATS:
        if c in phi:
            print(f"   {c:20s}{phi[c]:+.3f}", flush=True)
    print(f"   phi sum of squares = {sum(v * v for v in phi.values()):.3f} (should be ~ number of categories {len(phi)})",
          flush=True)

    # also print the values _DEFAULT_N / _DEFAULT_FREQ should have (needed for the freq/logq convention with SPOD_ZW=1)
    cnt = Counter(it["cat"] for it in keep if it["cat"] in CATS)
    ntot = sum(cnt.values())
    print("\n=== listed counts and shares (raw material for _DEFAULT_N / _DEFAULT_FREQ) ===", flush=True)
    print("   " + "  ".join(f"{c}={cnt[c]}" for c in CATS), flush=True)
    print("   " + "  ".join(f"{c}={cnt[c] / ntot:.3f}" for c in CATS), flush=True)

    out = {
        "rm_uri": rm_uri, "data_tag": args.data_tag, "n_items": len(items), "n_keep": tot,
        "rm_mean": {c: round(mu[c], 3) for c in CATS if c in mu},
        "phi_zscore": {c: round(phi[c], 3) for c in CATS if c in phi},
        "n_by_cat": {c: cnt[c] for c in CATS},
        "freq_by_cat": {c: round(cnt[c] / ntot, 3) for c in CATS},
    }
    name = f"rm_phi_{args.tag}.json"
    with open(os.path.join(workdir, name), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    dname = f"rm_scores_{args.tag}.jsonl"
    with open(os.path.join(workdir, dname), "w") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in ("uid", "cat", "tier", "rm_score")},
                               ensure_ascii=False) + "\n")

    # additionally copy the result to out_dir (any local dir); leave empty to keep only the copy in workdir.
    if args.out_dir:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            for fn, payload in ((name, json.dumps(out, ensure_ascii=False, indent=2)),
                                (dname, "\n".join(
                                    json.dumps({k: it[k] for k in ("uid", "cat", "tier", "rm_score")},
                                               ensure_ascii=False) for it in items))):
                with open(os.path.join(args.out_dir, fn), "w") as f:
                    f.write(payload)
                print(f"[PHI] wrote -> {os.path.join(args.out_dir, fn)}", flush=True)
        except Exception as exc:
            print(f"[PHI] out_dir write failed: {exc}", flush=True)

    print("\n[PHI] _RM_MEAN can be copied directly from this line:", flush=True)
    print("_RM_MEAN = {" + ", ".join(f"'{c}': {mu[c]:.3f}" for c in CATS if c in mu) + "}",
          flush=True)


if __name__ == "__main__":
    main()
