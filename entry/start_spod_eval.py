"""SPOD offline evaluation: given a GRPO output ckpt, generate on a fixed eval prompt set and score
with the same proxy RM, writing the raw "one row per sample" records to a local jsonl. Aggregation and
cross-arm comparison are not done here -- they are handled by spod_analyze.py (pure python, runs locally),
so changing the metric convention does not require re-running the GPU.

Why only raw records and no summary:
  the definitions of quality/diversity are still tweaked repeatedly (global vs drop-popular, the k of
  best-of-k, the pairing granularity of win rate); once the summary is hard-coded on the GPU side, every
  convention change would require re-running 4 arms x 856 generations.

WARNING the RM used for scoring must be the one from training (the same proxy RM), and the concat must be
  the raw text prompt + "\n" + response -- byte-for-byte consistent with SFT / RM / online GRPO, otherwise
  scoring is out of distribution.
WARNING category parsing directly imports spod_reward.spod_category, no separate regex: the evaluation
  convention and the convention that applied penalties during training must be the same function, otherwise
  "diversity improved" may just be an inconsistency between two parsers.

Two passes:
  greedy  do_sample=False, 1 per prompt -- corresponds to greedy mean score
  sample  temperature/top_p aligned with GRPO yaml's generating_args, k per prompt
          -- corresponds to mean score / best-of-k / category diversity

How to run (1 GPU):
  python entry/start_spod_eval.py --actor <local actor_ckpt> --rm <local RM_ckpt> --tag <arm>
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def resolve_model_path(uri):
    """Model checkpoint path. The open-source version uses a local HF checkpoint dir directly."""
    assert os.path.isdir(uri), f"model checkpoint dir not found: {uri}"
    return uri


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True, help="local dir of the actor ckpt being evaluated")
    parser.add_argument("--rm", required=True, help="local dir of the RM ckpt used for scoring")
    parser.add_argument("--tag", default="spod_eval", help="arm identifier in the output file name")
    parser.add_argument("--k", type=int, default=8)
    # default is train-set self-eval; for hold-out evaluation pass the test prompt file (--prompts)
    parser.add_argument("--prompts", default="grpo_prompts.jsonl",
                        help="prompt file name under data/")
    parser.add_argument("--temperature", type=float, default=0.99)   # aligned with GRPO yaml
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="")  # extra output dir (local), leave empty to only write workdir
    parser.add_argument("--tables", default=None)  # for compatibility with external launcher injection, received but unused
    args = parser.parse_args()

    actor_uri = args.actor
    rm_uri = args.rm
    workdir = os.environ.get("WORK_DIR", "./output")
    os.makedirs(workdir, exist_ok=True)
    print(f"[EVAL] actor = {actor_uri}", flush=True)
    print(f"[EVAL] rm    = {rm_uri}", flush=True)

    actor_path = resolve_model_path(actor_uri)
    rm_path = resolve_model_path(rm_uri)

    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForTokenClassification,
                              AutoTokenizer, set_seed)
    # Category parsing needs only the shipped operator, not an installed ROLL runtime.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "spod_eval_operator", os.path.join(HERE, "..", "roll_spod", "utils", "spod_reward.py"))
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    spod_category = operator.spod_category

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_jsonl(os.path.join(DATA_DIR, args.prompts))
    # prompt text must be exactly what chat_template's spod_raw produces: content + "\n"
    prompts = [(i, r["messages"][0]["content"] + "\n") for i, r in enumerate(rows)]
    print(f"[EVAL] prompts = {len(prompts)} ({args.prompts})  k = {args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(actor_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    actor = AutoModelForCausalLM.from_pretrained(actor_path, torch_dtype=torch.bfloat16).to(device)
    actor.eval()

    def generate(pass_name, num_return, do_sample):
        """Returns [(pid, idx, response_text)]. Generation-side padding must be on the left, otherwise
        right-padding feeds pad tokens into the decoder prefix and the first half of the output is garbage."""
        tok.padding_side = "left"
        out = []
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s:s + args.batch]
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            with torch.no_grad():
                gen = actor.generate(
                    **enc, do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    top_k=args.top_k if do_sample else None,
                    num_return_sequences=num_return,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
            plen = enc["input_ids"].shape[1]
            texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
            for j, (pid, _) in enumerate(chunk):
                for r in range(num_return):
                    out.append((pid, r, texts[j * num_return + r]))
            print(f"[EVAL] {pass_name} {min(s + args.batch, len(prompts))}/{len(prompts)}", flush=True)
        return out

    greedy = generate("greedy", 1, False)
    sampled = generate("sample", args.k, True)
    del actor
    if device == "cuda":
        torch.cuda.empty_cache()

    rm = AutoModelForTokenClassification.from_pretrained(
        rm_path, num_labels=1, torch_dtype=torch.bfloat16).to(device)
    rm.eval()
    rm_tok = AutoTokenizer.from_pretrained(rm_path)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm_tok.padding_side = "right"   # locate the last valid token via attention_mask.sum()-1

    def score(texts):
        """The logit of the last non-pad token is the reward -- exactly how GRPO-side hf_infer takes it."""
        vals = []
        for s in range(0, len(texts), args.batch):
            chunk = texts[s:s + args.batch]
            enc = rm_tok(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=768).to(device)   # 768 = the max_len used at RM training time
            with torch.no_grad():
                logits = rm(**enc).logits.squeeze(-1).float()
            last = enc["attention_mask"].sum(dim=1) - 1
            vals.extend(logits[torch.arange(logits.size(0), device=device), last].tolist())
        return vals

    recs = []
    for pass_name, items in (("greedy", greedy), ("sample", sampled)):
        texts = [prompts[pid][1] + t for pid, _, t in items]
        scores = score(texts)
        print(f"[EVAL] scored {pass_name}: n={len(scores)}", flush=True)
        for (pid, idx, t), sc in zip(items, scores):
            cat, ok = spod_category(t)
            recs.append({
                "tag": args.tag, "pass": pass_name, "pid": pid, "idx": idx,
                "track": rows[pid].get("track"),
                # only held-out files have subset (xtrack/late); None for train-set self-eval
                "subset": rows[pid].get("subset"), "cat": cat, "cat_ok": bool(ok),
                "rm_score": sc, "n_char": len(t), "text": t[:400],
            })

    name = f"eval_samples_{args.tag}.jsonl"
    local = os.path.join(workdir, name)
    with open(local, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[EVAL] wrote {len(recs)} rows -> {local}", flush=True)

    # additionally copy the result to out_dir (any local dir); leave empty to keep only the copy in workdir.
    if args.out_dir:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            dst = os.path.join(args.out_dir, name)
            with open(dst, "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[EVAL] wrote -> {dst}", flush=True)
        except Exception as exc:
            print(f"[EVAL] out_dir write failed: {exc}", flush=True)

    # only print a rough at-a-glance summary; the formal convention is all in spod_analyze.py
    smp = [r for r in recs if r["pass"] == "sample"]
    grd = [r for r in recs if r["pass"] == "greedy"]
    dist = {}
    for r in smp:
        dist[r["cat"]] = dist.get(r["cat"], 0) + 1
    n = len(smp) or 1
    print(f"[EVAL] greedy mean rm = {sum(x['rm_score'] for x in grd) / max(len(grd), 1):.4f}", flush=True)
    print(f"[EVAL] sample mean rm = {sum(x['rm_score'] for x in smp) / n:.4f}", flush=True)
    print(f"[EVAL] fmt_ok = {sum(1 for x in smp if x['cat_ok']) / n:.4f}", flush=True)
    print("[EVAL] cat share = " + "  ".join(
        f"{c}={v / n:.3f}" for c, v in sorted(dist.items(), key=lambda kv: -kv[1])), flush=True)


if __name__ == "__main__":
    main()
