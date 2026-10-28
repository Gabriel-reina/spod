"""proxy RM training: Bradley-Terry pairwise, produces a standard HF token-classification head.

Why AutoModelForTokenClassification(num_labels=1) rather than SequenceClassification:
the GRPO-side RewardWorker.forward_func_values expects exactly this protocol (strategy_name: hf_infer,
model_type: auto_token_classification, taking the logit of the last non-pad token). Choosing it means
zero glue downstream.

Input format = prompt + "\n" + response, plain-text concatenation, no chat template.
SFT / RM / online GRPO must be exactly identical, otherwise the RM's online scoring is
out-of-distribution. The prompt is generated once by data_pipeline/build_train_data.py and written to
file; here we only read, not concatenate.

Data is read directly from data/ inside the code package.

WARNING: this RM's class bias is a disease to keep, not a bug: it systematically over-scores some
categories independent of their frequency (per-category RM mean peaks at the rarest B6_infra_tooling and
bottoms at B5_library_std; category frequency and RM mean are negatively correlated). No class
rebalancing, no class weighting, no balancing term in the loss -- SPOD corrects this bias downstream.

How to run (2 GPUs = 2 workers, 1 GPU per worker):
  after setting RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT on each rank:
  python entry/start_spod_rm_train.py --world_size 2
  (the shipped data is already hold-out clean via the per-item "split" field)
"""
import argparse
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CATS = ["B1_backend_codegen", "B2_type_trait_solver", "B3_query_incremental",
        "B4_frontend_parse", "B5_library_std", "B6_infra_tooling"]
# the two trivial lower bounds it must beat (measured by data_pipeline/analyze_bias.py)
TRIVIAL_LEN_ONLY = 0.519
TRIVIAL_NCHANGE_ONLY = 0.735


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(pos, neg):
    """Mann-Whitney U, ties count as 0.5."""
    if not pos or not neg:
        return float("nan")
    hit = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return hit / (len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default="Qwen/Qwen3-1.7B-Base")  # base, same as SFT
    parser.add_argument("--revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                        help="pinned Hugging Face revision; ignored for a local model directory")
    parser.add_argument("--output_dir", default="./output_spod_rm")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr_llm", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--unfreeze_layers", type=int, default=4,
                        help="unfreeze only the last N layers + head; full-param finetune on this few unique texts would overfit")
    parser.add_argument("--pair_batch", type=int, default=8, help="pairs per rank per step, actual forward is 2x")
    parser.add_argument("--max_len", type=int, default=768)  # measured longest 626 tokens
    parser.add_argument("--seed", type=int, default=42)
    # rustc data is date-split hold-out (per-item "split" field); training pairs reference train-split
    # items only, so the default "" (no suffix) is already hold-out clean -- no separate _clean file.
    parser.add_argument("--data_tag", default="")
    parser.add_argument("--tables", default=None)  # accepted for external-launcher compatibility, unused
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ.setdefault("LOCAL_RANK", "0")
    workdir = os.path.abspath(args.output_dir)
    os.makedirs(workdir, exist_ok=True)

    # Remote-code execution is disabled. Pin Hub models; local checkpoint dirs need no revision.
    model_path = args.pretrain
    load_kwargs = {"trust_remote_code": False}
    if args.revision and not os.path.isdir(model_path):
        load_kwargs["revision"] = args.revision

    tag = args.data_tag
    items = {r["uid"]: r for r in load_jsonl(os.path.join(DATA_DIR, f"rm_items_prompted{tag}.jsonl"))}
    train_pairs = load_jsonl(os.path.join(DATA_DIR, f"rm_pairs_slim_train{tag}.jsonl"))
    val_pairs = load_jsonl(os.path.join(DATA_DIR, f"rm_pairs_slim_val{tag}.jsonl"))
    print(f"[rank{rank}] data_tag={tag!r} items={len(items)} train_pair={len(train_pairs)} "
          f"val_pair={len(val_pairs)} model={model_path}", flush=True)

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    multi = args.world_size > 1
    if multi:
        # timeout relaxed to 2h: NCCL watchdog defaults to 600s, longer inter-rank waits get misjudged as a comm hang
        from datetime import timedelta
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=args.world_size,
                                timeout=timedelta(seconds=7200))

    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # locate the last valid token via attention_mask.sum()-1

    net = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=1, torch_dtype=torch.float32, **load_kwargs)
    # fp32 weights + autocast(bf16) forward: at lr 1e-5 the update magnitude would be rounded away under
    # bf16 mantissa (8bit), so a fp32 master weight must be kept. 0.5B fp32 has plenty of memory on A100.
    net.config.pad_token_id = tokenizer.pad_token_id
    body = net.base_model
    for p in body.parameters():
        p.requires_grad = False
    for layer in body.layers[-args.unfreeze_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    if hasattr(body, "norm"):
        for p in body.norm.parameters():
            p.requires_grad = True
    # the head attribute name differs across architectures (Qwen2ForTokenClassification calls it score, BERT-family calls it classifier)
    head = getattr(net, "score", None) or getattr(net, "classifier", None)
    assert head is not None, "classification head not found, check the TokenClassification structure of your transformers version"
    for p in head.parameters():
        p.requires_grad = True
    net.to(device)
    model = DDP(net, device_ids=[local_rank]) if multi else net
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[rank{rank}] trainable params {n_train/1e6:.1f}M / "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)

    def text_of(uid):
        it = items[uid]
        return it["prompt"] + "\n" + it["response"]

    def score(uids, train_mode):
        """The logit of the last non-pad token is the reward -- identical to how GRPO takes it."""
        enc = tokenizer([text_of(u) for u in uids], truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = (model if train_mode else net)(**enc).logits.squeeze(-1)
        logits = logits.float()
        last = enc["attention_mask"].sum(dim=1) - 1
        return logits[torch.arange(logits.size(0), device=device), last]

    head_params = [p for p in head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    llm_params = [p for p in net.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": llm_params, "lr": args.lr_llm},
         {"params": head_params, "lr": args.lr_head}], weight_decay=0.0)
    # floor split guarantees equal step counts across ranks, otherwise a rank exiting early hangs the rest on all_reduce
    steps_per_epoch = len(train_pairs) // (args.world_size * args.pair_batch)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, steps_per_epoch, total_steps)
    print(f"[rank{rank}] steps/epoch={steps_per_epoch} total={total_steps}", flush=True)

    def evaluate():
        """Sharded eval + all_gather: all ranks participate.
        A rank0-only eval leaves the other ranks idling until the NCCL watchdog times out and aborts."""
        net.eval()
        my = list(range(rank, len(val_pairs), args.world_size)) if multi else list(range(len(val_pairs)))
        local = []
        with torch.no_grad():
            for i in range(0, len(my), args.pair_batch):
                chunk = [val_pairs[j] for j in my[i:i + args.pair_batch]]
                s = score([c["c"] for c in chunk] + [c["r"] for c in chunk], False)
                n = len(chunk)
                for k, c in enumerate(chunk):
                    local.append((c["kind"], float(s[k]), float(s[n + k])))
        if multi:
            gathered = [None] * args.world_size
            dist.all_gather_object(gathered, local)
            allr = [x for g in gathered for x in g]
        else:
            allr = local
        stat = {}
        for kind, sc, sr in allr:
            for key in ("ALL", kind):
                h, t = stat.get(key, (0, 0))
                stat[key] = (h + (1 if sc > sr else 0), t + 1)
        net.train()
        return {k: (v[0] / v[1], v[1]) for k, v in stat.items()}

    def score_all_items():
        """Score every library item (run alone on rank0, for bias diagnostics + downstream lookup)."""
        net.eval()
        uids = list(items)
        out = {}
        with torch.no_grad():
            for i in range(0, len(uids), 16):
                chunk = uids[i:i + 16]
                for u, v in zip(chunk, score(chunk, False).tolist()):
                    out[u] = v
        net.train()
        return out

    save_dir = os.path.join(workdir, "rm_hf")
    best = -1.0
    gstep = 0
    for epoch in range(args.epochs):
        order = list(range(len(train_pairs)))
        random.Random(args.seed + epoch).shuffle(order)      # same seed same order across ranks
        shard = order[rank::args.world_size][: steps_per_epoch * args.pair_batch] if multi \
            else order[: steps_per_epoch * args.pair_batch]
        for i in range(0, len(shard), args.pair_batch):
            chunk = [train_pairs[j] for j in shard[i:i + args.pair_batch]]
            s = score([c["c"] for c in chunk] + [c["r"] for c in chunk], True)
            n = len(chunk)
            # Bradley-Terry: only the difference enters the loss, the absolute scale is unidentifiable, no absolute term added
            loss = -F.logsigmoid(s[:n] - s[n:]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1
            if rank == 0 and gstep % 50 == 0:
                print(f"step {gstep}/{total_steps} loss={loss.item():.4f} "
                      f"lr_llm={scheduler.get_last_lr()[0]:.2e}", flush=True)
        m = evaluate()
        if rank == 0 or os.environ.get("SPOD_SAVE_LOCAL_REPLICA", "0") == "1":
            # Shared filesystems use one writer; separate worker disks may opt into local replicas.
            print(f"\n[epoch {epoch + 1}] val pair accuracy", flush=True)
            for k in sorted(m, key=lambda x: -m[x][1]):
                print(f"   {k:20s} {m[k][0]:.3f}  (n={m[k][1]})", flush=True)
            if m["ALL"][0] > best:
                best = m["ALL"][0]
                os.makedirs(save_dir, exist_ok=True)
                net.save_pretrained(save_dir)      # standard HF, consumed directly by GRPO
                tokenizer.save_pretrained(save_dir)
                print(f"best ckpt updated (val_pair_acc={best:.4f}) -> {save_dir}", flush=True)
        if multi:
            dist.barrier()

    if rank == 0:
        # ---- bias diagnostics: category popularity vs RM mean ----
        sc = score_all_items()
        with open(os.path.join(workdir, "rm_scores.jsonl"), "w") as f:
            for u, v in sc.items():
                f.write(json.dumps({"uid": u, "cat": items[u]["cat"],
                                    "tier": items[u]["tier"], "rm_score": v}) + "\n")
        keep = [u for u in sc if items[u]["tier"] != "reject" and items[u].get("split") == "train"]
        tot = len(keep)
        print("\n=== bias diagnostics: per-category RM mean (listed items, train split only) ===", flush=True)
        print(f"   {'category':20s}{'n':>4s}{'share':>8s}{'RM_mean':>10s}", flush=True)
        rows = []
        for c in CATS:
            sub = [sc[u] for u in keep if items[u]["cat"] == c]
            if sub:
                mu = sum(sub) / len(sub)
                rows.append((c, len(sub), 100 * len(sub) / tot, mu))
                print(f"   {c:20s}{len(sub):4d}{100*len(sub)/tot:7.1f}%{mu:10.3f}", flush=True)
        if len(rows) >= 2:
            hi, lo = max(rows, key=lambda r: r[3]), min(rows, key=lambda r: r[3])
            print(f"   highest {hi[0]}({hi[3]:.3f})  lowest {lo[0]}({lo[3]:.3f})  diff {hi[3]-lo[3]:.3f}", flush=True)
            xs = [r[2] for r in rows]
            ys = [r[3] for r in rows]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            r = num / den if den else 0.0
            print(f"   share vs mean correlation r={r:+.3f}"
                  f"{'   <- more popular scores higher, bias exists, the method has a target' if r > 0 else ''}", flush=True)
        pos = [sc[u] for u in sc if items[u]["tier"] == "strong"]
        neg = [sc[u] for u in sc if items[u]["tier"] == "reject"]
        print(f"\n   strong vs reject AUC = {auc(pos, neg):.4f}  (n_pos={len(pos)} n_neg={len(neg)})", flush=True)
        print("\n=== the two trivial lower bounds it must beat ===", flush=True)
        print(f"   text length only   {TRIVIAL_LEN_ONLY}", flush=True)
        print(f"   bundled change count only {TRIVIAL_NCHANGE_ONLY} (only 68 pairs decidable)", flush=True)
        print(f"   this run val ALL   {best:.3f}  "
              f"{'passed' if best > TRIVIAL_NCHANGE_ONLY else 'did not beat 0.735, the RM learned nothing beyond trivial'}",
              flush=True)

        print(f"\n[rank0] RM checkpoint saved at {save_dir}", flush=True)
    if multi:
        dist.barrier()
        dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
