"""SFT cold-start: give the model the basic ability to "emit, for a given baseline state, a format-valid speedup idea".

Why this step is required: GRPO's advantage is an in-group relative quantity; if the base model's 8 rollouts
under the same prompt are all format garbage, the in-group std is just noise, the reward differences carry no
information, and the method has nothing to act on.
SFT's only job is to move the output distribution to "format-valid + covering the 5 categories", not to chase a low loss.

Uses positive samples only (tier in strong/null), with categories already upsampled to balance on the data side (minority cap 3x, majority not cut):
  - positive only: SFT is behavior cloning, feeding it reject teaches it to output rejected changes
  - category balance: in the base library C1 is 33.6% / C5 is 4.8% (7:1). If after SFT the rollout categories
    collapse to C1, there is no category diversity in-group, and any later per-category operation degrades to a no-op.
    Balancing is to preserve category entropy.
  (Note the opposite of RM: on the RM side we absolutely do not balance -- the category bias is exactly the disease it should keep.)

Input format = prompt + "\n" + response, plain text, no chat template, exactly consistent with RM / online GRPO.
The prompt segment's label is set to -100, loss is computed only on the response segment.

Output is a standard HF bf16 dir, usable directly as GRPO's pretrain (no megatron->HF convert step needed).

How to run (2 GPUs):
  after each rank sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT:
  python entry/start_spod_sft_train.py --world_size 2 --data_tag _clean
"""
import argparse
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CATS = ["C1_sys_kernel", "C2_optimizer", "C3_architecture", "C4_attention", "C5_loss"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default="Qwen/Qwen2.5-0.5B")  # same base model as RM
    parser.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987",
                        help="pinned Hugging Face revision; ignored for a local model directory")
    parser.add_argument("--output_dir", default="./output_spod_sft")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=2, help="samples per rank per step")
    parser.add_argument("--max_len", type=int, default=768)  # measured max 626 tokens
    parser.add_argument("--seed", type=int, default=42)
    # data version suffix: "" = full data (it has seen the test28 gold, leaks);
    # "_clean" = remove all items attached to test28 states, forming a full-pipeline hold-out with grpo_prompts_test28 (used by open-source).
    parser.add_argument("--data_tag", default="_clean")
    parser.add_argument("--tables", default=None)  # for compatibility with external launcher injection, received but unused
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
    train_rows = load_jsonl(os.path.join(DATA_DIR, f"sft_train{tag}.jsonl"))
    val_rows = load_jsonl(os.path.join(DATA_DIR, f"sft_val{tag}.jsonl"))
    print(f"[rank{rank}] data_tag={tag!r} train={len(train_rows)} val={len(val_rows)} "
          f"model={model_path}", flush=True)
    if rank == 0:
        c = Counter(r["cat"] for r in train_rows)
        print("   training-set category distribution (after balancing): " +
              "  ".join(f"{k.split('_')[0]}={c.get(k, 0)}" for k in CATS), flush=True)

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    multi = args.world_size > 1
    if multi:
        from datetime import timedelta
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=args.world_size,
                                timeout=timedelta(seconds=7200))

    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    net = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, **load_kwargs)
    # fp32 weights + autocast(bf16) forward: the update magnitude at lr 1e-5 would be rounded away under bf16's 8-bit mantissa.
    # No gradient checkpointing: a 0.5B model at 139 items x maxlen768 x batch2 has plenty of memory headroom,
    # enabling it would instead require handling DDP + reentrant checkpoint unused-param issues.
    net.config.pad_token_id = tokenizer.pad_token_id
    net.to(device)
    model = DDP(net, device_ids=[local_rank]) if multi else net

    def encode(rows):
        """prompt segment label=-100, loss only on the response segment; append eos at response end to teach it to stop."""
        full, plens = [], []
        for r in rows:
            p = r["prompt"] + "\n"
            full.append(p + r["response"] + tokenizer.eos_token)
            plens.append(len(tokenizer(p, add_special_tokens=False)["input_ids"]))
        enc = tokenizer(full, truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, pl in enumerate(plens):
            labels[i, :min(pl, labels.size(1))] = -100
        enc["labels"] = labels
        return {k: v.to(device) for k, v in enc.items()}

    # foreach=False: the multi-tensor AdamW allocates full-size temporary tensors for the whole parameter group;
    # the 4B fp32 parameters themselves already take 64 GB, and that temp copy pushes a single GPU past 80 GB.
    # Single-tensor mode is mathematically equivalent, just a bit slower.
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.95), foreach=False)
    # floor division of the split keeps step counts equal across ranks, otherwise a rank that exits early hangs the rest at all_reduce
    steps_per_epoch = len(train_rows) // (args.world_size * args.batch)
    total_steps = steps_per_epoch * args.epochs
    assert steps_per_epoch > 0, "data is less than one step, check world_size/batch"
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(0.03 * total_steps)), total_steps)
    print(f"[rank{rank}] steps/epoch={steps_per_epoch} total={total_steps}", flush=True)

    def evaluate():
        """Sharded evaluation + all_gather, all ranks participate (rank0-exclusive would trigger the NCCL watchdog)."""
        net.eval()
        my = list(range(rank, len(val_rows), args.world_size)) if multi \
            else list(range(len(val_rows)))
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(my), args.batch):
                chunk = [val_rows[j] for j in my[i:i + args.batch]]
                if not chunk:
                    continue
                enc = encode(chunk)
                ntok = int((enc["labels"] != -100).sum())
                if ntok == 0:      # the whole batch got truncated to prompt, loss would be nan, skip
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(**enc)
                tot_loss += float(out.loss) * ntok
                tot_tok += ntok
        if multi:
            g = [None] * args.world_size
            dist.all_gather_object(g, (tot_loss, tot_tok))
            tot_loss = sum(x[0] for x in g)
            tot_tok = sum(x[1] for x in g)
        net.train()
        return tot_loss / max(1, tot_tok)

    save_dir = os.path.join(workdir, "sft_hf")
    best = float("inf")
    gstep = 0
    print(f"[rank{rank}] epoch 0 (untrained) val_loss={evaluate():.4f}", flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)      # same seed same order across ranks
        shard = order[rank::args.world_size][: steps_per_epoch * args.batch] if multi \
            else order[: steps_per_epoch * args.batch]
        run_loss = 0.0
        for i in range(0, len(shard), args.batch):
            chunk = [train_rows[j] for j in shard[i:i + args.batch]]
            enc = encode(chunk)
            assert int((enc["labels"] != -100).sum()) > 0, "whole batch label is all -100, max_len too small"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            run_loss += float(loss)
            gstep += 1
        vl = evaluate()
        if rank == 0 or os.environ.get("SPOD_SAVE_LOCAL_REPLICA", "0") == "1":
            # Shared filesystems use one writer; separate worker disks may opt into local replicas.
            print(f"[epoch {epoch + 1}] train_loss={run_loss/max(1,steps_per_epoch):.4f} "
                  f"val_loss={vl:.4f}  lr={scheduler.get_last_lr()[0]:.2e} step={gstep}/{total_steps}",
                  flush=True)
            # select best by val_loss: overfitting is fast on 139 items, the last-step ckpt is not necessarily the best
            if vl < best:
                best = vl
                os.makedirs(save_dir, exist_ok=True)
                net.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"best ckpt updated (val_loss={best:.4f}) -> {save_dir}", flush=True)
        if multi:
            dist.barrier()

    if rank == 0:
        # ---- smoke test: see what it actually emits; format validity is SFT's only acceptance metric ----
        net.eval()
        prompts = load_jsonl(os.path.join(DATA_DIR, "prompts.jsonl"))
        sample = random.Random(args.seed).sample(prompts, min(8, len(prompts)))
        ok, cats = 0, Counter()
        print("\n=== generation smoke test (8 prompts, 1 sample each) ===", flush=True)
        with torch.no_grad():
            for p in sample:
                enc = tokenizer(p["prompt"] + "\n", return_tensors="pt",
                                truncation=True, max_length=args.max_len).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net.generate(**enc, max_new_tokens=80, do_sample=True,
                                       temperature=0.8, top_p=0.95,
                                       pad_token_id=tokenizer.pad_token_id)
                gen = tokenizer.decode(out[0][enc["input_ids"].size(1):],
                                       skip_special_tokens=True).strip()
                hit = next((c for c in CATS if gen.startswith(f"[{c}]")), None)
                if hit:
                    ok += 1
                    cats[hit] += 1
                print(f"   [{p['pid']}] {'OK ' if hit else 'BAD'} {gen[:110]}", flush=True)
        print(f"\n   format validity {ok}/{len(sample)}   category hits {dict(cats)}", flush=True)
        print("   ^ validity below 5/8 means the cold-start failed, GRPO in-group std will be noise, add epochs and retrain first",
              flush=True)

        print(f"\n[rank0] SFT checkpoint saved at {save_dir}", flush=True)
    if multi:
        dist.barrier()
        dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
