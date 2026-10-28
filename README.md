# SPOD: Selective Popularity Debiasing for RLVR

SPOD adjusts reward-model scores during GRPO training using category-level popularity signals while preserving each category’s highest-scoring response in the rollout group.

```text
d_i = beta * std(r_group) * phi(category_i)
r_new_i = r_i           for the highest-scoring response of each category
r_new_i = r_i - d_i     for other responses
```

The repository includes two experiments, their public datasets, data-construction pipelines, SFT and reward-model training, GRPO and SPOD training, and evaluation.

| Experiment | Base model | Training prompts | Test prompts |
|---|---|---:|---:|
| nanoGPT speedrun | Qwen/Qwen2.5-0.5B | 79 | 28 |
| rustc-perf | Qwen/Qwen3-1.7B-Base | 1,414 | 177 |

## Environment

Use Linux, Python 3.10, CUDA 12.8 and two GPUs. The reference run uses two NVIDIA A100 80 GB GPUs. All training and evaluation seeds are 42.

| Package | Version |
|---|---|
| ROLL | `d3506ea9e280ae720fb74df9f82db6efac1c72d5` |
| PyTorch | 2.8.0 |
| vLLM | 0.11.0 |
| Transformers | 4.57.0 |
| FlashAttention | 2.8.1 |
| Transformer Engine | 2.8.0 |
| Click | 8.2.1 |

Install the environment from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# A100 uses CUDA architecture 80.
export FLASH_ATTN_CUDA_ARCHS=80
bash scripts/setup_environment.sh ../ROLL
```

The installer checks out the fixed public ROLL revision and installs its matching requirements and SPOD runtime dependencies. CUDA compilation requires a CUDA toolkit with `nvcc` and a C++ compiler.

## Run nanoGPT

```bash
source .venv/bin/activate
python scripts/install_roll_overlay.py --roll_dir ../ROLL --scenario nanogpt
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 bash scripts/run_reproduction.sh nanogpt
```

## Run rustc-perf

```bash
source .venv/bin/activate
python scripts/install_roll_overlay.py --roll_dir ../ROLL --scenario rustc
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 bash scripts/run_reproduction.sh rustc
```

Each command runs SFT, reward-model training, phi measurement, SFT evaluation, 50 steps of GRPO, 50 steps of SPOD, checkpoint conversion and evaluation. The released static phi table is used for SPOD; the measurement stage writes a separate table. Install each experiment’s overlay before running it. Use a separate ROLL checkout for simultaneous experiments.

| Arm | Method |
|---|---|
| `v` | GRPO |
| `bhap` | SPOD: `rmscore`, `MIN=0`, `EXEMPT=top1`, `beta=1.0` |

## Evaluation

The eight headline metrics are Hit@1, Hit@8, Coverage, TV_to_prior, Ideas, MaxClass, MaxSameClass and Collapse. Hit@1 selects the highest RM-scored response from eight samples. Quality metrics use prompts with gold labels; diversity metrics use all test prompts. Evaluation measures generated ideas against historical change categories.

nanoGPT results are written to `output/spod_eval`; rustc-perf results are written to `rustc_perf/output/spod_eval`.

```bash
python spod_metrics.py --dir output/spod_eval \
  --arms sft,v_step49,bhap_step49 --base v_step49
```

## Repository

| Path | Contents |
|---|---|
| `configs/` | nanoGPT GRPO configuration and runtime versions |
| `entry/` | Training, evaluation and checkpoint conversion |
| `roll_spod/` | Reward operator and ROLL reward worker |
| `scripts/` | Installation and experiment launchers |
| `data/`, `data_pipeline/` | nanoGPT datasets and construction pipeline |
| `rustc_perf/` | rustc-perf datasets, configuration and launchers |
| `tests/` | CPU regression checks |

SFT, RM and GRPO use raw `prompt + "\n" + response` text through the `spod_raw` template. The nanoGPT launchers use the `_clean` SFT/RM datasets, excluding test-state examples. rustc-perf uses a chronological training/test split.

## Tests

```bash
python -m unittest discover -s tests -q
```

## Data sources

nanoGPT data comes from public [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun records and GitHub pull requests. rustc-perf data comes from public [rustc-perf](https://github.com/rust-lang/rustc-perf) triage reports and [rust-lang/rust](https://github.com/rust-lang/rust) pull requests. Processed datasets and their construction scripts are included.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
