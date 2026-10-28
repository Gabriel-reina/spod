# SPOD on rustc-perf

This experiment trains Qwen3-1.7B-Base to propose Rust compiler performance improvements using public rustc-perf triage reports and rust-lang/rust pull requests.

From the repository root:

```bash
python scripts/install_roll_overlay.py --roll_dir ../ROLL --scenario rustc
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 bash scripts/run_reproduction.sh rustc
```

The dataset contains 1,591 states: 1,414 training prompts and 177 test prompts. Categories are `B1_backend_codegen`, `B2_type_trait_solver`, `B3_query_incremental`, `B4_frontend_parse`, `B5_library_std` and `B6_infra_tooling`.

The pipeline uses seed 42 and reports the same eight metrics as nanoGPT. Results are written under `rustc_perf/output/`. Rebuild the datasets from this directory with `python data_pipeline/build_all.py`.

See the [main README](../README.md) for installation, the reward operator, metrics and licenses.
