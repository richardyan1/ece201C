# ECE201C Project 3: Block Diffusion Decoding in RAPID-LLM

Richard Yan

This repository extends [RAPID-LLM](https://github.com/gkarfakis19/Rapid-LLM) (course branch) to
model block diffusion decoding and compares it with normal autoregressive (AR) decoding of
Llama-3.1-8B on one A100 80 GB GPU.

A block diffusion model writes its output in blocks of b tokens. Each block is refined over
R model calls, and every call processes all b positions of the block at once while reading the
KV cache of all earlier text. Compared with AR decoding (one token per call), a call reads the
weights and KV cache once for b tokens, but does about b times the compute.

## What was changed

All settings go in the model YAML under `project_param` (course helper H1). Without that block, or
with `block_size: 1` and `refine_rounds: 1`, every number is identical to the original RAPID-LLM.

```yaml
project_param:
  block_size: 8      # b: tokens per block, processed together in every call
  refine_rounds: 4   # R: model calls per block
```

| File | Change |
|---|---|
| `llm_util.py` (`autoregressive_decoder_gemm`) | Decode GEMM shapes use `query_tokens` = b instead of 1: QKV, attention (b queries per head against the full context), output projection, FFNs and output head. |
| `inference_timing.py` (`_build_decode_transformer_results`) | Norms, residuals and the output head process b tokens per call (`output_seq_len = query_tokens`). |
| `inference_timing.py` (`calc_time`) | The decode peak-memory pass uses b tokens per call. |
| `simulate_inference_graph.py` (`build_decode_graph`, `_generate_sample_points`, `_integrate_decode_samples`) | The decode loop walks blocks instead of tokens: block k attends to `prompt + (k+1)·b` keys, costs R calls, and sampling/integration run over the O/b blocks. Prints the model-call count. |
| `configs/model-config/course_project/Llama3.1-8B_block_diffusion_inf.yaml` | Llama-3.1-8B with `block_size: 8`, `refine_rounds: 4`, prompt 512, output 128. |

## Modeling assumptions

- Each block takes exactly R model calls. The optional extra call that saves the finished
  block's K/V is not modeled; adding it would multiply the call count by (R+1)/R.
- Every call attends to all earlier text plus all b positions of the current block.
- The output length must be a multiple of b.
- Prefill is unchanged; total time = prefill + decode.
- Time to first token = prefill + the first block (R calls).
- `sample_every` in the model YAML now counts blocks, not tokens.
- The decode tokens/s line printed by RAPID-LLM assumes one token per step and is not valid for
  b > 1; use total and decode time instead.
- MLA attention is not supported for b > 1.

## Setup and running

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q
```

Run one configuration (on Windows, set `PYTHONUTF8=1` first):

```bash
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/Llama3.1-8B_block_diffusion_inf.yaml
```

The run prints `[decode] block_size=8, refine_rounds=4, blocks=16, model calls=64` followed by the
prefill, decode and total time; peak memory is in `output/LLM/memory-summary/`.

## Reproducing the results

All results in the presentation can be regenerated with one command, run from the repository root
after the setup above.

```bash
pip install matplotlib                  # only for the slide 8 chart
python project3/reproduce.py            # full run, about 25 minutes
python project3/reproduce.py --quick    # skips the long-prompt study (study C), about 3 minutes
```

`reproduce.py` does two things in order:

1. Runs the simulations. It calls `project3/run_cases.py` once for each suite below, one after
   another (the runs share `output/LLM/`, so suites cannot run in parallel). Each run's config,
   console output and result files go to `project3/results/<suite>/<case>/`, and each suite's numbers
   go to `project3/results/<suite>/summary.csv`.
2. Builds the report. It calls `project3/make_report.py`, which reads the CSVs and writes
   `project3/report/REPORT.md` (every table and number from the slides, labelled by slide),
   `project3/report/slide08_batch_sweep.png` (the slide 8 chart) and `slide08_batch_sweep.csv` (its data).

With `--quick`, study C is not re-run and the report uses the committed study C results.
To regenerate only the report from the existing CSVs: `python project3/make_report.py`.
To run a single suite by hand: `python project3/run_cases.py <suite>`.

### Suites

| Suite | Cases | Run by `reproduce.py` |
|---|---|---|
| `baseline_after_change` | Normal (AR) decode, batch 1/16/64, prompt 512, output 128/256: the comparison for the required runs | yes |
| `diffusion_b8_R4` | Required evaluation: same cases with b = 8, R = 4 | yes |
| `study_a_b_R` | b = 4/8/16/32, R = 2/4/8 (R < b), batch 1/16/64, prompt 512, output 128 | yes |
| `study_b_batch` | Batch 1 to 128 (including 49 and 50, around the break-even), AR vs b = 8, R = 4, prompt 512, output 128 | yes |
| `study_c_prompt` | Prompt 2K/8K/32K, batch 1/8/32, AR vs b = 8, R = 4, output 128 | yes (not with `--quick`) |
| `baseline` | Same cases as `baseline_after_change`, recorded with the unmodified RAPID-LLM code | no (reference only) |
| `neutral_b1_R1` | Same cases with b = 1, R = 1 | no (side check) |

### Which output goes with which slide

| Presentation | Where to find it |
|---|---|
| Slide 2 (time per normal decode step) | `REPORT.md`, section "Slide 2" |
| Slide 6 (checks) | `REPORT.md`, section "Slide 6" (prefill comparison); tests: `pytest -q` (36 pass) |
| Slide 7 (required runs) | `REPORT.md`, section "Slide 7"; raw: `results/diffusion_b8_R4/summary.csv` and `results/baseline_after_change/summary.csv` |
| Slide 8 (where normal decode catches up) | `REPORT.md`, section "Slide 8"; chart `project3/report/slide08_batch_sweep.png`; raw: `results/study_b_batch/summary.csv` |
| Slide 9 (b and R grid) | `REPORT.md`, section "Slide 9"; raw: `results/study_a_b_R/summary.csv` |
| Slide 10 (long prompts) | `REPORT.md`, section "Slide 10"; raw: `results/study_c_prompt/summary.csv` |
| Slide 11 (memory) | `REPORT.md`, section "Slide 11" |

Cases whose peak memory is above 80 GB (the prefill of the whole batch, the same for both methods)
are marked `over 80 GB` in `REPORT.md` and have `fits_80gb = False` in the CSVs. In the block
diffusion rows of the CSVs, ignore the `tok_s_start` and `tok_s_end` columns (see the assumptions).

## Results

Required evaluation: Llama-3.1-8B, one A100 80 GB, prompt 512, b = 8, R = 4.

| Batch | Output | AR total | Block diffusion total | Speedup | Model calls (AR → BD) | Peak memory |
|---|---|---|---|---|---|---|
| 1 | 128 | 1.59 s | 0.81 s | 1.95× | 128 → 64 | 15.1 GiB |
| 1 | 256 | 3.14 s | 1.59 s | 1.97× | 256 → 128 | 15.1 GiB |
| 16 | 128 | 2.19 s | 1.50 s | 1.46× | 128 → 64 | 17.9 GiB |
| 16 | 256 | 3.94 s | 2.55 s | 1.54× | 256 → 128 | 17.9 GiB |
| 64 | 128 | 4.13 s | 4.43 s | 0.93× | 128 → 64 | 26.8 GiB |
| 64 | 256 | 6.51 s | 7.10 s | 0.92× | 256 → 128 | 26.8 GiB |

Findings:

- Small batch (memory-bound): a call costs about the same for b tokens as for one, so the
  decode speedup is close to b/R (e.g. 1.99× for b = 8, R = 4 and 14.5× for b = 32, R = 2 at
  batch 1).
- Large batch (compute-bound): each token is processed R times, so R dominates the cost. For
  b = 8, R = 4, AR decoding catches up between batch 48 (1.03× total) and batch 49 (0.93×): the cost of a call jumps there instead of growing smoothly.
- Long prompts: prefill dominates total time, so a 1.85× decode speedup at batch 1 shrinks to
  1.13× end to end at a 32K prompt.
- Peak memory is set by prefill and is the same as AR in every case; the decode-phase peak
  grows by at most 0.5 GiB.

## Repository layout

| Path | Contents |
|---|---|
| `project3/reproduce.py` | One-shot reproduction: runs the suites, then builds the report |
| `project3/run_cases.py` | Runs one suite and collects its results |
| `project3/make_report.py` | Builds `REPORT.md` and the slide 8 chart from the result CSVs |
| `project3/report/` | Generated report, chart and chart data |
| `project3/results/` | Configs, console output, result files and `summary.csv` for every run |
| `COURSE_README.md`, `PROJECTS.md` | Course project descriptions and code pointers |
| `RAPID-LLM_README.md` | Original RAPID-LLM README: setup, configs and how a decode step is priced |
