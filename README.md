# ECE201C Project 3: Block Diffusion Decoding in RAPID-LLM

Richard Yan

This repository extends [RAPID-LLM](https://github.com/gkarfakis19/Rapid-LLM) (course branch) to
model **block diffusion** decoding and compares it with normal autoregressive (AR) decoding of
Llama-3.1-8B on one A100 80 GB GPU.

A block diffusion model writes its output in blocks of **b** tokens. Each block is refined over
**R** model calls, and every call processes all b positions of the block at once while reading the
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

- Each block takes exactly **R** model calls. The optional extra call that saves the finished
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

`project3/run_cases.py` generates one model config per case, runs RAPID-LLM, and writes every
run's config, console output and result files to `project3/results/<suite>/`, plus a
`summary.csv` per suite:

```bash
python project3/run_cases.py diffusion_b8_R4
```

| Suite | Cases |
|---|---|
| `baseline` | AR decode, batch 1/16/64, prompt 512, output 128/256 (recorded with the unmodified code) |
| `baseline_after_change` | Same cases with the modified code and no `project_param` |
| `neutral_b1_R1` | Same cases with b = 1, R = 1 |
| `diffusion_b8_R4` | Required evaluation: same cases with b = 8, R = 4 |
| `study_a_b_R` | b = 4/8/16/32, R = 2/4/8 (R < b), batch 1/16/64, prompt 512, output 128 |
| `study_b_batch` | Batch 1–128, AR vs b = 8, R = 4, prompt 512, output 128 |
| `study_c_prompt` | Prompt 2K/8K/32K, batch 1/8/32, AR vs b = 8, R = 4, output 128 |

Run one suite at a time: all runs share the `output/LLM/` folder.

**Check:** `baseline_after_change` and `neutral_b1_R1` reproduce `baseline` exactly (result files
byte-for-byte identical).

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

- **Memory-bound (small batch):** a call costs about the same for b tokens as for one, so the
  decode speedup is close to b/R (e.g. 1.99× for b = 8, R = 4 and 14.5× for b = 32, R = 2 at
  batch 1).
- **Compute-bound (large batch):** each token is processed R times, so R dominates the cost. For
  b = 8, R = 4, AR decoding catches up between batch 48 (1.03× total) and 64 (0.93×).
- **Long prompts:** prefill dominates total time, so a 1.85× decode speedup at batch 1 shrinks to
  1.13× end to end at a 32K prompt.
- **Peak memory** is set by prefill and is the same as AR in every case; the decode-phase peak
  grows by at most 0.5 GiB.

## Repository layout

| Path | Contents |
|---|---|
| `project3/run_cases.py` | Runs the suites above and collects the results |
| `project3/results/` | Configs, console output, result files and `summary.csv` for every run |
| `COURSE_README.md`, `PROJECTS.md` | Course project descriptions and code pointers |
| `RAPID-LLM_README.md` | Original RAPID-LLM README: setup, configs and how a decode step is priced |

RAPID-LLM is licensed under the Apache License 2.0 (see `LICENSE`).
