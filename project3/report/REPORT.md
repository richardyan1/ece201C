# Project 3 results regenerated from `project3/results/`

Every section names the slide it reproduces. Llama-3.1-8B, one A100 80 GB.

## Slide 2: time per normal decode step (prompt 512)

| Batch | Step time |
| --- | --- |
| 1 | 12.1 ms |
| 16 | 13.5 ms |
| 64 | 17.8 ms |

## Slide 6: checks

- b=8 call at batch 64 (512 tokens): 40.5 ms; RAPID-LLM's own 512-token prefill (batch 1): 36.6 ms; difference 11%.
- Existing tests: run `pytest -q` (36 passed, 2 skipped).

## Slide 7: required runs, b = 8, R = 4, prompt 512

| Batch | Output | Normal total | Block diffusion total | Speedup | Model calls | Peak memory | Decode-phase peak change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1.59 s | 0.81 s | 1.95x | 128 -> 64 | 15.1 GiB | +0.00 GiB |
| 1 | 256 | 3.14 s | 1.59 s | 1.97x | 256 -> 128 | 15.1 GiB | +0.00 GiB |
| 16 | 128 | 2.19 s | 1.50 s | 1.46x | 128 -> 64 | 17.9 GiB | +0.03 GiB |
| 16 | 256 | 3.94 s | 2.55 s | 1.54x | 256 -> 128 | 17.9 GiB | +0.03 GiB |
| 64 | 128 | 4.13 s | 4.43 s | 0.93x (slower) | 128 -> 64 | 26.8 GiB | +0.11 GiB |
| 64 | 256 | 6.51 s | 7.10 s | 0.92x (slower) | 256 -> 128 | 26.8 GiB | +0.11 GiB |

## Slide 8: where normal decode catches up (prompt 512, output 128)

Break-even (decode speedup = 1, linear interpolation between the two batch sizes around it): batch 53.2.

| Batch | Diffusion call | Normal step | Ratio |
| --- | --- | --- | --- |
| 1 | 12.2 ms | 12.1 ms | 1.0x |
| 16 | 16.2 ms | 13.5 ms | 1.2x |
| 32 | 24.8 ms | 14.9 ms | 1.7x |
| 64 | 40.5 ms | 17.8 ms | 2.3x |
| 128 | 71.9 ms | 25.4 ms | 2.8x |

Chart data: `slide08_batch_sweep.csv`; chart: `slide08_batch_sweep.png`.

## Slide 9: decode-time speedup by block size b and rounds R (prompt 512, output 128)

| b | R | b/R | Calls | Batch 1 | Batch 16 | Batch 64 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 64 | 2.00x | 1.95x | 1.30x |
| 8 | 2 | 4 | 32 | 3.98x | 3.33x | 1.77x |
| 8 | 4 | 2 | 64 | 1.99x | 1.66x | 0.88x |
| 16 | 2 | 8 | 16 | 7.92x | 4.59x | 2.15x |
| 16 | 4 | 4 | 32 | 3.96x | 2.29x | 1.08x |
| 16 | 8 | 2 | 64 | 1.98x | 1.15x | 0.54x |
| 32 | 2 | 16 | 8 | 14.46x | 4.05x | 1.53x |
| 32 | 4 | 8 | 16 | 7.23x | 2.03x | 0.76x |
| 32 | 8 | 4 | 32 | 3.62x | 1.01x | 0.38x |

## Slide 10: long prompts, b = 8, R = 4, output 128 (decode speedup / total speedup)

| Prompt | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- |
| 512 | 1.99x / 1.95x | 1.93x / 1.73x | 1.20x / 1.13x |
| 2K | 1.98x / 1.84x | 1.88x / 1.42x | 1.22x / 1.07x |
| 8K | 1.95x / 1.53x | 1.74x / 1.14x | 1.27x / 1.03x (over 80 GB) |
| 32K | 1.85x / 1.13x | 1.53x / 1.03x (over 80 GB) | 1.30x / 1.01x (over 80 GB) |

- Prefill share of block-diffusion time at a 32K prompt, batch 1: 85% (5.37 s of 6.36 s).
- Peak memory of the over-80 GB cells: 110 to 393 GiB (prefill of the whole batch; identical for both methods).

## Slide 11: memory

- Largest decode-phase peak increase of block diffusion over normal decode, across studies A, B and C: 0.48 GiB. Max peak memory (prefill) is identical for both methods in every run.

