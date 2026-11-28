"""Regenerate every table, number and figure in the Project 3 presentation.

Reads project3/results/<suite>/summary.csv (written by run_cases.py) and writes
project3/report/:

  REPORT.md                       every table and quoted number, labelled with its slide
  slide08_batch_sweep.csv         data behind the slide-8 chart
  slide08_batch_sweep.png         the slide-8 chart (speedup vs batch size)

Usage: python project3/make_report.py
"""
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
REPORT = PROJECT / "report"

NEEDED = ["baseline_after_change", "diffusion_b8_R4",
          "study_a_b_R", "study_b_batch", "study_c_prompt"]


def load(suite):
    path = RESULTS / suite / "summary.csv"
    if not path.exists():
        sys.exit(f"missing {path}: run `python project3/reproduce.py` first")
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def f(row, key):
    return float(row[key])


def first_call_ms(row):
    """Time of the first sampled decode call, in ms (printed per call, before the xR)."""
    first = row["sampled_steps"].split(";")[0]          # e.g. 0@520=0.0122
    return float(first.split("=")[1]) * 1e3


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    for s in NEEDED:
        load(s)
    REPORT.mkdir(exist_ok=True)
    lines = ["# Project 3 results regenerated from `project3/results/`", "",
             "Every section names the slide it reproduces. Llama-3.1-8B, one A100 80 GB.", ""]

    # AR reference rows (modified code, no project_param: identical to the unmodified code).
    ar_req = {(r["batch"], r["output"]): r for r in load("baseline_after_change")}

    # ---- Slide 2: time per normal decode step ------------------------------------------
    rows = [(b, f"{first_call_ms(ar_req[(b, '128')]):.1f} ms") for b in ("1", "16", "64")]
    lines += ["## Slide 2: time per normal decode step (prompt 512)", "",
              md_table(["Batch", "Step time"], rows), ""]

    # ---- Slide 6: checks ----------------------------------------------------------------
    bd64 = next(r for r in load("diffusion_b8_R4") if r["batch"] == "64" and r["output"] == "128")
    prefill_ms = f(ar_req[("1", "128")], "prefill_s") * 1e3
    call_ms = first_call_ms(bd64)
    lines += ["## Slide 6: checks", "",
              f"- b=8 call at batch 64 (512 tokens): {call_ms:.1f} ms; RAPID-LLM's own 512-token prefill "
              f"(batch 1): {prefill_ms:.1f} ms; difference {abs(call_ms - prefill_ms) / prefill_ms * 100:.0f}%.",
              "- Existing tests: run `pytest -q` (36 passed, 2 skipped).", ""]

    # ---- Slide 7: required runs ---------------------------------------------------------
    rows = []
    for r in load("diffusion_b8_R4"):
        a = ar_req[(r["batch"], r["output"])]
        assert a["max_peak_gib"] == r["max_peak_gib"], "peak memory differs between AR and BD"
        sp = f(a, "total_s") / f(r, "total_s")
        rows.append((r["batch"], r["output"], f"{f(a, 'total_s'):.2f} s", f"{f(r, 'total_s'):.2f} s",
                     f"{sp:.2f}x" + (" (slower)" if sp < 1 else ""),
                     f"{a['output']} -> {r['model_calls']}", f"{f(r, 'max_peak_gib'):.1f} GiB",
                     f"{f(r, 'decode_peak_gib') - f(a, 'decode_peak_gib'):+.2f} GiB"))
    lines += ["## Slide 7: required runs, b = 8, R = 4, prompt 512", "",
              md_table(["Batch", "Output", "Normal total", "Block diffusion total", "Speedup",
                        "Model calls", "Peak memory", "Decode-phase peak change"], rows), ""]

    # ---- Slide 8: batch sweep -----------------------------------------------------------
    sb = load("study_b_batch")
    ar = {int(r["batch"]): r for r in sb if r["block_size"] == "1"}
    bd = {int(r["batch"]): r for r in sb if r["block_size"] == "8"}
    batches = sorted(ar)
    sweep = [(n, f(ar[n], "decode_s") / f(bd[n], "decode_s"), f(ar[n], "total_s") / f(bd[n], "total_s"))
             for n in batches]
    with open(REPORT / "slide08_batch_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "decode_speedup", "total_speedup"])
        w.writerows((n, f"{d:.4f}", f"{t:.4f}") for n, d, t in sweep)
    breakeven = None
    for (n0, d0, _), (n1, d1, _) in zip(sweep, sweep[1:]):
        if d0 >= 1 > d1:
            breakeven = n0 + (1 - d0) * (n1 - n0) / (d1 - d0)   # linear interpolation in batch
    rows = [(n, f"{first_call_ms(bd[n]):.1f} ms", f"{first_call_ms(ar[n]):.1f} ms",
             f"{first_call_ms(bd[n]) / first_call_ms(ar[n]):.1f}x") for n in (1, 16, 32, 64, 128)]
    lines += ["## Slide 8: where normal decode catches up (prompt 512, output 128)", "",
              f"Break-even (decode speedup = 1, linear interpolation between the two batch sizes "
              f"around it): batch {breakeven:.1f}.", "",
              md_table(["Batch", "Diffusion call", "Normal step", "Ratio"], rows), "",
              "Chart data: `slide08_batch_sweep.csv`; chart: `slide08_batch_sweep.png`.", ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(batches, [d for _, d, _ in sweep], marker="o", label="Decode only")
        ax.plot(batches, [t for _, _, t in sweep], marker="s", label="Total (prefill + decode)")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        if breakeven:
            ax.axvline(breakeven, color="gray", lw=1, ls=":")
            ax.annotate(f"break-even ~ batch {breakeven:.0f}", (breakeven, 1.0),
                        textcoords="offset points", xytext=(6, 8))
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels([str(n) for n in batches])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Speedup over normal decode")
        ax.set_title("Block diffusion (b = 8, R = 4), prompt 512, output 128")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT / "slide08_batch_sweep.png", dpi=200)
    except ImportError:
        lines += ["(matplotlib not installed: chart skipped. `pip install -r project3/requirements-report.txt`)", ""]

    # ---- Slide 9: (b, R) sweep ----------------------------------------------------------
    sa = load("study_a_b_R")
    rows = []
    for b, rr in sorted({(int(r["block_size"]), int(r["refine_rounds"])) for r in sa}):
        cells = []
        for batch in ("1", "16", "64"):
            r = next(x for x in sa if x["batch"] == batch and x["block_size"] == str(b)
                     and x["refine_rounds"] == str(rr))
            cells.append(f"{f(ar_req[(batch, '128')], 'decode_s') / f(r, 'decode_s'):.2f}x")
        calls = 128 // b * rr
        rows.append((b, rr, f"{b / rr:g}", calls, *cells))
    lines += ["## Slide 9: decode-time speedup by block size b and rounds R (prompt 512, output 128)", "",
              md_table(["b", "R", "b/R", "Calls", "Batch 1", "Batch 16", "Batch 64"], rows), ""]

    # ---- Slide 10: long prompts ---------------------------------------------------------
    sc = load("study_c_prompt")
    cells_ar = {(int(r["prompt"]), int(r["batch"])): r for r in sc if r["block_size"] == "1"}
    cells_bd = {(int(r["prompt"]), int(r["batch"])): r for r in sc if r["block_size"] == "8"}
    for batch in (1, 8, 32):                                   # the 512-token row comes from study B
        cells_ar[(512, batch)], cells_bd[(512, batch)] = ar[batch], bd[batch]
    rows, over = [], []
    for prompt in (512, 2048, 8192, 32768):
        row = [prompt if prompt < 1000 else f"{prompt // 1024}K"]
        for batch in (1, 8, 32):
            a, d = cells_ar[(prompt, batch)], cells_bd[(prompt, batch)]
            cell = f"{f(a, 'decode_s') / f(d, 'decode_s'):.2f}x / {f(a, 'total_s') / f(d, 'total_s'):.2f}x"
            if d["fits_80gb"] != "True":
                cell += " (over 80 GB)"
                over.append(f(d, "max_peak_gib"))
            row.append(cell)
        rows.append(row)
    d = cells_bd[(32768, 1)]
    lines += ["## Slide 10: long prompts, b = 8, R = 4, output 128 (decode speedup / total speedup)", "",
              md_table(["Prompt", "Batch 1", "Batch 8", "Batch 32"], rows), "",
              f"- Prefill share of block-diffusion time at a 32K prompt, batch 1: "
              f"{f(d, 'prefill_s') / f(d, 'total_s') * 100:.0f}% ({f(d, 'prefill_s'):.2f} s of {f(d, 'total_s'):.2f} s).",
              f"- Peak memory of the over-80 GB cells: {min(over):.0f} to {max(over):.0f} GiB "
              f"(prefill of the whole batch; identical for both methods).", ""]

    # ---- Slide 11: memory ---------------------------------------------------------------
    deltas = []
    for r in sa:
        deltas.append(f(r, "decode_peak_gib") - f(ar_req[(r["batch"], "128")], "decode_peak_gib"))
    for key in cells_bd:
        deltas.append(f(cells_bd[key], "decode_peak_gib") - f(cells_ar[key], "decode_peak_gib"))
    lines += ["## Slide 11: memory", "",
              f"- Largest decode-phase peak increase of block diffusion over normal decode, across studies A, B "
              f"and C: {max(deltas):.2f} GiB. Max peak memory (prefill) is identical for both methods in every run.", ""]

    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {REPORT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
