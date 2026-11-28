import copy
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
PYTHON = sys.executable
HW_CONFIG = "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
BASE_MODEL = REPO / "configs/model-config/course_project/Llama3.1-8B_inf.yaml"

PROMPT = 512
BATCHES = [1, 16, 64]
OUTPUTS = [128, 256]


def baseline_cases():
    cases = []
    for batch in BATCHES:
        for out in OUTPUTS:
            cases.append({
                "name": f"ar_B{batch}_P{PROMPT}_O{out}",
                "batch": batch,
                "prompt": PROMPT,
                "output": out,
                "overrides": {
                    "model_param": {
                        "global_batch_size": batch,
                        "seq_len": PROMPT + out,
                        "decode_len": out,
                    }
                },
            })
    return cases


def with_project_param(cases, prefix, block_size, refine_rounds):
    out = []
    for case in cases:
        case = copy.deepcopy(case)
        case["name"] = case["name"].replace("ar_", prefix, 1)
        case["overrides"]["project_param"] = {"block_size": block_size, "refine_rounds": refine_rounds}
        case["extra_cols"] = {"block_size": block_size, "refine_rounds": refine_rounds}
        out.append(case)
    return out


def make_case(batch, prompt, out, block_size=None, refine_rounds=None):
    model_param = {"global_batch_size": batch, "seq_len": prompt + out, "decode_len": out}
    overrides = {"model_param": model_param}
    if block_size is None:
        name = f"ar_B{batch}_P{prompt}_O{out}"
        extra = {"block_size": 1, "refine_rounds": 1}
    else:
        name = f"bd_b{block_size}_R{refine_rounds}_B{batch}_P{prompt}_O{out}"
        overrides["project_param"] = {"block_size": block_size, "refine_rounds": refine_rounds}
        extra = {"block_size": block_size, "refine_rounds": refine_rounds}
    return {"name": name, "batch": batch, "prompt": prompt, "output": out,
            "overrides": overrides, "extra_cols": extra}


def study_a_cases():
    cases = []
    for batch in BATCHES:
        for b in (4, 8, 16, 32):
            for r in (2, 4, 8):
                if r < b:
                    cases.append(make_case(batch, PROMPT, 128, b, r))
    return cases


STUDY_B_BATCHES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 49, 50, 64, 96, 128]


def study_b_cases():
    cases = []
    for batch in STUDY_B_BATCHES:
        cases.append(make_case(batch, PROMPT, 128))
        cases.append(make_case(batch, PROMPT, 128, 8, 4))
    return cases


def study_c_cases():
    cases = []
    for prompt in (2048, 8192, 32768):
        for batch in (1, 8, 32):
            cases.append(make_case(batch, prompt, 128))
            cases.append(make_case(batch, prompt, 128, 8, 4))
    return cases


SUITES = {
    "baseline": baseline_cases,
    "baseline_after_change": baseline_cases,
    "neutral_b1_R1": lambda: with_project_param(baseline_cases(), "bd_b1_R1_", 1, 1),
    "diffusion_b8_R4": lambda: with_project_param(baseline_cases(), "bd_b8_R4_", 8, 4),
    "study_a_b_R": study_a_cases,
    "study_b_batch": study_b_cases,
    "study_c_prompt": study_c_cases,
}


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def parse_metrics(stdout, results_txt, memory_txt):
    def grab(pattern, text, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    row = {
        "prefill_s": grab(r"\[prefill\] time: ([0-9.eE+-]+)s", stdout),
        "decode_s": grab(r"\[decode\] time: ([0-9.eE+-]+)s", stdout),
        "total_s": grab(r"\[total\] time: ([0-9.eE+-]+)s", stdout),
        "ttft_s": grab(r"Time to First Token: ([0-9.eE+-]+)s", results_txt),
        "tok_s_start": grab(r"Aggregate Decode Throughput Tok/s[^:]*: start=([0-9.]+)", results_txt),
        "tok_s_end": grab(r"Aggregate Decode Throughput Tok/s[^:]*:.*end=([0-9.]+)", results_txt),
        "prefill_peak_gib": grab(r"Prefill peak memory usage \(per gpu\): ([0-9.]+) GiB", memory_txt),
        "decode_peak_gib": grab(r"Final decode peak memory usage \(per gpu\): ([0-9.]+) GiB", memory_txt),
        "max_peak_gib": grab(r"Max peak memory usage \(per gpu\): ([0-9.]+) GiB", memory_txt),
        "fits_80gb": "[WARN] Peak memory exceeds capacity" not in memory_txt,
        "model_calls": grab(r"model calls=(\d+)", stdout, int),
    }
    steps = re.findall(r"\[decode\] sample step (\d+): seq_len=(\d+), time=([0-9.eE+-]+)s", stdout)
    row["sampled_steps"] = ";".join(f"{s}@{L}={t}" for s, L, t in steps)
    return row


def run_case(case, suite_dir, base_cfg):
    case_dir = suite_dir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = deep_update(copy.deepcopy(base_cfg), case["overrides"])
    cfg_path = case_dir / "model_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    out_dir = REPO / "output" / "LLM"
    for stale in (out_dir / "LLM_inference_results.txt",
                  out_dir / "memory-summary" / "memory_capacity_comparison.txt"):
        if stale.exists():
            stale.unlink()

    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run(
        [str(PYTHON), "run_perf.py", "--hardware_config", HW_CONFIG, "--model_config", str(cfg_path)],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    (case_dir / "stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}); see {case_dir / 'stdout.txt'}")
        return None

    results_txt = memory_txt = ""
    src = out_dir / "LLM_inference_results.txt"
    if src.exists():
        shutil.copy(src, case_dir / src.name)
        results_txt = src.read_text(encoding="utf-8")
    src = out_dir / "memory-summary" / "memory_capacity_comparison.txt"
    if src.exists():
        shutil.copy(src, case_dir / src.name)
        memory_txt = src.read_text(encoding="utf-8")

    row = {k: case[k] for k in ("name", "batch", "prompt", "output")}
    row.update(case.get("extra_cols", {}))
    row.update(parse_metrics(proc.stdout, results_txt, memory_txt))
    return row


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    cases = SUITES[suite]()
    with open(BASE_MODEL) as f:
        base_cfg = yaml.safe_load(f)
    suite_dir = PROJECT / "results" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        print(f"running {case['name']} ...", flush=True)
        row = run_case(case, suite_dir, base_cfg)
        if row:
            rows.append(row)
            print(f"  total={row['total_s']}s decode={row['decode_s']}s peak={row['max_peak_gib']}GiB")

    if rows:
        fields = list(rows[0].keys())
        with open(suite_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {suite_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
