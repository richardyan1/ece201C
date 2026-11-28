"""One-shot reproduction of the Project 3 results.

    python project3/reproduce.py            # everything (about 25 min; study C is most of it)
    python project3/reproduce.py --quick    # skips study_c_prompt (about 3 min)

Runs the suites in order (one at a time, they share output/LLM/), then calls make_report.py.
`baseline_after_change` is the normal-decode (autoregressive) reference for the comparisons.
"""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SUITES = ["baseline_after_change", "diffusion_b8_R4",
          "study_a_b_R", "study_b_batch", "study_c_prompt"]

if "--quick" in sys.argv:
    SUITES.remove("study_c_prompt")
for suite in SUITES:
    print(f"=== {suite} ===", flush=True)
    subprocess.run([sys.executable, str(PROJECT / "run_cases.py"), suite], check=True, cwd=PROJECT.parent)
if "--quick" in sys.argv:
    print("quick mode: study_c_prompt not re-run, report uses the committed results for it")
subprocess.run([sys.executable, str(PROJECT / "make_report.py")], check=True, cwd=PROJECT.parent)
print("\nOutputs: project3/results/<suite>/summary.csv, project3/report/REPORT.md, "
      "project3/report/slide08_batch_sweep.png")
