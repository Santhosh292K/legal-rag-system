"""
run_all_evaluations.py
Runs every script in evaluation/ in the order that makes their outputs
trustworthy, and prints a consolidated pass/fail summary at the end.

WHERE TO PUT THIS FILE: either directly in the rag project root (next to
main.py) or inside rag/evaluation/ — it locates the real project root
itself (see _find_rag_root below), so it works either way and regardless
of what directory you launch it from.

KNOWN GOTCHA THIS SCRIPT GUARDS AGAINST: there is a real, unrelated PyPI
package literally named "pipeline" (pypi.org/project/pipeline/). If it's
installed in your venv — even as a stray/transitive dependency you didn't
ask for — Python resolves `import pipeline` to THAT package instead of
this project's own pipeline/ directory, because evaluate.py's own
`sys.path.append(str(Path(__file__).parent.parent))` adds the project
root to the END of sys.path, after site-packages. The symptom is exactly
`ModuleNotFoundError: No module named 'pipeline.intent_classifier'` —
"pipeline" itself resolves (to the wrong package), but none of its
submodules exist there. This script (1) prepends the project root to
PYTHONPATH for every subprocess it launches, so the project's own
pipeline/ is found first regardless of what's in site-packages, and
(2) runs a one-line preflight check before the real steps that tells you
explicitly if `import pipeline` is still resolving to the wrong place, so
you're not left debugging a cryptic traceback. If the preflight warns,
the fix is: `pip show pipeline` in your venv to confirm it's the PyPI
package, then `pip uninstall pipeline`.

Order and why:
  1. validate_benchmark.py            — catch malformed/typo'd gold labels
                                         BEFORE anything is evaluated against
                                         them (a bad label silently corrupts
                                         every metric downstream).
  2. validate_benchmark.py --cases    — same check for benchmark_cases.json,
                                         if that file exists.
  3. split_benchmark.py               — creates the dev/test split so the
                                         final numbers in step 4 aren't
                                         reported on data anything was tuned
                                         against (leakage). Skipped if a
                                         split already exists, unless
                                         --resplit is passed.
  4. evaluate.py                      — main metrics run, against the held-
                                         out test split (evaluate.py's own
                                         default already prefers
                                         benchmark_scenarios_test.json).
  5. evaluate.py --ablation           — 7-variant ablation study.
  6. significance_test.py             — for each ablation variant vs the
                                         full pipeline, is the gap real or
                                         noise? Only runs if step 5 ran.
  7. evaluate_cases.py                — multi-document case-file scenarios,
                                         if benchmark_cases.json exists.

Every step runs as its own subprocess (same as running it by hand from the
CLI) so a crash or a missing dependency in one step can't silently corrupt
another step's in-process state. Each step's real stdout/stderr streams
live; nothing is swallowed or summarized away from you.

A step's failure does not stop the run by default — later steps that don't
depend on it still execute, and the final summary tells you exactly which
steps passed, which failed, and which were skipped and why. Pass
--stop-on-error to halt at the first failure instead.

Usage:
    python3 run_all_evaluations.py
    python3 run_all_evaluations.py --llm-judge
    python3 run_all_evaluations.py --max 10                 # fast smoke test
    python3 run_all_evaluations.py --skip-ablation --skip-cases
    python3 run_all_evaluations.py --resplit --seed 7
    python3 run_all_evaluations.py --stop-on-error
"""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _find_rag_root() -> Path:
    """Locates the rag project root by walking upward from this file's own
    (resolved, absolute) location, looking for a directory that has
    main.py, a pipeline/ package, and an evaluation/ folder."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parent.parents]:
        if (candidate / "main.py").is_file() \
           and (candidate / "pipeline").is_dir() \
           and (candidate / "evaluation").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find the rag project root by searching upward from "
        f"{here.parent} — expected a directory containing main.py, "
        f"pipeline/, and evaluation/. Place run_all_evaluations.py inside "
        f"the rag project (either at its root or inside its evaluation/ "
        f"folder)."
    )


RAG_ROOT = _find_rag_root()
EVAL_DIR = RAG_ROOT / "evaluation"

# Prepend the rag root to PYTHONPATH for every subprocess this script
# launches, so `import pipeline...` / `import main` resolve to THIS
# project's own packages first — ahead of any same-named package already
# installed in the venv's site-packages (see module docstring: this
# exact collision happens with the real PyPI package "pipeline").
_SUBPROCESS_ENV = dict(os.environ)
_SUBPROCESS_ENV["PYTHONPATH"] = os.pathsep.join(
    filter(None, [str(RAG_ROOT), _SUBPROCESS_ENV.get("PYTHONPATH", "")])
)

ABLATION_VARIANTS_VS_FULL = [
    "B: No IRAC Reranker",
    "C: No Hierarchy Chunking",
    "D: No Temporal Filter",
    "E: No Section Pinner",
    "F: Baseline Naive RAG",
    "G: No Knowledge Graph",
]
FULL_PIPELINE_LABEL = "A: Full Pipeline"
SIGNIFICANCE_METRICS = ["ndcg_at_10", "mrr", "rouge_l", "citation_f1"]


@dataclass
class StepResult:
    name: str
    status: str            # "PASS" | "FAIL" | "SKIPPED"
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class RunLog:
    steps: list = field(default_factory=list)

    def record(self, result: StepResult):
        self.steps.append(result)


def _print_header(title: str):
    print("\n" + "█" * 88)
    print(f"█  {title}")
    print("█" * 88)


def _preflight_check_pipeline_import(py: str) -> bool:
    """Runs `import pipeline` in a subprocess with the exact same cwd/env
    every real step below will use, and checks it resolves to THIS
    project's pipeline/ directory rather than some other installed
    package of the same name. Returns True if OK, False (with a specific,
    actionable message) if something is shadowing it."""
    result = subprocess.run(
        [py, "-c", "import pipeline; print(pipeline.__file__)"],
        cwd=str(RAG_ROOT), env=_SUBPROCESS_ENV,
        capture_output=True, text=True,
    )
    expected_prefix = str(RAG_ROOT / "pipeline")
    if result.returncode != 0:
        print("[run_all_evaluations] PREFLIGHT FAILED: `import pipeline` does not "
              "work at all in this environment:\n" + result.stderr.strip())
        return False
    found_at = result.stdout.strip()
    if not found_at.startswith(expected_prefix):
        print(
            "[run_all_evaluations] PREFLIGHT WARNING: `import pipeline` resolves "
            f"to:\n    {found_at}\n  instead of this project's own:\n"
            f"    {expected_prefix}\n"
            "  A different, unrelated package named 'pipeline' is installed in "
            "this Python environment and is shadowing the project's pipeline/ "
            "directory — this is why evaluate.py / evaluate_cases.py fail with "
            "\"No module named 'pipeline.intent_classifier'\" even though that "
            "file exists on disk.\n"
            "  Fix: run `pip show pipeline` in this venv to confirm it, then "
            "`pip uninstall pipeline`. Re-run this script after that.\n"
            "  Continuing anyway — steps 4/5/7 below (anything that imports "
            "main.py) will very likely still fail until this is fixed."
        )
        return False
    print(f"[run_all_evaluations] preflight OK — pipeline resolves to {found_at}")
    return True


def run_step(log: RunLog, name: str, cmd: list[str], stop_on_error: bool) -> bool:
    """Runs a subprocess, streaming its output live. cmd[1] is expected to
    be an absolute script path; cwd is always RAG_ROOT; PYTHONPATH is
    always primed with RAG_ROOT (see _SUBPROCESS_ENV). Returns True on
    exit code 0."""
    _print_header(name)
    print(f"$ (cwd={RAG_ROOT}) {' '.join(cmd)}\n")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(RAG_ROOT), env=_SUBPROCESS_ENV)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log.record(StepResult(name, "PASS", elapsed_s=elapsed))
            return True
        else:
            log.record(StepResult(
                name, "FAIL",
                detail=f"exit code {proc.returncode}",
                elapsed_s=elapsed,
            ))
            if stop_on_error:
                _abort(log, f"{name} failed (exit {proc.returncode}) and --stop-on-error was set")
            return False
    except FileNotFoundError as e:
        elapsed = time.time() - t0
        log.record(StepResult(name, "FAIL", detail=f"could not launch: {e}", elapsed_s=elapsed))
        if stop_on_error:
            _abort(log, f"{name} could not be launched: {e}")
        return False


def skip_step(log: RunLog, name: str, reason: str):
    print(f"\n[skip] {name} — {reason}")
    log.record(StepResult(name, "SKIPPED", detail=reason))


def _abort(log: RunLog, reason: str):
    print(f"\n[run_all_evaluations] Stopping early: {reason}")
    print_summary(log)
    sys.exit(1)


def print_summary(log: RunLog):
    _print_header("SUMMARY")
    name_w = max((len(s.name) for s in log.steps), default=20)
    print(f"{'Step':<{name_w}}  {'Status':<8} {'Time':>8}  Detail")
    print("-" * (name_w + 40))
    n_pass = n_fail = n_skip = 0
    for s in log.steps:
        if s.status == "PASS":
            n_pass += 1
        elif s.status == "FAIL":
            n_fail += 1
        else:
            n_skip += 1
        time_str = f"{s.elapsed_s:6.1f}s" if s.elapsed_s else "   -   "
        print(f"{s.name:<{name_w}}  {s.status:<8} {time_str}  {s.detail}")
    print("-" * (name_w + 40))
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"({len(log.steps)} total steps)")
    if n_fail:
        print("\nSome steps failed — scroll up to the step's own output above "
              "for the actual error; this script does not swallow or "
              "reinterpret any script's stderr.")


def main():
    parser = argparse.ArgumentParser(
        description="Run every evaluation/ script in order and print a consolidated summary."
    )
    parser.add_argument("--llm-judge", action="store_true",
                         help="Pass --llm-judge through to evaluate.py runs (main + ablation).")
    parser.add_argument("--max", type=int, default=None,
                         help="Pass --max N through to the main evaluate.py run (fast smoke test). "
                              "Does not limit the ablation study.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validate_benchmark.py steps.")
    parser.add_argument("--skip-split", action="store_true",
                         help="Skip split_benchmark.py even if no dev/test split exists yet.")
    parser.add_argument("--resplit", action="store_true",
                         help="Regenerate the dev/test split even if one already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to split_benchmark.py.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the 7-variant ablation study.")
    parser.add_argument("--skip-significance", action="store_true",
                         help="Skip paired bootstrap significance tests (only relevant if ablation runs).")
    parser.add_argument("--skip-cases", action="store_true", help="Skip evaluate_cases.py.")
    parser.add_argument("--skip-preflight", action="store_true",
                         help="Skip the `import pipeline` sanity check before running the real steps.")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="Halt immediately on the first failing step instead of continuing.")
    args = parser.parse_args()

    py = sys.executable
    log = RunLog()
    t_start = time.time()

    print(f"[run_all_evaluations] rag project root: {RAG_ROOT}")
    print(f"[run_all_evaluations] python: {py}")

    if not args.skip_preflight:
        _preflight_check_pipeline_import(py)

    scenarios_path = EVAL_DIR / "benchmark_scenarios.json"
    cases_path = EVAL_DIR / "benchmark_cases.json"
    dev_path = EVAL_DIR / "benchmark_scenarios_dev.json"
    test_path = EVAL_DIR / "benchmark_scenarios_test.json"

    def script(name: str) -> str:
        return str(EVAL_DIR / name)

    # ── 1. Validate scenarios ────────────────────────────────────────────────
    if args.skip_validate:
        skip_step(log, "validate_benchmark.py (scenarios)", "--skip-validate passed")
    elif not scenarios_path.exists():
        skip_step(log, "validate_benchmark.py (scenarios)", f"{scenarios_path} not found")
    else:
        ok = run_step(log, "validate_benchmark.py (scenarios)",
                       [py, script("validate_benchmark.py")], args.stop_on_error)
        if not ok:
            print("[run_all_evaluations] WARNING: benchmark validation failed — "
                  "downstream metrics may be computed against malformed/typo'd "
                  "gold labels. Continuing anyway (use --stop-on-error to halt here).")

    # ── 2. Validate cases ─────────────────────────────────────────────────────
    if args.skip_validate:
        skip_step(log, "validate_benchmark.py (cases)", "--skip-validate passed")
    elif not cases_path.exists():
        skip_step(log, "validate_benchmark.py (cases)", f"{cases_path} not found")
    else:
        run_step(log, "validate_benchmark.py (cases)",
                 [py, script("validate_benchmark.py"), str(cases_path), "--cases"],
                 args.stop_on_error)

    # ── 3. Split into dev/test ────────────────────────────────────────────────
    need_split = args.resplit or not (dev_path.exists() and test_path.exists())
    if args.skip_split:
        skip_step(log, "split_benchmark.py", "--skip-split passed")
    elif not scenarios_path.exists():
        skip_step(log, "split_benchmark.py", f"{scenarios_path} not found")
    elif not need_split:
        skip_step(log, "split_benchmark.py",
                   f"{dev_path.name}/{test_path.name} already exist (use --resplit to regenerate)")
    else:
        run_step(log, "split_benchmark.py",
                 [py, script("split_benchmark.py"), str(scenarios_path), "--seed", str(args.seed)],
                 args.stop_on_error)

    # ── 4. Main evaluation ────────────────────────────────────────────────────
    main_cmd = [py, script("evaluate.py"), "--save", str(RAG_ROOT / "results_full.json")]
    if args.llm_judge:
        main_cmd.append("--llm-judge")
    if args.max:
        main_cmd += ["--max", str(args.max)]
    run_step(log, "evaluate.py (main)", main_cmd, args.stop_on_error)

    # ── 5. Ablation study ─────────────────────────────────────────────────────
    ablation_ok = False
    if args.skip_ablation:
        skip_step(log, "evaluate.py --ablation", "--skip-ablation passed")
    else:
        ablation_cmd = [py, script("evaluate.py"), "--ablation",
                         "--save", str(RAG_ROOT / "results_ablation.json")]
        if args.llm_judge:
            ablation_cmd.append("--llm-judge")
        ablation_ok = run_step(log, "evaluate.py --ablation", ablation_cmd, args.stop_on_error)

    # ── 6. Significance tests: each variant vs full pipeline ─────────────────
    if args.skip_significance:
        skip_step(log, "significance_test.py (all variants)", "--skip-significance passed")
    elif args.skip_ablation or not ablation_ok:
        skip_step(log, "significance_test.py (all variants)",
                   "ablation study didn't run successfully")
    else:
        results_ablation = str(RAG_ROOT / "results_ablation.json")
        for variant in ABLATION_VARIANTS_VS_FULL:
            for metric in SIGNIFICANCE_METRICS:
                step_name = f"significance_test.py [{variant} vs Full | {metric}]"
                cmd = [
                    py, script("significance_test.py"),
                    results_ablation, results_ablation,
                    "--metric", metric,
                    "--system-a", FULL_PIPELINE_LABEL,
                    "--system-b", variant,
                ]
                run_step(log, step_name, cmd, args.stop_on_error)

    # ── 7. Case-file evaluation ───────────────────────────────────────────────
    if args.skip_cases:
        skip_step(log, "evaluate_cases.py", "--skip-cases passed")
    elif not cases_path.exists():
        skip_step(log, "evaluate_cases.py", f"{cases_path} not found")
    else:
        run_step(log, "evaluate_cases.py",
                 [py, script("evaluate_cases.py"), str(cases_path),
                  "--save", str(RAG_ROOT / "results_cases.json")],
                 args.stop_on_error)

    total = time.time() - t_start
    print_summary(log)
    print(f"\nTotal wall time: {total:.1f}s")
    print(f"Saved (if their step passed) under {RAG_ROOT}: "
          f"results_full.json, results_ablation.json, results_cases.json")

    n_fail = sum(1 for s in log.steps if s.status == "FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()