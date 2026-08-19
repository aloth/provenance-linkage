#!/usr/bin/env python3
"""Run the audit sequence end to end against the bundled artifacts.

Steps execute in the order the manuscript presents them, because each step
fixes an assumption the next one depends on. Scripts run from the bundle root
and are byte-identical to the versions that produced the published numbers;
they resolve their inputs relative to that root.

Steps whose inputs are deliberately not redistributed (third-party corpora,
human-subject records) report UNAVAILABLE rather than passing silently, so a
reader can distinguish "this check passed" from "this check could not run here".

Usage:  python3 run_audit.py [--verbose]

Exit code 0 when every runnable step completed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (path relative to bundle root, manuscript step, reason it may be unavailable)
STEPS = [
    # Expected verdict REFUTED: the archived dumps in this bundle carry the
    # CORRECTED key (t1_fix_mage_labels.py rewrote them in place on 2026-08-18),
    # so the stored label now agrees with src provenance. The 0.000/1.000
    # disagreement reported in the manuscript is reproducible only against the
    # pre-correction snapshot held in the conference-submission repository.
    ("outreach/triage/t1_mage_labels.py",
     "Step 1: label orientation (expects REFUTED on corrected dumps)",
     "dumps already carry the corrected key; see README section 'A note on Step 1'"),
    ("analysis/recompute_arr_tables.py",
     "Step 1 / Step 7: divergence and confusion tables", None),
    ("outreach/triage/t1_fix_mage_labels.py",
     "Step 1: corrected-key derivation", None),
    ("outreach/triage/t2_detection_checks.py",
     "Steps 2-3: split integrity and cheap baselines", None),
    ("analysis/logo_identifiability.py",
     "Step 4: leave-one-group-out identifiability", None),
    ("outreach/triage/t3_statistics_checks.py",
     "Step 4: analytical-set and statistics checks",
     "requires participant-level records, excluded on ethics grounds"),
    ("analysis/t5_corpus_composition.py",
     "Step 5: realized benchmark composition",
     "requires authorized local copies of RAID / M4 / MAGE"),
    ("analysis/auroc_audit.py",
     "Step 7: threshold-free discrimination", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="echo each script's stdout")
    args = ap.parse_args()

    results = []
    for rel, label, unavailable_reason in STEPS:
        path = ROOT / rel
        if not path.is_file():
            results.append(("MISSING", rel, label, ""))
            continue

        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if args.verbose and proc.stdout:
            print(f"----- {rel}\n{proc.stdout}")

        if proc.returncode == 0:
            status, detail = "OK", ""
        elif unavailable_reason:
            status, detail = "UNAVAILABLE", unavailable_reason
        else:
            status = "FAILED"
            detail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            detail = detail[0][:80]
        results.append((status, rel, label, detail))

    name_w = max(len(r[1]) for r in results)
    print("\n=== audit sequence ===")
    failed = 0
    for status, rel, label, detail in results:
        print(f"  {status:<12} {rel:<{name_w}}  {label}")
        if detail:
            print(f"               {detail}")
        if status in ("FAILED", "MISSING"):
            failed += 1

    ok = sum(1 for r in results if r[0] == "OK")
    unavail = sum(1 for r in results if r[0] == "UNAVAILABLE")
    print(f"\n{ok} completed, {unavail} unavailable by design, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
