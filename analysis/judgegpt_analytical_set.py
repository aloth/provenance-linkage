#!/usr/bin/env python3
"""Canonical derivation of the JudgeGPT analytical set (triage finding T9).

Why this file exists
--------------------
T9 recorded that the analytical denominators moved by exactly two judgments
depending on how the set was derived: 2,317 vs 2,315 judgments, 46 vs 44 human
judgments. Two rows out of 2,317 would normally be noise. Here they are not:
the human cell holds only 46 judgments, so two rows are 4.3 percent of the
entire human evidence base, and the class-wise human rate moves by 2.3
percentage points depending on a filter-ordering decision nobody had written
down.

This script fixes that decision in code, prints the full exclusion path, and
fails loudly if the authoritative snapshot ever stops reproducing the numbers
the chapter reports.

The decision
------------
Two results rows carry a FragmentID that does not resolve in fragments.csv:

    ResultID d1d7180d...  FragmentID ec94e219...  Origin=Human  2024-05-30
    ResultID 15efe22d...  FragmentID 7fc9d94d...  Origin=Human  2024-08-30

They are kept. The ground truth for a judgment is read from the results row
itself, falling back to the fragment record only when the results row is blank.
That fallback is safe because the two sources never disagree: across all 2,155
analytical rows where both fields are populated, results.Origin agrees with
fragments.Origin on 2,155 and results.IsFake agrees with fragments.IsFake on
2,155. Disagreements: zero. The missing fragment record therefore costs the
stimulus text, which no inferential model in chapter 06 uses, and not the label,
which every model uses.

Dropping them instead would silently discard 4.3 percent of the human evidence
for a reason unrelated to data quality, which is the worse error.

Note on the fragment denominator: the 1,052 distinct FragmentIDs in the
analytical set include those two unresolvable IDs, so 1,050 of them have a
fragment record. The chapter states this explicitly.

Usage:  python3 analysis/judgegpt_analytical_set.py [--data <dir>]
Exit code 1 if any expected count fails to reproduce.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATA = "data/research-data-2026-03-11/judgegpt"

EXPECTED = {
    "snapshot_judgments": 2546,
    "snapshot_participants": 539,
    "analytical_judgments": 2317,
    "analytical_participants": 395,
    "analytical_fragments": 1052,
    "analytical_fragments_with_record": 1050,
    "human_judgments": 46,
    "machine_judgments": 2271,
    "orphan_fragment_rows": 2,
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth_bool(value) -> bool | None:
    token = str(value).strip().lower()
    if token in {"1", "true", "yes"}:
        return True
    if token in {"0", "false", "no"}:
        return False
    return None


def as_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def derive(data_dir: Path) -> dict:
    results = read_csv(data_dir / "results.csv")
    fragments = {row["FragmentID"]: row for row in read_csv(data_dir / "fragments.csv")}
    participants = {row["ParticipantID"]: row for row in read_csv(data_dir / "participants.csv")}

    steps = Counter()
    analytical: list[dict] = []

    for row in results:
        steps["snapshot"] += 1
        fragment = fragments.get(row.get("FragmentID", ""), {})

        if truth_bool(row.get("ReportedAsBroken", "")) is True:
            steps["dropped_reported_broken"] += 1
            continue

        hm = as_float(row.get("HumanMachineScore"))
        lf = as_float(row.get("LegitFakeScore"))
        if hm is None or lf is None:
            steps["dropped_incomplete_scores"] += 1
            continue

        # Ground truth: results row first, fragment record only as fallback.
        origin = (row.get("Origin") or fragment.get("Origin") or "").strip().lower()
        is_fake = truth_bool(row.get("IsFake") or fragment.get("IsFake") or "")

        if origin not in {"human", "machine"} or is_fake is None:
            steps["dropped_missing_ground_truth"] += 1
            continue

        analytical.append(
            {
                "rid": row.get("ResultID", ""),
                "pid": row.get("ParticipantID", ""),
                "fid": row.get("FragmentID", ""),
                "origin": origin,
                "hm_correct": int(hm >= 0.5) if origin == "machine" else int(hm < 0.5),
                "lf_correct": int(lf >= 0.5) if is_fake else int(lf < 0.5),
                "linked_participant": row.get("ParticipantID", "") in participants,
                "has_fragment_record": bool(fragment),
            }
        )

    fids = {r["fid"] for r in analytical}
    human = [r for r in analytical if r["origin"] == "human"]
    machine = [r for r in analytical if r["origin"] == "machine"]
    orphans = [r for r in analytical if not r["has_fragment_record"]]

    def rate(rows: list[dict]) -> float:
        return 100.0 * sum(r["hm_correct"] for r in rows) / len(rows) if rows else float("nan")

    return {
        "steps": steps,
        "analytical": analytical,
        "counts": {
            "snapshot_judgments": steps["snapshot"],
            "snapshot_participants": len(participants),
            "analytical_judgments": len(analytical),
            "analytical_participants": len({r["pid"] for r in analytical if r["linked_participant"]}),
            "analytical_fragments": len(fids),
            "analytical_fragments_with_record": len({r["fid"] for r in analytical if r["has_fragment_record"]}),
            "human_judgments": len(human),
            "machine_judgments": len(machine),
            "orphan_fragment_rows": len(orphans),
        },
        "rates": {
            "human_correct_pct": rate(human),
            "machine_correct_pct": rate(machine),
            "balanced_pct": (rate(human) + rate(machine)) / 2,
        },
        "orphans": orphans,
    }


def check_label_sources(data_dir: Path) -> tuple[int, int]:
    """Verify the fallback is safe: results vs fragments ground truth, where both exist."""
    results = read_csv(data_dir / "results.csv")
    fragments = {row["FragmentID"]: row for row in read_csv(data_dir / "fragments.csv")}
    origin_dis = fake_dis = 0
    for row in results:
        fragment = fragments.get(row.get("FragmentID", ""))
        if not fragment:
            continue
        a, b = (row.get("Origin") or "").strip().lower(), (fragment.get("Origin") or "").strip().lower()
        if a and b and a != b:
            origin_dis += 1
        a, b = (row.get("IsFake") or "").strip().lower(), (fragment.get("IsFake") or "").strip().lower()
        if a and b and a != b:
            fake_dis += 1
    return origin_dis, fake_dis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    report = derive(data_dir)
    steps, counts, rates = report["steps"], report["counts"], report["rates"]

    print("JUDGEGPT ANALYTICAL SET - canonical derivation (triage T9)")
    print(f"Snapshot: {data_dir}")
    print("\nExclusion path:")
    print(f"  snapshot judgment rows            : {steps['snapshot']:5d}")
    print(f"  - reported broken                 : {steps['dropped_reported_broken']:5d}")
    print(f"  - incomplete scores               : {steps['dropped_incomplete_scores']:5d}")
    print(f"  - missing ground truth            : {steps['dropped_missing_ground_truth']:5d}")
    print(f"  = analytical judgments            : {counts['analytical_judgments']:5d}")

    print("\nDenominators:")
    for key in ("analytical_participants", "analytical_fragments",
                "analytical_fragments_with_record", "human_judgments", "machine_judgments"):
        print(f"  {key:34s}: {counts[key]:5d}")

    print("\nClass-wise source accuracy:")
    print(f"  human   : {rates['human_correct_pct']:.1f}%")
    print(f"  machine : {rates['machine_correct_pct']:.1f}%")
    print(f"  balanced: {rates['balanced_pct']:.1f}%")

    origin_dis, fake_dis = check_label_sources(data_dir)
    print("\nGround-truth fallback safety (results row vs fragment record):")
    print(f"  Origin disagreements: {origin_dis}    IsFake disagreements: {fake_dis}")
    if origin_dis or fake_dis:
        print("  WARNING: the two sources disagree; the fallback is no longer safe.")

    print(f"\nKept rows without a fragment record: {counts['orphan_fragment_rows']}")
    for r in report["orphans"]:
        print(f"  ResultID {r['rid'][:12]}...  origin={r['origin']}  fid={r['fid'][:12]}...")
    print("  Dropping these would remove 2 of 46 human judgments (4.3% of the human evidence)")
    print("  and move the human rate from 52.2% to 54.5%.")

    failures = [(k, v, counts[k]) for k, v in EXPECTED.items() if k in counts and counts[k] != v]
    print()
    if failures:
        for key, expected, got in failures:
            print(f"MISMATCH {key}: expected {expected}, got {got}")
        print("FAIL: the analytical set no longer reproduces the reported denominators.")
        return 1
    print("OK: all reported denominators reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
