#!/usr/bin/env python3
"""Leave-one-group-out identifiability audit (triage finding LOGO).

The disposition matrix recorded the suspicion that `cross_group_baseline()`
silently skips folds and that `metric_bundle()` averages a null recall into
balanced accuracy, so that the "chance" LOGO results in chapter 08 are not
two-class estimates.

Checking this against the artifacts gives a split verdict.

REFUTED: the skipping is not silent. `cross_group_baseline()` records
`status: "not estimable"` together with the reason, and reports `covered` out
of `total`, so an unestimated fold is visible in the JSON.

CONFIRMED, and worse than described: for the generator, style and format
partitions, EVERY estimated fold contains zero human items. Balanced accuracy
is (recall_human + recall_machine)/2, and recall_human is 0/0, which
`metric_bundle()`'s `div()` returns as 0.0. Balanced accuracy is therefore
pinned near 0.5 by construction, no matter how the model performs. The pooled
generator and style figures read balanced accuracy 0.500 at accuracy 1.000:
a model that was right on every single row is reported as performing at chance.

The cause is structural and is the same confound documented for chapter 06 as
finding T10. In this corpus the human class carries a blank generator, a blank
style and format "unknown". The human class therefore occupies exactly one group
in each of those partitions, and that group is precisely the fold that cannot be
trained (holding it out leaves a single-class training set). Every remaining
fold is 100 percent machine.

Consequence: only the language partition supports a genuine two-class LOGO
estimate. Language is also the only one of the four fields where human items
appear in more than one group.

Usage:  python3 analysis/logo_identifiability.py
Exit code 1 if the artifact stops matching the reported structure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACT = "analysis/classifier_leakage.json"

# Partitions where the human class occupies exactly one group, which is also the
# fold that cannot be trained. Balanced accuracy is not identifiable there.
NOT_IDENTIFIABLE = {"generator_disjoint", "style_proxy_source_disjoint", "format_disjoint"}
IDENTIFIABLE = {"language_disjoint"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / ARTIFACT).read_text(encoding="utf-8"))
    regrouped = data["regrouped_baselines"]

    print("LEAVE-ONE-GROUP-OUT IDENTIFIABILITY AUDIT")
    print("=" * 72)

    problems: list[str] = []
    summary: dict[str, dict] = {}

    for model, partitions in regrouped.items():
        print(f"\n##### {model}")
        for name, result in partitions.items():
            folds = result.get("folds", [])
            ok = [f for f in folds if f.get("status") == "ok"]
            skipped = [f for f in folds if f.get("status") != "ok"]
            pooled = result.get("pooled") or {}

            human_in_folds = sum(f.get("human_n", 0) for f in ok)
            folds_with_human = sum(1 for f in ok if f.get("human_n", 0) > 0)

            verdict = "IDENTIFIABLE" if folds_with_human else "NOT IDENTIFIABLE"
            print(f"  {name}")
            print(f"    covered {result.get('covered')}/{result.get('total')}  "
                  f"folds ok={len(ok)} not_estimable={len(skipped)}")
            print(f"    human items across estimated folds: {human_in_folds} "
                  f"(folds containing any human item: {folds_with_human}/{len(ok)})")
            if pooled:
                print(f"    pooled balanced_accuracy={pooled['balanced_accuracy']:.4f}  "
                      f"accuracy={pooled['accuracy']:.4f}  "
                      f"recall_human={pooled['recall_human']:.3f}  "
                      f"recall_machine={pooled['recall_machine']:.3f}")
            for f in skipped:
                print(f"    not estimable: {f.get('group')!r} -> {f.get('reason')}")
            print(f"    -> {verdict}")

            if verdict == "NOT IDENTIFIABLE" and pooled:
                print(f"       balanced accuracy is (0/0 -> 0.0 + {pooled['recall_machine']:.3f})/2 "
                      f"= {pooled['balanced_accuracy']:.4f}; it measures the machine class only.")

            summary.setdefault(name, {})[model] = verdict
            if name in NOT_IDENTIFIABLE and folds_with_human:
                problems.append(f"{model}/{name}: expected no human items in estimated folds, found {human_in_folds}")
            if name in IDENTIFIABLE and not folds_with_human:
                problems.append(f"{model}/{name}: expected human items in estimated folds, found none")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("  language_disjoint            : genuine two-class LOGO estimate")
    print("  generator/style/format       : balanced accuracy not identifiable;")
    print("                                 every estimated fold is 100% machine")
    print("\n  Only the language partition may be cited as a leave-one-group-out")
    print("  balanced-accuracy result. The others quantify machine-class recall")
    print("  under group shift and must be reported as such.")

    if problems:
        print("\nSTRUCTURE CHANGED:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nOK: artifact matches the reported structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
