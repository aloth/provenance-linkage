#!/usr/bin/env python3
"""Build the redistributable audit bundle from the dissertation artifacts.

The manuscript claims that the audit is reproducible from archived predictions
alone. This script produces the artifact that backs that claim.

Design decision: mirror the repository layout
---------------------------------------------
The audit scripts resolve their inputs relative to the repository root
(`data/classifier-artifacts/...`, `analysis/...`, `outreach/packages/...`).
The bundle therefore reproduces that layout rather than flattening it, so the
scripts run byte-identical to the versions that produced the published numbers.
Editing the scripts to accept a different layout would weaken the artifact:
a reader could no longer tell whether a discrepancy came from the analysis or
from the repackaging.

Redistribution boundary
-----------------------
The archived prediction dumps carry a `text_preview` field holding up to 200
characters of verbatim source text from RAID, M4, MAGE and the in-domain
corpus. Those corpora are third-party resources under their own licences, and
the in-domain corpus is released under restricted academic access. None of that
text may be redistributed here, so every such field is stripped.

Human-subject data is excluded wholesale. The red-team statistics package
contains participant-level records from the perception study; republishing them
is a research-ethics decision separate from corpus licensing, and this script
does not make it. Steps that depend on those files are reported as unavailable
rather than silently dropped.

Integrity
---------
Each emitted file is hashed (SHA-256), and the manifest records both the hash of
the emitted file and the hash of the source file it came from.

Usage:  python3 build_bundle.py [--out DIR] [--check]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[0]

DROP_FIELDS = ("text_preview", "text", "content")

# Scripts, copied into scripts/ AND left runnable from the mirrored root.
SCRIPTS = [
    "analysis/logo_identifiability.py",
    "analysis/auroc_audit.py",
    "analysis/t5_corpus_composition.py",
    "analysis/judgegpt_analytical_set.py",
    "outreach/triage/t1_mage_labels.py",
    "outreach/triage/t1_fix_mage_labels.py",
    "outreach/triage/t2_detection_checks.py",
    "outreach/triage/t3_statistics_checks.py",
]

# Directory trees copied verbatim except for prediction sanitizing.
DATA_GLOBS = [
    "data/classifier-artifacts/results/*.json",
    "data/classifier-artifacts/metrics/*.json",
    "analysis/classifier_leakage.json",
    "analysis/benchmark_audit.json",
    "outreach/packages/redteam-detection/predictions/*.json",
    "outreach/packages/redteam-detection/metrics/metrics/*.json",
    "outreach/packages/redteam-detection/metrics/metrics-grouped/*.json",
    "outreach/packages/redteam-detection/results/*.json",
]

# Deliberately NOT bundled. Kept as data so the manifest can state the reason.
EXCLUDED = {
    "outreach/packages/redteam-statistics/data/participants-deidentified.csv":
        "participant-level human-subject records; redistribution is an ethics "
        "decision outside the scope of this artifact",
    "outreach/packages/redteam-statistics/data/results.csv":
        "item-level human judgments from the perception study; same reason",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sanitize_predictions(obj) -> int:
    """Strip verbatim corpus text from a prediction dump, in place."""
    removed = 0
    preds = obj.get("predictions") if isinstance(obj, dict) else None
    if not isinstance(preds, list):
        return removed
    for rec in preds:
        if not isinstance(rec, dict):
            continue
        for field in DROP_FIELDS:
            if field in rec:
                del rec[field]
                removed += 1
        meta = rec.get("meta")
        if isinstance(meta, dict):
            for field in DROP_FIELDS:
                if field in meta:
                    del meta[field]
                    removed += 1
    return removed


def audit_residual_text(obj) -> list[str]:
    """Flag any remaining string over 80 chars inside prediction records."""
    flagged = []
    if not isinstance(obj, dict):
        return flagged
    for rec in obj.get("predictions", []) or []:
        if not isinstance(rec, dict):
            continue
        stack = [("", rec)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    stack.append((f"{prefix}.{k}" if prefix else k, v))
            elif isinstance(node, str) and len(node) > 80:
                flagged.append(prefix)
    return sorted(set(flagged))


def emit(src: Path, dst: Path, manifest_list: list, total: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = str(src.relative_to(REPO))
    source_hash = sha256_file(src)
    removed = 0

    if src.suffix == ".json":
        try:
            obj = json.loads(src.read_text())
        except json.JSONDecodeError:
            shutil.copy2(src, dst)
            manifest_list.append(
                {"path": rel, "source_sha256": source_hash, "sha256": sha256_file(dst)}
            )
            return
        removed = sanitize_predictions(obj)
        residual = audit_residual_text(obj)
        if residual:
            raise SystemExit(
                f"ABORT: {rel} still holds long strings at {residual}. "
                "Extend DROP_FIELDS before redistributing."
            )
        payload = json.dumps(obj, indent=1, sort_keys=True).encode()
        dst.write_bytes(payload)
        digest = sha256_bytes(payload)
    else:
        shutil.copy2(src, dst)
        digest = sha256_file(dst)

    total[0] += removed
    manifest_list.append(
        {
            "path": rel,
            "source_sha256": source_hash,
            "sha256": digest,
            "text_fields_removed": removed,
        }
    )


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle": "provenance-linkage-and-a-reproducible-benchmark-audit-sequence",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "layout": (
            "Mirrors the source repository layout so the audit scripts run "
            "unmodified. Run them from the bundle root."
        ),
        "redistribution_note": (
            "Prediction dumps are stripped of all verbatim corpus text. Only labels, "
            "predicted classes, logits and provenance markers are redistributed. "
            "Third-party corpora (RAID, M4, MAGE) and the restricted-access in-domain "
            "corpus are not included and must be obtained from their original sources."
        ),
        "fields_removed": list(DROP_FIELDS),
        "excluded": [{"path": k, "reason": v} for k, v in sorted(EXCLUDED.items())],
        "scripts": [],
        "data": [],
    }
    total = [0]

    for rel in SCRIPTS:
        src = REPO / rel
        if not src.is_file():
            print(f"  WARNING: missing script {rel}", file=sys.stderr)
            continue
        emit(src, out_dir / rel, manifest["scripts"], total)
        # convenience copy so readers find them in one place
        flat = out_dir / "scripts" / src.name
        flat.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, flat)

    local = PAPER / "analysis-links" / "recompute_arr_tables.py"
    if local.is_file():
        dst = out_dir / "analysis" / local.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)
        shutil.copy2(local, out_dir / "scripts" / local.name)
        manifest["scripts"].append(
            {"path": f"analysis/{local.name}", "sha256": sha256_file(dst)}
        )

    for pattern in DATA_GLOBS:
        for path in sorted(glob.glob(str(REPO / pattern))):
            src = Path(path)
            emit(src, out_dir / src.relative_to(REPO), manifest["data"], total)

    manifest["summary"] = {
        "scripts": len(manifest["scripts"]),
        "data_files": len(manifest["data"]),
        "text_fields_removed": total[0],
        "excluded_files": len(EXCLUDED),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(out_dir: Path) -> int:
    mpath = out_dir / "MANIFEST.json"
    if not mpath.is_file():
        print("no MANIFEST.json found", file=sys.stderr)
        return 2
    manifest = json.loads(mpath.read_text())
    bad = 0
    for section in ("scripts", "data"):
        for entry in manifest.get(section, []):
            path = out_dir / entry["path"]
            if not path.is_file():
                print(f"MISSING  {entry['path']}")
                bad += 1
            elif sha256_file(path) != entry["sha256"]:
                print(f"MISMATCH {entry['path']}")
                bad += 1

    for path in out_dir.rglob("*.json"):
        if path.name == "MANIFEST.json":
            continue
        try:
            residual = audit_residual_text(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if residual:
            print(f"TEXT LEAK {path.relative_to(out_dir)}: {residual}")
            bad += 1

    for rel in manifest.get("excluded", []):
        if (out_dir / rel["path"]).exists():
            print(f"SHOULD NOT BE PRESENT {rel['path']}")
            bad += 1

    print("OK" if bad == 0 else f"{bad} problem(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PAPER / "bundle"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    if args.check:
        raise SystemExit(check(out))

    manifest = build(out)
    print(json.dumps(manifest["summary"], indent=2))
    print(f"\nbundle written to {out}")


if __name__ == "__main__":
    main()
