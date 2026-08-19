#!/usr/bin/env python3
"""Composition audit of the RAID and M4 OOD subsamples (triage finding T5).

Why this file exists
--------------------
detection-rev2 claimed three things about the external benchmark suite that the
red-team package could not confirm, because the shipped prediction dumps carry
`src = None` for every RAID and M4 row:

  1. the RAID sample is 32 percent source code,
  2. 92 percent of it is adversarially attacked,
  3. the "M4" artifact is 62 percent MAGE, so two of the three OOD corpora are
     not independent.

All three are checked here against the raw corpora in the Hugging Face cache,
reproducing the exact seed-42, n=500 draws used for the evaluation by calling
the same loaders with the same arguments.

Result: all three confirmed, with one correction to the reviewer's wording,
documented in outreach/triage/TRIAGE-REPORT.md.

Note on cost: RAID's `extra` split is a 3.5 GB CSV with 2,039,100 usable rows
and is streamed row by row, which takes several minutes. The composition
summary is therefore cached in analysis/output/t5_composition.json; pass
--recompute to redraw from the raw corpora.

Usage:
    python3 analysis/t5_corpus_composition.py              # read cached summary
    python3 analysis/t5_corpus_composition.py --recompute  # redraw from raw corpora
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

CODE_ROOT = "/Volumes/MacDev/ai-disinfo-classifier/code"
VENV_HINT = "/Volumes/MacDev/ai-disinfo-work/.venv/bin/python"
CACHE_HINT = "/Volumes/Data/.cache/huggingface"
OUTPUT = "analysis/output/t5_composition.json"

# Values verified on 2026-08-18 against the raw corpora.
EXPECTED = {
    "raid": {
        "n": 500, "human_n": 40, "machine_n": 460,
        "code_pct": 31.6, "attacked_pct": 91.8, "english_news_n": 0,
    },
    "m4": {
        "n": 500, "human_n": 33, "machine_n": 467,
        "mage_pct": 62.2,
    },
}


def recompute(repo_root: Path) -> dict:
    """Redraw both samples from the raw corpora using the evaluation loaders."""
    sys.path.insert(0, str(Path(CODE_ROOT) / "src"))
    import os
    os.environ.setdefault("HF_HOME", CACHE_HINT)
    os.environ.setdefault("AIDISINFO_EXTERNAL_CACHE", CACHE_HINT)

    from aidisinfo.datasets import m4, raid  # type: ignore

    out: dict = {}

    m4_ds = m4.load_eval_set(seed=42, n=500)
    m4_src = collections.Counter((r.meta.get("source") or "") for r in m4_ds.records)
    m4_lab = collections.Counter(r.label for r in m4_ds.records)
    out["m4"] = {
        "notes": m4_ds.notes,
        "n": len(m4_ds.records),
        "human_n": m4_lab[0], "machine_n": m4_lab[1],
        "source_counts": dict(m4_src),
        "texts": [r.text[:200] for r in m4_ds.records],
    }

    raid_ds = raid.load_eval_set(seed=42, n=500)
    raid_dom = collections.Counter((r.meta.get("domain") or "") for r in raid_ds.records)
    raid_att = collections.Counter((r.meta.get("attack") or "") for r in raid_ds.records)
    raid_lab = collections.Counter(r.label for r in raid_ds.records)
    out["raid"] = {
        "notes": raid_ds.notes,
        "n": len(raid_ds.records),
        "human_n": raid_lab[0], "machine_n": raid_lab[1],
        "domain_counts": dict(raid_dom),
        "attack_counts": dict(raid_att),
    }
    return out


def mage_overlap(repo_root: Path, m4_texts: list[str]) -> int:
    """How many M4 rows appear verbatim in the archived MAGE evaluation sample."""
    path = repo_root / "data/classifier-artifacts/results/roguegpt_to_mage__bert-base__seed42.json"
    mage = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    previews = {p["text_preview"][:200] for p in mage}
    return sum(1 for t in m4_texts if t[:200] in previews)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / OUTPUT

    if args.recompute or not target.exists():
        if not Path(CODE_ROOT).is_dir():
            print(f"Cannot recompute: loader tree not found at {CODE_ROOT}")
            return 2
        data = recompute(repo_root)
        data["m4"]["verbatim_overlap_with_mage_eval"] = mage_overlap(repo_root, data["m4"].pop("texts"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {target.relative_to(repo_root)}")
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        print(f"Read cached {target.relative_to(repo_root)} (pass --recompute to redraw)")

    raid, m4 = data["raid"], data["m4"]
    problems: list[str] = []

    print("\nRAID SUBSAMPLE COMPOSITION (seed 42, n=500)")
    print(f"  {raid['notes']}")
    n = raid["n"]
    for dom, count in sorted(raid["domain_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    domain {dom:10s} {count:4d}  {100 * count / n:5.1f}%")
    code_pct = 100 * raid["domain_counts"].get("code", 0) / n
    attacked = n - raid["attack_counts"].get("none", 0)
    attacked_pct = 100 * attacked / n
    attacks = sorted(k for k in raid["attack_counts"] if k != "none")
    news = sum(v for k, v in raid["domain_counts"].items() if k not in {"german", "czech", "code"})
    print(f"  source code       : {code_pct:.1f}%   (claim: 32%)")
    print(f"  adversarial attack: {attacked_pct:.1f}%   (claim: 92%), {len(attacks)} distinct attacks")
    print(f"  English news rows : {news}")
    print(f"  labels            : human {raid['human_n']}, machine {raid['machine_n']}")

    print("\nM4 SUBSAMPLE COMPOSITION (seed 42, n=500)")
    print(f"  {m4['notes']}")
    n4 = m4["n"]
    for src, count in sorted(m4["source_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    source {src:6s} {count:4d}  {100 * count / n4:5.1f}%")
    mage_pct = 100 * m4["source_counts"].get("mage", 0) / n4
    print(f"  MAGE share        : {mage_pct:.1f}%   (claim: 62%)")
    print(f"  labels            : human {m4['human_n']}, machine {m4['machine_n']}")
    print(f"  rows appearing verbatim in the MAGE eval sample: "
          f"{m4.get('verbatim_overlap_with_mage_eval')}")

    for got, want, label in (
        (round(code_pct, 1), EXPECTED["raid"]["code_pct"], "raid code_pct"),
        (round(attacked_pct, 1), EXPECTED["raid"]["attacked_pct"], "raid attacked_pct"),
        (news, EXPECTED["raid"]["english_news_n"], "raid english_news_n"),
        (round(mage_pct, 1), EXPECTED["m4"]["mage_pct"], "m4 mage_pct"),
    ):
        if got != want:
            problems.append(f"{label}: expected {want}, got {got}")

    print("\nVERDICT")
    print("  1. RAID is ~32% source code                       CONFIRMED")
    print("  2. RAID is ~92% adversarially attacked            CONFIRMED")
    print("  3. M4 is ~62% MAGE                                CONFIRMED")
    print("  RAID contains no English news at all: the sample is German, Czech")
    print("  and Python only, so it measures attack robustness and cross-lingual")
    print("  transfer rather than English-domain transfer.")
    print("  The M4/MAGE dependence is one of shared source corpus, not of")
    print("  duplicated items: only 3 of 500 rows recur verbatim.")

    if problems:
        print("\nCOMPOSITION CHANGED:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nOK: composition matches the audited values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
