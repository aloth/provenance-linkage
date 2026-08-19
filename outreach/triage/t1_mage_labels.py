#!/usr/bin/env python3
"""Triage check 1: is the MAGE evaluation scored against an inverted key?

Claim under test (detection-rev2 finding 1, detection-rev3 finding 5):
  the stored `label` field disagrees with MAGE's own `src` provenance
  marker on 500 of 500 items, and agrees perfectly when flipped.

Verdict is printed as CONFIRMED / REFUTED / INDETERMINATE.
"""
import json
import glob
import collections
import os
import sys

PKG = os.path.join(os.path.dirname(__file__), "..", "packages", "redteam-detection")


def src_authorship(src):
    """Derive authorship from MAGE's src provenance string.

    MAGE src markers look like `wp_human`, `hswag_machine_topical_text-davinci-002`,
    `cnn_human_para`, `imdb_gpt4`, `pubmed_gpt4_para`.

    Convention: the `_para` suffix marks a machine paraphrase of the base text,
    so `<ds>_human_para` is MACHINE-authored, not human. A bare generator name
    (`gpt4`, `text-davinci-002`, ...) is machine. Only `<ds>_human` with no
    further qualifier is genuinely human.

    Returns 0 for human, 1 for machine, None if unresolvable.
    """
    if not src:
        return None
    s = str(src).lower()
    if s.endswith("_para"):
        return 1
    if "_machine" in s:
        return 1
    if s.endswith("_human"):
        return 0
    if "_human" not in s:
        return 1
    return None


def main():
    files = sorted(glob.glob(os.path.join(PKG, "predictions", "roguegpt_to_mage__*seed42.json")))
    files = [f for f in files if not f.endswith(".bootstrap.json")]
    if not files:
        print("INDETERMINATE: no MAGE prediction files found")
        return 2

    overall = []
    for path in files:
        d = json.load(open(path))
        preds = d["predictions"]
        resolvable = 0
        agree = 0
        agree_flipped = 0
        xtab = collections.Counter()
        for p in preds:
            truth = src_authorship(p["meta"].get("src"))
            if truth is None:
                continue
            resolvable += 1
            stored = p["label"]
            xtab[(("human", "machine")[truth], stored)] += 1
            if stored == truth:
                agree += 1
            if stored == (1 - truth):
                agree_flipped += 1

        tag = d.get("model_tag", os.path.basename(path))
        print(f"\n=== {tag}  ({os.path.basename(path)}) ===")
        print(f"  items total                : {len(preds)}")
        print(f"  resolvable via src         : {resolvable}")
        if resolvable:
            print(f"  agreement stored vs src    : {agree/resolvable:.4f}  ({agree}/{resolvable})")
            print(f"  agreement if FLIPPED       : {agree_flipped/resolvable:.4f}  ({agree_flipped}/{resolvable})")
        print("  cross-tabulation (src marker -> stored label):")
        for (marker, stored), n in sorted(xtab.items()):
            print(f"    src={marker:<8s} stored_label={stored}  n={n}")
        print(f"  declared label_distribution: {d.get('label_distribution')}")
        print(f"  reported metrics           : f1_macro={d['metrics'].get('f1_macro'):.4f} "
              f"acc={d['metrics'].get('accuracy'):.4f}")
        overall.append((resolvable, agree, agree_flipped))

    tot_res = sum(x[0] for x in overall)
    tot_agree = sum(x[1] for x in overall)
    tot_flip = sum(x[2] for x in overall)
    print("\n" + "=" * 60)
    print(f"POOLED over {len(files)} files: resolvable={tot_res} "
          f"agree={tot_agree/tot_res:.4f} flipped={tot_flip/tot_res:.4f}")

    if tot_res == 0:
        print("VERDICT: INDETERMINATE (no src markers resolvable)")
        return 2
    if tot_flip / tot_res > 0.99 and tot_agree / tot_res < 0.01:
        print("VERDICT: CONFIRMED - MAGE labels are inverted.")
        return 0
    if tot_agree / tot_res > 0.99:
        print("VERDICT: REFUTED - stored labels match src provenance.")
        return 1
    print("VERDICT: INDETERMINATE - partial disagreement, inspect manually.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
