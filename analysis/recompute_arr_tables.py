#!/usr/bin/env python3
"""Recompute the ARR cross-corpus tables under the corrected MAGE key.

The ARR submission (paper/main.tex in the ai-disinfo-text-classifier repo)
reported confusion matrices and McNemar tests for three checkpoints on four
corpora. Triage finding T1 established that the MAGE evaluation was scored
against an inverted label key: the stored `label` field disagreed with MAGE's
own `src` provenance marker on 500 of 500 sampled rows and agreed on all 500
after inversion.

This script re-derives every affected cell from the archived per-item
predictions, so the tables carried into the journal manuscript reflect the
corrected key rather than the published one. RAID, M4 and the in-domain split
are unaffected by T1 and are re-derived here only to confirm they reproduce.

Output: JSON on stdout, plus LaTeX table bodies when --latex is passed.
"""
import argparse
import collections
import glob
import json
import os

ARR = "/Volumes/Data/OneDrive/Research/Collaborations/ai-disinfo-text-classifier"
RESULTS = os.path.join(ARR, "paper-anon-code", "artifacts", "results")

MODELS = ["bert-base", "bigbird", "longformer"]
MODEL_LABEL = {
    "bert-base": "BERT",
    "bigbird": "BigBird",
    "longformer": "Longformer",
}
CORPORA = ["corpus", "raid", "m4", "mage"]
CORPUS_LABEL = {
    "corpus": "In-domain",
    "raid": "RAID",
    "m4": "M4",
    "mage": "MAGE",
}


def src_authorship(src):
    """Derive authorship from MAGE's src provenance string.

    Identical convention to outreach/triage/t1_mage_labels.py: `_para` marks a
    machine paraphrase of a base text and is therefore machine-authored, so
    `<ds>_human_para` is MACHINE. A bare generator name is machine. Only
    `<ds>_human` with no further qualifier is genuinely human.

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


def load(corpus, model):
    path = os.path.join(RESULTS, f"corpus_to_{corpus}__{model}__seed42.json")
    with open(path) as fh:
        return json.load(fh)


def truths_and_preds(corpus, model):
    """Return (truth, pred) lists with the corrected key applied to MAGE."""
    d = load(corpus, model)
    truth, pred, unresolved = [], [], 0
    for p in d["predictions"]:
        if corpus == "mage":
            t = src_authorship(p["meta"].get("src"))
            if t is None:
                unresolved += 1
                continue
        else:
            t = p["label"]
        truth.append(t)
        pred.append(p["pred"])
    return truth, pred, unresolved


def confusion(truth, pred):
    """TN/FP/FN/TP with respect to the machine class (label 1)."""
    c = collections.Counter(zip(truth, pred))
    return {
        "tn": c[(0, 0)],
        "fp": c[(0, 1)],
        "fn": c[(1, 0)],
        "tp": c[(1, 1)],
    }


def rates(cm):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    return {
        "machine_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "machine_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "human_precision": tn / (tn + fn) if (tn + fn) else float("nan"),
        "human_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def mcnemar(truth, pred_a, pred_b):
    """McNemar with continuity correction. b = A wrong/B right, c = A right/B wrong."""
    from math import erfc, sqrt

    b = sum(1 for t, x, y in zip(truth, pred_a, pred_b) if x != t and y == t)
    c = sum(1 for t, x, y in zip(truth, pred_a, pred_b) if x == t and y != t)
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if abs(b - c) >= 1 else 0.0
    p = erfc(sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p": min(p, 1.0)}


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    if p < 1e-15:
        return r"$<10^{-15}$"
    exp = 0
    v = p
    while v < 1:
        v *= 10
        exp += 1
    return rf"${v:.2f}\!\times\!10^{{-{exp}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table bodies")
    args = ap.parse_args()

    out = {"confusion": {}, "mcnemar": {}, "mage_key": {}}

    # --- T1 re-verification on the MAGE sample -----------------------------
    for model in MODELS:
        d = load("mage", model)
        stored_agree = flipped_agree = resolvable = 0
        for p in d["predictions"]:
            t = src_authorship(p["meta"].get("src"))
            if t is None:
                continue
            resolvable += 1
            stored_agree += int(p["label"] == t)
            flipped_agree += int(p["label"] == 1 - t)
        out["mage_key"][model] = {
            "resolvable": resolvable,
            "agreement_stored": stored_agree / resolvable if resolvable else None,
            "agreement_flipped": flipped_agree / resolvable if resolvable else None,
        }

    # --- confusion matrices ------------------------------------------------
    for corpus in CORPORA:
        out["confusion"][corpus] = {}
        for model in MODELS:
            truth, pred, unresolved = truths_and_preds(corpus, model)
            cm = confusion(truth, pred)
            cm.update(rates(cm))
            cm["n"] = len(truth)
            cm["unresolved"] = unresolved
            cm["human_items"] = sum(1 for t in truth if t == 0)
            cm["machine_items"] = sum(1 for t in truth if t == 1)
            out["confusion"][corpus][model] = cm

    # --- McNemar -----------------------------------------------------------
    pairs = [("bert-base", "bigbird"), ("bert-base", "longformer"), ("bigbird", "longformer")]
    for corpus in CORPORA:
        out["mcnemar"][corpus] = {}
        preds = {}
        truth_ref = None
        for model in MODELS:
            truth, pred, _ = truths_and_preds(corpus, model)
            preds[model] = pred
            truth_ref = truth
        for a, b in pairs:
            r = mcnemar(truth_ref, preds[a], preds[b])
            r["p_bonf"] = min(r["p"] * len(pairs), 1.0)
            out["mcnemar"][corpus][f"{a}_vs_{b}"] = r

    print(json.dumps(out, indent=2))

    if args.latex:
        print("\n% ---- confusion table body ----")
        for corpus in CORPORA:
            print(rf"\multirow{{3}}{{*}}{{{CORPUS_LABEL[corpus]}}}", end="")
            for i, model in enumerate(MODELS):
                cm = out["confusion"][corpus][model]
                lead = "" if i == 0 else " " * 30
                print(
                    rf"{lead} & {MODEL_LABEL[model]} & {cm['tn']} & {cm['fp']} & "
                    rf"{cm['fn']} & {cm['tp']} & {cm['machine_precision']:.3f} & "
                    rf"{cm['machine_recall']:.3f} \\"
                )
            print(r"\midrule")

        print("\n% ---- mcnemar table body ----")
        for corpus in CORPORA:
            print(rf"\multirow{{3}}{{*}}{{{CORPUS_LABEL[corpus]}}}", end="")
            for i, (a, b) in enumerate(pairs):
                r = out["mcnemar"][corpus][f"{a}_vs_{b}"]
                lead = "" if i == 0 else " " * 30
                bonf = fmt_p(r["p_bonf"])
                if r["p_bonf"] < 0.05:
                    bonf = rf"\textbf{{{bonf}}}"
                print(
                    rf"{lead} & {MODEL_LABEL[a]} vs.\ {MODEL_LABEL[b]} & {r['b']} & "
                    rf"{r['c']} & {fmt_p(r['p'])} & {bonf} \\"
                )
            print(r"\midrule")


if __name__ == "__main__":
    main()
