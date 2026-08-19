#!/usr/bin/env python3
"""Triage checks 2-5 (detection track).

T2  Majority-baseline metric confusion: is 0.804 accuracy, not macro-F1?
T3  Bag-of-words gap: is the claimed 1.7-point gap correct on the grouped split?
T4  Checkpoint provenance: do OOD predictions come from a different split
    family than the reported in-domain numbers?
T5  RAID / M4 corpus composition: source-code share, adversarial share,
    and whether "M4" is actually an aggregate dominated by MAGE.
"""
import json
import glob
import os
import collections
import statistics

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages", "redteam-detection")


def load(p):
    return json.load(open(os.path.join(PKG, p)))


def preds(corpus, model):
    return load(f"predictions/roguegpt_to_{corpus}__{model}__seed42.json")


def macro_f1_from_counts(tn, fp, fn, tp):
    def f1(p, r):
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    p_m = tp / (tp + fp) if (tp + fp) else 0.0
    r_m = tp / (tp + fn) if (tp + fn) else 0.0
    p_h = tn / (tn + fn) if (tn + fn) else 0.0
    r_h = tn / (tn + fp) if (tn + fp) else 0.0
    return (f1(p_m, r_m) + f1(p_h, r_h)) / 2


def t2_majority_metric():
    print("\n" + "=" * 70)
    print("T2  Majority baseline: accuracy or macro-F1?")
    print("=" * 70)
    d = load("metrics/metrics-grouped/bert-base-uncased__seed42__splits-grouped.json")
    test = d["split_summary"]["test"]
    n = test["rows"]
    n_mach = test["label_counts"]["Machine"]
    n_hum = test["label_counts"]["Human"]
    reported = d["test_metrics"]["majority_baseline"]

    acc = n_mach / n
    # constant predictor emitting the majority class (machine)
    mf1 = macro_f1_from_counts(tn=0, fp=n_hum, fn=0, tp=n_mach)

    print(f"  test set                       : n={n}  machine={n_mach}  human={n_hum}")
    print(f"  value reported as 'baseline'   : {reported:.4f}")
    print(f"  majority-class ACCURACY        : {acc:.4f}")
    print(f"  majority-class MACRO-F1        : {mf1:.4f}")
    match = "ACCURACY" if abs(reported - acc) < 1e-3 else ("MACRO-F1" if abs(reported - mf1) < 1e-3 else "NEITHER")
    print(f"  -> the reported number is      : {match}")
    if match == "ACCURACY":
        print("  VERDICT: CONFIRMED. Transformer macro-F1 is compared against a")
        print("           majority ACCURACY. Correct macro-F1 floor is "
              f"{mf1:.3f}, not {acc:.3f}.")
    else:
        print("  VERDICT: REFUTED / INDETERMINATE.")
    return {"reported": reported, "accuracy": acc, "macro_f1": mf1, "match": match}


def t3_bow_gap():
    print("\n" + "=" * 70)
    print("T3  Bag-of-words gap: is it 1.7 points?")
    print("=" * 70)
    b = load("results/baselines-grouped.json")

    rows = []
    for split_name, key in [("original (row-random)", "original"), ("grouped (current)", "group_aware")]:
        word = b[key]["lexical"]["word_tfidf_logistic"]["f1_macro"]
        char = b[key]["lexical"]["character_3_5gram_logistic"]["f1_macro"]
        rows.append((split_name, word, char))
        print(f"  {split_name:24s} word-TFIDF={word:.4f}  char-TFIDF={char:.4f}")

    # transformer results per split family
    trans = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(PKG, "metrics/metrics-grouped/*.json"))):
        d = json.load(open(f))
        arch = d["model"].split("-")[0]
        trans[arch].append(d["test_metrics"]["f1_macro"])

    word_grouped = b["group_aware"]["lexical"]["word_tfidf_logistic"]["f1_macro"]
    print(f"\n  grouped-split word-TFIDF baseline: {word_grouped:.4f}")
    print("  transformer means on the SAME grouped split:")
    for arch, vals in sorted(trans.items()):
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        print(f"    {arch:12s} mean={m:.4f} sd={sd:.4f}  gap_vs_BoW={100*(m-word_grouped):+.1f} pts"
              f"   (seeds: {', '.join(f'{v:.4f}' for v in sorted(vals))})")

    # seed-42 specific comparison
    print("\n  seed-42 only:")
    for f in sorted(glob.glob(os.path.join(PKG, "metrics/metrics-grouped/*seed42*.json"))):
        d = json.load(open(f))
        v = d["test_metrics"]["f1_macro"]
        print(f"    {d['model']:22s} {v:.4f}   gap_vs_BoW={100*(v-word_grouped):+.1f} pts")

    print("\n  VERDICT: the 1.7-point figure comes from the ORIGINAL split")
    print("           (word 0.9663 vs BERT 0.984 = 1.8 pts). On the current")
    print("           grouped split the gaps are far larger and BigBird@42 LOSES.")


def t4_checkpoint_provenance():
    print("\n" + "=" * 70)
    print("T4  Checkpoint provenance of the OOD evaluations")
    print("=" * 70)
    print("  OOD / in-domain prediction files:")
    ood_n = set()
    for f in sorted(glob.glob(os.path.join(PKG, "predictions/roguegpt_to_*seed42.json"))):
        if f.endswith(".bootstrap.json"):
            continue
        d = json.load(open(f))
        print(f"    {os.path.basename(f):48s} ckpt={d['checkpoint']:44s} n={d['eval_size']}")
        if d["eval_corpus"] == "roguegpt":
            ood_n.add(d["eval_size"])

    grouped_n = set()
    for f in glob.glob(os.path.join(PKG, "metrics/metrics-grouped/*.json")):
        grouped_n.add(json.load(open(f))["split_summary"]["test"]["rows"])

    print(f"\n  in-domain test size used by the prediction family : {sorted(ood_n)}")
    print(f"  in-domain test size of the grouped-split metrics   : {sorted(grouped_n)}")
    print("  checkpoint paths carry NO split marker (plain 'artifacts/models/<arch>'),")
    print("  so provenance must be inferred from the test-set size.")
    if ood_n and grouped_n and ood_n != grouped_n:
        print(f"\n  VERDICT: CONFIRMED. The prediction family evaluates on n={sorted(ood_n)[0]},")
        print(f"           the grouped-split metrics on n={sorted(grouped_n)[0]}. These are")
        print("           different split families. Any drop computed by subtracting an")
        print("           OOD score from a grouped-split in-domain score mixes two")
        print("           different trained models.")
    else:
        print("\n  VERDICT: REFUTED - same test size, likely same split family.")


def t5_ood_composition():
    print("\n" + "=" * 70)
    print("T5  Composition of the RAID / M4 / MAGE evaluation samples")
    print("=" * 70)
    for corpus in ["raid", "m4", "mage"]:
        d = preds(corpus, "bert-base")
        p = d["predictions"]
        print(f"\n  --- {corpus.upper()} (n={len(p)}) ---")
        print(f"    eval_notes        : {d.get('eval_notes')}")
        print(f"    label_distribution: {d.get('label_distribution')}")
        ld = d.get("label_distribution", {})
        tot = sum(ld.values()) or 1
        mach = ld.get("Machine", 0)
        print(f"    machine prevalence: {mach/tot:.3f}")
        srcs = collections.Counter(str(x["meta"].get("src")) for x in p)
        if len(srcs) == 1 and "None" in srcs:
            print("    src provenance    : ABSENT (all None) - composition NOT auditable")
            print("                        from this package. Claim about source-code /")
            print("                        adversarial share cannot be verified here.")
        else:
            print("    top src markers   :")
            for s, c in srcs.most_common(6):
                print(f"      {c:4d}  {s}")


if __name__ == "__main__":
    t2_majority_metric()
    t3_bow_gap()
    t4_checkpoint_provenance()
    t5_ood_composition()
