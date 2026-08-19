#!/usr/bin/env python3
"""T1 remediation: rewrite MAGE artifacts against the corrected label key.

Finding T1 (detection-rev2 #1, detection-rev3 #5, CONFIRMED by
outreach/triage/t1_mage_labels.py): the MAGE adapter stored MAGE's native
polarity (0=machine, 1=human) instead of the project convention
(0=human, 1=machine). Agreement with MAGE's own `src` provenance marker was
0.0000 stored and 1.0000 flipped, on 500/500 items in all three model files.

The manuscript (chapters/08-machine-perception.tex) already reports the
corrected values. This script brings the on-disk artifacts into agreement so
that package and text no longer diverge.

No model is retrained. Only the ground-truth key is corrected; every
`pred` and `logits` value is left untouched. All derived quantities
(metrics, label_distribution, bootstrap CIs, confusion matrices, McNemar
tables) are recomputed from the corrected key.

Usage:  python3 t1_fix_mage_labels.py [--apply]   (default: dry run)
"""
import json, os, sys, random, argparse, shutil, datetime

MODELS = ["bert-base", "bigbird", "longformer"]
ROOTS = [
    "data/classifier-artifacts/results",
    "outreach/packages/redteam-detection/predictions",
]

def prf(y, yh):
    out = {}
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = (p, r, f)
    return out

def metrics_block(y, yh):
    s = prf(y, yh)
    n = len(y)
    acc = sum(1 for a, b in zip(y, yh) if a == b) / n
    sup = [y.count(0), y.count(1)]
    return {
        "accuracy": acc,
        "precision_macro": (s[0][0] + s[1][0]) / 2,
        "recall_macro": (s[0][1] + s[1][1]) / 2,
        "f1_macro": (s[0][2] + s[1][2]) / 2,
        "precision_machine": s[1][0],
        "recall_machine": s[1][1],
        "f1_machine": s[1][2],
        "precision_human": s[0][0],
        "recall_human": s[0][1],
        "f1_human": s[0][2],
        "f1_weighted": (s[0][2] * sup[0] + s[1][2] * sup[1]) / n,
    }

def bootstrap(y, yh, keys, resamples=10000, seed=42):
    rng = random.Random(seed)
    n = len(y)
    draws = {k: [] for k in keys}
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        hh = [yh[i] for i in idx]
        m = metrics_block(yy, hh)
        for k in keys:
            draws[k].append(m[k])
    point = metrics_block(y, yh)
    out = {}
    for k in keys:
        v = sorted(draws[k])
        mean = sum(v) / len(v)
        var = sum((x - mean) ** 2 for x in v) / (len(v) - 1)
        out[k] = {
            "point": point[k],
            "mean": mean,
            "std": var ** 0.5,
            "ci95_lo": v[int(0.025 * len(v))],
            "ci95_hi": v[int(0.975 * len(v)) - 1],
        }
    return out

def chi2_sf(x):
    # survival function of chi-square with 1 df = erfc(sqrt(x/2))
    import math
    return math.erfc(math.sqrt(x / 2.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for root in ROOTS:
        d = os.path.join(repo, root)
        if not os.path.isdir(d):
            print(f"skip (absent): {root}")
            continue
        print(f"\n########## {root} ##########")
        per_model = {}
        for m in MODELS:
            path = os.path.join(d, f"roguegpt_to_mage__{m}__seed42.json")
            doc = json.load(open(path, encoding="utf-8"))
            P = doc["predictions"]
            if doc.get("label_key_corrected"):
                print(f"  {m}: already corrected, skipping")
                per_model[m] = None
                continue
            for p in P:
                p["label"] = 1 - p["label"]
            y = [p["label"] for p in P]
            yh = [p["pred"] for p in P]
            doc["metrics"] = metrics_block(y, yh)
            doc["label_distribution"] = {"Human": y.count(0), "Machine": y.count(1)}
            doc["label_key_corrected"] = True
            doc["label_key_note"] = (
                "MAGE encodes 0=machine, 1=human; the project convention is "
                "0=human, 1=machine. The stored key was corrected on "
                f"{stamp} (triage finding T1). Model outputs (pred, logits) "
                "are unchanged; all derived metrics were recomputed."
            )
            per_model[m] = (path, doc, y, yh)
            mm = doc["metrics"]
            print(f"  {m:11s} acc={mm['accuracy']:.4f} macroF1={mm['f1_macro']:.4f} "
                  f"P_mach={mm['precision_machine']:.3f} R_mach={mm['recall_machine']:.3f}")

        live = {k: v for k, v in per_model.items() if v}
        if not live:
            continue

        # confusion matrices
        cm = {}
        for m, (path, doc, y, yh) in live.items():
            tn = sum(1 for a, b in zip(y, yh) if a == 0 and b == 0)
            fp = sum(1 for a, b in zip(y, yh) if a == 0 and b == 1)
            fn = sum(1 for a, b in zip(y, yh) if a == 1 and b == 0)
            tp = sum(1 for a, b in zip(y, yh) if a == 1 and b == 1)
            cm[m] = {
                "n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "human_recall": tn / (tn + fp) if tn + fp else 0.0,
                "machine_recall": tp / (tp + fn) if tp + fn else 0.0,
                "human_precision": tn / (tn + fn) if tn + fn else 0.0,
                "machine_precision": tp / (tp + fp) if tp + fp else 0.0,
                "label_distribution": doc["label_distribution"],
                "source_file": os.path.basename(path),
            }
            print(f"  {m:11s} tn={tn} fp={fp} fn={fn} tp={tp}")

        # mcnemar
        pairs = []
        names = [m for m in MODELS if m in live]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ya, ha = live[a][2], live[a][3]
                hb = live[b][3]
                ca = [x == p for x, p in zip(ya, ha)]
                cb = [x == p for x, p in zip(ya, hb)]
                n01 = sum(1 for x, z in zip(ca, cb) if not x and z)
                n10 = sum(1 for x, z in zip(ca, cb) if x and not z)
                chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
                pv = chi2_sf(chi2)
                pairs.append({
                    "model_a": a, "model_b": b,
                    "acc_a": sum(ca) / len(ca), "acc_b": sum(cb) / len(cb),
                    "b_a_wrong_b_right": n01, "c_a_right_b_wrong": n10,
                    "chi2": chi2, "p_value": pv,
                    "p_bonferroni": min(1.0, pv * 3),
                    "significant_005_bonferroni": min(1.0, pv * 3) < 0.05,
                })
                print(f"  {a} vs {b}: chi2={chi2:.3f} p_bonf={min(1.0,pv*3):.3e}")

        if not args.apply:
            print("  [dry run] nothing written")
            continue

        for m, (path, doc, y, yh) in live.items():
            shutil.copy(path, path + ".bak-t1")
            json.dump(doc, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            bp = path.replace(".json", ".bootstrap.json")
            if os.path.exists(bp):
                bdoc = json.load(open(bp, encoding="utf-8"))
                shutil.copy(bp, bp + ".bak-t1")
                keys = list(bdoc["metrics"].keys())
                bdoc["metrics"] = bootstrap(y, yh, keys,
                                            resamples=bdoc.get("resamples", 10000),
                                            seed=bdoc.get("seed", 42))
                bdoc["label_key_corrected"] = True
                json.dump(bdoc, open(bp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print(f"  bootstrap rewritten: {os.path.basename(bp)}")

        cmp_ = os.path.join(d, "mage_confusion_matrices.json")
        shutil.copy(cmp_, cmp_ + ".bak-t1")
        json.dump(cm, open(cmp_, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        mnp = os.path.join(d, "_mcnemar__mage.json")
        mn = json.load(open(mnp, encoding="utf-8"))
        shutil.copy(mnp, mnp + ".bak-t1")
        mn["pairs"] = pairs
        mn["label_key_corrected"] = True
        json.dump(mn, open(mnp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("  confusion + mcnemar rewritten")

if __name__ == "__main__":
    main()
