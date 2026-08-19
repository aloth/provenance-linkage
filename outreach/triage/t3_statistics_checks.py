#!/usr/bin/env python3
"""Triage checks 6-10 (statistics track).

T6  Late-band sparsity: how many participants actually identify the post-30
    within-person slope, and are they straight-liners?
T7  McNemar presence: does the package contain any McNemar test at all?
T8  Boundary-MLE warnings: which models carry them, and are they disclosed?
T9  Analytical n reproduction: judgments / participants / fragments / origin split.
T10 Origin confounding: do human and machine items share common support on
    format, veracity and language?
"""
import json
import os
import glob
import subprocess
import collections
import statistics
import csv

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages", "redteam-statistics")


def read_csv(name):
    with open(os.path.join(PKG, "data", name), newline="") as fh:
        return list(csv.DictReader(fh))


def build_analytical():
    """Reproduce the analytical set: drop broken, require both scores, join metadata."""
    results = read_csv("results.csv")
    frags = {f["FragmentID"]: f for f in read_csv("fragments-metadata-only.csv")}
    parts = {p["ParticipantID"]: p for p in read_csv("participants-deidentified.csv")}

    def truthy(v):
        return str(v).strip().lower() in {"1", "true", "yes"}

    rows = []
    for r in results:
        if truthy(r.get("ReportedAsBroken")):
            continue
        hm = r.get("HumanMachineScore", "").strip()
        lf = r.get("LegitFakeScore", "").strip()
        if hm == "" or lf == "":
            continue
        fid = r.get("FragmentID", "").strip()
        pid = r.get("ParticipantID", "").strip()
        if fid not in frags or pid not in parts:
            continue
        row = dict(r)
        row["_frag"] = frags[fid]
        row["_part"] = parts[pid]
        rows.append(row)
    return rows


def t9_analytical_n():
    print("\n" + "=" * 70)
    print("T9  Analytical set: do the headline denominators reproduce?")
    print("=" * 70)
    rows = build_analytical()
    pids = {r["ParticipantID"] for r in rows}
    fids = {r["FragmentID"] for r in rows}
    origin = collections.Counter((r["_frag"].get("Origin") or "").strip() or "(blank)" for r in rows)

    print(f"  judgments  : {len(rows)}   (review claims 2,317)")
    print(f"  participants: {len(pids)}   (review claims 395)")
    print(f"  fragments  : {len(fids)}   (review claims 1,052)")
    print("  origin split of judgments:")
    for k, v in origin.most_common():
        print(f"    {k:<12s} {v}")
    print("  review claims 2,271 machine / 46 human judgments")
    return rows


def t6_late_band(rows):
    print("\n" + "=" * 70)
    print("T6  Late response-order band: who identifies the post-30 slope?")
    print("=" * 70)
    # reconstruct sequence exactly as judgegpt_mixed.py does:
    # sort by ParticipantID, timestamp, ResultID; cumcount+1 within participant
    rows_sorted = sorted(rows, key=lambda r: (r["ParticipantID"], r.get("Timestamp", ""), r.get("ResultID", "")))
    seq = collections.defaultdict(int)
    for r in rows_sorted:
        seq[r["ParticipantID"]] += 1
        r["_seq"] = seq[r["ParticipantID"]]

    by_p = collections.defaultdict(list)
    for r in rows_sorted:
        by_p[r["ParticipantID"]].append(r)

    def within_var(vals):
        m = sum(vals) / len(vals)
        return any(abs(v - m) > 1e-12 for v in vals)

    early_informative = 0
    late_informative = []
    for pid, rs in by_p.items():
        early = [min(r["_seq"], 20) for r in rs]
        late = [max(r["_seq"] - 30, 0) for r in rs]
        if within_var(early):
            early_informative += 1
        if within_var(late):
            late_informative.append(pid)

    print(f"  participants with within-person variation in early_wp10 : {early_informative}")
    print(f"  participants with within-person variation in late_wp10  : {len(late_informative)}")
    print("  (review claims 333 vs 4)")

    print("\n  profile of the late-band participants:")
    for pid in late_informative:
        rs = by_p[pid]
        hm = [float(r["HumanMachineScore"]) for r in rs if r["HumanMachineScore"].strip()]
        tta = [float(r["TimeToAnswer"]) for r in rs if r.get("TimeToAnswer", "").strip()]
        distinct = len(set(hm))
        sd = statistics.pstdev(hm) if len(hm) > 1 else 0.0
        med_t = statistics.median(tta) if tta else float("nan")
        flag = "  <-- STRAIGHT-LINER" if distinct <= 2 else ""
        print(f"    {pid[:12]}...  judgments={len(rs):3d}  max_seq={max(r['_seq'] for r in rs):3d}  "
              f"distinct_HM={distinct:3d}  sd={sd:.3f}  median_time={med_t:.4f}{flag}")

    n_straight = sum(1 for pid in late_informative
                     if len({float(r["HumanMachineScore"]) for r in by_p[pid]
                             if r["HumanMachineScore"].strip()}) <= 2)
    print(f"\n  straight-liners among late-band participants: {n_straight}/{len(late_informative)}")
    if len(late_informative) <= 6:
        print("  VERDICT: CONFIRMED. The post-30 within-person slope is identified by")
        print(f"           only {len(late_informative)} participants. A non-significant")
        print("           coefficient here is not evidence against fatigue.")


def t7_mcnemar():
    print("\n" + "=" * 70)
    print("T7  Is any McNemar test present in the statistics package?")
    print("=" * 70)
    hits = []
    for root, _dirs, files in os.walk(PKG):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as fh:
                    if b"mcnemar" in fh.read().lower():
                        hits.append(os.path.relpath(path, PKG))
            except OSError:
                pass
    print(f"  files mentioning 'mcnemar' in redteam-statistics: {len(hits)}")
    for h in hits:
        print(f"    {h}")

    det = os.path.join(os.path.dirname(PKG), "redteam-detection")
    dhits = []
    for root, _dirs, files in os.walk(det):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as fh:
                    if b"mcnemar" in fh.read().lower():
                        dhits.append(os.path.relpath(path, det))
            except OSError:
                pass
    print(f"\n  files mentioning 'mcnemar' in redteam-detection: {len(dhits)}")
    for h in sorted(dhits)[:10]:
        print(f"    {h}")

    if not hits and dhits:
        print("\n  VERDICT: CONFIRMED (with correction). No McNemar test in the")
        print("           STATISTICS package, but McNemar artifacts DO exist in the")
        print("           DETECTION package. The brief mislocates its own method:")
        print("           the reviewer was right that it is absent where described.")
    elif not hits and not dhits:
        print("\n  VERDICT: CONFIRMED. No McNemar test anywhere.")
    else:
        print("\n  VERDICT: REFUTED.")


def t8_boundary_warnings():
    print("\n" + "=" * 70)
    print("T8  Boundary-MLE warnings: present in JSON, absent from prose?")
    print("=" * 70)
    d = json.load(open(os.path.join(PKG, "results", "judgegpt_mixed.json")))
    flagged = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "warnings" and isinstance(v, list):
                    if any("boundary" in str(x).lower() for x in v):
                        flagged.append(path)
                else:
                    walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(d)
    print(f"  models with 'MLE may be on the boundary' warning: {len(flagged)}")
    for f in flagged:
        print(f"    {f}")

    # is it disclosed in the prose documents?
    print("\n  mentions of 'boundary' in the prose reports:")
    found_any = False
    for md in sorted(glob.glob(os.path.join(PKG, "*.md"))):
        with open(md, encoding="utf-8", errors="replace") as fh:
            txt = fh.read().lower()
        n = txt.count("boundary")
        if n:
            found_any = True
        print(f"    {os.path.basename(md):42s} {n}")
    if flagged and not found_any:
        print("\n  VERDICT: CONFIRMED. Load-bearing models carry boundary warnings that")
        print("           appear nowhere in the prose.")


def t10_common_support(rows):
    print("\n" + "=" * 70)
    print("T10  Common support: do human and machine items coexist in strata?")
    print("=" * 70)
    for field in ["Format", "IsFake", "ISOLanguage", "Style"]:
        table = collections.defaultdict(collections.Counter)
        for r in rows:
            f = r["_frag"]
            origin = (f.get("Origin") or "").strip() or "(blank)"
            key = (f.get(field) or "").strip() or "(blank)"
            table[key][origin] += 1
        print(f"\n  --- {field} ---")
        both = 0
        for key in sorted(table):
            c = table[key]
            origins = [o for o in c if o != "(blank)"]
            has_both = len(origins) >= 2
            both += has_both
            cells = "  ".join(f"{o}={c[o]}" for o in sorted(c))
            print(f"    {key:<18s} {cells}{'   <-- both' if has_both else ''}")
        print(f"    strata containing BOTH origins: {both}/{len(table)}")


if __name__ == "__main__":
    rows = t9_analytical_n()
    t6_late_band(rows)
    t7_mcnemar()
    t8_boundary_warnings()
    t10_common_support(rows)
