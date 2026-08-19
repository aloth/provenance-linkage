# Reproducibility bundle: a benchmark audit of an AI-text detection evaluation

This bundle accompanies the manuscript *One Prediction Set, Two Reported
Results: Provenance Linkage and a Reproducible Benchmark-Audit Sequence for
AI-Text Detection*. It contains the audit
scripts, the archived per-item predictions they consume, and the per-run metric
files, so that every count, rate, and ranking metric reported in the paper can
be re-derived without retraining a model.

## What is here

```
scripts/       nine audit scripts (the seven-step sequence plus the table recompute)
predictions/   archived per-item predictions, stripped of corpus text
metrics/       per-run metric files (accuracy, macro F1, per-class rates)
MANIFEST.json  SHA-256 for every file, plus the hash of each source file
```

## What is deliberately absent

No verbatim corpus text is redistributed. The archived prediction dumps carry a
`text_preview` field holding up to 200 characters of source text; every such
field was removed before packaging (5,976 fields across 32 files, recorded
per file in `MANIFEST.json`). RAID, M4, and MAGE are third-party resources under
their own licences, and the in-domain RogueGPT corpus is released under
restricted academic access. None of them may be republished here.

What remains is exactly what the audit consumes: stored label, predicted class,
the two logits, and the provenance marker. That is sufficient for Steps 1, 4, 6,
and 7 in full. Step 5, which reconstructs the realized composition of the
external draws, additionally requires authorized local copies of the raw
corpora; `t5_corpus_composition.py` documents the expected paths.

## A note on Step 1

`t1_mage_labels.py` returns **REFUTED** when run against this bundle, and that is
the expected result. The check asks whether the stored `label` field disagrees
with the `src` provenance marker. In the archived dumps shipped here it agrees,
because `t1_fix_mage_labels.py` rewrote those files in place on 2026-08-18 and
recorded the change in a `label_key_corrected` flag plus a `label_key_note` in
every affected file.

The 0.000-stored / 1.000-flipped disagreement reported in the manuscript is
therefore reproducible only against the pre-correction snapshot, which survives
in the conference-submission repository, not in this bundle. Readers can still
verify the finding two ways from what is here: the `label_key_note` field states
the original convention and the date of correction, and
`recompute_arr_tables.py` derives authorship from `src` independently of the
stored key, reproducing both orientations side by side (manuscript Table 2).

This is itself an instance of the paper's argument. An in-place correction that
is well documented inside the file is still an in-place correction: the artifact
no longer reproduces the defect it was collected to demonstrate. Preserving the
pre-correction snapshot alongside the corrected one would have been the better
disposition, and we say so rather than present the bundle as if the issue did
not arise.

## Verifying integrity

```
python3 build_bundle.py --check --out <bundle-dir>
```

This recomputes the SHA-256 of every shipped file against `MANIFEST.json` and
independently re-scans the prediction files for any string longer than 80
characters, which would indicate that corpus text slipped through the sanitizer.
A clean bundle prints `OK`.

`MANIFEST.json` records both the hash of each sanitized file and the hash of the
source file it was derived from. A reader with authorized access to the original
artifacts can therefore confirm that these dumps came from the archived
originals rather than from a regeneration.

## Reproducing the reported numbers

Scripts are plain Python 3 with no third-party dependencies beyond the standard
library, except where noted in the file header. Each writes JSON to stdout and
prints a verdict.

| Script | Manuscript location |
|---|---|
| `t1_mage_labels.py` | Step 1, label-orientation agreement (0.000 stored, 1.000 flipped) |
| `recompute_arr_tables.py` | Table 2 (divergence), Table 5 (confusion matrices) |
| `t1_fix_mage_labels.py` | Step 1 disposition, corrected-key derivation |
| `t2_detection_checks.py` | Steps 2 and 3, split integrity and cheap baselines |
| `logo_identifiability.py` | Step 4, leave-one-group-out identifiability |
| `t5_corpus_composition.py` | Step 5, realized composition (needs raw corpora) |
| `t3_statistics_checks.py` | Step 4 and analytical-set derivation |
| `auroc_audit.py` | Step 7, threshold-free discrimination |
| `judgegpt_analytical_set.py` | analytical-set filtering, reported in Limitations |

## Licence

Scripts: MIT. Derived metric and prediction files: CC BY 4.0. Neither licence
extends to the third-party corpora, which retain their original terms.
