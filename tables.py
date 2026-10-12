#!/usr/bin/env python
"""The tables of the paper that the scored runs support, from outputs/scores to
results/. Tables 6-11 carry each mask tier's mean coverage, the x-axis of Figure 5.

    python tables.py              # writes results/table<N>.txt

Table numbers follow the paper. Tables 1, 2 and 4 pool the complete runs among
1-5 (mean and sample SD across runs; one run has no SD); every other table is a
single run with mean and sample SD across subjects. A table with nothing scored
is skipped, a partly scored one shows the rows it has.
"""
import os

import numpy as np
from PIL import Image

import config as C
from piar import manifest as M
from piar.aggregate import across_runs, across_subjects, dataset_scores, fmt, load_rows

OUT = C.RESULTS
MAIN_RUNS = (1, 2, 3, 4, 5)
NAME = {"cfg": "CFG", "finextract": "FineXtract", "base": "Base Inpainting",
        "personalized": "Personalized Inpainting", "piar": "PIAR"}
DATASET = {"celebahq": "CelebA-HQ", "dreambooth": "DreamBooth", "customconcept101": "CustomConcept101"}
HELDOUT_DATASET = {"celebahq": "celebahq", "dreambooth": "dreambooth_heldout",
                   "customconcept101": "customconcept101"}
TIER = {"partial": "Partial", "small": "Small", "large": "Large", "xlarge": "Extra-large"}
MAIN_COLS = ("AS", "A-ESR@0.8", "CLIP", "LPIPS", "SSIM", "ArcFace")
FULL_COLS = ("AS", "A-ESR@0.7", "A-ESR@0.8", "CLIP", "LPIPS", "SSIM", "ArcFace")


# ------------------------------------------------------------------ rows --
def rows(run, kind, dataset, tier=None):
    p = M.scores_path(run, kind, dataset, tier)
    return load_rows(p) if os.path.exists(p) else []


def main_rows(run, dataset):
    """Table 1's units: masked arms at N=20 on the small tier, and the free-form
    baselines at their full budget. Both carry a 'method' key."""
    out = []
    for r in rows(run, "masked", dataset, C.MAIN_TIER):
        if r["n"] == C.N_MASKED and r["condition"] == "train":
            out.append({**r, "method": r["arm"]})
    for r in rows(run, "freeform", dataset):
        if r["budget_per_image"] == C.FREEFORM_PER_IMAGE:
            out.append(r)
    return out


def specificity_rows(run, dataset):
    """Table 2's units: the masked arms with the target present (Table 1's
    rows) and, for Personalized and PIAR, with the target absent."""
    out = []
    for r in rows(run, "masked", dataset, C.MAIN_TIER):
        if r["n"] == C.N_MASKED and r["condition"] == "train":
            out.append({**r, "cond": "present"})
    for r in rows(run, "wrong", dataset, C.MAIN_TIER):
        if r["n"] == C.N_MASKED:
            out.append({**r, "cond": "absent"})
    return out


# --------------------------------------------------------------- render --
def render(number, caption, header, body):
    """body: one row (label cells..., value cells...) per group that has scores.
    Writes results/table<number>.txt, or skips the table when nothing is scored."""
    if not body:
        print(f"Table {number} ({caption}): nothing scored yet\n")
        return
    os.makedirs(OUT, exist_ok=True)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    lines = [f"Table {number}: {caption}", "", "  ".join(str(h).ljust(w) for h, w in zip(header, widths))]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in body]
    txt = "\n".join(lines)
    with open(f"{OUT}/table{number}.txt", "w") as fh:
        fh.write(txt + "\n")
    print(txt + "\n")


def cells(stats, cols, faces=True):
    return [fmt(stats.get(c)) if (faces or c != "ArcFace") else "–" for c in cols]


def complete_runs(kind, dataset, rows_fn):
    """Runs whose score file exists and covers every subject of the dataset. A
    partially scored run would bias the pooled mean, so it is reported and skipped."""
    try:
        expected = len(M.subjects(dataset))
    except FileNotFoundError:
        expected = None                        # no manifest: cannot check, pool what exists
    out = []
    for r in MAIN_RUNS:
        if not os.path.exists(M.scores_path(r, kind, dataset, C.MAIN_TIER)):
            continue
        if expected is not None:
            got = len({x["subject"] for x in rows_fn(r, dataset)})
            if got < expected:
                print(f"run {r} {dataset}: {got} of {expected} subjects scored, not pooled")
                continue
        out.append(r)
    return out


# ---------------------------------------------------------------- tables --
def table_main(number, pooled):
    """Table 1 (pooled over runs) and Table 3 (run 1, across subjects)."""
    cols = MAIN_COLS if pooled else FULL_COLS
    body = []
    for ds in C.DATASETS:
        if pooled:
            runs = complete_runs("masked", ds, main_rows)
            stats = across_runs([dataset_scores(main_rows(r, ds), ("method",)) for r in runs]) if runs else {}
            tag = f"{len(runs)} run" + ("" if len(runs) == 1 else "s")
        else:
            stats = across_subjects(main_rows(1, ds), ("method",))
            tag = "run 1"
        first = len(body)
        for m in NAME:
            s = stats.get((m,))
            if s:
                body.append([DATASET[ds], NAME[m]] + cells(s, cols, ds == "celebahq"))
        if len(body) > first:
            body[-1][0] += f" [{tag}]"
    render(number, "Main reconstruction results" + (", mean ± SD across runs" if pooled else ", run 1, spread across subjects"),
           ["Dataset", "Method"] + list(cols), body)


def table_specificity(number, datasets, pooled, cols):
    body = []
    for ds in datasets:
        if pooled:
            runs = complete_runs("wrong", ds, specificity_rows)
            stats = across_runs([dataset_scores(specificity_rows(r, ds), ("arm", "cond")) for r in runs]) if runs else {}
        else:
            stats = across_subjects(specificity_rows(1, ds), ("arm", "cond"))
        for arm, cond, label in (("base", "present", "–"), ("personalized", "present", "Present"),
                                 ("personalized", "absent", "Absent"), ("piar", "present", "Present"),
                                 ("piar", "absent", "Absent")):
            s = stats.get((arm, cond))
            if s:
                body.append([DATASET[ds], NAME[arm], label] + cells(s, cols, ds == "celebahq"))
    render(number, "Checkpoint specificity" + (", mean ± SD across runs" if pooled else ", run 1, spread across subjects"),
           ["Dataset", "Method", "Target"] + list(cols), body)


def table_tiers(number, dataset, kind, run):
    """Tables 6-8 (mask sweep, run 1) and 9-11 (held-out, run 0)."""
    ds = HELDOUT_DATASET[dataset] if kind == "heldout" else dataset
    body = []
    for tier in C.MASK_TIERS:
        rs = [r for r in rows(run, kind, ds, tier) if r["n"] == C.N_MASKED]
        stats = across_subjects(rs, ("arm",))
        for arm in ("base", "personalized", "piar"):
            s = stats.get((arm,))
            if s:
                body.append([f"{TIER[tier]} ({100 * coverage(tier):.0f}%)", NAME[arm]]
                            + cells(s, FULL_COLS, dataset == "celebahq"))
    what = "held-out photographs" if kind == "heldout" else "fine-tuning photographs"
    render(number, f"{DATASET[dataset]}, {what} across mask tiers (run {run}, spread across subjects)",
           ["Mask tier (coverage)", "Method"] + list(FULL_COLS), body)


def table_budget(number, dataset, run=1):
    masked = [r for r in rows(run, "masked", dataset, C.MAIN_TIER) if r["condition"] == "train"]
    free = rows(run, "budget", dataset)
    sm = across_subjects(masked, ("n", "arm"))
    sf = across_subjects(free, ("budget_per_image", "method"))
    body = []
    for n in C.BUDGETS:
        values = ([fmt(sf[(n, m)]["AS"]) if (n, m) in sf else "–" for m in ("cfg", "finextract")]
                  + [fmt(sm[(n, a)]["AS"]) if (n, a) in sm else "–" for a in ("base", "personalized", "piar")])
        if any(v != "–" for v in values):
            body.append([str(n)] + values)
    render(number, f"{DATASET[dataset]}, AS across generation budgets (run {run}, spread across subjects)",
           ["N"] + [NAME[m] for m in NAME], body)


def table_ablation():
    cols = FULL_COLS[:-1]                      # no ArcFace: two of the three datasets are objects
    body15, body16 = [], []
    for ds in C.DATASETS:
        blind = [r for r in rows(0, "ablation_blind", ds) if r["n"] == C.N_MASKED]
        prompted = [r for r in rows(0, "ablation_prompted", ds) if r["n"] == C.N_MASKED]
        sb, sp = across_subjects(blind, ("w",)), across_subjects(prompted, ("w",))
        for w in C.ABLATION_W:
            if (w,) in sb:
                body15.append([DATASET[ds], f"{w:g}"] + cells(sb[(w,)], cols))
        if (C.W_PIAR,) in sb:
            body16.append([DATASET[ds], "Empty (default)"] + cells(sb[(C.W_PIAR,)], cols))
        if (C.W_PIAR,) in sp:
            body16.append([DATASET[ds], "Training prompt"] + cells(sp[(C.W_PIAR,)], cols))
    render(15, "Guidance weight, empty prompt (run 0, six subjects per dataset)", ["Dataset", "w"] + list(cols), body15)
    render(16, "Prompt given to PIAR at w=3 (run 0, six subjects per dataset)", ["Dataset", "Prompt"] + list(cols), body16)


# ------------------------------------------------------------- coverage --
def coverage(tier):
    covs = [(np.array(Image.open(M.mask_path(tier, m)).convert("L")) > 127).mean() for m in M.masks(tier)]
    return float(np.mean(covs))


def main():
    table_main(1, pooled=True)
    table_specificity(2, ["celebahq"], pooled=True, cols=MAIN_COLS)
    table_main(3, pooled=False)
    table_specificity(4, list(C.DATASETS), pooled=True, cols=FULL_COLS)
    table_specificity(5, list(C.DATASETS), pooled=False, cols=FULL_COLS)
    for i, ds in enumerate(C.DATASETS):
        table_tiers(6 + i, ds, "masked", run=1)
    for i, ds in enumerate(C.DATASETS):
        table_tiers(9 + i, ds, "heldout", run=0)
    for i, ds in enumerate(C.DATASETS):
        table_budget(12 + i, ds)
    table_ablation()


if __name__ == "__main__":
    main()
