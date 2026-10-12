#!/usr/bin/env python
"""Draw the quick-test figure from the demo tree (outputs/demo).

    python figure.py        # -> results/demo.png

For each dataset: the subject's fine-tuning images, then the attacked
photograph, its masked input and what each attack recovers, in the layout of
Figure 3 (GT, Masked, CFG, FineXtract, Base, Pers., PIAR), with each recovered
image's SSCD similarity to the photograph. The masked cells are the SSCD medoid
of the 20 candidates and the free-form cells the extracted image matched to the
photograph: the images the scores in outputs/demo/scores were computed on.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

import config as C
from piar import manifest as M

C.OUTPUTS, C.SCORES = C.DEMO, f"{C.DEMO}/scores"   # read the demo tree, wherever outputs/ is

CELL, PAD, TEXT = 256, 8, 24
COLUMNS = (("GT", None), ("Masked", None), ("CFG", "cfg"), ("FineXtract", "finextract"),
           ("Base", C.W_BASE), ("Pers.", C.W_PERSONALIZED), ("PIAR", C.W_PIAR))
TITLES = {"celebahq": "CelebA-HQ", "dreambooth": "DreamBooth", "customconcept101": "CustomConcept101"}


def rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def thumb(path):
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def block(dataset, run=0):
    """The demo subject's fine-tuning images, and (image path, SSCD) per column."""
    e = M.subjects(dataset)[0]
    ck = M.checkpoint_name(e, run)
    train = M.train_photos(e)
    target = train[C.DATASETS.index(dataset)]          # the photograph experiment.py demo attacked
    masked = [r for r in rows(M.scores_path(run, "masked", dataset, C.MAIN_TIER))
              if r["subject"] == e["subject"] and r["photo"] == os.path.basename(target) and r["n"] == C.N_MASKED]
    free = [r for r in rows(M.scores_path(run, "freeform", dataset))
            if r["subject"] == e["subject"] and r["photo"] == os.path.basename(target)]
    if len(masked) < 3 or len(free) < 2:
        raise SystemExit(f"{dataset}: {e['subject']} is not scored in {C.SCORES}; run `experiment.py demo` first")
    group = f"{M.masked_dir(run, 'masked', C.MAIN_TIER, ck)}/w{{w:g}}/{M.group_name(e['subject'], target, masked[0]['mask'])}"
    cells = []
    for label, key in COLUMNS:
        if label == "GT":
            cells.append((target, None))
        elif label == "Masked":
            cells.append((f"{group.format(w=C.W_BASE)}/masked_input.png", None))
        elif isinstance(key, str):
            r = next(r for r in free if r["method"] == key)
            cells.append((f"{M.freeform_dir(run, ck, key)}/{r['matched']}", r["sscd"]))
        else:
            r = next(r for r in masked if r["w"] == key)
            cells.append((f"{group.format(w=key)}/{r['selected']}", r["sscd"]))
    return train, cells


def draw(datasets, out):
    font, bold = ImageFont.load_default(size=16), ImageFont.load_default(size=19)
    blocks = [(ds, *block(ds)) for ds in datasets]
    ncol = max(len(COLUMNS), max(len(train) for _, train, _ in blocks))
    height = TEXT + CELL + PAD + TEXT + CELL + TEXT + PAD          # one dataset
    canvas = Image.new("RGB", (PAD + ncol * (CELL + PAD), PAD + len(blocks) * height), "white")
    d = ImageDraw.Draw(canvas)
    y = PAD
    for ds, train, cells in blocks:
        d.text((PAD, y), f"{TITLES.get(ds, ds)}: fine-tuning images", fill="black", font=bold)
        y += TEXT
        for i, p in enumerate(train):
            canvas.paste(thumb(p), (PAD + i * (CELL + PAD), y))
        y += CELL + PAD
        for i, (label, _) in enumerate(COLUMNS):
            d.text((PAD + i * (CELL + PAD), y), label, fill="black", font=bold)
        y += TEXT
        for i, (p, sscd) in enumerate(cells):
            x = PAD + i * (CELL + PAD)
            canvas.paste(thumb(p), (x, y))
            if sscd is not None:
                d.text((x, y + CELL + 3), f"SSCD {100 * sscd:.1f}", fill="black", font=font)
        y += CELL + TEXT + PAD
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(C.DATASETS))
    ap.add_argument("--out", default=f"{C.RESULTS}/demo.png")
    args = ap.parse_args()
    draw(args.datasets.split(","), args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
