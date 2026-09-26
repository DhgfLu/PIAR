#!/usr/bin/env python
"""The paper's experiments, end to end. Each command trains what is missing,
generates, and scores; rerunning skips what already exists.

    python experiment.py main --run 1 --sweep --gpu 0   # run 1: Tables 1-8, 12-14, Figure 6
    python experiment.py main --run 2 --gpu 0           # runs 2-5: Tables 1, 2, 4
    python experiment.py heldout --gpu 0                # Tables 9-11, Figure 6(a) dashed
    python experiment.py ablation --gpu 0               # Tables 15-16
    python tables.py                                    # every table, from results/

`main` covers, per dataset: victim training, the three masked arms on every
fine-tuning photograph, the two free-form baselines, the checkpoint-specificity
control, and scoring. `--sweep` adds the other three mask tiers and 50-candidate
pools for the budget sweep, which the paper takes from run 1.
"""
import argparse
import random
import subprocess
import sys

import config as C
from piar import manifest as M


def sh(script, *args):
    cmd = [sys.executable, f"{C.ROOT}/{script}", *[str(a) for a in args]]
    print("$", " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit(f"{script} failed")


def gpu_args(gpu):
    return ["--gpu", gpu] if gpu is not None else []


def main_run(run, sweep, datasets, gpu):
    g = gpu_args(gpu)
    for ds in datasets:
        sh("train.py", "--dataset", ds, "--run", run, *g)
        if sweep:
            sh("attack.py", "--dataset", ds, "--run", run, "--tier", "small", "--n", C.N_MASKED_SWEEP, *g)
            sh("attack.py", "--dataset", ds, "--run", run, "--tier", "all", *g)
        else:
            sh("attack.py", "--dataset", ds, "--run", run, *g)
        sh("attack.py", "--dataset", ds, "--run", run, "--target", "wrong", "--w", "1,3", *g)
        sh("extract.py", "--dataset", ds, "--run", run, *g)
        what = "masked,wrong,freeform" + (",budget" if sweep else "")
        sh("score.py", "--dataset", ds, "--run", run, "--what", what,
           "--tier", "all" if sweep else C.MAIN_TIER, *g)


def heldout(gpu):
    g = gpu_args(gpu)
    for ds in ("celebahq", "customconcept101", "dreambooth_heldout"):
        sh("train.py", "--dataset", ds, "--run", 0, *g)
        sh("attack.py", "--dataset", ds, "--run", 0, "--target", "heldout", "--tier", "all", *g)
        sh("score.py", "--dataset", ds, "--run", 0, "--what", "heldout", "--tier", "all", "--n", C.N_MASKED, *g)


def ablation(gpu):
    g = gpu_args(gpu)
    subjects = sorted(M.subjects("celebahq"), key=lambda e: e["index"])
    picked = sorted(random.Random(C.ABLATION_SEED).sample(subjects, C.ABLATION_N_SUBJECTS),
                    key=lambda e: e["index"])
    names = ",".join(e["subject"] for e in picked)
    print(f"ablation subjects: {names}")
    sh("train.py", "--dataset", "celebahq", "--run", 0, "--subjects", names, *g)
    sh("attack.py", "--dataset", "celebahq", "--run", 0, "--subjects", names,
       "--w", ",".join(f"{w:g}" for w in C.ABLATION_W), "--kind", "ablation_blind", *g)
    sh("attack.py", "--dataset", "celebahq", "--run", 0, "--subjects", names,
       "--w", f"{C.W_PIAR:g}", "--prompt", "training", "--kind", "ablation_prompted", *g)
    sh("score.py", "--dataset", "celebahq", "--run", 0, "--what", "ablation", *g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("main", help="one run of the main comparisons")
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--sweep", action="store_true", help="also the mask-coverage and budget sweeps")
    p.add_argument("--datasets", default=",".join(C.DATASETS))
    p.add_argument("--gpu", default=None)
    p = sub.add_parser("heldout", help="held-out photographs, all tiers")
    p.add_argument("--gpu", default=None)
    p = sub.add_parser("ablation", help="guidance weight and prompt ablations")
    p.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.cmd == "main":
        main_run(args.run, args.sweep, args.datasets.split(","), args.gpu)
    elif args.cmd == "heldout":
        heldout(args.gpu)
    else:
        ablation(args.gpu)


if __name__ == "__main__":
    main()
