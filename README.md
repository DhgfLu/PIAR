# PIAR: Partial-Image-Assisted Reconstruction

Code for *Partial Images Raise Further Privacy Concerns in Personalized Diffusion Models*.

```
config.py         paths you edit, protocol constants you don't
piar/             the method: pipeline, guidance, freeform, select, metrics, manifest, aggregate
prepare_data.py   build the study tree from data/ and your downloads
train.py          one victim checkpoint per subject
attack.py         Base / Personalized / PIAR on one checkpoint
extract.py        CFG / FineXtract on one checkpoint
score.py          every metric for every reconstruction
experiment.py     the paper's experiments end to end
tables.py         every table and figure CSV, from results/
data/             subject and image lists, the 40 masks
results/          score rows written by score.py, read by tables.py
```

## Setup

1. `conda env create -f environment.yml && conda activate piar`
2. Download the datasets: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ),
   the [DreamBooth dataset](https://github.com/google/dreambooth) (`dataset/`),
   [CustomConcept101](https://github.com/adobe-research/custom-diffusion) (`customconcept101/`).
3. Download the weights: [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4),
   [BrushNet](https://github.com/TencentARC/BrushNet) (`random_mask_brushnet_ckpt`),
   [SSCD](https://github.com/facebookresearch/sscd-copy-detection) (`sscd_disc_large.torchscript.pt`).
   LPIPS, CLIP ViT-B/32 and the InsightFace `buffalo_l` models download themselves on first use.
4. Point `config.RAW` and `config.WEIGHTS` at them, or set `PIAR_RAW_*`, `PIAR_SD14`, `PIAR_BRUSHNET`, `PIAR_SSCD`.
5. `python prepare_data.py`

## Run

```
python experiment.py main --run 1 --sweep --gpu 0    # then --run 2 ... --run 5
python experiment.py heldout --gpu 0
python experiment.py ablation --gpu 0
python tables.py
```

Each command skips what already exists and can be resumed. `tables.py` writes
`tables/table<N>.txt`, `.tex` and the Figure 6 CSVs from whatever runs are in `results/`.

One subject at a time: `python attack.py --dataset celebahq --subjects 7613`
(`--target heldout`, `--target wrong`, `--tier all`, `--w`, `--n`, `--prompt` select the variants).
