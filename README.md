# ITF
Information Theoretic Attribution
# ITF: Conditional-Information Attribution for Image Classifiers

Reference implementation of **ITF**, a set-level feature attribution method based on
**pointwise conditional mutual information** between a pixel set and the model's
predicted class. Unlike gradient-path methods, ITF measures the change in
`p(y|x)` when a set of pixels is replaced with a counterfactual sample — making
it directly aligned with removal-based faithfulness metrics. The repo also includes **Sparse_ITF**,
the set-uniform variant of ITF.

Evaluated on **ImageNet** and **Caltech-UCSD Birds (CUB)** across
VGG16, ResNet101, DenseNet121, and MobileNetV3, against the H-Sets paper's
reported results.

---

## Table of Contents

- [Method](#method)
- [Metrics](#metrics)
- [Results](#results)
- [Hyperparameters](#hyperparameters)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Downloading Datasets](#downloading-datasets)
- [Downloading SAM and Model Checkpoints](#downloading-sam-and-model-checkpoints)
- [Running the Pipeline](#running-the-pipeline)
- [Ablations](#ablations)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Method

For an input image `x ∈ ℝᵈ`, classifier `f`, and predicted class
`y = argmax_c f_c(x)`, ITF attributes a pixel set `S ⊆ [d]` by:

```
φ(S) = log p_f(y | x)  −  log E_{X_S ~ q(·|x_{−S})}[ p_f(y | X_S, x_{−S}) ]     (1)
```

where `q(·|x_{−S})` is a fill distribution for the removed pixels.
`φ(S) > 0` means the set `S` supports the prediction; `φ(S) < 0` means it argues
against it; `φ(S) = 0` means `y` is conditionally independent of `S`.

**Zero fill.** We use `q = δ(x_S = 0)`, which makes the expectation deterministic:
one forward pass per set, no Monte Carlo. This also aligns ITF with the
counterfactual used by Set-Faithfulness Correlation.

**Two-stage attribution.**

1. **Set proposals.** SAM's automatic mask generator (ViT-H) proposes semantic
   regions with area ∈ [50, 5000], stability > 0.9, NMS-IoU 0.7.
2. **Set + pixel scoring.** Compute `φ(S)` at each SAM proposal (top-B=5 refined
   at fine granularity), and at coarse 32×32 or 64×64 blocks everywhere.

**Two-level hierarchy.** Coarse blocks (32×32 for CUB, 64×64 for ImageNet) give
global structure; the top-B=5 SAM proposals are refined at 8×8 blocks inside
the mask. This trades ~8× compute for image-boundary fidelity and was validated
to have no metric impact (see `cache/itf_ablations/`).

### Sparse_ITF (set-uniform)

`--set_uniform` on `run_itf.py` produces the sparse variant used in the results table. It shares the identical per-set score φ(S) and identical top-B SAM proposals as two-level ITF; only the per-pixel map differs.

| | Two-level ITF | Sparse_ITF |
|---|---|---|
| Sets scored | coarse grid + top-B SAM proposals | top-B SAM proposals only |
| Coarse grid | 64×64 (IN) / 32×32 (CUB) | **none** — early return before coarse path |
| Refine blocks | 8×8 inside top-B union | none |
| Per-pixel assignment | block-constant per coarse/refine block | mask-constant per selected SAM mask |
| Nonzero support | every pixel | union of top-B masks only |
| Selection (`top_idx`) | `argsort(|φ(S)|)[:B]` | **identical** |
| Set function φ | zero-fill ITF | **identical** |

**Why two variants.** Two-level ITF is the recommended default: the dense map is the honest object for ROAD_AOPC, which removes and imputes pixels across the whole image. Sparse_ITF is the matched comparator for set-based baselines (H-Sets, Archipelago, MoXI): same set budget, same SAM proposals, comparable sparsity — so sparsity and per-pixel faithfulness are not confounded by map density.

The two variants share φ and the top-B selection, so **Harsanyi decomposition, top-B selection Jaccard, and Set-ROAD / Set-AOPC are identical for ITF and Sparse_ITF**. The choice affects the per-pixel map, not the set-level attribution.

**Harsanyi residual.** The per-pixel map `M` is not forced to be set-consistent
with `φ(S)`; the residual `φ(S) − Σ_{j∈S} M_j` is the Harsanyi interaction
(synergy if >0, redundancy if <0).

---

## Metrics

All three metrics are computed on the **same 1000 correctly-classified samples
per (dataset, model)** and the **same pixel map `M`**. The metric code is
identical to `scripts/hsets_metrics.py`.

### 1. Gini index (sparsity, higher = sparser)

For attribution vector `φ(x) ∈ ℝᵈ` with `|φ|` sorted ascending:

```
G(φ) = 1 − 2 Σ_{k=1}^{d}  |φ|_(k) / ‖φ‖₁  ·  (d − k + 0.5)/d
```

Range [0,1]. G=1 = single nonzero pixel; G=0 = uniform.

### 2. ROAD_AOPC (faithfulness under removal, higher = more faithful)

Iteratively remove the top-k most attributed pixels, impute with **noisy linear
imputation** (σ = 0.01, Rong et al. 2022), and measure the drop in
**classification accuracy** on the true class:

```
ROAD_AOPC = (1/(L+1)) Σ_{k=1}^{L} [ acc(x⁰) − acc(xᵏ) ]
```

Higher = a sharper accuracy drop under removing the top-ranked pixels = a more
faithful attribution.

### 3. Faithfulness Correlation (higher = more faithful)

Sample random pixel subsets `S ⊆ [d]`, replace with the zero baseline, and
measure the Pearson correlation between the **average attribution** on `S` and
the **change in the predicted logit**:

```
FC = ρ_Pearson ( {mean_{j∈S} φ_j},  { f_y(x) − f_y(x_S) } )
```

Uniform subset size `|S| = 20% · d`; 10 subsets per image; pooled across 1000 images.

**Important:** ROAD_AOPC and FC measure different notions of faithfulness.
ITF aligns with ROAD_AOPC (both use the zero/noisy-linear counterfactual);
per-pixel correlation methods like H-Sets align with FC. Neither metric is
universally "correct" — each measures a different property.

---

## Results

All numbers are **mean ± std across 5 seeds × 1000 samples**.
H-Sets reference values are the paper-reported figures.

### ROAD_AOPC (higher is better)

| Dataset  | Model       | ITF                 | Sparse_ITF       | H-Sets (paper) |
|----------|-------------|---------------------|------------------|----------------|
| CUB      | VGG16       | **0.748 ± 0.002**   | 0.712 ± 0.003    | 0.65 ± 0.00    |
| CUB      | ResNet101   | **0.766 ± 0.001**   | 0.671 ± 0.005    | 0.63 ± 0.01    |
| CUB      | DenseNet121 | **0.766 ± 0.003**   | 0.679 ± 0.003    | 0.65 ± 0.00    |
| CUB      | MobileNetV3 | **0.758 ± 0.002**   | 0.665 ± 0.007    | 0.60 ± 0.00    |
| ImageNet | VGG16       | **0.656 ± 0.004**   | 0.567 ± 0.007    | 0.34 ± 0.01    |
| ImageNet | ResNet101   | **0.611 ± 0.009**   | 0.520 ± 0.008    | 0.26 ± 0.02    |
| ImageNet | DenseNet121 | **0.623 ± 0.004**   | 0.526 ± 0.006    | 0.38 ± 0.01    |
| ImageNet | MobileNetV3 | **0.578 ± 0.008**   | 0.512 ± 0.009    | 0.37 ± 0.01    |

### Gini (sparsity; higher = sparser)

| Dataset  | Model       | ITF           | Sparse_ITF          | H-Sets (paper)     |
|----------|-------------|---------------|---------------------|--------------------|
| CUB      | VGG16       | 0.901 ± 0.002 | **0.950 ± 0.000**   | 0.93 ± 0.02        |
| CUB      | ResNet101   | 0.788 ± 0.003 | **0.939 ± 0.001**   | 0.92 ± 0.02        |
| CUB      | DenseNet121 | 0.778 ± 0.003 | **0.939 ± 0.001**   | 0.91 ± 0.02        |
| CUB      | MobileNetV3 | 0.774 ± 0.003 | **0.935 ± 0.001**   | 0.89 ± 0.02        |
| ImageNet | VGG16       | 0.752 ± 0.003 | 0.909 ± 0.003       | **0.95 ± 0.01**    |
| ImageNet | ResNet101   | 0.771 ± 0.002 | 0.913 ± 0.000       | **0.98 ± 0.01**    |
| ImageNet | DenseNet121 | 0.755 ± 0.001 | 0.908 ± 0.001       | **0.94 ± 0.01**    |
| ImageNet | MobileNetV3 | 0.762 ± 0.002 | 0.899 ± 0.001       | **0.95 ± 0.01**    |

### Faithfulness Correlation (higher = more faithful)

| Dataset  | Model       | ITF               | Sparse_ITF        | H-Sets (paper)      |
|----------|-------------|-------------------|-------------------|---------------------|
| CUB      | VGG16       | 0.070 ± 0.005     | 0.085 ± 0.009     | **0.124 ± 0.137**†  |
| CUB      | ResNet101   | 0.052 ± 0.003     | 0.033 ± 0.005     | **0.123 ± 0.145**†  |
| CUB      | DenseNet121 | **0.071 ± 0.010** | 0.058 ± 0.005     | 0.029 ± 0.126†      |
| CUB      | MobileNetV3 | 0.054 ± 0.003     | 0.037 ± 0.006     | **0.128 ± 0.119**†  |
| ImageNet | VGG16       | 0.026 ± 0.007     | **0.027 ± 0.008** | 0.009 ± 0.122†      |
| ImageNet | ResNet101   | 0.016 ± 0.007     | 0.014 ± 0.008     | **0.251 ± 0.081**   |
| ImageNet | DenseNet121 | 0.021 ± 0.006     | 0.015 ± 0.005     | **0.034 ± 0.135**†  |
| ImageNet | MobileNetV3 | 0.017 ± 0.004     | 0.013 ± 0.003     | **0.021 ± 0.143**†  |

**Interpretation.**

- **ITF wins ROAD_AOPC on all 8 configs.** This is expected: ITF's functional
  answers the same "what if we remove this set?" question that ROAD measures.
- **ITF produces less concentrated maps (Gini ~0.75–0.90) than H-Sets
  (~0.89–0.98).** This reflects ImageNet's distributed decision structure. H-Sets'
  architectural sparsity (5 sets × ~2000 pixels) forces concentration that ITF's
  block-based map cannot honestly reproduce.
- **ITF loses FC on most configs.** FC measures per-pixel logit correlation;
  ITF produces block-constant maps that don't track per-pixel gradients well.

- **Sparse_ITF dominates H-Sets on CUB**: at matched set budget (B=5) it is both sparser (Gini 0.935–0.950 vs 0.89–0.93) and more faithful (ROAD 0.665–0.712 vs 0.60–0.65) on all four CUB models. On ImageNet, H-Sets is ~0.04 sparser; the `top_B` ablation shows Sparse_ITF still wins ROAD at matched Gini (B=1–3) — the sparsity–faithfulness Pareto frontier does not pass through H-Sets.



---

## Hyperparameters

Locked after extensive ablation (see `cache/itf_ablations/`).

| Param | Value | Range tested | Effect | Rationale |
|---|---|---|---|---|
| `coarse_size` (ImageNet) | **64** | 8, 32, 48, 64, 96, 128, 224 | Gini peaks at 48–64; **collapses to 0.15 at 224** (single block ⇒ uniform) | Saves 4× cost vs 32 with same Gini |
| `coarse_size` (CUB) | **32** | 8, 32, 48, 64 | CUB has strong per-part evidence; 32×32 captures it | Validated in ablation grid |
| `top_B` | **5** | 1, 5, 9 | tb1 drifts corr +0.027; tb9 plateaus | Matches H-Sets' set budget |
| `block_size` (refine) | **8** | 4, 8, 16 | bs16 drifts corr +0.048; bs4 4× slower for no gain | Standard choice |
| `sampler` (fill) | **zero** | zero, mean, noise, noisy_linear | Zero fill is deterministic (MC=1) and matches H-Sets baseline | Enables apples-to-apples vs H-Sets |
| `signed` | **abs** | abs, signed | Within noise on CUB | Matches H-Sets' non-negative attribution |
| `K_sets`, `K_blocks` | **1** | (not meaningful; zero fill is deterministic) | — | MC budget for stochastic fills only |
| `num_samples` | **1000** per seed | — | — | Matches H-Sets protocol |
| `sample_seed` | **42, 43, 44, 45, 46** | — | 5 independent runs | Std across seeds only (zero-fill is deterministic) |
| `set_uniform` | False | True, False | Enables Sparse_ITF construction (no coarse path; mask-constant on top-B) | Apples-to-apples with H-Sets |

When `--set_uniform` is passed, `coarse_size`, `block_size`, and `hierarchy` are silently ignored — the pipeline returns before those code paths. This is correct behavior; see *Method → Sparse_ITF*.

**Consequence of the locks:** the pipeline is fully deterministic given a pool.
All reported std values **reflect only variation between the 5 sample pools**,
not attribution noise — this is a strict improvement over stochastic methods.

---

## Repository Structure

```
.
├── configs/default.py                 # normalization, hyperparams, paths
├── hsets/                             # (only if run_itf.py imports it)
│   ├── hessian_idg.py                 # provides _get_sam() lazy SAM loader
│   ├── attributions.py
│   └── modify_relu.py
├── scripts/
│   ├── check_accuracy.py              # load_*_model, val_transform(dataset=...)
│   ├── hsets_metrics.py               # gini, road_aopc, faithfulness_corr, to_2d
│   └── sample_correct.py              # --num_iterations, --seed
├── run_itf.py                         # single-config runner
├── run_sparse_itf.py                  # single-config runner (--set_uniform enables Sparse_ITF)
├── run_itf_5runs.sh                   # 4-GPU driver: 5 seeds × 8 configs
├── aggregate_itf_5runs.py             # reads JSON results, prints table
├── requirements.txt
├── datasets/                          # <- EMPTY; see Download section
├── checkpoints/                       # <- EMPTY; see Download section
├── sam_vit_h_4b8939.pth               # <- EMPTY; see Download section
├── cache/                             # small caches shipped (see .gitignore)
│   ├── {ds}_{model}/samples_seed*.json   # 40 files, ~2 MB
│   ├── itf_sam/                          # ~40 MB, SAM masks per image
│   ├── itf_{ds}_{model}_seed*_*.json     # 40 result JSONs
│   ├── itf_5runs/summary.json            # aggregated results
│   └── itf_ablations/                    # optional coarse_size scan JSONs
└── logs/itf_5runs_*.log                  # 45 logs, documentary
```

---

## Setup

### 1. Create environment

```bash
conda create -n hsets python=3.10 -y
conda activate hsets

# PyTorch FIRST (must match CUDA — change cu118 to your version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
              -c pytorch -c nvidia -y

# Everything else
pip install -r requirements.txt
```

### 2. Set environment variable

```bash
export SAM_CHECKPOINT=$HOME/checkpoints/sam_vit_h_4b8939.pth
echo 'export SAM_CHECKPOINT=$HOME/checkpoints/sam_vit_h_4b8939.pth' >> ~/.bashrc
```

### 3. Verify

```bash
python -c "import torch, numpy, captum, quantus, cv2; from segment_anything import sam_model_registry; \
print(f'torch={torch.__version__}  numpy={numpy.__version__}  cuda={torch.cuda.is_available()}')"
```

Requires `numpy < 2.0`. If you see `numpy 2.x`, run `pip install 'numpy<2.0'`.

---

## Downloading Datasets

`datasets/` is empty in the repository. You must download both datasets.

### ImageNet (ILSVRC2012 validation set)

The official site (`image-net.org`) is unreliable. Use the SJTU mirror.

```bash
mkdir -p datasets/imagenet/val
cd datasets/imagenet

# ~6.7 GB
wget https://medialab.sjtu.edu.cn/files/CoFAI-share/data/ILSVRC2012_img_val.tar

# Alternative: Academic Torrents
#   https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5

tar -xf ILSVRC2012_img_val.tar -C val

# Reorganize 50,000 flat files into 1,000 WordNet-ID class folders
cd val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ../..

# Verify
find datasets/imagenet/val -name '*.JPEG' | wc -l       # must equal 50000
find datasets/imagenet/val -maxdepth 1 -type d | wc -l  # must equal 1001 (1000 classes + val itself)
ls datasets/imagenet/val/n01440764/ | head -1           # must show a JPEG
```

Delete the tar after extraction to save ~6.7 GB: `rm datasets/imagenet/ILSVRC2012_img_val.tar`

### CUB-200-2011

Caltech's direct download works today.

```bash
mkdir -p datasets/cub
cd datasets

# ~1.2 GB
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
mv CUB_200_2011 cub_raw

# Use the official test split (5,794 images) as `val/`
python - <<'PY'
import os, shutil

SRC = "datasets/cub_raw"
DST = "datasets/cub/val"

id2path, id2train = {}, {}
with open(os.path.join(SRC, "images.txt")) as f:
    for line in f:
        i, p = line.strip().split(" ", 1); id2path[int(i)] = p
with open(os.path.join(SRC, "train_test_split.txt")) as f:
    for line in f:
        i, t = line.strip().split(); id2train[int(i)] = int(t)

n = 0
for i, is_train in id2train.items():
    if is_train:  # keep only test split
        continue
    rel = id2path[i]
    dst = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(SRC, "images", rel), dst)
    n += 1
print(f"Copied {n} images to {DST} (expect 5794)")
PY

# Verify
find datasets/cub/val -name '*.jpg' | wc -l        # must equal 5794
find datasets/cub/val -maxdepth 1 -type d | wc -l  # must equal 201 (200 classes + val)
```

Delete the raw folder after extraction: `rm -rf datasets/cub_raw datasets/CUB_200_2011.tgz`

---

## Downloading SAM and Model Checkpoints

### SAM ViT-H

```bash
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# ~2.4 GB
```

Confirm `$SAM_CHECKPOINT` points at it.

### Model Checkpoints

- **ImageNet**: torchvision `pretrained=True` weights **auto-download** on first call. ~500 MB per model, cached under `~/.cache/torch/`.
- **CUB**: 4 fine-tuned checkpoints needed. Obtainable via:
  - **Release assets** on the GitHub release page for this repo, or
  - Fine-tuning yourself using `scripts/finetune_cub.py` (Recipe C: SGD + differential LR, 50 epochs), or
  - Following the paper's training recipe.

Expected accuracy after fine-tuning:
- VGG16 ≈ 71–72%, ResNet101 ≈ 80–81%, DenseNet121 ≈ 80–81%, MobileNetV3 ≈ 75–76%.

Place under `checkpoints/cub_{vgg16,resnet101,densenet121,mobilenet_v3_large}.pth`.

---

## Running the Pipeline

### Quick check — 20 samples, ~3 min

Verify the environment and pipeline work.

```bash
cd ~/H-Sets
conda activate hsets
export SAM_CHECKPOINT=$HOME/checkpoints/sam_vit_h_4b8939.pth

CUDA_VISIBLE_DEVICES=0 python run_itf.py \
    --dataset cub --model vgg16 --num_samples 20 \
    --hierarchy two-level --coarse_size 32 --top_B 5 --block_size 8 \
    --seed 42 --tag smoke_test
```

For the sparse variant, the only change is `--set_uniform`:

```bash
CUDA_VISIBLE_DEVICES=0 python run_sparse_itf.py \
    --dataset cub --model vgg16 --num_samples 20 \
    --set_uniform --top_B 5 \
    --seed 42 --tag smoke_test_setuniform
```

Output schema is identical to the two-level run. Expect a **higher Gini** (fewer nonzero pixels) and a **slightly lower ROAD** (sparse maps cannot rank pixels outside the masks) — this is the structural cost, see *Method → Sparse_ITF*.

Expected output:
```
  [  5/ 20] gini=0.9xxx road=0.xxxx corr=0.xxxx  (X.Xs/img, ETA X.Xmin)
  [ 20/ 20] ...
  computed 20/20 (skipped 0 all-zero maps)

ITF cub/vgg16  tag=smoke_test  N=20
============================================================
  gini    +0.9xxx ± 0.xxxx
  road    +0.6xxx ± 0.xxxx
  corr    +0.0xxx ± 0.xxxx
Saved cache/itf_cub_vgg16_smoke_test.json
```

### Get the results instantly (~2 s, no GPU needed)

If `cache/itf_*_seed*_*.json` are present (shipped with the repo):

```bash
python aggregate_itf_5runs.py
```

Prints the full results table from this README.

### Reproduce the full 5-run sweep (~12 h on 4 GPUs)

Only needed if you want to regenerate from scratch.

```bash
cd ~/H-Sets
conda activate hsets
export SAM_CHECKPOINT=$HOME/checkpoints/sam_vit_h_4b8939.pth
mkdir -p logs cache/itf_5runs
pkill -f "run_itf.py" 2>/dev/null; sleep 2

nohup bash -c "IMG_CS=64 CUB_CS=32 ./run_itf_5runs.sh" \
    > logs/itf_5runs_master.log 2>&1 &
echo $! > logs/itf_5runs.pid
```

For the **Sparse_ITF** 5-run sweep (~8 h on 4 GPUs; Sparse_ITF is ~10× cheaper per image, so the total is dominated by the 4-model parallel schedule rather than by Sparse_ITF's own cost):

```bash
mkdir -p logs
for spec in "0:vgg16" "1:resnet101" "2:densenet121" "3:mobilenet_v3_large"; do
  gpu="${spec%%:*}"; m="${spec##*:}"
  ( for ds in imagenet cub; do for seed in 42 43 44 45 46; do
      out="cache/itf_${ds}_${m}_seed${seed}_setuniform.json"
      [ -f "$out" ] && { echo "[GPU$gpu] SKIP $ds/$m/seed$seed"; continue; }
      echo "[GPU$gpu] START $ds/$m/seed$seed"
      CUDA_VISIBLE_DEVICES=$gpu python run_itf.py \
        --dataset "$ds" --model "$m" --num_samples 1000 \
        --set_uniform --top_B 5 \
        --seed "$seed" --tag "seed${seed}_setuniform" \
        > "logs/itf_${ds}_${m}_seed${seed}_setuniform.log" 2>&1
      echo "[GPU$gpu] END $ds/$m/seed$seed exit=$?"
  done; done ) > "logs/setuniform_gpu${gpu}.log" 2>&1 &
done
wait
echo "=== ALL 40 CONFIGS COMPLETE $(date) ==="
```

Aggregate the 5 seed-level means per (dataset, model) — **do not** report the per-sample std inside a single JSON; the std you report in the table must be across the 5 pool means:

```bash
python - <<'PY'
import glob, json, statistics as st
from collections import defaultdict
rows = defaultdict(dict)
for f in glob.glob("cache/itf_*_seed*_setuniform.json"):
    d = json.load(open(f)); rows[(d["dataset"], d["model"])][d["seed"]] = d
for (ds, m), by_seed in sorted(rows.items()):
    for k in ("gini", "road", "corr"):
        vals = [by_seed[s][k]["mean"] for s in sorted(by_seed)]
        print(f"{ds:8s} {m:20s} {k:5s}  mean={st.mean(vals):+.4f}  std(5 pools)={st.stdev(vals):.4f}")
PY
```

**With shipped caches: ~10–12 h.**
**Without caches (regenerate all):** add ~6 h for `sample_correct.py` and SAM cold — **~16–18 h total**.

Monitor progress:

```bash
for f in logs/itf_5runs_gpu*.log; do
    gpu=$(basename "$f" | sed 's/itf_5runs_gpu//;s/.log//')
    done_n=$(grep -c "END seed" "$f" 2>/dev/null)
    current=$(grep "START seed" "$f" 2>/dev/null | tail -1 | grep -oE "seed[0-9]+_[a-z]+_[a-z0-9_]+")
    printf "  GPU%s: %2d/10 done, current=%s\n" "$gpu" "${done_n:-0}" "${current:-launching}"
done
```

After completion:

```bash
python aggregate_itf_5runs.py
```

### Regenerate sample pools (only if missing)

```bash
for ds in imagenet cub; do
  for m in vgg16 resnet101 densenet121 mobilenet_v3_large; do
    n=$(ls cache/${ds}_${m}/samples_seed*.json 2>/dev/null | wc -l)
    if [ "$n" -lt 5 ]; then
      python scripts/sample_correct.py --dataset $ds --model $m --num_iterations 5 --seed 42
    fi
  done
done
```

~25 min, produces 40 files.

---

## Ablations

Pre-computed ablations from the hyperparameter scan at N=20/50 (in `cache/itf_ablations/`):

| Ablation | Values | Finding |
|---|---|---|
| `coarse_size` | 8, 32, 48, 64, 96, 128, 224 | Gini peaks at 48–64; collapses at 224 (single block) |
| `top_B` | 1, 5, 9 | tb1 drifts corr +0.027; tb9 plateaus |
| `block_size` | 4, 8, 16 | bs16 drifts corr +0.048; bs4 4× slower for no gain |
| `signed` | abs, signed | Within noise on CUB |

To reproduce any ablation:

```bash
CUDA_VISIBLE_DEVICES=0 python run_itf.py \
    --dataset cub --model vgg16 --num_samples 20 \
    --coarse_size 48 --tag ablation_cs48
```

---
For ablations on **Sparse_ITF**, the only parameters that matter are `--top_B` and `--signed` (`--coarse_size`, `--block_size`, and `--hierarchy` are ignored under `--set_uniform`). Example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_itf.py \
    --dataset cub --model vgg16 --num_samples 200 \
    --set_uniform --top_B 3 \
    --seed 42 --tag ablation_setuniform_tb3
```

The `top_B` sweep on Sparse_ITF is what backs the "matched sparsity" claim in the Results table: at `B=1–3`, Sparse_ITF's ImageNet Gini rises to H-Sets' level (0.95–0.97) while its ROAD stays above H-Sets' 0.26–0.38.


## Troubleshooting

### `AssertionError: attribution vector all zero` (Quantus)

A sample produced a fully zero attribution map (small blocks + confident model ⇒ φ ≈ 0 for all blocks). **Not fatal.** The driver patches skip these and print `skipped N all-zero maps`. If skip rate exceeds 5% on a config, coarse block size is too small — bump it.

### `FileNotFoundError: cache/imagenet_vgg16/samples_seed42.json`

Run `python scripts/sample_correct.py --dataset imagenet --model vgg16 --num_iterations 5 --seed 42`.

### `FileNotFoundError: sam_vit_h_4b8939.pth`

`export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth` is not set. Set it.

### `RuntimeError: derivative for aten::hardsigmoid_backward is not implemented`

Only applies to **H-Sets** (Hessian computation), not ITF. ITF uses only forward passes and is immune.

### ImageNet accuracy is 65% not 71%

Wrong normalization. Ensure `configs/default.py` has `normalize_mean["imagenet"] = [0.485, 0.456, 0.406]`, and that `val_transform(dataset="imagenet")` is called with the `dataset=` argument (not the default).

### `RuntimeError: expected scalar type Float but found Half`

`--fp16` crashes on the VGG16 classifier. **fp16 is not supported**; use fp32 (default).

### Full-run hangs on ResNet101

**Does not apply to ITF** — ITF is forward-only. Only H-Sets suffers the `retain_graph` / Hessian BFS bug on skip-connection architectures.

**Sparse_ITF returns an all-zero map.**

Occurs when every SAM proposal scores φ(S) ≤ 0 on the predicted class — common on very confident models with small proposals. The driver counts these as skips; if the skip rate exceeds ~5% on a config, either raise `top_B` or lower `pred_iou_thresh` in `get_sam_proposals`. Do **not** switch to `--signed` — it changes the ranking key and breaks comparability with the unsigned H-Sets baseline.
---

## Citation

If you use this code, please cite both the ITF method and the base H-Sets framework it is benchmarked against:

```bibtex
@misc{itf2026,
  title  = {ITF: Conditional-Information Attribution for Image Classifiers},
  year   = {2026},
  note   = {Reference implementation}
}

@inproceedings{hsets2026,
  title     = {H-Sets: Hessian-Guided Discovery of Set-Level Feature Interactions in Image Classifiers},
  author    = {Mehrotra, Ayushi and Bhusal, Dipkamal and Clifford, Michael and Rastogi, Nidhi},
  booktitle = {CVPR},
  year      = {2026}
}

@inproceedings{road2022,
  title     = {A Consistent and Efficient Evaluation Strategy for Attribution Methods},
  author    = {Rong, Yao and Leemann, Tobias and Borisov, Vadim and Kasneci, Gjergji and Kasneci, Enkelejda},
  booktitle = {ICML},
  year      = {2022}
}
```

---

## License

MIT
