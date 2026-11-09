# scripts/hsets_metrics.py
"""Compute Gini / ROAD_AOPC / Faithfulness-Corr for H-Sets attributions.
Attributions are cached per (dataset, model, sample) and reused across metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantus.metrics.faithfulness.road import ROAD

from quantus.functions.perturb_func import noisy_linear_imputation

from quantus.helpers.model.model_interface import ModelInterface
import json, random, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from configs.default import CONFIG
from scripts.check_accuracy import (
    load_imagenet_model, load_cub_model, val_transform,
)
from hsets.hessian_idg import HessianIntegratedDirectionalGradients


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_2d(attr_chw):
    """[C,H,W] -> [H,W] by summing |attr| across channels, then min-max norm to [0,1]."""
    a = np.abs(attr_chw).sum(axis=0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


# ---------- Metric 1: Gini sparsity (paper Eq. 12) ----------
def gini(attr_flat):
    v = np.sort(np.abs(attr_flat).astype(np.float64))
    n = v.size
    if v.sum() == 0:
        return 0.0
    k = np.arange(1, n + 1, dtype=np.float64)
    return float(1.0 - 2.0 * np.sum(v / v.sum() * (n - k + 0.5) / n))


# ---------- Metric 2: ROAD AOPC (paper §I.2, Rong et al. 2022) ----------

def _class_logit(model, x_batched, y):
    """Paper's faithfulness correlation uses predicted LOGITS, not softmax."""
    with torch.no_grad():
        return model(x_batched)[0, y].item()


def _class_prob(model, x_batched, y):
    with torch.no_grad():
        return F.softmax(model(x_batched), dim=1)[0, y].item()






def road_aopc(model, x, y, attr_hw, device="cuda:0"):
    """ROAD accuracy-AOPC with Noisy Linear Imputation (Rong et al. 2022, Quantus).
    Reports mean over percentages of (1 - accuracy), matching the paper's protocol.
    """

    metric = ROAD(
        percentages=[0,20, 40, 60, 80, 100],
        noise=0.01,
        abs=False,
        normalise=False,
        perturb_func=noisy_linear_imputation,
    )
    scores = metric(
        model=model,
        x_batch=x.unsqueeze(0).cpu().numpy(),
        y_batch=np.array([y]),
        a_batch=attr_hw[None],
        device=device,
    )
    # scores is {percentage: mean_accuracy_at_that_percentage} for this sample
    # Per-sample AOPC = mean over percentages of (1 - acc)
    accs = np.array(list(scores.values()), dtype=float)
    return float((1.0 - accs).mean()) 
    



# ---------- Metric 3: Faithfulness correlation (paper §J.2) ----------
def faithfulness_corr(model, x, y, attr_chw,
                      num_runs=30, subset_frac=0.1, seed=123, device="cuda:0"):
    x = x.to(device)
    flat = np.abs(attr_chw).flatten().astype(np.float64)
    n_feat = flat.size
    sub_sz = max(1, int(subset_frac * n_feat))
    rng = np.random.default_rng(seed)

    logit_orig = _class_logit(model, x.unsqueeze(0), y)

    attrs, deltas = [], []
    for _ in range(num_runs):
        idx = rng.choice(n_feat, sub_sz, replace=False)
        x_flat = x.flatten().clone()
        x_flat[idx] = 0.0
        x_pert = x_flat.reshape(x.shape)
        logit_pert = _class_logit(model, x_pert.unsqueeze(0), y)
        attrs.append(flat[idx].mean())
        deltas.append(logit_orig - logit_pert)

    if np.std(attrs) < 1e-12 or np.std(deltas) < 1e-12:
        return 0.0
    return float(np.corrcoef(attrs, deltas)[0, 1])


def _clean_attr(a):
    """Force attribution to a clean float32 [C,H,W] ndarray.
    Handles legacy object-dtype caches written by the unpatched code."""
    a = np.asarray(a)
    # 0-d object array wrapping a single array
    if a.dtype == object and a.ndim == 0:
        a = a.item()
    # 1-element object array
    if isinstance(a, np.ndarray) and a.dtype == object and a.size == 1:
        a = a.flatten()[0]
    # N-element object array (your exact case: (3,224,224) of Python floats)
    if isinstance(a, np.ndarray) and a.dtype == object:
        return np.array(a.tolist(), dtype=np.float32)
    return np.asarray(a, dtype=np.float32)


def _load_attr(path):
    """Load attribution, tolerating legacy object-dtype caches."""
    try:
        a = np.load(path)                       # clean path
    except ValueError:
        a = np.load(path, allow_pickle=True)    # legacy object path
    return _clean_attr(a)



# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["imagenet", "cub"])
    ap.add_argument("--model", required=True,
                    choices=["vgg16", "resnet101", "densenet121", "mobilenet_v3_large"])
    ap.add_argument("--metric", required=True,
                    choices=["gini", "road", "corr", "all"])
    ap.add_argument("--num_samples", type=int, default=CONFIG["num_samples"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--regen", action="store_true",
                    help="ignore attribution cache and regenerate")
   # ap.add_argument("--road_step", type=int, default=3072,
    #                help="pixels removed per ROAD step (~2%% of 224*224*3)")
    #ap.add_argument("--road_steps", type=int, default=49,
     #               help="number of ROAD removal steps")
    args = ap.parse_args()

    set_seed(args.seed)
    CONFIG["dataset"] = args.dataset   # ensure downstream val_transform reads correctly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cache_dir = os.path.join(CONFIG["cache_dir"], f"{args.dataset}_{args.model}")
    attr_dir  = os.path.join(cache_dir, "attrs")
    os.makedirs(attr_dir, exist_ok=True)

    samp_path = os.path.join(cache_dir, "samples.json")
    if not os.path.isfile(samp_path):
        raise FileNotFoundError(
            f"{samp_path} missing — run scripts/sample_correct.py first")
    sample_idx = json.load(open(samp_path))["samples"][:args.num_samples]

    if args.dataset == "imagenet":
        ds = ImageFolder(CONFIG["imagenet_val_dir"], transform=val_transform(dataset="imagenet"))
        model = load_imagenet_model(args.model).to(device).eval()
        max_features = CONFIG["max_elements_imagenet"]
    else:
        ds = ImageFolder(CONFIG["cub_val_dir"], transform=val_transform(dataset="cub"))
        model, _ = load_cub_model(args.model)
        model = model.to(device).eval()
        max_features = CONFIG["max_elements_cub"]

    explainer = HessianIntegratedDirectionalGradients(model)
    explainer.max_features = max_features

    metrics_needed = ["gini", "road", "corr"] if args.metric == "all" else [args.metric]
    scores = {m: [] for m in metrics_needed}

    t0 = time.time()
    for k, idx in enumerate(sample_idx):
        x, y = ds[idx]
        x_dev = x.unsqueeze(0).to(device)
        attr_path = os.path.join(attr_dir, f"attr_{k:05d}.npy")

        #if os.path.isfile(attr_path) and not args.regen:
         #   attr = np.load(attr_path)
        #else:
         #   set_seed(args.seed + k)                 # per-sample determinism
          #  tk = time.time()
           # attr = explainer.attribute(
            #    x_dev,
             #   threshold=CONFIG["threshold"],
              #  steps=CONFIG["path_steps"],
               # k=CONFIG["k_sets"],
                #max_samples=CONFIG["mc_samples"],
          #  )[0]
           # np.save(attr_path, attr)


        if os.path.isfile(attr_path) and not args.regen:
            attr = _load_attr(attr_path)
            np.save(attr_path, attr)   # migrate legacy object-dtype → clean float32
        else:
            set_seed(args.seed + k)
            tk = time.time()
            attr = explainer.attribute(
                x_dev,
                threshold=CONFIG["threshold"],
                steps=CONFIG["path_steps"],
                k=CONFIG["k_sets"],
                max_samples=CONFIG["mc_samples"],
            )[0]
            attr = _clean_attr(attr)
            np.save(attr_path, attr)


            print(f"[{k+1:4d}/{len(sample_idx)}] attr in {time.time()-tk:5.1f}s")

        attr_hw = to_2d(attr)

        if "gini" in metrics_needed:
            scores["gini"].append(gini(attr_hw.ravel()))
        if "road" in metrics_needed:
            scores["road"].append(road_aopc(model, x, y, attr_hw, device=device))
        
        if "corr" in metrics_needed:
            scores["corr"].append(faithfulness_corr(model, x, y, attr, device=device))

        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  progress {k+1}/{len(sample_idx)}  elapsed={el/60:.1f}min  "
                  f"ETA={el/(k+1)*(len(sample_idx)-k-1)/60:.1f}min")

    # ---- Report ----
    print("\n" + "=" * 66)
    print(f"Dataset: {args.dataset}   Model: {args.model}   N = {len(sample_idx)}")
    print("=" * 66)
    report = {"dataset": args.dataset, "model": args.model,
              "n": len(sample_idx), "seed": args.seed, "metrics": {}}
    for m in metrics_needed:
        a = np.array(scores[m])
        print(f"  {m:6s}  mean = {a.mean():+.4f}   std = {a.std():.4f}")
        report["metrics"][m] = {"mean": float(a.mean()), "std": float(a.std())}

    out = os.path.join(cache_dir, f"metrics_{args.metric}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nSaved {out}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
