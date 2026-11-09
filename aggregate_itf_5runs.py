#!/usr/bin/env python3
"""Aggregate ITF 5-run results vs H-Sets (user repro + paper values)."""
import json, os
import numpy as np

SEEDS = [42, 43, 44, 45, 46]
CONFIGS = [
    ("imagenet","vgg16"), ("cub","vgg16"),
    ("imagenet","resnet101"), ("cub","resnet101"),
    ("imagenet","densenet121"), ("cub","densenet121"),
    ("imagenet","mobilenet_v3_large"), ("cub","mobilenet_v3_large"),
]
PAPER_HSETS = {
    ("cub","vgg16"):{"gini":0.93,"road":0.65,"corr":0.124},
    ("cub","resnet101"):{"gini":0.89,"road":0.63,"corr":0.123},
    ("cub","densenet121"):{"gini":0.89,"road":0.65,"corr":0.029},
    ("cub","mobilenet_v3_large"):{"gini":0.89,"road":0.60,"corr":0.128},
    ("imagenet","vgg16"):{"gini":0.95,"road":0.34,"corr":0.009},
    ("imagenet","resnet101"):{"gini":0.98,"road":0.26,"corr":0.251},
    ("imagenet","densenet121"):{"gini":0.94,"road":0.38,"corr":0.034},
    ("imagenet","mobilenet_v3_large"):{"gini":0.95,"road":0.37,"corr":0.021},
}

def load_user_hsets():
    out = {}
    p = "cache/metrics_runs/summary_5runs.json"
    if not os.path.isfile(p): return out
    for cfg, metrics in json.load(open(p)).items():
        if "/" not in cfg: continue
        ds, m = cfg.split("/")
        out[(ds, m)] = {k: metrics[k]["mean"] for k in ("gini","road","corr") if k in metrics}
    return out

def load_itf(ds, m):
    vals = {"gini":[], "road":[], "corr":[]}
    for seed in SEEDS:
        p = f"cache/itf_{ds}_{m}_seed{seed}_{ds}_{m}.json"
        if os.path.isfile(p):
            d = json.load(open(p))
            for k in vals:
                if k in d: vals[k].append(d[k]["mean"])
    return vals

def fmt(vals):
    if not vals: return "         —         "
    m = np.mean(vals); s = np.std(vals, ddof=1) if len(vals)>1 else 0.0
    return f"{m:+.4f} ± {s:.4f}"

def fmt1(v): return f"{v:+.4f}" if v is not None else "  —   "

def main():
    user_hsets = load_user_hsets()
    results = {}
    print("\nITF 5-run aggregate   (mean ± std across 5 seeds × 1000 samples)\n")
    print(f"{'config':<26} {'metric':<6} {'ITF':>20} {'H-Sets (user)':>16} "
          f"{'H-Sets (paper)':>16} {'Δ vs user':>12}")
    print("-" * 100)
    for ds, m in CONFIGS:
        itf = load_itf(ds, m)
        if not itf["gini"]:
            print(f"{ds}/{m:<20}  no results"); continue
        results[f"{ds}/{m}"] = {}
        for k in ("gini","road","corr"):
            u = user_hsets.get((ds,m),{}).get(k)
            pv = PAPER_HSETS.get((ds,m),{}).get(k)
            delta = (np.mean(itf[k]) - u) if (u is not None and itf[k]) else None
            results[f"{ds}/{m}"][k] = {
                "itf": {"mean": float(np.mean(itf[k])) if itf[k] else None,
                        "std":  float(np.std(itf[k], ddof=1)) if len(itf[k])>1 else 0.0,
                        "n":    len(itf[k])},
                "h_sets_user": u, "h_sets_paper": pv,
            }
            print(f"{ds}/{m:<22} {k:<6} {fmt(itf[k]):>20} {fmt1(u):>16} {fmt1(pv):>16} "
                  f"{fmt1(delta):>12}")
        print()
    os.makedirs("cache/itf_5runs", exist_ok=True)
    json.dump(results, open("cache/itf_5runs/summary.json","w"), indent=2)
    print("Saved cache/itf_5runs/summary.json")

if __name__ == "__main__":
    main()
