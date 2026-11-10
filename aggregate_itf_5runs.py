#!/usr/bin/env python3
"""Aggregate ITF and Sparse_ITF 5-run results vs H-Sets (user repro + paper values).

Reads
-----
  cache/itf_{ds}_{m}_seed{seed}_{ds}_{m}.json      # ITF (dense, two-level)
  cache/itf_{ds}_{m}_seed{seed}_setuniform.json    # Sparse_ITF (set-uniform)
  cache/metrics_runs/summary_5runs.json            # H-Sets user repro (optional)

Writes
------
  cache/itf_5runs/summary.json

Notes
-----
* mean ± std is across the 5 seed-level pool means (ddof=1). This is
  the value to report in the results table.
* The 'std' inside each ITF JSON is np.std over per-sample scores within
  a single pool (~sqrt(N) ≈ 31.6× larger than across-pool std). This
  script does NOT report that number.
* H-Sets paper stds are the 5-run stds from the H-Sets paper
  (Table 1 for Gini, Table 2 for ROAD, Table 13 for FC). They are not
  pooled the same way as ours, so the comparison is directional, not
  a paired test. See paper for paired tests.
"""
import json, os
import numpy as np

SEEDS = [42, 43, 44, 45, 46]
CONFIGS = [
    ("imagenet","vgg16"), ("cub","vgg16"),
    ("imagenet","resnet101"), ("cub","resnet101"),
    ("imagenet","densenet121"), ("cub","densenet121"),
    ("imagenet","mobilenet_v3_large"), ("cub","mobilenet_v3_large"),
]

# H-Sets paper-reported values (mean, std). Source: H-Sets paper
# Tables 1 (Gini), 2 (ROAD_AOPC), 13 (FC).
PAPER_HSETS = {
    ("cub","vgg16"):              {"gini":(0.93,0.02),"road":(0.65,0.00),"corr":(0.124,0.137)},
    ("cub","resnet101"):          {"gini":(0.89,0.04),"road":(0.63,0.01),"corr":(0.123,0.145)},
    ("cub","densenet121"):        {"gini":(0.89,0.05),"road":(0.65,0.00),"corr":(0.029,0.126)},
    ("cub","mobilenet_v3_large"): {"gini":(0.89,0.05),"road":(0.60,0.00),"corr":(0.128,0.119)},
    ("imagenet","vgg16"):         {"gini":(0.95,0.01),"road":(0.34,0.01),"corr":(0.009,0.122)},
    ("imagenet","resnet101"):     {"gini":(0.98,0.01),"road":(0.26,0.02),"corr":(0.251,0.081)},
    ("imagenet","densenet121"):   {"gini":(0.94,0.01),"road":(0.38,0.01),"corr":(0.034,0.135)},
    ("imagenet","mobilenet_v3_large"): {"gini":(0.95,0.01),"road":(0.37,0.01),"corr":(0.021,0.143)},
}


def load_user_hsets():
    out = {}
    p = "cache/metrics_runs/summary_5runs.json"
    if not os.path.isfile(p):
        return out
    for cfg, metrics in json.load(open(p)).items():
        if "/" not in cfg:
            continue
        ds, m = cfg.split("/")
        out[(ds, m)] = {k: metrics[k]["mean"] for k in ("gini","road","corr") if k in metrics}
    return out


def _load_variant(pattern_fn):
    """Return {ds: {m: {metric: [per-seed pool means]}}}, missing → empty lists."""
    vals = {}
    for ds, m in CONFIGS:
        vals[(ds, m)] = {"gini": [], "road": [], "corr": []}
        for seed in SEEDS:
            p = pattern_fn(ds, m, seed)
            if not os.path.isfile(p):
                continue
            d = json.load(open(p))
            for k in vals[(ds, m)]:
                if k in d:
                    vals[(ds, m)][k].append(d[k]["mean"])
    return vals


def load_itf():
    return _load_variant(lambda ds, m, s: f"cache/itf_{ds}_{m}_seed{s}_{ds}_{m}.json")


def load_sparse_itf():
    return _load_variant(lambda ds, m, s: f"cache/itf_{ds}_{m}_seed{s}_setuniform.json")


def mean_std(vals):
    if not vals:
        return (None, None)
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (mu, sd)


def fmt_cell(mu, sd, width=20):
    if mu is None:
        return f"{'—':>{width}}"
    return f"{mu:+.4f} ± {sd:.4f}".rjust(width)


def fmt_plain(mu, sd, width=16):
    if mu is None:
        return f"{'—':>{width}}"
    return f"{mu:+.4f}".rjust(width)


def bold(s):
    return f"**{s.strip()}**"


def main():
    user_hsets = load_user_hsets()
    itf     = load_itf()
    sitf    = load_sparse_itf()

    results = {}
    print("\nITF + Sparse_ITF 5-run aggregate   (mean ± std across 5 seeds × 1000 samples)\n")
    header = (f"{'config':<26} {'metric':<6} "
              f"{'ITF':>20} {'Sparse_ITF':>20} "
              f"{'H-Sets (user)':>16} {'H-Sets (paper)':>18}  "
              f"{'best':>12}")
    print(header)
    print("-" * len(header))

    for ds, m in CONFIGS:
        have_any = bool(itf[(ds,m)]["gini"] or sitf[(ds,m)]["gini"])
        if not have_any:
            print(f"{ds}/{m:<20}  no results")
            continue

        results[f"{ds}/{m}"] = {}
        for k in ("gini","road","corr"):
            # Our numbers
            itf_mu, itf_sd = mean_std(itf[(ds,m)][k])
            sitf_mu, sitf_sd = mean_std(sitf[(ds,m)][k])

            # H-Sets user repro
            u = user_hsets.get((ds,m), {}).get(k)

            # H-Sets paper (mean, std)
            paper = PAPER_HSETS.get((ds,m), {}).get(k)
            paper_mu = paper[0] if paper else None
            paper_sd = paper[1] if paper else None

            # Pick best variant per metric (highest mean wins; skip missing)
            candidates = [
                ("ITF", itf_mu),
                ("Sparse_ITF", sitf_mu),
                ("H-Sets(user)", u),
                ("H-Sets(paper)", paper_mu),
            ]
            best_name, best_mu = None, -np.inf
            for name, mu in candidates:
                if mu is not None and mu > best_mu:
                    best_name, best_mu = name, mu

            # Build row (bold the best column per metric; mark only ours)
            cells = []
            cells.append(fmt_cell(itf_mu, itf_sd))
            cells.append(fmt_cell(sitf_mu, sitf_sd))
            cells.append(fmt_plain(u, None))
            cells.append(f"{paper_mu:+.4f} ± {paper_sd:.3f}".rjust(18)
                         if paper_mu is not None else f"{'—':>18}")

            # Bold only the winning *of ours* (ITF/Sparse_ITF) to keep
            # the table focused on our variants; H-Sets best is shown by
            # the reference column but not bolded.
            our_best = max(
                [(name, mu) for name, mu in [("ITF", itf_mu), ("Sparse_ITF", sitf_mu)]
                 if mu is not None],
                key=lambda x: x[1],
                default=(None, None),
            )[0]
            if our_best == "ITF" and itf_mu is not None:
                cells[0] = bold(cells[0])
            elif our_best == "Sparse_ITF" and sitf_mu is not None:
                cells[1] = bold(cells[1])

            delta_vs_user = (itf_mu - u) if (itf_mu is not None and u is not None) else None
            best_str = best_name or "—"

            print(f"{ds}/{m:<22} {k:<6} {cells[0]} {cells[1]} {cells[2]} {cells[3]}  "
                  f"{best_str:>12}")

            results[f"{ds}/{m}"][k] = {
                "itf":         {"mean": itf_mu,  "std": itf_sd,
                                "n": len(itf[(ds,m)][k])},
                "sparse_itf":  {"mean": sitf_mu, "std": sitf_sd,
                                "n": len(sitf[(ds,m)][k])},
                "h_sets_user": u,
                "h_sets_paper": {"mean": paper_mu, "std": paper_sd},
                "best": best_name,
                "delta_itf_vs_user": delta_vs_user,
            }
        print()

    os.makedirs("cache/itf_5runs", exist_ok=True)
    json.dump(results, open("cache/itf_5runs/summary.json","w"), indent=2)
    print("Saved cache/itf_5runs/summary.json")

    # Warnings
    missing_sitf = [f"{ds}/{m}" for ds, m in CONFIGS if not sitf[(ds,m)]["gini"]]
    if missing_sitf:
        print(f"\n  [warn] Sparse_ITF results missing for: {', '.join(missing_sitf)}")
        print("         Run the Sparse_ITF sweep (README §'Sparse_ITF 5-run sweep').")


if __name__ == "__main__":
    main()
