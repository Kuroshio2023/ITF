# scripts/sample_correct.py
"""Pick N correctly-classified samples per (dataset, model). Idempotent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from configs.default import CONFIG
from scripts.check_accuracy import (
    load_imagenet_model, load_cub_model, val_transform,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["imagenet", "cub"])
    ap.add_argument("--model", required=True,
                    choices=["vgg16", "resnet101", "densenet121", "mobilenet_v3_large"])
    #ap.add_argument("--num_samples", type=int, default=CONFIG["num_samples"])
    #args = ap.parse_args()

    #set_seed(CONFIG["sample_seed"])

    ap.add_argument("--num_samples", type=int, default=CONFIG["num_samples"])
    ap.add_argument("--num_iterations", type=int, default=1,
                    help="Save N pools: samples_seed<s>.json for s=seed..seed+N-1")
    ap.add_argument("--seed", type=int, default=CONFIG["sample_seed"])
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "imagenet":
        ds = ImageFolder(CONFIG["imagenet_val_dir"], transform=val_transform(dataset="imagenet"))
        model = load_imagenet_model(args.model).to(device).eval()
    else:
        ds = ImageFolder(CONFIG["cub_val_dir"], transform=val_transform(dataset="cub"))
        model, _ = load_cub_model(args.model)
        model = model.to(device).eval()

    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

    correct = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            m = pred.eq(y).cpu().numpy()
            base = i * loader.batch_size
            correct.extend((base + np.where(m)[0]).tolist())

    print(f"{args.dataset}/{args.model}: {len(correct)}/{len(ds)} correct "
          f"({len(correct)/len(ds)*100:.2f}%)")
    if len(correct) < args.num_samples:
        raise RuntimeError(f"Only {len(correct)} correct; need {args.num_samples}")

   # rng = np.random.default_rng(args.seed)
   # sampled = rng.choice(correct, args.num_samples, replace=False).tolist()

   # out_dir = os.path.join(CONFIG["cache_dir"], f"{args.dataset}_{args.model}")
   # os.makedirs(out_dir, exist_ok=True)
   # out = os.path.join(out_dir, "samples.json")
   # with open(out, "w") as f:
   #     json.dump({
   #         "dataset": args.dataset, "model": args.model,
   #         "num_correct": len(correct), "num_total": len(ds),
   #         "num_sampled": len(sampled), "seed": CONFIG["sample_seed"],
   #         "samples": sampled,
   #     }, f, indent=2)
   # print(f"Saved {out}")

    out_dir = os.path.join(CONFIG["cache_dir"], f"{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    base_seed = args.seed if hasattr(args, "seed") else CONFIG["sample_seed"]
    for it in range(args.num_iterations):
        seed_it = base_seed + it
        rng = np.random.default_rng(seed_it)
        sampled = rng.choice(correct, args.num_samples, replace=False).tolist()

        out_seeded = os.path.join(out_dir, f"samples_seed{seed_it}.json")
        with open(out_seeded, "w") as f:
            json.dump({
                "dataset": args.dataset, "model": args.model,
                "num_correct": len(correct), "num_total": len(ds),
                "num_sampled": len(sampled), "seed": seed_it,
                "samples": sampled,
            }, f, indent=2)
        print(f"Saved {out_seeded}  ({len(sampled)} samples, seed={seed_it})")

        if it == 0:
            import shutil
            shutil.copy(out_seeded, os.path.join(out_dir, "samples.json"))

if __name__ == "__main__":
    main()
