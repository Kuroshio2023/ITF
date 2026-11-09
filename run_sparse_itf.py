"""run_itf.py — ITF attribution, apples-to-apples with H-Sets.

Zero-fill is MC-free: phi(S) = log p(y|x) - log p(y | x_S=0, x_-S).
Metrics imported unchanged from scripts.hsets_metrics (same code, same pool).
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

from configs.default import CONFIG
from scripts.check_accuracy import (
    load_imagenet_model, load_cub_model, val_transform,
)
from scripts.hsets_metrics import gini, road_aopc, faithfulness_corr
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# ---------- SAM: loaded once, proposals cached on disk ----------
_SAM = None
def _get_sam():
    global _SAM
    if _SAM is None:
        _SAM = sam_model_registry["vit_h"](
            checkpoint=os.environ["SAM_CHECKPOINT"]
        ).to("cuda")
    return _SAM


def _sam_cache_path(ds, model, idx):
    return f"cache/itf_sam/{ds}_{model}/sample_{idx:04d}.npz"


def get_sam_proposals(x, ds, model_name, idx):
    path = _sam_cache_path(ds, model_name, idx)
    if os.path.isfile(path):
        arr = np.load(path)["masks"]
        return [arr[i] for i in range(len(arr))]

    sam = _get_sam()
    gen = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32,
        pred_iou_thresh=0.88, stability_score_thresh=0.9,
        min_mask_region_area=50, crop_n_layers=0,
    )
    mean = np.array(CONFIG["normalize_mean"][ds]).reshape(3, 1, 1)
    std = np.array(CONFIG["normalize_std"][ds]).reshape(3, 1, 1)
    img = x[0].cpu().numpy()
    img = np.transpose(img * std + mean, (1, 2, 0))
    img = np.clip(img, 0, 1) * 255
    img = img.astype("uint8")
    H, W = img.shape[:2]
    masks = gen.generate(img)

    kept = []
    for m in masks:
        seg = m["segmentation"]; area = seg.sum()
        if area < 50 or area > 5000: continue
        border = (seg[0,:].any() or seg[-1,:].any()
                  or seg[:,0].any() or seg[:,-1].any())
        if border and area > 0.4 * H * W: continue
        kept.append(seg)
    kept.sort(key=lambda s: s.sum(), reverse=True)
    final = []
    for seg in kept:
        ok = True
        for s in final:
            if (seg & s).sum() / max((seg | s).sum(), 1) > 0.7:
                ok = False; break
        if ok: final.append(seg)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if final:
        np.savez_compressed(path, masks=np.stack(final))
    else:
        np.savez_compressed(path, masks=np.zeros((0, H, W), dtype=bool))
    return final


# ---------- ITF attributor ----------
class ITFAttributor:
    def __init__(
        self,
        model,
        device="cuda:0",
        coarse_size=32,
        block_size=8,
        top_B=5,
        hierarchy="two-level",
        signed=False,
        batch_size=32,
        set_uniform=False,
    ):
        self.model = model
        self.device = device
        self.cs = coarse_size
        self.bs = block_size
        self.top_B = top_B
        self.hierarchy = hierarchy
        self.signed = signed
        self.B = batch_size
        self.set_uniform = set_uniform
        self.dtype = next(model.parameters()).dtype
        self._HW = None

    @torch.no_grad()
    def _logp_full(self, x, y):
        return F.log_softmax(self.model(x), dim=1)[0, y].item()

    @torch.no_grad()
    def _phi_batch_zero(self, x, y, masks_np, log_p_full):
        x0 = x[0].to(self.dtype)
        phis = []
        for start in range(0, len(masks_np), self.B):
            batch_masks = masks_np[start : start + self.B]
            B = len(batch_masks)
            imputed = x0.unsqueeze(0).repeat(B, 1, 1, 1)
            mask_t = torch.from_numpy(np.stack(batch_masks)).bool().to(self.device)
            keep = (~mask_t).to(self.dtype)
            imputed = imputed * keep.unsqueeze(1)
            logits = self.model(imputed)
            probs = F.softmax(logits, dim=1)[:, y].float().cpu().numpy()
            phis.extend((log_p_full - np.log(probs + 1e-8)).tolist())
        return np.array(phis)

    def _blocks(self, size):
        H, W = self._HW
        masks, centers = [], []
        for i in range(0, H, size):
            for j in range(0, W, size):
                m = np.zeros((H, W), dtype=bool)
                m[i : i + size, j : j + size] = True
                masks.append(m)
                centers.append((i, j))
        return masks, centers

    def attribute(self, x, y, proposals):
        H, W = x.shape[2], x.shape[3]
        self._HW = (H, W)
        log_p_full = self._logp_full(x.to(self.dtype), y)

        # ---- Option D: set-uniform ITF over top-B SAM proposals ----
        if self.set_uniform:
            if not proposals:
                return np.zeros((H, W), dtype=np.float32), {}
            set_phis = self._phi_batch_zero(x, y, proposals, log_p_full)
            rank = set_phis if self.signed else np.abs(set_phis)
            top_idx = np.argsort(rank)[::-1][: self.top_B]
            M = np.zeros((H, W), dtype=np.float32)
            for ti in top_idx:
                M[proposals[ti]] = set_phis[ti]
            return M, {
                "set_scores": [float(set_phis[ti]) for ti in top_idx],
                "n_proposals": len(proposals),
            }

        # ---- Flat hierarchy ----
        if self.hierarchy == "flat":
            masks, centers = self._blocks(self.bs)
            phis = self._phi_batch_zero(x, y, masks, log_p_full)
            M = np.zeros((H, W), dtype=np.float32)
            for (i, j), p in zip(centers, phis):
                M[i : i + self.bs, j : j + self.bs] = p / (self.bs * self.bs)
            return M, {}

        # ---- Two-level: coarse grid + top-B refine ----
        cmasks, ccenters = self._blocks(self.cs)
        cphis = self._phi_batch_zero(x, y, cmasks, log_p_full)
        M = np.zeros((H, W), dtype=np.float32)
        for (i, j), p in zip(ccenters, cphis):
            M[i : i + self.cs, j : j + self.cs] = p / (self.cs * self.cs)

        if not proposals:
            return M, {"set_scores": [], "n_proposals": 0}

        set_phis = self._phi_batch_zero(x, y, proposals, log_p_full)
        rank_key = set_phis if self.signed else np.abs(set_phis)
        top_idx = np.argsort(rank_key)[::-1][: self.top_B]

        union = np.zeros((H, W), dtype=bool)
        for ti in top_idx:
            union |= proposals[ti]

        rmasks, rcenters = [], []
        for i in range(0, H, self.bs):
            for j in range(0, W, self.bs):
                block = np.zeros((H, W), dtype=bool)
                block[i : i + self.bs, j : j + self.bs] = True
                if (block & union).any():
                    rmasks.append(block)
                    rcenters.append((i, j))

        if rmasks:
            rphis = self._phi_batch_zero(x, y, rmasks, log_p_full)
            for (i, j), p in zip(rcenters, rphis):
                M[i : i + self.bs, j : j + self.bs] = p / (self.bs * self.bs)

        return M, {
            "set_scores": [float(set_phis[ti]) for ti in top_idx],
            "n_proposals": len(proposals),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["imagenet", "cub"])
    ap.add_argument("--model", required=True,
                    choices=["vgg16", "resnet101", "densenet121", "mobilenet_v3_large"])
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--hierarchy", default="two-level",
                    choices=["flat", "two-level"])
    ap.add_argument("--coarse_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--top_B", type=int, default=5)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sam_cache_only", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="Load samples_seed<N>.json instead of samples.json")
    ap.add_argument("--set_uniform", action="store_true",
                help="ITF set-uniform over top-B SAM proposals only. "
                     "This is the apples-to-apples map for H-Sets FC comparison.")
    args = ap.parse_args()

    CONFIG["dataset"] = args.dataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.dataset == "imagenet":
        model = load_imagenet_model(args.model).to(device).eval()
        ds = ImageFolder(CONFIG["imagenet_val_dir"],
                         transform=val_transform(dataset="imagenet"))
    else:
        model, _ = load_cub_model(args.model)
        model = model.to(device).eval()
        ds = ImageFolder(CONFIG["cub_val_dir"],
                         transform=val_transform(dataset="cub"))

    #samples = json.load(
    #    open(f"cache/{args.dataset}_{args.model}/samples.json")
    #)["samples"][:args.num_samples]

    if args.seed is not None:
        sample_file = f"cache/{args.dataset}_{args.model}/samples_seed{args.seed}.json"
    else:
        sample_file = f"cache/{args.dataset}_{args.model}/samples.json"
    if not os.path.isfile(sample_file):
        raise FileNotFoundError(f"{sample_file} missing")
    samples = json.load(open(sample_file))["samples"][:args.num_samples]
    print(f"Loaded {len(samples)} samples from {sample_file}")

    if args.sam_cache_only:
        print(f"Pre-warming SAM cache for {len(samples)} samples")
        for k, idx in enumerate(samples):
            x, _ = ds[idx]
            m = get_sam_proposals(x.unsqueeze(0).to(device),
                                  args.dataset, args.model, idx)
            if (k+1) % 5 == 0:
                print(f"  {k+1}/{len(samples)}  ({len(m)} masks)")
        print("Done")
        return

    # Two models: attribution in fp16 if requested, metrics ALWAYS fp32
    if args.fp16:
        import copy
        model_attr = copy.deepcopy(model).half()
    else:
        model_attr = model

   # attributor = ITFAttributor(
   #     model_attr, device=device,
   #     coarse_size=args.coarse_size, block_size=args.block_size,
   #     top_B=args.top_B, hierarchy=args.hierarchy,
   #     signed=args.signed, batch_size=args.batch_size,
   # )
   

    attributor = ITFAttributor(
        model_attr,
        device=device,
        coarse_size=args.coarse_size,
        block_size=args.block_size,
        top_B=args.top_B,
        hierarchy=args.hierarchy,
        signed=args.signed,
        batch_size=args.batch_size,
        set_uniform=args.set_uniform,
    )

    ginis, roads, corrs = [], [], []
    t0 = time.time()
    for k, idx in enumerate(samples):
        x, y = ds[idx]
        x = x.unsqueeze(0).to(device)

        proposals = get_sam_proposals(x, args.dataset, args.model, idx)
        M, _ = attributor.attribute(x, y, proposals)

        # abs + minmax, exactly like H-Sets' to_2d
        M_hw = np.abs(M).astype(np.float32)
        lo, hi = M_hw.min(), M_hw.max()
        M_hw = (M_hw - lo) / (hi - lo + 1e-12)

        if M_hw.max() < 1e-8:
            print(f"  [{k+1:3d}/{len(samples)}]  WARNING: all-zero map, skipping")
            continue




        g = gini(M_hw.ravel())
        r = road_aopc(model, x[0], y, M_hw, device=device)
        c = faithfulness_corr(model, x[0], y, M_hw[None], device=device)

        ginis.append(g); roads.append(r); corrs.append(c)
        if (k+1) % 5 == 0 or (k+1) == len(samples):
            el = time.time() - t0
            eta = el / (k+1) * (len(samples) - k - 1) / 60
            print(f"  [{k+1:3d}/{len(samples)}] gini={g:.4f} road={r:.4f} "
                  f"corr={c:.4f}  ({el/(k+1):.1f}s/img, ETA {eta:.1f}min)")

    skipped = len(samples) - len(ginis)
    print(f"\n  computed {len(ginis)}/{len(samples)} (skipped {skipped} all-zero maps)")

    
    out = {
        "dataset": args.dataset, "model": args.model, "n": len(ginis),
        "sampler": "zero", "hierarchy": args.hierarchy, "fp16": args.fp16,
        "signed": args.signed, "coarse_size": args.coarse_size, "seed": args.seed,
        "block_size": args.block_size, "top_B": args.top_B, "tag": args.tag,
        "gini": {"mean": float(np.mean(ginis)), "std": float(np.std(ginis))},
        "road": {"mean": float(np.mean(roads)), "std": float(np.std(roads))},
        "corr": {"mean": float(np.mean(corrs)), "std": float(np.std(corrs))},
    }
    print(f"\nITF {args.dataset}/{args.model}  tag={args.tag}  N={out['n']}")
    print("=" * 60)
    for k in ["gini", "road", "corr"]:
        print(f"  {k:<6}  {out[k]['mean']:+.4f} ± {out[k]['std']:.4f}")

    out_path = f"cache/itf_{args.dataset}_{args.model}_{args.tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
