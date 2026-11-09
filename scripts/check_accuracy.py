# scripts/check_accuracy.py
"""Top-1 val accuracy for imagenet and cub, for all four models."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, shutil
from configs.default import CONFIG



import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision import models as tv_models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["vgg16", "resnet101", "densenet121", "mobilenet_v3_large"]


#def val_transform():
#    return T.Compose([
#        T.Resize(CONFIG["resize"]),
#        T.CenterCrop(CONFIG["crop"]),
#        T.ToTensor(),
#        T.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"]),
#    ])

def val_transform(dataset=None):
    ds_name = dataset if dataset is not None else CONFIG["dataset"]
    mean = CONFIG["normalize_mean"][ds_name]
    std  = CONFIG["normalize_std"][ds_name]
    return T.Compose([
        T.Resize(CONFIG["resize"]),
        T.CenterCrop(CONFIG["crop"]),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_imagenet_model(name):
    table = {
        "vgg16":            (tv_models.vgg16,             tv_models.VGG16_Weights.IMAGENET1K_V1),
        "resnet101":        (tv_models.resnet101,         tv_models.ResNet101_Weights.IMAGENET1K_V1),
        "densenet121":      (tv_models.densenet121,       tv_models.DenseNet121_Weights.IMAGENET1K_V1),
        "mobilenet_v3_large":(tv_models.mobilenet_v3_large, tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    }
    fn, w = table[name]
    return fn(weights=w)


def load_cub_model(name):
    from scripts.finetune_cub import build_model
    ckpt_path = CONFIG["cub_checkpoint_template"].format(model=name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    model = build_model(name, 200)
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only = False)
    model.load_state_dict(ckpt["model_state_dict"])
    # Sanity: class map matches what we prepped
    import json
    with open(CONFIG["cub_class_map"]) as f:
        prep_map = json.load(f)
    assert ckpt["class_to_idx"] == prep_map, \
        "Checkpoint's class_to_idx does not match datasets/cub/class_to_idx.json"
    return model, ckpt.get("best_val_acc")


def run_accuracy(dataset, name):
    if dataset == "imagenet":
        ds = ImageFolder(CONFIG["imagenet_val_dir"], transform=val_transform(dataset="imagenet"))
        model = load_imagenet_model(name)
        note = ""
    else:
        ds = ImageFolder(CONFIG["cub_val_dir"], transform=val_transform(dataset="cub"))
        model, ckpt_acc = load_cub_model(name)
        note = f"  (checkpoint best: {ckpt_acc*100:.2f}%)"

    model.to(device).eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct, total, note


def main():
    print(f"Device: {device}\n")
    print(f"{'dataset/model':<32} {'acc':>8}  {'correct/total':>16}  note")
    print("-" * 80)

    for dataset in ["imagenet", "cub"]:
        for name in MODEL_NAMES:
            try:
                correct, total, note = run_accuracy(dataset, name)
                print(f"{dataset+'/'+name:<32} {correct/total*100:>7.2f}%  "
                      f"{correct:>6}/{total:<6}  {note}")
            except FileNotFoundError as e:
                print(f"{dataset+'/'+name:<32} {'SKIP':>8}  (missing: {os.path.basename(str(e))})")
            except Exception as e:
                print(f"{dataset+'/'+name:<32} {'ERROR':>8}  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
