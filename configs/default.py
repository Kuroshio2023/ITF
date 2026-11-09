# configs/default.py
"""Single source of truth. Edit DATASET + MODEL to switch experiments."""

CONFIG = {
    # ---- select experiment ----
    "dataset": "imagenet",          # "imagenet" | "cub"
    "model":   "resnet101",         # "vgg16" | "resnet101" | "densenet121" | "mobilenet_v3_large"
    "seed" :   42,

    # ---- data ----
    "imagenet_val_dir": "datasets/imagenet/val",
    "cub_raw_dir":      "datasets/cub",           # contains images/, *.txt
    "cub_train_dir":    "datasets/cub/train",
    "cub_val_dir":      "datasets/cub/val",
    "cub_class_map":    "datasets/cub/class_to_idx.json",
    "num_classes_imagenet": 1000,
    "num_classes_cub":      200,

    # ---- preprocessing (paper-mandated) ----
    "resize": 256,
    "crop":   224,
   # "normalize_mean": [0.5, 0.5, 0.5],
   # "normalize_std":  [0.5, 0.5, 0.5],

    "normalize_mean": {
        "imagenet": [0.485, 0.456, 0.406],
        "cub":      [0.5, 0.5, 0.5],
    },
    "normalize_std": {
        "imagenet": [0.229, 0.224, 0.225],
        "cub":      [0.5, 0.5, 0.5],
    },

    # ---- SAM ----
    "sam_checkpoint": "/data/checkpoints/sam_vit_h_4b8939.pth",
    "sam_min_mask_area": 1000,

    # ---- checkpoints ----
    "checkpoint_dir": "checkpoints",
    "cub_checkpoint_template": "checkpoints/cub_{model}.pth",

    # ---- H-Sets hyperparameters (paper defaults) ----
    "threshold":   0.5,
    "k_sets":      5,
    "max_elements_imagenet": 2000,
    "max_elements_cub":      3000,
    "path_steps":  50,
    "mc_samples":  50,

    # ---- evaluation ----
    "num_samples":   1000,
    "sample_seed":   42,
    "cache_dir":     "cache",
    "output_dir":    "output",
}
