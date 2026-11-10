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
