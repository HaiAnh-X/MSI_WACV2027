# Memory Subspace Iteration (MSI) for Efficient Video Object Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![WACV 2027](https://img.shields.io/badge/WACV-2027-success.svg)](https://wacv2027.thecvf.com/)

Official PyTorch implementation of **"Efficient Memory-constrained Video Object Segmentation via Subspace Optimization"** (WACV 2027).

---

## 📌 News
- **October 2026**: Codebase, checkpoints, and benchmark evaluation scripts released!

---

## 📖 Overview

Memory-based architectures for video object segmentation (such as **Segment Anything Model 2 - SAM 2**) establish strong state-of-the-art accuracy by employing a streaming **Memory Bank** to retrieve dense feature representations across temporal frames. However, maintaining uncompressed memory tensors over long video sequences incurs substantial storage and GPU memory footprints.

In this work, we demonstrate that **memory feature tensors exhibit pronounced temporal coherence**: consecutive frames reside in a stable low-rank subspace, and the principal informational content is preserved over time.

Motivated by this insight, we introduce **Memory Subspace Iteration (MSI)**:
- 🚀 **Training-Free & Plug-and-Play**: Seamlessly integrates into pre-trained video segmentation backbones (SAM 2, SAM 2.1) without retraining or fine-tuning.
- ⚡ **Online Subspace Iteration**: Employs Higher-Order SVD (HOSVD) power iteration, re-using orthonormal basis matrices across adjacent frames for rapid convergence.
- 📉 **Significant Resource Savings**: Reduces streaming memory bank footprint by up to **25%** and lowers power consumption by up to **27%**.
- 🎯 **Preserved Accuracy**: Restricts segmentation performance degradation to less than **1%** across 7 standard VOS benchmarks.

```
       [Raw Frame Vision Features]
                   │
                   ▼
       ┌────────────────────────┐
       │   Memory Encoder       │
       └───────────┬────────────┘
                   │  Dense Memory Tensor (B x C x H x W)
                   ▼
       ┌────────────────────────┐
       │  MSI Online Subspace   │  ◄── Warm-start Basis (U_prev)
       │       Iteration        │
       └───────────┬────────────┘
                   │  Low-rank Core Tensor S + Orthonormal Factors {U_i}
                   ▼
         [Compressed Memory Bank]
                   │
                   ▼ (On-demand Subspace Reconstruction)
         [Memory Cross-Attention]
```

---

## 📊 Benchmark Results

Comparison of **MSI** against vanilla **Baseline SAM 2.1** across standard VOS benchmarks evaluated under temporal sampling intervals $\Delta t \in \{1, 5\}$ on an NVIDIA A100 GPU (40GB). The primary metric is $\mathcal{J}\&\mathcal{F}$ (%).

### SAM 2.1 Base-Plus (`sam2.1_hiera_b+`)

| Method | $\Delta t$ | SA-V (T) | DAVIS '17 | SA-V (V) | MOSE | SeCVOS | LVOS | YT-VOS | Avg $\uparrow$ | Peak Mem (GB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline SAM 2.1** | 1 | **77.2** | **89.7** | **77.1** | 43.3 | 57.3 | 82.2 | **88.2** | **74.3** | 13.66 |
| **MSI + SAM 2.1** | 1 | 76.6 | **89.7** | 76.9 | **44.1** | **57.6** | **82.7** | 87.6 | 74.2 | **13.36** |
| **Baseline SAM 2.1** | 5 | 78.8 | **88.6** | **79.8** | **46.8** | 58.7 | **84.0** | **88.0** | **74.3** | 13.66 |
| **MSI + SAM 2.1** | 5 | **79.2** | 88.5 | 78.8 | 46.6 | **59.1** | 83.9 | 87.3 | 74.2 | **13.36** |

### SAM 2.1 Large (`sam2.1_hiera_l`)

| Method | $\Delta t$ | SA-V (T) | DAVIS '17 | SA-V (V) | MOSE | SeCVOS | LVOS | YT-VOS | Avg $\uparrow$ | Peak Mem (GB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline SAM 2.1** | 1 | **78.4** | **89.9** | **79.0** | **49.4** | 58.2 | **84.2** | **88.8** | **76.0** | 14.95 |
| **MSI + SAM 2.1** | 1 | 78.2 | 89.7 | 78.8 | 48.9 | **58.5** | 84.1 | **88.8** | 75.9 | **14.63** |
| **Baseline SAM 2.1** | 5 | **80.3** | 89.9 | 80.2 | **50.8** | **59.3** | **86.7** | 88.7 | **76.0** | 14.95 |
| **MSI + SAM 2.1** | 5 | 80.0 | **90.1** | **80.5** | 50.7 | **59.3** | 86.5 | 88.7 | 75.9 | **14.63** |

---

## 🛠️ Installation

### 1. Prerequisites
- Linux (Ubuntu 20.04/22.04 recommended)
- Python $\ge$ 3.10
- PyTorch $\ge$ 2.3.1 with CUDA $\ge$ 12.1

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/HaiAnh-X/WACV.git
cd WACV/sam2

# Create and activate conda environment
conda create -n msi python=3.10 -y
conda activate msi

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SAM 2 and MSI dependencies
pip install -e .
pip install hydra-core omegaconf tqdm opencv-python Pillow matplotlib pandas
```

---

## 📦 Model Checkpoints

Download official SAM 2.1 checkpoints into `checkpoints/`:

```bash
cd sam2/checkpoints
bash download_ckpts.sh
```

Alternatively, direct download links:
| Model Backbone | Checkpoint Link | Size |
| :--- | :--- | :---: |
| SAM 2.1 Hiera Base+ | [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) | 308 MB |
| SAM 2.1 Hiera Large | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | 856 MB |
| SAM 2.1 Hiera Small | [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) | 176 MB |
| SAM 2.1 Hiera Tiny | [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | 149 MB |

---

## 📂 Dataset Organization

Organize benchmark datasets with the following directory structure:

```
data/
├── DAVIS/
│   ├── JPEGImages/480p/
│   ├── Annotations/480p/
│   └── ImageSets/2017/val.txt
└── SA-V/
    ├── sav_val/
    │   ├── JPEGImages_24fps/
    │   ├── Annotations_6fps/
    │   └── sav_val_30.txt
    └── sav_test/
        ├── JPEGImages_24fps/
        ├── Annotations_6fps/
        └── sav_test_30.txt
```

---

## 🚀 Benchmark Reproduction & Evaluation

Our unified `vos_inference.py` supports running both **MSI** (enabled by default) and **Baseline SAM 2** (via `--disable_msi`), as well as configuring temporal strides $\Delta t$ and energy retention thresholds $\varepsilon$.

### 1. DAVIS 2017 Benchmark

#### MSI Evaluation ($\Delta t = 1$ and $\Delta t = 5$):
```bash
# MSI (Δt = 1)
python vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path/to/data/DAVIS/JPEGImages/480p \
  --input_mask_dir /path/to/data/DAVIS/Annotations/480p \
  --video_list_file /path/to/data/DAVIS/ImageSets/2017/val.txt \
  --memory_temporal_stride_for_eval 1 \
  --output_mask_dir benchmark_eval/davis_b+_msi_1

# Compute metrics
python tools/davis2017-evaluation/evaluation_method.py \
  --davis_path /path/to/data/DAVIS \
  --results_path benchmark_eval/davis_b+_msi_1 \
  --set val --task semi-supervised
```

#### Baseline SAM 2 Evaluation:
```bash
# Add --disable_msi to run vanilla SAM 2 baseline
python vos_inference.py \
  --disable_msi \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path/to/data/DAVIS/JPEGImages/480p \
  --input_mask_dir /path/to/data/DAVIS/Annotations/480p \
  --video_list_file /path/to/data/DAVIS/ImageSets/2017/val.txt \
  --memory_temporal_stride_for_eval 1 \
  --output_mask_dir benchmark_eval/davis_b+_base_1

# Compute metrics
python tools/davis2017-evaluation/evaluation_method.py \
  --davis_path /path/to/data/DAVIS \
  --results_path benchmark_eval/davis_b+_base_1 \
  --set val --task semi-supervised
```

---

### 2. Segment Anything Video (SA-V) Benchmark

For the SA-V dataset, objects can appear at intermediate frames and annotations are provided per object:

```bash
# SA-V Val with MSI (Δt = 1)
python vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path/to/data/SA-V/sav_val/JPEGImages_24fps \
  --input_mask_dir /path/to/data/SA-V/sav_val/Annotations_6fps \
  --video_list_file /path/to/data/SA-V/sav_val/sav_val_30.txt \
  --track_object_appearing_later_in_video \
  --per_obj_png_file \
  --memory_temporal_stride_for_eval 1 \
  --output_mask_dir benchmark_eval/sav_val_b+_msi_1

# Evaluate SA-V Val
python tools/sav_dataset/sav_evaluator.py \
  --gt_root /path/to/data/SA-V/sav_val/Annotations_6fps \
  --pred_root benchmark_eval/sav_val_b+_msi_1
```

For SA-V Test, substitute paths with `sav_test/`.

> **Note on Custom Ops (`Linear_MSI`)**:
> Pass `--wrap_linear` to wrap memory attention linear layers (`k_proj`, `v_proj`) with the custom `Linear_MSI` operator, matching the SAM 3 implementation. Both SAM 2 and SAM 3 in this repository support `--wrap_linear`.

---

## ⚡ Slurm Batch Execution (HPC Clusters)

We provide modular Slurm batch scripts in `sbatch/msi/` optimized for NVIDIA A100 GPUs:

```bash
# Submit all DAVIS 2017 benchmarks (MSI + Baseline, Δt=1, 5)
bash sbatch/msi/submit_davis.sh

# Submit SA-V Val benchmarks
bash sbatch/msi/submit_sav_val.sh

# Submit SA-V Test benchmarks
bash sbatch/msi/submit_sav_test.sh

# Check and compare results against paper benchmarks
python sbatch/msi/check_results.py
```

---

## 📚 Citation

If you find this work or codebase helpful in your research, please consider citing:

```bibtex
@inproceedings{wacv2027_msi,
  title     = {Efficient Memory-constrained Video Object Segmentation via Subspace Optimization},
  author    = {Nguyen, Linh and Collaborators},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

---

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. Built upon [Meta SAM 2](https://github.com/facebookresearch/sam2).
