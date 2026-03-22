# Zero-Latency — Trajectory Prediction Pipeline

A multi-modal transformer-based trajectory prediction model trained on the nuScenes dataset. Achieves competitive ADE/FDE against state-of-the-art methods.

---

## Results

| Metric | Value | Benchmark |
|---|---|---|
| Mean ADE | 1.31 m | Beats PECNet (2.10m) |
| Mean FDE | 1.98 m | Beats Trajectron++ SOTA (2.91m) |
| Median ADE | 0.64 m | — |
| Median FDE | 0.72 m | — |

Evaluated on full nuScenes v1.0-trainval (34,149 samples).

---

## Repository Structure

```
zero-latency/
├── dataset/
│   └── nuscenes_dataset.py      # nuScenes loader (supports mini + trainval + test)
├── modules/
│   ├── input_embedding.py
│   ├── temporal_transformer.py
│   ├── social/
│   │   └── social_transformer.py
│   ├── scene/
│   │   └── scene_context_encoder.py
│   └── decoder/
│       ├── goal_prediction.py
│       └── multimodal_decoder.py
├── utils/
│   └── losses.py                # best_of_k_loss, goal_classification_loss
├── train.py                     # Cloud training (RTX 5090 + Linux)
├── train-windows-rtx5070.py     # Local training (RTX 5070 Laptop + Windows)
├── evaluate2.py                 # Full evaluation with plots
├── infer_test.py                # nuScenes leaderboard submission
├── setup.sh                     # Cloud environment setup
└── checkpoints/                 # Saved model weights
    ├── best_1.pt                # Best validation loss checkpoint
    ├── best_2.pt                # Second best checkpoint
    └── latest.pt                # Most recent epoch
```

---

## Quick Start — Local (Windows, RTX 5070 Laptop)

### 1. Clone and install

```powershell
git clone https://github.com/Aurora-source/zero-latency.git
cd zero-latency
git checkout working-rikon

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### 2. Train on nuScenes mini

```powershell
python train-windows-rtx5070.py
```

### 3. Evaluate

```powershell
python evaluate2.py --dataroot "data/raw/nuscenes" --version v1.0-mini --batch_size 16
```

---

## Cloud Setup (Linux, RTX 5090 — vast.ai)

### 1. Run setup script

```bash
bash setup.sh
```

This will clone the repo, install all dependencies, install PyTorch cu128 nightly, and verify the installation.

### 2. Download nuScenes dataset using aria2

Install aria2:

```bash
apt-get install -y aria2
```

Copy your `links.txt` (from nuScenes download page) to the instance, then download:

```bash
# Download trainval dataset parts to nuscenes folder
mkdir -p /workspace/zero-latency/nuscenes
cd /workspace/zero-latency/nuscenes

# Download links.txt from Google Drive first
rclone copy "Rikon:nuscenes/links.txt" .

# Run aria2 to download all parts
aria2c -c \
  -j 2 \
  -x 8 \
  -s 8 \
  -k 1M \
  --file-allocation=falloc \
  --dir=. \
  -i links.txt
```

**aria2 flags explained:**
| Flag | Meaning |
|---|---|
| `-c` | Resume interrupted downloads |
| `-j 2` | 2 parallel downloads |
| `-x 8` | 8 connections per download |
| `-s 8` | Split each file into 8 segments |
| `--file-allocation=falloc` | Pre-allocate disk space (prevents fragmentation) |
| `--dir=.` | Download to current directory |
| `-i links.txt` | Read URLs from file |

### 3. Extract dataset parts (one at a time to manage disk space)

```bash
cd /workspace/zero-latency/nuscenes

# Extract metadata first (always required)
tar -xzf v1.0-trainval_meta.tgz && rm v1.0-trainval_meta.tgz

# Extract blob parts one at a time (delete after extracting to save space)
tar -xzf v1.0-trainval01_blobs.tgz && rm v1.0-trainval01_blobs.tgz
tar -xzf v1.0-trainval02_blobs.tgz && rm v1.0-trainval02_blobs.tgz
tar -xzf v1.0-trainval03_blobs.tgz && rm v1.0-trainval03_blobs.tgz
# ... repeat for all parts
```

Expected folder structure after extraction:
```
nuscenes/
├── maps/
│   └── prediction/
│       └── prediction_scenes.json
├── samples/
├── sweeps/
└── v1.0-trainval/
    ├── scene.json
    ├── sample.json
    ├── instance.json
    └── ...
```

### 4. Train on cloud

```bash
cd /workspace/zero-latency

# Train with trainval dataset (parts 1+2 = ~6000 samples)
NUSCENES_ROOT=/workspace/zero-latency/nuscenes \
TORCH_COMPILE_MODE=reduce-overhead \
BATCH_SIZE=32 \
EVAL_BATCH_SIZE=16 \
python train.py
```

---

## Saving and Uploading Checkpoints with rclone

### Setup rclone (one time per instance)

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config
# Follow prompts:
# n → new remote
# name: Rikon  (or your preferred name)
# Storage: drive
# Leave client_id and client_secret blank
# scope: 1 (full access)
# Use auto config: n  (we are on remote server)
# Copy the URL shown → open in laptop browser → approve → paste code back
# Configure as Shared Drive: n
# Save with: y
```

### Upload checkpoints after training

```bash
# Upload best checkpoint only (recommended — ~3GB)
rclone copy /workspace/zero-latency/checkpoints/best_1.pt \
  Rikon:zero-latency/checkpoints/ --progress

# Upload all checkpoints
rclone copy /workspace/zero-latency/checkpoints/ \
  Rikon:zero-latency/checkpoints/ --progress

# Verify upload
rclone ls Rikon:zero-latency/checkpoints/
```

### Download checkpoints to laptop (Windows)

```powershell
# Install rclone on Windows
winget install Rclone.Rclone

# Configure (same steps as above but use auto config: y on Windows)
rclone config

# Download best checkpoint
rclone copy "Rikon:zero-latency/checkpoints/best_1.pt" "C:\Users\Rikon\zero-latency\checkpoints\"
```

---

## Training Configuration

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `NUSCENES_ROOT` | `data/raw/nuscenes` | Path to nuScenes dataset |
| `NUSCENES_VERSION` | `v1.0-trainval` | Dataset version |
| `DATASET_LIMIT` | `6000` | Max samples to use |
| `BATCH_SIZE` | `32` | Micro batch size |
| `GRAD_ACCUM_STEPS` | `1` | Gradient accumulation steps |
| `LR` | `5e-5` | Learning rate |
| `RUN_EPOCHS` | `40` | Epochs per run |
| `RESUME` | `1` | Resume from best checkpoint |
| `TORCH_COMPILE` | `1` | Enable torch.compile (Linux only) |
| `CHECKPOINT_DIR` | `checkpoints` | Where to save checkpoints |

### Resume training

Training automatically resumes from `checkpoints/best_1.pt` on every run. To start fresh:

```bash
RESUME=0 python train.py
```

---

## Evaluation

### Run full evaluation with plots

```powershell
# On local machine (mini dataset)
python evaluate2.py --dataroot "data/raw/nuscenes" --version v1.0-mini --batch_size 16

# On local machine (full trainval on D: drive)
python evaluate2.py --dataroot "D:/v1.0-trainval" --version v1.0-trainval --batch_size 32
```

Saves plots to `evaluation_results/evaluation_results.png` including:
- Displacement error per timestep
- ADE/FDE histograms
- Cumulative error distribution
- Summary metrics table

### Generate leaderboard submission

```powershell
# Generate submission for official val split
python infer_test.py --dataroot "D:/v1.0-trainval" --split val

# Submit submission/submission_val.json to:
# https://eval.ai/web/challenges/challenge-page/591/overview
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 32 GB |
| RAM | 16 GB | 64 GB |
| Disk | 50 GB | 500 GB |
| CUDA | 12.8+ | 13.0+ |

### Tested configurations

| Setup | Batch size | Time per 40 epochs |
|---|---|---|
| RTX 5070 Laptop (8GB, Windows) | 4 | ~15 min (mini) |
| RTX 5090 Cloud (32GB, Linux) | 32-64 | ~8 min (mini) |

---

## Dataset

This project uses the [nuScenes dataset](https://www.nuscenes.org/).

Supported versions:
- `v1.0-mini` — 404 samples, 10 scenes (for local testing)
- `v1.0-trainval` — 34,149 samples, 850 scenes (for full training)
- `v1.0-test` — 6,008 samples, 150 scenes (no annotations, for leaderboard)

---

## setup.sh Reference

```bash
#!/bin/bash
# setup.sh — Cloud environment setup for zero-latency
# Run with: bash setup.sh

set -e  # stop on any error

echo "=== Cloning repo ==="
git clone https://github.com/Aurora-source/zero-latency.git
cd zero-latency
git checkout working-rikon

echo "=== Installing dependencies (excluding torch) ==="
pip install \
  cachetools==7.0.5 \
  colorama==0.4.6 \
  contourpy==1.3.3 \
  cycler==0.12.1 \
  descartes==1.1.0 \
  filelock==3.25.2 \
  fire==0.7.1 \
  fonttools==4.62.1 \
  fsspec==2026.2.0 \
  Jinja2==3.1.6 \
  joblib==1.5.3 \
  kiwisolver==1.5.0 \
  MarkupSafe==3.0.3 \
  matplotlib==3.10.8 \
  mpmath==1.3.0 \
  networkx==3.6.1 \
  numpy==2.4.3 \
  "nuscenes-devkit==1.2.0" --no-deps \
  opencv-python-headless==4.13.0.92 \
  packaging==26.0 \
  pandas==3.0.1 \
  parameterized==0.9.0 \
  pillow==12.1.1 \
  pycocotools==2.0.11 \
  pyparsing==3.3.2 \
  pyquaternion==0.9.9 \
  python-dateutil==2.9.0.post0 \
  PyYAML==6.0.3 \
  scikit-learn==1.8.0 \
  scipy==1.17.1 \
  setuptools==78.1.0 \
  shapely==2.1.2 \
  six==1.17.0 \
  sympy==1.14.0 \
  termcolor==3.3.0 \
  threadpoolctl==3.6.0 \
  tqdm==4.67.3 \
  typing_extensions==4.15.0 \
  tzdata==2025.3

echo "=== Installing PyTorch (cu128 nightly) ==="
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

echo "=== Installing remaining nuscenes-devkit deps ==="
pip install cachetools descartes fire \
  opencv-python-headless parameterized pycocotools \
  pyquaternion scikit-learn scipy shapely

echo "=== Verifying install ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
from nuscenes.nuscenes import NuScenes
print('nuscenes: OK')
"

echo "=== Setup complete! ==="
echo "Next: download nuscenes data using aria2 with links.txt"
echo "Then run: NUSCENES_ROOT=/workspace/zero-latency/nuscenes python train.py"
```
