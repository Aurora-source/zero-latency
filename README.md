# Zero-Latency — 

A multi-modal transformer-based trajectory prediction model trained on the nuScenes dataset. The model predicts future trajectories of traffic agents (vehicles, pedestrians, cyclists) using historical motion data and scene context. 


Orignal problem statement :
" Participants must develop a model that predicts the future coordinates (next 3 seconds) of pedestrians and cyclists based on 2 seconds of past motion. "

But we decide to improve the model by using the past 3 seconds of motion and then predicting the next 6 seconds of motion instead. Doing this may tank our ADE and FDE as we could have shown much better results for less time predicted but I thought that this model would be better in the long run. Also model can predict the motion of other vechicals too.

---

## Prediction results

The visualisation below shows the model predicting 6 seconds of future trajectories for all agents in a single scene. Solid lines are the model's best prediction (minADE mode), dashed lines are ground truth, and faint lines show all K=6 predicted modes. Also the results change for every scene , ADE and FDE are higher for scenes with higher moving objects. In the animation you can see the model taking previous 6 steps (3 secs) and then predicting the next 12 steps (6 secs)  ( TO REPRODUCE the exact scene presented here use python single_inference.py --seed 5575121 , the seed is important )

![Multi-agent trajectory prediction](visualisations/multi_agent_prediction.png)

<video src="https://github.com/user-attachments/assets/bb1ebaea-88d3-4d98-9b5a-b3a720f0771d" controls width="500"></video>

```python
    GROUND TRUTH MOVEMENT
  Agent  0 [[V] vehicle (Agent 0)]:   0.09 m  [static]
  Agent  1 [[V] vehicle (Agent 1)]:   0.14 m  [static]
  Agent  2 [[P] pedestrian (Agent 2)]:   6.97 m  [moving]
  Agent  3 [[P] pedestrian (Agent 3)]:   6.88 m  [moving]
  Agent  4 [[P] pedestrian (Agent 4)]:   7.32 m  [moving]
  Agent  5 [[P] pedestrian (Agent 5)]:   7.24 m  [moving]
  Agent  6 [[P] pedestrian (Agent 6)]:   8.34 m  [moving]
  Agent  7 [[P] pedestrian (Agent 7)]:   8.49 m  [moving]
  Agent  8 [[V] vehicle (Agent 8)]:   0.12 m  [static]
  Agent  9 [[V] vehicle (Agent 9)]:   3.53 m  [static]
```
 ```python
      PER-AGENT METRICS
  Agent  0 [V] [vehicle] | best_mode(minADE)=2 p=0.140 | ADE=0.115m  FDE=0.116m
  Agent  1 [V] [vehicle] | best_mode(minADE)=2 p=0.136 | ADE=0.315m  FDE=0.293m
  Agent  2 [P] [pedestrian] | best_mode(minADE)=0 p=0.131 | ADE=1.168m  FDE=1.246m
  Agent  3 [P] [pedestrian] | best_mode(minADE)=0 p=0.131 | ADE=0.860m  FDE=0.538m
  Agent  4 [P] [pedestrian] | best_mode(minADE)=3 p=0.133 | ADE=0.361m  FDE=0.240m
  Agent  5 [P] [pedestrian] | best_mode(minADE)=3 p=0.139 | ADE=0.605m  FDE=0.795m
  Agent  6 [P] [pedestrian] | best_mode(minADE)=2 p=0.132 | ADE=0.502m  FDE=0.744m
  Agent  7 [P] [pedestrian] | best_mode(minADE)=3 p=0.138 | ADE=0.518m  FDE=0.329m
  Agent  8 [V] [vehicle] | best_mode(minADE)=3 p=0.131 | ADE=0.244m  FDE=0.286m
  Agent  9 [V] [vehicle] | best_mode(minADE)=5 p=0.132 | ADE=1.288m  FDE=2.049m

  Mean ADE : 0.5975 m
  Mean FDE : 0.6637 m
```  
---

## Project overview

Zero-Latency is a trajectory prediction pipeline built on a multi-stage transformer architecture. Given a sequence of past agent positions and scene context, the model predicts multiple plausible future trajectories with associated probabilities.

The pipeline consists of six learned modules working in sequence:

1. **InputEmbedding** — encodes raw per-agent features (position, velocity, heading, agent type) into transformer-ready tokens
2. **TemporalTransformer** — models each agent's motion history over time
3. **SocialTransformer** — models interactions between agents at each timestep
4. **SceneContextEncoder** — fuses agent representations with map context via cross-attention
5. **GoalPredictionNetwork** — predicts a distribution over likely goal positions
6. **MultiModalDecoder** — generates K future trajectory modes conditioned on predicted goals

---

## Model architecture

```
Input: (batch, time= 6 steps (3 seconds), agents=10, features=8)
         ↓
InputEmbedding          ~0.5M params
         ↓
TemporalTransformer     ~110M params
         ↓
SocialTransformer       ~42M params
         ↓
SceneContextEncoder     ~28M params
         ↓
GoalPredictionNetwork   ~0.6M params
         ↓
MultiModalDecoder       ~82M params
         ↓
Output: (batch, K=6, agents, time= 12 steps (6 seconds), 2)
```

**Total parameters: ~263M**

Each future timestep is 0.5 seconds apart — the model predicts 6 seconds into the future taking in 3 seconds of past.

---

## Dataset

This project uses the [nuScenes dataset](https://www.nuscenes.org/). Registration is required to download. 

| Version | Samples | Scenes | Size |
|---|---|---|---|
| `v1.0-mini` | 404 | 10 | 3.9 GB |
| `v1.0-trainval` | 34,149 | 850 | ~300 GB (10 parts) |

Download links for `v1.0-trainval` are available in `nuscenes/links.txt`.

---

## Results

### Evaluation on v1.0-mini (404 samples)

```
Samples evaluated : 404
Loss              : 1.6108
ADE               : 0.5972 m
FDE               : 1.0135 m
```

### Evaluation on v1.0-trainval parts 9–10 (held-out, never seen during training)

The model was trained on parts 1–8 and evaluated on parts 9–10.

```
Samples evaluated : 34,149
Mean ADE          : 0.8412 m
Mean FDE          : 1.0704 m
Median ADE        : 0.5231 m
Median FDE        : 0.5819 m
ADE p90           : 1.9242 m
FDE p90           : 2.6094 m
Miss Rate FDE>2m  : 16.7%
Miss Rate FDE>4m  :  3.3%
```

### Evaluation plots

![Evaluation results](evaluation_results/evaluation_results.png)

---

## Repository structure

```
zero-latency/
├── checkpoints/                     # raw training checkpoints (saved during training)
│   └── best_1.pt
├── configs/
├── data/
│   ├── processed/
│   └── raw/
│       └── nuscenes/                # v1.0-mini dataset goes here
├── dataset/
├── evaluation_results/
│   └── evaluation_results.png
├── models/                          # exported inference-ready model weights
│   ├── model_fp16.pt                # half-precision export (~500 MB)
│   └── model_fp32.pt                # full-precision export (~1 GB)
├── modules/
│   ├── decoder/
│   ├── scene/
│   └── social/
├── nuscenes/                        # v1.0-trainval dataset goes here
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
├── pipeline-architecture/
├── utils/
├── visualisations/                  # output PNGs and MP4s from single_inference.py
│   ├── multi_agent_prediction.png
│   └── multi_agent_prediction.mp4
├── .gitignore
├── README.md
├── evaluate-mini.py
├── evaluate-trainval.py
├── export_model.py          # exports best checkpoint → models/ folder
├── requirements.txt
├── setup.sh
├── single_inference.py      # run inference + visualise a single scene
├── train-linux-32GB-VRAM.py
└── train-windows-8GB-VRAM.py
```

---

## Configuring file paths

Every script in this repo has a **CONFIG block near the top** — this is the only place you need to edit paths. Nothing is hardcoded anywhere else.

### `single_inference.py`

All paths are relative to the repo root and work on any machine after cloning. If you keep your dataset or model weights somewhere else, update the relevant variables at the top of the script. Or if you want to use absolute path use raw strings (`r"C:\..."`).

The script accepts command-line arguments so you can override behaviour without editing the file:

```powershell
python single_inference.py                     # mini dataset, random scene
python single_inference.py --trainval          # use trainval dataset instead
python single_inference.py --seed 42           # reproducible scene
python single_inference.py --no_anim           # skip MP4, PNG only
```

### `evaluate-mini.py` / `evaluate-trainval.py`

Both scripts have a config block at the top. All paths are relative to the repo root. If you have placed your datasets somewhere else just replace them. Or if you want to use absolute path use raw strings (`r"C:\..."`).

**`evaluate-mini.py`:**
```python
CHECKPOINT_PATH = "models/model_fp32.pt"
DATAROOT        = "data/raw/nuscenes"
VERSION         = "v1.0-mini"
BATCH_SIZE      = 4
```

**`evaluate-trainval.py`:**
```python
CHECKPOINT_PATH = "models/model_fp32.pt"
DATAROOT        = "nuscenes"
VERSION         = "v1.0-trainval"
BATCH_SIZE      = 64
OUTPUT_DIR      = "evaluation_results"
```

### `export_model.py`

Uses `argparse` — paths are passed as command-line arguments with sensible defaults:

```powershell
python export_model.py                                    # uses defaults below
python export_model.py --checkpoint checkpoints/best_1.pt --output models
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best_1.pt` | Path to the raw training checkpoint |
| `--output` | `models` | Folder where exported weights are written |

This writes two files into `models/`: `model_fp32.pt` and `model_fp16.pt`.

### `train-windows-8GB-VRAM.py`

Configured via environment variables — defaults target the mini dataset on a local Windows machine:

```python
dataset_root    = os.getenv("NUSCENES_ROOT",    "data/raw/nuscenes")
version         = os.getenv("NUSCENES_VERSION", "v1.0-mini")
checkpoint_dir  = os.getenv("CHECKPOINT_DIR",   "checkpoints")
dataset_limit   = int(os.getenv("DATASET_LIMIT", "404"))
run_epochs      = int(os.getenv("RUN_EPOCHS",    "40"))
```

Override any value by setting the env var before running, e.g.:

```powershell
$env:DATASET_LIMIT="999999"; python train-windows-8GB-VRAM.py
```

### `train-linux-32GB-VRAM.py`

Same env vars, but defaults target the full trainval dataset on a cloud server:

```python
dataset_root    = os.getenv("NUSCENES_ROOT",    "nuscenes")
version         = os.getenv("NUSCENES_VERSION", "v1.0-trainval")
checkpoint_dir  = os.getenv("CHECKPOINT_DIR",   "checkpoints")
dataset_limit   = int(os.getenv("DATASET_LIMIT", "999999"))
run_epochs      = int(os.getenv("RUN_EPOCHS",    "40"))
```

Override on the command line:

```bash
NUSCENES_ROOT=/workspace/zero-latency/nuscenes RESUME=0 python train-linux-32GB-VRAM.py
```

> **Tip:** All pre-trained model weights (`model_fp16.pt`, `model_fp32.pt`, `best_1.pt`) are available in the [shared Google Drive folder](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing). You do not need to train from scratch to run inference.

---

## Setup & installation

### Windows (8 GB VRAM)

I recommend using powershell / terminal to run / evaluate. After opening just cd to your desired workspace location then start from the first step.

#### 1. Clone the repo

```powershell
git clone https://github.com/Aurora-source/zero-latency.git
cd zero-latency
```

### `IMPORTANT NOTE :`
If you have never enabled set execution policy, then your command “venv\Scripts\activate” used to activate the virtual environment in the next step won’t work. Run this to fix the problem : 

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

#### 3. Install PyTorch (CUDA 12.8 nightly)

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 4. Install Python dependencies

```powershell
pip install cachetools==7.0.5 colorama==0.4.6 contourpy==1.3.3 cycler==0.12.1 descartes==1.1.0 filelock==3.25.2 fire==0.7.1 fonttools==4.62.1 fsspec==2026.2.0 Jinja2==3.1.6 joblib==1.5.3 kiwisolver==1.5.0 MarkupSafe==3.0.3 matplotlib==3.10.8 mpmath==1.3.0 networkx==3.6.1 numpy==2.4.3 opencv-python-headless==4.13.0.92 packaging==26.0 pandas==3.0.1 parameterized==0.9.0 pillow==12.1.1 pycocotools==2.0.11 pyparsing==3.3.2 pyquaternion==0.9.9 python-dateutil==2.9.0.post0 PyYAML==6.0.3 scikit-learn==1.8.0 scipy==1.17.1 setuptools==78.1.0 shapely==2.1.2 six==1.17.0 sympy==1.14.0 termcolor==3.3.0 threadpoolctl==3.6.0 tqdm==4.67.3 typing_extensions==4.15.0 tzdata==2025.3
```

```powershell
pip install "nuscenes-devkit==1.2.0" --no-deps
pip install cachetools descartes fire opencv-python-headless parameterized pycocotools pyquaternion scikit-learn scipy shapely
```

#### 5. Install FFmpeg (required for animation output)

`single_inference.py` saves an animated `.mp4` of the predicted trajectories. FFmpeg must be on your system PATH for this to work.

**Option A — winget (Windows 10/11):**
```powershell
winget install --id Gyan.FFmpeg -e
```

**Option B — manual install:**
1. Download a build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) (recommended: the *full* build from gyan.dev)
2. Extract to a folder such as `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Search "Environment Variables" in the Start menu
   - Edit `Path` under System Variables → New → `C:\ffmpeg\bin`
4. Open a new terminal and verify:

```powershell
ffmpeg -version
```

If `ffmpeg -version` prints a version string, the animation export will work. If FFmpeg is not found, `single_inference.py` still runs and saves the static PNG — only the MP4 is skipped.

#### 6. Download the nuScenes mini dataset

The mini dataset is available in the shared Google Drive folder (the mini dataset already has meta data with it, so no need to add it separately , while working on trainval dataset requires you to download meta data separately ) :

**[Download from Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing)**

Extract it to:

```
data/raw/nuscenes/
```

(all the folder are already created in the git repo, you just need to place the extracted dataset here, or if you don’t want to move it to this directory you can set env variable for that place ->

```powershell
$env:NUSCENES_ROOT="<directory_where_dataset_is_present>"
```
)

#### 7. Download model weights

All model weights, the project presentation are available in the shared Google Drive folder:

**[Download from Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing)**

The folder is organised as follows:

| Folder / File | Description |
|---|---|
| `checkpoints/best_1.pt` | Raw training checkpoint (~3.9 GB) |
| `models/model_fp32.pt` | Exported full-precision model — recommended for inference (~1 GB) |
| `models/model_fp16.pt` | Exported half-precision model — faster, slightly lower accuracy (~500 MB) |
| `nuscenes/` | nuScenes mini dataset with metadata pre-compressed as `.tar` |
| `zero-latency.pptx` | Hackathon presentation |

Place the downloaded files in the correct local folders:

```
checkpoints/
└── best_1.pt

models/
├── model_fp32.pt
└── model_fp16.pt
```

#### 8. (Optional, but needed to run evaluate-trainval.py and train-linux-32GB-VRAM.py) Download trainval metadata for full dataset evaluation

Download the trainval dataset and metadata using the links provided in the `nuscenes/links.txt`, simply copy one link and paste it in browser to start download ( you can download any of the parts out of 10 but metadata is mandetory ) and extract it into `nuscenes/` at the repo root:

```
nuscenes/
└── v1.0-trainval/
    ├── scene.json
    ├── sample.json
    └── ...
```
All parts merge automatically into `nuscenes/samples/` and `nuscenes/sweeps/`. I recommend not downloading this just for evaluation as we have already attached the results after evalutions on part (9 - 10) of the trainval dataset, it just the size of these datasets it too large. Rest is your wish.

#### 9. Verify setup

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
ffmpeg -version
```
If these return the version number ,then the setup is complete. Also just once check if you have correctly place the nuscenes datasets, checkpoints, models in there respective directories as recommended. 
---

### Cloud (Linux, RTX 5090, RAM – 256 GB, PyTorch (Vast) template)

#### 1. Run setup script

```bash
bash setup.sh
```

This script handles everything in one shot — clone, checkout, all Python dependencies, PyTorch cu128 nightly, and a verification check at the end. The full script is reproduced below for reference:

```bash
#!/bin/bash
set -e  
echo "Cloning repo"
git clone https://github.com/Aurora-source/zero-latency.git
cd zero-latency

echo "Installing dependencies (excluding torch)"
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

echo "    Installing PyTorch (cu128 nightly)    "
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

echo "    Installing remaining nuscenes-devkit deps    "
pip install cachetools descartes fire \
  opencv-python-headless parameterized pycocotools \
  pyquaternion scikit-learn scipy shapely

echo "    Verifying install    "
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
from nuscenes.nuscenes import NuScenes
print('nuscenes: OK')
"

echo "    Setup complete!    "
echo "Next: download nuscenes data using aria2 with links.txt"
echo "Then run: NUSCENES_ROOT=/workspace/zero-latency/nuscenes python train.py"
```

#### 2. Install FFmpeg

FFmpeg is usually pre-installed on most cloud images. Verify with:

```bash
ffmpeg -version
```

If it is missing:

```bash
apt-get update && apt-get install -y ffmpeg
```

#### 3. Set up rclone ([Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing) access)

rclone is used on the cloud server to pull the latest checkpoint from [Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing) before training starts, and to push the new best checkpoint back up when training finishes. On Windows you don't need rclone — just use [Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing) in the browser directly.

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config
# Follow the prompts:
#   n          → new remote
#   name:        Rikon  (or your preferred name — use this name in all rclone commands)
#   Storage:     drive
#   client_id:   (leave blank)
#   client_secret: (leave blank)
#   scope:       1  (full access)
#   auto config: n  (we are on a remote server with no browser)
#   Copy the URL shown → open it in your local browser → approve access → paste the code back
#   Shared Drive: n
#   Confirm with: y
```

#### 4. Download nuScenes dataset

`links.txt` at `data/links.txt` has the links to all 10 parts of trainval dataset, as well the metadata. We use aria to manage the downloads.

```bash
apt-get install -y aria2

cd /workspace/zero-latency/nuscenes

# Download all parts using links.txt from the repo
aria2c -c -j 2 -x 8 -s 8 -k 1M --file-allocation=falloc --dir=. \
  -i /workspace/zero-latency/nuscenes/links.txt
```

#### 5. Extract dataset (one part at a time to save disk space)

We don’t use the nuscenes-mini dataset on cloud, only trainval dataset.

```bash
tar -xzf v1.0-trainval_meta.tgz    && rm v1.0-trainval_meta.tgz
tar -xzf v1.0-trainval01_blobs.tgz && rm v1.0-trainval01_blobs.tgz
tar -xzf v1.0-trainval02_blobs.tgz && rm v1.0-trainval02_blobs.tgz
# repeat for remaining parts...
```

Expected structure after extraction:

```
nuscenes/
├── maps/
│   └── prediction/
│       └── prediction_scenes.json
├── samples/
├── sweeps/
└── v1.0-trainval/
```

#### 6. Download latest checkpoint

```bash
rclone copy "Rikon:zero-latency/checkpoints/best_1.pt" \
  /workspace/zero-latency/checkpoints/
```

---

### How to run – 

All scripts provided can be run on any platform, any device and any of nuscenes dataset ( mini / trainval ) you simply need to set the right env variable for that. If that’s too much pain then stick to using the default values, i.e. let’s say to run evaluate-mini.py it by default only uses the mini dataset. Go to Key environment variables at the bottom for more info.

### Exporting model weights {requires checkpoints/best_1.pt to be present} (Optional step if models downloaded from drive)

Before running inference, export the raw training checkpoint to inference-ready formats ( but there is no need if you have already download the exported models from models folder in our [Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing) and placed them at models/ folder )  :

```powershell
python export_model.py
```

This reads `checkpoints/best_1.pt` and writes two files into `models/`:

- `model_fp32.pt` — full-precision weights (~1 GB)
- `model_fp16.pt` — half-precision weights (~500 MB)


### Single-scene inference and visualisation { requires Nuscenes mini or trainval dataset and models/ ( model_fp32.pt is default ) }

```powershell
python single_inference.py                     # mini dataset, random scene
python single_inference.py --trainval          # use trainval dataset instead
python single_inference.py --seed 42           # reproducible scene
python single_inference.py --no_anim           # skip MP4, PNG only
```

Running the above, that one scene random from the available 404 scenes in the nuscenes dataset { mini or trainval , default mini } , in the output its has 3 major blocks showing the ground reality {also showing if agent is static or moving } (what actually happend ) , RAW OUTPUT SHAPES and Per-Agent Matrics (ADE and FDE for predicted values for each agent) , finally caluculates the MEAN ADE and FDE then saves:

- `visualisations/multi_agent_prediction.png` — static visualisation, 
- `visualisations/multi_agent_prediction.mp4` — animated visualisation, trajectories grow step by step (requires FFmpeg)

To view the generated graph / animated graph for the specific scene please go to visualisations folder after running the command. 

#### IMPORTANT NOTE :
To reproduce a specific scene, pass `--seed` with the sample index printed at the end of any previous run, or the ADE, FDE and graph will vary according to the scene and the motion in it :

```powershell
python single_inference.py --seed 177
# printed as: "To reproduce this scene: set SEED = 177"
```
##Sample OUTPUT :
To reproduce this exact scene run "python single_inference.py --seed 5575121"
```powershell
+++++++++++++++++++++++++++++++++++++++++++++++++++++++
Zero-Latency — Single Scene Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++
  repo root  : D:\zero-latency
  model      : D:\zero-latency\models\model_fp32.pt
  dataset    : v1.0-mini  (D:\zero-latency\data\raw\nuscenes)
  outputs    : D:\zero-latency\visualisations
  seed       : 5575121

device=cuda
Loading model: D:\zero-latency\models\model_fp32.pt
Model loaded.

Loading dataset: v1.0-mini
Total samples: 404

Searching for a random moving scene (min_movement=5.0m)...
SEED=5575121
Found moving scene at index 55 (checked 3 samples)

    GROUND TRUTH MOVEMENT
  Agent  0 [[V] vehicle (Agent 0)]:   0.09 m  [static]
  Agent  1 [[V] vehicle (Agent 1)]:   0.14 m  [static]
  Agent  2 [[P] pedestrian (Agent 2)]:   6.97 m  [moving]
  Agent  3 [[P] pedestrian (Agent 3)]:   6.88 m  [moving]
  Agent  4 [[P] pedestrian (Agent 4)]:   7.32 m  [moving]
  Agent  5 [[P] pedestrian (Agent 5)]:   7.24 m  [moving]
  Agent  6 [[P] pedestrian (Agent 6)]:   8.34 m  [moving]
  Agent  7 [[P] pedestrian (Agent 7)]:   8.49 m  [moving]
  Agent  8 [[V] vehicle (Agent 8)]:   0.12 m  [static]
  Agent  9 [[V] vehicle (Agent 9)]:   3.53 m  [static]

  Moving agents: [2, 3, 4, 5, 6, 7]

Running inference...

   RAW OUTPUT SHAPES
  traj  : (1, 10, 6, 12, 2)
  goals : (1, 10, 6, 2)
  probs : (1, 10, 6)

    PER-AGENT METRICS
  Agent  0 [V] [vehicle] | best_mode(minADE)=2 p=0.140 | ADE=0.115m  FDE=0.116m
  Agent  1 [V] [vehicle] | best_mode(minADE)=2 p=0.136 | ADE=0.315m  FDE=0.293m
  Agent  2 [P] [pedestrian] | best_mode(minADE)=0 p=0.131 | ADE=1.168m  FDE=1.246m
  Agent  3 [P] [pedestrian] | best_mode(minADE)=0 p=0.131 | ADE=0.860m  FDE=0.538m
  Agent  4 [P] [pedestrian] | best_mode(minADE)=3 p=0.133 | ADE=0.361m  FDE=0.240m
  Agent  5 [P] [pedestrian] | best_mode(minADE)=3 p=0.139 | ADE=0.605m  FDE=0.795m
  Agent  6 [P] [pedestrian] | best_mode(minADE)=2 p=0.132 | ADE=0.502m  FDE=0.744m
  Agent  7 [P] [pedestrian] | best_mode(minADE)=3 p=0.138 | ADE=0.518m  FDE=0.329m
  Agent  8 [V] [vehicle] | best_mode(minADE)=3 p=0.131 | ADE=0.244m  FDE=0.286m
  Agent  9 [V] [vehicle] | best_mode(minADE)=5 p=0.132 | ADE=1.288m  FDE=2.049m

  Mean ADE : 0.5975 m
  Mean FDE : 0.6637 m

Static PNG saved → D:\zero-latency\visualisations\multi_agent_prediction.png

Building animation (FPS=20, speed=0.25×, DPI=120) → D:\zero-latency\visualisations\multi_agent_prediction.mp4
Animation saved → D:\zero-latency\visualisations\multi_agent_prediction.mp4

To reproduce this scene: python single_inference.py --seed 5575121
Done.
```
### Training {requires Nuscenes mini or trainval dataset }

Running these scripts produces best_1.pt , best_2.pt and latest.pt in the checkpoints folder. Our general concern is only best_1.pt ( which gets only when the model is able to beat the best val_loss value, best_2.pt keeps second best model for backup in case the best_1.pt gets corrupted ).

**Windows (local, by default uses mini dataset):**
```powershell
python train-windows-8GB-VRAM.py
```
Training automatically resumes from `checkpoints/best_1.pt`.If best checkpoint file not found then the tranning starts from scratch, if best_1.pt present and you want to force tranning from scratch just use :

```powershell
$env:RESUME="0"; python train-windows-8GB-VRAM.py
```
Our tranning script has many features (to control batch sizes, eval batch sizes, lr and other things) that can be controlled through env variables both in linux and windows refer to configs section at the start of the readme or the Key environment variables section at the bottom.

**Cloud (Linux, by default uses trainval dataset) — one run takes ~2 hours on RTX 5090:**
```bash
NUSCENES_ROOT=/workspace/zero-latency/nuscenes \
TORCH_COMPILE_MODE=reduce-overhead \
BATCH_SIZE=72 \
EVAL_BATCH_SIZE=16 \
python train-linux-32GB-VRAM.py
```

Training automatically resumes from `checkpoints/best_1.pt`. To start from scratch:

```bash
RESUME=0 python train-linux-32GB-VRAM.py
```
Similarly all other env variable presented at the bottom of the readme can be used to modify the running config.

IMPORTANT – please don’t try to run this script as it requires a very huge ram , lets say you even want to work with 1 part of trainval data i.e 30 GB  you would require atleast 45 GB RAM and recommended 64 GB RAM to run, the script uses ram – caching i.e it caches the entire dataset to ram, for best performance. Though you can use the DATASET_LIMIT env to limit the dataset then run it safely.

### Evaluation { requires Nuscenes mini or trainval dataset and model or checkpoint - default model }

**Mini dataset:**
```powershell
python evaluate-mini.py
```

**Full trainval dataset:**
```powershell
python evaluate-trainval.py --dataroot "nuscenes" --batch_size 32
```

Arguments -–dataroot and --batch_size can used , or you can refer to configs section present at the top of the readme to change the default configurations. 

Saves plots to `evaluation_results/evaluation_results.png`.

To see the generated graphs please visit the evaluation_results folder after running the command 

### Syncing checkpoints via [Google Drive](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing)

**On the cloud server — pull the previous best checkpoint before training:**

```bash
rclone copy "Rikon:zero-latency/checkpoints/best_1.pt" \
  /workspace/zero-latency/checkpoints/ --progress
```

**On the cloud server — push the new best checkpoint after training finishes:**

```bash
rclone copy /workspace/zero-latency/checkpoints/best_1.pt \
  Rikon:zero-latency/checkpoints/ --progress
```

**On Windows** — no rclone needed. Open the [shared Google Drive folder](https://drive.google.com/drive/folders/16s7dJhrjQLzVtm-OpdlNWsWP6TRgp2OP?usp=sharing) in your browser to download weights, or navigate to `zero-latency/checkpoints/` in your own Drive to upload a new checkpoint after cloud training.

---

## Training configuration

### Key environment variables

Both training scripts read configuration from environment variables. Defaults differ between scripts — Windows targets the mini dataset, Linux targets the full trainval. Lets say you want to train using windows on the trainval dataset just change the NUSCENES_ROOT, NUSCENES_VERSION to your desired values. Defaults for all env variables mentioned :  

| Variable | Windows default | Linux default | Description |
|---|---|---|---|
| `NUSCENES_ROOT` | `data/raw/nuscenes` | `nuscenes` | Path to nuScenes dataset |
| `NUSCENES_VERSION` | `v1.0-mini` | `v1.0-trainval` | Dataset version |
| `CHECKPOINT_DIR` | `checkpoints` | `checkpoints` | Where to save/load checkpoints |
| `DATASET_LIMIT` | `404` | `999999` | Max samples to load per run |
| `RUN_EPOCHS` | `40` | `40` | Epochs per run |
| `RESUME` | `1` | `1` | Auto-resume from best checkpoint |
| `TORCH_COMPILE` | — | `1` | Enable torch.compile (Linux only) |

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 32 GB |
| RAM | 16 GB | 64 GB+ |
| Disk | 50 GB | 500 GB |
| CUDA | 12.8+ | 13.0+ |

### Tested configurations

| Setup | Script | Batch size | Time per run |
|---|---|---|---|
| RTX 5060 Laptop (8 GB VRAM, 16 GB RAM, Windows) | `train-windows-8GB-VRAM.py` | 4 | ~15 min (mini) |
| RTX 5070 Laptop (8 GB VRAM, 16 GB RAM, Windows) | `train-windows-8GB-VRAM.py` | 4 | ~12 min (mini) |
| RTX 5090 Cloud (32 GB VRAM ,256 GB RAM , Linux) | `train-linux-32GB-VRAM.py` | 72 | ~2 hours (trainval) |

