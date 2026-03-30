#!/bin/bash
set -e  
echo "Cloning repo"
git clone https://github.com/Aurora-source/zero-latency.git
cd zero-latency
git checkout working-rikon

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

echo "   Installing PyTorch (cu128 nightly)    "
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

echo "   Installing remaining nuscenes-devkit deps   "
pip install cachetools descartes fire \
  opencv-python-headless parameterized pycocotools \
  pyquaternion scikit-learn scipy shapely

echo "   Verifying install   "
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
from nuscenes.nuscenes import NuScenes
print('nuscenes: OK')
"

echo "   Setup complete!    "
echo "Next: download nuscenes data using aria2 with links.txt"
echo "Then run: NUSCENES_ROOT=/workspace/zero-latency/nuscenes python train.py"
