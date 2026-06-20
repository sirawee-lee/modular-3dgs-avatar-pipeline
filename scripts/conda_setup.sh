#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda create -n hugs python=3.8 -y

conda activate hugs

conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# Clone Gaussian Splatting CUDA submodules if not present
if [ ! -d "submodules/diff-gaussian-rasterization" ]; then
    echo "Cloning diff-gaussian-rasterization..."
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
fi
if [ ! -d "submodules/simple-knn" ]; then
    echo "Cloning simple-knn..."
    git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
fi

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy.git

# LLM normalization client (required for prompt normalization stage)
pip install ollama
