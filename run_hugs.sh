#!/bin/bash
# Script to run HUGS with proper environment
# This ensures all modules are available

# Activate hugs conda environment
source /home/sigma/anaconda3/etc/profile.d/conda.sh
conda activate hugs

# Set Python path to include compiled extensions
export PYTHONPATH="/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/submodules/diff-gaussian-rasterization:/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/submodules/simple-knn:$PYTHONPATH"

# Go to project directory
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

# Run HUGS with the correct Python from hugs environment
/home/sigma/anaconda3/envs/hugs/bin/python main.py "$@"
