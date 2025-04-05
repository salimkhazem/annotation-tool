#!/bin/bash

# Create directories
mkdir -p checkpoints
cd checkpoints
curl -O https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
bash download_ckpts.sh
cd .. 
echo "current directory is: $(pwd)"
echo "All models downloaded successfully!"

