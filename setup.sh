#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Starting Unified Eraser Agent Setup ---"

if ! command -v conda &> /dev/null
then
    echo "Miniconda not found. Installing..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
else
    echo "Miniconda already installed."
fi

if ! command -v git &> /dev/null
then
    echo "Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

eval "$(conda shell.bash hook)"
conda init

echo "--- Creating Unified Environment 'eraser_env' ---"

conda create -n eraser_env python=3.10 -y
conda activate eraser_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Setup Grounded-SAM-2
echo "--- Setting up Grounded-SAM-2 ---"
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install transformers==4.36.2 supervision pycocotools

# Moving download scripts
cd checkpoints
if [ -f "../../download_singleSAMCkpt.sh" ]; then
    mv ../../download_singleSAMCkpt.sh .
else
    echo "Warning: download_singleSAMCkpt.sh not found in parent directory."
fi
bash download_singleSAMCkpt.sh
cd ..
cd gdino_checkpoints
if [ -f "../../download_singleDinoCkpt.sh" ]; then
    mv ../../download_singleDinoCkpt.sh .
else
    echo "Warning: download_singleDinoCkpt.sh not found in parent directory."
fi
bash download_singleDinoCkpt.sh
cd ..

cd ..

# Setup LaMa
echo "--- Setting up LaMa ---"
git clone https://github.com/advimman/lama.git
pip install pytorch-lightning==1.2.9 omegaconf albumentations==0.5.2 \
            easydict pyyaml scikit-image opencv-python-headless webdataset

pip install fastapi uvicorn pydantic python-multipart
pip install "numpy<2.0"
pip install pandas scikit-learn kornia addict yapf timm

# LaMa Checkpoints
cd lama
echo "Downloading LaMa model..."
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip

echo "--- Setup Complete! ---"