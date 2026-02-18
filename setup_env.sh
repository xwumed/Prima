#!/bin/bash
# Setup script for Prima environment using uv

# Install uv if not present (requires curl)
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment with Python 3.10
echo "Creating virtual environment..."
uv venv .venv --python 3.10 --allow-existing

# Activate environment
source .venv/bin/activate

# Install build dependencies first (still good to have)
echo "Installing build dependencies..."
uv pip install torch==2.6.0 setuptools wheel packaging ninja

# Flash attention is optional and skipped due to missing CUDA build tools
# The model will fallback to standard attention
echo "Skipping flash-attn installation (using fallback)..."

# Install remaining dependencies (excluding flash-attn and torch to avoid issues)
echo "Installing other dependencies..."
# Explicitly install perceiver-pytorch to ensure it's not missed
uv pip install perceiver-pytorch==0.8.8

# Install gdown for model downloading
echo "Installing gdown..."
uv pip install gdown

grep -vE "flash-attn|torch==" requirements.txt > requirements_no_flash.txt
uv pip install -r requirements_no_flash.txt
rm requirements_no_flash.txt

# Install additional dependencies for feature extraction script if needed
# (Assuming numpy is already in requirements or transitive)

echo "Environment setup complete."
echo "To activate: source .venv/bin/activate"
