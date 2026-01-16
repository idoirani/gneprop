#!/bin/bash
# GNEprop setup script for Princeton Della cluster
#
# Usage:
#   chmod +x setup_della.sh
#   ./setup_della.sh

set -e

echo "=== GNEprop Della Setup ==="

# Load required modules
echo "Loading modules..."
module purge
module load anaconda3/2024.2
module load cudatoolkit/12.6

echo "Loaded modules:"
module list

# Check if environment already exists
if conda env list | grep -q "^gneprop "; then
    echo ""
    echo "Environment 'gneprop' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n gneprop -y
    else
        echo "Skipping environment creation. Activate with: conda activate gneprop"
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment (this may take 10-15 minutes)..."
conda env create -f environment_della.yml

# Install DGL separately (conda package has issues)
echo ""
echo "Installing DGL via pip..."
conda activate gneprop
pip install dgl -f https://data.dgl.ai/wheels/torch-2.0/cu121/repo.html

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use GNEprop:"
echo "  module load anaconda3/2024.2 cudatoolkit/12.6"
echo "  conda activate gneprop"
echo ""
echo "Add this to your ~/.bashrc for convenience:"
echo '  alias gneprop_env="module load anaconda3/2024.2 cudatoolkit/12.6 && conda activate gneprop"'
