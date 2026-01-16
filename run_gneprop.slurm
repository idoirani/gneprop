#!/bin/bash
#SBATCH --job-name=gneprop
#SBATCH --output=gneprop_%j.out
#SBATCH --error=gneprop_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Paths
export GNEPROP_CODE=/home/ii3409/GNEprop/gneprop
export GNEPROP_DATA=/scratch/gpfs/GITAI/ido/GNEprop

# Load modules
module purge
module load anaconda3/2024.2
module load cudatoolkit/12.6

# Activate environment
conda activate gneprop

# Change to code directory
cd $GNEPROP_CODE

# Create output directory
mkdir -p $GNEPROP_DATA/predictions

# Run inference - modify data_path as needed
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/New_sets_princeton_all_largest.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --save_path $GNEPROP_DATA/predictions/predictions_largest.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1

echo "Job completed at $(date)"
