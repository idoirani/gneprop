#!/bin/bash
#SBATCH --job-name=gneprop_finetune
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --time=06:00:00
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

# Create log directory
mkdir -p $GNEPROP_DATA/finetune_logs

# Fine-tune for higher recall
python gneprop_pyg.py \
    --dataset_path $GNEPROP_DATA/training_all_known_ab_clusters_with_target.csv \
    --supervised_pretrain_path $GNEPROP_DATA/checkpoints/20250811-202022/20250811-202022-fold_0/checkpoints/epoch=48-step=1845193.ckpt \
    --target_name active \
    --split_type scaffold \
    --use_mol_features \
    --auto_loss_weight \
    --balanced_batch_training \
    --freeze_ab_embeddings \
    --mp_to_freeze 1 \
    --lr 1e-4 \
    --max_epochs 50 \
    --metric val_ap \
    --use_early_stopping \
    --gpus 1 \
    --log_directory $GNEPROP_DATA/finetune_logs

echo "Fine-tuning completed at $(date)"
