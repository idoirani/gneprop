#!/bin/bash
#SBATCH --job-name=gneprop_finetune
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Paths
export GNEPROP_CODE=/home/ii3409/GNEprop/gneprop
export GNEPROP_DATA=/scratch/gpfs/GITAI/ido/GNEprop/Data
export GNEPROP_CHECK=/scratch/gpfs/GITAI/ido/GNEprop/checkpoints

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

# Verify all paths exist before running
echo "=== Verifying paths ==="
DATASET_PATH=$GNEPROP_DATA/training_all_known_ab_clusters_with_target.csv
CHECKPOINT_DIR=$GNEPROP_CHECK/20250811-202022

errors=0

if [ ! -d "$GNEPROP_CODE" ]; then
    echo "ERROR: Code directory not found: $GNEPROP_CODE"
    errors=$((errors + 1))
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found: $DATASET_PATH"
    errors=$((errors + 1))
else
    echo "OK: Dataset found ($DATASET_PATH)"
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    errors=$((errors + 1))
else
    # Count fold directories
    NUM_FOLDS=$(ls -d $CHECKPOINT_DIR/*fold_* 2>/dev/null | wc -l)
    echo "OK: Checkpoint directory found with $NUM_FOLDS folds ($CHECKPOINT_DIR)"
fi

if [ $errors -gt 0 ]; then
    echo "=== $errors error(s) found. Exiting. ==="
    exit 1
fi
echo "=== All paths verified ==="

# Fine-tune for higher recall
python gneprop_pyg.py \
    --dataset_path $GNEPROP_DATA/training_all_known_ab_clusters_with_target.csv \
    --supervised_pretrain_path_folds $GNEPROP_CHECK/20250811-202022 \
    --target_name activity_int \
    --split_type scaffold \
    --num_readout_layers 1 \
    --hidden_size 500 \
    --depth 5 \
    --auto_loss_weight \
    --balanced_batch_training \
    --freeze_ab_embeddings \
    --mp_to_freeze 1 \
    --lr 1e-4 \
    --max_epochs 50 \
    --metric val_ap \
    --use_early_stopping \
    --gpus 1 \
    --num_folds 8 \
    --log_directory $GNEPROP_DATA/finetune_logs

echo "Fine-tuning completed at $(date)"
