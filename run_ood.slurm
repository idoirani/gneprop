#!/bin/bash
#SBATCH --job-name=gneprop_ood
#SBATCH --output=ood_%j.out
#SBATCH --error=ood_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# Novel MOA Detection using OOD (Out-of-Distribution) Detection
# =============================================================================
#
# This script runs the Mahalanobis distance-based OOD detection pipeline
# to identify molecules with potentially novel mechanisms of action.
#
# Required inputs:
#   1. Known antibiotics CSV with SMILES and target labels
#   2. Query molecules CSV with SMILES
#   3. Self-supervised (SimCLR) checkpoint
#
# Output:
#   CSV with OOD scores for each query molecule (lower = more novel)
#
# =============================================================================

# Paths - MODIFY THESE FOR YOUR SETUP
export GNEPROP_CODE=/home/ii3409/GNEprop/gneprop
export GNEPROP_DATA=/scratch/gpfs/GITAI/ido/GNEprop
export GNEPROP_CHECK=/scratch/gpfs/GITAI/ido/GNEprop/checkpoints

# Self-supervised checkpoint path (the SimCLR pretrained model)
# This is the model trained on ~120M molecules from ZINC15
# Checkpoint 20210827-082422 is the self-supervised model
SSL_CHECKPOINT=$GNEPROP_CHECK/20210827-082422/version_0/checkpoints/epoch=26-step=2973239.ckpt

# Known antibiotics file options:
# Option 1: Use the original ~200 antibiotics from the paper
#           Note: The original file is Extended_data_table_known_antibiotics.xlsx
#           You may need to convert it to CSV first (see convert_antibiotics_xlsx.py)
# KNOWN_ANTIBIOTICS=$GNEPROP_DATA/support_data/Extended_data_table_known_antibiotics.csv

# Option 2: Use your extended ~800 compound list (816 compounds, 38 targets)
#           Columns: SMILES, Target (mechanism of action)
KNOWN_ANTIBIOTICS=$GNEPROP_DATA/Data/known_ab_for_gneprop.csv

# Query molecules to evaluate for novel MoA (320 compounds)
QUERY_MOLECULES=$GNEPROP_DATA/Data/PU_compounds.csv

# Output path
OUTPUT_PATH=$GNEPROP_DATA/predictions/ood_scores_PU_compounds.csv

# Column names (adjust based on your CSV structure)
SMILES_COL="SMILES"                    # For known antibiotics
QUERY_SMILES_COL="SMILES_FROM_INCHI"   # For query molecules (PU_compounds.csv)
TARGET_COL="Target"                    # Column with MoA/target labels in known antibiotics file

# =============================================================================

# Load modules
module purge
module load anaconda3/2024.2
module load cudatoolkit/12.6

# Activate environment
conda activate gneprop

# Change to code directory
cd $GNEPROP_CODE

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Verify all paths exist before running
echo "=== Verifying paths ==="
errors=0

if [ ! -d "$GNEPROP_CODE" ]; then
    echo "ERROR: Code directory not found: $GNEPROP_CODE"
    errors=$((errors + 1))
else
    echo "OK: Code directory found"
fi

if [ ! -f "$SSL_CHECKPOINT" ]; then
    echo "ERROR: SSL checkpoint not found: $SSL_CHECKPOINT"
    echo "       Please download the self-supervised checkpoint or update the path"
    errors=$((errors + 1))
else
    echo "OK: SSL checkpoint found"
fi

if [ ! -f "$KNOWN_ANTIBIOTICS" ]; then
    echo "ERROR: Known antibiotics file not found: $KNOWN_ANTIBIOTICS"
    errors=$((errors + 1))
else
    num_known=$(wc -l < "$KNOWN_ANTIBIOTICS")
    echo "OK: Known antibiotics file found ($((num_known - 1)) compounds)"
fi

if [ ! -f "$QUERY_MOLECULES" ]; then
    echo "ERROR: Query molecules file not found: $QUERY_MOLECULES"
    errors=$((errors + 1))
else
    num_query=$(wc -l < "$QUERY_MOLECULES")
    echo "OK: Query molecules file found ($((num_query - 1)) compounds)"
fi

if [ $errors -gt 0 ]; then
    echo "=== $errors error(s) found. Exiting. ==="
    exit 1
fi
echo "=== All paths verified ==="

# Run OOD detection
echo ""
echo "=== Starting Novel MOA Detection ==="
echo "Known antibiotics: $KNOWN_ANTIBIOTICS"
echo "Query molecules: $QUERY_MOLECULES"
echo "SSL checkpoint: $SSL_CHECKPOINT"
echo "Output: $OUTPUT_PATH"
echo ""

python run_ood_detection.py \
    --known_antibiotics_path "$KNOWN_ANTIBIOTICS" \
    --query_path "$QUERY_MOLECULES" \
    --ssl_checkpoint_path "$SSL_CHECKPOINT" \
    --output_path "$OUTPUT_PATH" \
    --smiles_col "$SMILES_COL" \
    --query_smiles_col "$QUERY_SMILES_COL" \
    --target_col "$TARGET_COL" \
    --batch_size 128 \
    --num_workers 8 \
    --gpus 1 \
    --save_representations \
    --save_centroids

echo ""
echo "=== OOD Detection completed at $(date) ==="
echo "Results saved to: $OUTPUT_PATH"
