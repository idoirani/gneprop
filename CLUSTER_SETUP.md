# GNEprop Cluster Setup Guide (Princeton Della)

## Directory Structure

```
Code:  /home/ii3409/GNEprop/gneprop/
Data:  /scratch/gpfs/GITAI/ido/GNEprop/
       ├── checkpoints/
       ├── Data/
       └── support_data/
```

## Files to Transfer

### Required (Minimal Setup) - ~1.5GB total

```
GNEprop/
├── gneprop_pyg.py          # Main module
├── clr.py                  # SimCLR module (imported by gneprop_pyg)
├── models.py               # Model architectures
├── data.py                 # Data loading
├── chemprop_featurizer.py  # Feature extraction
├── run_inference.py        # Inference script (created for you)
├── environment.yml         # Conda environment
│
├── gneprop/                # Core library (112KB)
│   ├── __init__.py
│   ├── augmentations.py
│   ├── chem_utils.py
│   ├── custom_objects.py
│   ├── explainability.py
│   ├── featurization.py
│   ├── image_utils.py
│   ├── ood.py
│   ├── options.py
│   ├── plot_utils.py
│   ├── scaffold.py
│   ├── utils.py
│   └── chemprop/
│       ├── __init__.py
│       └── features.py
│
├── checkpoints/
│   └── 20250811-202022/    # 8-fold ensemble (1.3GB)
│       ├── 20250811-202022-fold_0/checkpoints/*.ckpt
│       ├── 20250811-202022-fold_1/checkpoints/*.ckpt
│       ├── ... (folds 2-7)
│
└── Data/                   # Your data files (~20MB for CSVs)
    ├── New_sets_princeton_all_largest.csv
    ├── activity_known_compounds_princeton_all.csv
    ├── activity_known_ablated_1_compounds_princeton_all.csv
    └── New_sets_princeton_inc_singletons_random.csv
```

### Quick Transfer Command (from your Mac)

```bash
# Create a tar archive of required files (excludes zip backups)
cd /Users/ido/Dropbox/Bio/Antibiotics/GNEprop

tar -cvzf gneprop_transfer.tar.gz \
    gneprop_pyg.py clr.py models.py data.py chemprop_featurizer.py \
    run_inference.py environment.yml \
    gneprop/ \
    checkpoints/20250811-202022/ \
    --exclude='*.zip' \
    Data/*.csv

# Transfer to cluster
scp gneprop_transfer.tar.gz user@cluster:/path/to/destination/
```

---

## Cluster Setup

### 1. Extract Files

```bash
cd /path/to/destination
tar -xvzf gneprop_transfer.tar.gz
cd GNEprop  # or wherever files extracted
```

### 2. Create Conda Environment

```bash
# Load conda if needed (cluster-specific)
module load anaconda3  # or similar

# Create environment from yml
conda env create -f environment.yml

# Activate
conda activate gneprop
```

### 3. Install learn2learn (optional, for meta-learning)

```bash
pip install learn2learn
```

---

## Running Inference

### Setup paths (add to ~/.bashrc or run each session)

```bash
export GNEPROP_CODE=/home/ii3409/GNEprop/gneprop
export GNEPROP_DATA=/scratch/gpfs/GITAI/ido/GNEprop
```

### Basic Usage

```bash
# Activate environment
conda activate gneprop
cd $GNEPROP_CODE

# Create output directory
mkdir -p $GNEPROP_DATA/predictions

# Run inference on your data
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/New_sets_princeton_all_largest.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_largest.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1
```

### Run All Your Data Files

```bash
cd $GNEPROP_CODE
mkdir -p $GNEPROP_DATA/predictions

# File 1: New_sets_princeton_all_largest.csv (~59K molecules)
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/New_sets_princeton_all_largest.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_largest.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1

# File 2: activity_known_compounds_princeton_all.csv (~52K molecules)
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/activity_known_compounds_princeton_all.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_activity_known.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1

# File 3: activity_known_ablated_1_compounds_princeton_all.csv
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/activity_known_ablated_1_compounds_princeton_all.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_ablated.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1

# File 4: New_sets_princeton_inc_singletons_random.csv
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/New_sets_princeton_inc_singletons_random.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_singletons.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1
```

### SLURM Job Script (for Della)

Create `run_gneprop.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=gneprop_inference
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

# Activate environment
conda activate gneprop

# Change to code directory
cd $GNEPROP_CODE

# Create output directory
mkdir -p $GNEPROP_DATA/predictions

# Run inference
python run_inference.py \
    --data_path $GNEPROP_DATA/Data/New_sets_princeton_all_largest.csv \
    --checkpoint_dir $GNEPROP_DATA/checkpoints/20250811-202022 \
    --output $GNEPROP_DATA/predictions/predictions_largest.csv \
    --smiles_col SMILES_FROM_INCHI \
    --target_col binarized \
    --gpus 1
```

Submit with:
```bash
sbatch run_gneprop.slurm
```

---

## Output Format

The inference script produces a CSV with all original columns plus:

| Column | Description |
|--------|-------------|
| `prediction` | Mean prediction across 8-fold ensemble (0-1 for binary) |
| `epistemic_uncertainty` | Variance across ensemble (higher = less confident) |

If the target column exists, it also prints ROC-AUC and Average Precision metrics.

---

## Troubleshooting

### CUDA out of memory
Reduce batch size:
```bash
python run_inference.py ... --batch_size 64
```

### Module not found errors
Make sure you're in the GNEprop directory:
```bash
cd /path/to/GNEprop
python run_inference.py ...
```

### RDKit import errors
Ensure environment is activated:
```bash
conda activate gneprop
```

### Checkpoint not found
Verify checkpoint structure:
```bash
ls checkpoints/20250811-202022/*/checkpoints/*.ckpt
```

---

## Expected Runtime

| Dataset Size | Approximate Time (1 GPU) |
|--------------|-------------------------|
| 10K molecules | ~5 minutes |
| 50K molecules | ~15-20 minutes |
| 100K molecules | ~30-40 minutes |

(Times include loading 8 checkpoints sequentially)
