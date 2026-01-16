#!/usr/bin/env python
"""
GNEprop Inference Script
========================
Run predictions on molecular datasets using a trained GNEprop ensemble checkpoint.

Directory structure at Princeton Della:
    Code: /home/ii3409/GNEprop/gneprop/
    Data: /scratch/gpfs/GITAI/ido/GNEprop/

Usage:
    # From the code directory
    cd /home/ii3409/GNEprop/gneprop

    python run_inference.py \
        --data_path /scratch/gpfs/GITAI/ido/GNEprop/Data/New_sets_princeton_all_largest.csv \
        --checkpoint_dir /scratch/gpfs/GITAI/ido/GNEprop/checkpoints/20250811-202022 \
        --save_path /scratch/gpfs/GITAI/ido/GNEprop/predictions/predictions_largest.csv

Arguments:
    --data_path: Path to CSV file with SMILES column
    --checkpoint_dir: Directory containing fold checkpoints
    --output: Output CSV file path for predictions
    --smiles_col: Name of SMILES column in input (default: SMILES_FROM_INCHI)
    --target_col: Name of target column if present (default: binarized)
    --batch_size: Batch size for inference (default: 128)
    --gpus: Number of GPUs to use (default: 1)
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch

# Import GNEprop modules
from data import load_dataset_multi_format, MolDatasetOD
from gneprop_pyg import GNEprop, predict_ensemble
import gneprop.utils


def get_fold_checkpoints(checkpoint_dir):
    """
    Get all checkpoint paths from a multi-fold checkpoint directory.
    Handles the naming convention used in 20250811-202022.
    """
    checkpoints = []

    # Pattern 1: fold directories like "20250811-202022-fold_0/checkpoints/*.ckpt"
    pattern1 = os.path.join(checkpoint_dir, "*fold_*/checkpoints/*.ckpt")
    checkpoints.extend(glob.glob(pattern1))

    # Pattern 2: fold directories like "fold_0/version_0/checkpoints/*.ckpt"
    pattern2 = os.path.join(checkpoint_dir, "fold_*/version_*/checkpoints/*.ckpt")
    checkpoints.extend(glob.glob(pattern2))

    # Pattern 3: direct checkpoints in subdirectories
    pattern3 = os.path.join(checkpoint_dir, "*/checkpoints/*.ckpt")
    checkpoints.extend(glob.glob(pattern3))

    # Remove duplicates and sort
    checkpoints = sorted(list(set(checkpoints)))

    if not checkpoints:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    return checkpoints


def run_inference(data_path, checkpoint_dir, output_path, smiles_col='SMILES_FROM_INCHI',
                  target_col='binarized', batch_size=128, gpus=1):
    """
    Run inference on a dataset using an ensemble of checkpoints.

    Returns predictions and epistemic uncertainty (variance across ensemble).
    """
    print(f"Loading data from: {data_path}")
    print(f"SMILES column: {smiles_col}")
    print(f"Target column: {target_col}")

    # Load dataset
    dataset = load_dataset_multi_format(
        data_path,
        smiles_name=smiles_col,
        target_name=target_col,
        legacy=True
    )

    print(f"Loaded {len(dataset)} molecules")

    # Get checkpoint paths
    checkpoints = get_fold_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoint files:")
    for ckpt in checkpoints:
        print(f"  - {os.path.basename(os.path.dirname(os.path.dirname(ckpt)))}/{os.path.basename(ckpt)}")

    # Run ensemble prediction
    print("\nRunning ensemble predictions...")
    predictions, uncertainties = predict_ensemble(
        list_checkpoints=checkpoints,
        data=dataset,
        aggr='mean',
        gpus=gpus
    )

    # Load original data to preserve all columns
    original_df = pd.read_csv(data_path)

    # Add predictions to dataframe
    original_df['prediction'] = predictions
    original_df['epistemic_uncertainty'] = uncertainties

    # Save results
    original_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Print summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total molecules: {len(predictions)}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Std prediction: {predictions.std():.4f}")
    print(f"Min prediction: {predictions.min():.4f}")
    print(f"Max prediction: {predictions.max():.4f}")
    print(f"Mean uncertainty: {uncertainties.mean():.6f}")

    # If target column exists, compute metrics
    if target_col in original_df.columns and dataset.y is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score

        y_true = dataset.y
        if len(np.unique(y_true)) == 2:  # Binary classification
            try:
                auc = roc_auc_score(y_true, predictions)
                ap = average_precision_score(y_true, predictions)
                print(f"\n=== Performance Metrics ===")
                print(f"ROC-AUC: {auc:.4f}")
                print(f"Average Precision (AUPRC): {ap:.4f}")
            except Exception as e:
                print(f"Could not compute metrics: {e}")

    return predictions, uncertainties


def main():
    parser = argparse.ArgumentParser(description='GNEprop Inference Script')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file with SMILES')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--save_path', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--smiles_col', type=str, default='SMILES_FROM_INCHI',
                        help='Name of SMILES column in input CSV')
    parser.add_argument('--target_col', type=str, default='binarized',
                        help='Name of target column (optional, for metrics)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    args, _ = parser.parse_known_args()  # Ignore extra args from gneprop_pyg

    run_inference(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.save_path,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        batch_size=args.batch_size,
        gpus=args.gpus
    )


if __name__ == '__main__':
    main()
