#!/usr/bin/env python
"""
Novel MOA (Mechanism of Action) Detection using OOD (Out-of-Distribution) Detection

This script implements the Mahalanobis distance-based OOD detection pipeline
described in the GNEprop paper. It identifies molecules that are likely to have
a distinct MoA compared to a set of known antibiotics by:

1. Loading a pretrained self-supervised model (SimCLR)
2. Extracting molecular representations for known antibiotics (with target labels)
3. Fitting class-conditional Gaussian distributions on each target
4. For query molecules, computing an OOD score based on Mahalanobis distance

Molecules with very negative OOD scores are embedded far from all known target
clusters and are thus more likely to exhibit a distinct/novel MoA.

Usage:
    python run_ood_detection.py \
        --known_antibiotics_path support_data/Extended_data_table_antibiotics.csv \
        --query_path my_compounds.csv \
        --ssl_checkpoint_path /path/to/simclr/checkpoint.ckpt \
        --output_path predictions_ood.csv \
        --smiles_col SMILES \
        --target_col Target \
        --gpus 1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add the GNEprop directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import clr
import data
import gneprop_pyg
from gneprop.ood import compute_centroids, neg_distance_to_clusters


def load_ssl_model(checkpoint_path, device='cuda'):
    """Load the pretrained self-supervised SimCLR model."""
    print(f"Loading self-supervised model from: {checkpoint_path}")
    model = clr.SimCLR.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def extract_representations(model, smiles_list, batch_size=128, num_workers=4,
                           use_projection_layers=0, device='cuda'):
    """Extract molecular representations using the self-supervised model."""
    print(f"Extracting representations for {len(smiles_list)} molecules...")

    # Create dataset
    dataset = data.MolDatasetOD(smiles_list=list(smiles_list))
    dataloader = gneprop_pyg.convert_to_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Extract representations
    representations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representations"):
            batch = batch.to(device)
            repr_batch = model.get_representations(
                batch, use_projection_layers=use_projection_layers
            )
            representations.append(repr_batch.cpu().numpy())

    return np.vstack(representations)


def compute_ood_scores(query_reprs, mean_var_dict):
    """Compute OOD scores for query molecules."""
    print(f"Computing OOD scores for {len(query_reprs)} molecules...")

    ood_scores = []
    for i, repr_vec in enumerate(tqdm(query_reprs, desc="Computing OOD scores")):
        score = neg_distance_to_clusters(repr_vec, mean_var_dict)
        ood_scores.append(score)

    return np.array(ood_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Novel MOA Detection using OOD Detection"
    )

    # Required arguments
    parser.add_argument(
        "--known_antibiotics_path",
        type=str,
        required=True,
        help="Path to CSV with known antibiotics (SMILES and target labels)"
    )
    parser.add_argument(
        "--query_path",
        type=str,
        required=True,
        help="Path to CSV with query molecules to evaluate"
    )
    parser.add_argument(
        "--ssl_checkpoint_path",
        type=str,
        required=True,
        help="Path to self-supervised (SimCLR) model checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output CSV with OOD scores"
    )

    # Column names
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="SMILES",
        help="Name of SMILES column in input CSVs (default: SMILES)"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Target",
        help="Name of target/MoA column in known antibiotics CSV (default: Target)"
    )
    parser.add_argument(
        "--query_smiles_col",
        type=str,
        default=None,
        help="Name of SMILES column in query CSV (defaults to --smiles_col)"
    )

    # Model/processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for representation extraction (default: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--use_projection_layers",
        type=int,
        default=0,
        help="Number of projection layers to use for representations (default: 0, i.e., base encoder only)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (0 for CPU, default: 1)"
    )

    # Additional options
    parser.add_argument(
        "--save_representations",
        action="store_true",
        help="Also save the extracted representations as .npy files"
    )
    parser.add_argument(
        "--save_centroids",
        action="store_true",
        help="Save the computed centroids for visualization/analysis"
    )

    args = parser.parse_args()

    # Set device
    if args.gpus > 0 and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Default query SMILES column to main SMILES column
    if args.query_smiles_col is None:
        args.query_smiles_col = args.smiles_col

    # Load data
    print("\n=== Loading Data ===")
    print(f"Known antibiotics: {args.known_antibiotics_path}")
    known_df = pd.read_csv(args.known_antibiotics_path)
    print(f"  Loaded {len(known_df)} known antibiotics")
    print(f"  Targets: {known_df[args.target_col].nunique()} unique targets")
    print(f"  Target distribution:\n{known_df[args.target_col].value_counts()}")

    print(f"\nQuery molecules: {args.query_path}")
    query_df = pd.read_csv(args.query_path)
    print(f"  Loaded {len(query_df)} query molecules")

    # Load model
    print("\n=== Loading Model ===")
    model = load_ssl_model(args.ssl_checkpoint_path, device=device)

    # Extract representations for known antibiotics
    print("\n=== Processing Known Antibiotics ===")
    known_smiles = known_df[args.smiles_col].values
    known_reprs = extract_representations(
        model, known_smiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_projection_layers=args.use_projection_layers,
        device=device
    )
    print(f"Known antibiotics representation shape: {known_reprs.shape}")

    # Create a dataframe with index for centroid computation
    known_repr_df = pd.DataFrame({
        args.target_col: known_df[args.target_col].values
    })
    known_repr_df.index = range(len(known_repr_df))

    # Compute centroids (class-conditional Gaussians)
    print("\n=== Computing Target Centroids ===")
    mean_var_dict = compute_centroids(known_repr_df, known_reprs, args.target_col)
    print(f"Computed centroids for {len(mean_var_dict)} targets:")
    for target, stats in mean_var_dict.items():
        print(f"  {target}: mean shape={stats['mean'].shape}")

    # Extract representations for query molecules
    print("\n=== Processing Query Molecules ===")
    query_smiles = query_df[args.query_smiles_col].values
    query_reprs = extract_representations(
        model, query_smiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_projection_layers=args.use_projection_layers,
        device=device
    )
    print(f"Query molecules representation shape: {query_reprs.shape}")

    # Compute OOD scores
    print("\n=== Computing OOD Scores ===")
    ood_scores = compute_ood_scores(query_reprs, mean_var_dict)

    # Create output dataframe
    output_df = query_df.copy()
    output_df['ood_score'] = ood_scores

    # Lower (more negative) scores indicate more OOD (potentially novel MoA)
    output_df['ood_rank'] = output_df['ood_score'].rank(ascending=True)
    output_df['ood_percentile'] = output_df['ood_score'].rank(pct=True, ascending=True)

    # Sort by OOD score (most OOD first)
    output_df = output_df.sort_values('ood_score', ascending=True)

    # Save results
    print("\n=== Saving Results ===")
    output_df.to_csv(args.output_path, index=False)
    print(f"Saved OOD scores to: {args.output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"OOD Score range: [{ood_scores.min():.4f}, {ood_scores.max():.4f}]")
    print(f"OOD Score mean: {ood_scores.mean():.4f}")
    print(f"OOD Score std: {ood_scores.std():.4f}")
    print(f"\nTop 10 most OOD molecules (potential novel MoA):")
    print(output_df.head(10)[[args.query_smiles_col, 'ood_score', 'ood_rank']].to_string())

    # Save optional outputs
    if args.save_representations:
        output_dir = os.path.dirname(args.output_path)
        known_repr_path = os.path.join(output_dir, 'known_antibiotics_representations.npy')
        query_repr_path = os.path.join(output_dir, 'query_representations.npy')
        np.save(known_repr_path, known_reprs)
        np.save(query_repr_path, query_reprs)
        print(f"Saved representations to: {known_repr_path}, {query_repr_path}")

    if args.save_centroids:
        import pickle
        output_dir = os.path.dirname(args.output_path)
        centroids_path = os.path.join(output_dir, 'target_centroids.pkl')
        with open(centroids_path, 'wb') as f:
            pickle.dump(mean_var_dict, f)
        print(f"Saved centroids to: {centroids_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
