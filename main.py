"""Main script for spatial axis analysis"""
import anndata as ad
import numpy as np
import scanpy as sc
import argparse
import random
import os
from config import *
from core import (
    compute_marker_scores,
    compute_net_score,
    build_dual_transition_matrices,
    detect_seeds_dynamic,
    compute_bidirectional_axis,
    compute_local_flow,
    local_gradient_strength
)
from plotting import plot_spatial_results, plot_deg_heatmap
from utils import filter_tissue, export_deg_results







def process_sample(adata, sample_name, cv_markers=None, pv_markers=None, 
                   output_dir=".", **kwargs):
    """
    Process a single sample through the spatial axis analysis pipeline.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates and expression data
    sample_name : str
        Name of the sample
    cv_markers : list
        Pericentral marker genes
    pv_markers : list
        Periportal marker genes
    output_dir : str
        Directory for output files
    **kwargs : dict
        Additional parameters:
            - k (int): Number of neighbors for KNN
            - alpha (float): Drift strength
            - std_multiplier (float): Seed detection threshold
            - n_iter (int): Random walk iterations
            - restart_prob (float): Restart probability
            - top_n (int): Number of top genes for heatmap
            - flow_scale (float): Scale for flow arrows
            
    Returns
    -------
    AnnData
        Updated AnnData object with analysis results
    """
    # Set default parameters
    cv_markers = cv_markers or DEFAULT_CV_MARKERS
    pv_markers = pv_markers or DEFAULT_PV_MARKERS
    k = kwargs.get('k', DEFAULT_KNN_K)
    alpha = kwargs.get('alpha', DEFAULT_ALPHA)
    std_multiplier = kwargs.get('std_multiplier', DEFAULT_STD_MULTIPLIER)
    n_iter = kwargs.get('n_iter', DEFAULT_N_ITER)
    restart_prob = kwargs.get('restart_prob', DEFAULT_RESTART_PROB)
    top_n = kwargs.get('top_n', DEFAULT_TOP_N_GENES)
    flow_scale = kwargs.get('flow_scale', DEFAULT_FLOW_SCALE)
    
    print(f"\n{'='*60}")
    print(f"Processing: {sample_name}")
    print(f"{'='*60}")
    
    # Filter tissue spots
    adata = filter_tissue(adata)
    adata.obs["sample"] = sample_name
    
    # Get spatial coordinates
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
    coords = adata.obsm["spatial"]

    # Compute marker scores
    print("  - Computing marker scores...")
    pv_score = compute_marker_scores(adata, pv_markers)
    cv_score = compute_marker_scores(adata, cv_markers)
    net_score = compute_net_score(pv_score, cv_score)
    
    # Detect seeds
    print("  - Detecting seeds...")
    pv_idx, cv_idx = detect_seeds_dynamic(pv_score, cv_score, std_multiplier)
    
    # Build transition matrices
    print("  - Building transition matrices...")
    T_f, T_b, neighbors = build_dual_transition_matrices(coords, net_score, k, alpha)
    
    # Compute bidirectional axis
    print("  - Computing bidirectional axis...")
    axis, prob_pv, prob_cv, occupancy = compute_bidirectional_axis(
        T_f, T_b, pv_idx, cv_idx, n_iter, restart_prob
    )
    
    # Store results in AnnData
    adata.obs["markov_axis"] = axis
    adata.obs["pv_probability"] = prob_pv
    adata.obs["cv_probability"] = prob_cv
    adata.obs["trajectory_occupancy"] = occupancy
    
    # DEG analysis and visualization
    if len(pv_idx) > 0 and len(cv_idx) > 0:
        print("  - Running DEG analysis...")
        plot_deg_heatmap(adata, pv_idx, cv_idx, sample_name, top_n)
        export_deg_results(adata, sample_name, output_dir)
    
    # Compute flow and gradient
    print("  - Computing flow field...")
    flow = compute_local_flow(coords, axis, neighbors)
    grad = local_gradient_strength(axis, neighbors)
    
    # Generate spatial visualization
    print("  - Generating visualization...")
    output_file = os.path.join(output_dir, f"{sample_name}_bidirectional_axis.pdf")
    plot_spatial_results(coords, pv_score, cv_score, axis, prob_pv, prob_cv,
                        occupancy, flow, grad, output_file, flow_scale)
    
    print(f"  - Finished: {sample_name}\n")
    
    return adata


def main():
    """Main function to run the spatial axis analysis pipeline."""
    parser = argparse.ArgumentParser(description="Spatial Axis Analysis for Liver Zonation")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input h5ad file path")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for results")
    parser.add_argument("--sample_name", type=str, default="sample",
                       help="Sample name for output files")
    parser.add_argument("--k", type=int, default=DEFAULT_KNN_K,
                       help=f"Number of neighbors for KNN (default: {DEFAULT_KNN_K})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                       help=f"Drift strength (default: {DEFAULT_ALPHA})")
    parser.add_argument("--std_multiplier", type=float, default=DEFAULT_STD_MULTIPLIER,
                       help=f"Standard deviation multiplier for seed detection (default: {DEFAULT_STD_MULTIPLIER})")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N_GENES,
                       help=f"Number of top genes for heatmap (default: {DEFAULT_TOP_N_GENES})")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    adata = ad.read_h5ad(args.input)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Process sample
    result_adata = process_sample(
        adata,
        args.sample_name,
        output_dir=args.output_dir,
        k=args.k,
        alpha=args.alpha,
        std_multiplier=args.std_multiplier,
        top_n=args.top_n
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.sample_name}_processed.h5ad")
    result_adata.write(output_file)
    print(f"\nResults saved to {output_file}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()