"""Visualization functions for spatial axis analysis"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc


def plot_spatial_results(coords, pv_score, cv_score, axis, prob_pv, prob_cv,
                         occupancy, flow, grad, output_filename, flow_scale=1000):
    """
    Create comprehensive visualization of spatial axis analysis results.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates
    pv_score : np.ndarray
        Periportal marker scores
    cv_score : np.ndarray
        Pericentral marker scores
    axis : np.ndarray
        Computed axis values
    prob_pv : np.ndarray
        PV probabilities
    prob_cv : np.ndarray
        CV probabilities
    occupancy : np.ndarray
        Trajectory occupancy
    flow : np.ndarray
        Flow vectors
    grad : np.ndarray
        Gradient strength
    output_filename : str
        Path to save the figure
    flow_scale : float
        Scale factor for flow arrows
    """
    fig, axs = plt.subplots(2, 5, figsize=(24, 8))
    
    plots = [
        (axs[0, 0], pv_score, "Blues", "PV Marker Score"),
        (axs[0, 1], cv_score, "Reds", "CV Marker Score"),
        (axs[0, 2], prob_pv, "Purples", "PV Probability"),
        (axs[0, 3], prob_cv, "Oranges", "CV Probability"),
        (axs[0, 4], occupancy, "Greens", "Trajectory Occupancy"),
        (axs[1, 1], axis, "viridis", "Axis"),
        (axs[1, 2], grad, "magma", "Gradient Strength")
    ]
    
    for ax, data, cmap, title in plots:
        p = ax.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap, s=5)
        ax.set_title(title)
        ax.invert_yaxis()
        plt.colorbar(p, ax=ax)
    
    # Flow field
    mag = np.linalg.norm(flow, axis=1)
    mask = mag > np.percentile(mag, 80)
    
    axs[1, 0].quiver(coords[mask, 0], coords[mask, 1],
                     flow[mask, 0], flow[mask, 1],
                     scale=flow_scale, color="black", alpha=0.6)
    axs[1, 0].scatter(coords[:, 0], coords[:, 1],
                      c=axis, cmap="viridis", s=5, alpha=0.2)
    axs[1, 0].set_title("Flow Field (PV → CV)")
    axs[1, 0].invert_yaxis()
    
    # Histograms
    axs[1, 3].hist(axis, bins=50, color="gray")
    axs[1, 3].set_title("Axis Distribution")
    
    axs[1, 4].hist(occupancy, bins=50, color="green")
    axs[1, 4].set_title("Occupancy")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  - Saved visualization to {output_filename}")


def plot_deg_heatmap(adata, pv_idx, cv_idx, sample_name, top_n=10):
    """
    Generate heatmap for differentially expressed genes between seeds.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    pv_idx : np.ndarray
        PV seed indices
    cv_idx : np.ndarray
        CV seed indices
    sample_name : str
        Name of the sample
    top_n : int
        Number of top genes to display
        
    Returns
    -------
    tuple
        (pv_top_genes, cv_top_genes)
    """
    # Label seeds
    adata.obs["seed_group"] = "Other"
    adata.obs.iloc[pv_idx, adata.obs.columns.get_loc("seed_group")] = "PV_Seed"
    adata.obs.iloc[cv_idx, adata.obs.columns.get_loc("seed_group")] = "CV_Seed"
    
    # Run DEG analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby="seed_group",
        groups=["CV_Seed"],
        reference="PV_Seed",
        method="wilcoxon",
        key_added="seed_degs"
    )
    
    # Get top genes
    cv_top = list(adata.uns["seed_degs"]["names"]["CV_Seed"][:top_n])
    pv_top = list(adata.uns["seed_degs"]["names"]["CV_Seed"][-top_n:][::-1])
    
    # Prepare subset for heatmap
    subset = adata[adata.obs["seed_group"].isin(["PV_Seed", "CV_Seed"])].copy()
    subset.obs["seed_group"] = subset.obs["seed_group"].astype("category")
    subset.obs["seed_group"] = subset.obs["seed_group"].cat.reorder_categories(["PV_Seed", "CV_Seed"])
    
    # Filter valid genes
    plot_genes = [g for g in (pv_top + cv_top) if g in subset.var_names]
    
    # Generate heatmap
    sc.pl.heatmap(
        subset,
        var_names=plot_genes,
        groupby="seed_group",
        swap_axes=True,
        standard_scale="var",
        cmap="viridis",
        show=False
    )
    
    plt.title(f"Seed DEGs: {sample_name}")
    plt.savefig(f"{sample_name}_seed_heatmap.pdf", bbox_inches="tight")
    plt.close()
    
    return pv_top, cv_top