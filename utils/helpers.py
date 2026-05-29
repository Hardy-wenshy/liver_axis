"""Helper utility functions"""

import pandas as pd
import scanpy as sc


def filter_tissue(adata, tissue_key="in_tissue"):
    """
    Filter AnnData object to keep only tissue spots.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    tissue_key : str
        Column name indicating tissue spots
        
    Returns
    -------
    AnnData
        Filtered AnnData object
    """
    if tissue_key in adata.obs:
        adata = adata[adata.obs[tissue_key] == 1].copy()
    return adata


def export_deg_results(adata, sample_name, output_dir="."):
    """
    Export differential expression results to CSV.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with DEG results
    sample_name : str
        Name of the sample
    output_dir : str
        Directory to save the CSV file
    """
    if "seed_degs" not in adata.uns:
        print("  - No DEG results found. Run DEG analysis first.")
        return None
    
    deg_df = sc.get.rank_genes_groups_df(adata, group="CV_Seed", key="seed_degs")
    deg_df.insert(0, "sample_id", sample_name)
    deg_df.columns = ['sample_id', 'gene_name', 'log_fold_change', 
                      'p_value', 'p_adjusted', 'wilcoxon_score']
    deg_df = deg_df.sort_values(by="log_fold_change", ascending=False).reset_index(drop=True)
    
    csv_filename = f"{output_dir}/{sample_name}_seed_DEGs_statistics.csv"
    deg_df.to_csv(csv_filename, index=False)
    print(f"  - Exported DEG results to {csv_filename}")
    
    return deg_df