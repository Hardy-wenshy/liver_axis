"""
Makrer score computation for spatial transcriptomics
"""

import numpy as np
import scanpy as sc
import logging

def compute_marker_scores(adata,markers,normalize=True):
    """
    Parameters
    ------
    adata:Anndata
        Anndata object with gene expression matrix
    markers:list
        List of marker gene names
    normalize:bool
        Whether to normalize the expression matrix first
    
    Returns
    ------
    np.ndarray
        Array of marker socres for each cell/spot
    """
    if normalize:
        adata=adata.copy()
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)

    #filter markers that exist in the dataset
    genes=[g for g in markers if g in adata.var_names]

    invalid_markers=[g for g in markers if g not in adata.var_names]

    if invalid_markers:
        logging.warning(f'These markers are not in adata object:{invalid_markers}')

    if len(genes)==0:
        raise ValueError(f'No marker genes found.')

    X=adata[:,genes].X
    
    #Convert sparse to dense if needed

    if hasattr(X,'toarray'):
        X=X.toarray()

    return np.mean(X,axis=1).flatten()

def compute_net_score(pv_score,cv_score):
    """
    Parameters
    ------
    pv_score:np.ndarray
        Periportal marker scores
    cv_score:np.ndarray
        Pericentral marker scores

    Returns
    ------
    np.ndarray
        Net score(CV-PV)
    """
    return cv_score-pv_score



    
    

