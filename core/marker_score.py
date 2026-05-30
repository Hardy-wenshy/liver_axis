"""Marker score computation for spatial transcriptomics."""

import numpy as np


def compute_marker_scores(adata, markers):
    """
    Compute average expression score for a marker gene set.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression matrix.

    markers : list[str]
        Marker gene list.

    Returns
    -------
    np.ndarray
        Mean marker expression score per spot/cell.
    """

    genes = [g for g in markers if g in adata.var_names]

    if len(genes) == 0:
        raise ValueError("No marker genes found in adata.var_names.")

    X = adata[:, genes].X

    # sparse matrix -> dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.mean(X, axis=1).flatten()


def compute_net_score(pv_score, cv_score):
    """
    Compute zonation net score.

    Parameters
    ----------
    pv_score : np.ndarray
        Periportal marker score.

    cv_score : np.ndarray
        Pericentral marker score.

    Returns
    -------
    np.ndarray
        Net zonation score (CV - PV).
    """

    return cv_score - pv_score