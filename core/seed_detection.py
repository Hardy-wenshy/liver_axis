"""Seed detection for bidirectional axis analysis"""

import numpy as np


def detect_seeds_dynamic(pv_score, cv_score, std_multiplier=1.0):
    """
    Detect seed cells/points for PV and CV based on marker scores.
    
    Parameters
    ----------
    pv_score : np.ndarray
        Periportal marker scores
    cv_score : np.ndarray
        Pericentral marker scores
    std_multiplier : float
        Multiplier for standard deviation threshold
        
    Returns
    -------
    tuple
        (pv_indices, cv_indices) for seed cells
    """
    # Calculate thresholds
    pv_threshold = np.median(pv_score) + std_multiplier * np.std(pv_score)
    cv_threshold = np.median(cv_score) + std_multiplier * np.std(cv_score)
    
    print("PV score")
    print(np.mean(pv_score))
    print(np.std(pv_score))

    print("CV score")
    print(np.mean(cv_score))
    print(np.std(cv_score))

    # Identify high-score indices
    pv_idx = np.where(pv_score > pv_threshold)[0]
    cv_idx = np.where(cv_score > cv_threshold)[0]
    
    # Remove overlaps to ensure distinct seeds
    overlap = np.intersect1d(pv_idx, cv_idx)
    
    if len(overlap) > 0:
        pv_idx = np.setdiff1d(pv_idx, overlap)
        cv_idx = np.setdiff1d(cv_idx, overlap)
    
    print(f"  - Dynamic Seeds: PV={len(pv_idx)}, CV={len(cv_idx)}")
    
    return pv_idx, cv_idx