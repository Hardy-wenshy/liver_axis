"""Bidirectional occupancy-weighted axis computation"""

import numpy as np


def run_random_walk(T, seeds, n_iter=100, restart_prob=0.05):
    """
    Run random walk with restart on transition matrix.
    
    Parameters
    ----------
    T : scipy.sparse.csr_matrix
        Transition matrix
    seeds : np.ndarray
        Seed indices
    n_iter : int
        Number of iterations
    restart_prob : float
        Probability of restarting from seeds
        
    Returns
    -------
    np.ndarray
        Probability distribution after random walk
    """
    n = T.shape[0]
    
    if len(seeds) == 0:
        return np.ones(n) * 1e-15
    
    C0 = np.zeros(n)
    C0[seeds] = 1.0 / len(seeds)
    C = C0.copy()
    
    for _ in range(n_iter):
        C = (1 - restart_prob) * (C @ T) + restart_prob * C0
    
    return C


def compute_bidirectional_axis(T_f, T_b, pv_idx, cv_idx, n_iter=100, restart_prob=0.05):
    """
    Compute bidirectional occupancy-weighted axis.
    
    Parameters
    ----------
    T_f : scipy.sparse.csr_matrix
        Forward transition matrix
    T_b : scipy.sparse.csr_matrix
        Backward transition matrix
    pv_idx : np.ndarray
        Periportal seed indices
    cv_idx : np.ndarray
        Pericentral seed indices
    n_iter : int
        Number of random walk iterations
    restart_prob : float
        Restart probability for random walk
        
    Returns
    -------
    tuple
        (axis, prob_pv, prob_cv, occupancy_norm)
    """
    n = T_f.shape[0]
    
    # Run random walks
    prob_pv = run_random_walk(T_b, pv_idx, n_iter, restart_prob)
    prob_cv = run_random_walk(T_f, cv_idx, n_iter, restart_prob)
    
    # Compute balance and occupancy
    balance = prob_cv / (prob_cv + prob_pv + 1e-12)
    
    occupancy = prob_cv + prob_pv
    occ_p99 = np.percentile(occupancy, 99)
    occupancy_norm = np.clip(occupancy / (occ_p99 + 1e-12), 0, 1)
    
    # Compute axis
    axis = 0.5 + (balance - 0.5) * occupancy_norm
    
    # Normalize to [0, 1]
    p1 = np.percentile(axis, 1)
    p99 = np.percentile(axis, 99)
    axis = (axis - p1) / (p99 - p1 + 1e-12)
    axis = np.clip(axis, 0, 1)
    
    # Orient axis (PV low, CV high)
    pv_mean = np.mean(axis[pv_idx]) if len(pv_idx) > 0 else 0
    cv_mean = np.mean(axis[cv_idx]) if len(cv_idx) > 0 else 1
    
    if pv_mean > cv_mean:
        axis = 1 - axis
        prob_pv, prob_cv = prob_cv, prob_pv
    
    return axis, prob_pv, prob_cv, occupancy_norm