"""KNN graph and transition matrix construction"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def build_knn(coords, k=8):
    """
    Build k-nearest neighbors graph.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (n_cells x 2)
    k : int
        Number of neighbors
        
    Returns
    -------
    tuple
        (distances, indices) for nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coords)
    dist, idx = nbrs.kneighbors(coords)
    return dist, idx


def build_dual_transition_matrices(coords, net_score, k=8, alpha=2.0, min_prob=1e-4):
    """
    Build forward and backward transition matrices with drift based on net score.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates
    net_score : np.ndarray
        Net score (CV - PV) for each cell
    k : int
        Number of nearest neighbors
    alpha : float
        Drift strength parameter
    min_prob : float
        Minimum transition probability
        
    Returns
    -------
    tuple
        (T_forward, T_backward, neighbor_indices)
    """
    dist, idx = build_knn(coords, k)
    n = coords.shape[0]
    
    median_dist = np.median(dist)
    
    rows_f, cols_f, data_f = [], [], []
    rows_b, cols_b, data_b = [], [], []
    
    for i in range(n):
        neighbors = idx[i]
        dists = dist[i]
        
        wf_list = []
        wb_list = []
        js = []
        
        for j, d in zip(neighbors, dists):
            if i == j:
                continue
            
            delta = net_score[j] - net_score[i]
            spatial_w = np.exp(-d / (median_dist + 1e-12))
            
            wf = spatial_w * np.exp(alpha * delta)
            wb = spatial_w * np.exp(-alpha * delta)
            
            wf = max(wf, min_prob)
            wb = max(wb, min_prob)
            
            wf_list.append(wf)
            wb_list.append(wb)
            js.append(j)
        
        # Normalize probabilities
        wf_list = np.array(wf_list)
        wb_list = np.array(wb_list)
        wf_list = wf_list / (wf_list.sum() + 1e-12)
        wb_list = wb_list / (wb_list.sum() + 1e-12)
        
        rows_f.extend([i] * len(js))
        cols_f.extend(js)
        data_f.extend(wf_list)
        
        rows_b.extend([i] * len(js))
        cols_b.extend(js)
        data_b.extend(wb_list)
    
    T_f = csr_matrix((data_f, (rows_f, cols_f)), shape=(n, n))
    T_b = csr_matrix((data_b, (rows_b, cols_b)), shape=(n, n))
    
    return T_f, T_b, idx