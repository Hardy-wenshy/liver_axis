"""Flow field and gradient analysis"""

import numpy as np


def compute_local_flow(coords, axis, neighbor_idx):
    """
    Compute local flow field based on axis gradient.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates
    axis : np.ndarray
        Axis values for each cell
    neighbor_idx : np.ndarray
        Neighbor indices (n_cells x k)
        
    Returns
    -------
    np.ndarray
        Flow vectors for each cell
    """
    n = len(coords)
    flow = np.zeros_like(coords)
    
    for i in range(n):
        neighbors = neighbor_idx[i]
        vecs = coords[neighbors] - coords[i]
        grads = axis[neighbors] - axis[i]
        flow[i] = (vecs * grads[:, None]).mean(axis=0)
    
    return flow


def local_gradient_strength(axis, neighbor_idx):
    """
    Compute local gradient strength of the axis.
    
    Parameters
    ----------
    axis : np.ndarray
        Axis values for each cell
    neighbor_idx : np.ndarray
        Neighbor indices (n_cells x k)
        
    Returns
    -------
    np.ndarray
        Gradient strength for each cell
    """
    grad = np.zeros(len(axis))
    
    for i in range(len(axis)):
        grad[i] = np.std(axis[neighbor_idx[i]] - axis[i])
    
    return grad