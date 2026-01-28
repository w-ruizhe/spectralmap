import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --- 1. ALGORITHM: BEST POLYGON (Unchanged) ---
def count_points_in_triangle(A, B, C, all_points):
    triangle_poly = Path([A, B, C])
    mask = triangle_poly.contains_points(all_points, radius=1e-30)
    return np.sum(mask)

def get_best_polygon_by_points(points, min_corners=3, sensitivity=5.0):
    hull = ConvexHull(points)
    poly_indices = list(hull.vertices)
    history = []
    
    # 1. Iterative Reduction
    while len(poly_indices) > min_corners:
        min_points_lost = float('inf')
        remove_at_idx = -1
        L = len(poly_indices)
        
        for i in range(L):
            p0 = points[poly_indices[(i-1)%L]]
            p1 = points[poly_indices[i]]
            p2 = points[poly_indices[(i+1)%L]]
            
            # Using the FIXED function
            loss = count_points_in_triangle(p0, p1, p2, points)
            
            if loss < min_points_lost:
                min_points_lost = loss
                remove_at_idx = i
        
        # Record state BEFORE removal
        # We record the cost it WOULD take to remove this point
        history.append({
            'n_corners': len(poly_indices), 
            'indices': np.array(poly_indices),
            'points_lost': min_points_lost
        })
        
        # Execute removal
        del poly_indices[remove_at_idx]
    

    # Add final state (base case)
    history.append({
        'n_corners': len(poly_indices), 
        'indices': np.array(poly_indices), 
        'points_lost': float('inf')
    })
        
    # 2. Elbow Selection (Forward Scan: Complex -> Simple)
    
    # Default to the full hull if no elbow is found
    best_indices = history[0]['indices']
    
    # Iterate from 1 (the first simplified step) downwards
    for i in range(1, len(history)):
        
        step_curr = history[i]      # Current shape (N corners)
        step_prev = history[i-1]    # Previous shape (N+1 corners)
        
        # loss_curr: The cost to simplify FURTHER (remove the next point from current shape)
        loss_curr = step_curr['points_lost']
        
        # loss_prev: The cost we just paid to get here (remove the previous point)
        loss_prev = step_prev['points_lost']
        
        # Avoid division by zero (common when removing noise with 0 loss)
        if loss_prev <= 0: loss_prev = 0.1
        
        ratio = loss_curr / loss_prev
        
        print(f"Step {i} (Corners: {step_curr['n_corners']}): Loss {loss_curr} vs Prev {loss_prev} (Ratio: {ratio:.1f})")

        # LOGIC: If the cost to remove the NEXT point is way higher (e.g. >5x) 
        # than the cost of the previous removal, we have hit the structural corners.
        # Stop here and keep the current shape.
        if ratio > sensitivity:
            print(f"-> Elbow detected at {step_curr['n_corners']} corners!")
            best_indices = step_curr['indices']
            break
            
    return best_indices


def find_clusters(F_all_wl, F_cov_all_wl, n_neighbors=50):
    """ W: (n_grid, n_wl) """
    # --- 2. CONFIGURATION & SORTING ---
    X = F_all_wl
    log_X = np.log10(F_all_wl)
    pca = PCA(n_components=2)
    W = pca.fit_transform(log_X)

    corner_indices = get_best_polygon_by_points(W, min_corners=3, sensitivity=3)
    corner_coords = X[corner_indices]
    # Sort by Angle
    centroid = np.mean(corner_coords, axis=0)
    angles = np.arctan2(corner_coords[:, 1] - centroid[1], corner_coords[:, 0] - centroid[0])
    sort_order = np.argsort(angles)
    corner_indices = corner_indices[sort_order]
    corner_coords = corner_coords[sort_order]

    anchor_points = W[corner_indices]
    K = len(anchor_points)
    centers = anchor_points[:, :2]

    # --- 3. K-NEAREST NEIGHBOR ASSIGNMENT ---
    # SETTING: How many neighbors per anchor?
    N_NEIGHBORS = n_neighbors

    # Initialize labels as -1 (Unassigned)
    labels = np.full(len(W), -1)

    # Calculate distances from every point to every anchor
    # Shape: (N_points, K_anchors)
    all_dists = np.linalg.norm(W[:, None, :] - centers[None, :, :], axis=2)
    for k in range(K):
        # Get distances to this specific anchor k
        dists_to_k = all_dists[:, k]
        
        # Find indices of the N smallest distances
        # argsort gives indices from smallest to largest
        nearest_indices = np.argsort(dists_to_k)[:N_NEIGHBORS]
        
        # Assign these points to cluster k
        # Note: If a point is in the top N for multiple anchors, this simple loop
        # assigns it to the last one processed. To be strict about "closest",
        # we can check if it's already assigned and only overwrite if dist is smaller.
        # For visualization purposes, overwriting is usually fine, or we can do a mask check.
        labels[nearest_indices] = k

    # Refine overlaps: If a point belongs to multiple neighborhoods, assign to the CLOSEST one.
    # (Optional strictly correct step)
    assigned_mask = labels != -1
    if np.any(assigned_mask):
        # Only look at points that were picked by at least one neighbor check
        relevant_dists = all_dists[assigned_mask]
        # Re-assign based on strict minimum distance among the anchors
        labels[assigned_mask] = np.argmin(relevant_dists, axis=1)

    # 1. Plot UNASSIGNED points (Grey, faint)
    mask_unassigned = labels == -1
    # X: (N, D), C: (K, D)
    F_regionals = None
    F_regional_errs = None
    F_regional_covs = None
    V = np.zeros((np.unique(labels).size, F_all_wl.shape[0])) # (n_clusters, n_spatial_points)

    for i, label in enumerate(np.unique(labels)):
        ind = labels == label
        print(label, np.sum(ind))
        N = sum(ind)
        weights = np.ones(N) / N
        V[i, ind] = weights

    F_regionals = V @ F_all_wl # (n_clusters, n_wavelengths) use the original X (not log_X) for the mean
    F_regional_errs = np.einsum('ij,wjk,kl->wil', V, F_cov_all_wl, V.T)
    F_regional_covs = np.sqrt(np.diagonal(F_regional_errs, axis1=1, axis2=2)).T # (n_clusters, n_wavelengths)

    return F_regionals, F_regional_errs, labels