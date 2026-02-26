import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --- 1. ALGORITHM: BEST POLYGON ---
def count_points_in_triangle(A, B, C, all_points):
    triangle_poly = Path([A, B, C])
    mask = triangle_poly.contains_points(all_points, radius=1e-30)
    return np.sum(mask)

def get_best_polygon(points, n_corners=3):
    hull = ConvexHull(points)
    poly_indices = list(hull.vertices)

    if n_corners < 3:
        raise ValueError("n_corners must be >= 3 for polygon-based clustering")
    if n_corners > len(poly_indices):
        raise ValueError("n_corners cannot exceed number of convex hull vertices")

    # 1. Iterative reduction to an explicit target corner count
    while len(poly_indices) > n_corners:
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

        # Execute removal
        del poly_indices[remove_at_idx]

    return np.array(poly_indices)


def find_clusters(F_all_wl, F_cov_all_wl, n_neighbors=100, n_corners=3, plot=True):
    """ F_all_wl: (n_wl, n_grid) """
    # --- 2. CONFIGURATION & SORTING ---
    X = F_all_wl.T
    log_X = np.log10(X)
    pca = PCA(n_components=2)
    W = pca.fit_transform(log_X)

    if n_corners == 2:
        W_centered = W - np.mean(W, axis=0)
        _, _, vh = np.linalg.svd(W_centered, full_matrices=False)
        max_var_dir = vh[0]
        proj = W_centered @ max_var_dir
        i_min = int(np.argmin(proj))
        i_max = int(np.argmax(proj))

        if i_min == i_max:
            raise ValueError("Could not determine two distinct anchors for n_corners=2")

        corner_indices = np.array([i_min, i_max], dtype=int)
        corner_coords = W[corner_indices]
    elif n_corners >= 3:
        corner_indices = get_best_polygon(W, n_corners=n_corners)
        corner_coords = W[corner_indices]
        # Sort by Angle
        centroid = np.mean(corner_coords, axis=0)
        angles = np.arctan2(corner_coords[:, 1] - centroid[1], corner_coords[:, 0] - centroid[0])
        sort_order = np.argsort(angles)
        corner_indices = corner_indices[sort_order]
        corner_coords = corner_coords[sort_order]
    else:
        raise ValueError("n_corners must be >= 2")

    anchor_points = W[corner_indices]
    K = len(anchor_points)
    centers = anchor_points[:, :2]

    # --- 3. K-NEAREST NEIGHBOR ASSIGNMENT ---
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    # Initialize labels as -1 (unassigned)
    labels = np.full(len(W), -1)
    in_neighborhood = np.zeros((K, len(W)), dtype=bool)

    # Calculate distances from every point to every anchor
    # Shape: (N_points, K_anchors)
    all_dists = np.linalg.norm(W[:, None, :] - centers[None, :, :], axis=2)

    for k in range(K):
        dists_to_k = all_dists[:, k]
        nearest_indices = np.argsort(dists_to_k)[:n_neighbors]
        in_neighborhood[k, nearest_indices] = True
        labels[nearest_indices] = k

    # Refine overlaps with capacity limits:
    # move an overlapping point to a closer label only if that label has room.
    cluster_counts = np.array([np.sum(labels == k) for k in range(K)], dtype=int)
    overlap_indices = np.where(np.sum(in_neighborhood, axis=0) > 1)[0]

    for idx in overlap_indices:
        current_label = labels[idx]
        if current_label < 0:
            continue

        candidate_labels = np.where(in_neighborhood[:, idx])[0]
        if candidate_labels.size == 0:
            continue

        candidate_dists = all_dists[idx, candidate_labels]
        closest_label = candidate_labels[np.argmin(candidate_dists)]

        if closest_label == current_label:
            continue

        if cluster_counts[closest_label] < n_neighbors:
            labels[idx] = closest_label
            cluster_counts[current_label] -= 1
            cluster_counts[closest_label] += 1

    # X: (N, D), C: (K, D)
    V = np.zeros((K+1, X.shape[0])) # (n_corners, n_spatial_points)

    for i in range(K+1):
        ind = (labels == i-1)
        print(i-1, np.sum(ind))
        N = np.sum(ind)
        if N == 0:
            continue
        weights = np.ones(N) / N
        V[i, ind] = weights

    F_regionals = V @ X # (n_corners, n_wavelengths) use the original X (not log_X) for the mean

    F_regional_covs = np.einsum('ij,wjk,kl->wil', V, F_cov_all_wl, V.T)
    F_regional_errs = np.sqrt(np.diagonal(F_regional_covs, axis1=1, axis2=2)).T # (n_corners, n_wavelengths)
    
    # --- 4. PLOTTING ---
    if plot:
        plt.figure(figsize=(7, 2.5), dpi=300)

        plot_colors = plt.get_cmap('tab10').colors

        # 1. Plot UNASSIGNED points (grey, faint)
        mask_unassigned = labels == -1
        plt.scatter(W[mask_unassigned, 0], W[mask_unassigned, 1],
                    s=5, alpha=0.15, color='#BBBBBB', edgecolor='none', zorder=1)

        # 2. Plot ASSIGNED clusters
        for k in range(K):
            mask = labels == k
            color = plot_colors[k % len(plot_colors)]
            plt.scatter(W[mask, 0], W[mask, 1],
                        s=15, alpha=0.8, color=color, edgecolor='none', zorder=2,
                        label=f'Cluster {k+1}')

        # 3. Plot Polygon
        poly_draw = np.vstack([corner_coords, corner_coords[0]])
        plt.plot(poly_draw[:, 0], poly_draw[:, 1], 'k-', lw=1.5, alpha=0.7, zorder=3)

        # 4. Labels
        for k in range(K):
            color = plot_colors[k % len(plot_colors)]
            plt.text(centers[k, 0], centers[k, 1], f'{k+1}',
                    fontsize=10, fontname='Comic Sans MS', fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='white', linewidth=1, alpha=0.9),
                    path_effects=[pe.Stroke(linewidth=1.5, foreground='gray'), pe.Normal()], zorder=10)

        plt.title(f'Classification in PC Space (n_neighbors={n_neighbors})', fontsize=9)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.gca().set_aspect('equal')
        plt.tight_layout()

    return F_regionals, F_regional_errs, labels
