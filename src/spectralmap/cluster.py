from scipy.spatial import ConvexHull
from matplotlib.path import Path
import numpy as np


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