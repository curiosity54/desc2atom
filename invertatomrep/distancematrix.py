import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from scipy.optimize import linear_sum_assignment
from itertools import permutations
from scipy.optimize import minimize

# Utilities to retrieve distances from a distance matrix
def loss_fn(positions_flat, edges):
    """Optimizes only known edges"""
    positions = positions_flat.reshape(-1, 3)
    loss = 0.0
    for i, j, dij in edges:
        dist = np.linalg.norm(positions[i] - positions[j])
        loss += (dist - dij)**2
    return loss

def loss_fn_with_threshold(positions_flat, edges, n_atoms, threshold=1.0, penalty=1.0):
    """Optimizes known edges and also forces other edges to be above a certain threshold"""
    
    positions = positions_flat.reshape(-1, 3)
    loss = 0.0
    known_pairs = set((min(i, j), max(i, j)) for i, j, _ in edges)

    for i, j, dij in edges:
        dist = np.linalg.norm(positions[i] - positions[j])
        loss += (dist - dij) ** 2
    # unknown edges
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) not in known_pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    loss += penalty * (threshold - dist) ** 2

    return loss


def compute_distances(positions, edges):
    """Computes euclidean distances between pairs of atoms"""
    return [(i, j, np.linalg.norm(positions[i] - positions[j])) for (i, j, _) in edges]

def best_permutation_distance_matrices(D, D_prime, order=2):
    """
    Find ``perm`` that minimizes ``||D - D_prime[np.ix_(perm, perm)]||``
    over permutations (same objective as :func:`dref`).

    To align coordinate rows, use ``recovered_aligned = recovered[perm]`` so that
    row ``i`` of ``recovered_aligned`` is matched to row ``i`` of the reference.

    Parameters
    ----------
    D, D_prime : ndarray, shape (n, n)
        Pairwise distance matrices (symmetric, zero diagonal).
    order : {2, 8}
        Matrix norm (Frobenius for 2, elementwise max for 8).

    Returns
    -------
    perm : ndarray, shape (n,), dtype int64
        Best permutation indices.
    min_distance : float
        Minimum distance achieved (same as ``dref`` would return).
    """
    n = D.shape[0]
    assert D.shape == D_prime.shape, "Matrices must be same size"

    min_distance = np.inf
    best_perm = None
    for perm in permutations(range(n)):
        perm = list(perm)
        Dp_perm = D_prime[np.ix_(perm, perm)]
        if order == 2:
            dist = np.linalg.norm(D - Dp_perm, ord="fro")
        elif order == 8:
            dist = np.linalg.norm(D.flatten() - Dp_perm.flatten(), ord=8)
        else:
            raise NotImplementedError("only 2 and 8 norms")
        if dist < min_distance:
            min_distance = dist
            best_perm = perm

    return np.asarray(best_perm, dtype=np.int64), min_distance


def dref(D, D_prime, order = 2):
    """ 
    checking if the structures are identical up to relabeling of identical atom (permutations)
    dref = min(P)  (D  − P D'  P.T) Eq (7) from PRB 2013
    From errata, dref is computed as the sqrt of the 8 norm
    """
    _, min_distance = best_permutation_distance_matrices(D, D_prime, order=order)
    return min_distance
    
def dref_approx(D, D_prime):
    cost = np.linalg.norm(D[:, None, :] - D_prime[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    D_prime_perm = D_prime[np.ix_(col_ind, col_ind)]
    return np.linalg.norm(D - D_prime_perm, ord='fro')


def optimize_coords_from_dmat(distance_mat=None, nonzero_i=None, nonzero_j=None):
    if nonzero_i is None or nonzero_j is None:
        pass 
    n_atoms = distance_mat.shape[0]
    edges_left = [(i, j, distance_mat[i, j]) for i, j in zip(nonzero_i, nonzero_j)]

    # Initialize random 3D positions
    rng = np.random.default_rng(29641)
    initial_positions = rng.uniform(-2, 2, size=(n_atoms, 3)).flatten()
    
    # result = minimize(loss_fn, initial_positions, args=(edges_left,), method='L-BFGS-B')
    result = minimize(loss_fn_with_threshold, initial_positions, args=(edges_left,n_atoms,1.1), method='L-BFGS-B')
    positions_opt = result.x.reshape(-1, 3)
    return positions_opt, result 


# def points_from_distance_matrix(D):
#     N = D.shape[0]
#     # Multidimensional Scaling (MDS)
#     J = np.eye(N) - np.ones((N, N)) / N
#     B = -J @ (D ** 2) / 2
#     eigenvalues, eigenvectors = np.linalg.eigh(B)
#     points = np.zeros((N, N))
#     for i in range(N):
#         points[:, i] = eigenvectors[:, i] * np.sqrt(-eigenvalues[i])
#     return points[:,:3]


# points = points_from_distance_matrix(dist_matrix)
# print(points)


def linear_interpolate(start, end, alpha):
    return (1 - alpha) * start + alpha * end

def interpolate_distance_matrix(D1, D2, alpha):
    D = linear_interpolate(D1, D2, alpha)
    return D +D.T - np.diag(np.diag(D))  # Ensure symmetry and zero diagonal
    # n = D1.shape[0]
    # D = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         D[i, j] = linear_interpolate(D1[i, j], D2[i, j], alpha)
    #         D[j, i] = D[i, j]
    return D

def interpolate_coordinates(D1, D2, alpha):
    coords1 = reconstruct_coords(D1)
    coords2 = reconstruct_coords(D2)
    coords_interp = linear_interpolate(coords1, coords2, alpha)
    return coords_interp

    
