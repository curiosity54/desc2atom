import torch
import numpy as np
from itertools import permutations
from invertatomrep.descriptor import optimize_to_N_points
from ase.io import write
import ase
from ase import Atoms

def convert_coords_to_frames(coords):
    """Convert a list of coordinates to a list of frames."""
    frames = []
    for coord in coords:
        if isinstance(coord, torch.Tensor):
            coord = coord.detach().cpu().numpy()
        frames.append(ase.Atoms(['H'] * len(coord), coord))
    return frames

def append_origin_to_coords(neighbor_coords: torch.Tensor) -> torch.Tensor:
    """Stack central site at origin above neighbor rows: (n, 3) -> (n + 1, 3)."""
    origin = torch.zeros((1, 3), dtype=neighbor_coords.dtype, device=neighbor_coords.device)
    return torch.vstack((origin, neighbor_coords))


def extract_descriptor(descriptor_result, stack_parities=True):
    """Flatten descriptor outputs to a feature tensor (or batch of tensors).
    descriptor_result can be:
    - **Nested multi-order** ``{nu: {parity: tensor}}``: for each parity, concatenates
      all ``nu`` in ascending order, then concatenates parities in order ``-1``, ``1``
      when ``stack_parities`` is True. A given ``nu`` may be ``None`` (skip that order).
    - **Single-order parity dict** ``{parity: tensor}`` with keys in ``{1, -1}``.
    # - **List** of either of the above (one dict per structure) → ``torch.stack`` on the
    #   flattened tensors, or per-parity stacks when ``stack_parities`` is False.

    Tuple values ``(tensor, combinations)`` and ``numpy`` arrays are accepted where a
    tensor is expected and converted to ``torch.Tensor``.

    stack_parities: If True (default), concatenate parity blocks as ``-1`` then ``1``.
     If False, return ``{parity: tensor}`` (only parities present).
    """
    def _flatten_for_hstack(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return t
        if t.ndim == 0:
            return t.reshape(1, 1)
        if t.ndim == 1:
            return t.reshape(1, -1)
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)

    def _to_tensor(val):
        if isinstance(val, tuple):
            # happens when returning descriptor and l combinations
            # print(f"val is a tuple: {val}")
            val = val[0]
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        return val

    def _merge_parities(per_p: dict):
        if not per_p:
            raise ValueError("extract_descriptor: empty descriptor dict")
        if stack_parities:
            return torch.cat([per_p[k] for k in sorted(per_p)], dim=-1)
        return per_p

    def _from_nested_nu_parity(d: dict):
        nus = sorted(k for k in d if isinstance(k, int))
        parities = set()
        for nu in nus:
            inner = d[nu]
            if inner is None:
                continue
            if not isinstance(inner, dict):
                raise TypeError("extract_descriptor: expected {nu: {parity: tensor}} or nu→None")
            parities.update(inner)
        per_p = {}
        for p in sorted(parities):
            parity_wise = []
            for nu in nus:
                inner = d[nu]
                if inner is None or p not in inner:
                    continue
                parity_wise.append(_flatten_for_hstack(_to_tensor(inner[p])))
            if parity_wise:
                per_p[p] = torch.cat(parity_wise, dim=-1)
        return _merge_parities(per_p)

    def _from_parity_only(d: dict):
        return _merge_parities(
            {k: _flatten_for_hstack(_to_tensor(d[k])) for k in sorted(d)}
        )

    if isinstance(descriptor_result, list):
        if not descriptor_result:
            raise ValueError("extract_descriptor: empty list")
        parts = [extract_descriptor(x, stack_parities=stack_parities) for x in descriptor_result]
        if isinstance(parts[0], dict):
            keys = sorted({k for p in parts for k in p.keys()})
            return {k: torch.stack([p[k] for p in parts], dim=0) for k in keys}
        return torch.stack(parts, dim=0)

    if isinstance(descriptor_result, dict):
        if not descriptor_result:
            raise ValueError("Received empty dict")
        vals = list(descriptor_result.values())
        has_inner_dict = any(isinstance(v, dict) for v in vals)
        only_none_or_dict = all(v is None or isinstance(v, dict) for v in vals)
        if has_inner_dict and only_none_or_dict:
            return _from_nested_nu_parity(descriptor_result)
        if set(descriptor_result.keys()) <= {1, -1} and not any( isinstance(v, dict) for v in vals        ):
            return _from_parity_only(descriptor_result)
        raise TypeError(
            "extract_descriptor: expected {nu: {parity: tensor}} or "
            "{parity: tensor} with keys in {1, -1}, got incompatible dict layout"
        )

    raise TypeError("extract_descriptor expected dict or list of dicts")

def compute_distance_matrix(structure, dtype=torch.float64):
    """Compute the distance matrix for a structure."""
    if isinstance(structure, np.ndarray):
        structure_tensor = torch.tensor(structure, dtype=dtype)
    else:
        structure_tensor = structure.to(dtype)
    n_atoms = len(structure_tensor)
    distances = torch.zeros((n_atoms, n_atoms), dtype=dtype)
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            distances[i, j] = torch.norm(structure_tensor[i] - structure_tensor[j])
    
    return distances

def compute_volume(structure):
    """Compute the signed volume of a tetrahedron to detect chirality."""
    if len(structure) < 4:
        return 0.0

    # Convert to tensor if needed
    if isinstance(structure, np.ndarray):
        structure = torch.tensor(structure, dtype=torch.float64)

    # Use the first 4 atoms to form a tetrahedron
    p0, p1, p2, p3 = structure[:4]

    # Compute vectors
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0

    # Compute scalar triple product (signed volume)
    volume = torch.dot(v1, torch.linalg.cross(v2, v3))
    return volume.item()

def are_structures_symmetry_equivalent(structure1, structure2, tolerance=1e-6):
    """
    Check if two structures are symmetry equivalent by comparing distance matrices.
    
    Args:
        structure1: First structure (n, 3)
        structure2: Second structure (n, 3) 
        tolerance: Numerical tolerance for distance comparison
    Returns:
        bool: True if structures are symmetry equivalent
    """
    if len(structure1) != len(structure2):
        return False  
    
    # Then check distance matrices
    dist1 = compute_distance_matrix(structure1)
    dist2 = compute_distance_matrix(structure2)
    
    n_atoms = len(structure1)
    # loop over all relative permutations of the distance matrix
    for perm in permutations(range(n_atoms)):
        perm_dist2 = dist2[list(perm)][:, list(perm)]
        if torch.norm(dist1 - perm_dist2) < tolerance:
            return True
    
    return False


def generate_nu2_degen_structs(r,phi, center_species='C', ring_species='C'):
    """ phi: [list of angles, 2]"""
    from ase import Atoms
    structs = []
    n = phi.shape[0]
    for idx_str in range(2):
        layer = points_on_circle(r,phi[:,idx_str])   
        positions = np.zeros((n+1,3))
        positions[1:n+1,:2] = layer
        atom_string = center_species + ring_species*(n) 
        atoms = Atoms(atom_string, positions=positions, pbc=False)
        structs.append(atoms)

    return structs


def get_all_distances(
    atoms: Atoms,
    mic: bool = False,
    unordered_set: bool = False,
) -> list[tuple[int, int, float]] | list[float]:
    """Return all pair distances; with unordered_set=True return only distance values."""
    n = len(atoms)
    out: list[tuple[int, int, float]] | list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(atoms.get_distance(i, j, mic=mic))
            if unordered_set:
                out.append(d)
            else:
                out.append((i, j, d))
    if unordered_set:
        out = sorted(out)
    return out


def min_pairwise_distance(coords: torch.Tensor) -> float:
    """Minimum Euclidean distance over all distinct atom pairs; ``coords`` shape (N, 3)."""
    n = coords.shape[0]
    if n < 2:
        return float("inf")
    return float(torch.pdist(coords).min().item())


def unit_directions_from_origin(
    coords: torch.Tensor | np.ndarray,
    origin_tol: float = 1e-8,
    dtype: torch.dtype = torch.float64,
):
    """
    Compute unit direction vectors from the origin for each point.

    Returns
    -------
    unit_dirs : torch.Tensor, shape (N, 3)
        Unit vectors; rows with norm < origin_tol are set to zeros.
    norms : torch.Tensor, shape (N,)
        Original point norms.
    valid_mask : torch.Tensor, shape (N,), dtype=bool
        True where norm >= origin_tol (i.e. points not close to the origin, otherwise direction is ill defined).
    """
    if isinstance(coords, np.ndarray):
        pts = torch.tensor(coords, dtype=dtype)
    else:
        pts = coords.to(dtype=dtype)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {tuple(pts.shape)}")

    norms = torch.linalg.norm(pts, dim=1)
    valid_mask = norms >= float(origin_tol)
    unit_dirs = torch.zeros_like(pts, dtype=pts.dtype)
    if valid_mask.any():
        unit_dirs[valid_mask] = pts[valid_mask] / norms[valid_mask].unsqueeze(1)
    return unit_dirs, norms, valid_mask

import numpy as np
from invertatomrep.utils import unit_directions_from_origin

def gram_matrix(coords, center_idx=0, rcut=None, upper_triangle=True):
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).to(dtype=torch.float64)
    else:
        coords = coords.to(dtype=torch.float64)
    rel = coords - coords[center_idx].unsqueeze(0)
    if rcut is not None:
        keep = torch.linalg.norm(rel, dim=1) > 1e-8
        keep &= torch.linalg.norm(rel, dim=1) <= rcut
        rel = rel[keep]
    u, _, valid = unit_directions_from_origin(rel)
    G = u[valid] @ u[valid].T
    if upper_triangle:
        iu = np.triu_indices(len(u[valid]))
        sorted, _ = torch.sort(torch.round(G[iu], decimals=4))
        return sorted
    else:
        return torch.round(G, decimals=4)
