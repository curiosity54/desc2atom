import numpy as np
import torch
from typing import Optional, Tuple

__all__ = [
    "radial_cutoff",
    "smooth_radial_cutoff",
    "crystalnn_cutoff",
    "tensor_radial_local_environment",
    "tensor_smooth_radial_local_environment",
]


def tensor_radial_local_environment(
    points: torch.Tensor,
    atom_index: int,
    *,
    radius: float,
    central_atom: bool = True,
    include_central_site: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns local environment from (torch): relative vectors from site
    ``atom_index`` and uniform weights. 
    include_central_site
        If True/False, overrides ``central_atom`` for this call only. If None, use ``central_atom``.
    central_atom
        If True, the central site is included in the local environment.
    """
    include_central = central_atom if include_central_site is None else include_central_site
    center = points[atom_index]
    rel = points - center
    dist = torch.linalg.norm(rel, dim=-1)
    n = points.shape[0]
    device, dtype = points.device, points.dtype
    rc = torch.as_tensor(radius, device=device, dtype=dtype)
    not_center = torch.arange(n, device=device) != int(atom_index)
    in_shell = dist <= rc
    if include_central:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[atom_index] = True
        mask |= not_center & in_shell
    else:
        mask = not_center & in_shell
    sel = rel[mask]
    w = torch.ones(sel.shape[0], device=device, dtype=dtype)
    return sel, w


def _smooth_radial_weights(
    distances: torch.Tensor, radius: float, smearing_width: float
) -> torch.Tensor:
    """Cosine-smoothed radial weights (same as ``smooth_radial_cutoff``)."""
    r = torch.as_tensor(radius, device=distances.device, dtype=distances.dtype)
    smear = torch.as_tensor(smearing_width, device=distances.device, dtype=distances.dtype)
    f = torch.zeros_like(distances)
    inside = distances <= r - smear
    outside = distances >= r + smear
    transition = ~(inside | outside)
    f[inside] = 1.0
    x = (distances[transition] - (r - smear)) / (2 * smear)
    f[transition] = 0.5 * (1 + torch.cos(torch.pi * x))
    f[outside] = 0.0
    return f


def tensor_smooth_radial_local_environment(
    points: torch.Tensor,
    atom_index: int,
    *,
    radius: float,
    smearing_width: float,
    central_atom: bool = True,
    include_central_site: Optional[bool] = None,
    weight_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Smoothed local environment on a finite point cloud; same *semantics* as
    :func:`smooth_radial_cutoff` (pool radius ``radius + 3 * smearing_width``, cosine weights).
    """
    include_central = central_atom if include_central_site is None else include_central_site
    center = points[atom_index]
    rel = points - center
    dist = torch.linalg.norm(rel, dim=-1)
    n = points.shape[0]
    device, dtype = points.device, points.dtype
    smear = torch.as_tensor(smearing_width, device=device, dtype=dtype)
    r_cut = torch.as_tensor(radius, device=device, dtype=dtype)
    pool_r = r_cut + 3 * smear
    not_center = torch.arange(n, device=device) != int(atom_index)
    in_pool = dist <= pool_r
    if include_central:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[atom_index] = True
        mask |= not_center & in_pool
    else:
        mask = not_center & in_pool
    rel_sel = rel[mask]
    d_sel = dist[mask]
    f = _smooth_radial_weights(d_sel, radius, smearing_width)
    thr = torch.as_tensor(weight_threshold, device=device, dtype=dtype)
    keep = f > thr
    return rel_sel[keep], f[keep]


def radial_cutoff(radius=3.0, central_atom=False, **kwargs):
    """
    Get all neighbors to a site within a sphere of radius r.

    Returns a callable ``fn(structure, atom_index, *, return_tensor_local=False, ...)``.

    When ``return_tensor_local=True`` (keyword-only), ``structure`` must be a ``torch.Tensor``
    of shape ``(N, 3)`` and ``atom`` is the center index; the return value is
    ``(rel_vectors, weights)`` from :func:`tensor_radial_local_environment` instead of pymatgen
    sites. Use ``include_central_site=`` to override ``central_atom`` for that call only.
    """

    def neighbor_fn(structure, atom, *, return_tensor_local=False, include_central_site=None):
        if return_tensor_local:
            return tensor_radial_local_environment(
                structure,
                int(atom),
                radius=radius,
                central_atom=central_atom,
                include_central_site=include_central_site,
            )
        if not central_atom:
            return structure.get_neighbors(structure[int(atom)], radius)
        return [structure[int(atom)]] + structure.get_neighbors(
            structure[int(atom)], radius
        )

    return neighbor_fn



def smooth_radial_cutoff(radius=3.0, central_atom=False, smoothing_fn='cosine', smearing_width=0.1, **kwargs):
    """
    Return a function that computes smoothed neighbor contributions based on distance.

    The returned callable supports ``fn(structure, atom, *, return_tensor_local=False, ...)``.
    If ``return_tensor_local=True``, ``structure`` is a ``(N, 3)`` tensor and ``atom`` is the
    center index; returns ``(rel_vectors, weights)`` from :func:`tensor_smooth_radial_local_environment`
    (cosine smoothing only; same pool-radius rule as the pymatgen path).

    For pymatgen structures, the return is still ``list[tuple[Site, weight]]``.
    """
    
    def smoothing_weight(distance):
        if smoothing_fn == 'gaussian':
            return torch.exp(-0.5 * ((distance - radius) / smearing_width)**2)
        elif smoothing_fn == 'cosine':
            return shifted_cosine_cutoff(distance)
    #     elif smoothing_fn == 'cosine':
    #         delta = smearing_width
    #         if distance <= (radius - delta):
    #             return 1.0
    #         elif distance >= (radius + delta):
    #             return 0.0
    #         else:
    #             x = (distance - radius + delta) / (2 * delta)
    #             return 0.5 * (1 + torch.cos(np.pi * x))
    #     else:
    #         raise ValueError(f"Unknown smoothing_fn: {smoothing_fn}")
    def shifted_cosine_cutoff(r):
        # r = (r)
        f = torch.zeros_like(r)

        inside = r <= radius - smearing_width
        outside = r >= radius + smearing_width
        transition = ~(inside | outside)

        f[inside] = 1.0
        x = (r[transition] - (radius - smearing_width)) / (2 * smearing_width)
        f[transition] = 0.5 * (1 + torch.cos(np.pi * x))

        return f

    def get_smoothed_neighbors(structure, atom_index, weight_threshold=1e-6):
        center = structure[int(atom_index)]
        raw_neighbors = structure.get_neighbors(center, radius + 3*smearing_width) # get initial neighbors within a larger radius to ensure we capture all relevant neighbors
        if central_atom:
            raw_neighbors = [center] + raw_neighbors
        distances = torch.tensor([n.distance(center) for n in raw_neighbors])
        weights = smoothing_weight(distances)
        mask = weights > weight_threshold
        filtered_neighbors = [raw_neighbors[i] for i in np.where(mask)[0]]
        filtered_weights = weights[mask]
            
        # neighbors = []
        # if central_atom:
        #     neighbors.append((center, 1.0)) # atom, weight 
        # for neighbor in raw_neighbors:
        #     weight = smoothing_weight(neighbor.distance(center))
        #     if weight > weight_threshold:
        #         neighbors.append((neighbor, weight))
        
        return list(zip(filtered_neighbors, filtered_weights))

    def neighbor_fn(
        structure,
        atom,
        *,
        return_tensor_local=False,
        include_central_site=None,
        weight_threshold=1e-6,
    ):
        if return_tensor_local:
            if smoothing_fn != "cosine":
                raise ValueError(
                    "return_tensor_local for smooth_radial_cutoff only supports smoothing_fn='cosine'; "
                    f"got {smoothing_fn!r}"
                )
            return tensor_smooth_radial_local_environment(
                structure,
                int(atom),
                radius=radius,
                smearing_width=smearing_width,
                central_atom=central_atom,
                include_central_site=include_central_site,
                weight_threshold=weight_threshold,
            )
        return get_smoothed_neighbors(structure, int(atom), weight_threshold=weight_threshold)

    return neighbor_fn


def crystalnn_cutoff(radius=None, central_atom=False, **kwargs):
    """
    This function creates a cutoff function using the CrystalNN approach (with parameters set for pure geometric neighbor finding).

    Parameters:
        structure (Structure): The structure for which the cutoff is to be calculated.
        site_index (int): The site number of the atom for which the cutoff is to be calculated.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the CrystalNN approach.
    """
    # TODO: return central atom as neighbor! 

    def cutoff_function(structure, site_index):
        nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
        local_env =  nn.get_nn_info(structure, site_index)
        neighbors = [neighbor['site'] for neighbor in local_env]
        if central_atom:
            return [structure[int(site_index)]] + neighbors
        return neighbors
    return cutoff_function



