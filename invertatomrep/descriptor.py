import torch 
import e3nn.io as io
import e3nn
import e3nn.o3 as o3
# from ase.data import atomic_numbers, chemical_symbols
from ase.symbols import string2symbols, symbols2numbers
import invertatomrep.cutoff  
import warnings
import math
import numpy as np 
from collections import defaultdict
import itertools
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def enumerate_ls_nu(lmax, correlation_order=3, L=0, p=1, lcut=None, include_intermediates=True):
    """
    Returns all chains of form (l1, l2, L12, l3, L123, ..., ln) with the intermediate L's,
    such that final L123...n == L and parity is p = (-1)^(l1 + l2 + ... + ln).
    PASS p = -1 for L odd and 1 for L even to get POLAR TENSORS 

    Args:
        lmax: Maximum l value
        correlation_order: Order of correlation (nu)
        L: Target final angular momentum
        p: Target parity
        lcut: Cutoff for l values (defaults to lmax)
        include_intermediates: If True, include intermediate L values. If False, return only (l1, l2, ..., ln)
    """
    # sort l's in ascending order, i.e. l1 <= l2 <= ... <= ln
    assert correlation_order > 1, "Correlation order must be greater than 1"
        
    if lcut is None:
        lcut = lmax
    
    n = correlation_order
    results = []

    def recurse(idx, l_list, L_list, current_L):
        if idx == n:
            # Only accept if final angular momentum matches and parity is correct
            if current_L == L:
                # Check parity: (-1)^(l1 + l2 + ... + ln)
                parity = (-1) ** sum(l_list)
                if parity == p:
                    # Skip if any two l values are equal and the third is odd, as the term must be zero (see SI)
                    # This constraint only applies when L = 0
                    if correlation_order == 3 and L == 0 and p == -1:
                        l1, l2, l3 = l_list
                        unique_count = len(set([l1, l2, l3]))
                        if unique_count <= 2 and any(l % 2 == 1 for l in [l1, l2, l3]):
                            return  
                    
                    if include_intermediates:
                        # Include intermediate L values: (l1, l2, L12, l3, L123, ...)
                        out = []
                        out.append(l_list[0])
                        out.append(l_list[1])
                        for i in range(n-2):
                            out.append(L_list[i])
                            out.append(l_list[i+2])
                        results.append(tuple(out))
                    else:
                        # Return only the sorted l values: (l1, l2, l3, ...)
                        results.append(tuple(l_list))
            return

        # Ensure l_next >= l_list[-1] to maintain sorted order
        min_l_next = l_list[-1] if l_list else 0
        for l_next in range(min_l_next, lcut+1):
            # Triangle rule for combining previous total with l_next
            for L_next in range(abs(current_L - l_next), current_L + l_next + 1):
                recurse(idx+1, l_list + [l_next], L_list + [L_next], L_next)

    for l1 in range(0, lmax+1):
        for l2 in range(l1, lmax+1):  # Start l2 from l1 to ensure l1 <= l2
            for L12 in range(abs(l1 - l2), l1 + l2 + 1):
                recurse(2, [l1, l2], [L12], L12)

    if not include_intermediates:
        seen = set()
        unique_results = []
        for result in results:
            if result not in seen:
                seen.add(result)
                unique_results.append(result)
        return unique_results
    
    return results


def get_lm_slice(l):
    """
    Helper function to get the slice corresponding to m=-l..l in a dense
    storage block for Y^l_m-like coefficients
    """
    return slice(l * l, (l + 1) * (l + 1), 1)

def SPHparitylabel(L, pseudo = False, return_int=False):
    """ Returns the parity label for spherical harmonics (i.e. a polar tensor) of degree L."""
    if not pseudo:
        if return_int:
            return (-1)**L 
        else: 
            return 'e' if L%2==0 else 'o'
    else: 
        if return_int:
            return (-1)**(L+1) 
        else: 
            # pseudo tensors are odd for even L and even for odd L
            # i.e. they have the opposite parity of polar tensors
             return 'o' if L%2==0 else 'e'

def get_element_number(atom):
    pass 
    return symbols2numbers(['H'])
    return chemical_symbols[atomic_numbers[atom.symbol]]


##------- acdc based ------
def spherical_expansion_coeffs(lmax, atom_coords, values= None, factor=1.0, dtype = torch.float64):
    # This implements \sum_i v_i \delta(\br - \br_i) where v_i is the value at position \br_i
    # we computed v_i as the norm of teh vector from the origin to the atom position, i.e. v_i = ||\br_i|| * neighbor_weight
    sphten = io.SphericalTensor(lmax, p_val=1, p_arg=-1)
    if values is None:
        values =torch.linalg.norm(atom_coords, axis=1) # distance_ij 
        values = values*factor
    # TODO: species filtering , i.e. implement filter in DescriptorCalculator.get_local_environment

    spex = sphten.sum_of_diracs(atom_coords, values=values).to(dtype = dtype)
    
    return spex


def zeroth_order(atom_coords, dtype = torch.float64):
    # This implements \sum_i \delta(\hat{r} - \hat{r_i}) (i.e. not weighted by distance)
    # should be equivalent to the number of neighbors, but here depends on the normalizations used within sphtn
    sphten = io.SphericalTensor(0, p_val=1, p_arg=-1)
    values = torch.ones_like(atom_coords[:, 0], dtype=atom_coords.dtype, device=atom_coords.device)  # uniform value for all atoms    
    sp0 = sphten.sum_of_diracs(atom_coords, values=values).to(dtype = dtype)
    return sp0


def _squeeze_irrep_dim_tensor(t: torch.Tensor, radial_spex: bool) -> torch.Tensor:
    """Remove singleton ``(2L+1)`` axis for ``L=0`` blocks (DescriptorCalculator outputs only).
    radial_spex: bool, whether the descriptor is radial (i.e. has a singleton ``L=0`` axis on axis 1)
    Legacy invariant stacks end with ``(..., l_combinations, 1)``; radial stacks
    ``(..., 1, l_combinations, *radial_dims)``. ``equivariant_correlation`` itself is unchanged.
    """
    if radial_spex:
        if t.ndim >= 2 and t.shape[1] == 1:
            return t.squeeze(1)
    else:
        if t.ndim >= 2 and t.shape[-1] == 1:
            return t.squeeze(-1)
    return t


def squeeze_irrep_dim_nested(obj, radial_spex: bool):
    """Walk nested dict/tuple descriptor payloads and squeeze the singleton ``L=0`` irrep axis (see :func:`_squeeze_irrep_dim_tensor`)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return _squeeze_irrep_dim_tensor(obj, radial_spex)
    if isinstance(obj, dict):
        return {k: squeeze_irrep_dim_nested(v, radial_spex) for k, v in obj.items()}
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], torch.Tensor):
        return (
            _squeeze_irrep_dim_tensor(obj[0], radial_spex),
            obj[1],
        )
    if isinstance(obj, tuple):
        return tuple(squeeze_irrep_dim_nested(x, radial_spex) for x in obj)
    return obj


def equivariant_powerspectrum(rho1i, L, parity:int=0, cg=None, **kwargs):
    """ Compute all equivariant powerspectra up to lcut.
    lcut:int, max spherical harmonic degree to compute in the result
    parity: 'all', 'even', 'odd' corresponds to 0, 1, -1 (unlike usual ACDC where 1 = polar tensor and -1 = pseudo tensor)
    
    Returns:
        dict: Dictionary with parity as keys and tuples of (tensor_values, combinations) as values.
              combinations is a list of (l1, l2, L) tuples.
    """
    # maximum l value available in rho1i
    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-1], dtype=torch.float32))) - 1

    if cg is None:
        cg = CG(lmax=lmax_rho1i, device=rho1i.device)
    
    combination_string = "Lmn,im,in->iL"
    rho2_list = {1:[], -1:[]}
    combinations_list = {1:[], -1:[]}
    
    # Get sorted (l1, l2) combinations that can couple to L
    # If parity=0, we need to get combinations for both parities
    if parity == 0:
        l_combinations_1 = enumerate_ls_nu(lmax_rho1i, correlation_order=2, L=L, p=1, lcut=lmax_rho1i, include_intermediates=False)
        l_combinations_neg1 = enumerate_ls_nu(lmax_rho1i, correlation_order=2, L=L, p=-1, lcut=lmax_rho1i, include_intermediates=False)
        l_combinations = l_combinations_1 + l_combinations_neg1
    else:
        l_combinations = enumerate_ls_nu(lmax_rho1i, correlation_order=2, L=L, p=parity, lcut=lmax_rho1i, include_intermediates=False)
    
    for l1, l2 in l_combinations:
        try:
            cg_coeffs = cg.cg[l1, l2][L]
        except KeyError:
            warnings.warn(f"KeyError: {l1}, {l2}, {L}")
            continue
                
        # p1 = SPHparitylabel(l1, pseudo=False, return_int=True)
        # p2 = SPHparitylabel(l2, pseudo=False, return_int=True)
        
        rho2_ = torch.einsum(combination_string, cg_coeffs, rho1i[:, get_lm_slice(l1)], rho1i[:, get_lm_slice(l2)])
        
        # For polar tensors: output parity = (-1)^(l1 + l2)
        combination_parity = (-1)**(l1 + l2) # p1*p2
        rho2_list[combination_parity].append(rho2_)
        combinations_list[combination_parity].append((l1, l2, L))

    if parity == 0:
        result = {}
        if rho2_list[1]:  # Check if list is not empty
            result[1] = (torch.stack(rho2_list[1], dim=-2), combinations_list[1])
        if rho2_list[-1]:  # Check if list is not empty
            result[-1] = (torch.stack(rho2_list[-1], dim=-2), combinations_list[-1])
        
        if not result:  # If both lists are empty
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    if parity in rho2_list and rho2_list[parity]:  
        return {parity: (torch.stack(rho2_list[parity], dim=-2), combinations_list[parity])}
    else:
        available_parities = [p for p in rho2_list if rho2_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            return {available_parities[0]: (torch.stack(rho2_list[available_parities[0]], dim=-2), combinations_list[available_parities[0]])}
        else:
            warnings.warn(f"No valid combinations found for any parity")
            return None

def powerspectrum_acdc(spherical_expansion,  cg = None, **kwargs):
    """ ACDC style powerspectrum calc - returns only INVARIANTS"""
    result = equivariant_powerspectrum(spherical_expansion, 0, parity=1, cg=cg)
    if result is not None:
        return result[1][0] 
    return None

def equivariant_bispectrum(rho1i, L, lcut=None, parity:int=0, cg=None, rho2_L=None):
    """Compute equivariant bispectrum for given L.
    
    Returns:
        dict: Dictionary with parity as keys and tuples of (tensor_values, combinations) as values.
              combinations is a list of (l1, l2, L12, l3, L) tuples.
    """
    # maximum l value available in rho1i
    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-1], dtype=torch.float32))) - 1
    if lcut is None:
        lcut = lmax_rho1i
    if cg is None:
        cg = CG(lmax=lmax_rho1i, device=rho1i.device)
    
    if rho2_L is None:
        rho2_L = {l: equivariant_powerspectrum(rho1i, l, parity=0, cg=cg) for l in range(lcut+1)}
    
    rho3_list = {1:[], -1:[]}
    combinations_list = {1:[], -1:[]}
    
    # For bispectrum, we need to iterate over (l1, l2, L12, l3) combinations that can couple to L
    # We use enumerate_ls_nu with include_intermediates=True to get valid combinations directly
    if parity == 0:
        l_combinations_1 = enumerate_ls_nu(lcut, correlation_order=3, L=L, p=1, lcut=lcut, include_intermediates=True)
        l_combinations_m1 = enumerate_ls_nu(lcut, correlation_order=3, L=L, p=-1, lcut=lcut, include_intermediates=True)
        l_combinations = l_combinations_1 + l_combinations_m1
    else:
        l_combinations = enumerate_ls_nu(lcut, correlation_order=3, L=L, p=parity, lcut=lcut, include_intermediates=True)
    
    total_processed = 0
    for l1, l2, L12, l3 in l_combinations:
                try:
                    # First coupling: (l1, l2) -> L12
                    cg_coeffs_12 = cg.cg[l1, l2][L12]
                    # Second coupling: (L12, l3) -> L
                    cg_coeffs_123 = cg.cg[L12, l3][L]
                except KeyError:
                    continue
                
                # Compute the intermediate tensor product: rho1i[l1] * rho1i[l2] -> L12
                intermediate = torch.einsum("Lmn,im,in->iL", cg_coeffs_12, rho1i[:, get_lm_slice(l1)], rho1i[:, get_lm_slice(l2)])
                
                # Now combine with rho1i[l3] to get the final result: intermediate * rho1i[l3] -> L
                rho3_ = torch.einsum("Lmn,im,in->iL", cg_coeffs_123, intermediate, rho1i[:, get_lm_slice(l3)])
                
                # For polar tensors: output parity = (-1)^(l1 + l2 + l3)
                combination_parity = (-1)**(l1 + l2 + l3+L)
                rho3_list[combination_parity].append(rho3_)
                combinations_list[combination_parity].append((l1, l2, L12, l3, L))
                total_processed += 1
    if parity == 0:
        result = {}
        if rho3_list[1]:  # Check if list is not empty
            result[1] = (torch.stack(rho3_list[1], dim=-2), combinations_list[1])
        if rho3_list[-1]:  # Check if list is not empty
            result[-1] = (torch.stack(rho3_list[-1], dim=-2), combinations_list[-1])
        if not result:  # If both lists are empty
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    # For specific parity
    if parity in rho3_list and rho3_list[parity]:  # Check if list exists and is not empty
        return {parity: (torch.stack(rho3_list[parity], dim=-2), combinations_list[parity])}
    else:
        available_parities = [p for p in rho3_list if rho3_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            return {available_parities[0]: (torch.stack(rho3_list[available_parities[0]], dim=-2), combinations_list[available_parities[0]])}
        else:
            warnings.warn(f"No valid combinations found for any parity")
            return None


def equivariant_correlation(
    rho1i,
    L,
    correlation_order=3,
    lcut=None,
    parity:int=0,
    cg=None,
    rho_prev=None,
    dtype=torch.float64,
    return_combinations=False,
    descriptor_cache=None,
):
    """Compute equivariant correlation of arbitrary order for given L.
    
    Args:
        rho1i: Spherical expansion coefficients
        L: Target final angular momentum
        correlation_order: Order of correlation (2=power, 3=bispectrum, 4=trispectrum, etc.)
        lcut: Maximum l value to consider
        parity: Target parity (0=both, 1=even, -1=odd)
        cg: Clebsch-Gordan coefficients
        rho_prev: Pre-computed lower-order correlations (optional)
        descriptor_cache: Optional dict reused across nested calls to cache intermediate
            lower-order correlation results (same object passed through recursion).
    
    Returns:
        dict: Dictionary with parity as keys and tuples of (tensor_values, combinations) as values.
              combinations is a list of tuples containing the angular momentum chain.
    """
    if descriptor_cache is None:
        descriptor_cache = {}

    # Allow passing a list of correlation orders, returning a nested dict:
    # {nu: {parity: descriptor}} for each requested nu.
    if isinstance(correlation_order, (list, tuple)):
        if len(correlation_order) == 0:
            raise ValueError(
                "equivariant_correlation: correlation_order list is empty. "
                "For zeroth order, call DescriptorCalculator.get_descriptor with correlation_order=0 "
                "(or include 0 in the calculator's correlation_order list)."
            )

        orders = sorted(set(int(nu) for nu in correlation_order))
        if 0 in orders:
            raise ValueError(
                "equivariant_correlation cannot handle correlation_order=0 in a list (it only sees rho1i). "
                "Use DescriptorCalculator.get_descriptor with correlation_order list that includes 0."
            )

        lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-1], dtype=dtype))) - 1
        if lcut is None:
            lcut = lmax_rho1i
        if cg is None:
            cg = CG(lmax=lmax_rho1i, device=rho1i.device, dtype=dtype)

        out = {}
        cached_rho_prev_order = None
        cached_rho_prev = None  # dict: intermediate_L -> {parity_key: (tensor_vals, combinations)}

        # We only compute/store the lower-order tensors when they are needed as rho_prev.
        for nu_int in orders:
            if nu_int == 1:
                if parity in (0, 1):
                    out[1] = {1: rho1i[:, get_lm_slice(0)]}
                continue
            if nu_int == 2:
                out[2] = equivariant_correlation(
                    rho1i,
                    L,
                    correlation_order=2,
                    lcut=lcut,
                    parity=parity,
                    cg=cg,
                    rho_prev=None,
                    dtype=dtype,
                    return_combinations=return_combinations,
                    descriptor_cache=descriptor_cache,
                )
                # In this codebase, parity p=-1 terms may be empty; keep only p=1 when parity=0.
                if parity == 0 and isinstance(out[2], dict) and 1 in out[2]:
                    out[2] = {1: out[2][1]}
                continue

            lower_order = nu_int - 1
            if cached_rho_prev_order != lower_order:
                # Build rho_prev for all intermediate L values once, then reuse.
                cached_rho_prev = {}
                for l_target in range(lcut + 1):
                    cached_rho_prev[l_target] = equivariant_correlation(
                        rho1i,
                        l_target,
                        correlation_order=lower_order,
                        lcut=lcut,
                        parity=0,
                        cg=cg,
                        rho_prev=None,
                        dtype=dtype,
                        return_combinations=True,
                        descriptor_cache=descriptor_cache,
                    )
                cached_rho_prev_order = lower_order

            parity_dict_L = equivariant_correlation(
                rho1i,
                L,
                correlation_order=nu_int,
                lcut=lcut,
                parity=parity,
                cg=cg,
                rho_prev=cached_rho_prev,
                dtype=dtype,
                return_combinations=return_combinations,
                descriptor_cache=descriptor_cache,
            )
            # if parity == 0 and isinstance(parity_dict_L, dict) and 1 in parity_dict_L:
            #     out[nu_int] = {1: parity_dict_L[1]}
            # else:
            out[nu_int] = parity_dict_L

        return out

    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-1], dtype=dtype))) - 1
    if lcut is None:
        lcut = lmax_rho1i
    if cg is None:
        cg = CG(lmax=lmax_rho1i, device=rho1i.device, dtype=dtype)
    
    # Handle special cases
    if correlation_order == 1:
        return rho1i
    elif correlation_order == 2:
        return equivariant_powerspectrum(rho1i, L, parity=parity, cg=cg)
    
    if rho_prev is None:
        # Precompute lower-order correlations at all possible intermediate L values,
        # so the recursion can look up rho_prev[intermediate_L].
        rho_prev = {}
        for l in range(lcut + 1):
            corr_prev = correlation_order - 1
            cache_key = (
                corr_prev,
                l,
                0,  # parity=0
                True,  # return_combinations=True for rho_prev
                lcut,
                str(dtype),
                str(rho1i.device),
            )
            if cache_key in descriptor_cache:
                rho_prev[l] = descriptor_cache[cache_key]
            else:
                val = equivariant_correlation(
                    rho1i,
                    l,
                    correlation_order=corr_prev,
                    lcut=lcut,
                    parity=0,
                    cg=cg,
                    rho_prev=None,
                    dtype=dtype,
                    return_combinations=True,
                    descriptor_cache=descriptor_cache,
                )
                descriptor_cache[cache_key] = val
                rho_prev[l] = val
    
    rho_list = {1:[], -1:[]}
    combinations_list = {1:[], -1:[]}
    
    # Get valid combinations for the given correlation order
    if parity == 0:
        l_combinations_1 = enumerate_ls_nu(lcut, correlation_order=correlation_order, L=L, p=1, lcut=lcut, include_intermediates=True)
        l_combinations_m1 = enumerate_ls_nu(lcut, correlation_order=correlation_order, L=L, p=-1, lcut=lcut, include_intermediates=True)
        l_combinations = l_combinations_1 + l_combinations_m1
    else:
        l_combinations = enumerate_ls_nu(lcut, correlation_order=correlation_order, L=L, p=parity, lcut=lcut, include_intermediates=True)
    
    total_processed = 0
    for combination in l_combinations:
        l_values = []
        L_intermediates = []
        # Extract l values and intermediate L values from the combination
        # For correlation_order=3: (l1, l2, L12, l3) -> l_values=[l1, l2, l3], L_intermediates=[L12]
        # For correlation_order=4: (l1, l2, L12, l3, L123, l4) -> l_values=[l1, l2, l3, l4], L_intermediates=[L12, L123]
        l_values.append(combination[0])  # l1
        l_values.append(combination[1])    # l2
        
        # Extract remaining l values and intermediate L values
        for i in range(correlation_order - 2):
            L_intermediates.append(combination[2 + 2*i])      # L12, L123, ...
            l_values.append(combination[2 + 2*i + 1])        # l3, l4, ...
        # Add the final L
        L_intermediates.append(L)
        combination_tuple = tuple(combination) + (L,)
        
        try:
            # Try recursive approach if rho_prev is available
            current_tensor = None
            if rho_prev is not None:
                intermediate_L = L_intermediates[0]
                last_l = l_values[-1]
                
                if intermediate_L in rho_prev and rho_prev[intermediate_L] is not None:
                    rho_prev_result = rho_prev[intermediate_L]
                    target_combo = tuple(combination[:-1])
                    # Find the matching combination in rho_prev
                    for parity_key in rho_prev_result:
                        tensor_vals, combinations = rho_prev_result[parity_key]
                        for idx, combo in enumerate(combinations):
                            if combo == target_combo:
                                current_tensor = tensor_vals[:, idx, :]
                                break
                        if current_tensor is not None:
                            break
                    
                    if current_tensor is not None:
                        # Use recursive approach: combine with last l value
                        rho_ = torch.einsum("Lmn,im,in->iL", 
                                          cg.cg[intermediate_L, last_l][L], 
                                          current_tensor, 
                                          rho1i[:, get_lm_slice(last_l)])
                        current_tensor = None  # Skip fallback
            
            # Fallback: compute sequentially from rho1i (used for all cases where recursive fails)
            if current_tensor is None:
                current_tensor = torch.einsum("Lmn,im,in->iL", 
                                            cg.cg[l_values[0], l_values[1]][L_intermediates[0]], 
                                            rho1i[:, get_lm_slice(l_values[0])], 
                                            rho1i[:, get_lm_slice(l_values[1])])
                
                # Sequentially apply the remaining CG coefficients
                for i in range(1, correlation_order - 1):
                    current_tensor = torch.einsum("Lmn,im,in->iL", 
                                                cg.cg[L_intermediates[i-1], l_values[i+1]][L_intermediates[i]], 
                                                current_tensor, 
                                                rho1i[:, get_lm_slice(l_values[i+1])])
                rho_ = current_tensor
            
        except KeyError:
            continue
        
        l_sum = sum(l_values)
        combination_parity = (-1)**(l_sum + L)
        
        rho_list[combination_parity].append(rho_)
        combinations_list[combination_parity].append(combination_tuple)
        total_processed += 1
    
    if parity == 0:
        result = {}
        if rho_list[1]:  
            if return_combinations:
                result[1] = (torch.stack(rho_list[1], dim=-2), combinations_list[1])
            else:
                result[1] = torch.stack(rho_list[1], dim=-2)
        if rho_list[-1]:  
            if return_combinations:
                result[-1] = (torch.stack(rho_list[-1], dim=-2), combinations_list[-1])
            else:
                result[-1] = torch.stack(rho_list[-1], dim=-2)
        
        if not result:  
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    # For specific parity
    if parity in rho_list and rho_list[parity]:  # Check if list exists and is not empty
        if return_combinations:
            return {parity: (torch.stack(rho_list[parity], dim=-2), combinations_list[parity])}
        else:
            return {parity: torch.stack(rho_list[parity], dim=-2)}
    else:
        available_parities = [p for p in rho_list if rho_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            if return_combinations:
                return {available_parities[0]: (torch.stack(rho_list[available_parities[0]], dim=-2), combinations_list[available_parities[0]])}
            else:
                return {available_parities[0]: torch.stack(rho_list[available_parities[0]], dim=-2)}
        else:
            warnings.warn(f"No valid combinations found for any parity")
            return None


def bispectrum_acdc(spherical_expansion, parity = 0, cg =None, lcut=None, rho2_L = None):
    """ ACDC style bispectrum calc - returns only INVARIANTS"""
    
    bis = equivariant_bispectrum(spherical_expansion, 0, lcut=lcut, parity=parity, cg=cg, rho2_L=rho2_L)
    if bis is not None:
        if parity == 0:
            # Check which parities are available
            available_parities = list(bis.keys())
            if len(available_parities) == 2:
                return torch.cat([bis[1][0], bis[-1][0]], dim=1)
            elif len(available_parities) == 1:
                warnings.warn(f'Only found parity = {next(iter(bis.keys()))}')
                return next(iter(bis.values()))[0]
            else:
                return None
        return bis[parity][0]  
    return None

def equivariant_trispectrum(rho1i, L, lcut=3, parity:int=0, cg=None, rho3_L=None):
    """Compute equivariant trispectrum for given L (wrapper for backward compatibility).
    
    Returns:
        dict: Dictionary with parity as keys and tuples of (tensor_values, combinations) as values.
              combinations is a list of (l1, l2, L12, l3, L123, l4, L) tuples.
    """
    return equivariant_correlation(rho1i, L, correlation_order=4, lcut=lcut, parity=parity, cg=cg, rho_prev=rho3_L)

def trispectrum_acdc(spherical_expansion, parity=0, cg=None, lcut=None, rho3_L=None):
    """ ACDC style trispectrum calc - returns only INVARIANTS"""
    tris = equivariant_correlation(spherical_expansion, 0, correlation_order=4, cg=cg, lcut=lcut, rho_prev=rho3_L)
    if tris is not None:
        if parity == 0:
            # Check which parities are available
            available_parities = list(tris.keys())
            if len(available_parities) == 2:
                return torch.cat([tris[1][0], tris[-1][0]], dim=1)
            elif len(available_parities) == 1:
                return next(iter(tris.values()))[0]
            else:
                return None
        return tris[parity][0]  
    return None



def nu_invariant_acdc(nu:list, rho1i, lcut, parity=0, cg=None):
    #nu can be 1,2,3
    if nu == 1: 
        return rho1i
    elif nu == 2:
        nu2 = powerspectrum_acdc(rho1i,  cg = None)
        return torch.hstack(rho1i, nu2, dim=-1)
    if nu >2: 
        rho2_L = {l: equivariant_powerspectrum(rho1i, l, parity=0, cg=cg) for l in range(lcut+1)}
        nu2 = rho2_L[0]
        nu3 = bispectrum_acdc(rho1i,  cg = None, rho2_L=rho2_L, parity=parity, lcut=lcut)
        nu3_even = torch.hstack((rho1i, nu2))
        return torch.hstack(rho1i, nu2, nu3, dim=-1)

##------- e3nn based ------
def powerspectrum_rtp(spherical_expansion, lcut, parity = 'all'):
    irreps_out = '0'+parity if parity in ['e', 'o'] else ['0e','0o']
    rtp = e3nn.o3.ReducedTensorProducts('ij=ji', i=o3.Irreps.spherical_harmonics(lmax=lcut), 
                               filter_ir_out=irreps_out, filter_ir_mid=o3.Irrep.iterator(lcut))
    return rtp(spherical_expansion, spherical_expansion), rtp

def bispectrum_rtp(spherical_expansion, lcut, parity = 'all'):
    irreps_out = '0'+parity if parity in ['e', 'o'] else ['0e','0o']
    rtp = e3nn.o3.ReducedTensorProducts('ijk=jik=ikj', i=o3.Irreps.spherical_harmonics(lmax=lcut), 
                               filter_ir_out=irreps_out, filter_ir_mid=o3.Irrep.iterator(lcut))
    return rtp(spherical_expansion, spherical_expansion, spherical_expansion), rtp

#----- DescriptorCalculator -----
# has the functionality to 
# 1.compute the descriptor for a given atom in a structure (instantiate a pymatgen structure from absolute coordinates)
# (use_structure=True, atom_index=ATOM_INDEX). if ATOM_INDEX is None, use the first atom.
# 2.compute the descriptor for a given tensor of absolute coordinates (using the cutoff function to form relative vectors)
# (use_structure=False, atom_index=ATOM_INDEX), if ATOM_INDEX is not None, use the cutoff function to form relative vectors from the central atom to the neighbors.
# 3.compute the descriptor for a given tensor of relative coordinates (using the cutoff function to form relative vectors)
# (use_structure=False, atom_index=None)

class DescriptorCalculator(torch.nn.Module):
    def __init__(self, cutoff_type:str, lmax:int, rcut:float, correlation_order:int,
                 central_atom:bool=True, 
                 smoothing_fn:str=None, smearing_width:float=None, 
                 dtype=torch.float64, 
                 device='cpu', 
                #  center_origin:bool=True,
                 lcut=None,
                 cg = None
                 ):
        """
        cutoff_type: str, type of cutoff function to use (e.g. 'smooth_radial_cutoff')
        lmax: int, maximum spherical harmonic degree to compute
        central_atom: bool, whether to include the central atom in the local environment
        smoothing_fn: str, type of smoothing function to use (e.g. 'cosine', 'gaussian')
        smearing_width: float, width of the smoothing function (default is 0.1)
        dtype: torch.dtype, data type for computations (default is torch.float32)
        cg: CG object, for equivariant powerspectrum and bispectrum calculations

        Summary of possible use modes : 
        - 
        """
        super().__init__()
        self.lmax = lmax 
        self.correlation_order = correlation_order
        self.dtype = dtype
        self.device = device
        self.lcut = lcut if lcut is not None else lmax
        self.cutoff_type = cutoff_type

        if self.cutoff_type not in invertatomrep.cutoff.__all__:
            raise ValueError(f"Unknown cutoff type: {cutoff_type}. Available types: {invertatomrep.cutoff.__all__}")
        if self.cutoff_type.lower() == 'smooth_radial_cutoff':
            if smoothing_fn is None: 
                smoothing_fn = 'cosine'
            if smearing_width is None:
                smearing_width = 0.1
        
        warnings.warn(f"Using cutoff type: {self.cutoff_type} with smoothing function: {smoothing_fn} and smearing width: {smearing_width}")
        self.cutoff = getattr(invertatomrep.cutoff, cutoff_type)(radius = rcut, central_atom=central_atom, smoothing_fn=smoothing_fn, smearing_width=smearing_width)
        self.rcut = rcut
        self.smoothing_fn = smoothing_fn
        self.smearing_width = smearing_width

        self.central_atom = central_atom
        self.neighbor_weights = 1 # default weight for neighbors, modified by the cutoff function
        if cg is None:
            self.cg = CG(lmax=self.lmax, device=self.device) # for equivariant powerspectrum and bispectrum calculations
        else:
            self.cg = cg
        
    def get_local_environment(self,structure, atom_index = None):
        """
        Gets the neighbors of a given atom.
        structure: pymatgen Structure object
        atom_index (int): Index of central atom. ``None`` means use the first atom.

        Returns:
            list: coordinates and weights of the neighbors of the given atom.
        """
        if atom_index is None:
            warnings.warn("No atom index provided, using the first atom")
            atom_index = 0
        neighbors = self.cutoff(structure, atom_index)
        if self.cutoff_type.lower() == 'smooth_radial_cutoff':
            # self.neighbor_weights = {}
            neighbors, neighbor_weights = zip(*neighbors) 
        else: 
            neighbor_weights = [self.neighbor_weights] * len(neighbors)
        return neighbors, neighbor_weights


    def get_local_geometry(self, structure, atom_index = None):
        """
        Gets the relative interatomic vectors and weights of the neighbors of a given atom.

        Parameters:
            structure: PyMatGen Structure object
            atom_index (int): The atom site number. If ``None``, defaults to ``0`` 

        """
        local_env = self.get_local_environment(structure, atom_index) # neighbors and their weights
        if local_env is None:
            raise ValueError(f"No local environment found for atom index {atom_index} in the structure.")
        atoms, weights = local_env[0], local_env[1]
        relative_interatomic_vectors = torch.from_numpy(
            np.stack([atom.coords for atom in atoms], axis=0)
            - structure[atom_index].coords.reshape(1, 3)
        ).to(self.device, dtype=self.dtype)
        w = torch.as_tensor(weights, dtype=self.dtype, device=self.device).reshape(-1)
        return relative_interatomic_vectors, w

    def local_geometry_neighbors_relative(self, points: torch.Tensor, atom_index: int):
        """
        points: absolute cartesian coordinates of the points (not relative to the central atom)
       Returns the relative vectors from ``atom_index`` to neighbors (no self-term),
        with cutoff weights.
        """
        return self._local_geometry_from_tensor( points, atom_index, include_central_site=False)

    def local_geometry_with_central_at_origin(self, points: torch.Tensor, atom_index: int):
        """
        points: absolute cartesian coordinates of the points (not relative to the central atom)
        Returns the relative vectors including the central atom (whose relative position is ~(0,0,0)) 
        plus weights. 
        """
        return self._local_geometry_from_tensor(points, atom_index, include_central_site=True)

    def _local_geometry_from_tensor(
        self,
        points: torch.Tensor,
        atom_index: int,
        include_central_site: Optional[bool] = None,
    ):
        """
        ``points`` absolute Cartesian coordinates, shape ``(N, 3)``.  The cutoff functions form relative vectors
        ``points - points[atom_index]`` internally; DO NOT pass precomputed interatomic
        vectors.

        Returns ``(rel_vectors, weights)`` for the selected local environment.

        include_central_site
            If True, the central site is included (its relative vector is ~0). If False, only
            neighbors within the shell (no self row).
        """
        ct = self.cutoff_type.lower()
        if ct not in ("radial_cutoff", "smooth_radial_cutoff"):
            raise ValueError(
                f"Descriptor computation called from tensor objects with (use_structure=False, atom_index set), requires "
                f"radial_cutoff or smooth_radial_cutoff; got {self.cutoff_type}. "
                f"Use use_structure=True for other cutoff types (voronoi/crystalnn)"
            )
        if ct == "smooth_radial_cutoff" and getattr(self, "smoothing_fn", "cosine") != "cosine":
            raise ValueError(
                "Smooth cutoff only supports smoothing_fn='cosine' "
                f" got {self.smoothing_fn!r}. Use use_structure=True for gaussian."
            )
        return self.cutoff(
            points,
            atom_index,
            return_tensor_local=True,
            include_central_site=include_central_site,
        )

    def spherical_expansion_per_atom(self, frame_coords, atom_index, dtype = torch.float64):
        """
        Calculate spherical expansion coefficients given frame coords (absolute cartesian coordinates of the points)
        and the index of the central atom.
        frame_coords: absolute cartesian coordinates of the points (not relative to the central atom)
        if atom_index is None, use the first atom.
        """
        atom_coords, weights = self.get_local_geometry(frame_coords, atom_index)
        return spherical_expansion_coeffs(self.lmax, atom_coords, factor=weights, dtype=dtype   )

    def spherical_expansion(self, structures:list, dtype = torch.float64):
        # Can be vectorized better 
        nframes = len(structures)
        spex = [self.spherical_expansion_per_atom(structures[i],j, dtype=dtype) for i in range(nframes) for j in range(len(structures[i]))]
        return torch.stack(spex, dim=0).to(self.device, dtype=self.dtype)
    
    def _compute_invariants_from_spex_acdc(
        self,
        spex: torch.Tensor,
        parity: int,
        return_combinations: bool,
        coords_for_zeroth: Optional[torch.Tensor] = None,
    ):
        """Compute equivariant correlation on precomputed single-frame ``spex`` (..., irreps)."""
        radial_spex = spex.ndim >= 3
        if isinstance(self.correlation_order, (list, tuple)):
            orders = [int(x) for x in self.correlation_order]
            out: dict = {}
            if 0 in orders:
                if parity in (0, 1):
                    if coords_for_zeroth is None:
                        raise ValueError(
                            "correlation_order includes 0 but no coordinates were provided for zeroth_order"
                        )
                    out[0] = {1: zeroth_order(coords_for_zeroth)}
                else:
                    out[0] = {}
                orders = [x for x in orders if x != 0]
                if len(orders) == 0:
                    return squeeze_irrep_dim_nested(out, radial_spex)
            corr_out = equivariant_correlation(
                spex,
                L=0,
                correlation_order=orders,
                lcut=self.lmax,
                parity=parity,
                cg=self.cg,
                return_combinations=return_combinations,
            )
            out.update(corr_out)
            return squeeze_irrep_dim_nested(out, radial_spex)
        raw = equivariant_correlation(
            spex,
            L=0,
            correlation_order=self.correlation_order,
            lcut=self.lmax,
            parity=parity,
            cg=self.cg,
            return_combinations=return_combinations,
        )
        if isinstance(raw, dict):
            return squeeze_irrep_dim_nested(raw, radial_spex)
        return raw

    def compute_invariants_from_local_frame(
        self,
        relative_vec: torch.Tensor,
        weights: torch.Tensor,
        parity: int = 0,
        return_combinations: bool = False,
    ):
        """
        from a single local frame already reduced to relative vectors ``relative_vec`` and per-row
        ``weights`` (e.g. smooth cutoff weights), equivalent to
        ``get_descriptor(..., use_structure=False, coordinates="relative", local_weights=weights)``.
        """
        relative_vec = torch.as_tensor(relative_vec, dtype=self.dtype, device=self.device)
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device).reshape(-1)
        return self.get_descriptor(
            relative_vec,
            parity=parity,
            use_structure=False,
            atom_index=None,
            return_combinations=return_combinations,
            coordinates="relative",
            local_weights=weights,
        )

    def compute_invariants_from_structure(
        self,
        points: torch.Tensor,
        atom_index: Optional[int] = None,
        use_structure: bool = True,
        parity: int = 0,
        return_combinations: bool = False,
        include_central_site: bool = True,
    ):
        """
        **Reference descriptor** from full Cartesian coordinates: either a molecule / point cloud
        with optional ``atom_index`` for a local environment (``use_structure`` True: initializes pymatgen object;
        False: uses tensor of points to compute the descriptor), or global invariants on all points when ``atom_index`` is None.
        """
        return self.get_descriptor(
            points,
            parity=parity,
            use_structure=use_structure,
            atom_index=atom_index,
            return_combinations=return_combinations,
            include_central_site=include_central_site,
        )

   

    def get_descriptor(
        self,
        points,
        parity=0,
        use_structure=True,
        return_combinations=False,
        atom_index=None,
        include_central_site: bool = True,
        coordinates: Literal["absolute", "relative"] = "absolute",
        local_weights: Optional[torch.Tensor] = None,
    ):
        """
        Get the descriptor for given coordinates (which may be either absolute or relative to a central atom)
        Parameters:
            points:  coordinates (npoints, 3) (can be absolute or relative to a central atom)
            parity: Target parity (0=both, 1=even, -1=odd)
            use_structure: If True, build a pymatgen ``Molecule`` and use ``get_local_geometry`` / cutoff neighbors.
                If False, stay tensor-only: with ``atom_index`` set, neighbors are chosen by ``rcut`` (and smearing for smooth radial), same idea as ``get_local_geometry`` but via ``_local_geometry_from_tensor`` (no pymatgen).
            return_combinations: Whether to return angular momentum combinations along with values
            atom_index: Center atom for a local descriptor; used when ``use_structure`` is True (pymatgen path) or False (tensor path; requires radial or smooth radial cutoff).
            coordinates: ``"absolute"`` (default): ``points`` are Cartesian positions; with ``use_structure=False`` and ``atom_index`` set, the cutoff builds relative vectors from the center. 
                         ``"relative"``: ``points`` are **already** relative vectors in the local frame (same as compute_invariants_from_local_frame); ``atom_index`` must be ``None`` and ``use_structure`` must be ``False``.
                                          Optional ``local_weights`` for neighbors may be passed (same shape as ``points``) (defaults to ones).
            local_weights: Per-row weights for the spherical expansion when ``coordinates="relative"`` (e.g. smooth cutoff weights). If omitted, uses a vector of ones (hard cutoff / uniform weighting).
        Returns:
            torch.Tensor or dict: The descriptor for the given atom(s).
        """
        if coordinates not in ["absolute", "relative"]:
            raise ValueError(
                f"coordinates must be 'absolute' or 'relative', got {coordinates}"
            )
        if coordinates == "relative":
            # atom index must be None, 
            if use_structure:
                raise ValueError(
                    "coordinates='relative' requires use_structure=False "
                    "(points are already in the local frame, not a full structure)."
                )
            if atom_index is not None or use_structure:
                raise ValueError(
                    "coordinates='relative' is incompatible with atom_index OR use_structure"
                    "pass only the local-frame tensor as points."
                )

        if isinstance(self.correlation_order, int) and self.correlation_order == 0:
            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, dtype=self.dtype, device=self.device)
            else:
                points = points.to(device=self.device, dtype=self.dtype)
            return zeroth_order(points)

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
        else:
            points = points.to(device=self.device, dtype=self.dtype)

        from pymatgen.core import Molecule
        # if self.center_origin:
        #     points = points - points.mean(dim=0, keepdim=True)
        #     spex = spherical_expansion_coeffs(self.lmax, points).unsqueeze(0)

        if use_structure:#instantiate a pymatgen structure
            structure = Molecule(["C"] * len(points), points.detach().cpu().numpy())
            if atom_index is not None:
                spex = self.spherical_expansion_per_atom(structure, atom_index).unsqueeze(0)
            else:
                
                spex = self.spherical_expansion([structure])

        elif coordinates == "relative":
            if local_weights is None:
                warnings.warn("No local weights provided, using ones")
                w = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
            else:
                w = torch.as_tensor(
                    local_weights, dtype=points.dtype, device=points.device
                ).reshape(-1)
                if w.shape[0] != points.shape[0]:
                    raise ValueError(
                        f"local_weights length {w.shape[0]} must match points rows {points.shape[0]}"
                    )
            spex = spherical_expansion_coeffs(
                self.lmax, points, factor=w, dtype=points.dtype).unsqueeze(0)

        
        elif atom_index is not None: # if proving a tensor of absolute coordinates    "
            atom_coords, weights = self._local_geometry_from_tensor(
                points, int(atom_index), include_central_site=include_central_site
            )
            spex = spherical_expansion_coeffs(
                self.lmax, atom_coords, factor=weights, dtype=points.dtype
            ).unsqueeze(0)

        else: 
            raise NotImplementedError("This combination of coordinates and atom_index is not supported")
            warnings.warn('coordinates = "absolute" and atom_index is None. This will assume the given points are relative to some origin. Is this what you want?')
            spex = spherical_expansion_coeffs(self.lmax, points).unsqueeze(0)

        return self._compute_invariants_from_spex_acdc(
            spex, parity, return_combinations, coords_for_zeroth=points
        )

    # def get_descriptor_stacked(self,points, parity = 0):
    #     spex = spherical_expansion_coeffs(self.lmax, points).unsqueeze(0) 
    #     desc = [spex]
    #     if not isinstance(self.correlation_order, int):
    #         raise NotImplementedError(
    #             "get_descriptor_stacked currently supports only integer correlation_order. "
    #             "Use get_descriptor (which supports lists via equivariant_correlation)."
    #         )
    #     for nu in range(1, self.correlation_order + 1):
    #         desc.append(self.descriptor_fn[nu]())

    def get_descriptor_batch(
        self,
        coords,
        parity=0,
        use_structure=True,
        atom_index=None,
        return_combinations=False,
        include_central_site: bool = True,
        coordinates: Literal["absolute", "relative"] = "absolute",
        local_weights=None,
    ):
        desc = []
        if isinstance(coords, torch.Tensor):
            if coords.dim() == 2:
                environments = [coords]
            elif coords.dim() == 3:
                environments = [coords[i] for i in range(coords.shape[0])]
            else:
                raise ValueError(
                    "get_descriptor_batch: expected coords with shape (N, 3) or (B, N, 3); "
                    f"got {tuple(coords.shape)}"
                )
        else:
            environments = coords
        n_env = len(environments)

        def _weights_for_index(i: int):
            if local_weights is None:
                return None
            if isinstance(local_weights, torch.Tensor):
                if local_weights.dim() == 2 and local_weights.shape[0] == n_env:
                    return local_weights[i]
                if local_weights.dim() == 1 and n_env == 1:
                    return local_weights
            if isinstance(local_weights, (list, tuple)) and len(local_weights) == n_env:
                return local_weights[i]
            raise ValueError(
                "get_descriptor_batch: local_weights must be None, a (B, N) tensor aligned "
                "with batched coords, or a list of length B of per-environment weight vectors."
            )

        for i, env in enumerate(environments):
            desc.append(
                self.get_descriptor(
                    env,
                    parity=parity,
                    use_structure=use_structure,
                    atom_index=atom_index,
                    return_combinations=return_combinations,
                    include_central_site=include_central_site,
                    coordinates=coordinates,
                    local_weights=_weights_for_index(i),
                )
            )

        return desc

    def compute_optimization_descriptor(
        self,
        points: torch.Tensor,
        parity: int = 0,
        use_structure: bool = False,
        atom_index: Optional[int] = None,
        return_combinations: bool = False,
        local_weights: Optional[torch.Tensor] = None,
    ):
        """
        Compute descriptor during optimization
        # is passing relative coordinates, but not passing the weights.
        """
        if atom_index is not None:
            raise NotImplementedError("Computing descriptor around a specific atom is not implemented yet")
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
        else:
            points = points.to(device=self.device, dtype=self.dtype)

        return self.get_descriptor(
            points,
            parity=parity,
            use_structure=use_structure,
            atom_index=atom_index,
            return_combinations=return_combinations,
            coordinates='relative',
            local_weights=local_weights,
        )
class CG:
    def __init__(self, lmax: int, device: str = None, dtype= torch.float64):
        self.lmax = lmax
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        ls = range(self.lmax + 1)
        ps = [-1, 1]  # -1 for pseudo, 1 for polar
        irrep_list = [[1, (l, p)] for l, p in itertools.product(ls, ps)]
        self.tp = o3.FullTensorProduct(o3.Irreps(irrep_list), o3.Irreps(irrep_list))
        self.cg_paths = defaultdict(list)

        irreps_in1 = self.tp.irreps_in1
        irreps_in2 = self.tp.irreps_in2
        irreps_out = self.tp.irreps_out#.simplify()
        for instr in self.tp.instructions:
            i1, i2, i_out = instr.i_in1, instr.i_in2, instr.i_out
            ir1 = irreps_in1[i1].ir
            ir2 = irreps_in2[i2].ir
            ir_out = irreps_out[i_out].ir
            
            # l1 = irreps_in1[instr.i_in1].ir.l
            # l2 = irreps_in2[instr.i_in2].ir.l
            # L  = irreps_out[instr.i_out].ir.l
            # print(f"{l1:>3} {l2:>3} {L:>3}   ({instr.i_in1}, {instr.i_in2}, {instr.i_out})")

            self.cg_paths[(ir_out.l, ir_out.p)].append((ir1.l, ir2.l, ir1.p, ir2.p, i1, i2)) # accumulate allowable paths for each (L,p) pair
            # instr.i_in1: index of the irrep in irreps_in1
            # instr.i_in2: index of the irrep in irreps_in2
            # instr.i_out: index of the irrep in irreps_out
        # print(f"CG paths: {self.cg_paths}"


         # We only need to compute the change of basis once for each pair (l1, l2)
        self.cg = {}
        for l1 in range(self.lmax + 1):
            for l2 in range(self.lmax+1):
                self.cg[(l1, l2)] = {}
                        # self.tps[(l1,p1, l2, p2)] = e3nn.o3.ReducedTensorProducts('ij', i=f'{l1}'+SPHparitylabel(l1, pseudo=PSEUDO1), j=f'{l2}'+SPHparitylabel(l2, pseudo=PSEUDO2))
                tp = e3nn.o3.ReducedTensorProducts( 'ij', i=f'{l1}'+SPHparitylabel(l1, pseudo=False), j = f'{l2}'+SPHparitylabel(l2, pseudo=False))
                tp_slices = tp.irreps_out.slices()
                change_of_basis = tp.change_of_basis.to(self.device)
                for i_out, ir in enumerate(tp.irreps_out):
                    self.cg[(l1, l2)][ir.ir.l] =  change_of_basis[tp_slices[i_out], :, :].to(self.dtype)
                

def invert_descriptor(initial_points = None, 
                      npoints = None,  
                     device='cpu', DTYPE=torch.float64):
    # instantiate random points if not provided
    if initial_points is None:
        initial_points = torch.randn(npoints, 3)
    elif isinstance(initial_points, torch.Tensor):
        initial_points = initial_points.to(device).detach().to(dtype = DTYPE)
    else:
        initial_points = torch.tensor(initial_points, dtype=DTYPE, device=device)
    points = initial_points.clone().detach().to(device)
    points.requires_grad = True
    points = torch.nn.Parameter(points) 

    def loss_fn(pred_descriptor, true_descriptor, mask_index=None, mse=False):

        if mask_index is not None:
            mask = torch.ones_like(true_descriptor, dtype=torch.bool)
            mask[mask_index] = False
            masked_true = true_descriptor * mask
            masked_pred = pred_descriptor * mask
            if mse:
                return torch.mean((masked_true - masked_pred) ** 2)
                # MAE loss
                # return torch.mean(torch.abs(masked_true - masked_pred))
                # return torch.sum(torch.abs(masked_true - masked_pred)) / torch.sum(mask)
            return torch.mean(torch.abs(masked_true - masked_pred))
        else:
            if mse:
                return torch.mean((true_descriptor - pred_descriptor) ** 2)
            return torch.mean(torch.abs(true_descriptor - pred_descriptor))
    return points, loss_fn
    

def collapse_points(points, angle_tolerance=np.pi/6, radial_tolerance=None, origin_tol=1e-6):
    norms = torch.norm(points, dim=-1)
    origin_mask = norms < origin_tol # filter out points close to the origin
    origin_points = points[origin_mask]
    non_origin_points = points[~origin_mask]
    non_origin_norms = norms[~origin_mask]

    unit_vectors = non_origin_points / non_origin_norms[:, None]
    clusters = []
    keep = torch.ones(non_origin_points.shape[0], dtype=torch.bool, device=points.device)
    for i in range(len(non_origin_points)):
        if not keep[i]:
            continue

        ref_dir = unit_vectors[i]
        same_dir = torch.acos((unit_vectors @ ref_dir).clamp(-1, 1)) < angle_tolerance
        if radial_tolerance is not None: # BUT we dont care about radial separation - we only care about angles
            close_radius = torch.abs(non_origin_norms - non_origin_norms[i]) < radial_tolerance
            mask = same_dir & close_radius
        else:
            mask = same_dir
        cluster = non_origin_points[mask]
        keep[mask] = False  # Mark these points as used
        clusters.append(cluster.sum(dim=0))  # SUM the points in the cluster
    if origin_points.numel() > 0:
        clusters.append(origin_points.sum(dim=0))
    if len(clusters) == 0:
        return torch.zeros(0, 3, dtype=points.dtype, device=points.device)  # Return empty tensor if no clusters found
    return torch.stack(clusters)


def _pad_points(
    existing: torch.Tensor,
    n_add: int,
    min_dist: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    add ``n_add`` points from the centroid of ``existing`` on an
    expanding spiral in the xy-plane. Radii are
    ``max_radius + min_dist * (k+1)`` so new samples are not stacked at the origin
    and are spread in angle.
    """
    if n_add <= 0:
        return existing
    else:
        ex = existing.detach().to(dtype=dtype, device=device)
        center = ex.mean(dim=0)
        max_r = torch.linalg.norm(ex - center, dim=-1).max()
    rows = []
    mind = torch.as_tensor(min_dist, dtype=dtype, device=device)
    for k in range(n_add):
        ang = 2 * math.pi * k / max(n_add, 1)
        r = max_r + mind * (k + 1)
        dx = r * math.cos(ang)
        dy = r * math.sin(ang)
        rows.append(center + torch.tensor([dx, dy, 0.0], dtype=dtype, device=device))
    return torch.stack(rows, dim=0)


def detect_origin_in_structure(structure, origin_tol=1e-6):
    """
    Detect if a structure contains origin points.
    Args:
        structure: Tensor of shape (n, 3) - the original structure to analyze
        origin_tol: Tolerance for detecting origin points
        
    Returns:
        tuple: (actual_target_points, has_origin)
            - actual_target_points: Number of non-origin points
            - has_origin: Boolean indicating if origin was detected
    """
    if structure is None:
        raise ValueError("structure cannot be None - we need the original structure to detect origin points")
    
    origin_mask = torch.norm(structure, dim=-1) < origin_tol
    has_origin = origin_mask.any().item()
    
    if has_origin:
        non_origin_mask = torch.norm(structure, dim=-1) >= origin_tol
        actual_target_points = non_origin_mask.sum().item()
    else:
        actual_target_points = len(structure)
    
    return actual_target_points, has_origin



def optimize_to_N_points(target_descriptor, 
                         calculator, 
                         reference_structure = None , 
                         actual_target_points = None,
                         n_points = None,  
                         
                         device='cpu', 
                         DTYPE=torch.float64, 
                         angle_tolerance=np.pi/6, 
                         radial_tolerance=None, 
                         origin_tol=1e-6,
                         max_outer_iter=100, 
                         max_inner_iter=100, 
                         lr=1e-3, 
                         collapse_every = 10, 
                         patience=10, 
                         min_loss=1e-8, 
                         log_every=10, 
                         min_distance=0.5,
                         verbose=True,
                         parity = 0, 
                         
                         use_structure=False,
                         atom_index=None,
                         
                         mse = True,
                         seed = 134247,
                         initial_points=None,
                         jitter_scale=1e0
                         ):
    """
    target_descriptor: torch.Tensor of shape (D,)
    calculator: DescriptorCalculator
    reference_structure: torch.Tensor of shape (N, 3)
    actual_target_points: int (or None) - number of points to recover
    n_points: int (or None) - number of points to initialize
    center_origin: bool - whether to center the origin before computing the descriptor
    device: str - device to use
    DTYPE: torch.dtype - dtype to use
    angle_tolerance: float - angle tolerance for collapse
    radial_tolerance: float - radial tolerance for collapse
    origin_tol: float - origin tolerance for collapse
    mse: bool - whether to use MSE loss (if False, uses MAE loss)
    seed: int - seed for random number generator

    initial_points: optional tensor/array of shape (N_init, 3) used as optimization start.
        If provided, this overrides random initialization and can have N_init different from
        actual_target_points (collapse/pad path handles mismatch).
    max_outer_iter: int - rounds after end-of-round collapse + re-wrap with ``invert_descriptor``.
    max_inner_iter: int - gradient steps per outer round (main optimization).
    lr: float - learning rate
    collapse_every: int - collapse every n inner iterations
    patience: int - patience for early stopping
    min_loss: float - minimum loss for early stopping
    """

    torch.manual_seed(seed)
    if reference_structure is not None:
        actual_target_points, has_origin = detect_origin_in_structure(reference_structure, origin_tol)
        if verbose:
            print(f"Actual target points: {actual_target_points}, Has origin: {has_origin}")
        if has_origin and verbose:
            print(f"Detected origin point in reference structure. Recovering {actual_target_points} points instead of {len(reference_structure)}.")
    else:
        actual_target_points = n_points if actual_target_points is None else actual_target_points
        has_origin = False

    if not isinstance(target_descriptor, torch.Tensor):
        target_descriptor = torch.tensor(target_descriptor, dtype=calculator.dtype, device=device)
    elif target_descriptor.dtype != calculator.dtype:
        target_descriptor = target_descriptor.to(dtype=calculator.dtype)
    
    DTYPE = calculator.dtype
    if initial_points is None:
        n_init = n_points if n_points is not None else actual_target_points
        init_pts = torch.randn(n_init, 3, dtype=DTYPE, device=device)
    else:
        if not isinstance(initial_points, torch.Tensor):
            init_pts = torch.tensor(initial_points, dtype=DTYPE, device=device)
        else:
            init_pts = initial_points.to(dtype=DTYPE, device=device)
        if init_pts.ndim != 2 or init_pts.shape[-1] != 3:
            raise ValueError(
                f"initial_points must have shape (N, 3), got {tuple(init_pts.shape)}"
            )
    points_param, loss_fn = invert_descriptor(init_pts, actual_target_points, device=device, DTYPE=DTYPE)

    best_points = None
    best_loss = torch.inf
    early_stop_count = 0
    
    from invertatomrep.utils import extract_descriptor

    for outer in range(max_outer_iter):
        optimizer = torch.optim.Adam([points_param], lr=lr)
        # NOTE: we can add a different LR scheduling or make this parallel?
        improved_this_outer = False
        for step in range(max_inner_iter):
            optimizer.zero_grad()
            if not use_structure and atom_index is None:
                center_origin = True

            pred_descriptor = calculator.compute_optimization_descriptor(
                    points_param,
                    parity=parity,
                    use_structure=use_structure,
                    atom_index=atom_index,
                )
            #notice we're not passing any local weights here 
          
            pred_descriptor = extract_descriptor(pred_descriptor, stack_parities=True)
            # print(f"Pred descriptor: {pred_descriptor.shape}, target descriptor: {target_descriptor.shape}")

            loss = loss_fn(pred_descriptor, target_descriptor, mse=mse)
           
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_points = points_param.detach().clone()
                improved_this_outer = True

            if not use_structure:
                with torch.no_grad():
                    if atom_index is not None:
                        points_param.data = points_param.data - points_param.data[atom_index].reshape(1, 3)

            loss.backward()
            optimizer.step()

            if verbose and (step % log_every== 0 or step == max_inner_iter - 1):
                print(f"  Outer {outer+1}/{max_outer_iter}, Step {step+1}/{max_inner_iter}, Loss: {loss.item():.2e}")

            if collapse_every and (step + 1) % collapse_every == 0:
                with torch.no_grad():
                    collapsed = collapse_points(
                        points_param,
                        angle_tolerance=angle_tolerance,
                        radial_tolerance=radial_tolerance,
                        origin_tol=origin_tol,
                    )
                    if len(collapsed) < actual_target_points:
                        n_missing = actual_target_points - len(collapsed)
                        extra = _pad_points(
                            collapsed,
                            n_missing,
                            min_distance,
                            DTYPE,
                            device,
                        )
                        collapsed = torch.cat([collapsed, extra], dim=0)
                    elif len(collapsed) > actual_target_points:
                        indices = torch.randperm(len(collapsed), device=device)[
                            :actual_target_points
                        ]
                        collapsed = collapsed[indices]
                    # after stack/collapse, start a new phase with fresh Parameter and optimizer
                    points_param, loss_fn = invert_descriptor(
                        collapsed,
                        actual_target_points,
                        device=device,
                        DTYPE=DTYPE,
                    )
                    optimizer = torch.optim.Adam([points_param], lr=lr)

        if improved_this_outer:
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if verbose and outer % log_every == 0:
            print(f"Outer iteration {outer+1}/{max_outer_iter} completed. Best loss so far: {best_loss:.2e}")
        
        if best_loss < min_loss:
            if verbose:
                print(f"Stopping early: loss {best_loss:.2e} below minimum threshold {min_loss:.2e}")
            break
        
        if early_stop_count >= patience:
            if verbose:
                print(f"Stopping early: no improvement for {patience} iterations")
            break

        if collapse_every and (max_inner_iter % collapse_every != 0):
            with torch.no_grad():
                collapsed = collapse_points(
                    points_param.data,
                    angle_tolerance=angle_tolerance,
                    radial_tolerance=radial_tolerance,
                    origin_tol=origin_tol,
                )
                if len(collapsed) < actual_target_points:
                    n_missing = actual_target_points - len(collapsed)
                    extra = _pad_points(
                        collapsed,
                        n_missing,
                        min_distance,
                        DTYPE,
                        device,
                    )
                    collapsed = torch.cat([collapsed, extra], dim=0)
                elif len(collapsed) > actual_target_points:
                    indices = torch.randperm(len(collapsed), device=device)[
                        :actual_target_points
                    ]
                    collapsed = collapsed[indices]
                points_param, loss_fn = invert_descriptor(
                    collapsed,
                    actual_target_points,
                    device=device,
                    DTYPE=DTYPE,
                )
        # if loss is really low, we should make jitter_scale smaller
        jitter_scale = jitter_scale * 1e-1 if loss < 1e-10 else jitter_scale
        jitter = torch.randn_like(points_param) * jitter_scale if not improved_this_outer else torch.zeros_like(points_param)
        if jitter.norm()!=0:
            early_stop_count -=1
        print(f"Jitter:", jitter)
        points_param, loss_fn = invert_descriptor(points_param+jitter, actual_target_points, device=device, DTYPE=DTYPE)

    if best_points is not None and not use_structure:
        if atom_index is not None:
            best_points = best_points - best_points[int(atom_index)].reshape(1, 3)
        else:
            best_points = best_points #- best_points.mean(dim=0, keepdim=True)
    return best_points, best_loss


def stack_points_torch(
    points,
    angle_tolerance=np.pi/6,
    min_norm_fraction=0.5,
    origin_tol=1e-6,
    radial_tolerance=None,
):
    """
    Merge points on similar radial directions, then drop
    low-norm clusters. .
    """
    stacked = collapse_points(
        points,
        angle_tolerance=angle_tolerance,
        radial_tolerance=radial_tolerance,
        origin_tol=origin_tol,
    )
    if stacked.numel() == 0:
        warnings.warn("stack_points_torch returned 0 points")
        return stacked
    norms = torch.norm(stacked, dim=1)
    max_norm = norms.max()
    if max_norm <= origin_tol:
        return stacked
    keep = norms >= min_norm_fraction * max_norm
    return stacked[keep]


def optimize_invert_stack_invert(
    target_descriptor,
    calculator,
    *,
    n_points=12,
    lr=1e-2,
    refine_lr=1e-3,
    angle_tolerance=np.pi/6,
    min_norm_fraction=0.5,
    max_outer_iter=10,
    max_inner_iter=1000,
    refine_max_outer_iter=None,
    refine_max_inner_iter=None,
    patience=3,
    min_loss=1e-8,
    log_every=10,
    device="cpu",
    verbose=True,
    parity=0,
    use_structure=False,
    atom_index=None,
    mse=True,
    seed=134247,
    initial_points=None,
    jitter_scale=1e0,
    return_details=False,
    **optimize_kwargs,
):
    """
    1. Fit with ``n_points`` degrees of freedom (no mid-run collapse).
    2. ``stack_points_torch`` to merge angular clusters; N is whatever remains.
    3. Refit from stacked geometry at ``refine_lr``.

    No ``actual_target_points`` — final neighbor count comes from stacking.
    Extra kwargs are forwarded to both ``optimize_to_N_points`` calls.
    ``collapse_every`` is always forced to 0 (stacking is separate).
    """
    if refine_max_outer_iter is None:
        refine_max_outer_iter = max_outer_iter
    if refine_max_inner_iter is None:
        refine_max_inner_iter = max_inner_iter

    opt_kw = dict(
        device=device,
        verbose=verbose,
        parity=parity,
        use_structure=use_structure,
        atom_index=atom_index,
        mse=mse,
        patience=patience,
        min_loss=min_loss,
        log_every=log_every,
        jitter_scale=jitter_scale,
        angle_tolerance=angle_tolerance,
    )
    opt_kw.update(optimize_kwargs)
    if opt_kw.get("collapse_every", 0):
        warnings.warn(
            "optimize_invert_stack_invert: collapse_every must be 0; ignoring "
            f"passed value {opt_kw['collapse_every']}",
            stacklevel=2,
        )
    opt_kw["collapse_every"] = 0

    coarse_points, coarse_loss = optimize_to_N_points(
        target_descriptor=target_descriptor,
        calculator=calculator,
        n_points=n_points,
        actual_target_points=n_points,
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
        lr=lr,
        seed=seed,
        initial_points=initial_points,
        **opt_kw,
    )

    stacked_points = stack_points_torch(
        coarse_points,
        angle_tolerance=angle_tolerance,
        min_norm_fraction=min_norm_fraction,
        origin_tol=opt_kw.get("origin_tol", 1e-6),
        radial_tolerance=opt_kw.get("radial_tolerance"),
    )
    if len(stacked_points) == 0:
        if verbose:
            print("stack_points_torch returned 0 points; returning coarse fit")
        if return_details:
            return coarse_points, coarse_loss, {
                "coarse_points": coarse_points,
                "coarse_loss": coarse_loss,
                "stacked_points": stacked_points,
            }
        return coarse_points, coarse_loss

    if verbose:
        print(
            f"invert_stack_invert: coarse N={len(coarse_points)} "
            f"(loss {coarse_loss:.2e}) -> stacked N={len(stacked_points)}"
        )

    refined_points, refined_loss = optimize_to_N_points(
        target_descriptor=target_descriptor,
        calculator=calculator,
        n_points=len(stacked_points),
        actual_target_points=len(stacked_points),
        max_outer_iter=refine_max_outer_iter,
        max_inner_iter=refine_max_inner_iter,
        lr=refine_lr,
        seed=seed,
        initial_points=stacked_points,
        **opt_kw,
    )

    if return_details:
        return refined_points, refined_loss, {
            "coarse_points": coarse_points,
            "coarse_loss": coarse_loss,
            "stacked_points": stacked_points,
            "refined_points": refined_points,
            "refined_loss": refined_loss,
        }
    return refined_points, refined_loss


def normalize_quaternion(q):
    return q / q.norm(p=2, dim=-1, keepdim=True)


def rotate_points_quaternion(q, points):
    """
    Rotate 3D points using a quaternion.

    Parameters:
    q : torch.Tensor of shape (4,)
        Quaternion [w, x, y, z] (unit norm).
    points : torch.Tensor of shape (N, 3)
        3D points to rotate.

    Returns:
    torch.Tensor of shape (N, 3)
        Rotated points.
    """
    q = normalize_quaternion(q)
    w, x, y, z = q

    R = torch.tensor([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ], dtype=points.dtype, device=points.device)

    return torch.matmul(points, R.T)



def procrustes_align(X, Y):
    """
    Align Y to X using orthogonal Procrustes (both (N, 3) tensors).
    Returns aligned Y and rotation matrix.
    """
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Kabsch: H = X^T Y (both (N,3)) so each is centered; minimizes ||X - Y R^T||_F with R orthogonal
    H = X_centered.T @ Y_centered
    U, _, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    Y_aligned = (Y_centered @ R.T) + X_mean
    return Y_aligned, R


def align_and_compute_rmsd(
    true_coords,
    recovered_coords,
    reduction="none",
    mae=True,
    *,
    permute_identical_atoms=True,
    permute_max_atoms=15,
    dref_order=2,
    return_aligned=False,
):
    from invertatomrep.distancematrix import best_permutation_distance_matrices

    def _pairwise_distance_matrix(pos: torch.Tensor):
        """Symmetric matrix of Euclidean distances, same order as coords rows."""
        return torch.cdist(pos, pos).detach().cpu().numpy()

    def _pair_rmsd(true_points, recovered_points):
        if true_points.shape != recovered_points.shape:
            raise ValueError(
                f"Shape mismatch: true {true_points.shape} vs recovered {recovered_points.shape}"
            )
        if true_points.shape[-1] != 3:
            raise ValueError("Coordinates must have shape (..., 3).")
        recovered_points = recovered_points.to(
            device=true_points.device, dtype=true_points.dtype
        )

        n = true_points.shape[0]
        if permute_identical_atoms and n <= permute_max_atoms and n > 0:
            D = _pairwise_distance_matrix(true_points)
            Dp = _pairwise_distance_matrix(recovered_points)
            perm, dref = best_permutation_distance_matrices(D, Dp, order=dref_order)
            perm_t = torch.as_tensor(perm, device=recovered_points.device, dtype=torch.long)
            recovered_points = recovered_points[perm_t]

        aligned_recovered, _ = procrustes_align(true_points, recovered_points)
        if not mae:
            metric = torch.sqrt(torch.mean((aligned_recovered - true_points) ** 2))
        else:
            metric = torch.mean(torch.abs(aligned_recovered - true_points))
        if return_aligned:
            return metric, aligned_recovered
        return metric

    if isinstance(true_coords, torch.Tensor) and isinstance(recovered_coords, torch.Tensor):
        return _pair_rmsd(true_coords, recovered_coords)

    if not (hasattr(true_coords, "__len__") and hasattr(recovered_coords, "__len__")):
        raise TypeError("true_coords and recovered_coords must be tensors or sequences of tensors.")

    true_list = list(true_coords)
    recovered_list = list(recovered_coords)
    if len(true_list) != len(recovered_list):
        raise ValueError(
            f"Batch length mismatch: {len(true_list)} true vs {len(recovered_list)} recovered."
        )
    pairs = [_pair_rmsd(t, r) for t, r in zip(true_list, recovered_list)]
    if return_aligned:
        rmsds = torch.stack([p[0] for p in pairs])
        aligned_list = [p[1] for p in pairs]
        if reduction == "mean":
            return rmsds.mean(), aligned_list
        if reduction != "none":
            raise ValueError("reduction must be 'none' or 'mean'.")
        return rmsds, aligned_list
    rmsds = torch.stack(pairs)
    if reduction == "mean":
        return rmsds.mean()
    if reduction != "none":
        raise ValueError("reduction must be 'none' or 'mean'.")
    return rmsds

