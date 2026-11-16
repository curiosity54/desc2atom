import torch 
import e3nn.io as io
import e3nn
import e3nn.o3 as o3
import invertatomrep.cutoff
# from ase.data import atomic_numbers, chemical_symbols
import warnings
import numpy as np 
from typing import Literal, Optional, Union, List


from invertatomrep.descriptor import (
    CG,
    enumerate_ls_nu,
    get_lm_slice,
    squeeze_irrep_dim_nested,
    zeroth_order,
)

# Stacked correlation blocks (powerspectrum, bispectrum, equivariant_correlation):
#   values.shape == (n_samples, 2*L+1, n_combinations, *radial_dims)
# where L is the *output* coupled o3_lambda for that block, (2*L+1) is the equivariant m-channel,
# axis 2 indexes the same order as ``combinations[k]`` (allowed (l1,l2,…) chains from enumerate_ls_nu).

# --------RADIAL BASIS-------
# 1. Jacobi polynomials (NOTE, for our case, we will only use alpha = 0 and beta = 2 variants)
def radial_jacobi_z_from_r(r: torch.Tensor, r_cut: float, dtype = torch.float64) -> torch.Tensor:
    """Map ``r \\in [0, r_\\mathrm{cut}]`` to Jacobi argument ``z = 2r/r_\\mathrm{cut} - 1 \\in [-1, 1]``."""
    return (2.0 / float(r_cut)) * r.to(dtype=dtype) - 1.0


def radial_jacobi_basis_squared_norm(
    r_cut: float,
    n: int,
    *,
    alpha: float = 0.0,
    beta: float = 2.0,
) -> float:
    """``\\int_0^{r_\\mathrm{cut}} [R_n(r)]^2\\, r^2\\,dr`` with ``R_n(r)=P_n^{(\\alpha,\\beta)}(2r/r_\\mathrm{cut}-1)``.

    With ``z = 2r/r_\\mathrm{cut}-1``, ``r = r_\\mathrm{cut}(z+1)/2``, ``dr = (r_\\mathrm{cut}/2)\\,dz``,

    .. math::

        \\int_0^{r_\\mathrm{cut}} R_i R_j\\, r^2\\,dr
        = \\frac{r_\\mathrm{cut}^3}{8}\\int_{-1}^{1} P_i(z)\\,P_j(z)\\,(1+z)^\\beta(1-z)^\\alpha\\,dz.

    Write ``h_n^{(\\alpha,\\beta)} = \\int_{-1}^{1} [P_n]^2\\,(1-z)^\\alpha(1+z)^\\beta\\,dz``. For SciPy /
    DLMF Jacobi normalization (``n \\ge 0``, ``\\alpha,\\beta > -1``),

    .. math::

        h_n^{(\\alpha,\\beta)}
        = \\frac{2^{\\alpha+\\beta+1}}{2n+\\alpha+\\beta+1}\\,
          \\frac{\\Gamma(n+\\alpha+1)\\,\\Gamma(n+\\beta+1)}{\\Gamma(n+\\alpha+\\beta+1)\\,n!},
        \\qquad
        \\|R_n\\|^2 = \\frac{r_\\mathrm{cut}^3}{8}\\, h_n^{(\\alpha,\\beta)}.

    For ``(\\alpha,\\beta)=(0,2)``, ``\\|R_n\\|^2 = r_\\mathrm{cut}^3/(2n+3)`` (no ``\\Gamma`` calls).
    """
    from scipy.special import gamma

    if n < 0:
        raise ValueError("n must be >= 0.")
    R = float(r_cut)
    if R <= 0:
        raise ValueError("r_cut must be positive.")
    if alpha == 0.0 and beta == 2.0:
        return R**3 / (2.0 * float(n) + 3.0)
    apb = alpha + beta
    pref = 2.0 ** (apb + 1.0) / (2.0 * float(n) + apb + 1.0)
    rat = (gamma(n + alpha + 1) * gamma(n + beta + 1)) / (gamma(n + apb + 1) * gamma(n + 1))
    hz = float(pref * rat)
    return (R**3 / 8.0) * hz


def jacobi_polynomial_values_at_z(z: torch.Tensor, nmax: int, alpha: float = 0.0, beta: float = 2.0, dtype = torch.float64) -> torch.Tensor:
    """``P_n^{(\\alpha,\\beta)}(z)`` for ``n = 0,\\dots,n_{\\max}-1``; shape ``(..., nmax)``.

    Uses ``scipy.special.eval_jacobi`` (same normalization as the standard orthogonal Jacobi family on
    ``[-1,1]`` with weight ``(1-z)^\\alpha(1+z)^\\beta``).

    For spherical radial weight ``r^2\\,dr`` on ``[0,r_\\mathrm{cut}]``, use ``u=r/r_\\mathrm{cut}`` so
    ``\\int_0^{r_\\mathrm{cut}} f g\\, r^2\\,dr = r_\\mathrm{cut}^3 \\int_0^1 f g\\, u^2\\,du``.
    Polynomials in ``u`` orthogonal for ``u^2`` correspond to ``P_n^{(\\alpha,\\beta)}(2u-1)`` with
    ``u^\\beta(1-u)^\\alpha`` matching ``u^2``: hence ``\\beta=2``, ``\\alpha=0``.
    """
    from scipy.special import eval_jacobi

    if nmax < 1:
        raise ValueError("nmax must be >= 1")
    z = z.to(dtype=dtype)
    flat = z.reshape(-1)
    z_np = flat.detach().cpu().numpy()
    cols = []
    for n in range(nmax):
        vn = eval_jacobi(n, alpha, beta, z_np)
        cols.append(torch.as_tensor(np.asarray(vn), device=z.device, dtype=z.dtype))
    out = torch.stack(cols, dim=-1).reshape(*z.shape, nmax)
    return out.to(dtype=z.dtype)

# Util to verify the Jacobi basis is orthoNORMAL (it is orthogonal by definition, but the overlap of same basis is not 1)
def radial_jacobi_basis_gram_matrix(
    r_cut: float,
    nmax: int,
    *,
    n_quad: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Gram matrix with spherical radial weight on ``[0, r_\\mathrm{cut}]``:

    .. math::

        G_{ij} = \\int_0^{r_\\mathrm{cut}}
        R_i(r)\\, R_j(r)\\, r^2\\, dr,
        \\qquad
        R_k(r) = P_k^{(0,2)}\\!\\bigl(2r/r_\\mathrm{cut}-1\\bigr).

    Uses **Gauss–Jacobi** quadrature on ``z \\in [-1,1]`` with ``(\\alpha,\\beta)=(0,2)``, which
    integrates ``\\int_{-1}^1 f(z)\\,(1+z)^2\\,dz`` exactly when ``f`` is a polynomial of degree
    ``\\le 2 n_{\\mathrm{quad}} - 1``. Here ``f = P_i^{(0,2)} P_j^{(0,2)}`` has degree ``i+j``, so
    ``n_{\\mathrm{quad}} \\ge n_{\\max}`` suffices.

    Change of variables ``r = r_\\mathrm{cut}(z+1)/2`` gives the prefactor ``r_\\mathrm{cut}^3/8``.
    Diagonals match ``radial_jacobi_basis_squared_norm(r_cut, n)`` analytically; off-diagonals vanish.

    Returns
    -------
    Tensor of shape ``(n_{\\max}, n_{\\max})`` — numerically diagonal for classical Jacobi ``P^{(0,2)}``.
    """
    from scipy.special import eval_jacobi, roots_jacobi

    if r_cut <= 0 or nmax < 1:
        raise ValueError("r_cut must be positive and nmax >= 1.")
    n_q = int(n_quad) if n_quad is not None else max(nmax + 4, 24)
    if n_q < nmax:
        raise ValueError("n_quad must be >= nmax for exact polynomial quadrature here.")

    z_np, w_np = roots_jacobi(n_q, 0.0, 2.0)
    Phi = np.stack([eval_jacobi(j, 0.0, 2.0, z_np) for j in range(nmax)], axis=1)
    s = np.sqrt(np.maximum(w_np, 0.0))
    A = Phi * s[:, None]
    scale = (float(r_cut) ** 3) / 8.0
    G_np = scale * (A.T @ A)

    dev = device if device is not None else torch.device("cpu")
    return torch.as_tensor(G_np, dtype=dtype, device=dev)


def assert_radial_jacobi_basis_gram(
    r_cut: float,
    nmax: int,
    *,
    orthonormal: bool = False,
    n_quad: Optional[int] = None,
    atol_offdiag: float = 1e-10,
    atol_identity: float = 1e-10,
    rtol_identity: float = 1e-10,
) -> None:
    """Check Jacobi ``P^{(0,2)}`` radial Gram matrix from ``radial_jacobi_basis_gram_matrix``.

    * ``orthonormal=False``: assert off-diagonal entries are negligible (orthogonal unnormalized basis).
    * ``orthonormal=True``: assert ``D G D \\approx I`` with analytic ``D_{kk}=1/\\|P_k\\|``.

    Raises ``AssertionError`` if a bound is violated (use from tests or debug sessions).
    """
    G_raw = radial_jacobi_basis_gram_matrix(r_cut, nmax, n_quad=n_quad)
    dtype, dev = G_raw.dtype, G_raw.device

    if not orthonormal:
        off = G_raw - torch.diag(torch.diag(G_raw))
        max_off = float(off.abs().max())
        assert max_off <= atol_offdiag, (
            f"Jacobi radial Gram: max|off-diag|={max_off:g} exceeds atol_offdiag={atol_offdiag:g}"
        )
        return

    inv_sqrt = torch.tensor(
        [
            1.0 / np.sqrt(radial_jacobi_basis_squared_norm(float(r_cut), k, alpha=0.0, beta=2.0))
            for k in range(nmax)
        ],
        dtype=dtype,
        device=dev,
    )
    G_on = torch.diag(inv_sqrt) @ G_raw @ torch.diag(inv_sqrt)
    eye = torch.eye(nmax, dtype=dtype, device=dev)
    if not torch.allclose(G_on, eye, atol=atol_identity, rtol=rtol_identity):
        max_err = float((G_on - eye).abs().max())
        raise AssertionError(
            f"Orthonormal Jacobi radial Gram not identity: max abs error {max_err:g} "
            f"(atol={atol_identity:g}, rtol={rtol_identity:g})"
        )


def gto_zeta_sequence(r_mean: torch.Tensor, nmax: int) -> torch.Tensor:
    """``nmax`` inverse variances ``ζ_k`` for isotropic Gaussians ``exp(-ζ_k r^2)`` (same rule as ``spherical_expansion_coeffs``)."""
    r_mean = r_mean.to(dtype=torch.float64).clamp(min=1e-6)
    zeta_min = 0.15 / (r_mean**2)
    zeta_max = 8.0 / (r_mean**2)
    if nmax == 1:
        return zeta_min.reshape(1)
    ratio = (zeta_max / zeta_min) ** (1.0 / float(nmax - 1))
    return zeta_min * (ratio ** torch.arange(nmax, dtype=torch.float64, device=r_mean.device))


def eval_radial_basis_matrix(
    r: torch.Tensor,
    nmax: int,
    radial_basis: Literal["gto", "monomial", "jacobi", "jacobi_orthonormal"],
    *,
    r_mean: Optional[torch.Tensor] = None,
    r_cut: Optional[float] = None,
) -> torch.Tensor:
    """Evaluate raw radial factors at distances ``r``; shape ``(..., nmax)``.

    Monomial columns are ``r^n`` for ``n = 1..nmax``. GTO columns are ``exp(-ζ_k r^2)``.
    For GTO, ``r_mean`` sets ``ζ_k`` progression (defaults to ``r.mean()``).

    **Jacobi** (``radial_basis='jacobi'``): column ``k`` is ``P_k^{(0,2)}(2r/r_\\mathrm{cut}-1)`` — orthogonal
    under ``\\int_0^{r_\\mathrm{cut}} \\cdot\\, r^2\\,dr`` (continuum). Requires ``r_cut > 0``.

    **Jacobi orthonormal** (``radial_basis='jacobi_orthonormal'``): same polynomials divided by
    ``\\|\\cdot\\|_{r^2\\,dr}`` from ``radial_jacobi_basis_squared_norm`` so ``\\int \\tilde R_i \\tilde R_j\\, r^2\\,dr = \\delta_{ij}``.

    There is no separate classical name for ``(0,2)`` like “Legendre” for ``(0,0)``; they are Jacobi
    polynomials (shifted to ``r`` on ``[0,r_\\mathrm{cut}]``). See module discussion: weight ``u^2`` on
    ``u=r/r_\\mathrm{cut}`` pairs with ``u^{\\beta}(1-u)^{\\alpha}`` iff ``\\beta=2``, ``\\alpha=0``.
    """
    r = r.to(dtype=torch.float64)
    if radial_basis == "monomial":
        cols = [torch.pow(r, float(n)) for n in range(1, nmax + 1)]
        return torch.stack(cols, dim=-1)
    if radial_basis == "gto":
        rm = r.mean() if r_mean is None else r_mean
        zetas = gto_zeta_sequence(rm, nmax).to(device=r.device, dtype=r.dtype)
        cols = [torch.exp(-zetas[i] * r * r) for i in range(nmax)]
        return torch.stack(cols, dim=-1)
    if radial_basis in ("jacobi", "jacobi_orthonormal"):
        if r_cut is None or r_cut <= 0:
            raise ValueError(
                "radial_basis='jacobi' and 'jacobi_orthonormal' require a positive r_cut."
            )
        z = radial_jacobi_z_from_r(r, r_cut)
        Phi = jacobi_polynomial_values_at_z(z, nmax, alpha=0.0, beta=2.0)
        if radial_basis == "jacobi_orthonormal":
            inv = torch.tensor(
                [
                    1.0 / np.sqrt(radial_jacobi_basis_squared_norm(float(r_cut), n, alpha=0.0, beta=2.0))
                    for n in range(nmax)
                ],
                dtype=Phi.dtype,
                device=Phi.device,
            ).reshape((1,) * (Phi.ndim - 1) + (nmax,))
            Phi = Phi * inv
        return Phi
    raise ValueError(
        f"radial_basis must be 'gto', 'monomial', 'jacobi', or 'jacobi_orthonormal', got {radial_basis!r}"
    )


def radial_basis_orthogonal_matrix(
    Phi: torch.Tensor,
    weights_1d: torch.Tensor,
) -> torch.Tensor:
    """``Phi`` (N, K), weights for trapezoidal ``\\sum_i w_i f_i g_i``. Returns ``T`` (K, K) with
    ``(Phi @ T)^T diag(weights) (Phi @ T) = I`` (columns orthonormal in that discrete metric).

    Typical choice: ``weights_1d`` from ``trapezoid_weights_radial_measure`` (default ``radial_measure=2``,
    i.e. spherical ``r^2\\,dr`` on ``[0, r_\\mathrm{cut}]``). Raw monomials / Gaussians are **not**
    orthogonal in that inner product.
    """
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be (N, K), got {Phi.shape}")
    N, K = Phi.shape
    if weights_1d.shape != (N,):
        raise ValueError("weights_1d must be (N,)")
    sw = torch.sqrt(weights_1d.clamp(min=0.0))
    A = Phi * sw.unsqueeze(-1)
    G = A.T @ A
    eye = torch.eye(K, dtype=G.dtype, device=G.device)
    L = torch.linalg.cholesky(G + 1e-10 * eye)
    # T = L^{-T} so A @ T has orthonormal columns in Euclidean sense == Phi orthonormal w.r.t. W
    T = torch.linalg.solve_triangular(L, eye, upper=False).mT
    return T


def trapezoid_weights_radial_measure(
    r: torch.Tensor,
    radial_measure: float = 2.0,
) -> torch.Tensor:
    """Composite trapezoid weights for ``\\int q(r)\\, r^{\\texttt{radial_measure}}\\, dr`` on ``r``.

    ``radial_measure`` is the exponent in the radial Jacobian/volume factor (``2`` gives spherical
    ``\\propto r^2\\,dr``, i.e. matches Jacobi ``(0,2)`` / ``L^2([0,r_{\\mathrm{cut}}], r^2\\,dr)``).

    **Used by** ``plot_radial_bases`` and ``spherical_expansion_coeffs`` when ``orthogonalize=True`` for
    non-Jacobi bases: ``w_i`` define ``\\sum_i w_i \\phi_i(r_i)\\psi_j(r_i)`` as a discrete Gram inner
    product on ``[0, r_{\\mathrm{cut}}]``, paired with ``radial_basis_orthogonal_matrix``.
    """
    if r.ndim != 1:
        raise ValueError("r must be 1D")
    dr = torch.diff(r)
    c = torch.zeros_like(r)
    c[0] = 0.5 * dr[0]
    c[-1] = 0.5 * dr[-1]
    if len(r) > 2:
        c[1:-1] = 0.5 * (dr[:-1] + dr[1:])
    rp = torch.abs(r).clamp(min=0.0) ** radial_measure
    return c * rp


def plot_radial_bases(
    r_cut: float,
    nmax: int,
    radial_basis: Literal["gto", "monomial", "jacobi", "jacobi_orthonormal"] = "gto",
    *,
    n_grid: int = 400,
    r_mean: float = 2.0,
    orthogonalize: bool = True,
    radial_measure: float = 2.0,
    save_path: Optional[str] = None,
):
    """Plot radial factors vs ``r`` on ``[0, r_\\mathrm{cut}]``.

    For ``jacobi`` / ``jacobi_orthonormal``, columns are shifted Jacobi ``P_k^{(0,2)}`` on ``r``; they are
    orthogonal under ``\\int_0^{r_\\mathrm{cut}} \\cdot\\, r^{\\texttt{radial_measure}}\\,dr`` when
    ``radial_measure=2`` — **not** “curves don’t cross”. Use ``jacobi_orthonormal`` for ``P_k/\\|P_k\\|``
    (analytic norm via ``radial_jacobi_basis_squared_norm``).

    Discrete ``orthogonalize`` is skipped for Jacobi radial bases (see warning). For ``gto`` / ``monomial``,
    optional dashed curves show discrete Gram orthogonalization on the same grid.
    """
    import matplotlib.pyplot as plt

    r = torch.linspace(0.0, r_cut, n_grid, dtype=torch.float64)
    rm = torch.tensor(r_mean, dtype=torch.float64)
    Phi = eval_radial_basis_matrix(
        r,
        nmax,
        radial_basis,
        r_mean=rm,
        r_cut=float(r_cut) if radial_basis in ("jacobi", "jacobi_orthonormal") else None,
    )
    w = trapezoid_weights_radial_measure(r, radial_measure=radial_measure)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(nmax):
        if radial_basis == "jacobi":
            lbl = rf"$P_{{{k}}}^{{(0,2)}}(z)$"
        elif radial_basis == "jacobi_orthonormal":
            lbl = rf"$\tilde{{P}}_{{{k}}}^{{(0,2)}}(z)$"
        else:
            lbl = f"raw {k}"
        ax.plot(r.numpy(), Phi[:, k].numpy(), lw=1.2, label=lbl)

    if orthogonalize and radial_basis not in ("jacobi", "jacobi_orthonormal"):
        T = radial_basis_orthogonal_matrix(Phi, w)
        Psi = Phi @ T
        for k in range(nmax):
            ax.plot(r.numpy(), Psi[:, k].numpy(), "--", lw=1.2, label=f"ortho {k}")
    elif orthogonalize and radial_basis in ("jacobi", "jacobi_orthonormal"):
        warnings.warn(
            "plot_radial_bases: orthogonalize=True ignored for Jacobi radial bases "
            f"(continuum-orthogonal under ∫_0^{r_cut} · r^{radial_measure} dr up to quadrature).",
            stacklevel=2,
        )
    ax.set_xlabel("r")
    ax.set_ylabel("R_n(r)")
    if radial_basis == "jacobi_orthonormal":
        jacobi_note = r" Jacobi $P_n^{(0,2)}$ / $\|P_n\|_{r^2\,dr}$ (analytic norm)."
    elif radial_basis == "jacobi":
        jacobi_note = r" Jacobi $P_n^{(0,2)}$; continuum $r^2$-orthogonal."
    else:
        jacobi_note = ""
    ax.set_title(f"{radial_basis} radial basis (measure ∝ r^{radial_measure}){jacobi_note}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax

#----- spherical expansion from rascaline ------
def spherical_expansion_rascaline(lmax, nmax, cutoff, frame, dtype = torch.float64):
    import metatensor.torch as mts
    from featomic.torch import SphericalExpansion, systems_to_torch

    hypers = {
           "cutoff": {
        "radius": cutoff,
        "smoothing": {
            "type": "ShiftedCosine",
            "width": 0.1
        }
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": lmax,
        "radial": {
            "type": "Gto",
            "max_radial": nmax
        }
    }
        }
    
    spex = SphericalExpansion(**hypers).compute(systems_to_torch(frame))
    for k in ['neighbor_type', 'center_type', 'o3_sigma']:
        spex = mts.remove_dimension(spex, 'keys', k)
    
    spex = spex.components_to_properties('o3_mu')
    spex = spex.keys_to_properties('o3_lambda')
    # return spex
    spex = spex[0].values.reshape(spex[0].shape[0], (lmax+1)**2, nmax+1)
    return spex

##------- local ------
def spherical_expansion_coeffs(
    lmax,
    nmax,
    atom_coords,
    *,
    radial_basis: Literal["gto", "monomial", "jacobi", "jacobi_orthonormal"] = "jacobi_orthonormal",
    values=None,
    factor: float = 1.0,
    dtype=torch.float64,
    r_cut: Optional[float] = None,
    orthogonalize: bool = False,
    radial_measure: float = 2.0,
    n_orthog_grid: int = 512,
):
    """Expand ``\\sum_j \\delta(\\mathbf{x}-\\mathbf{r}_{ji})`` on ``Y_{\\ell m}(\\hat r)`` with a radial menu.

    With ``r_{ji} = \\|\\mathbf{r}_{ji}\\|`` (or ``values``), per-channel weights are:

    * **monomial**: ``r_{ji}^n`` for ``n = 1,\\dots,n_{\\max}`` (``n_{\\max}`` channels).
    * **gto**: ``n_{\\max}`` Gaussians ``\\exp(-\\zeta_k r_{ji}^2)`` with ``\\zeta_k`` from ``gto_zeta_sequence``.
    * **jacobi**: ``P_{k}^{(0,2)}(2r_{ji}/r_{\\mathrm{cut}}-1)``. Requires ``r_cut``.
    * **jacobi_orthonormal**: same polynomials scaled to unit ``L^2([0,r_{\\mathrm{cut}}], r^2\\,dr)`` norm
      (diagonal Gram ``=1``). Requires ``r_cut``.

    Raw monomials / Gaussians are not orthogonal in the spherical radial measure
    ``\\propto r^2\\,dr`` on ``[0, r_\\mathrm{cut}]``. Set ``orthogonalize=True`` and pass ``r_cut`` to
    build a fixed ``(n_{\\max}\\times n_{\\max})`` mixing ``T`` on a trapezoid grid so each column of
    ``\\Phi(r) @ T`` is orthonormal under ``\\sum_i w_i \\phi_i(r_i)\\phi_j(r_i)`` with
    ``w_i`` from ``trapezoid_weights_radial_measure(..., radial_measure=radial_measure)``.

    Each channel is one ``sum_of_diracs``; outputs are stacked on the **last** dimension.

    Parameters
    ----------
    nmax
        Monomial: largest exponent. GTO: number of radial Gaussian channels.
    r_cut
        Required when ``orthogonalize`` is True: upper limit for the Gram matrix grid.
    radial_measure
        Exponent ``k`` in ``r^k\\,dr`` for integral measure, used in ``trapezoid_weights_radial_measure`` when ``orthogonalize=True``
        (``2`` matches spherical / Jacobi ``r^2`` convention).
    """
    if nmax < 1:
        raise ValueError("nmax must be >= 1 (radial channel count / max monomial degree).")
    if radial_basis in ("jacobi", "jacobi_orthonormal") and (r_cut is None or r_cut <= 0):
        raise ValueError("radial_basis='jacobi' and 'jacobi_orthonormal' require a positive r_cut.")
    if orthogonalize and (r_cut is None or r_cut <= 0):
        raise ValueError("orthogonalize=True requires a positive r_cut (Gram matrix on [0, r_cut]).")

    sphten = io.SphericalTensor(lmax, p_val=1, p_arg=-1)

    if values is None:
        r_ji = torch.linalg.norm(atom_coords, dim=-1)
    else:
        r_ji = values
    r_ji = (r_ji * factor).to(dtype=dtype)

    r_mean = r_ji.mean().detach()
    Phi_neighbors = eval_radial_basis_matrix(
        r_ji,
        nmax,
        radial_basis,
        r_mean=r_mean,
        r_cut=float(r_cut) if radial_basis in ("jacobi", "jacobi_orthonormal") else None,
    ).to(dtype=dtype)

    if orthogonalize and radial_basis not in ("jacobi", "jacobi_orthonormal"):
        r_grid = torch.linspace(
            0.0,
            float(r_cut),
            int(n_orthog_grid),
            dtype=torch.float64,
            device=r_ji.device,
        )
        Phi_grid = eval_radial_basis_matrix(
            r_grid,
            nmax,
            radial_basis,
            r_mean=r_mean,
            r_cut=float(r_cut) if radial_basis in ("jacobi", "jacobi_orthonormal") else None,
        ).to(dtype=torch.float64)
        w = trapezoid_weights_radial_measure(r_grid, radial_measure=radial_measure).to(
            dtype=torch.float64
        )
        T = radial_basis_orthogonal_matrix(Phi_grid, w).to(dtype=dtype)
        Phi_neighbors = Phi_neighbors @ T
    elif orthogonalize and radial_basis in ("jacobi", "jacobi_orthonormal"):
        warnings.warn(
            "spherical_expansion_coeffs: orthogonalize=True ignored for Jacobi radial bases.",
            stacklevel=2,
        )

    radial_weights = [Phi_neighbors[..., k] for k in range(nmax)]

    blocks = [
        sphten.sum_of_diracs(atom_coords, values=w).to(dtype=dtype) for w in radial_weights
    ]
    return torch.stack(blocks, dim=-1)


_EINSUM_CG_RADIAL_NU1_PAIR = "Lmn,...mx,...ny->...Lxy"



def _rho_prev_dict_valid(
    rho_prev_result,
    equivariant_m_dim: Optional[int] = None,
) :
    """True if input matches the shape (n_samples, 2*L+1, n_combinations, *radials).

    If ``equivariant_m_dim`` is set (``2*L+1`` for this block's output L), also checks ``shape[1]``.
    """
    if rho_prev_result is None or not isinstance(rho_prev_result, dict):
        return False
    for val in rho_prev_result.values():

        if not isinstance(val, (list, tuple)) or len(val) != 2:
            return False
        tensor_vals, combinations = val
        if not torch.is_tensor(tensor_vals):
            return False
        if tensor_vals.ndim < 4 or tensor_vals.shape[2] != len(combinations):
            return False
        if equivariant_m_dim is not None and tensor_vals.shape[1] != equivariant_m_dim:
            return False
    return True


def _einsum_cg_radial_chain_equation(n_radial_axes_on_prev: int) -> str:
    """Return an einsum equation with one letter per radial axis already on ``field_prev``, plus one new axis.

    For ``k = n_radial_axes_on_prev`` radials on the left operand (after the ``m`` index), builds::

        Lmn,...m<r0..r_{k-1}>,...n<rk> -> ...L<r0..r_{k-1}><rk>

    so the output carries ``k+1`` radial indices (outer product over neighbor radial channels).
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n_radial_axes_on_prev + 1 > len(letters):
        raise ValueError("radial rank too large for einsum letter budget")
    r_prev = letters[:n_radial_axes_on_prev]
    r_new = letters[n_radial_axes_on_prev]
    sub_prev = "...m" + "".join(r_prev)
    sub_rho = "...n" + r_new
    sub_out = "...L" + "".join(r_prev) + r_new
    return f"Lmn,{sub_prev},{sub_rho}->{sub_out}"


def _einsum_cg_radial_nu1_pair(cg_wmn, rho_a, rho_b):
    """Clebsch–Gordan product of two nu=1 fields with one radial index each.

    ``cg_wmn`` is (L, m, n); ``rho_*`` are (..., m_i, r_i). Returns (..., L, r_a, r_b).
    ``...`` absorbs any leading batch / structure dimensions (same convention as ``imx``).
    """
    return torch.einsum(_EINSUM_CG_RADIAL_NU1_PAIR, cg_wmn, rho_a, rho_b)


def _einsum_cg_radial_contract_new_rho(cg_wmn, field_prev, rho_new, n_radial_axes_on_prev):
    """One CG step: couple ``field_prev`` (..., L_prev, m, *r[:k]) with a fresh nu=1 slice (..., n, r_new).

    ``n_radial_axes_on_prev`` is k (the number of radial axes already carried by ``field_prev``,
    not counting the m axis). Output (..., L_out, *r[:k], r_new).
    """
    return torch.einsum(
        _einsum_cg_radial_chain_equation(n_radial_axes_on_prev),
        cg_wmn,
        field_prev,
        rho_new,
    )


def equivariant_powerspectrum(rho1i, L, parity:int=0, cg=None, **kwargs):
    """ Compute all equivariant powerspectra up to lcut.
    lcut:int, max spherical harmonic degree to compute in the result
    parity: 'all', 'even', 'odd' corresponds to 0, 1, -1 (unlike usual ACDC where 1 = polar tensor and -1 = pseudo tensor)
    
    Returns:
        dict: Dictionary with parity as keys and tuples of (tensor_values, combinations) as values.
              combinations is a list of (l1, l2, L) tuples.
    """
    # maximum l value available in rho1i
    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-2], dtype=torch.float32))) - 1

    if cg is None:
        cg = CG(lmax=lmax_rho1i, device=rho1i.device)
    
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
        
        rho2_ = _einsum_cg_radial_nu1_pair(
            cg_coeffs, rho1i[:, get_lm_slice(l1)], rho1i[:, get_lm_slice(l2)]
        )
        
        # For polar tensors: output parity = (-1)^(l1 + l2)
        combination_parity = (-1)**(l1 + l2) # p1*p2
        rho2_list[combination_parity].append(rho2_)
        combinations_list[combination_parity].append((l1, l2, L))

    if parity == 0:
        result = {}
        if rho2_list[1]:  # Check if list is not empty
            result[1] = (torch.stack(rho2_list[1], dim=2), combinations_list[1])
        if rho2_list[-1]:  # Check if list is not empty
            result[-1] = (torch.stack(rho2_list[-1], dim=2), combinations_list[-1])
        
        if not result:  # If both lists are empty
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    if parity in rho2_list and rho2_list[parity]:  
        return {parity: (torch.stack(rho2_list[parity], dim=2), combinations_list[parity])}
    else:
        available_parities = [p for p in rho2_list if rho2_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            return {available_parities[0]: (torch.stack(rho2_list[available_parities[0]], dim=2), combinations_list[available_parities[0]])}
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
    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-2], dtype=torch.float32))) - 1
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
                intermediate = _einsum_cg_radial_nu1_pair(
                    cg_coeffs_12, rho1i[:, get_lm_slice(l1)], rho1i[:, get_lm_slice(l2)]
                )
                rho3_ = _einsum_cg_radial_contract_new_rho(
                    cg_coeffs_123, intermediate, rho1i[:, get_lm_slice(l3)], n_radial_axes_on_prev=2
                )
                
                # For polar tensors: output parity = (-1)^(l1 + l2 + l3)
                combination_parity = (-1)**(l1 + l2 + l3+L)
                rho3_list[combination_parity].append(rho3_)
                combinations_list[combination_parity].append((l1, l2, L12, l3, L))
                total_processed += 1
    if parity == 0:
        result = {}
        if rho3_list[1]:  # Check if list is not empty
            result[1] = (torch.stack(rho3_list[1], dim=2), combinations_list[1])
        if rho3_list[-1]:  # Check if list is not empty
            result[-1] = (torch.stack(rho3_list[-1], dim=2), combinations_list[-1])
        if not result:  # If both lists are empty
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    # For specific parity
    if parity in rho3_list and rho3_list[parity]:  # Check if list exists and is not empty
        return {parity: (torch.stack(rho3_list[parity], dim=2), combinations_list[parity])}
    else:
        available_parities = [p for p in rho3_list if rho3_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            return {available_parities[0]: (torch.stack(rho3_list[available_parities[0]], dim=2), combinations_list[available_parities[0]])}
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

        lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-2], dtype=dtype))) - 1
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

    lmax_rho1i = int(torch.sqrt(torch.tensor(rho1i.shape[-2], dtype=dtype))) - 1
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
                cached = descriptor_cache[cache_key]
                if _rho_prev_dict_valid(cached, equivariant_m_dim=2 * l + 1):
                    rho_prev[l] = cached
                else:
                    print(f"Invalid cache key: {cache_key}")
                    del descriptor_cache[cache_key]
            if cache_key not in descriptor_cache:
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
                prev_L_key = L_intermediates[correlation_order - 3]
                last_l = l_values[-1]

                if prev_L_key in rho_prev and rho_prev[prev_L_key] is not None:
                    rho_prev_result = rho_prev[prev_L_key]
                    target_combo = tuple(combination[:-1])
                    # Find the matching combination in rho_prev
                    m_prev = 2 * prev_L_key + 1
                    for parity_key in rho_prev_result:
                        tensor_vals, combinations = rho_prev_result[parity_key]
                        if (
                            not torch.is_tensor(tensor_vals)
                            or tensor_vals.ndim < 4
                            or tensor_vals.shape[1] != m_prev
                            or tensor_vals.shape[2] != len(combinations)
                        ):
                            continue
                        for idx, combo in enumerate(combinations):
                            if combo == target_combo:
                                current_tensor = tensor_vals[:, :, idx, ...]
                                break
                        if current_tensor is not None:
                            break

                    if current_tensor is not None:
                        cg_last = cg.cg[prev_L_key, last_l][L]
                        # Cached slice must carry the (2*prev_L_key+1) m-axis expected by this CG matrix
                        if cg_last.shape[1] != current_tensor.shape[1]:
                            current_tensor = None
                        else:
                            rho_ = _einsum_cg_radial_contract_new_rho(
                                cg_last,
                                current_tensor,
                                rho1i[:, get_lm_slice(last_l)],
                                n_radial_axes_on_prev=correlation_order - 1,
                            )
                            current_tensor = None  # Skip fallback
            
            # Fallback: compute sequentially from rho1i (used for all cases where recursive fails)
            if current_tensor is None:
                current_tensor = _einsum_cg_radial_nu1_pair(
                    cg.cg[l_values[0], l_values[1]][L_intermediates[0]],
                    rho1i[:, get_lm_slice(l_values[0])],
                    rho1i[:, get_lm_slice(l_values[1])],
                )
                # Each step adds one radial axis; after the first pair there are 2, then 3, ...
                for i in range(1, correlation_order - 1):
                    current_tensor = _einsum_cg_radial_contract_new_rho(
                        cg.cg[L_intermediates[i - 1], l_values[i + 1]][L_intermediates[i]],
                        current_tensor,
                        rho1i[:, get_lm_slice(l_values[i + 1])],
                        n_radial_axes_on_prev=i + 1,
                    )
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
                result[1] = (torch.stack(rho_list[1], dim=2), combinations_list[1])
            else:
                result[1] = torch.stack(rho_list[1], dim=2)
        if rho_list[-1]:  
            if return_combinations:
                result[-1] = (torch.stack(rho_list[-1], dim=2), combinations_list[-1])
            else:
                result[-1] = torch.stack(rho_list[-1], dim=2)
        
        if not result:  
            warnings.warn("No valid combinations found for any parity")
            return None
        return result
        
    # For specific parity
    if parity in rho_list and rho_list[parity]:  # Check if list exists and is not empty
        if return_combinations:
            return {parity: (torch.stack(rho_list[parity], dim=2), combinations_list[parity])}
        else:
            return {parity: torch.stack(rho_list[parity], dim=2)}
    else:
        available_parities = [p for p in rho_list if rho_list[p]]
        if available_parities:
            warnings.warn(f"No valid combinations found for parity {parity}, returning parity {available_parities[0]}")
            if return_combinations:
                return {available_parities[0]: (torch.stack(rho_list[available_parities[0]], dim=2), combinations_list[available_parities[0]])}
            else:
                return {available_parities[0]: torch.stack(rho_list[available_parities[0]], dim=2)}
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
                return torch.cat([bis[1][0], bis[-1][0]], dim=2)
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
                return torch.cat([tris[1][0], tris[-1][0]], dim=2)
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


#  
# ----- DescriptorCalculator -----
# has the functionality to 
# 1.compute the descriptor for a given atom in a structure (instantiate a pymatgen structure from absolute coordinates)
# (use_structure=True, atom_index=ATOM_INDEX). if ATOM_INDEX is None, use the first atom.
# 2.compute the descriptor for a given tensor of absolute coordinates (using the cutoff function to form relative vectors)
# (use_structure=False, atom_index=ATOM_INDEX), if ATOM_INDEX is not None, use the cutoff function to form relative vectors from the central atom to the neighbors.
# 3.compute the descriptor for a given tensor of relative coordinates (using the cutoff function to form relative vectors)
# (use_structure=False, atom_index=None)

class DescriptorCalculator(torch.nn.Module):
    def __init__(
        self,
        cutoff_type: str,
        lmax: int,
        rcut: float,
        correlation_order: int,
        central_atom: bool = True,
        smoothing_fn: str = None,
        smearing_width: float = None,
        dtype=torch.float64,
        device="cpu",
        lcut=None,
        cg=None,
        nmax: int = 1,
        radial_basis: Literal["gto", "monomial", "jacobi", "jacobi_orthonormal"] = "monomial",
        radial_orthogonalize: bool = False,
        radial_measure: float = 2.0,
        n_orthog_grid: int = 512,
    ):
        """
        cutoff_type: str, type of cutoff function to use (e.g. 'smooth_radial_cutoff')
        lmax: int, maximum spherical harmonic degree to compute
        central_atom: bool, whether to include the central atom in the local environment
        smoothing_fn: str, type of smoothing function to use (e.g. 'cosine', 'gaussian')
        smearing_width: float, width of the smoothing function (default is 0.1)
        dtype: torch.dtype, data type for computations (default is torch.float32)
        cg: CG object, for equivariant powerspectrum and bispectrum calculations
        nmax: Number of radial channels (see ``spherical_expansion_coeffs``).
        radial_basis: Radial factor menu on neighbor distances (``gto``, ``monomial``, ``jacobi``, …).
        radial_orthogonalize: If True, build discrete Gram orthonormal mixing on ``[0, rcut]`` for non-Jacobi bases (same neighbor cutoff radius).
        radial_measure: Exponent in ``r^{\\cdot}\\,dr`` for that Gram quadrature (default ``2``, spherical).
        n_orthog_grid: Grid size when ``radial_orthogonalize`` is True.

        Defaults ``nmax=1``, ``radial_basis='monomial'`` reproduce the legacy single radial channel
        (first monomial is neighbor distance, weighted like the old ``factor * ||r||`` path).

        Summary of possible use modes :
        -
        """
        super().__init__()
        self.lmax = lmax
        self.nmax = int(nmax)
        self.radial_basis = radial_basis
        self.radial_orthogonalize = radial_orthogonalize
        self.radial_measure = float(radial_measure)
        self.n_orthog_grid = int(n_orthog_grid)
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

    def _spherical_expansion_from_neighbors(
        self,
        atom_coords: torch.Tensor,
        weights: torch.Tensor,
        *,
        dtype: torch.dtype,
    ):
        """Local-frame neighbor vectors → stacked ``sum_of_diracs`` radial channels (see ``spherical_expansion_coeffs``)."""
        return spherical_expansion_coeffs(
            self.lmax,
            self.nmax,
            atom_coords,
            radial_basis=self.radial_basis,
            factor=weights,
            dtype=dtype,
            r_cut=float(self.rcut),
            orthogonalize=self.radial_orthogonalize,
            radial_measure=self.radial_measure,
            n_orthog_grid=self.n_orthog_grid,
        )

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
        return self._spherical_expansion_from_neighbors(
            atom_coords, weights, dtype=dtype
        )

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
            spex = self._spherical_expansion_from_neighbors(
                points, w, dtype=points.dtype
            ).unsqueeze(0)

        
        elif atom_index is not None: # if proving a tensor of absolute coordinates    "
            atom_coords, weights = self._local_geometry_from_tensor(
                points, int(atom_index), include_central_site=include_central_site
            )
            spex = self._spherical_expansion_from_neighbors(
                atom_coords, weights, dtype=points.dtype
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
