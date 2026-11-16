# from https://github.com/curiosity54/mlelec/blob/main/src/mlelec/utils/symmetry.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

I_SQRT_2 = 1.0 / np.sqrt(2)
SQRT_2 = np.sqrt(2)


def _rotation_matrix_from_angles(alpha, beta, gamma):
    """A Cartesian rotation matrix in the appropriate convention
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return torch.from_numpy(
        Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()
    )


def rotate_frame(frame, _rotation):
    """
    Utility function to also rotate a structure, given as an Atoms frame.
    NB: it will rotate positions and cell, and no other array.

    frame: ase.Atoms
        An atomic structure in ASE format, that will be modified in place

    Returns:
    -------
    The rotated frame.
    """
    frame = frame.copy()
    frame.positions = frame.positions @ _rotation.detach().numpy().T
    frame.cell = frame.cell @ _rotation.detach().numpy().T
    return frame
    
def invert_frame(frame, invert=True):
    """
    Utility function to invert a structure, given as an Atoms frame.
    """
    frame = frame.copy()
    frame.positions = frame.positions * -1
    frame.cell = frame.cell * -1
    return frame

def _wigner_d(L, alpha, beta, gamma):
    """Computes a Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    # print(wigner_d(L, alpha, beta, gamma))
    return torch.tensor(
        wigner_d(L, alpha, beta, gamma), dtype=torch.complex128
    ).reshape((2 * L + 1, 2 * L + 1))


def _r2c(sp):
    """Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l"""

    l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = torch.zeros(len(sp), dtype=torch.complex128)
    rc[l] = sp[l]
    for m in range(1, l + 1):
        rc[l + m] = (sp[l + m] + 1j * sp[l - m]) * I_SQRT_2 * (-1) ** m
        rc[l - m] = (sp[l + m] - 1j * sp[l - m]) * I_SQRT_2
    return rc


def _c2r(cp):
    """Complex to real SPH. Assumes a block with 2l+1 complex
    corresponding to Y^m_l with m indices from -l to +l"""
    l = (len(cp) - 1) // 2  # infers l from the vector size
    rs = torch.zeros(len(cp), dtype=torch.float64)
    rs[l] = np.real(cp[l])
    for m in range(1, l + 1):
        rs[l - m] = (-1) ** m * SQRT_2 * np.imag(cp[l + m])
        rs[l + m] = (-1) ** m * SQRT_2 * np.real(cp[l + m])
    return rs


def _real2complex(L: int):
    dim = 2 * L + 1
    mat = torch.zeros((dim, dim), dtype=torch.complex128)
    # m = 0
    mat[L, L] = 1.0

    if L == 0:
        return mat
    for m in range(1, L + 1):
        # m > 0
        mat[L + m, L + m] = I_SQRT_2 * (-1) ** m
        mat[L + m, L - m] = I_SQRT_2 * 1j * (-1) ** m
        # m < 0
        mat[L - m, L + m] = I_SQRT_2
        mat[L - m, L - m] = -I_SQRT_2 * 1j

    return mat


def _wigner_d_real(L, alpha, beta, gamma, device: str = None):
    # Ylm in xyz convention! (not the usual convention)
    r2c_mat = torch.hstack(
        [_r2c(torch.eye(2 * L + 1)[i])[:, None] for i in range(2 * L + 1)]
    )
    c2r_mat = torch.conj(r2c_mat).T
    wig = _wigner_d(int(L), alpha, beta, gamma)
    # print(c2r_mat.shape, wig.shape, r2c_mat.shape)
    return torch.real(c2r_mat @ torch.conj(wig) @ r2c_mat).to(device)
