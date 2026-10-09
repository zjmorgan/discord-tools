"""
Ewald summation of magnetic dipole-dipole interactions on a periodic
supercell.

The dipolar energy of the infinite periodic system is written as

    E = (mu0 / 4 pi) * 1/2 * sum_{i, j} mu_i . T_ij . mu_j,

where i and j run over all sites of the supercell and T_ij contains every
periodic image (including i = j: a moment interacts with its own images).
T depends only on the sublattice pair and the cell offset, so it is computed
once and can be used like an exchange tensor.

Ewald splits T into (Gaussian units, mu0/4pi = 1)

- real space:  sum over images r = r_ij + n of B(r) I - C(r) r r^T
- reciprocal:  (4 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 k k^T
               cos(k . r_ij)
- self:        -(4 alpha^3 / 3 sqrt(pi)) I for i = j
- surface:     (4 pi / 3 V) I for a spherical sample in vacuum, 0 for
               tin-foil (conducting) boundaries

with B(r) = [erfc(a r) + 2 a r / sqrt(pi) exp(-a^2 r^2)] / r^3 and
C(r) = [3 erfc(a r) + 2 a r / sqrt(pi) (3 + 2 a^2 r^2) exp(-a^2 r^2)] / r^5.
The total does not depend on the splitting parameter alpha.
"""

import numpy as np
from scipy.special import erfc

BOUNDARIES = ("tinfoil", "vacuum")


def ewald_dipolar_tensors(
    A, xyz, super_cell, boundary="tinfoil", alpha=None, precision=4.0
):
    """
    Ewald-summed dipolar interaction tensors on a periodic supercell.

    Parameters
    ----------
    A : array_like
        Direct-lattice transform, shape ``(3, 3)``: Cartesian position of
        fractional coordinates ``uvw`` is ``A @ uvw`` (columns are the unit
        cell vectors), in Angstrom.
    xyz : array_like
        Fractional coordinates of the atoms in the unit cell,
        shape ``(n_atoms, 3)``.
    super_cell : tuple of int
        Supercell replication ``(ni, nj, nk)``.
    boundary : {"tinfoil", "vacuum"}
        Boundary condition at infinity: conducting surroundings (no
        demagnetizing field) or a spherical sample in vacuum.
    alpha : float, optional
        Ewald splitting parameter in 1/Angstrom. Defaults to
        ``precision / r_c`` with ``r_c`` the smallest supercell width.
    precision : float
        Real- and reciprocal-space cutoffs satisfy ``alpha r_c = precision``
        and ``k_c / (2 alpha) = precision``; truncation errors scale as
        ``exp(-precision^2)``.

    Returns
    -------
    T : ndarray
        Cartesian tensors in 1/Angstrom^3, shape
        ``(n_atoms, n_atoms, ni, nj, nk, 3, 3)``: ``T[a, b, m]`` couples
        atom ``a`` in cell ``c`` to atom ``b`` in cell ``c + m`` (modulo the
        supercell). ``T[a, a, 0, 0, 0]`` is the self-image term.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}")

    A = np.asarray(A, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    N = np.asarray(super_cell, dtype=int)
    n_atoms = len(xyz)

    L = A * N[None, :]  # supercell vectors as columns
    V = abs(np.linalg.det(L))
    G = 2.0 * np.pi * np.linalg.inv(L).T  # reciprocal supercell vectors

    # distance between opposite faces of the supercell along each axis
    widths = V / np.linalg.norm(np.cross(L[:, [1, 2, 0]].T, L[:, [2, 0, 1]].T), axis=1)

    if alpha is None:
        alpha = precision / widths.min()
    r_cut = precision / alpha
    k_cut = 2.0 * alpha * precision

    # displacements in supercell fractional coordinates, wrapped to [-1/2, 1/2)
    m = np.stack(
        np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    frac = (
        xyz[None, :, None, :] - xyz[:, None, None, :] + m[None, None, :, :]
    ) / N
    frac -= np.round(frac)
    d = frac.reshape(-1, 3) @ L.T  # Cartesian, shape (P, 3)

    T = np.zeros((len(d), 3, 3))
    eye = np.eye(3)

    # real space
    n_img = np.ceil(r_cut / widths).astype(int) + 1
    images = np.stack(
        np.meshgrid(*[np.arange(-n, n + 1) for n in n_img], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3) @ L.T
    for start in range(0, len(d), 256):
        r = d[start : start + 256, None, :] + images[None, :, :]
        r2 = np.einsum("pij,pij->pi", r, r)
        mask = (r2 < r_cut**2) & (r2 > 1e-20)
        rr = np.sqrt(np.where(mask, r2, 1.0))
        ar = alpha * rr
        gauss = 2.0 * ar / np.sqrt(np.pi) * np.exp(-(ar**2))
        B = np.where(mask, (erfc(ar) + gauss) / rr**3, 0.0)
        C = np.where(
            mask,
            (3.0 * erfc(ar) + gauss * (3.0 + 2.0 * ar**2)) / rr**5,
            0.0,
        )
        T[start : start + 256] += B.sum(axis=1)[:, None, None] * eye
        T[start : start + 256] -= np.einsum("pi,pia,pib->pab", C, r, r)

    # reciprocal space
    n_k = np.ceil(k_cut / np.linalg.norm(G, axis=0)).astype(int) + 1
    hkl = np.stack(
        np.meshgrid(*[np.arange(-n, n + 1) for n in n_k], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    k = hkl @ G.T
    k2 = np.einsum("ki,ki->k", k, k)
    keep = (k2 > 1e-20) & (k2 < k_cut**2)
    k, k2 = k[keep], k2[keep]
    weight = 4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha**2)) / k2
    kk = weight[:, None, None] * k[:, :, None] * k[:, None, :]
    for start in range(0, len(d), 4096):
        cos = np.cos(d[start : start + 4096] @ k.T)
        T[start : start + 4096] += np.einsum("pk,kab->pab", cos, kk)

    T = T.reshape(n_atoms, n_atoms, *N, 3, 3)

    # self term
    for a in range(n_atoms):
        T[a, a, 0, 0, 0] -= 4.0 * alpha**3 / (3.0 * np.sqrt(np.pi)) * eye

    if boundary == "vacuum":
        T += 4.0 * np.pi / (3.0 * V) * eye

    return T


def dipolar_energy(T, mu):
    """
    Dipolar energy ``1/2 sum_ij mu_i . T_ij . mu_j`` of a configuration.

    Parameters
    ----------
    T : ndarray
        Tensors from :func:`ewald_dipolar_tensors`.
    mu : ndarray
        Cartesian moments, shape ``(n_atoms, ni, nj, nk, 3)``.

    Returns
    -------
    E : float
        Energy in units of (mu0/4pi) [mu]^2 / Angstrom^3.
    """
    mu = np.asarray(mu, dtype=float)
    n_atoms, ni, nj, nk, _ = mu.shape
    # field h_a(c) = sum_{b, m} T_ab(m) mu_b(c + m), by FFT convolution
    Tk = np.fft.fftn(T, axes=(2, 3, 4))
    muk = np.fft.fftn(mu, axes=(1, 2, 3))
    # sum_c mu_a(c) T_ab(m) mu_b(c+m) = (1/Nc) sum_q conj(mu_a(q)) T_ab(q)* mu_b(q)
    # with T_ab(m) correlating c and c + m
    hk = np.einsum("abxyzij,bxyzj->axyzi", np.conj(Tk), muk)
    E = np.einsum("axyzi,axyzi->", np.conj(muk), hk).real
    return 0.5 * E / (ni * nj * nk)
