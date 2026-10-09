"""
Automatic identification of the ordering mode and its order parameter.

The moments are Fourier transformed per sublattice on the supercell grid,
F_a(q) = sum_c mu_a(c) exp(-2 pi i q . c), and combined into the sublattice
correlation matrix

    C_ab(q) = sum_alpha F_a,alpha(q) conj(F_b,alpha(q)) / N,

a Hermitian ``n_atoms x n_atoms`` matrix at each wavevector. Its largest
eigenvalue over q picks the ordering wavevector q* and the eigenvector v the
sublattice pattern, which handles multi-sublattice order (e.g. the two Mn of
rutile MnF2 ordering antiparallel at q = 0) without choosing hkl by hand.

The order parameter of a configuration is

    m^2 = (n_atoms / N^2) sum_{q in Q*} sum_alpha |sum_a conj(v_a(q)) F_a,alpha(q)|^2,

summed over the star Q* of wavevectors whose eigenvalue is within a
tolerance of the maximum, so that switching between equivalent domains does
not change m. Perfect collinear commensurate order (or a single-q helix with
both +q and -q in the star) gives m = |mu|.
"""

import numpy as np


def sublattice_fourier(mu):
    """
    Per-sublattice Fourier transform of moments.

    Parameters
    ----------
    mu : array_like
        Moments, shape ``(..., n_atoms, ni, nj, nk, 3)``.

    Returns
    -------
    F : ndarray
        Complex, same shape as ``mu``.
    """
    return np.fft.fftn(np.asarray(mu, dtype=float), axes=(-4, -3, -2))


def correlation_matrix(mu):
    """
    Sublattice correlation matrix C_ab(q), averaged over leading axes.

    Parameters
    ----------
    mu : array_like
        Moments, shape ``(..., n_atoms, ni, nj, nk, 3)``.

    Returns
    -------
    C : ndarray
        Shape ``(ni, nj, nk, n_atoms, n_atoms)``.
    """
    mu = np.asarray(mu, dtype=float)
    n_atoms, ni, nj, nk = mu.shape[-5:-1]
    F = sublattice_fourier(mu).reshape(-1, n_atoms, ni, nj, nk, 3)
    C = np.einsum("sa...x,sb...x->...ab", F, np.conj(F)) / len(F)
    return C / (n_atoms * ni * nj * nk)


def ordering_modes(mu, rtol=0.05):
    """
    Dominant ordering wavevectors (the star Q*) and sublattice patterns.

    Parameters
    ----------
    mu : array_like
        One or more configurations, shape ``(..., n_atoms, ni, nj, nk, 3)``;
        correlations are averaged over the leading axes.
    rtol : float
        Wavevectors whose largest eigenvalue is within ``rtol`` of the
        overall maximum belong to the star.

    Returns
    -------
    q_index : ndarray of int
        Grid indices of the star, shape ``(n_q, 3)``; the wavevector in
        reciprocal-lattice units is ``q_index / (ni, nj, nk)``.
    modes : ndarray
        Normalized sublattice eigenvectors, shape ``(n_q, n_atoms)``.
    weight : float
        The largest eigenvalue (per-site structure factor of the mode).
    """
    C = correlation_matrix(mu)
    N = C.shape[:3]
    vals, vecs = np.linalg.eigh(C)  # ascending
    lam = vals[..., -1]
    lam_max = lam.max()
    star = np.argwhere(lam >= (1.0 - rtol) * lam_max)
    modes = np.array([vecs[tuple(q)][:, -1] for q in star])
    return star.astype(np.int64), modes, float(lam_max)


def order_parameter(mu, q_index, modes):
    """
    Order parameter m of each configuration for a given star and modes.

    Parameters
    ----------
    mu : array_like
        Moments, shape ``(..., n_atoms, ni, nj, nk, 3)``.
    q_index, modes : ndarray
        From :func:`ordering_modes`.

    Returns
    -------
    m : ndarray
        Shape ``mu.shape[:-5]``, in the units of ``mu``.
    """
    mu = np.asarray(mu, dtype=float)
    n_atoms, ni, nj, nk = mu.shape[-5:-1]
    n_sites = n_atoms * ni * nj * nk
    F = sublattice_fourier(mu)
    Fq = F[..., :, q_index[:, 0], q_index[:, 1], q_index[:, 2], :]
    # Fq: (..., n_atoms, n_q, 3); project each q on its sublattice mode
    proj = np.einsum("qa,...aqx->...qx", np.conj(modes), Fq)
    m2 = n_atoms * (np.abs(proj) ** 2).sum(axis=(-2, -1)) / n_sites**2
    return np.sqrt(m2)
