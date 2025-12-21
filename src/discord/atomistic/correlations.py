import numpy as np


def vector_vector(v):
    """Compute real-space vector–vector correlations C_ij(r).

    Parameters
    ----------
    v : ndarray
        Unit vector configuration with shape ``(nr, nb, nx, ny, nz, 3)``.

    Returns
    -------
    Cij : ndarray
        Correlations with shape ``(nr, nb, nb, nx, ny, nz)``.
    """

    v = np.asarray(v)
    nr, nb, nx, ny, nz, _ = v.shape
    N = nx * ny * nz

    F = np.fft.fftn(v, axes=(2, 3, 4))
    Cij = np.empty((nr, nb, nb, nx, ny, nz), dtype=np.float64)

    rx = -np.arange(nx) % nx
    ry = -np.arange(ny) % ny
    rz = -np.arange(nz) % nz

    for i in range(nb):
        for j in range(i, nb):
            acc = 0.0
            for a in range(3):
                prod = np.conj(F[:, i, ..., a]) * F[:, j, ..., a]
                acc += np.fft.ifftn(prod, axes=(1, 2, 3)).real

            corr = acc / N
            Cij[:, i, j] = corr
            if j != i:
                corr_sym = corr[:, rx][:, :, ry][:, :, :, rz]
                Cij[:, j, i] = corr_sym

    return Cij
