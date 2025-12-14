import numpy as np
from numba import njit


@njit
def matvec(A, v):
    return np.array(
        [
            A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2],
            A[1, 0] * v[0] + A[1, 1] * v[1] + A[1, 2] * v[2],
            A[2, 0] * v[0] + A[2, 1] * v[1] + A[2, 2] * v[2],
        ]
    )


@njit
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit
def unravel_site(idx, n_atoms, ni, nj, nk):
    njk = nj * nk
    nijk = ni * njk

    i_atom = idx // nijk
    r = idx % nijk
    i = r // njk
    r = r % njk
    j = r // nk
    k = r % nk
    return i_atom, i, j, k


@njit
def ravel_site(i_atom, i, j, k, n_atoms, ni, nj, nk):
    njk = nj * nk
    nijk = ni * njk
    return i_atom * nijk + i * njk + j * nk + k


@njit
def random_unit_vector():
    v0 = np.random.normal()
    v1 = np.random.normal()
    v2 = np.random.normal()
    v = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    if v == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return np.array([v0 / v, v1 / v, v2 / v])


@njit
def total_heisenberg_energy(
    s, nb_offsets, nb_atom, nb_ijk, nb_J, K, H, muB, g
):

    n_atoms, ni, nj, nk, _ = s.shape

    EJ = 0.0
    for i_atom in range(n_atoms):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    sl0 = s[i_atom, i, j, k, 0]
                    sl1 = s[i_atom, i, j, k, 1]
                    sl2 = s[i_atom, i, j, k, 2]
                    h0, h1, h2 = local_field_at_site(
                        s,
                        i_atom,
                        i,
                        j,
                        k,
                        nb_offsets,
                        nb_atom,
                        nb_ijk,
                        nb_J,
                        ni,
                        nj,
                        nk,
                    )
                    EJ -= 0.5 * (sl0 * h0 + sl1 * h1 + sl2 * h2)

    EK = 0.0
    for i_atom in range(n_atoms):
        K_l = K[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    v0 = s[i_atom, i, j, k, 0]
                    v1 = s[i_atom, i, j, k, 1]
                    v2 = s[i_atom, i, j, k, 2]
                    Kv0 = K_l[0, 0] * v0 + K_l[0, 1] * v1 + K_l[0, 2] * v2
                    Kv1 = K_l[1, 0] * v0 + K_l[1, 1] * v1 + K_l[1, 2] * v2
                    Kv2 = K_l[2, 0] * v0 + K_l[2, 1] * v1 + K_l[2, 2] * v2
                    EK -= v0 * Kv0 + v1 * Kv1 + v2 * Kv2

    EH = 0.0
    for i_atom in range(n_atoms):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    v0 = s[i_atom, i, j, k, 0]
                    v1 = s[i_atom, i, j, k, 1]
                    v2 = s[i_atom, i, j, k, 2]
                    EH -= muB * g * (v0 * H[0] + v1 * H[1] + v2 * H[2])

    return EJ + EK + EH


@njit
def local_field_at_site(
    s, i_atom, i, j, k, nb_offsets, nb_atom, nb_ijk, nb_J, ni, nj, nk
):
    hx = 0.0
    hy = 0.0
    hz = 0.0
    start = nb_offsets[i_atom]
    end = nb_offsets[i_atom + 1]

    for b in range(start, end):
        nn = nb_atom[b]
        di = nb_ijk[b, 0]
        dj = nb_ijk[b, 1]
        dk = nb_ijk[b, 2]

        ii = (i + di) % ni
        jj = (j + dj) % nj
        kk = (k + dk) % nk

        snn0 = s[nn, ii, jj, kk, 0]
        snn1 = s[nn, ii, jj, kk, 1]
        snn2 = s[nn, ii, jj, kk, 2]

        J = nb_J[b]
        hx += J[0, 0] * snn0 + J[0, 1] * snn1 + J[0, 2] * snn2
        hy += J[1, 0] * snn0 + J[1, 1] * snn1 + J[1, 2] * snn2
        hz += J[2, 0] * snn0 + J[2, 1] * snn1 + J[2, 2] * snn2

    return hx, hy, hz


@njit
def metropolis_heisenberg(
    idx,
    s,
    beta,
    E,
    n_local_sweeps,
    nb_offsets,
    nb_atom,
    nb_ijk,
    nb_J,
    K,
    H,
    muB,
    g,
    seed,
):

    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n = n_atoms * ni * nj * nk

    for _ in range(n_local_sweeps * n):
        flat_idx = np.random.randint(n)
        i_atom, i, j, k = unravel_site(flat_idx, n_atoms, ni, nj, nk)

        s_orig0 = s[i_atom, i, j, k, 0]
        s_orig1 = s[i_atom, i, j, k, 1]
        s_orig2 = s[i_atom, i, j, k, 2]

        s_cand0 = np.random.normal()
        s_cand1 = np.random.normal()
        s_cand2 = np.random.normal()

        norm = np.sqrt(
            s_cand0 * s_cand0 + s_cand1 * s_cand1 + s_cand2 * s_cand2
        )

        s_cand0 /= norm
        s_cand1 /= norm
        s_cand2 /= norm

        delta0 = s_cand0 - s_orig0
        delta1 = s_cand1 - s_orig1
        delta2 = s_cand2 - s_orig2

        h0, h1, h2 = local_field_at_site(
            s,
            i_atom,
            i,
            j,
            k,
            nb_offsets,
            nb_atom,
            nb_ijk,
            nb_J,
            ni,
            nj,
            nk,
        )
        dEJ = -(delta0 * h0 + delta1 * h1 + delta2 * h2)

        K_ion = K[i_atom]
        s_sum0 = s_cand0 + s_orig0
        s_sum1 = s_cand1 + s_orig1
        s_sum2 = s_cand2 + s_orig2

        K_s_sum0 = (
            K_ion[0, 0] * s_sum0 + K_ion[0, 1] * s_sum1 + K_ion[0, 2] * s_sum2
        )
        K_s_sum1 = (
            K_ion[1, 0] * s_sum0 + K_ion[1, 1] * s_sum1 + K_ion[1, 2] * s_sum2
        )
        K_s_sum2 = (
            K_ion[2, 0] * s_sum0 + K_ion[2, 1] * s_sum1 + K_ion[2, 2] * s_sum2
        )

        dEK = -(delta0 * K_s_sum0 + delta1 * K_s_sum1 + delta2 * K_s_sum2)

        dEH = -muB * g * (delta0 * H[0] + delta1 * H[1] + delta2 * H[2])

        dE = dEJ + dEK + dEH

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            s[i_atom, i, j, k, 0] = s_cand0
            s[i_atom, i, j, k, 1] = s_cand1
            s[i_atom, i, j, k, 2] = s_cand2
            E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))
