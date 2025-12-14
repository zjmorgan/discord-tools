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
                    sl = s[i_atom, i, j, k]
                    h_eff = local_field_at_site(
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
                    EJ -= 0.5 * dot(sl, h_eff)

    EK = 0.0
    for i_atom in range(n_atoms):
        K_l = K[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    v = s[i_atom, i, j, k]
                    Kv = matvec(K_l, v)
                    EK -= dot(v, Kv)

    EH = 0.0
    for i_atom in range(n_atoms):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    v = s[i_atom, i, j, k]
                    EH -= muB * g * dot(v, H)

    return EJ + EK + EH


@njit
def local_field_at_site(
    s, i_atom, i, j, k, nb_offsets, nb_atom, nb_ijk, nb_J, ni, nj, nk
):

    h = np.zeros(3)
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

        snn = s[nn, ii, jj, kk]
        h += matvec(nb_J[b], snn)

    return h


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

        s_orig = s[i_atom, i, j, k].copy()

        s_cand = np.empty(3)
        s_cand[0] = np.random.normal()
        s_cand[1] = np.random.normal()
        s_cand[2] = np.random.normal()

        norm = np.sqrt(dot(s_cand, s_cand))

        s_cand[0] /= norm
        s_cand[1] /= norm
        s_cand[2] /= norm

        delta = np.empty(3)
        delta[0] = s_cand[0] - s_orig[0]
        delta[1] = s_cand[1] - s_orig[1]
        delta[2] = s_cand[2] - s_orig[2]

        h_eff = local_field_at_site(
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
        dEJ = -dot(delta, h_eff)

        K_ion = K[i_atom]
        s_sum = np.empty(3)
        s_sum[0] = s_cand[0] + s_orig[0]
        s_sum[1] = s_cand[1] + s_orig[1]
        s_sum[2] = s_cand[2] + s_orig[2]

        K_s_sum = matvec(K_ion, s_sum)
        dEK = -dot(delta, K_s_sum)

        dEH = -muB * g * dot(delta, H)

        dE = dEJ + dEK + dEH

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            s[i_atom, i, j, k, 0] = s_cand[0]
            s[i_atom, i, j, k, 1] = s_cand[1]
            s[i_atom, i, j, k, 2] = s_cand[2]
            E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))
