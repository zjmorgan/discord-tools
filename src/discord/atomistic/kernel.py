import numpy as np
from numba import njit, prange


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


@njit(parallel=True)
def total_heisenberg_energy(
    s,
    delta_atoms,
    delta_ions,
    delta_bonds,
    nb_offsets,
    nb_atom,
    nb_ijk,
    nb_J,
    K,
    H,
    g,
    S,
    muB,
):
    """Total Heisenberg energy for a single replica.

    Note: g is currently a 1D array (per-atom scalar g-factor).
    Future: should be 3x3 tensor per atom for anisotropic g-factors.
    """

    n_atoms, ni, nj, nk, _ = s.shape

    EJ = 0.0
    for i_atom in prange(n_atoms):
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    sl0 = s[i_atom, i, j, k, 0]
                    sl1 = s[i_atom, i, j, k, 1]
                    sl2 = s[i_atom, i, j, k, 2]
                    delta = delta_bonds[i_atom, i, j, k]
                    h0, h1, h2 = local_field_at_site(
                        s,
                        delta_bonds,
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
                    EJ -= (
                        0.5
                        * S_sq_eff
                        * (sl0 * h0 + sl1 * h1 + sl2 * h2)
                        * delta
                    )

    EK = 0.0
    for i_atom in prange(n_atoms):
        K_l = K[i_atom]
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    s0 = s[i_atom, i, j, k, 0]
                    s1 = s[i_atom, i, j, k, 1]
                    s2 = s[i_atom, i, j, k, 2]
                    delta = delta_ions[i_atom, i, j, k]
                    Ks0 = K_l[0, 0] * s0 + K_l[0, 1] * s1 + K_l[0, 2] * s2
                    Ks1 = K_l[1, 0] * s0 + K_l[1, 1] * s1 + K_l[1, 2] * s2
                    Ks2 = K_l[2, 0] * s0 + K_l[2, 1] * s1 + K_l[2, 2] * s2
                    EK -= S_sq_eff * (s0 * Ks0 + s1 * Ks1 + s2 * Ks2) * delta

    EH = 0.0
    for i_atom in prange(n_atoms):
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    s0 = s[i_atom, i, j, k, 0]
                    s1 = s[i_atom, i, j, k, 1]
                    s2 = s[i_atom, i, j, k, 2]
                    delta = delta_atoms[i_atom, i, j, k]
                    EH -= (
                        muB
                        * g[i_atom]
                        * S_sq_eff
                        * (s0 * H[0] + s1 * H[1] + s2 * H[2])
                        * delta
                    )

    return EJ + EK + EH


@njit
def local_field_at_site(
    s,
    delta_bonds,
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

        delta_nn = delta_bonds[nn, ii, jj, kk]

        snn0 = s[nn, ii, jj, kk, 0]
        snn1 = s[nn, ii, jj, kk, 1]
        snn2 = s[nn, ii, jj, kk, 2]

        J = nb_J[b]
        hx += (J[0, 0] * snn0 + J[0, 1] * snn1 + J[0, 2] * snn2) * delta_nn
        hy += (J[1, 0] * snn0 + J[1, 1] * snn1 + J[1, 2] * snn2) * delta_nn
        hz += (J[2, 0] * snn0 + J[2, 1] * snn1 + J[2, 2] * snn2) * delta_nn

    return hx, hy, hz


@njit
def metropolis_heisenberg(
    idx,
    s,
    delta_atoms,
    delta_ions,
    delta_bonds,
    beta,
    E,
    n_local_sweeps,
    nb_offsets,
    nb_atom,
    nb_ijk,
    nb_J,
    K,
    H,
    g,
    S,
    muB,
    seed,
):
    """Local Metropolis sweeps for one replica.

    Proposes random unit-vector spins, computes ``dE`` using the same
    ``S(S+1)`` scaling as :func:`total_heisenberg_energy`, and updates
    the configuration and running energy ``E`` in place.
    """

    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n = n_atoms * ni * nj * nk

    for _ in range(n_local_sweeps * n):
        flat_idx = np.random.randint(n)
        i_atom, i, j, k = unravel_site(flat_idx, n_atoms, ni, nj, nk)

        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)

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

        delta_bond = delta_bonds[i_atom, i, j, k]

        h0, h1, h2 = local_field_at_site(
            s,
            delta_bonds,
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
        dEJ = (
            -S_sq_eff * (delta0 * h0 + delta1 * h1 + delta2 * h2) * delta_bond
        )

        delta_ion = delta_ions[i_atom, i, j, k]

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

        dEK = (
            -S_sq_eff
            * (delta0 * K_s_sum0 + delta1 * K_s_sum1 + delta2 * K_s_sum2)
            * delta_ion
        )

        delta_atom = delta_atoms[i_atom, i, j, k]

        dEH = (
            -muB
            * g[i_atom]
            * S_sq_eff
            * (delta0 * H[0] + delta1 * H[1] + delta2 * H[2])
        ) * delta_atom

        dE = dEJ + dEK + dEH

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            s[i_atom, i, j, k, 0] = s_cand0
            s[i_atom, i, j, k, 1] = s_cand1
            s[i_atom, i, j, k, 2] = s_cand2
            E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))


@njit
def wolff_heisenberg(
    idx,
    s,
    delta_atoms,
    delta_ions,
    delta_bonds,
    beta,
    E,
    nb_offsets,
    nb_atom,
    nb_ijk,
    nb_J,
    K,
    H,
    g,
    S,
    muB,
    seed,
):
    """Single Wolff-style cluster update for one replica.

    Cluster growth uses the exchange couplings only (``nb_J``) with a
    random reflection axis. After flipping the cluster, the full
    Heisenberg energy (exchange + anisotropy + Zeeman) is recomputed
    using :func:`total_heisenberg_energy` and returned.
    """

    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n_sites = n_atoms * ni * nj * nk

    flat_seed = -1
    for _ in range(n_sites):
        cand = np.random.randint(n_sites)
        i_atom0, i0, j0, k0 = unravel_site(cand, n_atoms, ni, nj, nk)
        if delta_atoms[i_atom0, i0, j0, k0] > 0.0:
            flat_seed = cand
            break

    if flat_seed < 0:
        return idx, s, E, int(np.random.randint(0, 2**31 - 1))

    n_vec = random_unit_vector()

    cluster_sites = np.empty(n_sites, dtype=np.int64)
    in_cluster = np.zeros(n_sites, dtype=np.uint8)

    front = 0
    back = 1
    cluster_sites[0] = flat_seed
    in_cluster[flat_seed] = 1

    while front < back:
        flat_i = cluster_sites[front]
        front += 1

        i_atom, i, j, k = unravel_site(flat_i, n_atoms, ni, nj, nk)

        if delta_atoms[i_atom, i, j, k] <= 0.0:
            continue

        if delta_bonds[i_atom, i, j, k] <= 0.0:
            continue

        s0 = s[i_atom, i, j, k, 0]
        s1 = s[i_atom, i, j, k, 1]
        s2 = s[i_atom, i, j, k, 2]
        n_dot_si = s0 * n_vec[0] + s1 * n_vec[1] + s2 * n_vec[2]

        if n_dot_si == 0.0:
            continue

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

            if delta_atoms[nn, ii, jj, kk] <= 0.0:
                continue

            flat_j = ravel_site(nn, ii, jj, kk, n_atoms, ni, nj, nk)
            if in_cluster[flat_j] == 1:
                continue

            if delta_bonds[nn, ii, jj, kk] <= 0.0:
                continue

            s0j = s[nn, ii, jj, kk, 0]
            s1j = s[nn, ii, jj, kk, 1]
            s2j = s[nn, ii, jj, kk, 2]

            n_dot_sj = s0j * n_vec[0] + s1j * n_vec[1] + s2j * n_vec[2]

            if n_dot_si * n_dot_sj <= 0.0:
                continue

            J = nb_J[b]
            Jn = matvec(J, n_vec)
            J_eff = n_vec[0] * Jn[0] + n_vec[1] * Jn[1] + n_vec[2] * Jn[2]

            if J_eff <= 0.0:
                continue

            E_bond = 2.0 * J_eff * n_dot_si * n_dot_sj

            if E_bond <= 0.0:
                continue

            p_add = 1.0 - np.exp(-beta * E_bond)

            if np.random.rand() < p_add:
                in_cluster[flat_j] = 1
                cluster_sites[back] = flat_j
                back += 1

    dEJ = 0.0
    dEK = 0.0
    dEH = 0.0

    def reflect(v0, v1, v2, nx, ny, nz):
        dot_nv = v0 * nx + v1 * ny + v2 * nz
        return (
            v0 - 2.0 * dot_nv * nx,
            v1 - 2.0 * dot_nv * ny,
            v2 - 2.0 * dot_nv * nz,
        )

    nx = n_vec[0]
    ny = n_vec[1]
    nz = n_vec[2]

    for idx_c in range(back):
        flat_i = cluster_sites[idx_c]
        i_atom, i, j, k = unravel_site(flat_i, n_atoms, ni, nj, nk)

        if delta_atoms[i_atom, i, j, k] <= 0.0:
            continue

        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        delta_bond_i = delta_bonds[i_atom, i, j, k]

        s0 = s[i_atom, i, j, k, 0]
        s1 = s[i_atom, i, j, k, 1]
        s2 = s[i_atom, i, j, k, 2]

        s0p, s1p, s2p = reflect(s0, s1, s2, nx, ny, nz)

        hx_old = 0.0
        hy_old = 0.0
        hz_old = 0.0
        hx_new = 0.0
        hy_new = 0.0
        hz_new = 0.0

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

            delta_nn = delta_bonds[nn, ii, jj, kk]
            if delta_nn <= 0.0:
                continue

            flat_j = ravel_site(nn, ii, jj, kk, n_atoms, ni, nj, nk)

            sj0 = s[nn, ii, jj, kk, 0]
            sj1 = s[nn, ii, jj, kk, 1]
            sj2 = s[nn, ii, jj, kk, 2]

            if in_cluster[flat_j] == 1:
                sj0p, sj1p, sj2p = reflect(sj0, sj1, sj2, nx, ny, nz)
            else:
                sj0p, sj1p, sj2p = sj0, sj1, sj2

            J = nb_J[b]

            Jsj0_old = J[0, 0] * sj0 + J[0, 1] * sj1 + J[0, 2] * sj2
            Jsj1_old = J[1, 0] * sj0 + J[1, 1] * sj1 + J[1, 2] * sj2
            Jsj2_old = J[2, 0] * sj0 + J[2, 1] * sj1 + J[2, 2] * sj2

            hx_old += Jsj0_old * delta_nn
            hy_old += Jsj1_old * delta_nn
            hz_old += Jsj2_old * delta_nn

            Jsj0_new = J[0, 0] * sj0p + J[0, 1] * sj1p + J[0, 2] * sj2p
            Jsj1_new = J[1, 0] * sj0p + J[1, 1] * sj1p + J[1, 2] * sj2p
            Jsj2_new = J[2, 0] * sj0p + J[2, 1] * sj1p + J[2, 2] * sj2p

            hx_new += Jsj0_new * delta_nn
            hy_new += Jsj1_new * delta_nn
            hz_new += Jsj2_new * delta_nn

        if delta_bond_i > 0.0:
            dEJ_site = (
                -0.5
                * S_sq_eff
                * (
                    (s0p * hx_new + s1p * hy_new + s2p * hz_new)
                    - (s0 * hx_old + s1 * hy_old + s2 * hz_old)
                )
                * delta_bond_i
            )
            dEJ += dEJ_site

        K_ion = K[i_atom]
        delta_ion = delta_ions[i_atom, i, j, k]

        Ks0_old = K_ion[0, 0] * s0 + K_ion[0, 1] * s1 + K_ion[0, 2] * s2
        Ks1_old = K_ion[1, 0] * s0 + K_ion[1, 1] * s1 + K_ion[1, 2] * s2
        Ks2_old = K_ion[2, 0] * s0 + K_ion[2, 1] * s1 + K_ion[2, 2] * s2

        Ks0_new = K_ion[0, 0] * s0p + K_ion[0, 1] * s1p + K_ion[0, 2] * s2p
        Ks1_new = K_ion[1, 0] * s0p + K_ion[1, 1] * s1p + K_ion[1, 2] * s2p
        Ks2_new = K_ion[2, 0] * s0p + K_ion[2, 1] * s1p + K_ion[2, 2] * s2p

        dEK_site = (
            -S_sq_eff
            * (
                (s0p * Ks0_new + s1p * Ks1_new + s2p * Ks2_new)
                - (s0 * Ks0_old + s1 * Ks1_old + s2 * Ks2_old)
            )
            * delta_ion
        )
        dEK += dEK_site

        delta_atom = delta_atoms[i_atom, i, j, k]

        dEH_site = (
            -muB
            * g[i_atom]
            * S_sq_eff
            * (
                (s0p * H[0] + s1p * H[1] + s2p * H[2])
                - (s0 * H[0] + s1 * H[1] + s2 * H[2])
            )
            * delta_atom
        )
        dEH += dEH_site

    dE = dEJ + dEK + dEH

    if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
        for idx_c in range(back):
            flat_i = cluster_sites[idx_c]
            i_atom, i, j, k = unravel_site(flat_i, n_atoms, ni, nj, nk)

            if delta_atoms[i_atom, i, j, k] <= 0.0:
                continue

            s0 = s[i_atom, i, j, k, 0]
            s1 = s[i_atom, i, j, k, 1]
            s2 = s[i_atom, i, j, k, 2]

            s0p, s1p, s2p = reflect(s0, s1, s2, nx, ny, nz)

            s[i_atom, i, j, k, 0] = s0p
            s[i_atom, i, j, k, 1] = s1p
            s[i_atom, i, j, k, 2] = s2p

        E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))
