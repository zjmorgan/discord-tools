import numpy as np
from numba import njit, prange


# ----------------------------
# Small vector/matrix helpers
# ----------------------------


@njit
def dot3(a0, a1, a2, b0, b1, b2):
    return a0 * b0 + a1 * b1 + a2 * b2


@njit
def norm3(a0, a1, a2):
    return np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)


@njit
def matvec3(A, v0, v1, v2):
    # returns A @ v as 3 scalars
    return (
        A[0, 0] * v0 + A[0, 1] * v1 + A[0, 2] * v2,
        A[1, 0] * v0 + A[1, 1] * v1 + A[1, 2] * v2,
        A[2, 0] * v0 + A[2, 1] * v1 + A[2, 2] * v2,
    )


@njit
def quad_form3(K, s0, s1, s2):
    # s^T K s
    t0, t1, t2 = matvec3(K, s0, s1, s2)
    return dot3(s0, s1, s2, t0, t1, t2)


@njit
def reflect_about_unit_axis(s0, s1, s2, u0, u1, u2):
    # s' = 2(u·s)u - s, with |u|=1
    udots = dot3(s0, s1, s2, u0, u1, u2)
    return (
        2.0 * udots * u0 - s0,
        2.0 * udots * u1 - s1,
        2.0 * udots * u2 - s2,
    )


@njit
def reflect_about_plane(s0, s1, s2, n0, n1, n2):
    # reflection through plane perpendicular to n:
    # s' = s - 2(n·s)n
    ndots = dot3(n0, n1, n2, s0, s1, s2)
    return (
        s0 - 2.0 * ndots * n0,
        s1 - 2.0 * ndots * n1,
        s2 - 2.0 * ndots * n2,
    )


# ----------------------------
# Indexing helpers
# ----------------------------


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


# ----------------------------
# RNG helpers
# ----------------------------


@njit
def random_unit_vector3():
    v0 = np.random.normal()
    v1 = np.random.normal()
    v2 = np.random.normal()
    n = norm3(v0, v1, v2)
    if n == 0.0:
        return 0.0, 0.0, 1.0
    return v0 / n, v1 / n, v2 / n


@njit
def sample_unit_vector_along_field(f0, f1, f2, beta):
    """
    Sample s on unit sphere from density ∝ exp(beta * f·s).
    Exact heatbath for linear energy E = -f·s.
    """
    fn = norm3(f0, f1, f2)
    if fn < 1e-14:
        return random_unit_vector3()

    h0 = f0 / fn
    h1 = f1 / fn
    h2 = f2 / fn

    alpha = beta * fn

    # sample cos(theta)
    if alpha < 1e-6:
        cos_theta = 2.0 * np.random.rand() - 1.0
    elif alpha > 50.0:
        cos_theta = 1.0 + np.log(np.random.rand()) / alpha
        if cos_theta < -1.0:
            cos_theta = -1.0
    else:
        u = np.random.rand()
        cos_theta = -1.0 + np.log1p(u * np.expm1(2.0 * alpha)) / alpha

    if cos_theta > 1.0:
        cos_theta = 1.0
    if cos_theta < -1.0:
        cos_theta = -1.0

    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * np.random.rand()

    # Orthonormal basis (v, w, h)
    if abs(h0) < 0.9:
        u0, u1, u2 = 1.0, 0.0, 0.0
    else:
        u0, u1, u2 = 0.0, 1.0, 0.0

    dot_uh = dot3(u0, u1, u2, h0, h1, h2)
    v0 = u0 - dot_uh * h0
    v1 = u1 - dot_uh * h1
    v2 = u2 - dot_uh * h2
    vn = norm3(v0, v1, v2)
    v0 /= vn
    v1 /= vn
    v2 /= vn

    w0 = h1 * v2 - h2 * v1
    w1 = h2 * v0 - h0 * v2
    w2 = h0 * v1 - h1 * v0

    c = np.cos(phi)
    s = np.sin(phi)

    s0 = cos_theta * h0 + sin_theta * (c * v0 + s * w0)
    s1 = cos_theta * h1 + sin_theta * (c * v1 + s * w1)
    s2 = cos_theta * h2 + sin_theta * (c * v2 + s * w2)
    return s0, s1, s2


# ----------------------------
# Exchange neighborhood field
# ----------------------------


@njit
def local_exchange_field_at_site(
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
    """
    h_exch = Σ_j J_ij s_j * delta_bonds[neighbor_site]
    (delta_bonds is treated as a *site* mask here, matching your existing code)
    """
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
        if delta_nn <= 0.0:
            continue

        sn0 = s[nn, ii, jj, kk, 0]
        sn1 = s[nn, ii, jj, kk, 1]
        sn2 = s[nn, ii, jj, kk, 2]

        J = nb_J[b]
        Jsn0, Jsn1, Jsn2 = matvec3(J, sn0, sn1, sn2)

        hx += Jsn0 * delta_nn
        hy += Jsn1 * delta_nn
        hz += Jsn2 * delta_nn

    return hx, hy, hz


# ----------------------------
# Local energy difference pieces
# ----------------------------


@njit
def dE_exchange_from_delta(
    delta0,
    delta1,
    delta2,
    hx_exch,
    hy_exch,
    hz_exch,
    S_sq_eff,
    delta_bond_center,
):
    # For a single-site change with neighbors fixed:
    # dEJ = -S_sq_eff * (delta_s · h_exch) * delta_bond_center
    return (
        -S_sq_eff
        * dot3(delta0, delta1, delta2, hx_exch, hy_exch, hz_exch)
        * delta_bond_center
    )


@njit
def dE_zeeman_from_delta(
    delta0, delta1, delta2, H, g_i, muB, S_sq_eff, delta_atom_center
):
    return (
        -muB
        * g_i
        * S_sq_eff
        * dot3(delta0, delta1, delta2, H[0], H[1], H[2])
        * delta_atom_center
    )


@njit
def dE_anisotropy_from_old_new(
    s0, s1, s2, sp0, sp1, sp2, K_i, S_sq_eff, delta_ion_center
):
    # EK = -S_sq_eff * (s^T K s) * delta_ion
    e_old = quad_form3(K_i, s0, s1, s2)
    e_new = quad_form3(K_i, sp0, sp1, sp2)
    return -S_sq_eff * (e_new - e_old) * delta_ion_center


@njit
def linear_field_energy_units(
    hx_exch,
    hy_exch,
    hz_exch,
    H,
    g_i,
    muB,
    S_sq_eff,
    delta_bond_center,
    delta_atom_center,
):
    # f for E_lin = - s · f
    f0 = (
        S_sq_eff * delta_bond_center * hx_exch
        + muB * g_i * S_sq_eff * delta_atom_center * H[0]
    )
    f1 = (
        S_sq_eff * delta_bond_center * hy_exch
        + muB * g_i * S_sq_eff * delta_atom_center * H[1]
    )
    f2 = (
        S_sq_eff * delta_bond_center * hz_exch
        + muB * g_i * S_sq_eff * delta_atom_center * H[2]
    )
    return f0, f1, f2


# ----------------------------
# Total energy (your convention)
# ----------------------------


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
    n_atoms, ni, nj, nk, _ = s.shape

    EJ = 0.0
    for i_atom in prange(n_atoms):
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    delta_bond_center = delta_bonds[i_atom, i, j, k]
                    if delta_bond_center <= 0.0:
                        continue

                    s0 = s[i_atom, i, j, k, 0]
                    s1 = s[i_atom, i, j, k, 1]
                    s2 = s[i_atom, i, j, k, 2]

                    hx, hy, hz = local_exchange_field_at_site(
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
                        * dot3(s0, s1, s2, hx, hy, hz)
                        * delta_bond_center
                    )

    EK = 0.0
    for i_atom in prange(n_atoms):
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        K_i = K[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    delta_ion = delta_ions[i_atom, i, j, k]
                    if delta_ion <= 0.0:
                        continue
                    s0 = s[i_atom, i, j, k, 0]
                    s1 = s[i_atom, i, j, k, 1]
                    s2 = s[i_atom, i, j, k, 2]
                    EK -= S_sq_eff * quad_form3(K_i, s0, s1, s2) * delta_ion

    EH = 0.0
    for i_atom in prange(n_atoms):
        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        g_i = g[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    delta_atom = delta_atoms[i_atom, i, j, k]
                    if delta_atom <= 0.0:
                        continue
                    s0 = s[i_atom, i, j, k, 0]
                    s1 = s[i_atom, i, j, k, 1]
                    s2 = s[i_atom, i, j, k, 2]
                    EH -= (
                        muB
                        * g_i
                        * S_sq_eff
                        * dot3(s0, s1, s2, H[0], H[1], H[2])
                        * delta_atom
                    )

    return EJ + EK + EH


# ----------------------------
# Metropolis update (random-direction)
# ----------------------------


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
    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n_sites = n_atoms * ni * nj * nk

    for _ in range(n_local_sweeps * n_sites):
        flat = np.random.randint(n_sites)
        i_atom, i, j, k = unravel_site(flat, n_atoms, ni, nj, nk)

        delta_atom_center = delta_atoms[i_atom, i, j, k]
        if delta_atom_center <= 0.0:
            continue

        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        g_i = g[i_atom]
        K_i = K[i_atom]

        s0 = s[i_atom, i, j, k, 0]
        s1 = s[i_atom, i, j, k, 1]
        s2 = s[i_atom, i, j, k, 2]

        # propose a new random unit vector
        sp0, sp1, sp2 = random_unit_vector3()

        d0 = sp0 - s0
        d1 = sp1 - s1
        d2 = sp2 - s2

        delta_bond_center = delta_bonds[i_atom, i, j, k]
        hx, hy, hz = local_exchange_field_at_site(
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

        dEJ = 0.0
        if delta_bond_center > 0.0:
            dEJ = dE_exchange_from_delta(
                d0, d1, d2, hx, hy, hz, S_sq_eff, delta_bond_center
            )

        delta_ion_center = delta_ions[i_atom, i, j, k]
        dEK = 0.0
        if delta_ion_center > 0.0:
            dEK = dE_anisotropy_from_old_new(
                s0, s1, s2, sp0, sp1, sp2, K_i, S_sq_eff, delta_ion_center
            )

        dEH = dE_zeeman_from_delta(
            d0, d1, d2, H, g_i, muB, S_sq_eff, delta_atom_center
        )

        dE = dEJ + dEK + dEH

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            s[i_atom, i, j, k, 0] = sp0
            s[i_atom, i, j, k, 1] = sp1
            s[i_atom, i, j, k, 2] = sp2
            E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))


# ----------------------------
# Heatbath (linear exact) + MH correction for anisotropy
# ----------------------------


@njit
def heatbath_heisenberg(
    idx,
    s,
    delta_atoms,
    delta_ions,
    delta_bonds,
    beta,
    E,
    n_heatbath_sweeps,
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
    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape

    for _ in range(n_heatbath_sweeps):
        for i_atom in range(n_atoms):
            S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
            g_i = g[i_atom]
            K_i = K[i_atom]
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        delta_atom_center = delta_atoms[i_atom, i, j, k]
                        if delta_atom_center <= 0.0:
                            continue

                        s0 = s[i_atom, i, j, k, 0]
                        s1 = s[i_atom, i, j, k, 1]
                        s2 = s[i_atom, i, j, k, 2]

                        delta_bond_center = delta_bonds[i_atom, i, j, k]
                        hx, hy, hz = local_exchange_field_at_site(
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

                        # linear field in energy units for E_lin = - s·f
                        f0, f1, f2 = linear_field_energy_units(
                            hx,
                            hy,
                            hz,
                            H,
                            g_i,
                            muB,
                            S_sq_eff,
                            delta_bond_center,
                            delta_atom_center,
                        )

                        sp0, sp1, sp2 = sample_unit_vector_along_field(
                            f0, f1, f2, beta
                        )

                        d0 = sp0 - s0
                        d1 = sp1 - s1
                        d2 = sp2 - s2

                        dEJ = 0.0
                        if delta_bond_center > 0.0:
                            dEJ = dE_exchange_from_delta(
                                d0,
                                d1,
                                d2,
                                hx,
                                hy,
                                hz,
                                S_sq_eff,
                                delta_bond_center,
                            )

                        dEH = dE_zeeman_from_delta(
                            d0,
                            d1,
                            d2,
                            H,
                            g_i,
                            muB,
                            S_sq_eff,
                            delta_atom_center,
                        )

                        # MH correction for anisotropy only
                        delta_ion_center = delta_ions[i_atom, i, j, k]
                        dEK = 0.0
                        if delta_ion_center > 0.0:
                            dEK = dE_anisotropy_from_old_new(
                                s0,
                                s1,
                                s2,
                                sp0,
                                sp1,
                                sp2,
                                K_i,
                                S_sq_eff,
                                delta_ion_center,
                            )
                            if not (
                                dEK <= 0.0
                                or np.random.rand() < np.exp(-beta * dEK)
                            ):
                                continue  # reject

                        # accept
                        s[i_atom, i, j, k, 0] = sp0
                        s[i_atom, i, j, k, 1] = sp1
                        s[i_atom, i, j, k, 2] = sp2
                        E += dEJ + dEH + dEK

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))


# ----------------------------
# Overrelaxation: choose axis (exchange-only or full-linear)
# ----------------------------


@njit
def overrelaxation_heisenberg(
    idx,
    s,
    delta_atoms,
    delta_ions,
    delta_bonds,
    beta,
    E,
    n_overrelaxation_sweeps,
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
    use_full_linear_axis=True,  # <--- key switch
):
    """
    If use_full_linear_axis=True:
      reflect about full linear field (exchange + Zeeman), so ΔEJ+ΔEH≈0 and only EK needs MH correction.

    If False:
      reflect about exchange-only field; then must MH-correct EK + EH (and EJ≈0 anyway).
    """
    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n_sites = n_atoms * ni * nj * nk

    for _ in range(n_overrelaxation_sweeps * n_sites):
        flat = np.random.randint(n_sites)
        i_atom, i, j, k = unravel_site(flat, n_atoms, ni, nj, nk)

        delta_atom_center = delta_atoms[i_atom, i, j, k]
        if delta_atom_center <= 0.0:
            continue

        S_sq_eff = S[i_atom] * (S[i_atom] + 1.0)
        g_i = g[i_atom]
        K_i = K[i_atom]

        s0 = s[i_atom, i, j, k, 0]
        s1 = s[i_atom, i, j, k, 1]
        s2 = s[i_atom, i, j, k, 2]

        delta_bond_center = delta_bonds[i_atom, i, j, k]
        hx, hy, hz = local_exchange_field_at_site(
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

        if use_full_linear_axis:
            f0, f1, f2 = linear_field_energy_units(
                hx,
                hy,
                hz,
                H,
                g_i,
                muB,
                S_sq_eff,
                delta_bond_center,
                delta_atom_center,
            )
            fn = norm3(f0, f1, f2)
            if fn < 1e-10:
                continue
            u0, u1, u2 = f0 / fn, f1 / fn, f2 / fn
        else:
            hn = norm3(hx, hy, hz)
            if hn < 1e-10:
                continue
            u0, u1, u2 = hx / hn, hy / hn, hz / hn

        sp0, sp1, sp2 = reflect_about_unit_axis(s0, s1, s2, u0, u1, u2)

        d0 = sp0 - s0
        d1 = sp1 - s1
        d2 = sp2 - s2

        # Exchange is microcanonical (≈0) if axis is exchange-only.
        # If axis is full-linear, exchange+field are microcanonical.
        dEJ = 0.0
        dEH = 0.0

        if not use_full_linear_axis:
            # still safe to compute; should be ~0 for EJ, but EH generally nonzero
            if delta_bond_center > 0.0:
                dEJ = dE_exchange_from_delta(
                    d0, d1, d2, hx, hy, hz, S_sq_eff, delta_bond_center
                )
            dEH = dE_zeeman_from_delta(
                d0, d1, d2, H, g_i, muB, S_sq_eff, delta_atom_center
            )

        # Anisotropy MH correction always needed if present
        delta_ion_center = delta_ions[i_atom, i, j, k]
        dEK = 0.0
        if delta_ion_center > 0.0:
            dEK = dE_anisotropy_from_old_new(
                s0, s1, s2, sp0, sp1, sp2, K_i, S_sq_eff, delta_ion_center
            )

        dE = dEJ + dEH + dEK

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            s[i_atom, i, j, k, 0] = sp0
            s[i_atom, i, j, k, 1] = sp1
            s[i_atom, i, j, k, 2] = sp2
            E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))


# --------------------
# Wolff cluster update
# --------------------


@njit
def wolff_add_prob(J, n0, n1, n2, si_proj, sj_proj, beta):
    # J_eff = n^T J n
    Jn0 = J[0, 0] * n0 + J[0, 1] * n1 + J[0, 2] * n2
    Jn1 = J[1, 0] * n0 + J[1, 1] * n1 + J[1, 2] * n2
    Jn2 = J[2, 0] * n0 + J[2, 1] * n1 + J[2, 2] * n2
    J_eff = n0 * Jn0 + n1 * Jn1 + n2 * Jn2
    if J_eff <= 0.0:
        return 0.0
    E_bond = 2.0 * J_eff * si_proj * sj_proj
    if E_bond <= 0.0:
        return 0.0
    return 1.0 - np.exp(-beta * E_bond)


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
    """
    Clean Wolff-style cluster:
      - grow cluster using exchange only (projected along random axis n)
      - propose reflection of cluster spins about plane ⟂ n
      - MH accept/reject based on full ΔE (exchange + anisotropy + field)

    """
    np.random.seed(seed)

    n_atoms, ni, nj, nk, _ = s.shape
    n_sites = n_atoms * ni * nj * nk

    # pick an active seed
    seed_flat = -1
    for _ in range(n_sites):
        cand = np.random.randint(n_sites)
        a, i, j, k = unravel_site(cand, n_atoms, ni, nj, nk)
        if delta_atoms[a, i, j, k] > 0.0:
            seed_flat = cand
            break
    if seed_flat < 0:
        return idx, s, E, int(np.random.randint(0, 2**31 - 1))

    n0, n1, n2 = random_unit_vector3()

    # queue + membership
    cluster = np.empty(n_sites, dtype=np.int64)
    in_cluster = np.zeros(n_sites, dtype=np.uint8)

    front = 0
    back = 1
    cluster[0] = seed_flat
    in_cluster[seed_flat] = 1

    # grow
    while front < back:
        flat_i = cluster[front]
        front += 1
        ai, ii, ji, ki = unravel_site(flat_i, n_atoms, ni, nj, nk)

        if delta_atoms[ai, ii, ji, ki] <= 0.0:
            continue
        if delta_bonds[ai, ii, ji, ki] <= 0.0:
            continue

        si0 = s[ai, ii, ji, ki, 0]
        si1 = s[ai, ii, ji, ki, 1]
        si2 = s[ai, ii, ji, ki, 2]
        si_proj = dot3(n0, n1, n2, si0, si1, si2)
        if si_proj == 0.0:
            continue

        start = nb_offsets[ai]
        end = nb_offsets[ai + 1]
        for b in range(start, end):
            aj = nb_atom[b]
            di = nb_ijk[b, 0]
            dj = nb_ijk[b, 1]
            dk = nb_ijk[b, 2]

            ij = (ii + di) % ni
            jj = (ji + dj) % nj
            kj = (ki + dk) % nk

            if delta_atoms[aj, ij, jj, kj] <= 0.0:
                continue
            if delta_bonds[aj, ij, jj, kj] <= 0.0:
                continue

            flat_j = ravel_site(aj, ij, jj, kj, n_atoms, ni, nj, nk)
            if in_cluster[flat_j] == 1:
                continue

            sj0 = s[aj, ij, jj, kj, 0]
            sj1 = s[aj, ij, jj, kj, 1]
            sj2 = s[aj, ij, jj, kj, 2]
            sj_proj = dot3(n0, n1, n2, sj0, sj1, sj2)

            if si_proj * sj_proj <= 0.0:
                continue

            p_add = wolff_add_prob(nb_J[b], n0, n1, n2, si_proj, sj_proj, beta)
            if p_add > 0.0 and np.random.rand() < p_add:
                in_cluster[flat_j] = 1
                cluster[back] = flat_j
                back += 1

    # MH accept/reject:
    # simplest and safest cleanup is to compute ΔE by local sums.
    # (You can also just compute full energy before/after if you prefer.)
    dE = 0.0

    for c in range(back):
        flat_i = cluster[c]
        ai, ii, ji, ki = unravel_site(flat_i, n_atoms, ni, nj, nk)
        if delta_atoms[ai, ii, ji, ki] <= 0.0:
            continue

        S_sq_eff = S[ai] * (S[ai] + 1.0)
        g_i = g[ai]
        K_i = K[ai]

        s0 = s[ai, ii, ji, ki, 0]
        s1 = s[ai, ii, ji, ki, 1]
        s2 = s[ai, ii, ji, ki, 2]

        sp0, sp1, sp2 = reflect_about_plane(s0, s1, s2, n0, n1, n2)

        # single-ion
        delta_ion = delta_ions[ai, ii, ji, ki]
        if delta_ion > 0.0:
            dE += dE_anisotropy_from_old_new(
                s0, s1, s2, sp0, sp1, sp2, K_i, S_sq_eff, delta_ion
            )

        # field
        delta_atom = delta_atoms[ai, ii, ji, ki]
        dE += (
            -muB
            * g_i
            * S_sq_eff
            * (
                dot3(sp0, sp1, sp2, H[0], H[1], H[2])
                - dot3(s0, s1, s2, H[0], H[1], H[2])
            )
            * delta_atom
        )

        # exchange: compute local exchange energy change using old/new neighbor spins depending on membership
        delta_bond_center = delta_bonds[ai, ii, ji, ki]
        if delta_bond_center > 0.0:
            # old local field
            hx_old, hy_old, hz_old = 0.0, 0.0, 0.0
            hx_new, hy_new, hz_new = 0.0, 0.0, 0.0

            start = nb_offsets[ai]
            end = nb_offsets[ai + 1]
            for b in range(start, end):
                aj = nb_atom[b]
                di = nb_ijk[b, 0]
                dj = nb_ijk[b, 1]
                dk = nb_ijk[b, 2]
                ij = (ii + di) % ni
                jj = (ji + dj) % nj
                kj = (ki + dk) % nk

                delta_nn = delta_bonds[aj, ij, jj, kj]
                if delta_nn <= 0.0:
                    continue

                flat_j = ravel_site(aj, ij, jj, kj, n_atoms, ni, nj, nk)

                sj0 = s[aj, ij, jj, kj, 0]
                sj1 = s[aj, ij, jj, kj, 1]
                sj2 = s[aj, ij, jj, kj, 2]

                # neighbor spin in proposed config
                if in_cluster[flat_j] == 1:
                    sj0p, sj1p, sj2p = reflect_about_plane(
                        sj0, sj1, sj2, n0, n1, n2
                    )
                else:
                    sj0p, sj1p, sj2p = sj0, sj1, sj2

                J = nb_J[b]
                Jsj0, Jsj1, Jsj2 = matvec3(J, sj0, sj1, sj2)
                Jsj0p, Jsj1p, Jsj2p = matvec3(J, sj0p, sj1p, sj2p)

                hx_old += Jsj0 * delta_nn
                hy_old += Jsj1 * delta_nn
                hz_old += Jsj2 * delta_nn

                hx_new += Jsj0p * delta_nn
                hy_new += Jsj1p * delta_nn
                hz_new += Jsj2p * delta_nn

            # local exchange energy contribution uses your global convention; for ΔE, a consistent local form is:
            # ΔE_i = -S_sq_eff * delta_bond_center * (s'_i·h_new - s_i·h_old)
            dE += (
                -S_sq_eff
                * delta_bond_center
                * (
                    dot3(sp0, sp1, sp2, hx_new, hy_new, hz_new)
                    - dot3(s0, s1, s2, hx_old, hy_old, hz_old)
                )
            )

    if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
        for c in range(back):
            flat_i = cluster[c]
            ai, ii, ji, ki = unravel_site(flat_i, n_atoms, ni, nj, nk)
            if delta_atoms[ai, ii, ji, ki] <= 0.0:
                continue
            s0 = s[ai, ii, ji, ki, 0]
            s1 = s[ai, ii, ji, ki, 1]
            s2 = s[ai, ii, ji, ki, 2]
            sp0, sp1, sp2 = reflect_about_plane(s0, s1, s2, n0, n1, n2)
            s[ai, ii, ji, ki, 0] = sp0
            s[ai, ii, ji, ki, 1] = sp1
            s[ai, ii, ji, ki, 2] = sp2
        E += dE

    return idx, s, E, int(np.random.randint(0, 2**31 - 1))
