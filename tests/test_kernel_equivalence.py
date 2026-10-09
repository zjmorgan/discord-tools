"""
Distribution-equivalence tests for the Monte Carlo update kernels.

Energy bookkeeping tests only check that the tracked energy matches the
configuration; they cannot detect a kernel that samples the wrong
distribution. Here each kernel (or kernel combination) is run as a Markov
chain at fixed temperature and its averages are compared against

- an exact Boltzmann average obtained by quadrature (two-spin system), and
- a Metropolis-only reference chain (small simple-cubic lattice).

The kernels are called directly on hand-built neighbor tables so these tests
do not depend on mantid or :class:`discord.material.Crystal`.
"""

import numpy as np
import pytest

from discord.atomistic import kernel
from discord.atomistic.simulation import wolff_axis_projectors

# The kernels take muB as an argument; unit value keeps the field scale simple.
MUB = 1.0


def build_system(shape, J, K, H, S, bonds):
    """
    Single-atom basis with the given neighbor offsets, all sharing coupling J.
    """
    n_bonds = len(bonds)
    nb_offsets = np.array([0, n_bonds], dtype=np.int64)
    nb_atom = np.zeros(n_bonds, dtype=np.int64)
    nb_ijk = np.array(bonds, dtype=np.int64)
    nb_J = np.repeat(np.asarray(J, float)[None], n_bonds, axis=0)
    deltas = np.ones((1, *shape))
    return dict(
        delta_atoms=deltas,
        delta_ions=deltas.copy(),
        delta_bonds=deltas.copy(),
        nb_offsets=nb_offsets,
        nb_atom=nb_atom,
        nb_ijk=nb_ijk,
        nb_J=nb_J,
        K=np.asarray(K, float)[None],
        H=np.asarray(H, float),
        g=np.array([2.0]),
        S=np.array([S]),
    )


def total_energy(s, sys):
    return kernel.total_heisenberg_energy(
        s,
        sys["delta_atoms"],
        sys["delta_ions"],
        sys["delta_bonds"],
        sys["nb_offsets"],
        sys["nb_atom"],
        sys["nb_ijk"],
        sys["nb_J"],
        sys["K"],
        sys["H"],
        sys["g"],
        sys["S"],
        MUB,
    )


def new_seed(rng):
    """
    Fresh kernel seed per call. Chaining the seed each kernel returns is a
    map on 31-bit integers, which falls into a cycle after ~1e4 calls and
    then replays the same random stream.
    """
    return int(rng.integers(2**31 - 1))


def run_chain(sys, schedule, beta, n_steps, n_thermal, seed):
    """
    Run a chain where each step applies ``schedule``, a list of
    ``(method, n)`` pairs, and record the energy and squared magnetization
    per site after every step.
    """
    rng = np.random.default_rng(seed)
    shape = sys["delta_atoms"].shape
    s = rng.normal(size=(*shape, 3))
    s /= np.linalg.norm(s, axis=-1)[..., None]
    E = total_energy(s, sys)
    n_sites = s[..., 0].size

    common = (
        sys["nb_offsets"],
        sys["nb_atom"],
        sys["nb_ijk"],
        sys["nb_J"],
        sys["K"],
        sys["H"],
        sys["g"],
        sys["S"],
        MUB,
    )
    deltas = (sys["delta_atoms"], sys["delta_ions"], sys["delta_bonds"])

    E_series = np.empty(n_steps)
    m2_series = np.empty(n_steps)
    # "wolff_aligned" draws axes within eigenspaces of K (random if isotropic)
    projectors = {
        "wolff": wolff_axis_projectors(sys["K"], "random"),
        "wolff_aligned": wolff_axis_projectors(sys["K"], "mixed")[:-1]
        if np.ptp(np.linalg.eigvalsh(sys["K"][0])) > 0
        else wolff_axis_projectors(sys["K"], "random"),
    }

    for step in range(n_thermal + n_steps):
        for method, n in schedule:
            if method in projectors:
                for _ in range(n):
                    _, s, E, _, _ = kernel.wolff_heisenberg(
                        0,
                        s,
                        *deltas,
                        beta,
                        E,
                        *common,
                        new_seed(rng),
                        projectors[method],
                    )
            else:
                func = {
                    "metropolis": kernel.metropolis_heisenberg,
                    "heatbath": kernel.heatbath_heisenberg,
                    "overrelaxation": kernel.overrelaxation_heisenberg,
                }[method]
                _, s, E, _, _ = func(
                    0, s, *deltas, beta, E, n, *common, new_seed(rng)
                )
        if step >= n_thermal:
            i = step - n_thermal
            E_series[i] = E / n_sites
            m = s.reshape(-1, 3).mean(axis=0)
            m2_series[i] = m @ m

    assert np.isclose(E, total_energy(s, sys), rtol=1e-8, atol=1e-8)
    return E_series, m2_series


def mean_and_error(x, n_bins=50):
    """Mean and binned standard error (robust to autocorrelation)."""
    n = len(x) // n_bins * n_bins
    bins = x[:n].reshape(n_bins, -1).mean(axis=1)
    return bins.mean(), bins.std(ddof=1) / np.sqrt(n_bins)


SCHEDULES = {
    "metropolis": [("metropolis", 1)],
    "heatbath": [("heatbath", 1)],
    "overrelaxation+metropolis": [("overrelaxation", 3), ("metropolis", 1)],
    "wolff": [("wolff", 1)],
    "wolff+metropolis": [("wolff", 4), ("metropolis", 1)],
    "wolff_aligned+metropolis": [("wolff_aligned", 4), ("metropolis", 1)],
}

# Parameter sets: isotropic ferromagnet with no single-ion terms, an
# easy-axis antiferromagnet (Wolff clusters grow along anti-aligned bonds),
# and an anisotropic case (exchange anisotropy + easy axis + field) that
# exercises the MH corrections in every kernel.
J0 = 0.1
CASES = {
    "isotropic": dict(
        J=J0 * np.eye(3), K=np.zeros((3, 3)), H=np.zeros(3), S=2.5
    ),
    "antiferromagnetic": dict(
        J=-J0 * np.eye(3),
        K=np.diag([0.0, 0.0, 0.02]),
        H=np.zeros(3),
        S=2.5,
    ),
    "anisotropic": dict(
        J=J0 * np.diag([1.0, 1.0, 1.4]),
        K=np.diag([0.0, 0.0, 0.02]),
        H=np.array([0.0, 0.0, 0.01]),
        S=2.5,
    ),
}


# --------------------------------------------------------------------------
# Exact reference: two spins coupled by two periodic bonds
# --------------------------------------------------------------------------


def dimer_exact_energy(case, beta, n_quad=48):
    """
    Exact <E>/site for the 2x1x1 periodic dimer by quadrature over both
    spheres (Gauss-Legendre in cos(theta), uniform in phi).
    """
    S_eff = case["S"] * (case["S"] + 1.0)
    J, K, H = case["J"], case["K"], case["H"]

    u, w = np.polynomial.legendre.leggauss(n_quad)
    phi = 2.0 * np.pi * (np.arange(n_quad) + 0.5) / n_quad
    U, PHI = np.meshgrid(u, phi, indexing="ij")
    st = np.sqrt(1.0 - U**2)
    v = np.stack(
        [st * np.cos(PHI), st * np.sin(PHI), U], axis=-1
    ).reshape(-1, 3)
    wt = np.repeat(w, n_quad)

    # Both periodic bonds connect the same pair: E_J = -S_eff (s0.J s1 + s1.J s0)
    Jsym = J + J.T
    EJ = -S_eff * np.einsum("ai,ij,bj->ab", v, Jsym, v)
    E1 = -S_eff * np.einsum("ai,ij,aj->a", v, K, v) - MUB * 2.0 * np.sqrt(S_eff) * (
        v @ H
    )
    E = EJ + E1[:, None] + E1[None, :]

    weight = wt[:, None] * wt[None, :] * np.exp(-beta * (E - E.min()))
    return (weight * E).sum() / weight.sum() / 2.0


@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("case_name", list(CASES))
def test_dimer_matches_exact(case_name, schedule):
    case = CASES[case_name]
    beta = 1.5
    sys = build_system(
        (2, 1, 1), bonds=[(1, 0, 0), (-1, 0, 0)], **case
    )

    E_series, _ = run_chain(
        sys, SCHEDULES[schedule], beta, n_steps=100_000, n_thermal=1000,
        seed=1,
    )
    E_mc, err = mean_and_error(E_series)
    E_exact = dimer_exact_energy(case, beta)

    assert abs(E_mc - E_exact) < 4.0 * err, (
        f"{schedule}: <E>={E_mc:.5f} ± {err:.5f}, exact {E_exact:.5f} "
        f"({(E_mc - E_exact) / err:+.1f} sigma)"
    )


# --------------------------------------------------------------------------
# Lattice: compare each schedule against Metropolis near T_c
# --------------------------------------------------------------------------

SC_BONDS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]


@pytest.fixture(scope="module")
def lattice_reference():
    cache = {}

    def get(case_name, beta):
        key = (case_name, beta)
        if key not in cache:
            sys = build_system((4, 4, 4), bonds=SC_BONDS, **CASES[case_name])
            E, m2 = run_chain(
                sys, [("metropolis", 1)], beta, n_steps=40_000,
                n_thermal=2000, seed=2,
            )
            cache[key] = (mean_and_error(E), mean_and_error(m2))
        return cache[key]

    return get


@pytest.mark.parametrize(
    "schedule",
    [
        "heatbath",
        "overrelaxation+metropolis",
        "wolff",
        "wolff+metropolis",
        "wolff_aligned+metropolis",
    ],
)
@pytest.mark.parametrize("case_name", list(CASES))
def test_lattice_matches_metropolis(case_name, schedule, lattice_reference):
    # S_eff*J0 = 0.875, so T_c ~ 1.443*0.875 ~ 1.26 (beta_c ~ 0.79) for the
    # isotropic 3D Heisenberg ferromagnet; sit just above it.
    beta = 0.75
    (E_ref, E_ref_err), (m2_ref, m2_ref_err) = lattice_reference(
        case_name, beta
    )

    sys = build_system((4, 4, 4), bonds=SC_BONDS, **CASES[case_name])
    E, m2 = run_chain(
        sys, SCHEDULES[schedule], beta, n_steps=20_000, n_thermal=2000,
        seed=3,
    )
    (E_mc, E_err), (m2_mc, m2_err) = mean_and_error(E), mean_and_error(m2)

    sig_E = np.hypot(E_err, E_ref_err)
    sig_m2 = np.hypot(m2_err, m2_ref_err)
    assert abs(E_mc - E_ref) < 4.0 * sig_E, (
        f"{schedule}: <E>={E_mc:.5f}, ref {E_ref:.5f} "
        f"({(E_mc - E_ref) / sig_E:+.1f} sigma)"
    )
    assert abs(m2_mc - m2_ref) < 4.0 * sig_m2, (
        f"{schedule}: <m^2>={m2_mc:.5f}, ref {m2_ref:.5f} "
        f"({(m2_mc - m2_ref) / sig_m2:+.1f} sigma)"
    )


def test_wolff_clusters_grow_for_antiferromagnet():
    # On the bipartite simple-cubic lattice the antiferromagnet maps onto the
    # ferromagnet by a sublattice flip, so cluster sizes must match; with
    # isotropic exchange every cluster flip is accepted.
    beta = 0.75
    stats = {}
    for name, sign in [("ferro", 1.0), ("antiferro", -1.0)]:
        sys = build_system(
            (4, 4, 4),
            J=sign * J0 * np.eye(3),
            K=np.zeros((3, 3)),
            H=np.zeros(3),
            S=2.5,
            bonds=SC_BONDS,
        )
        rng = np.random.default_rng(5)
        s = rng.normal(size=(1, 4, 4, 4, 3))
        s /= np.linalg.norm(s, axis=-1)[..., None]
        E = total_energy(s, sys)
        common = (
            sys["nb_offsets"],
            sys["nb_atom"],
            sys["nb_ijk"],
            sys["nb_J"],
            sys["K"],
            sys["H"],
            sys["g"],
            sys["S"],
            MUB,
        )
        deltas = (sys["delta_atoms"], sys["delta_ions"], sys["delta_bonds"])
        accepted, attempted = 0, 0
        for step in range(3000):
            _, s, E, n_acc, n_try = kernel.wolff_heisenberg(
                0, s, *deltas, beta, E, *common, new_seed(rng), np.eye(3)[None]
            )
            if step >= 500:
                accepted += n_acc
                attempted += n_try
        stats[name] = (attempted / 2500, accepted / attempted)

    (size_f, acc_f), (size_af, acc_af) = stats["ferro"], stats["antiferro"]
    assert size_f > 5 and size_af > 5
    assert abs(size_af - size_f) < 0.2 * size_f
    assert acc_f == 1.0 and acc_af == 1.0


def wolff_cluster_stats(sys, beta, projectors, n_steps=3000, n_thermal=500):
    """Mean cluster size and acceptance of pure Wolff dynamics."""
    rng = np.random.default_rng(6)
    s = rng.normal(size=(*sys["delta_atoms"].shape, 3))
    s /= np.linalg.norm(s, axis=-1)[..., None]
    E = total_energy(s, sys)
    common = (
        sys["nb_offsets"],
        sys["nb_atom"],
        sys["nb_ijk"],
        sys["nb_J"],
        sys["K"],
        sys["H"],
        sys["g"],
        sys["S"],
        MUB,
    )
    deltas = (sys["delta_atoms"], sys["delta_ions"], sys["delta_bonds"])
    accepted, attempted = 0, 0
    for step in range(n_steps):
        _, s, E, n_acc, n_try = kernel.wolff_heisenberg(
            0, s, *deltas, beta, E, *common, new_seed(rng), projectors
        )
        if step >= n_thermal:
            accepted += n_acc
            attempted += n_try
    return attempted / (n_steps - n_thermal), accepted / max(attempted, 1)


def test_anisotropy_aligned_axes_accept_large_clusters():
    # Easy-axis antiferromagnet in its ordered phase: random axes tilt large
    # clusters off the easy axis and are rejected; axes in eigenspaces of K
    # leave the anisotropy energy unchanged, so with isotropic exchange and
    # no field every cluster flip is accepted.
    sys = build_system(
        (4, 4, 4),
        J=-J0 * np.eye(3),
        K=np.diag([0.0, 0.0, 0.05]),
        H=np.zeros(3),
        S=2.5,
        bonds=SC_BONDS,
    )
    beta = 1.5
    size_r, acc_r = wolff_cluster_stats(
        sys, beta, wolff_axis_projectors(sys["K"], "random")
    )
    size_a, acc_a = wolff_cluster_stats(
        sys, beta, wolff_axis_projectors(sys["K"], "anisotropy")
    )
    assert acc_a == 1.0
    assert acc_r < 0.5
    assert size_a > 10


def test_wolff_axis_projectors():
    # Uniaxial: the easy axis and the (degenerate) perpendicular plane.
    P = wolff_axis_projectors(np.diag([0.0, 0.0, 0.1])[None], "anisotropy")
    ranks = sorted(int(round(np.trace(p))) for p in P)
    assert ranks == [1, 2]
    assert any(np.allclose(p, np.diag([0, 0, 1])) for p in P)

    # Rotated orthorhombic frame shared by two sites: three axes.
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    K = np.stack([Q @ np.diag(d) @ Q.T for d in ([1, 2, 3], [0, 5, 1])])
    P = wolff_axis_projectors(K, "anisotropy")
    assert len(P) == 3
    for p in P:
        for Ki in K:
            assert np.allclose(p @ Ki, Ki @ p)
    assert np.allclose(P.sum(axis=0), np.eye(3))

    # "mixed" adds uniformly random axes.
    assert len(wolff_axis_projectors(K, "mixed")) == 4

    # Isotropic K: "random" and "mixed" work, "anisotropy" is rejected.
    K_iso = np.zeros((1, 3, 3))
    assert np.allclose(wolff_axis_projectors(K_iso, "mixed"), np.eye(3))
    with pytest.raises(ValueError, match="anisotropic"):
        wolff_axis_projectors(K_iso, "anisotropy")

    # Uniaxial sites with different easy axes share the x, y, z frame.
    K_xz = np.stack([np.diag([0, 0, 1.0]), np.diag([1.0, 0, 0])])
    P = wolff_axis_projectors(K_xz, "anisotropy")
    assert len(P) == 3

    # Sites with different principal axes are rejected.
    c = np.cos(np.pi / 6)
    s = np.sin(np.pi / 6)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    K_bad = np.stack([np.diag([0, 0, 1.0]), R @ np.diag([0, 0, 1.0]) @ R.T])
    with pytest.raises(ValueError, match="principal axes"):
        wolff_axis_projectors(K_bad, "anisotropy")
