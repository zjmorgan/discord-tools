import numpy as np
import pytest

from discord.dipolar import ewald_dipolar_tensors, dipolar_energy
from discord.parameters.constants import D, muB

CUBIC = {
    "sc": [[0, 0, 0]],
    "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]],
    "fcc": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
}


@pytest.mark.parametrize("lattice", list(CUBIC))
def test_uniform_cubic_lorentz_cavity(lattice):
    # Uniformly magnetized cubic lattice: the dipolar field inside a Lorentz
    # sphere vanishes by symmetry, so per moment E = 0 for a vacuum sphere
    # and E = -(2 pi / 3) mu^2 rho with tin-foil boundaries.
    a = 2.0
    xyz = np.array(CUBIC[lattice], dtype=float)
    N = (4, 4, 4)
    rho = len(xyz) / a**3
    mu = np.zeros((len(xyz), *N, 3))
    mu[..., 2] = 1.0
    n = mu[..., 0].size

    T = ewald_dipolar_tensors(a * np.eye(3), xyz, N, "tinfoil")
    assert dipolar_energy(T, mu) / n == pytest.approx(
        -2.0 * np.pi / 3.0 * rho, rel=1e-5
    )

    T = ewald_dipolar_tensors(a * np.eye(3), xyz, N, "vacuum")
    assert abs(dipolar_energy(T, mu) / n) < 1e-5 * rho


def triclinic():
    A = np.array([[3.0, 0.7, 0.4], [0.0, 2.6, 0.5], [0.0, 0.0, 2.9]])
    xyz = np.random.default_rng(0).random((3, 3))
    return A, xyz, (3, 4, 2)


def test_independent_of_splitting_parameter():
    A, xyz, N = triclinic()
    tensors = [
        ewald_dipolar_tensors(A, xyz, N, alpha=alpha, precision=5.0)
        for alpha in (0.35, 0.55, 0.8)
    ]
    scale = np.abs(tensors[0]).max()
    for T in tensors[1:]:
        assert np.abs(T - tensors[0]).max() < 1e-8 * scale


def test_tensor_symmetries():
    A, xyz, N = triclinic()
    T = ewald_dipolar_tensors(A, xyz, N)
    # each tensor is symmetric, and T_ab(m) = T_ba(-m)
    assert np.allclose(T, np.swapaxes(T, -1, -2))
    T_ba = np.swapaxes(T, 0, 1)
    T_ba_neg = np.roll(np.flip(T_ba, axis=(2, 3, 4)), 1, axis=(2, 3, 4))
    assert np.allclose(T, T_ba_neg)
    # The bare dipole tensor is traceless, so the vacuum-sphere lattice sum is
    # too; tin-foil removes the surface term (4 pi / 3 V) I for every pair.
    n_atoms = len(xyz)
    n_sites = n_atoms * np.prod(N)
    V = abs(np.linalg.det(A)) * np.prod(N)
    trace = np.trace(T, axis1=-2, axis2=-1).sum()
    assert trace == pytest.approx(-4.0 * np.pi * n_atoms * n_sites / V, rel=1e-4)


def test_vacuum_matches_direct_spherical_sum():
    # The conditionally convergent direct sum over growing spheres of image
    # cells converges to the vacuum-boundary Ewald result.
    rng = np.random.default_rng(1)
    A = np.array([[1.0, 0.2, 0.1], [0.0, 1.1, 0.15], [0.0, 0.0, 0.9]])
    xyz = np.array([[0.0, 0.0, 0.0], [0.3, 0.6, 0.45]])
    N = (2, 2, 2)
    mu = rng.normal(size=(2, *N, 3))
    E_ewald = dipolar_energy(
        ewald_dipolar_tensors(A, xyz, N, "vacuum", precision=5.0), mu
    )

    cells = np.stack(
        np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    pos = np.array([[A @ (x + c) for c in cells] for x in xyz]).reshape(-1, 3)
    m = mu.reshape(-1, 3)
    L = A * np.array(N)[None, :]

    R = 40
    n = np.stack(
        np.meshgrid(*[np.arange(-R, R + 1)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    shifts = n @ L.T
    radius = R * np.linalg.norm(L, axis=0).min()
    shifts = shifts[np.linalg.norm(shifts, axis=1) <= radius]

    E = 0.0
    for i in range(len(pos)):
        r = pos[None, :, :] - pos[i] + shifts[:, None, :]
        r2 = (r**2).sum(-1)
        ok = r2 > 1e-12
        r2 = np.where(ok, r2, 1.0)
        e = (m[i] @ m.T)[None, :] / r2**1.5 - 3.0 * np.einsum(
            "sjk,k->sj", r, m[i]
        ) * np.einsum("sjk,jk->sj", r, m) / r2**2.5
        E += 0.5 * np.where(ok, e, 0.0).sum()

    assert E == pytest.approx(E_ewald, rel=1e-3)


# --------------------------------------------------------------------------
# Crystal integration (requires mantid)
# --------------------------------------------------------------------------


def mnf2(L=3):
    pytest.importorskip("mantid")
    from discord.material import Crystal

    crystal = Crystal(
        [4.873, 4.873, 3.130, 90, 90, 90],
        "P 42/m n m",
        [["Mn", 0, 0.0, 0.0]],
        (L, L, L),
        S=2.5,
    )
    crystal.generate_bonds(d_cut=4.8)
    K, J = crystal.initialize_magnetic_parameters()
    K[:] = np.diag([0, 0, 0.091])
    J[0] = 0.028 * np.eye(3)
    J[1] = -0.152 * np.eye(3)
    crystal.assign_magnetic_parameters(K, J)
    return crystal


def kernel_energy(crystal, s):
    from discord.atomistic import kernel

    nb_J, K, H = crystal.get_magnetic_parameters()
    nb_offsets, nb_atom, nb_ijk = crystal.get_compressed_sparse_row()
    return kernel.total_heisenberg_energy(
        s,
        *crystal.get_delta_arrays(),
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        crystal.get_g_factors(),
        crystal.get_spin_quantum_numbers(),
        muB,
    )


@pytest.mark.parametrize("boundary", ["tinfoil", "vacuum"])
def test_crystal_dipolar_energy_matches_ewald(boundary):
    crystal = mnf2()
    n_bonds = len(crystal.nb_atom)
    s = crystal.get_spin_vectors()[0]

    E_exchange = kernel_energy(crystal, s)

    crystal.add_dipolar_interactions(boundary)
    E_total = kernel_energy(crystal, s)

    # independent evaluation: (mu0/4pi) 1/2 sum mu.T.mu with mu in muB
    T = ewald_dipolar_tensors(
        crystal.A, crystal.xyz, crystal.N, boundary
    )
    moment = crystal.g * np.sqrt(crystal.S * (crystal.S + 1))
    mu = np.einsum("a,ij,a...j->a...i", moment, crystal.C, s)
    E_dipolar = D * dipolar_energy(T, mu)

    assert E_total - E_exchange == pytest.approx(E_dipolar, rel=1e-8)

    # every site couples to all others; removal restores the exchange bonds
    n_sites = crystal.get_total_sites()
    assert len(crystal.nb_atom) == n_bonds + crystal.n_atoms * (n_sites - 1)
    crystal.remove_dipolar_interactions()
    assert len(crystal.nb_atom) == n_bonds
    assert kernel_energy(crystal, s) == pytest.approx(E_exchange)


def test_dipolar_kept_when_exchange_reassigned():
    crystal = mnf2()
    crystal.add_dipolar_interactions()
    n_bonds = len(crystal.nb_atom)
    crystal.assign_magnetic_parameters(crystal.K, 2 * crystal.J)
    assert len(crystal.nb_atom) == n_bonds


def test_monte_carlo_with_dipolar_tracks_energy():
    from discord.atomistic.simulation import MonteCarlo

    crystal = mnf2()
    crystal.add_dipolar_interactions()
    mc = MonteCarlo(crystal, T=[20, 80], n_replicas=2, seed=3)
    mc.parallel_tempering(
        n_local_sweeps=1,
        n_heatbath_sweeps=1,
        n_overrelaxation_sweeps=1,
        n_cluster_sweeps=2,
        wolff_axes="mixed",
        n_outer=15,
        n_thermal=5,
    )
    for i in range(2):
        assert kernel_energy(crystal, mc.s[i]) == pytest.approx(
            mc.E[i], rel=1e-9, abs=1e-9
        )


@pytest.mark.parametrize(
    "pattern, component, expected",
    [
        ("uniform", 2, -2.09440),  # -2 pi / 3
        ("layered", 2, +4.84372),  # moments along the stacking direction
        ("layered", 0, -2.42186),  # moments perpendicular to it
    ],
)
def test_simple_cubic_reference_fields(pattern, component, expected):
    # Reference values from the rmc-discord test suite: half the dipolar
    # field (in units of mu / a^3) at a site of a simple cubic lattice for a
    # uniform state and for planes alternating along z.
    a = 4.04
    N = (8, 8, 16)
    T = ewald_dipolar_tensors(a * np.eye(3), np.array([[0.5, 0.5, 0.5]]), N)
    mu = np.zeros((*N, 3))
    if pattern == "uniform":
        mu[..., component] = 1.0
    else:
        mu[:, :, 0::2, component] = 1.0
        mu[:, :, 1::2, component] = -1.0
    h = np.einsum("xyzij,xyzj->i", T[0, 0], mu)
    assert h[component] * a**3 / 2 == pytest.approx(expected, abs=2e-5)
