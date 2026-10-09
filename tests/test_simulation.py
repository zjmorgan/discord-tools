import pytest
import numpy as np

from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo

from discord.atomistic import kernel
from discord.parameters.constants import muB


def total_energy(s, bi, bj, d_ijk, J, K, H, g, S, muB):

    n_atoms, ni, nj, nk, _ = s.shape

    Jij = []
    for i_atom in range(n_atoms):
        mask = bi == i_atom
        j_atom = bj[mask]
        di = d_ijk.T[0][mask]
        dj = d_ijk.T[1][mask]
        dk = d_ijk.T[2][mask]
        Jnn = J[mask]
        Jij.append((Jnn, j_atom, di, dj, dk))

    EJ = 0.0
    for i_atom in range(n_atoms):
        S_eff = S[i_atom] * (S[i_atom] + 1.0)
        Jnn, j_atom, di, dj, dk = Jij[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    Sn = s[i_atom, i, j, k, :]
                    Snn = s[
                        j_atom, (i + di) % ni, (j + dj) % nj, (k + dk) % nk, :
                    ]
                    h_eff = np.einsum("ijk,ik->j", Jnn, Snn)
                    EJ += -0.5 * S_eff * (Sn @ h_eff)

    EK = 0.0
    for i_atom in range(n_atoms):
        S_eff = S[i_atom] * (S[i_atom] + 1.0)
        Sn = s[i_atom]
        EK += -S_eff * np.einsum(
            "...i,ij,...j->", Sn, K[i_atom], Sn, optimize=True
        )

    EH = 0.0
    for i_atom in range(n_atoms):
        S_eff = S[i_atom] * (S[i_atom] + 1.0)
        Sn = s[i_atom]
        EH += (
            -muB
            * g[i_atom]
            * np.sqrt(S_eff)
            * np.tensordot(Sn, H, axes=([3], [0])).sum()
        )
    return EJ + EK + EH


@pytest.mark.parametrize("g", [2])
def test_MnF2(g):
    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    crystal = Crystal(cell, space_group, sites, S=2.5)

    crystal.generate_bonds(d_cut=4.8)
    K, J = crystal.initialize_magnetic_parameters()
    K[:] = np.diag([0, 0, 0.091])
    J[0] = 0.028 * np.eye(3)
    J[1] = -0.152 * np.eye(3)
    crystal.assign_magnetic_parameters(K, J)

    mc = MonteCarlo(crystal)

    J = mc.crystal.J

    nb_J, K, H = mc.crystal.get_magnetic_parameters()
    nb_offsets, nb_atom, nb_ijk = mc.crystal.get_compressed_sparse_row()

    S = crystal.get_spin_quantum_numbers()
    g = crystal.get_g_factors()

    for i in range(crystal.s.shape[0]):
        E = kernel.total_heisenberg_energy(
            crystal.s[i],
            crystal.delta_atoms,
            crystal.delta_ions,
            crystal.delta_bonds,
            nb_offsets,
            nb_atom,
            nb_ijk,
            nb_J,
            K,
            H,
            g,
            S,
            muB,
        )
        E0 = total_energy(
            crystal.s[i],
            crystal.bi,
            crystal.bj,
            crystal.d_ijk,
            J[crystal.inverses],
            K,
            H,
            g,
            S,
            muB,
        )
        assert np.isclose(E, E0)

    mc.parallel_tempering(
        n_local_sweeps=1,
        n_cluster_sweeps=1,
        n_overrelaxation_sweeps=1,
        n_heatbath_sweeps=1,
        n_outer=100,
        n_thermal=70,
    )

    s = mc.crystal.get_spin_vectors()[0]

    E0 = total_energy(
        s,
        crystal.bi,
        crystal.bj,
        crystal.d_ijk,
        J[crystal.inverses],
        K,
        H,
        g,
        S,
        muB,
    )

    E = kernel.total_heisenberg_energy(
        mc.crystal.s[0],
        mc.crystal.delta_atoms,
        mc.crystal.delta_ions,
        mc.crystal.delta_bonds,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        g,
        S,
        muB,
    )

    assert np.isclose(E, E0)
    assert np.isclose(E, mc.E[0])


def test_checkpoint_roundtrip_continues_averages(tmp_path):
    h5py = pytest.importorskip("h5py")

    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    crystal = Crystal(cell, space_group, sites, S=2.5)

    crystal.generate_bonds(d_cut=4.8)
    K, J = crystal.initialize_magnetic_parameters()
    K[:] = np.diag([0, 0, 0.091])
    J[0] = 0.028 * np.eye(3)
    J[1] = -0.152 * np.eye(3)
    crystal.assign_magnetic_parameters(K, J)

    mc = MonteCarlo(crystal)

    ckpt = tmp_path / "mc_checkpoint.h5"
    mc.parallel_tempering(
        n_local_sweeps=1,
        n_cluster_sweeps=0,
        n_overrelaxation_sweeps=0,
        n_heatbath_sweeps=0,
        n_outer=40,
        n_thermal=20,
        checkpoint_final=True,
        checkpoint_path=str(ckpt),
    )

    with h5py.File(ckpt, "r") as f:
        n0 = int(f["state"].attrs["n_samples_accumulated"])
        assert n0 > 0

    # Resume and run more; sample count should increase.
    mc2 = MonteCarlo(crystal)
    mc2.parallel_tempering(
        n_local_sweeps=1,
        n_cluster_sweeps=0,
        n_overrelaxation_sweeps=0,
        n_heatbath_sweeps=0,
        n_outer=60,
        n_thermal=20,
        resume_from=str(ckpt),
        checkpoint_final=True,
        checkpoint_path=str(ckpt),
    )

    with h5py.File(ckpt, "r") as f:
        n1 = int(f["state"].attrs["n_samples_accumulated"])
        assert n1 > n0


def _small_mnf2(L=3):
    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    crystal = Crystal(cell, space_group, sites, (L, L, L), S=2.5)

    crystal.generate_bonds(d_cut=4.8)
    K, J = crystal.initialize_magnetic_parameters()
    K[:] = np.diag([0, 0, 0.091])
    J[0] = 0.028 * np.eye(3)
    J[1] = -0.152 * np.eye(3)
    crystal.assign_magnetic_parameters(K, J)
    return crystal


def test_seed_reproducible_and_resume_matches_uninterrupted(tmp_path):
    pytest.importorskip("h5py")

    sweeps = dict(
        n_local_sweeps=1,
        n_cluster_sweeps=1,
        n_overrelaxation_sweeps=1,
        n_heatbath_sweeps=1,
    )

    def run(**kwargs):
        mc = MonteCarlo(_small_mnf2(), n_replicas=4, seed=1234)
        result = mc.parallel_tempering(**sweeps, n_thermal=10, **kwargs)
        return mc, result

    mc_a, res_a = run(n_outer=30)
    mc_b, res_b = run(n_outer=30)
    assert np.array_equal(mc_a.s, mc_b.s)
    assert np.array_equal(res_a["E(ave)"], res_b["E(ave)"])

    # Stop at 20 steps, resume to 30: identical to the uninterrupted run.
    ckpt = tmp_path / "ckpt.h5"
    run(n_outer=20, checkpoint_final=True, checkpoint_path=str(ckpt))
    mc_c = MonteCarlo(_small_mnf2(), n_replicas=4)
    res_c = mc_c.parallel_tempering(
        **sweeps, n_outer=30, n_thermal=10, resume_from=str(ckpt),
        outdir=str(tmp_path),
    )
    assert np.allclose(mc_c.s, mc_a.s)
    assert np.allclose(res_c["E(ave)"], res_a["E(ave)"])


def test_results_include_series_and_errors(tmp_path):
    pytest.importorskip("h5py")

    mc = MonteCarlo(_small_mnf2(), n_replicas=4, seed=7)
    hkl = np.array([[1, 0, 0]])
    ckpt = tmp_path / "ckpt.h5"
    result = mc.parallel_tempering(
        hkl,
        n_local_sweeps=1,
        n_outer=60,
        n_thermal=20,
        checkpoint_final=True,
        checkpoint_path=str(ckpt),
    )

    n = 40
    series = result["series"]
    assert result["n_samples"] == n
    assert series["E"].shape == (n, 4)
    assert series["M"].shape == (n, 4, 3)
    assert series["I"].shape == (n, 4, 1)

    # Moments are in Bohr magnetons: g sqrt(S(S+1)) = 2 sqrt(8.75) for Mn2+.
    mu = 2.0 * np.sqrt(2.5 * 3.5)
    assert np.allclose(mc.crystal.get_effective_moment(), mu)
    assert np.all(np.linalg.norm(series["M"], axis=-1) <= mu + 1e-12)

    # Series means reproduce the running-sum averages.
    assert np.allclose(series["E"].mean(axis=0), result["E(ave)"])
    assert np.allclose(series["M"].mean(axis=0), result["M(ave)"])

    for key, shape in [
        ("E(err)", (4,)),
        ("tau(E)", (4,)),
        ("M(err)", (4, 3)),
        ("tau(M)", (4, 3)),
        ("C(err)", (4,)),
        ("chi(err)", (4, 3, 3)),
        ("I(err)", (4, 1)),
    ]:
        assert result[key].shape == shape, key
        assert np.all(np.isfinite(result[key])), key
    assert np.all(result["tau(E)"] >= 0.5)
    assert np.all(result["C(err)"] > 0)

    # Series survive a checkpoint round trip, so errors remain available.
    mc2 = MonteCarlo(_small_mnf2(), n_replicas=4)
    result2 = mc2.parallel_tempering(
        hkl, n_local_sweeps=1, n_outer=80, n_thermal=20,
        resume_from=str(ckpt), outdir=str(tmp_path),
    )
    assert result2["n_samples"] == 60
    assert np.array_equal(result2["series"]["E"][:n], series["E"])

    mc.save_results(result, str(tmp_path / "out"))
    data = np.loadtxt(tmp_path / "out_heat_capacity.txt")
    assert data.shape == (4, 3)


def test_per_site_fluctuations_independent_of_supercell():
    # In the paramagnetic phase the correlation length is short, so per-site
    # C and chi must agree between supercells (54 vs 128 sites here).
    results = {}
    for L in (3, 4):
        mc = MonteCarlo(_small_mnf2(L), T=[150, 300], n_replicas=4, seed=L)
        results[L] = mc.parallel_tempering(
            n_local_sweeps=1, n_outer=1500, n_thermal=200
        )
    a, b = results[3], results[4]

    sig_C = np.hypot(a["C(err)"], b["C(err)"])
    assert np.all(np.abs(a["C"] - b["C"]) < 4 * sig_C)

    diag = np.arange(3)
    chi_a, chi_b = a["chi"][:, diag, diag], b["chi"][:, diag, diag]
    sig_chi = np.hypot(
        a["chi(err)"][:, diag, diag], b["chi(err)"][:, diag, diag]
    )
    assert np.all(np.abs(chi_a - chi_b) < 4 * sig_chi)


def test_per_temperature_schedule_and_timing():
    mc = MonteCarlo(_small_mnf2(), n_replicas=4, seed=11)
    result = mc.parallel_tempering(
        n_local_sweeps=[1, 1, 0, 0],
        n_heatbath_sweeps=[0, 0, 2, 0],
        n_overrelaxation_sweeps=[0, 3, 0, 1],
        n_cluster_sweeps=[0, 0, 0, 2],
        n_outer=30,
        n_thermal=10,
    )
    timing = result["timing"]
    assert timing["n_steps"] == 20

    # Kernel calls per step follow the schedule (Wolff: one per cluster).
    assert np.array_equal(timing["metropolis"]["calls_per_step"], [1, 1, 0, 0])
    assert np.array_equal(timing["heatbath"]["calls_per_step"], [0, 0, 1, 0])
    assert np.array_equal(
        timing["overrelaxation"]["calls_per_step"], [0, 1, 0, 1]
    )
    assert np.array_equal(timing["wolff"]["calls_per_step"], [0, 0, 0, 2])

    for method, active in [
        ("metropolis", [0, 1]),
        ("heatbath", [2]),
        ("overrelaxation", [1, 3]),
        ("wolff", [3]),
    ]:
        entry = timing[method]
        idle = np.setdiff1d(np.arange(4), active)
        assert np.all(entry["kernel_time_per_step"][active] > 0)
        assert np.all(entry["kernel_time_per_step"][idle] == 0)
        assert np.all((entry["acceptance"][active] >= 0))
        assert np.all((entry["acceptance"][active] <= 1))
        assert np.all(np.isnan(entry["acceptance"][idle]))
    assert timing["wolff"]["cluster_size"][3] >= 1

    # Wall-time breakdown of the update phase is consistent.
    assert timing["updates"] > 0 and timing["kernel_mean"] > 0
    assert timing["load_imbalance"] >= 0
    assert np.isclose(
        timing["updates"],
        timing["kernel_mean"]
        + timing["load_imbalance"]
        + timing["dispatch_overhead"],
    )
    assert timing["total_wall_time_per_step"] >= timing["updates"]

    # Tracked energies stay consistent with per-temperature schedules.
    nb_J, K, H = mc.crystal.get_magnetic_parameters()
    nb_offsets, nb_atom, nb_ijk = mc.crystal.get_compressed_sparse_row()
    for i in range(4):
        E = kernel.total_heisenberg_energy(
            mc.s[i],
            mc.crystal.delta_atoms,
            mc.crystal.delta_ions,
            mc.crystal.delta_bonds,
            nb_offsets,
            nb_atom,
            nb_ijk,
            nb_J,
            K,
            H,
            mc.crystal.get_g_factors(),
            mc.crystal.get_spin_quantum_numbers(),
            muB,
        )
        assert np.isclose(E, mc.E[i])


def test_schedule_validation():
    mc = MonteCarlo(_small_mnf2(), n_replicas=3, seed=0)
    with pytest.raises(ValueError, match="one entry per temperature"):
        mc.parallel_tempering(n_local_sweeps=[1, 1], n_outer=2, n_thermal=1)
    with pytest.raises(ValueError, match="not ergodic"):
        mc.parallel_tempering(
            n_local_sweeps=[1, 0, 1],
            n_overrelaxation_sweeps=1,
            n_outer=2,
            n_thermal=1,
        )


def test_sample_interval():
    mc = MonteCarlo(_small_mnf2(), n_replicas=3, seed=2)
    result = mc.parallel_tempering(
        n_local_sweeps=1, n_outer=40, n_thermal=10, sample_interval=3
    )
    # Production steps 10..39 sampled at 10, 13, ..., 37; timing covers all.
    assert result["n_samples"] == 10
    assert result["series"]["E"].shape == (10, 3)
    assert result["timing"]["n_steps"] == 30


def test_wolff_anisotropy_axes_option():
    mc = MonteCarlo(_small_mnf2(), n_replicas=3, seed=4)
    result = mc.parallel_tempering(
        n_local_sweeps=1,
        n_cluster_sweeps=2,
        wolff_axes="anisotropy",
        n_outer=20,
        n_thermal=5,
    )
    assert np.all(result["timing"]["wolff"]["calls_per_step"] == 2)

    # Restricted axes alone are not ergodic.
    with pytest.raises(ValueError, match="random axes"):
        mc.parallel_tempering(
            n_local_sweeps=0,
            n_cluster_sweeps=1,
            wolff_axes="anisotropy",
            n_outer=2,
            n_thermal=1,
        )
