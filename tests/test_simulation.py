import pytest
import numpy as np

from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo

from discord.atomistic import kernel
from discord.parameters.constants import muB


def total_energy(s, bi, bj, d_ijk, J, K, H, muB, g):

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
        Jnn, j_atom, di, dj, dk = Jij[i_atom]
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    Sn = s[i_atom, i, j, k, :]
                    Snn = s[
                        j_atom, (i + di) % ni, (j + dj) % nj, (k + dk) % nk, :
                    ]
                    h_eff = np.einsum("ijk,ik->j", Jnn, Snn)
                    EJ += -0.5 * Sn @ h_eff

    EK = 0.0
    for i_atom in range(n_atoms):
        Sn = s[i_atom]
        EK += -np.einsum("...i,ij,...j->", Sn, K[i_atom], Sn, optimize=True)

    EH = -muB * g * np.tensordot(s, H, axes=([4], [0])).sum()
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

    J = mc.crystal.J * crystal.S * (crystal.S + 1)

    nb_J, K, H = mc.crystal.get_magnetic_parameters()
    nb_offsets, nb_atom, nb_ijk = mc.crystal.get_compressed_sparse_row()

    for i in range(crystal.s.shape[0]):
        E = kernel.total_heisenberg_energy(
            crystal.s[i], nb_offsets, nb_atom, nb_ijk, nb_J, K, H, muB, g
        )
        E0 = total_energy(
            crystal.s[i],
            crystal.bi,
            crystal.bj,
            crystal.d_ijk,
            J[crystal.inverses],
            K,
            H,
            muB,
            g,
        )
        assert np.isclose(E, E0)

    mc.parallel_tempering(n_local_sweeps=2, n_outer=100, n_thermal=70, g=g)

    s = mc.crystal.get_spin_vectors()[0]

    E0 = total_energy(
        s,
        crystal.bi,
        crystal.bj,
        crystal.d_ijk,
        J[crystal.inverses],
        K,
        H,
        muB,
        g,
    )

    E = kernel.total_heisenberg_energy(
        mc.crystal.s[0], nb_offsets, nb_atom, nb_ijk, nb_J, K, H, muB, g
    )

    assert np.isclose(E, E0)
    assert np.isclose(E, mc.E[0])
