import pytest
import numpy as np

from discord.material import Crystal
from discord.scattering.intensity import StructureFactor

def configuration_FM(crystal):
    crystal.s[..., 0:2] = 0.0
    crystal.s[..., 2] = 1.0

def configuration_AFM(crystal):
    ni, nj, nk = crystal.get_super_cell_shape()
    n_atoms = crystal.get_number_atoms()
    for i_atom in range(n_atoms):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    sign = (-1) ** i_atom
                    crystal.s[:, i_atom, i, j, k, 0:2] = 0.0
                    crystal.s[:, i_atom, i, j, k, 2] = sign

@pytest.fixture
def crystal_MnF2():
    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    return Crystal(cell, space_group, sites, S=2.5)


def test_struct_fact_MnF2_ferro(crystal_MnF2):
    configuration_FM(crystal_MnF2)
    struct_fact = StructureFactor(crystal_MnF2)
    hkl = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 0, 2]])
    I = struct_fact.magnetic_intensity(hkl)
    assert I[0, 1] > I[0, 0]
    assert np.isclose(I[0, 0], 0)
    assert np.isclose(I[0, 2], 0)
    assert np.isclose(I[0, 3], 0)

def test_struct_fact_MnF2_antiferro(crystal_MnF2):
    configuration_AFM(crystal_MnF2)
    struct_fact = StructureFactor(crystal_MnF2)
    hkl = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 0, 2]])
    I = struct_fact.magnetic_intensity(hkl)
    assert I[0, 0] > I[0, 1]
    assert np.isclose(I[0, 1], 0)
    assert np.isclose(I[0, 2], 0)
    assert np.isclose(I[0, 3], 0)