import pytest
import numpy as np

from discord.material import Crystal


@pytest.fixture
def crystal_MnF2():
    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    return Crystal(cell, space_group, sites, S=2.5)


@pytest.mark.parametrize("a,c", [(4.873, 3.130)])
def test_crystal_MnF2_transforms(a, c, crystal_MnF2):
    A = crystal_MnF2.get_direct_cartesian_transform()
    assert np.allclose(A, np.diag([a, a, c]))
    B = crystal_MnF2.get_reciprocal_cartesian_transform()
    assert np.allclose(B, np.diag([1 / a, 1 / a, 1 / c]))
    G = crystal_MnF2.get_direct_metric_tensor()
    assert np.allclose(G, np.diag([a**2, a**2, c**2]))
    G_ = crystal_MnF2.get_reciprocal_metric_tensor()
    assert np.allclose(G_, np.diag([1 / a**2, 1 / a**2, 1 / c**2]))
    R = crystal_MnF2.get_direct_reciprocal_rotation()
    assert np.allclose(R, np.eye(3))
    C = crystal_MnF2.get_moment_cartesian_transform()
    assert np.allclose(C, np.eye(3))

def test_crystal_MnF2_atoms(crystal_MnF2):
    assert crystal_MnF2.get_number_atoms() == 2