import pytest
import numpy as np

from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo

# @pytest.fixture
# def crystal_MnF2():
cell = [4.873, 4.873, 3.130, 90, 90, 90]
space_group = "P 42/m n m"
sites = [["Mn", 0, 0.0, 0.0]]
crystal = Crystal(cell, space_group, sites, S=2.5)

bonds = crystal.generate_bonds(d_cut=4.8)
K, J = crystal.initialize_magnetic_parameters()
K[:] = np.diag([0, 0, 0.091])
J[0] = 0.028 * np.eye(3)
J[1] = -0.152 * np.eye(3)
crystal.assign_magnetic_parameters(K, J)
# return crystal
