import pytest
import numpy as np

from discord.material import Crystal
from discord.scattering.intensity import StructureFactor


@pytest.fixture
def struct_fact_MnF2():
    cell = [4.873, 4.873, 3.130, 90, 90, 90]
    space_group = "P 42/m n m"
    sites = [["Mn", 0, 0.0, 0.0]]
    return StructureFactor(Crystal(cell, space_group, sites))


def test_struct_fact_MnF2_ferro(struct_fact):
    pass
