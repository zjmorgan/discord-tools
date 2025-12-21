import time

import numpy as np
import matplotlib.pyplot as plt

from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo
from discord.visualization import Visualize

cell = [4.873, 4.873, 3.130, 90, 90, 90]
space_group = "P 42/m n m"
sites = [["Mn", 0.0, 0.0, 0.0]]
crystal = Crystal(cell, space_group, sites, (8, 8, 8), S=2.5)

crystal.generate_bonds(d_cut=4.8)
K, J = crystal.initialize_magnetic_parameters()
K[:] = np.diag([0, 0, 0.091])
J[0] = 0.028 * np.eye(3)
J[1] = -0.152 * np.eye(3)
crystal.assign_magnetic_parameters(K, J)

hkl = np.array([[1, 0, 0]])

mc = MonteCarlo(crystal)

t0 = time.time()
result = mc.parallel_tempering(
    hkl, n_local_sweeps=2, n_outer=10000, n_thermal=7000
)
t1 = time.time()

print(f"Time: {t1 - t0:.2f} seconds")

T = result["T"]

chi_11 = result["chi"][:, 0, 0]
chi_22 = result["chi"][:, 1, 1]
chi_33 = result["chi"][:, 2, 2]
chi_23 = result["chi"][:, 1, 2]
chi_13 = result["chi"][:, 0, 2]
chi_12 = result["chi"][:, 0, 1]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.plot(T, chi_11, "-o", label="$\chi_{11}$")
ax.plot(T, chi_22, "-o", label="$\chi_{22}$")
ax.plot(T, chi_33, "-o", label="$\chi_{33}$")
ax.plot(T, chi_23, "-o", label="$\chi_{23}$")
ax.plot(T, chi_13, "-o", label="$\chi_{13}$")
ax.plot(T, chi_12, "-o", label="$\chi_{12}$")
ax.legend(shadow=True)
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("$\chi_{ij}$ [$\mu_B^2$/eV]")
fig.savefig("MnF2_susceptibility.png")

Mx = result["M(ave)"][:, 0]
My = result["M(ave)"][:, 1]
Mz = result["M(ave)"][:, 2]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.plot(T, Mx, "-o", label="$M_{x}$")
ax.plot(T, My, "-o", label="$M_{y}$")
ax.plot(T, Mz, "-o", label="$M_{z}$")
ax.legend(shadow=True)
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("$M_{i}$ [$\mu_B$]")
fig.savefig("MnF2_magnetization.png")

C = result["C"]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.plot(T, C, "-o")
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("$C$ [eV/K]")
fig.savefig("MnF2_heat_capacity.png")

E = result["E(ave)"]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.plot(T, E, "-o")
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("$E$ [eV]")
fig.savefig("MnF2_energy.png")

I = result["I(ave)"][:, 0]
sig = result["I(std)"][:, 0]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.errorbar(T, I, sig, fmt="-o")
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("$I$ [arb. units]")
fig.savefig("MnF2_intensity.png")

s = crystal.get_spin_vectors()[0]

viz = Visualize(crystal)
viz.plot_spins(s, filename="MnF2_ground_state.png")
