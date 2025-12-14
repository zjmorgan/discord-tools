import time
import colorsys

import numpy as np
import matplotlib.pyplot as plt

from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo

cell = [4.873, 4.873, 3.130, 90, 90, 90]
space_group = "P 42/m n m"
sites = [["Mn", 0.0, 0.0, 0.0]]
crystal = Crystal(cell, space_group, sites, S=2.5)

crystal.generate_bonds(d_cut=4.8)
K, J = crystal.initialize_magnetic_parameters()
K[:] = np.diag([0, 0, 0.091])
J[0] = 0.028 * np.eye(3)
J[1] = -0.152 * np.eye(3)
crystal.assign_magnetic_parameters(K, J)

mc = MonteCarlo(crystal)

t0 = time.time()
result = mc.parallel_tempering(n_local_sweeps=2, n_outer=1000, n_thermal=700)
t1 = time.time()

print(f"Time: {t1 - t0:.2f} seconds")

T = result["T"]

chi_xx = result["chi"][:, 0, 0]
chi_yy = result["chi"][:, 1, 1]
chi_zz = result["chi"][:, 2, 2]
chi_yz = result["chi"][:, 1, 2]
chi_xz = result["chi"][:, 0, 2]
chi_xy = result["chi"][:, 0, 1]

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.minorticks_on()
ax.plot(T, chi_xx, "-o", label="$\chi_{xx}$")
ax.plot(T, chi_yy, "-o", label="$\chi_{yy}$")
ax.plot(T, chi_zz, "-o", label="$\chi_{zz}$")
ax.plot(T, chi_yz, "-o", label="$\chi_{yz}$")
ax.plot(T, chi_xz, "-o", label="$\chi_{xz}$")
ax.plot(T, chi_xy, "-o", label="$\chi_{xy}$")
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

s = crystal.get_spin_vectors()[0]
r = crystal.get_atom_positions()

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(111, projection="3d")

x = r[..., 0].flatten()
y = r[..., 1].flatten()
z = r[..., 2].flatten()

u = s[..., 0].flatten()
v = s[..., 1].flatten()
w = s[..., 2].flatten()

phi = np.arctan2(v, u)
hue = (phi + np.pi) / (2 * np.pi)

lightness = 0.15 + 0.7 * (w + 1) / 2.0
saturation = np.ones_like(hue)

colors = [
    colorsys.hls_to_rgb(hh, ll, ss)
    for hh, ll, ss in zip(hue, lightness, saturation)
]

ax.quiver(
    x,
    y,
    z,
    u,
    v,
    w,
    normalize=True,
    pivot="middle",
    color=colors,
)

ax.set_xlabel("x [$\AA$]")
ax.set_ylabel("y [$\AA$]")
ax.set_zlabel("z [$\AA$]")
fig.savefig("MnF2_ground_state.png")
