import numpy as np
from numba import njit

@njit
def matvec(A, v):
    return np.array([
        A[0, 0]*v[0] + A[0, 1]*v[1] + A[0, 2]*v[2],
        A[1, 0]*v[0] + A[1, 1]*v[1] + A[1, 2]*v[2],
        A[2, 0]*v[0] + A[2, 1]*v[1] + A[2, 2]*v[2],
    ])

@njit
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit
def unravel_site(idx, n, Nx, Ny, Nz):
    Nyz = Ny * Nz
    Nxyz = Nx * Nyz

    mu = idx // Nxyz
    r = idx % Nxyz
    i = r // Nyz
    r = r % Nyz
    j = r // Nz
    k = r % Nz
    return mu, i, j, k

@njit
def ravel_site(mu, i, j, k, n, Nx, Ny, Nz):
    Nyz = Ny * Nz
    Nxyz = Nx * Nyz
    return mu * Nxyz + i * Nyz + j * Nz + k

@njit
def random_unit_vector():
    v0 = np.random.normal()
    v1 = np.random.normal()
    v2 = np.random.normal()
    v = np.sqrt(v0*v0 + v1*v1 + v2*v2)
    if v == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return np.array([v0 / v, v1 / v, v2 / v])
