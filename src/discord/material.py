from itertools import product

import numpy as np

from scipy.linalg import cholesky

from mantid.geometry import CrystalStructure

from discord.parameters.constants import muB


class Crystal:

    def __init__(self, cell, space_group, sites, super_cell=(4, 4, 4), S=0.5):

        self.cell = " ".join(6 * ["{}"]).format(*cell)

        self.space_group = space_group

        self.N = super_cell

        n_sites = len(sites)

        S_site = np.array(S)

        self.S_site = S_site if S_site.ndim == 1 else np.full(n_sites, S)

        assert len(self.S_site) == n_sites, "Mismatched quantum number length"

        self.process_sites(sites)

        self.initialize_random_spin_configurations(1)

    def process_sites(self, sites):
        self.sites = []
        for site in sites:
            atom, x, y, z = site
            element = "".join(c for c in atom if c.isalpha())
            self.sites.append([element, x, y, z, 1.0, 0.0])

        self.scatterers = ";".join(
            [" ".join(6 * ["{}"]).format(*site) for site in self.sites]
        )

        cs = CrystalStructure(self.cell, self.space_group, self.scatterers)

        sg = cs.getSpaceGroup()
        uc = cs.getUnitCell()

        self.G = uc.getG().copy()
        self.generate_matrices()

        self.xyz = []
        self.atoms = []

        mu = self.get_effective_moment()
        self.mu = []
        self.S = []

        for j, site in enumerate(sites):
            atom, x, y, z = site
            for pos in sg.getEquivalentPositions([x, y, z]):
                self.atoms.append(atom)
                self.xyz.append(self._wrap(np.array(pos)))
                self.mu.append(mu[j])
                self.S.append(self.S_site[j])

        self.n_atoms = len(self.atoms)

        self.xyz = np.array(self.xyz)
        self.atoms = np.array(self.atoms)

        self.mu = np.array(self.mu)
        self.S = np.array(self.S)

    def _wrap(self, val):
        val = np.array(val)
        mask = val < 0
        val[mask] += 1
        val[np.isclose(val, 0)] = 0
        return val

    def generate_matrices(self):
        self.G_ = np.linalg.inv(self.G)
        self.A = cholesky(self.G, lower=False)
        self.B = cholesky(self.G_, lower=False)
        self.R = np.dot(np.linalg.inv(self.A).T, np.linalg.inv(self.B))
        self.C = np.dot(self.A, np.diag(1 / np.sqrt(np.diag(self.G))))
        self.C_ = np.linalg.inv(self.C)

    def get_super_cell_shape(self):
        return self.N

    def get_direct_reciprocal_rotation(self):
        return self.R

    def get_direct_cartesian_transform(self):
        return self.A

    def get_reciprocal_cartesian_transform(self):
        return self.B

    def get_moment_cartesian_transform(self):
        return self.C

    def get_moment_crystal_axis_transform(self):
        return self.C_

    def get_direct_metric_tensor(self):
        return self.G

    def get_reciprocal_metric_tensor(self):
        return self.G_

    def generate_bonds(self, d_cut=5, radius=4):
        i, j = np.indices((self.n_atoms, self.n_atoms))
        i, j = i.flatten(), j.flatten()

        uvw = self.xyz[j] - self.xyz[i]

        offsets = np.array(list(product(range(-radius, radius + 1), repeat=3)))

        bonds = uvw + offsets[:, np.newaxis]
        bi = np.tile(i, offsets.shape[0])
        bj = np.tile(j, offsets.shape[0])

        u = np.einsum("ij,klj->ikl", self.A, bonds).reshape(3, -1)
        d = np.linalg.norm(u, axis=0)

        mask = (d < d_cut) & (d > 0)

        self.bi = bi[mask]
        self.bj = bj[mask]

        uni, self.inverses = np.unique(d[mask].round(3), return_inverse=True)

        self.n_bonds = self.bi.size
        self.n_types = uni.size

        all_offsets = np.repeat(offsets, uvw.shape[0], axis=0)
        self.d_ijk = all_offsets[mask].astype(int)

        self.bonds = {i: uni[i] for i in range(self.n_types)}
        return self.bonds

    def initialize_magnetic_parameters(self):
        self.K = np.zeros((self.n_atoms, 3, 3))
        self.J = np.zeros((self.n_types, 3, 3))
        self._build_neighbor_arrays()
        return self.K, self.J

    def assign_magnetic_parameters(self, K, J, H=np.zeros(3)):
        self.K = K
        assert self.K.shape == (self.n_atoms, 3, 3)
        self.J = J
        assert self.J.shape == (self.n_types, 3, 3)
        self.H = np.array(H)
        assert self.H.size == 3
        self._build_neighbor_arrays()

    def _build_neighbor_arrays(self):
        counts = np.bincount(self.bi, minlength=self.n_atoms)
        self.nb_offsets = np.empty(self.n_atoms + 1, dtype=np.int64)
        self.nb_offsets[0] = 0
        self.nb_offsets[1:] = np.cumsum(counts)

        n_bonds = self.nb_offsets[-1]

        self.nb_atom = np.empty(n_bonds, dtype=np.int64)
        self.nb_ijk = np.empty((n_bonds, 3), dtype=np.int64)
        self.nb_J = np.empty((n_bonds, 3, 3), dtype=np.float64)

        J_ij = self.J[self.inverses]

        cursor = self.nb_offsets.copy()
        for i in range(self.bi.size):
            l = self.bi[i]
            pos = cursor[l]
            self.nb_atom[pos] = self.bj[i]
            self.nb_ijk[pos] = self.d_ijk[i]
            self.nb_J[pos] = J_ij[i]
            cursor[l] += 1

    def get_compressed_sparse_row(self):
        return self.nb_offsets, self.nb_atom, self.nb_ijk

    def get_magnetic_parameters(self):
        return self.nb_J, self.K, self.H

    def get_number_bonds(self):
        return self.n_bonds

    def get_number_atoms(self):
        return self.n_atoms

    def get_unit_cell_atom_types(self):
        return self.atoms

    def get_unit_atom_position(self):
        return self.xyz

    def initialize_random_spin_configurations(self, n_replicas):
        n_ijk = self.get_super_cell_shape()
        n_atoms = self.get_number_atoms()

        s = np.random.normal(size=(n_replicas, n_atoms, *n_ijk, 3))
        s /= np.linalg.norm(s, axis=5)[..., np.newaxis]
        self.s = s

    def get_effective_moment(self, g=2):
        return g * np.sqrt(self.S_site * (self.S_site + 1)) * muB

    def get_spin_moments(self):
        return np.einsum("j,ij...->ij...", self.mu, self.s)

    def get_spin_quantum_numbers(self):
        return self.S

    def get_spin_vectors(self):
        return self.s

    def set_spin_vectors(self, s):
        self.s = s

    def net_moment(self):
        return self.get_spin_moments().sum(axis=(1, 2, 3, 4))

    def get_total_sites(self):
        return self.n_atoms * np.prod(self.N)

    def get_atom_positions(self):
        ni, nj, nk = self.N
        offsets = np.stack(
            np.meshgrid(
                np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"
            ),
            axis=-1,
        )

        A = self.get_direct_cartesian_transform()

        uvw = self.xyz[:, None, None, None, :] + offsets[None, ...]
        r = np.einsum("ij,...j->...i", A, uvw)

        return r
