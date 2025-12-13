from itertools import product

import numpy as np

from scipy.linalg import cholesky


from mantid.geometry import CrystalStructure

kB = 0.08617  # meV/K
muB = 0.05788  # meV/T


class Crystal:

    def __init__(self, cell, space_group, sites, super_cell=(4, 4, 4)):

        self.cell = " ".join(6 * ["{}"]).format(*cell)

        self.space_group = space_group

        self.Ns = super_cell

        self.process_sites(sites)

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

        self.n_atoms = len(self.sites)

        self.xyz = np.empty((self.n_atoms, 3), dtype="float")
        self.atoms = []

        for j, site in enumerate(sites):
            atom, x, y, z = site
            for pos in sg.getEquivalentPositions([x, y, z]):
                self.atoms.append(atom)
                self.xyz[j] = self._wrap(np.array(pos))

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

    def get_super_cell_shape(self):
        return self.Ns

    def get_direct_reciprocal_rotation(self):
        return self.R

    def get_direct_cartesian_transform(self):
        return self.A

    def get_reciprocal_cartesian_transform(self):
        return self.B

    def get_moment_cartesian_transform(self):
        return self.C

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
        d = np.liang.norm(u, axis=0)

        mask = (d < d_cut) & (d > 0)

        self.bi = bi[mask]
        self.bj = bj[mask]

        uni, self.inverses = np.unique(d[mask].round(3), return_inverse=True)

        self.n_bonds = uvw.shape[0]

        all_offsets = np.repeat(offsets, self.n_bonds, axis=0)
        self.d_ijk = all_offsets[mask].T.astype(int)

    def get_n_bonds(self):
        return self.n_pairs

    def get_n_atoms(self):
        return self.n_atoms

    def build_neighbor_arrays(self):
        counts = np.bincount(self.bi, minlength=self.n_atoms)
        nb_offsets = np.empty(self.n_atoms + 1, dtype=np.int64)
        nb_offsets[0] = 0
        nb_offsets[1:] = np.cumsum(counts)

        n_bonds = nb_offsets[-1]

        nb_atom = np.empty(n_bonds, dtype=np.int64)
        nb_ijk = np.empty((n_bonds, 3), dtype=np.int64)

        cursor = nb_offsets.copy()
        for i in range(self.bi.size):
            l = self.bi[i]
            pos = cursor[l]
            nb_atom[pos] = self.bj[i]
            nb_ijk[pos] = self.d_ijk[i]
            cursor[l] += 1

        return nb_offsets, nb_atom, nb_ijk
