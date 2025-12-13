import numpy as np

from discord.parameters import tables

p = 2.695 # fm/muB


class StrctureFactor:

    def __init__(self, crystal):
        self.crystal = crystal

        self.extract_crystal_info()

    def extract_crystal_info(self):
        self.xyz = self.crystal.get_unit_atom_position()
        self.atoms = self.crystal.get_unic_cell_atom_types()

        self.R = self.crystal.get_direct_reciprocal_rotation()
        self.A = self.crystal.get_direct_cartesian_transform()
        self.B = self.crystal.get_reciprocal_cartesian_transform()
        self.C = self.crystal.get_moment_cartesian_transform()
        self.G_ = self.crystal.get_direct_metric_tensor()
        self.G_ = self.crystal.get_reciprocal_metric_tensor()

        self.Ns = self.crystal.get_super_cell_shape()

    def phase_factor(self, hkl):
        return np.exp(2j * np.pi * np.einsum("ji,ki->jk", self.xyz, hkl))

    def indices(self, hkl):
        Ns = np.asarray(self.Ns)
        return np.mod(np.round(hkl * Ns).astype(int), Ns)

    def d_spacing(self, hkl):
        return 1.0 / np.sqrt(np.einsum("ij,ki,kj->k", self.G_, hkl, hkl))

    def magnetic_form_factor(self, hkl, g=2):
        k = 2 / g - 1
        d = self.d_spacing(hkl)
        s = 0.5 / d
        f = np.empty((len(self.atoms), d.size))
        for i, atom in enumerate(self.atoms):
            A, a, B, b, C, c, D = tables.j0.get(atom, [0] * 7)
            j0 = (
                + A * np.exp(-a * s**2)
                + B * np.exp(-b * s**2)
                + C * np.exp(-c * s**2)
                + D
            )
            A, a, B, b, C, c, D = tables.j2.get(atom, [0] * 7)
            j2 = A*s**2*np.exp(-a*s**2)+\
                   B*s**2*np.exp(-b*s**2)+\
                   C*s**2*np.exp(-c*s**2)+\
                   D*s**2
            f[i] = j0 + k * j2
        f[f < 0] = 0.0
        return f

    def magnetic_structure_factor(self, M, hkl, xyz, Ns):
        fQ = self.magnetic_form_factor(hkl)
        pf = self.phase_factor(hkl)

        i_idx, j_idx, k_idx = self.indices(hkl, Ns)

        vol = np.prod(Ns)

        Mk = np.fft.ifftn(M, axes=(1, 2, 3)) / vol
        MQ = Mk[:, i_idx, j_idx, k_idx, :]

        F = p * np.sum(MQ * (pf * fQ)[..., None], axis=0)
        return F

    def magnetic_intensity(self, hkl, F):
        d = self.d_spacing(hkl, self.G_)
        dhkl = hkl * d

        Fc = np.einsum("ij,kj->ki", self.C, F)
        Qc = np.einsum("ij,kj->ki", self.R @ self.B, dhkl)

        Q = np.linalg.norm(Qc, axis=1, keepdims=True)
        Q[Q == 0] = 1.0
        Q_hat = Qc / Q

        Fc_dot_Q_hat = np.sum(Fc.conj() * Q_hat, axis=1)
        Fc_perp = Fc - Q_hat * Fc_dot_Q_hat

        I = np.sum(Fc_perp.conj() * Fc_perp, axis=1).real
        return I
