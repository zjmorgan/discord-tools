import numpy as np

from discord.parameters import tables

from discord.parameters.constants import p

class StructureFactor:

    def __init__(self, crystal):
        self.crystal = crystal

    def phase_factor(self, hkl):
        xyz = self.crystal.get_unit_atom_position()
        return np.exp(2j * np.pi * np.einsum("ji,ki->jk", xyz, hkl))

    def indices(self, hkl):
        N = np.asarray(self.crystal.get_super_cell_shape())
        return np.mod(np.round(hkl * N).astype(int), N)

    def d_spacing(self, hkl):
        G_ = self.crystal.get_reciprocal_metric_tensor()
        return 1.0 / np.sqrt(np.einsum("ij,ki,kj->k", G_, hkl, hkl))

    def magnetic_form_factor(self, hkl, g=2):
        k = 2 / g - 1

        d = self.d_spacing(hkl)
        s = 0.5 / d

        atoms = self.crystal.get_unit_cell_atom_types()
        n_atoms = self.crystal.get_number_atoms()

        f = np.empty((n_atoms, d.size))
        for i, atom in enumerate(atoms):
            A, a, B, b, C, c, D = tables.j0.get(atom, [0] * 7)
            j0 = (
                +A * np.exp(-a * s**2)
                + B * np.exp(-b * s**2)
                + C * np.exp(-c * s**2)
                + D
            )
            A, a, B, b, C, c, D = tables.j2.get(atom, [0] * 7)
            j2 = (
                A * s**2 * np.exp(-a * s**2)
                + B * s**2 * np.exp(-b * s**2)
                + C * s**2 * np.exp(-c * s**2)
                + D * s**2
            )
            f[i] = j0 + k * j2
        f[f < 0] = 0.0
        return f

    def magnetic_structure_factor(self, hkl):
        fQ = self.magnetic_form_factor(hkl)
        pf = self.phase_factor(hkl)

        ijk = self.indices(hkl)

        vol = np.prod(self.crystal.get_super_cell_shape())

        M = self.crystal.get_spin_moments()

        Mk = np.fft.ifftn(M, axes=(2, 3, 4)) * vol
        MQ = Mk[..., ijk[:, 0], ijk[:, 1], ijk[:, 2], :]

        F = p * np.sum(MQ * (pf * fQ)[..., np.newaxis], axis=1)
        
        return F

    def magnetic_intensity(self, hkl):
        F = self.magnetic_structure_factor(hkl)

        d = self.d_spacing(hkl)
        dhkl = hkl * d[:, np.newaxis]

        C = self.crystal.get_moment_cartesian_transform()
        R = self.crystal.get_direct_reciprocal_rotation()
        B = self.crystal.get_reciprocal_cartesian_transform()

        Fc = np.einsum("ij,klj->kli", C, F)
        Qc = np.einsum("ij,kj->ki", R @ B, dhkl)

        Q = np.linalg.norm(Qc, axis=1, keepdims=True)
        Q[Q == 0] = 1.0
        Q_hat = Qc / Q

        Fc_dot_Q_hat = np.sum(Fc.conj() * Q_hat, axis=2)
        Fc_perp = Fc - Q_hat * Fc_dot_Q_hat[..., np.newaxis]

        I = np.sum(Fc_perp.conj() * Fc_perp, axis=2).real
        return I
