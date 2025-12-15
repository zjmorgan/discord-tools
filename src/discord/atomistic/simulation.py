import numpy as np

from multiprocessing import Pool

from discord.scattering.intensity import StructureFactor

from discord.atomistic import kernel
from discord.parameters.constants import kB, muB


class MonteCarlo:
    """Replica-exchange Monte Carlo simulation."""

    def __init__(self, crystal, T=[10, 300], n_replicas=30):
        self.crystal = crystal

        self.T = np.linspace(*T, n_replicas)

    def get_n_replicas(self):
        return len(self.T)

    def make_seeds(self, n_replicas):
        root = np.random.SeedSequence()
        children = root.spawn(n_replicas)
        return [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]

    def replica_exchange(self):
        n_replica = self.get_n_replicas()
        for offset in (0, 1):
            for i in range(offset, n_replica - 1, 2):
                j = i + 1
                beta0, beta1 = self.beta[i], self.beta[j]
                E0, E1 = self.E[i], self.E[j]
                d = (beta0 - beta1) * (E1 - E0)
                if np.random.rand() < np.exp(-d):
                    self.s[i], self.s[j] = self.s[j].copy(), self.s[i].copy()
                    self.E[i], self.E[j] = self.E[j], self.E[i]
                    self.seeds[i], self.seeds[j] = self.seeds[j], self.seeds[i]

    def metropolis_hastings(
        self,
        n_local_sweeps,
        n_replicas,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        S,
        g=2,
    ):
        args = [
            (
                i,
                self.s[i],
                self.beta[i],
                self.E[i],
                n_local_sweeps,
                nb_offsets,
                nb_atom,
                nb_ijk,
                nb_J,
                K,
                H,
                S,
                muB,
                g,
                self.seeds[i],
            )
            for i in range(n_replicas)
        ]

        results = self.pool.starmap(kernel.metropolis_heisenberg, args)
        results.sort(key=lambda x: x[0])

        for i, s, E, seed in results:
            self.s[i] = s
            self.E[i] = E
            self.seeds[i] = seed

    def sample_parameters(self, hkl):
        n_sites = self.crystal.get_total_sites()

        M = self.crystal.net_moment()
        self.M_sum += M / n_sites
        self.M_sq_sum += M[:, :, None] * M[:, None, :] / n_sites**2

        self.E_sum += self.E / n_sites
        self.E_sq_sum += self.E**2 / n_sites**2

        if hkl is not None:
            struct_fact = StructureFactor(self.crystal)
            I = struct_fact.magnetic_intensity(hkl)
            self.I_sum += I
            self.I_sq_sum += I**2

    def ensemble_average(self, n_sample):
        M_ave = self.M_sum / n_sample
        M_sq_ave = self.M_sq_sum / n_sample

        M_var = M_sq_ave - np.einsum("ri,rj->rij", M_ave, M_ave)
        M_std = np.sqrt(M_var[:, np.arange(3), np.arange(3)])

        chi = self.beta[:, None, None] * M_var
        chi = 0.5 * (chi + np.swapaxes(chi, 1, 2))

        E_ave = self.E_sum / n_sample
        E_sq_ave = self.E_sq_sum / n_sample

        E_var = E_sq_ave - E_ave**2
        E_std = np.sqrt(E_var)

        C = self.beta**2 * E_var

        I_ave = None
        I_std = None
        if self.I_sum is not None:
            I_ave = self.I_sum / n_sample
            I_sq_ave = self.I_sq_sum / n_sample

            I_std = np.sqrt(I_sq_ave - I_ave**2)

        parameters = {
            "T": self.T,
            "M(ave)": M_ave,
            "M(std)": M_std,
            "chi": chi,
            "E(ave)": E_ave,
            "E(std)": E_std,
            "C": C,
            "I(ave)": I_ave,
            "I(std)": I_std,
        }

        return parameters

    def parallel_tempering(
        self,
        hkl=None,
        n_local_sweeps=2,
        n_outer=1000,
        n_thermal=700,
        g=2,
    ):
        n_sample = n_outer - n_thermal
        assert n_sample > 0, "Outer steps less than thermalization steps"

        n_replicas = self.get_n_replicas()

        self.beta = 1.0 / (kB * self.T)
        self.seeds = self.make_seeds(n_replicas)

        nb_offsets, nb_atom, nb_ijk = self.crystal.get_compressed_sparse_row()

        nb_J, K, H = self.crystal.get_magnetic_parameters()

        S = self.crystal.get_spin_quantum_numbers()

        self.M_sum = np.zeros((n_replicas, 3))
        self.M_sq_sum = np.zeros((n_replicas, 3, 3))

        self.E_sum = np.zeros(n_replicas)
        self.E_sq_sum = np.zeros(n_replicas)

        self.I_sum = None
        self.I_sq_sum = None

        if hkl is not None:
            self.I_sum = np.zeros((n_replicas, len(hkl)))
            self.I_sq_sum = np.zeros((n_replicas, len(hkl)))

        self.crystal.initialize_random_spin_configurations(n_replicas)

        self.s = self.crystal.get_spin_vectors()
        self.E = np.zeros(n_replicas)

        for i in range(n_replicas):
            self.E[i] = kernel.total_heisenberg_energy(
                self.s[i], nb_offsets, nb_atom, nb_ijk, nb_J, K, H, S, muB, g
            )

        with Pool(processes=n_replicas) as self.pool:
            for i_outer in range(n_outer):
                print(f"{i_outer}/{n_outer}")

                self.metropolis_hastings(
                    n_local_sweeps,
                    n_replicas,
                    nb_offsets,
                    nb_atom,
                    nb_ijk,
                    nb_J,
                    K,
                    H,
                    S,
                    g,
                )

                self.replica_exchange()

                if i_outer >= n_thermal:

                    self.crystal.set_spin_vectors(self.s)

                    self.sample_parameters(hkl)

        return self.ensemble_average(n_sample)
