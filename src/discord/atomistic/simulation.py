import numpy as np
import os
from datetime import datetime, timezone
import json

from multiprocessing import Pool

from discord.scattering.intensity import StructureFactor

from discord.atomistic import kernel, correlations
from discord.parameters.constants import kB, muB
from discord.atomistic.plotting import plot_results

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


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

    def _require_h5py(self):
        if h5py is None:
            raise ImportError(
                "HDF5 checkpoints require 'h5py'. Install with: pip install h5py"
            )

    def save_checkpoint_h5(
        self,
        path,
        *,
        i_outer,
        n_outer,
        n_thermal,
        hkl=None,
        nb_offsets=None,
        nb_atom=None,
        nb_ijk=None,
        nb_J=None,
        delta_atoms=None,
        delta_ions=None,
        delta_bonds=None,
        compression="gzip",
        compression_opts=4,
    ):
        """
        Save a restartable checkpoint (state + running averages) to HDF5.

        Layout:
        - /meta      : format/version metadata
        - /state     : Markov chain state (T/beta, spins, energies, seeds, step)
        - /material  : material/model parameters (K, J, H, g, S, deltas, neighbors)
        - /averages  : running sums and sample counter for continuing averages
        """

        self._require_h5py()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _ds(group, name, data):
            return group.create_dataset(
                name,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
            )

        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["format"] = "discord.atomistic.checkpoint"
            meta.attrs["version"] = 1
            meta.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

            crystal_group = f.create_group("crystal")
            # Minimal information to reconstruct a Crystal instance.
            cell_params = None
            if hasattr(self.crystal, "cell"):
                try:
                    cell_params = [
                        float(x) for x in str(self.crystal.cell).split()
                    ]
                except Exception:
                    cell_params = None

            sites = None
            if hasattr(self.crystal, "sites"):
                try:
                    sites = [
                        [
                            str(s[0]),
                            float(s[1]),
                            float(s[2]),
                            float(s[3]),
                        ]
                        for s in self.crystal.sites
                    ]
                except Exception:
                    sites = None

            crystal_def = {
                "cell": cell_params,
                "space_group": getattr(self.crystal, "space_group", None),
                "sites": sites,
                "super_cell": list(self.crystal.get_super_cell_shape()),
                "S_site": (
                    getattr(self.crystal, "S_site", None).tolist()
                    if hasattr(self.crystal, "S_site")
                    else None
                ),
                "g_site": (
                    getattr(self.crystal, "g_site", None).tolist()
                    if hasattr(self.crystal, "g_site")
                    else None
                ),
            }
            crystal_group.attrs["definition_json"] = json.dumps(crystal_def)

            state = f.create_group("state")
            state.attrs["i_outer"] = int(i_outer)
            state.attrs["n_outer"] = int(n_outer)
            state.attrs["n_thermal"] = int(n_thermal)
            state.attrs["n_samples_accumulated"] = int(
                getattr(self, "n_samples_accumulated", 0)
            )

            _ds(state, "T", np.asarray(self.T))
            _ds(state, "beta", np.asarray(self.beta))
            _ds(state, "E", np.asarray(self.E))
            _ds(state, "seeds", np.asarray(self.seeds, dtype=np.int64))
            _ds(state, "s", np.asarray(self.s))
            if hkl is not None:
                _ds(state, "hkl", np.asarray(hkl))

            material = f.create_group("material")
            # These are needed to re-run kernels and to validate compatibility.
            if nb_offsets is not None:
                _ds(material, "nb_offsets", np.asarray(nb_offsets))
            if nb_atom is not None:
                _ds(material, "nb_atom", np.asarray(nb_atom))
            if nb_ijk is not None:
                _ds(material, "nb_ijk", np.asarray(nb_ijk))
            if nb_J is not None:
                _ds(material, "nb_J", np.asarray(nb_J))

            if delta_atoms is not None:
                _ds(material, "delta_atoms", np.asarray(delta_atoms))
            if delta_ions is not None:
                _ds(material, "delta_ions", np.asarray(delta_ions))
            if delta_bonds is not None:
                _ds(material, "delta_bonds", np.asarray(delta_bonds))

            # Store higher-level parameters alongside derived ones.
            if hasattr(self.crystal, "K"):
                _ds(material, "K", np.asarray(self.crystal.K))
            if hasattr(self.crystal, "J"):
                _ds(material, "J", np.asarray(self.crystal.J))
            if hasattr(self.crystal, "H"):
                _ds(material, "H", np.asarray(self.crystal.H))

            _ds(material, "g", np.asarray(self.crystal.get_g_factors()))
            _ds(
                material,
                "S",
                np.asarray(self.crystal.get_spin_quantum_numbers()),
            )
            material.attrs["super_cell"] = np.asarray(
                self.crystal.get_super_cell_shape(), dtype=np.int64
            )
            material.attrs["n_atoms"] = int(self.crystal.get_number_atoms())

            av = f.create_group("averages")
            # Running sums (only present after parallel_tempering initializes)
            for name in (
                "M_sum",
                "M_sq_sum",
                "E_sum",
                "E_sq_sum",
                "I_sum",
                "I_sq_sum",
                "C_ij_sum",
                "C_ij_sq_sum",
            ):
                if hasattr(self, name):
                    val = getattr(self, name)
                    if val is not None:
                        _ds(av, name, np.asarray(val))

    def load_checkpoint_h5(
        self,
        path,
        *,
        apply_material_to_crystal=True,
        expect_hkl=None,
    ):
        """Load checkpoint from HDF5 and populate state + averages.

        Returns a dict with control info: i_outer, n_outer, n_thermal,
        n_samples_accumulated.
        """

        self._require_h5py()

        with h5py.File(path, "r") as f:
            meta = f["meta"]
            if meta.attrs.get("format", "") != "discord.atomistic.checkpoint":
                raise ValueError("Unrecognized checkpoint format")
            if int(meta.attrs.get("version", 0)) != 1:
                raise ValueError("Unsupported checkpoint version")

            state = f["state"]
            self.T = np.array(state["T"])
            self.beta = np.array(state["beta"])
            self.E = np.array(state["E"])
            self.seeds = np.array(state["seeds"], dtype=np.int64)
            self.s = np.array(state["s"])
            self.n_samples_accumulated = int(
                state.attrs.get("n_samples_accumulated", 0)
            )

            # Initialize expected accumulator attributes with safe defaults.
            # Checkpoints omit datasets for None-valued accumulators (e.g. I_sum).
            n_replicas = len(self.T)
            _, n_atoms, ni, nj, nk, _ = self.s.shape

            self.M_sum = np.zeros((n_replicas, 3))
            self.M_sq_sum = np.zeros((n_replicas, 3, 3))
            self.E_sum = np.zeros(n_replicas)
            self.E_sq_sum = np.zeros(n_replicas)

            self.I_sum = None
            self.I_sq_sum = None

            self.C_ij_sum = np.zeros(
                (n_replicas, n_atoms, n_atoms, ni, nj, nk)
            )
            self.C_ij_sq_sum = np.zeros(
                (n_replicas, n_atoms, n_atoms, ni, nj, nk)
            )

            hkl = None
            if "hkl" in state:
                hkl = np.array(state["hkl"])
            if expect_hkl is not None:
                if hkl is None:
                    raise ValueError("Checkpoint does not contain hkl")
                if hkl.shape != np.asarray(
                    expect_hkl
                ).shape or not np.allclose(hkl, np.asarray(expect_hkl)):
                    raise ValueError("Provided hkl does not match checkpoint")

            if apply_material_to_crystal and "material" in f:
                material = f["material"]

                # Restore derived neighbor arrays and Hamiltonian parameters.
                # We avoid requiring bond regeneration (generate_bonds) here.
                if "nb_offsets" in material:
                    self.crystal.nb_offsets = np.array(material["nb_offsets"])
                if "nb_atom" in material:
                    self.crystal.nb_atom = np.array(material["nb_atom"])
                if "nb_ijk" in material:
                    self.crystal.nb_ijk = np.array(material["nb_ijk"])
                if "nb_J" in material:
                    self.crystal.nb_J = np.array(material["nb_J"])

                if "K" in material:
                    self.crystal.K = np.array(material["K"])
                if "J" in material:
                    self.crystal.J = np.array(material["J"])
                if "H" in material:
                    self.crystal.H = np.array(material["H"])

                if (
                    "delta_atoms" in material
                    and "delta_ions" in material
                    and "delta_bonds" in material
                ):
                    self.crystal.set_delta_arrays(
                        np.array(material["delta_atoms"]),
                        np.array(material["delta_ions"]),
                        np.array(material["delta_bonds"]),
                    )

            if "averages" in f:
                av = f["averages"]
                # Only set what exists in the file.
                for name in (
                    "M_sum",
                    "M_sq_sum",
                    "E_sum",
                    "E_sq_sum",
                    "I_sum",
                    "I_sq_sum",
                    "C_ij_sum",
                    "C_ij_sq_sum",
                ):
                    if name in av:
                        setattr(self, name, np.array(av[name]))

            self.crystal.set_spin_vectors(self.s)

            crystal_def = None
            if "crystal" in f:
                try:
                    crystal_def = json.loads(
                        f["crystal"].attrs.get("definition_json", "null")
                    )
                except Exception:
                    crystal_def = None

            return {
                "i_outer": int(state.attrs.get("i_outer", -1)),
                "n_outer": int(state.attrs.get("n_outer", -1)),
                "n_thermal": int(state.attrs.get("n_thermal", -1)),
                "n_samples_accumulated": int(
                    state.attrs.get("n_samples_accumulated", 0)
                ),
                "hkl": hkl,
                "crystal": crystal_def,
            }

    @classmethod
    def from_checkpoint_h5(
        cls,
        path,
        *,
        apply_material_to_crystal=True,
    ):
        """Construct a MonteCarlo instance from an HDF5 checkpoint.

        This rebuilds a Crystal using stored crystallographic metadata when
        available, then loads state/averages. Derived neighbor arrays and
        Hamiltonian parameters are restored from the checkpoint.
        """

        if h5py is None:
            raise ImportError(
                "HDF5 checkpoints require 'h5py'. Install with: pip install h5py"
            )

        with h5py.File(path, "r") as f:
            crystal_def = None
            if "crystal" in f:
                try:
                    crystal_def = json.loads(
                        f["crystal"].attrs.get("definition_json", "null")
                    )
                except Exception:
                    crystal_def = None

            if not crystal_def or crystal_def.get("cell") is None:
                raise ValueError(
                    "Checkpoint does not include enough crystal metadata to reconstruct a Crystal"
                )

            from discord.material import Crystal

            cell = crystal_def["cell"]
            space_group = crystal_def.get("space_group")
            sites = crystal_def.get("sites") or []
            super_cell = tuple(crystal_def.get("super_cell") or (4, 4, 4))
            S_site = crystal_def.get("S_site")
            g_site = crystal_def.get("g_site")

            crystal = Crystal(
                cell,
                space_group,
                sites,
                super_cell=super_cell,
                S=S_site if S_site is not None else 0.5,
                g=g_site if g_site is not None else 2,
            )

        # Create with correct replica count, then overwrite T/beta from checkpoint.
        mc = cls(crystal, T=[0, 1], n_replicas=1)
        mc.load_checkpoint_h5(
            path,
            apply_material_to_crystal=apply_material_to_crystal,
            expect_hkl=None,
        )

        # Ensure internal T-grid matches checkpoint rather than the constructor.
        mc.T = np.array(mc.T)
        return mc

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
        delta_atoms,
        delta_ions,
        delta_bonds,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        g,
        S,
    ):
        args = [
            (
                i,
                self.s[i],
                delta_atoms,
                delta_ions,
                delta_bonds,
                self.beta[i],
                self.E[i],
                n_local_sweeps,
                nb_offsets,
                nb_atom,
                nb_ijk,
                nb_J,
                K,
                H,
                g,
                S,
                muB,
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

    def overrelaxation(
        self,
        n_overrelaxation_sweeps,
        n_replicas,
        delta_atoms,
        delta_ions,
        delta_bonds,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        g,
        S,
    ):
        """
        Perform overrelaxation sweeps on all replicas.

        Overrelaxation reflects each spin across its local effective field,
        providing a microcanonical update that decorrelates configurations
        faster than Metropolis updates while approximately preserving energy.
        """
        args = [
            (
                i,
                self.s[i],
                delta_atoms,
                delta_ions,
                delta_bonds,
                self.beta[i],
                self.E[i],
                n_overrelaxation_sweeps,
                nb_offsets,
                nb_atom,
                nb_ijk,
                nb_J,
                K,
                H,
                g,
                S,
                muB,
                self.seeds[i],
            )
            for i in range(n_replicas)
        ]

        results = self.pool.starmap(kernel.overrelaxation_heisenberg, args)
        results.sort(key=lambda x: x[0])

        for i, s, E, seed in results:
            self.s[i] = s
            self.E[i] = E
            self.seeds[i] = seed

    def heatbath(
        self,
        n_heatbath_sweeps,
        n_replicas,
        delta_atoms,
        delta_ions,
        delta_bonds,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        g,
        S,
    ):
        """
        Perform heatbath (Gibbs sampling) sweeps on all replicas.

        Heatbath samples new spin directions from the conditional Boltzmann
        distribution given the effective field. When anisotropy is present,
        the kernel uses an exact heatbath for the linear terms and a
        Metropolis-Hastings correction for the anisotropy term.
        """
        args = [
            (
                i,
                self.s[i],
                delta_atoms,
                delta_ions,
                delta_bonds,
                self.beta[i],
                self.E[i],
                n_heatbath_sweeps,
                nb_offsets,
                nb_atom,
                nb_ijk,
                nb_J,
                K,
                H,
                g,
                S,
                muB,
                self.seeds[i],
            )
            for i in range(n_replicas)
        ]

        results = self.pool.starmap(kernel.heatbath_heisenberg, args)
        results.sort(key=lambda x: x[0])

        for i, s, E, seed in results:
            self.s[i] = s
            self.E[i] = E
            self.seeds[i] = seed

    def wolff_cluster_updates(
        self,
        n_clusters,
        n_replicas,
        delta_atoms,
        delta_ions,
        delta_bonds,
        nb_offsets,
        nb_atom,
        nb_ijk,
        nb_J,
        K,
        H,
        g,
        S,
    ):
        """
        Perform Wolff-style cluster updates on all replicas.

        This is analogous to :meth:`metropolis_hastings` but uses the
        cluster kernel instead of local single-spin updates. One call
        performs ``n_clusters`` cluster flips per replica.
        """

        for _ in range(n_clusters):
            args = [
                (
                    i,
                    self.s[i],
                    delta_atoms,
                    delta_ions,
                    delta_bonds,
                    self.beta[i],
                    self.E[i],
                    nb_offsets,
                    nb_atom,
                    nb_ijk,
                    nb_J,
                    K,
                    H,
                    g,
                    S,
                    muB,
                    self.seeds[i],
                )
                for i in range(n_replicas)
            ]

            results = self.pool.starmap(kernel.wolff_heisenberg, args)
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

        C_ij = correlations.vector_vector(self.s)

        self.C_ij_sum += C_ij
        self.C_ij_sq_sum += C_ij**2

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

        C_ij_ave = self.C_ij_sum / n_sample
        C_ij_sq_ave = self.C_ij_sq_sum / n_sample
        C_ij_std = np.sqrt(C_ij_sq_ave - C_ij_ave**2)

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
            "C_ij(ave)": C_ij_ave,
            "C_ij(std)": C_ij_std,
        }

        return parameters

    def parallel_tempering(
        self,
        hkl=None,
        n_local_sweeps=2,
        n_cluster_sweeps=0,
        n_overrelaxation_sweeps=0,
        n_heatbath_sweeps=0,
        n_outer=1000,
        n_thermal=700,
        n_interval=None,
        checkpoint_interval=None,
        checkpoint_final=None,
        checkpoint_path=None,
        resume_from=None,
        outdir="checkpoints",
        prefix="mc",
    ):
        assert n_outer > 0

        if checkpoint_final is None:
            checkpoint_final = (
                checkpoint_interval is not None or resume_from is not None
            )

        # Prepare output directory if we're writing anything.
        if (
            n_interval is not None
            or checkpoint_interval is not None
            or checkpoint_final
        ):
            os.makedirs(outdir, exist_ok=True)

        i_outer_start = 0

        if resume_from is not None:
            info = self.load_checkpoint_h5(
                resume_from, apply_material_to_crystal=True, expect_hkl=hkl
            )
            i_outer_start = info["i_outer"] + 1
            n_thermal = (
                int(info["n_thermal"]) if info["n_thermal"] >= 0 else n_thermal
            )

        n_replicas = self.get_n_replicas()

        if resume_from is None:
            assert (
                n_outer - n_thermal
            ) > 0, "Outer steps less than thermalization steps"

            self.beta = 1.0 / (kB * self.T)
            self.seeds = self.make_seeds(n_replicas)
            self.n_samples_accumulated = 0

            self.M_sum = np.zeros((n_replicas, 3))
            self.M_sq_sum = np.zeros((n_replicas, 3, 3))

            self.E_sum = np.zeros(n_replicas)
            self.E_sq_sum = np.zeros(n_replicas)

            self.I_sum = None
            self.I_sq_sum = None
            if hkl is not None:
                self.I_sum = np.zeros((n_replicas, len(hkl)))
                self.I_sq_sum = np.zeros((n_replicas, len(hkl)))

            n_atoms = self.crystal.get_number_atoms()
            N = self.crystal.get_super_cell_shape()
            self.C_ij_sum = np.zeros((n_replicas, n_atoms, n_atoms, *N))
            self.C_ij_sq_sum = np.zeros((n_replicas, n_atoms, n_atoms, *N))

            self.crystal.initialize_random_spin_configurations(n_replicas)

            self.s = self.crystal.get_spin_vectors()
            self.E = np.zeros(n_replicas)

        nb_offsets, nb_atom, nb_ijk = self.crystal.get_compressed_sparse_row()
        nb_J, K, H = self.crystal.get_magnetic_parameters()
        delta_atoms, delta_ions, delta_bonds = self.crystal.get_delta_arrays()
        S = self.crystal.get_spin_quantum_numbers()
        g = self.crystal.get_g_factors()

        if resume_from is None:
            for i in range(n_replicas):
                self.E[i] = kernel.total_heisenberg_energy(
                    self.s[i],
                    delta_atoms,
                    delta_ions,
                    delta_bonds,
                    nb_offsets,
                    nb_atom,
                    nb_ijk,
                    nb_J,
                    K,
                    H,
                    g,
                    S,
                    muB,
                )

        with Pool(processes=n_replicas) as self.pool:
            last_i_outer = i_outer_start - 1
            for i_outer in range(i_outer_start, n_outer):
                last_i_outer = i_outer
                print(f"{i_outer}/{n_outer}")

                # Common parameters for all MC update methods
                mc_params = (
                    n_replicas,
                    delta_atoms,
                    delta_ions,
                    delta_bonds,
                    nb_offsets,
                    nb_atom,
                    nb_ijk,
                    nb_J,
                    K,
                    H,
                    g,
                    S,
                )

                if n_cluster_sweeps > 0:
                    self.wolff_cluster_updates(n_cluster_sweeps, *mc_params)

                if n_overrelaxation_sweeps > 0:
                    self.overrelaxation(n_overrelaxation_sweeps, *mc_params)

                if n_heatbath_sweeps > 0:
                    self.heatbath(n_heatbath_sweeps, *mc_params)

                if n_local_sweeps > 0:
                    self.metropolis_hastings(n_local_sweeps, *mc_params)

                self.replica_exchange()

                if i_outer >= n_thermal:
                    self.crystal.set_spin_vectors(self.s)
                    self.sample_parameters(hkl)
                    self.n_samples_accumulated += 1

                    if (
                        n_interval is not None
                        and (i_outer + 1) % n_interval == 0
                    ):
                        result = self.ensemble_average(
                            self.n_samples_accumulated
                        )
                        plot_results(
                            result,
                            prefix=prefix,
                            outdir=outdir,
                            show=False,
                        )

                if (
                    checkpoint_interval is not None
                    and (i_outer + 1) % checkpoint_interval == 0
                ):
                    checkpoint_path_interval = (
                        checkpoint_path
                        if checkpoint_path is not None
                        else os.path.join(outdir, f"{prefix}_checkpoint.h5")
                    )
                    self.save_checkpoint_h5(
                        checkpoint_path_interval,
                        i_outer=i_outer,
                        n_outer=n_outer,
                        n_thermal=n_thermal,
                        hkl=hkl,
                        nb_offsets=nb_offsets,
                        nb_atom=nb_atom,
                        nb_ijk=nb_ijk,
                        nb_J=nb_J,
                        delta_atoms=delta_atoms,
                        delta_ions=delta_ions,
                        delta_bonds=delta_bonds,
                    )

            if checkpoint_final:
                checkpoint_path_final = (
                    checkpoint_path
                    if checkpoint_path is not None
                    else os.path.join(outdir, f"{prefix}_checkpoint.h5")
                )
                self.save_checkpoint_h5(
                    checkpoint_path_final,
                    i_outer=last_i_outer,
                    n_outer=n_outer,
                    n_thermal=n_thermal,
                    hkl=hkl,
                    nb_offsets=nb_offsets,
                    nb_atom=nb_atom,
                    nb_ijk=nb_ijk,
                    nb_J=nb_J,
                    delta_atoms=delta_atoms,
                    delta_ions=delta_ions,
                    delta_bonds=delta_bonds,
                )

        self.crystal.set_spin_vectors(self.s)

        # When resuming, n_samples_accumulated may differ from (n_outer-n_thermal)
        # if the run is extended.
        n_samples = int(getattr(self, "n_samples_accumulated", 0))
        if n_samples <= 0:
            n_samples = max(1, n_outer - n_thermal)
        return self.ensemble_average(n_samples)

    def save_results(self, result, filename):
        """
        Save Monte Carlo simulation results to text files.

        Parameters
        ----------
        result : dict
            Dictionary of results from parallel_tempering method.
        filename : str
            Base filename (without extension) for saving results
        """
        T = result["T"]

        chi_11 = result["chi"][:, 0, 0]
        chi_22 = result["chi"][:, 1, 1]
        chi_33 = result["chi"][:, 2, 2]
        chi_23 = result["chi"][:, 1, 2]
        chi_13 = result["chi"][:, 0, 2]
        chi_12 = result["chi"][:, 0, 1]

        np.savetxt(
            filename + "_susceptibility.txt",
            np.column_stack(
                (T, chi_11, chi_22, chi_33, chi_23, chi_13, chi_12)
            ),
            header="T chi_11 chi_22 chi_33 chi_23 chi_13 chi_12",
        )

        Mx = result["M(ave)"][:, 0]
        My = result["M(ave)"][:, 1]
        Mz = result["M(ave)"][:, 2]
        Mx_std = result["M(std)"][:, 0]
        My_std = result["M(std)"][:, 1]
        Mz_std = result["M(std)"][:, 2]

        np.savetxt(
            filename + "_magnetization.txt",
            np.column_stack((T, Mx, My, Mz, Mx_std, My_std, Mz_std)),
            header="T Mx My Mz Mx_std My_std Mz_std",
        )

        E = result["E(ave)"]
        E_std = result["E(std)"]

        np.savetxt(
            filename + "_energy.txt",
            np.column_stack((T, E, E_std)),
            header="T E E_std",
        )

        C = result["C"]

        np.savetxt(
            filename + "_heat_capacity.txt",
            np.column_stack((T, C)),
            header="T C",
        )

        if result["I(ave)"] is not None:
            I = result["I(ave)"][:, 0]
            sig = result["I(std)"][:, 0]

            np.savetxt(
                filename + "_intensity.txt",
                np.column_stack((T, I, sig)),
                header="T I sig",
            )
