import numpy as np
import os
import time
from datetime import datetime, timezone
import json

from multiprocessing import Pool

from discord.scattering.intensity import StructureFactor

from discord.atomistic import kernel, correlations, statistics
from discord.parameters.constants import kB, muB
from discord.atomistic.plotting import plot_results

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


# Update methods in the order they are applied within each step
METHODS = ("wolff", "overrelaxation", "heatbath", "metropolis")

KERNELS = {
    "wolff": kernel.wolff_heisenberg,
    "overrelaxation": kernel.overrelaxation_heisenberg,
    "heatbath": kernel.heatbath_heisenberg,
    "metropolis": kernel.metropolis_heisenberg,
}


def wolff_axis_projectors(K, mode="random", rtol=1e-6):
    """
    Projectors defining the Wolff embedding-axis distribution.

    The kernel draws one projector P uniformly and uses n = P g / |P g|
    with g a uniform random unit vector.

    Parameters
    ----------
    K : array_like
        Single-ion anisotropy tensors, shape ``(n_atoms, 3, 3)``.
    mode : {"random", "anisotropy", "mixed"}
        ``"random"``: uniformly random axes (P = I). ``"anisotropy"``: axes
        drawn within the eigenspaces of K (a random direction in a
        degenerate plane), so cluster reflections leave the anisotropy
        energy unchanged. ``"mixed"``: both, with equal weight per
        projector; it falls back to ``"random"`` for isotropic K.
    rtol : float
        Relative tolerance for degeneracy and shared-frame checks.

    Returns
    -------
    projectors : ndarray
        Shape ``(n_proj, 3, 3)``.

    Raises
    ------
    ValueError
        For ``"anisotropy"`` with isotropic K, or when the anisotropic sites
        do not share principal axes.
    """
    identity = np.eye(3)[None]
    if mode == "random":
        return identity.copy()
    if mode not in ("anisotropy", "mixed"):
        raise ValueError(f"Unknown wolff_axes mode {mode!r}")

    K = np.asarray(K, dtype=float)
    K = 0.5 * (K + np.swapaxes(K, 1, 2))
    traceless = K - np.trace(K, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    norms = np.linalg.norm(traceless, axis=(1, 2))
    scale = np.linalg.norm(K, axis=(1, 2)).max(initial=0.0)
    anisotropic = norms > rtol * max(scale, np.finfo(float).tiny)

    if not anisotropic.any():
        if mode == "mixed":
            return identity.copy()
        raise ValueError(
            "wolff_axes='anisotropy' needs anisotropic single-ion tensors K"
        )

    # A generic combination of commuting tensors has their common eigenbasis,
    # degenerate only where all of them are.
    weights = np.random.default_rng(0).uniform(0.5, 1.5, size=len(K))
    combo = np.einsum("i,ijk->jk", weights * anisotropic, traceless)
    vals, vecs = np.linalg.eigh(combo)
    spread = vals.max() - vals.min()
    groups = [[0]]
    for a in range(1, 3):
        if vals[a] - vals[groups[-1][-1]] > rtol * spread:
            groups.append([a])
        else:
            groups[-1].append(a)
    projectors = np.array([vecs[:, g] @ vecs[:, g].T for g in groups])

    for i in np.flatnonzero(anisotropic):
        Ki = K[i]
        for P in projectors:
            c = np.trace(P @ Ki @ P) / np.trace(P)
            residual = np.linalg.norm(P @ Ki @ P - c * P) + np.linalg.norm(
                P @ Ki @ (np.eye(3) - P)
            )
            if residual > rtol * np.linalg.norm(Ki):
                raise ValueError(
                    "Single-ion tensors do not share principal axes; "
                    "use wolff_axes='random'"
                )

    if mode == "mixed":
        projectors = np.concatenate([projectors, identity])
    return np.ascontiguousarray(projectors)


# Hamiltonian and lattice arrays, sent to each worker once at pool start
_WORKER = {}


def _init_worker(static):
    _WORKER.clear()
    _WORKER.update(static)


def _run_schedule(i, s, E, beta, counts, seed):
    """
    Apply one step of the update schedule to replica ``i`` in a worker.

    ``counts[m]`` is the number of sweeps (Wolff: clusters) of
    ``METHODS[m]``; each kernel call gets a fresh seed from ``seed``.

    Returns ``(i, s, E, stats)`` with ``stats[m] = (calls, accepted,
    attempted, seconds)``.
    """
    w = _WORKER
    deltas = (w["delta_atoms"], w["delta_ions"], w["delta_bonds"])
    params = (
        w["nb_offsets"],
        w["nb_atom"],
        w["nb_ijk"],
        w["nb_J"],
        w["K"],
        w["H"],
        w["g"],
        w["S"],
        muB,
    )
    rng = np.random.default_rng(seed)
    stats = np.zeros((len(METHODS), 4))

    for m, method in enumerate(METHODS):
        n = int(counts[m])
        if n == 0:
            continue
        func = KERNELS[method]
        n_calls = n if method == "wolff" else 1
        seeds = rng.integers(0, 2**32, size=n_calls, dtype=np.uint64)
        accepted = attempted = 0
        t0 = time.perf_counter()
        if method == "wolff":
            for c in range(n):
                _, s, E, n_acc, n_try = func(
                    i,
                    s,
                    *deltas,
                    beta,
                    E,
                    *params,
                    int(seeds[c]),
                    w["axis_projectors"],
                )
                accepted += n_acc
                attempted += n_try
            calls = n
        else:
            _, s, E, accepted, attempted = func(
                i, s, *deltas, beta, E, n, *params, int(seeds[0])
            )
            calls = 1
        stats[m] = calls, accepted, attempted, time.perf_counter() - t0

    return i, s, E, stats


# Upper-triangle (i, j) pairs in the column order used by save_results
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class MonteCarlo:
    """Replica-exchange Monte Carlo simulation."""

    def __init__(self, crystal, T=[10, 300], n_replicas=30, seed=None):
        self.crystal = crystal

        self.T = np.linspace(*T, n_replicas)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_n_replicas(self):
        return len(self.T)

    def kernel_seeds(self, n_replicas):
        """
        Fresh, independent seeds for one kernel call on each replica.

        Kernels reseed numba's generator on entry. Chaining the seed each
        kernel returns is a map on 32-bit integers that falls into a cycle
        after ~1e4 calls and then replays the same random stream, so seeds
        are always drawn from the parent generator instead.
        """
        return self.rng.integers(0, 2**32, size=n_replicas, dtype=np.uint64)

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
        - /state     : Markov chain state (T/beta, spins, energies, RNG, step)
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
            meta.attrs["version"] = 2
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
            state.attrs["rng_state_json"] = json.dumps(
                self.rng.bit_generator.state
            )
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
            if getattr(self.crystal, "K_dipolar", None) is not None:
                _ds(material, "K_dipolar", np.asarray(self.crystal.K_dipolar))

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

            series = f.create_group("series")
            for key, values in self.get_series().items():
                if values is not None:
                    _ds(series, key, values)

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
            version = int(meta.attrs.get("version", 0))
            if version not in (1, 2):
                raise ValueError("Unsupported checkpoint version")

            state = f["state"]
            self.T = np.array(state["T"])
            self.beta = np.array(state["beta"])
            self.E = np.array(state["E"])
            # Version 1 stored chained per-replica seeds; those are not
            # reusable, so such checkpoints continue with a fresh generator.
            if "rng_state_json" in state.attrs:
                bit_generator = np.random.PCG64()
                bit_generator.state = json.loads(
                    state.attrs["rng_state_json"]
                )
                self.rng = np.random.Generator(bit_generator)
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
                # Dipolar bonds are part of nb_J; restore their self term too.
                self.crystal.K_dipolar = (
                    np.array(material["K_dipolar"])
                    if "K_dipolar" in material
                    else None
                )

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

            self.series = {"E": [], "M": [], "I": []}
            if "series" in f:
                for key in self.series:
                    if key in f["series"]:
                        self.series[key] = list(np.array(f["series"][key]))

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
                if self.rng.random() < np.exp(-d):
                    self.s[i], self.s[j] = self.s[j].copy(), self.s[i].copy()
                    self.E[i], self.E[j] = self.E[j], self.E[i]

    def _per_replica(self, n, name):
        """Broadcast a sweep count (int or one per temperature) to replicas."""
        n_replicas = self.get_n_replicas()
        counts = np.asarray(n)
        if counts.ndim == 0:
            counts = np.full(n_replicas, counts)
        if counts.shape != (n_replicas,):
            raise ValueError(
                f"{name} must be an int or have one entry per temperature "
                f"({n_replicas}), got shape {counts.shape}"
            )
        if np.any(counts < 0) or np.any(counts != np.round(counts)):
            raise ValueError(f"{name} must be non-negative integers")
        return counts.astype(np.int64)

    def _reset_stats(self):
        n_replicas = self.get_n_replicas()
        self.stats = {"n_steps": 0, "kernel_max": 0.0, "kernel_mean": 0.0}
        for method in METHODS:
            self.stats[method] = {
                "calls": np.zeros(n_replicas, dtype=np.int64),
                "accepted": np.zeros(n_replicas, dtype=np.int64),
                "attempted": np.zeros(n_replicas, dtype=np.int64),
                "kernel_time": np.zeros(n_replicas),
            }
        self.stats["wall_time"] = {
            "updates": 0.0,
            "exchange": 0.0,
            "measurement": 0.0,
        }

    def update_replicas(self, schedule, record=False):
        """
        Apply one step of the update schedule to every replica.

        One task per replica runs all of its methods (in ``METHODS`` order)
        inside a worker, so each step costs a single round trip to the pool.

        Parameters
        ----------
        schedule : ndarray of int
            Shape ``(n_replicas, len(METHODS))``: sweeps (Wolff: clusters)
            of each method per replica.
        record : bool
            Accumulate kernel time, acceptance and the wall time of the
            phase into ``self.stats``.
        """
        n_replicas = self.get_n_replicas()
        seeds = self.kernel_seeds(n_replicas)
        tasks = [
            (i, self.s[i], self.E[i], self.beta[i], schedule[i], int(seeds[i]))
            for i in range(n_replicas)
        ]

        t0 = time.perf_counter()
        results = self.pool.starmap(_run_schedule, tasks)
        elapsed = time.perf_counter() - t0

        kernel_total = np.zeros(n_replicas)
        for i, s, E, stats in results:
            self.s[i] = s
            self.E[i] = E
            kernel_total[i] = stats[:, 3].sum()
            if record:
                for m, method in enumerate(METHODS):
                    entry = self.stats[method]
                    entry["calls"][i] += int(stats[m, 0])
                    entry["accepted"][i] += int(stats[m, 1])
                    entry["attempted"][i] += int(stats[m, 2])
                    entry["kernel_time"][i] += stats[m, 3]

        if record:
            self.stats["wall_time"]["updates"] += elapsed
            self.stats["kernel_max"] += kernel_total.max()
            self.stats["kernel_mean"] += kernel_total.mean()

    def timing_summary(self):
        """
        Cost and acceptance of each update method per temperature.

        Accumulated over production steps only (after thermalization, which
        also absorbs numba compilation), and restarted on resume.

        Returns
        -------
        summary : dict or None
            ``"n_steps"``: production steps recorded. For each method in
            ``METHODS``: ``"calls_per_step"``, ``"kernel_time_per_step"``
            (seconds inside the worker, per temperature) and
            ``"acceptance"``; for Wolff also ``"cluster_size"``.
            Wall seconds per step: ``"updates"`` (the parallel update
            phase), ``"exchange"``, ``"measurement"`` (amortized over
            ``sample_interval``) and ``"total_wall_time_per_step"``. The
            update phase splits into ``"kernel_mean"`` (average kernel time
            of a replica), ``"load_imbalance"`` (slowest replica minus the
            average) and ``"dispatch_overhead"`` (the rest: transfer and
            scheduling).
        """
        stats = getattr(self, "stats", None)
        if stats is None or stats["n_steps"] == 0:
            return None

        n_steps = stats["n_steps"]
        wall = stats["wall_time"]
        summary = {"n_steps": n_steps}
        for method in METHODS:
            st = stats[method]
            with np.errstate(invalid="ignore", divide="ignore"):
                acceptance = st["accepted"] / st["attempted"]
                entry = {
                    "calls_per_step": st["calls"] / n_steps,
                    "kernel_time_per_step": st["kernel_time"] / n_steps,
                    "acceptance": np.where(
                        st["attempted"] > 0, acceptance, np.nan
                    ),
                }
                if method == "wolff":
                    entry["cluster_size"] = np.where(
                        st["calls"] > 0, st["attempted"] / st["calls"], np.nan
                    )
            summary[method] = entry
        summary["updates"] = wall["updates"] / n_steps
        summary["kernel_mean"] = stats["kernel_mean"] / n_steps
        summary["load_imbalance"] = (
            stats["kernel_max"] - stats["kernel_mean"]
        ) / n_steps
        summary["dispatch_overhead"] = (
            wall["updates"] - stats["kernel_max"]
        ) / n_steps
        summary["exchange"] = wall["exchange"] / n_steps
        summary["measurement"] = wall["measurement"] / n_steps
        summary["total_wall_time_per_step"] = sum(wall.values()) / n_steps
        return summary

    def sample_parameters(self, hkl):
        n_sites = self.crystal.get_total_sites()

        M = self.crystal.net_moment()
        self.series["E"].append(self.E / n_sites)
        self.series["M"].append(M / n_sites)

        self.M_sum += M / n_sites
        self.M_sq_sum += M[:, :, None] * M[:, None, :] / n_sites**2

        self.E_sum += self.E / n_sites
        self.E_sq_sum += self.E**2 / n_sites**2

        if hkl is not None:
            struct_fact = StructureFactor(self.crystal)
            I = struct_fact.magnetic_intensity(hkl)
            self.series["I"].append(np.array(I, dtype=float))
            self.I_sum += I
            self.I_sq_sum += I**2

        C_ij = correlations.vector_vector(self.s)

        self.C_ij_sum += C_ij
        self.C_ij_sq_sum += C_ij**2

    def ensemble_average(self, n_sample):
        # Accumulators hold per-site e = E/N and m = M/N, so the per-site
        # fluctuation quantities are C = kB beta^2 N var(e) and
        # chi = beta N var(m); both are size independent away from T_c.
        n_sites = self.crystal.get_total_sites()

        M_ave = self.M_sum / n_sample
        M_sq_ave = self.M_sq_sum / n_sample

        M_var = M_sq_ave - np.einsum("ri,rj->rij", M_ave, M_ave)
        M_std = np.sqrt(M_var[:, np.arange(3), np.arange(3)])

        chi = n_sites * self.beta[:, None, None] * M_var
        chi = 0.5 * (chi + np.swapaxes(chi, 1, 2))

        E_ave = self.E_sum / n_sample
        E_sq_ave = self.E_sq_sum / n_sample

        E_var = E_sq_ave - E_ave**2
        E_std = np.sqrt(E_var)

        C = n_sites * kB * self.beta**2 * E_var

        I_ave = None
        I_std = None
        if self.I_sum is not None:
            I_ave = self.I_sum / n_sample
            I_sq_ave = self.I_sq_sum / n_sample

            I_std = np.sqrt(I_sq_ave - I_ave**2)

        C_ij_ave = self.C_ij_sum / n_sample
        C_ij_sq_ave = self.C_ij_sq_sum / n_sample
        C_ij_std = np.sqrt(np.maximum(C_ij_sq_ave - C_ij_ave**2, 0.0))

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

        parameters.update(self.error_analysis(n_sample))
        parameters["timing"] = self.timing_summary()

        return parameters

    def get_series(self):
        """
        Recorded time series, one entry per sample, indexed by temperature.

        Returns
        -------
        series : dict
            ``"E"``: energy per site, shape ``(n_samples, n_replicas)``;
            ``"M"``: moment per site, ``(n_samples, n_replicas, 3)``;
            ``"I"``: intensities at ``hkl``, ``(n_samples, n_replicas,
            n_hkl)``, or ``None`` if no ``hkl`` was given.
        """
        series = getattr(self, "series", None) or {}
        out = {}
        for key in ("E", "M", "I"):
            values = series.get(key, [])
            out[key] = np.asarray(values) if len(values) > 0 else None
        return out

    @staticmethod
    def _jackknife_blocks(n, tau):
        """Number of jackknife blocks so each block spans ~8 tau_int."""
        block_len = max(1, int(np.ceil(8.0 * tau)))
        return int(np.clip(n // block_len, 2, 50))

    def error_analysis(self, n_sample=None):
        """
        Autocorrelation times and statistical errors from the time series.

        Errors of means are corrected by the integrated autocorrelation time
        (``2 * tau_int * var / N``). Errors of fluctuation quantities (C, chi)
        use a blocked jackknife with blocks of ~8 tau_int per temperature.

        Returns an empty dict when the time series are unavailable (e.g.
        after resuming from a version 1 checkpoint) or do not match the
        number of accumulated samples.
        """
        series = self.get_series()
        E_series, M_series = series["E"], series["M"]
        if E_series is None or (
            n_sample is not None and len(E_series) != n_sample
        ):
            return {}

        n, n_replicas = E_series.shape
        n_sites = self.crystal.get_total_sites()

        E_ave, E_err, tau_E = statistics.mean_error(E_series)
        M_ave, M_err, tau_M = statistics.mean_error(M_series)
        tau_E2, _ = statistics.integrated_autocorrelation_time(E_series**2)
        MM_series = M_series[..., :, None] * M_series[..., None, :]

        C_err = np.zeros(n_replicas)
        chi_err = np.zeros((n_replicas, 3, 3))
        for r in range(n_replicas):
            beta = self.beta[r]

            n_blocks = self._jackknife_blocks(n, max(tau_E[r], tau_E2[r]))
            _, C_err[r] = statistics.jackknife(
                lambda e, e2: n_sites * kB * beta**2 * (e2 - e**2),
                E_series[:, r],
                E_series[:, r] ** 2,
                n_blocks=n_blocks,
            )

            n_blocks = self._jackknife_blocks(n, tau_M[r].max())
            _, err = statistics.jackknife(
                lambda m, mm: n_sites * beta * (mm - np.outer(m, m)),
                M_series[:, r],
                MM_series[:, r],
                n_blocks=n_blocks,
            )
            chi_err[r] = 0.5 * (err + err.T)

        out = {
            "n_samples": n,
            "E(err)": E_err,
            "tau(E)": tau_E,
            "tau(E^2)": tau_E2,
            "M(err)": M_err,
            "tau(M)": tau_M,
            "C(err)": C_err,
            "chi(err)": chi_err,
            "I(err)": None,
            "tau(I)": None,
            "series": series,
        }

        if series["I"] is not None and len(series["I"]) == n:
            _, out["I(err)"], out["tau(I)"] = statistics.mean_error(
                series["I"]
            )

        return out

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
        sample_interval=1,
        wolff_axes="random",
        checkpoint_interval=None,
        checkpoint_final=None,
        checkpoint_path=None,
        resume_from=None,
        outdir="checkpoints",
        prefix="mc",
    ):
        """
        Replica-exchange Monte Carlo over the temperature grid ``self.T``.

        Each outer step applies Wolff, overrelaxation, heatbath and
        Metropolis updates (in that order) to every replica, then attempts
        replica exchanges; after ``n_thermal`` steps one sample is recorded
        every ``sample_interval`` steps (autocorrelation times are then in
        units of samples).

        Each ``n_*_sweeps`` is an int (same at every temperature) or a
        sequence with one entry per temperature, so the update mix can be
        tuned per temperature. Replica ``i`` always holds temperature
        ``T[i]`` because exchanges swap configurations. Sweep methods count
        lattice sweeps; ``n_cluster_sweeps`` counts Wolff clusters. Every
        temperature needs at least one Metropolis, heatbath or Wolff update
        (with ``wolff_axes="anisotropy"``, Metropolis or heatbath).

        ``wolff_axes`` sets the Wolff embedding-axis distribution: see
        :func:`wolff_axis_projectors`. ``"anisotropy"`` keeps cluster flips
        from paying single-ion anisotropy energy, which matters for large
        clusters in anisotropic magnets at low temperature.

        Returns the ensemble averages (see :meth:`ensemble_average`),
        including statistical errors (:meth:`error_analysis`) and per-method
        cost and acceptance under ``"timing"`` (:meth:`timing_summary`).
        """
        assert n_outer > 0
        assert sample_interval >= 1

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

            self.series = {"E": [], "M": [], "I": []}

            self.crystal.initialize_random_spin_configurations(
                n_replicas, rng=self.rng
            )

            self.s = self.crystal.get_spin_vectors()
            self.E = np.zeros(n_replicas)

        schedule = {
            "wolff": self._per_replica(n_cluster_sweeps, "n_cluster_sweeps"),
            "overrelaxation": self._per_replica(
                n_overrelaxation_sweeps, "n_overrelaxation_sweeps"
            ),
            "heatbath": self._per_replica(
                n_heatbath_sweeps, "n_heatbath_sweeps"
            ),
            "metropolis": self._per_replica(n_local_sweeps, "n_local_sweeps"),
        }
        # Wolff restricted to anisotropy axes alone is not ergodic
        ergodic = schedule["heatbath"] + schedule["metropolis"]
        if wolff_axes != "anisotropy":
            ergodic = ergodic + schedule["wolff"]
        if np.any(ergodic == 0):
            raise ValueError(
                "Every temperature needs at least one Metropolis or heatbath "
                "update per step, or a Wolff update with random axes "
                "(overrelaxation alone is not ergodic); "
                f"missing at T = {self.T[ergodic == 0]}"
            )
        self._reset_stats()

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

        static = dict(
            axis_projectors=wolff_axis_projectors(K, wolff_axes),
            delta_atoms=delta_atoms,
            delta_ions=delta_ions,
            delta_bonds=delta_bonds,
            nb_offsets=nb_offsets,
            nb_atom=nb_atom,
            nb_ijk=nb_ijk,
            nb_J=nb_J,
            K=K,
            H=H,
            g=g,
            S=S,
        )
        schedule_matrix = np.stack([schedule[m] for m in METHODS], axis=1)

        with Pool(
            processes=n_replicas, initializer=_init_worker, initargs=(static,)
        ) as self.pool:
            last_i_outer = i_outer_start - 1
            for i_outer in range(i_outer_start, n_outer):
                last_i_outer = i_outer
                print(f"{i_outer}/{n_outer}")

                record = i_outer >= n_thermal
                sample = record and (i_outer - n_thermal) % sample_interval == 0

                self.update_replicas(schedule_matrix, record)

                t0 = time.perf_counter()
                self.replica_exchange()
                t1 = time.perf_counter()

                if sample:
                    self.crystal.set_spin_vectors(self.s)
                    self.sample_parameters(hkl)
                    self.n_samples_accumulated += 1

                if record:
                    wall = self.stats["wall_time"]
                    wall["exchange"] += t1 - t0
                    wall["measurement"] += time.perf_counter() - t1
                    self.stats["n_steps"] += 1

                if sample:
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

        columns = [T, chi_11, chi_22, chi_33, chi_23, chi_13, chi_12]
        header = "T chi_11 chi_22 chi_33 chi_23 chi_13 chi_12"
        if "chi(err)" in result:
            err = result["chi(err)"]
            columns += [err[:, i, j] for i, j in VOIGT]
            header += " " + " ".join(f"chi_{i+1}{j+1}_err" for i, j in VOIGT)

        np.savetxt(
            filename + "_susceptibility.txt",
            np.column_stack(columns),
            header=header,
        )

        Mx = result["M(ave)"][:, 0]
        My = result["M(ave)"][:, 1]
        Mz = result["M(ave)"][:, 2]
        Mx_std = result["M(std)"][:, 0]
        My_std = result["M(std)"][:, 1]
        Mz_std = result["M(std)"][:, 2]

        columns = [T, Mx, My, Mz, Mx_std, My_std, Mz_std]
        header = "T Mx My Mz Mx_std My_std Mz_std"
        if "M(err)" in result:
            columns += list(result["M(err)"].T) + list(result["tau(M)"].T)
            header += " Mx_err My_err Mz_err tau_Mx tau_My tau_Mz"

        np.savetxt(
            filename + "_magnetization.txt",
            np.column_stack(columns),
            header=header,
        )

        E = result["E(ave)"]
        E_std = result["E(std)"]

        columns = [T, E, E_std]
        header = "T E E_std"
        if "E(err)" in result:
            columns += [result["E(err)"], result["tau(E)"]]
            header += " E_err tau_E"

        np.savetxt(
            filename + "_energy.txt",
            np.column_stack(columns),
            header=header,
        )

        C = result["C"]

        columns = [T, C]
        header = "T C"
        if "C(err)" in result:
            columns += [result["C(err)"]]
            header += " C_err"

        np.savetxt(
            filename + "_heat_capacity.txt",
            np.column_stack(columns),
            header=header,
        )

        if result["I(ave)"] is not None:
            I = result["I(ave)"][:, 0]
            sig = result["I(std)"][:, 0]

            np.savetxt(
                filename + "_intensity.txt",
                np.column_stack((T, I, sig)),
                header="T I sig",
            )
