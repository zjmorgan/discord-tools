import numpy as np

from multiprocessing import Pool

from discord.parameters.constants import kB

class MonteCarlo:
    
    def __init__(self, crystal, T=[0, 300], n_replicas=31):
        self.crystal = crystal

        self.T = np.linspace(*T, n_replicas)

    def get_n_replicas(self):
        return len(self.T)

    def make_seeds(self, n_replicas):
        root = np.random.SeedSequence()
        children = root.spawn(n_replicas)
        return [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]

    def replica_exchange(self, beta, E, S, seeds):
        n_replica = self.get_n_replicas()
        for offset in (0, 1):
            for i in range(offset, n_replica - 1, 2):
                j = i_replica + 1
                beta0, beta1 = betas[i], beta[j]
                E0, E1 = E[i], E[j]
                d = (beta0 - beta1) * (E1 - E0)
                if np.random.rand() < np.exp(-d):
                    S[i], S[j] = S[j], S[i]
                    E[i], E[j] = E[j], E[i]
                    seeds[i], seeds[j] = seeds[j], seeds[i]
    
    def parallel_tempering(self)
        S_reps, 
        nb_offsets, nb_m, nb_di, nb_dj, nb_dk, nb_J,
        E_reps, K, H, g,
        factors, h, k, l, x, y, z, N1, N2, N3, G,
        n_local_sweeps=2, n_outer=10000, n_therm=7000,
    ):
        beta = 1.0 / (kB * self.Ts + np.finfo(float).eps)
        n_replicas = self.get_n_replicas()
    
        # store magnetization per site (vector) at each T
        M_samples = [[] for _ in range(R)]
        # store intensity samples at each T
        I_samples = [[] for _ in range(R)]
        # store energy samples at each T (for C(T))
        E_samples = [[] for _ in range(R)]
    
        with Pool(processes=n_replicas) as pool:
            for i_outer in range(n_outer):
                print(f"{i_outer}/{n_outer}")
    
                args = [
                    (
                        r,
                        S_reps[r].copy(),
                        betas[r],
                        float(E_reps[r]),
                        N_local_sweeps,
                        nb_offsets, nb_m, nb_di, nb_dj, nb_dk, nb_J,
                        K, H, muB, g,
                        seeds[r],
                    )
                    for r in range(R)
                ]
    
                results = pool.map(metropolis, args)
                results.sort(key=lambda x: x[0])
    
                for r, S_new, E_new, seed in results:
                    S_reps[r] = S_new
                    E_reps[r] = E_new
                    seeds[r] = seed
    
                # Swap neighboring replicas
                for offset in (0, 1):
                    for r in range(offset, R - 1, 2):
                        beta_r, beta_r1 = betas[r], betas[r + 1]
                        E_r, E_r1 = E_reps[r], E_reps[r + 1]
                        d = (beta_r - beta_r1) * (E_r1 - E_r)
                        if np.random.rand() < np.exp(-d):
                            S_reps[r], S_reps[r + 1] = S_reps[r + 1], S_reps[r].copy()
                            E_reps[r], E_reps[r + 1] = E_reps[r + 1], E_reps[r]
                            seeds[r], seeds[r + 1] = seeds[r + 1], seeds[r]
    
                # Measurements after thermalization
                if outer >= N_therm:
                    for r in range(R):
                        # magnetization per site (dimensionless spins)
                        M = S_reps[r].mean(axis=(0, 1, 2, 3))  # shape (3,)
                        M_samples[r].append(M)
    
                        # intensity sample for this config
                        S_cfg = S_reps[r]          # (n, N1, N2, N3, 3)
                        Mx_cfg = S_cfg[..., 0]     # (n, N1, N2, N3)
                        My_cfg = S_cfg[..., 1]
                        Mz_cfg = S_cfg[..., 2]
    
                        Fx, Fy, Fz = magnetic_structure_factor(
                            factors,
                            Mx_cfg, My_cfg, Mz_cfg,
                            h, k, l,
                            x, y, z,
                            N1, N2, N3,
                            G,
                        )
                        I_hkl = magnetic_intensity(h, k, l, Fx, Fy, Fz, G)[0]
                        I_samples[r].append(I_hkl)
    
                        # energy sample for specific heat
                        E_samples[r].append(E_reps[r])
    
        # convert samples -> averages for M and I
        M_mean = []
        MM_mean = []   # <M_i M_j> per site
        I_mean = []
        I_err  = []
    
        for r in range(R):
            arrM = np.array(M_samples[r])      # (Nsamples, 3)
            M_mean.append(arrM.mean(axis=0))
            MM_mean.append(
                (arrM[:, :, None] * arrM[:, None, :]).mean(axis=0)
            )
    
            arrI = np.array(I_samples[r])      # (Nsamples,)
            if arrI.size > 0:
                I_mean.append(arrI.mean())
                if arrI.size > 1:
                    var = arrI.var(ddof=1)
                    I_err.append(np.sqrt(var / arrI.size))
                else:
                    I_err.append(0.0)
            else:
                I_mean.append(0.0)
                I_err.append(0.0)
    
        M_mean = np.array(M_mean)     # (R, 3)
        MM_mean = np.array(MM_mean)   # (R, 3, 3)
        I_mean = np.array(I_mean)     # (R,)
        I_err  = np.array(I_err)      # (R,)
    
        # specific heat from energy fluctuations:
        # C(T) = β^2 / N_sites * ( <E^2> - <E>^2 ), per spin
        N_sites = (
            S_reps.shape[1] *
            S_reps.shape[2] *
            S_reps.shape[3] *
            S_reps.shape[4]
        )
    
        E_mean = np.zeros(R, dtype=float)
        C_per_site = np.zeros(R, dtype=float)
    
        for r in range(R):
            arrE = np.array(E_samples[r], dtype=float)
            if arrE.size > 0:
                E_mean[r] = arrE.mean()
                E2_mean = (arrE ** 2).mean()
                C_per_site[r] = (betas[r] ** 2 / N_sites) * (E2_mean - E_mean[r] ** 2)
            else:
                E_mean[r] = E_reps[r]
                C_per_site[r] = 0.0
    
        return (
            np.array(Ts),
            M_mean,
            MM_mean,
            I_mean,
            I_err,
            S_reps,
            E_reps,
            E_mean,
            C_per_site,
        )
