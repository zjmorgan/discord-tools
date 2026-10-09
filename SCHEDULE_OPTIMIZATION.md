# Automatic Update-Schedule Optimization (protocol spec, draft)

Goal: for a given system and temperature grid, choose the Monte Carlo update mix at each temperature so that parallel tempering reaches target error bars on thermodynamic observables (C, χ, order parameter) in the least wall time, without hand tuning. Pilot, select, freeze, then production.

Status: design agreed. Implemented so far: the infrastructure in [MONTE_CARLO_METHODS.md](MONTE_CARLO_METHODS.md) (per-temperature schedules, τ_int and error estimates, per-method timing), Wolff cluster sweeps (plan item 1) and the automatic order parameter (plan item 3). Next: the pilot/selection module (item 2).

## 1. Objective

For an observable $O$ at temperature $T_r$, a run of $N$ steps gives a squared error $\varepsilon^2 \approx 2\tau_{O,r}\,\mathrm{var}(O_r)/N$ (τ in steps). All temperatures advance together, so the wall time to reach targets $\varepsilon_{O,r}$ everywhere is

$$W = t_\text{step}\cdot \max_{r,O}\; \frac{2\,\tau_{O,r}\,\mathrm{var}(O_r)}{\varepsilon_{O,r}^2}.$$

The variances are properties of the system, not of the schedule, so the schedule only enters through

$$J = t_\text{step}\cdot \max_{r,O} w_{O,r}\,\tau_{O,r}, \qquad w_{O,r} = \mathrm{var}(O_r)/\varepsilon_{O,r}^2,$$

with the weights estimated once in the pilot. Default: equal relative targets for C and χ at every temperature.

Replicas run in parallel, so the step time is set by the slowest replica, not the sum:

$$t_\text{step} = o + \max_r c_r(s_r) + t_\text{exchange} + t_\text{measure}/k,$$

where $c_r(s_r)$ is the kernel time of schedule $s_r$ at temperature $r$, $o$ the dispatch overhead, and $k$ the sample interval (all reported by `result["timing"]`).

Consequence: extra work at an "easy" temperature is free up to the slowest replica's time, and useless once that temperature is already uncorrelated (τ ≈ 1/2).

## 2. Decomposition: per-temperature recipe + one global budget

A **recipe** $s$ is a small unit of work: one ergodic canonical move (Metropolis or heatbath) plus $n_\text{OR}$ overrelaxation sweeps plus $f_W$ Wolff "cluster sweeps" (clusters until $f_W N$ sites have been flipped, following Gouvêa et al.). The schedule at temperature $r$ repeats its recipe $m_r$ times per step.

Cost-normalized efficiency of recipe $s$ at temperature $r$, measured with the recipe applied once per step:

$$\eta_r(s) = \frac{1}{\tau_r(s)\,c_r(s)}, \qquad \tau_r = \max_O w_{O,r}\,\tau_{O,r}.$$

If repeating a recipe $m$ times divides τ by $m$ (true when local dynamics, not replica diffusion, limits τ), then giving every replica the same kernel budget $b$ per step yields $\tau_r \approx 1/(\eta_r b)$, and

$$J(b) \approx \frac{o + b + t_\text{exchange} + t_\text{measure}/k}{b \cdot \min_r \eta_r(s_r)}.$$

So the problem separates:

1. at each temperature, pick $s_r^* = \arg\max_s \eta_r(s)$;
2. pick one scalar budget $b$ (equivalently the repeat counts $m_r = \lfloor b / c_r(s_r^*) \rfloor$, at least 1) by a 1-D search on $J(b)$.

Step 2 must be checked in real runs, because the $1/m$ model fails once τ is limited by round trips in temperature space (more work per step does not make replicas diffuse faster; Bittner et al. 2008), and τ cannot go below 1/2.

## 3. Candidate recipes

- Ergodic base: Metropolis or heatbath (one sweep).
- Overrelaxation: $n_\text{OR} \in \{0, 1, 2, 4, 8, 16\}$.
- Wolff: $f_W \in \{0, 1/4, 1, 2\}$ cluster sweeps.
- Wolff axis mode: one global choice between `random` and `mixed`. No prior knowledge of the anisotropy is assumed: `mixed` is built from K automatically (easy axis and easy plane plus random axes, falling back to `random` for isotropic K), so it also covers easy-axis to easy-plane crossovers with temperature. Pure `anisotropy` axes are never safer than `mixed` and are not searched. The pilot runs the Wolff recipes under both modes and keeps the mode with the better $J$ after per-temperature selection.

About 2 × 6 × 4 = 48 recipes per axis mode (the 12 without Wolff are shared), fewer after excluding Wolff for systems where clusters percolate trivially (frustrated lattices) or never grow. The grid is deliberately coarse: τ estimates are noisy (Section 4), so finer resolution is not measurable.

## 4. Pilot design

1. **Shared thermalization.** One PT run with a default recipe until equilibrated (energy drift test), then checkpoint. All pilots start from this checkpoint, so no candidate pays for equilibration.
2. **One pilot run per recipe, recipe uniform across temperatures, exchanges on.** Each run measures $\tau_{O,r}$ and $c_r$ at all R temperatures at once. Exchanges stay on because isolated low-temperature chains can be non-ergodic in practice; the cost is that a temperature's τ is slightly influenced by its neighbours' recipes (checked in step 5).
3. **Successive halving.** The relative error of a τ estimate is roughly $\sqrt{2(2W+1)/N} \approx \sqrt{24\tau/N}$, so 20 % precision needs $N \approx 600\,\tau$. Start all recipes short; after each round drop, per temperature, recipes whose η is clearly worse than the best (by more than twice the estimate's error); double the length of the survivors. A recipe stops being run once it is dropped at all temperatures.
4. **Select** $s_r^*$ per temperature, then **search the budget** $b$ over a few values (e.g. 1, 2, 4, 8 × the largest $c_r(s_r^*)$) in short joint runs, keeping the best measured $J$.
5. **Verify** with one joint run of the frozen schedule; report the measured τ and $J$, not the model's.
6. **Freeze** and run production. Adaptation never happens during production, so the production chain is an ordinary MCMC chain (no adaptive-MCMC convergence conditions needed).

Pilot cost is part of the reported total and should stay below ~10–20 % of production.

## 5. Observables

- Energy (for C) always.
- Order parameter: the structure factor at the ordering wavevector $q^*$, found automatically as the maximum of $S(q)$ from the lowest-temperature replica after thermalization (or a user-supplied `hkl`). Net M is not the slow mode for antiferromagnets.
- Magnetization components (for χ).
- τ of $E^2$ and of the order parameter's square where C and χ are the targets (already computed for E).

## 6. Benchmark harness

Score = $W$ for fixed relative targets (e.g. 1 % on C and χ at every temperature), computed from measured τ, variance and $t_\text{step}$ of a production-length run, plus the pilot cost. Several seeds per entry, speedups with errors.

| System | Why |
| --- | --- |
| 3D Heisenberg ferromagnet (SC) | Known T_c; Wolff territory |
| Easy-axis and easy-plane ferromagnets | Ising- and XY-like limits; aligned Wolff axes |
| MnF2 (rutile, easy-axis AFM) | Real material; antiferromagnetic clusters |
| Pyrochlore Heisenberg AFM | Frustrated; overrelaxation should dominate, Wolff useless |
| Site-diluted ferromagnet | Disorder |
| MnF2 with dipolar interactions (optional) | Long-range couplings change the cost balance |

Baselines:

- Metropolis only;
- 1 Metropolis + 10 overrelaxation (common Heisenberg spin-glass practice);
- the recipe table in MONTE_CARLO_METHODS.md;
- Bittner et al.: sweeps per temperature ∝ τ of the energy, single move type;
- S^z-only Wolff (Gouvêa et al.) as the reference for aligned axes.

Fixed hardware: one core per replica; report core count and lattice sizes. Temperature grids are fixed (same for all methods) in this study; temperature optimization is orthogonal and can be combined later.

## 7. Implementation plan

1. Kernel: Wolff "cluster sweep" count (clusters until $f_W N$ sites flipped).
2. `discord.atomistic.tuning`: recipes, pilot runner with checkpoint branching, successive halving, budget search, `ScheduleResult` (frozen schedule plus the measured efficiency table, saved to JSON for the paper).
3. Automatic $q^*$ and order-parameter series.
4. `benchmarks/`: system definitions, baselines, scoring, plots.

## 8. Decisions

- Weights $w_{O,r}$: equal relative error targets at every temperature.
- Both Metropolis and heatbath are recipe bases; the pilot picks per temperature.
- Wolff axis mode is global, chosen by the pilot from `random` and `mixed`; no anisotropy type is assumed.
- Pilots run at the production lattice size.

## Related work (for the paper)

- E. Bittner, A. Nußbaumer, W. Janke, PRL 101, 130603 (2008): sweeps between exchanges ∝ canonical τ of the energy, from pilot Metropolis runs; closest prior work (count of one move type, not the mix of moves).
- H. G. Katzgraber, S. Trebst, D. A. Huse, M. Troyer, J. Stat. Mech. P03018 (2006): feedback-optimized temperature sets.
- S. Syed, A. Bouchard-Côté, G. Deligiannidis, A. Doucet, JRSS-B 84, 321 (2022): non-reversible PT and adaptive schedules (the alternating even/odd exchange used here is their deterministic even–odd scheme).
- D. Zhao, N. S. Pillai (2024): policy-gradient temperature tuning.
- D. Nguyen et al., arXiv:1802.08798: automatic selection among MCMC samplers by efficiency per time (general statistics; conceptual precedent).
- J. D. Alzate-Cardona et al., JPCM 31, 095802 (2019): adaptive Metropolis cone width (tuning within one move).
- M. E. Gouvêa, G. M. Wysin et al., PRB 59, 6229 (1999): S^z-only Wolff clusters for easy-axis magnets, combined with Metropolis and overrelaxation.
- M. D'Onorio De Meo, S. K. Oh, PRB 46, 257 (1992): Wolff for anisotropic continuous spins.
- J. Kent-Dobias, J. P. Sethna, PRE 98, 063306 (2018): Wolff in arbitrary fields via a ghost transformation.
- Z. Morgan, F. Ye, JOM 74 (2022): Toward Discord.

Still to check before submission: papers citing Bittner 2008 and Gouvêa 1999.
