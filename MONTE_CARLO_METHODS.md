# Monte Carlo Methods in discord-tools

## Overview

The simulation now supports **four Monte Carlo update methods** that can be combined for efficient sampling:

### 1. Metropolis-Hastings (Local Updates)
- **Type**: Canonical (samples Boltzmann distribution)
- **Algorithm**: Proposes random spin direction, accepts with probability min(1, exp(-β·ΔE))
- **Use case**: Basic ergodic updates, good at high temperatures
- **Parameter**: `n_local_sweeps`

### 2. Wolff Cluster Updates
- **Type**: Canonical (samples Boltzmann distribution)  
- **Algorithm**: Grows and flips clusters of aligned spins
- **Use case**: Critical slowing down near phase transitions, large-scale moves
- **Parameter**: `n_cluster_sweeps`

### 3. Overrelaxation
- **Type**: Microcanonical (approximately energy-conserving)
- **Algorithm**: Reflects spin across local exchange field deterministically
- **Use case**: Faster decorrelation, especially at low temperatures
- **Note**: Not ergodic alone - must combine with Metropolis or Wolff
- **Parameter**: `n_overrelaxation_sweeps`

### 4. Heatbath (Gibbs Sampling)
- **Type**: Canonical (samples Boltzmann distribution)
- **Algorithm**: Samples new spin from conditional distribution given effective field
- **Use case**: Efficient at moderate/high temperatures, natural detailed balance
- **Parameter**: `n_heatbath_sweeps`

## Usage Example

```python
from discord.material import Crystal
from discord.atomistic.simulation import MonteCarlo

# Setup crystal
crystal = Crystal(cell, space_group, sites, S=2.5)
crystal.generate_bonds(d_cut=4.8)
# ... assign magnetic parameters ...

# Create Monte Carlo object
mc = MonteCarlo(crystal)

# Run simulation with all methods
mc.parallel_tempering(
    n_local_sweeps=1,          # Metropolis sweeps per step
    n_cluster_sweeps=1,        # Wolff cluster updates per step  
    n_overrelaxation_sweeps=1, # Overrelaxation sweeps per step
    n_heatbath_sweeps=1,       # Heatbath sweeps per step
    n_outer=1000,              # Total MC steps
    n_thermal=700,             # Thermalization steps
)
```

## Recommended Combinations

### Fast equilibration (high T):
```python
n_local_sweeps=1
n_heatbath_sweeps=2  
n_overrelaxation_sweeps=1
n_cluster_sweeps=0
```

### Near critical point:
```python
n_cluster_sweeps=2
n_local_sweeps=1
n_overrelaxation_sweeps=1
n_heatbath_sweeps=0
```

### Low temperature:
```python
n_cluster_sweeps=1
n_overrelaxation_sweeps=2
n_local_sweeps=1
n_heatbath_sweeps=1
```

## Hamiltonian and Units

With unit spin vectors **s**ᵢ and S̃ᵢ² = Sᵢ(Sᵢ+1):

E = −½ Σᵢ Σⱼ S̃ᵢ² **s**ᵢ·Jᵢⱼ**s**ⱼ − Σᵢ S̃ᵢ² **s**ᵢ·Kᵢ**s**ᵢ − Σᵢ gᵢ μ_B S̃ᵢ **s**ᵢ·**H**

Energies are in meV (`kB` in meV/K, `muB` in meV/T), fields in T, and moments gᵢ S̃ᵢ **s**ᵢ in μ_B. Reported `E` is meV/site, `M` is μ_B/site, `C` is meV/K/site and `chi` is μ_B²/meV/site.

## Energy Tracking

All methods maintain accurate incremental energy tracking with errors at floating-point precision (~10⁻¹⁵).

## Error Analysis

Each sample records the energy and moment per site (and intensities at `hkl`) for every temperature; they are returned as `result["series"]` and stored in checkpoints. From these, `parallel_tempering` also returns:

- `tau(E)`, `tau(E^2)`, `tau(M)`, `tau(I)`: integrated autocorrelation times (Sokal automatic windowing, convention τ_int = 1/2 for uncorrelated samples), in units of outer steps
- `E(err)`, `M(err)`, `I(err)`: errors of the means, `sqrt(2 τ_int var / N)`
- `C(err)`, `chi(err)`: blocked-jackknife errors with blocks of ~8 τ_int

`C` and `chi` are per site, `C = k_B β² N var(e)` and `chi = β N var(m)` with `e = E/N` and `m = M/N`, so results from different supercells can be compared directly (they coincide away from T_c, and the peaks grow with N near it).

`E(std)`, `M(std)` and `I(std)` remain the standard deviation of the distribution, not the error of the mean. The estimators live in `discord.atomistic.statistics`.

## Random Numbers

`MonteCarlo(..., seed=...)` makes a run reproducible. A single generator in the parent process supplies a fresh seed to every kernel call and drives replica exchange; its state is saved in checkpoints (format version 2), so a resumed run continues exactly as an uninterrupted one.

## Correctness Tests

`tests/test_kernel_equivalence.py` checks that every kernel samples the Boltzmann distribution, not just that its energy bookkeeping is consistent. Each schedule is run as a fixed-temperature chain and compared against an exact quadrature result (two-spin system) and a Metropolis-only reference (4×4×4 simple-cubic lattice near T_c), for both isotropic and anisotropic (exchange anisotropy + easy axis + field) Hamiltonians. Any new update method should be added to these tests before it is used.

## Order of Operations

Each MC step executes methods in this order:
1. Wolff cluster updates
2. Overrelaxation sweeps  
3. Metropolis-Hastings sweeps
4. Heatbath sweeps
5. Replica exchange (parallel tempering)

## Technical Notes

- **Overrelaxation** preserves exchange energy exactly (isotropic systems) but modifies anisotropy/Zeeman energy
- **Heatbath** uses rejection sampling for the cone distribution at low temperatures
- **Wolff** bond probabilities include the S(S+1) factor, and the cluster is accepted with min(1, exp(-β(ΔE − W))), where W is the boundary-bond Hastings term. For isotropic ferromagnetic exchange W cancels the exchange part of ΔE, so only anisotropy and field are MH-corrected. Bonds inside the cluster are counted once in ΔE.
- All methods respect delta masks for magnetic dilution and disorder
