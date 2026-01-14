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

## Energy Tracking

All methods maintain accurate incremental energy tracking with errors at floating-point precision (~10⁻¹⁵). The Wolff bug (factor of 0.5 error) has been fixed.

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
- **Wolff** energy calculation correctly handles bonds within clusters (no 0.5 factor)
- All methods respect delta masks for magnetic dilution and disorder
