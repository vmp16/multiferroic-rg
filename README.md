# Multiferroic Rhombohedral Graphene Modeling

This project implements numerical and analytical modeling of **ABC-stacked Rhombohedral Graphene** using the **McCann model** formulated in Cartesian coordinates and incorporating **Huang's interaction energy** to study its topological, transport, and multiferroic properties.

Specifically, this repository contains code to:
1. **Reproduce Anomalous Hall Effect (AHE) butterfly hysteresis** under external interlayer potential sweeps.
2. **Investigate nonlinear conductivity** driven by the **Quantum Metric Dipole (QMD)**.

---

## Physical Background

### McCann Model in Cartesian Coordinates
The system is modeled using the effective low-energy $N$-layer Hamiltonian for ABC-stacked graphene:
* Formulation in Cartesian coordinates ($k_x$, $k_y$).
* Includes in-layer hopping ($\gamma_0$), vertical nearest-layer hopping ($\gamma_1$), next-nearest layer hopping ($\gamma_2$), trigonal warping ($\gamma_3$), and electron-hole symmetry breaking terms ($\gamma_4$).
* Enables analytic calculation of energy bands, eigenstates, and Hamiltonian derivatives.

### Huang's Interaction Energy
To capture the self-consistent polarization and gap state changes, the model incorporates Huang's interaction energy. This is essential for studying the multiferroic state and explaining the hysteretic response of the valley/spin polarization.

### Topological & Non-linear Transport Phenomena
* **Anomalous Hall Effect (AHE)**: Intrinsic linear Hall conductivity calculated via the Berry curvature integration over the Brillouin zone.
* **Quantum Metric Dipole (QMD)**: Computes the first-order momentum-space gradient of the Quantum Metric tensor. This dipole characterizes the nonlinear Hall/longitudinal transport under an AC electric field.

---

## Repository Structure

```
multiferroic-rg/
├── model/
│   ├── model.py          # McCannCarts system class
│   ├── config.py         # Global parameters
│   └── analysis/         # Physics analysis package (split by topic)
│       ├── __init__.py   # Re-exports all public symbols
│       ├── utils.py      # Utility helpers (progress_bar)
│       ├── kspace.py     # k-mesh, Fermi distribution, system construction
│       ├── thermo.py     # DOS, flavor densities, kinetic energies
│       ├── interactions.py  # SU(4) and anisotropic interaction potentials
│       ├── meanfield.py  # Self-consistent mean-field (SCF) solvers
│       ├── geometry.py   # QGT, quantum metric, Berry curvature, AHE, QMD
│       ├── montecarlo.py # Monte Carlo ground-state search
│       └── phases.py     # Phase diagram tables, SCF from tables, flavor counting
├── scripts/
│   ├── fixed_bands/      # Band structure with manually set gaps; DOS and carrier density
│   ├── correlations/     # Mean-field calculations for electronic correlations
│   └── transport/        # Transport signatures
│       ├── ahe_scan_U.py        # AHE butterfly hysteresis scan
│       ├── gao_scan_mu.py       # Gao NLT conductivity vs chemical potential
│       ├── holder_scan_mu.py    # Holder NLT conductivity vs chemical potential
│       └── qmd_scan_mu.py       # QMD conductivity vs chemical potential
├── tests/                # Unit tests, shape verifications, debugging scripts
├── figures/              # Generated plots (AHE, DOS, bands, QMD)
└── data/                 # Calculated .npz data archives
```

### `model/analysis` package

All public functions are re-exported through `model/analysis/__init__.py`, so existing code using `from model.analysis import <name>` continues to work unchanged.

| Sub-module | Contents |
|---|---|
| `utils` | `progress_bar` |
| `kspace` | `get_kmesh`, `fermi_distrib`, `deriv_fermi_distrib`, `build_systems_with_displacement` |
| `thermo` | `precompute_flavor_bands`, `get_dos`, `get_flavor_density_from_bands`, `get_flavor_mu_from_densities`, `get_flavor_dos`, `get_flavor_kinetic_energy_from_bands`, `get_part_density` |
| `interactions` | `V_int_SU4`, `V_int_anisotropic`, `compute_mf_symmetric_potentials`, `compute_mf_anisotropic_potentials` |
| `meanfield` | `solve_self_consistent_mean_field`, `solve_phase_mean_field` |
| `geometry` | `velocity_element`, `get_QG_tensor_component`, `get_quantum_metric_component`, `get_Berry_curv`, `get_QG_tensor`, `get_quantum_metric`, `get_G_tensor`, `get_qmd`, `get_ahe`, `sym_decomp_cond`, `get_QGT_chunk`, `get_qm_from_qgt`, `get_bc_from_qgt`, `get_G_tensor_from_qgt`, `get_qmd_from_qgt`, `integrate_qmd_chunk` |
| `montecarlo` | `sample_flavor_densities`, `build_density_interp_tables`, `evaluate_realizations`, `find_minimum_energy_configuration` |
| `phases` | `build_density_and_energy_tables`, `build_tables_histogram`, `solve_scf_from_tables`, `internal_energy_from_tables`, `count_occupied_flavors`, `count_degenerate_flavors` |

---

## How to Run

Scan the Anomalous Hall conductivity vs interlayer potential $U$:
```bash
python scripts/transport/ahe_scan_U.py
```

Scan the Quantum Metric Dipole (QMD) conductivity tensor vs chemical potential:
```bash
python scripts/transport/qmd_scan_mu.py
```

---

## Notes on Legacy Files

The `_to_delete/` folder holds files that have been superseded and are pending removal:
* `analysis.py` — flat module replaced by the `model/analysis/` package
* `analysis_old.py` — early prototype; all functions present in the new package

Scripts in `scripts/transport/` ending in `_fixed.py` and tests prefixed `nlt_*_single.py` or `test_nlt_*` used an old element-wise NLT API (`get_nlt_gao`, `get_nlt_holder`, `get_nlt_qmd`) that was removed when the code was modernised to use chunk-based integration. These files now raise `NotImplementedError` on execution; the current equivalents are the `*_scan_mu.py` scripts.
