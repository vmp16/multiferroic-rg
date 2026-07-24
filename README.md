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

* **`model/`**: Core physical implementation.
  * [`model.py`](file:///home/martinpv/DIPC/multiferroic-rg/model/model.py): Defines the `McCannCarts` system class with Cartesian coordinate evaluations, analytic energy bands, eigenstates, and derivatives.
  * [`analysis.py`](file:///home/martinpv/DIPC/multiferroic-rg/model/analysis.py): Implements functions to calculate the Fermi distribution, velocity matrix elements, Berry curvature, Quantum Metric Tensor, Quantum Metric Dipole (QMD), linear AHE conductivity, and density of states (DOS).
  * [`config.py`](file:///home/martinpv/DIPC/multiferroic-rg/model/config.py): Global parameters (hoppings, temperature, valley indices, spin-valley gap splits, and onsite energy offsets).

* **`scripts/`**: Scripts for sweeps and visualizations.
  * **`fixed_bands/`**: Building of the band structure with manually determined energy gaps and shifts to reproduce Huang's bands. Extracting the DOS and calculating the charge carrier density at a given Fermi level and vice-versa.
  * **`correlations/`**: Performing mean-field calculations to characterize the effect of electronic correlations, giving the values of the energy gaps and shifts.
  * **`transport/`**: Calculates different relevant transport signatures of the system, including the linear Anomalous Hall Effect butterfly hysteresis [`ahe_scan_U.py`] and the QMD's non-linear conductivity [`qmd_scan_mu.py`].

  <!-- * **AHE & Hysteresis**:
    * [`ahe_scan_U.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/ahe_scan_U.py) / [`ahe_scan_plot.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/ahe_scan_plot.py): Simulates and plots the AHE butterfly hysteresis loops by scanning the interlayer potential $U$.
  * **QMD & Nonlinear Conductivity**:
    * [`qmd_scan_U.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/qmd_scan_U.py) / [`qmd_scan_mu.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/qmd_scan_mu.py): Scans QMD conductivity tensor elements as a function of $U$ or chemical potential $\mu$.
  * **Bands & DOS**:
    * [`bandplot.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/bandplot.py) / [`dos_plot.py`](file:///home/martinpv/DIPC/multiferroic-rg/scripts/dos_plot.py): Plots dispersion relation surfaces and density of states. -->

* **`tests/`**: Unit tests, shape verifications, and parameter debugging scripts.
* **`figures/`**: Generated plots for AHE butterfly hysteresis, DOS, energy bands, and QMD sweeps.
* **`data/`**: Calculated `.npz` data archives.

---

## How to Run

To run a scan of the Anomalous Hall conductivity vs interlayer potential $U$ and save the data:
```bash
python scripts/ahe_scan_U.py
```

To run a scan of the Quantum Metric Dipole (QMD) conductivity tensor elements:
```bash
python scripts/qmd_scan_U.py
```
