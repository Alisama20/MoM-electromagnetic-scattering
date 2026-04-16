# Method of Moments for 2D Electromagnetic Scattering

Numerical solution of the **EFIE** and **MFIE** for TM-polarised electromagnetic scattering by a perfectly conducting (PEC) circular cylinder, using the **Frequency-Domain Method of Moments (FD-MoM)**.

> **Course project** — *Complementos Matemáticos y Numéricos*, Master's in Physics (Radiation, Nanotechnology, Particles & Astrophysics), University of Granada.

---

## Physical Problem

A PEC cylinder of radius $a$ is illuminated by a TM-polarised plane wave $E_z^i = E_0 e^{-jk\rho\cos\phi}$. The induced surface current $J_z(\phi')$ is found by solving either of two integral equations:

### EFIE (Electric Field Integral Equation)

$$
\hat{n}(\mathbf{r}) \times \mathbf{E}^i(\mathbf{r}) = \hat{n}(\mathbf{r}) \times \left( j\omega\mu \int_S \left[ \mathbf{J}(\mathbf{r}') + \frac{1}{k^2} \nabla'(\nabla' \cdot \mathbf{J}(\mathbf{r}')) \right] G(\mathbf{r},\mathbf{r}') \, d\mathbf{r}' \right)
$$

### MFIE (Magnetic Field Integral Equation)

$$
\hat{n}(\mathbf{r}) \times \mathbf{H}^i(\mathbf{r}) = \frac{J(\mathbf{r})}{2} - \hat{n}(\mathbf{r}) \times \int_S \mathbf{J}(\mathbf{r}') \times \nabla' G(\mathbf{r},\mathbf{r}') \, d\mathbf{r}'
$$

where $G(\boldsymbol{\rho},\boldsymbol{\rho}') = -\frac{j}{4} H_0^{(2)}(k|\boldsymbol{\rho}-\boldsymbol{\rho}'|)$ is the 2D Green's function.

---

## Methods

### EFIE with pulse basis and point matching — `EFIEpuldosmatchingTM.py`

The boundary is discretised into $N$ segments with **pulse basis functions** and tested at segment centres (*point matching*). The circulant structure of the impedance matrix is exploited via `scipy.linalg.solve_circulant` for $O(N \log N)$ solution.

### EFIE with triangular basis (Galerkin) — `EFIEtriangulosftestTM.py`

Uses **triangular (rooftop) basis** and **testing functions** with Gauss quadrature for a more accurate Galerkin formulation.

### MFIE with pulse basis and point matching — `MFIE-CILINDRO2DINF-PONITMATCHyPULSOS-TM.py`

Same discretisation as the EFIE pulse code but applied to the MFIE. Convergence comparisons reveal the MFIE's higher sensitivity to mesh density.

### Thin-wire antenna (Pocklington) — `ThinWire.py`

Solves the **Pocklington integral equation** for the current distribution on a thin-wire dipole antenna using MoM with pulse basis functions.

### 3D EFIE with RWG basis (experimental) — `IntentoMoM3D.py`

Prototype implementation of a 3D MoM solver using **RWG (Rao–Wilton–Glisson)** basis functions on a triangular mesh, with singularity extraction and Duffy transforms. Accelerated with Numba.

---

## Results

### Surface currents

<p align="center">
<img src="figures/CorrientesEFIE.png" width="45%">
<img src="figures/CorrientesMFIE.png" width="45%">
</p>

Left: EFIE surface current vs analytical solution. Right: MFIE surface current vs analytical solution. The EFIE shows faster convergence with fewer segments.

### Radar Cross Section (RCS)

<p align="center">
<img src="figures/RCS2DEFIE.png" width="45%">
<img src="figures/RCS2DMFIE.png" width="45%">
</p>

2D bistatic RCS ($\sigma_{2D}/\lambda$) computed from EFIE (left) and MFIE (right), compared with the analytical Mie series.

---

## Report

| File | Description |
|------|-------------|
| [MemoriaES.pdf](MemoriaES.pdf) | Spanish report — full derivation of EFIE/MFIE, MoM discretisation, results and convergence analysis |

The LaTeX sources are in the [`latex/`](latex/) folder.

---

## Repository Structure

```
.
├── EFIEpuldosmatchingTM.py              # EFIE — pulse basis, point matching
├── EFIEtriangulosftestTM.py             # EFIE — triangular basis, Galerkin
├── MFIE-CILINDRO2DINF-PONITMATCHyPULSOS-TM.py  # MFIE — pulse basis, point matching
├── ThinWire.py                          # Pocklington thin-wire antenna
├── IntentoMoM3D.py                      # 3D EFIE with RWG (experimental)
├── MemoriaES.pdf                        # Academic report (Spanish)
├── latex/
│   ├── main.tex                         # LaTeX source
│   ├── bibliografia.bib                 # Bibliography
│   └── escudoUGRmonocromo.png           # UGR logo
├── figures/                             # Result figures
│   ├── CorrientesEFIE.png
│   ├── CorrientesMFIE.png
│   ├── RCS2DEFIE.png
│   └── RCS2DMFIE.png
└── LICENSE
```

---

## Requirements

```
numpy  scipy  matplotlib
```

For the 3D solver (`IntentoMoM3D.py`):

```
numpy  scipy  matplotlib  trimesh  numba
```

```bash
pip install numpy scipy matplotlib trimesh numba
```

---

## Usage

```bash
# 2D EFIE (pulse basis) — prints current and RCS, shows plots
python EFIEpuldosmatchingTM.py

# 2D EFIE (triangular basis)
python EFIEtriangulosftestTM.py

# 2D MFIE
python MFIE-CILINDRO2DINF-PONITMATCHyPULSOS-TM.py

# Thin-wire dipole antenna
python ThinWire.py

# 3D EFIE with RWG (experimental)
python IntentoMoM3D.py
```

---

## Author

**A. S. Amari**

Developed as part of the coursework for *Mathematical and Numerical Complements* —
Master's Degree in Physics: Radiation, Nanotechnology, Particles and Astrophysics,
University of Granada.
