# 500K single-track case — 4-solver comparison

**Mesh**: 50×200×50 = 500,000 cells, dx = 4 μm
**Physics**: 300W laser, full LPBF (rays>0, real melt + vapor)
**OF version**: v2412 fresh build with our matrixDumper hooks
**Truth**: CHOLMOD LU on lab Xeon (rel resid ~2.7e-15 across all 6)
**Note**: AMGx_e12+IR ≈ LU to rel 1e-11 (from lab Xeon LU verification)


## melting @ t = 320 ns

`‖x_LU‖∞ = 1.284e+06 Pa,  ‖x_LU‖₂ = 1.018e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 7 | n/a | 9.21e-09 | **4.459e+01** | 3.47e-05 | **4.484e+02** | 4.40e-06 |
| AMGx PCG | 1e-8 | 497 | 22741 | 1.08e-08 | **3.079e+01** | 2.40e-05 | **1.825e+02** | 1.79e-06 |
| AMGx PCG + 1 IR | 1e-12 | 616 | 99337 | 1.02e-15 | **1.250e-05** | 9.73e-12 | **1.250e-06** | 9.73e-13 |
| CHOLMOD direct (truth) | — | direct | 73200 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

## melting @ t = 380 ns

`‖x_LU‖∞ = 1.284e+06 Pa,  ‖x_LU‖₂ = 1.018e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 35 | n/a | 9.65e-09 | **3.007e+01** | 2.34e-05 | **2.281e+02** | 2.24e-06 |
| AMGx PCG | 1e-8 | 535 | 38437 | 1.23e-08 | **2.013e+01** | 1.57e-05 | **1.078e+02** | 1.06e-06 |
| AMGx PCG + 1 IR | 1e-12 | 689 | 88218 | 1.02e-15 | **1.019e-05** | 7.94e-12 | **1.019e-06** | 7.94e-13 |
| CHOLMOD direct (truth) | — | direct | 74700 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

## melting @ t = 410 ns

`‖x_LU‖∞ = 1.284e+06 Pa,  ‖x_LU‖₂ = 1.019e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 40 | n/a | 9.58e-09 | **1.793e+01** | 1.40e-05 | **2.111e+02** | 2.07e-06 |
| AMGx PCG | 1e-8 | 539 | 23085 | 1.20e-08 | **1.795e+01** | 1.40e-05 | **9.608e+01** | 9.43e-07 |
| AMGx PCG + 1 IR | 1e-12 | 735 | 44780 | 1.02e-15 | **2.180e-06** | 1.70e-12 | **2.180e-07** | 1.70e-13 |
| CHOLMOD direct (truth) | — | direct | 74500 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

## evap_early @ t = 700 ns

`‖x_LU‖∞ = 1.285e+06 Pa,  ‖x_LU‖₂ = 1.019e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 17 | n/a | 9.18e-09 | **1.408e+01** | 1.10e-05 | **2.753e+02** | 2.70e-06 |
| AMGx PCG | 1e-8 | 709 | 32746 | 8.94e-09 | **1.111e+01** | 8.65e-06 | **1.170e+02** | 1.15e-06 |
| AMGx PCG + 1 IR | 1e-12 | 887 | 52350 | 1.02e-15 | **1.449e-05** | 1.13e-11 | **1.449e-06** | 1.13e-12 |
| CHOLMOD direct (truth) | — | direct | 69500 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

## evap @ t = 900 ns

`‖x_LU‖∞ = 1.373e+06 Pa,  ‖x_LU‖₂ = 1.019e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 43 | n/a | 9.94e-09 | **1.446e+01** | 1.05e-05 | **1.823e+02** | 1.79e-06 |
| AMGx PCG | 1e-8 | 435 | 9025 | 1.13e-08 | **1.241e+01** | 9.04e-06 | **9.967e+01** | 9.78e-07 |
| AMGx PCG + 1 IR | 1e-12 | 559 | 21678 | 1.02e-15 | **1.569e-06** | 1.14e-12 | **1.569e-07** | 1.14e-13 |
| CHOLMOD direct (truth) | — | direct | 72500 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

## evap_late @ t = 1060 ns

`‖x_LU‖∞ = 1.285e+06 Pa,  ‖x_LU‖₂ = 1.018e+08 Pa`

| solver | tol | iter | wall (ms) | rel_resid (‖A·x-b‖/‖b‖) | max\|x-x_LU\| (Pa) | rel max | ‖x-x_LU‖₂ (Pa) | rel L2 |
|---|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 65 | n/a | 9.99e-09 | **2.325e+01** | 1.81e-05 | **2.387e+02** | 2.34e-06 |
| AMGx PCG | 1e-8 | 773 | 38867 | 1.20e-08 | **5.628e+01** | 4.38e-05 | **3.331e+02** | 3.27e-06 |
| AMGx PCG + 1 IR | 1e-12 | 1123 | 100695 | 1.03e-15 | **1.129e-05** | 8.79e-12 | **1.129e-06** | 8.79e-13 |
| CHOLMOD direct (truth) | — | direct | 70800 | 2.70e-15 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |

---
## Summary table — max errors across 6 timesteps

| solver | tol | max iter | wall range (ms) | max(max\|err\|) Pa | max(rel max) | max(‖err‖₂) Pa | max(rel L2) |
|---|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 65 | n/a | **4.459e+01** | 3.47e-05 | **4.484e+02** | 4.40e-06 |
| AMGx PCG | 1e-8 | 773 | 9025-38867 | **5.628e+01** | 4.38e-05 | **3.331e+02** | 3.27e-06 |
| AMGx PCG + 1 IR | 1e-12 | 1123 | 21678-100695 | **1.449e-05** | 1.13e-11 | **1.449e-06** | 1.13e-12 |
| CHOLMOD direct | — | direct | 69500-74700 | **0.000e+00** | 0.00e+00 | **0.000e+00** | 0.00e+00 |