import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2, hankel1

# ======================================================
# PARAMETERS
# ======================================================

lambda_val = 1.0
k = 2 * np.pi / lambda_val
a = 2.0 * lambda_val  # Cylinder radius to match Gibson Fig 5.9
eta = 120 * np.pi
N = 180               # Number of segments/nodes
gamma_eul = 1.7810724179

# ======================================================
# GEOMETRY
# ======================================================

phi_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
dl = (2 * np.pi * a) / N

# ======================================================
# IMPEDANCE MATRIX Z AND EXCITATION VECTOR b
# (Galerkin with triangular basis functions)
# ======================================================

Z = np.zeros((N, N), dtype=complex)
b = np.zeros(N, dtype=complex)

# Gauss quadrature (2 points per segment)
g_p = np.array([0.211324865, 0.788675135])
g_w = np.array([0.5, 0.5])

# Fill by segment pairs (each segment contributes to 2 triangles)
for i in range(N):  # Test segment m (between node i and i+1)
    for j in range(N):  # Source segment n (between node j and j+1)
        for pm in range(2):
            tm = g_p[pm]
            phi_m = phi_nodes[i] + tm * (2 * np.pi / N)
            xm, ym = a * np.cos(phi_m), a * np.sin(phi_m)

            # On segment i: Triangle f_i is descending ramp (1-t)
            # Triangle f_{i+1} is ascending ramp (t)
            f_test = [1 - tm, tm]
            nodes_m = [i, (i + 1) % N]

            for pn in range(2):
                tn = g_p[pn]
                phi_n = phi_nodes[j] + tn * (2 * np.pi / N)
                xn, yn = a * np.cos(phi_n), a * np.sin(phi_n)

                dist = np.sqrt((xm - xn)**2 + (ym - yn)**2)
                f_basis = [1 - tn, tn]
                nodes_n = [j, (j + 1) % N]

                # EFIE kernel
                if i == j:  # Self-term approximation
                    h0 = (1 - 1j * (2/np.pi) * (np.log(k * gamma_eul * dl / 4) - 1))
                else:
                    h0 = hankel2(0, k * dist)

                # Distribute into the 4 matrix elements
                for mi in range(2):
                    for ni in range(2):
                        Z[nodes_m[mi], nodes_n[ni]] += (
                            g_w[pm] * g_w[pn] * f_test[mi]
                            * f_basis[ni] * h0 * dl**2
                        )

    # Excitation vector b
    for p in range(2):
        t = g_p[p]
        phi_q = phi_nodes[i] + t * (2 * np.pi / N)
        E_inc = np.exp(1j * k * a * np.cos(phi_q))  # Incidence from phi=0
        factor = (4 / (k * eta)) * g_w[p] * E_inc * dl
        b[i] += factor * (1 - t)           # Descending part of node i
        b[(i + 1) % N] += factor * t       # Ascending part of node i+1

# ======================================================
# MoM SOLUTION
# ======================================================

Jz_tri = np.linalg.solve(Z, b)

# ======================================================
# ANALYTICAL SOLUTION (Mode series)
# ======================================================

num_modes = 40
phi_plot = np.linspace(0, 2*np.pi, N, endpoint=False)
Jz_exact = np.zeros(N, dtype=complex)
for n in range(num_modes):
    eps_n = 1 if n == 0 else 2
    Hn = hankel1(n, k * a)
    Jz_exact += eps_n * ((-1j)**n) * np.cos(n * phi_plot) / Hn
Jz_exact *= (2 / (np.pi * k * a * eta))

# ======================================================
# PLOT
# ======================================================

plt.figure(figsize=(9, 6))
plt.plot(np.degrees(phi_plot), np.abs(Jz_tri) * 1000,
         'b.', label='EFIE (Galerkin triangles)')
plt.plot(np.degrees(phi_plot), np.abs(Jz_exact) * 1000,
         'r-', label='Exact mode series')
plt.xlabel('Azimuthal position (degrees)')
plt.ylabel('Current density (mA/m)')
plt.title(f'Induced surface current ($a={a/lambda_val}\\lambda$) — Galerkin triangular basis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/CurrentGalerkin.png", dpi=150)
plt.show()
