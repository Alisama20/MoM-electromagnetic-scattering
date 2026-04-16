import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2, hankel1
from scipy.constants import e

# 1. Parámetros del Problema (Sección 5.3.1.5)
lambda_val = 1.0
k = 2 * np.pi / lambda_val
a = 2.0 * lambda_val  # Radio del cilindro para coincidir con Fig 5.9 [1]
eta = 120 * np.pi
N = 180               # Número de segmentos/nodos [1]
gamma_eul = 1.7810724179 

# 2. Geometría
phi_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
dl = (2 * np.pi * a) / N

# 3. Matriz de Impedancia Z y Vector b (Galerkin - Ec. 5.99 y 5.100) [3, 4]
Z = np.zeros((N, N), dtype=complex)
b = np.zeros(N, dtype=complex)

# Cuadratura de Gauss (2 puntos por segmento) [5]
g_p = np.array([0.211324865, 0.788675135]) 
g_w = np.array([0.5, 0.5])

# Llenado por pares de segmentos (Cada segmento contribuye a 2 triángulos) [3]
for i in range(N): # Segmento de prueba m (entre nodo i y i+1)
    for j in range(N): # Segmento fuente n (entre nodo j y j+1)
        for pm in range(2):
            tm = g_p[pm]
            phi_m = phi_nodes[i] + tm * (2 * np.pi / N)
            xm, ym = a * np.cos(phi_m), a * np.sin(phi_m)
            
            # En el segmento i: Triángulo f_i es rampa descendente (1-t)
            # Triángulo f_{i+1} es rampa ascendente (t) [2]
            f_test = [1 - tm, tm] 
            nodes_m = [i, (i + 1) % N]

            for pn in range(2):
                tn = g_p[pn]
                phi_n = phi_nodes[j] + tn * (2 * np.pi / N)
                xn, yn = a * np.cos(phi_n), a * np.sin(phi_n)
                
                dist = np.sqrt((xm - xn)**2 + (ym - yn)**2)
                f_basis = [1 - tn, tn]
                nodes_n = [j, (j + 1) % N]

                # Kernel de la EFIE (Ec. 5.5) [6]
                if i == j: # Auto-término aproximado (Ec. 5.15) [7]
                    h0 = (1 - 1j * (2/np.pi) * (np.log(k * gamma_eul * dl / 4) - 1))
                else:
                    h0 = hankel2(0, k * dist)

                # Distribución en los 4 elementos de la matriz Z [3]
                for mi in range(2):
                    for ni in range(2):
                        Z[nodes_m[mi], nodes_n[ni]] += g_w[pm] * g_w[pn] * f_test[mi] * f_basis[ni] * h0 * dl**2

    # Vector de excitación b (Ec. 5.100) [4]
    for p in range(2):
        t = g_p[p]
        phi_q = phi_nodes[i] + t * (2 * np.pi / N)
        E_inc = np.exp(1j * k * a * np.cos(phi_q)) # Incidencia desde phi=0 [3]
        factor = (4 / (k * eta)) * g_w[p] * E_inc * dl
        b[i] += factor * (1 - t)         # Parte descendente del nodo i
        b[(i + 1) % N] += factor * t     # Parte ascendente del nodo i+1

# 4. Solución MoM
Jz_tri = np.linalg.solve(Z, b)

# 5. Solución Analítica (Serie de Modos - Ec. 5.95) [8]
num_modos = 40
phi_plot = np.linspace(0, 2*np.pi, N, endpoint=False)
Jz_exacta = np.zeros(N, dtype=complex)
for n in range(num_modos):
    eps_n = 1 if n == 0 else 2
    Hn = hankel1(n, k * a)
    Jz_exacta += eps_n * ((-1j)**n) * np.cos(n * phi_plot) / Hn
Jz_exacta *= (2 / (np.pi * k * a * eta))

# 6. Graficación (mA/m)
plt.figure(figsize=(9, 6))
plt.plot(np.degrees(phi_plot), np.abs(Jz_tri) * 1000, 'b.', label='EFIE (Galerkin Triangles)')
plt.plot(np.degrees(phi_plot), np.abs(Jz_exacta) * 1000, 'r-', label='Exact Mode Series')
plt.xlabel('Azimuthal Position (Degrees)')
plt.ylabel('Current Density (mA/m)')
plt.title(f'Induced Surface Current ($a={a/lambda_val}\lambda$) - Gibson Fig 5.9a')
plt.legend()
plt.grid(True)
plt.show()