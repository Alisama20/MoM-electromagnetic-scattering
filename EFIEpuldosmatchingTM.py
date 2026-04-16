import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1, hankel2, jv
from scipy.constants import epsilon_0, mu_0, c
from scipy.linalg import solve_circulant



# ======================================================
# PARAMETROS
# ======================================================

lambda_val = 1.0
k = 2 * np.pi / lambda_val
a = 2.0 * lambda_val
eta = np.sqrt(mu_0 / epsilon_0)

gamma = np.exp(np.euler_gamma)

rho = 1000 * a

num_modos = 40

phi_obs = np.linspace(0, 2*np.pi, 360)

eps = 1e-20 #EVITAR LOG(0) EN RCS


# ======================================================
# CONSTRUCCION MOM
# ======================================================

def solve_mom(N):

    phi = np.linspace(0, 2*np.pi, N, endpoint=False)

    dl = 2 * np.pi * a / N

    x = a * np.cos(phi)
    y = a * np.sin(phi)

    
    dist = np.sqrt((x[0]-x)**2 + (y[0]-y)**2)

    row = dl * hankel2(0, k * dist)

    row[0] = dl * (
        1 - 1j * (2/np.pi) * (np.log(k*gamma*dl/4) - 1)
    )

    #ECITACION
    E_inc = np.exp(1j * k * x)
    b = (4/(k*eta)) * E_inc

    #RESOLVER
    J = solve_circulant(row, b)

    return phi, J


# ======================================================
# CAMPO LEJANO NUMERICO
# ======================================================

def far_field_numeric(phi, J, N):

    dphi = 2*np.pi/N

    omega = 2*np.pi*c/lambda_val

    C = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k))

    Es = np.zeros_like(phi_obs, dtype=complex)

    for i, phi_o in enumerate(phi_obs):

        phase = np.exp(1j*k*a*np.cos(phi - phi_o))

        integral = np.sum(J * phase) * a * dphi

        Es[i] = (
            C
            * np.exp(-1j*k*rho)
            / np.sqrt(rho)
            * integral
        )

    return Es


# ======================================================
# CAMPO LEJANO ANALITICO
# ======================================================

def far_field_exact():

    n = np.arange(0, num_modos+1)

    epsilon_n = np.ones_like(n)
    epsilon_n[1:] = 2

    C = -np.sqrt(2/np.pi) * np.exp(
        -1j*(k*rho - np.pi/4)
    ) / np.sqrt(k*rho)

    Es = np.zeros_like(phi_obs, dtype=complex)

    Jn = jv(n, k*a)
    Hn = hankel1(n, k*a)

    coef = epsilon_n * (-1)**n * (Jn/Hn)

    for i, phi_o in enumerate(phi_obs):

        serie = np.sum(
            coef * np.cos(n*phi_o)
        )

        Es[i] = C * serie

    return Es


# ======================================================
# RCS 2D
# ======================================================

def compute_rcs(Es):

    sigma = 2*np.pi*rho*np.abs(Es)**2

    sigma_dB = 10*np.log10(np.maximum(sigma, eps))

    return sigma, sigma_dB


# ======================================================
# Error L2
# ======================================================

def l2_error(num, exact):

    return np.linalg.norm(num-exact)






# ======================================================
# Programa principal
# ======================================================

def main():

    N_values = [10, 20, 30, 40, 50, 100,1000]

    # ===============================
    # CORRIENTE ANALITICA
    # ===============================

    def exact_current(phi):

        n = np.arange(0, num_modos+1)

        epsilon_n = np.ones_like(n)
        epsilon_n[1:] = 2

        factor = epsilon_n * (1j)**n

        Hn = hankel2(n, k*a).reshape(-1,1)

        cos_n = np.cos(np.outer(n, phi))

        J = np.sum(
            (factor.reshape(-1,1) * cos_n)/Hn,
            axis=0
        )

        J *= (2/(np.pi*k*a*eta))

        return J


    # ===============================
    # FIGURA 1: CORRIENTE
    # ===============================

    plt.figure(figsize=(10,7))


    # Solución exacta de referencia (alta resolución)
    phi_ref = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    J_exact_ref = exact_current(phi_ref)



    for N in N_values:

        print(f"Corriente: N = {N}")

        phi, J = solve_mom(N)

        J_exact = exact_current(phi)

        error = l2_error(J, J_exact)

        plt.plot(
            np.degrees(phi),
            np.abs(J)*1000,
            'o-',
            linewidth=1.5,
            markersize=4,
            label=f"N = {N} | Error $L^2$ ={error:.2e}"
        )


    # Curva exacta
    plt.plot(
        np.degrees(phi_ref),
        np.abs(J_exact_ref)*1000,
        'k--',
        linewidth=3,
        label="Exacta (40 modos)"
    )

    plt.xlabel("Ángulo azimutal (grados)")
    plt.ylabel("Densidad de corriente (mA/m)")

    plt.title(r" EFIE: $|\boldsymbol{J}_z (\phi)|$ de un cilindro inifito de radio 2$\lambda$ mediante FD-MoM ($point$ $matching$ + pulsos)")

    plt.grid(True)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


    # ===============================
    # FIGURA 2: RCS
    # ===============================

    # Campo exacto
    Es_exact = far_field_exact()
    _, sigma_exact_dB = compute_rcs(Es_exact)

    phi_deg = np.degrees(phi_obs)

    plt.figure(figsize=(10,7))


    for N in N_values:

        phi, J = solve_mom(N)

        Es_num = far_field_numeric(phi, J, N)

        _, sigma_num_dB = compute_rcs(Es_num)

        error = l2_error(
            sigma_num_dB,
            sigma_exact_dB
        )

        plt.plot(
            phi_deg,
            sigma_num_dB,
            linewidth=2,
            label=f"N = {N} | Error $L^2$ = {error:.2e}"
        )


    plt.plot(
        phi_deg,
        sigma_exact_dB,
        'k--',
        linewidth=3,
        label="Exacta (40 modos)"
    )
    

    plt.xlabel("Ángulo de observación (grados)")
    plt.ylabel("RCS 2D (dBsm)")

    plt.title(r" EFIE: RCS 2D de un cilindro inifito de radio 2$\lambda$ mediante FD-MoM ($point$ $matching$ + pulsos)")

    plt.grid(True)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()



# ======================================================
# Ejecutar
# ======================================================

if __name__ == "__main__":
    main()
