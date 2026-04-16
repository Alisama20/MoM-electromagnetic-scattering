import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1, hankel2, jv
from scipy.constants import epsilon_0, mu_0, c, pi
from scipy.linalg import solve_circulant


# ======================================================
# Parámetros globales
# ======================================================

lambda_val = 1.0
k = 2 * pi / lambda_val
a = 2.0 * lambda_val
eta = np.sqrt(mu_0 / epsilon_0)

rho = 1000 * a
num_modos = 40

phi_obs = np.linspace(0, 2*pi, 360)

eps = 1e-20


# ======================================================
# MFIE MoM
# ======================================================

def solve_mfie(N):

    phi = np.linspace(0, 2*pi, N, endpoint=False)

    dl = 2*pi*a/N

    x = a*np.cos(phi)
    y = a*np.sin(phi)

    nx = np.cos(phi)
    ny = np.sin(phi)


    dx = x[0] - x
    dy = y[0] - y

    dist = np.sqrt(dx**2 + dy**2)

    #INGNORA NANs Y DIVISIONES POR CERO (AUTOTERMINO)
    with np.errstate(divide='ignore', invalid='ignore'):

        rx = dx/dist
        ry = dy/dist

        dot_n_R = nx[0]*rx + ny[0]*ry

        kernel = -1j*k*hankel2(1, k*dist)/4

        row = -dot_n_R*kernel*dl


    #AUTIERMINO
    row[0] = 0.5


    #EXCITACION
    b = (1/eta)*np.cos(phi)*np.exp(1j*k*x)

    J = solve_circulant(row, b)

    return phi, J


# ======================================================
# CAMPO LEJANO NUMERICO
# ======================================================

def far_field_numeric(phi, J, N):

    dphi = 2*pi/N

    omega = 2*pi*c/lambda_val

    C = -omega*mu_0*np.sqrt(1j/(8*pi*k))

    Es = np.zeros_like(phi_obs, dtype=complex)


    for i, phi_o in enumerate(phi_obs):

        phase = np.exp(1j*k*a*np.cos(phi-phi_o))

        integral = np.sum(J*phase)*a*dphi

        Es[i] = (
            C*np.exp(-1j*k*rho)
            / np.sqrt(rho)
            * integral
        )

    return Es


# ======================================================
# CAMPO LEJANO ANALITICO
# ======================================================

def far_field_exact():

    n = np.arange(num_modos+1)

    eps_n = np.ones_like(n)
    eps_n[1:] = 2


    C = -np.sqrt(2/pi)*np.exp(-1j*(k*rho-pi/4))/np.sqrt(k*rho)

    Es = np.zeros_like(phi_obs, dtype=complex)


    Jn = jv(n, k*a)
    Hn = hankel1(n, k*a)

    coef = eps_n*(-1)**n*(Jn/Hn)


    for i, phi_o in enumerate(phi_obs):

        Es[i] = C*np.sum(coef*np.cos(n*phi_o))


    return Es


# ======================================================
# RCS
# ======================================================

def compute_rcs(Es):

    sigma = 2*pi*rho*np.abs(Es)**2

    sigma_dB = 10*np.log10(np.maximum(sigma, eps))

    return sigma, sigma_dB


# ======================================================
# ERROR L2
# ======================================================

def l2_error(num, exact):

    return np.linalg.norm(num-exact)



# ======================================================
# CORRIENTE EXACTA
# ======================================================

def exact_current(phi):

    n = np.arange(num_modos+1)

    eps_n = np.ones_like(n)
    eps_n[1:] = 2

    factor = eps_n*(1j)**n

    Hn = hankel2(n, k*a)[:,None]

    cos_n = np.cos(np.outer(n, phi))


    J = np.sum(
        (factor[:,None]*cos_n)/Hn,
        axis=0
    )

    J *= 2/(pi*k*a*eta)

    return J


# ======================================================
# PROGRAMA PRINCIPAL
# ======================================================

def main():

    N_values = [10, 20, 30, 40, 50, 100, 1000]


    # ===============================
    # FIGURA 1: CORRIENTE
    # ===============================

    plt.figure(figsize=(10,7))


    phi_ref = np.linspace(0,2*pi,2000,endpoint=False)
    J_exact_ref = exact_current(phi_ref)




    for N in N_values:

        print(f"Corriente MFIE: N = {N}")

        phi, J = solve_mfie(N)

        J_ex = exact_current(phi)

        err = l2_error(J, J_ex)


        plt.plot(
            np.degrees(phi),
            np.abs(J)*1000,
            'o-',
            markersize=4,
            label=f"N={N} | Error L2={err:.2e}"
        )

    plt.plot(
        np.degrees(phi_ref),
        np.abs(J_exact_ref)*1000,
        'k--',
        linewidth=3,
        label="Exacta (40 modos)"
    )


    plt.xlabel(r"Ángulo azimutal $\phi$ (grados)")
    plt.ylabel(r"$|J_z|$ (mA/m)^2")

    plt.title(r" MFIE: $|\boldsymbol{J}_z (\phi)|$ de un cilindro inifito de radio 2$\lambda$ mediante FD-MoM ($point$ $matching$ + pulsos)")

    plt.grid(True)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.show()



    # ===============================
    # FIGURA 2: RCS
    # ===============================

    Es_exact = far_field_exact()
    _, sigma_exact_dB = compute_rcs(Es_exact)

    phi_deg = np.degrees(phi_obs)


    plt.figure(figsize=(10,7))




    for N in N_values:

        print(f"RCS MFIE: N = {N}")

        phi, J = solve_mfie(N)

        Es_num = far_field_numeric(phi, J, N)

        _, sigma_num_dB = compute_rcs(Es_num)

        err = l2_error(sigma_num_dB, sigma_exact_dB)


        plt.plot(
            phi_deg,
            sigma_num_dB,
            linewidth=2,
            label=f"N={N} | Error L2={err:.2e}"
        )


    plt.plot(
        phi_deg,
        sigma_exact_dB,
        'k--',
        linewidth=3,
        label="Exacta (40 modos)"
    )


    plt.xlabel(r"Ángulo de observación $\phi^s$ (grados)")
    plt.ylabel(r"$\sigma_{2D}$ (dBsm)")

    plt.title(r" MFIE: RCS 2D de un cilindro inifito de radio 2$\lambda$ mediante FD-MoM ($point$ $matching$ + pulsos)")

    plt.grid(True)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.show()



# ======================================================
# Ejecutar
# ======================================================

if __name__ == "__main__":
    main()
