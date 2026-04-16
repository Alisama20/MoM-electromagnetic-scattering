import numpy as np
import matplotlib.pyplot as plt

# ============================
# Constantes y parámetros
# ============================

pi = np.pi
j = 1j

# Parámetros globales (como COMMON /CNSTNS/)
a = 0.005
a2 = a * a
b = 2 * pi      # beta normalizado
b2 = b * b


# ============================
# Kernel de Pocklington
# ============================

def kernel(z, zp):
    """
    Kernel K(z, z') según Pocklington compactado:
    K = exp(-j b R) * [ (1 + j b R)(2 R^2 - 3 a^2) + b^2 a^2 R^2 ] / R^5
    """
    R = np.sqrt((z - zp)**2 + a2)
    R2 = R * R
    R5 = R2 * R2 * R
    return np.exp(-j * b * R) * (
        (1.0 + j * b * R) * (2.0 * R2 - 3.0 * a2) + b2 * a2 * R2
    ) / R5


# ============================
# Campo incidente (anillo magnético)
# ============================

def einc(z):
    bar = 2.23
    R1 = np.sqrt(z*z + a2)
    R2 = np.sqrt(z*z + 4.9729 * a2)
    return (1.0 / (2.0 * np.log(bar))) * (
        np.exp(-j * b * R1) / R1 - np.exp(-j * b * R2) / R2
    )


# ============================
# Integración por Simpson (compleja)
# ============================

def ingrlk(nint, ll, ul, z):
    """
    Integral de K(z, z') respecto a z' desde ll hasta ul
    usando Simpson compuesto con nint subintervalos.
    """
    h = (ul - ll) / nint
    hover2 = 0.5 * h

    # puntos tipo "4*f" (mitad de cada subsegmento)
    half = kernel(z, ll + hover2)

    # puntos tipo "2*f" (extremos intermedios)
    s = 0.0 + 0.0j

    nintm1 = nint - 1
    for i in range(1, nintm1 + 1):
        zp = ll + i * h
        s += kernel(z, zp)
        half += kernel(z, zp + hover2)

    return (h / 6.0) * (kernel(z, ll) + kernel(z, ul) + 4.0 * half + 2.0 * s)


# ============================
# Resolución del sistema lineal (Gauss)
# ============================

def solve_system(Z, V):
    """
    Sustitución de la subrutina SISTEMAS:
    resuelve Z I = V mediante numpy.linalg.solve.
    """
    return np.linalg.solve(Z, V)


# ============================
# Programa principal
# ============================

def main():
    # Número de segmentos
    N = int(input("Número de segmentos, N = "))

    # Longitud normalizada
    L = 0.47
    delz = L / N

    zi_inf = -0.5 * L
    delz02 = delz / 2.0
    z1j = zi_inf + delz02  # centro del primer segmento (j=1)

    # Primera columna de Z y vector V
    ZZ = np.zeros(N, dtype=complex)
    V = np.zeros(N, dtype=complex)

    for ii in range(N):
        zisup = zi_inf + delz
        zim = zi_inf + delz02

        # NINT variable como en el Fortran
        nint = int(80.00001 - (ii + 1) * (60.0 / N))

        # Integral del kernel en el segmento ii, punto campo z1j
        val = ingrlk(nint, zi_inf, zisup, z1j)
        ZZ[ii] = -4.7724j * val

        # Voltaje incidente en el centro del segmento
        V[ii] = -einc(zim)

        zi_inf = zisup

    # Construcción de la matriz Toeplitz Z a partir de ZZ
    Z = np.zeros((N, N), dtype=complex)
    for ii in range(N):
        for jj in range(N):
            ic = abs(jj - ii)
            Z[ii, jj] = ZZ[ic]

    # Resolver Z I = V
    I = solve_system(Z, V)

    import matplotlib.pyplot as plt

    # Coordenadas z de los centros de segmento
    z = np.linspace(-L/2 + delz/2, L/2 - delz/2, N)

    plt.figure(figsize=(8,5))
    plt.plot(z, np.abs(I), '-o', markersize=4)
    plt.xlabel("z (longitudes de onda)")
    plt.ylabel("|I(z)|")
    plt.title("Distribución de corriente en el hilo (|I| vs z)")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()


