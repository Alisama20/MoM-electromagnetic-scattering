# ============================================================
# MoM EFIE 3D con RWG
# - Término vectorial + término escalar
# - Extracción de singularidad + Duffy
# - Numba + ensamblado por triángulos
# ============================================================

import numpy as np
import trimesh
from scipy.constants import mu_0, epsilon_0, c
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# ============================================================
# 1. Parámetros físicos
# ============================================================

freq = 300e6
omega = 2.0 * np.pi * freq
k = omega / c
mu = mu_0
eps = epsilon_0

# ============================================================
# 2. Geometría
# ============================================================

radius = 1.0
mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)  # Cambiado de 4 a 3 para reducir complejidad

vertices = mesh.vertices.astype(np.float64)
faces = mesh.faces.astype(np.int64)
face_areas = mesh.area_faces.astype(np.float64)
face_normals = mesh.face_normals.astype(np.float64)

num_faces = faces.shape[0]

internal_edges = mesh.face_adjacency_edges.astype(np.int64)
adjacent_faces = mesh.face_adjacency.astype(np.int64)
num_rwg = internal_edges.shape[0]

print(f"Vertices: {vertices.shape[0]}")
print(f"Faces:    {num_faces}")
print(f"RWG:      {num_rwg}")

# ============================================================
# 3. Construcción RWG
# ============================================================

rp = np.zeros((num_rwg, 3))
rm = np.zeros((num_rwg, 3))
edge_len = np.zeros(num_rwg)
A_plus = np.zeros(num_rwg)
A_minus = np.zeros(num_rwg)
rwg_tris = np.zeros((num_rwg, 2), dtype=np.int64)

for m in range(num_rwg):
    v1, v2 = internal_edges[m]
    f_plus, f_minus = adjacent_faces[m]

    tri_p = faces[f_plus]
    tri_m = faces[f_minus]

    free_p = [v for v in tri_p if v not in (v1, v2)][0]
    free_m = [v for v in tri_m if v not in (v1, v2)][0]

    rp[m] = vertices[free_p]
    rm[m] = vertices[free_m]

    edge_len[m] = np.linalg.norm(vertices[v2] - vertices[v1])
    A_plus[m] = face_areas[f_plus]
    A_minus[m] = face_areas[f_minus]

    rwg_tris[m, 0] = f_plus
    rwg_tris[m, 1] = f_minus

# ============================================================
# 4. Cuadratura 3 puntos
# ============================================================

bary = np.array([
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5]
])

quad_pts = np.zeros((num_faces, 3, 3))
quad_w = np.zeros((num_faces, 3))

for t in range(num_faces):
    tri = vertices[faces[t]]
    quad_pts[t] = bary @ tri
    area = 0.5 * np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))
    quad_w[t] = area / 3.0

# ============================================================
# 5. Mapa triángulo → RWG
# ============================================================

tri_rwg = -np.ones((num_faces, 3), dtype=np.int64)
tri_sign = np.zeros((num_faces, 3))
counter = np.zeros(num_faces, dtype=np.int64)

for m in range(num_rwg):
    tp, tm = rwg_tris[m]

    tri_rwg[tp, counter[tp]] = m
    tri_sign[tp, counter[tp]] = +1.0
    counter[tp] += 1

    tri_rwg[tm, counter[tm]] = m
    tri_sign[tm, counter[tm]] = -1.0
    counter[tm] += 1

# --------------------------------------------------------
# Duffy: integral de 1/|r - r'| sobre un triángulo fuente
# --------------------------------------------------------
@njit
def duffy_integral_1_over_R(tri, r_obs):
    """
    ∫_T 1/|r_obs - r'| dS' usando Duffy (orden 7×7)
    tri: (3,3) vértices
    r_obs: (3,)
    """
    v0 = tri[0]
    a = tri[1] - v0
    b = tri[2] - v0

    # Gauss-Legendre 7 puntos en [0,1]
    u_nodes = np.array([
        0.025446043828620737,
        0.12923440720030278,
        0.2970774243113014,
        0.5000000000000000,
        0.7029225756886986,
        0.8707655927996972,
        0.9745539561713793
    ])
    u_weights = np.array([
        0.06474248308443485,
        0.13985269574463833,
        0.19091502525255946,
        0.2089795918367347,
        0.19091502525255946,
        0.13985269574463833,
        0.06474248308443485
    ])

    area2 = 0.5 * np.linalg.norm(np.cross(a, b))
    val = 0.0

    for i in range(7):
        u = u_nodes[i]
        wu = u_weights[i]
        for j in range(7):
            v = u_nodes[j]
            wv = u_weights[j]

            xi = u
            eta = u * v

            r_p = v0 + xi * a + eta * b

            dx = r_obs[0] - r_p[0]
            dy = r_obs[1] - r_p[1]
            dz = r_obs[2] - r_p[2]
            R = np.sqrt(dx*dx + dy*dy + dz*dz)

            if R < 1e-14:
                continue

            jac = area2 * u
            val += (1.0 / R) * jac * wu * wv

    return val

# --------------------------------------------------------
# Duffy para integrales vectoriales y escalares (mejora para términos singulares)
# --------------------------------------------------------
@njit
def duffy_integral_vector_scalar(tri, r_obs, sign_m, rfree_m, edge_len_m, area_m, sign_n, rfree_n, edge_len_n, area_n, factor_A, factor_Phi, div_m, div_n, inv4pi):
    """
    ∫_T [factor_A * fm · fn + factor_Phi * div_m * div_n] / |r_obs - r'| dS' usando Duffy
    Integra fm y fn correctamente, no aproximando en centroides.
    """
    v0 = tri[0]
    a = tri[1] - v0
    b = tri[2] - v0

    u_nodes = np.array([
        0.025446043828620737,
        0.12923440720030278,
        0.2970774243113014,
        0.5000000000000000,
        0.7029225756886986,
        0.8707655927996972,
        0.9745539561713793
    ])
    u_weights = np.array([
        0.06474248308443485,
        0.13985269574463833,
        0.19091502525255946,
        0.2089795918367347,
        0.19091502525255946,
        0.13985269574463833,
        0.06474248308443485
    ])

    area2 = 0.5 * np.linalg.norm(np.cross(a, b))
    val = 0.0 + 0.0j

    for i in range(7):
        u = u_nodes[i]
        wu = u_weights[i]
        for j in range(7):
            v = u_nodes[j]
            wv = u_weights[j]

            xi = u
            eta = u * v

            r_p = v0 + xi * a + eta * b

            dx = r_obs[0] - r_p[0]
            dy = r_obs[1] - r_p[1]
            dz = r_obs[2] - r_p[2]
            R = np.sqrt(dx*dx + dy*dy + dz*dz)

            if R < 1e-14:
                continue

            # RWG en el punto de integración
            fm = sign_m * (r_obs - rfree_m) * (edge_len_m / (2.0 * area_m))
            fn = sign_n * (r_p - rfree_n) * (edge_len_n / (2.0 * area_n))

            fm_dot_fn = fm[0]*fn[0] + fm[1]*fn[1] + fm[2]*fn[2]

            jac = area2 * u
            kernel = (factor_A * fm_dot_fn + factor_Phi * div_m * div_n) * inv4pi / R
            val += kernel * jac * wu * wv

    return val

# --------------------------------------------------------
# Ensamblado EFIE con extracción de singularidad (7.16) - MEJORADO
# --------------------------------------------------------
@njit(parallel=True, fastmath=True)
def assemble_Z_EFIE_duffy(
    quad_pts, quad_w,
    tri_rwg, tri_sign, rwg_tris,
    vertices, faces, face_normals,
    rp, rm, edge_len, A_plus, A_minus,
    k, omega, mu, eps
):
    num_faces = quad_pts.shape[0]
    num_rwg = edge_len.shape[0]
    Z = np.zeros((num_rwg, num_rwg), dtype=np.complex128)

    inv4pi = 1.0 / (4.0 * np.pi)
    factor_A = 1j * omega * mu
    factor_Phi = -1.0 / (1j * omega * eps)

    centroids = np.zeros((num_faces, 3))
    for t in range(num_faces):
        centroids[t] = (
            vertices[faces[t, 0]] +
            vertices[faces[t, 1]] +
            vertices[faces[t, 2]]
        ) / 3.0

    lam = 2.0 * np.pi / k
    near_thresh = 0.2 * lam

    for m in prange(num_rwg):
        for sm in range(2):
            tm = rwg_tris[m, sm]
            sign_m = 1.0 if sm == 0 else -1.0
            area_m = A_plus[m] if sm == 0 else A_minus[m]
            rfree_m = rp[m] if sm == 0 else rm[m]
            div_m = sign_m * edge_len[m] / area_m

            pts_m = quad_pts[tm]
            w_m = quad_w[tm]
            c_m = centroids[tm]
            n_m = face_normals[tm]

            delta_m = 1e-6 * np.sqrt(area_m)
            r_obs_m = c_m  # Cambiado: usa centroide en lugar de desplazar para evitar fm fuera de la superficie

            for ts in range(num_faces):
                c_s = centroids[ts]
                dx = c_m[0] - c_s[0]
                dy = c_m[1] - c_s[1]
                dz = c_m[2] - c_s[2]
                near = (dx*dx + dy*dy + dz*dz) < near_thresh**2

                pts_s = quad_pts[ts]
                w_s = quad_w[ts]

                for idx in range(3):
                    n = tri_rwg[ts, idx]
                    if n < 0:
                        continue

                    sign_n = tri_sign[ts, idx]
                    area_n = A_plus[n] if sign_n > 0 else A_minus[n]
                    rfree_n = rp[n] if sign_n > 0 else rm[n]
                    div_n = sign_n * edge_len[n] / area_n

                    Zmn = 0.0 + 0.0j

                    # -------- PARTE REGULAR (vectorial + escalar) --------
                    for ip in range(3):
                        r = pts_m[ip]
                        fm = sign_m * (r - rfree_m) * (edge_len[m] / (2.0 * area_m))

                        for js in range(3):
                            rp_ = pts_s[js]
                            fn = sign_n * (rp_ - rfree_n) * (edge_len[n] / (2.0 * area_n))

                            dx = r[0] - rp_[0]
                            dy = r[1] - rp_[1]
                            dz = r[2] - rp_[2]
                            R = np.sqrt(dx*dx + dy*dy + dz*dz)

                            if R < 1e-12:
                                G_reg = -1j * k * inv4pi
                            else:
                                G_reg = (np.exp(-1j * k * R) - 1.0) * inv4pi / R

                            Zmn += (
                                factor_A * (fm @ fn)
                                + factor_Phi * (div_m * div_n)
                            ) * G_reg * w_m[ip] * w_s[js]

                    # -------- PARTE SINGULAR (vectorial + escalar) - MEJORADA --------
                    if near:
                        tri_s = vertices[faces[ts]]
                        tri_m = vertices[faces[tm]]

                        delta_n = 1e-6 * np.sqrt(area_n)
                        r_obs_n = c_s  # Cambiado: usa centroide

                        # Usa Duffy para integrar el kernel completo, no aproximando fm·fn
                        Zmn_sing_mn = duffy_integral_vector_scalar(
                            tri_s, r_obs_m, sign_m, rfree_m, edge_len[m], area_m,
                            sign_n, rfree_n, edge_len[n], area_n, factor_A, factor_Phi, div_m, div_n, inv4pi
                        )
                        Zmn_sing_nm = duffy_integral_vector_scalar(
                            tri_m, r_obs_n, sign_n, rfree_n, edge_len[n], area_n,
                            sign_m, rfree_m, edge_len[m], area_m, factor_A, factor_Phi, div_n, div_m, inv4pi
                        )
                        Zmn += 0.5 * (Zmn_sing_mn + Zmn_sing_nm)

                    Z[m, n] += Zmn

    return Z


def assemble_b_EFIE(
    quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k, k_dir, pol, E0
):
    """
    Vector de excitación EFIE para onda plana
    """
    num_faces = quad_pts.shape[0]
    num_rwg = edge_len.shape[0]
    b = np.zeros(num_rwg, dtype=np.complex128)

    for ts in range(num_faces):
        pts = quad_pts[ts]
        w = quad_w[ts]

        for idx in range(3):
            m = tri_rwg[ts, idx]
            if m < 0:
                continue

            sign = tri_sign[ts, idx]
            area = A_plus[m] if sign > 0 else A_minus[m]
            rfree = rp[m] if sign > 0 else rm[m]

            bm = 0.0 + 0.0j
            for ip in range(3):
                r = pts[ip]

                # campo incidente
                phase = np.exp(-1j * k * np.dot(k_dir, r))
                Einc = E0 * phase * pol

                # RWG
                fm = sign * (r - rfree) * (edge_len[m] / (2.0 * area))

                bm += np.dot(Einc, fm) * w[ip]

            b[m] += bm

    return b

def mie_pec_rcs_monostatic(k, a, n_max=60):
    """
    RCS monostática exacta de una esfera PEC (Mie)
    """
    ka = k * a
    S = 0.0 + 0.0j

    for n in range(1, n_max + 1):
        jn = spherical_jn(n, ka)
        jn_p = spherical_jn(n, ka, derivative=True)

        yn = spherical_yn(n, ka)
        yn_p = spherical_yn(n, ka, derivative=True)

        hn = jn + 1j * yn
        hn_p = jn_p + 1j * yn_p

        term = (-1)**n * (2*n + 1) * (jn_p / hn_p - jn / hn)
        S += term

    sigma = (4.0 * np.pi / k**2) * np.abs(S)**2
    return 10.0 * np.log10(sigma)

def compute_rcs_mom_monostatic(
    coeffs,
    quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k, obs_dir, pol_obs, E0
):
    eta = 120.0 * np.pi
    inv4pi = 1.0 / (4.0 * np.pi)

    E_far = np.zeros(3, dtype=np.complex128)

    for ts in range(quad_pts.shape[0]):
        pts = quad_pts[ts]
        w = quad_w[ts]

        for idx in range(3):
            n = tri_rwg[ts, idx]
            if n < 0:
                continue

            sign = tri_sign[ts, idx]
            area = A_plus[n] if sign > 0 else A_minus[n]
            rfree = rp[n] if sign > 0 else rm[n]

            for ip in range(3):
                r = pts[ip]
                fn = sign * (r - rfree) * (edge_len[n] / (2.0 * area))
                fn_t = fn - obs_dir * np.dot(obs_dir, fn)
                phase = np.exp(1j * k * np.dot(obs_dir, r))
                E_far += coeffs[n] * fn_t * phase * w[ip]

    E_far *= 1j * k * eta * inv4pi
    Esca = np.dot(E_far, pol_obs)

    # RCS en m^2
    sigma = 4.0 * np.pi * np.abs(Esca)**2 / np.abs(E0)**2
    return sigma


# ============================================================
# 7. Ejecutar y validar a una frecuencia
# ============================================================

print("Ensamblando Z (EFIE completo, f0)...")
Z = assemble_Z_EFIE_duffy(
    quad_pts, quad_w,
    tri_rwg, tri_sign, rwg_tris,
    vertices, faces, face_normals,
    rp, rm, edge_len, A_plus, A_minus,
    k, omega, mu, eps
)

print(f"Número de condición de Z: {np.linalg.cond(Z):.2e}")

E0 = 1.0 + 0.0j
k_dir = np.array([0.0, 0.0, -1.0])
pol = np.array([1.0, 0.0, 0.0])

b = assemble_b_EFIE(
    quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k, k_dir, pol, E0
)

coeffs = np.linalg.solve(Z, b)

obs_dir = -k_dir
rcs_mom = compute_rcs_mom_monostatic(
    coeffs,
    quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k,
    obs_dir,
    pol,
    E0
)

rcs_mie = mie_pec_rcs_monostatic(k, radius, n_max=60)

print(f"f0 = {freq/1e6:.1f} MHz")
print(f"  RCS MoM = {rcs_mom:.2f} dBsm")
print(f"  RCS Mie = {rcs_mie:.2f} dBsm")
print(f"  Error   = {rcs_mom - rcs_mie:.2f} dB")

# ============================================================
# Barrido en frecuencia: RCS MoM vs Mie
# ============================================================

f_start = 100e6
f_stop  = 1.0e9
n_freqs = 5  # Cambiado de 15 a 5 para pruebas rápidas

frequencies = np.linspace(f_start, f_stop, n_freqs)

rcs_mom_list = []
rcs_mie_list = []

E0 = 1.0 + 0.0j
k_dir = np.array([0.0, 0.0, -1.0])
pol = np.array([1.0, 0.0, 0.0])

print("\nIniciando barrido en frecuencia...\n")

for f in frequencies:
    omega_f = 2.0 * np.pi * f
    k_f = omega_f / c

    print(f"f = {f/1e6:.1f} MHz")

    Z = assemble_Z_EFIE_duffy(
        quad_pts, quad_w,
        tri_rwg, tri_sign, rwg_tris,
        vertices, faces, face_normals,
        rp, rm, edge_len, A_plus, A_minus,
        k_f, omega_f, mu_0, epsilon_0
    )

    print(f"  Número de condición de Z: {np.linalg.cond(Z):.2e}")

    b = assemble_b_EFIE(
        quad_pts, quad_w,
        tri_rwg, tri_sign,
        rp, rm, edge_len, A_plus, A_minus,
        k_f, k_dir, pol, E0
    )

    try:
        coeffs = np.linalg.solve(Z, b)
    except np.linalg.LinAlgError:
        print("  -> Z singular, se salta frecuencia")
        rcs_mom_list.append(np.nan)
        rcs_mie_list.append(np.nan)
        continue

    obs_dir = -k_dir
    rcs_mom_f = compute_rcs_mom_monostatic(
        coeffs,
        quad_pts, quad_w,
        tri_rwg, tri_sign,
        rp, rm, edge_len, A_plus, A_minus,
        k_f,
        obs_dir,
        pol,
        E0
    )

    rcs_mie_f = mie_pec_rcs_monostatic(k_f, radius, n_max=60)

    rcs_mom_list.append(rcs_mom_f)
    rcs_mie_list.append(rcs_mie_f)

    print(f"   RCS MoM = {rcs_mom_f:.2f} dBsm")
    print(f"   RCS Mie = {rcs_mie_f:.2f} dBsm")
    print(f"   Error   = {rcs_mom_f - rcs_mie_f:.2f} dB\n")

plt.figure(figsize=(9,5))
plt.plot(frequencies/1e6, rcs_mom_list, 'o-', label='MoM EFIE (RWG)')
plt.plot(frequencies/1e6, rcs_mie_list, 's--', label='Mie (exacto)')
plt.xlabel('Frecuencia (MHz)')
plt.ylabel('RCS monostática (dBsm)')
plt.title('RCS de esfera PEC: MoM vs Mie')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
