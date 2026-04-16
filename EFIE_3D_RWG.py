# ============================================================
# 3D MoM EFIE with RWG basis functions
# - Vector + scalar terms
# - Singularity extraction + Duffy transform
# - Numba-accelerated assembly
# ============================================================

import numpy as np
import trimesh
from scipy.constants import mu_0, epsilon_0, c
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# ============================================================
# 1. Physical parameters
# ============================================================

freq = 300e6
omega = 2.0 * np.pi * freq
k0 = omega / c
mu = mu_0
eps = epsilon_0

# ============================================================
# 2. Geometry — PEC sphere
# ============================================================

radius = 0.5  # 0.5 m sphere — ka ~ 3.14 at 300 MHz
mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)

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
print(f"ka:       {k0 * radius:.2f}")

# ============================================================
# 3. RWG basis function data
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
# 4. Quadrature — 7-point rule on triangles
# ============================================================

# Symmetric 7-point Gauss rule (degree 5) on the reference triangle
_bary7 = np.array([
    [1/3, 1/3, 1/3],
    [0.059715871789770, 0.470142064105115, 0.470142064105115],
    [0.470142064105115, 0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.470142064105115, 0.059715871789770],
    [0.797426985353087, 0.101286507323456, 0.101286507323456],
    [0.101286507323456, 0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.101286507323456, 0.797426985353087],
])
_wt7 = np.array([
    0.225,
    0.132394152788506, 0.132394152788506, 0.132394152788506,
    0.125939180544827, 0.125939180544827, 0.125939180544827,
])

NQUAD = 7
quad_pts = np.zeros((num_faces, NQUAD, 3))
quad_w = np.zeros((num_faces, NQUAD))

for t in range(num_faces):
    tri = vertices[faces[t]]
    area = face_areas[t]
    quad_pts[t] = _bary7 @ tri
    quad_w[t] = _wt7 * area

# ============================================================
# 5. Triangle → RWG map
# ============================================================

tri_rwg = -np.ones((num_faces, 3), dtype=np.int64)
tri_sign = np.zeros((num_faces, 3))
counter = np.zeros(num_faces, dtype=np.int64)

for m in range(num_rwg):
    tp, tm_ = rwg_tris[m]

    tri_rwg[tp, counter[tp]] = m
    tri_sign[tp, counter[tp]] = +1.0
    counter[tp] += 1

    tri_rwg[tm_, counter[tm_]] = m
    tri_sign[tm_, counter[tm_]] = -1.0
    counter[tm_] += 1


# ============================================================
# 6. Duffy integral for 1/R singularity extraction
# ============================================================

@njit
def duffy_integrals(tri, r_obs):
    """Integrate [1, r'] / |r_obs - r'| over triangle tri using Duffy (7x7).
    Returns out[0] = ∫ 1/R dS',  out[1:4] = ∫ r'/R dS'.
    """
    v0 = tri[0]
    a = tri[1] - v0
    b = tri[2] - v0

    u_nodes = np.array([
        0.025446043828620737, 0.12923440720030278,
        0.2970774243113014, 0.5,
        0.7029225756886986, 0.8707655927996972,
        0.9745539561713793
    ])
    u_weights = np.array([
        0.06474248308443485, 0.13985269574463833,
        0.19091502525255946, 0.2089795918367347,
        0.19091502525255946, 0.13985269574463833,
        0.06474248308443485
    ])

    # |a x b| is twice the triangle area; Duffy Jacobian is |a x b| * u
    cross_norm = np.linalg.norm(np.cross(a, b))
    out = np.zeros(4)

    for i in range(7):
        u = u_nodes[i]
        wu = u_weights[i]
        for j in range(7):
            v = u_nodes[j]
            wv = u_weights[j]

            xi = u
            eta = u * v
            rx = v0[0] + xi * a[0] + eta * b[0]
            ry = v0[1] + xi * a[1] + eta * b[1]
            rz = v0[2] + xi * a[2] + eta * b[2]

            dx = r_obs[0] - rx
            dy = r_obs[1] - ry
            dz = r_obs[2] - rz
            R = np.sqrt(dx*dx + dy*dy + dz*dz)

            if R < 1e-14:
                continue

            w = cross_norm * u * wu * wv / R
            out[0] += w
            out[1] += w * rx
            out[2] += w * ry
            out[3] += w * rz

    return out


# ============================================================
# 7. EFIE assembly with singularity extraction
# ============================================================

@njit(parallel=True)
def assemble_Z_EFIE(
    quad_pts, quad_w,
    tri_rwg, tri_sign, rwg_tris,
    vertices, faces, face_areas,
    rp, rm, edge_len, A_plus, A_minus,
    k, omega, mu, eps
):
    num_faces = quad_pts.shape[0]
    nq = quad_pts.shape[1]
    num_rwg = edge_len.shape[0]
    Z = np.zeros((num_rwg, num_rwg), dtype=np.complex128)

    inv4pi = 1.0 / (4.0 * np.pi)
    factor_A = 1j * omega * mu
    factor_Phi = 1.0 / (1j * omega * eps)

    # Precompute centroids
    centroids = np.zeros((num_faces, 3))
    for t in range(num_faces):
        for d in range(3):
            centroids[t, d] = (
                vertices[faces[t, 0], d] +
                vertices[faces[t, 1], d] +
                vertices[faces[t, 2], d]
            ) / 3.0

    lam = 2.0 * np.pi / k
    near_thresh2 = (0.25 * lam) ** 2

    for m in prange(num_rwg):
        for sm in range(2):
            tm = rwg_tris[m, sm]
            sign_m = 1.0 if sm == 0 else -1.0
            area_m = A_plus[m] if sm == 0 else A_minus[m]
            rfree_m = rp[m] if sm == 0 else rm[m]
            div_m = sign_m * edge_len[m] / area_m

            for ts in range(num_faces):
                # Check if source triangle has any RWG functions
                has_rwg = False
                for idx in range(3):
                    if tri_rwg[ts, idx] >= 0:
                        has_rwg = True
                        break
                if not has_rwg:
                    continue

                # Near-field check
                dx_c = centroids[tm, 0] - centroids[ts, 0]
                dy_c = centroids[tm, 1] - centroids[ts, 1]
                dz_c = centroids[tm, 2] - centroids[ts, 2]
                is_near = (dx_c*dx_c + dy_c*dy_c + dz_c*dz_c) < near_thresh2

                for idx in range(3):
                    n = tri_rwg[ts, idx]
                    if n < 0:
                        continue

                    sign_n = tri_sign[ts, idx]
                    area_n = A_plus[n] if sign_n > 0 else A_minus[n]
                    rfree_n = rp[n] if sign_n > 0 else rm[n]
                    div_n = sign_n * edge_len[n] / area_n

                    Zmn = 0.0 + 0.0j

                    # Regular part: G_reg = (exp(-jkR) - 1) / R
                    for ip in range(nq):
                        r = quad_pts[tm, ip]
                        fm = np.empty(3)
                        for d in range(3):
                            fm[d] = sign_m * (r[d] - rfree_m[d]) * (edge_len[m] / (2.0 * area_m))

                        for js in range(nq):
                            rp_ = quad_pts[ts, js]
                            fn = np.empty(3)
                            for d in range(3):
                                fn[d] = sign_n * (rp_[d] - rfree_n[d]) * (edge_len[n] / (2.0 * area_n))

                            dx = r[0] - rp_[0]
                            dy = r[1] - rp_[1]
                            dz = r[2] - rp_[2]
                            R = np.sqrt(dx*dx + dy*dy + dz*dz)

                            if R < 1e-14:
                                G_reg = -1j * k * inv4pi
                            else:
                                G_reg = (np.exp(-1j * k * R) - 1.0) * inv4pi / R

                            fm_dot_fn = fm[0]*fn[0] + fm[1]*fn[1] + fm[2]*fn[2]

                            Zmn += (
                                factor_A * fm_dot_fn
                                + factor_Phi * div_m * div_n
                            ) * G_reg * quad_w[tm, ip] * quad_w[ts, js]

                    # Singular part: 1/R integrated via Duffy or standard quadrature
                    if is_near:
                        for ip in range(nq):
                            r_obs = quad_pts[tm, ip]
                            fm = np.empty(3)
                            for d in range(3):
                                fm[d] = sign_m * (r_obs[d] - rfree_m[d]) * (edge_len[m] / (2.0 * area_m))

                            # Approximate fn at centroid of source triangle for the
                            # vector part; scalar part uses constant divergence.
                            fn_c = np.empty(3)
                            for d in range(3):
                                fn_c[d] = sign_n * (centroids[ts, d] - rfree_n[d]) * (edge_len[n] / (2.0 * area_n))

                            fm_dot_fn_c = fm[0]*fn_c[0] + fm[1]*fn_c[1] + fm[2]*fn_c[2]

                            tri_s = np.empty((3, 3))
                            for vi in range(3):
                                for d in range(3):
                                    tri_s[vi, d] = vertices[faces[ts, vi], d]

                            I = duffy_integrals(tri_s, r_obs)
                            I_sing = I[0]

                            Zmn += (
                                factor_A * fm_dot_fn_c
                                + factor_Phi * div_m * div_n
                            ) * inv4pi * I_sing * quad_w[tm, ip]

                    else:
                        # Standard quadrature for 1/R (far enough)
                        for ip in range(nq):
                            r = quad_pts[tm, ip]
                            fm = np.empty(3)
                            for d in range(3):
                                fm[d] = sign_m * (r[d] - rfree_m[d]) * (edge_len[m] / (2.0 * area_m))

                            for js in range(nq):
                                rp_ = quad_pts[ts, js]
                                fn = np.empty(3)
                                for d in range(3):
                                    fn[d] = sign_n * (rp_[d] - rfree_n[d]) * (edge_len[n] / (2.0 * area_n))

                                dx = r[0] - rp_[0]
                                dy = r[1] - rp_[1]
                                dz = r[2] - rp_[2]
                                R = np.sqrt(dx*dx + dy*dy + dz*dz)

                                if R < 1e-14:
                                    continue

                                G_1R = inv4pi / R

                                fm_dot_fn = fm[0]*fn[0] + fm[1]*fn[1] + fm[2]*fn[2]

                                Zmn += (
                                    factor_A * fm_dot_fn
                                    + factor_Phi * div_m * div_n
                                ) * G_1R * quad_w[tm, ip] * quad_w[ts, js]

                    Z[m, n] += Zmn

    return Z


def assemble_b_EFIE(
    quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k, k_dir, pol, E0
):
    """Excitation vector for a plane-wave incident field."""
    num_faces = quad_pts.shape[0]
    nq = quad_pts.shape[1]
    num_rwg = edge_len.shape[0]
    b = np.zeros(num_rwg, dtype=np.complex128)

    for ts in range(num_faces):
        for idx in range(3):
            m = tri_rwg[ts, idx]
            if m < 0:
                continue

            sign = tri_sign[ts, idx]
            area = A_plus[m] if sign > 0 else A_minus[m]
            rfree = rp[m] if sign > 0 else rm[m]

            bm = 0.0 + 0.0j
            for ip in range(nq):
                r = quad_pts[ts, ip]
                phase = np.exp(-1j * k * np.dot(k_dir, r))
                Einc = E0 * phase * pol
                fm = sign * (r - rfree) * (edge_len[m] / (2.0 * area))
                bm += np.dot(Einc, fm) * quad_w[ts, ip]

            b[m] += bm

    return b


# ============================================================
# 8. Mie series — exact PEC sphere monostatic RCS
# ============================================================

def mie_pec_rcs_monostatic(k, a, n_max=60):
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
    return sigma


def compute_rcs_mom_monostatic(
    coeffs, quad_pts, quad_w,
    tri_rwg, tri_sign,
    rp, rm, edge_len, A_plus, A_minus,
    k, obs_dir, pol_obs, E0
):
    eta0 = 120.0 * np.pi
    inv4pi = 1.0 / (4.0 * np.pi)
    nq = quad_pts.shape[1]

    E_far = np.zeros(3, dtype=np.complex128)

    for ts in range(quad_pts.shape[0]):
        for idx in range(3):
            n = tri_rwg[ts, idx]
            if n < 0:
                continue

            sign = tri_sign[ts, idx]
            area = A_plus[n] if sign > 0 else A_minus[n]
            rfree = rp[n] if sign > 0 else rm[n]

            for ip in range(nq):
                r = quad_pts[ts, ip]
                fn = sign * (r - rfree) * (edge_len[n] / (2.0 * area))
                fn_t = fn - obs_dir * np.dot(obs_dir, fn)
                phase = np.exp(1j * k * np.dot(obs_dir, r))
                E_far += coeffs[n] * fn_t * phase * quad_w[ts, ip]

    E_far *= 1j * k * eta0 * inv4pi
    Esca = np.dot(E_far, pol_obs)

    sigma = 4.0 * np.pi * np.abs(Esca)**2 / np.abs(E0)**2
    return sigma


# ============================================================
# 9. Run: frequency sweep — MoM vs Mie
# ============================================================

if __name__ == "__main__":

    f_start = 100e6
    f_stop = 600e6
    n_freqs = 6

    frequencies = np.linspace(f_start, f_stop, n_freqs)

    rcs_mom_list = []
    rcs_mie_list = []

    E0 = 1.0 + 0.0j
    k_dir = np.array([0.0, 0.0, -1.0])
    pol = np.array([1.0, 0.0, 0.0])

    print("\nStarting frequency sweep...\n")

    for f in frequencies:
        omega_f = 2.0 * np.pi * f
        k_f = omega_f / c
        ka = k_f * radius

        print(f"f = {f/1e6:.0f} MHz  (ka = {ka:.2f})")

        Z = assemble_Z_EFIE(
            quad_pts, quad_w,
            tri_rwg, tri_sign, rwg_tris,
            vertices, faces, face_areas,
            rp, rm, edge_len, A_plus, A_minus,
            k_f, omega_f, mu_0, epsilon_0
        )

        cond = np.linalg.cond(Z)
        print(f"  Condition number: {cond:.2e}")

        b = assemble_b_EFIE(
            quad_pts, quad_w,
            tri_rwg, tri_sign,
            rp, rm, edge_len, A_plus, A_minus,
            k_f, k_dir, pol, E0
        )

        try:
            coeffs = np.linalg.solve(Z, b)
        except np.linalg.LinAlgError:
            print("  -> Z singular, skipping frequency")
            rcs_mom_list.append(np.nan)
            rcs_mie_list.append(np.nan)
            continue

        obs_dir = -k_dir
        rcs_mom_f = compute_rcs_mom_monostatic(
            coeffs, quad_pts, quad_w,
            tri_rwg, tri_sign,
            rp, rm, edge_len, A_plus, A_minus,
            k_f, obs_dir, pol, E0
        )

        rcs_mie_f = mie_pec_rcs_monostatic(k_f, radius, n_max=60)

        rcs_mom_list.append(rcs_mom_f)
        rcs_mie_list.append(rcs_mie_f)

        rcs_mom_dB = 10.0 * np.log10(rcs_mom_f) if rcs_mom_f > 0 else -100
        rcs_mie_dB = 10.0 * np.log10(rcs_mie_f) if rcs_mie_f > 0 else -100

        print(f"  RCS MoM  = {rcs_mom_dB:.2f} dBsm")
        print(f"  RCS Mie  = {rcs_mie_dB:.2f} dBsm")
        print(f"  Error    = {rcs_mom_dB - rcs_mie_dB:.2f} dB\n")

    # Convert to dBsm for plotting
    rcs_mom_dB = [10*np.log10(x) if x > 0 else np.nan for x in rcs_mom_list]
    rcs_mie_dB = [10*np.log10(x) if x > 0 else np.nan for x in rcs_mie_list]

    plt.figure(figsize=(9, 5))
    plt.plot(frequencies/1e6, rcs_mom_dB, 'o-', label='MoM EFIE (RWG)')
    plt.plot(frequencies/1e6, rcs_mie_dB, 's--', label='Mie (exact)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Monostatic RCS (dBsm)')
    plt.title(f'PEC sphere RCS (r = {radius} m): MoM vs Mie')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/RCS_3D_sphere.png", dpi=150)
    print("Figure saved to figures/RCS_3D_sphere.png")
    plt.show()
