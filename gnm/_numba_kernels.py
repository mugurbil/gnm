# -*- coding: utf-8 -*-
"""
Numba JIT-compiled kernels for the Gauss-Newton-Metropolis (GNM) sampler.
Accelerates Cholesky factorization, triangular solves, delayed-rejection
transition probabilities (_log_P and log_K), and dynamic cubic-Hermite
back-off optimization down to native LLVM machine code.
"""

import math
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(fastmath=False)
def njit_cholesky_and_solve(J, f, x, has_prior, H, Hm, reg, base_reg=1e-2):
    """
    Fused native kernel for _proposal_params:
      1. Forms P = J^T J (+ reg*I) (+ H if has_prior)
      2. Computes lower-triangular Cholesky factor L of P (with adaptive
         Levenberg-Marquardt regularization if rank-deficient)
      3. Computes L_inv_T = L^{-T}, log_det_L = sum(log(diag(L))), and
         Gauss-Newton proposal mean mu.
    """
    m_rows, n = J.shape
    JJ = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for r in range(m_rows):
                s += J[r, i] * J[r, j]
            JJ[i, j] = s
            JJ[j, i] = s

    P = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = JJ[i, j]
            if has_prior:
                val += H[i, j]
            if i == j and reg > 0.0:
                val += reg
            P[i, j] = val

    # Unregularized / regularized Cholesky decomposition in-place
    L = np.zeros((n, n), dtype=np.float64)
    success = False
    damp = 0.0
    for _ in range(14):
        ok = True
        max_diag = 1.0
        min_diag = 1e300
        for i in range(n):
            for j in range(i + 1):
                s = P[i, j]
                if i == j and damp > 0.0:
                    s += damp
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s <= 1e-14:
                        ok = False
                        break
                    d_val = math.sqrt(s)
                    L[i, i] = d_val
                    if d_val > max_diag:
                        max_diag = d_val
                    if d_val < min_diag:
                        min_diag = d_val
                else:
                    L[i, j] = s / L[j, j]
            if not ok:
                break
        if ok and (min_diag > 1e-6 * max_diag or damp > 0.0):
            success = True
            break
        damp = base_reg if damp == 0.0 else damp * 10.0

    if not success:
        for i in range(n):
            for j in range(n):
                L[i, j] = math.sqrt(base_reg) if i == j else 0.0

    # Compute log_det_L
    log_det_L = 0.0
    for i in range(n):
        log_det_L += math.log(abs(L[i, i]))

    # Compute L_inv (lower triangular) via forward substitution, then L_inv_T = L_inv.T
    L_inv_T = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        L_inv_T[col, col] = 1.0 / L[col, col]
        for row in range(col + 1, n):
            s = 0.0
            for k in range(col, row):
                s += L[row, k] * L_inv_T[col, k]
            L_inv_T[col, row] = -s / L[row, row]

    # Compute J^T f
    JTf = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for r in range(m_rows):
            s += J[r, i] * f[r, 0]
        JTf[i] = s

    rhs = np.empty(n, dtype=np.float64)
    if has_prior:
        for i in range(n):
            jj_x = 0.0
            for j in range(n):
                jj_x += JJ[i, j] * x[j, 0]
            rhs[i] = Hm[i, 0] - JTf[i] + jj_x
    else:
        for i in range(n):
            rhs[i] = JTf[i]

    # Forward solve L y = rhs
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = rhs[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]

    # Backward solve L^T step = y  (using L_inv_T @ y)
    mu = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(i, n):
            s += L_inv_T[i, k] * y[k]
        mu[i, 0] = s if has_prior else (x[i, 0] - s)

    return L, L_inv_T, log_det_L, mu


@njit(fastmath=False)
def njit_log_post(x, f, has_prior, m, H, ln_H):
    """Compute log posterior density and ||f||^2 in native code."""
    m_rows = f.shape[0]
    f_sq = 0.0
    for r in range(m_rows):
        v = f[r, 0]
        f_sq += v * v
    log_lik = -0.5 * f_sq
    if not has_prior:
        return log_lik, f_sq

    n = x.shape[0]
    quad = 0.0
    for i in range(n):
        di = x[i, 0] - m[i, 0]
        h_d = 0.0
        for j in range(n):
            h_d += H[i, j] * (x[j, 0] - m[j, 0])
        quad += di * h_d
    return ln_H - 0.5 * quad + log_lik, f_sq


@njit(fastmath=False)
def njit_propose(x, mu, L_inv_T, t, z_std):
    """Compute proposal z = (1-t)x + t*mu + sqrt(2t - t^2) * L^{-T} z_std."""
    n = x.shape[0]
    scale = math.sqrt(2.0 * t - t * t)
    z = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(i, n):
            s += L_inv_T[i, k] * z_std[k, 0]
        z[i, 0] = (1.0 - t) * x[i, 0] + t * mu[i, 0] + scale * s
    return z


@njit(fastmath=False)
def njit_log_K_single(z_x, x_x, x_mu, x_L, x_log_det_L, t):
    """Compute log K(Z | X, t) in native code."""
    n = z_x.shape[0]
    var_scale = 2.0 * t - t * t
    log_det_Lt = x_log_det_L - 0.5 * n * math.log(var_scale)
    quad = 0.0
    for j in range(n):
        s = 0.0
        for i in range(j, n):
            m_i = (1.0 - t) * x_x[i, 0] + t * x_mu[i, 0]
            s += x_L[i, j] * (z_x[i, 0] - m_i)
        quad += s * s
    return log_det_Lt - 0.5 * (quad / var_scale)


@njit(fastmath=False)
def _njit_log_P_recursive(target_idx, source_idx, k, X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, r_arr):
    if k == 0:
        return LOG_P_arr[source_idx] + njit_log_K_single(
            X_arr[target_idx],
            X_arr[source_idx],
            MU_arr[source_idx],
            L_arr[source_idx],
            LOG_DET_arr[source_idx],
            r_arr[0],
        )

    log_P_zk_z = _njit_log_P_recursive(k, source_idx, k - 1, X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, r_arr)
    log_P_z_zk = _njit_log_P_recursive(source_idx, k, k - 1, X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, r_arr)

    if (not math.isfinite(log_P_zk_z)) or (log_P_zk_z <= log_P_z_zk):
        return -math.inf

    log_diff = log_P_zk_z + math.log1p(-math.exp(log_P_z_zk - log_P_zk_z))
    return log_diff + njit_log_K_single(
        X_arr[target_idx],
        X_arr[source_idx],
        MU_arr[source_idx],
        L_arr[source_idx],
        LOG_DET_arr[source_idx],
        r_arr[k],
    )


@njit(fastmath=False)
def njit_stage_step(
    k,
    k_max,
    z,
    f_z,
    J_z,
    has_prior,
    m,
    H,
    Hm,
    ln_H,
    reg,
    log_P_z_x_prev,
    X_arr,
    MU_arr,
    L_arr,
    LOG_DET_arr,
    LOG_P_arr,
    F_SQ_arr,
    f0,
    J0,
    r_arr,
    dynamic,
    step_size,
    t1,
    t2,
    u_rand,
):
    """
    Fused native kernel for one stage k of the GNM delayed-rejection loop:
      1. Computes proposal parameters (L_z, L_inv_T_z, log_det_L_z, mu_z, log_p_z, f_sq_z)
      2. Stores candidate Z at index k + 1 of the stage buffers
      3. Evaluates recursive delayed-rejection acceptance probability A_z_x
      4. If accepted (u_rand <= A_z_x), updates index 0 of the stage buffers in-place
      5. If rejected and k < k_max, computes the next back-off step size r_arr[k + 1]
    """
    cand_idx = k + 1
    L_z, L_inv_T_z, log_det_L_z, mu_z = njit_cholesky_and_solve(
        J_z, f_z, z, has_prior, H, Hm, reg, 1e-2
    )
    log_p_z, f_sq_z = njit_log_post(z, f_z, has_prior, m, H, ln_H)

    X_arr[cand_idx] = z
    MU_arr[cand_idx] = mu_z
    L_arr[cand_idx] = L_z
    LOG_DET_arr[cand_idx] = log_det_L_z
    LOG_P_arr[cand_idx] = log_p_z
    F_SQ_arr[cand_idx] = f_sq_z

    r_k = r_arr[k]
    log_K_zx = njit_log_K_single(
        X_arr[cand_idx], X_arr[0], MU_arr[0], L_arr[0], LOG_DET_arr[0], r_k
    )
    log_P_z_x = log_P_z_x_prev + log_K_zx

    log_N = _njit_log_P_recursive(0, cand_idx, k, X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, r_arr)
    if not math.isfinite(log_N):
        A_z_x = 0.0
    elif log_N >= log_P_z_x:
        A_z_x = 1.0
    else:
        A_z_x = math.exp(log_N - log_P_z_x)

    if u_rand <= A_z_x:
        X_arr[0] = z
        MU_arr[0] = mu_z
        L_arr[0] = L_z
        LOG_DET_arr[0] = log_det_L_z
        LOG_P_arr[0] = log_p_z
        F_SQ_arr[0] = f_sq_z
        return True, L_inv_T_z, log_P_z_x, A_z_x

    log_P_z_x_next = log_P_z_x + math.log1p(-min(A_z_x, 1.0 - 1e-16))

    # Compute next back-off step size only if another stage remains
    if k < k_max:
        if not dynamic:
            r_new = r_k * step_size
        else:
            f_0 = F_SQ_arr[0]
            f_t = f_sq_z
            n = X_arr.shape[1]
            m_rows = f0.shape[0]
            dp_0 = 0.0
            dp_r = 0.0
            for r_idx in range(m_rows):
                jv0 = 0.0
                jvr = 0.0
                for j in range(n):
                    vj = (z[j, 0] - X_arr[0, j, 0]) / r_k
                    jv0 += J0[r_idx, j] * vj
                    jvr += J_z[r_idx, j] * vj
                dp_0 += 2.0 * f0[r_idx, 0] * jv0
                dp_r += 2.0 * f_z[r_idx, 0] * jvr

            t = r_k
            a = (t * dp_r - 2.0 * f_t + 2.0 * f_0 + t * dp_0) / (t * t * t)
            b = (f_t - f_0 - t * dp_0) / (t * t)
            c = dp_0
            A = 3.0 * a
            B = b - t * a
            C = B * B - A * c

            if C == 0.0:
                t_new = t1 * t if (c > 0.0 or dp_r > 0.0) else t2 * t
            elif A == 0.0:
                t_new = t1 * t if B == 0.0 else (-c / (2.0 * B))
            elif C > 0.0:
                t_new = (-B + math.sqrt(C)) / A
            else:
                t_new = t1 * t

            if (not math.isfinite(t_new)) or t_new < t1 * t:
                t_new = t1 * t
            elif t_new > t2 * t:
                t_new = t2 * t
            r_new = t_new
        r_arr[k + 1] = r_new

    return False, L_inv_T_z, log_P_z_x_next, A_z_x


def warmup_numba_kernels():
    """Pre-compile Numba kernels on a tiny 1x1 dummy input so first sample() call has zero JIT latency."""
    if not HAS_NUMBA:
        return
    x = np.zeros((1, 1), dtype=np.float64)
    f = np.zeros((1, 1), dtype=np.float64)
    J = np.ones((1, 1), dtype=np.float64)
    m = np.zeros((1, 1), dtype=np.float64)
    H = np.ones((1, 1), dtype=np.float64)
    Hm = np.zeros((1, 1), dtype=np.float64)
    L, L_inv_T, log_det_L, mu = njit_cholesky_and_solve(J, f, x, True, H, Hm, 0.0, 1e-2)
    log_p, f_sq = njit_log_post(x, f, True, m, H, 0.0)
    z = njit_propose(x, mu, L_inv_T, 1.0, x)

    X_arr = np.zeros((3, 1, 1), dtype=np.float64)
    MU_arr = np.zeros((3, 1, 1), dtype=np.float64)
    L_arr = np.ones((3, 1, 1), dtype=np.float64)
    LOG_DET_arr = np.zeros(3, dtype=np.float64)
    LOG_P_arr = np.zeros(3, dtype=np.float64)
    F_SQ_arr = np.zeros(3, dtype=np.float64)
    r_arr = np.ones(3, dtype=np.float64)
    njit_stage_step(
        0, 1, z, f, J, True, m, H, Hm, 0.0, 0.0, log_p,
        X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, F_SQ_arr,
        f, J, r_arr, True, 0.2, 0.05, 0.5, 0.5,
    )


warmup_numba_kernels()
