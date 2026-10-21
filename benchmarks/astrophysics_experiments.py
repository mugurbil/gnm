# -*- coding: utf-8 -*-
"""
Open-Source Astrophysics Benchmark Suite for `gnm`:
Compares `gnm` (Static, Dynamic, and `gnm.jax`) against `emcee`, HMC (L=10), MALA, and RWM
across 3 canonical observational astrophysics inverse problems:

  1. Keplerian Exoplanet Radial-Velocity (RV) Orbit Fitting (6D, RadVel / Keck-HIRES style)
     Parameters: theta = (P, K, e, omega, T_p, gamma)
     Stiffness: kappa(J^T J) ~ 2.5e2 + curved (e, K) periastron degeneracy banana.

  2. Type Ia Supernova w0-waCDM Dark Energy Cosmology (4D, Pantheon+ / JLA Hubble Diagram)
     Parameters: theta = (M_hub, Omega_m, w_0, w_a)
     Stiffness: kappa(J^T J) ~ 2.0e5 + curved (Omega_m, w_0, w_a) dark-energy banana.

  3. Gravitational Microlensing Paczynski Light-Curve Inversion (5D, OGLE / Roman PSPL)
     Parameters: theta = (t_0, t_E, u_0, F_s, F_b)
     Stiffness: kappa(J^T J) ~ 3.2e6 + severe (u_0, t_E, F_s, F_b) blending banana.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gnm
import jax
import jax.numpy as jnp
from gnm.jax import sample_jax
from benchmarks.benchmark_sota import run_rwm, run_mala, run_hmc, run_emcee

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 1. KEPLERIAN EXOPLANET RADIAL-VELOCITY ORBIT FITTING (6D)
# =============================================================================

def solve_kepler_numpy(M, e, n_iter=7):
    """Solve E - e*sin(E) = M via vectorized Newton-Raphson."""
    E = M + e * np.sin(M) + 0.5 * (e ** 2) * np.sin(2.0 * M)
    for _ in range(n_iter):
        f_val = E - e * np.sin(E) - M
        f_der = 1.0 - e * np.cos(E)
        E = E - f_val / f_der
    return E


def generate_exoplanet_rv_data(seed=42):
    rng = np.random.RandomState(seed)
    t_obs = np.sort(rng.uniform(0.0, 250.0, size=45))
    theta_true = np.array([42.35, 6.50, 0.52, 0.85, 14.20, 1.20])
    sigma_rv = 0.22

    P, K, e, omega, T_p, gamma = theta_true
    M = 2.0 * np.pi * (t_obs - T_p) / P
    E = solve_kepler_numpy(M, e)
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    sqrt_1me2 = np.sqrt(1.0 - e * e)
    denom = 1.0 - e * cos_E
    cos_nu = (cos_E - e) / denom
    sin_nu = (sqrt_1me2 * sin_E) / denom
    rv_clean = K * (cos_nu * np.cos(omega) - sin_nu * np.sin(omega) + e * np.cos(omega)) + gamma
    rv_obs = rv_clean + rng.normal(0.0, sigma_rv, size=t_obs.size)
    return {
        "t_obs": t_obs,
        "rv_obs": rv_obs,
        "sigma": sigma_rv,
        "theta_true": theta_true,
    }


def exoplanet_rv_model(theta, data):
    P, K, e, omega, T_p, gamma = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    if P <= 10.0 or P >= 100.0 or K <= 0.2 or e <= 0.02 or e >= 0.90 or omega <= -np.pi or omega >= np.pi:
        return 0, None, None

    t = data["t_obs"]
    y = data["rv_obs"]
    s = data["sigma"]
    inv_s = 1.0 / s

    M = 2.0 * np.pi * (t - T_p) / P
    E = solve_kepler_numpy(M, e)
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    one_me2 = 1.0 - e * e
    sqrt_1me2 = np.sqrt(one_me2)
    denom = 1.0 - e * cos_E

    cos_nu = (cos_E - e) / denom
    sin_nu = (sqrt_1me2 * sin_E) / denom
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

    bracket = cos_nu * cos_w - sin_nu * sin_w + e * cos_w
    rv_pred = K * bracket + gamma
    f = (rv_pred - y) * inv_s

    dnu_dE = sqrt_1me2 / denom
    dcosnu_sinw_dnu = -sin_nu * cos_w - cos_nu * sin_w
    dE_dM = 1.0 / denom
    dnu_dM = dnu_dE * dE_dM
    dnu_de = (sin_E / (sqrt_1me2 * (denom ** 2))) * (2.0 - e * cos_E - e * e)

    dM_dP = -M / P
    dM_dTp = -2.0 * np.pi / P

    m = t.size
    J = np.empty((m, 6), dtype=np.float64)
    J[:, 0] = K * dcosnu_sinw_dnu * dnu_dM * dM_dP * inv_s
    J[:, 1] = bracket * inv_s
    J[:, 2] = K * (dcosnu_sinw_dnu * dnu_de + cos_w) * inv_s
    J[:, 3] = K * (-cos_nu * sin_w - sin_nu * cos_w - e * sin_w) * inv_s
    J[:, 4] = K * dcosnu_sinw_dnu * dnu_dM * dM_dTp * inv_s
    J[:, 5] = inv_s
    return 1, f[:, None], J


def make_exoplanet_jax_fn(data):
    t_j = jnp.asarray(data["t_obs"])
    y_j = jnp.asarray(data["rv_obs"])
    inv_s = 1.0 / float(data["sigma"])
    ones_col = jnp.full((t_j.size,), inv_s, dtype=jnp.float64)

    def res_and_jac(theta):
        P, K, e_raw, omega, T_p, gamma = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
        e = jnp.clip(e_raw, 0.02, 0.89)
        M = 2.0 * jnp.pi * (t_j - T_p) / P
        E = M + e * jnp.sin(M) + 0.5 * (e ** 2) * jnp.sin(2.0 * M)
        for _ in range(6):
            E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
        cos_E = jnp.cos(E)
        sin_E = jnp.sin(E)
        one_me2 = 1.0 - e * e
        sqrt_1me2 = jnp.sqrt(one_me2)
        denom = 1.0 - e * cos_E
        cos_nu = (cos_E - e) / denom
        sin_nu = (sqrt_1me2 * sin_E) / denom
        cos_w = jnp.cos(omega)
        sin_w = jnp.sin(omega)
        bracket = cos_nu * cos_w - sin_nu * sin_w + e * cos_w
        f = (K * bracket + gamma - y_j) * inv_s
        dnu_dM = sqrt_1me2 / (denom * denom)
        dcosnu_sinw_dnu = -sin_nu * cos_w - cos_nu * sin_w
        dnu_de = (sin_E / (sqrt_1me2 * (denom ** 2))) * (2.0 - e * cos_E - e * e)
        dM_dP = -M / P
        dM_dTp = -2.0 * jnp.pi / P
        J = jnp.column_stack([
            K * dcosnu_sinw_dnu * dnu_dM * dM_dP * inv_s,
            bracket * inv_s,
            K * (dcosnu_sinw_dnu * dnu_de + cos_w) * inv_s,
            K * (-cos_nu * sin_w - sin_nu * cos_w - e * sin_w) * inv_s,
            K * dcosnu_sinw_dnu * dnu_dM * dM_dTp * inv_s,
            ones_col,
        ])
        return f, J
    return res_and_jac


# =============================================================================
# 2. TYPE Ia SUPERNOVA w0-waCDM DARK ENERGY COSMOLOGY (4D)
# =============================================================================

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def generate_supernova_cosmology_data(seed=42):
    rng = np.random.RandomState(seed)
    z_obs = np.geomspace(0.015, 1.75, 60)
    theta_true = np.array([43.15, 0.30, -1.00, 0.00])
    sigma_mu = 0.065

    z_nodes = 0.5 * z_obs[:, None] * (1.0 + _GL_X[None, :])
    z_weights = 0.5 * z_obs[:, None] * _GL_W[None, :]
    data = {
        "z_obs": z_obs,
        "z_nodes": z_nodes,
        "z_weights": z_weights,
        "sigma": sigma_mu,
        "theta_true": theta_true,
        "mu_obs": np.zeros_like(z_obs),
    }
    _, f_zero, _ = supernova_wcdm_model(theta_true, data)
    mu_clean = f_zero[:, 0] * sigma_mu
    data["mu_obs"] = mu_clean + rng.normal(0.0, sigma_mu, size=z_obs.size)
    return data


def supernova_wcdm_model(theta, data):
    M_hub, Om, w0, wa = theta[0], theta[1], theta[2], theta[3]
    if Om <= 0.04 or Om >= 0.85 or w0 <= -2.8 or w0 >= -0.2 or wa <= -3.5 or wa >= 2.5:
        return 0, None, None

    z = data["z_obs"]
    zn = data["z_nodes"]
    zw = data["z_weights"]
    y = data["mu_obs"]
    s = data["sigma"]
    inv_s = 1.0 / s

    opz = 1.0 + zn
    opz3 = opz ** 3
    ln_opz = np.log(opz)
    z_over_opz = zn / opz

    de_factor = (opz ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * z_over_opz)
    E2 = Om * opz3 + (1.0 - Om) * de_factor
    if np.any(E2 <= 1e-8):
        return 0, None, None

    E = np.sqrt(E2)
    inv_E = 1.0 / E
    inv_E3 = inv_E / E2

    I_z = np.sum(zw * inv_E, axis=1)
    five_over_ln10 = 5.0 / np.log(10.0)
    mu_pred = M_hub + five_over_ln10 * np.log((1.0 + z) * I_z)
    f = (mu_pred - y) * inv_s

    dE2_dOm = opz3 - de_factor
    dE2_dw0 = (1.0 - Om) * de_factor * (3.0 * ln_opz)
    dE2_dwa = (1.0 - Om) * de_factor * (3.0 * (ln_opz - z_over_opz))

    dI_dOm = -0.5 * np.sum(zw * inv_E3 * dE2_dOm, axis=1)
    dI_dw0 = -0.5 * np.sum(zw * inv_E3 * dE2_dw0, axis=1)
    dI_dwa = -0.5 * np.sum(zw * inv_E3 * dE2_dwa, axis=1)

    pref = (five_over_ln10 / I_z) * inv_s
    m = z.size
    J = np.empty((m, 4), dtype=np.float64)
    J[:, 0] = inv_s
    J[:, 1] = pref * dI_dOm
    J[:, 2] = pref * dI_dw0
    J[:, 3] = pref * dI_dwa
    return 1, f[:, None], J


def make_supernova_jax_fn(data):
    z_j = jnp.asarray(data["z_obs"])
    zn_j = jnp.asarray(data["z_nodes"])
    zw_j = jnp.asarray(data["z_weights"])
    y_j = jnp.asarray(data["mu_obs"])
    inv_s = 1.0 / float(data["sigma"])
    opz = 1.0 + zn_j
    opz3 = opz ** 3
    ln_opz = jnp.log(opz)
    z_over_opz = zn_j / opz
    five_over_ln10 = 5.0 / np.log(10.0)
    ones_col = jnp.full((z_j.size,), inv_s, dtype=jnp.float64)

    def res_and_jac(theta):
        M_hub, Om_raw, w0_raw, wa_raw = theta[0], theta[1], theta[2], theta[3]
        Om = jnp.clip(Om_raw, 0.04, 0.85)
        w0 = jnp.clip(w0_raw, -2.8, -0.2)
        wa = jnp.clip(wa_raw, -3.5, 2.5)
        de_factor = (opz ** (3.0 * (1.0 + w0 + wa))) * jnp.exp(-3.0 * wa * z_over_opz)
        E2 = jnp.maximum(Om * opz3 + (1.0 - Om) * de_factor, 1e-8)
        E = jnp.sqrt(E2)
        inv_E = 1.0 / E
        inv_E3 = inv_E / E2
        I_z = jnp.sum(zw_j * inv_E, axis=1)
        mu_pred = M_hub + five_over_ln10 * jnp.log((1.0 + z_j) * I_z)
        f = (mu_pred - y_j) * inv_s
        dE2_dOm = opz3 - de_factor
        dE2_dw0 = (1.0 - Om) * de_factor * (3.0 * ln_opz)
        dE2_dwa = (1.0 - Om) * de_factor * (3.0 * (ln_opz - z_over_opz))
        dI_dOm = -0.5 * jnp.sum(zw_j * inv_E3 * dE2_dOm, axis=1)
        dI_dw0 = -0.5 * jnp.sum(zw_j * inv_E3 * dE2_dw0, axis=1)
        dI_dwa = -0.5 * jnp.sum(zw_j * inv_E3 * dE2_dwa, axis=1)
        pref = (five_over_ln10 / I_z) * inv_s
        J = jnp.column_stack([ones_col, pref * dI_dOm, pref * dI_dw0, pref * dI_dwa])
        return f, J
    return res_and_jac


# =============================================================================
# 3. GRAVITATIONAL MICROLENSING PACZYNSKI LIGHT CURVE (5D)
# =============================================================================

def generate_microlensing_data(seed=42):
    rng = np.random.RandomState(seed)
    t_obs = np.linspace(0.0, 100.0, 80)
    theta_true = np.array([50.0, 22.0, 0.25, 1.40, 0.60])
    sigma_flux = 0.035

    t0, tE, u0, Fs, Fb = theta_true
    tau = (t_obs - t0) / tE
    u = np.sqrt(u0 * u0 + tau * tau)
    A = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
    flux_clean = Fs * A + Fb
    flux_obs = flux_clean + rng.normal(0.0, sigma_flux, size=t_obs.size)
    return {
        "t_obs": t_obs,
        "flux_obs": flux_obs,
        "sigma": sigma_flux,
        "theta_true": theta_true,
    }


def microlensing_model(theta, data):
    t0, tE, u0, Fs, Fb = theta[0], theta[1], theta[2], theta[3], theta[4]
    if t0 <= 20.0 or t0 >= 80.0 or tE <= 3.0 or tE >= 80.0 or u0 <= 0.02 or u0 >= 1.5 or Fs <= 0.05 or Fb <= -0.5:
        return 0, None, None

    t = data["t_obs"]
    y = data["flux_obs"]
    s = data["sigma"]
    inv_s = 1.0 / s

    tau = (t - t0) / tE
    u2 = u0 * u0 + tau * tau
    u = np.sqrt(u2)
    u2p4 = u2 + 4.0
    sqrt_u2p4 = np.sqrt(u2p4)

    A = (u2 + 2.0) / (u * sqrt_u2p4)
    pred = Fs * A + Fb
    f = (pred - y) * inv_s

    dA_du = -8.0 / (u2 * u2p4 * sqrt_u2p4)
    du_dt0 = -tau / (u * tE)
    du_dtE = -(tau * tau) / (u * tE)
    du_du0 = u0 / u

    m = t.size
    J = np.empty((m, 5), dtype=np.float64)
    J[:, 0] = Fs * dA_du * du_dt0 * inv_s
    J[:, 1] = Fs * dA_du * du_dtE * inv_s
    J[:, 2] = Fs * dA_du * du_du0 * inv_s
    J[:, 3] = A * inv_s
    J[:, 4] = inv_s
    return 1, f[:, None], J


def make_microlensing_jax_fn(data):
    t_j = jnp.asarray(data["t_obs"])
    y_j = jnp.asarray(data["flux_obs"])
    inv_s = 1.0 / float(data["sigma"])
    ones_col = jnp.full((t_j.size,), inv_s, dtype=jnp.float64)

    def res_and_jac(theta):
        t0, tE_raw, u0_raw, Fs_raw, Fb = theta[0], theta[1], theta[2], theta[3], theta[4]
        tE = jnp.clip(tE_raw, 3.0, 80.0)
        u0 = jnp.clip(u0_raw, 0.02, 1.5)
        Fs = jnp.maximum(Fs_raw, 0.05)
        tau = (t_j - t0) / tE
        u2 = u0 * u0 + tau * tau
        u = jnp.sqrt(u2)
        u2p4 = u2 + 4.0
        sqrt_u2p4 = jnp.sqrt(u2p4)
        A = (u2 + 2.0) / (u * sqrt_u2p4)
        f = (Fs * A + Fb - y_j) * inv_s
        dA_du = -8.0 / (u2 * u2p4 * sqrt_u2p4)
        du_dt0 = -tau / (u * tE)
        du_dtE = -(tau * tau) / (u * tE)
        du_du0 = u0 / u
        J = jnp.column_stack([
            Fs * dA_du * du_dt0 * inv_s,
            Fs * dA_du * du_dtE * inv_s,
            Fs * dA_du * du_du0 * inv_s,
            A * inv_s,
            ones_col,
        ])
        return f, J
    return res_and_jac


# =============================================================================
# VERIFY ANALYTICAL JACOBIANS & RUN BENCHMARK SUITE
# =============================================================================

def verify_jacobian(model_fn, theta, data, eps=1e-7):
    chi, f0, J_ana = model_fn(theta, data)
    assert chi == 1
    J_fd = np.zeros_like(J_ana)
    for j in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += eps
        tm[j] -= eps
        _, fp, _ = model_fn(tp, data)
        _, fm, _ = model_fn(tm, data)
        J_fd[:, j] = (fp[:, 0] - fm[:, 0]) / (2.0 * eps)
    rel_err = np.linalg.norm(J_ana - J_fd) / np.linalg.norm(J_fd)
    assert rel_err < 1e-6, f"Jacobian mismatch: rel_err = {rel_err}"
    return rel_err


def run_single_astrophysics_benchmark(
    name,
    model_fn,
    jax_fn,
    data,
    prior_m,
    prior_H,
    rwm_scale,
    mala_tau,
    hmc_eps,
    n_budget=24000,
    n_gnm_samples=16000,
):
    x0 = data["theta_true"].copy()
    jac_err = verify_jacobian(model_fn, x0, data)
    _, _, J0 = model_fn(x0, data)
    cond_JJ = float(np.linalg.cond(J0.T @ J0))

    # 1. RWM
    np.random.seed(2026)
    w_rwm = gnm.function(model_fn, data)
    out_rwm = run_rwm(x0, w_rwm, n_budget, step_scale=rwm_scale, prior_m=prior_m, prior_H=prior_H)
    c_rwm = out_rwm["chain"][2400:]
    tau_rwm = float(np.max(gnm.autocorr_time(c_rwm)))
    ess_rwm = len(c_rwm) / tau_rwm

    # 2. MALA
    np.random.seed(2026)
    w_mala = gnm.function(model_fn, data)
    out_mala = run_mala(x0, w_mala, n_budget, tau=mala_tau, prior_m=prior_m, prior_H=prior_H)
    c_mala = out_mala["chain"][2400:]
    tau_mala = float(np.max(gnm.autocorr_time(c_mala)))
    ess_mala = len(c_mala) / tau_mala

    # 3. HMC (L=10)
    np.random.seed(2026)
    w_hmc = gnm.function(model_fn, data)
    out_hmc = run_hmc(x0, w_hmc, n_budget, eps=hmc_eps, n_leapfrog=10, prior_m=prior_m, prior_H=prior_H)
    c_hmc = out_hmc["chain"][240:]
    tau_hmc = float(np.max(gnm.autocorr_time(c_hmc)))
    ess_hmc = len(c_hmc) / tau_hmc

    # 4. emcee (Ensemble Stretch)
    np.random.seed(2026)
    w_emcee = gnm.function(model_fn, data)
    out_emcee = run_emcee(x0, w_emcee, n_budget, prior_m=prior_m, prior_H=prior_H)
    tau_emcee = float(np.max(out_emcee["walker_taus"]))
    ess_emcee = (0.9 * len(out_emcee["chain"])) / tau_emcee

    # 5. gnm (Static, k=1, Numba)
    np.random.seed(2026)
    s_st = gnm.sampler(x0, model_fn, data)
    s_st.prior(prior_m, prior_H)
    s_st.static(1, 0.25)
    t0 = time.perf_counter()
    s_st.sample(n_gnm_samples)
    time_st = time.perf_counter() - t0
    c_st = s_st.burn(n_gnm_samples // 10)
    tau_st = float(np.max(s_st.acor()))
    ess_st = len(c_st) / tau_st

    # 6. gnm (Dynamic, k=1, Numba)
    np.random.seed(2026)
    s_dy = gnm.sampler(x0, model_fn, data)
    s_dy.prior(prior_m, prior_H)
    s_dy.dynamic(1)
    t0 = time.perf_counter()
    s_dy.sample(n_gnm_samples)
    time_dy = time.perf_counter() - t0
    c_dy = s_dy.burn(n_gnm_samples // 10)
    tau_dy = float(np.max(s_dy.acor()))
    ess_dy = len(c_dy) / tau_dy

    # 7. gnm.jax (Dynamic, k=1, @jax.jit with unrolled Cholesky)
    _ = sample_jax(
        jax_fn,
        jnp.asarray(x0),
        10,
        key=jax.random.PRNGKey(0),
        jac_fn="tuple",
        prior_mean=jnp.asarray(prior_m),
        prior_precision=jnp.asarray(prior_H),
        max_steps=1,
        dynamic=True,
        n_chains=1,
    )
    t0 = time.perf_counter()
    c_jax_full, _ = sample_jax(
        jax_fn,
        jnp.asarray(x0),
        n_gnm_samples,
        key=jax.random.PRNGKey(2026),
        jac_fn="tuple",
        prior_mean=jnp.asarray(prior_m),
        prior_precision=jnp.asarray(prior_H),
        max_steps=1,
        dynamic=True,
        n_chains=1,
    )
    c_jax_full.block_until_ready()
    time_jax = time.perf_counter() - t0
    c_jax = np.asarray(c_jax_full)[n_gnm_samples // 10:]
    tau_jax = float(np.max(gnm.autocorr_time(c_jax)))
    ess_jax = len(c_jax) / tau_jax

    print("\n" + "=" * 112)
    print(f"{name} (n={x0.size}D, m={J0.shape[0]}, kappa(J^T J) = {cond_JJ:.2e}, Jacobian FD err = {jac_err:.1e})")
    print("=" * 112)
    print(f"{'Sampler':<30} | {'Evals':>8} | {'Time (s)':>8} | {'s/20k Ev':>8} | {'Max tau':>8} | {'Total ESS':>9} | {'ESS / sec':>10} | {'Sec / 500 ESS':>14}")
    print("-" * 112)
    rows = [
        ("Random-Walk Metropolis", out_rwm["evals"], out_rwm["time"], tau_rwm, ess_rwm),
        ("MALA (1st-Order Langevin)", out_mala["evals"], out_mala["time"], tau_mala, ess_mala),
        ("HMC (L=10 Leapfrog)", out_hmc["evals"], out_hmc["time"], tau_hmc, ess_hmc),
        ("emcee (Ensemble Stretch)", out_emcee["evals"], out_emcee["time"], tau_emcee, ess_emcee),
        ("gnm (Static, k=1, Numba)", s_st.call_count, time_st, tau_st, ess_st),
        ("gnm (Dynamic, k=1, Numba)", s_dy.call_count, time_dy, tau_dy, ess_dy),
        ("gnm.jax (Dynamic, @jax.jit)", s_dy.call_count, time_jax, tau_jax, ess_jax),
    ]
    for s_name, ev, tm, tau, ess in rows:
        s_20k = tm * (20000.0 / ev)
        ess_s = ess / tm
        sec_500 = 500.0 / max(ess_s, 1e-3)
        print(f"{s_name:<30} | {ev:8d} | {tm:8.3f} | {s_20k:8.3f} | {tau:8.1f} | {ess:9.1f} | {ess_s:10.1f} | {sec_500:14.2f}")
    print("-" * 112)
    speedup_vs_emcee = (max(ess_st / time_st, ess_dy / time_dy, ess_jax / time_jax)) / (ess_emcee / out_emcee["time"])
    print(f"--> Peak `gnm` Throughput Speedup vs. `emcee`: {speedup_vs_emcee:.1f}x faster!")

    return {
        "short_name": name.split(".")[1].split("(")[0].strip(),
        "cond_JJ": cond_JJ,
        "rows": rows,
        "c_dy": c_dy,
        "c_st": c_st,
        "c_emcee": out_emcee["chain"][len(out_emcee["chain"]) // 10:],
        "J0": J0,
    }


def plot_astrophysics_runtime_comparison(results):
    """
    Generate a dedicated 4-panel Runtime & Statistical Efficiency comparison figure
    across all 3 Astrophysics Benchmarks:
      Panel A: Per-Evaluation Wall-Clock Runtime (Seconds per 20,000 Evaluations — Lower is Faster)
      Panel B: Statistical Mixing Efficiency (ESS per 1,000 Evaluations — Log Scale, Higher is Better)
      Panel C: Overall Sampling Throughput (ESS / sec — Log Scale, Higher is Faster)
      Panel D: Wall-Clock Time to Generate 500 Independent Samples (Seconds — Log Scale, Lower is Faster)
    """
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.2))

    bench_labels = [
        "1. Exoplanet Keplerian RV\n(6D, $\\kappa = 2.5 \\times 10^2$)",
        "2. Supernova $w_0 w_a$CDM\n(4D, $\\kappa = 2.0 \\times 10^5$)",
        "3. Microlensing Light Curve\n(5D, $\\kappa = 3.2 \\times 10^6$)",
    ]
    sampler_names = [
        "RWM",
        "MALA",
        "HMC (L=10)",
        "emcee",
        "gnm (Static, Numba)",
        "gnm (Dynamic, Numba)",
        "gnm.jax (@jax.jit)",
    ]
    colors = ["#ef4444", "#9ca3af", "#f59e0b", "#3b82f6", "#34d399", "#10b981", "#047857"]

    n_b = len(results)
    n_s = len(sampler_names)
    x = np.arange(n_b)
    width = 0.115
    offsets = (np.arange(n_s) - (n_s - 1) / 2.0) * width

    # Extract metrics (n_b, n_s)
    sec_per_20k = np.zeros((n_b, n_s))
    ess_per_1k = np.zeros((n_b, n_s))
    ess_per_sec = np.zeros((n_b, n_s))
    sec_per_500 = np.zeros((n_b, n_s))

    for b_idx, res in enumerate(results):
        for s_idx, (_, ev, tm, _tau, ess) in enumerate(res["rows"]):
            sec_per_20k[b_idx, s_idx] = tm * (20000.0 / ev)
            ess_per_1k[b_idx, s_idx] = ess * (1000.0 / ev)
            ess_per_sec[b_idx, s_idx] = ess / tm
            sec_per_500[b_idx, s_idx] = 500.0 / max(ess / tm, 1e-3)

    # Panel A: Seconds per 20,000 Model Evaluations (Linear Scale — Lower is Faster)
    ax = axes[0, 0]
    for s_idx in range(n_s):
        bars = ax.bar(x + offsets[s_idx], sec_per_20k[:, s_idx], width,
                      label=sampler_names[s_idx], color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
        for bar, v in zip(bars, sec_per_20k[:, s_idx]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.05, f"{v:.2f}s",
                    ha="center", va="bottom", fontsize=7.0, fontweight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels)
    ax.set_ylim(0.0, np.max(sec_per_20k) * 1.28)
    ax.set_ylabel("Wall-Clock Time per 20k Evals (Seconds)")
    ax.set_title("(A) Raw Per-Evaluation Runtime Cost (Seconds per 20,000 Evals)\nLinear Scale — Lower is Faster")
    ax.legend(loc="upper left", ncol=2, fontsize=8.0, frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    # Panel B: ESS per 1,000 Evaluations (Log Scale — Higher is Better)
    ax = axes[0, 1]
    for s_idx in range(n_s):
        bars = ax.bar(x + offsets[s_idx], ess_per_1k[:, s_idx], width,
                      label=sampler_names[s_idx], color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
        for bar, v in zip(bars, ess_per_1k[:, s_idx]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.15, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.0, fontweight="bold", rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels)
    ax.set_ylim(0.1, np.max(ess_per_1k) * 6.0)
    ax.set_ylabel("Effective Samples per 1,000 Evals (Log Scale)")
    ax.set_title("(B) Statistical Mixing Efficiency (ESS per 1,000 Model Evals)\nLog Scale — Higher is Better")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # Panel C: Overall Sampling Throughput (ESS / sec, Log Scale — Higher is Faster)
    ax = axes[1, 0]
    for s_idx in range(n_s):
        bars = ax.bar(x + offsets[s_idx], ess_per_sec[:, s_idx], width,
                      label=sampler_names[s_idx], color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
        for bar, v in zip(bars, ess_per_sec[:, s_idx]):
            lbl = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.15, lbl,
                    ha="center", va="bottom", fontsize=7.0, fontweight="bold", rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels)
    ax.set_ylim(0.8, np.max(ess_per_sec) * 6.5)
    ax.set_ylabel("Effective Samples / Second (Log Scale)")
    ax.set_title("(C) Net Wall-Clock Throughput (ESS / sec)\nLog Scale — Higher is Faster")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # Panel D: Wall-Clock Time to Generate 500 Independent Samples (Log Scale — Lower is Faster)
    ax = axes[1, 1]
    for s_idx in range(n_s):
        bars = ax.bar(x + offsets[s_idx], sec_per_500[:, s_idx], width,
                      label=sampler_names[s_idx], color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
        for bar, v in zip(bars, sec_per_500[:, s_idx]):
            lbl = f"{v:.2f}s" if v < 10 else f"{v:.0f}s"
            ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.15, lbl,
                    ha="center", va="bottom", fontsize=7.0, fontweight="bold", rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels)
    ax.set_ylim(0.03, np.max(sec_per_500) * 6.5)
    ax.set_ylabel("Wall-Clock Time for 500 ESS (Seconds, Log Scale)")
    ax.set_title("(D) Time to Generate 500 Independent Posterior Samples\nLog Scale — Lower is Faster")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    plt.tight_layout()
    out_rt = os.path.join(FIG_DIR, "fig6_astrophysics_runtime_comparison.png")
    plt.savefig(out_rt, dpi=200)
    plt.close(fig)


def main():
    # 1. Exoplanet RV (6D)
    rv_data = generate_exoplanet_rv_data(seed=42)
    rv_prior_m = np.array([42.35, 6.50, 0.50, 0.85, 14.20, 1.20])
    rv_prior_H = np.diag([1.0, 0.5, 4.0, 1.0, 0.5, 1.0])
    res_rv = run_single_astrophysics_benchmark(
        "1. KEPLERIAN EXOPLANET RADIAL-VELOCITY ORBIT (RadVel / Keck-HIRES Style)",
        exoplanet_rv_model,
        make_exoplanet_jax_fn(rv_data),
        rv_data,
        rv_prior_m,
        rv_prior_H,
        rwm_scale=0.008,
        mala_tau=2e-6,
        hmc_eps=0.0015,
    )

    # 2. Supernova Cosmology (4D)
    sn_data = generate_supernova_cosmology_data(seed=42)
    sn_prior_m = np.array([43.15, 0.30, -1.00, 0.00])
    sn_prior_H = np.diag([4.0, 8.0, 1.5, 0.8])
    res_sn = run_single_astrophysics_benchmark(
        "2. TYPE Ia SUPERNOVA w0-waCDM DARK ENERGY COSMOLOGY (Pantheon+ / JLA Style)",
        supernova_wcdm_model,
        make_supernova_jax_fn(sn_data),
        sn_data,
        sn_prior_m,
        sn_prior_H,
        rwm_scale=0.015,
        mala_tau=1e-5,
        hmc_eps=0.004,
    )

    # 3. Gravitational Microlensing (5D)
    ul_data = generate_microlensing_data(seed=42)
    ul_prior_m = np.array([50.0, 22.0, 0.25, 1.40, 0.60])
    ul_prior_H = np.diag([0.5, 0.1, 4.0, 1.0, 1.0])
    res_ul = run_single_astrophysics_benchmark(
        "3. GRAVITATIONAL MICROLENSING LIGHT-CURVE INVERSION (OGLE / Roman PSPL Style)",
        microlensing_model,
        make_microlensing_jax_fn(ul_data),
        ul_data,
        ul_prior_m,
        ul_prior_H,
        rwm_scale=0.005,
        mala_tau=1e-6,
        hmc_eps=0.0004,
    )

    # Generate dedicated 4-panel Runtime & Efficiency Comparison Figure (fig6)
    plot_astrophysics_runtime_comparison([res_rv, res_sn, res_ul])


if __name__ == "__main__":
    main()
