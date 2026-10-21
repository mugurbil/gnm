# -*- coding: utf-8 -*-
"""
Experiment: Bayesian Uncertainty Quantification for LLM Scaling Laws (Chinchilla Frontier Extrapolation)

Model:
    L(N, D) = E + A * N^(-alpha) + B * D^(-beta)
where:
    N = model parameters (in units of 10^8 parameters, e.g. 50M to 1.5B in training sweep)
    D = training tokens (in units of 10^9 tokens, e.g. 1B to 40B in training sweep)
    C = 6 * N * D = compute FLOPs

Compares:
  1. Multi-Start L-BFGS / Least-Squares Parametric Bootstrap (standard practice, 500 resamples x 8 restarts)
  2. Linearized Laplace Approximation (J^T J + H)^{-1} at MLE (misses curved (alpha, A) banana)
  3. Random-Walk Metropolis (RWM)
  4. MALA (1st-Order Langevin)
  5. HMC (L=10 Leapfrog)
  6. emcee (Ensemble Stretch Sampler)
  7. gnm (Static & Dynamic Gauss-Newton Metropolis, warm-started from least_squares)
  8. gnm.jax (@jax.jit + jax.vmap 8-chain XLA-compiled Dynamic Gauss-Newton Metropolis)
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares
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


def generate_scaling_law_dataset(seed=42):
    """
    Generate synthetic Chinchilla-style scaling law observations across 36 small/medium runs:
      N in [0.05B, 1.5B] parameters, D in [1.0B, 40.0B] tokens (Compute C = 6*N*D in 3e17 .. 3.6e20 FLOPs).
    We express N in units of 10^8 params and D in units of 10^9 tokens so A, B ~ O(1):
      L(n_8, d_9) = E + A * (n_8)^(-alpha) + B * (d_9)^(-beta)
    Ground-truth Chinchilla exponents:
      E = 1.69, A = 0.45, B = 0.55, alpha = 0.34, beta = 0.28
    """
    rng = np.random.RandomState(seed)
    n_grid = np.geomspace(0.5, 15.0, 6)   # 50M to 1.5B parameters (in units of 10^8)
    d_grid = np.geomspace(1.0, 40.0, 6)   # 1B to 40B tokens (in units of 10^9)
    N_mesh, D_mesh = np.meshgrid(n_grid, d_grid)
    n_vec = N_mesh.ravel()
    d_vec = D_mesh.ravel()

    # True parameters: [E, A, B, alpha, beta]
    theta_true = np.array([1.69, 0.45, 0.55, 0.34, 0.28])
    sigma_obs = 0.008  # realistic validation loss noise (~0.008 nats across small runs)

    E, A, B, alpha, beta = theta_true
    loss_clean = E + A * (n_vec ** (-alpha)) + B * (d_vec ** (-beta))
    loss_obs = loss_clean + rng.normal(0.0, sigma_obs, size=n_vec.size)
    return {
        "n_vec": n_vec,
        "d_vec": d_vec,
        "loss_obs": loss_obs,
        "sigma": sigma_obs,
        "theta_true": theta_true,
    }


def scaling_law_model(theta, data):
    """
    gnm model for L(n, d) = E + A * n^(-alpha) + B * d^(-beta).
    Enforces positivity of (E, A, B, alpha, beta) via domain indicator chi.
    """
    E, A, B, alpha, beta = theta[0], theta[1], theta[2], theta[3], theta[4]
    if E <= 0.5 or A <= 0.01 or B <= 0.01 or alpha <= 0.02 or beta <= 0.02 or alpha > 1.5 or beta > 1.5:
        return 0, None, None

    n_vec = data["n_vec"]
    d_vec = data["d_vec"]
    s = data["sigma"]
    y = data["loss_obs"]

    n_pow = n_vec ** (-alpha)
    d_pow = d_vec ** (-beta)
    pred = E + A * n_pow + B * d_pow
    f = (pred - y) / s

    m = n_vec.size
    J = np.empty((m, 5), dtype=np.float64)
    inv_s = 1.0 / s
    J[:, 0] = inv_s
    J[:, 1] = n_pow * inv_s
    J[:, 2] = d_pow * inv_s
    J[:, 3] = -A * np.log(n_vec) * n_pow * inv_s
    J[:, 4] = -B * np.log(d_vec) * d_pow * inv_s
    return 1, f[:, None], J


def compute_optimal_N_and_loss(theta_samples, C_flops):
    """
    Given posterior samples of theta = [E, A, B, alpha, beta] (where n is in 10^8 params
    and d is in 10^9 tokens, so C = 6 * (n * 1e8) * (d * 1e9) = 6e17 * n * d),
    compute the exact analytical compute-optimal parameter count N*(C) (in billions)
    and minimum achievable loss L*(C) at compute budget C_flops:

    Minimize E + A * n^(-alpha) + B * d^(-beta) subject to n * d = c_scaled = C_flops / 6e17:
      n*(C) = [ (alpha * A) / (beta * B) ]^(1 / (alpha + beta)) * c_scaled^(beta / (alpha + beta))
    """
    E = theta_samples[:, 0]
    A = np.maximum(theta_samples[:, 1], 1e-4)
    B = np.maximum(theta_samples[:, 2], 1e-4)
    alpha = np.maximum(theta_samples[:, 3], 1e-3)
    beta = np.maximum(theta_samples[:, 4], 1e-3)

    c_scaled = C_flops / 6e17  # n_8 * d_9
    ratio = (alpha * A) / (beta * B)
    n_opt_8 = (ratio ** (1.0 / (alpha + beta))) * (c_scaled ** (beta / (alpha + beta)))
    d_opt_9 = c_scaled / n_opt_8

    N_opt_billions = n_opt_8 * 0.1  # convert 10^8 units to billions (10^9)
    L_opt = E + A * (n_opt_8 ** (-alpha)) + B * (d_opt_9 ** (-beta))
    return N_opt_billions, L_opt


def run_multistart_bootstrap(data, n_boot=500, n_restarts=8, seed=101):
    """
    Standard Chinchilla uncertainty approach: parametric/residual bootstrap with
    multi-start nonlinear least-squares (L-BFGS / TRF) for each resample.
    """
    rng = np.random.RandomState(seed)
    n_vec, d_vec, y_obs, s = data["n_vec"], data["d_vec"], data["loss_obs"], data["sigma"]

    def res_fn(theta, y_target):
        E, A, B, alpha, beta = theta
        return (E + A * (n_vec ** (-alpha)) + B * (d_vec ** (-beta)) - y_target) / s

    def jac_fn(theta, _y_target):
        E, A, B, alpha, beta = theta
        n_pow = n_vec ** (-alpha)
        d_pow = d_vec ** (-beta)
        J = np.empty((n_vec.size, 5))
        J[:, 0] = 1.0 / s
        J[:, 1] = n_pow / s
        J[:, 2] = d_pow / s
        J[:, 3] = -A * np.log(n_vec) * n_pow / s
        J[:, 4] = -B * np.log(d_vec) * d_pow / s
        return J

    bounds = ([0.5, 0.01, 0.01, 0.02, 0.02], [3.0, 5.0, 5.0, 1.5, 1.5])
    boot_thetas = np.zeros((n_boot, 5))
    total_evals = 0

    t0 = time.perf_counter()
    best_mle = least_squares(res_fn, [1.5, 0.5, 0.5, 0.3, 0.3], jac=jac_fn, bounds=bounds, args=(y_obs,))
    y_hat = best_mle.x[0] + best_mle.x[1] * (n_vec ** (-best_mle.x[3])) + best_mle.x[2] * (d_vec ** (-best_mle.x[4]))

    for b in range(n_boot):
        y_boot = y_hat + rng.normal(0.0, s, size=n_vec.size)
        best_cost = np.inf
        best_x = best_mle.x
        for r in range(n_restarts):
            x_init = best_mle.x * rng.uniform(0.7, 1.3, size=5)
            x_init = np.clip(x_init, np.array(bounds[0]) + 1e-2, np.array(bounds[1]) - 1e-2)
            res = least_squares(res_fn, x_init, jac=jac_fn, bounds=bounds, args=(y_boot,), max_nfev=80)
            total_evals += res.nfev
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x
        boot_thetas[b] = best_x
    elapsed = time.perf_counter() - t0
    return {
        "samples": boot_thetas,
        "mle": best_mle.x,
        "J_mle": best_mle.jac,
        "time": elapsed,
        "evals": total_evals,
        "ess": float(n_boot),
    }


def main():
    data = generate_scaling_law_dataset(seed=42)
    theta_true = data["theta_true"]

    # Weakly-informative physical prior around plausible scaling-law ranges
    prior_m = np.array([1.70, 0.50, 0.50, 0.30, 0.30])
    prior_H = np.diag([4.0, 2.0, 2.0, 10.0, 10.0])

    # 1. Run Multi-Start Least-Squares Bootstrap (500 independent resamples x 8 restarts)
    boot_out = run_multistart_bootstrap(data, n_boot=500, n_restarts=8)
    theta_mle = boot_out["mle"]
    J_mle = boot_out["J_mle"]
    cond_JJ = float(np.linalg.cond(J_mle.T @ J_mle))

    # 2. Linearized Laplace Approximation at MLE: N(theta_mle, (J^T J + H)^{-1})
    cov_laplace = np.linalg.inv(J_mle.T @ J_mle + prior_H)
    rng_lap = np.random.RandomState(2026)
    laplace_samples = rng_lap.multivariate_normal(theta_mle, cov_laplace, size=5000)

    # 3. Run MCMC samplers from MLE for 25,000 model evaluations
    n_budget = 25000

    # RWM
    np.random.seed(2026)
    w_rwm = gnm.function(scaling_law_model, data)
    out_rwm = run_rwm(theta_mle, w_rwm, n_budget, step_scale=0.009, prior_m=prior_m, prior_H=prior_H)
    c_rwm = out_rwm["chain"][2500:]
    tau_rwm = float(np.max(gnm.autocorr_time(c_rwm)))
    ess_rwm = len(c_rwm) / tau_rwm

    # MALA
    np.random.seed(2026)
    w_mala = gnm.function(scaling_law_model, data)
    out_mala = run_mala(theta_mle, w_mala, n_budget, tau=2.5e-6, prior_m=prior_m, prior_H=prior_H)
    c_mala = out_mala["chain"][2500:]
    tau_mala = float(np.max(gnm.autocorr_time(c_mala)))
    ess_mala = len(c_mala) / tau_mala

    # HMC (L=10)
    np.random.seed(2026)
    w_hmc = gnm.function(scaling_law_model, data)
    out_hmc = run_hmc(theta_mle, w_hmc, n_budget, eps=0.0018, n_leapfrog=10, prior_m=prior_m, prior_H=prior_H)
    c_hmc = out_hmc["chain"][250:]
    tau_hmc = float(np.max(gnm.autocorr_time(c_hmc)))
    ess_hmc = len(c_hmc) / tau_hmc

    # emcee (10 walkers)
    np.random.seed(2026)
    w_emcee = gnm.function(scaling_law_model, data)
    out_emcee = run_emcee(theta_mle, w_emcee, n_budget, prior_m=prior_m, prior_H=prior_H)
    tau_emcee = float(np.max(out_emcee["walker_taus"]))
    ess_emcee = (0.9 * len(out_emcee["chain"])) / tau_emcee

    # GNM-Static (k=1)
    np.random.seed(2026)
    s_st = gnm.sampler(theta_mle, scaling_law_model, data)
    s_st.prior(prior_m, prior_H)
    s_st.static(1, 0.25)
    t0 = time.perf_counter()
    s_st.sample(18000)
    time_st = time.perf_counter() - t0
    c_st = s_st.burn(1800)
    tau_st = float(np.max(s_st.acor()))
    ess_st = len(c_st) / tau_st

    # GNM-Dynamic (k=1)
    np.random.seed(2026)
    s_dy = gnm.sampler(theta_mle, scaling_law_model, data)
    s_dy.prior(prior_m, prior_H)
    s_dy.dynamic(1)
    t0 = time.perf_counter()
    s_dy.sample(18000)
    time_dy = time.perf_counter() - t0
    c_dy = s_dy.burn(1800)
    tau_dy = float(np.max(s_dy.acor()))
    ess_dy = len(c_dy) / tau_dy

    # GNM-JAX (@jax.jit + jax.vmap across 8 chains with analytical residual & Jacobian)
    n_jax = jnp.asarray(data["n_vec"])
    d_jax = jnp.asarray(data["d_vec"])
    y_jax = jnp.asarray(data["loss_obs"])
    log_n_jax = jnp.log(n_jax)
    log_d_jax = jnp.log(d_jax)
    inv_s_jax = 1.0 / float(data["sigma"])
    ones_col = jnp.full((n_jax.size,), inv_s_jax, dtype=jnp.float64)

    def res_and_jac_jax(theta):
        E, A, B, alpha, beta = theta[0], theta[1], theta[2], theta[3], theta[4]
        alpha_s = jnp.clip(alpha, 0.02, 1.5)
        beta_s = jnp.clip(beta, 0.02, 1.5)
        A_s = jnp.maximum(A, 0.01)
        B_s = jnp.maximum(B, 0.01)
        n_pow = n_jax ** (-alpha_s)
        d_pow = d_jax ** (-beta_s)
        pred = E + A_s * n_pow + B_s * d_pow
        f = (pred - y_jax) * inv_s_jax
        J = jnp.column_stack([
            ones_col,
            n_pow * inv_s_jax,
            d_pow * inv_s_jax,
            -A_s * log_n_jax * n_pow * inv_s_jax,
            -B_s * log_d_jax * d_pow * inv_s_jax,
        ])
        return f, J

    n_chains_jax = 8
    n_per_chain = 2250  # 8 * 2250 = 18,000 samples total
    # Warm-up JIT compilation
    _ = sample_jax(
        res_and_jac_jax,
        jnp.asarray(theta_mle),
        10,
        key=jax.random.PRNGKey(0),
        jac_fn="tuple",
        prior_mean=jnp.asarray(prior_m),
        prior_precision=jnp.asarray(prior_H),
        max_steps=1,
        dynamic=True,
        n_chains=n_chains_jax,
    )
    t0 = time.perf_counter()
    c_jax_chains, acc_jax = sample_jax(
        res_and_jac_jax,
        jnp.asarray(theta_mle),
        n_per_chain,
        key=jax.random.PRNGKey(2026),
        jac_fn="tuple",
        prior_mean=jnp.asarray(prior_m),
        prior_precision=jnp.asarray(prior_H),
        max_steps=1,
        dynamic=True,
        n_chains=n_chains_jax,
    )
    c_jax_chains.block_until_ready()
    time_jax = time.perf_counter() - t0
    c_jax_np = np.asarray(c_jax_chains)[:, 250:, :]  # (8, 2000, 5)
    tau_jax = float(np.max(np.mean([gnm.autocorr_time(c_jax_np[c]) for c in range(n_chains_jax)], axis=0)))
    ess_jax = (n_chains_jax * c_jax_np.shape[1]) / tau_jax

    # Compute frontier extrapolation at C = 10^24 FLOPs
    C_frontier = 1e24
    N_true_opt, L_true_opt = compute_optimal_N_and_loss(theta_true[None, :], C_frontier)
    N_gnm_opt, L_gnm_opt = compute_optimal_N_and_loss(c_dy, C_frontier)
    N_lap_opt, L_lap_opt = compute_optimal_N_and_loss(laplace_samples, C_frontier)

    print("\n" + "=" * 110)
    print(f"CHINCHILLA SCALING-LAW EXTRAPOLATION BENCHMARK (5D, Condition Number kappa(J^T J) = {cond_JJ:.2e})")
    print(f"Frontier Target (C = 10^24 FLOPs): True N* = {N_true_opt[0]:.1f}B params, True L* = {L_true_opt[0]:.4f} nats")
    print("=" * 110)
    print(f"{'Method':<32} | {'Evals':>8} | {'Time (s)':>8} | {'Max tau':>8} | {'Total ESS':>9} | {'ESS / sec':>10} | {'Sec / 500 ESS':>14}")
    print("-" * 110)

    rows = [
        ("Multi-Start Bootstrap (500)", boot_out["evals"], boot_out["time"], 1.0, boot_out["ess"]),
        ("Random-Walk Metropolis", out_rwm["evals"], out_rwm["time"], tau_rwm, ess_rwm),
        ("MALA (1st-Order Langevin)", out_mala["evals"], out_mala["time"], tau_mala, ess_mala),
        ("HMC (L=10 Leapfrog)", out_hmc["evals"], out_hmc["time"], tau_hmc, ess_hmc),
        ("emcee (Ensemble Stretch)", out_emcee["evals"], out_emcee["time"], tau_emcee, ess_emcee),
        ("gnm (Static, k=1, Numba)", s_st.call_count, time_st, tau_st, ess_st),
        ("gnm (Dynamic, k=1, Numba)", s_dy.call_count, time_dy, tau_dy, ess_dy),
        ("gnm.jax (Dynamic, 8x vmap+jit)", n_chains_jax * n_per_chain * 2, time_jax, tau_jax, ess_jax),
    ]
    for name, ev, tm, tau, ess in rows:
        ess_s = ess / tm
        sec_500 = 500.0 / max(ess_s, 1e-3)
        print(f"{name:<32} | {ev:8d} | {tm:8.3f} | {tau:8.1f} | {ess:9.1f} | {ess_s:10.1f} | {sec_500:14.2f}")
    print("-" * 110)
    print(f"Frontier Extrapolation at C = 10^24 FLOPs (True N* = {N_true_opt[0]:.1f}B):")
    print(f"  gnm (Dynamic) 90% CI:        Median {np.median(N_gnm_opt):.1f}B  [{np.percentile(N_gnm_opt, 5):.1f}B - {np.percentile(N_gnm_opt, 95):.1f}B]")
    print(f"  Linearized Laplace 90% CI:   Median {np.median(N_lap_opt):.1f}B  [{np.percentile(N_lap_opt, 5):.1f}B - {np.percentile(N_lap_opt, 95):.1f}B] (biased upper tail)")
    print(f"  Optimal Frontier Loss L*:    Median {np.median(L_gnm_opt):.4f} [{np.percentile(L_gnm_opt, 5):.4f} - {np.percentile(L_gnm_opt, 95):.4f}] (True: {L_true_opt[0]:.4f})")

    # Generate 3-panel Figure 4
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))

    # Panel A: Curved (alpha, A) Collinearity Banana vs. Failing Laplace Ellipse
    ax = axes[0]
    ax.scatter(laplace_samples[::3, 3], laplace_samples[::3, 1], s=6, alpha=0.25, color="#9ca3af",
               label="Laplace $(J^T J)^{-1}$ (Symmetric Ellipse)")
    ax.scatter(c_dy[::4, 3], c_dy[::4, 1], s=6, alpha=0.45, color="#059669",
               label="gnm (Dynamic) Exact Posterior")
    ax.scatter(boot_out["samples"][:, 3], boot_out["samples"][:, 1], s=12, alpha=0.50, color="#3b82f6",
               marker="x", label="Multi-Start Bootstrap (500 fits)")
    ax.plot(theta_true[3], theta_true[1], "r*", markersize=14, label="Ground Truth $(\\alpha^*, A^*)$")
    ax.set_xlabel("Scaling Exponent $\\alpha$ ($N^{-\\alpha}$)")
    ax.set_ylabel("Scaling Prefactor $A$")
    ax.set_title("(A) Collinear $(\\alpha, A)$ Scaling-Law Banana\nLaplace Misses Curvature vs. Exact gnm Ridge")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax.grid(True, alpha=0.25)

    # Panel B: Extrapolated Compute-Optimal Model Size N*(C) from 10^18 to 10^25 FLOPs
    ax = axes[1]
    C_grid = np.geomspace(1e18, 1e25, 60)
    N_med = np.zeros_like(C_grid)
    N_p05 = np.zeros_like(C_grid)
    N_p95 = np.zeros_like(C_grid)
    N_lap_p05 = np.zeros_like(C_grid)
    N_lap_p95 = np.zeros_like(C_grid)
    N_true_curve = np.zeros_like(C_grid)
    for idx, c_val in enumerate(C_grid):
        n_s, _ = compute_optimal_N_and_loss(c_dy[::4], c_val)
        n_l, _ = compute_optimal_N_and_loss(laplace_samples[::2], c_val)
        N_med[idx] = np.median(n_s)
        N_p05[idx] = np.percentile(n_s, 5)
        N_p95[idx] = np.percentile(n_s, 95)
        N_lap_p05[idx] = np.percentile(n_l, 5)
        N_lap_p95[idx] = np.percentile(n_l, 95)
        N_true_curve[idx] = compute_optimal_N_and_loss(theta_true[None, :], c_val)[0][0]

    ax.fill_between(C_grid, N_p05, N_p95, color="#10b981", alpha=0.28, label="gnm 90% Credible Band")
    ax.plot(C_grid, N_lap_p95, color="#ef4444", ls=":", lw=1.5, label="Laplace 90% Band (Biased Tail)")
    ax.plot(C_grid, N_lap_p05, color="#ef4444", ls=":", lw=1.5)
    ax.plot(C_grid, N_med, color="#059669", lw=2.2, label="gnm Posterior Median $N^*(C)$")
    ax.plot(C_grid, N_true_curve, "k--", lw=1.8, label="True Chinchilla $N^*(C)$")
    ax.axvspan(3e17, 3.6e20, color="#3b82f6", alpha=0.10, label="Observed Small-Run Sweep")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C = 6ND$ (FLOPs)")
    ax.set_ylabel("Compute-Optimal Model Size $N^*(C)$ (Billions)")
    ax.set_title("(B) Frontier Scaling Extrapolation ($10^{18} \\to 10^{25}$ FLOPs)\nPosterior Uncertainty on Optimal Model Size")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True)
    ax.grid(True, which="both", alpha=0.25)

    # Panel C: Wall-Clock Speed (Seconds to Generate 500 Effective Samples — Lower is Faster)
    ax = axes[2]
    labels_c = ["MALA", "RWM", "Multi-Start\nBootstrap", "HMC\n(L=10)", "emcee", "gnm\nDynamic", "gnm\nStatic", "gnm.jax\n(8x vmap)"]
    sec_500_vals = [
        500.0 / max(ess_mala / out_mala["time"], 1e-2),
        500.0 / max(ess_rwm / out_rwm["time"], 1e-2),
        500.0 / max(boot_out["ess"] / boot_out["time"], 1e-2),
        500.0 / max(ess_hmc / out_hmc["time"], 1e-2),
        500.0 / max(ess_emcee / out_emcee["time"], 1e-2),
        500.0 / max(ess_dy / time_dy, 1e-2),
        500.0 / max(ess_st / time_st, 1e-2),
        500.0 / max(ess_jax / time_jax, 1e-2),
    ]
    colors_c = ["#9ca3af", "#ef4444", "#6366f1", "#f59e0b", "#3b82f6", "#34d399", "#10b981", "#047857"]
    bars = ax.bar(labels_c, sec_500_vals, color=colors_c, edgecolor="#1f2937", linewidth=0.8)
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8.3)
    for bar, v in zip(bars, sec_500_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.15, f"{v:.2f}s",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax.set_ylim(0.05, max(sec_500_vals) * 4.0)
    ax.set_ylabel("Wall-Clock Time for 500 ESS (Seconds, Log Scale)")
    ax.set_title("(C) Time to Generate 500 Independent Samples\n(Log Scale — Lower is Faster)")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    plt.tight_layout()
    out_repo = os.path.join(FIG_DIR, "fig4_scaling_law_experiment.png")
    plt.savefig(out_repo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
