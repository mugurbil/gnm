# -*- coding: utf-8 -*-
"""
Experimental Benchmark Suite for `gnm` (Dynamic Gauss-Newton Metropolis)
========================================================================
Tests:
  Part 1 (Verification / "Works as Intended"):
    1A. Exact posterior density recovery on 1D Bimodal Double-Well vs. quadrature
    1B. Back-off ablation (k_max = 0, 1, 2, Static vs Dynamic) on Curved Rosenbrock (b=100)
        against exact analytical moments
    1C. Strict Affine-Invariance test under extreme coordinate anisotropy (scale 1 to 1e-6)

  Part 2 (Comparison vs. State-of-the-Art MCMC under Equal Function-Evaluation Budget):
    Compares RWM, MALA, HMC, `emcee` (Ensemble Stretch Move), GNM-Static, and GNM-Dynamic on:
    2A. Stiff Curved Rosenbrock Valley (2D, b=100)
    2B. Anisotropic Nonlinear Least-Squares System (6D, condition number 10^6)
    2C. Ill-Posed 4D Multi-Exponential Time-Series Inversion
"""

import os
import sys
import time
import json
import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gnm
import emcee

la = np.linalg


# ============================================================================
# Baseline State-of-the-Art Samplers (RWM, MALA, HMC, emcee)
# ============================================================================

def eval_log_prob_and_grad(x, fn_wrap, prior_m=None, prior_H=None):
    """Evaluate log pi(x) and its gradient grad log pi(x) = -J(x)^T f(x) - H(x - m)."""
    chi, f, J = fn_wrap(x)
    if not chi or not np.all(np.isfinite(f)):
        return -np.inf, np.zeros_like(x)
    f_vec = f.reshape(-1)
    log_p = -0.5 * float(np.dot(f_vec, f_vec))
    grad = -np.dot(J.T, f_vec)
    if prior_m is not None and prior_H is not None:
        diff = x.reshape(-1) - prior_m.reshape(-1)
        H_diff = np.dot(prior_H, diff)
        log_p -= 0.5 * float(np.dot(diff, H_diff))
        grad -= H_diff
    return log_p, grad


def run_rwm(x0, fn_wrap, n_evals, step_scale=None, prior_m=None, prior_H=None):
    """Random-Walk Metropolis with isotropic Gaussian proposals."""
    x = np.array(x0, dtype=float).reshape(-1)
    n = x.size
    if step_scale is None:
        step_scale = 2.38 / np.sqrt(n)
    chain = np.zeros((n_evals, n))
    log_p, _ = eval_log_prob_and_grad(x, fn_wrap, prior_m, prior_H)
    accepted = 0
    t0 = time.perf_counter()
    for i in range(n_evals):
        z = x + step_scale * np.random.randn(n)
        log_pz, _ = eval_log_prob_and_grad(z, fn_wrap, prior_m, prior_H)
        if np.isfinite(log_pz) and np.log(np.random.rand()) < (log_pz - log_p):
            x = z
            log_p = log_pz
            accepted += 1
        chain[i] = x
    elapsed = time.perf_counter() - t0
    return {
        "chain": chain,
        "accept_rate": accepted / n_evals,
        "evals": fn_wrap.count,
        "time": elapsed,
    }


def run_mala(x0, fn_wrap, n_evals, tau=0.01, prior_m=None, prior_H=None):
    """Metropolis-Adjusted Langevin Algorithm (1st-order gradient MCMC)."""
    x = np.array(x0, dtype=float).reshape(-1)
    n = x.size
    chain = np.zeros((n_evals, n))
    log_p, grad_x = eval_log_prob_and_grad(x, fn_wrap, prior_m, prior_H)
    accepted = 0
    sqrt_tau = np.sqrt(tau)
    t0 = time.perf_counter()
    for i in range(n_evals):
        mean_x = x + 0.5 * tau * grad_x
        z = mean_x + sqrt_tau * np.random.randn(n)
        log_pz, grad_z = eval_log_prob_and_grad(z, fn_wrap, prior_m, prior_H)
        if np.isfinite(log_pz):
            mean_z = z + 0.5 * tau * grad_z
            log_q_z_given_x = -0.5 * np.sum((z - mean_x) ** 2) / tau
            log_q_x_given_z = -0.5 * np.sum((x - mean_z) ** 2) / tau
            log_alpha = (log_pz + log_q_x_given_z) - (log_p + log_q_z_given_x)
            if np.log(np.random.rand()) < log_alpha:
                x, log_p, grad_x = z, log_pz, grad_z
                accepted += 1
        chain[i] = x
    elapsed = time.perf_counter() - t0
    return {
        "chain": chain,
        "accept_rate": accepted / n_evals,
        "evals": fn_wrap.count,
        "time": elapsed,
    }


def run_hmc(x0, fn_wrap, n_evals, eps=0.03, n_leapfrog=10, prior_m=None, prior_H=None):
    """Hamiltonian Monte Carlo (HMC) with L leapfrog steps per proposal."""
    x = np.array(x0, dtype=float).reshape(-1)
    n = x.size
    n_steps = max(1, n_evals // n_leapfrog)
    chain = np.zeros((n_steps, n))
    log_p, grad_x = eval_log_prob_and_grad(x, fn_wrap, prior_m, prior_H)
    accepted = 0
    t0 = time.perf_counter()
    for i in range(n_steps):
        p = np.random.randn(n)
        current_K = 0.5 * np.sum(p ** 2)
        z = x.copy()
        grad_z = grad_x.copy()
        p_z = p + 0.5 * eps * grad_z
        valid = True
        for l in range(n_leapfrog):
            z = z + eps * p_z
            log_pz, grad_z = eval_log_prob_and_grad(z, fn_wrap, prior_m, prior_H)
            if not np.isfinite(log_pz):
                valid = False
                break
            if l < n_leapfrog - 1:
                p_z = p_z + eps * grad_z
        if valid:
            p_z = p_z + 0.5 * eps * grad_z
            proposed_K = 0.5 * np.sum(p_z ** 2)
            log_alpha = (log_pz - proposed_K) - (log_p - current_K)
            if np.log(np.random.rand()) < log_alpha:
                x, log_p, grad_x = z, log_pz, grad_z
                accepted += 1
        chain[i] = x
    elapsed = time.perf_counter() - t0
    return {
        "chain": chain,
        "accept_rate": accepted / n_steps,
        "evals": fn_wrap.count,
        "time": elapsed,
    }


def run_emcee(x0, fn_wrap, n_evals, prior_m=None, prior_H=None):
    """Goodman & Weare (2010) Affine-Invariant Ensemble Sampler via `emcee`."""
    x0 = np.array(x0, dtype=float).reshape(-1)
    n = x0.size
    n_walkers = max(2 * n, 8)
    n_steps = max(1, n_evals // n_walkers)

    def log_prob_fn(theta):
        lp, _ = eval_log_prob_and_grad(theta, fn_wrap, prior_m, prior_H)
        return lp

    p0 = x0[None, :] + 1e-2 * np.random.randn(n_walkers, n)
    sampler = emcee.EnsembleSampler(n_walkers, n, log_prob_fn)
    t0 = time.perf_counter()
    sampler.run_mcmc(p0, n_steps, progress=False)
    elapsed = time.perf_counter() - t0

    # Compute mean autocorrelation time across walkers
    chain_3d = sampler.get_chain()  # (n_steps, n_walkers, n)
    walker_taus = np.mean([gnm.autocorr_time(chain_3d[:, w, :]) for w in range(n_walkers)], axis=0)
    flat_chain = sampler.get_chain(flat=True)
    return {
        "chain": flat_chain,
        "chain_3d": chain_3d,
        "walker_taus": walker_taus,
        "accept_rate": float(np.mean(sampler.acceptance_fraction)),
        "evals": fn_wrap.count,
        "time": elapsed,
    }


# ============================================================================
# Part 1: Verification Experiments ("Does gnm work as intended?")
# ============================================================================

def exp_1a_double_well_accuracy():
    """1D Bimodal Double-Well: compare GNM density against exact numerical quadrature."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 1A: 1D Bimodal Double-Well Exact Posterior Recovery (y=2, s=1)")
    print("=" * 78)
    y_obs = 2.0
    sigma = 1.0

    def well_model(x, args):
        return 1, [(x[0] ** 2 - args['y']) / args['s']], [[(2.0 * x[0]) / args['s']]]

    args = {'y': y_obs, 's': sigma}

    # Exact numerical quadrature ground truth
    ref = gnm.sampler([1.0], well_model, args)
    ref.prior([0.0], [[1.0]])
    Z_norm, _ = integrate.quad(lambda u: ref.posterior([u]), -6.0, 6.0, limit=200)
    exact_x2, _ = integrate.quad(lambda u: (u ** 2) * ref.posterior([u]) / Z_norm, -6.0, 6.0, limit=200)

    n_bins = 80
    grid_edges = np.linspace(-3.0, 3.0, n_bins + 1)
    grid_centers = 0.5 * (grid_edges[:-1] + grid_edges[1:])
    dx = grid_edges[1] - grid_edges[0]
    p_exact = np.array([ref.posterior([u]) / Z_norm for u in grid_centers])

    configs = [
        ("GNM (k=0, no back-off)", "static", (0, 0.25)),
        ("GNM-Static (k=1, q=0.25)", "static", (1, 0.25)),
        ("GNM-Static (k=2, q=0.25)", "static", (2, 0.25)),
        ("GNM-Dynamic (k=1)", "dynamic", (1,)),
        ("GNM-Dynamic (k=2)", "dynamic", (2,)),
    ]

    results = []
    print(f"{'Method':<26} | {'Accept':>7} | {'tau_int':>7} | {'ESS':>8} | {'ESS/eval':>8} | {'TV Dist':>7} | {'E[x^2] (Exact=' + f'{exact_x2:.3f})':>20}")
    print("-" * 102)
    for name, mode, params in configs:
        np.random.seed(42)
        s = gnm.sampler([0.5], well_model, args)
        s.prior([0.0], [[1.0]])
        if mode == "static":
            s.static(*params)
        else:
            s.dynamic(*params)
        s.sample(42000)
        s.burn(2000)
        chain = s.chain[:, 0]
        tau = float(s.acor()[0])
        ess = len(chain) / tau
        ess_per_eval = ess / s.call_count
        _, p_hat, _ = s.error_bars(n_bins, [-3.0], [3.0])
        tv_dist = 0.5 * float(np.sum(np.abs(p_hat - p_exact)) * dx)
        est_x2 = float(np.mean(chain ** 2))
        print(f"{name:<26} | {s.accept_rate:7.3f} | {tau:7.2f} | {ess:8.0f} | {ess_per_eval:8.4f} | {tv_dist:7.4f} | {est_x2:20.4f}")
        results.append({
            "name": name,
            "accept_rate": s.accept_rate,
            "tau": tau,
            "ess": ess,
            "ess_per_eval": ess_per_eval,
            "tv_dist": tv_dist,
            "est_x2": est_x2,
            "exact_x2": exact_x2,
        })
    return results


def exp_1b_rosenbrock_backoff_ablation():
    """
    2D Curved Rosenbrock Banana: f(x) = [a - x0, sqrt(b)*(x1 - x0^2)]^T with a=1, b=100.
    Exact analytical posterior moments (no prior needed since J is 2x2 full rank):
      x0 ~ N(1, 1)         => E[x0] = 1.0, Std(x0) = 1.0
      x1 | x0 ~ N(x0^2, 1/b) => E[x1] = a^2 + 1 = 2.0, Var(x1) = 4a^2 + 2 + 1/b = 6.01 (Std = 2.4515)
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 1B: Back-Off Ablation on Stiff Rosenbrock Banana (b=100)")
    print("Exact Analytical Ground Truth: E[x0]=1.000, Std(x0)=1.000, E[x1]=2.000, Std(x1)=2.452")
    print("=" * 78)

    def rosen_vec(x, args):
        a, sb = args['a'], args['sb']
        f = [a - x[0], sb * (x[1] - x[0] ** 2)]
        J = [[-1.0, 0.0], [-2.0 * sb * x[0], sb]]
        return 1, f, J

    args = {'a': 1.0, 'sb': 10.0}  # b = sb^2 = 100

    configs = [
        ("GNM (k=0, no back-off)", "static", (0, 0.25)),
        ("GNM-Static (k=1, q=0.25)", "static", (1, 0.25)),
        ("GNM-Static (k=2, q=0.25)", "static", (2, 0.25)),
        ("GNM-Static (k=3, q=0.35)", "static", (3, 0.35)),
        ("GNM-Dynamic (k=1)", "dynamic", (1,)),
        ("GNM-Dynamic (k=2)", "dynamic", (2,)),
        ("GNM-Dynamic (k=3)", "dynamic", (3,)),
    ]

    print(f"{'Configuration':<26} | {'Accept':>7} | {'Max tau':>8} | {'ESS':>8} | {'ESS/eval':>8} | {'E[x0]':>6} | {'Std(x0)':>7} | {'E[x1]':>6} | {'Std(x1)':>7}")
    print("-" * 106)
    results = []
    for name, mode, params in configs:
        np.random.seed(123)
        s = gnm.sampler([1.0, 1.0], rosen_vec, args)
        if mode == "static":
            s.static(*params)
        else:
            s.dynamic(*params)
        s.sample(25000)
        s.burn(2000)
        taus = s.acor()
        max_tau = float(np.max(taus))
        ess = len(s.chain) / max_tau
        ess_eval = ess / s.call_count
        m0, sd0 = float(np.mean(s.chain[:, 0])), float(np.std(s.chain[:, 0]))
        m1, sd1 = float(np.mean(s.chain[:, 1])), float(np.std(s.chain[:, 1]))
        print(f"{name:<26} | {s.accept_rate:7.3f} | {max_tau:8.2f} | {ess:8.0f} | {ess_eval:8.4f} | {m0:6.3f} | {sd0:7.3f} | {m1:6.3f} | {sd1:7.3f}")
        results.append({
            "name": name,
            "accept_rate": s.accept_rate,
            "max_tau": max_tau,
            "ess": ess,
            "ess_per_eval": ess_eval,
            "mean_x0": m0,
            "std_x0": sd0,
            "mean_x1": m1,
            "std_x1": sd1,
        })
    return results


def exp_1c_affine_invariance_stress():
    """
    Verify strict affine invariance of GNM vs non-affine samplers when coordinates
    are rescaled by 10^0, 10^2, 10^4, 10^6 (condition number kappa(J^T J) up to 10^12).
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 1C: Affine-Invariance Stress Test (Coordinate Scale Factor S)")
    print("=" * 78)
    scales = [1.0, 1e2, 1e4, 1e6]
    print(f"{'Scale S (kappa=S^2)':<22} | {'GNM-Dynamic Accept':>18} | {'GNM-Dynamic tau':>15} | {'MALA Accept':>12} | {'MALA tau':>10} | {'HMC Accept':>11} | {'HMC tau':>10}")
    print("-" * 112)
    results = []
    for S in scales:
        def scaled_rosen(x, _args):
            # Transform x0_phys = x[0], x1_phys = S * x[1]
            x0 = x[0]
            x1 = S * x[1]
            sb = 5.0
            f = [1.0 - x0, sb * (x1 - x0 ** 2)]
            J = [[-1.0, 0.0], [-2.0 * sb * x0, sb * S]]
            return 1, f, J

        np.random.seed(99)
        s_gnm = gnm.sampler([1.0, 1.0 / S], scaled_rosen)
        s_gnm.dynamic(2)
        s_gnm.sample(10000)
        s_gnm.burn(1000)
        tau_gnm = float(np.max(s_gnm.acor()))

        np.random.seed(99)
        fn_mala = gnm.function(scaled_rosen)
        res_mala = run_mala([1.0, 1.0 / S], fn_mala, 10000, tau=0.005)
        tau_mala = float(np.max(gnm.autocorr_time(res_mala["chain"][1000:])))

        np.random.seed(99)
        fn_hmc = gnm.function(scaled_rosen)
        res_hmc = run_hmc([1.0, 1.0 / S], fn_hmc, 10000, eps=0.02, n_leapfrog=10)
        tau_hmc = float(np.max(gnm.autocorr_time(res_hmc["chain"][100:])))

        label = f"S = 1e{int(np.log10(S))} (kappa=1e{int(2*np.log10(S))})"
        print(f"{label:<22} | {s_gnm.accept_rate:18.3f} | {tau_gnm:15.2f} | {res_mala['accept_rate']:12.3f} | {tau_mala:10.1f} | {res_hmc['accept_rate']:11.3f} | {tau_hmc:10.1f}")
        results.append({
            "scale": S,
            "gnm_accept": s_gnm.accept_rate,
            "gnm_tau": tau_gnm,
            "mala_accept": res_mala["accept_rate"],
            "mala_tau": tau_mala,
            "hmc_accept": res_hmc["accept_rate"],
            "hmc_tau": tau_hmc,
        })
    return results


# ============================================================================
# Part 2: Head-to-Head Benchmark vs. State-of-the-Art Samplers
# ============================================================================

def benchmark_problem(title, x0, model_fn, args, prior_m=None, prior_H=None,
                      n_eval_budget=40000, rwm_scale=0.1, mala_tau=0.002, hmc_eps=0.015):
    print("\n" + "=" * 78)
    print(f"BENCHMARK: {title} (Budget = {n_eval_budget:,} model evaluations)")
    print("=" * 78)
    print(f"{'Sampler':<26} | {'Accept':>7} | {'Evals':>7} | {'Max tau':>8} | {'ESS':>8} | {'ESS / 1k evals':>14} | {'ESS / sec':>10}")
    print("-" * 96)

    records = []

    # 1. RWM
    np.random.seed(2026)
    fn_rwm = gnm.function(model_fn, args)
    out_rwm = run_rwm(x0, fn_rwm, n_eval_budget, step_scale=rwm_scale, prior_m=prior_m, prior_H=prior_H)
    c_rwm = out_rwm["chain"][len(out_rwm["chain"]) // 10:]
    tau_rwm = float(np.max(gnm.autocorr_time(c_rwm)))
    ess_rwm = len(c_rwm) / tau_rwm

    # 2. MALA
    np.random.seed(2026)
    fn_mala = gnm.function(model_fn, args)
    out_mala = run_mala(x0, fn_mala, n_eval_budget, tau=mala_tau, prior_m=prior_m, prior_H=prior_H)
    c_mala = out_mala["chain"][len(out_mala["chain"]) // 10:]
    tau_mala = float(np.max(gnm.autocorr_time(c_mala)))
    ess_mala = len(c_mala) / tau_mala

    # 3. HMC (L=10)
    np.random.seed(2026)
    fn_hmc = gnm.function(model_fn, args)
    out_hmc = run_hmc(x0, fn_hmc, n_eval_budget, eps=hmc_eps, n_leapfrog=10, prior_m=prior_m, prior_H=prior_H)
    c_hmc = out_hmc["chain"][len(out_hmc["chain"]) // 10:]
    tau_hmc = float(np.max(gnm.autocorr_time(c_hmc)))
    ess_hmc = len(c_hmc) / tau_hmc

    # 4. emcee (Ensemble Stretch Move)
    np.random.seed(2026)
    fn_emcee = gnm.function(model_fn, args)
    out_emcee = run_emcee(x0, fn_emcee, n_eval_budget, prior_m=prior_m, prior_H=prior_H)
    tau_emcee = float(np.max(out_emcee["walker_taus"]))
    n_post_emcee = int(0.9 * len(out_emcee["chain"]))
    ess_emcee = n_post_emcee / tau_emcee

    # 5. GNM-Static (k=2)
    np.random.seed(2026)
    s_stat = gnm.sampler(x0, model_fn, args)
    if prior_m is not None and prior_H is not None:
        s_stat.prior(prior_m, prior_H)
    s_stat.static(2, 0.25)
    n_gnm_steps = int(n_eval_budget / 1.5)
    t0 = time.perf_counter()
    s_stat.sample(n_gnm_steps)
    t_stat = time.perf_counter() - t0
    s_stat.burn(n_gnm_steps // 10)
    tau_stat = float(np.max(s_stat.acor()))
    ess_stat = len(s_stat.chain) / tau_stat

    # 6. GNM-Dynamic (k=2)
    np.random.seed(2026)
    s_dyn = gnm.sampler(x0, model_fn, args)
    if prior_m is not None and prior_H is not None:
        s_dyn.prior(prior_m, prior_H)
    s_dyn.dynamic(2)
    t0 = time.perf_counter()
    s_dyn.sample(n_gnm_steps)
    t_dyn = time.perf_counter() - t0
    s_dyn.burn(n_gnm_steps // 10)
    tau_dyn = float(np.max(s_dyn.acor()))
    ess_dyn = len(s_dyn.chain) / tau_dyn

    rows = [
        ("Random-Walk Metropolis", out_rwm["accept_rate"], out_rwm["evals"], tau_rwm, ess_rwm, out_rwm["time"]),
        ("MALA (1st-Order Langevin)", out_mala["accept_rate"], out_mala["evals"], tau_mala, ess_mala, out_mala["time"]),
        ("HMC (L=10 Leapfrog)", out_hmc["accept_rate"], out_hmc["evals"], tau_hmc, ess_hmc, out_hmc["time"]),
        ("emcee (Ensemble Stretch)", out_emcee["accept_rate"], out_emcee["evals"], tau_emcee, ess_emcee, out_emcee["time"]),
        ("gnm (Static, k=2)", s_stat.accept_rate, s_stat.call_count, tau_stat, ess_stat, t_stat),
        ("gnm (Dynamic, k=2)", s_dyn.accept_rate, s_dyn.call_count, tau_dyn, ess_dyn, t_dyn),
    ]

    for name, acc, ev, tau, ess, sec in rows:
        ess_1k = (ess / ev) * 1000.0
        ess_sec = ess / sec
        print(f"{name:<26} | {acc:7.3f} | {ev:7d} | {tau:8.2f} | {ess:8.0f} | {ess_1k:14.2f} | {ess_sec:10.0f}")
        records.append({
            "sampler": name,
            "accept_rate": acc,
            "evals": ev,
            "max_tau": tau,
            "ess": ess,
            "ess_per_1k_evals": ess_1k,
            "ess_per_sec": ess_sec,
        })
    return records


def main():
    r1a = exp_1a_double_well_accuracy()
    r1b = exp_1b_rosenbrock_backoff_ablation()
    r1c = exp_1c_affine_invariance_stress()

    # Suite 2A: 2D Stiff Curved Rosenbrock (b = 100)
    def rosen_2d(x, args):
        a, sb = args['a'], args['sb']
        return 1, [a - x[0], sb * (x[1] - x[0] ** 2)], [[-1.0, 0.0], [-2.0 * sb * x[0], sb]]

    r2a = benchmark_problem(
        "2A. Stiff 2D Rosenbrock Banana (b=100)",
        x0=[1.0, 1.0],
        model_fn=rosen_2d,
        args={'a': 1.0, 'sb': 10.0},
        n_eval_budget=30000,
        rwm_scale=0.15,
        mala_tau=0.002,
        hmc_eps=0.015,
    )

    # Suite 2B: 6D Anisotropic Coupled Rosenbrock System (condition number 10^6)
    scales_6d = np.array([1.0, 10.0, 100.0, 1.0, 10.0, 1000.0])

    def aniso_6d(x, _args):
        u = scales_6d * x
        f = np.empty(6)
        J = np.zeros((6, 6))
        for b in range(3):
            i0, i1 = 2 * b, 2 * b + 1
            f[i0] = 1.0 - u[i0]
            f[i1] = 2.0 * (u[i1] - u[i0] ** 2)
            J[i0, i0] = -scales_6d[i0]
            J[i1, i0] = -4.0 * u[i0] * scales_6d[i0]
            J[i1, i1] = 2.0 * scales_6d[i1]
        return 1, f, J

    x0_6d = np.ones(6) / scales_6d

    r2b = benchmark_problem(
        "2B. 6D Anisotropic Coupled Rosenbrock (Condition Number 10^6)",
        x0=x0_6d,
        model_fn=aniso_6d,
        args=None,
        n_eval_budget=30000,
        rwm_scale=0.002,
        mala_tau=1e-6,
        hmc_eps=1e-3,
    )

    # Suite 2C: 4D Ill-Posed Multi-Exponential Time Series
    np.random.seed(3)
    t_grid = np.linspace(0.0, 2.0, 15)
    x_true = np.array([1.0, 2.0, 0.5, 2.5])

    def exp_ts_model(x, args):
        t = args['t']
        y = args['y']
        s = args['s']
        e1 = np.exp(-x[2] * t)
        e2 = np.exp(-x[3] * t)
        pred = x[0] * e1 + x[1] * e2
        f = (pred - y) / s
        J = np.column_stack([e1 / s, e2 / s, -x[0] * t * e1 / s, -x[1] * t * e2 / s])
        return 1, f, J

    y_clean = x_true[0] * np.exp(-x_true[2] * t_grid) + x_true[1] * np.exp(-x_true[3] * t_grid)
    y_obs = y_clean + 0.05 * np.random.randn(t_grid.size)
    ts_args = {'t': t_grid, 'y': y_obs, 's': 0.1}
    prior_m = x_true.copy()
    prior_H = np.eye(4) * 0.5

    r2c = benchmark_problem(
        "2C. 4D Ill-Posed Multi-Exponential Time Series",
        x0=x_true,
        model_fn=exp_ts_model,
        args=ts_args,
        prior_m=prior_m,
        prior_H=prior_H,
        n_eval_budget=30000,
        rwm_scale=0.05,
        mala_tau=0.0005,
        hmc_eps=0.008,
    )


if __name__ == "__main__":
    main()
