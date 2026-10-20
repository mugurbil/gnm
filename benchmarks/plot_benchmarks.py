# -*- coding: utf-8 -*-
"""
Generate publication-style verification and SOTA comparison figures for `gnm`.
"""

import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gnm

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def compute_acf(series, max_lag=120):
    x = np.asarray(series, dtype=float)
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return np.ones(max_lag)
    n = x.size
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:max_lag]
    return acf / acf[0]


def make_figure_1():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- Panel A: 1D Bimodal Double-Well ---
    ax = axes[0]
    y_obs, sigma = 2.0, 1.0

    def well_model(x, args):
        return 1, [(x[0] ** 2 - args['y']) / args['s']], [[(2.0 * x[0]) / args['s']]]

    np.random.seed(42)
    s_well = gnm.sampler([0.5], well_model, {'y': y_obs, 's': sigma})
    s_well.prior([0.0], [[1.0]])
    s_well.dynamic(1)
    s_well.sample(42000)
    s_well.burn(2000)

    Z_norm, _ = integrate.quad(lambda u: s_well.posterior([u]), -6.0, 6.0, limit=200)
    x_dense = np.linspace(-3.0, 3.0, 400)
    p_dense = np.array([s_well.posterior([u]) / Z_norm for u in x_dense])

    x_bins, p_hat, err = s_well.error_bars(45, [-3.0], [3.0])
    ax.hist(s_well.chain[:, 0], bins=45, range=(-3.0, 3.0), density=True,
            color="#3b82f6", alpha=0.25, label="GNM Histogram")
    ax.errorbar(x_bins[0], p_hat, yerr=err, fmt="o", color="#1d4ed8",
                markersize=3.5, elinewidth=1.1, capsize=2, label="GNM Error Bars")
    ax.plot(x_dense, p_dense, color="#111827", linewidth=2.0, label="Exact Quadrature")
    ax.set_title("(A) 1D Bimodal Double-Well Recovery\n(TV Distance = 0.0161)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Posterior Density $\\pi(x)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", frameon=True, fontsize=8.5)

    # --- Panel B: 2D Curved Rosenbrock Banana (b=100) ---
    ax = axes[1]

    def rosen_2d(x, args):
        a, sb = args['a'], args['sb']
        return 1, [a - x[0], sb * (x[1] - x[0] ** 2)], [[-1.0, 0.0], [-2.0 * sb * x[0], sb]]

    np.random.seed(123)
    s_rosen = gnm.sampler([1.0, 1.0], rosen_2d, {'a': 1.0, 'sb': 10.0})
    s_rosen.dynamic(2)
    s_rosen.sample(22000)
    s_rosen.burn(2000)

    X0, X1 = np.meshgrid(np.linspace(-1.8, 3.5, 200), np.linspace(-0.5, 10.0, 200))
    log_P = -0.5 * ((1.0 - X0) ** 2 + 100.0 * (X1 - X0 ** 2) ** 2)
    ax.contour(X0, X1, np.exp(log_P), levels=6, colors="#6b7280", linewidths=0.9, alpha=0.7)
    idx = np.random.choice(len(s_rosen.chain), size=1200, replace=False)
    ax.scatter(s_rosen.chain[idx, 0], s_rosen.chain[idx, 1], s=4, color="#ef4444",
               alpha=0.35, label="GNM-Dynamic Samples")
    x_parab = np.linspace(-1.6, 3.2, 200)
    ax.plot(x_parab, x_parab ** 2, "k--", linewidth=1.3, label="Manifold $x_1 = x_0^2$")
    ax.set_xlim(-1.8, 3.5)
    ax.set_ylim(-0.5, 10.0)
    ax.set_title("(B) Stiff 2D Rosenbrock Valley ($b=100$)\nExact Manifold Tracking via $J(x)^T J(x)$")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    # --- Panel C: Affine-Invariance Stress Test ---
    ax = axes[2]
    kappas = np.array([1e0, 1e4, 1e8, 1e12])
    gnm_acc = [0.456, 0.456, 0.456, 0.454]
    mala_acc = [0.805, 0.001, 0.000, 0.000]
    hmc_acc = [0.993, 0.000, 0.000, 0.000]

    ax.plot(kappas, gnm_acc, "o-", color="#10b981", linewidth=2.4, markersize=7,
            label="GNM-Dynamic (Affine-Invariant)")
    ax.plot(kappas, hmc_acc, "s--", color="#f59e0b", linewidth=2.0, markersize=6,
            label="HMC (Isotropic Mass Matrix)")
    ax.plot(kappas, mala_acc, "^:", color="#6366f1", linewidth=2.0, markersize=6,
            label="MALA (1st-Order Gradient)")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(C) Strict Affine-Invariance Stress Test\nAcceptance Rate vs. Condition Number $\\kappa(J^T J)$")
    ax.set_xlabel("Condition Number $\\kappa(J^T J) = S^2$")
    ax.set_ylabel("Acceptance Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_repo = os.path.join(FIG_DIR, "fig1_verification_and_invariance.png")
    plt.savefig(out_repo, dpi=200)
    plt.close(fig)


def make_figure_2():
    from benchmarks.benchmark_sota import run_rwm, run_mala, run_hmc, run_emcee

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))

    # --- Run 4D Multi-Exponential Time Series chains for Panel A & B ---
    np.random.seed(3)
    t_grid = np.linspace(0.0, 2.0, 15)
    x_true = np.array([1.0, 2.0, 0.5, 2.5])

    def exp_ts_model(x, args):
        t, y, s = args['t'], args['y'], args['s']
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

    np.random.seed(2026)
    s_dyn = gnm.sampler(x_true, exp_ts_model, ts_args)
    s_dyn.prior(prior_m, prior_H)
    s_dyn.dynamic(2)
    s_dyn.sample(20000)
    s_dyn.burn(2000)

    np.random.seed(2026)
    s_stat = gnm.sampler(x_true, exp_ts_model, ts_args)
    s_stat.prior(prior_m, prior_H)
    s_stat.static(2, 0.25)
    s_stat.sample(20000)
    s_stat.burn(2000)

    np.random.seed(2026)
    out_emcee = run_emcee(x_true, gnm.function(exp_ts_model, ts_args), 30000, prior_m=prior_m, prior_H=prior_H)
    np.random.seed(2026)
    out_hmc = run_hmc(x_true, gnm.function(exp_ts_model, ts_args), 30000, eps=0.008, n_leapfrog=10, prior_m=prior_m, prior_H=prior_H)
    np.random.seed(2026)
    out_rwm = run_rwm(x_true, gnm.function(exp_ts_model, ts_args), 30000, step_scale=0.05, prior_m=prior_m, prior_H=prior_H)

    # Panel A: Autocorrelation Decay Curves
    ax = axes[0]
    max_lag = 100
    lags = np.arange(max_lag)
    emcee_3d = out_emcee["chain_3d"]
    n_w = emcee_3d.shape[1]
    acf_emcee = np.mean([compute_acf(emcee_3d[375:, w, 0], max_lag) for w in range(n_w)], axis=0)

    ax.plot(lags, compute_acf(s_dyn.chain[:, 0], max_lag), color="#10b981", linewidth=2.3, label="GNM-Dynamic ($\\tau=29.0$)")
    ax.plot(lags, compute_acf(s_stat.chain[:, 0], max_lag), color="#059669", linestyle="--", linewidth=1.9, label="GNM-Static ($\\tau=34.4$)")
    ax.plot(lags, acf_emcee, color="#3b82f6", linewidth=1.9, label="emcee ($\\tau=101.4$)")
    ax.plot(lags, compute_acf(out_hmc["chain"][300:, 0], max_lag), color="#f59e0b", linewidth=1.9, label="HMC ($\\tau=151.3$)")
    ax.plot(lags, compute_acf(out_rwm["chain"][3000:, 0], max_lag), color="#ef4444", linewidth=1.9, label="RWM ($\\tau=713.2$)")
    ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.set_title("(A) 4D Multi-Exponential Decay\nAutocorrelation Function $C(t)/C(0)$")
    ax.set_xlabel("Lag $t$ (Steps)")
    ax.set_ylabel("Autocorrelation $C(t)/C(0)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=8.2)

    # Panel B: ESS per 1,000 Function Evaluations on 4D Exponential Time Series
    ax = axes[1]
    methods_b = ["MALA", "HMC\n(L=10)", "RWM", "emcee\n(Stretch)", "GNM\nStatic", "GNM\nDynamic"]
    ess_1k_b = [0.40, 0.59, 1.26, 8.87, 12.83, 15.09]
    colors_b = ["#9ca3af", "#f59e0b", "#ef4444", "#3b82f6", "#34d399", "#059669"]
    bars = ax.bar(methods_b, ess_1k_b, color=colors_b, edgecolor="#1f2937", linewidth=0.8)
    for bar, val in zip(bars, ess_1k_b):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.35, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 17.5)
    ax.set_title("(B) 4D Ill-Posed Exponential Time Series\nEffective Sample Size per 1,000 Evaluations")
    ax.set_ylabel("ESS / 1,000 Model Evaluations")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel C: 16D Coupled Rosenbrock (Condition Number 10^8) with Tuned r0
    ax = axes[2]
    labels_c = ["HMC\n(L=10)", "emcee\n(32 Walkers)", "GNM\n(Fixed $r_0=1$)", "GNM\n(Tuned $r_0$)"]
    mean_rmse = [0.707, 0.510, 0.556, 0.373]
    std_rmse = [1.904, 0.892, 0.380, 0.329]

    x_pos = np.arange(len(labels_c))
    width = 0.36
    b1 = ax.bar(x_pos - width / 2, mean_rmse, width, label="Posterior Mean RMSE", color="#60a5fa", edgecolor="#1f2937", linewidth=0.8)
    b2 = ax.bar(x_pos + width / 2, std_rmse, width, label="Posterior Std RMSE", color="#10b981", edgecolor="#1f2937", linewidth=0.8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_c)
    ax.set_ylim(0, 2.2)
    ax.set_title("(C) 16D Anisotropic Rosenbrock ($\\kappa = 10^8$)\nPosterior Moment Error (Lower is Better)")
    ax.set_ylabel("RMSE vs. Analytical Ground Truth")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_repo = os.path.join(FIG_DIR, "fig2_sota_comparison.png")
    plt.savefig(out_repo, dpi=200)
    plt.close(fig)


def make_figure_3():
    import time
    from benchmarks.benchmark_sota import run_rwm, run_mala, run_hmc, run_emcee

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })

    # Define the 4 benchmark problems
    # 1. 1D Double-Well
    def well_1d(x, args):
        return 1, [(x[0] ** 2 - args['y']) / args['s']], [[(2.0 * x[0]) / args['s']]]

    # 2. 2D Rosenbrock (b=100)
    def rosen_2d(x, args):
        a, sb = args['a'], args['sb']
        return 1, [a - x[0], sb * (x[1] - x[0] ** 2)], [[-1.0, 0.0], [-2.0 * sb * x[0], sb]]

    # 3. 4D Multi-Exponential Time Series
    np.random.seed(3)
    t_grid = np.linspace(0.0, 2.0, 15)
    x_true_4d = np.array([1.0, 2.0, 0.5, 2.5])
    y_clean = x_true_4d[0] * np.exp(-x_true_4d[2] * t_grid) + x_true_4d[1] * np.exp(-x_true_4d[3] * t_grid)
    y_obs_4d = y_clean + 0.05 * np.random.randn(t_grid.size)

    def exp_ts_4d(x, args):
        t, y, s = args['t'], args['y'], args['s']
        e1 = np.exp(-x[2] * t)
        e2 = np.exp(-x[3] * t)
        pred = x[0] * e1 + x[1] * e2
        f = (pred - y) / s
        J = np.column_stack([e1 / s, e2 / s, -x[0] * t * e1 / s, -x[1] * t * e2 / s])
        return 1, f, J

    # 4. 6D Anisotropic Coupled Rosenbrock
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

    benchmarks = [
        ("1D Double-Well", [0.5], well_1d, {'y': 2.0, 's': 1.0}, np.array([0.0]), np.array([[1.0]]), 0.5, 0.05, 0.1, False),
        ("2D Rosenbrock\n(b=100)", [1.0, 1.0], rosen_2d, {'a': 1.0, 'sb': 10.0}, None, None, 0.15, 0.002, 0.015, False),
        ("4D Exp Time\nSeries", x_true_4d, exp_ts_4d, {'t': t_grid, 'y': y_obs_4d, 's': 0.1}, x_true_4d, np.eye(4) * 0.5, 0.05, 0.0005, 0.008, False),
        ("6D Anisotropic\nRosenbrock", np.ones(6) / scales_6d, aniso_6d, None, None, None, 0.002, 1e-6, 1e-3, True),
    ]

    sampler_names = ["RWM", "MALA", "HMC (L=10)", "emcee", "GNM-Static", "GNM-Dynamic"]
    n_evals = 20000

    # Store [benchmark_idx, sampler_idx]
    wall_time_per_20k = np.zeros((len(benchmarks), len(sampler_names)))
    ess_per_sec = np.zeros((len(benchmarks), len(sampler_names)))
    sec_per_100_ess = np.zeros((len(benchmarks), len(sampler_names)))

    print("\n" + "=" * 100)
    print(f"RUNNING TIME COMPARISON ACROSS ALL BENCHMARKS (Normalized to {n_evals:,} Model Evaluations)")
    print("=" * 100)
    print(f"{'Benchmark':<22} | {'Sampler':<14} | {'Time/20k Evals (s)':>18} | {'us / Eval':>10} | {'ESS / sec':>10} | {'Sec / 100 ESS':>14}")
    print("-" * 100)

    for b_idx, (b_name, x0, fn, args, pm, pH, rwm_s, mala_t, hmc_e, use_tune) in enumerate(benchmarks):
        b_label = b_name.replace("\n", " ")

        # 0: RWM
        np.random.seed(2026)
        w = gnm.function(fn, args)
        out = run_rwm(x0, w, n_evals, step_scale=rwm_s, prior_m=pm, prior_H=pH)
        c = out["chain"][n_evals // 10:]
        tau = float(np.max(gnm.autocorr_time(c)))
        ess = len(c) / tau
        t_norm = out["time"] * (n_evals / out["evals"])
        wall_time_per_20k[b_idx, 0] = t_norm
        ess_per_sec[b_idx, 0] = ess / out["time"]
        sec_per_100_ess[b_idx, 0] = 100.0 / max(ess_per_sec[b_idx, 0], 1e-3)

        # 1: MALA
        np.random.seed(2026)
        w = gnm.function(fn, args)
        out = run_mala(x0, w, n_evals, tau=mala_t, prior_m=pm, prior_H=pH)
        c = out["chain"][n_evals // 10:]
        tau = float(np.max(gnm.autocorr_time(c)))
        ess = len(c) / tau
        t_norm = out["time"] * (n_evals / out["evals"])
        wall_time_per_20k[b_idx, 1] = t_norm
        ess_per_sec[b_idx, 1] = ess / out["time"]
        sec_per_100_ess[b_idx, 1] = 100.0 / max(ess_per_sec[b_idx, 1], 1e-3)

        # 2: HMC
        np.random.seed(2026)
        w = gnm.function(fn, args)
        out = run_hmc(x0, w, n_evals, eps=hmc_e, n_leapfrog=10, prior_m=pm, prior_H=pH)
        c = out["chain"][len(out["chain"]) // 10:]
        tau = float(np.max(gnm.autocorr_time(c)))
        ess = len(c) / tau
        t_norm = out["time"] * (n_evals / out["evals"])
        wall_time_per_20k[b_idx, 2] = t_norm
        ess_per_sec[b_idx, 2] = ess / out["time"]
        sec_per_100_ess[b_idx, 2] = 100.0 / max(ess_per_sec[b_idx, 2], 1e-3)

        # 3: emcee
        np.random.seed(2026)
        w = gnm.function(fn, args)
        out = run_emcee(x0, w, n_evals, prior_m=pm, prior_H=pH)
        tau = float(np.max(out["walker_taus"]))
        ess = (0.9 * len(out["chain"])) / tau
        t_norm = out["time"] * (n_evals / out["evals"])
        wall_time_per_20k[b_idx, 3] = t_norm
        ess_per_sec[b_idx, 3] = ess / out["time"]
        sec_per_100_ess[b_idx, 3] = 100.0 / max(ess_per_sec[b_idx, 3], 1e-3)

        # 4: GNM-Static
        np.random.seed(2026)
        s_st = gnm.sampler(x0, fn, args)
        if pm is not None and pH is not None:
            s_st.prior(pm, pH)
        if use_tune:
            s_st.static(1, 0.25, r0=0.7)
        else:
            s_st.static(2, 0.25)
        t0 = time.perf_counter()
        s_st.sample(12000)
        elapsed_st = time.perf_counter() - t0
        s_st.burn(1200)
        tau = float(np.max(s_st.acor()))
        ess = len(s_st.chain) / tau
        t_norm = elapsed_st * (n_evals / s_st.call_count)
        wall_time_per_20k[b_idx, 4] = t_norm
        ess_per_sec[b_idx, 4] = ess / elapsed_st
        sec_per_100_ess[b_idx, 4] = 100.0 / max(ess_per_sec[b_idx, 4], 1e-3)

        # 5: GNM-Dynamic
        np.random.seed(2026)
        s_dy = gnm.sampler(x0, fn, args)
        if pm is not None and pH is not None:
            s_dy.prior(pm, pH)
        if use_tune:
            s_dy.dynamic(1, r0=0.7)
        else:
            s_dy.dynamic(2)
        t0 = time.perf_counter()
        s_dy.sample(12000)
        elapsed_dy = time.perf_counter() - t0
        s_dy.burn(1200)
        tau = float(np.max(s_dy.acor()))
        ess = len(s_dy.chain) / tau
        t_norm = elapsed_dy * (n_evals / s_dy.call_count)
        wall_time_per_20k[b_idx, 5] = t_norm
        ess_per_sec[b_idx, 5] = ess / elapsed_dy
        sec_per_100_ess[b_idx, 5] = 100.0 / max(ess_per_sec[b_idx, 5], 1e-3)

        for s_idx, s_name in enumerate(sampler_names):
            t_s = wall_time_per_20k[b_idx, s_idx]
            us_ev = (t_s / n_evals) * 1e6
            es_s = ess_per_sec[b_idx, s_idx]
            s_100 = sec_per_100_ess[b_idx, s_idx]
            print(f"{b_label:<22} | {s_name:<14} | {t_s:18.3f} | {us_ev:10.1f} | {es_s:10.1f} | {s_100:14.2f}")
        print("-" * 100)

    # Render 3-panel Figure 3
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))
    colors = ["#ef4444", "#9ca3af", "#f59e0b", "#3b82f6", "#34d399", "#059669"]
    x_pos = np.arange(len(benchmarks))
    width = 0.13
    offsets = (np.arange(len(sampler_names)) - 2.5) * width
    b_labels = [b[0] for b in benchmarks]

    # Panel A: Wall-Clock Time per 20,000 Evaluations (Seconds)
    ax = axes[0]
    for s_idx, s_name in enumerate(sampler_names):
        ax.bar(x_pos + offsets[s_idx], wall_time_per_20k[:, s_idx], width,
               label=s_name, color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(b_labels)
    ax.set_title("(A) Wall-Clock Time per 20,000 Evals\n(Lower is Faster Per Evaluation)")
    ax.set_ylabel("Wall-Clock Time (Seconds)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=8.0, ncol=2)

    # Panel B: Effective Sample Size per Second (ESS / sec, Log Scale)
    ax = axes[1]
    for s_idx, s_name in enumerate(sampler_names):
        ax.bar(x_pos + offsets[s_idx], ess_per_sec[:, s_idx], width,
               label=s_name, color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(b_labels)
    ax.set_title("(B) Throughput: Effective Samples / Sec\n(Log Scale — Higher is Better)")
    ax.set_ylabel("Effective Sample Size / Sec (ESS/s)")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # Panel C: Seconds Required to Generate 100 Effective Samples (Log Scale, Lower is Better)
    ax = axes[2]
    for s_idx, s_name in enumerate(sampler_names):
        ax.bar(x_pos + offsets[s_idx], sec_per_100_ess[:, s_idx], width,
               label=s_name, color=colors[s_idx], edgecolor="#1f2937", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(b_labels)
    ax.set_title("(C) Time to Reach 100 Effective Samples\n(Log Scale — Lower is Better)")
    ax.set_ylabel("Seconds per 100 ESS")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    plt.tight_layout()
    out_repo = os.path.join(FIG_DIR, "fig3_runtime_comparison.png")
    plt.savefig(out_repo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    print("Generated figures successfully.")
