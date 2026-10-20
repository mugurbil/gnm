# -*- coding: utf-8 -*-
"""Comprehensive test suite for gnm."""

import os
import numpy as np
import pytest
from scipy import integrate

import gnm


def well_model(x, args):
    y = args['y']
    s = args['s']
    return 1, [(x[0] ** 2 - y[0]) / s], [[(2.0 * x[0]) / s]]


def rosenbrock_model(x, args):
    a = args['a']
    b = args['b']
    z = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    dx = -2.0 * (a - x[0]) + 2.0 * b * (x[1] - x[0] ** 2) * (-2.0 * x[0])
    dy = 2.0 * b * (x[1] - x[0] ** 2)
    return 1, [z], [[dx, dy]]


def test_1d_well_posterior_accuracy():
    """Verify sampled 1D double-well matches exact quadrature posterior within error bars."""
    np.random.seed(42)
    data = {'y': [1.0], 's': 1.0}
    jagger = gnm.sampler([0.1], well_model, data)
    jagger.prior([0.0], [[1.0]])

    assert jagger.Jtest([-3.0], [3.0], N=100) == 0

    jagger.static(1, 0.2)
    jagger.sample(12000)
    jagger.burn(2000)

    assert 0.2 < jagger.accept_rate < 0.95
    assert jagger.chain.shape == (10000, 1)

    x_grid, p_est, err = jagger.error_bars(40, [-3.0], [3.0])
    norm, _ = integrate.quad(lambda u: jagger.posterior([u]), -6.0, 6.0)
    p_exact = np.array([jagger.posterior([u]) / norm for u in x_grid[0]])

    # L1 error between histogram estimate and exact posterior should be small
    dx = 6.0 / 40.0
    l1_err = np.sum(np.abs(p_est - p_exact)) * dx
    assert l1_err < 0.08


def test_dynamic_backoff_and_deep_recursion():
    """Verify dynamic back-off with multiple steps (k_max=3) and log-space _log_P."""
    np.random.seed(7)
    data = {'y': [2.0], 's': 0.5}
    jagger = gnm.sampler([0.5], well_model, data)
    jagger.prior([0.0], [[1.0]])
    jagger.dynamic(3)
    jagger.sample(2000)
    jagger.burn(200)

    assert jagger.chain.shape == (1800, 1)
    assert jagger.accept_rate > 0.3
    # Verify some back-off steps (k >= 1) were accepted
    assert np.sum(jagger.step_count[2:]) > 0


def test_large_residual_no_underflow():
    """
    When ||f(x)||^2 / 2 > 1500, exp(-||f(x)||^2 / 2) underflows float64 to 0.0.
    Verify log-space _log_P still functions properly without NaN or false rejection.
    """
    np.random.seed(11)

    def offset_model(x):
        # First residual component is a constant 60.0 -> ||f||^2 / 2 >= 1800
        return 1, [60.0, x[0] - 1.0], [[0.0], [1.0]]

    jagger = gnm.sampler([0.0], offset_model)
    jagger.dynamic(2)
    jagger.sample(500)
    assert jagger.accept_rate > 0.5
    assert np.all(np.isfinite(jagger.chain))


def test_rank_deficient_jacobian_without_prior():
    """
    Rosenbrock has m=1 residual and n=2 parameters, so J'J is rank-1 (singular).
    Verify _cholesky_regularized allows sampling even without a prior.
    """
    np.random.seed(19)
    args = {'a': 1.0, 'b': 1.0}
    jagger = gnm.sampler([0.5, 0.5], rosenbrock_model, args)
    jagger.static(1, 0.2)
    jagger.sample(300)
    assert jagger.chain.shape == (300, 2)
    assert jagger.accept_rate > 0.1


def test_auto_jacobian_and_flexible_signatures():
    """Test models that omit args, return (f, J), or return only f(x) (finite-diff Jacobian)."""
    np.random.seed(23)

    # 1. Residual-only function with no args (uses automatic central finite differences)
    def residual_only(x):
        return [x[0], x[0] ** 2 + x[1]]

    jagger_fd = gnm.sampler([0.1, -0.1], residual_only)
    assert jagger_fd.Jtest([-2.0, -2.0], [2.0, 2.0], N=50) == 0
    jagger_fd.sample(500)
    assert jagger_fd.chain.shape == (500, 2)
    assert jagger_fd.accept_rate > 0.4

    # 2. 2-tuple (f_x, J_x) return signature
    def two_tuple_model(x):
        return [x[0], x[0] ** 2 + x[1]], [[1.0, 0.0], [2.0 * x[0], 1.0]]

    jagger_2t = gnm.sampler([0.1, -0.1], two_tuple_model)
    assert jagger_2t.Jtest([-2.0, -2.0], [2.0, 2.0], N=50) == 0


def test_jtest_detects_wrong_jacobian_and_bad_bounds():
    """Verify Jtest catches an incorrect Jacobian and returns 1 on invalid bounds."""
    def wrong_model(x, args):
        return 1, [x[0] ** 2, x[0] * x[1]], [[2.0 * x[0], 0.0], [x[0], x[1]]]

    fn = gnm.function(wrong_model, {})
    err = fn.Jtest([1.0, 1.0], [2.0, 2.0], N=20)
    assert err > 0

    # Invalid bounds x_min >= x_max should return 1 instead of calling exit(0)
    assert fn.Jtest([2.0, 2.0], [1.0, 1.0], N=5) == 1


def test_save_and_load_resume(tmp_path):
    """Verify save() and load() preserve numpy types so sample() can resume afterward."""
    np.random.seed(31)
    data = {'y': [1.0], 's': 1.0}
    jagger = gnm.sampler([0.2], well_model, data)
    jagger.prior([0.0], [[1.0]])
    jagger.sample(200)

    save_file = tmp_path / "chain.dat"
    jagger.save(str(save_file))

    jagger2 = gnm.sampler([0.2], well_model, data)
    jagger2.prior([0.0], [[1.0]])
    jagger2.load(str(save_file))
    assert jagger2.n_samples == 200
    assert isinstance(jagger2._X['J'], np.ndarray)

    # Resume sampling after load
    jagger2.sample(100)
    assert jagger2.n_samples == 300
    assert jagger2.chain.shape == (300, 1)


def test_builtin_acor_and_multidim_error_bars():
    """Verify built-in FFT autocorrelation time and n-D error_bars projection."""
    np.random.seed(101)
    args = {'a': 1.0, 'b': 1.0}
    jagger = gnm.sampler([0.1, 0.1], rosenbrock_model, args)
    jagger.prior([0.0, 0.0], np.eye(2))
    jagger.sample(2000)

    taus = jagger.acor()
    assert taus.shape == (2,)
    assert np.all(taus >= 1.0)
    assert np.all(np.isfinite(taus))

    x_grid, p_x, err = jagger.error_bars(25, [-2.0, -1.0], [2.0, 3.0])
    assert x_grid.shape == (2, 25)
    assert np.isclose(x_grid[0, 0], -2.0 + 4.0 * 0.5 / 25)
    assert np.isclose(x_grid[1, 0], -1.0 + 4.0 * 0.5 / 25)
    assert p_x.shape == (25,)
    assert err.shape == (25,)


def test_from_least_squares():
    """Verify initializing sampler from scipy.optimize.least_squares MAP point."""
    np.random.seed(55)
    args = {'a': 1.0, 'b': 2.0}
    jagger = gnm.sampler.from_least_squares(
        rosenbrock_model,
        x0=[-1.5, 2.0],
        args=args,
        prior_mean=[1.0, 1.0],
        prior_precision=np.eye(2),
    )
    # MAP of Rosenbrock(1, 2) with N([1,1], I) prior is exactly (1, 1)
    assert np.allclose(jagger._X['x'].reshape(-1), [1.0, 1.0], atol=1e-5)
    jagger.sample(300)
    assert jagger.chain.shape == (300, 2)


def test_r0_auto_and_tune_in_high_dimensions():
    """Verify dimension-scaled r0='auto' and burn-in Robbins-Monro tune()."""
    np.random.seed(77)
    n_blocks = 4  # 8D coupled Rosenbrock

    def rosen_8d(x):
        f = np.empty(2 * n_blocks)
        J = np.zeros((2 * n_blocks, 2 * n_blocks))
        for b in range(n_blocks):
            i0, i1 = 2 * b, 2 * b + 1
            f[i0] = 1.0 - x[i0]
            f[i1] = 2.0 * (x[i1] - x[i0] ** 2)
            J[i0, i0] = -1.0
            J[i1, i0] = -4.0 * x[i0]
            J[i1, i1] = 2.0
        return 1, f, J

    jagger = gnm.sampler(np.ones(2 * n_blocks), rosen_8d, r0="auto")
    assert np.isclose(jagger.r0, 2.0 / 8.0)

    jagger.dynamic(1)
    tuned_r0 = jagger.tune(400, target_accept=0.45)
    assert 1e-4 <= tuned_r0 <= 1.0
    assert jagger.n_samples == 0  # reset_chain=True keeps chain clean after tuning

    jagger.sample(600)
    assert jagger.chain.shape == (600, 8)
    assert jagger.accept_rate > 0.30


def test_jax_kernel_and_vmap():
    """Verify XLA-compiled gnm.jax kernel with automatic jax.jacfwd and multi-chain vmap."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from gnm.jax import sample_jax

    def well_res(x):
        return jnp.array([(x[0] ** 2 - 2.0) / 1.0])

    chain, acc = sample_jax(
        well_res,
        jnp.array([0.5]),
        4000,
        key=jax.random.PRNGKey(42),
        prior_mean=jnp.array([0.0]),
        prior_precision=jnp.array([[1.0]]),
        max_steps=2,
        dynamic=True,
    )
    assert chain.shape == (4000, 1)
    assert 0.5 < acc <= 1.0
    assert abs(float(jnp.mean(chain[500:, 0] ** 2)) - 1.246) < 0.15

    # Multi-chain parallel vmap on 2D Rosenbrock
    def rosen_res(x):
        return jnp.array([1.0 - x[0], 5.0 * (x[1] - x[0] ** 2)])

    chains, acc_vmap = sample_jax(
        rosen_res,
        jnp.array([1.0, 1.0]),
        600,
        key=jax.random.PRNGKey(7),
        n_chains=4,
    )
    assert chains.shape == (4, 600, 2)
    assert acc_vmap > 0.2


