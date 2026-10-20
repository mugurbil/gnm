# -*- coding: utf-8 -*-
"""
`gnm.jax`: XLA-compiled (@jax.jit) Dynamic Gauss-Newton Metropolis kernel.

Runs the full Gauss-Newton proposal, Cholesky preconditioning, and delayed-rejection
(static geometric or dynamic cubic-Hermite back-off) inside `jax.lax.scan`, with
automatic forward-mode differentiation (`jax.jacfwd`) when only `residual_fn(x)`
is supplied, and multi-chain vectorization via `jax.vmap`.
"""

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class GNMState(NamedTuple):
    x: jax.Array          # (n,)
    f: jax.Array          # (m,)
    J: jax.Array          # (m, n)
    L: jax.Array          # (n, n) lower-triangular Cholesky factor
    mu: jax.Array         # (n,) Gauss-Newton proposal mean
    log_det_L: jax.Array  # () scalar
    log_p: jax.Array      # () scalar
    f_sq: jax.Array       # () scalar


def _proposal_params_jax(
    x: jax.Array,
    f: jax.Array,
    J: jax.Array,
    has_prior: bool,
    m_prior: jax.Array,
    H_prior: jax.Array,
    Hm_prior: jax.Array,
    ln_H: float,
    reg: float,
) -> GNMState:
    n = x.shape[0]
    JJ = J.T @ J
    P = JJ + (H_prior if has_prior else 0.0) + reg * jnp.eye(n, dtype=x.dtype)
    # Adaptive Levenberg-Marquardt floor to guarantee positive definiteness under JIT
    eig_floor = 1e-10
    L = jnp.linalg.cholesky(P)
    bad_chol = jnp.any(jnp.isnan(L)) | (jnp.min(jnp.diag(L)) <= 1e-8)
    L = jnp.where(
        bad_chol,
        jnp.linalg.cholesky(P + 1e-2 * jnp.eye(n, dtype=x.dtype)),
        L,
    )
    log_det_L = jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
    JTf = J.T @ f
    rhs = (Hm_prior - JTf + JJ @ x) if has_prior else JTf
    y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
    step = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    mu = step if has_prior else (x - step)

    f_sq = jnp.dot(f, f)
    log_lik = -0.5 * f_sq
    if has_prior:
        diff = x - m_prior
        log_p = ln_H - 0.5 * jnp.dot(diff, H_prior @ diff) + log_lik
    else:
        log_p = log_lik + 0.0 * eig_floor
    return GNMState(x=x, f=f, J=J, L=L, mu=mu, log_det_L=log_det_L, log_p=log_p, f_sq=f_sq)


def _propose_jax(state: GNMState, t: jax.Array, z_std: jax.Array) -> jax.Array:
    m = (1.0 - t) * state.x + t * state.mu
    scale = jnp.sqrt(2.0 * t - t * t)
    delta = jax.scipy.linalg.solve_triangular(state.L.T, scale * z_std, lower=False)
    return m + delta


def _log_K_jax(Z: GNMState, X: GNMState, t: jax.Array) -> jax.Array:
    n = X.x.shape[0]
    var_scale = 2.0 * t - t * t
    m = (1.0 - t) * X.x + t * X.mu
    Lt_diff = X.L.T @ (Z.x - m)
    quad = jnp.dot(Lt_diff, Lt_diff) / var_scale
    return X.log_det_L - 0.5 * n * jnp.log(var_scale) - 0.5 * quad


def _log_P_jax(target: GNMState, source: GNMState, k: int, Z_stages: list, r_stages: list) -> jax.Array:
    if k == 0:
        return source.log_p + _log_K_jax(target, source, r_stages[0])
    Zk = Z_stages[k]
    log_P_zk_z = _log_P_jax(Zk, source, k - 1, Z_stages, r_stages)
    log_P_z_zk = _log_P_jax(source, Zk, k - 1, Z_stages, r_stages)
    valid = jnp.isfinite(log_P_zk_z) & (log_P_zk_z > log_P_z_zk)
    diff = jnp.where(valid, log_P_z_zk - log_P_zk_z, -1.0)
    log_diff = jnp.where(valid, log_P_zk_z + jnp.log1p(-jnp.exp(diff)), -jnp.inf)
    return log_diff + _log_K_jax(target, source, r_stages[k])


def _backoff_step_jax(X: GNMState, Z: GNMState, r: jax.Array, dynamic: bool, step_size: float, t1: float = 0.05, t2: float = 0.5) -> jax.Array:
    if not dynamic:
        return r * step_size
    v = (Z.x - X.x) / r
    dp_0 = 2.0 * jnp.dot(X.f, X.J @ v)
    dp_r = 2.0 * jnp.dot(Z.f, Z.J @ v)
    f_0, f_t = X.f_sq, Z.f_sq
    a = (r * dp_r - 2.0 * f_t + 2.0 * f_0 + r * dp_0) / (r ** 3)
    b = (f_t - f_0 - r * dp_0) / (r ** 2)
    c = dp_0
    A = 3.0 * a
    B = b - r * a
    C = B * B - A * c
    t_quad = (-B + jnp.sqrt(jnp.maximum(C, 0.0))) / jnp.where(A == 0.0, 1.0, A)
    t_new = jnp.where(C > 0.0, t_quad, t1 * r)
    t_new = jnp.where(jnp.isfinite(t_new), jnp.clip(t_new, t1 * r, t2 * r), t1 * r)
    return t_new


@partial(
    jax.jit,
    static_argnames=("eval_both", "n_samples", "has_prior", "max_steps", "dynamic"),
)
def _run_single_chain_jit(
    init_x: jax.Array,
    chain_key: jax.Array,
    m_p: jax.Array,
    H_p: jax.Array,
    Hm_p: jax.Array,
    ln_H: float,
    step_size: float,
    r0: float,
    reg: float,
    *,
    eval_both: Callable,
    n_samples: int,
    has_prior: bool,
    max_steps: int,
    dynamic: bool,
):
    n = init_x.shape[0]
    prop_fn = partial(
        _proposal_params_jax,
        has_prior=has_prior,
        m_prior=m_p,
        H_prior=H_p,
        Hm_prior=Hm_p,
        ln_H=ln_H,
        reg=reg,
    )
    f0, J0 = eval_both(init_x)
    init_state = prop_fn(init_x, f0, J0)

    def _scan_step(carry, step_key):
        X, n_acc = carry
        z_keys = jax.random.split(step_key, 2 * (max_steps + 1))
        r_0 = jnp.asarray(r0, dtype=jnp.float64)

        # Stage 0
        z0_std = jax.random.normal(z_keys[0], shape=(n,), dtype=jnp.float64)
        u0 = jax.random.uniform(z_keys[1], dtype=jnp.float64)
        z0 = _propose_jax(X, r_0, z0_std)
        f_z0, J_z0 = eval_both(z0)
        Z0 = prop_fn(z0, f_z0, J_z0)

        Z_stages = [X, Z0]
        r_stages = [r_0]
        log_P_zx = X.log_p + _log_K_jax(Z0, X, r_0)
        log_N = _log_P_jax(X, Z0, 0, Z_stages, r_stages)
        A0 = jnp.where(
            jnp.isfinite(log_N),
            jnp.clip(jnp.exp(jnp.minimum(0.0, log_N - log_P_zx)), 0.0, 1.0),
            0.0,
        )
        accepted = u0 <= A0
        curr_X = jax.tree_util.tree_map(lambda a, b: jnp.where(accepted, a, b), Z0, X)
        log_P_zx = log_P_zx + jnp.log1p(-jnp.minimum(A0, 1.0 - 1e-16))

        # Delayed-rejection back-off stages k = 1 .. max_steps
        for k in range(1, max_steps + 1):
            r_k = _backoff_step_jax(X, Z_stages[-1], r_stages[-1], dynamic, step_size)
            r_stages.append(r_k)
            zk_std = jax.random.normal(z_keys[2 * k], shape=(n,), dtype=jnp.float64)
            uk = jax.random.uniform(z_keys[2 * k + 1], dtype=jnp.float64)

            def _do_stage_k(_):
                zk = _propose_jax(X, r_k, zk_std)
                f_zk, J_zk = eval_both(zk)
                return prop_fn(zk, f_zk, J_zk)

            Zk = jax.lax.cond(accepted, lambda _: Z_stages[-1], _do_stage_k, operand=None)
            Z_stages.append(Zk)
            log_P_zx_k = log_P_zx + _log_K_jax(Zk, X, r_k)
            log_N_k = _log_P_jax(X, Zk, k, Z_stages, r_stages)
            Ak = jnp.where(
                (~accepted) & jnp.isfinite(log_N_k),
                jnp.clip(jnp.exp(jnp.minimum(0.0, log_N_k - log_P_zx_k)), 0.0, 1.0),
                0.0,
            )
            acc_k = (~accepted) & (uk <= Ak)
            curr_X = jax.tree_util.tree_map(lambda a, b: jnp.where(acc_k, a, b), Zk, curr_X)
            accepted = accepted | acc_k
            log_P_zx = log_P_zx_k + jnp.log1p(-jnp.minimum(Ak, 1.0 - 1e-16))

        return (curr_X, n_acc + accepted.astype(jnp.int32)), curr_X.x

    step_keys = jax.random.split(chain_key, n_samples)
    (_, total_acc), chain = jax.lax.scan(_scan_step, (init_state, jnp.int32(0)), step_keys)
    return chain, total_acc / float(n_samples)


_EVAL_BOTH_CACHE = {}


def _get_eval_both(residual_fn, jac_fn):
    cache_key = (residual_fn, jac_fn)
    if cache_key not in _EVAL_BOTH_CACHE:
        if jac_fn == "tuple":
            _EVAL_BOTH_CACHE[cache_key] = residual_fn
        elif jac_fn is not None:
            _EVAL_BOTH_CACHE[cache_key] = lambda x: (residual_fn(x), jac_fn(x))
        else:
            j_auto = jax.jacfwd(residual_fn)
            _EVAL_BOTH_CACHE[cache_key] = lambda x: (residual_fn(x), j_auto(x))
    return _EVAL_BOTH_CACHE[cache_key]


def sample_jax(
    residual_fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    n_samples: int,
    *,
    key: Optional[jax.Array] = None,
    jac_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
    prior_mean: Optional[jax.Array] = None,
    prior_precision: Optional[jax.Array] = None,
    max_steps: int = 2,
    dynamic: bool = True,
    step_size: float = 0.25,
    r0: float = 1.0,
    reg: float = 1e-6,
    n_chains: int = 1,
) -> Tuple[jax.Array, float]:
    """
    Run XLA-compiled Dynamic Gauss-Newton Metropolis sampling via `jax.lax.scan`.

    Args:
        residual_fn: JAX-traceable function `f(x)` returning 1D residual vector `(m,)`,
            or returning `(f(x), J(x))` if `jac_fn="tuple"`.
        x0: Initial parameter vector of shape `(n,)` (or `(n_chains, n)`).
        n_samples: Number of MCMC samples per chain.
        key: JAX PRNGKey (defaults to `jax.random.PRNGKey(0)`).
        jac_fn: Optional callable `J(x)`, or `"tuple"` if `residual_fn(x)` returns `(f, J)`.
            If None, exact forward-mode automatic differentiation `jax.jacfwd(residual_fn)` is used.
        prior_mean: Optional Gaussian prior mean of shape `(n,)`.
        prior_precision: Optional Gaussian prior precision matrix H of shape `(n, n)`.
        max_steps: Maximum number of delayed-rejection back-off steps (default 2).
        dynamic: If True, use cubic-Hermite dynamic back-off; else static geometric back-off.
        step_size: Geometric reduction factor when `dynamic=False`.
        r0: Initial stage-0 step size in (0, 1].
        reg: Baseline Levenberg-Marquardt damping.
        n_chains: Number of parallel chains (vectorized via `jax.vmap` if > 1).

    Returns:
        `(chain, accept_rate)` where `chain` has shape `(n_samples, n)` (or `(n_chains, n_samples, n)`).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    n = x0_arr.shape[-1]
    eval_both = _get_eval_both(residual_fn, jac_fn)

    has_prior = (prior_mean is not None) and (prior_precision is not None)
    if has_prior:
        m_p = jnp.asarray(prior_mean, dtype=jnp.float64).reshape(n)
        H_p = jnp.asarray(prior_precision, dtype=jnp.float64).reshape(n, n)
        _, logdet = jnp.linalg.slogdet(H_p)
        ln_H = float(0.5 * logdet)
        Hm_p = H_p @ m_p
    else:
        m_p = jnp.zeros(n, dtype=jnp.float64)
        H_p = jnp.zeros((n, n), dtype=jnp.float64)
        Hm_p = jnp.zeros(n, dtype=jnp.float64)
        ln_H = 0.0

    runner = partial(
        _run_single_chain_jit,
        m_p=m_p,
        H_p=H_p,
        Hm_p=Hm_p,
        ln_H=ln_H,
        step_size=float(step_size),
        r0=float(r0),
        reg=float(reg),
        eval_both=eval_both,
        n_samples=int(n_samples),
        has_prior=has_prior,
        max_steps=int(max_steps),
        dynamic=bool(dynamic),
    )

    if n_chains == 1 and x0_arr.ndim == 1:
        chain, acc_rate = runner(x0_arr, key)
        return chain, float(acc_rate)

    chain_keys = jax.random.split(key, n_chains)
    if x0_arr.ndim == 1:
        init_xs = x0_arr[None, :] + 1e-2 * jax.random.normal(key, shape=(n_chains, n), dtype=jnp.float64)
    else:
        init_xs = x0_arr
    chains, acc_rates = jax.vmap(runner)(init_xs, chain_keys)
    return chains, float(jnp.mean(acc_rates))
