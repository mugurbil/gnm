# -*- coding: utf-8 -*-

"""
Functions needed by the gnm sampler
1. test
2. update_params
3. log_K
4. multi_normal
5. det
6. log_det_tril
7. optimize
8. autocorr_time
9. function
"""

__all__ = [
    'test',
    'update_params',
    'log_K',
    'multi_normal',
    'det',
    'log_det_tril',
    'optimize',
    'autocorr_time',
    'function',
    'Function',
]

import inspect
import math
import numpy as np

la = np.linalg

try:
    from scipy.linalg import solve_triangular as _solve_triangular
except ImportError:
    _solve_triangular = None


def test():
    from . import quickstart  # noqa: F401


def _solve_lower(L, b):
    """Solve L x = b where L is lower triangular."""
    n = L.shape[0]
    if n == 1:
        return b / L[0, 0]
    if n == 2:
        x0 = b[0, 0] / L[0, 0]
        x1 = (b[1, 0] - L[1, 0] * x0) / L[1, 1]
        res = np.empty((2, 1), dtype=float)
        res[0, 0] = x0
        res[1, 0] = x1
        return res
    if _solve_triangular is not None and n > 32:
        return _solve_triangular(L, b, lower=True, check_finite=False)
    return la.solve(L, b)


def _solve_upper_T(L, b):
    """Solve L.T x = b where L is lower triangular."""
    n = L.shape[0]
    if n == 1:
        return b / L[0, 0]
    if n == 2:
        x1 = b[1, 0] / L[1, 1]
        x0 = (b[0, 0] - L[1, 0] * x1) / L[0, 0]
        res = np.empty((2, 1), dtype=float)
        res[0, 0] = x0
        res[1, 0] = x1
        return res
    if _solve_triangular is not None and n > 32:
        return _solve_triangular(L, b, lower=True, trans='T', check_finite=False)
    return la.solve(L.T, b)


def update_params(state, t):
    """
    Update parameters
        updates mean and precision to the step size
    Inputs:
        state :
            mu :
                mean
            L  :
                cholesky factor of the precision matrix
        t :
            step size
    Outputs:
        mu :
            updated mean
        L  :
            updated cholesky factor of the precision matrix
    """
    mu = (1.0 - t) * state['x'] + t * state['mu']
    L = state['L'] / math.sqrt(2.0 * t - t * t)
    return mu, L


def log_det_tril(L):
    """
    Compute log(det(L)) for a lower-triangular Cholesky factor L with positive
    diagonal entries, avoiding floating-point overflow/underflow.
    """
    L_arr = np.asarray(L, dtype=float)
    if L_arr.size == 1:
        return math.log(abs(float(L_arr.flat[0])))
    return float(np.sum(np.log(np.abs(L_arr.diagonal()))))


def log_K(Z, X, t):
    """
    Log K
        Log of the proposal probability density function for gnm
    Inputs :
        Z :
            proposed to
        X :
            proposed from
        t :
            step size
    Outputs :
        log of the probability density function
    """
    var_scale = 2.0 * t - t * t
    m = (1.0 - t) * X['x'] + t * X['mu']
    diff = Z['x'] - m
    L = X['L']
    n = diff.size
    log_det_L = X['log_det_L'] if 'log_det_L' in X else log_det_tril(L)
    log_det_Lt = log_det_L - 0.5 * n * math.log(var_scale)
    if n == 1:
        d0 = L[0, 0] * diff[0, 0]
        quad = (d0 * d0) / var_scale
    elif n == 2:
        d0 = L[0, 0] * diff[0, 0] + L[1, 0] * diff[1, 0]
        d1 = L[1, 1] * diff[1, 0]
        quad = (d0 * d0 + d1 * d1) / var_scale
    else:
        Lt_diff = np.dot(L.T, diff).ravel()
        quad = float(np.dot(Lt_diff, Lt_diff)) / var_scale
    return log_det_Lt - 0.5 * quad


def multi_normal(X, t):
    """
    Multivariate normal sampler:
        Generates normal samples with mean m, precision matrix LL'
    Inputs:
        X :
            propose from
        t :
            step size
    Outputs:
        normal with mean m and precision LL'
    """
    m = (1.0 - t) * X['x'] + t * X['mu']
    scale = math.sqrt(2.0 * t - t * t)
    z = np.random.standard_normal(m.shape) * scale
    if 'L_inv_T' in X:
        return np.dot(X['L_inv_T'], z) + m
    return _solve_upper_T(X['L'], z) + m


def det(L):
    """
    Determinant
        Compute the determinant given a lower triangular matrix
    Inputs:
        L :
            lower triangular matrix
    Outputs:
        det_L :
            determinant of L
    """
    L = np.asarray(L)
    if L.size == 1:
        return np.array(L)
    size_L = L.shape
    if size_L[0] != size_L[1]:
        print('Error: Not a square matrix.')
        return 0
    if not np.allclose(np.tril(L), L):
        print('Error: Input is not a lower triangular matrix.')
        return 0
    return float(np.prod(np.diag(L)))


def optimize(t, f_0, d_f_0, f_t, d_f_t, t1=0.05, t2=0.5):
    """
    Third order approximation to find the minimum of the function
        f : function to be optimized over
    Inputs :
        t :
            previous step size
        f_0 : f(0),
            function value at 0
        d_f_0: f'(0),
            the derivative of the function at 0
        f_t : f(t),
            function value at t
        d_f_t : f'(t),
            the derivative of the function at t
        t1 :
            step size reduction if minimum is at 0 or it can't be found
        t2 :
            step size reduction if minimum is at 1
    Outputs :
        alpha_new :
            the new step size that minimizes the function
    """
    if t <= 0:
        print("Error: please enter non-negative t")
        return t

    a = (t * d_f_t - 2.0 * f_t + 2.0 * f_0 + t * d_f_0) / (t ** 3)
    b = (f_t - f_0 - t * d_f_0) / (t ** 2)
    c = d_f_0
    A = 3.0 * a
    B = b - t * a
    C = B ** 2 - A * c

    if C == 0.0:
        if c > 0 or d_f_t > 0:
            t_new = t1 * t
        else:
            t_new = t2 * t
    elif A == 0.0:
        if B == 0.0:
            t_new = t1 * t
        else:
            t_new = -c / (2.0 * B)
    elif C > 0.0:
        t_new = (-B + np.sqrt(C)) / A
    else:
        t_new = t1 * t

    # check the bounds on new step size
    if not np.isfinite(t_new) or t_new < t1 * t:
        t_new = t1 * t
    elif t_new > t2 * t:
        t_new = t2 * t

    return float(t_new)


def autocorr_time(chain, k=5):
    """
    Compute the integrated autocorrelation time for each parameter in a chain
    using Sokal's adaptive windowing algorithm via FFT.

    Inputs:
        chain : ndarray of shape (n_samples, n_dims) or (n_samples,)
        k     : window multiplier c (default 5, matching Goodman & Weare / acor)
    Outputs:
        tau   : ndarray of shape (n_dims,) with integrated autocorrelation times
    """
    x = np.asarray(chain, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_samples, n_dims = x.shape
    if n_samples < 2:
        return np.ones(n_dims)

    # Next power of 2 for zero-padded FFT
    n_fft = 1 << (2 * n_samples - 1).bit_length()
    taus = np.zeros(n_dims)

    for j in range(n_dims):
        series = x[:, j] - np.mean(x[:, j])
        var = np.var(series)
        if var == 0.0 or not np.isfinite(var):
            taus[j] = 1.0
            continue
        f = np.fft.rfft(series, n=n_fft)
        acf = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:n_samples]
        acf /= acf[0]

        # Sokal's self-consistent window: smallest M such that M >= k * tau(M)
        tau_est = 1.0
        for m in range(1, n_samples):
            tau_est += 2.0 * acf[m]
            if m >= k * tau_est:
                break
        taus[j] = max(tau_est, 1.0)

    return taus


def _finite_diff_jacobian(eval_f, x, f_x, eps=1e-7):
    """Compute central finite-difference Jacobian when user model only returns f(x)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    f_x = np.asarray(f_x, dtype=float).reshape(-1)
    n = x.size
    m = f_x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        step = eps * max(1.0, abs(x[j]))
        dx = np.zeros(n, dtype=float)
        dx[j] = step
        chi_r, f_r = eval_f(x + dx)
        chi_l, f_l = eval_f(x - dx)
        if chi_r and chi_l:
            J[:, j] = (np.asarray(f_r).reshape(-1) - np.asarray(f_l).reshape(-1)) / (2.0 * step)
        elif chi_r:
            J[:, j] = (np.asarray(f_r).reshape(-1) - f_x) / step
        elif chi_l:
            J[:, j] = (f_x - np.asarray(f_l).reshape(-1)) / step
        else:
            return False, J
    return True, J


class function(object):

    def __init__(self, f, args=None, jac=None, fd_eps=1e-7):
        """
    Init
        Initialize the developer function class
    Inputs :
        f : user defined function
            ---
            Inputs of f :
                x :
                    input value
                args : (optional)
                    the arguments that the function takes
            Outputs of f :
                Either:
                  1. (chi_x, f_x, J_x) - classic gnm 3-tuple
                  2. (f_x, J_x)        - 2-tuple (chi_x inferred from finiteness)
                  3. f_x               - residual vector only (Jacobian computed via
                                         `jac` callable or central finite differences)
            Demo :
                chi_x, f_x, J_x = f(x, args)
            ---
        args :
            the arguments that the user defined function takes (optional)
        jac :
            optional callable J(x, args) or J(x) when f only returns f_x
        fd_eps :
            step size for finite-difference Jacobian when f only returns f_x
        """
        self._f = f
        self._args = args
        self._jac = jac
        self._fd_eps = fd_eps
        self._count = 0

        # Inspect whether f accepts 1 argument (x) or 2+ arguments (x, args)
        try:
            sig = inspect.signature(f)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
            self._pass_args = has_varargs or (len(params) >= 2)
        except (ValueError, TypeError):
            self._pass_args = True

    def _invoke_raw(self, x):
        if self._pass_args and self._args is not None:
            return self._f(x, self._args)
        elif self._pass_args:
            try:
                return self._f(x, self._args)
            except TypeError:
                self._pass_args = False
                return self._f(x)
        else:
            return self._f(x)

    def _eval_residual_only(self, x):
        out = self._invoke_raw(x)
        if isinstance(out, tuple):
            if len(out) == 3:
                chi_x, f_x, _ = out
                return bool(chi_x), np.asarray(f_x, dtype=float).reshape(-1, 1)
            elif len(out) == 2:
                first, second = out
                if isinstance(first, (bool, int, np.bool_)) and not isinstance(first, np.ndarray):
                    return bool(first), np.asarray(second, dtype=float).reshape(-1, 1)
                f_arr = np.asarray(first, dtype=float).reshape(-1, 1)
                return bool(np.all(np.isfinite(f_arr))), f_arr
        f_arr = np.asarray(out, dtype=float).reshape(-1, 1)
        return bool(np.all(np.isfinite(f_arr))), f_arr

    def __call__(self, x):
        """
    Call
        Calls the user defined function
    Inputs:
        x :
            input value
    Outputs:
        chi_x, f_x, J_x = f(x, args)
        """
        self._count += 1
        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            x_1d = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x.ravel()
        else:
            x_1d = np.asarray(x, dtype=np.float64).ravel()

        out = self._invoke_raw(x_1d)

        if isinstance(out, tuple):
            n_out = len(out)
            if n_out == 3:
                chi_x, f_x, J_x = out
                if not chi_x:
                    return False, None, None
                if isinstance(f_x, np.ndarray) and f_x.dtype == np.float64:
                    f_arr = f_x if (f_x.ndim == 2 and f_x.shape[1] == 1) else f_x.reshape(-1, 1)
                else:
                    f_arr = np.asarray(f_x, dtype=np.float64).reshape(-1, 1)
                if isinstance(J_x, np.ndarray) and J_x.dtype == np.float64 and J_x.ndim == 2:
                    J_arr = J_x
                else:
                    J_arr = np.asarray(J_x, dtype=np.float64)
                return True, f_arr, J_arr
            elif n_out == 2:
                first, second = out
                # Disambiguate (chi_x, f_x) vs (f_x, J_x)
                if isinstance(first, (bool, np.bool_)) and self._jac is not None:
                    chi_x = bool(first)
                    f_x = np.reshape(np.array(second, dtype=float), (-1, 1))
                    J_x = np.array(
                        self._jac(x_1d, self._args) if self._pass_args and self._args is not None else self._jac(x_1d),
                        dtype=float,
                    )
                    return chi_x, f_x, J_x
                f_x = np.reshape(np.array(first, dtype=float), (-1, 1))
                J_x = np.array(second, dtype=float)
                chi_x = bool(np.all(np.isfinite(f_x)) and np.all(np.isfinite(J_x)))
                return chi_x, f_x, J_x

        # Residual-only output: compute Jacobian via self._jac or finite differences
        f_x = np.reshape(np.array(out, dtype=float), (-1, 1))
        if not np.all(np.isfinite(f_x)):
            return False, f_x, np.zeros((f_x.size, x.size))
        if self._jac is not None:
            try:
                J_raw = self._jac(x, self._args) if (self._pass_args and self._args is not None) else self._jac(x)
            except TypeError:
                J_raw = self._jac(x)
            J_x = np.array(J_raw, dtype=float)
            chi_x = bool(np.all(np.isfinite(J_x)))
            return chi_x, f_x, J_x

        ok, J_x = _finite_diff_jacobian(self._eval_residual_only, x, f_x, eps=self._fd_eps)
        return ok and bool(np.all(np.isfinite(J_x))), f_x, J_x

    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
              p=2, l_max=50, r=0.5):
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
    Inputs :
        x_min :
            lower bound on the domain
        x_max :
            upper bound on the domain
        ** Warning **
            x_min and x_max must be arrays of the same dimension
    Optional inputs :
        dx : (2*10^-4)
            the ratio of dx to the size of the domain
        N : (1000)
            number of test points
        eps_max : (10^-4)
            the maximum value error is allowed to be to confirm convergence
        p : (2)
            to specify the norm of the error (p-norm)
        l_max : (50)
            maximum number of tries to reduce dx
        r : (0.5)
            dx will be multiplied by this constant each step the error
            exceeds error_bound until l_max is reached
        ** Warning **
            keep in mind machine precision when changing l_max and r
    Outputs :
        error :
          * 1 if it did not pass the checks,
          * 0 if converged,
          * eps, the error of the numerical gradient point if no convergence
        """
        x_min = np.reshape(np.array(x_min, dtype=float), (-1))
        x_max = np.reshape(np.array(x_max, dtype=float), (-1))
        # begin checks
        if x_min.size != x_max.size or np.any(x_min >= x_max):
            print("Error: All values of x_min should be less than the "
                  "corresponding values for x_max.")
            return 1
        # end checks
        # begin test
        k = 0
        while k < N:
            # reset dx each time
            D_x = (x_max - x_min) * dx

            # Sample random points in the state space
            x = np.random.uniform(x_min + D_x, x_max - D_x)

            # Compute jacobian at x
            l = 0
            test_convergence = 1
            while test_convergence:
                chi_x, f_x, J_x = self.__call__(x)
                D_f_x = J_x * 0.0
                for j in range(np.size(x)):
                    # calculate the derivative of each component of f
                    d_x = D_x * 0.0
                    d_x[j] = D_x[j]
                    chi_x_r, f_x_r, J_x_r = self.__call__(x + d_x)
                    chi_x_l, f_x_l, J_x_l = self.__call__(x - d_x)
                    # check if the function is defined on these points
                    if not (chi_x and chi_x_r and chi_x_l):
                        # discard this trial if one of the values is not defined
                        test_convergence = 0  # break outer loop
                        break
                    d_f = (f_x_r - f_x_l) / (2.0 * d_x[j])
                    D_f_x[:, j] = d_f[:, 0]
                if not test_convergence:
                    break
                eps = la.norm(D_f_x - J_x, p) / np.size(J_x)
                if eps < eps_max:
                    test_convergence = 0  # break outer loop
                    k += 1
                else:
                    D_x = D_x * r
                    if l > l_max:  # numerical gradient did not converge
                        return float(eps)
                    l += 1
        return 0
        # end test

    @property
    def f(self):
        return self._f

    @property
    def args(self):
        return self._args

    @property
    def count(self):
        return self._count


Function = function
