# -*- coding: utf-8 -*-

"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

The Gauss-Newton-Metropolis Algorithm with back-off strategy is
specialized in sampling highly non-linear posterior distributions.
"""

__all__ = ["sampler", "Sampler"]

import json
import math
import sys
import numpy as np

la = np.linalg

from ._numba_kernels import (
    HAS_NUMBA,
    njit_cholesky_and_solve,
    njit_log_post,
    njit_propose,
    njit_stage_step,
)
from .utils import (
    _solve_lower,
    _solve_upper_T,
    autocorr_time,
    function,
    log_K,
    multi_normal,
    optimize,
)


def _cholesky_regularized(P, base_reg=1e-2, max_tries=12):
    """
    Compute the lower-triangular Cholesky factor L of a symmetric matrix P.
    If P is only positive semi-definite or rank-deficient (e.g. m < n without
    a prior, or at a critical point where J(x) = 0), add adaptive
    Levenberg-Marquardt damping lambda * I so Cholesky succeeds with a
    reasonable proposal variance in the nullspace.
    """
    n = P.shape[0]
    if n == 1:
        p00 = float(P[0, 0])
        if p00 > 1e-12:
            res = np.empty((1, 1), dtype=float)
            res[0, 0] = math.sqrt(p00)
            return res
        res = np.empty((1, 1), dtype=float)
        res[0, 0] = math.sqrt(max(p00 + base_reg, base_reg))
        return res
    if n == 2:
        p00 = float(P[0, 0])
        p10 = float(P[1, 0])
        p11 = float(P[1, 1])
        if p00 > 1e-12:
            l00 = math.sqrt(p00)
            l10 = p10 / l00
            rem = p11 - l10 * l10
            if rem > 1e-12 * max(1.0, p00):
                l11 = math.sqrt(rem)
                if min(l00, l11) > 1e-6 * max(1.0, l00, l11):
                    res = np.zeros((2, 2), dtype=float)
                    res[0, 0] = l00
                    res[1, 0] = l10
                    res[1, 1] = l11
                    return res

    try:
        L = la.cholesky(P)
        diag = L.diagonal()
        if float(diag.min()) > 1e-6 * max(1.0, float(diag.max())):
            return L
    except la.LinAlgError:
        pass

    eye = np.eye(n, dtype=float)
    reg = float(base_reg)
    for _ in range(max_tries):
        try:
            return la.cholesky(P + reg * eye)
        except la.LinAlgError:
            reg *= 10.0
    eigvals, eigvecs = la.eigh(0.5 * (P + P.T))
    eigvals = np.maximum(eigvals, base_reg)
    return la.cholesky((eigvecs * eigvals) @ eigvecs.T)


class sampler(object):

    def __init__(self, x, model, args=None, jac=None, reg=0.0, r0=1.0):
        """
    Init
        Initialize the GNM sampler class
    Inputs  :
        x :
            initial guess
        model :
            user defined data model function
        args : (optional)
            arguments for the model
        jac : (optional)
            callable Jacobian if model only returns residual vector f(x)
        reg : (optional, default 0.0)
            baseline Levenberg-Marquardt regularization added to J'J + H
        r0 : (optional, default 1.0 or "auto")
            initial stage-0 step size in (0, 1]. Use "auto" for dimension-scaled
            min(1.0, 2.0 / n) in higher dimensions.
        """
        self._args = args
        self._reg = float(reg)
        self._f = function(model, args, jac=jac)

        x = np.reshape(np.array(x, dtype=float), (-1, 1))
        self._n = np.size(x)  # size of input space
        try:
            chi_x, f_x, J_x = self._f(x)
        except Exception as e:
            raise TypeError(
                "model() could not be evaluated on initial guess: " + str(e)
            )

        if not chi_x:
            raise ValueError("initial guess out of range")

        self._mn = np.size(f_x)
        J_x = np.array(J_x, dtype=float)
        if np.shape(J_x) != (self._mn, self._n):
            raise TypeError(
                "Shape of Jacobian, "
                + str(np.shape(J_x))
                + ", is not correct, (%d, %d)." % (self._mn, self._n)
            )

        self._X = {'x': x, 'f': f_x, 'J': J_x}  # state of x

        # prior parameters
        self._prior = False
        self._m = np.zeros((self._n, 1), dtype=np.float64)
        self._H = np.zeros((self._n, self._n), dtype=np.float64)
        self._Hm = np.zeros((self._n, 1), dtype=np.float64)
        self._ln_H_ = 0.0

        # back-off parameters
        self._max_steps = 1
        self._step_size = 0.1
        self._dynamic = False
        self._opts = {}
        self.r0 = r0

        # sampler outputs
        self._chain = None
        self._step_count = None
        self._n_samples = 0
        self._n_accepted = 0

    @classmethod
    def from_least_squares(cls, model, x0, args=None, jac=None, prior_mean=None,
                           prior_precision=None, reg=0.0, **lsq_kwargs):
        """
        Initialize the GNM sampler at the MAP/MLE estimate found via
        scipy.optimize.least_squares, eliminating burn-in on stiff problems.
        """
        from scipy.optimize import least_squares

        fn_wrap = function(model, args, jac=jac)

        def res_fun(x):
            chi, f_val, _ = fn_wrap(x)
            if not chi:
                return np.full(fn_wrap(x0)[1].size, 1e10)
            r = f_val.reshape(-1)
            if prior_mean is not None and prior_precision is not None:
                L_p = la.cholesky(np.asarray(prior_precision, dtype=float))
                m_p = np.asarray(prior_mean, dtype=float).reshape(-1)
                r = np.concatenate([r, L_p.T @ (x - m_p)])
            return r

        def jac_fun(x):
            _, _, J_val = fn_wrap(x)
            if prior_mean is not None and prior_precision is not None:
                L_p = la.cholesky(np.asarray(prior_precision, dtype=float))
                return np.vstack([J_val, L_p.T])
            return J_val

        res = least_squares(res_fun, np.asarray(x0, dtype=float).reshape(-1),
                            jac=jac_fun, **lsq_kwargs)
        inst = cls(res.x, model, args=args, jac=jac, reg=reg)
        if prior_mean is not None and prior_precision is not None:
            inst.prior(prior_mean, prior_precision)
        return inst

    def prior(self, m, H):
        """
    Set prior
        Set prior values
    Inputs :
        m :
            mean of the prior
        H :
            precision matrix of the prior
    Hiddens :
        ln_H_ : log(det(H))/2
            calculate this once to use everytime log prior is called
        Hm    : < H, m >
            calculate this once to use everytime proposal is called
        """
        if self._prior is True:
            raise Warning("prior information is already set")
        self._prior = True

        # mean
        self._m = np.reshape(np.array(m, dtype=float), (-1, 1))
        if np.size(self._m) != self._n:
            raise TypeError("mean has to be an array of size n")

        # precision
        self._H = np.array(H, dtype=float)
        if np.shape(self._H) != (self._n, self._n):
            raise TypeError("precision has to be a matrix of shape n by n")

        # precalculations using slogdet to avoid overflow in high dimensions
        sign, logdet = la.slogdet(self._H)
        if sign <= 0:
            raise ValueError("prior precision matrix H must be positive definite")
        self._ln_H_ = 0.5 * float(logdet)
        self._Hm = np.dot(self._H, self._m)

    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
              p=2, l_max=50, r=0.5):
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
        """
        if np.size(x_min) != self._n:
            raise TypeError(
                "dimension of x_min, %d, does not match the "
                "dimension of input, %d" % (np.size(x_min), self._n)
            )
        if np.size(x_max) != self._n:
            raise TypeError(
                "dimension of x_max, %d, does not match the "
                "dimension of input, %d." % (np.size(x_max), self._n)
            )
        return self._f.Jtest(x_min, x_max, dx=dx, N=N, eps_max=eps_max, p=p,
                             l_max=l_max, r=r)

    def static(self, max_steps, step_size, r0=None):
        """
    Set Back-off to Static
        Set the sampler parameters for static back off
    Inputs :
        max_steps :
            maximum optimization steps to be taken
        step_size :
            the step size of the back-off
        r0 : (optional)
            initial stage-0 step size in (0, 1] or "auto"
        """
        self._dynamic = False
        if r0 is not None:
            self.r0 = r0
        try:
            self._max_steps = int(max_steps)
        except Exception:
            print("Error: Input 1 (max_steps) has to be an int.")
            return 0
        if self._max_steps < 0:
            print("Warning: Input 1 (max_steps) has to be non-negative.")
            print("Setting max_steps to 0.")
            self._max_steps = 0

        if self._max_steps > 0:
            try:
                assert step_size == float(step_size)
            except AssertionError:
                print("Warning: Input 2 (step_size) is not a float. Converted.")
                step_size = float(step_size)
            except Exception:
                print("Error: Input 2 (step_size) has to be a float.")
                return 0

            if not (0.0 < step_size < 1.0):
                print("Warning: Input 2 (step_size) has to be between 0 and 1.")
                print("Setting step_size to 0.2.")
                step_size = 0.2
            self._step_size = float(step_size)
            if self._step_size ** self._max_steps < 10 ** (-15):
                print("Warning: Back-off gets dangerously small.")

    def dynamic(self, max_steps, opts=None, r0=None):
        """
    Dynamic Switch
        Set the sampler parameters for dynamic back off
    Inputs :
        max_steps :
            maximum back-off steps to be taken
    Optional Inputs:
        opts : ({})
            dictionary containing fancy options (e.g. 't1', 't2', 'r0')
        r0 : (optional)
            initial stage-0 step size in (0, 1] or "auto"
        """
        self._dynamic = True
        try:
            self._max_steps = int(max_steps)
        except Exception:
            raise TypeError("input 1 (max_steps) has to be an integer")
        if self._max_steps < 0:
            raise Warning(
                "input 1 (max_steps) has to be non-negative. Setting (max_steps) to 0."
            )
        self._opts = {} if opts is None else dict(opts)
        if r0 is not None:
            self.r0 = r0
        elif 'r0' in self._opts:
            self.r0 = self._opts['r0']

    def tune(self, n_tune=1000, target_accept=0.45, reset_chain=True):
        """
    Tune Initial Step Size r0 (Burn-In Adaptation)
        Uses Robbins-Monro stochastic approximation during burn-in to adapt the
        initial stage-0 step size r0 in (0, 1] toward `target_accept`, then
        freezes r0 so subsequent sample() calls strictly satisfy detailed balance.
    Inputs :
        n_tune : (1000)
            number of burn-in tuning iterations
        target_accept : (0.45)
            target stage-0 acceptance probability
        reset_chain : (True)
            if True, discard tuning samples and reset counters while keeping
            the warmed-up state X and tuned r0
    Outputs :
        r0 : float
            the tuned stage-0 step size
        """
        n_tune = int(n_tune)
        if self._r0 == 1.0 and self._n > 2:
            self._r0 = min(1.0, 2.0 / self._n)
        log_r0 = float(np.log(self._r0))
        # When back-off stages (k_max >= 1) are enabled, stage-0 should propose bold
        # Gauss-Newton jumps (target stage-0 acceptance ~ 0.234 in n >= 4) so back-off
        # handles local curvature overshoots rather than shrinking r0 to tiny RW steps.
        stage0_target = (
            min(float(target_accept), 0.234)
            if (self._max_steps >= 1 and self._n >= 4)
            else float(target_accept)
        )

        saved_chain = self._chain
        saved_step_count = self._step_count
        saved_n_samples = self._n_samples
        saved_n_accepted = self._n_accepted

        X = self._proposal_params(self._X)
        if HAS_NUMBA:
            n = self._n
            X_arr = np.empty((2, n, 1), dtype=np.float64)
            MU_arr = np.empty((2, n, 1), dtype=np.float64)
            L_arr = np.empty((2, n, n), dtype=np.float64)
            LOG_DET_arr = np.empty(2, dtype=np.float64)
            LOG_P_arr = np.empty(2, dtype=np.float64)
            F_SQ_arr = np.empty(2, dtype=np.float64)
            r_arr = np.empty(2, dtype=np.float64)

            X_arr[0] = X['x']
            MU_arr[0] = X['mu']
            L_arr[0] = X['L']
            LOG_DET_arr[0] = X['log_det_L']
            LOG_P_arr[0] = X['log_p']
            F_SQ_arr[0] = X['f_sq']
            L_inv_T_curr = X['L_inv_T']
            f0 = X['f']
            J0 = X['J']

            z_batch = np.random.standard_normal((n_tune, n, 1))
            u_batch = np.random.rand(n_tune)
            log_min = math.log(1e-4)

            for i in range(n_tune):
                r0_val = min(1.0, max(1e-4, math.exp(log_r0)))
                self._r0 = r0_val
                r_arr[0] = r0_val
                chi_z = False
                for attempt in range(200):
                    z_std = z_batch[i] if attempt == 0 else np.random.standard_normal((n, 1))
                    z = njit_propose(X_arr[0], MU_arr[0], L_inv_T_curr, r0_val, z_std)
                    chi_z, f_z, J_z = self._f(z)
                    if chi_z:
                        break
                if not chi_z:
                    alpha_0 = 0.0
                else:
                    accepted, L_inv_T_z, _, alpha_0 = njit_stage_step(
                        0, 0, z, f_z, J_z,
                        self._prior, self._m, self._H, self._Hm, self._ln_H_, self._reg,
                        LOG_P_arr[0],
                        X_arr, MU_arr, L_arr, LOG_DET_arr, LOG_P_arr, F_SQ_arr,
                        f0, J0, r_arr, False, 0.1, 0.05, 0.5, u_batch[i],
                    )
                    if accepted:
                        L_inv_T_curr = L_inv_T_z
                        f0 = f_z
                        J0 = J_z

                gamma = 0.5 / ((1.0 + i / 50.0) ** 0.65)
                log_r0 = min(0.0, max(log_min, log_r0 + gamma * (alpha_0 - stage0_target)))

            self._r0 = min(1.0, max(1e-4, math.exp(log_r0)))
            self._X = {
                'x': X_arr[0].copy(),
                'f': f0,
                'J': J0,
                'L': L_arr[0].copy(),
                'L_inv_T': L_inv_T_curr,
                'log_det_L': float(LOG_DET_arr[0]),
                'mu': MU_arr[0].copy(),
                'log_p': float(LOG_P_arr[0]),
                'f_sq': float(F_SQ_arr[0]),
            }
        else:
            for i in range(n_tune):
                self._r0 = float(np.clip(np.exp(log_r0), 1e-4, 1.0))
                chi_z = False
                for _ in range(200):
                    z = multi_normal(X, self._r0)
                    chi_z, f_z, J_z = self._f(z)
                    if chi_z:
                        break
                if not chi_z:
                    alpha_0 = 0.0
                else:
                    Z = self._proposal_params({'x': z, 'f': f_z, 'J': J_z})
                    log_P_z_x = float(X['log_p']) + log_K(Z, X, self._r0)
                    log_P_x_z = float(Z['log_p']) + log_K(X, Z, self._r0)
                    if not np.isfinite(log_P_x_z):
                        alpha_0 = 0.0
                    elif log_P_x_z >= log_P_z_x:
                        alpha_0 = 1.0
                    else:
                        alpha_0 = float(np.exp(log_P_x_z - log_P_z_x))
                    if np.random.rand() <= alpha_0:
                        X = Z

                gamma = 0.5 / ((1.0 + i / 50.0) ** 0.65)
                log_r0 = float(np.clip(log_r0 + gamma * (alpha_0 - target_accept), np.log(1e-4), 0.0))

            self._r0 = float(np.clip(np.exp(log_r0), 1e-4, 1.0))
            self._X = X

        if reset_chain:
            self._chain = saved_chain
            self._step_count = saved_step_count
            self._n_samples = saved_n_samples
            self._n_accepted = saved_n_accepted
        return self._r0

    def sample(self, n_samples, divs=1, visual=False, safe=False):
        """
    Sample
        Sampling
    Inputs  :
        n_samples :
            number of samples to generate
    Optional Inputs :
        divs : (1)
            number of divisions
        visual :
            show progress
        safe :
            save the chain at every division
        """
        n_samples = int(n_samples)
        divs = int(divs)
        if visual:
            print("Sampling: 0%")
        chunk = n_samples // divs
        for i in range(divs):
            if chunk > 0:
                self._sample(chunk)
            if visual:
                sys.stdout.write("\033[F")  # cursor up
                print("Sampling: " + str(int((i + 1) * 100.0 / divs)) + "%")
            if safe:
                self.save(path="chain_{:}.dat".format(i))
        rem = n_samples % divs
        if rem != 0:
            self._sample(rem)
            if safe:
                self.save(path="chain_{:}.dat".format(divs))

    def vsample(self, n_samples, divs=100, safe=False):
        """Convenience wrapper for visual sampling (used in time_series examples)."""
        return self.sample(n_samples, divs=divs, visual=True, safe=safe)

    def save(self, path="chain.dat"):
        """
    Save
        Save data to file
    Inputs  :
        path :
            specifies the path name of the file to be saved to
        """
        dic = {
            'chain': self._chain.tolist() if self._chain is not None else [],
            'step_count': self._step_count.tolist() if self._step_count is not None else [],
            'n_samples': int(self._n_samples),
            'n_accepted': int(self._n_accepted),
            'x': self._X['x'].tolist(),
            'f': self._X['f'].tolist(),
            'J': self._X['J'].tolist(),
        }
        with open(path, 'w') as file:
            json.dump(dic, file)

    def load(self, path="chain.dat"):
        """
    Load
        Load data from file
    Inputs  :
        path :
            specifies the path name of the file to be loaded from
        """
        with open(path, 'r') as file:
            dic = json.load(file)

        self._chain = np.array(dic['chain'], dtype=float)
        self._step_count = np.array(dic['step_count'], dtype=float)
        self._n_samples = int(dic['n_samples'])
        self._n_accepted = int(dic['n_accepted'])
        self._X = {
            'x': np.reshape(np.array(dic['x'], dtype=float), (-1, 1)),
            'f': np.reshape(np.array(dic['f'], dtype=float), (-1, 1)),
            'J': np.array(dic['J'], dtype=float),
        }

    def burn(self, n_burned):
        """
    Burn
        Burn the initial samples to adjust for convergence of the chain
        cut the first (n_burned) burn-in samples
    Inputs  :
        n_burned :
            number of samples to cut
        """
        self._chain = self._chain[int(n_burned):]
        return self._chain

    burn_in = burn

    def acor(self, k=5):
        """
    Autocorrelation time of the chain
        return the autocorrelation time for each parameter
    Inputs :
        k :
            parameter in self-consistent window
    Outputs :
        t :
            autocorrelation time of the chain
        """
        try:
            import acor as _acor_mod
            n = np.shape(self._chain)[1]
            t = np.zeros(n)
            for i in range(n):
                t[i] = _acor_mod.acor(self._chain[:, i], k)[0]
            return t
        except ImportError:
            return autocorr_time(self._chain, k=k)

    def posterior(self, x):
        """
    Posterior density
        ** not normalized **
        This is used to plot the theoretical curve for tests.
    Inputs  :
        x :
            input value
    Outputs :
        p : p(x)=pi(x)*exp{-||f(x)||^2/(2)}
            posterior probability of x
        """
        x = np.reshape(np.array(x, dtype=float), (-1, 1))

        chi_x, f_x, _ = self._f(x)
        if chi_x:
            p = float(np.exp(-0.5 * la.norm(f_x) ** 2))
            if self._prior:
                m = self._m
                H = self._H
                quad = float(np.dot((x - m).T, np.dot(H, x - m)).reshape(-1)[0])
                p = p * float(np.exp(-0.5 * quad))
            return p
        return 0.0

    def error_bars(self, n_bins, d_min, d_max):
        """
    Error Bars
        create bars and error bars to plot
    Inputs  :
        n_bins :
            number of bins
        d_min, d_max :
            domain min and max vectors of length n_dims
    Outputs :
        x     :
            domain centers of shape (n_dims, n_bins)
        p_x   :
            estimated posterior using the chain on the domain
        error :
            estimated error for p_x
        """
        chain = self._chain
        len_chain = len(chain)
        n_dims = np.shape(chain)[1] if chain.ndim > 1 else 1

        if n_bins != int(n_bins):
            raise TypeError("number of bins has to be an integer")
        n_bins = int(n_bins)

        d_min = np.reshape(np.array(d_min, dtype=float), (-1, 1))
        d_max = np.reshape(np.array(d_max, dtype=float), (-1, 1))
        if np.size(d_min) != n_dims:
            raise TypeError("domain minimum has wrong size")
        if np.size(d_max) != n_dims:
            raise TypeError("domain maximum has wrong size")

        v = d_max - d_min
        v_2 = float(np.dot(v.T, v).reshape(-1)[0])
        if v_2 <= 0.0:
            raise ValueError("d_min and d_max must not be identical")

        # Vectorized projection onto the line segment from d_min to d_max
        proj = np.dot(chain - d_min[:, 0], v[:, 0]) / v_2 * n_bins
        bin_indices = np.floor(proj).astype(int)
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        p_x = np.bincount(bin_indices[valid], minlength=n_bins).astype(float)

        dx = np.sqrt(v_2) / n_bins
        p_x = p_x / (len_chain * dx)

        error = np.sqrt(np.maximum(0.0, p_x * (1.0 / dx - p_x)) / len_chain)
        grid_frac = (0.5 + np.arange(n_bins)) / n_bins
        x = d_min + v * grid_frac[None, :]
        return x, p_x, error

    # internal methods
    def _sample(self, n_samples, max_domain_tries=200):
        """
    Sample
        Generate samples for posterior distribution using Gauss-Newton
        proposal parameters
        """
        try:
            n_samples = int(n_samples)
        except Exception:
            raise TypeError("number of samples has to be an integer")

        X = self._proposal_params(self._X)
        k_max = self._max_steps
        n = self._n

        chain = np.zeros((n_samples, n), dtype=np.float64)
        n_accepted = 0
        step_count = np.zeros(k_max + 2, dtype=np.float64)

        if HAS_NUMBA:
            max_stages = k_max + 2
            X_arr = np.empty((max_stages, n, 1), dtype=np.float64)
            MU_arr = np.empty((max_stages, n, 1), dtype=np.float64)
            L_arr = np.empty((max_stages, n, n), dtype=np.float64)
            LOG_DET_arr = np.empty(max_stages, dtype=np.float64)
            LOG_P_arr = np.empty(max_stages, dtype=np.float64)
            F_SQ_arr = np.empty(max_stages, dtype=np.float64)
            r_arr = np.empty(max_stages, dtype=np.float64)

            X_arr[0] = X['x']
            MU_arr[0] = X['mu']
            L_arr[0] = X['L']
            LOG_DET_arr[0] = X['log_det_L']
            LOG_P_arr[0] = X['log_p']
            F_SQ_arr[0] = X['f_sq']
            L_inv_T_curr = X['L_inv_T']
            f0 = X['f']
            J0 = X['J']

            has_prior = self._prior
            m_prior = self._m
            H_prior = self._H
            Hm_prior = self._Hm
            ln_H_prior = self._ln_H_
            reg_val = self._reg
            r0_val = float(self._r0)
            dynamic_flag = self._dynamic
            step_size_val = float(self._step_size)
            t1_val = float(self._opts.get('t1', 0.05))
            t2_val = float(self._opts.get('t2', 0.5))
            f_eval = self._f

            # Pre-generate standard normal and uniform draws in chunks of 4096
            chunk_size = min(n_samples, 4096)
            z_batch = np.random.standard_normal((chunk_size, k_max + 1, n, 1))
            u_batch = np.random.rand(chunk_size, k_max + 1)
            batch_offset = 0

            for i in range(n_samples):
                idx = i - batch_offset
                if idx >= chunk_size:
                    batch_offset = i
                    rem_samples = min(n_samples - i, 4096)
                    z_batch = np.random.standard_normal((rem_samples, k_max + 1, n, 1))
                    u_batch = np.random.rand(rem_samples, k_max + 1)
                    chunk_size = rem_samples
                    idx = 0

                accepted = False
                r_arr[0] = r0_val
                log_P_z_x = LOG_P_arr[0]
                x_curr = X_arr[0]
                mu_curr = MU_arr[0]

                k = 0
                while k <= k_max:
                    chi_z = False
                    r_k = r_arr[k]
                    for attempt in range(max_domain_tries):
                        z_std = z_batch[idx, k] if attempt == 0 else np.random.standard_normal((n, 1))
                        z = njit_propose(x_curr, mu_curr, L_inv_T_curr, r_k, z_std)
                        chi_z, f_z, J_z = f_eval(z)
                        if chi_z:
                            break
                    if not chi_z:
                        break

                    accepted, L_inv_T_z, log_P_z_x, _ = njit_stage_step(
                        k,
                        k_max,
                        z,
                        f_z,
                        J_z,
                        has_prior,
                        m_prior,
                        H_prior,
                        Hm_prior,
                        ln_H_prior,
                        reg_val,
                        log_P_z_x,
                        X_arr,
                        MU_arr,
                        L_arr,
                        LOG_DET_arr,
                        LOG_P_arr,
                        F_SQ_arr,
                        f0,
                        J0,
                        r_arr,
                        dynamic_flag,
                        step_size_val,
                        t1_val,
                        t2_val,
                        u_batch[idx, k],
                    )
                    if accepted:
                        L_inv_T_curr = L_inv_T_z
                        f0 = f_z
                        J0 = J_z
                        break
                    k += 1

                if accepted:
                    chain[i, :] = X_arr[0, :, 0]
                    n_accepted += 1
                    step_count[k + 1] += 1
                else:
                    chain[i, :] = X_arr[0, :, 0]
                    step_count[0] += 1

            self._X = {
                'x': X_arr[0].copy(),
                'f': f0,
                'J': J0,
                'L': L_arr[0].copy(),
                'L_inv_T': L_inv_T_curr,
                'log_det_L': float(LOG_DET_arr[0]),
                'mu': MU_arr[0].copy(),
                'log_p': float(LOG_P_arr[0]),
                'f_sq': float(F_SQ_arr[0]),
            }
        else:
            for i in range(n_samples):
                accepted = False
                r_ = [float(self._r0)]
                Z_ = [X]
                self._r_ = r_
                self._log_P_cache = {}
                log_P_z_x = float(X['log_p'])

                k = 0
                while k <= k_max:
                    chi_z = False
                    for _ in range(max_domain_tries):
                        z = multi_normal(X, r_[-1])
                        chi_z, f_z, J_z = self._f(z)
                        if chi_z:
                            break
                    if not chi_z:
                        break

                    Z = self._proposal_params({'x': z, 'f': f_z, 'J': J_z})
                    Z_.append(Z)
                    self._Z_ = Z_

                    log_P_z_x += log_K(Z, X, r_[-1])

                    self._N_is_0 = False
                    log_N = self._log_P(X, Z, k)

                    if self._N_is_0 or not math.isfinite(log_N):
                        A_z_x = 0.0
                    elif log_N >= log_P_z_x:
                        A_z_x = 1.0
                    else:
                        A_z_x = math.exp(log_N - log_P_z_x)

                    if np.random.rand() <= A_z_x:
                        accepted = True
                        break
                    else:
                        log_P_z_x += math.log1p(-min(A_z_x, 1.0 - 1e-16))
                        self._back_off()
                        k += 1

                if accepted:
                    chain[i, :] = z[:, 0]
                    X = Z
                    n_accepted += 1
                    step_count[k + 1] += 1
                else:
                    chain[i, :] = X['x'][:, 0]
                    step_count[0] += 1

            self._X = X

        if self._n_samples == 0 or self._chain is None:
            self._chain = chain
            self._step_count = step_count
        else:
            self._chain = np.concatenate((self._chain, chain), axis=0)
            self._step_count = np.add(self._step_count, step_count)
        self._n_samples += n_samples
        self._n_accepted += n_accepted

    def _proposal_params(self, state):
        """
    Proposal parameters
        Calculate parameters needed for the proposal.
        """
        if HAS_NUMBA:
            x = np.ascontiguousarray(state['x'], dtype=np.float64)
            f = np.ascontiguousarray(state['f'], dtype=np.float64)
            J = np.ascontiguousarray(state['J'], dtype=np.float64)
            L, L_inv_T, log_det_L, mu = njit_cholesky_and_solve(
                J, f, x, self._prior, self._H, self._Hm, self._reg, 1e-2
            )
            log_p, f_sq = njit_log_post(
                x, f, self._prior, self._m, self._H, self._ln_H_
            )
            state['x'] = x
            state['f'] = f
            state['J'] = J
            state['L'] = L
            state['L_inv_T'] = L_inv_T
            state['log_det_L'] = float(log_det_L)
            state['mu'] = mu
            state['log_p'] = float(log_p)
            state['f_sq'] = float(f_sq)
            return state

        x = state['x']
        f = state['f']
        J = state['J']
        n = self._n
        JJ = np.dot(J.T, J)
        if self._reg > 0.0:
            JJ = JJ + self._reg * np.eye(n)

        if self._prior:
            L = _cholesky_regularized(self._H + JJ)
            rhs = self._Hm - np.dot(J.T, f) + np.dot(JJ, x)
        else:
            L = _cholesky_regularized(JJ)
            rhs = np.dot(J.T, f)

        if n == 1:
            l00 = float(L[0, 0])
            inv00 = 1.0 / l00
            L_inv_T = np.array([[inv00]], dtype=float)
            step = rhs * (inv00 * inv00)
            log_det_L = math.log(abs(l00))
        elif n == 2:
            l00 = float(L[0, 0])
            l10 = float(L[1, 0])
            l11 = float(L[1, 1])
            inv00 = 1.0 / l00
            inv11 = 1.0 / l11
            inv01 = -l10 * inv00 * inv11
            L_inv_T = np.array([[inv00, inv01], [0.0, inv11]], dtype=float)
            y0 = rhs[0, 0] * inv00
            y1 = (rhs[1, 0] - l10 * y0) * inv11
            step = np.array([[inv00 * y0 + inv01 * y1], [inv11 * y1]], dtype=float)
            log_det_L = math.log(abs(l00) * abs(l11))
        else:
            if not hasattr(self, '_eye_n') or self._eye_n.shape[0] != n:
                self._eye_n = np.eye(n, dtype=float)
            aug = np.empty((n, n + 1), dtype=float)
            aug[:, 0:1] = rhs
            aug[:, 1:] = self._eye_n
            sol = _solve_lower(L, aug)
            y = sol[:, 0:1]
            L_inv_T = sol[:, 1:].T
            step = np.dot(L_inv_T, y)
            log_det_L = float(np.sum(np.log(np.abs(L.diagonal()))))

        mu = step if self._prior else (x - step)

        state['L'] = L
        state['L_inv_T'] = L_inv_T
        state['log_det_L'] = log_det_L
        state['mu'] = mu
        state['log_p'] = self._log_post(x, f)
        return state

    def _log_P(self, X, Z, k):
        """
    Log of the probability of transition from z to x with k steps
        log ( P_k (x, z) )
        Computed in log-space using log1p(-exp(...)) to prevent float64 underflow.
        """
        r_ = self._r_
        if k == 0:
            return float(Z['log_p']) + log_K(X, Z, r_[0])

        cache = getattr(self, '_log_P_cache', None)
        key = (id(Z), k)
        if cache is not None and key in cache:
            log_diff = cache[key]
            if log_diff == -math.inf:
                self._N_is_0 = True
                return -math.inf
            return log_diff + log_K(X, Z, r_[k])

        Z_ = self._Z_
        log_P_zk_z = self._log_P(Z_[k], Z, k - 1)
        log_P_z_zk = self._log_P(Z, Z_[k], k - 1)

        if (not math.isfinite(log_P_zk_z)) or (log_P_zk_z <= log_P_z_zk):
            if cache is not None:
                cache[key] = -math.inf
            self._N_is_0 = True
            return -math.inf

        log_diff = log_P_zk_z + math.log1p(-math.exp(log_P_z_zk - log_P_zk_z))
        if cache is not None:
            cache[key] = log_diff
        return log_diff + log_K(X, Z, r_[k])

    def _back_off(self):
        """
    Back off
        Calculate the back off step size
        """
        q = self._step_size
        r = self._r_[-1]
        Z_ = self._Z_
        if self._dynamic:
            x0 = Z_[0]['x']
            f0 = Z_[0]['f']
            J0 = Z_[0]['J']
            xr = Z_[-1]['x']
            fr = Z_[-1]['f']
            Jr = Z_[-1]['J']

            f0_flat = f0.ravel()
            fr_flat = fr.ravel()
            p_0_sq = float(np.dot(f0_flat, f0_flat))
            p_r_sq = float(np.dot(fr_flat, fr_flat))

            # Directional derivative along the step vector v = (xr - x0) / r
            v = (xr - x0) / r
            dp_0 = 2.0 * float(np.dot(f0_flat, np.dot(J0, v).ravel()))
            dp_r = 2.0 * float(np.dot(fr_flat, np.dot(Jr, v).ravel()))

            t1 = self._opts.get('t1', 0.05)
            t2 = self._opts.get('t2', 0.5)
            r_new = optimize(r, p_0_sq, dp_0, p_r_sq, dp_r, t1=t1, t2=t2)
        else:
            r_new = r * q
        self._r_.append(r_new)

    def _log_post(self, x, f_x):
        """
    Log of the posterior density
        This is used to calculate acceptance probability for sampling.
        """
        f_flat = f_x.ravel()
        log_likelihood = -0.5 * float(np.dot(f_flat, f_flat))

        if self._prior:
            diff = x - self._m
            quad = float(np.dot(diff.ravel(), np.dot(self._H, diff).ravel()))
            return self._ln_H_ - 0.5 * quad + log_likelihood
        return log_likelihood

    @property
    def chain(self):
        return self._chain

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_accepted(self):
        return self._n_accepted

    @property
    def accept_rate(self):
        return float(self._n_accepted) / self._n_samples if self._n_samples > 0 else 0.0

    @property
    def step_count(self):
        return self._step_count

    @property
    def call_count(self):
        return self._f.count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def step_size(self):
        return self._step_size

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, value):
        if isinstance(value, str) and value.lower() == "auto":
            self._r0 = min(1.0, 2.0 / max(1, self._n))
        else:
            val = float(value)
            if not (0.0 < val <= 1.0):
                raise ValueError("r0 must be in (0, 1] or 'auto'")
            self._r0 = val


Sampler = sampler
