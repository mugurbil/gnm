1.1.0 : (2026-10-19)
++++++++++++++++++++
- Modernized for Python 3 (`pyproject.toml`, `range`, `print`, relative imports).
- Fixed numerical underflow in recursive delayed-rejection `_log_P` via log-space `log1p(-exp(...))`.
- Fixed directional derivative calculation in `_back_off` for dynamic cubic interpolation.
- Added regularized Cholesky decomposition (`_cholesky_regularized`) to handle rank-deficient/underdetermined Jacobians ($m < n$) and critical points.
- Added triangular solver support (`scipy.linalg.solve_triangular`) and overflow-safe log-determinant (`log_det_tril` / `slogdet`).
- Added built-in FFT integrated autocorrelation time estimator (`autocorr_time` / `sampler.acor`) with no external C dependency.
- Added flexible model signatures (supporting residual-only functions with automatic central finite-difference Jacobians) and `sampler.from_least_squares`.
- Fixed `load()` array deserialization and multidimensional `error_bars` projection.

1.0.0 : (2017-6-1)
+++++++++++++++++++
- Initial release.
