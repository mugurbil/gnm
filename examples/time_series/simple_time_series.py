# -*- coding: utf-8 -*-

print("---------------------------\n" +
      "--Exponential Time Series--\n" +
      "---------------------------")

import numpy as np
la = np.linalg
import gnm
import time
import matplotlib.pyplot as plt

'''
Exponential time series is an ill-posed problem.
'''

# set the parameters
seed = 3
t_max = 10
x_max = [6.0, 5.0, 5.0, 5.0]
x_min = [-1.0, -2.0, -0.0, -1.0]
m = [12.0, 8.0, 0.5, 1.0]
H = np.identity(4) * 0.5
sigma = 0.1
n_burn = 10000
n_samples = 1000000 + n_burn
n_grid = 100
n_bins = 400

np.random.seed(seed)


def exp_time_series(x, t):
    n = x.size // 2
    f = np.zeros((t.size, 1))
    J = np.zeros((t.size, x.size))
    for j in range(t.size):
        for i in range(n):
            f[j] += x[i] * np.exp(-x[i + n] * t[j])
            J[j, i] = np.exp(-x[i + n] * t[j])
            J[j, i + n] = -x[i] * t[j] * np.exp(-x[i + n] * t[j])
    return f, J


def model(x, args):
    t = args['t']
    y = args['y']
    s = args['s']
    a, b = exp_time_series(x, t)
    return 1, (a - y) / s, b / s


# create the data with noise
x = np.array([1.0, 2.5, 0.5, 3.1])
t = np.array(range(t_max))
y, _ = exp_time_series(x, t)
for i in range(y.size):
    y[i] = y[i] * (1 + np.random.randn() * 0.1)
args = {'t': t, 'y': y, 's': sigma}

jagger = gnm.sampler(m, model, args)
jagger.prior(m, H)
jagger.dynamic(1)

# sample the likelihood
print("Sampling...")
start_time = time.time()
jagger.vsample(n_samples)
jagger.burn(n_burn)
end_time = time.time()
acor = la.norm(jagger.acor())
T = end_time - start_time
print("Acceptance Percentage    : {:.3}".format(jagger.accept_rate))
print("Elapsed Time             : %d h %d m %d s" % (T / 3600, T / 60 % 60, T % 60))
print("Auto-correlation time    : {:.2e}".format(acor))
print("Sample size              : {:.2e}".format(n_samples - n_burn))
print("Effective sample size    : {:.2e}".format((n_samples - n_burn) / acor))
print("Number of function calls : {:.2e}".format(jagger.call_count))
print()

# histogram of samples
print('Plotting Samples...')
plt.hist(
    jagger.chain[:, 0],
    n_bins,
    color='b',
    range=[x_min[0], x_max[0]],
    density=True,
    label='sampled',
    alpha=0.3,
)

# error bars
z, fhat, epsf = jagger.error_bars(n_bins, x_min, x_max)
plt.plot(z[0], fhat, color='b', marker='s', linewidth=0, alpha=0.0)
plt.errorbar(z[0], fhat, yerr=epsf, fmt='k.')

plt.title('Marginal Posterior for Exponential Time Series for $w_1$')
plt.xlabel('Weight ($w_1$)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.savefig('weight.pdf', dpi=500)
plt.clf()

print('Plotting Samples...')
plt.hist(
    jagger.chain[:, 1],
    n_bins,
    color='b',
    range=[x_min[1], x_max[1]],
    density=True,
    label='sampled',
    alpha=0.3,
)
plt.plot(z[1], fhat, color='b', marker='s', linewidth=0, alpha=0.0)
plt.errorbar(z[1], fhat, yerr=epsf, fmt='k.')

plt.title('Marginal Posterior for Exponential Time Series for $\\lambda_1$')
plt.xlabel('Exponent ($\\lambda_1$)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.savefig('exponent.pdf', dpi=500)
plt.clf()

plt.hist2d(jagger.chain[:, 0], jagger.chain[:, 2], bins=n_bins, density=True)
plt.colorbar()
plt.title('Exp Time Series Posterior Probability')
plt.xlabel('Weight ($w_1$)')
plt.ylabel('Exponent ($\\lambda_1$)')
plt.savefig('time_series_4d1.pdf', dpi=500)
plt.clf()

plt.hist2d(jagger.chain[:, 1], jagger.chain[:, 3], bins=n_bins, density=True)
plt.colorbar()
plt.title('Exp Time Series Posterior Probability 4D-2')
plt.xlabel('Weight ($w$)')
plt.ylabel('Exponent ($\\lambda$)')
plt.savefig('time_series_4d2.pdf', dpi=500)
plt.clf()

plt.plot(jagger.step_count / jagger.n_samples, 'ro')
plt.title('Step of Acceptance')
plt.xlabel('Step')
plt.ylabel('Percentage')
plt.savefig('step.pdf', dpi=500)
plt.clf()
