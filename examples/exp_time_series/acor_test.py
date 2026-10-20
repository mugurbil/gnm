# -*- coding: utf-8 -*-

import json
import os
import time
from optparse import OptionParser
import numpy as np
from scipy import integrate
import gnm
from model import funky

'''
Calculating the auto-correlation time which is an ill-posed problem.
Generating the theoretical (quadrature) curve for the auto-correlation.
'''
print('\nAuto-correlation time: test\n')

parser = OptionParser()
parser.add_option('-c', dest='count', type='int',
                  default=0, help='count of experiment')
parser.add_option('--quad', action='store_true', dest='quad',
                  default=False, help='do quadrature')
parser.add_option('--rint', action='store_true', dest='rint',
                  default=False, help='recursive quadrature (slow)')
parser.add_option('-k', dest='k', type='int',
                  default=0, help='quadrature dimension')
parser.add_option('-g', dest='num_grid_pts', type='int',
                  default=100, help='number of grid points')
parser.add_option('-l', dest='xmin', type='float',
                  default=-1.0, help='lower bound of range')
parser.add_option('-u', dest='xmax', type='float',
                  default=3.0, help='upper bound of range')
parser.add_option('--bar', action='store_true', dest='pbar',
                  default=False, help='quadrature progress bar')

(options, _args) = parser.parse_args()
xmin = options.xmin
xmax = options.xmax
num_grid_pts = options.num_grid_pts

plot_range = [[-2.0, 2.0], [2.0, 7.0], [0.0, 7.0], [4.0, 9.0]]

try:
    print('Importing Data...\n')
    folder = 'acor_data_%d/' % options.count
    path = os.path.join(folder, 'data')
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    N = data['N']
    m = data['m']
    H = data['H']
    sigma = data['s']
    y = np.asarray(data['y'])
except Exception:
    print("Data could not be found.")
    raise SystemExit(0)


def model_fn(x, args):
    chi, f_val, J_val = funky(x, args['t'])
    return chi, (f_val - args['y']) / args['s'], J_val / args['s']


sampler = gnm.sampler(m, model_fn, {'t': data['args'], 'y': y, 's': sigma})
sampler.prior(m, H)


def func(x0, x1, x2, x3, x4, x5):
    x = [x0, x1, x2, x3, x4, x5][:2 * N]
    return sampler.posterior(x)


def integrand(x, k):
    if N == 1:
        if k == 0:
            return lambda y0: func(x, y0, 0, 0, 0, 0)
        return lambda y0: func(y0, x, 0, 0, 0, 0)
    if N == 2:
        if k == 0:
            return lambda a, b, c: func(x, a, b, c, 0, 0)
        elif k == 1:
            return lambda a, b, c: func(a, x, b, c, 0, 0)
        elif k == 2:
            return lambda a, b, c: func(a, b, x, c, 0, 0)
        return lambda a, b, c: func(a, b, c, x, 0, 0)
    if N == 3:
        if k == 0:
            return lambda a, b, c, d, e: func(x, a, b, c, d, e)
        elif k == 1:
            return lambda a, b, c, d, e: func(a, x, b, c, d, e)
        elif k == 2:
            return lambda a, b, c, d, e: func(a, b, x, c, d, e)
        elif k == 3:
            return lambda a, b, c, d, e: func(a, b, c, x, d, e)
        elif k == 4:
            return lambda a, b, c, d, e: func(a, b, c, d, x, e)
        return lambda a, b, c, d, e: func(a, b, c, d, e, x)


if options.quad:
    if N > 3:
        print('quad works for N<=3, try rint or update the code')
        raise SystemExit(0)
    k = options.k
    print('Dimension %d of %d' % ((k + 1), (2 * N)))
    print('Doing Quadrature...')

    xmin = plot_range[k][0]
    xmax = plot_range[k][1]
    p_range = [plot_range[i] for i in range(2 * N) if i != k]
    start_time = time.time()
    integral_vec = np.empty([num_grid_pts + 1])
    dx = (xmax - xmin) / num_grid_pts

    def opts0(*_a, **_kw):
        return {'epsabs': 0, 'limit': 1}

    for i in range(num_grid_pts + 1):
        xnow = xmin + i * dx
        integral_vec[i], _ = integrate.nquad(
            integrand(xnow, k), p_range, opts=[opts0 for _ in range(2 * N - 1)]
        )

    end_time = time.time()
    T = end_time - start_time
    print('Elapsed Time : %d h %d m %d s' % (T / 3600, T / 60 % 60, T % 60))
    print()

    print('Calculating normalization...')
    normalization = np.average(integral_vec) * (xmax - xmin)
    integral_vec = integral_vec / normalization
    print('Normalization (Z) : {:.3}'.format(normalization))
    print()

    path = os.path.join(folder, 'curve' + str(k))
    with open(path, 'w') as file:
        json.dump(integral_vec.tolist(), file)
