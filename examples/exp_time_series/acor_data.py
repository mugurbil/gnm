# -*- coding: utf-8 -*-

import json
import os
from optparse import OptionParser
import numpy as np
import gnm  # noqa: F401
from model import funky

la = np.linalg

'''
Generating data for the problem.
'''
print()
print('Auto-correlation time: data')

# command line options to set parameters
parser = OptionParser()
parser.add_option('-c', dest='count', type='int',
                  default=1, help='count of experiment')
parser.add_option('-N', dest='N', type='int',
                  default=2, help='upper limit of the sum of acor')
parser.add_option('-s', dest='seed', type='int',
                  default=5, help='random number generator seed')
parser.add_option('-T', dest='t_max', type='int',
                  default=2, help='upper bound for time')
parser.add_option('-n', dest='t_num', type='int',
                  default=20, help='number of time steps')
parser.add_option('--Jtest', action='store_true', dest='Jtest',
                  default=False, help='test the derivative')

(options, _args) = parser.parse_args()
N = options.N
np.random.seed(options.seed)

f = funky
t_args = np.arange(options.t_num) * options.t_max / float(options.t_num)

print('Creating data...')
sigma = 0.1
x = np.zeros(2 * N)
for i in range(N):
    x[i] = i + 2.0
    x[i + N] = (i + 2) ** 2

_, y, J = f(x, t_args)
for i in range(y.size):
    y[i] = y[i] * (1.0 + np.random.randn() * sigma)
print()

m = [1.1, 5.2, 3.1, 5.1]
H = [[1.20, 0.20, 0.25, 0.23],
     [0.20, 1.64, 0.02, 0.11],
     [0.25, 0.02, 0.93, 0.13],
     [0.23, 0.11, 0.13, 1.01]]

prior_info = {
    'N': N,
    'args': t_args.tolist(),
    's': sigma,
    'y': y.tolist(),
    'm': m,
    'H': H,
}

folder = 'acor_data_%d/' % options.count
if os.path.exists(folder):
    write = str(input("Warning: directory exists. Overwrite? (y,n) "))
    while write not in ('y', 'n'):
        write = str(input('Please type y or n: '))
else:
    os.mkdir(folder)
    write = 'y'
if write == 'y':
    print('Writing to file...')
    path = os.path.join(folder, 'data')
    with open(path, 'w') as prior_file:
        json.dump(prior_info, prior_file)
    print('Done!\n')
elif write == 'n':
    print('Terminated.\n')
