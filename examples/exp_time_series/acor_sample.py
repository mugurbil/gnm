# -*- coding: utf-8 -*-

import json
import os
import time
from optparse import OptionParser
import numpy as np
import gnm
from model import funky

'''
Calculating the auto-correlation time which is an ill-posed problem.
Generating the theoretical (quadrature) curve for the auto-correlation.
'''
print()
print('Auto-correlation time: sampling')

parser = OptionParser()
parser.add_option('-c', dest='count', type='int',
                  default=0, help='count of experiment')
parser.add_option('-s', dest='seed', type='int',
                  default=5, help='random number generator seed')
parser.add_option('-n', dest='num_samples', type='int',
                  default=10000, help='number of samples')
parser.add_option('-b', dest='num_burn', type='int',
                  default=1000, help='number of samples burned')
parser.add_option('-m', dest='max_steps', type='int',
                  default=4, help='max back off steps')
parser.add_option('-z', dest='step_size', type='float',
                  default=0.1, help='step size of back off')

(opts, _arg) = parser.parse_args()
np.random.seed(opts.seed)

try:
    print('Importing Data...\n')
    folder = 'acor_data_%d/' % opts.count
    path = os.path.join(folder, 'data')
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    t_args = data['args']
    m = data['m']
    H = data['H']
    sigma = data['s']
    y = np.asarray(data['y'])
except Exception:
    print("Data could not be imported.")
    raise SystemExit(0)


def model_fn(x, args):
    chi, f_val, J_val = funky(x, args['t'])
    return chi, (f_val - args['y']) / args['s'], J_val / args['s']


sampler = gnm.sampler(m, model_fn, {'t': t_args, 'y': y, 's': sigma})
sampler.prior(m, H)
sampler.static(opts.max_steps, opts.step_size)

print('Sampling {:.2e} points...'.format(opts.num_samples))
start_time = time.time()
sampler.sample(opts.num_samples)
sampler.burn(opts.num_burn)
end_time = time.time()
T = end_time - start_time
print('Acceptance Percentage : {:.3f}'.format(sampler.accept_rate))
print('Elapsed Time          : %d h %d m %d s' % (T / 3600, T / 60 % 60, T % 60))
print()

path = os.path.join(folder, 'chain')
with open(path, 'w') as file:
    json.dump(sampler.chain.tolist(), file)

path = os.path.join(folder, 'stats')
with open(path, 'w') as file:
    json.dump({'accept_rate': sampler.accept_rate}, file)
