# -*- coding: utf-8 -*-

'''
Plot the results
'''

import json
import os
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import gnm

parser = OptionParser()
parser.add_option('-c', dest='count', type='int',
                  default=0, help='count of experiment')
parser.add_option('-n', dest='num_bins', type='int',
                  default=100, help='number of bins')
parser.add_option('-l', dest='xmin', type='float',
                  default=-5.0, help='lower bound of range')
parser.add_option('-u', dest='xmax', type='float',
                  default=5.0, help='upper bound of range')
parser.add_option('-k', dest='k', type='int',
                  default=None, help='the dimension of plot')
parser.add_option('--all', action='store_true', dest='all',
                  default=False, help='plot all dimensions')
parser.add_option('--theory', action='store_true', dest='theo',
                  default=False, help='theoretical curve')
parser.add_option('--two', action='store_true', dest='two',
                  default=False, help='2d histogram')

(opts, _args) = parser.parse_args()

xmin = opts.xmin
xmax = opts.xmax
plot_range = [xmin, xmax]
num_bins = opts.num_bins

folder = 'acor_data_%d/' % opts.count
path = os.path.join(folder, 'chain')
with open(path, 'r') as file:
    chain = np.array(json.load(file))
n = len(chain)
N = len(chain[0]) // 2

path = os.path.join(folder, 'stats')
with open(path, 'r') as file:
    stats = json.load(file)

tau = gnm.autocorr_time(chain, k=5)
print(tau)

if opts.all:
    k_list = list(range(2 * N))
elif opts.k is not None:
    k_list = [opts.k]
else:
    k_list = []

for k in k_list:
    plt.hist(
        chain[:, k],
        num_bins,
        color='b',
        range=plot_range,
        density=True,
        label='sampled',
        alpha=0.3,
    )

    if opts.theo:
        path = os.path.join(folder, 'curve' + str(k))
        with open(path, 'r') as data_file:
            theoCurve = json.load(data_file)
        x_space = np.linspace(xmin, xmax, num=len(theoCurve))
        plt.plot(x_space, theoCurve, color='r', linewidth=1, label='quadrature')

    plt.title('Histogram (n={:.0e}, P(A)={:.3})'.format(n, stats['accept_rate']))
    if k < N:
        x_label = '$w_%d$' % k
    else:
        x_label = '$\\lambda_%d$' % (k - N)
    plt.xlabel(x_label)
    plt.ylabel('Probability')
    plt.legend()
    path = os.path.join(folder, 'acor_' + x_label.strip('$') + '.pdf')
    plt.savefig(path, dpi=500)
    plt.clf()

if opts.two:
    for k in range(N):
        plt.hist2d(chain[:, k], chain[:, k + N], bins=2 * num_bins, density=True)
        plt.colorbar()
        plt.title('2D Histogram (n={:.0e},P(A)={:.3})'.format(n, stats['accept_rate']))
        plt.xlabel('Weight ($w_%d$)' % k)
        plt.ylabel('Exponent ($\\lambda_%d$)' % k)
        path = os.path.join(folder, 'acor_2d_%d.pdf' % k)
        plt.savefig(path, dpi=500)
        plt.clf()
