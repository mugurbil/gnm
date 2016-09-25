#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple example with 1D Well
print("-----------------------\n"+
      "--------1D Well--------\n"+
      "-----------------------")

import numpy as np
import gnm
import time
import json
import matplotlib.pyplot as plt
import acor

# command line prompts
from optparse import OptionParser
parser = OptionParser()
# random seed
parser.add_option('-r', dest='seed', type='int',
                default=2, help='random seed')
# initial guess
parser.add_option('--guess', dest='x_0', type='float',
                default=0., help='initial guess')
# domain info 
parser.add_option('--dmin', dest='d_min', type='float',
                default=-3., help='domain minimum')
parser.add_option('--dmax', dest='d_max', type='float',
                default=3., help='domain maximum')                
# prior info
parser.add_option('-m', dest='m', type='float',
                default=0, help='mean of prior')
parser.add_option('-H', dest='H', type='float',
                default=1, help='precision of prior')
# data info
parser.add_option('-y', dest='y', type='float',
                default=4, help='observed data point')
parser.add_option('-s', dest='s', type='float',
                default=1, help='observed standard deviation')
# back off info
parser.add_option('--max', dest='max', type=int,
                default=5, help='max number of steps for back-off')
parser.add_option('--size', dest='size', type=float,
                default=0.5, help='step size for back-off')
parser.add_option('--fancy', dest='fancy', action='store_true',
                default=False, help='turn on dynamic back-off')
# # visual bar
# parser.add_option('--bar', dest='pbar', action='store_false',
#                 default=True, help='toggle off progressbar')
# sampling info
parser.add_option('-n', dest='n_samples', type='int',
                default=1*10**6, help='number of samples')
parser.add_option('-b', dest='n_burn', type='int',
                default=10**3, help='number of samples to burn')
# plotting info
parser.add_option('-g', dest='n_grid', type='int',
                default=100, help='number of grid points')
(opts, args) = parser.parse_args()
# end command line

np.random.seed(opts.seed)

# user function 
def model(x, args):
    y = args['y']
    s = args['s']
    return 1, [(x[0]**2-y)/s], [[(2.*x[0])/s]]
# initial guess
x_0 = [opts.x_0]
# observed data and error
data = {'y':[opts.y], 's':opts.s}

# creating sampler object
jagger = gnm.sampler(x_0, model, data)

# setting prior mean and precision 
m = [opts.m]   # vector
H = [[opts.H]] # matrix
jagger.prior(m, H)

# sampler object
if opts.fancy:
    jagger.dynamic(opts.max)
else: 
    jagger.static(opts.max, opts.size)
start_time = time.time()
jagger.vsample(opts.n_samples)

jagger.burn(opts.n_burn)

end_time = time.time()
T = end_time - start_time # sample time
print("Ellapsed time            : %d h %d m %d s" % (T/3600,T/60%60,T%60))
print("Acceptence rate          : {:.3f}".format(jagger.accept_rate))
print("Auto-correlation time    : {:.2e}".format(jagger.acor()[0]))
print("Sample size              : {:.2e}".format(opts.n_samples - opts.n_burn))
print("Effective sample size    : {:.2e}".format((opts.n_samples - opts.n_burn)/jagger.acor()[0]))
print("Number of function calls : {:.2e}".format(jagger.call_count))

x, p_x, err = jagger.error_bars(opts.n_grid, [opts.d_min],[opts.d_max])

curve = np.zeros(opts.n_grid)
cnorm = 0.

for i in xrange(opts.n_grid) :
    curve[i] = jagger.posterior(x[0][i])
    cnorm += curve[i]

curve = curve/cnorm*opts.n_grid/(opts.d_max-opts.d_min)

if opts.max == 0 :
    sz = "no back-off"
else :  
    sz = "step_size=" + str(opts.size)

plt.plot(x[0], curve, color = 'k', linewidth = 2, label='Theoretical')
plt.plot(x[0], p_x[0], color = 'b', marker = 'o', linewidth = 0, 
	label='Sampled')    
plt.errorbar(x[0], p_x[0], yerr = err[0], fmt = 'b.') 
title = ("1D Well: $p(x|y=%d)=exp(\\frac{-x^2}{2}" % opts.y +
	     "+\\frac{-(x^2-%d)^2}{2})$" % opts.y )
plt.title(title)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend(loc ='upper center') 
plt.savefig('test%d.pdf' % opts.max, dpi = 500)   
plt.show() 

