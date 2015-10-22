#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
la = np.linalg
import time
from optparse import OptionParser
import json
import os
import gnm

'''
Calculating the auto-corrolation time which is an ill-posed problem.
Generating the theoretical (quadrature) curve for the auto-corrolation.
'''
print
print 'Auto-corrolation time: data'

# command line options to set parameters
parser = OptionParser()
# experiment number
parser.add_option('-c', dest='count', type='int',
				default=1, help='count of experiment')
# note that N can be 1, 2 or 3 only for quad
parser.add_option('-N', dest='N', type='int', 
				default=2, help='upper limit of the sum of acor')
# seeding
parser.add_option('-s', dest='seed', type='int', 
				default=5, help='random number generator seed')
# arguments
parser.add_option('-T', dest='t_max', type='int', 
				default=2, help='upper bound for time')

parser.add_option('-n', dest='t_num', type='int', 
				default=20, help='number of time steps')
# Jtest
parser.add_option('--Jtest', action='store_true', dest='Jtest',
				default=False, help='test the derivative')

(options, args) = parser.parse_args()
# set the parameters
N = options.N
# seed the random number generator
np.random.seed(options.seed)

# make function instance
from acor_func import funky
args = np.array(range(options.t_num))*options.t_max/float(options.t_num)
f    = gnm.F(funky,args)

# test the jacobian
if options.Jtest:
	print 'Testing Jacobian...'
	x_max            = [2. for i in xrange(2*N) ]
	x_min            = [0. for i in xrange(2*N) ]
	converged, error = f.Jtest(x_min,x_max)
	if converged == 1 :
		print 'Converged!' 
	else : 
		print 'Did not converge, check jacobian :('
	print 'error : {:.3e}'.format(error)
	print

print 'Creating data...' 
sigma   = 0.1
# create the free parameters
x = np.zeros(2*N)
for i in xrange(N):
	x[i]   = (i+2.)
	x[i+N] = (i+2)**2
# create the data
_,y,J = f(x)
# add noise
for i in xrange(y.size):
	y[i] = y[i]*(1.+np.random.randn()*sigma)
print

# creating prior info
m = [1.1,5.2,3.1,5.1]
H = [[1.20,0.20,0.25,0.23],
	 [0.20,1.64,0.02,0.11],
 	 [0.25,0.02,0.93,0.13],
	 [0.23,0.11,0.13,1.01]]

prior_info = {}
prior_info['N'] = N
prior_info['args'] = args.tolist()
prior_info['s'] = sigma
prior_info['y'] = y.tolist()
prior_info['m'] = m
prior_info['H'] = H

folder = 'acor_data_%d/' % options.count
if os.path.exists(folder):
	write = str(raw_input("Warning: directory exists. Overwrite? (y,n) "))
	while not (write == 'y' or write == 'n') :
		write = str(raw_input('Please type y or n: '))
else :
	os.mkdir(folder)
	write = 'y'
if write == 'y' :
	print 'Writing to file...'
	path = os.path.join(folder, 'data')
	prior_file = open(path, 'w')
	json.dump(prior_info, prior_file)
	print 'Done!\n'
elif write == 'n' :
	print 'Terminated.\n'

