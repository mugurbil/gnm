# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
from math import exp,pi
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
print 'Auto-corrolation time: sampling'

# command line options to set parameters
parser = OptionParser()
# experiment number
parser.add_option('-c', dest='count', type='int',
				default=0, help='count of experiment')
# seeding
parser.add_option('-s', dest='seed', type='int', 
				default=5, help='random number generator seed')
# for the sampler
parser.add_option('-n', dest='num_samples', type='int', 
				default=10000, help='number of samples')

parser.add_option('-b', dest='num_burn', type='int', 
				default=1000, help='number of samples burned')

parser.add_option('-m', dest='max_steps', type='int', 
				default=4, help='max back off steps')

parser.add_option('-z', dest='step_size', type='float', 
				default=0.1, help='step size of back off')

(opts, arg) = parser.parse_args()
# seed the random number generator
np.random.seed(opts.seed)

# get the data
try: 
	print 'Importing Data...\n'
	folder = 'acor_data_%d/' % opts.count
	path = os.path.join(folder, 'data')
	data_file = open(path, 'r')
	data = json.load(data_file)
	data_file.close()
	args = data['args']
	m = data['m']
	H = data['H']
	sigma = data['s']
	y = data['y']
except: 
	print "Data could not be imported."
	exit(0)

# make function instance
from acor_func import funky
f = gnm.F(funky,args)

# creating sampler object
sampler = gnm.sampler(m,H,y,sigma,f)

# sample the likelihood
print 'Sampling {:.2e} points...'.format(opts.num_samples)
start_time  = time.time()
chain,stats = sampler.sample(m,opts.num_samples,
								max_steps=opts.max_steps,
								step_size=opts.step_size)
chain       = chain[opts.num_burn:]
end_time    = time.time()
T 			= end_time-start_time
print 'Acceptence Percentage : {:.3}'.format(stats['accept_rate'])
print 'Ellapsed Time         : %d h %d m %d s' % (T/3600,T/60%60,T%60)
print 

# write data to file
path = os.path.join(folder, 'chain')
file = open(path, 'w')
json.dump(chain.tolist(), file)
file.close()

path = os.path.join(folder, 'stats')
file = open(path, 'w')
json.dump(stats, file)
file.close()



