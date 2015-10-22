#!/usr/bin/env python
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
print '\nAuto-corrolation time: test\n'

# command line options to set parameters
parser = OptionParser()
# experiment number
parser.add_option('-c', dest='count', type='int',
				default=0, help='count of experiment')
# for quadrature
parser.add_option('--quad', action='store_true', dest='quad',
				default=False, help='do quadrature')
# for recursive integration
parser.add_option('--rint', action='store_true', dest='rint', 
				default=False, help='recursive quadrature (slow)')
# options for both
parser.add_option('-k', dest='k', type='int', 
				default=0, help='quadrature dimension')

parser.add_option('-g', dest='num_grid_pts', type='int', 
				default=100, help='number of grid points')

parser.add_option('-l', dest='xmin', type='float', 
				default=-1., help='lower bound of range')

parser.add_option('-u', dest='xmax', type='float', 
				default=3., help='upper bound of range')

parser.add_option('--bar', action='store_true', dest='pbar', 
				default=False, help='quadrature progress bar')

(options, args) = parser.parse_args()
# set the parameters for the quadrature
xmin          = options.xmin
xmax          = options.xmax
plot_range    = [xmin,xmax]
num_grid_pts  = options.num_grid_pts

plot_range = [[-2.,2.],[2.,7.],[0.,7.],[4.,9.]]

# get the data
try: 
	print 'Importing Data...\n'
	folder = 'acor_data_%d/' % options.count
	path = os.path.join(folder, 'data')
	data_file = open(path, 'r')
	data = json.load(data_file)
	data_file.close()
	N = data['N']
	m = data['m']
	H = data['H']
	sigma = data['s']
	y = data['y']
except: 
	print "Data could not be found."
	exit(0)

# make function instance
from acor_func import funky
f    = gnm.F(funky,data['args'])

# creating sampler object
sampler = gnm.sampler(m,H,y,sigma,f)

# to use the right format for scipy
def func(x0,x1,x2,x3,x4,x5):
	x = [x0,x1,x2,x3,x4,x5]
	x = x[:2*N]
	return	sampler.posterior(x)

def integrand(x):
	if N == 1:
		if   k == 0 :
			return lambda y : func(x,y,0,0,0,0)
		else : 
			return lambda y : func(y,x,0,0,0,0)
	if N == 2:
		if   k == 0:
			return lambda a,b,c : func(x,a,b,c,0,0)
		elif k == 1:
			return lambda a,b,c : func(a,x,b,c,0,0)		
		elif k == 2:
			return lambda a,b,c : func(a,b,x,c,0,0)
		else :
			return lambda a,b,c : func(a,b,c,x,0,0)
	if N == 3:
		if   k == 0:
			return lambda a,b,c,d,e : func(x,a,b,c,d,e)
		elif k == 1:
			return lambda a,b,c,d,e : func(a,x,b,c,d,e)		
		elif k == 2:
			return lambda a,b,c,d,e : func(a,b,x,c,d,e)
		elif k == 3:
			return lambda a,b,c,d,e : func(a,b,c,x,d,e)
		elif k == 4:
			return lambda a,b,c,d,e : func(a,b,c,d,x,e)
		else :
			return lambda a,b,c,d,e : func(a,b,c,d,e,x)
			
# do quadrature 
if options.quad:
	if N > 3 : 
		print 'quad works for N<=3, try rint or update the code'
		exit()
	k = options.k
	print 'Dimension %d of %d' % ((k+1),(2*N))
	print 'Doing Quadrature...'
	# initialize progress-bar
	if options.pbar :
		try: 
			from progressbar import *
		except: 
			options.pbar = False
			print 'Import Error: progressbar could not be found'
	if options.pbar :
		widgets = ['Integrating: ', Percentage(),Bar('>'), ' ']
		pbar    = ProgressBar(widgets=widgets)
		pbar.start()

	xmin = plot_range[k][0]
	xmax = plot_range[k][1]
	p_range = []
	for i in xrange(2*N):
		if i != k :
			p_range.append(plot_range[i])
	start_time   = time.time()
	integral_vec = np.empty([num_grid_pts+1])
	dx           = (xmax-xmin)/num_grid_pts

	def opts0(*args, **kwargs):
	    return {'epsabs':0, 'limit':1}

	for i in xrange(num_grid_pts+1):
		xnow              = xmin + i * dx
		integral_vec[i],_ = integrate.nquad(integrand(xnow), p_range,
	    					 opts=[opts0 for j in xrange(2*N-1)])
		if options.pbar :
			pbar.update(i*100./(num_grid_pts+1))

	end_time = time.time()
	T 		 = end_time-start_time
	if options.pbar :
		pbar.finish()
	print 'Ellapsed Time : %d h %d m %d s' % (T/3600,T/60%60,T%60)
	print 

	# normalize the vector
	print 'Calculating normalization...'
	normalization = np.average(integral_vec )*(xmax-xmin)
	integral_vec  = integral_vec / normalization
	print 'Normalization (Z) : {:.3}'.format(normalization)
	print

	# save the vector
	path = os.path.join(folder, 'curve'+str(k))
	file = open(path, 'w')
	json.dump(integral_vec.tolist(), file)
	file.close()

# recursive integrator
def rint(f, z, xmin, xmax, n, k):
	dx   = (xmax-xmin)/n
	x    = xmin
	rsum = 0.
	for i in xrange(n+1):
		y = z+[x]
		if k == 1 :
			r = f(y)
		else : 
			r = rint(f,y,xmin,xmax,n,k-1)
		rsum += r
		x    += dx
	return rsum*dx

# do recursive quadrature 
if options.rint:
	k = options.k
	print 'Dimension %d of %d' % ((k+1),(2*N))
	print 'Doing Recursive Quadrature...'
	# initialize progress-bar
	if options.pbar :
		try: 
			from progressbar import *
		except: 
			options.pbar = False
			print 'Import Error: progressbar could not be found'
	if options.pbar :
		widgets = ['Integrating: ', Percentage(),Bar('>'), ' ']
		pbar    = ProgressBar(widgets=widgets)
		pbar.start()

	start_time   = time.time()
	integral_vec = np.empty([num_grid_pts+1])
	dx           = (xmax-xmin)/num_grid_pts
	x            = xmin

	for i in xrange(num_grid_pts+1):
		integral_vec[i] = rint(sampler.posterior,[x],xmin-options.pad,
								xmax+options.pad,num_grid_pts,2*N-1)
		if options.pbar :
			pbar.update(i*100./(num_grid_pts+1))
		x += dx

	end_time = time.time()
	T 		 = end_time-start_time
	if options.pbar :
		pbar.finish()
	print 'Ellapsed Time : %d h %d m %d s' % (T/3600,T/60%60,T%60)
	print 

	# normalize the vector
	print 'Calculating normalization...'
	normalization = np.average(integral_vec )*(xmax-xmin)
	integral_vec  = integral_vec / normalization
	print 'Normalization (Z) : {:.3}'.format(normalization)
	print

	# save the vector
	path = os.path.join(folder, 'rint'+str(k))
	file = open(path, 'w')
	json.dump(integral_vec.tolist(), file)
	file.close()
