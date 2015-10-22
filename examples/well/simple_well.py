#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple example with 1D Well
print("----------------------------\n"+
	  "-------Simple 1D Well-------\n"+
	  "----------------------------")

import numpy as np
import gnm
import time
import matplotlib.pyplot as plt

# random seeding
np.random.seed(1)
# arguments for the user function
args = {'a':1.}
# domain
d_min = -3.
d_max = 3.
# user-defined prior mean and precision 
m = [0.]   # vector
H = [[1.]] # matrix
# observed data and error
y = [1.]   # vector
sigma = 1. # float
# initial guess
x_0 = m
# back-off info
max_steps = 5
step_size = 0.2
fancy = False
# sampling
n_samples = 1.1*10**5
n_burn = 10**4
# plotting 
n_grid = 100

# function 
def mod_f(x, args):
    a = args['a']
    return 1, [a*x[0]**2], [[a*2.*x[0]]]

# gnm function instance
f = gnm.func(mod_f, args)
print("\nTesting Jacobian...")
converged, error = f.Jtest(d_min, d_max)
if converged :
    print('Converged!')
print("Error: {:.3e}".format(error))

# sampler object
gnm_sampler = gnm.sampler(m, H, y, sigma, f)
# gnm_sampler.set(x_0, max_steps=max_steps, step_size=step_size)
# when sampler is not set, initial guess is the mean of prior

# start sampling
print("\nSampling...")
start_time = time.time()
gnm_sampler.sample(n_samples)
end_time = time.time()

# burn the initial 
gnm_sampler.burn(n_burn)

# print results
print("Acceptence Rate : {:.3f}".format(gnm_sampler.accept_rate))
print("Ellapsed Time   : {:.1f}s".format(end_time - start_time))
print("Number Sampled  : {:.1e}".format(n_samples))
print("Number Burned   : {:.1e}".format(n_burn))
print("Number Used     : {:.1e}".format(n_samples - n_burn))

# create plot info 
x, p_x, err = gnm_sampler.error_bars(n_grid, [[d_min,d_max]])

# create theoretical plot
#   # initialize curve
curve = np.zeros(n_grid)
cnorm = 0.

#   # create theoretical curve
for i in xrange(n_grid) :
    curve[i] = gnm_sampler.posterior(x[0][i])
    cnorm += curve[i]

#   # normalize curve
curve = curve/cnorm/6.*n_grid

# plotting
plt.plot(x[0], curve, color = 'k', linewidth = 2, label="Theoretical")
plt.plot(x[0], p_x[0], color = 'b', label="Sampled", linewidth=0)    
plt.errorbar(x[0], p_x[0], yerr = err[0], fmt = 'b.') 
plt.legend(loc ="lower center") 
plt.grid(True)
# title for plot
title = ("Simple Well: $\\pi(x|y=%d)=exp(\\frac{-x^2}{2}"
	     "+\\frac{-(x^2-%d)^2}{2})$" % (y[0],y[0]) )
plt.title(title)
plt.xlabel("x")
plt.ylabel("Probability")
plt.show() 
print 

