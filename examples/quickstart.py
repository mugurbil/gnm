#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
__Quickstart__

'''
import numpy as np # for random seeding
import gnm
import matplotlib.pyplot as plt # for plotting

# random seeding
np.random.seed(1)

# define your model function 
def f(x, args):
    a = args['a']
    return 1, [a*x[0]**2], [[a*2.*x[0]]]

arguments_1 = {'a': 1.}

# create a gnm function instance of the function
f_1 = gnm.func(f, arguments_1)

# check the function is defined properly
x_min = -3. # domain minimum
x_max = 3.  # domain maximum
error = f_1.Jtest(x_min, x_max)
assert error == 0

# sampler object
m = [0.]   # prior mean - vector
H = [[1.]] # prior precision - matrix
# observed data and error
y = [1.]   # data - vector
sigma = 1. # data standart deviation - float
gnm_sampler = gnm.sampler(m, H, y, sigma, f_1)

# sample 
n_samples = 1.1 * 10 ** 4
gnm_sampler.sample(n_samples)

# burn the initial 
n_burn = 10 ** 3
gnm_sampler.burn(n_burn)

# print results
print("Acceptence Rate : {:.3f}".format(gnm_sampler.accept_rate))
print("Number Sampled  : {:.1e}".format(n_samples))
print("Number Burned   : {:.1e}".format(n_burn))
print("Number Used     : {:.1e}".format(n_samples - n_burn))

# plot
n_grid = 100
x, p_x, err = gnm_sampler.error_bars(n_grid, [[x_min,x_max]])
plt.plot(x[0], p_x[0], color = 'b', label="Sampled", linewidth=0)    
plt.errorbar(x[0], p_x[0], yerr = err[0], fmt = 'b.') 

# create theoretical plot
# ** Warning: not for practical use **
curve = np.zeros(n_grid) # initialize curve
cnorm = 0.
for i in xrange(n_grid) : # create theoretical curve
    curve[i] = gnm_sampler.posterior(x[0][i])
    cnorm += curve[i]
curve = curve/cnorm/6.*n_grid # normalize curve
plt.plot(x[0], curve, color = 'k', linewidth = 2, label="Theoretical")

# plot options
plt.legend(loc ="lower center") 
plt.grid(True)
title = ("Simple Well: $\\pi(x|y=%d)=exp(\\frac{-x^2}{2}"
	     "+\\frac{-(x^2-%d)^2}{2})$" % (y[0],y[0]) )
plt.title(title)
plt.xlabel("x")
plt.ylabel("Probability")
plt.show() 

