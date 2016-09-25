#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple example with 1D Well
print("---------------------------------\n"+
	  "-------Rosenbrock Function-------\n"+
	  "---------------------------------")

import numpy as np
from scipy import integrate
import gnm
import time
import matplotlib.pyplot as plt

# random seeding
np.random.seed(3)

# initial guess
x_0 = [0.1, 0.1]
# user function 
def model(x, args):
    a = args['a']
    b = args['b']
    z = (a-x[0])**2+b*(x[1]-x[0]**2)**2
    dx = -2*(a-x[0])+2*b*(x[1]-x[0]**2)*(-2*x[0])
    dy = 2*b*(x[1]-x[0]**2)
    return 1, [z], [[dx, dy]]

# observed data and error = arguments for the user function
args = {'a':1., 'b':1.}    
# sampler object
jagger = gnm.sampler(x_0, model, args)
# user-defined prior mean and precision 

m = [0., 0.]   # vector
H = [[1., 0.], 
     [0., 1.]] # matrix
jagger.prior(m, H)

# domain for Jtest
d_min = [-3., -3.]
d_max = [3., 3.]
# test the model's function-Jacobian match
error = jagger.Jtest(d_min, d_max)
assert error == 0 

# back-off info
max_steps = 0
dilation = 0.1
jagger.static(max_steps, dilation)

# start sampling
print("Sampling...")
n_samples = 1.1*10**5
jagger.sample(n_samples)

# burn the initial samples
n_burn = 10**3
jagger.burn(n_burn)

# print results
print("Acceptence Rate : {:.3f}".format(jagger.accept_rate))
print("Number Sampled  : {:.1e}".format(n_samples))
print("Number Burned   : {:.1e}".format(n_burn))
print("Number Used     : {:.1e}".format(n_samples - n_burn))

# create plot info
n_grid = 100
# domain for error_bars
D_min = [-2., 0.]
D_max = [2., 0.]
x, p_x, err = jagger.error_bars(n_grid, D_min, D_max)
plt.plot(x[0], p_x, color = 'b', marker='o', label="Sampled", linewidth=0)    
plt.errorbar(x[0], p_x, yerr = err, fmt = 'b.') 

# theoretical curve (quadrature)
def integrand(a):
    f = lambda b: jagger.posterior([a, b])
    return f

x_min = D_min[0]
x_max = D_max[0]
integral_vector = np.empty([n_grid])
dx  = (x_max-x_min)/n_grid
# integrate
for i in xrange(n_grid):
    x_now = x_min + i * dx
    integral, error = integrate.quad(integrand(x_now), -10, 10)
    integral_vector[i] = integral
# normalize
normalization = np.average(integral_vector)*(x_max-x_min)
normed_vector = integral_vector/normalization


plt.plot(x[0], normed_vector, color = 'k', linewidth = 2, label="Theoretical")

# plot options
plt.legend(loc ="lower center") 
plt.grid(True)
title = ("Rosenbrock")
plt.title(title)
plt.xlabel("x")
plt.ylabel("Probability")
plt.show() 

plt.hist2d(jagger.chain[:,0], jagger.chain[:,1], bins=200, normed=True)
plt.show()
print("--------------FIN!--------------")