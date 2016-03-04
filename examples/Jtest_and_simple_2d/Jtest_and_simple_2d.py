#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple example with 2D function 
print("----------------------------\n"+
      "----Jtest and Simple 2D-----\n"+
      "----------------------------")

'''
This program shows the usage of Jtest and how it can be helpful in seeing 
mistakes in the definition of the function. One should observe that the 
correct function converges and the error is small. The wrong function has 
an error in its Jacobian and therefore does not pass the Jtest. 

Then we show the sample usage of the gnm-sampler plotting the theoretical 
curve against the sampled distribution for check. This problem uses a simple
nonlinear 2-dimensional model function: f(x,y) = [x, a*x^2+y]'. 
'''

import numpy as np 
import gnm 
import time
from scipy import integrate 
import matplotlib.pyplot as plt

# set the parameters
seed           = 1
x_max          = [2.,2.] # for Jtest
x_min          = [1.,1.] # for Jtest
m              = [0.,0.] # mean of prior
H              = [[1.,0.],
                  [0.,1.]] # precision of prior
arguments      = {'a':1, 'b':2}
num_samples    = 10000
num_burn       = 200
angle          = 1*np.pi/2 # angle is measured ccw from +x axis
plot_range     = [[-3.,-3.],[3.,3.]]
num_grid_pts   = 1000 # for theoretical plot
num_bins       = 50 # for histogram
max_steps      = 2 
step_size      = 0.5

# to check the error after Jtest
def check(error):
    if error:
        print("Failed to converge :(")
        print("Error : " + str(error))
    else: 
        print("Converged!")
    print

np.random.seed(seed)

# function with incorect Jacobian 
print("Jtest with Wrong Function:")
def wrong_funky(x, args):
    a   = args["a"]
    b   = args["b"]
    f_x = [a*x[0]**2, b*x[0]*x[1]]
    J_x = [[a*2*x[0], 0.], [x[0]*b, x[1]*b]]
    return 1, f_x, J_x
f_wrong = gnm.function(wrong_funky, arguments)
error_w = f_wrong.Jtest(x_min, x_max)
check(error_w)

# correct user-defined function
# f(x,y) = (x,ax^2+y)'
print("Jtest with Correct Function:")
def funky(x, args):
    a   = args["a"]
    f_x = [x[0], a*x[0]**2+x[1]]
    J_x = [[1., 0], [2*a*x[0], 1.]]
    return 1, f_x, J_x
f = gnm.function(funky, arguments)
error = f.Jtest(x_min, x_max)
check(error)

# sample the likelihood
start_time    = time.time()
sampler       = gnm.sampler(m, funky, arguments, m=m, H=H)
sampler.sample(num_samples)
# chain         = sampler.burn_in(num_burn)
end_time      = time.time()
print 'Acceptence Percentage: ' + str(sampler.accept_rate)
print 'Ellapsed Time        : ' + str(end_time-start_time)
print 

# histogram of samples
matrix = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])
chain = np.dot(sampler.chain, matrix)
plot_sampled = plt.hist(chain[:,0], num_bins,color = 'b',
                       range=[plot_range[0][0], plot_range[1][0]],normed=True,
                       label='sampled',alpha=0.3)

### START QUADRATURE ###
def integrator(integrand,xmin,xmax,n_points,theta,factor=2):
    '''
    Creating theoretical curve for 2D model functions
    integrator function
    '''
    integral_vector = np.empty([n_points+1])
    dx  = (xmax-xmin)/n_points
    # integrate
    for i in xrange(n_points+1):
        xnow = xmin + i * dx
        integral, error = integrate.quad(rotate(integrand,
                xnow,theta), xmin*factor, xmax*factor)
        integral_vector[i] = integral
    # normalize
    normalization = np.average(integral_vector)*(xmax-xmin)
    normalized_vector = integral_vector/normalization
    return normalized_vector

def rotate(f,x,theta):
    '''
    Returns a function that takes as input the 1D vector along the angle
    given a function that takes in 2D input
    '''
    f_R = lambda b: f(np.array([[x*np.cos(theta)-b*np.sin(theta)],
                                [x*np.sin(theta)+b*np.cos(theta)]]))
    return f_R

x_space = np.linspace(plot_range[0][0],plot_range[1][0],num=num_grid_pts+1)
quadrature_curve = integrator(sampler.posterior, plot_range[0][0],
        plot_range[1][0], num_grid_pts, angle)
plot_theoretical = plt.plot(x_space, quadrature_curve, color = 'r',
                            linewidth =1, label='theoretical')
### END QUADRATURE ###

# error bars
z, fhat, epsf = sampler.error_bars(num_bins, plot_range[0], plot_range[1])
z = np.dot(z.T, matrix).T       # rotate
fhat = np.dot(fhat.T, matrix).T # rotate
epsf = np.dot(epsf.T, matrix).T # rotate
p3 = plt.plot(z[0], fhat[0] , color = 'b', marker = 's', linewidth = 0, 
                alpha=0.0)
plt.errorbar(z[0], fhat[0] , yerr = epsf[0], fmt = 'k.')

# plot labels
plt.title("Simple 2D-Function Posterior PDF")
plt.xlabel("Location (Angle to +x CCW {:.0f}$^o$)".format(angle*180/np.pi))
plt.ylabel("Posterior Probability")
plt.legend()
plt.savefig('simple_2d.pdf',dpi = 500) 
plt.show()

