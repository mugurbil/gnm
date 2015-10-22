#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
from model_func import F 
from gnm import gnm_sampler
from utils import *
import time

'''
This program shows the usage of Jtest and how it can be helpful in seeing 
mistakes in the definition of the function. One should observe that the 
correctfunction converges and the error is small. The wrong function has 
an error in its Jacobian and therefore does not pass the Jtest. 

Then we show the sample usage of the gnm_sampler plotting the theoretical 
curve against the sampled distribution for check. 
'''
# set the parameters
seed           = 1
arguments      = {'a':1.,'b':2}
x_max          = [2.,2.]
x_min          = [1.,1.]
m              = [0.,0.]
H              = [[1.,0.],
                  [0.,1.]]
y              = [0.,0.]
sigma          = 1.
num_samples    = 10000
num_burn       = 200
angle          = 0*pi/2
plot_range     = [-3.,3]
num_grid_pts   = 1000
num_bins       = 50
max_steps      = 5

np.random.seed(seed)

# function with incorect Jacobian 
def wrongFunky(x, args):
    a   = args["a"]
    b   = args["b"]
    f_x = np.array([x[0,0]**2*a,x[0,0]*x[1,0]*b]) 
    J_x = np.array([[a*2*x[0,0],0.],[x[0,0]*b,x[1,0]*b]])
    return 1, f_x, J_x

print 
f_wrong = F(wrongFunky,arguments)
print('Wrong Funky')
converged_w, error_w = f_wrong.Jtest(x_min,x_max)
print('converged:' + str(converged_w))
print('error    :' + str(error_w))
print

# correct user-defined function
# f(x,y) = (x,ax^2+y)'
def funky(x, args):
    a   = args["a"]
    f_x = np.array([[x[0,0],a*x[0,0]**2+x[1,0]]])
    J_x = np.array([[1.,0],[2*a*x[0,0], 1.]])
    return 1, f_x, J_x
f = F(funky,arguments)
print('Correct Funky')
converged, error = f.Jtest(x_min,x_max)
print('converged:' + str(converged))
print('error    :' + str(error))
print

# sample the likelihood
start_time    = time.time()
sampler       = gnm_sampler(m,H,y,sigma,f)
chain, stats  = sampler.sample(m,num_samples,max_steps=max_steps, 
                                step_size=0.5)
# chain         = sampler.burn_in(num_burn)
end_time      = time.time()
print 'Acceptence Percentage: ' + str(stats['accept_rate'])
print 'Ellapsed Time        : ' + str(end_time-start_time)
print 

# plot the results
try:
    import matplotlib.pyplot as plt
except ImportError:
    print 'Install matplotlib to generate plots'
    exit()

# histogram of samples
matrix    = np.array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])
chain     = np.dot(chain,matrix)
plt_smpld = plt.hist(chain[:,0],num_bins,color = 'b',
        range=plot_range,normed=True,label='sampled',alpha=0.3)

# quadrature
def integrator(integrand,xmin,xmax,numberPoints,theta,
        normed=1,factor=2):
    '''
    Theoretical curve plotter
        Creating theoretical curve for 2D model functions
        integrator function
    Inputs:
        integrand:
        xmin:
        xmax:
        numberPoints:
        theta:
            angle
        normed: (optional) 
            norms the output to make it a probability distribution
            write normed=0 if you do not want the output normed
    Outputs:
    '''
    integralVector          = np.empty([numberPoints+1])
    dx                      = (xmax-xmin)/numberPoints
    for i in xrange(numberPoints+1):
        xnow                = xmin + i * dx
        integral, error     = integrate.quad(rotate(integrand,
                xnow,theta),xmin*factor,xmax*factor)
        integralVector[i]   = integral
    if normed:
        normalization       = np.average(integralVector)*(xmax-xmin)
        print 'Normalization is Z = '+str(normalization)
        integralVector      = integralVector/normalization
    return integralVector
    
def rotate(f,x,theta):
    '''
    returns a function that takes as input the 1D vector along the angle
    given a function that takes in 2D input
    '''
    f_R = lambda b: f(np.array([[x*cos(theta)-b*sin(theta)],[x*sin(theta)+b*cos(theta)]]))
    return f_R

# theoretical curve
x_space   = np.linspace(plot_range[0],plot_range[1],num=num_grid_pts+1)
theoCurve = integrator(sampler.posterior,plot_range[0],
        plot_range[1],num_grid_pts,angle)
plt_theo  = plt.plot(x_space,theoCurve,color = 'r', linewidth =1,
        label='theoretical')

# error bars
z, fhat, epsf = error_bars(0,chain,num_bins,plot_range)
p3 = plt.plot(z,fhat,color = 'b',marker = 's',linewidth = 0,alpha=0.0)
plt.errorbar(z,fhat,yerr = epsf,fmt = 'k.')

# plot labels
plt.title('mc_galactic: Histogram of Samples (P(A)='
            +str(stats['accept_rate'])+')')
plt.xlabel('Location')
plt.ylabel('Probability')
plt.legend()
plt.savefig('model.pdf',dpi = 500)
plt.show()

p3 = plt.plot(stats['step_of_accept'])
plt.title('Step of Acceptence')
plt.xlabel('Chain Time')
plt.ylabel('Step')
plt.savefig('step.pdf',dpi = 500)
plt.show()

p4 = plt.semilogy(stats['step_count'])
plt.title('Count of Samples Accepted at Each Step')
plt.xlabel('Step')
plt.ylabel('Number of Samples')
plt.savefig('count.pdf',dpi = 500)
plt.show()

for i in range(max_steps+1):
    avg = 0
    for j in xrange(len(stats['prob_accept'][i])):
        avg += stats['prob_accept'][i][j]
    avg = float(avg)/len(stats['prob_accept'][i])

    plt.plot(stats['prob_accept'][i][1000:1100])
    plt.title('Acceptence Probability at Step %d (Average %f)' % (i,avg))
    plt.xlabel('Location')
    plt.ylabel('Probability')
    plt.savefig('p_a_%d.pdf' % i,dpi = 500)
    plt.show()


