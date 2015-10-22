#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
la = np.linalg
import gnm
import time

'''
Calculating the auto-corrolation time which is an ill-posed problem.
'''


# set the parameters
seed           = 3
t_max          = 10
N              = 1
x_max          = [2. for i in xrange(2*N) ]
x_min          = [0. for i in xrange(2*N) ]
m              = [0. for i in xrange(2*N) ]
H              = np.identity(2*N)
sigma          = 0.1
num_samples    = 100000
num_burn       = 2000
angle          = 0.*np.pi/2
plot_range     = [-1.0,3.0]
num_grid_pts   = 100
num_bins       = 100

np.random.seed(seed)

args = np.array(range(t_max))

def funky(x,t):
	n   = x.size/2
	f 	= np.zeros((t.size,1))
	J   = np.zeros((t.size,x.size))
	for j in xrange(t.size):
		for i in xrange(n):
			f[j]    += x[i]*exp(-x[i+n]*t[j])
			J[j,i]   = exp(-x[i+n]*t[j])
			J[j,i+n] = -x[i]*t[j]*exp(-x[i+n]*t[j])
	return 1,f,J

f = gnm.F(funky,args)

# create the data 
x     = [0.1,0.1]
_,y,J = f(x)
for i in xrange(y.size):
	y[i] = y[i]*(1.+np.random.randn()*0.1)

print
print('Auto-corrolation time')
print 
print 'Testing Jacobian...'
converged, error = f.Jtest(x_min,x_max)
print('converged:' + str(converged))
print('error    :' + str(error))
print

# sample the likelihood
print 'Sampling...'
start_time    = time.time()
sampler       = gnm.sampler(m,H,y,sigma,f)
sampler.sample(m,num_samples)
sampler.burn(num_burn)
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

# quadrature
def integrator(integrand,xmin,xmax,numberPoints,theta,
        normed=1,factor=2):
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
    f_R = lambda b: f(np.array([[x*cos(theta)-b*sin(theta)],[x*sin(theta)+b*cos(theta)]]))
    return f_R

# histogram of samples
print 'Plotting Samples...'
plt_smpld = plt.hist(chain[:,0],num_bins,color = 'b',
        range=plot_range,normed=True,label='sampled',alpha=0.3)

# theoretical curve
print 'Plotting Theoretical Curve...'
x_space   = np.linspace(plot_range[0],plot_range[1],num=num_grid_pts+1)
theoCurve = integrator(sampler.posterior,plot_range[0],
        plot_range[1],num_grid_pts,angle)
plt_theo  = plt.plot(x_space,theoCurve,color = 'r', linewidth =1,
        label='theoretical')
print 

# error bars
z, fhat, epsf = error_bars(0,chain,num_bins,plot_range)
p3 = plt.plot(z,fhat,color = 'b',marker = 's',linewidth = 0,alpha=0.0)
plt.errorbar(z,fhat,yerr = epsf,fmt = 'k.')

# plot labels
plt.title('Acor: Histogram of Samples (P(A)='+ str(stats['accept_rate'])+')')
plt.xlabel('Location')
plt.ylabel('Probability')
plt.legend()
plt.savefig('acor.pdf',dpi = 500)
plt.clf()

# histogram of samples
print 'Plotting Samples...'
plt_smpld = plt.hist(chain[:,1],num_bins,color = 'b',
        range=plot_range,normed=True,label='sampled',alpha=0.3)

# theoretical curve
print 'Plotting Theoretical Curve...'
x_space   = np.linspace(plot_range[0],plot_range[1],num=num_grid_pts+1)
theoCurve = integrator(sampler.posterior,plot_range[0],
        plot_range[1],num_grid_pts,pi/2.)
plt_theo  = plt.plot(x_space,theoCurve,color = 'r', linewidth =1,
        label='theoretical')
print 

# error bars
z, fhat, epsf = error_bars(1,chain,num_bins,plot_range)
p3 = plt.plot(z,fhat,color = 'b',marker = 's',linewidth = 0,alpha=0.0)
plt.errorbar(z,fhat,yerr = epsf,fmt = 'k.')

# plot labels
plt.title('Acor: Histogram of Samples (P(A)='+ str(stats['accept_rate'])+')')
plt.xlabel('Location')
plt.ylabel('Probability')
plt.legend()
plt.savefig('acor2.pdf',dpi = 500)
plt.clf()

p3 = plt.plot(stats['step_of_accept'])
plt.title('Step of Acceptence')
plt.xlabel('Chain Time')
plt.ylabel('Step')
plt.savefig('step.pdf',dpi = 500)
plt.clf()

p4 = plt.semilogy(stats['step_count'])
plt.title('Count of Samples Accepted at Each Step')
plt.xlabel('Step')
plt.ylabel('Number of Samples')
plt.savefig('count.pdf',dpi = 500)
plt.clf()

for i in range(5+1):
    avg = 0
    for j in xrange(len(stats['prob_accept'][i])):
        avg += stats['prob_accept'][i][j]
    avg = float(avg)/len(stats['prob_accept'][i])

    plt.plot(stats['prob_accept'][i][num_samples/2:num_samples/2+2*num_grid_pts])
    plt.title('Acceptence Probability at Step %d (Average %f)' % (i,avg))
    plt.xlabel('Location')
    plt.ylabel('Probability')
    plt.savefig('p_a_%d.pdf' % i,dpi = 500)
    plt.clf()

from matplotlib.colors import LogNorm
from pylab import *

plt.hist2d(chain[:,0], chain[:,1], bins=num_bins, normed=True)
colorbar()
plt.title('Acor: Histogram of Samples 2D (P(A)='+ str(stats['accept_rate'])+')')
plt.xlabel('Weight ($w$)')
plt.ylabel('Exponent ($\lambda$)')
plt.savefig('acor2d.pdf',dpi = 500)
plt.clf()
