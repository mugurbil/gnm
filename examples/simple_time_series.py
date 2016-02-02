#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("---------------------------\n"+
      "--Exponential Time Series--\n"+
      "---------------------------")

import numpy as np
la = np.linalg
import gnm
import time
from scipy import integrate
import matplotlib.pyplot as plt

'''
Exponential time series is an ill-posed problem.
'''

# set the parameters
seed         = 3
t_max        = 10
x_max        = [6.,5.,5.,5.]
x_min        = [-1.,-2.0,-0.,-1.]
m            = [4.,2.,0.5,1.]
H            = np.identity(4)*0.5
sigma        = 0.1
n_samples    = 10000 
n_burn       = 400
angle        = 0.*np.pi/2
n_grid_pts   = 100
n_bins       = 100

np.random.seed(seed)

def exp_time_series(x,t):
	n   = x.size/2
	f 	= np.zeros((t.size,1))
	J   = np.zeros((t.size,x.size))
	for j in xrange(t.size):
		for i in xrange(n):
			f[j]    += x[i]*np.exp(-x[i+n]*t[j])
			J[j,i]   = np.exp(-x[i+n]*t[j])
			J[j,i+n] = -x[i]*t[j]*np.exp(-x[i+n]*t[j])
	return f,J

def model(x, args):
    t = args['t']
    y = args['y']
    s = args['s']
    a,b = exp_time_series(x,t)
    return 1,(a-y)/s,b/s

# create the data with noise
x = np.array([1.,2.5,0.5,3.1])
t = np.array(range(t_max))
y,_ = exp_time_series(x,t)
for i in xrange(y.size):
	y[i] = y[i]*(1+np.random.randn()*0.1)
args = {'t':t, 'y':y, 's':sigma}


jagger = gnm.sampler(m,model,args,m=m,H=H)
#jagger.dynamic(1)
jagger.static(1,0.1)

print("Testing Jacobian...")
error = jagger.Jtest(x_min,x_max)
if error:
    print("error :" + str(error))
else: 
    print("Converged!")
print


# sample the likelihood
print("Sampling...")
start_time    = time.time()
jagger.vsample(n_samples)
jagger.burn(n_burn)
end_time      = time.time()
print("Acceptence Percentage: " + str(jagger.accept_rate))
print("Ellapsed Time        : " + str(end_time-start_time))
print 

# histogram of samples
print 'Plotting Samples...'
plt_smpld = plt.hist(jagger.chain[:,0],n_bins,color = 'b',
                     range=[x_min[0],x_max[0]],normed=True,
                     label='sampled',alpha=0.3)

# # theoretical curve
# print 'Plotting Theoretical Curve...'
# x_space   = np.linspace(x_min[0],x_max[0],num=n_grid_pts+1)
# curve1    = gnm.integrator(0, 1, jagger.posterior, x_min, x_max, n_grid_pts, 0)
# plt_theo  = plt.plot(x_space, curve1, color = 'r', linewidth =1,
#                      label='theoretical')
# print 

# error bars
z, fhat, epsf = jagger.error_bars(n_bins,x_min,x_max)
p3 = plt.plot(z[0],fhat[0], color = 'b', marker = 's', linewidth = 0, alpha=0.0)
plt.errorbar(z[0],fhat[0], yerr = epsf[0], fmt = 'k.')

# plot labels
plt.title('Marginal Posterior for Exponential Time Series for $w_1$')
plt.xlabel('Weight ($w_1$)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.savefig('weight.pdf',dpi = 500)
plt.clf()

# histogram of samples
print 'Plotting Samples...'
plt_smpld = plt.hist(jagger.chain[:,1],n_bins,color ='b',
                     range=[x_min[1],x_max[1]],normed=True,
                     label='sampled',alpha=0.3)

# # theoretical curve
# print 'Plotting Theoretical Curve...'
# x_space   = np.linspace(x_min[1], x_max[1], num=n_grid_pts+1)
# curve2    = gnm.integrator(1, 0, jagger.posterior, x_min, x_max, n_grid_pts, np.pi/2)
# plt_theo  = plt.plot(x_space, curve2, color = 'r', linewidth =1,
#                      label='theoretical')
# print 

# error bars
p3 = plt.plot(z[1],fhat[1], color = 'b', marker = 's', linewidth = 0, alpha=0.0)
plt.errorbar(z[1],fhat[1], yerr = epsf[1],fmt = 'k.')

# plot labels
plt.title('Marginal Posterior for Exponential Time Series for $\lambda_1$')
plt.xlabel('Exponent ($\lambda_1$)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.savefig('exponent.pdf',dpi = 500)
plt.clf()

from matplotlib.colors import LogNorm
from pylab import *

plt.hist2d(jagger.chain[:,0], jagger.chain[:,2], bins=n_bins, normed=True)
colorbar()
plt.title('Exp Time Series Posterior Probability')
plt.xlabel('Weight ($w_1$)')
plt.ylabel('Exponent ($\lambda_1$)')
plt.savefig('time_series_4d1.pdf',dpi = 500)
plt.clf()

plt.hist2d(jagger.chain[:,1], jagger.chain[:,3], bins=n_bins, normed=True)
colorbar()
plt.title('Exp Time Series Posterior Probability 4D-2')
plt.xlabel('Weight ($w$)')
plt.ylabel('Exponent ($\lambda$)')
plt.savefig('time_series_4d2.pdf',dpi = 500)
plt.clf()

print la.norm(jagger.acor())
print jagger.call_count

# p3 = plt.hist(jagger.step_of_accept, [-1,0,1,2,3,4,5], normed=1)
# plt.title('Step of Acceptence')
# plt.xlabel('Step')
# plt.ylabel('Percentage')
# plt.savefig('step2.pdf',dpi = 500)
# plt.clf()

p4 = plt.plot(jagger.step_count/jagger.n_samples, 'ro')
plt.title('Step of Acceptence')
plt.xlabel('Step')
plt.ylabel('Percentage')
plt.savefig('step.pdf',dpi = 500)
plt.clf()

chain = jagger.chain
N = len(chain[0])
n = len(chain)

A = np.zeros(N)

for i in xrange(n):
  A += chain[i,:]

A = A/float(n)

win = 1000
t_max = 10000
C = np.zeros((t_max,N))

for t in xrange(t_max):
  for k in xrange(win):
      for j in xrange(N):
          C[t,j] += (chain[k,j]-A[j])*(chain[k+t,j]-A[j])
  C[t,:] = C[t,:]/(win-1)

for i in xrange(N):
    pv = plt.plot(C[:,i])
    plt.xlabel('Time')
    plt.ylabel('C(t)')
    if i<N/2:
        name = '$w_%d$' % i
    else : 
        name = '$\lambda_%d$' % (i-N/2)
    plt.title('Covariance Function for '+str(name))
    plt.savefig('acf%d.pdf' % i)
    plt.clf()

    

