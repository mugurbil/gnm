#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple example with 1D Well
print("-----------------------\n"+
      "--------1D Well--------\n"+
      "-----------------------")

import numpy as np
import gnm
import time
import json
import matplotlib.pyplot as plt
import acor

# command line prompts
from optparse import OptionParser
parser = OptionParser()
# random seed
parser.add_option('-r', dest='seed', type='int',
                default=1, help='random seed')
# argument for the user function
parser.add_option('-a', dest='a', type='float',
                default=1., help='args')
# domain info 
parser.add_option('--dmin', dest='d_min', type='float',
                default=-3., help='domain minimum')
parser.add_option('--dmax', dest='d_max', type='float',
                default=3., help='domain maximum')                
# prior info
parser.add_option('-m', dest='m', type='float',
                default=0, help='mean of prior')
parser.add_option('-H', dest='H', type='float',
                default=1, help='precision of prior')
# data info
parser.add_option('-y', dest='y', type='float',
                default=2, help='observed data point')
parser.add_option('-s', dest='s', type='float',
                default=1, help='observed standard deviation')
# back off info
parser.add_option('--max', dest='max', type=int,
                default=5, help='max number of steps for back-off')
parser.add_option('--size', dest='size', type=float,
                default=0.2, help='step size for back-off')
parser.add_option('--fancy', dest='fancy', action='store_true',
                default=False, help='turn on fancy')
# # visual bar
# parser.add_option('--bar', dest='pbar', action='store_false',
#                 default=True, help='toggle off progressbar')
# sampling info
parser.add_option('-n', dest='n_samples', type='int',
                default=1.1*10**5, help='number of samples')
parser.add_option('-b', dest='n_burn', type='int',
                default=10**4, help='number of samples to burn')
# plotting info
parser.add_option('-g', dest='n_grid', type='int',
                default=100, help='number of grid points')
(opts, args) = parser.parse_args()
# end command line

np.random.seed(opts.seed)

# function 
args = {'a':opts.a}

def mod_f(x,args):
    a = args['a']
    return 1, [a*x[0]**2], [[a*2.*x[0]]]

# gnm function instance
f = gnm.F(mod_f,args)
print("\nTesting Jacobian...")
converged, error = f.Jtest(opts.d_min, opts.d_max)
if converged :
    print("Converged!")
else : 
    print("Check function definition. \nExiting...")
    exit()
print("Error: {:.3e}\n".format(error))

# user-defined prior mean and precision 
m = [opts.m]   # vector
H = [[opts.H]] # matrix

# observed data and error
y = [opts.y]   # vector
sigma = opts.s # float

# initial guess 
x_0 = m

# sampler object
gnm_sampler = gnm.sampler(m, H, y, sigma, f)
gnm_sampler.set(x_0, max_steps=opts.max, step_size=opts.size, 
                fancy=opts.fancy, opts={})

start_time = time.time()
gnm_sampler.vsample(opts.n_samples)

gnm_sampler.burn(opts.n_burn)

end_time = time.time()
T = end_time - start_time # sample time
print("Acceptence Rate : {:.3f}".format(gnm_sampler.accept_rate))
print("Step Size       : {:.2f}".format(opts.size))
print("Max Steps       : {:}".format(opts.max))
print("Ellapsed Time   : %d h %d m %d s" % (T/3600,T/60%60,T%60))
print("Number Sampled  : {:.1e}".format(opts.n_samples))
print("Number Burned   : {:.1e}".format(opts.n_burn))
print("Number Used     : {:.1e}\n".format(opts.n_samples - opts.n_burn))


x, p_x, err = gnm_sampler.error_bars(opts.n_grid, [[opts.d_min,opts.d_max]])

curve = np.zeros(opts.n_grid)
cnorm = 0.

for i in xrange(opts.n_grid) :
    curve[i] = gnm_sampler.posterior(x[0][i])
    cnorm += curve[i]

curve = curve/cnorm/6.*opts.n_grid

if opts.max == 0 :
    sz = "no back-off"
else :  
    sz = "step_size=" + str(opts.size)

plt.plot(x[0], curve, color = 'k', linewidth = 2, label='Theoretical')
plt.plot(x[0], p_x[0], color = 'b', marker = ',', linewidth = 0, 
	label='Sampled')    
plt.errorbar(x[0], p_x[0], yerr = err[0], fmt = 'b.') 
title = ("1D Well: $\\pi(x|y=%d)=exp(\\frac{-x^2}{2}" % y[0] +
	     "+\\frac{-(x^2-%d)^2}{2})$" % y[0] )
plt.title(title)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend(loc ='lower center') 
plt.savefig('test%d.pdf' % opts.max, dpi = 500)   
plt.show() 

# p3 = plt.plot(stats['step_of_accept'])
# plt.show()

# if max_steps > 0 :
#     p4 = plt.semilogy(stats['step_count'])
#     plt.savefig('steps%d.pdf'%max_steps,dpi = 500)   
#     plt.clf()
#     # plt.show()

# for i in range(5):
#     avg = 0
#     for j in xrange(len(stats['prob_accept'][i])):
#         avg += stats['prob_accept'][i][j]
#     avg = float(avg)/len(stats['prob_accept'][i])

#     plt.plot(stats['prob_accept'][i][10**(5-i):min(10**(5-i)+1000,len(stats['prob_accept'][i]))])
#     plt.title('Acceptence Probability at Step %d (Average %f)' % (i,avg))
#     plt.xlabel('Location')
#     plt.ylabel('Probability')
#     plt.savefig('p_a_%d_%d.pdf' % (max_steps,i),dpi = 500)
#     plt.clf()


# def fInt(a,b,n,f):
#     '''
#     Quick and dirty integrator
#     a,b - integration interval
#     n   - number of points for integration
#     f   - integrand
#     '''    
#     dx = float(b-a)/n  
#     I  = 0             # result
#     x  = a + .5*dx
#     for j in range(n):
#         I = I + f(x)
#         x = x + dx
#     I = I*dx
#     return I    
# file = open('chain%d'%max_steps, 'w')
# json.dump(x_m.tolist(), file)
# file.close()

# file = open('stats%d'%max_steps, 'w')
# json.dump(stats, file)
# file.close()

# print "Loading File..."
# file = open('chain%d'%max_steps, 'r')
# x_m = json.load(file)
# file.close()
# x_m = np.array(x_m[5*10**6:9*10**6])

# file = open('stats%d'%max_steps, 'r')
# stats = json.load(file)
# file.close()
# print "File Loaded."

# def un_post(x):
#     '''
#     un-normalized posterior distribution
#     '''
#     flag,f_x,J_x = f(x)     
#     return exp(-x**2/2.)*exp(-(f_x-y[0])**2/2.)

# xmin = -3.
# xmax = 3.    
# N_F  = fInt(xmin,xmax,6000,un_post) # normalization factor

# def post(x):
#     '''
#     normalized posterior
#     '''
#     return un_post(x)/N_F
# L    = len(x_m) # number of samples    
# dx   = .02
# nb   = int((xmax-xmin)/dx)   # number of bins
# dx   = float((xmax-xmin)/nb)
# N    = np.zeros(nb)   # bin counts
# for i in range(L):
#     if x_m[i,0] > xmin and x_m[i,0] < xmax:
#         j = int((x_m[i,0]-xmin)/dx)
#         N[j] = N[j]+1

# fhat = np.zeros(nb,np.float64)    # esitmate of posterior
# fBar = np.zeros(nb,np.float64)    # average of posterior in each bin
# xl   = xmin
# xr   = xmin + dx
# epsf = np.zeros(nb,np.float64)    # error bars
# x    = np.zeros(nb)               # centers of bins

# for i in range(nb):
#     fhat[i] = float(N[i]/(L*dx))
#     fBar[i] = fInt(xl,xr,int(dx*100),post)/dx
#     xl = xl+dx
#     xr = xr+dx
#     p  = float(N[i])/L
#     epsf[i] = sqrt(p*(1-p)/(dx*dx*L))
#     x[i] = xmin + (.5+i)*dx
    
    
# try:
#     import acor as acor
#     tau = acor.acor(x_m[:,0], 5)  # compute the autocorrelation time
#     print 'Acor : {:.1f} \n'.format(tau[0])
#     # epsf = epsf*np.log(tau[0])      # adjust the error with autocorrelation time
# except:
#     print 'Could not work acor.'


          
