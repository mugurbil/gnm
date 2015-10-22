#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Plot the results
'''

# import 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
import os

import gnm 

# command line options to set parameters
from optparse import OptionParser
parser = OptionParser()
# experiment number
parser.add_option('-c', dest='count', type='int',
                default=0, help='count of experiment')
# for plotting
parser.add_option('-n', dest='num_bins', type='int', 
                default=100, help='number of bins')

parser.add_option('-l', dest='xmin', type='float', 
                default=-5., help='lower bound of range')

parser.add_option('-u', dest='xmax', type='float', 
                default=5., help='upper bound of range')

parser.add_option('-k', dest='k', type='int',
                default=None, help='the dimension of plot')

parser.add_option('--all', action='store_true', dest='all', 
                default='False', help='plot all dimensions')

parser.add_option('--theory', action='store_true', dest='theo', 
                default='False', help='theoretical curve')

parser.add_option('--two', action='store_true', dest='two', 
                default='False', help='2d histogram')

(opts, args) = parser.parse_args()

# set the parameters
xmin       = opts.xmin
xmax       = opts.xmax
plot_range = [xmin,xmax]
num_bins   = opts.num_bins

# load the samples
folder = 'acor_data_%d/' % opts.count
path  = os.path.join(folder, 'chain')
file  = open(path, 'r')
chain = json.load(file)
chain = np.array(chain)
n     = len(chain)
N     = len(chain[0])/2
file.close()

path = os.path.join(folder, 'stats')
file = open(path, 'r')
stats = json.load(file)
file.close()

import acor
tau   = np.zeros(2*N)
for i in range(2*N):
    tau[i] = acor.acor(chain[:,i],5)[0]
print tau

if opts.all==True :
    k_list = range(2*N)
elif opts.k != None :
    k_list = [opts.k]
else :
    k_list = []

for k in k_list:

    # histogram of samples
    plt_smpld = plt.hist(chain[:,k],num_bins,color = 'b',
            range=plot_range,normed=True,label='sampled',alpha=0.3)

    if opts.theo==True : 
        # theoretical curve
        path = os.path.join(folder, 'curve'+str(k))
        data_file = open(path, 'r')
        theoCurve = json.load(data_file)
        data_file.close()
        x_space   = np.linspace(xmin,xmax,num=len(theoCurve)) 
        plt_theo  = plt.plot(x_space,theoCurve,color = 'r', linewidth =1,
                                label='quadrature')

    # error bars
    z, fhat, epsf = gnm.error_bars(k,chain,num_bins,plot_range)
    epsf = epsf*sqrt(sqrt(tau[k])) # adjust the error with autocorrelation time

    plt_err = plt.plot(z,fhat,color = 'b',marker = 's',
                        linewidth = 0,alpha=0.0)
    plt.errorbar(z,fhat,yerr = epsf,fmt = 'k.')

    # plot labels
    plt.title('Histogram (n={:.0e}, P(A)={:.3})'.format(n,stats['accept_rate']))
    if k < N :
        x_label = '$w_%d$' % (k)
    else : 
        x_label = '$\lambda_%d$' % (k-N)
    plt.xlabel( x_label )
    plt.ylabel('Probability')
    plt.legend()
    path = os.path.join(folder, 'acor_'+x_label.strip('\$')+'.pdf')
    plt.savefig(path, dpi = 500)
    plt.clf()

if opts.two==True :
    for k in xrange(N):
        # 2d histogram
        plt.hist2d(chain[:,k], chain[:,k+N], bins=2*num_bins, normed=True)
        colorbar()
        plt.title('2D Histogram (n={:.0e},P(A)={:.3})'.format(n,stats['accept_rate']))
        plt.xlabel('Weight ($w_%d$)' % k)
        plt.ylabel('Exponent ($\lambda_%d$)' % k)
        path = os.path.join(folder, 'acor_2d_%d.pdf' % k)
        plt.savefig(path, dpi = 500)
        plt.clf()


