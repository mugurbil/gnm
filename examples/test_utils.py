#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
import numpy as np
from optimize import *

m = np.array([2,3])
L = np.array([[1,0],[2,3]])
C = np.linalg.inv(np.dot(L,L.T))

print multi_normal(m,L)
print np.random.multivariate_normal(m,C)
print det(L)


xmin = 0.
xmax = 1.
from scipy import integrate
    
def integrand(x1,x2,x3):
	return x1*x2*x3
n, err = integrate.nquad(integrand,[[xmin,xmax] for i in xrange(3)])
print n


g_2 = 3. 
h_2 = 2.
f = lambda x: sin(g_2*x+h_2)
df = lambda x: g_2*cos(g_2*x+h_2)
f_0 = f(0)
df_0 = df(0)
alpha = 1.
f_alpha = f(alpha)
df_alpha = df(alpha)

g = 3.
h = -1.
k = lambda x: -(x+h)**3
dk = lambda x: -3*(x+h)**2
k_0 = k(0)
dk_0 = dk(0)
alpha = 1.
k_alpha = k(alpha)
dk_alpha = dk(alpha)

x = optimize(alpha, f_0, df_0, f_alpha, df_alpha)
print x



# integralVector        = np.empty([numberPoints+1])
# dx                    = (xmax-xmin)/numberPoints
# for i in range(numberPoints+1):
#     xnow              = xmin + i * dx
#     integral, err     = integrate.nquad(integrand,,xmin*factor,xmax*factor)
#     integralVector[i] = integral
# if normed:
#     normalization     = np.average(integralVector)*(xmax-xmin)
#     print 'Normalization is Z = '+str(normalization)
#     integralVector    = integralVector/normalization
# return integralVector


