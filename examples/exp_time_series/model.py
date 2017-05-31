# -*- coding: utf-8 -*-

import numpy as np

# w_i = x_i
# l_i = x_(n+i)
# f_k = sum_i (w_i * exp(-l_i * t_k))
# d(f_k) = exp(-l_i * t_k)
# /d(w_i)
# d(f_k) = -w_i * t_k *exp(-l_i * t_k)
# /d(l_i)
def funky(x,t):
	t   = np.array(t)
	n   = x.size/2
	f 	= np.zeros((t.size))
	J   = np.zeros((t.size, x.size))
	for j in xrange(t.size):
		for i in xrange(n):
			f[j]    += x[i]*np.exp(-x[i+n]*t[j])
			J[j,i]   = np.exp(-x[i+n]*t[j])
			J[j,i+n] = -x[i]*t[j]*np.exp(-x[i+n]*t[j])
	return 1,f,J