# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import json
import time

# load the samples
file  = open('chain', 'r')
chain = json.load(file)
chain = np.array(chain)
n     = len(chain)
N     = len(chain[0])
file.close()

# A = np.zeros(N)

# for i in xrange(n):
# 	A += chain[i,:]

# A = A/float(n)


# win = 1000
# t_max = n-win
# C = np.zeros((t_max,N))

# for t in xrange(t_max):
# 	for k in xrange(win):
# 		for j in xrange(N):
# 			C[t,j] += (chain[k,j]-A[j])*(chain[k+t,j]-A[j])
# 	C[t,:] = C[t,:]/(win-1)

# file = open('acov', 'w')
# json.dump(C.tolist(), file)
# file.close()

file = open('acov', 'r')
C = json.load(file)
C = np.array(C)
file.close()

for i in xrange(N):
	pv = plt.plot(C[700000:701000,i])
	plt.xlabel('Time')
	plt.ylabel('C(t)')
	if i<N/2:
		name = '$w_%d$' % i
	else : 
		name = '$\lambda_%d$' % (i-N/2)
	plt.title('Covariance Function for '+str(name))
	plt.savefig('acf%d.pdf' % i)
	plt.clf()

	