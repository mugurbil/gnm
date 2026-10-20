# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt

with open('chain', 'r') as file:
    chain = np.array(json.load(file))
n = len(chain)
N = len(chain[0])

with open('acov', 'r') as file:
    C = np.array(json.load(file))

for i in range(N):
    plt.plot(C[700000:701000, i])
    plt.xlabel('Time')
    plt.ylabel('C(t)')
    if i < N // 2:
        name = '$w_%d$' % i
    else:
        name = '$\\lambda_%d$' % (i - N // 2)
    plt.title('Covariance Function for ' + str(name))
    plt.savefig('acf%d.pdf' % i)
    plt.clf()
