import numpy as np
import gnm

x_0 = [0, 0, 0, 0, 0] # initial guess
def model(x, args):
    s = args['s']
    return 1, np.dot(s,x)/np.sqrt(2), s/np.sqrt(2)
arg = {'s':np.eye(5)}
jagger = gnm.sampler(x_0, model, arg)
jagger.sample(1000)