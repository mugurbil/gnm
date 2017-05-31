# -*- coding: utf-8 -*-

# Simple example with 1D Well to test installation

import numpy as np
import gnm

# random seeding
np.random.seed(3)

# initial guess
x_0 = [0.1]

# user function 
def model(x, args):
    y = args['y']
    s = args['s']
    return 1, [(x[0]**2-y)/s], [[(2.*x[0])/s]]

# observed data and error = arguments for the user function
data = {'y':[1.], 's':1.}    

# sampler object
jagger = gnm.sampler(x_0, model, data)

# user-defined prior mean and precision 
m = [0.]   # vector
H = [[1.]] # matrix
jagger.prior(m, H)

# domain for Jtest
d_min = [-3.]
d_max = [3.]
# test the model's function-Jacobian match
error = jagger.Jtest(d_min, d_max)
assert error == 0 

# back-off info
max_steps = 1
dilation = 0.1
jagger.static(max_steps, dilation)

# start sampling
print("Sampling...")
n_samples = 1.1*10**4
jagger.sample(n_samples)
print("Done!")

# burn the initial samples
n_burn = 10**3
jagger.burn(n_burn)

# print results
print("Acceptence Rate : {:.3f}".format(jagger.accept_rate))
print("Number Sampled  : {:.1e}".format(n_samples))
print("Number Burned   : {:.1e}".format(n_burn))
print("Number Used     : {:.1e}".format(n_samples - n_burn))

# plot
try: 
	import matplotlib.pyplot as plt
	# create plot info 
	n_grid = 100
	# domain for error_bars
	D_min = [-3.]
	D_max = [3.]
	x, p_x, err = jagger.error_bars(n_grid, D_min, D_max)
	plt.plot(x[0], p_x, color = 'b', marker='o', label="Sampled", linewidth=0)    
	plt.errorbar(x[0], p_x, yerr = err, fmt = 'b.') 

	# create theoretical plot
	#   # initialize curve
	curve = np.zeros(n_grid)
	cnorm = 0.
	#   # create theoretical curve
	for i in xrange(n_grid) :
	    curve[i] = jagger.posterior(x[0][i])
	    cnorm += curve[i]
	#   # normalize curve
	curve = curve/cnorm*n_grid/(D_max[0]-D_min[0])
	plt.plot(x[0], curve, color = 'k', linewidth = 2, label="Theoretical")

	# plot options
	plt.legend(loc ="lower center") 
	plt.grid(True)
	title = ("Simple Well: $p(x)=exp(\\frac{-x^2}{2}"
		     "+\\frac{-(x^2-%d)^2}{2})$" % (data['y'][0]) )
	plt.title(title)
	plt.xlabel("x")
	plt.ylabel("Probability")
	print("Note: Python will stop running when the plot is closed.")
	plt.show() 

except: 
	raise Warning("matplotlib not found.")
	print("Ending.")
