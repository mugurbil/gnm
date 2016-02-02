#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

The Gauss-Newton-Metropolis Algorithm with back-off strategy is
specialized in sampling highly non-linear posterior distributions. 
"""

__all__ = ["sampler"]

import numpy as np
la = np.linalg

import copy # for sample
import sys # for vsample

from .utils import *

class sampler:

    def __init__(self, x, model, args, m=0, H=0):
    	""" 
    Init
        Initialize the GNM sampler class
	Inputs  :
        x : 
            initial guess
        model :
            user defined data model function
        m : 
            mean of the prior
        H : 
            precision matrix of the prior
    Hiddens :
        ln_H_ : log(det(H))/2
            calculate this once to use everytime log prior is called
        Hm    : < H, m >
            calculate this once to use everytime proposal is called
        """
        self._args = args
        self._f = function(model, args)

        x = np.reshape(np.array(x), (-1,1)) 
        self._n = np.size(x)
        chi_x, f_x, J_x = self._f(x)     
        try:
            assert chi_x == True
        except AssertionError:
            print("Error: Initial guess out of range.")
            return 0

    	self.prior(m, H)
    	# self.prior(0*x, (0*x)*x.T)

        self._max_steps = 5
        self._step_size = 0.5
        self._fancy = False
        self._opts = {}

        # sampler outputs
        self._chain = None
        self._n_samples = 0
        self._n_accepted = 0

        self._state_x = self._proposal_params({'x':x,'f':f_x,'J':J_x }) 

    def prior(self, m, H):
        """
    Set prior
        Set prior values
    Inputs :    
        m : 
            mean of the prior
        H : 
            precision matrix of the prior
    Hiddens :
        ln_H_ : log(det(H))/2
            calculate this once to use everytime log prior is called
        Hm    : < H, m >
            calculate this once to use everytime proposal is called
        """
        self._m     = np.reshape(np.array(m), (-1,1))
        try : 
            assert np.size(self._m) == self._n
        except : 
            print("Error: Mean has to be an array of size n.")
        self._H     = np.array(H)
        try : 
            assert np.shape(self._H) == (self._n, self._n)
        except : 
            print("Error: Precision has to be a matrix of shape n by n.")
            exit()
        self._ln_H_ = np.log(la.det(self._H))/2.
        self._Hm    = np.dot(self._H,self._m)

    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
            p=2, l_max=50, r=0.5):
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
        """
    	return self._f.Jtest(x_min, x_max, dx=dx, N=N, eps_max=eps_max, p=p, 
    				  l_max=l_max, r=r)

    def static(self, max_steps, step_size):
        """
    Set Back-off to Static
        Set the sampler parameters for static back off
    Inputs :
        max_steps :
            maximum optimization steps to be taken
        step_size : 
            the step size of the back-off
        """
        self._fancy = False
        # begin checks      
        try : 
            assert max_steps == int(max_steps)
        except AssertionError :
            print("Warning: max_steps is not an int. Converted.")
            max_steps = int(max_steps)
        except :
            print("Error: max_steps has to be an int.")
            return 0

        try : 
            assert max_steps >= 0
        except : 
            print("Warning: max_steps has to be non-negative.")
            print("Setting max_steps to 0.")
            max_steps = 0
        self._max_steps = max_steps

        if max_steps > 0 :
            try : 
                assert step_size == float(step_size)
            except AssertionError :
                print("Warning: step_size is not a float. Converted.")
                step_size = float(step_size)
            except :
                print("Error: step_size has to be a float.")
                return 0

            try : 
                assert 0. < step_size < 1.
            except : 
                print("Warning: step_size has to be between 0 and 1.")
                print("Setting step_size to 0.5.")
                step_size = 0.5
            self._step_size = step_size
            if step_size**max_steps < 10**(-15):
                print("Warning: Back-off gets dangerously small.")
        # end checks

    def dynamic(self, max_steps, opts={}):
        """
    Dynamic Switch
        Set the sampler parameters for dynamic back off
    Inputs :
        max_steps :
            maximum optimization steps to be taken
    Optional Inputs: 
        opts       : ({})
        dictionary containing fancy options
        """
        self._fancy = True
        self._opts = opts
        self._max_steps = max_steps

    def sample(self, n_samples):
        """
    Sample
        Generate samples for posterior distribution using Gauss-Newton 
        proposal parameters
    Inputs : 
        n_samples :
            number of samples to generate
    Hidden Outputs :
        chain  :
            chain of samples
        n_samples :
            length of chain
        n_accepted :
            number of proposals accepted
        step_count :
            count of the steps accepted
        """
        try : 
            n_samples = int(n_samples)
        except :
            print("Error: Number of samples has to be an int.")
            exit()

        # fetch info
        state_x = self._state_x
        x = state_x['x']
        max_steps = self._max_steps
        q = self._step_size
        opts = self._opts

        # initialize lists 
        chain = np.zeros((n_samples, self._m.size)) 
        n_accepted = 0
        step_count = np.zeros(max_steps+2)

        # begin outer loop
        for i in xrange(n_samples):
            accepted = False           # check if sample is accepted
            r        = 1.              # initial step size
            r_list   = []              # list of step sizes
            z_list   = [state_x]       # list of steps 
            D_x_z    = state_x['post'] # denominator of acceptance prob

            steps = 0 # back off steps taken so far
            while steps <= max_steps:

                # get proposal
                f_defined_z = False
                while f_defined_z == False :
                    mu_x, L_x = update_params(state_x, r)
                    z         = multi_normal(mu_x, L_x)
                    f_defined_z, f_z, J_z = self._f(z)
                    if not f_defined_z: 
                        r = r * q
                        steps += 1
                r_list.append(r)

                state_z = self._proposal_params({'x':z,'f':f_z,'J':J_z })
                z_list.append(state_z)
                mu_z, L_z = update_params(state_z, r)

                D_x_z  = D_x_z + log_expo(z, mu_x, L_x)
                # step 2 is to compute N_z_x where it is more difficult 
                # since we cannot make use of previous N_z_x
                N_z_x  = state_z['post'] + log_expo(x, mu_z, L_z)
                N_is_0 = False # check to see if N_z_x = 0
                for j in xrange(1, steps+1):
                    r_j       = r_list[j]
                    mu_z, L_z = update_params(state_z, r_j)
                    z_j, f_j  = get_vals(z_list[j])
                    mu_j, L_j = update_params(z_list[j], r_j)

                    P_z_j = log_expo(z_j, mu_z, L_z)
                    D_z_j = state_z['post']   + P_z_j
                    N_j_z = z_list[j]['post'] + log_expo(z, mu_j, L_j)

                    if N_j_z > D_z_j :
                        A_z_j = 1. 
                        N_is_0 = True
                        break
                    else:
                        A_z_j = min(1., np.exp(N_j_z - D_z_j))
                    N_z_x  = N_z_x + P_z_j + np.log(1. - A_z_j)
                # end of j for loop
                if N_is_0 == True :
                    A_x_z = 0.
                elif N_z_x > D_x_z :
                    A_x_z = 1.
                else :
                    A_x_z = min(1., np.exp(N_z_x - D_x_z))

                if  np.random.rand() <= A_x_z:
                    accepted = True
                    break
                else : 
                    D_x_z  = D_x_z + np.log(1. - A_x_z)
                    r      = self._back_off(z_list, r_list)
                    steps += 1                     
            # end of steps for loop
            if accepted == True :
                n_accepted += 1    
                chain[i,:] = z[:,0] 
                state_x = copy.deepcopy(state_z)
                x,f_x   = get_vals(state_x)
                # for statistics
                step_count[steps+1] += 1
            else :
                chain[i,:]  = x[:,0] 
                step_count[0] += 1
        # end outer loop

        # update stored info
        self._state_x = state_x

        # outputs
        if self._n_samples == 0 :
            self._chain = chain
            self._step_count = step_count
        else :
            self._chain = np.append(self._chain, chain, axis=0)
            self._step_count = np.add(self._step_count, step_count)

        self._n_samples += n_samples
        self._n_accepted += n_accepted
    # end sampler

    def vsample(self, n_samples, divs=100):
        """
    Vsample
        Provides simple sampling info while sampling
    Inputs  : 
        n_samples :
            number of samples to generate
    Optional Inputs : 
        divs : (100)
            number of divisions
        """
        print("Sampling: 0%")
        for i in xrange(divs):
            self.sample(int(n_samples/divs))
            sys.stdout.write("\033[F") # curser up
            print("Sampling: "+str(int(i*100./divs)+1)+'%')
        self.sample(n_samples % divs)
        
    def burn(self, n_burned):
        """
    Burn
        Burn the inital samples to adjust for convergence of the chain
        cut the first (n_burned) burn-in samples
    Inputs  :
        chain : 
            the full Markov chain
        n_burned :
            number of samples to cut 
    Hidden Outputs :
        chain : 
            chain with the firt n_burned samples cut
        """
        self._chain = self._chain[n_burned:]

    def acor(self, k = 5):
        """
    Autocorrelation time of the chain
        return the autocorrelation time for each parameters
    Inputs :
        k : 
            parameter in self-consistent window
    Outputs :
        t :
            autocorrelation time of the chain
        """
        try:
            import acor
        except ImportError:
            print "Can't import acor, please download."
            return 0
        n = np.shape(self._chain)[1]
        t = np.zeros(n)
        for i in xrange(n):
            t[i] = acor.acor(self._chain[:,i],k)[0]
        return t

    def posterior(self, x):
        """
    Posterior density 
        ** not normalized **
        This is used to plot the theoretical curve for tests.
    Inputs  :
        x : 
            input value 
    Outputs : 
        p_x_y : p(x|y)=p(x)*exp{-||f(x)-y||^2/(2s^2)}
            posterior probability of x given y
        """
        m = self._m
        H = self._H

        x = np.reshape(np.array(x), (-1,1))

        f_defined, f_x, J_x = self._f(x)
        if f_defined :
            p_x   = np.exp(-np.dot((x-m).T,np.dot(H,x-m))/2.)
            p_x_y = p_x*np.exp(-la.norm(f_x)**2/2.)
            return p_x_y
        else :
            return 0
    
    def error_bars(self, n_bins, d_min, d_max):
        """
    Error Bars
        create bars and error bars to plot
    Inputs  :
        n_bins   :
            number of bins
        plot_range : (shape) = (number of dimensions, 2)
            matrix which contain the min and max for each dimension as rows
    Outputs :
        x     : 
            domain
        p_x   :
            estimated posterior using the chain on the domain
        error : 
            estimated error for p_x
        """
        # fetch data
        chain    = self._chain
        length   = len(chain)
        try:
            n_dims = np.shape(chain)[1]
        except:
            n_dims = 1

        # begin checks
        try: 
            assert n_bins == int(n_bins)
        except: 
            print("Number of bins has to be an integer.")
            return 0
        d_min = np.reshape(np.array(d_min), (-1,1))
        d_max = np.reshape(np.array(d_max), (-1,1))
        try: 
            assert np.size(d_min) == n_dims
        except: 
            print("Domain minimum has wrong size.")
            return 0
        try: 
            assert np.size(d_max) == n_dims
        except: 
            print("Domain maximum has wrong size.")
            return 0
        # end checks


        # initialize outputs
        p_x   = np.zeros((n_dims, n_bins))    # esitmate of posterior
        error = np.zeros((n_dims, n_bins))    # error bars
        x     = np.zeros((n_dims, n_bins))    # centers of bins

        # loop through dimensions
        for dim in xrange(n_dims):
            x_min = d_min[dim]
            x_max = d_max[dim]
            dx    = float(x_max-x_min)/n_bins
            # bin count 
            for i in xrange(length):
                if chain[i][dim] > x_min and chain[i][dim] < x_max:
                    bin_no = int((chain[i][dim]-x_min)/dx)
                    p_x[dim][bin_no] += 1.
            # end count
            p_x[dim] = p_x[dim]/(length*dx)
            # find error
            for i in xrange(n_bins):
                p             = p_x[dim][i]
                error[dim][i] = np.sqrt(p*(1./dx-p)/(length))
                x[dim][i]     = x_min+(0.5+i)*dx
            # end find
        # end outer loop
        return x, p_x, error

    """
    Properties:
    1.  chain
    2.  n_samples
    3.  n_accepted
    4.  accept_rate
    5.  step_count
    6.  call_count
    7.  max_steps
    8.  step_size
    """

    @property
    def chain(self):
        return self._chain

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_accepted(self):
        return self._n_accepted

    @property
    def accept_rate(self):
        return float(self._n_accepted)/self._n_samples

    @property
    def step_count(self):
        return self._step_count

    @property
    def call_count(self):
        return self._f.count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def step_size(self):
        return self._step_size
    
    
    

    # internal methods
    
    def _log_post(self,x,f_x):
        """
    Log of the posterior density
        This is used to calculete acceptance probability for sampling.
    Inputs  :
        x   : 
            input value
        f_x : f(x), 
            function value at x
    Outputs : 
        log(p_x_y) : p(x|y)=log[p(x)]-||f(x)-y||^2/(2s^2)
            posterior probability of x given y
        """
        m = self._m
        H = self._H

        # prior part           -(x-m)'H(x-m)/2
        part1 = -np.dot((x-m).T,np.dot(H,x-m))/2.
        # least squares part   -||f(x)-y||^2/(2s^2)
        part2 = -la.norm(f_x)**2/(2.)
        return self._ln_H_+part1+part2

    def _proposal_params(self, state):
        """
    Proposal parameters
        Calculate parameters needed for the proposal. 
    Inputs  :
        state : 
            x  :   
                the present sample, the place to linearize around
            f  : f(x), 
                function value at x
            J  : f'(x), 
                the jacobian of the function evaluated at x
    Outputs :
        state :
            mu   : 
                the mean vector
            L    :
                the lower triangular cholesky factor of P 
            post : log_post
                log of the posterior density
        """
        m  = self._m
        H  = self._H
        Hm = self._Hm
        x  = state['x']
        f  = state['f']
        J  = state['J']

        b  = -np.dot(J,x)+(f)             # b = [f(x)-f'(x)x]
        L  = la.cholesky(H+np.dot(J.T,J)) # P = H+A'A = LL'   
        mu = la.solve(L.T,la.solve(L,Hm-np.dot(J.T,b)))
        state['L']  = L                   # mu_x = P^-1(Hm-A'b)
        state['mu'] = mu
        # state['ln_L'] = np.log(det(L))
        state['post'] = self._log_post(x,f)
        return state

    def _back_off(self, list_states, r_list):
        """
    Back off
        Calculate the back off step size
    Inputs : 
        list_states :
            list of states in current proposal
        r_list : 
            list of back offs in current proposal
        step_size : 
            step size reduction
        fancy : 
            set to True if you want to use the fancy back-off
    Outputs : 
        """
        step_size = self._step_size
        fancy = self._fancy
        r = r_list[len(r_list)-1]
        if fancy:
            p_0   = la.norm(list_states[0]['f'])
            dp_0  = p_0*2*la.norm(list_states[0]['J'])
            p_r   = la.norm(list_states[len(list_states)-1]['f'])
            dp_r  = p_0*2*la.norm(list_states[len(list_states)-1]['J'])
            r_new = optimize(r,p_0**2,dp_0,p_r**2,dp_r)
            return r_new
        else :
            return r*step_size    
