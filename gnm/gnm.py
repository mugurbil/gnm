#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

This is a Gauss-Newton-Metropolis Algorithm with back-off strategy is
specialized in sampling highly non-linear posterior distributions. 
User should supply a Gaussian prior distribution with mean vector m 
and precision matrix H (inverse of the covariance matrix). The likely-hood 
function is given by a model function f and observed data y. The observational 
error is assumed to be i.i.d N(0,sigma**2). If we have no information 
for the prior, just set H to be very small.
'''

__all__ = ['sampler']

import numpy as np
la = np.linalg

import copy # for sample
import sys # for vsample

from .utils import *

class sampler:

    def __init__(self, m, H, y, sigma, f):
    	''' 
    Init
        Initialize the GNM sampler class
	Inputs  :
		m : 
			mean of the prior
		H : 
			precision matrix of the prior
		s : sigma 
            standard deviation of observation error
        y :
            observed data
        f :
            developer defined data model function instance of the
            user defined model function
    Hiddens :
        ln_H_ : log(det(H))/2
            calculate this once to use everytime log prior is called
        Hm    : < H, m >
            calculate this once to use everytime proposal is called
        '''
        self._m     = np.reshape(np.array(m), (-1,1))
        self._H     = np.array(H)
        try : 
            assert self._H.ndim == 2
        except : 
            print("Error: Precision has to be a matrix.")
            exit()
        self._s     = float(sigma)
        self._y     = np.reshape(np.array(y), (-1,1))
        self._f     = f
        # precompute variables to be used later
        self._ln_H_ = np.log(la.det(self._H))/2.
        self._Hm    = np.dot(self._H,self._m)
        self.reset()

    def guess(self, x):
        """"
    Set initial guess
        Set the initial guess
    Inputs :
        x0 : 
            initial guess, np-array of size n 
        """
        x = np.reshape(np.array(x), (-1,1)) 
        try: 
            assert self._m.size == x.size
        except AssertionError:
            print("Error: Size of initial guess does not match the prior.")
            return 0

        f_defined_x,f_x,J_x = self._f(x)     
        try:
            assert f_defined_x == True
        except AssertionError:
            print("Error: Initial guess out of range.")
            return 0
        self._state_x = self._proposal_params({'x':x,'f':f_x,'J':J_x }) 

    def set(self, max_steps=5, step_size=0.2, fancy=False, opts={}):
        '''
    Set sampler
        Set the sampler parameters
    Inputs :
        x0 : 
            initial guess, np-array of size n   
    Optional Inputs :
        max_steps : (5)
            maximum optimization steps to be taken
        step_size : (0.2)
            the step size of the back-off
        fancy     : (False)
            set to True if you want to use the fancy back-off
        opt       : ({})
            dictionary containing fancy options
        '''
        # begin checks      
        try : 
            assert max_steps == int(max_steps)
        except AssertionError :
            print 'Warning: max_steps is not an int. Converted.'
            max_steps = int(max_steps)
        except :
            print 'Error: max_steps has to be an int.'
            return 0

        try : 
            assert max_steps >= 0
        except : 
            print 'Warning: max_steps has to be non-negative.'
            print 'Setting max_steps to 0.'
            max_steps = 0
        self._max_steps = max_steps

        if max_steps > 0 :
            try : 
                assert step_size == float(step_size)
            except AssertionError :
                print 'Warning: step_size is not a float. Converted.'
                step_size = float(step_size)
            except :
                print 'Error: step_size has to be a float.'
                return 0

            try : 
                assert 0. < step_size < 1.
            except : 
                print 'Warning: step_size has to be between 0 and 1.'
                print 'Setting step_size to 0.2.'
                step_size = 0.2
            self._step_size = step_size

        self._fancy = fancy
        self._opts = opts
        # end checks
        # opt checks...

    def reset(self):
        '''
    Reset
        Reset and initialize the parameters
        '''
        # sampler parameters
        x = self._m
        f_defined_x,f_x,J_x = self._f(x)     
        try:
            assert f_defined_x == True
        except AssertionError:
            print("Error: Prior mean out of range.")
            return 0
        self._state_x = self._proposal_params({'x':x,'f':f_x,'J':J_x}) 
        self._max_steps = 5
        self._step_size = 0.2
        self._fancy = False
        self._opts = {}

        # sampler outputs
        self._chain = None
        self._num_samples = 0
        self._num_accepted = 0
        self._step_of_accept = None
        self._step_count = None
        self._prob_accept = []

    def sample(self, num_samples):
        '''
    Sample
        Generate samples for posterior distribution using Gauss-Newton 
        proposal parameters
    Inputs  : 
        num_samples :
            number of samples to generate
    Hidden Outputs :
        chain  :
            chain of samples
        num_samples :
            length of chain
        num_accepted :
            number of proposals accepted
        step_of_accept :
            list of the step of acceptance
        step_count :
            count of the steps accepted
        prob_accept :
            probability of acceptence at each proposal
        '''
        try : 
            num_samples = int(num_samples)
        except :
            print("Error: Number of samples has to be an int.")
            exit()

        # fetch info
        state_x = self._state_x
        x = state_x['x']
        max_steps = self._max_steps
        step_size = self._step_size
        fancy = self._fancy
        opts = self._opts

        # initialize lists 
        chain = np.zeros((num_samples,self._m.size)) 
        num_accepted = 0
        step_of_accept = np.zeros(num_samples)
        step_count = np.zeros(max_steps+1)
        prob_accept = [[] for i in xrange(max_steps+1)]

        # begin outer loop
        for i in xrange(num_samples):
            accepted = False           # check if sample is accepted
            r        = 1.              # initial step size
            r_list   = []              # list of steps sizes
            z_list   = [state_x]       # list of steps 
            D_x_z    = state_x['post'] # denominator of acceptance prob

            steps = 0 # back off steps taken so far
            while steps <= max_steps:

                # get proposal
                f_defined_z = False
                while f_defined_z == False :
                    mu_x, L_x = update_params(state_x,r)
                    z         = multi_normal(mu_x,L_x)
                    f_defined_z, f_z, J_z = self._f(z)
                    if not f_defined_z: 
                        r = r * step_size
                        steps += 1
                r_list.append(r)

                state_z   = self._proposal_params({'x':z,'f':f_z,'J':J_z })
                z_list.append(state_z)
                mu_z, L_z = update_params(state_z,r)

                D_x_z  = D_x_z + log_expo(z,mu_x,L_x)
                # step 2 is to compute N_z_x where it is more difficult 
                # since we cannot make use of previous N_z_x
                N_z_x  = state_z['post'] + log_expo(x,mu_z,L_z)
                N_is_0 = False # check to see if N_z_x = 0
                for j in xrange(1,steps+1):
                    r         = r_list[j]
                    mu_z, L_z = update_params(state_z,r)
                    z_j,f_j   = get_vals(z_list[j])
                    mu_j,L_j  = update_params(z_list[j],r)

                    P_z_j = log_expo(z_j,mu_z,L_z)
                    D_z_j = state_z['post']   + P_z_j
                    N_j_z = z_list[j]['post'] + log_expo(z,mu_j,L_j)

                    if N_j_z > D_z_j :
                        A_z_j = 1. 
                        N_is_0 = True
                        break
                    else:
                        A_z_j = min(1.,np.exp(N_j_z-D_z_j))
                    N_z_x  = N_z_x + P_z_j + np.log(1.-A_z_j)
                # end of j for loop
                if N_is_0 == True :
                    A_x_z = 0.
                elif N_z_x > D_x_z :
                    A_x_z = 1.
                else :
                    A_x_z = min(1.,np.exp(N_z_x-D_x_z))

                # for statistics
                prob_accept[steps].append(float(A_x_z))

                if  np.random.rand() <= A_x_z:
                    accepted = True
                    break
                else : 
                    D_x_z  = D_x_z + np.log(1.-A_x_z)
                    r      = self._back_off(z_list,r_list,step_size,fancy)
                    steps += 1                     
            # end of steps for loop
            if accepted == True :
                num_accepted += 1    
                chain[i,:] = z[:,0] 
                state_x = copy.deepcopy(state_z)
                x,f_x   = get_vals(state_x)
                # for statistics
                step_of_accept[i] = steps
                step_count[steps] = step_count[steps] + 1
            else :
                chain[i,:]  = x[:,0] 
                # for statistics
                step_of_accept[i] = -1              
        # end outer loop

        # update stored info
        self._state_x = state_x

        # outputs
        if self._num_samples == 0 :
            self._chain = chain
            self._step_of_accept = step_of_accept
            self._step_count = step_count
        else :
            self._chain = np.append(self._chain, chain, axis=0)
            self._step_of_accept = np.append(self._step_of_accept, 
                                            step_of_accept, axis=0)
            self._step_count = np.append(self._step_count, step_count, 
                                        axis=0)
        self._num_samples += num_samples
        self._num_accepted += num_accepted
        self._prob_accept.extend(prob_accept)
    # end sampler

    def vsample(self, num_samples, divs=100):
        '''
    Vsample
        Provides simple sampling info while sampling
    Inputs  : 
        num_samples :
            number of samples to generate
    Optional Inputs : 
        divs : (100)
            number of divisions
        '''
        print("Sampling: 0%")
        for i in xrange(divs):
            self.sample(int(num_samples/divs))
            sys.stdout.write("\033[F") # curser up
            print("Sampling: "+str(int(i*100./divs)+1)+'%')
        self.sample(num_samples % divs)
        
    def burn(self, num_burned):
        '''
    Burn
        Burn the inital samples to adjust for convergence of the chain
        cut the first (num_burned) burn-in samples
    Inputs  :
        chain : 
            the full Markov chain
        num_burned :
            number of samples to cut 
    Hidden Outputs :
        chain : 
            chain with the firt num_burned samples cut
        '''
        self._chain = self._chain[num_burned:]

    def acor(self, k = 5):
        '''
    Autocorrelation time of the chain
        return the autocorrelation time for each parameters
    Inputs :
        k : 
            parameter in self-consistent window
    Outputs :
        t :
            autocorrelation time of the chain
        '''
        try:
            import acor
        except ImportError:
            print "Can't import acor, please download."
            return 0
        n = np.size(self._chain)[1]
        t = np.zeros(n)
        for i in xrange(n):
            t[i] = acor.acor(self._chain[:,i],k)[0]
        return t

    def posterior(self, x):
        '''
    Posterior density 
        ** not normalized **
        This is used to plot the theoretical curve for tests.
    Inputs  :
        x : 
            input value 
    Outputs : 
        p_x_y : p(x|y)=p(x)*exp{-||f(x)-y||^2/(2s^2)}
            posterior probability of x given y
        '''
        y = self._y
        s = self._s
        m = self._m
        H = self._H

        x = np.reshape(np.array(x), (-1,1))

        f_defined,f_x,J_x = self._f(x)
        if f_defined :
            p_x   = np.exp(-np.dot((x-m).T,np.dot(H,x-m))/2.)
            p_x_y = p_x*np.exp(-la.norm(f_x-y)**2/(2.*s**2))
            return p_x_y
        else :
            return 0
    
    def error_bars(self, num_bins, plot_range):
        '''
    Error Bars
        create bars and error bars to plot
    Inputs  :
        num_bins   :
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
        '''
        # fetch data
        chain    = self._chain
        length   = len(chain)
        try:
            num_dims = np.size(chain)[1]
        except:
            num_dims = 1

        # begin checks
        try: 
            assert num_bins == int(num_bins)
        except: 
            print "Number of bins has to be an integer."
            return 0
        plot_range = np.array(plot_range)
        try: 
            assert np.shape(plot_range) == (num_dims, 2)
        except: 
            print "Plot range shape has to be (number of dimensions, 2)."
            return 0
        # end checks

        # initialize outputs
        p_x   = np.zeros((num_dims, num_bins))    # esitmate of posterior
        error = np.zeros((num_dims, num_bins))    # error bars
        x     = np.zeros((num_dims, num_bins))    # centers of bins

        # loop through dimensions
        for dim in xrange(num_dims):
            x_min = plot_range[dim][0]
            x_max = plot_range[dim][1]
            dx    = float(x_max-x_min)/num_bins
            # bin count 
            for i in xrange(length):
                if chain[i][dim] > x_min and chain[i][dim] < x_max:
                    bin_no = int((chain[i][dim]-x_min)/dx)
                    p_x[dim][bin_no] += 1.
            # end count
            p_x[dim] = p_x[dim]/(length*dx)
            # find error
            for i in xrange(num_bins):
                p             = p_x[dim][i]
                error[dim][i] = np.sqrt(p*(1./dx-p)/(length))
                x[dim][i]     = x_min+(0.5+i)*dx
            # end find
        # end outer loop
        return x, p_x, error

    """
    Properties:
    1.  chain
    2.  num_samples
    3.  num_accepted
    4.  accept_rate
    5.  step_of_accept
    6.  step_count
    7.  prob_accept
    """

    @property
    def chain(self):
        return self._chain

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_accepted(self):
        return self._num_accepted

    @property
    def accept_rate(self):
        return float(self._num_accepted)/self._num_samples

    @property
    def step_of_accept(self):
        return self._step_of_accept

    @property
    def step_count(self):
        return self._step_count
    
    @property
    def prob_accept(self):
        return self._prob_accept

    # internal methods
    
    def _log_post(self,x,f_x):
        '''
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
        '''
        m     = self._m
        H     = self._H
        s     = self._s
        y     = self._y

        # prior part           -(x-m)'H(x-m)/2
        part1 = -np.dot((x-m).T,np.dot(H,x-m))/2.
        # least squares part   -||f(x)-y||^2/(2s^2)
        part2 = -la.norm(f_x-y)**2/(2.*s**2)
        return self._ln_H_+part1+part2

    def _proposal_params(self,state):
        '''
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
        '''
        m  = self._m
        H  = self._H
        s  = self._s
        y  = self._y
        Hm = self._Hm
        x  = state['x']
        f  = state['f']
        J  = state['J']

        A  = J/s                          # A = f'(x)/s
        b  = -np.dot(A,x)+(f-y)/s         # b = [f(x)-y-f'(x)x]/s
        L  = la.cholesky(H+np.dot(A.T,A)) # P = H+A'A = LL'   
        mu = la.solve(L.T,la.solve(L,Hm-np.dot(A.T,b)))
        state['L']  = L                   # mu_x = P^-1(Hm-A'b)
        state['mu'] = mu
        # state['ln_L'] = np.log(det(L))
        state['post'] = self._log_post(x,f)
        return state

    def _back_off(self,list_states,r_list,step_size,fancy):
        '''
    Back off
        Calculate the back off step size
    Inputs : 
        list_states :
            list of states in current proposal
        r_list : 
            list of back offs in current proposal
        step_size : 
            current step size
        fancy : 
            set to True if you want to use the fancy back-off
    Outputs : 
        '''
        r = r_list[len(r_list)-1]
        if fancy:
            p_0   = la.norm(list_states[0]['f']-self._y)
            dp_0  = p_0*2*la.norm(list_states[0]['J'])
            p_r   = la.norm(list_states[len(list_states)-1]['f']-self._y)
            dp_r  = p_0*2*la.norm(list_states[len(list_states)-1]['J'])
            r_new = optimize(r,p_0**2,dp_0,p_r**2,dp_r)
            return r_new
        else :
            return r*step_size    
