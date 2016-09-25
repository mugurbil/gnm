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
import sys # for sample visual
import json # for save/load

from .utils import *

class sampler(object):

    def __init__(self, x, model, args):
    	""" 
    Init
        Initialize the GNM sampler class
	Inputs  :
        x : 
            initial guess
        model :
            user defined data model function
        args :
            arguments for the model
        """
        self._args = args
        self._f = function(model, args)

        x = np.reshape(np.array(x), (-1,1)) 
        self._n = np.size(x) # size of input space
        try: 
            x = np.reshape(np.array(x), (-1)) 
            out_x = model(x, self._args)
        except TypeError as e:
            raise TypeError(str(e)[:-7]+" needed)")

        """
        except IndexError as e:
            raise IndexError("initial guess size does not fit model()\n    "
                    +str(e))
        except Exception as e:
            print("Error: Model function could not be evaluated.")
            print("     - Check size of intial guess.")
            print("     - Check definition of the model.")
            print(str(e))
            print(type(e))
            raise RuntimeError("model() could not be evaluated\n   ")
        """

        try:
            chi_x, f_x, J_x = out_x
            f_x = np.reshape(np.array(f_x), (-1,1))
        except:
            raise TypeError("model() needs to have 3 outputs: chi_x, f_x, J_x")
        try:
            assert chi_x == True
        except AssertionError:
            raise ValueError("initial guess out of range")
        try: 
            self._mn = np.size(f_x)
            J_x = np.array(J_x)
            assert np.shape(J_x) == (self._mn, self._n)
        except:
            raise TypeError("Shape of Jacobian, " + str(np.shape(J_x)) + 
                  ", is not correct, (%d, %d)." % (self._mn, self._n))

        x = np.reshape(np.array(x), (-1,1)) 
        self._X = {'x':x,'f':f_x,'J':J_x} # state of x

        # prior parameters
        self._prior = False

        # back-off parameters
        self._max_steps = 1
        self._step_size = 0.1
        self._dynamic = False
        self._opts = {}

        # sampler outputs
        self._chain = None
        self._n_samples = 0
        self._n_accepted = 0

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
        if self._prior == True:
            raise Warning("prior information is already set")
        else: 
            self._prior = True

        # mean
        self._m = np.reshape(np.array(m), (-1,1))
        try : 
            assert np.size(self._m) == self._n
        except : 
            raise TypeError("mean has to be an array of size n")

        # precision
        self._H     = np.array(H)
        try : 
            assert np.shape(self._H) == (self._n, self._n)
        except : 
            raise TypeError("precision has to be a matrix of shape n by n")

        # precalculations
        self._ln_H_ = np.log(la.det(self._H))/2.
        self._Hm    = np.dot(self._H, self._m)

    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
            p=2, l_max=50, r=0.5):
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
        """
        # check inputs x_min and x_max
        try :
            assert np.size(x_min) == self._n
        except :
            raise TypeError("dimension of x_min, %d, does not match the "
                  "dimension of input, %d" % (np.size(x_min), self._n))
        try :
            assert np.size(x_max) == self._n
        except :
            raise TypeError("dimension of x_max, %d, does not match the "
                  "dimension of input, %d." % (np.size(x_max), self._n))
        # end checks and call developer function
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
        self._dynamic = False
        # begin checks      
        try : 
            self._max_steps = int(max_steps)
        except :
            print("Error: Input 1 (max_steps) has to be an int.")
            return 0
        try : 
            assert self._max_steps >= 0
        except : 
            print("Warning: Input 1 (max_steps) has to be non-negative.")
            print("Setting max_steps to 0.")
            self._max_steps = 0

        if max_steps > 0 :
            try : 
                assert step_size == float(step_size)
            except AssertionError :
                print("Warning: Input 2 (step_size) is not a float. Converted.")
                step_size = float(step_size)
            except :
                print("Error: Input 2 (step_size) has to be a float.")
                return 0

            try : 
                assert 0. < step_size < 1.
            except : 
                print("Warning: Input 2 (step_size) has to be between 0 and 1.")
                print("Setting step_size to 0.2.")
                step_size = 0.2
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
            maximum back-off steps to be taken
    Optional Inputs: 
        opts : ({})
        dictionary containing fancy options
        """
        self._dynamic = True

        # begin checks      
        try : 
            self._max_steps = int(max_steps)
        except :
            print("Error: Input 1 (max_steps) has to be an int.")
            return 0
        try : 
            assert self._max_steps >= 0
        except : 
            print("Warning: Input 1 (max_steps) has to be non-negative.")
            print("Setting max_steps to 0.")
            self._max_steps = 0

        self._opts = opts
        # end checks

    def sample(self, n_samples, divs=1, visual=False, safe=False):
        """
    Sample
        Sampling 
    Inputs  : 
        n_samples :
            number of samples to generate
    Optional Inputs : 
        divs : (1)
            number of divisions
        visual : 
            show progress 
        safe : 
            save the chain at every division
        """
        if visual: 
            print("Sampling: 0%")
        for i in xrange(divs):
            self._sample(int(n_samples/divs))
            if visual: 
                sys.stdout.write("\033[F") # curser up
                print("Sampling: "+str(int(i*100./divs)+1)+'%')
            if safe: 
                self.save(path="chain_{:}.dat".format(i))
        if n_samples % divs != 0:
            self._sample(n_samples % divs)
            if safe: 
                self.save(path="chain_{:}.dat".format(divs))

    def save(self, path="chain.dat"):
        """
    Save
        Save data to file
    Inputs  : 
        path :
            specifies the path name of the file to be loaded to
        """
        # create dictionary for data
        dic = {}
        dic['chain'] = self._chain.tolist()
        dic['step_count'] = self._step_count.tolist()
        dic['n_samples'] = self._n_samples
        dic['n_accepted'] = self._n_accepted
        dic['x'] = self._X['x'].tolist()
        dic['f'] = self._X['f'].tolist()
        dic['J'] = self._X['J'].tolist()

        # write data to file
        file = open(path, 'w')
        json.dump(dic, file)
        file.close()

    def load(self, path="chain.dat"):
        """
    Load
        Load data from file
    Inputs  : 
        path :
            specifies the path name of the file to be loaded from
        """
        # read data from file
        file = open(path, 'r')
        dic = json.load(file)
        file.close()

        # get data from dictionary
        self._chain = np.array(dic['chain'])
        self._step_count = np.array(dic['step_count'])
        self._n_samples = dic['n_samples']
        self._n_accepted = dic['n_accepted']
        self._X = {}
        self._X['x'] = dic['x']
        self._X['f'] = dic['f']
        self._X['J'] = dic['J']
        
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
        p : p(x)=pi(x)*exp{-||f(x)||^2/(2)}
            posterior probability of x
        """
        x = np.reshape(np.array(x), (-1,1))

        chi_x, f_x, J_x = self._f(x)
        if chi_x :
            p = np.exp(-la.norm(f_x)**2/2.)
            if self._prior:
                m = self._m
                H = self._H
                p = p * np.exp(-np.dot((x-m).T,np.dot(H,x-m))/2.)
            return p
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
        chain = self._chain
        len_chain = len(chain)
        try:
            n_dims = np.shape(chain)[1]
        except:
            n_dims = 1

        # begin checks
        try: 
            assert n_bins == int(n_bins)
        except: 
            raise TypeError("number of bins has to be an integer")
        d_min = np.reshape(np.array(d_min), (-1,1))
        d_max = np.reshape(np.array(d_max), (-1,1))
        try: 
            assert np.size(d_min) == n_dims
        except: 
            raise TypeError("domain minimum has wrong size")
        try: 
            assert np.size(d_max) == n_dims
        except: 
            raise TypeError("domain maximum has wrong size")
        # end checks

        # initialize outputs
        p_x = np.zeros(n_bins) # esitmate of posterior
        error = np.zeros(n_bins) # error bars
        x = np.zeros((n_dims, n_bins)) # centers of bins

        # set dx
        v = d_max-d_min
        v_2 = np.dot(v.T, v)[0][0]

        # bin count
        for i in xrange(len_chain):
            bin_no = int(np.floor(np.dot(chain[i].T-d_min,v)/v_2*n_bins)[0])
            if n_bins > bin_no > -1:
                p_x[bin_no] += 1.
        # end count
        dx = np.sqrt(v_2)/n_bins
        p_x = p_x/(len_chain*dx)
        # find error
        for i in xrange(n_bins):
            p = p_x[i]
            error[i] = np.sqrt(p*(1./dx-p)/(len_chain))
            x[:,i] = (d_min+v*(0.5+i)/n_bins)[0]
        # end find
        return x, p_x, error
    # end error_bars
    
    # internal methods
    def _sample(self, n_samples):
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
        X = self._proposal_params(self._X)
        k_max = self._max_steps

        # initialize 
        chain = np.zeros((n_samples, self._n)) 
        n_accepted = 0
        step_count = np.zeros(k_max+2)

        # begin outer loop
        for i in xrange(n_samples):
            accepted  = False       # check if sample is accepted
            r_        = [1]         # list of step sizes
            Z_        = [X]         # initialize list of Z s
            self._r_  = r_ 
            log_P_z_x = 0. + X['log_p'] 

            k = 0 # back-off steps taken so far
            while k <= k_max:
                # get proposal
                chi_z = False
                while not chi_z:
                    z = multi_normal(X, r_[-1])
                    chi_z, f_z, J_z = self._f(z)
                Z = self._proposal_params({'x':z,'f':f_z,'J':J_z})
                Z_.append(Z)
                self._Z_ = Z_

                log_P_z_x += log_K(Z, X, r_[-1])

                # N is the Numerator of the acceptance, N = P_x_z
                self._N_is_0 = False # check to see if N = 0, to use in _log_P
                log_N = self._log_P(X, Z, k)

                # calculating acceptance probability
                if self._N_is_0 == True :
                    A_z_x = 0.
                elif log_N >= log_P_z_x :
                    A_z_x = 1.
                else :
                    A_z_x = np.exp(log_N - log_P_z_x)

                # acceptance rejection
                if  np.random.rand() <= A_z_x:
                    accepted = True
                    break
                else : 
                    log_P_z_x += np.log(1. - A_z_x)
                    self._back_off()
                    k += 1                     
            # end of steps for loop
            if accepted == True :
                chain[i,:] = z[:,0] 
                X = Z
                # for statistics
                n_accepted += 1    
                step_count[k+1] += 1
            else :
                chain[i,:]  = X['x'][:,0]
                # for statistics
                step_count[0] += 1
        # end outer loop

        # update stored info
        self._X = X

        # outputs
        if self._n_samples == 0 :
            self._chain = chain
            self._step_count = step_count
        else :
            self._chain = np.append(self._chain, chain, axis=0)
            self._step_count = np.add(self._step_count, step_count)
        self._n_samples += n_samples
        self._n_accepted += n_accepted
    # end sample

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
            log_p : log(p(x))
                log of the posterior density
        """
        x  = state['x']
        f  = state['f']
        J  = state['J']
        JJ = np.dot(J.T,J)     

        if self._prior: 
            m  = self._m
            H  = self._H
            Hm = self._Hm
            # LL' = P = H+J'J 
            L  = la.cholesky(H+JJ)   
            # mu = (P^-1)(Hm-J'f+J'Jx)
            mu = la.solve(L.T,la.solve(L,Hm-np.dot(J.T,f)+np.dot(JJ,x))) 
        else: 
            # P = J'J
            L = la.cholesky(JJ)
            # mu = x-(P^-1)J'f
            mu = x-la.solve(L.T,la.solve(L,np.dot(J.T,f)))

        state['L'] = L        
        state['mu'] = mu
        state['log_p'] = self._log_post(x,f)
        return state

    def _log_P(self, X , Z, k):
        """
    Log of the probability of transition from z to x with k steps
        log ( P_k (x, z) )
    Inputs : 
        X :
            state to be proposed to
        Z : 
            state to be proposed from
        k : 
            number of recursions, depth
        """
        r_ = self._r_
        Z_ = self._Z_
        # zero case
        if k == 0 :
            log_P = Z['log_p'] + log_K(X, Z, r_[k])
        # recursice case
        else :
            P_zk_z = np.exp( self._log_P(Z_[k], Z, k-1) )
            P_z_zk = np.exp( self._log_P(Z, Z_[k], k-1) ) 
            # flag
            if P_zk_z <= P_z_zk :
                self._N_is_0 = True
                log_P = -np.inf
            else : 
                log_P = np.log( P_zk_z - P_z_zk ) + log_K(X, Z, r_[k])
        return log_P

    def _back_off(self):
        """
    Back off
        Calculate the back off step size
    Inputs : 
        Z_ :
            list of states in current proposal
        r_ : 
            list of back offs in current proposal
        q : 
            step size reduction
        dynamic : 
            set to True if you want to use the dynamic back-off
    Outputs : 
        """
        q = self._step_size
        r = self._r_[-1]
        Z_ = self._Z_
        if self._dynamic:
            p_0   = la.norm(Z_[0]['f'])
            dp_0  = p_0*2*la.norm(Z_[0]['J'])
            p_r   = la.norm(Z_[-1]['f'])
            dp_r  = p_0*2*la.norm(Z_[-1]['J'])
            r_new = optimize(r, p_0**2, dp_0, p_r**2, dp_r)
        else :
            r_new = r * q  
        self._r_.append(r_new)  

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
        log(p_x) : log[pi(x)]-||f(x)||^2/(2)
            log of the posterior probability
        """
        # least squares part   -||f(x)||^2/2
        log_likelihood = (-la.norm(f_x)**2)/(2.)

        # prior part           -(x-m)'H(x-m)/2
        if self._prior: 
            m = self._m
            H = self._H
            log_prior = self._ln_H_-np.dot((x-m).T,np.dot(H,x-m))/2.
            return log_prior+log_likelihood
        else:
            return log_likelihood

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