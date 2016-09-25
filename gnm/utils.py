#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Here we provide the functions that the gnm sampler needs
"""

__all__ = ['test','update_params','log_K','multi_normal', 'det','optimize','function']

import numpy as np
la = np.linalg

def test():
    import quickstart
    
def update_params(state, t):
    """
    Update parameters
        updates mean and precision to the step size
    Inputs:
        state :
            mu :
                mean
            L  :
                cholesky factor of the precision matrix
        t : 
            step size
    Outputs:
        mu :
            updated mean
        L  :
            updated cholesky factor of the precision matrix  
    """
    mu = (1.-t)*state['x'] + t*state['mu']
    L  = state['L'] / np.sqrt(2.*t - t**2)
    return mu, L

def log_K(Z, X, t):
    """
    Log K
        Log of the proposal probability density function for gnm
    Inputs :
        Z :
            proposed to
        x : 
            proposed from 
    Outputs :   
        log of the probability density function
    """
    m, L = update_params(X, t)
    z = Z['x']
    return np.log(det(L))-la.norm(np.dot(L.T,z-m))**2/2. 

def multi_normal(X, t):
    """
    Multivariate normal sampler:
        Generates normal samples with mean m, precision matrix LL' 
    Inputs:
        x : 
            propose from 
    Outputs:
        normal with mean m and precision LL'
    """
    m, L = update_params(X, t)
    z = np.random.standard_normal(np.shape(m)) # generate i.i.d N(0,1)
    return  la.solve(L.T,z)+m

def det(L):
    """
    Determinant
        Compute the determinant given a lower triangular matrix
    Inputs: 
        L :
            lower triangular matrix
    Outputs: 
        det_L : 
            determinant of L
    """
    size_L = L.shape
    if np.size(L) == 1:
        return np.array(L)
    else:    
        try: 
            assert np.all(np.tril(L)==L)
        except AssertionError:
            print 'Error: Input is not a lower triangular matrix.'
            return 0
        try:
            assert size_L[0] == size_L[1]
        except AssertionError:
            print 'Error: Not a square matrix.'
            return 0
        det_L = 1.
        for i in xrange(size_L[1]):
            det_L = det_L*L[i,i]
        return det_L

def optimize(t, f_0, d_f_0, f_t, d_f_t, t1=0.05, t2=0.5):
    """
    Third order approximation to find the minimum of the function
        f : function to be optimized over
    Inputs :  
        t : 
            previous step size
        f_0 : f(0),
            function value at 0
        d_f_0: f'(0),
            the derivative of the function at 0
        f_t : f(t),
            function value at t
        d_f_t : f'(t),
            the derivative of the function at t
        t1 : 
            step size reduction if minimum is at 0 or it can't be found  
        t2 : 
            step size reduction if minimum is at 1
    Outputs :
        alpha_new : 
            the new step size that minimizes the function
    """
    if t <= 0 :
        print("Error: please enter non-negative t")
        return t

    a = (t*d_f_t-2*f_t+2*f_0+t*d_f_0)/(t**3)
    b = (f_t-f_0-t*d_f_0)/(t**2)
    c = d_f_0
    A = 3*a
    B = b-t*a
    C = B**2-A*c

    if C == 0. :
        if c>0 or d_f_t>0 :
            t_new = t1*t
        else :
            t_new = t2*t
    elif A == 0 :
        t_new = -c/2./B 
    elif C > 0 :
        t_new = (-B+np.sqrt(C))/A
    else : 
        t_new = t1*t

    # check the bounds on new step size
    if   t_new < t1*t :
        t_new = t1*t
    elif t_new > t2*t :
        t_new = t2*t

    return t_new


class function(object):

    def __init__(self, f, args):
        """
    Init 
        Initialize the developer function class 
    Inputs :
        f : user defined function
            ---
            Inputs of f :  
                x : 
                    input value
                args : 
                    the arguments that the function takes
            Outputs of f :
                chi_x : 
                    boolean flag indicating whether the function is 
                    defined at x or not
                f_x : f(x), 
                    function value at x
                J_x : f'(x), 
                    the jacobian of the function evaluated at x
            Demo : 
                chi_x, f_x, J_x = f(x,args)
            ---
        args : 
            the arguments that the user defined function takes        
        """
        self._f     = f 
        self._args  = args
        self._count = 0

    def __call__(self, x):
        """
    Call
        Calls the user defined function
    Inputs:
        x : 
            input value
    Outputs: 
        chi_x, f_x, J_x = f(x,args)
        """
        self._count += 1
        x = np.reshape(np.array(x), (-1)) 
        chi_x, f_x, J_x = self._f(x, self.args)
        f_x = np.reshape(np.array(f_x), (-1,1))
        return chi_x, f_x, np.array(J_x)
        
    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
            p=2, l_max=50, r=0.5): 
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
    Inputs :
        x_min : 
            lower bound on the domain
        x_max : 
            upper bound on the domain
        ** Warning **
            x_min and x_max must be arrays of the same dimension
    Optional inputs :
        dx : (2*10^-4)
            the ratio of dx to the size of the domain
        N : (1000)
            number of test points
        eps_max : (10^-4)
            the maximum value error is allowed to be to confirm convergence
        p : (2)
            to specify the norm of the error (p-norm)
        l_max : (40)
            maximum number of tries to reduce dx
        r : (0.5)
            dx will be multiplied by this constant each step the error 
            exceeds error_bound until l_max is reached
        ** Warning **
            keep in mind machine precision when changing l_max and r 
    Outputs : 
        error : 
          * 1 if did it did not pass the checks,
          * 0 if converged, 
          * eps, the error of the numerical gradient point if no convergence
        """
        x_min = np.reshape(np.array(x_min), (-1)) 
        x_max = np.reshape(np.array(x_max), (-1)) 
        # begin checks
        try: 
            for i in xrange(np.size(x_min)):
                assert x_min[i] < x_max[i]
        except: 
            print("Error: All values of x_min should be less than the "
                 +"corresponding values for x_max.")
            exit(0)
        # end checks
        # begin test
        k = 0 
        while k < N :
            # reset dx each time
            D_x = (x_max - x_min) * dx

            # Sample random points in the state space
            x = np.random.uniform(x_min + D_x, x_max - D_x) 

            # Compute jacobian at x
            l = 0
            test_convergence = 1
            while test_convergence:
                chi_x, f_x, J_x = self.__call__(x)
                D_f_x = J_x * 0.
                for j in xrange(np.size(x)):
                    # calculate the derivative of each component of f
                    d_x = D_x * 0. 
                    d_x[j] = D_x[j] 
                    chi_x_r, f_x_r, J_x_r = self.__call__(x + d_x) 
                    chi_x_l, f_x_l, J_x_l = self.__call__(x - d_x)
                    # check if the function is defined on these points
                    if( not(chi_x and chi_x_r and chi_x_l)):
                    # discard this trial if one of the values is not defined
                        test_convergence = 0 # break outer loop
                        break
                    d_f = (f_x_r - f_x_l) / (2. * d_x[j])
                    D_f_x[:,j] = d_f[:,0]
                eps = la.norm(D_f_x - J_x, p) / np.size(J_x)
                if (eps < eps_max):
                    test_convergence = 0 # break outer loop
                    k += 1
                else:
                    D_x = D_x * r
                    if (l > l_max): # numerical gradient did not converge 
                        return eps
                    l += 1
        return 0
        # end test

    """
    Properties:
    1.  f
    2.  args
    3.  count
    """
    
    @property
    def f(self):
        return self._f

    @property
    def args(self):
        return self._args

    @property
    def count(self):
        return self._count
## end function ##