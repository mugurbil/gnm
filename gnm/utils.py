#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Here we provide the functions that the gnm sampler needs
'''

__all__=['update_params','get_vals','log_expo','multi_normal',
            'det','optimize','function']

import numpy as np
la = np.linalg
try:
    from scipy import integrate
except:
    integrate = None

def update_params(state, t):
    '''
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
    '''
    mu = (1.-t)*state['x'] + t*state['mu']
    L  = state['L'] / np.sqrt(2.*t - t**2)
    return mu, L

def get_vals(state):
    '''
    Get Values
        Get the values out from the state to use for other purposes
    Inputs:
        state :
            the state that keeps the values
    Outputs:
        x :
            the value
        f  : f(x)
            the function value at x 
    '''
    x  = state['x']
    f  = state['f']
    return x, f

def log_expo(z,mu,L):
    '''
    Log Expo
        Log of the proposal probability density function for gnm
    Inputs :
        z :
            location
        mu :
            mean of the proposal
        L :
            cholesky factor of the precision matrix
    Outputs : 
        log of the probability density function
    '''
    return np.log(det(L))-la.norm(np.dot(L.T,z-mu))**2/2. 

def multi_normal(m,L):
    ''' 
    Multivariate normal sampler:
        Generates normal samples with mean m, precision matrix LL' 
    Inputs:
        m :
            mean
        L :
            cholesky factor of the precision matrix
    Outputs:
        normal with mean m and precision LL'
    '''
    Z = np.random.standard_normal(np.shape(m)) # generate i.i.d N(0,1)
    return  la.solve(L.T,Z)+m

def det(L):
    '''
    Determinant
        Compute the determinant given a lower triangular matrix
    Inputs: 
        L :
            lower triangular matrix
    Outputs: 
        det_L : 
            determinant of L
    '''
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

def optimize(alpha, f_0, d_f_0, f_alpha, d_f_alpha, 
        t1=0.1, t2=0.5):
    '''
    Third order approximation to find the minimum of the function
        f : function to be optimized over
    Inputs :  
        alpha : 
            previous step size
        f_0 : f(0),
            function value at 0
        d_f_0: f'(0),
            the derivative of the function at 0
        f_alpha : f(alpha),
            function value at alpha
        d_f_alpha : f'(alpha),
            the derivative of the function at alpha
        gamma1, gamma2 :
            the new step size will be picked in [gamma1*alpha,gamma2*alpha] 
        t1 : 
            step size reduction if minimum is at 0 or it can't be found  
        t2 : 
            step size reduction if minimum is at 1
    Outputs :
        alpha_new : 
            the new step size that minimizes the function
    '''
    if alpha == 0 :
        print("Error: please enter non-zero alpha")
        return alpha

    a = (alpha*d_f_alpha-2*f_alpha+2*f_0+alpha*d_f_0)/(alpha**3)
    b = (f_alpha-f_0-alpha*d_f_0)/(alpha**2)
    c = d_f_0
    A = 3*a
    B = b-alpha*a
    C = B**2-A*c

    if C == 0. :
        if c>0 or d_f_alpha>0 :
            alpha_new = t1*alpha
        else :
            alpha_new = t2*alpha
    elif A == 0 :
        alpha_new = -c/2./B 
    elif C > 0 :
        alpha_new = (-B+np.sqrt(C))/A
    else : 
        alpha_new = t1*alpha

    # check the bounds on new step size
    if   alpha_new < t1*alpha :
        alpha_new = t1*alpha
    elif alpha_new > t2*alpha :
        alpha_new = t2*alpha

    return alpha_new


class function:

    def __init__(self, f, args):
        '''
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
                f_defined_x : 
                    boolean flag indicating whether the function is 
                    defined at x or not
                f_x : f(x), 
                    function value at x
                J_x : f'(x), 
                    the jacobian of the function evaluated at x
            Demo : 
                f_defined_x, f_x, J_x = f(x,args)
            ---
        args : 
            the arguments that the user defined function takes        
        '''
        self._f     = f 
        self._args  = args
        self._count = 0

    def __call__(self, x):
        '''
    Call
        Calls the user defined function
    Inputs:
        x : 
            input value
    Outputs: 
        f_defined_x, f_x, J_x = f(x,args)
        '''
        self._count += 1
        x = np.reshape(np.array(x), (-1)) 
        f_defined_x, f_x, J_x = self._f(x, self.args)
        f_x = np.reshape(np.array(f_x), (-1,1))
        return f_defined_x, f_x, np.array(J_x)
        
    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
            p=2, l_max=50, r=0.5): 
        """
    Gradient Checker
        Test the function's jacobian against the numerical jacobian
    Inputs :
        xmin : 
            lower bound on the domain
        xmax : 
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
        # begin checks
        try :
            assert (np.size(x_min) == np.size(x_max))
        except :
            print("Error: Dimensions of x_min (%d) and x_max (%d) "
                  "are not the same." % ( np.size(x_min), np.size(x_max) ) )
            return 1 
        x_min = np.reshape(np.array(x_min), (-1)) 
        x_max = np.reshape(np.array(x_max), (-1)) 
        try: 
            for i in xrange(np.size(x_min)):
                assert x_min[i] < x_max[i]
        except: 
            print("Error: All values of x_min should be less than the "
                +"corresponding values for x_max.")
            return 1
        try :
            f_defined_x = False
            while not f_defined_x : 
                x = np.random.uniform(x_min,x_max)
                f_defined_x, f_x, J_x = self.__call__(x)
                assert J_x.ndim == 2
        except : 
            print("Error: Jacobian has to be a matrix.")
            return 1
        # end checks
        # begin test
        k = 0 
        while k < N :
            # reset dx each time
            Dx = (x_max - x_min) * dx

            # Sample random points in the state space
            x = np.random.uniform(x_min + Dx, x_max - Dx) 

            # Compute jacobian at x
            l = 0
            test_convergence = 1
            while test_convergence:
                f_defined_x, f_x, J_x = self.__call__(x)
                grad_f_x = J_x * 0.
                for j in xrange(np.size(x)):
                    # calculate the derivative of each component of f
                    delta_x = Dx * 0. 
                    delta_x[j] = Dx[j] 
                    f_defined_xr, f_xr, J_xr = self.__call__(x + delta_x) 
                    f_defined_xl, f_xl, J_xl = self.__call__(x - delta_x)
                    # check if the function is defined on these points
                    if( not(f_defined_x and f_defined_xr and f_defined_xl)):
                    # discard this trial if one of the values is not defined
                        test_convergence = 0 # break outer loop
                        break
                    delta_f = (f_xr - f_xl) / (2. * delta_x[j])
                    grad_f_x[:,j] = delta_f[:,0]
                eps = la.norm(grad_f_x - J_x, p) / np.size(J_x)
                if (eps < eps_max):
                    test_convergence = 0 # break outer loop
                    k += 1
                else:
                    Dx = Dx * r
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
