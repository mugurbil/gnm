#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
func is a class that creates instances of functions with specified arguments.

There are 3 operations on func: __init__, __call__, and Jtest 
There are 2 properties on func: f, args
Sample usage:
    f_0 = gnm.func(f,args)
    converged, error = f_0.Jtest()
'''

__all__ = ["func"]

import numpy as np
la = np.linalg

class func:

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
        self._f    = f 
        self._args = args

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
        x = np.reshape(np.array(x), (-1)) 
        f_defined_x, f_x, J_x = self._f(x, self.args)
        f_x = np.reshape(np.array(f_x), (-1,1))
        return f_defined_x, f_x, np.array(J_x)
        
    def Jtest(self, x_min, x_max, dx=0.0002, N=1000, eps_max=0.0001,
            p=2, l_max=50, d=0.5): 
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
    """
    
    @property
    def f(self):
        return self._f

    @property
    def args(self):
        return self._args

