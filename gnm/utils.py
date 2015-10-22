#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Here we provide the functions that the gnm sampler needs
'''

__all__=['update_params','get_vals','log_expo','multi_normal',
            'det','optimize']

import numpy as np
la = np.linalg

def update_params(state,t):
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
    return mu,L

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
        gamma1 = 0.1**12, gamma2 = 0.5, t1=0.01, t2=0.5):
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
    if   alpha_new < gamma1*alpha :
        alpha_new = t1*alpha
    elif alpha_new > gamma2*alpha :
        alpha_new = t2*alpha

    return alpha_new


