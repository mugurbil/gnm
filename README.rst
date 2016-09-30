gnm: The MCMC Jagger 
====================

+----------------------+--------------------------------------------------------------------------------+
|.. image:: guitar.png | **Rock and Rolling awesome Python package for affine-invariant MCMC sampling** |
+----------------------+--------------------------------------------------------------------------------+

The gnm is a stable, well tested Python implementation of the affine-invariant sampler for Markov chain Monte Carlo (MCMC) that uses the Gauss-Newton-Metropolis (GNM) Algorithm.

The code is open source.

This python package is an affine invariant Markov chain Monte Carlo (MCMC) sampler based on the dynamic Gauss-Newton-Metropolis (GNM) algorithm. The GNM algorithm is specialized in sampling highly non-linear posterior probability distribution functions of the form :math:`e^{-||f(x)||^2/2}`, and the package is an implementation of this algorithm.

On top of the back-off strategy in the original GNM algorithm, there is the dynamic hyper-parameter optimization feature added to the algorithm and included in the package to help increase performance of the back-off and therefore the sampling. Also, there are the Jacobian tester, error bars creator and many more features for the ease of use included in the code. 

The problem is introduced and a guide to installation is given in the introduction. Then how to use the python package is explained. The algorithm is given and finally there are some examples using exponential time series to show the performance of the algorithm and the back-off strategy. 

Documentation
-------------

Read the guide at Documentation_.

.. _Documentation: https://github.com/mugurbil/gnm/tree/master/Documentation/#user-guide

Attribution
-----------

Goodman_

.. _Goodman: http://www.math.nyu.edu/faculty/goodman/

Copyright
---------

Copyright 2016 `Mehmet Ugurbil`_ and contributors.

.. _Mehmet Ugurbil: http://www.cims.nyu.edu/~mu388


License
-------

The gnm is free software made available under the `MIT LICENSE`_.

.. _MIT LICENSE: LICENSE.rst
