gnm
===

**The Python sampling toolkit for affine-invariant MCMC**

The gnm is a stable, well tested Python implementation of the affine-invariant sampler for Markov chain Monte Carlo (MCMC) that uses the Gauss-Newton-Metropolis (GNM) Algorithm.

The code is open source.

Requirements
------------

To use the gnm package, the package numpy_ needs to be installed. 

.. _numpy: http://www.numpy.org/

For the examples and plotting the results, matplotlib_ is required. 

.. _matplotlib: http://matplotlib.org/

The autocorrelation feature demands the installation of the acor_ package.

.. _acor: http://www.math.nyu.edu/faculty/goodman/software/acor/

From the default packages, one will need os, setuptools (or distutils), re, sys, copy, and json. These packages likely come with any python installation.

Installation
------------

The easiest way to install gnm would be to use pip::

$ pip install gnm

To check that the package is working, you can run quickstart.py::

$ python -c 'import gnm; gnm.test()'

A plot should pop up.

.. image:: https://github.com/mugurbil/gnm/blob/master/Documentation/gnm_test.png

Manual Installation
-------------------

To download manually, use git clone or download as zip (see right hand side)::

$ git clone https://github.com/mugurbil/gnm.git

Then you can install manually by going into the gnm directory, and then runing setup.py::

$ python setup.py install

Documentation
-------------

Read the guide at Documentation_.

.. _Documentation: http://www.cims.nyu.edu/~mu388

Attribution
-----------

Goodman_

.. _Goodman: http://www.math.nyu.edu/faculty/goodman/

Copyright
---------

Copyright 2016 Mehmet_Ugurbil_ and contributors.

.. _Mehmet_Ugurbil: http://www.cims.nyu.edu/~mu388


License
-------

The gnm is free software made available under the MIT LICENSE_.

.. _LICENSE: LICENSE.rst
