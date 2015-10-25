gnm
===

**The Python sampling toolkit for affine-invariant MCMC**

gnm is a stable, well tested Python implementation of the affine-invariant sampler for Markov chain Monte Carlo (MCMC) that uses the Gauss-Newton-Metropolis (GNM) Algorithm.

The code is open source.

Installation
------------

To use the gnm package you need to have the package numpy installed. To
use the examples and plot the results, you will need matplotlib. To use the
acor feature, you will need acor.
From the default packages, you will need os, setuptools (or distutils), re,
sys, and copy. These packages likely come with your python installation.
The easiest way to install gnm would be to use pip.
$ pip install gnm
If you want to download manually, you can nd the package from the website
http://cims.nyu.edu/~mu388/gnm. Then you can install manually by going
into the gnm directory, and then runing setup.py.
$ python setup.py install
To clean the repository after installation, one can run clean with setup.py.
$ python setup.py clean


Documentation
-------------

Read the guide at Documentation.

Attribution
-----------



License
-------

Copyright 2015-? Mehmet Ugurbil and contributors.

gnm is free software made available under the MIT License. For details see
the LICENSE file.
