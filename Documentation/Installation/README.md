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

A plot_ should pop up.

.. _plot: https://github.com/mugurbil/gnm/blob/master/Documentation/gnm_test.png

Manual Installation
-------------------

To download manually, use git clone or download as zip (see right hand side)::

$ git clone https://github.com/mugurbil/gnm.git

Then you can install manually by going into the gnm directory, and then runing setup.py::

$ python setup.py install