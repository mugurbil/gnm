User Guide
==========

	1) Installation_

	.. _Installation: https://github.com/mugurbil/gnm/tree/master/Documentation#1-installation

		a) Requirements_

		.. _Requirements: https://github.com/mugurbil/gnm/tree/master/Documentation#a-requirements

		b) `Pip Installation`_

		.. _Pip Installation: https://github.com/mugurbil/gnm/tree/master/Documentation#b-pip-installation

		c) `Manual Installation`_

		.. _Manual Installation: https://github.com/mugurbil/gnm/tree/master/Documentation#c-manual-installation

		d) `Test the Installation`_

		.. _Test the Installation: https://github.com/mugurbil/gnm/tree/master/Documentation#d-test-the-installation

	2) Quickstart_

	.. _Quickstart: https://github.com/mugurbil/gnm/tree/master/Documentation#2-quickstart

1. Installation
===============

a) Requirements
---------------

To use the gnm package, the package numpy_ needs to be installed. 

.. _numpy: http://www.numpy.org/

For the examples and plotting the results, matplotlib_ is required. 

.. _matplotlib: http://matplotlib.org/

The autocorrelation feature demands the installation of the acor_ package.

.. _acor: http://www.math.nyu.edu/faculty/goodman/software/acor/

From the default packages, one will need os, setuptools (or distutils), re, sys, copy, and json. These packages likely come with any python installation.

b) Pip Installation
-------------------

The easiest way to install gnm would be to use pip_::

$ pip install gnm

.. _pip: https://pip.pypa.io/en/stable/

c) Manual Installation
----------------------

To download manually, use git clone or download as zip (see right hand side)::

$ git clone https://github.com/mugurbil/gnm.git

Then you can install manually by going into the gnm directory, and then runing setup.py::

$ python setup.py install

d) Test the Installation
------------------------

To check that the package is working, you can run a quickt est::

$ python -c 'import gnm; gnm.test()'

There should be some information on the terminal, and a plot should pop up that looks like this:

.. image:: https://github.com/mugurbil/gnm/blob/master/Documentation/gnm_test.png

2. Quickstart
=============

