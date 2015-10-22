#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os 

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command

def read(file): 
    f = open(file)
    r = f.read()
    f.close()
    return r

def readme():
    return(read('README.rst')+'\n\nChange Log\n---------\n\n'
            +read('HISTORY.rst'))

class CleanCommand(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

import re
vre = re.compile("__version__ = \"(.*?)\"")
m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "gnm", "__init__.py"))
version = vre.findall(m)[0]

setup(
	name='gnm',
	version='0.0.1',
	description='Rock n Rolling awesome affine-invariant MCMC Sampler',
	long_description=readme(),
	url='http://github.com/mugurbil/gnm',
	author='Mehmet Ugurbil',
	author_email='mu388@nyu.edu',
	license='MIT',
	packages=['gnm'],
    package_data={'': ['LICENSE', 'AUTHORS.rst']},
    include_package_data=True,
	install_requires=["numpy"],
    classifiers=[
        "Development Status :: 0 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Topic :: MCMC Sampling"
    ],
    cmdclass={'clean': CleanCommand,}
    )
