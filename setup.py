#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import re

try:
    from setuptools import setup, Command
    setup
except ImportError:
    from distutils.core import setup, Command
    setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def read(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def read(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

def readme():
    return(read("README.rst")+"\n\n"+
            "Change Log\n"+
            "----------\n\n"+
            read("HISTORY.rst"))

class CleanCommand(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

vre = re.compile("__version__ = \"(.*?)\"")
m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "gnm", "__init__.py"))
version = vre.findall(m)[0]

setup(
	name='gnm',
	version=version,
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7"
    ],
    cmdclass={'clean': CleanCommand,}
    )
