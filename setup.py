# -*- coding: utf-8 -*-

import os
import re
import sys
from setuptools import Command, setup


def read(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()


def readme():
    return read("README.md")


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


vre = re.compile(r'__version__ = "(.*?)"')
m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnm", "__init__.py"))
version = vre.findall(m)[0]

setup(
    name='gnm',
    version=version,
    description='Rock n Rolling awesome affine-invariant MCMC Sampler',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/mugurbil/gnm',
    author='Mehmet Ugurbil',
    author_email='mugurbil@gmail.com',
    license='MIT',
    keywords=[
        'mcmc',
        'bayesian-inference',
        'gauss-newton',
        'metropolis-hastings',
        'affine-invariant',
        'inverse-problems',
        'nonlinear-least-squares',
    ],
    project_urls={
        'Homepage': 'https://github.com/mugurbil/gnm',
        'Repository': 'https://github.com/mugurbil/gnm',
        'Paper': 'https://arxiv.org/abs/2001.03530',
        'Issues': 'https://github.com/mugurbil/gnm/issues',
        'Author': 'https://mugurbil.github.io/',
    },
    packages=['gnm'],
    package_data={'': ['LICENSE.rst', 'AUTHORS.rst', 'HISTORY.rst']},
    include_package_data=True,
    install_requires=["numpy>=1.20"],
    extras_require={
        "scipy": ["scipy>=1.7"],
        "fast": ["numba>=0.57"],
        "dev": ["pytest>=7.0", "scipy>=1.7", "matplotlib>=3.5", "numba>=0.57"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={'clean': CleanCommand},
)
