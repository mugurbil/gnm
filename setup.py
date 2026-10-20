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
    packages=['gnm'],
    package_data={'': ['LICENSE.rst', 'AUTHORS.rst', 'HISTORY.rst']},
    include_package_data=True,
    install_requires=["numpy"],
    extras_require={
        "scipy": ["scipy"],
        "dev": ["pytest", "scipy", "matplotlib"],
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
