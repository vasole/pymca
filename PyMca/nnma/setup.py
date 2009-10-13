#!/usr/bin/env python
#encoding:latin-1

#from distutils.core import setup
from setuptools import setup

setup(name='py_nnma',
      version='2.1',
      description='Nonnegative matrix approximation',
      author='Uwe Schmitt',
      author_email='uschmitt@mineway.de',
      packages=['py_nnma'],
      install_requires = ["numpy>=1.0.0", "scipy>=0.6.0"],
     )


