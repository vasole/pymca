import glob
import os
import sys
if "bdist_wheel" in sys.argv:
    from setuptools import setup, Extension
else:
    try:
        from setuptools import setup, Extension
    except ImportError:
        from distutils.core import setup, Extension
import numpy

if sys.platform.startswith("win"):
    extra_compile_args = ["/openmp"]
    extra_link_args= []
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args=['-fopenmp']

include_dirs = [numpy.get_include()]
#define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
define_macros = []

def buildExtension():
    module = Extension(name="kmeans",
                    sources=['kmeans.pyx'],
                    include_dirs=include_dirs,
                    define_macros=define_macros,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    #language="c++",
                    )
    return module

ext_modules = [buildExtension()]

setup(name='kmeans', ext_modules=ext_modules)
