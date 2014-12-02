#!/usr/bin/env python
import glob
import os
import sys
import numpy
from distutils.core import setup, Extension
try:
    from Cython.Distutils import build_ext
except:
    build_ext = None


c_files = glob.glob(os.path.join('src', 'InsidePolygonWithBounds.c'))
c_files += glob.glob(os.path.join('src', 'MinMaxImpl.c'))

if build_ext:
    src = glob.glob(os.path.join('cython', '_ctools.pyx'))
    src = glob.glob(os.path.join('cython', 'minMax.pyx'))
else:
    src = glob.glob(os.path.join('cython', '*.c'))

src += c_files

if sys.platform == 'win32':
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = []
    extra_link_args = []

setup(
    name='ctools',
    ext_modules=[Extension(
        name="_ctools",
        sources=src,
        include_dirs=[numpy.get_include(),
                      os.path.join(os.getcwd(), "include")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )],
    cmdclass={'build_ext': build_ext},
)
