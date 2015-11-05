#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import glob
import os
import sys
import numpy
from distutils.core import setup, Extension
try:
    html = False
    if html:
        import Cython.Compiler.Options
        Cython.Compiler.Options.annotate = True    
    from Cython.Distutils import build_ext
except:
    build_ext = None

c_files = [os.path.join('src', 'polspl.c'),
           os.path.join('src', 'bessel0.c')]
if build_ext:
    src = [os.path.join('cython', '_xas.pyx')]
else:
    src = glob.glob(os.path.join('cython', '*.c'))

src += c_files

if sys.platform == 'win32':
    extra_compile_args = []
    extra_link_args = []
else:
    # OpenMP and auto-vectorization flags for Colormap and MinMax
    # extra_compile_args = ['-fopenmp', '-ftree-vectorize']
    # extra_link_args = ['-fopenmp']
    extra_compile_args = []
    extra_link_args = []

setup(
    name='_xas',
    ext_modules=[Extension(
        name="_xas",
        sources=src,
        include_dirs=[numpy.get_include(),
                      os.path.join(os.getcwd(), "include")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )],
    cmdclass={'build_ext': build_ext},
)
