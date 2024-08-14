#!/usr/bin/env python
#/*##########################################################################
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

"""Setup script for the SPS module distribution."""

import os, sys, glob

try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError(text)

import platform
from distutils.core import setup
from distutils.extension import Extension

if platform.system() == 'Linux' :
    extra_compile_args = ['-pthread']
    #extra_compile_args = []
elif platform.system() == 'SunOS' :
    #extra_compile_args = ['-pthreads']
    extra_compile_args = []
else:
    extra_compile_args = []

ext_modules = [Extension(
                        name = 'spslut',
                        sources=['Src/sps_lut.c',
                                 'Src/spslut_py.c'],
                        extra_compile_args = extra_compile_args,
                        include_dirs  = ['Include', numpy.get_include()],
                   )]
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
else:
    define_macros = []
    ext_modules.append( Extension(
                            name = 'sps',
                            sources=['Src/sps.c',
                                     'Src/sps_py.c'],
                            extra_compile_args = extra_compile_args,
                            include_dirs  = ['Include', numpy.get_include()]))
setup (
        name         = "sps",
        version      = "1.0",
        description  = "shared memory and spec",
        author       = "BLISS Group",
        author_email = "rey@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

        # Description of the modules and packages in the distribution
        ext_modules  = ext_modules)
