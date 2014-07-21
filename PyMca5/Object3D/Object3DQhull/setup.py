#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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

"""Setup script for the Object3DQhull module distribution."""

import os, sys, glob
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError, text

from distutils.core import setup
from distutils.extension import Extension

sources = glob.glob('./src/*.c')

#Work with doubles
define_macros = []
setup (
        name         = "Object3DQhull",
        version      = "1.0",
        description  = "Interface to Qhull library.",
        author       = "V.A. Sole - Software Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr",

        # Description of the modules and packages in the distribution
        ext_modules  = [
                       Extension(
                            name          = 'Object3DQhull',
                            sources       = sources,
                            define_macros = define_macros,
                            libraries  = [],
                            include_dirs  = [numpy.get_include()]
                       ),
       ],
)

#Work with floats
define_macros = [('Object3DFloat', None)]
setup (
        name         = "Object3DQhullf",
        version      = "1.0",
        description  = "Interface to Qhull library using floats.",
        author       = "V.A. Sole - BLISS Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

        # Description of the modules and packages in the distribution
        ext_modules  = [
                       Extension(
                            name          = 'Object3DQhullf',
                            sources       = sources,
                            define_macros = define_macros,
                            libraries  = [],
                            include_dirs  = [numpy.get_include()]
                       ),
       ],
)
