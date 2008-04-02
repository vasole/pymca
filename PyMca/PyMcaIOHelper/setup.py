#!/usr/bin/env python

"""Setup script for the PyMcaIOHelper module distribution."""

import os, sys, glob
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError, text

from distutils.core import setup
from distutils.extension import Extension

sources = glob.glob('*.c')
define_macros = []
setup (
        name         = "PyMcaIOHelper",
        version      = "1.0",
        description  = "PyMca Input Output helper module",
        author       = "V.A. Sole - BLISS Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

        # Description of the modules and packages in the distribution
        ext_modules  = [
                       Extension(
                            name          = 'PyMcaIOHelper',
                            sources       = sources,
                            define_macros = define_macros,
                            libraries  = [], 
                            include_dirs  = [numpy.get_include()]
                       ),
       ],
)
