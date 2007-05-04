#!/usr/bin/env python

"""Setup script for the SPECFITFUNS module distribution."""

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
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
else:
    define_macros = []
setup (
        name         = "SpecfitFuns",
        version      = "2.1",
        description  = "fit functions module",
        author       = "V.A. Sole - BLISS Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

        # Description of the modules and packages in the distribution
        ext_modules  = [
                       Extension(
                            name          = 'SpecfitFuns',
                            sources       = sources,
                            define_macros = define_macros,
                            include_dirs  = [numpy.get_include()]
                       ),
       ],
)
