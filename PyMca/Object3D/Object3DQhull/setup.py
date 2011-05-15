#!/usr/bin/env python

"""Setup script for the BlissQhull module distribution."""

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
        author       = "V.A. Sole - BLISS Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

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
