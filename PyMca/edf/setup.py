#!/usr/bin/env python

"""Setup script for the SPECFILE module distribution."""

import os, sys, glob

from distutils.core import setup
from distutils.extension import Extension

sources = glob.glob('*.c')
if sys.platform == "win32":
    define_macros = [('WIN32',None)]
else:
    define_macros = []

setup (
        name         = "FastEdf",
        version      = "1.0",
        description  = "fit functions module",
        author       = "BLISS Group",
        author_email = "sole@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",

        # Description of the modules and packages in the distribution

        #extra_path   = 'Pybliss',
        ext_modules  = [
                       Extension(
                            name          = 'FastEdf',
                            sources       = sources,
                            define_macros = define_macros
                       ),
       ],
)
