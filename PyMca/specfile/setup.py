#!/usr/bin/env python
"""Setup script for the SPECFILE module distribution."""

import os, sys, glob
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError(text)

from distutils.core import setup
from distutils.extension import Extension

SPECFILE_USE_GNU_SOURCE = os.getenv("SPECFILE_USE_GNU_SOURCE")
if SPECFILE_USE_GNU_SOURCE is None:
    SPECFILE_USE_GNU_SOURCE = 0
    if sys.platform.lower().startswith("linux"):
        print("WARNING:")
        print("A cleaner locale independent implementation")
        print("may be achieved setting SPECFILE_USE_GNU_SOURCE to 1")
        print("For instance running this script as:")
        print("SPECFILE_USE_GNU_SOURCE=1 python setup.py build")
else:
    SPECFILE_USE_GNU_SOURCE = int(SPECFILE_USE_GNU_SOURCE)

srcfiles = [ 'sfheader','sfinit','sflists','sfdata','sfindex',
             'sflabel' ,'sfmca', 'sftools','locale_management','specfile_py']

sources = [] 
for ffile in srcfiles:
  sources.append('src/'+ffile+'.c')

if sys.platform == "win32":
    define_macros = [('WIN32',None)]    
elif os.name.lower().startswith('posix'):
    define_macros = [('SPECFILE_POSIX', None)]
    #this one is more efficient but keeps the locale
    #changed for longer time
    #define_macros = [('PYMCA_POSIX', None)]
    #the best choice is to have _GNU_SOURCE defined
    #as a compilation flag because that allows the
    #use of strtod_l
    if SPECFILE_USE_GNU_SOURCE:
        define_macros = [('_GNU_SOURCE', 1)]        
else:
    define_macros = []
setup (
        name         = "specfile",
        version      = "3.2",
        description  = "module to read SPEC datafiles",
        author       = "BLISS Group",
        author_email = "rey@esrf.fr",
        url          = "http://www.esrf.fr/computing/bliss/",
        ext_modules  = [
                       Extension(
                            name          = 'specfile',
                            sources       = sources,
                            define_macros = define_macros,
                            include_dirs  = ['include', numpy.get_include()],
                       ),
       ],
)
