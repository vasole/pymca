try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError, text

from distutils.core import setup
from distutils.extension import Extension

data_files  = ('signal',['__init__.py', 'median.py'])
setup( name = 'signal',
      ext_modules = [Extension(
                  name = 'signal.mediantools',
                  sources = ['mediantools.c', 'medianfilter.c'],
                  define_macros = [],
                  include_dirs = [numpy.get_include()]
                  ),],
       data_files=data_files)
