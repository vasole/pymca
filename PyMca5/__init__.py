import os
if os.path.exists(os.path.join(\
    os.path.dirname(os.path.dirname(__file__)), 'py2app_setup.py')):
    raise ImportError('PyMca cannot be imported from source directory')

try:
    from .PyMcaIO import specfilewrapper, EdfFile, specfile
except:
    print("WARNING importing IO directly")
    from PyMcaIO import specfilewrapper, EdfFile, specfile

from .PyMcaCore import Plugin1DBase, StackPluginBase
