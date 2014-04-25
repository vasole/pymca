import os
if os.path.exists(os.path.join(\
    os.path.dirname(os.path.dirname(__file__)), 'py2app_setup.py')):
    raise ImportError('PyMca cannot be imported from source directory')

from .PyMcaCore import Plugin1DBase, StackPluginBase, PyMcaDirs, DataObject

try:
    from .PyMcaIO import specfilewrapper, EdfFile, specfile, ConfigDict
except:
    print("WARNING importing IO directly")
    from PyMcaIO import specfilewrapper, EdfFile, specfile, ConfigDict

from .PyMcaMath.fitting import SpecfitFuns, Gefit, Specfit
from .PyMcaMath.fitting import SpecfitFunctions
from .PyMcaPhysics import Elements
