#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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
__version__ = "5.1.3"

import os
import sys
from PyMca5.PyMcaDataDir import PYMCA_DATA_DIR
try:
    from fisx.DataDir import FISX_DATA_DIR
except ImportError:
    FISX_DATA_DIR = None

DEBUG = 0

if sys.platform.startswith("win"):
    try:
        # make sure hdf5plugins are imported when using PyMca
        import hdf5plugin
    except:
        # but at the same time do not disturb other people
        pass
    import ctypes
    from ctypes.wintypes import MAX_PATH
elif sys.platform.startswith("darwin"):
    try:
        # make sure hdf5plugins are imported when using PyMca
        import hdf5plugin
    except:
        # but at the same time do not disturb other people
        pass

if os.path.exists(os.path.join(\
    os.path.dirname(os.path.dirname(__file__)), 'py2app_setup.py')):
    raise ImportError('PyMca cannot be imported from source directory')

def version():
    return __version__

def getDefaultSettingsFile():
    """
    Return the path to the default settings file (PyMca.ini).

    The file itself may not exist, but this function tries to create
    the containing directory if not already created.
    """
    filename = "PyMca.ini"
    if sys.platform == 'win32':
        # recipe based on: http://bugs.python.org/issue1763#msg62242
        dll = ctypes.windll.shell32
        buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
        if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
            directory = buf.value
        else:
            # the above should have worked
            home = os.getenv('USERPROFILE')
            try:
                l = len(home)
                directory = os.path.join(home, "My Documents")
            except:
                home = '\\'
                directory = '\\'
        if os.path.isdir('%s' % directory):
            directory = os.path.join(directory, "PyMca")
        else:
            directory = os.path.join(home, "PyMca")
        if not os.path.exists('%s' % directory):
            os.mkdir('%s' % directory)
        finalfile = os.path.join(directory, filename)
    else:
        home = os.getenv('HOME')
        directory = os.path.join(home, "PyMca")
        if not os.path.exists('%s' % directory):
            os.mkdir('%s' % directory)
        finalfile =  os.path.join(directory, filename)
    return finalfile

def getDefaultUserPluginsDirectory():
    """
    Return the default directory to look for user defined plugins.

    The directory will be created if not existing. In case of error it returns None.
    """
    try:
        settingsDir = os.path.dirname(getDefaultSettingsFile())
        if os.path.exists(settingsDir):
            userPluginDir = os.path.join(settingsDir, "plugins")
            if not os.path.exists(userPluginDir):
                os.mkdir(userPluginDir)
            return userPluginDir
        else:
            return None
    except:
        print("WARNING: Cannot initialize plugis directory")
        return None

def getUserDataFile(fileName, directory=""):
    """
    Look for an alternative to the given filename in the default user data directory
    """
    userDataDir = None
    try:
        settingsDir = os.path.dirname(getDefaultSettingsFile())
        if os.path.exists(settingsDir):
            userDataDir = os.path.join(settingsDir, "data")
            if not os.path.exists(userDataDir):
                os.mkdir(userDataDir)
    except:
        print("WARNING: cannot initialize user data directory")
        
    if userDataDir is None:
        return fileName

    baseName = os.path.basename(fileName)
    if len(directory):
        userDataFile = os.path.join(userDataDir, directory, baseName)
    else:
        userDataFile = os.path.join(userDataDir, baseName)
    if os.path.exists(userDataFile):
        if DEBUG:
            print("Using user data file: %s" % userDataFile)
        return userDataFile
    else:
        if DEBUG:
            print("Using data file: %s" % fileName)
        return fileName

def getDataFile(fileName, directory=None):
    """
    Look for the provided file name in directories following the priority:

    0 - The provided file
    1 - User data directory (~/PyMca/PyMcaData)
    2 - PyMca data directory (PyMcaDataDir.PYMCA_DATA_DIR)
    3 - fisx data directory (fisx.DataDir.FISX_DATA_DIR)
    """

    # return the input file name if exists
    if os.path.exists(fileName):
        if DEBUG:
            print("Filename as supplied <%s>" % newFileName)
        return fileName

    # the list of sub-directories where to look for the file
    if directory is None:
        directoryList = [""]
    else:
        directoryList = [directory, ""]

    # user
    for subdirectory in directoryList:
        newFileName = getUserDataFile(fileName, directory=subdirectory)
        if os.path.exists(newFileName):
            if DEBUG:
                print("Filename from user <%s>" % newFileName)
            return newFileName

    # PyMca
    for subdirectory in directoryList:
        newFileName = os.path.join(PYMCA_DATA_DIR,
                                   subdirectory,
                                   os.path.basename(fileName))
        if os.path.exists(newFileName):
            if DEBUG:
                print("Filename from PyMca Data Directory <%s>" % newFileName)
            return newFileName

    # fisx
    if FISX_DATA_DIR is not None:
        for subdirectory in directoryList:
            newFileName = os.path.join(FISX_DATA_DIR,
                                       subdirectory,
                                       os.path.basename(fileName))
            if os.path.exists(newFileName):
                if DEBUG:
                    print("Filename from fisx Data Directory <%s>" % \
                                                  newFileName)
                return newFileName

    # file not found
    txt = "File not found:  <%s>" % fileName
    if FISX_DATA_DIR is None:
        txt += " Please install fisx module (command 'pip install fisx [--user]')."
    raise IOError(txt)

# workaround matplotlib MPLCONFIGDIR issues under windows
if sys.platform.startswith("win"):
    try:
        #try to avoid matplotlib config dir problem under windows
        if os.getenv("MPLCONFIGDIR") is None:
            os.environ['MPLCONFIGDIR'] = os.path.dirname(getDefaultSettingsFile())
    except:
        print("WARNING: Could not set MPLCONFIGDIR.", sys.exc_info()[1])

# mandatory modules for backwards compatibility
from .PyMcaCore import Plugin1DBase, StackPluginBase, PyMcaDirs, DataObject

#convenience modules that could be directly imported
# using from PyMca5.PyMca import
try:
    from .PyMcaIO import specfilewrapper, EdfFile, specfile, ConfigDict
except:
    print("WARNING importing IO directly")
    from PyMcaIO import specfilewrapper, EdfFile, specfile, ConfigDict

from .PyMcaMath.fitting import SpecfitFuns, Gefit, Specfit
from .PyMcaMath.fitting import SpecfitFunctions

from .PyMcaPhysics.xrf import Elements

#all the rest can be imported using from PyMca5.PyMca import ...
from . import PyMca
