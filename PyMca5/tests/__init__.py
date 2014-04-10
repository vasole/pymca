import os
if os.path.exists('PyMca'):
    if os.path.exists('setup.py'):
        if os.path.exists('py2app_setup.py'):
            txt ='Tests cannnot be imported from top source directory'
            raise ImportError(txt)
from PyMca5.tests.TestAll import main as testAll
from PyMca5.tests.ConfigDictTest import test as testConfigDict
from PyMca5.tests.EdfFileTest import test as testEdfFile
from PyMca5.tests.ElementsTest import test as testElements
from PyMca5.tests.GefitTest import test as testGefit
from PyMca5.tests.PCAToolsTest import test as testPCATools
from PyMca5.tests.SpecfileTest import test as testSpecfile
from PyMca5.tests.specfilewrapperTest import test as testSpecfilewrapper
