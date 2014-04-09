#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
import unittest
import os
import sys
import glob
import unittest

def getSuite(auto=True):
    pythonFiles = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
    sys.path.insert(0, os.path.dirname(__file__))
    testSuite = unittest.TestSuite()
    for fname in pythonFiles:
        if os.path.basename(fname) in ["__init__.py", "TestAll.py"]:
            continue
        modName = os.path.splitext(os.path.basename(fname))[0]
        try:
            module = __import__(modName)
        except ImportError:
            print("Failed to import %s" % fname)
            continue
        if hasattr(module, "getSuite"):
            testSuite.addTest(module.getSuite(auto))
    return testSuite

def main(auto=True):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        auto = False
    else:
        auto = True
    main(auto)
