#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
import os
from PyMca5 import PyMcaDataDir
from PyMca5.PyMcaGui import PyMcaQt as qt

class ChangeLog(qt.QTextDocument):
    def __init__(self, parent=None, textfile = None):
        qt.QTextDocument.__init__(self, parent)
        if textfile is not None:
            self.setTextFile(textfile)

    def setTextFile(self, textfile):
        if not os.path.exists(textfile):
            textfile = os.path.join(PyMcaDataDir.PYMCA_DATA_DIR, textfile)
        f = open(textfile)
        lines = f.readlines()
        f.close()
        text = ""
        for line in lines:
            text += "%s" % line
        self.setPlainText(text)

def test():
    app = qt.QApplication([])
    w = qt.QTextEdit()
    if len(sys.argv) > 1:
        log = ChangeLog(textfile=sys.argv[-1])
    else:
        log = ChangeLog(textfile='changelog.txt')
    w.setDocument(log)
    w.show()
    app.exec()

if __name__ == "__main__":
    test()
