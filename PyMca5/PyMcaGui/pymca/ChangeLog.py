#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
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
    app.exec_()

if __name__ == "__main__":
    test()
