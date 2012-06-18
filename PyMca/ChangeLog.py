#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import sys
import os
from PyMca import PyMcaDataDir
from PyMca import PyMcaQt as qt

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
