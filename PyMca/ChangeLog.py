#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem to you.
#############################################################################*/
import sys
import os
import PyQt4.Qt as qt

class ChangeLog(qt.QTextDocument):
    def __init__(self, parent=None, textfile = None):
        qt.QTextDocument.__init__(self, parent)
        if textfile is not None:
            self.setTextFile(textfile)

    def setTextFile(self, textfile):
        if not os.path.exists(textfile):
            textfile = os.path.join(os.path.dirname(__file__), textfile)
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
    log = ChangeLog(textfile='changelog.txt')
    w.setDocument(log)
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
