#!/usr/bin/env python
__revision__ = "$Revision: 1.4 $"
#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
from MyEdfFileSelector import *

class EdfFileSimpleViewer(EdfFileSelector):
    def __init__(self, parent=None, name="EdfSelector", fl=0):
        EdfFileSelector.__init__(self,parent,name,fl,justviewer=1)   
        if 1 or  qt.qVersion() < '3.0.0':
            wid= self.getParamWidget("array")
            wid.iCombo.setMinimumWidth(wid.iCombo.sizeHint().width()*3)
        
    def setFileList(self, filelist):
        for file in filelist:
            self.openFile(file, justloaded=1)
            


if __name__ == "__main__":
    import sys
    import EdfFileLayer
    import getopt
    app=qt.QApplication(sys.argv) 
    winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
    app.setPalette(winpalette)
    options=''
    longoptions=[]
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)      
    for opt,arg in opts:
        pass
    filelist=args
    qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
    w=EdfFileSimpleViewer()
    if qt.qVersion() < '4.0.0' :
        app.setMainWidget(w)
    d = EdfFileLayer.EdfFileLayer()
    w.setData(d)
    w.show()
    if len(filelist):w.setFileList(filelist)
    if qt.qVersion() < '4.0.0' :
        app.exec_loop()
    else:
        app.exec_()
