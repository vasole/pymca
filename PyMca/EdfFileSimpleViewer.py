#!/usr/bin/env python
__revision__ = "$Revision: 1.7 $"
#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
# is a problem for you.
#############################################################################*/
import sys
import PyMcaQt as qt
    
QTVERSION = qt.qVersion()
DEBUG = 0

import QSourceSelector
import QDataSource
import QEdfFileWidget
class EdfFileSimpleViewer(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.sourceList = []
        filetypelist = ["EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "All Files (*)"]

        self.sourceSelector = QSourceSelector.QSourceSelector(self, filetypelist = filetypelist)
        self.sourceSelector.specButton.hide()
        self.selectorWidget = {}
        self.selectorWidget[QEdfFileWidget.SOURCE_TYPE] = QEdfFileWidget.\
                                        QEdfFileWidget(self,justviewer=1)
        self.mainLayout.addWidget(self.sourceSelector)
        self.mainLayout.addWidget(self.selectorWidget[QEdfFileWidget.SOURCE_TYPE])

        if QTVERSION < '4.0.0':
            self.connect(self.sourceSelector, 
                    qt.PYSIGNAL("SourceSelectorSignal"), 
                    self._sourceSelectorSlot)
        else:
            self.connect(self.sourceSelector, 
                    qt.SIGNAL("SourceSelectorSignal"), 
                    self._sourceSelectorSlot)

    def _sourceSelectorSlot(self, ddict):
        if DEBUG:
            print("_sourceSelectorSlot(self, ddict)")
            print("ddict = ",ddict)
        if ddict["event"] == "NewSourceSelected":
            source = QDataSource.QDataSource(ddict["sourcelist"])
            self.sourceList.append(source)
            sourceType = source.sourceType
            self.selectorWidget[sourceType].setDataSource(source)
        elif ddict["event"] == "SourceSelected":
            found = 0
            for source in self.sourceList:
                if source.sourceName == ddict["sourcelist"]:
                    found = 1
                    break
            if not found:
                if DEBUG:
                    print("WARNING: source not found")
                return
            sourceType = source.sourceType
            self.selectorWidget[sourceType].setDataSource(source)
        elif ddict["event"] == "SourceClosed":
            found = 0
            for source in self.sourceList:
                if source.sourceName == ddict["sourcelist"]:
                    found = 1
                    break
            if not found:
                if DEBUG:
                    print("WARNING: source not found")
                return
            sourceType = source.sourceType
            del self.sourceList[self.sourceList.index(source)]
            for source in self.sourceList:
                if sourceType == source.sourceType:
                    self.selectorWidget[sourceType].setDataSource(source)
                    return
            #there is no other selection of that type
            if len(self.sourceList):
                source = self.sourceList[0]
                sourceType = source.sourceType
                self.selectorWidget[sourceType].setDataSource(source)
            else:
                self.selectorWidget[sourceType].setDataSource(None)

    def setFileList(self, filelist):
        for ffile in filelist:
            self.sourceSelector.openFile(ffile, justloaded = 1)
                    
            
def main():
    import sys
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
    if QTVERSION < '4.0.0' :
        app.setMainWidget(w)
        w.show()
        if len(filelist):w.setFileList(filelist)
        app.exec_loop()
    else:
        if len(filelist):w.setFileList(filelist)
        w.show()
        app.exec_()

if __name__ == "__main__":
    main()
