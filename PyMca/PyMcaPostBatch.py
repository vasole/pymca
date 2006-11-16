#!/usr/bin/env python
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import os
try:
    import DataSource
    DataReader = DataSource.DataSource
except:
    import EdfFileDataSource
    DataReader = EdfFileDataSource.EdfFileDataSource
import RGBCorrelatorWidget
qt = RGBCorrelatorWidget.qt
import QtBlissGraph

import Numeric

class PyMcaPostBatch(qt.QWidget):
    def __init__(self, parent = None, bgrx = True, graph = None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("PyMCA RGB Correlator")
        self.mainLayout = qt.QHBoxLayout(self)

        self.controller = RGBCorrelatorWidget.RGBCorrelatorWidget(self)
        self.mainLayout.addWidget(self.controller)

        if graph is None:
            self.graph = QtBlissGraph.QtBlissGraph(self)
            self.mainLayout.addWidget(self.graph)
        else:
            self.graph = graph

        self.addImage = self.controller.addImage
        self.reset    = self.controller.reset
        self.addImage = self.controller.addImage

    def addBatchDatFile(self, filename, ignoresigma=True):
        if ignoresigma:step = 2
        else:step=1
        f = open(filename)
        lines = f.readlines()
        f.close()
        labels = lines[0].replace("\n","").split("  ")
        i = 1
        while (not len( lines[-i].replace("\n",""))):
               i += 1
        nlabels = len(labels)
        nrows = len(lines) - i
        totalArray = Numeric.zeros((nrows, nlabels), Numeric.Float)
        for i in range(nrows):
            totalArray[i, :] = map(float, lines[i+1].split())

        nrows = int(max(totalArray[:,0]) + 1)
        ncols = int(max(totalArray[:,1]) + 1)
        singleArray = Numeric.zeros((nrows* ncols, 1), Numeric.Float)
        for i in range(2, nlabels, step):
            singleArray[:, 0] = totalArray[:,i] * 1
            self.addImage(Numeric.resize(singleArray, (nrows, ncols)), labels[i])

    def addFileList(self, filelist):
        """
        Expected to work just with EDF files
        """
        for fname in filelist:
            source = DataReader(fname)
            for key in source.getSourceInfo()['KeyList']:
                dataObject = source.getDataObject(key)
                self.controller.addImage(dataObject.data,
                                         os.path.basename(fname)+" "+key)

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    graph = QtBlissGraph.QtBlissGraph()
    w = PyMcaPostBatch(graph=graph)
    def slot(ddict):
        if ddict.has_key('image'):
            image_buffer = ddict['image'].tostring()
            size = ddict['size']
            graph.pixmapPlot(image_buffer,size)
            graph.replot()
            
    app.connect(w.controller, qt.SIGNAL("RGBCorrelatorWidgetSignal"), slot)
    import getopt
    options=''
    longoptions=[]
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)      
    for opt,arg in opts:
        pass
    filelist=args
    if len(filelist) == 1:
        w.addBatchDatFile(filelist[0])
    else:
        w.addFileList(filelist)
    graph.show()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
