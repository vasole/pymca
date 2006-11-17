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
import RGBCorrelatorWidget
qt = RGBCorrelatorWidget.qt
import RGBCorrelatorGraph
try:
    import DataSource
    DataReader = DataSource.DataSource
except:
    import EdfFileDataSource
    DataReader = EdfFileDataSource.EdfFileDataSource
import Numeric

class RGBCorrelator(qt.QWidget):
    def __init__(self, parent = None, graph = None, bgrx = True):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("PyMCA RGB Correlator")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(RGBCorrelatorGraph.IconDict['gioconda16'])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(6)
        self.splitter   = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Horizontal)
        self.controller = RGBCorrelatorWidget.RGBCorrelatorWidget(self.splitter)
        if graph is None:
            self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self.splitter)
            self.graph = self.graphWidget.graph
        else:
            self.graph = graph
        #self.splitter.setStretchFactor(0,1)
        #self.splitter.setStretchFactor(1,1)
        self.mainLayout.addWidget(self.splitter)
        
        self.addImage = self.controller.addImage
        self.reset    = self.controller.reset
        self.addImage = self.controller.addImage
        self.connect(self.controller,
                     qt.SIGNAL("RGBCorrelatorWidgetSignal"),
                     self.correlatorSignalSlot)


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

    def correlatorSignalSlot(self, ddict):
        if ddict.has_key('image'):
            image_buffer = ddict['image'].tostring()
            size = ddict['size']
            if not self.graph.yAutoScale:
                #store graph settings
                ylimits = self.graph.gety1axislimits()
            if not self.graph.xAutoScale:
                #store graph settings
                xlimits = self.graph.getx1axislimits()
            #zoomstack = graph.zoomStack * 1
            self.graph.pixmapPlot(image_buffer,size)
            #graph.zoomStack = 1 *  zoomstack
            if not self.graph.yAutoScale:
                self.graph.sety1axislimits(ylimits[0], ylimits[1])
            if not self.graph.xAutoScale:
                self.graph.setx1axislimits(xlimits[0], xlimits[1])
            self.graph.replot()

    def closeEvent(self, event):
        ddict = {}
        ddict['event'] = "RGBCorrelatorClosed"
        ddict['id']    = id(self)
        self.emit(qt.SIGNAL("RGBCorrelatorSignal"),ddict)

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))
    if 0:
        graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph()
        graph = graphWidget.graph
        w = RGBCorrelator(graph=graph)
    else:
        w = RGBCorrelator()
        w.resize(800, 600)
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
    elif len(filelist):
        w.addFileList(filelist)
    else:
        filelist = qt.QFileDialog.getOpenFileNames(None,
                                                   "Select EDF files",
                                                   os.getcwd())
        if len(filelist):
            filelist = map(str, filelist)
            w.addFileList(filelist)
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
