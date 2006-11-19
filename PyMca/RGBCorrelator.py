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
QWTVERSION4 = RGBCorrelatorGraph.QWTVERSION4
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
        self._y1AxisInverted = False
        if graph is None:
            self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self.splitter)
            self.graph = self.graphWidget.graph
            if not QWTVERSION4:
                #add flip Icon
                self.connect(self.graphWidget.hFlipToolButton,
                             qt.SIGNAL("clicked()"),
                             self._hFlipIconSignal)
                self._handleGraph    = True
            else:
                self._handleGraph = False
        else:
            self.graph = graph
            self._handleGraph = False 
        #self.splitter.setStretchFactor(0,1)
        #self.splitter.setStretchFactor(1,1)
        self.mainLayout.addWidget(self.splitter)
        
        self.addImage = self.controller.addImage
        self.reset    = self.controller.reset
        self.addImage = self.controller.addImage
        self.setImageShape = self.controller.setImageShape
        self.update   = self.controller.update
        self.connect(self.controller,
                     qt.SIGNAL("RGBCorrelatorWidgetSignal"),
                     self.correlatorSignalSlot)

    if not QWTVERSION4:
        def _hFlipIconSignal(self):
            if self._handleGraph:
                if not self.graph.yAutoScale:
                    qt.QMessageBox.information(self, "Open",
                            "Please set Y Axis to AutoScale first")
                    return
                if not self.graph.xAutoScale:
                    qt.QMessageBox.information(self, "Open",
                            "Please set X Axis to AutoScale first")
                    return
                if self._y1AxisInverted:
                    self._y1AxisInverted = False
                else:
                    self._y1AxisInverted = True
                self.graph.setY1AxisInverted(self._y1AxisInverted)
                self.graph.zoomReset()
                self.controller.update()
                return

    def correlatorSignalSlot(self, ddict):
        if ddict.has_key('image'):
            self.image_buffer = ddict['image'].tostring()
            size = ddict['size']
            if not self.graph.yAutoScale:
                ylimits = self.graph.getY1AxisLimits()
            if not self.graph.xAutoScale:
                xlimits = self.graph.getX1AxisLimits()
            if self._handleGraph:
                self.graph.pixmapPlot(self.image_buffer,size, xmirror = 0,
                                  ymirror = not self._y1AxisInverted)
            else:
                self.graph.pixmapPlot(self.image_buffer,size)
            if not self.graph.yAutoScale:
                self.graph.setY1AxisLimits(ylimits[0], ylimits[1], replot=False)
            if not self.graph.xAutoScale:
                self.graph.setX1AxisLimits(xlimits[0], xlimits[1], replot=False)
            
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
    if len(filelist):
        try:
            import DataSource
            DataReader = DataSource.DataSource
        except:
            import EdfFileDataSource
            DataReader = EdfFileDataSource.EdfFileDataSource
        for fname in filelist:
            source = DataReader(fname)
            for key in source.getSourceInfo()['KeyList']:
                dataObject = source.getDataObject(key)
                w.addImage(dataObject.data, os.path.basename(fname)+" "+key)
    else:
        array1 = Numeric.arange(10000)
        array2 = Numeric.resize(Numeric.arange(10000), (100, 100))
        array2 = Numeric.transpose(array2)
        array3 = array1 * 1
        w.addImage(array1)
        w.addImage(array2)
        w.addImage(array3)
        w.setImageShape([100, 100])
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
