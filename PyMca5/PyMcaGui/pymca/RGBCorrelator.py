#!/usr/bin/env python
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
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
import numpy
from PyMca5.PyMcaGui.pymca import RGBCorrelatorWidget
qt = RGBCorrelatorWidget.qt
if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = str
from PyMca5.PyMcaGui import RGBCorrelatorGraph
from PyMca5.PyMcaGui import QPyMcaMatplotlibSave
MATPLOTLIB = True


class RGBCorrelator(qt.QWidget):
    def __init__(self, parent = None, graph = None, bgrx = True):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("PyMCA RGB Correlator")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(RGBCorrelatorGraph.IconDict['gioconda16'])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(6)
        self.splitter   = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Horizontal)
        self.controller = RGBCorrelatorWidget.RGBCorrelatorWidget(self.splitter)
        self._y1AxisInverted = False
        self._imageBuffer = None
        self._matplotlibSaveImage = None
        standaloneSaving = True
        if graph is None:
            if MATPLOTLIB:
                standaloneSaving = False
            self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self.splitter,
                                            standalonesave=standaloneSaving)
            if not standaloneSaving:
                self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
                self._saveMenu = qt.QMenu()
                self._saveMenu.addAction(QString("Standard"),    self.graphWidget._saveIconSignal)
                self._saveMenu.addAction(QString("Matplotlib") , self._saveMatplotlibImage)
            self.graph = self.graphWidget.graph
            #add flip Icon
            self.connect(self.graphWidget.hFlipToolButton,
                         qt.SIGNAL("clicked()"),
                         self._hFlipIconSignal)
            self._handleGraph    = True
        else:
            self.graph = graph
            self._handleGraph = False 
        #self.splitter.setStretchFactor(0,1)
        #self.splitter.setStretchFactor(1,1)
        self.mainLayout.addWidget(self.splitter)
        
        self.reset    = self.controller.reset
        self.addImage = self.controller.addImage
        self.removeImage = self.controller.removeImage
        self.addImageSlot = self.controller.addImageSlot
        self.removeImageSlot = self.controller.removeImageSlot
        self.replaceImageSlot = self.controller.replaceImageSlot
        self.setImageShape = self.controller.setImageShape
        self.update   = self.controller.update
        self.transposeImages   = self.controller.transposeImages
        self.connect(self.controller,
                     qt.SIGNAL("RGBCorrelatorWidgetSignal"),
                     self.correlatorSignalSlot)

    def _hFlipIconSignal(self):
        if self._handleGraph:
            if self.graph.isYAxisInverted():
                self.graph.invertYAxis(False)
            else:
                self.graph.invertYAxis(True)
            self.graph.replot()
            #this is not needed
            #self.controller.update()
            return

    def correlatorSignalSlot(self, ddict):
        if 'image' in ddict:
            # keep the image buffer as an array
            self._imageBuffer = ddict['image']
            size = ddict['size']
            self._imageBuffer.shape = size[1],size[0],4
            self._imageBuffer[:,:,3] = 255
            self.graph.addImage(self._imageBuffer)
            self.graph.replot()

    def _saveToolButtonSignal(self):
        self._saveMenu.exec_(self.cursor().pos())

    def _saveMatplotlibImage(self):
        if self._matplotlibSaveImage is None:
            self._matplotlibSaveImage = QPyMcaMatplotlibSave.SaveImageSetup(None,
                                                                            None)
            self._matplotlibSaveImage.setWindowTitle("Matplotlib RGBCorrelator")

        #Qt is BGR while the others are RGB ...
        self._matplotlibSaveImage.setPixmapImage(self._imageBuffer, bgr=True)
        self._matplotlibSaveImage.show()
        self._matplotlibSaveImage.raise_()


    def closeEvent(self, event):
        ddict = {}
        ddict['event'] = "RGBCorrelatorClosed"
        ddict['id']    = id(self)
        self.controller.close()
        if self._matplotlibSaveImage is not None:
            self._matplotlibSaveImage.close()
        self.emit(qt.SIGNAL("RGBCorrelatorSignal"),ddict)
        qt.QWidget.closeEvent(self, event)

    def show(self):
        if self.controller.isHidden():
            self.controller.show()
        qt.QWidget.show(self)

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
        print("This is a just test method using 100 x 100 matrices.")
        print("Run PyMcaPostBatch to have file loading capabilities.") 
        array1 = numpy.arange(10000)
        array2 = numpy.resize(numpy.arange(10000), (100, 100))
        array2 = numpy.transpose(array2)
        array3 = array1 * 1
        w.addImage(array1)
        w.addImage(array2)
        w.addImage(array3)
        w.setImageShape([100, 100])
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
