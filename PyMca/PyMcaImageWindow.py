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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
from Icons import IconDict
import Numeric
import time
import RGBCorrelator
import RGBImageCalculator
from RGBImageCalculator import qt
QTVERSION = qt.qVersion()
DEBUG = 0

class PyMcaImageWindow(RGBImageCalculator.RGBImageCalculator):
    def __init__(self, parent = None,
                 name = "PyMca Image Window",
                 correlator = None):
        RGBImageCalculator.RGBImageCalculator.__init__(self, parent,
                                                       math = False,
                                                       replace = True)
        self.setWindowTitle(name)
        self.correlator = correlator
        self.ownCorrelator = False
        #self.mathBox.hide()
        self.dataObjectsList = []
        self.dataObjectsDict = {}
        self._plotEnabled    = True

    def _connectCorrelator(self):
        if QTVERSION > '4.0.0':
            self.ownCorrelator = True
            self.correlator = RGBCorrelator.RGBCorrelator()
            self.correlator.setWindowTitle("ImageWindow RGB Correlator")
            self.connect(self, qt.SIGNAL("addImageClicked"),
                         self.correlator.addImageSlot)
            self.connect(self, qt.SIGNAL("removeImageClicked"),
                         self.correlator.removeImageSlot)
            self.connect(self, qt.SIGNAL("replaceImageClicked"),
                         self.correlator.replaceImageSlot)


    def _addImageClicked(self):
        if self.correlator is None: self._connectCorrelator()
        if self._imageData is None:return
        if self._imageData == []:return

        if not RGBImageCalculator.RGBImageCalculator._addImageClicked(self):
            #if self.ownCorrelator:
                if self.correlator.isHidden():
                    self.correlator.show()

    def setDispatcher(self, w):
        if QTVERSION < '4.0.0':
            self.connect(w, qt.PYSIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.PYSIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.PYSIGNAL("replaceSelection"),
                             self._replaceSelection)
        else:
            self.connect(w, qt.SIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.SIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.SIGNAL("replaceSelection"),
                             self._replaceSelection)
            
    def _addSelection(self, selectionlist):
        if DEBUG:print "_addSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            #if not sel.has_key("scanselection"): continue
            #if not sel["scanselection"]:continue
            #if len(key.split(".")) > 2: continue
            dataObject = sel['dataobject']

            #only one-dimensional selections considered
            if dataObject.info["selectiontype"] != "2D": continue
            if dataObject.data is None:
                print "nothing to plot"
            self.dataObjectsList = [legend]
            self.dataObjectsDict = {legend:dataObject}
            self._imageData = dataObject.data
            self.name.setText(legend)
            if self._plotEnabled:
                self.plotImage(True)

    def setPlotEnabled(self, value=True):
        self._plotEnabled = value
        if value:
            if self._imageData is not None:
                self.plotImage(True)
            else:
                self.graphWidget.graph.clear()
                pass
            
    def _removeSelection(self, selectionlist):
        if DEBUG:print "_removeSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]
        for sel in sellist:
            legend = sel['legend']
            if legend in self.dataObjectsList:
                self.dataObjectsList = []
            if legend in self.dataObjectsDict.keys():
                self.dataObjectsDict = {}
                #For the time being I prefer to leave the last image plotted
                #self._imageData = 0 * self._imageData
                #self.plotImage(True)

    def _replaceSelection(self, selectionlist):
        if DEBUG:print "_replaceSelection(self, selectionlist)",selectionlist
        self._addSelection(selectionlist)

    def closeEvent(self, event):
        if self.ownCorrelator:
            self.correlator.close()
        RGBImageCalculator.RGBImageCalculator.closeEvent(self, event)    


class TimerLoop:
    def __init__(self, function = None, period = 1000):
        self.__timer = qt.QTimer()
        if function is None: function = self.test
        self._function = function
        self.__setThread(function, period)
        #self._function = function 
    
    def __setThread(self, function, period):
        self.__timer = qt.QTimer()
        qt.QObject.connect(self.__timer,
                       qt.SIGNAL("timeout()"),
                       function)
        self.__timer.start(period)

    def test(self):
        print "Test function called"

if __name__ == "__main__":
    import DataObject
    import weakref
    import time
    def buildDataObject(arrayData):
        dataObject = DataObject.DataObject()
        dataObject.data = arrayData
        dataObject.info['selectiontype'] = "2D"
        dataObject.info['Key'] = id(dataObject)
        return dataObject

    def buildSelection(dataObject, name = "image_data0"):
        key = dataObject.info['Key']
        def dataObjectDestroyed(ref, dataObjectKey=key):
            if DEBUG: print "dataObject distroyed key = ", key
        dataObjectRef=weakref.proxy(dataObject, dataObjectDestroyed)
        selection = {}
        selection['SourceType'] = 'SPS'
        selection['SourceName'] = 'spec'
        selection['Key']        = name
        selection['legend']     = selection['SourceName'] + "  "+ name
        selection['imageselection'] = True
        selection['dataobject'] = dataObjectRef
        return selection

    a = 1000
    b = 1000
    period = 1000
    x1 = Numeric.arange(a * b).astype(Numeric.Float)
    x1.shape= [a, b]
    x2 = Numeric.transpose(x1)

    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                        app,qt.SLOT("quit()"))
    if len(sys.argv) > 1:PYMCA=True
    else:PYMCA = False

    if PYMCA:
        import PyMca
        w = PyMca.PyMca()
        w.show()
    else:
        w = PyMcaImageWindow()
        w.show()
    counter = 0
    def function(period = period):
        global counter
        flag = counter % 6
        if flag == 0:
            #add x1
            print "Adding X1"
            dataObject = buildDataObject(x1)
            selection = buildSelection(dataObject, 'X1')
            if PYMCA:
                w.dispatcherAddSelectionSlot(selection)
            else:
                w._addSelection(selection)
        elif flag == 1:
            #add x2
            print "Adding X2"
            dataObject = buildDataObject(x2)
            selection = buildSelection(dataObject, 'X2')
            if PYMCA:
                w.dispatcherAddSelectionSlot(selection)
            else:
                w._addSelection(selection)
        elif flag == 2:
            #add x1
            print "Changing X1"
            dataObject = buildDataObject(x2)
            selection = buildSelection(dataObject, 'X1')
            if PYMCA:
                w.dispatcherAddSelectionSlot(selection)
            else:
                w._addSelection(selection)
        elif flag == 1:
            #add x2
            print "Changing X2"
            dataObject = buildDataObject(x2)
            selection = buildSelection(dataObject, 'X1')
            if PYMCA:
                w.dispatcherAddSelectionSlot(selection)
            else:
                w._addSelection(selection)
        elif flag == 4:
            #replace x1
            print "Replacing by new X1"
            dataObject = buildDataObject(x1-x2)
            selection = buildSelection(dataObject, 'X1')
            if PYMCA:
                w.dispatcherReplaceSelectionSlot(selection)
            else:
                w._replaceSelection(selection)
        else:
            #replace by x2
            print "Replacing by new X2"
            dataObject = buildDataObject(x2-x1)
            selection = buildSelection(dataObject, 'X2')
            if PYMCA:
                w.dispatcherReplaceSelectionSlot(selection)
            else:
                w._replaceSelection(selection)
        counter += 1

    loop = TimerLoop(function = function, period = period)
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        sys.exit(app.exec_())
