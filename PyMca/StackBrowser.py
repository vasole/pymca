#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import numpy
from PyMca import MaskImageWidget
from PyMca import FrameBrowser
from PyMca import DataObject
qt = MaskImageWidget.qt

DEBUG = 0

class StackBrowser(MaskImageWidget.MaskImageWidget):
    def __init__(self, *var, **kw):
        ddict = {}
        ddict['usetab'] = kw.get("usetab", False)
        ddict.update(kw)
        ddict['standalonesave'] = True
        MaskImageWidget.MaskImageWidget.__init__(self, *var, **ddict)
        self.setWindowTitle("Stack Browser")
        self.dataObjectsList = []
        self.dataObjectsDict = {}

        self.nameBox = qt.QWidget(self)
        self.nameBox.mainLayout = qt.QHBoxLayout(self.nameBox)

        self.nameLabel = qt.QLabel(self.nameBox)
        self.nameLabel.setText("Image Name = ")
        self.name = qt.QLineEdit(self.nameBox)
        self.nameBox.mainLayout.addWidget(self.nameLabel)
        self.nameBox.mainLayout.addWidget(self.name)

        self.slider = FrameBrowser.HorizontalSliderWithBrowser(self)
        self.slider.setRange(0, 0)

        self.mainLayout.addWidget(self.nameBox)
        self.mainLayout.addWidget(self.slider)
        self.connect(self.slider,
                     qt.SIGNAL("valueChanged(int)"),
                     self._showImageSliderSlot)
        self.slider.hide()
        self.buildAndConnectImageButtonBox(replace=True)

    def setStackDataObject(self, stack, index=None, stack_name=None):
        if hasattr(stack, "info") and hasattr(stack, "data"):            
            dataObject = stack
        else:
            dataObject = DataObject.DataObject()
            dataObject.info = {}
            dataObject.data = stack
        if dataObject.data is None:
            return
        if stack_name is None:
            legend = dataObject.info.get('SourceName', "Stack")
        else:
            legend = stack_name
        if index is None:
            mcaIndex = dataObject.info.get('McaIndex', 0)
        else:
            mcaIndex = index
        shape = dataObject.data.shape
        self.dataObjectsList = [legend]
        self.dataObjectsDict = {legend:dataObject}
        self._browsingIndex = mcaIndex
        if mcaIndex == 0:
            if len(shape) == 2:
                self._nImages = 1
                self.setImageData(dataObject.data)
                self.slider.hide()
                self.name.setText(legend)
            else:
                self._nImages = 1
                for dimension in dataObject.data.shape[:-2]:
                    self._nImages *= dimension
                #This is a problem for dynamic data        
                #dataObject.data.shape = self._nImages, shape[-2], shape[-1]
                data = self._getImageDataFromSingleIndex(0)
                self.setImageData(data)
                self.slider.setRange(0, self._nImages - 1)
                self.slider.setValue(0)
                self.slider.show()
                self.name.setText(legend+" 0")
        elif mcaIndex in [len(shape)-1, -1]:
            mcaIndex = -1
            self._browsingIndex = mcaIndex
            if len(shape) == 2:
                self._nImages = 1
                self.setImageData(dataObject.data)
                self.slider.hide()
                self.name.setText(legend)
            else:
                self._nImages = 1
                for dimension in dataObject.data.shape[2:]:
                    self._nImages *= dimension
                #This is a problem for dynamic data        
                #dataObject.data.shape = self._nImages, shape[-2], shape[-1]
                data = self._getImageDataFromSingleIndex(0)
                self.setImageData(data)
                self.slider.setRange(0, self._nImages - 1)
                self.slider.setValue(0)
                self.slider.show()
                self.name.setText(legend+" 0")
        else:
            raise ValueError("Unsupported 1D index %d"  % mcaIndex)
        if self._nImages > 1:
            self.showImage(0)
        else:
            self.plotImage()

    def _getImageDataFromSingleIndex(self, index):
        if not len(self.dataObjectsList):
            print "nothing to show"
            return
        legend = self.dataObjectsList[0]
        if type(legend) == type([]):
            legend = legend[index]
        dataObject = self.dataObjectsDict[legend]
        shape = dataObject.data.shape
        if len(shape) == 2:
            if index > 0:
                raise IndexError, "Only one image in stack"
            return dataObject.data
        if self._browsingIndex == 0:
            if len(shape) == 3:
                data = dataObject.data[index:index+1,:,:]
                data.shape = data.shape[1:]
                return data
            #I have to deduce the appropriate indices from the given index
            #always assuming C order
            acquisitionShape =  dataObject.data.shape[:-2]
            if len(shape) == 4:
                j = index % acquisitionShape[-1]
                i = int(index/(acquisitionShape[-1]*acquisitionShape[-2]))
                return dataObject.data[i, j]
        elif self._browsingIndex == -1:
            if len(shape) == 3:
                data = dataObject.data[:,:,index:index+1]
                data.shape = data.shape[0], data.shape[1]
                return data
        raise IndexError, "Unhandled dimension"

    def _showImageSliderSlot(self, index):
        self.showImage(index, moveslider=False)

    def _buildTitle(self, legend, index):
        return "%s %d" % (legend, index)
            
    def showImage(self, index=0, moveslider=True):
        if not len(self.dataObjectsList):
            return
        legend = self.dataObjectsList[0]
        dataObject = self.dataObjectsDict[legend]
        data = self._getImageDataFromSingleIndex(index)       
        self.setImageData(data, clearmask=False)
        txt = self._buildTitle(legend, index)
        self.graphWidget.graph.setTitle(txt)
        self.name.setText(txt)
        if moveslider:
            self.slider.setValue(index)

if __name__ == "__main__":
    #create a dummy stack
    nrows = 100
    ncols = 200
    nchannels = 1024
    a = numpy.ones((nrows, ncols), numpy.float)
    stackData = numpy.zeros((nrows, ncols, nchannels), numpy.float)
    for i in xrange(nchannels):
        stackData[:, :, i] = a * i

    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                        app,qt.SLOT("quit()"))
    w = StackBrowser()
    w.setStackDataObject(stackData, index=0)
    w.show()
    app.exec_()
