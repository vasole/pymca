#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""

A Stack plugin is a module that will be automatically added to the PyMca stack windows
in order to perform user defined operations on the data stack.

These plugins will be compatible with any stack window that provides the functions:
    #data related
    getStackDataObject
    getStackData
    getStackInfo
    setStack

    #images related
    addImage
    removeImage
    replaceImage

    #mask related
    setSelectionMask
    getSelectionMask

    #displayed curves
    getActiveCurve
    getGraphXLimits
    getGraphYLimits

    #information method
    stackUpdated
    selectionMaskUpdated
"""
import sys
import os
import logging
import traceback
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui import CalculationThread
# only EDGE and JUMP are always small enough to be shown ...
from PyMca5.PyMcaGui import StackPluginResultsWindow
from PyMca5.PyMcaGui import StackXASBatchWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMca import XASStackBatch

_logger = logging.getLogger(__name__)


class XASStackBatchPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        function = self.calculate
        info = "Perform XAS Data reduction using a configuration file"
        icon = None
        self.methodDict["Calculate"] =[function,
                                       info,
                                       icon]
        self.__methodKeys = ["Calculate"]
        self.configurationWidget = None
        self.workerInstance = XASStackBatch.XASStackBatch()
        self._widget = None
        self.thread = None

    def stackUpdated(self):
        _logger.debug("StackXASBatchPlugin.stackUpdated() called")
        if self._widget is not None:
            self._widget.close()
        self._widget = None

    def selectionMaskUpdated(self):
        if self._widget is None:
            return
        if self._widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self._widget.setSelectionMask(mask)

    def mySlot(self, ddict):
        _logger.debug("mySlot %s %s", ddict['event'], ddict.keys())
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "addAllClicked":
            for i in range(len(ddict["images"])):
                self.addImage(ddict['images'][i], ddict['titles'][i])            
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)

    #Methods implemented by the plugin
    def getMethods(self):
        if self._widget is None:
            return [self.__methodKeys[0]]
        else:
            return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    # The specific part
    def calculate(self):
        if self.configurationWidget is None:
            self.configurationWidget = \
                            StackXASBatchWindow.StackXASBatchDialog()
        ret = self.configurationWidget.exec()
        if ret:
            self._executeFunctionAndParameters()

    def _executeFunctionAndParameters(self):
        self._parameters = self.configurationWidget.getParameters()
        self._widget = None
        if _logger.getEffectiveLevel() == logging.DEBUG:
            self.thread = CalculationThread.CalculationThread(\
                            calculation_method=self.actualCalculation)
            self.thread.result = self.actualCalculation()
            self.threadFinished()
        else:
            self.thread = CalculationThread.CalculationThread(\
                            calculation_method=self.actualCalculation)
            self.thread.finished.connect(self.threadFinished)
            self.thread.start()
            message = "Please wait. Calculation going on."
            CalculationThread.waitingMessageDialog(self.thread,
                                parent=self.configurationWidget,
                                message=message)

    def actualCalculation(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is not None:
            x, spectrum, legend, info = activeCurve
        else:
            x = None
            spectrum = None
        stack = self.getStackDataObject()
        configurationFile = self._parameters['configuration']
        net = self._parameters['mask']
        self.workerInstance.setConfigurationFile(configurationFile)
        if net:
            mask = self.getStackSelectionMask()
        else:
            mask = None
        outputDir = self._parameters["output_dir"]
        if outputDir in [None, ""]:
           outputDir=None
        if not (self._parameters["file_root"] in [None, ""]):
            fileRoot = self._parameters["file_root"].replace(" ","")
        if fileRoot in [None, ""]:
            fileRoot = None
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        result = self.workerInstance.processMultipleSpectra(x=x,
                                                            y=stack,
                                                            weight=None,
                                                            ysum=spectrum,
                                                            mask=mask,
                                                            directory=outputDir,
                                                            name=fileRoot)
        return result

    def threadFinished(self):
        try:
            self._threadFinished()
        except:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec()

    def _threadFinished(self):
        result = self.thread.result
        self.thread = None
        if type(result) == type((1,)):
            #if we receive a tuple there was an error
            if len(result):
                if result[0] == "Exception":
                    # somehow this exception is not caught
                    raise Exception(result[1], result[2])#, result[3])
                    return
        imageNames = result['names']
        images = result["images"]
        nImages = images.shape[0]
        self._widget = StackPluginResultsWindow.StackPluginResultsWindow(\
                                        usetab=False)
        self._widget.buildAndConnectImageButtonBox(replace=True,
                                                  multiple=True)
        qt = StackPluginResultsWindow.qt
        self._widget.sigMaskImageWidgetSignal.connect(self.mySlot)
        self._widget.setStackPluginResults(images,
                                          image_names=imageNames)
        self._showWidget()

    def _showWidget(self):
        if self._widget is None:
            return
        #Show
        self._widget.show()
        self._widget.raise_()

        #update
        self.selectionMaskUpdated()

MENU_TEXT = "XAS Batch"
def getStackPluginInstance(stackWindow, **kw):
    ob = XASStackBatchPlugin(stackWindow)
    return ob
    
