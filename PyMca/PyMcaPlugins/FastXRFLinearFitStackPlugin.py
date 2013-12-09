#/*##########################################################################
# Copyright (C) 2013 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
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
import os
import time
from PyMca import StackPluginBase
from PyMca import FastXRFLinearFit
from PyMca import FastXRFLinearFitWindow
from PyMca import CalculationThread
from PyMca import StackPluginResultsWindow
from PyMca import PyMcaFileDialogs
import PyMca.PyMca_Icons as PyMca_Icons
from PyMca import PyMcaQt as qt
from PyMca import ArraySave

DEBUG = 0

class FastXRFLinearFitStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        function = self.calculate
        info = "Fit stack with current fit configuration"
        icon = PyMca_Icons.fit
        self.methodDict["Fit Stack"] =[function,
                                       info,
                                       icon]
        self.__methodKeys = ["Fit Stack"]
        self.configurationWidget = \
                    FastXRFLinearFitWindow.FastXRFLinearFitDialog()
        self.fitInstance = None
        self.widget = None
        self.thread = None

    def stackUpdated(self):
        if DEBUG:
            print("FastXRFLinearFitStackPlugin.stackUpdated() called")
        self.widget = None

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def mySlot(self, ddict):
        if DEBUG:
            print("mySlot ", ddict['event'], ddict.keys())
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)
    
    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    # The specific part
    def calculate(self):
        ret = self.configurationWidget.exec_()
        if ret:
            self._executeFunctionAndParameters()

    def _executeFunctionAndParameters(self):
        parameters = self.configurationWidget.getParameters()
        self.widget = None
        if self.fitInstance is None:
            self.fitInstance = FastXRFLinearFit.FastXRFLinearFit()
        self._fitConfigurationFile = parameters['configuration']
        #self._fitConfigurationFile="E:\DATA\COTTE\CH1777\G4-4720eV-NOWEIGHT-Constant-batch.cfg"
        
        if DEBUG:
            self.thread = CalculationThread.CalculationThread(\
                            calculation_method=self.actualCalculation)
            self.thread.result = self.actualCalculation()
            self.threadFinished()
        else:
            self.thread = CalculationThread.CalculationThread(\
                            calculation_method=self.actualCalculation)
            #qt.QObject.connect(self.thread,
            #             qt.SIGNAL('finished()'),
            #             self.threadFinished)
            self.thread.start()
            message = "Please wait. Calculation going on."
            CalculationThread.waitingMessageDialog(self.thread,
                                parent=self.configurationWidget,
                                message=message)
            while self.thread.isRunning():
                time.sleep(2)
            self.threadFinished()

    def actualCalculation(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is not None:
            x, spectrum, legend, info = activeCurve
        else:
            x = None
            spectrum = None
        if not self.isStackFinite():
            # one has to check for NaNs in the used region(s)
            # for the time being only in the global image
            # spatial_mask = numpy.isfinite(image_data)
            spatial_mask = numpy.isfinite(self.getStackOriginalImage())
        stack = self.getStackDataObject()

        self.fitInstance.setFitConfigurationFile(self._fitConfigurationFile)
        result = self.fitInstance.fitMultipleSpectra(x=None,
                                                     y=stack,
                                                     concentrations=False,
                                                     ysum=spectrum)
        return result

    def threadFinished(self):
        result = self.thread.result
        self.thread = None
        if type(result) == type((1,)):
            #if we receive a tuple there was an error
            if len(result):
                if result[0] == "Exception":
                    # somehow this exception is not caught
                    raise Exception(result[1], result[2])
                    return
        images = result['parameters']
        imageNames = result['names']
        nImages = images.shape[0]
        self.widget = StackPluginResultsWindow.StackPluginResultsWindow(\
                                        usetab=False)
        self.widget.buildAndConnectImageButtonBox()
        qt = StackPluginResultsWindow.qt
        qt.QObject.connect(self.widget,
                           qt.SIGNAL('MaskImageWidgetSignal'),
                           self.mySlot)

        self.widget.setStackPluginResults(images,
                                          image_names=imageNames)
        self._showWidget()

        # save to output directory
        parameters = self.configurationWidget.getParameters()
        outputDir = parameters["output_dir"]
        if outputDir in [None, ""]:
            if DEBUG:
                print("Nothing to be saved")
                return
        if not os.path.isdir(outputDir):
            raise IOError("<%s>  is not a valid output directory" % outputDir)
        imagesDir = os.path.join(outputDir, "IMAGES")
        imageList = [None] * nImages
        for i in range(nImages):
            imageList[i] = images[i]
        fileName = os.path.join(imagesDir, "images.edf")
        ArraySave.save2DArrayListAsEDF(imageList, fileName, labels=imageNames)
        fileName = os.path.join(imagesDir, "images.csv")
        ArraySave.save2DArrayListAsASCII(imageList, fileName, csv=True, labels=imageNames)                    

    def _showWidget(self):
        if self.widget is None:
            return
        #Show
        self.widget.show()
        self.widget.raise_()

        #update
        self.selectionMaskUpdated()

MENU_TEXT = "Fast XRF Stack Fitting"
def getStackPluginInstance(stackWindow, **kw):
    ob = FastXRFLinearFitStackPlugin(stackWindow)
    return ob
