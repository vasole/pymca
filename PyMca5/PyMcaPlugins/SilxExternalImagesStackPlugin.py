# /*#########################################################################
# Copyright (C) 2004-2017 V.A. Sole, European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Plugin allowing to view an external image in a plot widget, with tools to
crop and rotate the image.

The mask of the plot widget is synchronized with the master stack widget."""

__authors__ = ["V.A. Sole", "P. Knobel"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"


import os
import numpy

from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaIO import EDFStack
from PyMca5.PyMcaGui import PyMcaFileDialogs
from PyMca5.PyMcaGui import SilxExternalImagesWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons


class SilxExternalImagesStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow)
        self.methodDict = {'Load': [self._loadImageFiles,
                                    "Load Images",
                                    PyMca_Icons.fileopen],
                           'Show': [self._showWidget,
                                    "Show Image Browser",
                                    PyMca_Icons.brushselect]}
        self.__methodKeys = ['Load', 'Show']
        self.widget = None

    def stackUpdated(self):
        self.widget = None

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def onWidgetSignal(self, ddict):
        """triggered by self.widget.sigExternalImagesWindowSignal"""
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(
                self.widget.getSelectionMask()
            )
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
        if self.widget is None:
            return [self.__methodKeys[0]]
        else:
            return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _loadImageFiles(self):
        if self.getStackDataObject() is None:
            return
        getfilter = True
        fileTypeList = ["PNG Files (*png)",
                        "JPEG Files (*jpg *jpeg)",
                        "IMAGE Files (*)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "EDF Files (*)"]
        message = "Open image file"
        filenamelist, filefilter = PyMcaFileDialogs.getFileList(
                parent=None,
                filetypelist=fileTypeList,
                message=message,
                getfilter=getfilter,
                single=False,
                currentfilter=None)
        if not filenamelist:
            return
        imagelist = []
        imagenames = []
        mask = self.getStackSelectionMask()
        if mask is None:
            r, n = self.getStackROIImagesAndNames()
            shape = r[0].shape
        else:
            shape = mask.shape

        self.widget = SilxExternalImagesWindow.SilxExternalImagesWindow()
        self.widget.sigExternalImagesWindowSignal.connect(
                self.onWidgetSignal)

        if filefilter.startswith("EDF"):
            for filename in filenamelist:
                # read the edf file
                edf = EDFStack.EdfFileDataSource.EdfFileDataSource(filename)

                # the list of images
                keylist = edf.getSourceInfo()['KeyList']
                if len(keylist) < 1:
                    msg = qt.QMessageBox(None)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Cannot read image from file")
                    msg.exec_()
                    return

                for key in keylist:
                    #get the data
                    dataObject = edf.getDataObject(key)
                    data = dataObject.data
                    if data.shape[0] not in shape or data.shape[1] not in shape:
                        # print("Ignoring %s (inconsistent shape)" % key)
                        continue
                    imagename = dataObject.info.get('Title', "")
                    if imagename != "":
                        imagename += " "
                    imagename += os.path.basename(filename) + " " + key
                    imagelist.append(data)
                    imagenames.append(imagename)
            if not imagelist:
                msg = qt.QMessageBox(None)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot read a valid image from the file")
                msg.exec_()
                return
        else:
            # Try pure Image formats
            for filename in filenamelist:
                image = qt.QImage(filename)
                if image.isNull():
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Cannot read file %s as an image" % filename)
                    msg.exec_()
                    return
                imagelist.append(image)
                imagenames.append(os.path.basename(filename))

            if not imagelist:
                msg = qt.QMessageBox(None)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot read a valid image from the file")
                msg.exec_()
                return

            for i, image in enumerate(imagelist):
                imagelist[i] = self.qImageToRgba(image)

        self.widget.setImages(imagelist,
                              labels=imagenames,
                              width=shape[1], height=shape[0])   # fixme: should be width and height of stack image
        self._showWidget()

    def _showWidget(self):
        if self.widget is None:
            return

        #Show
        self.widget.show()
        self.widget.raise_()

        #update
        self.selectionMaskUpdated()

    @staticmethod
    def qImageToRgba(qimage):
        width = qimage.width()
        height = qimage.height()
        if qimage.format() == qt.QImage.Format_Indexed8:
            pixmap0 = numpy.fromstring(qimage.bits().asstring(width * height),
                                       dtype=numpy.uint8)
            pixmap = numpy.zeros((height * width, 4), numpy.uint8)
            pixmap[:, 0] = pixmap0[:]
            pixmap[:, 1] = pixmap0[:]
            pixmap[:, 2] = pixmap0[:]
            pixmap[:, 3] = 255
            pixmap.shape = height, width, 4
        else:
            qimage = qimage.convertToFormat(qt.QImage.Format_ARGB32)
            pixmap = numpy.fromstring(qimage.bits().asstring(width * height * 4),
                                      dtype=numpy.uint8)
            pixmap.shape = height, width, -1
            # Qt uses BGRA, convert to RGBA   # TODO: check this claim (qt doc says 0xAARRGGBB)
            tmpBuffer = numpy.array(pixmap[:, :, 0], copy=True, dtype=pixmap.dtype)
            pixmap[:, :, 0] = pixmap[:, :, 2]
            pixmap[:, :, 2] = tmpBuffer

        return pixmap


MENU_TEXT = "External Images Tool"


def getStackPluginInstance(stackWindow, **kw):
    ob = SilxExternalImagesStackPlugin(stackWindow)
    return ob
