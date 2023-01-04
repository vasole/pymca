#/*##########################################################################
# Copyright (C) 2022-2023 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy

import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.misc import TableWidget
logger = logging.getLogger(__name__)

class ImageListStatsWidget(TableWidget.TableWidget):
    def __init__(self, parent=None, cut=False, paste=False):
        super(ImageListStatsWidget, self).__init__(parent=parent, cut=cut, paste=paste)
        self.imageList = None
        self.imageMask = None
        labels = ["Name", "Maximum", "Minimum", "N", "Mean", "std"]
        self._stats = [x.lower() for x in labels]
        self.setColumnCount(len(self._stats))
        for i in range(len(labels)):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(labels[i],
                                           qt.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.setHorizontalHeaderItem(i,item)
        rheight = self.horizontalHeader().sizeHint().height()
        self.setMinimumHeight(5*rheight)

    def setImageList(self, images, image_names=None):
        if images is None:
            self.imageList = None
            self.imageMask = None
            self.updateStats()
            return
        if type(images) == type([]):
            self.imageList = images
            if image_names is None:
                self.imageNames = []
                for i in range(nimages):
                    self.imageNames.append("Image %02d" % i)
            else:
                self.imageNames = image_names
        elif len(images.shape) == 3:
            nimages = images.shape[0]
            self.imageList = [0] * nimages
            for i in range(nimages):
                self.imageList[i] = images[i,:]
                if 0:
                    #leave the data as they originally come
                    if self.imageList[i].max() < 0:
                        self.imageList[i] *= -1
                        if self.spectrumList is not None:
                            self.spectrumList [i] *= -1
            if image_names is None:
                self.imageNames = []
                for i in range(nimages):
                    self.imageNames.append("Image %02d" % i)
            else:
                self.imageNames = image_names

        newMask = None
        if self.imageList is not None:
            if len(self.imageList):
                if self.imageMask is not None:
                    if self.imageMask.shape == self.imageList[0].shape:
                        # we keep the mask
                        logger.info("Keeping previously defined mask")
                        newMask = self.imageMask

        self.imageMask = newMask
        self.updateStats()

    def setSelectionMask(self, mask=None):
        self.imageMask = mask
        self.updateStats()

    def updateStats(self):
        if self.imageList in [None, []]:
            self.setRowCount(0)
            return
        statsList = []
        mask = self.imageMask
        if mask is None:
            mask = numpy.zeros(self.imageList[0].shape, dtype=numpy.uint8)
        mask = mask.flatten()
        results = []
        for idx, imageName in enumerate(self.imageNames):
            result = {}
            image = self.imageList[idx].flatten()
            if mask.min() == mask.max():
                if mask.min() == 0:
                    # whole image
                    pass
                else:
                    # only masked image
                    image = image[mask > 0]
            else:
                image = image[mask > 0]
            image = numpy.array(image[numpy.isfinite(image)], dtype=numpy.float64)
            result['name'] = imageName
            result['maximum'] = image.max()
            result['minimum'] = image.min()
            result['n'] = image.size
            result['mean'] = image.mean()
            result['std'] = image.std()
            results.append(result)
        self._fillTable(results)

    def _fillTable(self, results):
        nRows = self.rowCount()
        nColumns = self.columnCount()
        self.setRowCount(len(results))
        for row, result in enumerate(results):
            for column, stat in enumerate(self._stats):
                if column == 0:
                    text = result[stat]
                else:
                    text = "%g" % result[stat]
                item = self.item(row, column)
                if item is None:
                    item = qt.QTableWidgetItem(text, qt.QTableWidgetItem.Type)
                    item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
                    item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                    self.setItem(row, column, item)
                else:
                    item.setText(text)

def main():
    w = ImageListStatsWidget()
    data = numpy.arange(20000)
    data.shape = 2, 100, 100
    data[1, 0:100,0:50] = 100
    w.setImageList(data, image_names=["I1", "I2"])
    w.show()
    return w

if __name__ == "__main__":
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    w = main()
    app.exec()


