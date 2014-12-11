#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
import os
import numpy
import traceback
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.misc import SelectionTable
from PyMca5.PyMcaGui.plotting import MaskScatterWidget

class ScatterPlotCorrelatorWidget(MaskScatterWidget.MaskScatterWidget):
    def __init__(self, parent=None,
                       labels=("Legend", "X", "Y"),
                       types=("Text","RadioButton", "RadioButton"),
                       **kw):
        super(ScatterPlotCorrelatorWidget, self).__init__(None, **kw)
        self._splitter = qt.QSplitter(parent)
        self._splitter.setOrientation(qt.Qt.Horizontal)

        # add a toolbar on top of the table
        self.container = qt.QWidget(self._splitter)
        self.container.mainLayout = qt.QVBoxLayout(self.container)
        self.container.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.container.mainLayout.setSpacing(0)
        self.toolBar = qt.QToolBar(self.container)

        # the selection table
        self.table = SelectionTable.SelectionTable(self.container,
                                                   labels=labels,
                                                   types=types)
        # the plot widget
        self.container.mainLayout.addWidget(self.toolBar)
        self.container.mainLayout.addWidget(self.table)
        self._splitter.addWidget(self.container)
        self._splitter.addWidget(self)

        # internal variables
        self._itemList = []
        self._itemLabels = []

        # connect
        self.table.sigSelectionTableSignal.connect(self.selectionTableSlot)

    def show(self):
        if self._splitter.isHidden():
            self._splitter.show()
        else:
            super(ScatterPlotCorrelatorWidget, self).show()

    def addSelectableItem(self, item, label = None):
        # we always keep a copy
        item = numpy.array(item, dtype=numpy.float32, copy=True)
        if label is None:
            label = "Unnamed 00"
            i = 0
            while(label in self._itemLabels):
                i += 1
                label = "Unnamed %02d" % i

        if len(self._itemList):
            if item.size != self._itemList[0].size:
                raise IndexError("Invalid size")
        if label in self._itemLabels:
            self._itemList[self._itemLabels.index(label)] = item
        else:
            self._itemList.append(item)
            self._itemLabels.append(label)
            nItems = len(self._itemList)
            self.table.setRowCount(nItems)
            self.table.fillLine(nItems - 1, [label, "", ""])
            self.table.resizeColumnToContents(0)
            self.table.resizeColumnToContents(1)
            self.table.resizeColumnToContents(2)

        ddict = self.table.getSelection()
        index = self._itemLabels.index(label)
        xKey = qt.safe_str(self.table.horizontalHeaderItem(1).text()).lower()
        yKey = qt.safe_str(self.table.horizontalHeaderItem(2).text()).lower()
        if index in (ddict[xKey] + ddict[yKey]):
            self.selectionTableSlot(ddict)

    def selectionTableSlot(self, ddict):
        legendKey = qt.safe_str(self.table.horizontalHeaderItem(0).text()).lower()
        xKey = qt.safe_str(self.table.horizontalHeaderItem(1).text()).lower()
        yKey = qt.safe_str(self.table.horizontalHeaderItem(2).text()).lower()
        x = self._itemList[ddict[xKey][0]]
        y = self._itemList[ddict[yKey][0]]
        xLabel = self._itemLabels[ddict[xKey][0]]
        yLabel = self._itemLabels[ddict[yKey][0]]
        self.setSelectionCurveData(x, y, legend="ScatterPlotWidget",
                                   color="k", symbol=None,
                                   replot=False,
                                   replace=False,
                                   xlabel=xLabel,
                                   ylabel=yLabel)
        # active curve handling is disabled
        self.setGraphXLabel(xLabel)
        self.setGraphYLabel(yLabel)
        self._updatePlot(replot=True)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = ScatterPlotCorrelatorWidget(labels=["Legend",
                                            "X",
                                            "Y"],
                                    types=["Text",
                                           "RadioButton",
                                           "RadioButton"],
                                    maxNRois=2)
    w.show()
    # fill some data
    import numpy
    import numpy.random
    x = numpy.arange(1000.)
    w.addSelectableItem(x, "range(1000)")
    w.addSelectableItem(x * x, "range(1000) ** 2")
    x = numpy.random.random(1000)
    w.addSelectableItem(x, "random(1000)")
    w.setPolygonSelectionMode()
    app.exec_()
