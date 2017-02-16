#/*##########################################################################
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
#############################################################################*/


import ScanWindow

class PyMcaScanWindow(ScanWindow.ScanWindow):
    """

    """
    def __init__(self, parent=None, name="Scan Window", fit=True, backend=None,
                 plugins=True, control=True, position=True, roi=True,
                 specfit=None, info=False):
        ScanWindow.ScanWindow.__init__(self,
                parent, name, fit, backend,
                plugins, control, position, roi,
                specfit, info)
        self.setWindowType("SCAN")

        self.dataObjectsDict = {}

    @property
    def dataObjectsList(self):
        return self.getAllCurves(just_legend=True)

    @property
    def _curveList(self):
        return self.getAllCurves(just_legend=True)

    def setDispatcher(self, w):
        w.sigAddSelection.connect(self._addSelection)
        w.sigRemoveSelection.connect(self._removeSelection)
        w.sigReplaceSelection.connect(self._replaceSelection)

    def _addSelection(self, selectionlist, replot=True):
        # TODO
        pass

    def _removeSelection(self, selectionlist):
        # TODO
        pass

    def _replaceSelection(self, selectionlist):
        # TODO
        pass

    def removeCurves(self, removeList):
        for legend in removeList:
            self.removeCurve(legend)
            if legend in self.dataObjectsDict:
                del self.dataObjectsDict[legend]

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, **kw):
        # TODO?? (original ScanWindow keeps old info when replacing a curve)
        pass

    def newCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, **kw):
        # TODO?? (maybe merge with addCurve to have a single method to add a curve)
        pass
