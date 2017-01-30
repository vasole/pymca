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
"""This module defines a :class:`ScanWindow` inheriting a *silx*
:class:`PlotWindow` with additional tools and actions.
The main addition is a :class:`PluginsToolButton` button added to the toolbar,
to open a menu with plugins."""

# TODO: more ScanWindow tools

import numpy
import os
import sys
import traceback

from silx.gui.plot import PlotWindow

import PyMca5
from PyMca5.PyMcaGui import PyMcaQt as qt
from PluginsToolButton import PluginsToolButton   # TODO: relative import
from ScanFitToolButton import ScanFitToolButton   # TODO: relative import
from PyMca5.PyMcaGui.pymca import ScanFit


if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str

PLUGINS_DIR = None

if os.path.exists(os.path.join(os.path.dirname(PyMca5.__file__), "PyMcaPlugins")):
    from PyMca5 import PyMcaPlugins
    PLUGINS_DIR = os.path.dirname(PyMcaPlugins.__file__)
else:
    directory = os.path.dirname(__file__)
    while True:
        if os.path.exists(os.path.join(directory, "PyMcaPlugins")):
            PLUGINS_DIR = os.path.join(directory, "PyMcaPlugins")
            break
        directory = os.path.dirname(directory)
        if len(directory) < 5:
            break
userPluginsDirectory = PyMca5.getDefaultUserPluginsDirectory()
if userPluginsDirectory is not None:
    if PLUGINS_DIR is None:
        PLUGINS_DIR = userPluginsDirectory
    else:
        PLUGINS_DIR = [PLUGINS_DIR, userPluginsDirectory]


class ScanWindow(PlotWindow):
    def __init__(self, parent=None, name="Scan Window", fit=True, backend=None,
                 plugins=True, control=True, position=True, roi=True,
                 specfit=None, newplot=True, info=False, **kw):

        super(ScanWindow, self).__init__(parent,
                                         backend=backend,
                                         roi=roi,
                                         control=control,
                                         position=position,
                                         )  # **kw)
        # fixme: newplot,?
        self.setDataMargins(0, 0, 0.025, 0.025)
        self.setPanWithArrowKeys(True)
        self._plotType = "SCAN"
        # these two objects are the same
        self.dataObjectsList = list(self._curves.keys())
        # but this is tricky
        self.dataObjectsDict = {}

        self.setWindowTitle(name)
        self.matplotlibDialog = None

        _simpleOpsToolbar = qt.QToolBar("Simple operations")
        self.addToolBar(_simpleOpsToolbar)

        if plugins:
            self.pluginsToolButton = PluginsToolButton(plot=self, parent=self)
            _simpleOpsToolbar.addWidget(self.pluginsToolButton)

            if PLUGINS_DIR is not None:
                if isinstance(PLUGINS_DIR, list):
                    pluginDir = PLUGINS_DIR
                else:
                    pluginDir = [PLUGINS_DIR]
                self.pluginsToolButton.getPlugins(
                        method="getPlugin1DInstance",
                        directoryList=pluginDir)

        if fit:
            self.scanFit = ScanFit.ScanFit(specfit=specfit)
            scanFitToolButton = ScanFitToolButton(self)

            _simpleOpsToolbar.addWidget(scanFitToolButton)


def test():
    app = qt.QApplication([])
    w = ScanWindow()
    x = numpy.arange(1000.)
    y = 10 * x + 10000. * numpy.exp(-0.5*(x-500)*(x-500)/400)
    w.addCurve(x, y, legend="dummy", resetzoom=True, replace=True)
    w.resetZoom()
    app.lastWindowClosed.connect(app.quit)
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
