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

import os

from silx.gui.plot import PlotWindow

import PyMca5
from PyMca5.PyMcaGui import PyMcaQt as qt
from PluginsToolButton import PluginsToolButton   # TODO: relative import
from ScanFitToolButton import ScanFitToolButton   # TODO: relative import
import SimpleActions   # TODO: fix relative import
import ScanWindowInfoWidget  # TODO: fix relative import
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
    """:class:`PlotWindow` augmented with plugins, fitting actions,
    a widget for displaying scan metadata and simple curve processing actions.
    """
    def __init__(self, parent=None, name="Scan Window", fit=True, backend=None,
                 plugins=True, control=True, position=True, roi=True,
                 specfit=None, info=False):
        super(ScanWindow, self).__init__(parent,
                                         backend=backend,
                                         roi=roi,
                                         control=control,
                                         position=position,
                                         mask=False,
                                         colormap=False)
        self.setDataMargins(0, 0, 0.025, 0.025)
        self.setPanWithArrowKeys(True)
        self._plotType = "SCAN"     # needed by legacy plugins

        self.setWindowTitle(name)

        # Toolbar
        # self.toolBar().setIconSize(qt.QSize(15, 18))
        self._toolbar = qt.QToolBar(self)
        # self._toolbar.setIconSize(qt.QSize(15, 18))

        self.addToolBar(self._toolbar)

        if fit:
            # attr needed by scanFitToolButton
            self.scanFit = ScanFit.ScanFit(specfit=specfit)
            scanFitToolButton = ScanFitToolButton(self)
            self._toolbar.addWidget(scanFitToolButton)

        self.avgAction = SimpleActions.AverageAction(plot=self)
        self.derivativeAction = SimpleActions.DerivativeAction(plot=self)
        self.smoothAction = SimpleActions.SmoothAction(plot=self)
        self.swapSignAction = SimpleActions.SwapSignAction(plot=self)
        self.yMinToZero = SimpleActions.YMinToZeroAction(plot=self)
        self.subtractAction = SimpleActions.SubtractAction(plot=self)

        self._toolbar.addAction(self.avgAction)
        self._toolbar.addAction(self.derivativeAction)
        self._toolbar.addAction(self.smoothAction)
        self._toolbar.addAction(self.swapSignAction)
        self._toolbar.addAction(self.yMinToZero)
        self._toolbar.addAction(self.subtractAction)

        if plugins:
            pluginsToolButton = PluginsToolButton(plot=self)

            if PLUGINS_DIR is not None:
                if isinstance(PLUGINS_DIR, list):
                    pluginDir = PLUGINS_DIR
                else:
                    pluginDir = [PLUGINS_DIR]
                pluginsToolButton.getPlugins(
                        method="getPlugin1DInstance",
                        directoryList=pluginDir)
            self._toolbar.addWidget(pluginsToolButton)

        self.scanWindowInfoWidget = None
        self.infoDockWidget = None
        if info:
            self.scanWindowInfoWidget = ScanWindowInfoWidget.\
                                            ScanWindowInfoWidget()
            self.infoDockWidget = qt.QDockWidget(self)
            self.infoDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.infoDockWidget.setWidget(self.scanWindowInfoWidget)
            self.infoDockWidget.setWindowTitle(self.windowTitle() + " Info")
            self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                               self.infoDockWidget)

            self.sigActiveCurveChanged.connect(self.__updateInfoWidget)

    def __updateInfoWidget(self, previous_legend, legend):
        x, y, legend, info, params = self.getCurve(legend)
        self.scanWindowInfoWidget.updateFromXYInfo(x, y, info)

    # TODO: toggleInfoWidget (method and control menu entry)


def test():
    import numpy
    app = qt.QApplication([])
    w = ScanWindow(info=True)
    x = numpy.arange(1000.)
    y1 = 10 * x + 10000. * numpy.exp(-0.5*(x-500)*(x-500)/400)
    y2 = y1 + 5000. * numpy.exp(-0.5*(x-700)*(x-700)/200)
    y3 = y1 + 7000. * numpy.exp(-0.5*(x-200)*(x-200)/1000)
    w.addCurve(x, y1, legend="dummy1",
               info={"SourceName": "Synthetic data 1 (linear+gaussian)",
                     "hkl": [1.1, 1.2, 1.3],
                     "Header": ["#S 1 toto"]})
    w.addCurve(x, y2, legend="dummy2",
               info={"SourceName": "Synthetic data 2",
                     "hkl": [2.1, 2.2, 2.3],
                     "Header": ["#S 2"]})
    w.addCurve(x, y3, legend="dummy3",
               info={"SourceName": "Synthetic data 3",
                     "hkl": ["3.1", 3.2, 3.3],
                     "Header": ["#S 3"]})
    w.resetZoom()
    app.lastWindowClosed.connect(app.quit)
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
