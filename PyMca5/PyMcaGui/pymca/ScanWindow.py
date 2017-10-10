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
import copy
import logging
import numpy

from silx.gui.plot import PlotWindow

import PyMca5
from PyMca5.PyMcaGui.pymca import ScanWindowInfoWidget
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.PluginsToolButton import PluginsToolButton
from PyMca5.PyMcaGui.math import SimpleActions
from PyMca5.PyMcaGui.pymca import ScanFit
from PyMca5.PyMcaGui.pymca.ScanFitToolButton import ScanFitToolButton
from PyMca5.PyMcaCore import DataObject


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


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class BaseScanWindow(PlotWindow):
    """:class:`PlotWindow` augmented with plugins, fitting actions,
    a widget for displaying scan metadata and simple curve processing actions.
    """
    def __init__(self, parent=None, name="Scan Window", fit=True, backend=None,
                 plugins=True, control=True, position=True, roi=True,
                 specfit=None, info=False):
        super(BaseScanWindow, self).__init__(parent,
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
            self.infoDockWidget.setWindowTitle("Scan Info")
            self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                               self.infoDockWidget)

            self.sigActiveCurveChanged.connect(self.__updateInfoWidget)

    def _customControlButtonMenu(self):
        """Display Options button sub-menu. Overloaded to add
        _toggleInfoAction"""
        # overloaded from PlotWindow to add "Show/Hide Info"
        controlMenu = self.controlButton.menu()
        controlMenu.clear()
        controlMenu.addAction(self.getLegendsDockWidget().toggleViewAction())

        if self.infoDockWidget is not None:
            controlMenu.addAction(self.infoDockWidget.toggleViewAction())
        controlMenu.addAction(self.getRoiAction())
        controlMenu.addAction(self.getMaskAction())
        controlMenu.addAction(self.getConsoleAction())

        controlMenu.addSeparator()
        controlMenu.addAction(self.getCrosshairAction())
        controlMenu.addAction(self.getPanWithArrowKeysAction())

    def __updateInfoWidget(self, previous_legend, legend):
        x, y, legend, info, params = self.getCurve(legend)
        self.scanWindowInfoWidget.updateFromXYInfo(x, y, info)

    def setWindowType(self, wtype=None):
        if wtype not in [None, "SCAN", "MCA"]:
            raise AttributeError("Unsupported window type %s." % wtype)
        self._plotType = wtype


class ScanWindow(BaseScanWindow):
    """ScanWindow, adding dataObject management to BaseScanWindow
    """

    def __init__(self, parent=None, name="Scan Window", fit=True, backend=None,
                 plugins=True, control=True, position=True, roi=True,
                 specfit=None, info=False):
        BaseScanWindow.__init__(self,
                                parent, name, fit, backend,
                                plugins, control, position, roi,
                                specfit, info)

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

    def _addSelection(self, selectionlist, resetzoom=True, replot=None):
        """Add curves to plot and data objects to :attr:`dataObjectsDict`
        """
        _logger.debug("_addSelection(self, selectionlist) " +
                      str(selectionlist))
        if replot is not None:
            _logger.warning(
                    'deprecated replot argument, use resetzoom instead')
            resetzoom = replot and resetzoom

        sellist = selectionlist if isinstance(selectionlist, list) else \
            [selectionlist]

        if len(self.getAllCurves(just_legend=True)):
            activeCurve = self.getActiveCurve(just_legend=True)
        else:
            activeCurve = None
        nSelection = len(sellist)
        for selectionIndex in range(nSelection):
            sel = sellist[selectionIndex]
            key = sel['Key']
            legend = sel['legend']  # expected form sourcename + scan key
            if "scanselection" not in sel or not sel["scanselection"] or \
                            sel['scanselection'] == "MCA":
                continue
            if len(key.split(".")) > 2:
                continue
            dataObject = sel['dataobject']
            # only one-dimensional selections considered
            if dataObject.info["selectiontype"] != "1D":
                continue

            # there must be something to plot
            if not hasattr(dataObject, 'y'):
                continue
            if getattr(dataObject, 'x', None) is None:
                ylen = len(dataObject.y[0])
                if not ylen:
                    # nothing to be plot
                    continue
                xdata = numpy.arange(ylen).astype(numpy.float)
            elif len(dataObject.x) > 1:
                # mesh plot
                continue
            else:
                xdata = dataObject.x[0]

            if sel.get('SourceType') == "SPS":
                ycounter = -1
                if 'selection' not in dataObject.info:
                    dataObject.info['selection'] = copy.deepcopy(sel['selection'])
                for ydata in dataObject.y:
                    xlabel = None
                    ylabel = None
                    ycounter += 1
                    # normalize ydata with monitor
                    if dataObject.m is not None and len(dataObject.m[0]) > 0:
                        if len(dataObject.m[0]) != len(ydata):
                            raise ValueError("Monitor data length different than counter data")
                        index = numpy.nonzero(dataObject.m[0])[0]
                        if not len(index):
                            continue
                        xdata = numpy.take(xdata, index)
                        ydata = numpy.take(ydata, index)
                        mdata = numpy.take(dataObject.m[0], index)
                        # A priori the graph only knows about plots
                        ydata = ydata / mdata
                    ylegend = 'y%d' % ycounter
                    if isinstance(dataObject.info['selection'], dict):
                        if 'x' in dataObject.info['selection']:
                            # proper scan selection
                            ilabel = dataObject.info['selection']['y'][ycounter]
                            ylegend = dataObject.info['LabelNames'][ilabel]
                            ylabel = ylegend
                            if sel['selection']['x'] is not None:
                                if len(dataObject.info['selection']['x']):
                                    xlabel = dataObject.info['LabelNames'] \
                                        [dataObject.info['selection']['x'][0]]
                    dataObject.info["xlabel"] = xlabel
                    dataObject.info["ylabel"] = ylabel
                    newLegend = legend + " " + ylegend
                    self.dataObjectsDict[newLegend] = dataObject
                    self.addCurve(xdata, ydata, legend=newLegend, info=dataObject.info,
                                  xlabel=xlabel, ylabel=ylabel)
                    if self.scanWindowInfoWidget is not None:
                        if not self.infoDockWidget.isHidden():
                            activeLegend = self.getActiveCurve(just_legend=True)
                            if activeLegend == newLegend:
                                self.scanWindowInfoWidget.updateFromDataObject \
                                    (dataObject)
                            else:
                                # TODO: better to implement scanWindowInfoWidget.clear
                                dummyDataObject = DataObject.DataObject()
                                dummyDataObject.y = [numpy.array([])]
                                dummyDataObject.x = [numpy.array([])]
                                self.scanWindowInfoWidget.updateFromDataObject(dummyDataObject)
            else:
                # we have to loop for all y values
                ycounter = -1
                for ydata in dataObject.y:
                    ylen = len(ydata)
                    if ylen == 1 and len(xdata) > 1:
                        ydata = ydata[0] * numpy.ones(len(xdata)).astype(numpy.float)
                    elif len(xdata) == 1:
                        xdata = xdata[0] * numpy.ones(ylen).astype(numpy.float)
                    ycounter += 1
                    newDataObject = DataObject.DataObject()
                    newDataObject.info = copy.deepcopy(dataObject.info)
                    if dataObject.m is None or len(dataObject.m[0]) == 0:
                        mdata = numpy.ones(len(ydata)).astype(numpy.float)
                    elif len(dataObject.m[0]) == len(ydata):
                        index = numpy.nonzero(dataObject.m[0])[0]
                        if not len(index):
                            continue
                        xdata = numpy.take(xdata, index)
                        ydata = numpy.take(ydata, index)
                        mdata = numpy.take(dataObject.m[0], index)
                        # A priori the graph only knows about plots
                        ydata = ydata / mdata
                    elif len(dataObject.m[0]) == 1:
                        mdata = numpy.ones(len(ydata)).astype(numpy.float)
                        mdata *= dataObject.m[0][0]
                        index = numpy.nonzero(dataObject.m[0])[0]
                        if not len(index):
                            continue
                        xdata = numpy.take(xdata, index)
                        ydata = numpy.take(ydata, index)
                        mdata = numpy.take(dataObject.m[0], index)
                        # A priori the graph only knows about plots
                        ydata = ydata / mdata
                    else:
                        raise ValueError("Monitor data length different than counter data")

                    newDataObject.x = [xdata]
                    newDataObject.y = [ydata]
                    newDataObject.m = [mdata]
                    newDataObject.info['selection'] = copy.deepcopy(sel['selection'])
                    ylegend = 'y%d' % ycounter
                    xlabel = None
                    ylabel = None
                    if isinstance(sel['selection'], dict) and 'x' in sel['selection']:
                        # proper scan selection
                        newDataObject.info['selection']['x'] = sel['selection']['x']
                        newDataObject.info['selection']['y'] = [sel['selection']['y'][ycounter]]
                        newDataObject.info['selection']['m'] = sel['selection']['m']
                        ilabel = newDataObject.info['selection']['y'][0]
                        ylegend = newDataObject.info['LabelNames'][ilabel]
                        ylabel = ylegend
                        if len(newDataObject.info['selection']['x']):
                            ilabel = newDataObject.info['selection']['x'][0]
                            xlabel = newDataObject.info['LabelNames'][ilabel]
                        else:
                            xlabel = "Point number"
                    if ('operations' in dataObject.info) and len(dataObject.y) == 1:
                        newDataObject.info['legend'] = legend
                        symbol = 'x'
                    else:
                        symbol = None
                        newDataObject.info['legend'] = legend + " " + ylegend
                        newDataObject.info['selectionlegend'] = legend
                    yaxis = None
                    if "plot_yaxis" in dataObject.info:
                        yaxis = dataObject.info["plot_yaxis"]
                    elif 'operations' in dataObject.info:
                        if dataObject.info['operations'][-1] == 'derivate':
                            yaxis = 'right'
                    self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
                    self.addCurve(xdata, ydata, legend=newDataObject.info['legend'],
                                  info=newDataObject.info,
                                  symbol=symbol,
                                  yaxis=yaxis,
                                  xlabel=xlabel,
                                  ylabel=ylabel)
        try:
            if activeCurve is None:
                self.setActiveCurve(self._curveList[0])
        finally:
            if resetzoom:
                self.resetZoom()

    def _removeSelection(self, selectionlist):
        _logger.debug("_removeSelection(self, selectionlist) " +
                      str(selectionlist))

        sellist = selectionlist if isinstance(selectionlist, list) else \
            [selectionlist]

        removelist = []
        for sel in sellist:
            key = sel['Key']
            if "scanselection" not in sel or not sel["scanselection"]:
                continue
            if sel['scanselection'] == "MCA":
                continue
            if len(key.split(".")) > 2:
                continue

            legend = sel['legend']  # expected form sourcename + scan key
            if isinstance(sel['selection'], dict) and 'y' in sel['selection']:
                for lName in ['cntlist', 'LabelNames']:
                    if lName in sel['selection']:
                        for index in sel['selection']['y']:
                            removelist.append(legend + " " +
                                              sel['selection'][lName][index])

        if len(removelist):
            self.removeCurves(removelist)

    def _replaceSelection(self, selectionlist):
        """Delete existing curves and data objects, then add new selection.
        """
        _logger.debug("_replaceSelection(self, selectionlist) " +
                      str(selectionlist))

        sellist = selectionlist if isinstance(selectionlist, list) else \
            [selectionlist]

        doit = False
        for sel in sellist:
            if "scanselection" not in sel or not sel["scanselection"]:
                continue
            if sel['scanselection'] == "MCA":
                continue
            if len(sel["Key"].split(".")) > 2:
                continue
            dataObject = sel['dataobject']
            if dataObject.info["selectiontype"] == "1D":
                if hasattr(dataObject, 'y'):
                    doit = True
                    break
        if not doit:
            return
        self.clearCurves()
        self.dataObjectsDict = {}
        self._addSelection(selectionlist, resetzoom=True)

    def removeCurves(self, removeList):
        for legend in removeList:
            self.removeCurve(legend)
            if legend in self.dataObjectsDict:
                del self.dataObjectsDict[legend]

    def addCurve(self, x, y, legend=None, info=None, replace=False,
                 resetzoom=False, replot=True, color=None, symbol=None,
                 linestyle=None, xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, **kw):
        """Add a curve. If a curve with the same legend already exists,
        the unspecified parameters (color, symbol, linestyle, yaxis) are
        assumed to be identical to the parameters of the existing curve."""
        if legend in self._curveList:
            if info is None:
                info = {}
            oldStuff = self.getCurve(legend)
            if oldStuff is not None:
                oldX, oldY, oldLegend, oldInfo, oldParams = oldStuff
            else:
                oldInfo = {}
            if color is None:
                color = info.get("plot_color",
                                 oldInfo.get("plot_color", None))
            if symbol is None:
                symbol = info.get("plot_symbol",
                                  oldInfo.get("plot_symbol", None))
            if linestyle is None:
                linestyle = info.get("plot_linestyle",
                                     oldInfo.get("plot_linestyle", None))
            if yaxis is None:
                yaxis = info.get("plot_yaxis",
                                 oldInfo.get("plot_yaxis", None))
        else:
            if info is None:
                info = {}
            if color is None:
                color = info.get("plot_color", None)
            if symbol is None:
                symbol = info.get("plot_symbol", None)
            if linestyle is None:
                linestyle = info.get("plot_linestyle", None)
            if yaxis is None:
                yaxis = info.get("plot_yaxis", None)
        if legend in self.dataObjectsDict:
            # the info is changing
            super(ScanWindow, self).addCurve(
                    x, y, legend=legend, info=info, replot=replot,
                    replace=replace, color=color, symbol=symbol,
                    linestyle=linestyle, xlabel=xlabel, ylabel=ylabel,
                    yaxis=yaxis, xerror=xerror, yerror=yerror,
                    resetzoom=resetzoom, **kw)
        else:
            # create the data object
            self.newCurve(
                    x, y, legend=legend, info=info, replot=replot,
                    replace=replace, color=color, symbol=symbol,
                    linestyle=linestyle, xlabel=xlabel, ylabel=ylabel,
                    yaxis=yaxis, xerror=xerror, yerror=yerror,
                    resetzoom=resetzoom, **kw)

    def newCurve(self, x, y, legend=None, info=None, replace=False,
                 resetzoom=False, replot=True, color=None, symbol=None,
                 linestyle=None, xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, **kw):
        """
        Create and add a data object to :attr:`dataObjectsDict`
        """
        if legend is None:
            legend = "Unnamed curve 1.1"
        if xlabel is None:
            xlabel = "X"
        if ylabel is None:
            ylabel = "Y"
        if info is None:
            info = {}
        if color is not None:
            info["plot_color"] = color
        if symbol is not None:
            info["plot_symbol"] = symbol
        if linestyle is not None:
            info["plot_linestyle"] = linestyle
        if yaxis is not None:
            info["plot_yaxis"] = yaxis

        newDataObject = DataObject.DataObject()
        newDataObject.x = [x]
        newDataObject.y = [y]
        newDataObject.m = None
        newDataObject.info = copy.deepcopy(info)
        newDataObject.info['legend'] = legend
        newDataObject.info['SourceName'] = legend
        newDataObject.info['Key'] = ""
        newDataObject.info['selectiontype'] = "1D"
        newDataObject.info['LabelNames'] = [xlabel, ylabel]
        newDataObject.info['selection'] = {'x': [0], 'y': [1]}

        sel = {'SourceType': "Operation",
               'SourceName': legend,
               'Key': "",
               'legend': legend,
               'dataobject': newDataObject,
               'scanselection': True,
               'selection': {'x': [0], 'y': [1], 'm': [],
                             'cntlist': [xlabel, ylabel]},
               'selectiontype': "1D"}
        sel_list = [sel]
        if replace:
            self._replaceSelection(sel_list)
        else:
            self._addSelection(sel_list, resetzoom=replot)


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
