#/*##########################################################################
# Copyright (C) 2004-2019 V.A. Sole, European Synchrotron Radiation Facility
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
"""This module defines a QToolButton opening a plugin menu when clicked:

    - :class:`PluginsToolButton`

This button takes a plot object as constructor parameter.
The plot can be a legacy *PyMca* plot widget, or a *silx* plot widget.
The minor API incompatibilities between the plugins and the *silx*
plot widget are solved using a proxy class (:class:`PlotProxySilx`).

This button inherits :class:`PluginLoader` to load the plugins.

It also acts as a plot proxy to dynamically provide the plot methods needed
by the plugins. This is done in its method :meth:`PluginsToolButton.__getattr__`
which looks for needed methods in :attr:`PluginsToolButton.plot`.
"""

# TODO: we should probably support future plugins using the new silx plot API
# (e.g. expecting 5 return values from getActiveCurve() ...)

import logging
import os
import sys
import traceback
import weakref

from PyMca5.PyMcaGui import PyMcaQt as qt

from PyMca5.PyMcaGraph.PluginLoader import PluginLoader
from PyMca5.PyMcaGui.plotting.PyMca_Icons import IconDict


_logger = logging.getLogger(__name__)


def _toggleLogger():
    """Toggle logger level for logging.DEBUG to logging.WARNING
    and vice-versa."""
    if _logger.getEffectiveLevel() == logging.DEBUG:
        _logger.setLevel(logging.WARNING)
    else:
        _logger.setLevel(logging.DEBUG)


class PluginsToolButton(qt.QToolButton, PluginLoader):
    """Toolbutton providing a context menu loaded with PyMca plugins.
    It behaves as a proxy for accessing the plot methods from the plugins.

    :param plot: reference to related plot widget
    :param parent: Parent QWidget widget
    """

    def __init__(self, plot, parent=None, method="getPlugin1DInstance"):

        qt.QToolButton.__init__(self, parent)
        self.setIcon(qt.QIcon(qt.QPixmap(IconDict["plugin"])))
        if method == "getPlugin1DInstance":
            self.setToolTip("Call/Load 1D Plugins")
        elif method == "getPlugin2DInstance":
            self.setToolTip("Call/Load 2D Plugins")

        # fill attr pluginList and pluginInstanceDict with existing plugins
        PluginLoader.__init__(self, method=method)

        # plugins expect a legacy API, not the silx Plot API
        self.plot = weakref.proxy(plot, self._ooPlotDestroyed)
        self._plotType = getattr(self.plot, "_plotType", None)

        self.clicked.connect(self._pluginClicked)

    def _ooPlotDestroyed(self, obj=None):
        self.setEnabled(False)

    def __getattr__(self, attr):
        """Plot API for plugins: forward calls for unknown
        methods to :attr:`plot`."""
        try:
            return getattr(self.plot, attr)
        except AttributeError:
            # blame plot class for missing attribute, not PluginsToolButton
            raise AttributeError(
                    self.plot.__class__.__name__ + " has no attribute " + attr)

    def _connectPlotSignals(self):
        for name, plugin in self.pluginInstanceDict.items():
            if hasattr(plugin, "activeCurveChanged") and callable(plugin.activeCurveChanged):
                # Can we just assume it has the proper signature?
                self.plot.sigActiveCurveChanged.connect(plugin.activeCurveChanged)
            if hasattr(plugin, "activeImageChanged") and callable(plugin.activeImageChanged):
                # Can we just assume it has the proper signature?
                self.plot.sigActiveImageChanged.connect(plugin.activeImageChanged)

    def _disconnectPlotSignals(self):
        for name, plugin in self.pluginInstanceDict.items():
            if hasattr(plugin, "activeCurveChanged") and callable(plugin.activeCurveChanged):
                # Can we just assume it has the proper signature?
                self.plot.sigActiveCurveChanged.disconnect(plugin.activeCurveChanged)
            if hasattr(plugin, "activeImageChanged") and callable(plugin.activeImageChanged):
                # Can we just assume it has the proper signature?
                self.plot.sigActiveImageChanged.disconnect(plugin.activeImageChanged)

    def getPlugins(self, method=None, directoryList=None, exceptions=False):
        """method overloaded to update signal connections when loading plugins"""
        self._disconnectPlotSignals()
        PluginLoader.getPlugins(self, method, directoryList, exceptions)
        self._connectPlotSignals()

    def _pluginClicked(self):
        actionNames = []
        menu = qt.QMenu(self)
        menu.addAction("Reload Plugins")
        actionNames.append("Reload Plugins")
        menu.addAction("Set User Plugin Directory")
        actionNames.append("Set User Plugin Directory")

        if _logger.getEffectiveLevel() == logging.DEBUG:
            text = "Toggle DEBUG mode OFF"
        else:
            text = "Toggle DEBUG mode ON"

        menu.addAction(text)
        menu.addSeparator()
        actionNames.append(text)
        callableKeys = ["Dummy0", "Dummy1", "Dummy2"]
        pluginInstances = self.pluginInstanceDict
        for pluginName in self.pluginList:
            if pluginName in ["PyMcaPlugins.Plugin1DBase", "Plugin1DBase"]:
                continue
            module = sys.modules[pluginName]
            if hasattr(module, 'MENU_TEXT'):
                text = module.MENU_TEXT
            else:
                text = os.path.basename(module.__file__)
                if text.endswith('.pyc'):
                    text = text[:-4]
                elif text.endswith('.py'):
                    text = text[:-3]

            methods = pluginInstances[pluginName].getMethods(
                    plottype=self._plotType)
            if not len(methods):
                continue
            elif len(methods) == 1:
                pixmap = pluginInstances[pluginName].getMethodPixmap(methods[0])
                tip = pluginInstances[pluginName].getMethodToolTip(methods[0])
                if pixmap is not None:
                    action = qt.QAction(qt.QIcon(qt.QPixmap(pixmap)), text, self)
                else:
                    action = qt.QAction(text, self)
                if tip is not None:
                    action.setToolTip(tip)
                menu.addAction(action)
            else:
                menu.addAction(text)
            actionNames.append(text)
            callableKeys.append(pluginName)
        menu.hovered.connect(self._actionHovered)
        a = menu.exec_(qt.QCursor.pos())
        if a is None:
            return None

        idx = actionNames.index(a.text())
        if a.text() == "Reload Plugins":
            n, message = self.getPlugins(exceptions=True)
            if n < 1:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                msg.setWindowTitle("No plugins")
                msg.setInformativeText(" Problem loading plugins ")
                msg.setDetailedText(message)
                msg.exec()
            return
        if a.text() == "Set User Plugin Directory":
            dirName = qt.QFileDialog.getExistingDirectory(
                    self,
                    "Enter user plugins directory",
                    os.getcwd())
            if len(dirName):
                pluginsDir = self.getPluginDirectoryList()
                pluginsDirList = [pluginsDir[0], dirName]
                self.setPluginDirectoryList(pluginsDirList)
            return
        if "Toggle DEBUG mode" in a.text():
            _toggleLogger()
            return
        key = callableKeys[idx]

        methods = pluginInstances[key].getMethods(
                plottype=self._plotType)
        if len(methods) == 1:
            idx = 0
        else:
            actionNames = []
            # allow the plugin designer to specify the order
            #methods.sort()
            menu = qt.QMenu(self)
            for method in methods:
                text = method
                pixmap = pluginInstances[key].getMethodPixmap(method)
                tip = pluginInstances[key].getMethodToolTip(method)
                if pixmap is not None:
                    action = qt.QAction(qt.QIcon(qt.QPixmap(pixmap)), text, self)
                else:
                    action = qt.QAction(text, self)
                if tip is not None:
                    action.setToolTip(tip)
                menu.addAction(action)
                actionNames.append((text, pixmap, tip, action))
            #qt.QObject.connect(menu, qt.SIGNAL("hovered(QAction *)"), self._actionHovered)
            menu.hovered.connect(self._actionHovered)
            a = menu.exec_(qt.QCursor.pos())
            if a is None:
                return None
            idx = -1
            for action in actionNames:
                if a.text() == action[0]:
                    idx = actionNames.index(action)
        try:
            pluginInstances[key].applyMethod(methods[idx])
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Plugin error")
            msg.setText("An error has occured while executing the plugin:")
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec()

    def _actionHovered(self, action):
        # from PyMca5 PlotWindow
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)
        else:
            qt.QToolTip.hideText()
