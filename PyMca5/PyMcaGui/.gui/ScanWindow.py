#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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

import numpy
import os
import sys
import traceback

import PyMca5
from PluginsToolButton import PluginsToolButton

from silx.gui import qt
from silx.gui.plot import PlotWindow

DEBUG = 0    # fixme: replace with logging

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
                 plugins=True, newplot=True, roi=True,
                 control=True, position=True, info=False, **kw):

        super(ScanWindow, self).__init__(parent,
                                         backend=backend,
                                         roi=roi,
                                         fit=fit,
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

        if plugins:
            self.pluginsToolButton = PluginsToolButton(plot=self, parent=self)
            self.pluginsToolButton.clicked.connect(self._pluginClicked)

            self.toolBar().addWidget(self.pluginsToolButton)

            if PLUGINS_DIR is not None:
                if isinstance(PLUGINS_DIR, list):
                    pluginDir = PLUGINS_DIR
                else:
                    pluginDir = [PLUGINS_DIR]
                self.pluginsToolButton.getPlugins(method="getPlugin1DInstance",
                               directoryList=pluginDir)
                # plugin name
                for pname in self.pluginsToolButton.pluginList:
                    print(pname, ":")
                    # method name
                    # for mname in self.pluginsToolButton.pluginInstanceDict[pname].methodDict:
                    #     print("\t", mname, ":")
                    #     print("\t\t",  self.pluginsToolButton.pluginInstanceDict[pname].methodDict[mname])

    def _pluginClicked(self):
        # actionNames = [] # actionNames = [a.text for a in menu.actions()]
        actionNames = []
        menu = qt.QMenu(self)
        menu.addAction("Reload Plugins")
        actionNames.append("Reload Plugins")
        menu.addAction("Set User Plugin Directory")
        actionNames.append("Set User Plugin Directory")
        global DEBUG
        if DEBUG:
            text = "Toggle DEBUG mode OFF"
        else:
            text = "Toggle DEBUG mode ON"
        menu.addAction(text)
        menu.addSeparator()
        actionNames.append(text)
        callableKeys = ["Dummy0", "Dummy1", "Dummy2"]
        pluginInstances = self.pluginsToolButton.pluginInstanceDict
        for pluginName in self.pluginsToolButton.pluginList:
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

            methods = pluginInstances[pluginName].getMethods(plottype=self._plotType)
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
            n, message = self.pluginsToolButton.getPlugins(exceptions=True)
            if n < 1:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                msg.setWindowTitle("No plugins")
                msg.setInformativeText(" Problem loading plugins ")
                msg.setDetailedText(message)
                msg.exec_()
            return
        if a.text() == "Set User Plugin Directory":
            dirName = qt.QFileDialog.getExistingDirectory(
                    self,
                    "Enter user plugins directory",
                    os.getcwd())
            if len(dirName):
                pluginsDir = self.pluginsToolButton.getPluginDirectoryList()
                pluginsDirList = [pluginsDir[0], dirName]
                self.pluginsToolButton.setPluginDirectoryList(pluginsDirList)
            return
        if "Toggle DEBUG mode" in a.text():
            if DEBUG:
                DEBUG = 0
            else:
                DEBUG = 1
            return
        key = callableKeys[idx]

        methods = pluginInstances[key].getMethods(plottype=self._plotType)
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
            msg.exec_()

    def _actionHovered(self, action):
        # from PyMca5 PlotWindow
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)
        else:
            qt.QToolTip.hideText()


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
