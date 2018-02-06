#/*##########################################################################
# Copyright (C) 20016-2017 European Synchrotron Radiation Facility
#
# This file is part of tomogui. Interface for tomography developed at
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
__author__ = ["H. Payno"]
__contact__ = "henri.payno@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""
Plugin to call tomogui (https://gitlab.esrf.fr/tomoTools/tomogui) in order to
run tomography reconstruction based on silx (https://github.com/silx-kit/silx)
or freeart (https://gitlab.esrf.fr/freeart/freeart)
"""
from PyMca5 import StackPluginBase
import logging
logger = logging.getLogger(__name__)
try:
    import tomogui
    # neeed at least tomogui 0.2 to work
    if tomogui._version.MINOR < 2:
        logger.warning('tomogui version is to old, please install v0.2 at least')
        tomogui = None
    else:
        from tomogui.gui.NewProjectDialog import NewProjectDialog
        from tomogui.gui.ProjectWidget import ProjectWindow
        from silx.gui import qt
except ImportError:
    logger.info('Tomography plugin disabled, tomogui not found')
    tomogui = None

if tomogui:
    DEBUG = 0

    MENU_TEXT = "Reconstruction"

    class TomographyPlugin(StackPluginBase.StackPluginBase):
        def __init__(self, stackWindow, **kw):
            StackPluginBase.DEBUG = DEBUG
            StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
            self.methodDict = {}
            self.methodDict["runReconstruction"] = [self._runRecons,
                                                    "Run the reconstruction for the given sinogram",
                                                    None]
            self._widget = None

        def getMethods(self, plottype=None):
            """
            A list with the NAMES  associated to the callable methods
            that are applicable to the specified plot.

            Plot type can be "SCAN", "MCA", None, ...
            """
            names = list(self.methodDict.keys())
            names.sort()
            return names

        def getMethodToolTip(self, name):
            """
            Returns the help associated to the particular method name or None.
            """
            return self.methodDict[name][1]

        def getMethodPixmap(self, name):
            """
            Returns the pixmap associated to the particular method name or None.
            """
            return self.methodDict[name][2]

        def applyMethod(self, name):
            """
            The plugin is asked to apply the method associated to name.
            """
            self.methodDict[name][0]()
            return

        def _runRecons(self):
            diag = NewProjectDialog()
            if diag.exec_():
                reconsType = NewProjectDialog.resToRtype(diag.result())
                if reconsType:
                    self.mainWindow = ProjectWindow()
                    self.mainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)
                    self.mainWindow.setSinoToRecons(
                        reconsType=reconsType,
                        sinograms=[self.getStackData()])
                    # by default do not reconstruct log
                    self.mainWindow.setLogRecons(False)
                    self.mainWindow.show()
            else:
                logger.info('reconstruction has been cancel')


def getStackPluginInstance(stackWindow, **kw):
    if tomogui:
        ob = TomographyPlugin(stackWindow)
        return ob
    else:
        return None
