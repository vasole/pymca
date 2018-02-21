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
"""Plugin to apply a median filter on the ROI stack.

The mask of the plot widget is synchronized with the master stack widget.
"""

__authors__ = ["H. Payno"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"


from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5 import StackPluginBase
import logging
logger = logging.getLogger(__file__)

try:
    from PyMca5.PyMcaGui.pymca.TomographyRecons import TomoReconsDialog
    from tomogui.gui.ProjectWidget import ProjectWindow as TomoguiProjWindow
except ImportError as e:
    logger.warning(e)
    TomoReconsDialog = None

class ReconsStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'selectRecons': [self.selectRecons,
                                            "select reconstruction"]}
        self.__methodKeys = ['selectRecons']
        self.sinograms = {}
        self._tomoguiWindow = None

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

    def _updateSinoDict(self):
        self.sinograms = {}
        sinograms, self.sinoNames = self.getStackROIImagesAndNames()
        for sinoName, sinogram in zip(self.sinoNames, sinograms):
            self.sinograms[sinoName] = sinogram

    def selectRecons(self):
        def getSinograms(sinoNames):
            res = []
            for sinoName in sinoNames:
                res.append(self.sinograms[sinoName])
            return res

        if TomoReconsDialog is None:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setWindowTitle('Cannot load plugin')
            msg.setText("tomogui is not installed, can't run reconstruction.")
            msg.setInformativeText("To install tomogui see the installation "
                                   "procedure: \n"
                                   "http://edna-site.org/pub/doc/tomogui/latest/install.html")
            msg.exec_()
            return

        self._updateSinoDict()
        diag = TomoReconsDialog(entries=self.sinoNames)
        if diag.exec_():
            if self._tomoguiWindow is None:
                self._tomoguiWindow = TomoguiProjWindow()

            self._tomoguiWindow.clean()
            reconsType = diag.getReconstructionType()
            sinoNames = diag.getSinogramsToRecons()
            self._tomoguiWindow.setSinoToRecons(reconsType=reconsType,
                                                sinograms=getSinograms(sinoNames),
                                                names=sinoNames)
            if diag.hasIt() is True:
                it = self.sinograms[diag.getIt()]
                self._tomoguiWindow.setIt(it=it, name=diag.getIt())
            if diag.hasI0() is True:
                i0 = self.sinograms[diag.getI0()]
                self._tomoguiWindow.setI0(i0=i0, name=diag.getI0())
            # by default do not reconstruct log
            self._tomoguiWindow.setLogRecons(False)
            self._tomoguiWindow.show()


MENU_TEXT = "Reconstruction"


def getStackPluginInstance(stackWindow):
    ob = ReconsStackPlugin(stackWindow)
    return ob
