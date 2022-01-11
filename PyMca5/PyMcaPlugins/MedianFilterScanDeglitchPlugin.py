#/*##########################################################################
# Copyright (C) 2004-2020 T.Rueter, European Synchrotron Radiation Facility
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
"""This plugin uses a median filter smoothing to remove glitches from curves.
A configuration widget can be used to configure the median filter (width and
threshold), and the user can choose to apply it on the active curve or on all
curves.

"""
__author__ = "T. Rueter - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from PyMca5 import Plugin1DBase
from PyMca5.PyMcaMath.PyMcaSciPy.signal import medfilt1d
from PyMca5.PyMcaGui import PyMcaQt as qt
import numpy
import logging

_logger = logging.getLogger(__name__)


class MedianFilterScanDeglitchPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self,  plotWindow,  **kw):
        Plugin1DBase.Plugin1DBase.__init__(self,  plotWindow,  **kw)
        self.methodDict = {
            'Apply to active curve':
                [self.removeSpikesActive,
                 'Apply sliding median filter to active curve',
                 None],
            'Apply to all curves':
                [self.removeSpikesAll,
                 'Apply sliding median filter to all curves',
                 None],
            'Configure median filter':
                [self.configureFilter,
                 'Set threshold and width of the filter',
                 None]
        }
        self._methodList = ['Configure median filter',
                            'Apply to active curve',
                            'Apply to all curves']
        self.threshold = 0.66
        self.width = 9
        self._widget = None

    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...
        """
        return list(self._methodList)

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

    def configureFilter(self):
        if self._widget is None:
            # construct a widget
            msg = qt.QDialog()
            msg.setWindowTitle("Deglitch Configuration")
            msgLayout = qt.QGridLayout()
            buttonLayout = qt.QHBoxLayout()

            inpThreshold = qt.QDoubleSpinBox()
            inpThreshold.setRange(0.,10.)
            inpThreshold.setSingleStep(.1)
            inpThreshold.setValue(self.threshold)
            inpThreshold.setToolTip('Increase width for broad spikes')

            inpWidth = qt.QSpinBox()
            inpWidth.setRange(1,101)
            inpWidth.setSingleStep(2)
            inpWidth.setValue(self.width)
            inpWidth.setToolTip('Set low threshold for multiple spikes of different markedness')

            labelWidth = qt.QLabel('Width (must be odd)')
            labelThreshold = qt.QLabel('Threshold (multiple of deviation)')
            buttonOK = qt.QPushButton('Ok')
            buttonOK.clicked.connect(msg.accept)
            buttonCancel = qt.QPushButton('Cancel')
            buttonCancel.clicked.connect(msg.reject)

            allActiveBG = qt.QButtonGroup()
            buttonAll = qt.QCheckBox('Apply to All')
            buttonActive = qt.QCheckBox('Apply to Active')
            allActiveBG.addButton(buttonAll, 0)
            allActiveBG.addButton(buttonActive, 1)
            buttonActive.setChecked(True)

            buttonLayout.addWidget(qt.HorizontalSpacer())
            buttonLayout.addWidget(buttonOK)
            buttonLayout.addWidget(buttonCancel)

            msgLayout.addWidget(labelWidth,0,0)
            msgLayout.addWidget(inpWidth,0,1)
            msgLayout.addWidget(labelThreshold,1,0)
            msgLayout.addWidget(inpThreshold,1,1)
            msgLayout.addWidget(buttonActive,2,0)
            msgLayout.addWidget(buttonAll,2,1)
            msgLayout.addLayout(buttonLayout,3,0,1,2)
            msg.setLayout(msgLayout)
            msg.inputWidth = inpWidth
            msg.inputThreshold = inpThreshold
            msg.applyToAll =  buttonAll
            self._widget = msg
            self._widget.buttonActive = buttonActive
            self._widget.buttonAll = buttonAll
        if self._widget.exec():
            self.threshold = float(self._widget.inputThreshold.value())
            self.width = int(self._widget.inputWidth.value())
            if not (self.width%2):
                self.width += 1
            if self._widget.buttonAll.isChecked():
                _logger.debug('AllChecked')
                self.removeSpikesAll()
            elif self._widget.buttonActive.isChecked():
                _logger.debug('ActiveChecked')
                self.removeSpikesActive()

    def removeSpikesAll(self):
        self.medianThresholdFilter(False, self.threshold, self.width)

    def removeSpikesActive(self):
        self.medianThresholdFilter(True, self.threshold, self.width)

    def medianThresholdFilter(self, activeOnly, threshold, length):
        if activeOnly:
            active = self.getActiveCurve()
            if not active:
                return
            else:
                x, y, legend, info = active[:4]
                self.removeCurve(legend)
                spectra = [active]
        else:
            spectra = self.getAllCurves()
        for (idx, spec) in enumerate(spectra):
            x, y, legend, info = spec[:4]
            xlabel=info.get("xlabel", None)
            ylabel=info.get("ylabel", None)
            filtered = medfilt1d(y, length)
            diff = filtered-y
            mean = diff.mean()
            sigma = (y-mean)**2
            sigma = numpy.sqrt(sigma.sum()/float(len(sigma)))
            ynew = numpy.where(abs(diff) > threshold * sigma, filtered, y)
            legend = info.get('selectionlegend',legend) + ' SR'
            if (idx==0) and (len(spectra)!=1):
                self.addCurve(x, ynew, legend, info,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              replace=True, replot=False)
            elif idx == (len(spectra)- 1):
                self.addCurve(x, ynew, legend, info,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              replace=False, replot=True)
            else:
                self.addCurve(x, ynew, legend, info,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              replace=False, replot=False)
        #self._plotWindow.replot()


MENU_TEXT = "Remove glitches from curves"
def getPlugin1DInstance(plotWindow,  **kw):
    ob = MedianFilterScanDeglitchPlugin(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca5.PyMcaGui import PlotWindow
    app = qt.QApplication([])

    sw = PlotWindow.PlotWindow()

    x = numpy.linspace(0, 1999, 2000)
    y0 = x/100. + 100.*numpy.exp(-(x-500)**2/1000.) + 50.*numpy.exp(-(x-1200)**2/5000.) + 100.*numpy.exp(-(x-1700)**2/500.) + 10 * numpy.random.random(2000)
    y1 = x/100. + 100.*numpy.exp(-(x-600)**2/1000.) + 50.*numpy.exp(-(x-1000)**2/5000.) + 100.*numpy.exp(-(x-1500)**2/500.) + 10 * numpy.random.random(2000)

    for idx in range(20):
        y0[idx*100] = 500. * numpy.random.random(1)
        y1[idx*100] = 500. * numpy.random.random(1)

    sw.addCurve(x, y0, legend="Curve0")
    sw.addCurve(x, y1, legend="Curve1")
    sw.setActiveCurve("Curve0")

    plugin = getPlugin1DInstance(sw)
    plugin.configureFilter()

    sw.show()
    app.exec()
