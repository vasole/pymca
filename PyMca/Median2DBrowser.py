#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import numpy
try:
    from PyMca import StackBrowser
    from PyMca.PyMcaSciPy.signal import median
except ImportError:
    print("Median2DBrowser importing directly!")
    import StackBrowser
    from PyMcaSciPy.signal import median

medfilt2d = median.medfilt2d
qt = StackBrowser.qt
DEBUG = 0

class MedianParameters(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.label = qt.QLabel(self)
        self.label.setText("Median filter width: ")
        self.widthSpin = qt.QSpinBox(self)
        self.widthSpin.setMinimum(1)
        self.widthSpin.setMaximum(99)
        self.widthSpin.setValue(1)
        self.widthSpin.setSingleStep(2)
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.widthSpin)

class Median2DBrowser(StackBrowser.StackBrowser):
    def __init__(self, *var, **kw):
        StackBrowser.StackBrowser.__init__(self, *var, **kw)
        self.setWindowTitle("Image Browser with Median Filter")
        self._medianParameters = {'use':True,
                                  'row_width':5,
                                  'column_width':5}
        self._medianParametersWidget = MedianParameters(self)
        self._medianParametersWidget.widthSpin.setValue(5)        
        self.layout().addWidget(self._medianParametersWidget)
        self.connect(self._medianParametersWidget.widthSpin,
                     qt.SIGNAL('valueChanged(int)'),
                     self.setKernelWidth)

    def setKernelWidth(self, value):
        kernelSize = numpy.asarray(value)
        if len(kernelSize.shape) == 0:
            kernelSize = [kernelSize.item()] * 2
        self._medianParameters['row_width'] = kernelSize[0]
        self._medianParameters['column_width'] = kernelSize[1]
        self._medianParametersWidget.widthSpin.setValue(int(kernelSize[0]))
        current = self.slider.value()
        self.showImage(current, moveslider=False)

    def _buildTitle(self, legend, index):
        a = self._medianParameters['row_width']
        b = self._medianParameters['column_width']
        title = StackBrowser.StackBrowser._buildTitle(self, legend, index)
        if max(a, b) > 1:
            return "Median Filter (%d,%d) of %s" % (a, b, title)
        else:
            return title

    def setImageData(self, data, **kw):
        if self._medianParameters['use']:
            if max(self._medianParameters['row_width'],
                   self._medianParameters['column_width']) > 1:
                data = medfilt2d(data,[self._medianParameters['row_width'],
                                   self._medianParameters['column_width']])
        StackBrowser.StackBrowser.setImageData(self, data, **kw)

if __name__ == "__main__":
    #create a dummy stack
    nrows = 100
    ncols = 200
    nchannels = 1024
    a = numpy.ones((nrows, ncols), numpy.float)
    stackData = numpy.zeros((nrows, ncols, nchannels), numpy.float)
    for i in range(nchannels):
        if i % 10:
            stackData[:, :, i] = a * i
        else:
            stackData[:, :, i] = 10 * a * i

    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                        app,qt.SLOT("quit()"))
    w = Median2DBrowser()
    w.setStackDataObject(stackData, index=0)
    w.show()
    app.exec_()
