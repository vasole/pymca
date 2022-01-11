#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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

from PyMca5.PyMcaCore.Plugin2DBase import Plugin2DBase
from PyMca5.PyMcaGui import PyMcaQt as qt
from silx.gui.plot import Plot2D
from silx.math.medianfilter import medfilt2d


class Medfilt2DPlugin(Plugin2DBase):
    def __init__(self, plotWindow):
        Plugin2DBase.__init__(self, plotWindow)
        self.methods = {
            "Median filter": [self._medfilt2D,
                              "Open a plot showing a filtered image",
                              None]
        }
        self.widget = None

    def getMethods(self, plottype=None):
        """

        :return:  A list with the NAMES  associated to the callable methods
         that are applicable to the specified type plot. The list can be empty.
        :rtype: list[string]
        """
        return list(self.methods.keys())

    def _getMethod(self, name):
        method = self.methods.get(name)
        if method is None:
            raise NameError("method %s not found" % name)
        return method

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.

        :param name: The method for which a tooltip is asked
        :rtype: string
        """
        return self._getMethod(name)[1]

    def getMethodPixmap(self, name):
        """
        :param name: The method for which a pixmap is asked
        :rtype: QPixmap or None
        """
        return self._getMethod(name)[2]

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        return self._getMethod(name)[0]()

    def _medfilt2D(self):
        if self.widget is None:
            self.widget = Plot2DMedFilt()
            self.widget.show()

        active_image = self._plotWindow.getActiveImage()
        if active_image is None:
            return
        data = active_image.getData()
        if data.ndim > 2:
            raise NotImplementedError("Median filter not implemented for RGB images")
        self.widget.setColormap(active_image.getColormap())
        self.widget.setLegend("medfilt2d(%s)" % active_image.getLegend())
        self.widget.setRawData(data)

    def activeImageChanged(self, prev, new):
        if self.widget is None or not self.widget.isVisible() or new is None:
            return
        self._medfilt2D()


class Plot2DMedFilt(Plot2D):
    # TODO: we could allow setting different X- and Y-filter width.
    def __init__(self, parent=None):
        Plot2D.__init__(self, parent=parent)
        self.toolBar().addSeparator()
        label = qt.QLabel(self)
        label.setText("Median filter width:")

        self.spinbox = qt.QSpinBox(self)
        self.spinbox.setMinimum(1)
        self.spinbox.setValue(1)
        self.spinbox.setSingleStep(2)
        self.spinbox.valueChanged[int].connect(self._medfiltWidthChanged)
        self.spinbox.setEnabled(False)

        self.toolBar().addWidget(label)
        self.toolBar().addWidget(self.spinbox)

        self._data = None
        self._legend = None
        self._colormap = None

    def setLegend(self, legend):
        """

        :param str legend:
        :return:
        """
        self._legend = legend

    def setRawData(self, data, legend=None):
        """Set raw image data to be filtered.

        :param numpy.ndarray data:
        :param str legend:
        :return:
        """
        if data is None:
            self.spinbox.setEnabled(False)
            self._data = None
            return
        if data.ndim != 2:
            raise TypeError("Data must be a 2D array")
        self._data = data
        self.spinbox.setMaximum(max(data.shape))
        self.spinbox.setEnabled(True)
        self._applyFilter()

    def setColormap(self, colormap):
        self._colormap = colormap

    @property
    def medfilt_width(self):
        return self.spinbox.value()

    def _medfiltWidthChanged(self, width):
        self._applyFilter()

    def _applyFilter(self):
        # medfilt2D requires the data to be C-contiguous with silx <= 0.9
        self.addImage(medfilt2d(numpy.ascontiguousarray(self._data),
                                kernel_size=self.medfilt_width),
                      colormap=self._colormap,
                      legend=self._legend)


MENU_TEXT = "2D median filter"


def getPlugin2DInstance(plotWindow, **kw):
    """
    """
    ob = Medfilt2DPlugin(plotWindow)
    return ob


if __name__ == "__main__":
    # python -m PyMca5.PyMcaPlugins.Medfilt2DPlugin
    from PyMca5.PyMcaGui.PluginsToolButton import PluginsToolButton
    from PyMca5 import PyMcaPlugins
    import os
    from silx.test.utils import add_relative_noise
    from silx.gui.plot import PlotWidget

    # build a plot widget with a plugin toolbar button
    app = qt.QApplication([])
    master_plot = PlotWidget()
    toolb = qt.QToolBar(master_plot)
    plugins_tb_2d = PluginsToolButton(plot=master_plot,
                                      parent=toolb,
                                      method="getPlugin2DInstance")
    plugins_tb_2d.getPlugins(
                    method="getPlugin2DInstance",
                    directoryList=[os.path.dirname(PyMcaPlugins.__file__)])
    toolb.addWidget(plugins_tb_2d)
    master_plot.addToolBar(toolb)
    master_plot.show()

    # add a noisy image
    a, b = numpy.meshgrid(numpy.linspace(-10, 10, 500),
                          numpy.linspace(-10, 5, 400),
                          indexing="ij")
    myimg = numpy.asarray(numpy.sin(a * b) / (a * b),
                          dtype='float32')
    myimg = add_relative_noise(myimg,
                               max_noise=10.)    # %
    master_plot.addImage(myimg)

    app.exec()
