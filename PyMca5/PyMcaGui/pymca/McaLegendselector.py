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
__author__ = "P. Knobel - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import copy
import numpy
from silx.gui.plot import LegendSelector


class McaLegendsDockWidget(LegendSelector.LegendsDockWidget):
    """Subclassing of the silx LegendsDockWidget to handle
    curve renaming for McaWindow.
    """
    def renameCurve(self, oldLegend, newLegend):
        """Change the name of a curve using remove and addCurve.
        The name must also be changed in dataObjetsDict and calDict.

        :param str oldLegend: The legend of the curve to be changed
        :param str newLegend: The new legend of the curve
        """
        is_active = self.plot.getActiveCurve(just_legend=True) == oldLegend
        xChannels, yOrig, infoOrig = self.plot.getDataAndInfoFromLegend(oldLegend)
        curve = self.plot.getCurve(oldLegend)
        x = curve.getXData()
        y = curve.getYData()
        info = curve.getInfo()
        calib = info.get('McaCalib', [0.0, 1.0, 0.0])
        calibrationOrder = info.get('McaCalibOrder', 2)
        if calibrationOrder == 'TOF':
            xFromChannels = calib[2] + calib[0] / pow(xChannels-calib[1], 2)
        else:
            xFromChannels = calib[0] + \
                            calib[1] * xChannels + calib[2] * xChannels * xChannels
        if numpy.allclose(xFromChannels, x):
            x = xChannels
        newInfo = copy.deepcopy(info)
        newInfo['legend'] = newLegend
        newInfo['SourceName'] = newLegend
        newInfo['Key'] = ""
        newInfo['selectiontype'] = "1D"
        self.plot.removeCurve(oldLegend)
        self.plot.addCurve(x,
                           y,
                           legend=newLegend,
                           info=newInfo,
                           color=curve.getColor(),
                           symbol=curve.getSymbol(),
                           linewidth=curve.getLineWidth(),
                           linestyle=curve.getLineStyle(),
                           xlabel=curve.getXLabel(),
                           ylabel=curve.getYLabel(),
                           xerror=curve.getXErrorData(copy=False),
                           yerror=curve.getYErrorData(copy=False),
                           z=curve.getZValue(),
                           selectable=curve.isSelectable(),
                           fill=curve.isFill(),
                           resetzoom=False)
        if is_active:
            self.plot.setActiveCurve(newLegend)

        # make sure the dicts are also renamed
        self._renameInDataObjectsDict(oldLegend, newLegend)
        self._renameInCalDict(oldLegend, newLegend)

    def _renameInDataObjectsDict(self, oldLegend, newLegend):
        # This seems to be useless but I don't know why.
        # dataObjectDict is already properly renamed.
        if oldLegend in self.plot.dataObjectsDict:
            self.plot.dataObjectsDict[newLegend] = copy.deepcopy(
                     self.plot.dataObjectsDict[oldLegend])
            self.plot.dataObjectsDict[newLegend].info['legend'] = newLegend
            del self.plot.dataObjectsDict[oldLegend]

    def _renameInCalDict(self, oldLegend, newLegend):
        if oldLegend in self.plot.caldict:
            self.plot.caldict[newLegend] = copy.deepcopy(self.plot.caldict[oldLegend])
            del self.plot.caldict[oldLegend]
