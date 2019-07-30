# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Set of QToolBar to attach to a :class:`PlotWindow` instance.

Available toolbars:

- ProvileToolBar: Profiling tools
- LimitsToolBar: Text field to display and change the plot limits.

"""


# import ######################################################################

import logging
import sys

import numpy

try:
    from .. import PyMcaQt as qt
except ImportError:
    from PyMca5.PyMcaGui import PyMcaQt as qt

from .PyMca_Icons import IconDict
from .ProfileScanWidget import ProfileScanWidget

from . import _ImageProfile


# ProfileToolBar ##############################################################

class ProfileToolBar(qt.QToolBar):
    """QToolBar providing profile tools operating on a :class:`PlotWindow`.

    Attributes:

    - plotWindow: Associated :class:`PlotWindow`.
    - profileWindow: Associated :class:`ProfileScanWidget`.
    - actionGroup: :class:`QActionGroup` of available actions.
    """
    # TODO when available, listen to active image change to refresh profile

    _POLYGON_LEGEND = '__ProfileToolBar_ROI_Polygon'

    def __init__(self, plotWindow, profileWindow=None,
                 title='Profile Selection', parent=None):
        """

        :param plotWindow: :class:`PlotWindow` instance on which to operate.
        :param profileWindow: :class:`ProfileScanWidget` instance where to
                              display the profile curve or None to create one.
        :param str title: See :class:`QToolBar`.
        :param parent: See :class:`QToolBar`.
        """
        super(ProfileToolBar, self).__init__(title, parent)
        assert plotWindow is not None
        self.plotWindow = plotWindow
        self.plotWindow.sigPlotSignal.connect(self._plotWindowSlot)

        self._overlayColor = None

        self._roiInfo = None
        if profileWindow is None:
            self.profileWindow = ProfileScanWidget(actions=False)
        else:
            self.profileWindow = profileWindow

        # Actions
        self.browseAction = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["normal"])),
            'Browsing Mode', None)
        self.browseAction.setToolTip(
            'Enables zooming interaction mode')
        self.browseAction.setCheckable(True)

        self.hLineAction = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["horizontal"])),
            'Horizontal Profile Mode', None)
        self.hLineAction.setToolTip(
            'Enables horizontal profile selection mode')
        self.hLineAction.setCheckable(True)

        self.vLineAction = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["vertical"])),
            'Vertical Profile Mode', None)
        self.vLineAction.setToolTip(
            'Enables vertical profile selection mode')
        self.vLineAction.setCheckable(True)

        self.lineAction = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["diagonal"])),
            'Fee Line Profile Mode', None)
        self.lineAction.setToolTip(
            'Enables line profile selection mode')
        self.lineAction.setCheckable(True)

        self.clearAction = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["image"])),
            'Clear Profile', None)
        self.clearAction.setToolTip(
            'Clear the profile Region of interest')
        self.clearAction.setCheckable(False)
        self.clearAction.triggered.connect(self.clearProfile)

        # ActionGroup
        self.actionGroup = qt.QActionGroup(self)
        self.actionGroup.addAction(self.browseAction)
        self.actionGroup.addAction(self.hLineAction)
        self.actionGroup.addAction(self.vLineAction)
        self.actionGroup.addAction(self.lineAction)
        self.actionGroup.triggered.connect(self._actionGroupTriggeredSlot)

        self.browseAction.setChecked(True)

        # Add actions to ToolBar
        self.addAction(self.browseAction)
        self.addAction(self.hLineAction)
        self.addAction(self.vLineAction)
        self.addAction(self.lineAction)
        self.addAction(self.clearAction)

        # Add width spin box to toolbar
        self.addWidget(qt.QLabel('W:'))
        self.lineWidthSpinBox = qt.QSpinBox(self)
        self.lineWidthSpinBox.setRange(0, 1000)
        self.lineWidthSpinBox.setValue(1)
        self.lineWidthSpinBox.valueChanged[int].connect(
            self._lineWidthSpinBoxValueChangedSlot)
        self.addWidget(self.lineWidthSpinBox)

    def _lineWidthSpinBoxValueChangedSlot(self, value):
        self.updateProfile()

    def _actionGroupTriggeredSlot(self, action):
        if action == self.browseAction:
            self.plotWindow.setZoomModeEnabled(True, color=self.overlayColor)
        elif action == self.hLineAction:
            self.plotWindow.setDrawModeEnabled(True, shape='hline',
                                               color=self.overlayColor)
        elif action == self.vLineAction:
            self.plotWindow.setDrawModeEnabled(True, shape='vline',
                                               color=self.overlayColor)
        elif action == self.lineAction:
            self.plotWindow.setDrawModeEnabled(True, shape='line',
                                               color=self.overlayColor)
        else:
            logging.error(
                'ProfileToolBar._actionGroupTriggered got unknown action')

    def _plotWindowSlot(self, event):
        if event['event'] not in ('drawingProgress', 'drawingFinished'):
            return

        roiStart, roiEnd = event['points'][0], event['points'][1]

        checkedAction = self.actionGroup.checkedAction()
        if checkedAction == self.hLineAction:
            roiStart = - sys.maxsize, roiStart[1]
            roiEnd = sys.maxsize, roiEnd[1]
            lineProjectionMode = 'X'
        elif checkedAction == self.vLineAction:
            roiStart = roiStart[0], - sys.maxsize
            roiEnd = roiEnd[0], sys.maxsize
            lineProjectionMode = 'Y'
        elif checkedAction == self.lineAction:
            lineProjectionMode = 'D'
        else:
            return

        self._roiInfo = roiStart, roiEnd, lineProjectionMode
        self.updateProfile()

    @property
    def overlayColor(self):
        """The color to use for the ROI.
        """
        return self._overlayColor

    @overlayColor.setter
    def overlayColor(self, color):
        self._overlayColor = color
        self.updateProfile()

    def updateProfile(self):
        """Update the displayed profile and profile ROI.
        """
        self.plotWindow.removeItem(self._POLYGON_LEGEND, replot=True)

        if self._roiInfo is None:
            return

        imageData = self.plotWindow.getActiveImage()
        if imageData is None:
            return
        data, legend, info, pixmap = imageData

        origin = info['plot_xScale'][0], info['plot_yScale'][0]
        scale = info['plot_xScale'][1], info['plot_yScale'][1]

        roiWidth = self.lineWidthSpinBox.value()
        roiStart, roiEnd, lineProjectionMode = self._roiInfo

        profile = _ImageProfile.getProfileCurve(data,
                                                roiStart,
                                                roiEnd,
                                                roiWidth,
                                                origin,
                                                scale,
                                                lineProjectionMode)
        if profile is None:
            return

        # Update ROI polygon
        self.plotWindow.addItem(profile['roiPolygonX'],
                                profile['roiPolygonY'],
                                legend=self._POLYGON_LEGEND,
                                color=self.overlayColor,
                                shape='polygon', fill=True,
                                replace=False, replot=True)

        # Title
        if lineProjectionMode == 'X':
            yMin = profile['roiPolygonY'].min()
            yMax = profile['roiPolygonY'].max() - 1
            if roiWidth <= 1:
                profileName = 'Y = %g' % yMin
            else:
                profileName = 'Y = [%g, %g]' % (yMin, yMax)
            xLabel = 'Columns'

        elif lineProjectionMode == 'Y':
            xMin = profile['roiPolygonX'].min()
            xMax = profile['roiPolygonX'].max() - 1
            if roiWidth <= 1:
                profileName = 'X = %g' % xMin
            else:
                profileName = 'X = [%g, %g]' % (xMin, xMax)
            xLabel = 'Rows'

        else:
            x0, y0 = profile['startPoint']
            x1, y1 = profile['endPoint']
            if roiWidth < 1 or x1 == x0 or y1 == y0:
                profileName = 'From (%g, %g) to (%g, %g)' % (
                    x0, y0, x1, y1)
            else:
                m = (y1 - y0) / float((x1 - x0))
                b = y0 - m * x0
                profileName = 'y = %g * x %+g ; width=%d' % (
                    m, b, roiWidth)
            xLabel = 'Distance'

        # Update profile curve
        coords, values = profile['profileCoords'], profile['profileValues']
        index = numpy.isfinite(values)
        coords, values = coords[index], values[index]
        self.profileWindow.addCurve(coords, values,
                                    legend=profileName, xlabel=xLabel,
                                    replace=True, replot=True)
        self.profileWindow.show()

    def clearProfile(self):
        self._roiInfo = None
        self.updateProfile()


# LimitsToolBar ##############################################################

class LimitsToolBar(qt.QToolBar):
    """QToolBar displaying and controlling the limits of a :class:`PlotWindow`.
    """

    class _FloatEdit(qt.QLineEdit):
        """Field to edit a float value."""
        def __init__(self, value=None, *args, **kwargs):
            qt.QLineEdit.__init__(self, *args, **kwargs)
            self.setValidator(qt.CLocaleQDoubleValidator(None))
            self.setFixedWidth(100)
            self.setAlignment(qt.Qt.AlignLeft)
            if value is not None:
                self.setValue(value)

        def value(self):
            return float(self.text())

        def setValue(self, value):
            self.setText('%g' % value)

    def __init__(self, plotWindow, title='Limits', parent=None):
        """

        :param plotWindow: :class:`PlotWindow` instance on which to operate.
        :param str title: See :class:`QToolBar`.
        :param parent: See :class:`QToolBar`.
        """
        super(LimitsToolBar, self).__init__(title, parent)
        assert plotWindow is not None
        self._plotWindow = plotWindow
        self._plotWindow.sigPlotSignal.connect(self._plotWindowSlot)

        self._initWidgets()

    @property
    def plotWindow(self):
        """The :class:`PlotWindow` the toolbar is attached to."""
        return self._plotWindow

    def _initWidgets(self):
        """Create and init Toolbar widgets."""
        xMin, xMax = self.plotWindow.getGraphXLimits()
        yMin, yMax = self.plotWindow.getGraphYLimits()

        self.addWidget(qt.QLabel('Limits: '))
        self.addWidget(qt.QLabel(' X: '))
        self._xMinFloatEdit = self._FloatEdit(xMin)
        self._xMinFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMinFloatEdit)

        self._xMaxFloatEdit = self._FloatEdit(xMax)
        self._xMaxFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMaxFloatEdit)

        self.addWidget(qt.QLabel(' Y: '))
        self._yMinFloatEdit = self._FloatEdit(yMin)
        self._yMinFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMinFloatEdit)

        self._yMaxFloatEdit = self._FloatEdit(yMax)
        self._yMaxFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMaxFloatEdit)

    def _plotWindowSlot(self, event):
        """Listen to :class:`PlotWindow` events."""
        if event['event'] not in ('limitsChanged',):
            return

        xMin, xMax = self.plotWindow.getGraphXLimits()
        yMin, yMax = self.plotWindow.getGraphYLimits()

        self._xMinFloatEdit.setValue(xMin)
        self._xMaxFloatEdit.setValue(xMax)
        self._yMinFloatEdit.setValue(yMin)
        self._yMaxFloatEdit.setValue(yMax)

    def _xFloatEditChanged(self):
        """Handle X limits changed from the GUI."""
        xMin, xMax = self._xMinFloatEdit.value(), self._xMaxFloatEdit.value()
        if xMax < xMin:
            xMin, xMax = xMax, xMin

        self.plotWindow.setGraphXLimits(xMin, xMax)

    def _yFloatEditChanged(self):
        """Handle Y limits changed from the GUI."""
        yMin, yMax = self._yMinFloatEdit.value(), self._yMaxFloatEdit.value()
        if yMax < yMin:
            yMin, yMax = yMax, yMin

        self.plotWindow.setGraphYLimits(yMin, yMax)
