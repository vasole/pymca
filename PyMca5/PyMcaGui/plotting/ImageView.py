# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
ImageView: QWidget displaying a single 2D image with histograms on its sides.
"""


# import ######################################################################

import numpy as np
import warnings
import weakref

from .. import PyMcaQt as qt
from .PyMca_Icons import IconDict
from .ColormapDialog import ColormapDialog


# utils #######################################################################

def _getBackendClass(name=None):
    """Return the class of backend corresponding to the name

    :param str name: The name of the backend or None for default backend.
    :return: The class of the backend
    :rtype: Subclass of PlotBackend
    :raises: RuntimeError if the name does not correspond to a backend.
    """
    if name is None:
        name = 'mpl'

    name = name.lower()
    if name in ('mpl', 'matplotlib'):
        from PyMca5.PyMcaGraph.backends.MatplotlibBackend import \
            MatplotlibBackend
        backend = MatplotlibBackend
    elif name in ('osmesa', 'mesa'):
        from PyMca5.PyMcaGraph.backends.OSMesaGLBackend import OSMesaGLBackend
        backend = OSMesaGLBackend
    elif name in ('gl', 'opengl'):
        from PyMca5.PyMcaGraph.backends.OpenGLBackend import OpenGLBackend
        backend = OpenGLBackend
    else:
        raise RuntimeError("Cannot find backend with name: %s" % name)

    return backend

_COLORMAP_CURSOR_COLORS = {
    'gray': 'pink',
    'reversed gray': 'pink',
    'temperature': 'black',
    'red': 'gray',
    'green': 'gray',
    'blue': 'gray'}

def _cursorColorForColormap(colormapName):
    """Get a color suitable for overlay over a colormap.

    :param str colormapName: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    return _COLORMAP_CURSOR_COLORS.get(colormapName, 'black')


# ColormapDialogHelper ########################################################

class _ColormapDialogHelper(qt.QObject):
    """Helper class wrapping ColormapDialog for use with ImageView."""

    colormapChanged = qt.pyqtSignal(dict)
    """Signal a change in colormap.

    Provides a description of the colormap as a dict.
    See PlotBackend.getDefaultColormap for details.
    """

    DIALOG_HISTO_NB_BINS = 50
    """Number of bins of the histogram displayed in the colormap dialog."""

    # Workaround: PlotBackend and ColormapDialog cmap names mismatch
    # Names as used in PlotBackend in the order of colormapDialog
    _COLORMAP_NAMES = ('gray', 'reversed gray', 'temperature',
                      'red', 'green', 'blue', 'temperature')

    def __init__(self, imageView):
        super(_ColormapDialogHelper, self).__init__()
        self._imageViewRef = weakref.ref(imageView)
        self._cmapDialog = None

    def _dialog(self):
        if self._cmapDialog is None:  # Reuse same widget
            imageView = self._imageViewRef()
            self._cmapDialog = ColormapDialog(imageView)

            # Colormap dialog is modal as we don't care about data/colormap
            # update while the dialog is opened to simplify implementation.
            self._cmapDialog.setModal(True)
            self._cmapDialog.finished.connect(self._close)

            # Disable gamma mapping
            gammaBtn = self._cmapDialog.buttonGroup.button(2)
            assert gammaBtn.text() == 'Gamma'
            gammaBtn.setCheckable(False)
            gammaBtn.setVisible(False)

        return self._cmapDialog

    def _close(self, result=0):
        self._cmapDialog.sigColormapChanged.disconnect(self._colormapChanged)

    def showDialog(self):
        """Slot for Change colormap action."""
        dialog = self._dialog()

        imageView = self._imageViewRef()
        assert imageView is not None

        # Data dependent information
        histogram = imageView.imageHistogram(self.DIALOG_HISTO_NB_BINS)
        if histogram is not None:
            counts, binEdges = histogram
            # Bin edges to bins center (all bins have the same width)
            bins = binEdges[:-1] + 0.5 * (binEdges[1] - binEdges[0])
            dialog.plotHistogram((bins, counts))
            dialog.setDataMinMax(binEdges[0], binEdges[-1])

        # Set dialog colormap
        colormap = imageView.colormap()

        index = self._COLORMAP_NAMES.index(colormap['name'])
        dialog.setColormap(index)

        dialog.setMinValue(colormap['vmin'])
        dialog.setMaxValue(colormap['vmax'])
        dialog.setAutoscale(colormap['autoscale'])
        dialog.setColormapType(
            1 if colormap['normalization'].startswith('log') else 0)

        # Only connect colormapChanged now, to avoid being called while
        # setting-up the colormap dialog.
        self._cmapDialog.sigColormapChanged.connect(self._colormapChanged)

        dialog.show()

    def _colormapChanged(self, info):
        cmapIndex, autoscale, vMin, vMax, dataMin, dataMax, normIndex = info

        if normIndex not in (0, 1):
            warnings.warn('Unsupported colormap, using linear', UserWarning)
            normIndex = 0

        colormap = {
            'name': self._COLORMAP_NAMES[cmapIndex],
            'autoscale': autoscale,
            'vmin': vMin,
            'vmax': vMax,
            'normalization': 'log' if normIndex == 1 else 'linear',
            'colors': 256
        }
        self.colormapChanged.emit(colormap)


# RadarView ###################################################################

class RadarView(qt.QGraphicsView):
    """Present a synthetic view of a 2D image and the current visible area.

    Used coordinates are as in QGraphicsView: x goes from left to right and 
    y goes from top to bottom.
    This widget preserves the aspect ratio of the data.
    """

    visibleRectDragged = qt.pyqtSignal(float, float, float, float)
    """Signals that the visible rectangle has been dragged.

    It provides: left, top, width, height in data coordinates.
    """

    _DATA_PEN = qt.QPen(qt.QColor('white'))
    _DATA_BRUSH = qt.QBrush(qt.QColor('light gray'))
    _VISIBLE_PEN = qt.QPen(qt.QColor('red'))
    _VISIBLE_BRUSH = qt.QBrush(qt.QColor(0, 0, 0, 0))
    _TOOLTIP = 'Radar View:\nVisible area (in red)\nof the image (in gray).'

    _PIXMAP_SIZE = 256

    class _DraggableRectItem(qt.QGraphicsRectItem):
        """RectItem which signals its change through visibleRectDragged."""
        def __init__(self, *args, **kwargs):
            super(RadarView._DraggableRectItem, self).__init__(*args, **kwargs)
            self.setFlag(qt.QGraphicsItem.ItemIsMovable)
            self.setFlag(qt.QGraphicsItem.ItemSendsGeometryChanges)
            self._ignoreChange = False
            self._constraint = 0, 0, 0, 0

        def setConstraintRect(self, left, top, width, height):
            """Set the constraint rectangle for dragging.

            The coordinates are in the _DraggableRectItem coordinate system.

            This constraint only applies to modification through interaction
            (i.e., this constraint is not applied to change through API).

            If the _DraggableRectItem is smaller than the constraint rectangle,
            the _DraggableRectItem remains within the constraint rectangle.
            If the _DraggableRectItem is wider than the constraint rectangle,
            the constraint rectangle remains within the _DraggableRectItem.
            """
            self._constraint = left, left + width, top, top + height

        def setPos(self, *args, **kwargs):
            """Overridden to ignore changes from API in itemChange."""
            self._ignoreChange = True
            super(RadarView._DraggableRectItem, self).setPos(*args, **kwargs)
            self._ignoreChange = False

        def moveBy(self, *args, **kwargs):
            """Overridden to ignore changes from API in itemChange."""
            self._ignoreChange = True
            super(RadarView._DraggableRectItem, self).moveBy(*args, **kwargs)
            self._ignoreChange = False

        def itemChange(self, change, value):
            """Callback called before applying changes to the item."""
            if (change == qt.QGraphicsItem.ItemPositionChange and
                    not self._ignoreChange):
                # Makes sure that the visible area is in the data
                # or that data is in the visible area if area is too wide
                x, y = value.x(), value.y()
                xMin, xMax, yMin, yMax = self._constraint

                if self.rect().width() <= (xMax - xMin):
                    if x < xMin:
                        value.setX(xMin)
                    elif x > xMax - self.rect().width():
                        value.setX(xMax - self.rect().width())
                else:
                    if x > xMin:
                        value.setX(xMin)
                    elif x < xMax - self.rect().width():
                        value.setX(xMax - self.rect().width())

                if self.rect().height() <= (yMax - yMin):
                    if y < yMin:
                        value.setY(yMin)
                    elif y > yMax - self.rect().height():
                        value.setY(yMax - self.rect().height())
                else:
                    if y > yMin:
                        value.setY(yMin)
                    elif y < yMax - self.rect().height():
                        value.setY(yMax - self.rect().height())

                if self.pos() != value:
                    # Notify change through signal
                    views = self.scene().views()
                    assert len(views) == 1
                    views[0].visibleRectDragged.emit(
                        value.x() + self.rect().left(),
                        value.y() + self.rect().top(),
                        self.rect().width(),
                        self.rect().height())

                return value

            return super(RadarView._DraggableRectItem, self).itemChange(
                change, value)

    def __init__(self, parent=None):
        self._scene = qt.QGraphicsScene()
        self._dataRect = self._scene.addRect(0, 0, 1, 1,
                                             self._DATA_PEN,
                                             self._DATA_BRUSH)
        self._visibleRect = self._DraggableRectItem(0, 0, 1, 1)
        self._visibleRect.setPen(self._VISIBLE_PEN)
        self._visibleRect.setBrush(self._VISIBLE_BRUSH)
        self._scene.addItem(self._visibleRect)

        super(RadarView, self).__init__(self._scene, parent)
        self.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('border: 0px')
        self.setToolTip(self._TOOLTIP)

    def sizeHint(self):
        """Overridden to avoid sizeHint to depend on content size."""
        return self.minimumSizeHint()

    def wheelEvent(self, event):
        """Overridden to disable vertical scrolling with wheel."""
        event.ignore()

    def resizeEvent(self, event):
        """Overridden to fit current content to new size."""
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)
        super(RadarView, self).resizeEvent(event)

    def setDataRect(self, left, top, width, height):
        """Set the bounds of the data rectangular area.
        
        This sets the coordinate system.
        """
        self._dataRect.setRect(left, top, width, height)
        self._visibleRect.setConstraintRect(left, top, width, height)
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)

    def setVisibleRect(self, left, top, width, height):
        """Set the visible rectangular area.

        The coordinates are relative to the data rect.
        """
        self._visibleRect.setRect(0, 0, width, height)
        self._visibleRect.setPos(left, top)
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)


# ImageView ###################################################################

class ImageView(qt.QWidget):
    """Display a single image with horizontal and vertical histograms.

    QAction available as attributes of the class:

    - actionResetZoom: Displays the full image in the plot area (State-less).
    - actionKeepDataAspectRatio: Controls image aspect ratio (Checkable).
    - actionChangeColormap: Opens colormap chooser (State-less).
    - actionInvertYAxis: Controls Y axis direction (Checkable).

    The commands associated to those QActions are also available through
    methods.
    """

    HISTOGRAMS_COLOR = 'blue'
    """Color to use for the side histograms."""

    HISTOGRAMS_HEIGHT = 200
    """Height in pixels of the side histograms."""

    IMAGE_MIN_SIZE = 200
    """Minimum size in pixels of the image area."""

    # Qt signals
    valueChanged = qt.pyqtSignal(float, float, float)
    """Signals that the data value under the cursor has changed.

    It provides: row, column, data value.

    When the cursor is over an histogram, either row or column is Nan
    and the provided data value is the histogram value
    (i.e., the sum along the corresponding row/column).
    Row and columns are either Nan or integer values.
    """

    def __init__(self, parent=None, windowFlags=qt.Qt.Widget, backend=None):
        self._data = None  # Store current image
        self._cache = None  # Store currently visible data information
        self._cacheHisto = None  # cache histogram for colormap dialog
        self._updatingLimits = False

        super(ImageView, self).__init__(parent, windowFlags)
        self._initWidgets(_getBackendClass(backend))

        # Sync PlotBackend and ImageView
        self.setColormap(self._imagePlot.getDefaultColormap())

        self._initActions()

    def _initWidgets(self, backendClass):
        """Set-up layout and plots."""
        # Monkey-patch for histogram size
        # alternative: create a layout that does not use widget size hints
        def sizeHint():
            return qt.QSize(self.HISTOGRAMS_HEIGHT, self.HISTOGRAMS_HEIGHT)

        self._histoHPlot = backendClass()
        self._histoHPlot.setZoomModeEnabled(True)
        self._histoHPlot.setCallback(self._histoHPlotCB)
        self._histoHPlot.getWidgetHandle().sizeHint = sizeHint
        self._histoHPlot.getWidgetHandle().minimumSizeHint = sizeHint

        self._imagePlot = backendClass()
        self._imagePlot.setZoomModeEnabled(True)  # Color is set in setColormap
        self._imagePlot.setCallback(self._imagePlotCB)

        self._histoVPlot = backendClass()
        self._histoVPlot.setZoomModeEnabled(True)
        self._histoVPlot.setCallback(self._histoVPlotCB)
        self._histoVPlot.getWidgetHandle().sizeHint = sizeHint
        self._histoVPlot.getWidgetHandle().minimumSizeHint = sizeHint

        self._radarView = RadarView()
        self._radarView.visibleRectDragged.connect(self._radarViewCB)

        layout = qt.QGridLayout()
        layout.addWidget(self._imagePlot.getWidgetHandle(), 0, 0)
        layout.addWidget(self._histoVPlot.getWidgetHandle(), 0, 1)
        layout.addWidget(self._histoHPlot.getWidgetHandle(), 1, 0)
        layout.addWidget(self._radarView, 1, 1)

        layout.setColumnMinimumWidth(0, self.IMAGE_MIN_SIZE)
        layout.setColumnStretch(0, 1)
        layout.setColumnMinimumWidth(1, self.HISTOGRAMS_HEIGHT)
        layout.setColumnStretch(1, 0)

        layout.setRowMinimumHeight(0, self.IMAGE_MIN_SIZE)
        layout.setRowStretch(0, 1)
        layout.setRowMinimumHeight(1, self.HISTOGRAMS_HEIGHT)
        layout.setRowStretch(1, 0)

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

    def _updateHistograms(self):
        """Update histograms content."""
        if self._data is not None:
            xMin, xMax = self._imagePlot.getGraphXLimits()
            yMin, yMax = self._imagePlot.getGraphYLimits()

            height, width = self._data.shape
            if (xMin <= width and xMax >= 0 and
                yMin <= height and yMax >= 0):
                # The image is at least partly in the plot area
                visibleXMin = 0 if xMin < 0 else int(xMin)
                visibleXMax = width if xMax >= width else int(xMax + 1)
                visibleYMin = 0 if yMin < 0 else int(yMin)
                visibleYMax = height if yMax >= height else int(yMax + 1)

                if (self._cache is None or
                        visibleXMin != self._cache['visibleXMin'] or
                        visibleXMax != self._cache['visibleXMax'] or
                        visibleYMin != self._cache['visibleYMin'] or
                        visibleYMax != self._cache['visibleYMax']):
                    # The visible area of data has changed, update histograms

                    # Rebuild histograms for visible area
                    visibleData = self._data[visibleYMin:visibleYMax,
                                             visibleXMin:visibleXMax]
                    histoHVisibleData = np.sum(visibleData, axis=0)
                    histoVVisibleData = np.sum(visibleData, axis=1)
                    
                    self._cache = {
                        'visibleXMin': visibleXMin,
                        'visibleXMax': visibleXMax,
                        'visibleYMin': visibleYMin,
                        'visibleYMax': visibleYMax,

                        'histoH': histoHVisibleData,
                        'histoHMin': np.min(histoHVisibleData),
                        'histoHMax': np.max(histoHVisibleData),

                        'histoV': histoVVisibleData,
                        'histoVMin': np.min(histoVVisibleData),
                        'histoVMax': np.max(histoVVisibleData)
                    }

                    # Convert to histogram curve and update plots
                    coords = np.arange(2 * histoHVisibleData.size)
                    xCoords = (coords + 1) // 2 + visibleXMin
                    xData = np.take(histoHVisibleData, coords // 2)
                    self._histoHPlot.addCurve(xCoords, xData,
                                              replace=False, replot=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)

                    coords = np.arange(2 * histoVVisibleData.size)
                    yCoords = (coords + 1) // 2 + visibleYMin
                    yData = np.take(histoVVisibleData, coords // 2)
                    self._histoVPlot.addCurve(yData, yCoords,
                                              replace=False, replot=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)
            else:
                self._cache = None

                self._histoHPlot.clearCurves()
                self._histoVPlot.clearCurves()

    def _updateRadarView(self, xMin, xMax, yMin, yMax):
        """Update radar view visible area.
        
        Takes care of y coordinate conversion.
        """
        if self.isYAxisInverted():
            top = yMin
        elif self._data is not None:
            top = self._data.shape[0] - yMax
        else:
            top = 0
        self._radarView.setVisibleRect(xMin, top, xMax -xMin, yMax - yMin)

    # Plots event listeners
    def _imagePlotCB(self, eventDict):
        """Callback for imageView plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._data is not None:
                height, width = self._data.shape
                x, y = int(eventDict['x']), int(eventDict['y'])
                if x >= 0 and x < width and y >= 0 and y < height:
                    self.valueChanged.emit(float(x), float(y),
                                           self._data[y][x])
        elif eventDict['event'] == 'limitsChanged':
            # Do not handle histograms limitsChanged while
            # updating their limits from here.
            self._updatingLimits = True

            # Refresh histograms
            self._updateHistograms()

            # could use eventDict['xdata'], eventDict['ydata'] instead
            xMin, xMax = self._imagePlot.getGraphXLimits()
            yMin, yMax = self._imagePlot.getGraphYLimits()

            # Set horizontal histo limits
            if self._cache is not None:
                vMin = self._cache['histoHMin']
                vMax = self._cache['histoHMax']
                vOffset = 0.1 * (vMax - vMin)
                if vOffset == 0.:
                    vOffset = 1.
                self._histoHPlot.setGraphYLimits(
                    vMin - vOffset,
                    vMax + vOffset)
            self._histoHPlot.setGraphXLimits(xMin, xMax)
            self._histoHPlot.replot()

            # Set vertical histo limits
            if self._cache is not None:
                vMin = self._cache['histoVMin']
                vMax = self._cache['histoVMax']
                vOffset = 0.1 * (vMax - vMin)
                if vOffset == 0.:
                    vOffset = 1.
                self._histoVPlot.setGraphXLimits(
                    vMin - vOffset,
                    vMax + vOffset)
            self._histoVPlot.setGraphYLimits(yMin, yMax)
            self._histoVPlot.replot()

            self._updateRadarView(xMin, xMax, yMin, yMax)

            self._updatingLimits = False

    def _histoHPlotCB(self, eventDict):
        """Callback for horizontal histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                minValue = self._cache['visibleXMin']
                data = self._cache['histoH']
                width = data.shape[0]
                x = int(eventDict['x'])
                if x >= minValue and x < minValue + width:
                    self.valueChanged.emit(float('nan'), float(x),
                                           data[x - minValue])
        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['xdata'] != self._imagePlot.getGraphXLimits()):
                xMin, xMax = eventDict['xdata']
                self._imagePlot.setGraphXLimits(xMin, xMax)
                self._imagePlot.replot()

    def _histoVPlotCB(self, eventDict):
        """Callback for vertical histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                minValue = self._cache['visibleYMin']
                data = self._cache['histoV']
                height = data.shape[0]
                y = int(eventDict['y'])
                if y >= minValue and y < minValue + height:
                    self.valueChanged.emit(float(y), float('nan'),
                                           data[y - minValue])
        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['ydata'] != self._imagePlot.getGraphYLimits()):
                yMin, yMax = eventDict['ydata']
                self._imagePlot.setGraphYLimits(yMin, yMax)
                self._imagePlot.replot()

    def _radarViewCB(self, left, top, width, height):
        """Slot for radar view visible rectangle changes."""
        if not self._updatingLimits:
            # Takes care of Y axis conversion
            if self.isYAxisInverted():
                yMin = top
                yMax = yMin + height
            else:
                yMax = self._data.shape[0] - top
                yMin = yMax - height
            self._imagePlot.setLimits(left, left + width, yMin, yMax)
            self._imagePlot.replot()

    # Actions
    def _setYAxisInverted(self, inverted):
        self._imagePlot.invertYAxis(inverted)
        self._histoVPlot.invertYAxis(inverted)
        xMin, xMax = self._imagePlot.getGraphXLimits()
        yMin, yMax = self._imagePlot.getGraphYLimits()
        self._updateRadarView(xMin, xMax, yMin, yMax)

        self._imagePlot.replot()
        self._histoVPlot.replot()
        self._radarView.update()

    def _setKeepDataAspectRatio(self, keepRatio):
        iconName = 'solidcircle' if keepRatio else 'solidellipse'
        self.actionKeepDataAspectRatio.setIcon(
            qt.QIcon(qt.QPixmap(IconDict[iconName])))

    def _initActions(self):
        # Reset zoom
        self.actionResetZoom = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict['zoomreset'])),
            'Reset Zoom (Ctrl-0)',
            self,
            triggered=self.resetZoom)
        self.actionResetZoom.setShortcut('Ctrl+0')

        # keep data aspect ratio
        self.actionKeepDataAspectRatio = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict['solidellipse'])),
            'Keep data aspect ratio',
            self,
            toggled=self._imagePlot.keepDataAspectRatio)
        # No need to ask for replot here
        # No need to sync histogram limits, this is automatic
        # Change icon
        self.actionKeepDataAspectRatio.toggled.connect(
            self._setKeepDataAspectRatio)
        self.actionKeepDataAspectRatio.setCheckable(True)
        self.actionKeepDataAspectRatio.setChecked(
            self._imagePlot.isKeepDataAspectRatio())

        # Change colormap
        cmapDialog = _ColormapDialogHelper(self)
        cmapDialog.colormapChanged.connect(self.setColormap)

        self.actionChangeColormap = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict['colormap'])),
            'Change colormap',
            self,
            triggered=cmapDialog.showDialog)
        self.actionChangeColormap._cmapDialog = cmapDialog  # Store a ref

        # Invert Y axis
        self.actionInvertYAxis = qt.QAction(
            qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"])),
            'Flip Horizontal',
            self,
            toggled=self._setYAxisInverted)
        self.actionInvertYAxis.setCheckable(True)
        self.actionInvertYAxis.setChecked(self._imagePlot.isYAxisInverted())

    # API
    def resetZoom(self):
        # Triggers limitsChanges which update histograms
        self._imagePlot.resetZoom()

    def isKeepDataAspectRatio(self):
        return self.actionKeepDataAspectRatio.isChecked()

    def setKeepDataAspectRatio(self, keepRatio):
        self.actionKeepDataAspectRatio.setChecked(keepRatio)

    def isYAxisInverted(self):
        return self.actionInvertYAxis.isChecked()

    def setYAxisInverted(self, inverted):
        self.actionInvertYAxis.setChecked(inverted)

    # Colormap API
    def colormap(self):
        """Get the current colormap description.
        
        :return: A dict (See PlotBackend getDefaultColormap for details).
        """
        return self._colormap.copy()

    def setColormap(self, colormap):
        """Set the current colormap.

        :param dict colormap: colormap description
                              (See PlotBackend.getDefaultColormap for details).
        """
        if colormap is None:
            colormap = self._imagePlot.getDefaultColormap()
        assert colormap['name'] in self._imagePlot.getSupportedColormaps()
        self._colormap = colormap.copy()

        cursorColor = _cursorColorForColormap(colormap['name'])
        self._imagePlot.setZoomModeEnabled(True, color=cursorColor)

        if self._data is not None:  # Force refresh
            self._imagePlot.addImage(self._data, legend='image',
                                     replace=False, replot=False,
                                     colormap=self.colormap())
            self._imagePlot.replot()

    # Image API
    def setImage(self, image, copy=True, reset=True):
        """Set the image to display.
        
        :param image: A 2D array representing the image or None to empty plot.
        :type image: numpy.ndarray-like with 2 dimensions or None.
        :param bool copy: Whether to copy image data (default) or not.
        :param bool reset: Whether to reset zoom and ROI (default) or not.
        """
        self._cache = None
        self._cacheHisto = None

        if image is None:
            self._data = None
            return

        self._data = np.array(image, order='C', copy=copy)
        assert self._data.size != 0
        assert len(self._data.shape) == 2
        height, width = self._data.shape

        self._imagePlot.addImage(self._data, legend='image',
                                 replace=False, replot=False,
                                 colormap=self.colormap())
        self._updateHistograms()

        self._radarView.setDataRect(0, 0, width, height)

        if reset:
            self.resetZoom()
        else:
            self._histoHPlot.replot()
            self._histoVPlot.replot()
            self._imagePlot.replot()

    def imageHistogram(self, bins):
        """Calls numpy.histogram on the current image.

        Intended for ColormapDialog histogram.

        :param bins: See numpy.histogram for more details.
        :return: None if no image is set or the histogram of the image.
        :rtype: None or 2 arrays: counts and binEdges.
        """
        if self._data is not None:
            if self._cacheHisto is None or self._cacheHisto[0] != bins:
                self._cacheHisto = bins, np.histogram(self._data, bins)
            return self._cacheHisto[1]
        else:
            return None


# ImageViewMainWindow #########################################################

class ImageViewMainWindow(qt.QMainWindow):
    """QMainWindow embedding an ImageView.
    
    Surrounds the ImageView with an associated toolbar and status bar.
    """

    def __init__(self, parent=None, windowFlags=qt.Qt.Widget, backend=None):
        self._dataInfo = None
        super(ImageViewMainWindow, self).__init__(parent, windowFlags)

        self.imageView = ImageView(backend=backend)
        self.setCentralWidget(self.imageView)

        toolbar = qt.QToolBar('Image View')
        toolbar.addAction(self.imageView.actionResetZoom)
        toolbar.addAction(self.imageView.actionKeepDataAspectRatio)
        toolbar.addAction(self.imageView.actionChangeColormap)
        toolbar.addAction(self.imageView.actionInvertYAxis)
        self.addToolBar(toolbar)

        self.statusBar()
        self.imageView.valueChanged.connect(self._statusBarSlot)

    def _statusBarSlot(self, row, column, value):
        """Update status bar with coordinates/value from plots."""
        if math.isnan(row):
            msg = 'Column: %d, Sum: %g' % (int(column), value)
        elif math.isnan(column):
            msg = 'Row: %d, Sum: %g' % (int(row), value)
        else:
            msg = 'Position: (%d, %d), Value: %g' % (int(row), int(column),
                                                     value)
        if self._dataInfo is not None:
            msg = self._dataInfo + ', ' + msg

        self.statusBar().showMessage(msg)

    def setImage(self, image, *args, **kwargs):
        if hasattr(image, 'dtype') and hasattr(image, 'shape'):
            assert len(image.shape) == 2
            height, width = image.shape
            self._dataInfo = 'Data: %dx%d (%s)' % (width, height,
                                                   str(image.dtype))
            self.statusBar().showMessage(self._dataInfo)
        else:
            self._dataInfo = None

        self.imageView.setImage(image, *args, **kwargs)


# main ########################################################################

if __name__ == "__main__":
    import argparse
    import math
    import os.path
    import sys

    from PyMca5.PyMcaIO.EdfFile import EdfFile

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description='Browse the images of an EDF file.')
    parser.add_argument(
        '-b', '--backend',
        choices=('mpl', 'opengl', 'osmesa'),
        help="""The plot backend to use: Matplotlib (mpl, the default),
        OpenGL 2.1 (opengl, requires appropriate OpenGL drivers) or
        Off-screen Mesa OpenGL software pipeline (osmesa,
        requires appropriate OSMesa library).""")
    parser.add_argument('filename', help='EDF filename of the image to open')
    args = parser.parse_args()

    # Open the input file
    if not os.path.isfile(args.filename):
        raise RuntimeError('No input file: %s' % args.filename)

    edfFile = EdfFile(args.filename)
    nbFrames = edfFile.GetNumImages()
    if nbFrames == 0:
        raise RuntimeError('Cannot read image(s) from file: %s' % args.filename)

    # Set-up Qt application and main window
    app = qt.QApplication([])

    mainWindow = ImageViewMainWindow(backend=args.backend)
    mainWindow.setImage(edfFile.GetData(0))

    if nbFrames > 1:  # Add a toolbar for multi-frame EDF support
        multiFrameToolbar = qt.QToolBar('Multi-frame')
        multiFrameToolbar.addWidget(qt.QLabel(
            'Frame [0-%d]:' % (nbFrames - 1)))

        spinBox = qt.QSpinBox()
        spinBox.setRange(0, nbFrames-1)
        def updateImage(index):
            mainWindow.setImage(edfFile.GetData(index), reset=False)
        spinBox.valueChanged[int].connect(updateImage)
        multiFrameToolbar.addWidget(spinBox)

        mainWindow.addToolBar(multiFrameToolbar)

    mainWindow.show()

    sys.exit(app.exec_())
