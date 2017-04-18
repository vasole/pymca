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
__authors__ = ["V.A. Sole", "P. Knobel"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"

from collections import OrderedDict

import sys
import os
import numpy


from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
from PyMca5.PyMcaGui.plotting.PyMca_Icons import IconDict

from silx.gui.plot import PlotWidget
from silx.gui.plot import PlotActions
from silx.gui.plot import PlotToolButtons
from silx.gui.plot import MaskToolsWidget


class SilxExternalImagesWindow(qt.QMainWindow):
    """Widget displaying a stack of images and allowing to apply simple
    editing on the images: cropping to current zoom, 90 degrees rotations,
    horizontal or vertical flipping.

    A slider enables browsing through the images.

    All images must have the same shape. The operations are applied to
    all images, not only the one currently displayed.
    """
    # TODO: rotation
    sigMaskChanged = qt.pyqtSignal()
    sigImageModified = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QMainWindow.__init__(self, parent=parent)

        centralWidget = qt.QWidget(self)
        layout = qt.QVBoxLayout(centralWidget)
        centralWidget.setLayout(layout)

        self.plot = PlotWidget(parent=centralWidget)
        self.plot.setWindowFlags(qt.Qt.Widget)
        layout.addWidget(self.plot)

        self._maskToolsDockWidget = None

        # Init actions
        self.group = qt.QActionGroup(self)
        self.group.setExclusive(False)

        self.resetZoomAction = self.group.addAction(
                PlotActions.ResetZoomAction(plot=self.plot, parent=self))
        self.addAction(self.resetZoomAction)

        self.zoomInAction = PlotActions.ZoomInAction(plot=self.plot, parent=self)
        self.addAction(self.zoomInAction)

        self.zoomOutAction = PlotActions.ZoomOutAction(plot=self.plot, parent=self)
        self.addAction(self.zoomOutAction)

        self.xAxisAutoScaleAction = self.group.addAction(
            PlotActions.XAxisAutoScaleAction(plot=self.plot, parent=self))
        self.addAction(self.xAxisAutoScaleAction)

        self.yAxisAutoScaleAction = self.group.addAction(
            PlotActions.YAxisAutoScaleAction(plot=self.plot, parent=self))
        self.addAction(self.yAxisAutoScaleAction)

        self.colormapAction = self.group.addAction(
                PlotActions.ColormapAction(plot=self.plot, parent=self))
        self.addAction(self.colormapAction)

        self.keepDataAspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=self.plot)

        self.group.addAction(self.getMaskAction())

        self._separator = qt.QAction('separator', self)
        self._separator.setSeparator(True)
        self.group.addAction(self._separator)

        self.cropIcon = qt.QIcon(qt.QPixmap(IconDict["crop"]))
        self.cropButton = qt.QToolButton(self)
        self.cropButton.setIcon(self.cropIcon)
        self.cropButton.setToolTip("Crop image to the currently zoomed window")
        self.cropButton.clicked.connect(self._cropIconChecked)

        self.hFlipIcon = qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"]))
        self.hFlipToolButton = qt.QToolButton(self)
        self.hFlipToolButton.setIcon(self.hFlipIcon)
        self.hFlipToolButton.setToolTip("Flip image and data, not the scale.")

        self._flipMenu = qt.QMenu()
        self._flipMenu.addAction(QString("Invert Y axis direction"),
                                 self._hFlipIconSignal)
        self._flipMenu.addAction(QString("Flip Image Left-Right"),
                                 self._flipLeftRight)
        self._flipMenu.addAction(QString("Flip Image Up-Down"),
                                 self._flipUpDown)

        self.hFlipToolButton.setMenu(self._flipMenu)
        self.hFlipToolButton.setPopupMode(qt.QToolButton.InstantPopup)

        # TODO: rotate...

        # Creating the toolbar also create actions for toolbuttons
        self._toolbar = self._createToolBar(title='Plot', parent=None)
        self.addToolBar(self._toolbar)

        self.slider = qt.QSlider(centralWidget)
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        layout.addWidget(self.slider)
        self.slider.valueChanged[int].connect(self.showImage)

        self.setCentralWidget(centralWidget)

        self._images = []
        """List of images, as 2D numpy arrays or 3D numpy arrays (RGB(A)).
        """
        self._labels = []
        """List of image labels.
        """

        self._scale = (1.0, 1.0)
        """Current image scale (Xscale, Yscale) (in axis units per image pixel).
        The scale is adjusted to keep constant width and height for the image
        when a crop operation is applied."""

        self._maskIsSet = False

    def sizeHint(self):
        return qt.QSize(400, 400)

    def _createToolBar(self, title, parent):
        """Create a QToolBar from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param qt.QWidget parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(title, parent)

        # Order widgets with actions
        objects = self.group.actions()

        # Add standard push buttons to list
        index = objects.index(self.colormapAction)
        objects.insert(index + 1, self.keepDataAspectRatioButton)
        # objects.insert(index + 2, self.yAxisInvertedButton)

        for obj in objects:
            if isinstance(obj, qt.QAction):
                toolbar.addAction(obj)
            else:
                # keep reference to toolbutton's action for changing visibility
                if obj is self.keepDataAspectRatioButton:
                    self.keepDataAspectRatioAction = toolbar.addWidget(obj)
                elif obj is self.yAxisInvertedButton:
                    self.yAxisInvertedAction = toolbar.addWidget(obj)
                else:
                    raise RuntimeError()

        # custom widgets added to the end
        toolbar.addWidget(self.cropButton)
        toolbar.addWidget(self.hFlipToolButton)

        return toolbar

    def getMaskToolsDockWidget(self):
        """DockWidget with image mask panel (lazy-loaded)."""
        if self._maskToolsDockWidget is None:
            self._maskToolsDockWidget = MaskToolsWidget.MaskToolsDockWidget(
                plot=self.plot, name='Mask')
            self._maskToolsDockWidget.hide()
            self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                               self._maskToolsDockWidget)
            self._maskToolsDockWidget.setFloating(True)
        return self._maskToolsDockWidget

    def getMaskAction(self):
        """QAction toggling image mask dock widget

        :rtype: QAction
        """
        return self.getMaskToolsDockWidget().toggleViewAction()

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
                    Mask type: array of uint8 of dimension 2,
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 2-tuple if successful.
                 The mask can be cropped or padded to fit active image,
                 the returned shape is that of the active image.
        """
        return self.getMaskToolsDockWidget().setSelectionMask(mask,
                                                              copy=copy)

    def getSelectionMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self.getMaskToolsDockWidget().getSelectionMask(copy=copy)

    def _cropIconChecked(self):  # fixme: why does this require qImages?
        """Crop all images in :attr:`qImages` to the X and Y ranges
        currently displayed (crop to zoomed area)."""
        # current image
        image = self._images[self.slider.value()]
        ncols = image.shape[1]
        nrows = image.shape[0]
        width = ncols * self._scale[0]   # X
        height = nrows * self._scale[1]  # Y

        xmin, xmax = map(int, self.plot.getGraphXLimits())
        ymin, ymax = map(int, self.plot.getGraphYLimits())

        xmin = max(min(xmax, xmin), 0)
        xmax = min(max(xmax, xmin), width)
        ymin = max(min(ymax, ymin), 0)
        ymax = min(max(ymax, ymin), height)

        cols_min = int(xmin / self._scale[0])
        cols_max = int(xmax / self._scale[0])
        rows_min = int(ymin / self._scale[1])
        rows_max = int(ymax / self._scale[1])

        croppedImages = []
        for img in self._images:
            image = img[rows_min:rows_max, cols_min:cols_max]
            croppedImages.append(image)

        # newXscale = self._scale[0] * width / (xmax - xmin)
        # newYscale = self._scale[1] * height / (ymax - ymin)

        # # change the scale to keep the same height and width (now done in setImages
        # self._scale = newXscale, newYscale

        # replace current image by the new one
        self.setImages(croppedImages, self._labels,
                       width=width, height=height)

        self.showImage(self.slider.value())

    # def _flipIconChecked(self):
    #     if not self.plot.isYAxisAutoScale():
    #         qt.QMessageBox.information(
    #                 self, "Open",
    #                 "Please set Y Axis to AutoScale first")
    #         return
    #     if not self.plot.isXAxisAutoScale():
    #         qt.QMessageBox.information(
    #                 self, "Open",
    #                 "Please set X Axis to AutoScale first")
    #         return
    #     if not self.qImages:
    #         assert not self.numpyImages, "numpy images and qimages not in sync"
    #         return
    #     self._flipMenu.exec_(self.cursor().pos())

    def _hFlipIconSignal(self):
        isYAxisInverted = self.plot.isYAxisInverted()
        # self.plot.resetzoom()
        self.plot.setYAxisInverted(not isYAxisInverted)

        # inform the other widgets
        self.sigImageModified.emit(
            {'event': "hFlipSignal",
             'current': self.plot.isYAxisInverted(),
             'id': id(self)}
        )

    def _flipUpDown(self):
        flippedImages = []
        for img in self._images:
            flippedImages.append(numpy.flipud(img))
        self._images = flippedImages

        self.sigImageModified.emit(
                {'event': "flipUpDownSignal"}
        )

        self.showImage(self.slider.value())

    def _flipLeftRight(self):
        flippedImages = []
        for img in self._images:
            flippedImages.append(numpy.fliplr(img))
        self._images = flippedImages

        self.sigImageModified.emit(
                {'event': "flipLeftRightSignal"}
        )

        self.showImage(self.slider.value())

    def showImage(self, index=0):
        """Show image corresponding to index. Update slider to index."""
        if not self._images:
            return
        assert index < len(self._images)

        self.plot.remove(legend="current")
        self.plot.addImage(self._images[index],
                           legend="current",
                           scale=self._scale,
                           replace=False)
        self.plot.setGraphTitle(self._labels[index])
        self.slider.setValue(index)

    def setImages(self, images, labels=None,
                  height=None, width=None):
        """Set the list of images.

        :param images: List of 2D or 3D (for RGBA data) numpy arrays
            of image data. All images must have the same shape.
        :type images: List of ndarrays
        :param labels: list of image names
        :param height: Image height in Y axis units. If None, use the
            image shape.
        :param width: Image width in X axis units. If None, use the
            image shape.
        """
        self._images = images
        if labels is None:
            labels = ["Image %d" % (i + 1) for i in range(len(images))]

        self._labels = labels

        pixel_height, pixel_width = images[0].shape[0:2]
        if height is None:
            height = pixel_height
        if width is None:
            width = pixel_width

        if not self._maskIsSet:
            self.resetMask(width, height)

        self._scale = (float(width) / pixel_width,
                       float(height) / pixel_height)

        current = self.slider.value()
        self.slider.setMaximum(len(self._images) - 1)
        if current < len(self._images):
            self.showImage(current)
        else:
            self.showImage(0)

    def resetMask(self, width, height):
        """Initialize a mask with a given width and height.

        :param width:
        :param height:
        :return:
        """
        transparent_active_image = numpy.zeros((height, width, 4))
        # set alpha for total transparecy
        transparent_active_image[:, :, -1] = 0
        self.plot.addImage(transparent_active_image, legend="mask support")
        self.plot.setActiveImage("mask support")

        self.setSelectionMask(numpy.zeros((height, width)))
        self._maskIsSet = True


    # def _updateProfileCurve(self, ddict):
    #     if not self._depthSelection:
    #         return MaskImageWidget.MaskImageWidget._updateProfileCurve(self,
    #                                                                    ddict)
    #     nImages = len(self.imageNames)
    #     for i in range(nImages):
    #         image=self.imageList[i]
    #         overlay = False
    #         if i == 0:
    #             overlay = MaskImageWidget.OVERLAY_DRAW
    #             replace = True
    #             if len(self.imageNames) == 1:
    #                 replot = True
    #             else:
    #                 replot = False
    #         elif i == (nImages -1):
    #             replot = True
    #             replace=False
    #         else:
    #             replot = False
    #             replace= False
    #         curve = self._getProfileCurve(ddict, image=image, overlay=overlay)
    #         if curve is None:
    #             return
    #         xdata, ydata, legend, info = curve
    #         newLegend = self.imageNames[i]+ " " + legend
    #         self._profileSelectionWindow.addCurve(xdata, ydata,
    #                                               legend=newLegend,
    #                                               info=info,
    #                                               replot=replot,
    #                                               replace=replace)

    def getCurrentIndex(self):
        return self.slider.value()


def test():
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)

    container = SilxExternalImagesWindow()
    data = numpy.arange(10000)
    data.shape = 50, 200
    data[25, :] = 0
    data[:, 130] = 10000
    container.setImages([data])
    container.show()

    def theSlot(ddict):
        print(ddict)

    container.sigImageModified.connect(theSlot)

    app.exec_()

if __name__ == "__main__":
    test()
