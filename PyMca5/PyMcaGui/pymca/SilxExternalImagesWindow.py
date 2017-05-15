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


import numpy

from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
from PyMca5.PyMcaGui.plotting.PyMca_Icons import IconDict
from PyMca5.PyMcaGui.plotting import SilxMaskImageWidget

from silx.gui.plot import PlotWidget
from silx.gui.plot import PlotActions
from silx.gui.plot import PlotToolButtons


class SilxExternalImagesWindow(SilxMaskImageWidget.SilxMaskImageWidget):
    """Widget displaying a stack of images and allowing to apply simple
    editing on the images: cropping to current zoom, 90 degrees rotations,
    horizontal or vertical flipping.

    A slider enables browsing through the images.

    All images must have the same shape. The operations are applied to
    all images, not only the one currently displayed.
    """
    def __init__(self, parent=None):
        SilxMaskImageWidget.SilxMaskImageWidget.__init__(self, parent=parent)

        self.addImageButton.clicked.connect(self._addImageClicked)
        self.removeImageButton.clicked.connect(self._removeImageClicked)
        self.replaceImageButton.clicked.connect(self._replaceImageClicked)

        # Image selection slider
        self.slider = qt.QSlider(self.centralWidget())
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        self.centralWidget().layout().insertWidget(1, self.slider)
        self.slider.valueChanged[int].connect(self.showImage)


        # Additional actions added to action group
        self.cropIcon = qt.QIcon(qt.QPixmap(IconDict["crop"]))
        self.cropButton = qt.QToolButton(self)
        self.cropButton.setIcon(self.cropIcon)
        self.cropButton.setToolTip("Crop image to current zoomed area")
        self.cropButton.clicked.connect(self._cropIconChecked)

        self.hFlipIcon = qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"]))
        self.hFlipToolButton = qt.QToolButton(self)
        self.hFlipToolButton.setIcon(self.hFlipIcon)
        self.hFlipToolButton.setToolTip("Flip image and data, not the scale.")
        self._flipMenu = qt.QMenu()
        self._flipMenu.addAction(QString("Flip Image Left-Right"),
                                 self._flipLeftRight)
        self._flipMenu.addAction(QString("Flip Image Up-Down"),
                                 self._flipUpDown)

        self.hFlipToolButton.setMenu(self._flipMenu)
        self.hFlipToolButton.setPopupMode(qt.QToolButton.InstantPopup)

        self.rotateLeftIcon = qt.QIcon(qt.QPixmap(IconDict["rotate_left"]))
        self.rotateRightIcon = qt.QIcon(qt.QPixmap(IconDict["rotate_right"]))
        self.rotateButton = qt.QToolButton(self)
        self.rotateButton.setIcon(self.rotateLeftIcon)
        self.rotateButton.setToolTip("Rotate image by 90 degrees")
        self._rotateMenu = qt.QMenu()
        self.rotateLeftAction = qt.QAction(self.rotateLeftIcon,
                                           QString("Rotate left"),
                                           self)
        self.rotateLeftAction.triggered.connect(self._rotateLeft)
        self._rotateMenu.addAction(self.rotateLeftAction)
        self.rotateRightAction = qt.QAction(self.rotateRightIcon,
                                            QString("Rotate right"),
                                            self)
        self.rotateRightAction.triggered.connect(self._rotateRight)
        self._rotateMenu.addAction(self.rotateRightAction)

        self.rotateButton.setMenu(self._rotateMenu)
        self.rotateButton.setPopupMode(qt.QToolButton.InstantPopup)

        # Creating the toolbar also create actions for toolbuttons
        self._imageEditingToolbar = self._createImageEditingToolBar(title='Plot', parent=None)
        self.addToolBar(self._imageEditingToolbar)

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

        self._origin = (0., 0.)
        """Current image origin: coordinate (x, y) of sample located at
        (row, column) = (0, 0)"""

        self._maskIsSet = False

    def _createImageEditingToolBar(self, title, parent):
        """Create a QToolBar with crop, rotate and flip operations

        :param str title: The title of the QMenu
        :param qt.QWidget parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(title, parent)

        # custom widgets added to the end
        toolbar.addWidget(self.cropButton)
        toolbar.addWidget(self.hFlipToolButton)
        toolbar.addWidget(self.rotateButton)

        return toolbar

    @staticmethod
    def _getImageData(image):
        """Convert RGBA image to 2D array of grayscale values
        (Luma coding)

        :param image: RGBA image, as a numpy array of shapes
             (nrows, ncols, 3/4)
        :return:
        """
        if len(image.shape) == 2:
            return image
        assert len(image.shape) == 3

        imageData = image[:, :, 0] * 0.299 +\
                    image[:, :, 1] * 0.587 +\
                    image[:, :, 2] * 0.114
        return imageData

    def _addImageClicked(self):
        imageData = self._getImageData(self._images[self.slider.value()])
        ddict = {
            'event': "addImageClicked",
            'image': imageData,
            'title': self.plot.getGraphTitle(),
            'id': id(self)}
        self.sigMaskImageWidget.emit(ddict)

    def _replaceImageClicked(self):
        imageData = self._getImageData(self._images[self.slider.value()])
        ddict = {
            'event': "replaceImageClicked",
            'image': imageData,
            'title': self.plot.getGraphTitle(),
            'id': id(self)}
        self.sigMaskImageWidget.emit(ddict)

    def _removeImageClicked(self):
        imageData = self._getImageData(self._images[self.slider.value()])
        ddict = {
            'event': "removeImageClicked",
            'image': imageData,
            'title': self.plot.getGraphTitle(),
            'id': id(self)}
        self.sigMaskImageWidget.emit(ddict)

    def _getCurrentHeightWidth(self):
        image = self._images[self.slider.value()]
        ncols = image.shape[1]
        nrows = image.shape[0]
        width = ncols * self._scale[0]   # X
        height = nrows * self._scale[1]  # Y
        return height, width

    def _cropIconChecked(self):  # fixme: why does this require qImages?
        """Crop all images in :attr:`qImages` to the X and Y ranges
        currently displayed (crop to zoomed area)."""
        height, width = self._getCurrentHeightWidth()

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

        # replace current image data by the new one, keep (width, height)
        self.setImages(croppedImages, self._labels,
                       origin=self._origin,
                       width=width, height=height)

        self.sigMaskImageWidget.emit(
                {'event': "cropSignal"})

    def _flipUpDown(self):
        flippedImages = []
        for img in self._images:
            flippedImages.append(numpy.flipud(img))
        self._images = flippedImages

        self.sigMaskImageWidget.emit(
                {'event': "flipUpDownSignal"})

        self.showImage(self.slider.value())

    def _flipLeftRight(self):
        flippedImages = []
        for img in self._images:
            flippedImages.append(numpy.fliplr(img))
        self._images = flippedImages

        self.sigMaskImageWidget.emit(
                {'event': "flipLeftRightSignal"})

        self.showImage(self.slider.value())

    def _rotateRight(self):
        """Rotate the image 90 degrees clockwise.

        Depending on the Y axis orientation, the array must be
        rotated by 90 or 270 degrees."""
        rotatedImages = []
        if not self.plot.isYAxisInverted():
            for img in self._images:
                rotatedImages.append(numpy.rot90(img, 1))
        else:
            for img in self._images:
                rotatedImages.append(numpy.rot90(img, 3))

        self.sigMaskImageWidget.emit(
                {'event': "rotateRight"})

        height, width = self._getCurrentHeightWidth()

        self.setImages(rotatedImages, self._labels,
                       origin=self._origin,
                       width=width, height=height)

    def _rotateLeft(self):
        """Rotate the image 90 degrees counterclockwise.

        Depending on the Y axis orientation, the array must be
        rotated by 90 or 270 degrees."""
        height, width = self._getCurrentHeightWidth()

        rotatedImages = []
        if not self.plot.isYAxisInverted():
            for img in self._images:
                rotatedImages.append(numpy.rot90(img, 3))
        else:
            for img in self._images:
                rotatedImages.append(numpy.rot90(img, 1))

        self.sigMaskImageWidget.emit(
                {'event': "rotateLeft"})

        self.setImages(rotatedImages, self._labels,
                       origin=self._origin,
                       width=width, height=height)

    def showImage(self, index=0):
        """Show image corresponding to index. Update slider to index."""
        if not self._images:
            return
        assert index < len(self._images)

        self.plot.remove(legend="current")
        self.plot.addImage(self._images[index],
                           legend="current",
                           origin=self._origin,
                           scale=self._scale,
                           replace=False)
        self.plot.setGraphTitle(self._labels[index])
        self.slider.setValue(index)

    def setImages(self, images, labels=None,
                  origin=None, height=None, width=None):
        """Set the list of images.

        :param images: List of 2D or 3D (for RGBA data) numpy arrays
            of image data. All images must have the same shape.
        :type images: List of ndarrays
        :param labels: list of image names
        :param origin: Image origin: coordinate (x, y) of sample located at
            (row, column) = (0, 0). If None, use (0., 0.)
        :param height: Image height in Y axis units. If None, use the
            image height in number of pixels.
        :param width: Image width in X axis units. If None, use the
            image width in number of pixels.
        """
        self._images = images
        if labels is None:
            labels = ["Image %d" % (i + 1) for i in range(len(images))]

        self._labels = labels

        height_pixels, width_pixels = images[0].shape[0:2]
        height = height or height_pixels
        width = width or width_pixels

        self._scale = (float(width) / width_pixels,
                       float(height) / height_pixels)

        self._origin = origin or (0., 0.)

        current = self.slider.value()
        self.slider.setMaximum(len(self._images) - 1)
        if current < len(self._images):
            self.showImage(current)
        else:
            self.showImage(0)

    def resetMask(self, width, height,
                  origin=None, scale=None):
        """Initialize a mask with a given width and height.

        The mask may be synchronized with another widget.
        The mask size must therefore match the master widget's image
        size (in pixels).

        :param width: Mask width
        :param height: Mask height
        :param origin: Tuple of (X, Y) coordinates of the sample (0, 0)
        :param scale: Tuple of (xscale, yscale) scaling factors, in axis units
            per pixel.
        """
        transparent_active_image = numpy.zeros((int(height), int(width), 4))
        # set alpha for total transparency
        transparent_active_image[:, :, -1] = 0

        origin = origin or (0., 0.)
        scale = scale or (1., 1.)

        self.plot.addImage(transparent_active_image,
                           origin=origin,
                           scale=scale,
                           legend="mask support",
                           replace=False)
        self.plot.setActiveImage("mask support")

        self.setSelectionMask(numpy.zeros((int(height), int(width))))
        self._maskIsSet = True

    def getCurrentIndex(self):
        return self.slider.value()


def test():
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)

    data = numpy.arange(10000)
    data.shape = 50, 200
    data[8:12, 48:52] = 10000
    data[6:14, 146:154] = 10000
    data[34:46, 44:56] = 0
    data[32:48, 142:158] = 0

    container = SilxExternalImagesWindow()
    container.setImages([data])
    container.show()

    def theSlot(ddict):
        print(ddict)

    container.sigMaskImageWidget.connect(theSlot)

    app.exec_()

if __name__ == "__main__":
    test()
