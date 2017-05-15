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

        toolbar = qt.QToolBar("Image edition", parent=None)
        # custom widgets added to the end
        toolbar.addWidget(self.cropButton)
        toolbar.addWidget(self.hFlipToolButton)
        toolbar.addWidget(self.rotateButton)
        self.addToolBar(toolbar)

    def _cropIconChecked(self):
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
