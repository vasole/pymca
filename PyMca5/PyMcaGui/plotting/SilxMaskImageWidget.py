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
"""Base class for SilxStackRoiWindow and SilxExternalImagesWindow"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"

# TODO: in save actions, colormap access needs to be updated for silx plot

import numpy
import os

from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
from PyMca5.PyMcaGui.plotting.PyMca_Icons import IconDict
from PyMca5.PyMcaIO import ArraySave
from PyMca5.PyMcaCore import PyMcaDirs
from PyMca5 import spslut

try:
    from PyMca5.PyMcaGui.pymca import QPyMcaMatplotlibSave
except ImportError:
    QPyMcaMatplotlibSave = None

from silx.gui.plot import PlotWidget
from silx.gui.plot import PlotActions
from silx.gui.plot import PlotToolButtons
from silx.gui.plot.MaskToolsWidget import MaskToolsDockWidget
from silx.gui import icons


class SaveImageListAction(qt.QAction):
    """Save current image  and mask (if any) in a :class:`MaskImageWidget`
    to EDF or CSV"""
    def __init__(self, title, maskImageWidget,
                 clipped=False, subtract=False):
        super(SaveImageListAction, self).__init__(QString(title),
                                                  maskImageWidget)
        self.maskImageWidget = maskImageWidget
        self.triggered[bool].connect(self.onTrigger)

        self.outputDir = PyMcaDirs.outputDir
        """Default output dir. After each save operation, this is updated
        to re-use the same folder for next save."""
        self.clipped = clipped
        """If True, clip data range to colormap min and max."""
        self.subtract = subtract
        """If True, subtract data min value."""

    def onTrigger(self):
        filename, saveFilter = self.getOutputFileNameFilter()
        if not filename:
            return

        if filename.lower().endswith(".csv"):
            csvseparator = "," if "," in saveFilter else\
                ";" if ";" in saveFilter else\
                "\t"
        else:
            csvseparator = None
        images, labels = self.getImagesLabels()

        self.saveImageList(filename, images, labels, csvseparator)

    def saveImageList(self, filename, imageList, labels,
                      csvseparator=None):
        if not imageList:
            qt.QMessageBox.information(
                    self,
                    "No Data",
                    "Image list is empty.\nNothing to be saved")
            return

        if filename.lower().endswith(".edf"):
            ArraySave.save2DArrayListAsEDF(imageList, filename, labels)
        elif filename.lower().endswith(".csv"):
            assert csvseparator is not None
            ArraySave.save2DArrayListAsASCII(imageList, filename, labels,
                                             csv=True,
                                             csvseparator=csvseparator)
        else:
            ArraySave.save2DArrayListAsASCII(imageList, filename, labels,
                                             csv=False)

    def getImagesLabels(self):
        """Return images to be saved and corresponding labels.

        Images are:
             - image currently displayed clipped to visible colormap range
             - mask

        If :attr:`subtract` is True, subtract the minimum image sample value
        to all samples."""
        imageList = []
        labels = []
        imageData = self.maskImageWidget.getImageData()
        label = self.maskImageWidget.plot.getGraphTitle()
        if not label:
            label = "Image01"
        label.replace(' ', '_')
        if self.clipped and self.colormap is not None:                           # fixme --------------------------------------------------
            colormapIndex, autoscale, vmin, vmax = self.colormap[0:4]
            if not autoscale:
                imageData = imageData.clip(vmin, vmax)
                label += ".clip(%f,%f)" % (vmin, vmax)
        if self.subtract:
            vmin = imageData.min()
            imageData = imageData - vmin
            label += "-%f" % vmin
        imageList.append(imageData)
        labels.append(label)

        mask = self.maskImageWidget.getSelectionMask()
        if mask is not None and mask.max() > 0:
            imageList.append(mask)
            labels.append(label + "_Mask")

        return imageList, labels

    def getOutputFileNameFilter(self):
        """Open a file dialog to get the output file name, and return the
        file name and the selected format filter.

        Remember output directory in attribute :attr:`outputDir`"""
        if os.path.exists(self.outputDir):
            initdir = self.outputDir
        else:
            # folder deleted, reset
            initdir = PyMcaDirs.outputDir
        filedialog = qt.QFileDialog(self.maskImageWidget)
        filedialog.setFileMode(filedialog.AnyFile)
        filedialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict["gioconda16"])))
        formatlist = ["ASCII Files *.dat",
                      "EDF Files *.edf",
                      'CSV(, separated) Files *.csv',
                      'CSV(; separated) Files *.csv',
                      'CSV(tab separated) Files *.csv']
        if hasattr(qt, "QStringList"):
            strlist = qt.QStringList()
        else:
            strlist = []
        for f in formatlist:
            strlist.append(f)
        saveFilter = formatlist[0]
        if hasattr(filedialog, "setFilters"):
            filedialog.setFilters(strlist)
            filedialog.selectFilter(saveFilter)
        else:
            filedialog.setNameFilters(strlist)
            filedialog.selectNameFilter(saveFilter)
        filedialog.setDirectory(initdir)
        ret = filedialog.exec_()
        if not ret:
            return "", ""
        filename = filedialog.selectedFiles()[0]
        if filename:
            filename = qt.safe_str(filename)
            self.outputDir = os.path.dirname(filename)
            if hasattr(filedialog, "selectedFilter"):
                saveFilter = qt.safe_str(filedialog.selectedFilter())
            else:
                saveFilter = qt.safe_str(filedialog.selectedNameFilter())
            filterused = "." + saveFilter[-3:]
            PyMcaDirs.outputDir = os.path.dirname(filename)
            if len(filename) < 4 or filename[-4:] != filterused:
                filename += filterused
        else:
            filename = ""
        return filename, saveFilter


class SaveMatplotlib(qt.QAction):
    """Save current image  and mask (if any) in a :class:`MaskImageWidget`
    to EDF or CSV"""
    def __init__(self, title, maskImageWidget):
        super(SaveMatplotlib, self).__init__(QString(title),
                                             maskImageWidget)
        self.maskImageWidget = maskImageWidget
        self.triggered[bool].connect(self.onTrigger)

        self._matplotlibSaveImage = None

    def onTrigger(self):
        imageData = self.maskImageWidget.getImageData()
        if self._matplotlibSaveImage is None:
            self._matplotlibSaveImage = QPyMcaMatplotlibSave.SaveImageSetup(
                    None, image=None)
        title = "Matplotlib " + self.maskImageWidget.plot.getGraphTitle()
        self._matplotlibSaveImage.setWindowTitle(title)
        ddict = self._matplotlibSaveImage.getParameters()

        if self.colormap is not None:     # Fixme !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            colormapType = ddict['linlogcolormap']
            try:
                colormapIndex, autoscale, vmin, vmax,\
                        dataMin, dataMax, colormapType = self.colormap
                if colormapType == spslut.LOG:
                    colormapType = 'logarithmic'
                else:
                    colormapType = 'linear'
            except:
                colormapIndex, autoscale, vmin, vmax = self.colormap[0:4]
            ddict['linlogcolormap'] = colormapType
            if not autoscale:
                ddict['valuemin'] = vmin
                ddict['valuemax'] = vmax
            else:
                ddict['valuemin'] = 0
                ddict['valuemax'] = 0

        # this sets the actual dimensions
        origin = self.maskImageWidget._origin
        scale = self.maskImageWidget._scale

        ddict['xpixelsize'] = scale[0]
        ddict['xorigin'] = origin[0]
        ddict['ypixelsize'] = scale[1]
        ddict['yorigin'] = origin[1]

        ddict['xlabel'] = self.maskImageWidget.plot.getXLabel()
        ddict['ylabel'] = self.maskImageWidget.plot.getYLabel()
        limits = self.maskImageWidget.plot.getGraphXLimits()
        ddict['zoomxmin'] = limits[0]
        ddict['zoomxmax'] = limits[1]
        limits = self.maskImageWidget.plot.getGraphYLimits()
        ddict['zoomymin'] = limits[0]
        ddict['zoomymax'] = limits[1]

        self._matplotlibSaveImage.setParameters(ddict)
        self._matplotlibSaveImage.setImageData(imageData)
        self._matplotlibSaveImage.show()
        self._matplotlibSaveImage.raise_()


class SilxMaskImageWidget(qt.QMainWindow):
    """

    """
    sigMaskImageWidget = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle("PyMca - Image Selection Tool")

        centralWidget = qt.QWidget(self)
        layout = qt.QVBoxLayout(centralWidget)
        centralWidget.setLayout(layout)

        # Plot
        self.plot = PlotWidget(parent=centralWidget)
        self.plot.setWindowFlags(qt.Qt.Widget)
        layout.addWidget(self.plot)

        # Mask Widget
        self._maskToolsDockWidget = None

        # Image selection slider
        self.slider = qt.QSlider(self.centralWidget())
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        layout.addWidget(self.slider)
        self.slider.valueChanged[int].connect(self.showImage)

        # ADD/REMOVE/REPLACE IMAGE buttons
        buttonBox = qt.QWidget(self)
        buttonBoxLayout = qt.QHBoxLayout(buttonBox)
        buttonBoxLayout.setContentsMargins(0, 0, 0, 0)
        buttonBoxLayout.setSpacing(0)
        self.addImageButton = qt.QPushButton(buttonBox)
        icon = qt.QIcon(qt.QPixmap(IconDict["rgb16"]))
        self.addImageButton.setIcon(icon)
        self.addImageButton.setText("ADD IMAGE")
        self.addImageButton.setToolTip("Add image to RGB correlator")
        buttonBoxLayout.addWidget(self.addImageButton)

        self.removeImageButton = qt.QPushButton(buttonBox)
        self.removeImageButton.setIcon(icon)
        self.removeImageButton.setText("REMOVE IMAGE")
        self.removeImageButton.setToolTip("Remove image from RGB correlator")
        buttonBoxLayout.addWidget(self.removeImageButton)

        self.replaceImageButton = qt.QPushButton(buttonBox)
        self.replaceImageButton.setIcon(icon)
        self.replaceImageButton.setText("REPLACE IMAGE")
        self.replaceImageButton.setToolTip(
                "Replace all images in RGB correlator with this one")
        buttonBoxLayout.addWidget(self.replaceImageButton)

        self.addImageButton.clicked.connect(self._addImageClicked)
        self.removeImageButton.clicked.connect(self._removeImageClicked)
        self.replaceImageButton.clicked.connect(self._replaceImageClicked)

        layout.addWidget(buttonBox)

        self.setCentralWidget(centralWidget)

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

        self.yAxisInvertedButton = PlotToolButtons.YAxisOriginToolButton(
            parent=self, plot=self.plot)
        # self.yAxisInvertedButton.clicked.connect(self._hFlipIconSignal)

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

        # Creating the toolbar also create actions for toolbuttons
        self._toolbar = self._createToolBar(title='Plot', parent=None)
        self.addToolBar(self._toolbar)

        self._buildStandaloneSaveMenu()

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

    def sizeHint(self):
        return qt.QSize(500, 400)

    def _createToolBar(self, title, parent):
        """Create a QToolBar with crop, rotate and flip operations

        :param str title: The title of the QMenu
        :param qt.QWidget parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(title, parent)

        # Order widgets with actions
        objects = self.group.actions()

        # Add standard push buttons to list
        index = objects.index(self.colormapAction)
        objects.insert(index + 1, self.keepDataAspectRatioButton)
        objects.insert(index + 2, self.yAxisInvertedButton)

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
        return toolbar

    def _buildStandaloneSaveMenu(self):
        toolbar = qt.QToolBar("save", self)

        self.saveToolbutton = qt.QToolButton(toolbar)
        self.saveToolbutton.setIcon(icons.getQIcon("document-save"))
        self.saveToolbutton.clicked.connect(self._saveToolButtonSignal)
        self.saveToolbutton.setToolTip('Save Graph')

        toolbar.addWidget(self.saveToolbutton)

        self.addToolBar(toolbar)

        self._saveMenu = qt.QMenu()
        self._saveMenu.addAction(
                SaveImageListAction("Image Data", self))
        self._saveMenu.addAction(
                SaveImageListAction("Colormap Clipped Seen Image Data",
                                    self, clipped=True))
        self._saveMenu.addAction(
                SaveImageListAction("Clipped and Subtracted Seen Image Data",
                                    self, clipped=True, subtract=True))
        # standard silx save action
        self._saveMenu.addAction(PlotActions.SaveAction(plot=self.plot, parent=self))

        if QPyMcaMatplotlibSave is not None:
            self._saveMenu.addAction(SaveMatplotlib("Matplotlib", self))

    def _saveToolButtonSignal(self):
        self._saveMenu.exec_(self.cursor().pos())

    def _getMaskToolsDockWidget(self):
        """DockWidget with image mask panel (lazy-loaded)."""
        if self._maskToolsDockWidget is None:
            self._maskToolsDockWidget = MaskToolsDockWidget(
                plot=self.plot, name='Mask')
            self._maskToolsDockWidget.hide()
            self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                               self._maskToolsDockWidget)
            self._maskToolsDockWidget.setFloating(True)
            self._maskToolsDockWidget.sigMaskChanged.connect(
                    self._emitMaskImageWidgetSignal)
        return self._maskToolsDockWidget

    def getMaskAction(self):
        """QAction toggling image mask dock widget

        :rtype: QAction
        """
        return self._getMaskToolsDockWidget().toggleViewAction()

    def _emitMaskImageWidgetSignal(self):
        self.sigMaskImageWidget.emit(
            {"event": "selectionMaskChanged",
             "current": self.getSelectionMask(),
             "id": id(self)})

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
        # don't emit signal for programmatic mask change,
        # only for interactive mask drawing
        # (avoid infinite loop)
        self._getMaskToolsDockWidget().sigMaskChanged.disconnect(
                    self._emitMaskImageWidgetSignal)
        ret = self._getMaskToolsDockWidget().setSelectionMask(mask,
                                                              copy=copy)
        self._getMaskToolsDockWidget().sigMaskChanged.connect(
                    self._emitMaskImageWidgetSignal)
        return ret

    def getSelectionMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self._getMaskToolsDockWidget().getSelectionMask(copy=copy)

    def getImageData(self):
        """Return image data to be sent to RGB correlator

        Convert RGBA image to 2D array of grayscale values
        (Luma coding)

        :param image: RGBA image, as a numpy array of shapes
             (nrows, ncols, 3/4)
        :return:
        """
        index = self.slider.value()
        image = self._images[index]
        if len(image.shape) == 2:
            return image
        assert len(image.shape) == 3

        imageData = image[:, :, 0] * 0.299 +\
                    image[:, :, 1] * 0.587 +\
                    image[:, :, 2] * 0.114
        return imageData

    def _addImageClicked(self):
        imageData = self.getImageData()
        ddict = {
            'event': "addImageClicked",
            'image': imageData,
            'title': self.plot.getGraphTitle(),
            'id': id(self)}
        self.sigMaskImageWidget.emit(ddict)

    def _replaceImageClicked(self):
        imageData = self.getImageData()
        ddict = {
            'event': "replaceImageClicked",
            'image': imageData,
            'title': self.plot.getGraphTitle(),
            'id': id(self)}
        self.sigMaskImageWidget.emit(ddict)

    def _removeImageClicked(self):
        imageData = self.getImageData()
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


if __name__ == "__main__":
    app = qt.QApplication([])
    w = SilxMaskImageWidget()
    w.show()
    w.plot.addImage([[0, 1, 2], [2, 1, -1]])
    app.exec_()
