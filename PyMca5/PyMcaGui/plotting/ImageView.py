# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
The classes in this module are deprecated. Use
:class:`silx.gui.plot.ImageView` and
:class:`silx.gui.plot.ImageViewMainWindow` instead.

This module can be used to open an EDF or TIFF file
from the shell command line.
To view an image file:
``python -m PyMca5.PyMcaGui.plotting.ImageView <file to open>``
To get help:
``python -m PyMca5.PyMcaGui.plotting.ImageView -h``
"""


# import ######################################################################
import logging
import traceback
try:
    from .. import PyMcaQt as qt
except ImportError:
    from PyMca5.PyMcaGui import PyMcaQt as qt

from silx.gui.plot.ImageView import ImageView as SilxImageView
from silx.gui.plot.ImageView import ImageViewMainWindow as SilxImageViewMainWindow


_logger = logging.getLogger(__name__)
_logger.warning("%s is deprecated, you are advised to use "
                "silx.gui.plot.ImageView instead",
                __name__)
for line in traceback.format_stack(limit=3):
    _logger.warning(line.rstrip())


class ImageView(SilxImageView):
    def __init__(self, parent=None, windowFlags=None, backend=None):
        """

        :param parent:
        :param windowFlags: windowFlags (e.g. qt.Qt.Widget, qt.Qt.Window...)
            If None, the silx default behavior is used: behave as a widget if
            parent is not None, else behave as a Window.
        :param backend:
        """
        super(ImageView, self).__init__(parent=parent, backend=backend)

        # SilxImageView does not have a windowFlags parameter.
        # A silx PlotWidget behaves as a Widget if parent is not None,
        # else it behaves as a QMainWindow.
        if windowFlags is not None:
            self.setWindowFlags(windowFlags)


class ImageViewMainWindow(SilxImageViewMainWindow):
    def __init__(self, parent=None, windowFlags=qt.Qt.Widget, backend=None):
        super(ImageViewMainWindow, self).__init__(parent, backend)
        if windowFlags is not None:
            self.setWindowFlags(windowFlags)



# main ########################################################################

if __name__ == "__main__":
    import argparse
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
    parser.add_argument(
        '-o', '--origin', nargs=2,
        type=float, default=(0., 0.),
        help="""Coordinates of the origin of the image: (x, y).
        Default: 0., 0.""")
    parser.add_argument(
        '-s', '--scale', nargs=2,
        type=float, default=(1., 1.),
        help="""Scale factors applied to the image: (sx, sy).
        Default: 1., 1.""")
    parser.add_argument('filename', help='EDF filename of the image to open')
    args = parser.parse_args()

    # Open the input file
    if not os.path.isfile(args.filename):
        raise RuntimeError('No input file: %s' % args.filename)

    edfFile = EdfFile(args.filename)
    nbFrames = edfFile.GetNumImages()
    if nbFrames == 0:
        raise RuntimeError(
            'Cannot read image(s) from file: %s' % args.filename)

    # Set-up Qt application and main window
    app = qt.QApplication([])

    mainWindow = ImageViewMainWindow(backend=args.backend)
    mainWindow.setImage(edfFile.GetData(0),
                        origin=args.origin,
                        scale=args.scale)

    if nbFrames > 1:
        # Add a toolbar for multi-frame EDF support
        multiFrameToolbar = qt.QToolBar('Multi-frame')
        multiFrameToolbar.addWidget(qt.QLabel(
            'Frame [0-%d]:' % (nbFrames - 1)))

        spinBox = qt.QSpinBox()
        spinBox.setRange(0, nbFrames-1)

        def updateImage(index):
            mainWindow.setImage(edfFile.GetData(index),
                                origin=args.origin,
                                scale=args.scale,
                                reset=False)
        spinBox.valueChanged[int].connect(updateImage)
        multiFrameToolbar.addWidget(spinBox)

        mainWindow.addToolBar(multiFrameToolbar)

    mainWindow.show()

    sys.exit(app.exec_())
