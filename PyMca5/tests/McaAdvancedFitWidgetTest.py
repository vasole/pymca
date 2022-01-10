#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2019-2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
import os
import sys
import time
import unittest
import PyMca5.PyMcaGui.PyMcaQt as qt
from PyMca5.PyMcaGui.misc.testutils import TestCaseQt

if os.environ.get('WITH_OPENGL_TEST', 'True') in ['False', '0', 0, 'FALSE']:
    OPENGL = False
else:
    try:
        import OpenGL
        OPENGL = True
    except:
        OPENGL = False

# Debian packaging for armhf does not support OpenGL extensions in the
# implementation of the packaging machine. Disable the OpenGL tests.
ARM32 = False
if OPENGL:
    import platform
    if platform.machine().startswith("arm"):
        if platform.architecture()[0].startswith('32'):
            ARM32 = True

try:
    import silx.gui
    SILX = True
except ImportError:
    SILX = False

class TestMcaAdvancedFitWidget(TestCaseQt):
    def setUp(self):
        super(TestMcaAdvancedFitWidget, self).setUp()

    def _workOnBackend(self, backend):
        from PyMca5.PyMcaGui.physics.xrf import McaAdvancedFit
        from PyMca5.PyMcaGraph.Plot import Plot
        Plot.defaultBackend = backend
        widget = McaAdvancedFit.McaAdvancedFit()
        widget.show()
        self.qapp.processEvents()

        # read the data
        from PyMca5 import PyMcaDataDir
        dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
        from PyMca5.PyMcaPhysics.xrf import ConcentrationsTool
        from PyMca5.PyMcaIO import ConfigDict

        dataFile = os.path.join(dataDir, "Steel.spe")
        self.assertTrue(os.path.isfile(dataFile),
                        "File %s is not an actual file" % dataFile)
        sf = specfile.Specfile(dataFile)
        self.assertTrue(len(sf) == 1, "File %s cannot be read" % dataFile)
        self.assertTrue(sf[0].nbmca() == 1,
                        "Spe file should contain MCA data")
        y = counts = sf[0].mca(1)
        x = channels = numpy.arange(y.size).astype(numpy.float64)
        sf = None
        widget.setData(x, y)
        self.qapp.processEvents()

        # read the fit configuration
        configFile = os.path.join(dataDir, "Steel.cfg")
        self.assertTrue(os.path.isfile(configFile),
                        "File %s is not an actual file" % configFile)
        configuration = ConfigDict.ConfigDict()
        configuration.read(configFile)
        widget.configure(configuration)
        self.qapp.processEvents()
        time.sleep(1)

        # switch log axis
        #   widget.graphWindow.yLogButton clicked
        isLogy0 = widget.graphWindow.isYAxisLogarithmic()
        self.mouseClick(widget.graphWindow.yLogButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        isLogy1 = widget.graphWindow.isYAxisLogarithmic()
        self.assertTrue(isLogy0 != isLogy1,
                        "Y scale not toggled!")
        time.sleep(1)
        # reset zoom
        widget.graphWindow.resetZoom()

        # swith energy axis:
        #   widget.graphWindow._energyIconSignal
        #   widget.graphWindow.energyButton clicked
        label0 = widget.graphWindow.getGraphXLabel()
        self.mouseClick(widget.graphWindow.energyButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        label1 = widget.graphWindow.getGraphXLabel()
        self.assertTrue(label0 != label1,
                        "Energy scale not toggled!")
        self.assertTrue(label0.lower() in ["channel", "energy"],
                        "Unexpected plot X label <%s>" % label0)
        self.assertTrue(label1.lower() in ["channel", "energy"],
                        "Unexpected plot X label <%s>" % label0)

        # reset zoom
        widget.graphWindow.resetZoom()

        # fit:
        #   callback widget.fit
        #   widget.graphWindow.fitButton clicked
        #   widget.graphWindow._fitIconSignal
        self.assertTrue(not widget._fitdone(),
                        "Bad fit widget state. Fit should not be finished")
        self.mouseClick(widget.graphWindow.fitButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        self.assertTrue(widget._fitdone(),
                        "Bad fit widget state. Fit should be finished")

        # toggle matrix spectrum
        curveList0 = widget.graphWindow.getAllCurves(just_legend=True)
        self.mouseClick(widget.matrixSpectrumButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        curveList1 = widget.graphWindow.getAllCurves(just_legend=True)
        self.assertTrue(abs(len(curveList0) - len(curveList1)) == 1,
                        "Matrix spectrum not working!!")

        # toggle peaks
        curveList0 = widget.graphWindow.getAllCurves(just_legend=True)
        for curve in ["Data", "Fit", "Continuum", "Pile-up"]:
            self.assertTrue(curve in curveList0,
                            "Curve <%s> expected but not found" % curve)
        self.mouseClick(widget.peaksSpectrumButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        curveList1 = widget.graphWindow.getAllCurves(just_legend=True)
        self.assertTrue(len(curveList0) != len(curveList1),
                        "Peaks spectrum not working!!")

        time.sleep(1)
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

    @unittest.skipUnless(SILX, "silx not installed")
    def testInteractionSilxMpl(self, backend="silx-mpl"):
        return self._workOnBackend(backend)

    @unittest.skipIf(ARM32, "OpenGL tests disabled on ARM-32 bit")
    @unittest.skipUnless(SILX and OPENGL, "silx and/or OpenGL disabled")
    def testInteractionSilxGL(self, backend="silx-gl"):
        return self._workOnBackend(backend)

    def testInteractionMpl(self, backend="mpl"):
        return self._workOnBackend(backend)

    @unittest.skipIf(ARM32, "OpenGL tests disabled on ARM-32 bit")
    @unittest.skipUnless(OPENGL, "OpenGL not imported or disabled")
    @unittest.skipUnless(SILX, "silx not installed")
    def testInteractionOpenGL(self, backend="gl"):
        return self._workOnBackend(backend)

def getSuite(auto=True):
    with_qt_test = True
    skip_msg = ""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        skip_msg = 'Widgets tests disabled (DISPLAY env. variable not set)'
        with_qt_test = False

    elif os.environ.get('WITH_QT_TEST', 'True') in ['False', 'FALSE', 0, "0"]:
        skip_msg = "Widgets tests skipped by WITH_QT_TEST env var"
        with_qt_test = False

    testSuite = unittest.TestSuite()

    if not with_qt_test:
        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    skip_msg)
        testSuite.addTest(SkipGUITest())
        return testSuite

    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase( \
            TestMcaAdvancedFitWidget))
    else:
        # use a predefined order
        testSuite.addTest(TestMcaAdvancedFitWidget("testInteractionMpl"))
        testSuite.addTest(TestMcaAdvancedFitWidget("testInteractionSilxMpl"))
        testSuite.addTest(TestMcaAdvancedFitWidget("testInteractionOpenGL"))
        testSuite.addTest(TestMcaAdvancedFitWidget("testInteractionSilxGL"))
    return testSuite

def test(auto=False):
    return unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        auto = False
    else:
        auto = True
    if os.environ.get('WITH_QT_TEST', 'True') not in ['False', 'FALSE', 0, "0"]:
        app = qt.QApplication([])
    result = test(auto)
    app = None
    sys.exit(not result.wasSuccessful())
