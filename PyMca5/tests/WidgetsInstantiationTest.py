#/*##########################################################################
# Copyright (C) 2004-2022 European Synchrotron Radiation Facility
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

import logging
import os
import sys
import unittest
import PyMca5.PyMcaGui.PyMcaQt as qt
from PyMca5.PyMcaGui.misc.testutils import TestCaseQt

_logger = logging.getLogger(__name__)


class TestQtWrapper(unittest.TestCase):
    """Minimalistic test to check that Qt has been loaded."""

    def testQObject(self):
        """Test that QObject is there."""
        obj = qt.QObject()
        self.assertTrue(obj is not None)

class TestPlotWidget(TestCaseQt):
    def setUp(self):
        super(TestPlotWidget, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.plotting import PlotWidget
        widget = PlotWidget.PlotWidget()
        widget.show()
        self.qapp.processEvents()

class TestRGBCorrelatorGraph(TestCaseQt):
    def setUp(self):
        super(TestRGBCorrelatorGraph, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.plotting import RGBCorrelatorGraph
        widget = RGBCorrelatorGraph.RGBCorrelatorGraph()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestRGBCorrelatorWidget(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.pymca import RGBCorrelatorWidget
        widget = RGBCorrelatorWidget.RGBCorrelatorWidget()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestXASNormalizationWindow(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xas import XASNormalizationWindow
        spectrum = [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5,  5]
        widget = XASNormalizationWindow.XASNormalizationWindow(None, spectrum)
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestMaskImageWidget(TestCaseQt):
    def setUp(self):
        super(TestMaskImageWidget, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.plotting import MaskImageWidget
        widget = MaskImageWidget.MaskImageWidget()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestPlotWindow(TestCaseQt):
    def setUp(self):
        super(TestPlotWindow, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.plotting import PlotWindow
        widget = PlotWindow.PlotWindow()
        widget.show()
        self.qapp.processEvents()

class TestScanWindow(TestCaseQt):
    def setUp(self):
        super(TestScanWindow, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.pymca import ScanWindow
        widget = ScanWindow.ScanWindow()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestMcaCalWidget(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xrf import McaCalWidget
        widget = McaCalWidget.McaCalWidget(y=[1,2,3,4])
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestMcaWindow(TestCaseQt):
    def setUp(self):
        super(TestMcaWindow, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.pymca import McaWindow
        widget = McaWindow.McaWindow()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestMaterialEditor(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xrf import MaterialEditor
        widget = MaterialEditor.MaterialEditor()
        widget.show()
        self.qapp.processEvents()

class TestFitParam(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xrf import FitParam
        widget = FitParam.FitParamWidget()
        widget.show()
        self.qapp.processEvents()

class TestPeakIdentifier(TestCaseQt):
    def setUp(self):
        super().setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xrf import PeakIdentifier
        widget = PeakIdentifier.PeakIdentifier()
        widget.show()
        self.qapp.processEvents()

class TestMcaAdvancedFit(TestCaseQt):
    def setUp(self):
        super(TestMcaAdvancedFit, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.physics.xrf import McaAdvancedFit
        widget = McaAdvancedFit.McaAdvancedFit()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

class TestPyMcaMain(TestCaseQt):
    def setUp(self):
        super(TestPyMcaMain, self).setUp()

    def testShow(self):
        from PyMca5.PyMcaGui.pymca import PyMcaMain
        widget = PyMcaMain.PyMcaMain()
        widget.show()
        self.qapp.processEvents()
        from PyMca5.PyMcaGui.plotting import PyMcaPrintPreview
        PyMcaPrintPreview.resetSingletonPrintPreview()

def getSuite(auto=True):
    test_suite = unittest.TestSuite()

    with_qt_test = True
    skip_msg = ""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        skip_msg = 'Widgets tests disabled (DISPLAY env. variable not set)'
        with_qt_test = False

    elif os.environ.get('WITH_QT_TEST', 'True') == 'False':
        skip_msg = "Widgets tests skipped by WITH_QT_TEST env var"
        with_qt_test = False

    if not with_qt_test:
        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    skip_msg)
        test_suite.addTest(SkipGUITest())
        return test_suite

    for TestCaseCls in (TestQtWrapper,
                        TestPlotWidget,
                        TestPlotWindow,
                        TestRGBCorrelatorGraph,
                        TestRGBCorrelatorWidget,
                        TestXASNormalizationWindow,
                        TestMaskImageWidget,
                        TestScanWindow,
                        TestMcaCalWidget,
                        TestMcaWindow,
                        TestMaterialEditor,
                        TestFitParam,
                        TestPeakIdentifier,
                        TestMcaAdvancedFit,
                        TestPyMcaMain,
                        ):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestCaseCls))
    return test_suite

def test(auto=False):
    return unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        auto = False
    else:
        auto = True
    app = qt.QApplication([])
    result = test(auto)
    app = None
    sys.exit(not result.wasSuccessful())
