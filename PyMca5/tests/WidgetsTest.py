


import unittest
import PyMca5.PyMcaGui.PyMcaQt as qt
from silx.gui.test.utils import TestCaseQt

from PyMca5.PyMcaGui.pymca import ScanWindow
from PyMca5.PyMcaGui.pymca import McaWindow
from PyMca5.PyMcaGui.physics.xrf import McaAdvancedFit


class TestQtWrapper(unittest.TestCase):
    """Minimalistic test to check that Qt has been loaded."""

    def testQObject(self):
        """Test that QObject is there."""
        obj = qt.QObject()
        self.assertTrue(obj is not None)


class TestScanWindow(TestCaseQt):
    def setUp(self):
        super(TestScanWindow, self).setUp()

    def testShow(self):
        widget = ScanWindow.ScanWindow()
        widget.show()
        self.qapp.processEvents()


class TestMcaWindow(TestCaseQt):
    def setUp(self):
        super(TestMcaWindow, self).setUp()

    def testShow(self):
        widget = McaWindow.McaWindow()
        widget.show()
        self.qapp.processEvents()


class TestMcaAdvancedFit(TestCaseQt):
    def setUp(self):
        super(TestMcaAdvancedFit, self).setUp()

    def testShow(self):
        widget = McaAdvancedFit.McaAdvancedFit()
        widget.show()
        self.qapp.processEvents()


def getSuite(auto=True):
    test_suite = unittest.TestSuite()
    for TestCaseCls in (TestQtWrapper, TestScanWindow,
                        TestMcaWindow, TestMcaAdvancedFit):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestCaseCls))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='getSuite')
