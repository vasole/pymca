#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import unittest
import os
import sys
import numpy
import gc
import shutil
import tempfile
try:
    import h5py
    HAS_H5PY = True
except:
    HAS_H5PY = None

DEBUG = 0

class testMcaStackExport(unittest.TestCase):
    def setUp(self):
        """
        Get the data directory
        """
        self._importSuccess = False
        self._outputDir = None
        self._h5File = None
        try:
            from PyMca5 import PyMcaDataDir
            self._importSuccess = True
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None

    def tearDown(self):
        gc.collect()
        if self._h5File is not None:
            fileName = self._h5File
            if os.path.exists(fileName):
                os.remove(fileName)
        if self._outputDir is not None:
            shutil.rmtree(self._outputDir, ignore_errors=True)
            if os.path.exists(self._outputDir):
                raise IOError("Directory <%s> not deleted" % self._outputDir)

    @unittest.skipIf(not HAS_H5PY, "skipped h5py missing")
    def testSingleStackExport(self):
        from PyMca5 import PyMcaDataDir
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaIO import ConfigDict
        from PyMca5.PyMcaCore import DataObject
        from PyMca5.PyMcaCore import StackBase
        from PyMca5.PyMcaCore import McaStackExport
        spe = os.path.join(self.dataDir, "Steel.spe")
        cfg = os.path.join(self.dataDir, "Steel.cfg")
        sf = specfile.Specfile(spe)
        self.assertTrue(len(sf) == 1, "File %s cannot be read" % spe)
        self.assertTrue(sf[0].nbmca() == 1,
                        "Spe file should contain MCA data")

        y = counts = sf[0].mca(1)
        x = channels = numpy.arange(y.size).astype(numpy.float64)
        sf = None
        configuration = ConfigDict.ConfigDict()
        configuration.read(cfg)
        calibration = configuration["detector"]["zero"], \
                      configuration["detector"]["gain"], 0.0
        initialTime = configuration["concentrations"]["time"]
        # create the data
        nRows = 5
        nColumns = 10
        nTimes = 3
        data = numpy.zeros((nRows, nColumns, counts.size), dtype = numpy.float64)
        live_time = numpy.zeros((nRows * nColumns), dtype=numpy.float64)
        xpos = 10 + numpy.zeros((nRows * nColumns), dtype=numpy.float64)
        ypos = 100 + numpy.zeros((nRows * nColumns), dtype=numpy.float64)
        mcaIndex = 0
        for i in range(nRows):
            for j in range(nColumns):
                data[i, j] = counts
                live_time[i * nColumns + j] = initialTime * \
                                              (1 + mcaIndex % nTimes)
                xpos[mcaIndex] += j
                ypos[mcaIndex] += i
                mcaIndex += 1

        # create the stack data object
        stack = DataObject.DataObject()
        stack.data = data
        stack.info = {}
        stack.info["McaCalib"] = calibration
        stack.info["McaLiveTime"] = live_time
        stack.x = [channels]
        stack.info["positioners"] = {"x": xpos,
                                     "y": ypos}

        tmpDir = tempfile.gettempdir()
        self._h5File = os.path.join(tmpDir, "SteelStack.h5")
        if os.path.exists(self._h5File):
            os.remove(self._h5File)
        McaStackExport.exportStackList(stack, self._h5File)

        # read back the stack
        from PyMca5.PyMcaIO import HDF5Stack1D
        stackRead = HDF5Stack1D.HDF5Stack1D([self._h5File],
                                            {"y":"/measurement/detector_00"})

        # let's play
        sb = StackBase.StackBase()
        sb.setStack(stackRead)

        # positioners
        data = stackRead.info["positioners"]["x"]
        self.assertTrue(numpy.allclose(data, xpos),
                "Incorrect readout of x positions")
        data = stackRead.info["positioners"]["y"]
        self.assertTrue(numpy.allclose(data, ypos),
                "Incorrect readout of y positions")

        # calibration and live time
        x, y, legend, info = sb.getStackOriginalCurve()
        readCalib = info["McaCalib"]
        readLiveTime = info["McaLiveTime"]
        self.assertTrue(abs(readCalib[0] - calibration[0]) < 1.0e-10,
                "Calibration zero. Expected %f got %f" % \
                             (calibration[0], readCalib[0]))
        self.assertTrue(abs(readCalib[1] - calibration[1]) < 1.0e-10,
                "Calibration gain. Expected %f got %f" % \
                             (calibration[1], readCalib[0]))
        self.assertTrue(abs(readCalib[2] - calibration[2]) < 1.0e-10,
                "Calibration 2nd order. Expected %f got %f" % \
                             (calibration[2], readCalib[2]))
        self.assertTrue(abs(live_time.sum() - readLiveTime) < 1.0e-5,
                "Incorrect sum of live time data")

    @unittest.skipIf(not HAS_H5PY, "skipped h5py missing")
    def testSingleArrayExport(self):
        from PyMca5.PyMcaCore import StackBase
        from PyMca5.PyMcaCore import McaStackExport
        tmpDir = tempfile.gettempdir()
        self._h5File = os.path.join(tmpDir, "Array.h5")
        data = numpy.arange(3*1024).reshape(3, 1024)
        McaStackExport.exportStackList([data], self._h5File)
        # read back the stack
        from PyMca5.PyMcaIO import HDF5Stack1D
        stackRead = HDF5Stack1D.HDF5Stack1D([self._h5File],
                                            {"y":"/measurement/detector_00"})
        # let's play
        sb = StackBase.StackBase()
        sb.setStack(stackRead)

        # check the data
        self.assertTrue(numpy.allclose(data, stackRead.data),
                "Incorrect data readout")


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(\
            testMcaStackExport))
    else:
        # use a predefined order
        testSuite.addTest(testMcaStackExport("testSingleStackExport"))
        testSuite.addTest(testMcaStackExport("testSingleArrayExport"))
    return testSuite

def test(auto=False):
    return unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        auto = False
    else:
        auto = True
    result = test(auto)
    sys.exit(not result.wasSuccessful())
