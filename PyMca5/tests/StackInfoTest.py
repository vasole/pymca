#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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

try:
    import h5py
    HAS_H5PY = True
except:
    HAS_H5PY = None
if sys.version_info < (3,):
    from StringIO import StringIO
else:
    from io import StringIO

class testStackInfo(unittest.TestCase):
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
        if self._outputDir is not None:
            shutil.rmtree(self._outputDir, ignore_errors=True)
            if os.path.exists(self._outputDir):
                raise IOError("Directory <%s> not deleted" % self._outputDir)
        if self._h5File is not None:
            fileName = self._h5File
            if os.path.exists(fileName):
                os.remove(fileName)

    def testDataDirectoryPresence(self):
        self.assertTrue(self._importSuccess,
                        'Unsuccessful PyMca5.PyMcaDataDir import')
        self.assertTrue(self.dataDir is not None,
                        'Unassigned PyMca5.PyMcaDataDir.PYMCA_DATA_DIR')
        self.assertTrue(os.path.exists(self.dataDir),
                        'Directory "%s" does not exist' % self.dataDir)
        self.assertTrue(os.path.isdir(self.dataDir),
                        '"%s" expected to be a directory' % self.dataDir)

    def testDataFilePresence(self):
        for fileName in ["Steel.spe", "Steel.cfg"]:
            dataFile = os.path.join(self.dataDir, fileName)
            self.assertTrue(os.path.exists(dataFile),
                            "File %s does not exists" % dataFile)
            self.assertTrue(os.path.isfile(dataFile),
                            "File %s is not an actual file" % dataFile)

    def testBatchFitHdf5Stack(self):
        import tempfile
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        from PyMca5.PyMcaIO import ConfigDict
        from PyMca5.PyMcaIO import HDF5Stack1D
        from PyMca5.PyMcaPhysics.xrf import McaAdvancedFitBatch
        spe = os.path.join(self.dataDir, "Steel.spe")
        cfg = os.path.join(self.dataDir, "Steel.cfg")
        sf = specfile.Specfile(spe)
        self.assertTrue(len(sf) == 1, "File %s cannot be read" % spe)
        self.assertTrue(sf[0].nbmca() == 1,
                        "Spe file should contain MCA data")
        y = counts = sf[0].mca(1)
        x = channels = numpy.arange(y.size).astype(numpy.float)
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
        data = numpy.zeros((nRows, nColumns, counts.size), dtype = numpy.float)
        live_time = numpy.zeros((nRows * nColumns), dtype=numpy.float)

        mcaIndex = 0
        for i in range(nRows):
            for j in range(nColumns):
                data[i, j] = counts
                live_time[i * nColumns + j] = initialTime * \
                                              (1 + mcaIndex % nTimes)
                mcaIndex += 1
        self._h5File = os.path.join(tempfile.gettempdir(), "Steel.h5")

        # write the stack to an HDF5 file
        if os.path.exists(self._h5File):
            os.remove(self._h5File)
        h5 = h5py.File(self._h5File, "w")
        h5["/entry/instrument/detector/calibration"] = calibration
        h5["/entry/instrument/detector/channels"] = channels
        h5["/entry/instrument/detector/data"] = data
        h5["/entry/instrument/detector/live_time"] = live_time

        # add nexus conventions
        h5["/entry"].attrs["NX_class"] = u"NXentry"
        h5["/entry/instrument"].attrs["NX_class"] = u"NXinstrument"
        h5["/entry/instrument/detector/"].attrs["NX_class"] = u"NXdetector"
        h5["/entry/instrument/detector/data"].attrs["interpretation"] = \
                                                              u"spectrum"
        h5.flush()
        h5.close()
        h5 = None

        # check that the data can be read as a stack
        fileList = [self._h5File]
        selection = {"y":"/instrument/detector/data"}
        stack = HDF5Stack1D.HDF5Stack1D(fileList, selection)
        info = stack.info
        for key in ["McaCalib", "McaLiveTime"]:
            self.assertTrue(key in info,
                        "Key <%s>  not present but it should be there")

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
        self.assertTrue(live_time.size == readLiveTime.size,
                        "Incorrect size of live time data")
        self.assertTrue(numpy.allclose(live_time, readLiveTime),
                        "Incorrect live time read")
        self.assertTrue(numpy.allclose(stack.x, channels),
                        "Incorrect channels read")
        self.assertTrue(numpy.allclose(stack.data, data),
                        "Incorrect data read")
        # perform the batch fit
        self._outputDir = os.path.join(tempfile.gettempdir(), "SteelTestDir")
        if not os.path.exists(self._outputDir):
            os.mkdir(self._outputDir)
        cfgFile = os.path.join(tempfile.gettempdir(), "SteelNew.cfg")
        if os.path.exists(cfgFile):
            os.remove(cfgFile)
        # we need to make sure we use fundamental parameters and
        # the time read from the file
        configuration["concentrations"]["usematrix"] = 0
        configuration["concentrations"]["useautotime"] = 1
        configuration.write(cfgFile)

        batch = McaAdvancedFitBatch.McaAdvancedFitBatch(cfgFile,
                                        filelist=[self._h5File],
                                        outputdir=self._outputDir,
                                        concentrations=True,
                                        selection=selection)
        batch.processList()

        # recover the results
        imageFile = os.path.join(self._outputDir, "IMAGES", "Steel.dat")
        self.assertTrue(os.path.isfile(imageFile),
                "Batch fit result file <%s> not present" % imageFile)
        sf = specfile.Specfile(imageFile)
        labels = sf[0].alllabels()
        scanData = sf[0].data()
        sf = None
        self.assertTrue(scanData.shape[-1] == (nRows * nColumns),
           "Expected %d values got %d" % (nRows * nColumns, scanData.shape[-1]))

        referenceResult = {}
        for point in range(scanData.shape[-1]):
            for label in labels:
                idx = labels.index(label)
                if label in ["Point", "row", "column"]:
                    continue
                elif point == 0:
                    referenceResult[label] = scanData[idx, point]
                elif label.endswith("-mass-fraction"):
                    #print("label = ", label)
                    #print("reference = ", referenceResult[label])
                    #print("current = ", scanData[idx, point])
                    reference = referenceResult[label]
                    current = scanData[idx, point]
                    #print("ratio = ", current / reference)
                    #print("time ratio = ", readLiveTime[point] / readLiveTime[0])
                    if point % nTimes:
                        self.assertTrue(reference != current,
                            "Incorrect concentration for point %d" % point)
                        corrected = current * \
                                    (readLiveTime[point] / readLiveTime[0])
                        delta = 100 * abs((reference - corrected) / reference)
                        self.assertTrue(delta < 0.01,
                             "Incorrect concentration(t) for point %d" % point)
                    else:
                        self.assertTrue(reference == current,
                            "Incorrect concentration for point %d" % point)
                elif label not in ["Point", "row", "column"]:
                    reference = referenceResult[label]
                    current = scanData[idx, point]
                    self.assertTrue( reference == current,
                                    "Incorrect value for point %d" % point)

        # Batch fitting went well
        # Test the fast XRF
        from PyMca5.PyMcaPhysics.xrf import FastXRFLinearFit
        ffit = FastXRFLinearFit.FastXRFLinearFit()
        configuration["concentrations"]["usematrix"] = 0
        configuration["concentrations"]["useautotime"] = 1
        configuration['fit']['stripalgorithm'] = 1
        outputDict = ffit.fitMultipleSpectra(y=stack,
                                             weight=0,
                                             configuration=configuration,
                                             concentrations=True,
                                             refit=1)
        print("keys = ", outputDict.keys())
        names = outputDict["names"]
        parameters = outputDict["parameters"]
        uncertainties = outputDict["uncertainties"]
        concentrations = outputDict["concentrations"]
        cCounter = 0
        for i in range(len(names)):
            name = names[i]
            if name.startswith("C(") and name.endswith(")"):
                # it is a concentrations parameter
                cCounter += 1
                continue
            else:
                print(name, parameters[i][0, 0])
                delta = (parameters[i] - parameters[i][0, 0])
                self.assertTrue(delta.max() == 0,
                    "Different fit value for parameter %s delta %f" % \
                                (name, delta.max()))
                self.assertTrue(delta.min() == 0,
                    "Different fit value for parameter %s delta %f" % \
                                (name, delta.min()))
                delta = (uncertainties[i] - uncertainties[i][0, 0])
                self.assertTrue(delta.max() == 0,
                    "Different sigma value for parameter %s delta %f" % \
                                (name, delta.max()))
                self.assertTrue(delta.min() == 0,
                    "Different sigma value for parameter %s delta %f" % \
                                (name, delta.min()))
        outputDict = ffit.fitMultipleSpectra(y=stack,
                                             weight=0,
                                             configuration=configuration,
                                             concentrations=True,
                                             refit=0)
        print("keys = ", outputDict.keys())
        names = outputDict["names"]
        parameters = outputDict["parameters"]
        uncertainties = outputDict["uncertainties"]
        concentrations = outputDict["concentrations"]
        cCounter = 0
        for i in range(len(names)):
            name = names[i]
            if name.startswith("C(") and name.endswith(")"):
                # it is a concentrations parameter
                cCounter += 1
                continue
            else:
                print(name, parameters[i][0, 0])
                delta = (parameters[i] - parameters[i][0, 0])
                self.assertTrue(delta.max() == 0,
                    "Different fit value for parameter %s delta %f" % \
                                (name, delta.max()))
                self.assertTrue(delta.min() == 0,
                    "Different fit value for parameter %s delta %f" % \
                                (name, delta.min()))
                delta = (uncertainties[i] - uncertainties[i][0, 0])
                self.assertTrue(delta.max() == 0,
                    "Different sigma value for parameter %s delta %f" % \
                                (name, delta.max()))
                self.assertTrue(delta.min() == 0,
                    "Different sigma value for parameter %s delta %f" % \
                                (name, delta.min()))


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testStackInfo))
    else:
        # use a predefined order
        testSuite.addTest(testStackInfo("testDataDirectoryPresence"))
        testSuite.addTest(testStackInfo("testDataFilePresence"))
        testSuite.addTest(testStackInfo("testBatchFitHdf5Stack"))
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
