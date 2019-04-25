#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import unittest
import sys
import os
import numpy
from random import randint
import tempfile
import shutil
from PyMca5.tests import XrfData
import PyMca5.PyMcaGui.PyMcaQt as qt
from PyMca5.PyMcaGui.misc.testutils import TestCaseQt
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class testPyMcaBatch(TestCaseQt):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='pymca')
        super(testPyMcaBatch, self).setUp()

    def tearDown(self):
        shutil.rmtree(self.path)
        super(testPyMcaBatch, self).tearDown()

    def _checkForUnreleasedWidgets(self):
        # A simple import already creates widgets:
        #from PyMca5.PyMcaGui.pymca import PyMcaBatch
        pass

    def testCommand(self):
        from PyMca5.PyMcaGui.pymca import PyMcaBatch
        cmd = PyMcaBatch.Command('command')
        cmd.addOption('a', value=1, format='"{:04d}"')
        cmd.b = 2
        parts = {'command', '--a="0001"', '--b=2'}
        self.assertEqual(parts, set(str(cmd).split(' ')))
        cmd.a = 10
        parts = {'command', '--a="0010"', '--b=2'}
        self.assertEqual(parts, set(str(cmd).split(' ')))
        cmd['a'] = 5
        cmd['c'] = 'test'
        parts = {'command', '--a=5', '--b=2', '--c=test'}
        self.assertEqual(parts, set(str(cmd).split(' ')))
        dict1 = cmd.getAllOptions()
        dict2 = {'a': 5, 'b': 2, 'c': 'test'}
        self.assertEqual(dict1, dict2)
        dict1 = cmd.getAllOptionsBut('a', 'b')
        dict2 = {'c': 'test'}
        self.assertEqual(dict1, dict2)
        dict1 = cmd.getOptions('a', 'b')
        dict2 = {'a': 5, 'b': 2}
        self.assertEqual(dict1, dict2)

    def testSubCommands(self):
        """
        Check multi processing slicing of 2D maps for different
        map dimensions, number of files and number of processes.
        Assumes a file contains one or more rows (i.e. columns are
        never split over files).
        """
        for i in range(10000):
            nRows = randint(1, 10)
            nColumns = randint(1, 10)
            nRowsPerFile = randint(1, 5)
            if (nRows % nRowsPerFile) == 0:
                nFiles = nRows//nRowsPerFile
            else:
                nFiles = nRows
            nBatches = randint(1, 30)
            chunks = (i % 2) == 0
            self.assertSubCommands(nRows, nColumns, nFiles, nBatches, chunks)

    def assertSubCommands(self, nRows, nColumns, nFiles, nBatches, bchunks):
        """
        Checks whether each spectrum is processed exactly once and chunk
        indices are sequential.
        """
        from PyMca5.PyMcaGui.pymca import PyMcaBatch
        msg = '\nnRows={}, nColumns={}, nFiles={}, nBatches={}'\
              .format(nRows, nColumns, nFiles, nBatches)
        coverage = numpy.zeros((nRows, nColumns), dtype=int)
        nRowsPerFile = nRows//nFiles
        chunks = []

        def runProcess(cmd):
            iFiles = list(range(cmd.filebeginoffset, nFiles-cmd.fileendoffset, cmd.filestep))
            iCols = list(range(cmd.mcaoffset, nColumns, cmd.mcastep))
            iRows = list(range(nRowsPerFile))
            for ifile in iFiles:
                for irow in iRows:
                    for icol in iCols:
                        coverage[ifile*nRowsPerFile+irow, icol] += 1
            self.assertTrue(bool(iFiles), msg + '\n no files processed')
            if bool(iCols):
                chunks.append(cmd.chunk)

        cmd = PyMcaBatch.Command()
        PyMcaBatch.SubCommands(cmd, nFiles, nBatches, runProcess, chunks=bchunks)

        # Check: each spectrum is processed exactly once
        self.assertTrue((coverage == 1).all(), msg + '\n {}'.format(coverage))

        # Check: chunk indices sequential
        self.assertTrue((numpy.diff(sorted(chunks)) == 1).all(), msg)
        self.assertTrue(len(chunks) <= nBatches, msg)

    def testFastFitEdfMap(self):
        info = self._generateData(fast=True, typ='edf')
        self._fitXrfMap(info, fast=True)

    def testSlowFitEdfMap(self):
        info = self._generateData(typ='edf')
        self._fitXrfMap(info)

    def testSlowMultiFitEdfMap(self):
        info = self._generateData(typ='edf')
        result1 = self._fitXrfMap(info, nBatches=4, outputdir='fitresults1')
        result2 = self._fitXrfMap(info, nBatches=1, outputdir='fitresults2')
        # TODO: rtol is pretty bad!!!
        self._compareBatchFitResult(result1, result2, rtol=1e-3)

    @unittest.skipIf(not HAS_H5PY, "skipped h5py missing")
    def testFastFitHdf5Map(self):
        info = self._generateData(fast=True, typ='hdf5')
        self._fitXrfMap(info, fast=True)

    @unittest.skipIf(not HAS_H5PY, "skipped h5py missing")
    def testSlowFitHdf5Map(self):
        info = self._generateData(typ='hdf5')
        result1 = self._fitXrfMap(info, legacy=False, outputdir='fitresults1')
        result2 = self._fitXrfMap(info, legacy=True, outputdir='fitresults2')
        self._compareBatchFitResult(result1, result2, rtol=1e-5)

    @unittest.skipIf(not HAS_H5PY, "skipped h5py missing")
    def testSlowMultiFitHdf5Map(self):
        info = self._generateData(typ='hdf5')
        result1 = self._fitXrfMap(info, nBatches=4, outputdir='fitresults1')
        result2 = self._fitXrfMap(info, nBatches=1, outputdir='fitresults2')
        # TODO: rtol is pretty bad!!!
        self._compareBatchFitResult(result1, result2, rtol=1e-3)

    def _fitXrfMap(self, info, fast=False, nBatches=0,
                   outputdir='fitresults', **kwargs):
        outputdir = os.path.join(self.path, outputdir)
        if fast:
            imageFile = self._fastFitXrfMap(info, outputdir, **kwargs)
        elif nBatches > 0:
            imageFile = self._slowMultiFitXrfMap(info, outputdir,
                                                 nBatches, **kwargs)
        else:
            imageFile = self._slowFitXrfMap(info, outputdir, **kwargs)
        labels, scanData = self._parseDatResults(imageFile)
        self._verifyBatchFitResult(labels, scanData, info['liveTimeCorrection'],
                                   multiprocessing=nBatches > 1)
        return labels, scanData

    def _fastFitXrfMap(self, info, outputdir):
        from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
        from PyMca5.PyMcaPhysics.xrf.XRFBatchFitOutput import OutputBuffer
        kwargs = {'y': info['input'],
                  'livetime': info['liveTime'],
                  'weight': 0,
                  'configuration': info['configuration'],
                  'concentrations': True,
                  'refit': 1}
        outbuffer = OutputBuffer(outputDir=outputdir,
                                 dat=True, edf=False, h5=False,
                                 diagnostics=True)
        batch = FastXRFLinearFit()
        with outbuffer.saveContext():
            outbuffer = batch.fitMultipleSpectra(outbuffer=outbuffer, **kwargs)
        return outbuffer.filename('.dat')
    
    def _slowFitXrfMap(self, info, outputdir, legacy=False, nBatches=1):
        if legacy:
            from PyMca5.PyMcaPhysics.xrf.LegacyMcaAdvancedFitBatch import McaAdvancedFitBatch
            os.mkdir(outputdir)
        else:
            from PyMca5.PyMcaPhysics.xrf.McaAdvancedFitBatch import McaAdvancedFitBatch
        kwargs = {'filelist': info['input'],
                  'outputdir': outputdir,
                  'concentrations': True,
                  'selection': info['selection'],
                  'quiet': True}
        if not legacy:
            kwargs['dat'] = True
            kwargs['edf'] = False
            kwargs['h5'] = False
            kwargs['diagnostics'] = True
        batch = McaAdvancedFitBatch(info['cfgname'], **kwargs)
        batch.processList()
        if legacy:
            imageFile = os.path.join(outputdir, "IMAGES", "xrfmap.dat")
        else:
            imageFile = batch.outbuffer.filename('.dat')
        return imageFile
    
    def _slowMultiFitXrfMap(self, info, outputdir, nBatches, legacy=False):
        from PyMca5.PyMcaGui.pymca import PyMcaBatch
        from PyMca5.PyMcaPhysics.xrf import McaAdvancedFitBatch 
        os.mkdir(outputdir)
        kwargs = {'actions': True,
                  'showresult': False,
                  'filelist': info['input'],
                  'config': info['cfgname'],
                  'outputdir': outputdir,
                  'selection': info['selection'],
                  'nproc': nBatches}
        if not legacy:
            kwargs['dat'] = True
            kwargs['edf'] = False
            kwargs['h5'] = False
            kwargs['diagnostics'] = True
            kwargs['concentrations'] = True
        else:
            raise NotImplementedError
        widget = PyMcaBatch.McaBatchGUI(**kwargs)
        #widget.show()
        self.qapp.processEvents()
        widget.start()

        # Wait until result is created
        from time import sleep
        rootname = McaAdvancedFitBatch.getRootName(kwargs['filelist'])
        imageFile = os.path.join(outputdir, rootname, rootname+'.dat')
        while not os.path.exists(imageFile):
            sleep(1)
            self.qapp.processEvents()
        
        # Wait until result is finished writting
        bytes0 = os.stat(imageFile).st_size
        while True:
            sleep(1)
            bytes1 = os.stat(imageFile).st_size
            if bytes1 == bytes0:
                break
            else:
                bytes0 = bytes1
        
        widget.close()
        self.qapp.processEvents()
        #self.qapp.exec_()
        return imageFile

    def _generateData(self, fast=False, typ='hdf5'):
        # Generate data (in memory + save in requested format)
        nDet = 1  # TODO: currently only works with 1 detector
        nRows = 5
        nColumns = 10
        nTimes = 3
        filename = os.path.join(self.path, 'xrfmap')
        if typ == 'edf':
            genFunc = XrfData.generateEdfMap
            filename += '.edf'
        elif typ == 'specmesh':
            genFunc = XrfData.generateSpecMesh
            filename += '.dat'
        elif typ == 'hdf5':
            genFunc = XrfData.generateHdf5Map
            filename += '.h5'
        else:
            raise ValueError('Unknown data type {} for XRF map'.format(repr(typ)))
        # TODO: cannot provide live time when fitting .edf list of files
        liveTimeIsProvided = fast or typ != 'edf'

        def modfunc(configuration):
            configuration["concentrations"]["usematrix"] = 0
            configuration["concentrations"]["useautotime"] = int(liveTimeIsProvided)
            if fast:
                configuration['fit']['stripalgorithm'] = 1
        info = genFunc(filename, nDet=nDet, nRows=nRows,
                       nColumns=nColumns, nTimes=nTimes,
                       modfunc=modfunc)
        
        # Concentrations are multiplied by this factor to
        # normalize live time to preset time
        # TODO: currently only works with 1 detector
        info['liveTime'] = info['liveTime'][0, ...]
        if liveTimeIsProvided:
            info['liveTimeCorrection'] = float(info['presetTime'])/info['liveTime']
        else:
            info['liveTimeCorrection'] = numpy.ones_like(info['liveTime'])

        # Batch fit input (list of strings or stack object)
        filelist = info['filelist']
        if typ == 'edf':
            if fast:
                from PyMca5.PyMca import EDFStack
                info['input'] = EDFStack.EDFStack(filelist, dtype=numpy.float32)
            else:
                info['input'] = filelist
                info['selection'] = None
        elif typ == 'specmesh':
            raise NotImplementedError
        elif typ == 'hdf5':
            datasets = ['/xrf/mca{:02d}/data'.format(k) for k in range(nDet)]
            
            if fast:
                from PyMca5.PyMcaIO import HDF5Stack1D
                info['selection'] = selection = {'y': datasets[0]}
                info['input'] = HDF5Stack1D.HDF5Stack1D(filelist, selection)
            else:
                info['selection'] = {'x':[], 'm':[], 'y': [datasets[0]]}
                info['input'] = filelist
        
        # Batch fit configuration
        info['cfgname'] = os.path.join(self.path, 'xrfmap.cfg')
        return info

    def _compareBatchFitResult(self, result1, result2, rtol=0, atol=0):
        labels1, scanData1 = result1
        labels2, scanData2 = result2
        self.assertEqual(set(labels1), set(labels2))
        for label, data in zip(labels1, scanData1):
            idx = labels2.index(label)
            numpy.testing.assert_allclose(data, scanData2[idx, :],
                                          err_msg=label, rtol=rtol,
                                          atol=atol)

    def _convertLegacyLabel(self, label):
        if label.endswith('-mass-fraction'):
            label = label.replace('-mass-fraction', '')
            label = 'w({})'.format(label)
        label = label.replace('-', '_')
        return label

    def _parseDatResults(self, filename):
        """
        :param str filename:
        :returns tuple: list(nparams), ndarray(nparams, nrows, ncolumns)
        """
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        self.assertTrue(os.path.isfile(filename),
                        "Batch fit result file <%s> not present" % filename)
        sf = specfile.Specfile(filename)
        labels = sf[0].alllabels()
        scanData = sf[0].data()
        sf = None
        nParams, nPoints = scanData.shape
        idxRow = labels.index('row')
        idxColumn = labels.index('column')
        nRows = int(numpy.round(max(scanData[idxRow, :]))) + 1
        nColumns = int(numpy.round(max(scanData[idxColumn, :]))) + 1
        colfast = scanData[idxRow, 0] == scanData[idxRow, 1]
        if colfast:
            order = 'C'
        else:
            order = 'F'
        scanData = scanData.reshape((nParams, nRows, nColumns), order=order)
        labels = list(map(self._convertLegacyLabel, labels))
        return labels, scanData

    def _verifyBatchFitResult(self, labels, paramStack, liveTimeCorrection,
                              multiprocessing=False):
        """
        :param list labels: parameter names
        :param ndarray paramStack: nParams x nRows x nColumns
        :param ndarray liveTimeCorrection: nRows x nColumns
        :param bool multiprocessing: merged result of multiple processes
        """
        nParams, nRows, nColumns = paramStack.shape
        self.assertTrue((nRows, nColumns), liveTimeCorrection.shape)
        for label, param in zip(labels, paramStack):
            if label in ["Point", "row", "column"]:
                continue
            if label.startswith("w("):
                # Same spectrum in each pixel but live time changes.
                # This means peak areas are the same but concentrations
                # are corrected for this live time.
                param = param/liveTimeCorrection
                if multiprocessing:
                    # TODO: pretty bad!!!
                    rtol = 1e-3
                else:
                    rtol = 1e-5
            else:
                # Same spectrum in each pixel
                if multiprocessing:
                    rtol = 1e-4
                else:
                    rtol = 0
            numpy.testing.assert_allclose(param, param[0, 0], err_msg=label,
                                          rtol=rtol, atol=0)


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testPyMcaBatch))
    else:
        # use a predefined order
        testSuite.addTest(testPyMcaBatch("testCommand"))
        testSuite.addTest(testPyMcaBatch("testSubCommands"))
        testSuite.addTest(testPyMcaBatch("testFastFitHdf5Map"))
        testSuite.addTest(testPyMcaBatch("testFastFitEdfMap"))
        testSuite.addTest(testPyMcaBatch("testSlowFitHdf5Map"))
        testSuite.addTest(testPyMcaBatch("testSlowFitEdfMap"))
        testSuite.addTest(testPyMcaBatch("testSlowMultiFitEdfMap"))
        testSuite.addTest(testPyMcaBatch("testSlowMultiFitHdf5Map"))
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
