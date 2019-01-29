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
import tempfile
import shutil
import os
import numpy
import itertools
from contextlib import contextmanager
try:
    from PyMca5.PyMcaPhysics.xrf import McaStackView
except ImportError:
    McaStackView = None
try:
    import h5py
except ImportError:
    h5py = None


class testMcaStackView(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='pymca')

    def tearDown(self):
        shutil.rmtree(self.path)

    @unittest.skipIf(McaStackView is None,
                     'PyMca5.PyMcaPhysics.xrf.McaStackView cannot be imported')
    def testViewUtils(self):
        n = 20
        slices = [slice(None), slice(1, -2),
                  slice(8, 2, -1), slice(0, n, 3),
                  slice(0, n, -2), slice(n-2, 2, -3),
                  slice(n-1, None, -1), slice(None, -2, 3)]
        lst = list(range(n))
        for idx in slices:
            idxn = McaStackView.sliceNormalize(idx, n)
            self.assertEqual(lst[idx], lst[idxn])
        for idx in slices:
            self.assertEqual(McaStackView.sliceLen(idx, n), len(lst[idx]))
        for idx in slices:
            idxi = McaStackView.sliceReverse(idx, n)
            self.assertEqual(lst[idx][::-1], lst[idxi])
        for idx in slices:
            idxc = McaStackView.sliceComplement(idx, n)
            self.assertEqual(list(sorted(lst[idx]+idxc)), lst)
        for idx in slices:
            start, stop, step = idx.indices(n)
            lst1 = list(range(start, stop, int(numpy.sign(step))))
            it = McaStackView.chunkIndexGen(start, stop, step)
            self.assertEqual(len(lst1), sum(ns for it, ns in it))
            it = McaStackView.chunkIndexGen(start, stop, step)
            lst2 = [i for idxc, ns in it for i in lst[idxc]]
            self.assertEqual(lst1, lst2)

    @unittest.skipIf(McaStackView is None,
                     'PyMca5.PyMcaPhysics.xrf.McaStackView cannot be imported')
    def testfullChunkIndex(self):
        for ndim in [2, 3, 4]:
            shape = range(4, 4+ndim)
            data = numpy.zeros(shape, dtype=int)
            ndim = len(shape)
            for nChunksMax in range(numpy.prod(shape[1:])+2):
                for chunkAxes in range(ndim):
                    data[()] = 0
                    chunkIndex, chunkAxes, axesOrder, nChunksMax2 =\
                    McaStackView.fullChunkIndex(shape, nChunksMax, chunkAxes=(chunkAxes,))
                    self.assertTrue(nChunksMax2 <= max(nChunksMax, 1))
                    it = McaStackView.iterChunkIndex(chunkIndex, chunkAxes, axesOrder)
                    for i, (idxChunk, idxShape, nChunks) in enumerate(it, 1):
                        data[idxChunk] += i
                        self.assertEqual(data[idxChunk].shape, idxShape)
                        self.assertTrue(nChunks <= nChunksMax2)
                    # Verify data coverage:
                    self.assertFalse((data == 0).any())
                    # Verify chunk access order:
                    arr = data.transpose(axesOrder[::-1]+chunkAxes).flatten()
                    lst1 = [k for k, g in itertools.groupby(arr)]
                    lst2 = list(range(1, i+1))
                    self.assertEqual(lst1, lst2)

    @unittest.skipIf(McaStackView is None,
                     'PyMca5.PyMcaPhysics.xrf.McaStackView cannot be imported')
    def testFullViewNumpy(self):
        for ndim in [2, 3, 4]:
            shape = range(6, 6+ndim)
            data = numpy.random.uniform(size=shape)
            self._assertFullView(data)
    
    @unittest.skipIf(McaStackView is None,
                     'PyMca5.PyMcaPhysics.xrf.McaStackView cannot be imported')
    @unittest.skipIf(h5py is None,
                     'h5py cannot be imported')
    def testFullViewH5(self):
        for ndim in [2, 3]:
            shape = range(6, 6+ndim)
            data = numpy.random.uniform(size=shape)
            with self.h5Open('testFullView') as f:
                name = 'data{}'.format(ndim)
                f[name] = data
                self._assertFullView(f[name])

    def _assertFullView(self, data):
        mcaSlice = slice(2, -1)
        for nMca in range(numpy.prod(data.shape[1:])+2):
            for mcaAxis in range(data.ndim):
                dataView = McaStackView.FullView(data,
                                                 readonly=False,
                                                 mcaAxis=mcaAxis,
                                                 mcaSlice=mcaSlice,
                                                 nMca=nMca)
                npAdd = numpy.arange(data.size).reshape(data.shape)
                addView = McaStackView.FullView(npAdd,
                                                readonly=True,
                                                mcaAxis=mcaAxis,
                                                mcaSlice=mcaSlice,
                                                nMca=nMca)
                idxFull = dataView.idxFull
                idxFullComplement = dataView.idxFullComplement
                for readonly in [True, False]:
                    dataView.readonly = readonly
                    dataOrg = numpy.copy(data)
                    iters = dataView.items(), addView.items()
                    chunks = McaStackView.izipChunkItems(*iters)
                    for (key, chunk), (addKey, add) in chunks:
                        chunk += add
                    numpy.testing.assert_array_equal(data[idxFullComplement],
                                                     dataOrg[idxFullComplement])
                    if readonly:
                        numpy.testing.assert_array_equal(data[idxFull],
                                                         dataOrg[idxFull])
                    else:
                        numpy.testing.assert_array_equal(data[idxFull],
                                                         dataOrg[idxFull]+npAdd[idxFull])

    @contextmanager
    def h5Open(self, name):
        filename = os.path.join(self.path, name+'.h5')
        with h5py.File(filename, mode='a') as f:
            yield f


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(testMcaStackView))
    else:
        # use a predefined order
        testSuite.addTest(testMcaStackView('testViewUtils'))
        testSuite.addTest(testMcaStackView('testfullChunkIndex'))
        testSuite.addTest(testMcaStackView('testFullViewNumpy'))
        testSuite.addTest(testMcaStackView('testFullViewH5'))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == '__main__':
    test()
