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
from contextlib import contextmanager
try:
    import h5py
except:
    h5py = None
try:
    from PyMca5.PyMcaIO import NexusUtils
except ImportError:
    NexusUtils = None
try:
    from PyMca5.PyMcaIO import ConfigDict
except ImportError:
    ConfigDict = None


class testNexusUtils(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='pymca')

    def tearDown(self):
        shutil.rmtree(self.path)

    @contextmanager
    def h5open(self, name):
        filename = os.path.join(self.path, name+'.h5')
        with NexusUtils.nxRoot(filename, mode='a') as h5group:
            yield h5group

    def validateNxRoot(self, h5group):
        attrs = ['NX_class', 'creator', 'HDF5_Version', 'file_name',
                 'file_time', 'file_update_time', 'h5py_version']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        self.assertEqual(h5group.attrs['NX_class'], 'NXroot')
        self.assertEqual(h5group.name, '/')

    def validateNxEntry(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['start_time', 'end_time']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXentry')
        self.assertEqual(h5group.parent.name, '/')

    def validateNxProcess(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['program', 'version', 'configuration', 'date', 'results']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXprocess')
        self.assertEqual(h5group.parent.attrs['NX_class'], 'NXentry')
        self.validateNxNote(h5group['configuration'])
        self.validateNxCollection(h5group['results'])

    def validateNxNote(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['date', 'data', 'type']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXnote')

    def validateNxCollection(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        self.assertEqual(h5group.attrs['NX_class'], 'NXcollection')

    def validateNxData(self, h5group, axes, signals):
        attrs = ['NX_class', 'axes', 'signal', 'auxiliary_signals']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = list(next(iter(zip(*axes)))) + list(next(iter(zip(*signals))))
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXdata')

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxRoot(self):
        with self.h5open('testNxRoot') as h5group:
            self.validateNxRoot(h5group)
    
    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxEntry(self):
        with self.h5open('testNxEntry') as h5group:
            entry = NexusUtils.nxEntry(h5group, 'entry0001')
            self.assertRaises(RuntimeError, NexusUtils.nxEntry,
                              entry, 'entry0002')
            self.validateNxEntry(entry)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    @unittest.skipIf(ConfigDict is None,
                     'PyMca5.PyMcaIO.ConfigDict cannot be imported')
    def testNxProcess(self):
        with self.h5open('testNxProcess') as h5group:
            entry = NexusUtils.nxEntry(h5group, 'entry0001')
            configdict = ConfigDict.ConfigDict(initdict={'a': 1, 'b': 2})
            process = NexusUtils.nxProcess(entry, 'process0001', configdict=configdict)
            self.assertRaises(RuntimeError, NexusUtils.nxProcess,
                              h5group, 'process0002', configdict=configdict)
            self.validateNxProcess(process)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxData(self):
        with self.h5open('testNxEntry') as h5group:
            entry = NexusUtils.nxEntry(h5group, 'entry0001')
            process = NexusUtils.nxProcess(entry, 'process0001')
            data = NexusUtils.nxData(process['results'], 'data')
            s = (4, 3, 2)
            axes = [('y', numpy.arange(s[0]), {'units': 'um'}),
                    ('x', numpy.arange(s[1]), {}),
                    ('z', {'shape': (s[2],), 'dtype': int}, None)]
            signals = [('Fe K', numpy.zeros(s), {'interpretation': 'image'}),
                       ('Ca K', {'data': numpy.zeros(s)}, {}),
                       ('S K', {'shape': s, 'dtype': int}, None)]
            NexusUtils.nxDataAddAxes(data, axes)
            NexusUtils.nxDataAddSignals(data, signals)

            self.validateNxData(data, axes, signals)
            signals = NexusUtils.nxDataGetSignals(data)
            self.assertEqual(signals, ['Fe K', 'Ca K', 'S K'])

            NexusUtils.markDefault(data['Ca K'])
            data = entry[NexusUtils.DEFAULT_PLOT_NAME]
            signals = NexusUtils.nxDataGetSignals(data)
            self.assertEqual(signals, ['Ca K', 'Fe K', 'S K'])
            self.assertEqual(data['y'].attrs['units'], 'um')
            self.assertEqual(data['Fe K'].attrs['interpretation'], 'image')
            for name in signals:
                self.assertEqual(data[name].shape, s)
            for n, name in zip(s, list(next(iter(zip(*axes))))):
                self.assertEqual(data[name].shape, (n,))

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxStringAttribute(self):
        self._checkStringTypes(attribute=True, asarray=False)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxStringDataset(self):
        self._checkStringTypes(attribute=False, asarray=False)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxOldStringAttribute(self):
        self._checkStringTypes(attribute=True, asarray=True)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNxOldStringDataset(self):
        self._checkStringTypes(attribute=False, asarray=True)

    def _checkStringTypes(self, attribute=True, asarray=False):
        try:
            ustring = unicode
        except NameError:
            ustring = str
        with self.h5open('testNxString{:d}'.format(attribute)) as h5group:
            h5group = h5group.create_group('test')
            if attribute:
                outdict = h5group.attrs
            else:
                outdict = h5group
            self._writeStringTypes(outdict, asarray=asarray)
            for k, v in outdict.items():
                if asarray:
                    if 'ext' in k or k == 'mixed(list)':
                        expectedType = bytes
                    else:
                        expectedType = ustring
                else:
                    expectedType = ustring
                if not attribute:
                    v = v[()]
                if isinstance(v, numpy.ndarray):
                    self.assertTrue('list' in k or '1d-array' in k, msg=k)
                    stype = h5py.check_dtype(vlen=v.dtype)
                else:
                    self.assertTrue('scalar' in k or '0d-array' in k, msg=k)
                    stype = type(v)
                msg = '{} type {} instead of {}'.format(k, stype, expectedType)
                self.assertEqual(stype, expectedType, msg=msg)

    def _writeStringTypes(self, outdict, asarray=False):
        abc_ascii = b'abc'
        abc_ext = b'\xe423'
        abc_unicode = u'\u0101bc'
        if asarray:
            scalarFunc = sequenceFunc = NexusUtils.asNxChar
        else:
            scalarFunc = NexusUtils.asNxCharScalar
            sequenceFunc = NexusUtils.asNxCharArray
        outdict['ascii(scalar)'] = scalarFunc(abc_ascii)
        outdict['ext(scalar)'] = scalarFunc(abc_ext)
        outdict['unicode(scalar)'] = scalarFunc(abc_unicode)
        outdict['ascii(list)'] = sequenceFunc([abc_ascii, abc_ascii])
        outdict['ext(list)'] = sequenceFunc([abc_ext, abc_ext])
        outdict['unicode(list)'] = sequenceFunc([abc_unicode, abc_unicode])
        outdict['mixed(list)'] = sequenceFunc([abc_unicode, abc_ascii, abc_ext])
        outdict['ascii(0d-array)'] = sequenceFunc(numpy.array(abc_ascii))
        outdict['ext(0d-array)'] = sequenceFunc(numpy.array(abc_ext))
        outdict['unicode(0d-array)'] = sequenceFunc(numpy.array(abc_unicode))
        outdict['ascii(1d-array)'] = sequenceFunc(numpy.array([abc_ascii, abc_ascii]))
        outdict['ext(1d-array)'] = sequenceFunc(numpy.array([abc_ext, abc_ext]))
        outdict['unicode(1d-array)'] = sequenceFunc(numpy.array([abc_unicode, abc_unicode]))
        outdict['mixed(1d-array)'] = sequenceFunc(numpy.array([abc_unicode, abc_ascii]))


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(testNexusUtils))
    else:
        # use a predefined order
        testSuite.addTest(testNexusUtils('testNxStringAttribute'))
        testSuite.addTest(testNexusUtils('testNxStringDataset'))
        testSuite.addTest(testNexusUtils('testNxOldStringAttribute'))
        testSuite.addTest(testNexusUtils('testNxOldStringDataset'))
        testSuite.addTest(testNexusUtils('testNxRoot'))
        testSuite.addTest(testNexusUtils('testNxEntry'))
        testSuite.addTest(testNexusUtils('testNxProcess'))
        testSuite.addTest(testNexusUtils('testNxData'))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == '__main__':
    test()
