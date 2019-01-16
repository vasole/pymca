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
        with NexusUtils.nxroot(filename, mode='a') as h5group:
            yield h5group

    def validate_nxroot(self, h5group):
        attrs = ['NX_class', 'creator', 'HDF5_Version', 'file_name',
                 'file_time', 'file_update_time', 'h5py_version']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        self.assertEqual(h5group.attrs['NX_class'], 'NXroot')
        self.assertEqual(h5group.name, '/')

    def validate_nxentry(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['start_time', 'end_time']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXentry')
        self.assertEqual(h5group.parent.name, '/')

    def validate_nxprocess(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['program', 'version', 'configuration', 'date', 'results']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXprocess')
        self.assertEqual(h5group.parent.attrs['NX_class'], 'NXentry')
        self.validate_nxnote(h5group['configuration'])
        self.validate_nxcollection(h5group['results'])

    def validate_nxnote(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = ['date', 'data', 'type']
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXnote')

    def validate_nxcollection(self, h5group):
        attrs = ['NX_class']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        self.assertEqual(h5group.attrs['NX_class'], 'NXcollection')

    def validate_nxdata(self, h5group, axes, signals):
        attrs = ['NX_class', 'axes', 'signal', 'auxiliary_signals']
        self.assertEqual(set(h5group.attrs.keys()), set(attrs))
        files = list(zip(*axes)[0]) + list(zip(*signals)[0])
        self.assertEqual(set(h5group.keys()), set(files))
        self.assertEqual(h5group.attrs['NX_class'], 'NXdata')

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNXroot(self):
        with self.h5open('testNXroot') as h5group:
            self.validate_nxroot(h5group)
    
    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNXentry(self):
        with self.h5open('testNXentry') as h5group:
            entry = NexusUtils.nxentry(h5group, 'entry0001')
            self.assertRaises(RuntimeError, NexusUtils.nxentry,
                              entry, 'entry0002')
            self.validate_nxentry(entry)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    @unittest.skipIf(ConfigDict is None,
                     'PyMca5.PyMcaIO.ConfigDict cannot be imported')
    def testNXprocess(self):
        with self.h5open('testNXprocess') as h5group:
            entry = NexusUtils.nxentry(h5group, 'entry0001')
            configdict = ConfigDict.ConfigDict(initdict={'a': 1, 'b': 2})
            process = NexusUtils.nxprocess(entry, 'process0001', configdict=configdict)
            self.assertRaises(RuntimeError, NexusUtils.nxprocess,
                              h5group, 'process0002', configdict=configdict)
            self.validate_nxprocess(process)

    @unittest.skipIf(NexusUtils is None,
                     'PyMca5.PyMcaIO.NexusUtils cannot be imported')
    def testNXdata(self):
        with self.h5open('testNXentry') as h5group:
            entry = NexusUtils.nxentry(h5group, 'entry0001')
            process = NexusUtils.nxprocess(entry, 'process0001')
            data = NexusUtils.nxdata(process['results'], 'data')
            s = (4, 3, 2)
            axes = [('y', numpy.arange(s[0]), {'units': 'um'}),
                    ('x', numpy.arange(s[1]), {}),
                    ('z', numpy.arange(s[2]), None)]
            signals = [('Fe K', numpy.zeros(s), {'interpretation': 'image'}),
                       ('Ca K', numpy.zeros(s), {}),
                       ('S K', numpy.zeros(s), None)]
            NexusUtils.nxdata_add_axes(data, axes)
            NexusUtils.nxdata_add_signals(data, signals)

            self.validate_nxdata(data, axes, signals)
            signals = NexusUtils.nxdata_get_signals(data)
            self.assertEqual(signals, ['Fe K', 'Ca K', 'S K'])

            NexusUtils.mark_default(data['Ca K'])
            data = entry[NexusUtils.DEFAULT_PLOT_NAME]
            signals = NexusUtils.nxdata_get_signals(data)
            self.assertEqual(signals, ['Ca K', 'Fe K', 'S K'])
            self.assertEqual(data['y'].attrs['units'], 'um')
            self.assertEqual(data['Fe K'].attrs['interpretation'], 'image')


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(testNexusUtils))
    else:
        # use a predefined order
        testSuite.addTest(testNexusUtils('testNXroot'))
        testSuite.addTest(testNexusUtils('testNXentry'))
        testSuite.addTest(testNexusUtils('testNXprocess'))
        testSuite.addTest(testNexusUtils('testNXdata'))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == '__main__':
    test()
