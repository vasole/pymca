import os
import shutil
import tempfile
import unittest
import h5py

from PyMca5.PyMcaIO import HDF5Utils


def _cause_segfault(*args, **kwargs):
    import ctypes

    i = ctypes.c_char(b"a")
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = b"a"
        c += 1


def _safe_cause_segfault(*args, **kwargs):
    return HDF5Utils.run_in_subprocess(_cause_segfault, *args, **kwargs)


class testHDF5Utils(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp(prefix="pymca")

    def tearDown(self):
        shutil.rmtree(self.path)

    def testHdf5GroupKeys(self):
        filename = os.path.join(self.path, "test.h5")
        with h5py.File(filename, "w", track_order=True) as f:
            for i in range(5):
                f[str(i)] = i

        names = list(map(str, range(5)))
        self.assertEqual(HDF5Utils.get_hdf5_group_keys(filename), names)
        self.assertEqual(HDF5Utils.safe_hdf5_group_keys(filename), names)

    def testSegFault(self):
        self.assertEqual(_safe_cause_segfault(default=123), 123)


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testHDF5Utils))
    else:
        # use a predefined order
        testSuite.addTest(testHDF5Utils("testHdf5GroupKeys"))
        testSuite.addTest(testHDF5Utils("testSegFault"))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == "__main__":
    test()
