# -*- coding: utf-8 -*-

# import ######################################################################

import numpy as np
import time
try:
    import unittest
except ImportError:
    import unittest2 as unittest

from PyMca5.PyMcaGraph import ctools


# TODOs:
# test with inf, nan for floats
# benchmark perf

# common ######################################################################

class _TestColormap(unittest.TestCase):
    # Array data types to test
    SIGNED_DTYPES = (np.float16, np.float32, np.float64,
                     np.int8, np.int16,
                     np.int32, np.int64)
    UNSIGNED_DTYPES = (np.uint8, np.uint16,
                       np.uint32, np.uint64)
    DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES

    # Array sizes to test
    SIZES = 2, 10, 256, 1024  # , 2048, 4096

    # Colormaps definitions
    _LUT_RED_256 = np.zeros((256, 4), dtype=np.uint8)
    _LUT_RED_256[:, 0] = np.arange(256, dtype=np.uint8)
    _LUT_RED_256[:, 3] = 255

    _LUT_RGB_3 = np.array(((255, 0, 0, 255),
                           (0, 255, 0, 255),
                           (0, 0, 255, 255)), dtype=np.uint8)

    _LUT_RGB_768 = np.zeros((768, 4), dtype=np.uint8)
    _LUT_RGB_768[0:256, 0] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[256:512, 1] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[512:768, 1] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[:, 3] = 255

    COLORMAPS = {
        'red 256': _LUT_RED_256,
        'rgb 3': _LUT_RGB_3,
        'rgb 768': _LUT_RGB_768,
    }

    @staticmethod
    def _log(*args):
        """Logging used by test for debugging."""
        pass
        # print(args)

    @staticmethod
    def buildControlPixmap(data, colormap, min_, max_):
        """Generate a pixmap used to test C pixmap."""
        min_, max_ = float(min_), float(max_)
        range_ = max_ - min_

        if range_ <= 0:  # Then set pixmap to min color
            indices = np.zeros(data.shape, dtype=np.uint8)
        else:
            clipData = np.clip(data, min_, max_)  # Clip first avoid overflow
            scale = len(colormap) / range_
            normData = scale * np.asarray(clipData - min_, np.float64)

            # Clip again to makes sure <= len(colormap) - 1
            indices = np.asarray(np.clip(normData,
                                         0, len(colormap) - 1),
                                 dtype=np.uint32)

        pixmap = np.take(colormap, indices, axis=0)
        pixmap.shape = data.shape + (4,)
        return np.ascontiguousarray(pixmap)


# TestLinearColormap ##########################################################

class TestLinearColormap(_TestColormap):
    """Test fill pixmap with colormap in C with linear mode.

    Test with different: data types, sizes, colormaps (with different sizes),
    mapping range.
    """

    # Colormap ranges to map
    RANGES = (None, None), (1, 10)

    def _testColormap(self, data, colormap, min_, max_):
        """Test pixmap built with C code against Python control code."""
        startTime = time.time()
        pixmap = ctools.dataToRGBAColormap(data, colormap, min_, max_)
        duration = time.time() - startTime

        if min_ is None:
            min_ = data.min()
        if max_ is None:
            max_ = data.max()
        pixmapControl = self.buildControlPixmap(data, colormap, min_, max_)

        self.assertTrue(np.all(pixmap == pixmapControl))

        return duration

    # @unittest.skip("")
    def testLinear1DData(self):
        """Test pixmap generation for 1D data of different size and types."""
        self._log("testLinear1DData")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(size, dtype=dtype)
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

    # @unittest.skip("")
    def testLinear2DData(self):
        """Test pixmap generation for 2D data of different size and types."""
        self._log("testLinear2DData")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(size * size, dtype=dtype)
                        data = np.nan_to_num(data)
                        data.shape = size, size
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('2D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1, ::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('2D', cmapName, dtype, size, (min_, max_),
                                  duration)

    @unittest.skip("Not for reproductible tests")
    def testLinear1DDataRandom(self):
        """Test pixmap generation for 1D data of different size and types."""
        self._log("testLinear1DDataRandom")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        try:
                            dtypeMax = np.iinfo(dtype).max
                        except ValueError:
                            dtypeMax = np.finfo(dtype).max
                        data = np.asarray(np.random.rand(size) * dtypeMax,
                                          dtype=dtype)
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D Random', cmapName, dtype, size,
                                  (min_, max_), duration)


# TestLog10Colormap ###########################################################

class TestLog10Colormap(_TestColormap):
    """Test fill pixmap with colormap in C with log mode.

    Test with different: data types, sizes, colormaps (with different sizes),
    mapping range.
    """
    # Colormap ranges to map
    RANGES = (None, None), (1, 10)

    @staticmethod
    def _minPos(data):
        posValue = data[np.nonzero(data > 0)]
        if posValue.size != 0:
            return posValue.min()
        else:
            return None  # if no value above 0

    def _testColormap(self, data, colormap, min_, max_):
        """Test pixmap built with C code against Python control code."""
        startTime = time.time()
        pixmap = ctools.dataToRGBAColormap(data, colormap, min_, max_,
                                           isLog10Mapping=True)
        duration = time.time() - startTime

        # Convert to log
        dataLog = np.nan_to_num(np.log10(data, dtype=np.float64))
        if min_ is None:
            min_ = self._minPos(data)
        if max_ is None:
            max_ = data.max()
        min_ = 0. if min_ <= 0. else np.log10(min_, dtype=np.float64)
        max_ = 0. if max_ <= 0. else np.log10(max_, dtype=np.float64)

        pixmapControl = self.buildControlPixmap(dataLog, colormap, min_, max_)

        self.assertTrue(np.all(pixmap == pixmapControl))

        return duration

    # @unittest.skip("")
    def testLog101DDataAllPositive(self):
        """Test pixmap generation for all positive 1D data."""
        self._log("testLog101DData")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(size, dtype=dtype) + 1
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

    # @unittest.skip("")
    def testLog102DDataAllPositive(self):
        """Test pixmap generation for all positive 2D data."""
        self._log("testLog102DData")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(size * size, dtype=dtype) + 1
                        data = np.nan_to_num(data)
                        data.shape = size, size
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('2D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1, ::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('2D', cmapName, dtype, size, (min_, max_),
                                  duration)

    # @unittest.skip("")
    def testLog10AllNegative(self):
        """Test pixmap generation for all negative 1D data."""
        self._log("testLog10AllNegative")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.SIGNED_DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(-size, 0, dtype=dtype)
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

    # @unittest.skip("")
    def testLog10CrossingZero(self):
        """Test pixmap generation for 1D data with negative and zero."""
        self._log("testLog10CrossingZero")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.SIGNED_DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(-size/2, size/2 + 1, dtype=dtype)
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

                        # Reverse order
                        data = data[::-1]
                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D', cmapName, dtype, size, (min_, max_),
                                  duration)

    @unittest.skip("Not for reproductible tests")
    def testLog1DDataRandom(self):
        """Test pixmap generation for 1D data of different size and types."""
        self._log("testLog1DDataRandom")
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        try:
                            dtypeMax = np.iinfo(dtype).max
                            dtypeMin = np.iinfo(dtype).min
                        except ValueError:
                            dtypeMax = np.finfo(dtype).max
                            dtypeMin = np.finfo(dtype).min
                        if dtypeMin < 0:
                            data = np.asarray(-dtypeMax/2. +
                                              np.random.rand(size) * dtypeMax,
                                              dtype=dtype)
                        else:
                            data = np.asarray(np.random.rand(size) * dtypeMax,
                                              dtype=dtype)

                        duration = self._testColormap(data, colormap,
                                                      min_, max_)

                        self._log('1D Random', cmapName, dtype, size,
                                  (min_, max_), duration)


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])
