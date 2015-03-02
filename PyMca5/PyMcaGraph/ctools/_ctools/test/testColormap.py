# -*- coding: utf-8 -*-

# import ######################################################################

import numpy as np
import time
try:
    import unittest
except ImportError:
    import unittest2 as unittest

from PyMca5.PyMcaGraph import ctools


# TestColormap ################################################################

class TestColormap(unittest.TestCase):
    """Test fill pixmap with colormap in C.

    Test with different: data types, sizes, colormaps (with different sizes),
    mapping range. 
    """
    # Array data types to test
    DTYPES = (np.float32, np.float64,
              np.int8, np.uint8,
              np.int16, np.uint16,
              np.int32, np.uint32,
              np.int64, np.uint64)

    # Array sizes to test
    SIZES = 2, 10 , 256, 1024 # , 2048, 4096

    # Colormaps definitions
    _LUT_RED_256 = np.zeros((256, 4), dtype=np.uint8)
    _LUT_RED_256[:, 0] = np.arange(256, dtype=np.uint8)
    _LUT_RED_256[:,3] = 255

    _LUT_RGB_3 = np.array(((255, 0, 0, 255),
                           (0, 255, 0, 255),
                           (0, 0, 255,255)), dtype=np.uint8)

    _LUT_RGB_768 = np.zeros((768, 4), dtype=np.uint8)
    _LUT_RGB_768[0:256, 0] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[256:512, 1] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[512:768, 1] = np.arange(256, dtype=np.uint8)
    _LUT_RGB_768[:,3] = 255

    # Colormaps to test
    COLORMAPS = {
        'red 256': _LUT_RED_256,
        'rgb 3': _LUT_RGB_3,
        'rgb 768': _LUT_RGB_768,
    }

    # Colormap ranges to map
    RANGES = (None, None), (1, 10)


    def _log(self, *args):
        """Logging used by test for debugging."""
        pass
        # print(args)

    def _testColormap(self, data, colormap, min_, max_):
        """Test pixmap built with C code against Python control code."""
        startTime = time.time()
        pixmap = ctools.dataToRGBAColormap(data, colormap, min_, max_)
        duration = time.time() - startTime

        pixmapControl = self._controlPixmap(data, colormap, min_, max_)
        self.assertTrue(np.all(pixmap == pixmapControl))

        return duration

    def _controlPixmap(self, data, colormap, min_, max_):
        """Generate a pixmap used to test C pixmap."""
        if min_ is None:
            min_ = data.min()
        if max_ is None:
            max_ = data.max()
        min_, max_ = float(min_), float(max_)
        range_ = max_ - min_

        if range_ <= 0:  # Then set pixmap to min color
            indices = np.zeros(data.shape, dtype=np.uint8)
        else:
            clipData = np.clip(data, min_, max_) # Clip first avoid overflow
            normData = (np.asarray(clipData, np.float64) - min_) / range_

            # Clip again to makes sure <= len(colormap) - 1
            indices = np.asarray(np.clip(len(colormap) * normData,
                                         0, len(colormap) - 1),
                                 dtype=np.uint32)

        pixmap = np.take(colormap, indices, axis=0)
        pixmap.shape = data.shape + (4,)
        return np.ascontiguousarray(pixmap)

    # @unittest.skip("")
    def test1DData(self):
        """Test pixmap generation for 1D data of different size and types."""
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
    def test2DData(self):
        """Test pixmap generation for 2D data of different size and types."""
        for cmapName, colormap in self.COLORMAPS.items():
            for size in self.SIZES:
                for dtype in self.DTYPES:
                    for min_, max_ in self.RANGES:
                        # Increasing values
                        data = np.arange(size * size, dtype=dtype)
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
    def test1DDataRandom(self):
        """Test pixmap generation for 1D data of different size and types."""
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


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])
