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

import numpy
import logging

_logger = logging.getLogger(__name__)


def sliceLen(slc, n):
    start, stop, step = slc.indices(n)
    if step < 0:
        one = -1
    else:
        one = 1
    return max(0, (stop - start + step - one) // step)


def chunkIndices(start, stop=None, step=None):
    """
    :param start:
    :param stop:
    :param step:
    :returns list: list of (slice,n)
    """
    if stop is None:
        start, stop, step = 0, start, 1
    elif step is None:
        step = 1
    if step < 0:
        func = max
        one = -1
    else:
        func = min
        one = 1
    for a in range(start, stop, step):
        b = func(a+step, stop)
        yield slice(a, b), abs(b-a)


def izipChunkIter(*iters):
    """
    Like Python 3's zip but making sure next is called
    on all items when StopIteration occurs
    """
    bloop = [True]  # because of python 2
    #bloop = True
    def _next(it):
        #nonlocal bloop
        try:
            return next(it)
        except StopIteration:
            bloop[0] = False
            #bloop = False
            return None
    while bloop[0]:
        ret = tuple(_next(it) for it in iters)
        if bloop[0]:
            yield ret


class McaStackView(object):

    def __init__(self, data, nbuffer=None, mcaslice=None, dtype=None, modify=False):
        """
        :param array data: 3D array (n0, n1, n2)
        :param num nbuffer: number of elements from (n0, n1) in buffer
        :param slice mcaslice: slice MCA axis=2
        :param dtype:
        :param bool modify: modify original on access
        """
        # Buffer shape
        n2 = data.shape[-1]
        if mcaslice:
            nChan = sliceLen(mcaslice, n2)
        else:
            nChan = n2
            mcaslice = slice(None)
        self._mcaslice = mcaslice
        self._buffershape = nbuffer, nChan
        self._buffer = None

        # Buffer dtype
        if dtype is None:
            dtype = data.dtype
        self._dtype = dtype
        self._change_type = data.dtype != dtype

        # Data
        self._data = data
        self._modify = modify
        self._isndarray = isinstance(data, numpy.ndarray)

    def _prepareAccess(self, copy=True):
        _logger.debug('Iterate MCA stack in chunks of {} spectra'
                      .format(self._buffershape[0]))
        needs_copy = copy or self._change_type
        yields_copy = needs_copy or not self._isndarray
        post_copy = yields_copy and self._modify
        if yields_copy and self._buffer is None:
            self._buffer = numpy.empty(self._buffershape, self._dtype)
        return needs_copy, yields_copy, post_copy

    def chunks(self, copy=True):
        raise NotImplementedError


class McaStackMaskView(McaStackView):

    def __init__(self, data, mask=None, **kwargs):
        """
        3D stack (n0, n1, n2) with mask (n0, n1) and slice n2

        :param array data: numpy array or h5py dataset
        :param array mask: shape = (n0, n1)
        :param \**kwargs: see McaStackView
        """
        if mask is None:
            self._indices = None
            self._mask = Ellipsis
            nbuffer = data.shape[0]*data.shape[1]
        else:
            if isinstance(data, numpy.ndarray):
                # Support multiple advanced indexes
                self._indices = None
                self._mask = mask
                nbuffer = int(mask.sum())
            else:
                # Does not support multiple advanced indexes
                self._indices = numpy.where(mask)
                self._mask = None
                nbuffer = len(self._indices[0])
        super(McaStackMaskView, self).__init__(data, nbuffer=nbuffer, **kwargs)

    def chunks(self, copy=True):
        needs_copy, _, post_copy = self._prepareAccess(copy=copy)
        if self._indices is None:
            ret = self._get_with_mask(copy=needs_copy)
        else:
            ret = self._get_with_indices()
        yield ret
        if post_copy:
            if self._indices is None:
                self._set_with_mask(*ret)
            else:
                self._set_with_indices(*ret)

    def _get_with_mask(self, copy=True):
        idx = self._mask, self._mcaslice
        if copy:
            chunk = self._buffer
            chunk[()] = self._data[idx]
        else:
            chunk = self._data[idx]
        return idx, chunk

    def _set_with_mask(self, idx, chunk):
        self._data[idx] = chunk

    def _get_with_indices(self):
        chunk = self._buffer
        mcaslice = self._mcaslice
        idx = self._indices + (mcaslice,)
        j0keep = -1
        for i, (j0, j1) in enumerate(zip(*self._indices)):
            if j0 != j0keep:
                tmpData = self._data[j0]
                j0keep = j0
            chunk[i] = tmpData[j1, mcaslice]
        return idx, chunk

    def _set_with_indices(self, idx, chunk):
        lst0, lst1, mcaslice = idx
        for v, j0, j1 in zip(chunk, lst0, lst1):
            self._data[j0, j1, mcaslice] = v


class McaStackChunkView(McaStackView):

    def __init__(self, data, nmca=None, **kwargs):
        """
        3D stack (n0, n1, n2) to be iterated over in nmca spectra

        :param array data: numpy array or h5py dataset
        :param num nmca: number of spectra in one chunk
        :param \**kwargs: see McaStackView
        """
        # Outer loop (non-chunked dimension)
        n0, n1, n2 = data.shape
        if nmca is None:
            nmca = min(n0, n1)
        if abs(n0-nmca) < abs(n1-nmca):
            self._loopout = list(range(n1))
            self._axes = 1,0
            n = n0
        else:
            self._loopout = list(range(n0))
            self._axes = 0,1
            n = n1
        
        # Inner loop (chunked dimension)
        nbuffer = min(nmca, n)
        nchunks = n//nbuffer + int(bool(n % nbuffer))
        nbuffer = n//nchunks + int(bool(n % nchunks))
        self._loopin = list(chunkIndices(0, n, nbuffer))

        super(McaStackChunkView, self).__init__(data, nbuffer=nbuffer, **kwargs)

    def chunks(self, copy=True):
        _, yields_copy, post_copy = self._prepareAccess(copy=copy)
        idx_data = [slice(None), slice(None), self._mcaslice]
        i,j = self._axes
        buffer = self._buffer
        for idxout in self._loopout:
            idx_data[i] = idxout
            for idxin, n in self._loopin:
                idx_data[j] = idxin
                idx = tuple(idx_data)
                if yields_copy:
                    chunk = buffer[:n, :]
                    chunk[()] = self._data[idx]
                else:
                    chunk = self._data[idx]
                yield idx, chunk
                if post_copy:
                    self._data[idx] = chunk
