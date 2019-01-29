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
from abc import ABCMeta, abstractmethod, abstractproperty
from six import with_metaclass
import numbers
import itertools

_logger = logging.getLogger(__name__)


def sliceNormalize(slc, n):
    """
    Slice with positive integers

    :param slice slc:
    :param int n:
    :returns slice:
    """
    start, stop, step = slc.indices(n)
    if slc.stop is None and step < 0:
        stop = None
    return slice(start, stop, step)


def sliceLen(slc, n):
    """
    Length after slicing range(n)

    :param slice slc:
    :param int n:
    :returns int:
    """
    start, stop, step = slc.indices(n)
    if step < 0:
        one = -1
    else:
        one = 1
    return max(0, (stop - start + step - one) // step)


def sliceReverse(slc, n):
    """
    Returns slice that yields same items in reversed order

    :param slice slc:
    :param int n:
    :returns slice:
    """
    start, stop, step = slc.indices(n)
    if step < 0:
        one = 1
    else:
        one = -1
    stop = (stop-start+one)//step*step+start
    start += one
    if start == -1:
        start = None
    return slice(stop, start, -step)


def sliceComplement(slc, n):
    """
    Returns indices not in slice

    :param slice slc:
    :param int n:
    :returns list(int):
    """
    lst1 = list(range(n))
    lst2 = lst1[slc]
    return [i for i in lst1 if i not in lst2]


def chunkIndexGen(start, stop, step):
    """
    Index equivalent to list(range(start, stop, sign(step))) but given
    in chunks of "step" items (last chunk may have less items)

    :param start:
    :param stop:
    :param step:
    :returns generator(tuple): generates (index(slice), nElements(int))
    """
    if step is None:
        step = 1
    if not isinstance(start, numbers.Integral):
        raise TypeError('{} object cannot be interpreted as an integer'
                        .format(type(start)))
    if not isinstance(stop, numbers.Integral):
        raise TypeError('{} object cannot be interpreted as an integer'
                        .format(type(stop)))
    if not isinstance(step, numbers.Integral):
        raise TypeError('{} object cannot be interpreted as an integer'
                        .format(type(step)))
    if step < 0:
        func = max
        one = -1
    else:
        func = min
        one = 1
    for a in range(start, stop, step):
        b = func(a+step, stop)
        n = abs(b-a)
        if b == -1:
            b = None
        yield slice(a, b, one), n


def possitive_index(i, n):
    """
    :param int i:
    :param int n:
    """
    if i < 0:
        return i + max((-i)//n, 1)*n
    else:
        return i


def fullChunkIndex(shape, nChunksMax, chunkAxes=None, axesOrder=None, chunkAxesSlice=None, defaultOrder='C'):
    """
    Returns a number of lists (as many as there are dimensions)
    which cartesian product represents all chunk indices

    :param tuple shape: array shape to be sliced
    :param int nChunksMax: maximal number of chunks
    :param tuple(int) chunkAxes: dimensions that define the chunk
    :param tuple(int) axesOrder: order of other dimensions to be sliced
    :param tuple(slice) chunkAxesSlice: slice chunk dimensions
    :param str defaultOrder: 'C' (last index varies the fastest) or 'F' (first index varies the fastest)
    :returns tuple: chunkIndex(list(list(slice,int))), chunkAxes, axesOrder, nBuffer(may be smaller than nChunksMax)
    """
    # Check whether dimensions are compatible
    ndim = len(shape)
    if chunkAxes is None:
        chunkAxes = tuple()
    chunkAxes = tuple(possitive_index(i, ndim) for i in chunkAxes)
    if chunkAxesSlice is None:
        chunkAxesSlice = (slice(None),)*len(chunkAxes)
    else:
        if len(chunkAxes) != len(chunkAxesSlice):
            raise ValueError('Chunk slicing does not correspond with chunk dimensions')
    aAxesOrder = range(ndim)
    if defaultOrder == 'C':
        aAxesOrder = aAxesOrder[::-1]
    aAxesOrder = tuple(i for i in aAxesOrder if i not in chunkAxes)
    if axesOrder is None:
        axesOrder = aAxesOrder
    else:
        axesOrder = tuple(possitive_index(i, ndim) for i in axesOrder)
        if set(axesOrder) != set(aAxesOrder):
            raise ValueError('axesOrder and chunkAxes do not correspond')
    nChunksMax = max(nChunksMax, 1)

    # A number of lists (as many as there are dimensions)
    # which cartesian product represents all chunk indices
    chunkIndex1 = []
    for axis, idx in zip(chunkAxes, chunkAxesSlice):
        nAxis = shape[axis]
        idxAxis = [(idx, sliceLen(idx, nAxis))]
        chunkIndex1.append(idxAxis)
    nItems = 1
    nBuffer = 1
    chunkIndex2 = []
    for axis in axesOrder:
        nAxis = shape[axis]
        nItemsNew = nItems*nAxis
        if nItemsNew <= nChunksMax:
            idxAxis = [(slice(None), nAxis)]
            nBuffer *= nAxis
            #print('Axis {} (size={}): {}x{} chunks'.format(axis, nAxis, 1, nAxis))
        elif nItems > nChunksMax:
            idxAxis = list(chunkIndexGen(0, nAxis, 1))
            #print('Axis {} (size={}): {}x{} chunks'.format(axis, nAxis, len(idxAxis), 1))
        else:
            # Axis will be split in pieces with length "step"
            step = nChunksMax//nItems
            # We have "n" such pieces (last piece can have smaller length)
            n = (nAxis//step) + int(bool(nAxis % step))
            # Maximize the length of the last piece
            # example: nAxis=51 and step=40 -> step = 26
            step = (nAxis//n) + int(bool(nAxis % n))
            nBuffer *= step
            idxAxis = list(chunkIndexGen(0, nAxis, step))
            #print('Axis {} (size={}): {}x{} chunks'.format(axis, nAxis, len(idxAxis), step))
        nItems = nItemsNew
        chunkIndex2.append(idxAxis)
    chunkIndex = chunkIndex1 + chunkIndex2[::-1]
    return chunkIndex, chunkAxes, axesOrder, nBuffer


def iterChunkIndex(chunkIndex, chunkAxes, axesOrder):
    """
    Iterator over the cartesian product of chunkIndex (yields index and shape)

    :param list(list(slice,int)) chunkIndex:
    :param tuple chunkAxes:
    :param tuple axesOrder:
    :returns generator: index(tuple), shape(tuple), nChunks(int)
    """
    axes = chunkAxes+axesOrder[::-1]
    ndim = len(axes)
    idxData = [None]*ndim
    chunkShape = [None]*ndim
    for idxChunk in itertools.product(*chunkIndex):
        nChunks = 1
        for axis, (idx, n) in zip(axes, idxChunk):
            idxData[axis] = idx
            chunkShape[axis] = n
            if axis in axesOrder:
                nChunks *= n
        yield tuple(idxData), tuple(chunkShape), nChunks


def izipChunkItems(*iterables):
    """
    Zip iterators but making sure next is called
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
        ret = tuple(_next(it) for it in iterables)
        if bloop[0]:
            yield ret


def chunks_in_memory(shape, dtype, axis=-1, margin=0.01, maximal=None):
    """
    Number of chunks that fit into memory (with a margin)

    :param tuple shape: nD array
    :param dtype:
    :param axis: axes contibuting to one chunk
    :param margin:
    :param maximal:
    :returns: number of slices that fit in memory
    """
    try:
        from psutil import virtual_memory
    except ImportError:
        try:
            from PyMca5.PyMcaMisc.PhysicalMemory import getAvailablePhysicalMemoryOrNone as getMem
        except ImportError:
            from PyMca5.PyMcaMisc.PhysicalMemory import getPhysicalMemoryOrNone as getMem
        nbytes_mem = getMem()
    else:
        nbytes_mem = virtual_memory().available
    if nbytes_mem is None:
        return maximal
    shape_slice = list(shape)
    if isinstance(axis, (tuple, list)):
        for ax in axis:
            shape_slice.pop(ax)
    else:
        shape_slice.pop(axis)
    if not shape_slice:
        raise ValueError('Required: len(axis)<len(shape)')
    n_items = numpy.prod(shape_slice)
    itemsize = numpy.array(0, dtype=dtype).itemsize
    nbytes_chunk = n_items*itemsize
    n_chunks = int((nbytes_mem*margin)/nbytes_chunk)
    if maximal:
        return max(n_chunks, maximal)
    else:
        return n_chunks


class ChunkedView(with_metaclass(ABCMeta, object)):

    def __init__(self, data, nMca=None, mcaAxis=None, mcaSlice=None,
                 dtype=None, readonly=True):
        """
        :param array data: nD array (numpy.ndarray or h5py.Dataset)
        :param num nMca: maximal number of MCA spectra to be buffered
        :param int mcaAxis:
        :param slice mcaSlice: slice along the MCA axis
        :param dtype:
        :param bool readonly:
        """
        # Buffer shape
        if mcaAxis is None:
            mcaAxis = -1
        n2 = data.shape[mcaAxis]
        if mcaSlice:
            nChan = sliceLen(mcaSlice, n2)
        else:
            nChan = n2
            mcaSlice = slice(None)
        self._mcaSlice = mcaSlice
        self._mcaAxis = mcaAxis
        self._bufferShape = nMca, nChan
        self._buffer = None

        # Buffer dtype
        if dtype is None:
            dtype = data.dtype
        self._dtype = dtype
        self._differentType = data.dtype != dtype

        # Data
        self._data = data
        self.readonly = readonly
        self._isNdarray = isinstance(data, numpy.ndarray)

    @property
    def nChanOrg(self):
        return self._data.shape[self._mcaAxis]

    @property
    def nChan(self):
        return self._bufferShape[1]

    @property
    def idxFull(self):
        idx = [slice(None)] * self._data.ndim
        idx[self._mcaAxis] = self._mcaSlice
        return tuple(idx)

    @property
    def idxFullComplement(self):
        idx = [slice(None)] * self._data.ndim
        idx[self._mcaAxis] = sliceComplement(self._mcaSlice, self.nChanOrg)
        return tuple(idx)

    def _prepareAccess(self):
        _logger.debug('Iterate MCA stack in chunks of {} spectra'
                      .format(self._bufferShape[0]))
        post_copy = not self.readonly
        if self._buffer is None:
            self._buffer = numpy.empty(self._bufferShape, self._dtype)
        return post_copy

    @abstractmethod
    def items(self):
        pass


class MaskedView(ChunkedView):

    def __init__(self, data, mask=None, nMca=None, **kwargs):
        """
        :param array data: nD array (numpy.ndarray or h5py.Dataset)
        :param array mask: shape of data without the MCA dimension
        :param num nMca: number of spectra per chunk
        :param \**kwargs: see ChunkedView
        """
        if mask is None:
            self._indices = None
            self._mask = Ellipsis
            nMca = data.shape[0]*data.shape[1]
        else:
            if isinstance(data, numpy.ndarray):
                # Support multiple advanced indexes
                self._indices = None
                self._mask = mask
                nMca = int(mask.sum())
            else:
                # Does not support multiple advanced indexes
                self._indices = numpy.where(mask)
                self._mask = None
                nMca = len(self._indices[0])
        super(MaskedView, self).__init__(data, nMca=nMca, **kwargs)

    def items(self, keyType='all'):
        post_copy = self._prepareAccess()
        if self._indices is None:
            k, v = self._get_with_mask()
        else:
            k, v = self._get_with_indices()
        if keyType == 'select':
            k = self._mask
        yield k, v
        if post_copy:
            if self._indices is None:
                self._set_with_mask(k, v)
            else:
                self._set_with_indices(k, v)

    def _get_with_mask(self):
        idx = self._mask, self._mcaSlice
        chunk = self._buffer
        chunk[()] = self._data[idx]
        return idx, chunk

    def _set_with_mask(self, idx, chunk):
        self._data[idx] = chunk

    def _get_with_indices(self):
        chunk = self._buffer
        mcaSlice = self._mcaSlice
        idx = self._indices + (mcaSlice,)
        j0keep = -1
        for i, (j0, j1) in enumerate(zip(*self._indices)):
            if j0 != j0keep:
                tmpData = self._data[j0]
                j0keep = j0
            chunk[i] = tmpData[j1, mcaSlice]
        return idx, chunk

    def _set_with_indices(self, idx, chunk):
        lst0, lst1, mcaSlice = idx
        for v, j0, j1 in zip(chunk, lst0, lst1):
            self._data[j0, j1, mcaSlice] = v


class FullView(ChunkedView):

    def __init__(self, data, nMca=None, mcaAxis=None, mcaSlice=None,
                 axesOrder=None, **kwargs):
        """
        :param array data: nD array (numpy.ndarray or h5py.Dataset)
        :param num nMca: number of spectra per chunk
        :param int mcaAxis: MCA channel dimension
        :param tuple axesOrder: order of other dimensions to be sliced (C order by default)
        :param \**kwargs: see ChunkedView
        """
        if mcaAxis is None:
            mcaAxis = -1
        if mcaSlice is None:
            mcaSlice = slice(None)
        chunkIndex, chunkAxes, axesOrder, nMca = fullChunkIndex(data.shape, nMca,
                                                                chunkAxes=(mcaAxis,),
                                                                chunkAxesSlice=(mcaSlice,),
                                                                axesOrder=axesOrder)
        self._chunkIndex = chunkIndex
        self._chunkAxes = chunkAxes
        self._axesOrder = axesOrder
        super(FullView, self).__init__(data, nMca=nMca, mcaAxis=mcaAxis, mcaSlice=mcaSlice, **kwargs)

    def items(self, keyType='all'):
        """Yields (index(tuple), shape(tuple)), chunk(array)
        """
        nChan = self.nChan
        data = self._data
        chunkIndex = self._chunkIndex
        chunkAxes = self._chunkAxes
        axesOrder = self._axesOrder
        it = iterChunkIndex(chunkIndex, chunkAxes, axesOrder)
        axesOrder = tuple(sorted(axesOrder))
        transposeAxes = axesOrder + chunkAxes
        itransposeAxes = tuple(numpy.argsort(transposeAxes).tolist())

        post_copy = self._prepareAccess()
        buffer = self._buffer
        for idxChunk, idxShape, nMca in it:
            value = buffer[:nMca, :]
            value[()] = numpy.transpose(data[idxChunk], transposeAxes)\
                             .reshape(nMca, nChan)
            if keyType == 'select':
                key = tuple(idxChunk[i] for i in axesOrder),\
                      tuple(idxShape[i] for i in axesOrder)
            else:
                key = idxChunk, idxShape
            yield key, value
            if post_copy:
                idxShape = tuple(idxShape[i] for i in transposeAxes)
                data[idxChunk] = numpy.transpose(value.reshape(idxShape),
                                                 itransposeAxes)
