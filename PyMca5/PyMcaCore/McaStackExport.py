#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import numpy
import posixpath
import h5py
import logging
_logger = logging.getLogger(__name__)

try:
    from PyMca5.PyMcaIO import NexusUtils
    HAS_NEXUS_UTILS = True
except:
    # this should only happen if somebody uses this module out of the distribution
    HAS_NEXUS_UTILS = False
    _logger.info("PyMca5.PyMcaIO.NexusUtils could not be imported")

def exportStackList(stackList, filename, channels=None, calibration=None):
    if hasattr(stackList, "data") and hasattr(stackList, "info"):
        stackList = [stackList]
    if isinstance(filename, h5py.File):
        h5 = filename
        _exportStackList(stackList,
                         h5,
                         channels=channels,
                         calibration=calibration)
    else:
        h5 = h5py.File(filename, "w-")
        try:
            if HAS_NEXUS_UTILS:
                NexusUtils.nxRootInit(h5)
            _exportStackList(stackList,
                             h5,
                             channels=channels,
                             calibration=calibration)
        finally:
            h5.close()

def _exportStackList(stackList, h5, path=None, channels=None, calibration=None):
    if path is None:
        # initialize the entry
        entryName = "stack"
    else:
        entryName = path
    if entryName not in h5 and HAS_NEXUS_UTILS:
        NexusUtils.nxEntryInit(h5, entryName)
    entry = h5.require_group(entryName)
    att = "NX_class"
    if att not in entry.attrs:
        entry.attrs[att] = u"NXentry"
    instrumentName = "instrument"
    instrument = entry.require_group(instrumentName)
    if att not in instrument.attrs:
        instrument.attrs[att] = u"NXinstrument"

    # save all the stacks
    dataTargets = []
    i = 0
    for stack in stackList:
        detectorName = "detector_%02d" % i
        detector = instrument.require_group(detectorName)
        if att not in detector.attrs:
            detector.attrs[att] = u"NXdetector"
        detectorPath = posixpath.join("/",
                                      entryName,
                                      instrumentName,
                                      detectorName)
        exportStack(stack,
                    h5,
                    detectorPath,
                    channels=channels,
                    calibration=calibration)
        dataPath = posixpath.join(detectorPath, "data")
        dataTargets.append(dataPath)
        i += 1

    # create NXdata
    measurement = entry.require_group("measurement")
    if att not in measurement.attrs:
        measurement.attrs[att] = u"NXdata"
    att = "default"
    if "default" not in entry.attrs:
        entry.attrs["default"] = u"measurement"
    i = 0
    auxiliary = []
    for target in dataTargets:
        name = posixpath.basename(posixpath.dirname(target))
        measurement[name] = h5py.SoftLink(target)
        if i == 0:
            measurement.attrs["signal"] = name
        else:
            auxiliary.append(name)
    if len(auxiliary):
        if sys.version_info < (3,):
            dtype = h5py.special_dtype(vlen=unicode)
        else:
            dtype = h5py.special_dtype(vlen=str)
        measurement.attrs["auxiliary_signals"] = numpy.array(auxiliary,
                                                             dtype=dtype)
    h5.flush()
    return entryName

def exportStack(stack, h5object, path, channels=None, calibration=None):
    """
    Exports the stack to the given HDF5 file object and path
    """
    h5g = h5object.require_group(path)

    # destination should be an NXdetector group
    att = "NX_class"
    if att not in h5g.attrs:
        h5g.attrs[att] = u"NXdetector"
    elif h5g.attrs[att] != u"NXdetector":
        _logger.warning("Invalid destination NXclass %s" % h5g.attrs[att])

    # put the data themselves
    if hasattr(stack, "data") and hasattr(stack, "info"):
        data = stack.data
    elif hasattr(stack, "shape") and hasattr(stack, "dtype"):
        # numpy like object received
        data = stack
    else:
        raise TypeError("Unrecognized stack object received")

    dataset = h5g.require_dataset("data",
                                  shape=data.shape,
                                  dtype=data.dtype)
    dataset[:] = data

    # support a simple array of data
    if hasattr(stack, "info"):
        info = stack.info
    else:
        info = {}

    # provide a hint for the data type
    mcaIndex = info.get('McaIndex', -1)
    if mcaIndex < 0:
        mcaIndex = len(data.shape) + mcaIndex
    if len(data.shape) > 1:
        if mcaIndex == 0:
            if len(data.shape) == 3:
                dataset.attrs["interpretation"] = u"image"
        else:
            dataset.attrs["interpretation"] = u"spectrum"

    # get the calibration
    if calibration is None:
        calibration = info.get('McaCalib', [0.0, 1.0, 0.0])
    h5g["calibration"] = numpy.array(calibration, copy=False)

    # get the time
    for key in ["McaLiveTime", "live_time"]:
        if key in info and info[key] is not None:
            # TODO: live time can actually be elapsed time!!!
            h5g["live_time"] =  numpy.array(info[key], copy=False)

    for key in ["preset_time", "elapsed_time"]:
        if key in info and info[key] is not None:
            h5g[key] =  numpy.array(info[key], copy=False)

    # get the channels
    if channels is None:
        if hasattr(stack, "x"):
            if hasattr(stack.x, "__len__"):
                if len(stack.x):
                    channels = stack.x[0]

    if channels is not None:
        h5g["channels"] = numpy.array(channels, copy=False)

    # the positioners
    posKey = "positioners"
    if posKey in info and info[posKey] is not None:
        posGroupPath = posixpath.join(posixpath.dirname(path), posKey)
        posGroup = h5object.require_group(posGroupPath)
        att = "NX_class"
        if att not in posGroup.attrs:
            posGroup.attrs[att] = u"NXcollection"
        for key in info[posKey]:
            if key not in posGroup:
                posGroup[key] = numpy.array(info[posKey][key], copy=False)
