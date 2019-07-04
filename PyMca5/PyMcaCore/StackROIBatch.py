#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
__doc__ = """
Module to calculate a set of ROIs on a stack of data.
"""
import os
import numpy
import time
import logging
import copy
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO.OutputBuffer import OutputBuffer
from PyMca5.PyMcaCore import McaStackView


_logger = logging.getLogger(__name__)


class StackROIBatch(object):

    def __init__(self):
        self.config = ConfigDict.ConfigDict()

    def configure(self, newdict=None):
        if newdict:
            self.config.update(newdict)
        return copy.deepcopy(self.config)

    def setConfiguration(self, configuration):
        self.configure(configuration)

    def getConfiguration(self):
        return self.configure()

    def setConfigurationFile(self, ffile):
        if not os.path.exists(ffile):
            raise IOError("File <%s> does not exists" % ffile)
        configuration = ConfigDict.ConfigDict()
        configuration.read(ffile)
        self.setConfiguration(configuration)

    def batchROIMultipleSpectra(self, x=None, y=None, configuration=None,
                                net=True, xAtMinMax=False, index=None,
                                xLabel=None, outbuffer=None, save=True,
                                **outbufferinitargs):
        """
        This method performs the actual fit. The y keyword is the only mandatory input argument.

        :param x: 1D array containing the x axis (usually the channels) of the spectra.
        :param y: 3D array containing the data, usually [nrows, ncolumns, nchannels]
        :param weight: 0 Means no weight, 1 Use an average weight, 2 Individual weights (slow)
        :param net: 0 Means no subtraction, 1 Calculate
        :param xAtMinMax: if True, calculate X at maximum and minimum Y. Default is false.
        :param index: Index of dimension where to apply the ROIs.
        :param xLabel: Type of ROI to be used.
        :param outbuffer:
        :param save: set to False to postpone saving the in-memory buffers
        :return OutputBuffer:
        """
        data, x, index = self._parseData(x=x, y=y, index=index)
        roiList, config = self._prepareRoiList(configuration=configuration,
                                               xLabel=xLabel)

        # Calculation needs buffer for memory allocation (memory or H5)
        if outbuffer is None:
            outbuffer = OutputBuffer(**outbufferinitargs)
        with outbuffer.Context(save=save):
            outbuffer['configuration'] = config

            self._extractRois(data, x, index,
                              roiList=roiList,
                              roiDict=config["ROI"]["roidict"],
                              outbuffer=outbuffer,
                              xAtMinMax=xAtMinMax)

        return outbuffer

    def _extractRois(self, data, x, index, roiList=None, roiDict=None,
                     outbuffer=None, xAtMinMax=False):
        nRois = len(roiList)
        nRows = data.shape[0]
        nColumns = data.shape[1]
        if xAtMinMax:
            roiShape = (nRois * 4, nRows, nColumns)
            names = [None] * 4 * nRois
        else:
            roiShape = (nRois * 2, nRows, nColumns)
            names = [None] * 2 * nRois

        # Helper variables for roi calculation
        idx = [None] * nRois  # indices along axis=index for each ROI
        xw = [None] * nRois  # x-values for each ROI
        iXMinList = [None] * nRois  # min(xw) for each ROI
        iXMaxList = [None] * nRois  # max(xw) for each ROI
        for j, roi in enumerate(roiList):
            if roi == "ICR":
                xw[j] = x
                idx[j] = numpy.arange(len(x))
                iXMinList[j] = idx[j][0]
                iXMaxList[j] = idx[j][-1]
            else:
                roiFrom = roiDict[roi]["from"]
                roiTo = roiDict[roi]["to"]
                idx[j] = numpy.nonzero((roiFrom <= x) & (x <= roiTo))[0]
                if len(idx[j]):
                    xw[j] = x[idx[j]]
                    iXMinList[j] = numpy.argmin(xw[j])
                    iXMaxList[j] = numpy.argmax(xw[j])
                else:
                    xw[j] = None
            names[j] = "ROI " + roi
            names[j + nRois] = "ROI " + roi + " Net"
            if xAtMinMax:
                roiType = roiDict[roi]["type"]
                names[j + 2 * nRois] = "ROI " + roi + (" %s at Max." % roiType)
                names[j + 3 * nRois] = "ROI " + roi + (" %s at Min." % roiType)

        # Allocate memory for result
        roidtype = numpy.float
        results = outbuffer.allocateMemory('roi',
                                           shape=roiShape,
                                           dtype=roidtype,
                                           labels=names,
                                           dataAttrs=None,
                                           groupAttrs={'default': True},
                                           memtype='ram')

        # Allocate memory of partial result
        jStep = min(1000, data.shape[1])
        chunk = numpy.zeros((jStep, data.shape[index]), roidtype)
        for i in range(0, data.shape[0]):
            jStart = 0
            while jStart < data.shape[1]:
                jEnd = min(jStart + jStep, data.shape[1])
                chunk[:(jEnd - jStart)] = data[i, jStart: jEnd]
                for j, roi in enumerate(roiList):
                    if xw[j] is None:
                        # no points in the ROI            
                        rawSum = 0.0
                        netSum = 0.0
                    else:
                        tmpArray = chunk[:(jEnd - jStart), idx[j]]
                        rawSum = tmpArray.sum(axis=-1, dtype=numpy.float)
                        deltaX = xw[j][iXMaxList[j]] - xw[j][iXMinList[j]]
                        left = tmpArray[:, iXMinList[j]]
                        right = tmpArray[:, iXMaxList[j]]
                        deltaY = right - left
                        if abs(deltaX) > 0.0:
                            slope = deltaY / float(deltaX)
                            background = left * len(xw[j])+ slope * \
                                        (xw[j] - xw[j][iXMinList[j]]).sum(dtype=numpy.float) 
                            netSum = rawSum - background
                        else:
                            netSum = 0.0
                    results[j][i, :(jEnd - jStart)] = rawSum
                    results[j + nRois][i, :(jEnd - jStart)] = netSum
                    if xAtMinMax:
                        if xw[j] is None:
                            # what can be the Min and the Max when there is nothing in the ROI?
                            _logger.warning("No Min. Max for ROI <%s>. Empty ROI" % roiLine)
                        else:
                            # maxImage
                            results[j + 2 * nRois][i, :(jEnd - jStart)] = \
                                    xw[j][numpy.argmax(tmpArray, axis=1)]
                            # minImage
                            results[j + 3 * nRois][i, :(jEnd - jStart)] = \
                                    xw[j][numpy.argmin(tmpArray, axis=1)]
                jStart = jEnd

    def _parseData(self, x=None, y=None, index=None):
        if y is None:
            raise RuntimeError("y keyword argument is mandatory!")
        if hasattr(y, "info") and hasattr(y, "data"):
            data = y.data
            mcaIndex = y.info.get("McaIndex", -1)
        else:
            data = y
            mcaIndex = -1
        if index is None:
            index = mcaIndex
        if index < 0:
            index = len(data.shape) - 1

        #workaround a problem with h5py
        try:
            if index in [0]:
                testException = data[0:1]
            else:
                if len(data.shape) == 2:
                    testException = data[0:1, -1]
                elif len(data.shape) == 3:
                    testException = data[0:1, 0:1, -1]
        except AttributeError:
            txt = "%s" % type(data)
            if 'h5py' in txt:
                _logger.info("Implementing h5py workaround")
                import h5py
                data = h5py.Dataset(data.id)
            else:
                raise

        # only usual spectra case supported
        if index != (len(data.shape) - 1):
            raise IndexError("Only stacks of spectra supported")
        if len(data.shape) != 3:
            txt = "For the time being only "
            txt += "three dimensional arrays supported"
            raise NotImplementedError(txt)
        if len(data.shape) != 3:
            txt = "For the time being only "
            txt += "three dimensional arrays supported"
            raise NotImplementedError(txt)

        # make sure to get x data
        if x is None:
            x = numpy.arange(data.shape[index]).astype(numpy.float32)
        elif x.size != data.shape[index]:
            raise NotImplementedError("All the spectra should share same X axis")
        return data, x, index

    def _prepareRoiList(self, configuration=None, xLabel=None):
        # read the current configuration
        if configuration is not None:
            self.setConfiguration(configuration)
        config = self.getConfiguration()

        # prepare roi list
        roiList0 = config["ROI"]["roilist"]
        if type(roiList0) not in [type([]), type((1,))]:
            roiList0 = [roiList0]

        # operate only on compatible ROIs
        roiList = []
        for roi in roiList0:
            if roi.upper() == "ICR":
                roiList.append(roi)
            roiType = config["ROI"]["roidict"][roi]["type"]
            if xLabel is None:
                roiList.append(roi)
            elif xLabel.lower() == roiType.lower():
                roiList.append(roi)
            else:
                _logger.info("ROI <%s> ignored")
        return roiList, config


def getFileListFromPattern(pattern, begin, end, increment=None):
    if type(begin) == type(1):
        begin = [begin]
    if type(end) == type(1):
        end = [end]
    if len(begin) != len(end):
        raise ValueError(\
            "Begin list and end list do not have same length")
    if increment is None:
        increment = [1] * len(begin)
    elif type(increment) == type(1):
        increment = [increment]
    if len(increment) != len(begin):
        raise ValueError(\
            "Increment list and begin list do not have same length")
    fileList = []
    if len(begin) == 1:
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            fileList.append(pattern % (j))
    elif len(begin) == 2:
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            for k in range(begin[1], end[1] + increment[1], increment[1]):
                fileList.append(pattern % (j, k))
    elif len(begin) == 3:
        raise ValueError("Cannot handle three indices yet.")
        for j in range(begin[0], end[0] + increment[0], increment[0]):
            for k in range(begin[1], end[1] + increment[1], increment[1]):
                for l in range(begin[2], end[2] + increment[2], increment[2]):
                    fileList.append(pattern % (j, k, l))
    else:
        raise ValueError("Cannot handle more than three indices.")
    return fileList


if __name__ == "__main__":
    import glob
    import sys
    from PyMca5.PyMca import EDFStack
    from PyMca5.PyMca import ArraySave
    import getopt
    _logger.setLevel(logging.DEBUG)
    options     = ''
    longoptions = ['cfg=', 'outdir=',
                   'tif=', #'listfile=',
                   'filepattern=', 'begin=', 'end=', 'increment=',
                   "outfileroot="]
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except:
        _logger.error(sys.exc_info()[1])
        sys.exit(1)
    fileRoot = ""
    outputDir = None
    fileindex = 0
    filepattern = None
    begin = None
    end = None
    increment = None
    tif = 0
    for opt, arg in opts:
        if opt in ('--cfg'):
            configurationFile = arg
        elif opt in '--begin':
            if "," in arg:
                begin = [int(x) for x in arg.split(",")]
            else:
                begin = [int(arg)]
        elif opt in '--end':
            if "," in arg:
                end = [int(x) for x in arg.split(",")]
            else:
                end = int(arg)
        elif opt in '--increment':
            if "," in arg:
                increment = [int(x) for x in arg.split(",")]
            else:
                increment = int(arg)
        elif opt in '--filepattern':
            filepattern = arg.replace('"', '')
            filepattern = filepattern.replace("'", "")
        elif opt in '--outdir':
            outputDir = arg
        elif opt in '--outfileroot':
            fileRoot = arg
        elif opt in ['--tif', '--tiff']:
            tif = int(arg)
    if filepattern is not None:
        if (begin is None) or (end is None):
            raise ValueError(
                "A file pattern needs at least a set of begin and end indices")
    if filepattern is not None:
        fileList = getFileListFromPattern(filepattern, begin, end,
                                          increment=increment)
    else:
        fileList = args
    if len(fileList):
        dataStack = EDFStack.EDFStack(fileList, dtype=numpy.float32)
    else:
        print("OPTIONS:", longoptions)
        sys.exit(0)
    if outputDir is None:
        print("RESULTS WILL NOT BE SAVED: No output directory specified")
    t0 = time.time()
    worker = StackROIBatch()
    worker.setConfigurationFile(configurationFile)
    outbuffer = OutputBuffer(outputDir=outputDir,
                             fileEntry=fileRoot,
                             edf=1, tif=tif)
    with outbuffer.saveContext():
        worker.batchROIMultipleSpectra(y=dataStack,
                                       outbuffer=outbuffer)
