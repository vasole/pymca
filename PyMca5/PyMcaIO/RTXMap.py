#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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
import os
import numpy
import types
import logging

_logger = logging.getLogger(__name__)

#the file format is based on XML
import xml.etree.ElementTree as ElementTree
from PyMca5.PyMcaCore import DataObject
from PyMca5.PyMcaIO import ArtaxFileParser
from PyMca5.PyMcaIO import SpecFileStack
#SOURCE_TYPE = "EdfFileStack"
SOURCE_TYPE ="SpecFileStack"

myFloat = ArtaxFileParser.myFloat

class RTXMap(DataObject.DataObject):
    '''
    Class to read ARTAX .rtx files
    '''
    def __init__(self, filename):
        '''
        Parameters:
        -----------
        filename : str
            Name of the .rtx file.
        '''
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        if 0:
            # this works but it is very slow
            DataObject.DataObject.__init__(self)
            stack = SpecFileStack.SpecFileStack(filename)
            self.data = stack.data
            self.info = stack.info
            return

        sf = ArtaxFileParser.ArtaxFileParser(filename)
        # get some Artax map specific information
        tScanInfo = sf.artaxTScanInfo

        # the rest is specfile like
        nScans = sf.scanno()
        scan = sf[0]
        calib0 = scan.header("@CALIB")
        ctime = scan.header("@CTIME")
        positioners = None
        motorNames = sf.allmotors()
        # assuming dictionaries are ordered
        if len(motorNames):
            positioners = {}
            for mne in motorNames:
                positioners[mne] = numpy.zeros((nScans,),
                                               dtype=numpy.float32)
        if ctime:
            liveTime = numpy.zeros((nScans,), dtype=numpy.float32)
        spectrum = sf[0].mca(1)
        data = numpy.zeros((1, nScans, len(spectrum)),
                            dtype=numpy.float32)
        for i in range(nScans):
            scan = sf[i]
            if ctime:
                ctime = scan.header("@CTIME")[0]
                liveTime[i] = myFloat(ctime.split()[-2])
            if positioners:
                motorPos = scan.allmotorpos()
                for mneIdx in range(len(motorNames)):
                    mne = motorNames[mneIdx]
                    positioners[mne][i] = motorPos[mneIdx]
            data[0, i] = scan.mca(1)
        DataObject.DataObject.__init__(self)
        self.data = data
        if positioners:
            self.info["positioners"] = positioners
        if ctime:
            self.info["McaLiveTime"] = liveTime
        self.info["McaCalib"] = [myFloat(x) for x in calib0[0].split()[1:]]
        self.info["SourceName"] = os.path.abspath(filename)
        
        if tScanInfo.get("Mapping", False):
            _logger.info("Regular Artax Map")
        else:
            _logger.info("Non-regular Artax Map")

        # we are supposed to have a regular map
        # let's figure out the shape?
        if tScanInfo["XFirst"] == tScanInfo["XLast"] and \
           tScanInfo["YFirst"] != tScanInfo["YLast"] and \
           tScanInfo["ZFirst"] != tScanInfo["ZLast"]:
            _logger.info("YZ scan if Y and Z are finite")
            meshType = "YZ"
        elif tScanInfo["XFirst"] != tScanInfo["XLast"] and \
             tScanInfo["YFirst"] != tScanInfo["YLast"] and \
             tScanInfo["ZFirst"] == tScanInfo["ZLast"]:
            _logger.info("XY scan if X and Y are finite")
            meshType = "XY"            
        elif tScanInfo["XFirst"] != tScanInfo["XLast"] and \
             tScanInfo["YFirst"] == tScanInfo["YLast"] and \
             tScanInfo["ZFirst"] != tScanInfo["ZLast"]:
            _logger.info("XZ scan if X and Z are finite")
            meshType = "XZ"
        elif tScanInfo["XFirst"] != tScanInfo["XLast"] and \
             tScanInfo["YFirst"] != tScanInfo["YLast"] and \
             numpy.isnan(tScanInfo["ZFirst"]) and \
             numpy.isnan(tScanInfo["ZLast"]):
            _logger.info("XY scan if X and Y are finite")
            meshType = "XY"
        else:
            meshType = None

        if meshType:
            for axis in meshType:
                key=axis+"First"
                if not numpy.isfinite(tScanInfo[key]):
                    meshType = None
                    break

        if meshType == "XY":
            _logger.info("XY scan")
            x = positioners["x"]
            y = positioners["y"]
            xFirst = tScanInfo["XFirst"]
            xLast = tScanInfo["XLast"]
            yFirst = tScanInfo["YFirst"]
            yLast = tScanInfo["YLast"]
        elif meshType == "XZ":
            _logger.info("XZ scan")
            x = positioners["x"]
            y = positioners["z"]
            xFirst = tScanInfo["XFirst"]
            xLast = tScanInfo["XLast"]
            yFirst = tScanInfo["ZFirst"]
            yLast = tScanInfo["ZLast"]
        elif meshType == "YZ":
            _logger.info("YZ scan")
            x = positioners["y"]
            y = positioners["z"]
            xFirst = tScanInfo["YFirst"]
            xLast = tScanInfo["YLast"]
            yFirst = tScanInfo["ZFirst"]
            yLast = tScanInfo["ZLast"]
        else:
            return

        if len(x) == 1 or len(y) == 1:
            return

        reasonableDeltaX = numpy.abs(x.max() - x.min()) / (1.0 + len(x))
        reasonableDeltaY = numpy.abs(y.max() - y.min()) / (1.0 + len(y))
        if numpy.abs(x[1] - x[0]) > reasonableDeltaX:
            x0 = x[0]
            i = 1
            while (i < len(x)) and (numpy.abs(x[i] - x[0]) > reasonableDeltaX):
                i += 1
            nColumns = i
        else:
            x0 = x[0]
            i = 0
            while (i < len(x)) and (numpy.abs(x[i] - x[0]) < reasonableDeltaX):
                i += 1
            nColumns = i
        # the scan can be in zig-zag
        # it is safer to rely on the scatter view
        if nScans % nColumns == 0:
            nRows = nScans // nColumns
            self.data.shape = nRows, nColumns, -1
            self.info["xScale"] = [xFirst, (xLast - xFirst) / nColumns]
            self.info["yScale"] = [yFirst, (yLast - yFirst) / nRows]

def isRTXMap(filename):
    try:
        if filename[-3:].lower() not in ["rtx", "spx"]:
            return False
        with open(filename, 'rb') as f:
            # expected to read an xml file
            someChar = f.read(20).decode()
        if "<" in someChar and "xml version" in someChar:
            return True
    except:
        pass
    return False

def test(filename):
    print("isRTXMap?  = ", isRTXMap(filename))
    a = RTXMap(filename)
    print("info = ", a.info)
    print("Data shape = ", a.data.shape)

if __name__ == "__main__":
    test(sys.argv[1])
