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
        import time
        t0 = time.time()
        f = ElementTree.parse(filename)
        root = f.getroot()
        node = root.find(".//ClassInstance[@Type='TScanInfo']")
        scanInfoKeys = ["XFirst",
                        "YFirst",
                        "ZFirst",
                        "XLast",
                        "YLast",
                        "ZLast",
                        "MeasNo",
                        "Mapping"]
        scanInfo = {}
        if node:
            for child in node:
                if child.tag in scanInfoKeys:
                    key = child.tag
                    if key == "Mapping":
                        if child.text.upper() == "TRUE":
                            scanInfo[key] = True
                        else:
                            scanInfo[key] = False
                    else:
                        scanInfo[key] = myFloat(child.text)

        for key in scanInfoKeys:
            if key in scanInfo:
                continue
            if key == "Mapping":
                scanInfo[key] = False
            else:
                scanInfo[key] = numpy.nan
        
        t0 = time.time()
        if 0:
            # this works but it is very slow
            child = None
            node = None
            root = None
            f = None
            DataObject.DataObject.__init__(self)
            stack = SpecFileStack.SpecFileStack(filename)
            self.data = stack.data
            self.info = stack.info
            print("ELAPSED = ", time.time() - t0)
            return
        sf = ArtaxFileParser.ArtaxFileParser(filename)
        nscans = sf.scanno()
        scan = sf[0]
        calib0 = scan.header("@CALIB")
        ctime = scan.header("@CTIME")
        positioners = None
        motorNames = sf.allmotors()
        # assuming dictionaries are ordered
        if len(motorNames):
            positioners = {}
            for mne in motorNames:
                positioners[mne] = numpy.zeros((nscans,),
                                               dtype=numpy.float32)
        liveTime = numpy.zeros((nscans,), dtype=numpy.float32)
        spectrum = sf[0].mca(1)
        data = numpy.zeros((1, nscans, len(spectrum)),
                            dtype=numpy.float32)
        for i in range(nscans):
            scan = sf[i]
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
        self.info["McaLiveTime"] = liveTime
        self.info["McaCalib"] = [myFloat(x) for x in calib0[0].split()[1:]]
        self.info["SourceName"] = os.path.abspath(filename)
        print("ELAPSED = ", time.time() - t0)
        if scanInfo.get("Mapping", False):
            _logger.info("Regular Artax Map")
        else:
            _logger.info("Non-regular Artax Map")
        return
        # we are supposed to have a regular map
        # let's figure out the shape?
        if scanInfo["XFirst"] == scanInfo["XLast"] and \
           scanInfo["YFirst"] != scanInfo["YLast"] and \
           scanInfo["ZFirst"] != scanInfo["ZLast"]:
            _logger.info("YZ scan if Y and Z are finite")
            meshType = "YZ"
        elif scanInfo["XFirst"] != scanInfo["XLast"] and \
             scanInfo["YFirst"] != scanInfo["YLast"] and \
             (scanInfo["ZFirst"] == scanInfo["ZLast"]:
            _logger.info("XY scan if X and Y are finite")
            meshType = "XY"            
        elif scanInfo["XFirst"] != scanInfo["XLast"] and \
             scanInfo["YFirst"] == scanInfo["YLast"] and \
             (scanInfo["ZFirst"] != scanInfo["ZLast"]:
            _logger.info("XZ scan if X and Z are finite")
            meshType = "XZ"
        elif scanInfo["XFirst"] != scanInfo["XLast"] and \
             scanInfo["YFirst"] != scanInfo["YLast"] and \
             numpy.isnan(scanInfo["ZFirst"]) and \
             numpy.isnan(scanInfo["ZLast"]):
            _logger.info("XY scan if X and Y are finite")
        else:
            meshType = None

        if meshType:
            for axis in meshType:
                key=axis+"First"
                if not numpy.isfinite(scanInfo[key]):
                    meshType = None
                    break
        # the scan can be in zig-zag
        # it is safer to rely on the scatter view

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

