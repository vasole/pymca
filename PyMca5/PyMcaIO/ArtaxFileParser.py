#!/usr/bin/env python
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
import os
import numpy
import types
import logging

_logger = logging.getLogger(__name__)

#spx and rtx file formats based on XML
import xml.etree.ElementTree as ElementTree
from PyMca5.PyMcaIO import SpecFileAbstractClass

def myFloat(x):
    try:
        return float(x)
    except ValueError:
        if ',' in x:
            try:
                return float(x.replace(',','.'))
            except:
                return float(x)
        elif '.' in x:
            try:
                return float(x.replace('.',','))
            except:
                return float(x)
        else:
            raise

class ArtaxFileParser(object):
    '''
    Class to read ARTAX .spx or .rtx files
    '''
    def __init__(self, filename):
        '''
        Parameters:
        -----------
        filename : str
            Name of the .spx or .rtx file.
        '''

        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        if not isArtaxFile(filename):
            raise IOError("This does not look as an Artax file")

        f = ElementTree.parse(filename)
        root = f.getroot()
        self._classDict = {}
        for classType in [#'TRTProject',
                          #'TRTBase',
                          #'TScanInfo',
                          #'TRTImageData',
                          'TRTSpectrum']:
            content = root.findall(".//ClassInstance[@Type='%s']" % classType)
            self._classDict[classType] = content

        self.__artaxTScanInfo = self.__getArtaxTScanInfo(root)

        self._cacheScan = None
        self._file = os.path.abspath(filename)
        if self.scanno():
            self._cacheScan = ArtaxScan(self._classDict["TRTSpectrum"][0], 0, self._file)
            self._lastScan = 0

    @property
    def artaxTScanInfo(self):
        return self.__artaxTScanInfo

    def __getArtaxTScanInfo(self, root):
        # obtain some Artax Map specific information
        node = root.find(".//ClassInstance[@Type='TScanInfo']")
        scanInfoKeys = ["XFirst",
                        "YFirst",
                        "ZFirst",
                        "XLast",
                        "YLast",
                        "ZLast",
                        "MeasNo", # number of spectra
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
        # done with the Artax map specific information
        return scanInfo
        
    def scanno(self):
        return len(self._classDict["TRTSpectrum"])

    def __getitem__(self, item):
        if item == self._lastScan and self._cacheScan:
            scan = self._cacheScan
        else:
            scan = ArtaxScan(self._classDict["TRTSpectrum"][item], item, self._file)
            self._lastScan = item
            self._cacheScan = scan
        return scan

    def list(self):
        return "1:%d" % self.scanno()

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        n = key.split(".")
        return self.__getitem__(int(n[0])-1)

    def allmotors(self):
        if self.scanno():
            return self._cacheScan._motorNames * 1
        else:
            return []

class ArtaxScan(object):
    def __init__(self, spectrumNode, number, filename):
        self._node = spectrumNode
        self._number = number
        self.__data = None
        command = ""
        if "Name" in spectrumNode.keys():
            command = "%s" % self._node.attrib["Name"]
        else:
            command = "TRTSpectrum %d" % number

        # we expect only one spectrum (if not we would use findall)
        self._spectra = [self._node.find(".//Channels")]

        # get the position(s) at which the spectrum was collected
        self._motorNames = []
        self._motorValues = []
        keyToSearch = ".//ClassInstance[@Type='TRTAxesHeader']//AxesParameter"
        positionsNode = self._node.find(keyToSearch)
        if positionsNode:
            for child in positionsNode:
                if "AxisName" in child.attrib:
                    motorName = child.attrib["AxisName"]
                    if "AxisPosition" in child.attrib:
                        motorValue = myFloat(child.attrib["AxisPosition"])
                        self._motorNames.append(motorName)
                        self._motorValues.append(motorValue)
        elif "PDHID" in command and "(X " in command and " Y " in command:
            # Aaron's approximation to Artax files
            # we should not crash because of it
            try:
                import re
                expr = r"(?:[XY] \d+(?:[.]\d*)?|[.]\d+)"
                XY = re.findall(expr, command)
                if len(XY) == 2:
                    X, Y = XY
                    self._motorNames = ["x", "y"]
                    self._motorValues = [myFloat(X.split(" ")[-1]),
                                         myFloat(Y.split(" ")[-1])]
            except:
                _logger.warning("Could not extract positions from %s" % command)

        # get the additional information
        info = {}
        infoKeys = ['HighVoltage', 'TubeCurrent',
                    'RealTime', 'LifeTime', 'DeadTime',
                    'ZeroPeakPosition', 'ZeroPeakFrequency', 'PulseDensity',
                    'Amplification', 'ShapingTime',
                    'Date','Time',
                    'ChannelCount','CalibAbs', 'CalibLin']
        classTypeList = ['TRTSpectrumHeader',
                         'TRTGeneratorHeader',
                         'TRTSpectrumHardwareHeader']
        for classType in classTypeList:
            nodeToSearch = ".//ClassInstance[@Type='%s']" % classType
            target = self._node.find(nodeToSearch)
            if target is None:
                _logger.debug("Unused class <%s>" % classType)
                continue
            for child in target:
                if child.tag in ["Date", "Time"]:
                    info[child.tag] = child.text
                elif child.tag in infoKeys:
                    info[child.tag] = myFloat(child.text)

        for key in infoKeys:
            if key not in info:
                _logger.debug("key not found %s" % key)

        self._command = command
        scanheader = []
        scanheader.append("#S %d %s" % (self._number + 1, self.command()))
        i = 0
        for key in infoKeys:
            scanheader.append("#U%d %s %s" % (i, key, info.get(key, "Unknown")))
            i += 1
        liveTime = info.get('LifeTime', None)
        realTime = info.get('RealTime', liveTime)
        if liveTime is not None:
            scanheader.append("#@CTIME %f %f %f" % (myFloat(realTime)/1000.,
                                                    myFloat(liveTime)/1000.,
                                                    myFloat(realTime)/1000.))

        scanheader.append("#@CALIB %f %f 0" % (myFloat(info.get('CalibAbs', 0.0)),
                                               myFloat(info.get('CalibLin', 1.0))))
        self._scanHeader = scanheader

        self._fileHeader = ["#F %s" % filename]
        if len(self._motorNames):
            spacing = " " * 4
            motorsLine = "#O0%s" % spacing
            for mne in self._motorNames:
                motorsLine += "%s%s" % (spacing, mne)
            self._fileHeader.append(motorsLine)

    def _readSpectrum(self, channelsNode):
        if self.__data is None:
            self.__data = numpy.array([myFloat(x) for x in channelsNode.text.split(',')],
                                      dtype=numpy.float32)
        return self.__data

    def nbmca(self):
        return len(self._spectra)

    def mca(self, number):
        return self._readSpectrum(self._spectra[number - 1])

    def alllabels(self):
        return []

    def allmotorpos(self):
        return self._motorValues

    def command(self):
        return self._command

    def date(self):
        return self._data["TimeStamp"][self._number]

    def fileheader(self):
        return self._fileHeader

    def header(self, key):
        _logger.debug("Requested key = %s", key)
        _logger.debug("Requested key = %s", key)
        if key in ['S', '#S']:
            return self.fileheader()[0]
        elif key == 'N':
            return []
        elif key == 'L':
            return []
        elif key in ['D', '@CTIME', '@CALIB', '@CHANN']:
            for item in self._scanHeader:
                if item.startswith("#" + key):
                    return [item]
            return []
        elif key == "" or key == " ":
            return self._scanHeader
        else:
            return []

    def order(self):
        return 1

    def number(self):
        return self._number + 1

    def lines(self):
        return 0

def isArtaxFile(filename):
    try:
        if filename[-3:].lower() not in ["rtx", "spx"]:
            return False
        with open(filename, 'rb') as f:
            # expected to read an xml file
            someChar = f.read(20).decode()
        if "xml version" in someChar:
            return True
    except:
        pass
    return False

def test(filename):
    if isArtaxFile(filename):
        sf=ArtaxFileParser(filename)
    else:
        print("Not an Artax .spx or .rtx File")
        return
    sf = ArtaxFileParser(filename)
    print(sf[0].nbmca())
    print(sf[0].mca(1))
    print(sf[0].header('S'))
    #print(sf[0].header('D'))
    #print(sf[0].alllabels())
    print(sf.allmotors())
    print(sf[0].allmotorpos())
    print(sf[0].header('@CTIME'))
    print(sf[0].header('@CALIB'))
    print(sf[0].header(''))

if __name__ == "__main__":
    test(sys.argv[1])
