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
import logging

_logger = logging.getLogger(__name__)


class BufferedFile(object):
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.__buffer = f.read()
        f.close()
        self.__buffer = self.__buffer.replace(b"\x00", b"")
        self.__buffer = self.__buffer.replace(b"\r", b"\n")
        self.__buffer = self.__buffer.replace(b"\n\n", b"\n")
        self.__buffer = self.__buffer.split(b"\n")
        self.__currentLine = 0

    def readline(self):
        if self.__currentLine >= len(self.__buffer):
            return ""
        if self.__currentLine == 0:
            # if we try to decode the first two characters we get an error
            line = "TT" + self.__buffer[self.__currentLine][2:].decode("utf-8")
        else:
            line = self.__buffer[self.__currentLine].decode("utf-8")
        self.__currentLine += 1
        return line

    def close(self):
        self.__buffer = [""]
        self.__currentLine = 0
        return

class OlympusCSVFileParser(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        _fileObject = BufferedFile(filename)

        #Several measurement per file
        ddict = {}
        line = _fileObject.readline()
        if not line.startswith("TTTestID"):
            raise IOError("This does not look an Olympus CSV file")
        splitLine = line[2:].split("\t")
        nSpectra = len(splitLine) - 1
        channel = 0
        while(len(splitLine) > 1):
            key = splitLine[0]
            if len(key):
                ddict[key] = splitLine[1:]
                if key == "NumData":
                    nChannels = int(splitLine[1])
                    ddict['data'] = numpy.zeros((nSpectra, nChannels), numpy.float64)
            else:
                ddict['data'][:, channel] = [float(x) for x in splitLine[1:]]
                channel += 1
            line = _fileObject.readline()
            splitLine = line.split("\t")
        _fileObject.close()
        interestingMotors = ["VacPressure",
                             "TubeVoltageSet",
                             "AmbientPressure",
                             "Realtime",
                             "Livetime",
                             "FilterPosition",
                             "TubeCurrentMon",
                             "TubeVoltageMon",
                             "ExposureNum",
                             "TubeCurrentSet"]
        self._motorNames = []
        for key in ddict.keys():
            #print("Key = ", key, "ken = ", len(ddict[key]))
            if key in interestingMotors:
                self._motorNames.append(key)
        self._data = ddict

    def __getitem__(self, item):
        motorValues = []
        for key in self._motorNames:
            motorValues.append(float(self._data[key][item]))
        return OlympusCSVScan(self._data, item, motorValues=motorValues)

    def scanno(self):
        """
        Gives back the number of scans in the file
        """
        return self._data['data'].shape[0]

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
        return self._motorNames

class OlympusCSVScan(object):
    def __init__(self, data, number, motorValues=None):
        if motorValues is None:
            motorValues = []
        self._data = data
        self._number = number
        self._motorValues = motorValues
        self._scanHeader = [self.fileheader()[0]]
        if "TimeStamp" in self._data:
            self._scanHeader.append("#D %s" % \
                                    self._data["TimeStamp"][self._number])
        
        #return the life time, the preset the elapsed?
        # to be safe, I return the LiveTime
        if "Livetime" in self._data:
            self._scanHeader.append("#@CTIME %s %s %s" % (self._data["Livetime"][self._number],
                                         self._data["Livetime"][self._number],
                                         self._data["Livetime"][self._number]))
        self._scanHeader.append("#@CHANN %d 0 %d 1" % (self._data["data"].shape[1],
                                           self._data["data"].shape[1]-1))
        if "Offset" in self._data:
            if "Slope" in self._data:
                self._scanHeader.append("#@CALIB %s %s 0.0" % \
                                        (self._data["Offset"][self._number],
                                        self._data["Slope"][self._number]))

    def nbmca(self):
        return 1

    def mca(self, number):
        # it gives the last column (some files have three columns)
        # corresponding to channels, counts and (probably) corrected counts
        if number <= 0:
            raise IndexError("Mca numbering starts at 1")
        elif number > self.nbmca():
            raise IndexError("Only %d MCA's" % number)
        return self._data['data'][self._number]

    def alllabels(self):
        return []

    def allmotorpos(self):
        return self._motorValues

    def command(self):
        return self._data['TestID'][self._number]

    def date(self):
        return self._data["TimeStamp"][self._number]

    def fileheader(self):
        a = "#S %d %s" % (self._number + 1, self.command()) 
        return [a]

    def header(self, key):
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

def isOlympusCSVFile(filename):
    f = open(filename, 'rb')
    try:
        line = f.read(14)
    except:
        f.close()
        return False
    f.close()
    line = line.replace(b"\x00", b"")
    try:
        if filename.lower().endswith(".csv"):
            # expected chain b"\xff\xfeTestID"
            if line[2:].startswith(b"TestID"):
                return True
    except:
        pass
    return False

def test(filename):
    if isOlympusCSVFile(filename):
        sf=OlympusCSVFileParser(filename)
    else:
        print("Not an Olympus CSV File")
        return
    print(sf[0].header('S'))
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    #print(sf[0].allmotorsvalues())
    print(sf[0].nbmca())
    print(sf[0].mca(1))

if __name__ == "__main__":
    test(sys.argv[1])
