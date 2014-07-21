#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
from PyMca5.PyMcaIO import SpecFileAbstractClass

class Fit2DChiFileParser(SpecFileAbstractClass.SpecFileAbstractClass):
    def __init__(self, filename):
        SpecFileAbstractClass.SpecFileAbstractClass.__init__(self, filename)
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)
        f = open(filename, 'r')
        self.__buffer = f.read()
        f.close()
        self.__buffer=self.__buffer.replace("\r", "\n")
        self.__buffer=self.__buffer.replace("\n\n", "\n")
        self.__buffer = self.__buffer.split("\n")
        header = [self.__buffer[0]]
        a = self.__buffer[0].split(":")
        command = a[-1]
        labels = []
        npoints = 0
        self.__currentLine = 1
        lenBuffer = len(self.__buffer)
        while (npoints == 0) and (self.__currentLine < lenBuffer):
            line = self.__buffer[self.__currentLine]
            self.__currentLine += 1
            header.append(line)
            try:
                npoints = int(line)
            except:
                labels.append(line)
                pass
        if len(labels[-1]) == 0:
            labels[-1] = "Intensity"
        if npoints == 0:
            raise IOError("Problem reading file. Number of points is 0.")
        data = numpy.zeros((npoints, len(labels)), numpy.float32)
        for i in range(npoints):
            if self.__currentLine < lenBuffer:
                line = self.__buffer[self.__currentLine]
                try:
                    data[i,:] = [float(x) for x in line.split()]
                except ValueError:
                    if i == 0:
                        values = [float(x) for x in line.split()]
                        nActualValues = len(values)
                        if nActualValues < len(labels):
                            labels = labels[-nActualValues:]
                            data = numpy.zeros((npoints, len(labels)),
                                               numpy.float32)
                            data[i,:] = values
                            self.__currentLine += 1
                            continue
                    raise
            self.__currentLine += 1

        scanheader = ['#S 1  ' + command]
        self.scandata = [SpecFileAbstractClass.SpecFileAbstractScan(data,
                                scantype="SCAN",
                                identification="1.1",
                                labels=labels,
                                scanheader=scanheader)]
        x0 = data[0, 0]
        x1 = data[-1,0]
        delta = (x1-x0)/npoints
        data = data[:,1]
        scanheader = ['#S 2  ' + command]
        scanheader.append("#@CALIB %f %f 0" % (x0, delta))
        self.scandata.append(SpecFileAbstractClass.SpecFileAbstractScan(data,
                                scantype="MCA",
                                identification="2.1",
                                scanheader=scanheader))

    def list(self):
        #We have two "scans"
        return "1:2"

def isFit2DChiFile(filename):
    #Obviously I should put a better test than this one
    if not filename.upper().endswith(".CHI"):
        return False
    return True

def test(filename):
    if isFit2DChiFile(filename):
        sf=Fit2DChiFileParser(filename)
    else:
        print("Not a Fit2D .Chi File")
    print(sf[0].alllabels())
    print(dir(sf[0]))



if __name__ == "__main__":
    test(sys.argv[1])

