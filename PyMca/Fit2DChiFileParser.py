#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
import sys
import os
import numpy
import SpecFileAbstractClass

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
                    data[i,:] = map(float, line.split())
                except ValueError:
                    if i == 0:
                        values = map(float, line.split())
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
        
