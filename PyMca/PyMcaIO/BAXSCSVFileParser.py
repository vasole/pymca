#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import sys
import os
import numpy
from PyMca.PyMcaIO import SpecFileAbstractClass

DEBUG = 0

class BufferedFile(object):
    def __init__(self, filename):
        f = open(filename, 'r')
        self.__buffer = f.read()
        f.close()
        self.__buffer = self.__buffer.replace("\r", "\n")
        self.__buffer = self.__buffer.replace("\n\n", "\n")
        self.__buffer = self.__buffer.split("\n")
        self.__currentLine = 0

    def readline(self):
        if self.__currentLine >= len(self.__buffer):
            return ""
        line = self.__buffer[self.__currentLine]
        self.__currentLine += 1
        return line

    def close(self):
        self.__buffer = [""]
        self.__currentLine = 0
        return

class BAXSCSVFileParser(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        _fileObject = BufferedFile(filename)

        #Only one measurement per file
        header = []
        ddict = {}
        line = _fileObject.readline()
        if not (line.startswith("Bruker AXS") or \
               (("KTI" in line) and ("Spectrum" in line))):
            raise IOError("This does look as a Bruker AXS Handheld CSV file")
        while not line.startswith("Channel#"):
            header.append(line)
            line = _fileObject.readline()
        header.append(line)
        line = _fileObject.readline()
        line = line.replace('"',"")
        splitLine = line.split(",")
        data = []
        while(len(splitLine)):
            if len(splitLine[0]):
                try:
                    data.append([float(x) for x in splitLine if len(x) > 0])
                except ValueError:
                    break
            else:
                break
            line = _fileObject.readline()
            line = line.replace('"',"")
            splitLine = line.split(",")
        _fileObject.close()
        dataColumnNames = [x for x in header[-1].split(",") if len(x) > 0]
        data = numpy.array(data, dtype=numpy.float)
        #print(header)
        #print(data)
        labels=[]
        #create an abstract scan object
        self._scan = [BAXSCSVScan(data,
                              scantype='MCA',
                              scanheader=header,
                              #labels=labels,
                              #motor_values=self.motorValues,
                              point=False)]

    def __getitem__(self, item):
        return self._scan[item]

    def scanno(self):
        """
        Gives back the number of scans in the file
        """
        return len(self_scan)
        
    def list(self):
        return "1:1"

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        n = key.split(".")
        return self.__getitem__(int(n[0])-1)

    def allmotors(self):
        if DEBUG:
            print("BAXCSVFileParser allmotors called")
        return []

class BAXSCSVScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype='MCA',
                 identification="1.1", scanheader=None, labels=None,
                 motor_values=None, point=False):
        SpecFileAbstractClass.SpecFileAbstractScan.__init__(self,
                    data, scantype=scantype, identification=identification,
                    scanheader=scanheader, labels=labels, point=point)
        self._data = data

    def nbmca(self):
        return 1
        
    def mca(self, number):
        # it gives the last column (some files have three columns)
        # corresponding to channels, counts and (probably) corrected counts
        if number <= 0:
            raise IndexError("Mca numbering starts at 1")
        elif number > self.nbmca():
            raise IndexError("Only %d MCA's" % number)
        return self._data[:, -1]

def isBAXSCSVFile(filename):
    f = open(filename, 'r')
    try:
        line = f.readline()
    except:
        f.close()
        return False
    try:
        if filename.lower().endswith(".csv"):
            if line.startswith("Bruker AXS") or \
               (("KTI" in line) and ("Spectrum" in line)):
                return True
    except:
        pass
    return False

def test(filename):
    if isBAXSCSVFile(filename):
        sf=BAXSCSVFileParser(filename)
    else:
        print("Not a Bruker AXS File")
    print(sf[0].header('S'))
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    #print(sf[0].allmotorsvalues())
    print(sf[0].nbmca())
    print(sf[0].mca(1))

if __name__ == "__main__":
    test(sys.argv[1])
