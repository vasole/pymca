#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
from PyMca5.PyMcaIO import JcampReader
from PyMca5.PyMcaIO import SpecFileAbstractClass

class JcampFileParser(SpecFileAbstractClass.SpecFileAbstractClass):
    def __init__(self, filename):
        instance = JcampReader.JcampReader(filename)
        info = instance.info
        jcampDict = info
        x, y = instance.data
        title = jcampDict.get('TITLE', "Unknown scan")
        xLabel = jcampDict.get('XUNITS', 'channel')
        yLabel = jcampDict.get('YUNITS', 'counts')
        scanheader = ["#S 1 %s" % title,
                      "#N 2",
                      "#L %s  %s" % (xLabel, yLabel)]
        try:
            fileheader = instance._header
        except:
            print("JCampFileParser cannot access '_header' attribute")
            fileheader=None
        data = numpy.zeros((x.size, 2), numpy.float32)
        data[:, 0] = x
        data[:, 1] = y
        self.scanData = self.scandata = JCAMPFileScan(data, scantype="SCAN",
                                                      scanheader=scanheader,
                                                      labels=[xLabel, yLabel],
                                                      fileheader=fileheader)

    def __getitem__(self, item):
        if item not in [-1, 0]:
            raise IndexError("Only one scan in the file")
        return self.scanData

    def scanno(self):
        return 1

class JCAMPFileScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype="SCAN",
                 scanheader=None, labels=None, fileheader=None):
        SpecFileAbstractClass.SpecFileAbstractScan.__init__(self, data,
                            scantype=scantype, scanheader=scanheader,
                            labels=labels)
        self._fileHeader = fileheader

    def fileheader(self, key=''):
        return self._fileHeader

    def header(self, key=''):
        if   key == 'S':
            return self.scanheader[0]
        elif key == 'N':
            return self.scanheader[-2]
        elif key == 'L':
            return self.scanheader[-1]
        elif key == '@CALIB':
            output = []
            if self.scanheader is None: return output
            for line in self.scanheader:
                if line.startswith(key) or\
                   line.startswith('#'+key):
                    output.append(line)
            return output
        elif key == '@CTIME':
            # expected to send Preset Time, Live Time, Real (Elapsed) Time
            output = []
            if self.scanheader is None: return output
            for line in self.scanheader:
                if line.startswith(key) or\
                   line.startswith('#'+key):
                    output.append(line)
            return output
        elif key == "" or key == " ":
            return self.scanheader

def isJcampFile(filename):
    return JcampReader.isJcampFile(filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python JCAMPFileParser.py filename")
        sys.exit(0)
    print(" isJCAMPFile = ", isJcampFile(sys.argv[1]))
    sf = JcampFileParser(sys.argv[1])
    print("nscans = ", sf.scanno())
    print("list = ", sf.list())
    print("select = ", sf.select(sf.list()[0]))
