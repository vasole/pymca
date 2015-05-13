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
        try:
            fileheader = instance._header
        except:
            print("JCampFileParser cannot access '_header' attribute")
            fileheader=None
        data = numpy.zeros((x.size, 2), numpy.float32)
        data[:, 0] = x
        data[:, 1] = y
        self.scandata = []
        scanheader = ["#S 1 %s" % title]
        scanheader.append("#N 2")
        scanheader.append("#L %s  %s" % (xLabel, yLabel))
        scanData = JCAMPFileScan(data,
                                 scantype="SCAN",
                                 scanheader=scanheader,
                                 labels=[xLabel, yLabel],
                                 fileheader=fileheader)
        self.scandata.append(scanData)
        scanheader = ["#S 2 %s" % title]
        if jcampDict['XYDATA'].upper() ==  '(X++(Y..Y))':
            # we can deal with the X axis via its calibration
            scanheader.append("#@CHANN %d  %d  %d  1" % (len(x), 0, len(x) - 1))
            scanheader.append("#@CALIB %f %f 0" % (x[0], x[1] - x[0]))
            scantype = "MCA"
        scanData = JCAMPFileScan(data, scantype="MCA",
                                                      scanheader=scanheader,
                                                      #labels=[xLabel, yLabel],
                                                      fileheader=fileheader)
        self.scandata.append(scanData)

    def __getitem__(self, item):
        return self.scandata[item]

    def list(self):
        return "1:%d" % len(self.scandata)

    def scanno(self):
        return len(self.scandata)

class JCAMPFileScan(SpecFileAbstractClass.SpecFileAbstractScan):
    def __init__(self, data, scantype="SCAN",
                 scanheader=None, labels=None, fileheader=None):
        SpecFileAbstractClass.SpecFileAbstractScan.__init__(self, data,
                            scantype=scantype, scanheader=scanheader,
                            labels=labels)
        self._data = data
        self._fileHeader = fileheader

    def fileheader(self, key=''):
        return self._fileHeader

    def nbmca(self):
        if self.scantype == 'SCAN':
            return 0
        else:
            return 1

    def mca(self, number):
        if number not in [1]:
            raise ValueError("Specfile mca numberig starts at 1")
        return self._data[:, number]

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
