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
__doc__="""
Minimalistic support to read and write JCAMP-DX files.

The first three lines of a JCAMP-DX file have to be:

##TITLE=
##JCAMP-DX= JCAMP version number $$ Optional comment about the writing software
##DATA TYPE=

The next two lines were considered mandatory in Applied Spectroscopy 42 (1998) 151-162.
##ORIGIN=
##OWNER=

Then came several optional lines
##XUNITS=
##YUNITS=
##XFACTOR=
##YFACTOR=
##FIRSTX=
##LASTX=
##NPOINTS=
##FIRSTY=
##XYDATA=(X++(Y..Y))

data block
##END=
"""
import os
import sys
import re
import numpy
patternKey=re.compile(r'^[#][#]\s*(?P<name>[^=]+)=(?P<value>.*)$')
#patternNumber = re.compile(r'([+-]?\d+\.?\d*)')
patternNumber = re.compile(r'[+-]?[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?')
DEBUG = 0
if DEBUG:
    text = '1.23 +2 456-7.98+5 10+3.4E+01 98-7.6E-2+3'
    print("RESULT:")
    print(re.findall(patternNumber, text))
    print("EXPECTED:")
    print(['1.23', '+2', '456', '-7.98', '+5', '10', '+3.4E+01', '98', '-7.6E-2', '+3'])

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

class JcampReader(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise IOError("File %s does not exists")

        _fileObject = BufferedFile(filename)
        # only one measurement per file
        ddict = {}
        header = []
        # test we are actually using a JCAMP-DX file
        testKeys = ["TITLE", "JCAMP-DX", "DATA TYPE"]
        for i, key in enumerate(testKeys):
            line = _fileObject.readline()
            info = re.findall(patternKey, line)
            if len(info):
                actualKey = key.replace(" ","")
                if info[0][0].replace(" ", "").upper().startswith(actualKey):
                    header.append(line)
                    ddict[key] = info[0][1]
            else:
                raise IOError("This does not look as a JCAMP-DX file")
        line = _fileObject.readline()
        while not line.startswith("##XYDATA"):
            header.append(line)
            line = _fileObject.readline()
        key, value = re.findall(patternKey, line)[0]
        ddict["XYDATA"] = value.upper()
        header.append(line)
        # we are at the data block
        data = []
        line = _fileObject.readline()
        while not line.startswith("##END"):
            data.append(line)
            line = _fileObject.readline()
        _fileObject.close()
        self._header = header
        self.info = self.parseHeader()
        self.data = self.parseData(data)

    def parseHeader(self, keyList=None):
        if keyList is None:
            keyList = ["TITLE", "JCAMP-DX", "DATA TYPE",
                       "ORIGIN", "OWNER",
                       "XUNITS", "YUNITS",
                       "XFACTOR", "YFACTOR",
                       "FIRSTX", "LASTX", "DELTAX", "NPOINTS",
                       "FIRSTY",
                       "XYDATA"]
        ddict = {}
        for line in self._header:
            for key in keyList:
                info = re.findall(patternKey, line)
                if len(info):
                    actualKey = key.replace(" ","")
                    if info[0][0].replace(" ", "").upper().startswith(actualKey):
                        ddict[key] = info[0][1]
        return ddict

    def parseData(self, dataLines):
        if self.info['XYDATA'].upper().strip() not in ["(X++(Y..Y))", "(XY..XY)"]:
            raise IOError("Format <%s> not supported yet" %  self.info['XYDATA'])
        if self.info['XYDATA'].upper().strip() == "(X++(Y..Y))":
            xValues = []
            yValues = []
            nValues = []
            for line in dataLines:
                values = [float(x) for x in re.findall(patternNumber, line)]
                xValues.append(values[0])
                yValues += values[1:]
                nValues.append(len(values) - 1)
            # the y values are all there, but the x values are not
            lastX = float(self.info["LASTX"])
            try:
                # try to apply the formula given in the article
                # the problem is that DELTAX is not mandatory
                firstX = float(self.info["FIRSTX"])
                deltaX = float(self.info["DELTAX"])
                nPoints = int(self.info.get("NPOINTS", 0))
                if nPoints != len(yValues):
                    print("Number of points does not match number of values")
                    nPoints = len(yValues)
                # this formula is given in the article
                x = firstX + numpy.arange(nPoints) * \
                    ((lastX - firstX) / (nPoints - 1.0))
            except KeyError:
                xValues.append(lastX)
                x = numpy.zeros((len(yValues),), dtype=numpy.float)
                start = 0
                nDataLines = len(nValues)
                for i in range(nDataLines):
                    n = nValues[i]
                    end = start + n
                    if i == (nDataLines - 1):
                        endpoint = True
                    else:
                        endpoint = False
                    x[start:end] = numpy.linspace(xValues[i],
                                              xValues[i+1],
                                              n, endpoint=endpoint)
                    start = end
        else:
            # XY, XY, ...
            values = []
            for line in dataLines:
                values += [float(x) for x in re.findall(patternNumber, line)]
            values = numpy.array(values)
            x = values[0::2]
            yValues = values[1::2]
        xFactor = float(self.info.get("XFACTOR", 1.0))
        yFactor = float(self.info.get("YFACTOR", 1.0))
        return x * xFactor, numpy.array(yValues) * yFactor

def isJcampFile(filename):
    try:
        testKeys = ["TITLE", "JCAMP-DX", "DATA TYPE"]
        # if read mode is 'rb' python 3 does not work
        fid = open(filename, mode='r')
        for i, key in enumerate(testKeys):
            line = fid.readline()
            info = re.findall(patternKey, line)
            if len(info):
                actualKey = key.replace(" ","")
                if info[0][0].replace(" ", "").upper().startswith(actualKey):
                    continue
            else:
                return False
        return True
    except:
        return False

if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    print("is JCAMP-DX File?", isJcampFile(filename))
    instance = JcampReader(filename)
    print(instance.info)
    x, y = instance.data
    try:
        import matplotlib.pylab as plt
        plt.figure(0)
        plt.plot(x, y)
        plt.xlabel(instance.info.get('XUNITS', 'X'))
        plt.ylabel(instance.info.get('YUNITS', 'Y'))
        plt.show()
    except:
        pass
