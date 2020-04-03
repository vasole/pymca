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
import types

#spx file format is based on XML
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

class SPXFileParser(SpecFileAbstractClass.SpecFileAbstractClass):
    def __init__(self, filename):
        SpecFileAbstractClass.SpecFileAbstractClass.__init__(self, filename)
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)
        f = ElementTree.parse(filename)
        root = f.getroot()
        info = {}
        #I do not use find all, providing support for only one spectrum
        infoKeys = ['HighVoltage', 'TubeCurrent',
                    'RealTime', 'LifeTime', 'DeadTime',
                    'ZeroPeakPosition', 'ZeroPeakFrequency', 'PulseDensity',
                    'Amplification', 'ShapingTime',
                    'Date','Time',
                    'ChannelCount','CalibAbs', 'CalibLin']

        for key in infoKeys:
            keyToSearch = './/ClassInstance/%s' % key
            content = root.find(keyToSearch)
            if content is not None:
                info[key] = content.text
        axes = root.find('.//ClassInstance/AxesParameter')
        data = numpy.array([float(x) for x in root.find('.//Channels').text.split(',')])
        data.shape = len(data), 1

        scanheader = ['#S 1  ' + info.get('name', "Unknown name")]
        i = 0
        if axes is not None:
            for axis in axes:
                scanheader.append("#U%d %s  %f  %s" % (i,
                                                 axis.attrib['AxisName'],
                                                 myFloat(axis.attrib['AxisPosition']),
                                                 axis.attrib['AxisUnit']))
                i += 1
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

        self.scandata = [SpecFileAbstractClass.SpecFileAbstractScan(data,
                                scantype="MCA",
                                scanheader=scanheader)]

def test(filename):
    SPXFileParser(filename)

if __name__ == "__main__":
    test(sys.argv[1])

