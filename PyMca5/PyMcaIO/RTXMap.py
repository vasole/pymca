#!/usr/bin/env python
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
import os
import numpy
import types

#spx file format is based on XML
import xml.etree.ElementTree as ElementTree
from PyMca5.PyMcaCore import DataObject
SOURCE_TYPE = "EdfFileStack"

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
        DataObject.DataObject.__init__(self)
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)

        f = ElementTree.parse(filename)
        root = f.getroot()
        #print("Tag  = ", root.tag)
        #print("Tag  = ", root.attrib)
        info = {}
        if 1:
            # this works
            for key in ["XFirst", "YFirst", "ZFirst", "XLast", "YLast", "ZLast", "MeasNo"]:
                element = root.find(".//ClassInstance/%s" % key)
                if element is not None:
                    info[key] = myFloat(element.text)
        if "MeasNo" in info:
            nSpectra = int(info["MeasNo"])
        else:
            nSpectra = None
            
        if 1:
            # This works
            i = 0
            axesList = root.findall('.//ClassInstance/AxesParameter')
            nRows = None
            nColumns = 0
            y = None
            done = False
            for axes in axesList:
                if not done:
                    for axis in axes:
                        if axis.attrib['AxisName'] == 'y':
                            if i == 0:
                                y = myFloat(axis.attrib['AxisPosition'])
                            if y == myFloat(axis.attrib['AxisPosition']):
                                nColumns += 1
                            else:
                                done = True
                    #print("#U%d %s  %f  %s" % (i,
                    #                                 axis.attrib['AxisName'],
                     #                                myFloat(axis.attrib['AxisPosition']),
                     #                                axis.attrib['AxisUnit']))
                i += 1
            #print("Found %d positions" % i)
            #print("nColumns = %d" % nColumns)
            if nColumns == 0:
                nColumns = 1
            nRows = i //nColumns
            #print("nRows = %d"  % nRows)
            if nSpectra is None:
                nSpectra = nRows * nColumns
            else:
                if nSpectra != nRows * nColumns:
                    print("WARNING Header info does not match")
        data = None
        if 1:
            # This works
            for key in ["Channels"]:
                keyToSearch = './/ClassInstance/%s' % key
                i = 0
                for content in root.findall(keyToSearch):
                    if content is not None:
                        spectrum = numpy.array([myFloat(x) for x in content.text.split(',')])
                        nChannels = len(spectrum)
                        if data is None:
                            data = numpy.zeros((nSpectra, nChannels))
                        if i < nSpectra:
                            data[i] = spectrum
                        i += 1
                print("Found %d spectra" % i)
        #print nSpectra
        #print nRows * nColumns
        data.shape = nRows, nColumns, nChannels
        self.data = data
        self.info = info

def test(filename):
    a = RTXMap(filename)
    print("info = ", a.info)
    print("Data shape = ", a.data.shape)

if __name__ == "__main__":
    test(sys.argv[1])

