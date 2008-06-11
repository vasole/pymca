#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
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
"""
This clases just put in evidence the Specfile methods called from
PyMca.
It can be used to wrap other formats as specile
"""
import os
import numpy
DEBUG = 0
class SpecFileAbstractClass:
    def __init__(self, filename):
        if not os.path.exists(filename):
            return None

    def list(self):
        """
        If there is only one scan returns 1:1
        with two scans returns 1:2
        """
        if DEBUG:
            print "list method called"
        return "1:1"

    def __getitem__(self, item):
        """
        Returns the scan data
        """
        if DEBUG:
            print "__getitem__ called"
        return self.scandata[item]

    def select(self, key):
        """
        key is of the from s.o
        scan number, scan order
        """
        n = key.split(".")
        return self.__getitem__(int(n[0])-1)

    def scanno(self):
        """
        Gives back the number of scans in the file
        """
        return 0

class SpecFileAbstractScan:
    def __init__(self, data, scantype=None, identification=None, scanheader=None, labels=None):
        if identification is None:identification='1.1'
        if scantype is None:scantype='SCAN'
        self.scanheader = scanheader
        (rows, cols) = data.shape
        if scantype == 'SCAN':
            self.__data = numpy.zeros((rows, cols +1 ), numpy.float)
            self.__data[:,0] = numpy.arange(rows) * 1.0
            self.__data[:,1:] = data * 1
            self.__cols = cols + 1
            self.labels = ['Point']
        else:
            self.__data = data
            self.__cols = cols
            self.labels = []
        self.scantype = scantype
        self.rows = rows
        if labels is None:
            for i in range(cols):
                self.labels.append('Column %d'  % i)
        else:
            for label in labels:
                self.labels.append('%s' % label)
        n = identification.split(".")
        self.__number = int(n[0])
        self.__order  = int(n[1])

    def alllabels(self):
        """
        These are the labels associated to the counters
        """
        if self.scantype == 'SCAN':
            return self.labels
        else:
            return []

    def allmotorpos(self):
        return []
        
    def cols(self):
        return self.__cols
    
    def command(self):
        if DEBUG:
            print "command called"
        text = ""
        if self.scanheader is not None:
            if len(self.scanheader):
                text = self.scanheader[0]
        return text
        
    def data(self):
        return numpy.transpose(self.__data)
    
    def datacol(self,col):
        return self.__data[:,col]
        
    def dataline(self,line):
        return self.__data[line,:]
        
    
    def date(self):
        text = 'sometime'
        return text
            
    def fileheader(self):
        if DEBUG:
            print "file header called"
        labels = '#L '
        for label in self.labels:
            labels += '  '+label
        if self.scantype == 'SCAN':
            return ['#S1 Unknown command','#N %d' % self.cols,labels] 
        else:
            if self.scanheader is None:
                return ['#S1 Unknown command']
            else:
                if DEBUG:
                    print "returning ",self.scanheader
                return self.scanheader
    
    def header(self,key):
        if   key == 'S': return self.fileheader()[0]
        elif key == 'N':return self.fileheader()[-2]
        elif key == 'L':return self.fileheader()[-1]
        elif key == '@CALIB':
            output = []
            if self.scanheader is None: return output
            for line in self.scanheader:
                if line.startswith("#@CALIB"):
                    output = [line]
                    break
            return output
        elif key == "" or key == " ":return self.fileheader()
        else:
            #print "requested key = ",key 
            return []
    
    def order(self):
        return self.__order
        
    def number(self):
        return self.__number
    
    def lines(self):
        if self.scantype == 'SCAN':
            return self.rows
        else:
            return 0
            
    def nbmca(self):
        if self.scantype == 'SCAN':
            return 0
        else:
            return self.__cols
        
    def mca(self,number):
        return self.__data[:,number-1]
    


def test():
    pass

if __name__ == "__main__":
    test()
        
