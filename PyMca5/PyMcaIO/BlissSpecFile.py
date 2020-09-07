#!/usr/bin/env python
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""
This class just puts in evidence the Specfile methods called from
PyMca.
It can be used to wrap other formats as specile
"""
import os
import sys
import numpy
import logging
_logger = logging.getLogger(__name__)

try:
    import RedisTools as redis
    HAS_REDIS = True
except:
    try:
        import PyMca5.PyMcaCore.RedisTools as redis
        HAS_REDIS = True
    except:
        _logger.info("Cannot import PyMca5.PyMcaCore.RedisTools")
        HAS_REDIS = False


class BlissSpecFile(object):
    def __init__(self, filename):
        """
        filename is the name of the bliss session
        """
        if not HAS_REDIS:
            raise ImportError("Could not import RedisTools")
        if filename not in redis.get_sessions_list():
            return None
        self._scan_nodes = []
        self._session = filename
        self._filename = redis.get_session_filename(self._session)
        self._scan_nodes = redis.get_session_scan_list(self._session,
                                                  self._filename)

    def list(self):
        """
        If there is only one scan returns 1:1
        with two scans returns 1:2
        """
        _logger.debug("list method called")
        scanlist = ["%s" % scan.name.split("_")[0] for scan in self._scan_nodes]
        return ",".join(scanlist)

    def __getitem__(self, item):
        """
        Returns the scan data
        """
        _logger.debug("__getitem__ called")
        return BlissSpecScan(self._scan_nodes[item])

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
        return len(self._scan_nodes)

    def allmotors(self):
        return []

class BlissSpecScan(object):
    def __init__(self, scanNode):
        self._node = scanNode
        self._identification = scanNode.name.split("_")[0] + ".1"
        self._spectra = redis.get_spectra(scanNode)
        self._counters = redis.get_scan_data(scanNode)
        self._motors = redis.scan_info(self._node).get("positioners", {})

    def alllabels(self):
        """
        These are the labels associated to the counters
        """
        return [key for key in self._counters]

    def allmotors(self):
        positioners = self._motors.get("positioners_start", {})
        return [key for key in positioners if not hasattr(positioners[key], "endswith")]

    def allmotorpos(self):
        positioners = self._motors.get("positioners_start", {})
        return [positioners[key] for key in positioners if not hasattr(positioners[key], "endswith")]

    def cols(self):
        return len(self._counters)

    def command(self):
        _logger.debug("command called")
        return redis.scan_info(self._node).get("title",
                                                      "No COMMAND")
    def data(self):
        return numpy.transpose(self.__data)

    def datacol(self, col):
        keys = list(self._counters.keys())
        return self._counters[keys[col]]

    def dataline(self,line):
        return self.__data[line,:]

    def date(self):
        text = 'sometime'
        return redis.scan_info(self._node).get("start_time", text)

    def fileheader(self):
        _logger.debug("file header called")
        labels = '#L '
        for label in self._counters:
            labels += '  '+label
        return ['#S %s  %s' %(self._node.name.split("_")[0], self.command()),
                '#D %s' % self.date(),
                '#N %d' % self.cols(),
                labels]

    def header(self,key):
        if key == 'S':
            return self.fileheader()[0]
        elif key == 'D':
            return self.fileheader()[1]
        elif key == 'N':
            return self.fileheader()[-2]
        elif key == 'L':
            return self.fileheader()[-1]
        elif key == '@CALIB':
            output = []
            return output
        elif key == '@CTIME':
            # expected to send Preset Time, Live Time, Real (Elapsed) Time
            output = []
            return output
        elif key == "" or key == " ":
            return self.fileheader()
        else:
            output = []
            return output

    def order(self):
        return 1

    def number(self):
        return int(self._node.name.split("_")[0])

    def lines(self):
        if self.scantype == 'SCAN':
            return self.rows
        else:
            return 0

    def nbmca(self):
        return len(self._spectra)

    def mca(self,number):
        if number <= 0:
            raise IndexError("Mca numbering starts at 1")
        elif number > self.nbmca():
            raise IndexError("Only %d MCAs in file" % self.nbmca())
        return self._spectra[number - 1]

def isBlissSpecFile(filename):
    if os.path.exists(filename):
        return False
    try:
        if filename in redis.get_sessions_list():
            return True
    except:
        pass
    return False    

def test(filename):
    sf = BlissSpecFile(filename)
    print(sf[0].header('D'))
    print(sf[0].alllabels())
    print(sf[0].nbmca())
    if sf[0].nbmca():
        print(sf[0].mca(1))
    print(sf[0].header('S'))
    print(sf[0].allmotors())
    print(sf[0].allmotorpos())
    print(sf[0].header('@CTIME'))
    print(sf[0].header('@CALIB'))
    print(sf[0].header(''))

if __name__ == "__main__":
    test(sys.argv[1])
